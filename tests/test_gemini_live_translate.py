"""
Summary:
    test_gemini_live_translate.py is the executable specification for TASK-11.1 and
    TASK-11.2: the server-side Gemini Live Translate WebSocket leg and the PCM adapter
    that sits on either side of it.

    The leg is an interpreter transport, not a reasoning engine, so what matters here is
    the wire contract: an environment-selected model, an AUDIO-only response with both
    transcriptions enabled, a BCP-47 target language, `echoTargetLanguage=false`, and
    *no* source-language configuration at all - Gemini detects the source per utterance,
    which is what lets a code-switching speaker be interpreted without reconfiguration.

Covers:
    - REQ-17 setup message, model resolution, reconnect, and explicit close.
    - REQ-17 audio contract: 16 kHz mono PCM16 in ~100 ms chunks, 24 kHz mono out.
    - REQ-17 fail-closed behaviour when the Live endpoint is unreachable.
"""

import base64
import json
import os
import unittest
from unittest.mock import patch

from src.translation.audio import (
    GEMINI_OUTPUT_SAMPLE_RATE,
    INPUT_CHUNK_BYTES,
    INPUT_CHUNK_MS,
    INPUT_SAMPLE_RATE,
    AgoraPublishAdapter,
    GeminiInputEncoder,
    downmix_to_mono,
    resample_pcm16,
)
from src.translation.gemini_live import (
    DEFAULT_LIVE_TRANSLATE_MODEL,
    GeminiLiveTranslateSession,
    LegConfig,
    LegState,
    TranslationUnavailableError,
    parse_server_message,
    resolve_model,
    to_bcp47,
)


def silence(sample_count: int, channels: int = 1, value: int = 0) -> bytes:
    """Builds `sample_count` frames of PCM16 at the given channel count."""
    return (value.to_bytes(2, "little", signed=True)) * sample_count * channels


class FakeConnection:
    """A stand-in for one Gemini Live WebSocket, recording everything sent to it."""

    def __init__(self, fail_on_send_call: int = 0):
        self.sent = []
        self.closed = False
        self.fail_on_send_call = fail_on_send_call
        self._send_calls = 0

    def send(self, message):
        self._send_calls += 1
        if self.fail_on_send_call and self._send_calls == self.fail_on_send_call:
            raise ConnectionResetError("socket closed by peer")
        self.sent.append(message)

    def close(self):
        self.closed = True


class RecordingConnector:
    """A connector factory that hands out FakeConnections and counts connect attempts."""

    def __init__(self, fail_on_send_call: int = 0):
        self.connections = []
        self.urls = []
        self.fail_on_send_call = fail_on_send_call

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        conn = FakeConnection(fail_on_send_call=self.fail_on_send_call)
        # Only the first connection is made to fail, so a reconnect can succeed.
        self.fail_on_send_call = 0
        self.connections.append(conn)
        return conn


def leg(target_language="ja", recipients=("peer-b",)) -> LegConfig:
    """Builds a single A->B translation leg configuration."""
    return LegConfig(
        leg_id="leg-a-to-b",
        speaker_id="peer-a",
        target_language=target_language,
        recipients=tuple(recipients),
    )


class TestLanguageCodes(unittest.TestCase):
    """BCP-47 resolution for the three supported languages (REQ-17)."""

    def test_supported_languages_map_to_bcp47(self):
        self.assertEqual(to_bcp47("ja"), "ja-JP")
        self.assertEqual(to_bcp47("hi"), "hi-IN")
        self.assertTrue(to_bcp47("en").startswith("en-"))

    def test_already_qualified_codes_pass_through(self):
        """A caller that already holds a region-qualified tag must not have it rewritten."""
        self.assertEqual(to_bcp47("en-IN"), "en-IN")
        self.assertEqual(to_bcp47("ja-JP"), "ja-JP")

    def test_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            to_bcp47("klingon")


class TestModelResolution(unittest.TestCase):
    """The Live Translate model is environment-configured, never frozen in code."""

    def test_default_model_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_LIVE_TRANSLATE_MODEL", None)
            self.assertEqual(resolve_model(), DEFAULT_LIVE_TRANSLATE_MODEL)

    def test_environment_overrides_the_default(self):
        with patch.dict(os.environ, {"GEMINI_LIVE_TRANSLATE_MODEL": "models/custom-live"}):
            self.assertEqual(resolve_model(), "models/custom-live")


class TestSetupMessage(unittest.TestCase):
    """The Live session setup contract (REQ-17, spec 3.1)."""

    def setUp(self):
        self.session = GeminiLiveTranslateSession(
            config=leg(), api_key="test-key", connector=RecordingConnector()
        )
        self.setup = self.session.build_setup_message()["setup"]
        self.generation_config = self.setup["generationConfig"]

    def test_response_is_audio_only(self):
        self.assertEqual(self.generation_config["responseModalities"], ["AUDIO"])

    def test_both_transcriptions_are_enabled(self):
        """Input and output transcription both feed REQ-17 transcript events."""
        self.assertIn("inputAudioTranscription", self.generation_config)
        self.assertIn("outputAudioTranscription", self.generation_config)

    def test_translation_config_is_nested_in_generation_config(self):
        """
        The Live Translate contract nests `translationConfig`, `inputAudioTranscription`
        and `outputAudioTranscription` inside `generationConfig`. At the top level of
        `setup` they are silently ignored, which produces the worst possible failure: a
        session that connects, streams, and answers in the wrong language with no error.
        """
        self.assertNotIn("translationConfig", self.setup)
        self.assertNotIn("inputAudioTranscription", self.setup)
        self.assertNotIn("outputAudioTranscription", self.setup)
        self.assertIn("translationConfig", self.generation_config)

    def test_target_language_is_bcp47(self):
        self.assertEqual(
            self.generation_config["translationConfig"]["targetLanguageCode"], "ja-JP"
        )

    def test_echo_target_language_defaults_off(self):
        """
        Default off so target-language speech is not echoed back into the translated
        track - the case that matters most for the international_work English pivot.
        """
        self.assertIs(
            self.generation_config["translationConfig"]["echoTargetLanguage"], False
        )

    def test_default_model_is_a_live_translate_model(self):
        """
        Live Translate is a distinct model family. A general native-audio Live model
        connects and talks, but does not interpret, so the wrong default here is another
        failure that looks like success.
        """
        self.assertIn("translate", DEFAULT_LIVE_TRANSLATE_MODEL)

    def test_no_source_language_is_configured(self):
        """
        Source language is auto-detected per utterance. Any fixed source key would break
        a code-switching speaker the moment they change language mid-turn.
        """
        blob = json.dumps(self.setup).lower()
        for forbidden in ("sourcelanguage", "source_language", "sourcelanguagecode"):
            self.assertNotIn(forbidden, blob)

    def test_setup_carries_the_resolved_model(self):
        self.assertIn(resolve_model().split("/")[-1], self.setup["model"])


class TestSessionLifecycle(unittest.TestCase):
    """Connect / send / reconnect / close (REQ-17)."""

    def test_connect_sends_setup_first(self):
        connector = RecordingConnector()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k", connector=connector)

        session.connect()

        self.assertEqual(session.state, LegState.ACTIVE)
        first = json.loads(connector.connections[0].sent[0])
        self.assertIn("setup", first)

    def test_api_key_never_appears_in_a_client_visible_field(self):
        """GEMINI_API_KEY is server-only (REQ-08/REQ-17): it must not leak into config."""
        session = GeminiLiveTranslateSession(
            config=leg(), api_key="super-secret", connector=RecordingConnector()
        )
        self.assertNotIn("super-secret", json.dumps(session.build_setup_message()))

    def test_send_audio_transmits_base64_pcm_at_16k(self):
        connector = RecordingConnector()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k", connector=connector)
        session.connect()

        chunk = silence(INPUT_SAMPLE_RATE // 10)
        self.assertTrue(session.send_audio(chunk))

        payload = json.loads(connector.connections[0].sent[-1])
        blob = payload["realtimeInput"]["audio"]
        self.assertEqual(blob["mimeType"], "audio/pcm;rate=%d" % INPUT_SAMPLE_RATE)
        self.assertEqual(base64.b64decode(blob["data"]), chunk)

    def test_send_audio_reconnects_once_after_a_dropped_socket(self):
        """A dropped socket degrades this leg only, and recovers without a restart."""
        connector = RecordingConnector(fail_on_send_call=2)  # 1 == setup, 2 == first audio
        session = GeminiLiveTranslateSession(config=leg(), api_key="k", connector=connector)
        session.connect()

        self.assertTrue(session.send_audio(silence(1600)))

        self.assertEqual(len(connector.connections), 2)
        self.assertEqual(session.state, LegState.ACTIVE)
        self.assertEqual(session.reconnect_count, 1)

    def test_unreachable_endpoint_fails_closed(self):
        """
        REQ-17: an unreachable Live endpoint must not block session start. The session
        object reports `unavailable` with a reason instead of propagating a raw error.
        """
        def broken_connector(url, **kwargs):
            raise OSError("name resolution failed")

        session = GeminiLiveTranslateSession(
            config=leg(), api_key="k", connector=broken_connector
        )

        with self.assertRaises(TranslationUnavailableError):
            session.connect()

        self.assertEqual(session.state, LegState.UNAVAILABLE)
        self.assertIn("name resolution failed", session.unavailable_reason)

    def test_missing_api_key_is_unavailable_not_a_crash(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            session = GeminiLiveTranslateSession(config=leg(), api_key=None)
            with self.assertRaises(TranslationUnavailableError):
                session.connect()
            self.assertEqual(session.state, LegState.UNAVAILABLE)

    def test_close_is_explicit_and_idempotent(self):
        connector = RecordingConnector()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k", connector=connector)
        session.connect()

        session.close()
        session.close()

        self.assertTrue(connector.connections[0].closed)
        self.assertEqual(session.state, LegState.CLOSED)

    def test_sending_after_close_is_refused(self):
        connector = RecordingConnector()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k", connector=connector)
        session.connect()
        session.close()

        self.assertFalse(session.send_audio(silence(1600)))


class TestServerMessageParsing(unittest.TestCase):
    """Normalizing Live server messages into the three things the router cares about."""

    def test_input_transcription_is_normalized_and_final(self):
        """
        `inputTranscription` IS the finalized transcript - the Live contract has no
        `finished` flag, and finality is carried by which field arrives. Treating it as
        interim would leave `is_final` permanently false, and the once-only ingestion
        into `TeachingAgent` and the artifacts would then never fire at all.
        """
        raw = json.dumps({"serverContent": {"inputTranscription": {"text": "ohayou"}}})
        events = parse_server_message(raw)
        self.assertEqual(events[0]["type"], "input_transcript")
        self.assertEqual(events[0]["text"], "ohayou")
        self.assertTrue(events[0]["is_final"])

    def test_interim_input_transcription_is_not_final(self):
        """`interimInputTranscription` updates while the speaker is still talking."""
        raw = json.dumps({
            "serverContent": {"interimInputTranscription": {"text": "ohay"}}
        })
        events = parse_server_message(raw)
        self.assertEqual(events[0]["type"], "input_transcript")
        self.assertFalse(events[0]["is_final"])

    def test_transcription_language_code_is_carried_through(self):
        """
        Gemini reports the language it actually detected. That is the only true source
        language on a leg with no configured source, so it must not be dropped in favour
        of the language we merely expected this speaker to use.
        """
        raw = json.dumps({
            "serverContent": {
                "inputTranscription": {"text": "namaste", "languageCode": "hi-IN"}
            }
        })
        self.assertEqual(parse_server_message(raw)[0]["language_code"], "hi-IN")

    def test_output_transcription_is_normalized(self):
        raw = json.dumps({"serverContent": {"outputTranscription": {"text": "good morning"}}})
        events = parse_server_message(raw)
        self.assertEqual(events[0]["type"], "output_transcript")
        self.assertEqual(events[0]["text"], "good morning")

    def test_audio_parts_are_decoded_to_pcm_bytes(self):
        pcm = silence(240, value=99)
        raw = json.dumps({
            "serverContent": {
                "modelTurn": {"parts": [{
                    "inlineData": {
                        "mimeType": "audio/pcm;rate=%d" % GEMINI_OUTPUT_SAMPLE_RATE,
                        "data": base64.b64encode(pcm).decode("ascii"),
                    }
                }]}
            }
        })
        events = parse_server_message(raw)
        self.assertEqual(events[0]["type"], "audio")
        self.assertEqual(events[0]["audio"], pcm)

    def test_turn_complete_finalizes_the_transcript(self):
        raw = json.dumps({"serverContent": {"turnComplete": True}})
        events = parse_server_message(raw)
        self.assertTrue(events[0]["is_final"])

    def test_unparseable_message_yields_no_events_rather_than_raising(self):
        self.assertEqual(parse_server_message("not json at all"), [])


class TestAudioAdapter(unittest.TestCase):
    """TASK-11.2: Agora frame -> Gemini PCM, and Gemini PCM -> Agora publish frames."""

    def test_input_chunk_size_is_100ms_of_16k_mono(self):
        self.assertEqual(INPUT_CHUNK_MS, 100)
        self.assertEqual(INPUT_CHUNK_BYTES, INPUT_SAMPLE_RATE * 2 * INPUT_CHUNK_MS // 1000)

    def test_downmix_averages_stereo_to_mono(self):
        stereo = b"".join([
            (100).to_bytes(2, "little", signed=True),
            (300).to_bytes(2, "little", signed=True),
        ])
        mono = downmix_to_mono(stereo, channels=2)
        self.assertEqual(len(mono), 2)
        self.assertEqual(int.from_bytes(mono, "little", signed=True), 200)

    def test_downmix_of_mono_is_a_passthrough(self):
        mono = silence(160, value=42)
        self.assertEqual(downmix_to_mono(mono, channels=1), mono)

    def test_resample_changes_duration_proportionally(self):
        one_second_48k = silence(48000, value=1000)
        resampled = resample_pcm16(one_second_48k, 48000, INPUT_SAMPLE_RATE)
        self.assertEqual(len(resampled), INPUT_SAMPLE_RATE * 2)

    def test_resample_is_a_passthrough_at_matching_rates(self):
        pcm = silence(160, value=7)
        self.assertEqual(resample_pcm16(pcm, 16000, 16000), pcm)

    def test_encoder_emits_only_whole_100ms_chunks(self):
        encoder = GeminiInputEncoder(source_rate=48000, channels=1)

        # 50 ms of 48 kHz audio is half a chunk: nothing may be emitted yet.
        self.assertEqual(encoder.push(silence(2400)), [])
        chunks = encoder.push(silence(2400))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), INPUT_CHUNK_BYTES)

    def test_encoder_downmixes_and_resamples_stereo_input(self):
        encoder = GeminiInputEncoder(source_rate=48000, channels=2)
        chunks = encoder.push(silence(4800, channels=2))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), INPUT_CHUNK_BYTES)

    def test_encoder_flush_returns_the_partial_remainder(self):
        encoder = GeminiInputEncoder(source_rate=16000, channels=1)
        encoder.push(silence(800))  # 50 ms
        remainder = encoder.flush()
        self.assertEqual(len(remainder), 1600)
        self.assertEqual(encoder.flush(), b"")

    def test_publish_adapter_converts_24k_to_agora_frames(self):
        adapter = AgoraPublishAdapter(publish_rate=48000, frame_ms=20)
        frames = adapter.push(silence(GEMINI_OUTPUT_SAMPLE_RATE // 10))  # 100 ms @ 24k

        self.assertEqual(len(frames), 5)  # 100 ms / 20 ms
        for frame in frames:
            self.assertEqual(len(frame), 48000 * 2 * 20 // 1000)

    def test_publish_adapter_buffers_across_calls(self):
        adapter = AgoraPublishAdapter(publish_rate=24000, frame_ms=20)
        self.assertEqual(adapter.push(silence(240)), [])  # 10 ms
        self.assertEqual(len(adapter.push(silence(240))), 1)


if __name__ == "__main__":
    unittest.main()
