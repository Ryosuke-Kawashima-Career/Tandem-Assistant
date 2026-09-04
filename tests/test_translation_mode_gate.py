"""
Summary:
    test_translation_mode_gate.py is the executable specification for TASK-11.5 and
    TASK-11.6: the per-participant translated-audio gate, and the fail-closed contract
    when Gemini Live Translate is unreachable at session start.

    The gate exists because the two modes want opposite defaults. A work call wants the
    translation on immediately - that is the point of the call. A tandem lesson does not:
    handing a learner a fluent translation of their partner is exactly the crutch the
    lesson is trying to remove, so learners opt in. Either default can be flipped from
    the REQ-06 direct-AI controls.

    Transcripts are deliberately outside the gate: they still feed `TeachingAgent` and
    artifacts, and they remain available as an on-demand subtitle, so muting the
    translated voice must not also silence the pipeline behind it.

Covers:
    - REQ-17 per-mode default gate state and per-participant override.
    - REQ-17 gate governs audio publication only, never transcript publication.
    - REQ-17 session start proceeds when Live Translate is unavailable.
"""

import unittest
from unittest.mock import MagicMock

from src.rtc.data_stream import DataStreamManager
from src.sessions.models import SessionMode, SessionRecord
from src.translation.gemini_live import LegState, TranslationUnavailableError
from src.translation.router import (
    EVENT_INPUT_TRANSCRIPT,
    EVENT_STATUS,
    Participant,
    TranslationRouter,
)

PEER_A = Participant(participant_id="peer-a", language="hi")
PEER_B = Participant(participant_id="peer-b", language="ja")


class StubLeg:
    """A leg that connects successfully and swallows audio."""

    def __init__(self, config, **kwargs):
        self.config = config
        self.state = LegState.IDLE
        self.closed = False

    def connect(self):
        self.state = LegState.ACTIVE
        return True

    def send_audio(self, chunk):
        return True

    def close(self):
        self.closed = True
        self.state = LegState.CLOSED


def router_for(mode, data_stream=None, factory=StubLeg, sink=None):
    """Builds a started router in the given mode with both peers present."""
    session = SessionRecord.create(channel="c1", mode=mode, languages=["hi", "ja"])
    router = TranslationRouter(
        session=session,
        data_stream=data_stream,
        session_factory=factory,
        transcript_sink=sink,
    )
    router.start([PEER_A, PEER_B])
    return router


class TestModeDefaults(unittest.TestCase):
    """TASK-11.5: the mode picks the default, not the client (REQ-17)."""

    def test_mode_owns_the_default_so_callers_do_not_re_derive_it(self):
        self.assertTrue(SessionMode.INTERNATIONAL_WORK.translated_audio_default)
        self.assertFalse(SessionMode.LANGUAGE_LEARNING.translated_audio_default)

    def test_work_mode_defaults_the_gate_on(self):
        router = router_for("international_work")
        self.assertTrue(router.translated_audio_enabled("peer-a"))
        self.assertTrue(router.translated_audio_enabled("peer-b"))

    def test_learning_mode_defaults_the_gate_off(self):
        """A learner is not passively handed a translation of their partner."""
        router = router_for("language_learning")
        self.assertFalse(router.translated_audio_enabled("peer-a"))
        self.assertFalse(router.translated_audio_enabled("peer-b"))

    def test_an_unknown_participant_reports_the_mode_default(self):
        self.assertTrue(router_for("international_work").translated_audio_enabled("ghost"))


class TestGateToggle(unittest.TestCase):
    """TASK-11.5: the toggle is per participant and works in both directions."""

    def test_a_learner_can_opt_in(self):
        router = router_for("language_learning")

        router.set_translated_audio_enabled("peer-a", True)

        self.assertTrue(router.translated_audio_enabled("peer-a"))
        self.assertFalse(router.translated_audio_enabled("peer-b"))

    def test_a_work_participant_can_opt_out(self):
        router = router_for("international_work")

        router.set_translated_audio_enabled("peer-b", False)

        self.assertFalse(router.translated_audio_enabled("peer-b"))
        self.assertTrue(router.translated_audio_enabled("peer-a"))

    def test_toggling_emits_a_status_event_so_every_client_agrees(self):
        data_stream = MagicMock(spec=DataStreamManager)
        router = router_for("language_learning", data_stream=data_stream)
        data_stream.reset_mock()

        router.set_translated_audio_enabled("peer-a", True)

        payloads = [c.args[1] for c in data_stream.send_translation_event.call_args_list
                    if c.args[0] == EVENT_STATUS]
        self.assertTrue(any(p.get("participant_id") == "peer-a"
                            and p.get("translated_audio_enabled") is True
                            for p in payloads))


class TestGateGovernsAudioOnly(unittest.TestCase):
    """TASK-11.5: the gate mutes the voice, never the pipeline behind it."""

    def setUp(self):
        self.publisher = MagicMock()
        self.sink = MagicMock()
        self.data_stream = MagicMock(spec=DataStreamManager)

    def build(self, mode):
        router = router_for(mode, data_stream=self.data_stream, sink=self.sink)
        router.audio_publisher = self.publisher
        self.data_stream.reset_mock()
        return router

    def test_gate_off_suppresses_translated_audio_publication(self):
        router = self.build("language_learning")
        leg_a = router.leg_for_speaker("peer-a")

        recipients = router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)

        self.assertEqual(recipients, [])
        self.publisher.assert_not_called()

    def test_gate_on_publishes_translated_audio(self):
        router = self.build("international_work")
        leg_a = router.leg_for_speaker("peer-a")

        recipients = router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)

        self.assertEqual(recipients, ["peer-b"])
        self.publisher.assert_called()

    def test_a_mixed_gate_publishes_only_to_the_recipients_who_opted_in(self):
        session = SessionRecord.create(channel="c1", mode="international_work",
                                       languages=["hi", "ja", "en"])
        router = TranslationRouter(session=session, session_factory=StubLeg)
        router.audio_publisher = self.publisher
        router.start([PEER_A, PEER_B, Participant("peer-c", "en")])
        router.set_translated_audio_enabled("peer-b", False)

        leg_a = router.leg_for_speaker("peer-a")
        recipients = router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)

        self.assertEqual(recipients, ["peer-c"])

    def test_transcripts_publish_with_the_gate_off(self):
        router = self.build("language_learning")
        leg_a = router.leg_for_speaker("peer-a")

        router.handle_input_transcript(leg_a.leg_id, "namaste", is_final=True)

        published = [c.args[0] for c in self.data_stream.send_translation_event.call_args_list]
        self.assertIn(EVENT_INPUT_TRANSCRIPT, published)

    def test_finalized_transcripts_still_reach_the_agent_with_the_gate_off(self):
        router = self.build("language_learning")
        leg_a = router.leg_for_speaker("peer-a")

        router.handle_input_transcript(leg_a.leg_id, "namaste", is_final=True)

        self.sink.assert_called_once()

    def test_the_gate_does_not_change_the_leg_topology(self):
        """Turning audio off must not tear a leg down: the transcripts still come from it."""
        router = self.build("international_work")
        before = router.leg_states()

        router.set_translated_audio_enabled("peer-a", False)
        router.set_translated_audio_enabled("peer-b", False)

        self.assertEqual(router.leg_states(), before)


class TestUnavailableFailsClosed(unittest.TestCase):
    """TASK-11.6: Live Translate being down must not block session start (REQ-17)."""

    def unavailable_factory(self, reason="401 API key rejected"):
        def factory(config, **kwargs):
            leg = StubLeg(config)

            def connect():
                leg.state = LegState.UNAVAILABLE
                raise TranslationUnavailableError(reason)

            leg.connect = connect
            return leg
        return factory

    def test_start_returns_normally_when_every_leg_is_unavailable(self):
        router = TranslationRouter(
            session=SessionRecord.create(channel="c1", mode="international_work",
                                         languages=["hi", "ja"]),
            session_factory=self.unavailable_factory(),
        )

        states = router.start([PEER_A, PEER_B])

        self.assertEqual(set(states.values()), {LegState.UNAVAILABLE.value})
        self.assertFalse(router.is_available)

    def test_each_unavailable_leg_reports_its_reason(self):
        data_stream = MagicMock(spec=DataStreamManager)
        router = TranslationRouter(
            session=SessionRecord.create(channel="c1", mode="international_work",
                                         languages=["hi", "ja"]),
            data_stream=data_stream,
            session_factory=self.unavailable_factory("quota exceeded"),
        )

        router.start([PEER_A, PEER_B])

        payloads = [c.args[1] for c in data_stream.send_translation_event.call_args_list]
        self.assertEqual(len(payloads), 2)
        for payload in payloads:
            self.assertEqual(payload["state"], "unavailable")
            self.assertIn("quota exceeded", payload["reason"])

    def test_an_unavailable_session_still_accepts_audio_without_raising(self):
        """
        Original Agora audio keeps flowing; the router simply has nowhere to send a copy.
        A raise here would take the voice call down with the translator.
        """
        router = TranslationRouter(
            session=SessionRecord.create(channel="c1", mode="international_work",
                                         languages=["hi", "ja"]),
            session_factory=self.unavailable_factory(),
        )
        router.start([PEER_A, PEER_B])

        self.assertEqual(router.route_audio("peer-a", b"\x00\x00" * 1600,
                                            sample_rate=16000), 0)

    def test_an_unavailable_session_publishes_no_translated_audio(self):
        publisher = MagicMock()
        router = TranslationRouter(
            session=SessionRecord.create(channel="c1", mode="international_work",
                                         languages=["hi", "ja"]),
            session_factory=self.unavailable_factory(),
        )
        router.audio_publisher = publisher
        router.start([PEER_A, PEER_B])
        leg_a = router.leg_for_speaker("peer-a")

        self.assertEqual(router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400), [])
        publisher.assert_not_called()

    def test_a_partially_available_session_reports_itself_available(self):
        calls = {"n": 0}

        def factory(config, **kwargs):
            leg = StubLeg(config)
            calls["n"] += 1
            if calls["n"] == 1:
                def connect():
                    leg.state = LegState.UNAVAILABLE
                    raise TranslationUnavailableError("transient")
                leg.connect = connect
            return leg

        router = TranslationRouter(
            session=SessionRecord.create(channel="c1", mode="international_work",
                                         languages=["hi", "ja"]),
            session_factory=factory,
        )
        router.start([PEER_A, PEER_B])

        self.assertTrue(router.is_available)


class TestTranslationApi(unittest.TestCase):
    """
    TASK-11.5: the REQ-06 controls reach the gate over HTTP.

    The toggle is a per-participant server-side decision, not a client-side mute: the
    router decides who is published to, so a browser that simply stopped playing the
    track would still be paying for its bandwidth and would disagree with every other
    client about whether translation is on.
    """

    def setUp(self):
        from src.server import app, server_instance

        self.client = app.test_client()
        self.server = server_instance
        self.channel = "translation-api-test"
        self.server.sessions.create_session(channel=self.channel, mode="language_learning",
                                            languages=["hi", "ja"])
        self.addCleanup(self.server.sessions.end_session, self.channel)
        self.addCleanup(self.server.stop_translation, self.channel)

    def start(self, participants=None):
        return self.client.post("/api/translation/start", json={
            "channel": self.channel,
            "participants": participants or [
                {"participant_id": "peer-a", "language": "hi"},
                {"participant_id": "peer-b", "language": "ja"},
            ],
        })

    def test_start_reports_the_planned_legs_and_their_states(self):
        response = self.start()

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(len(body["legs"]), 2)

    def test_start_without_a_session_is_a_404(self):
        response = self.client.post("/api/translation/start", json={
            "channel": "no-such-channel", "participants": []
        })
        self.assertEqual(response.status_code, 404)

    def test_status_reports_the_mode_default_gate(self):
        self.start()

        body = self.client.get(f"/api/translation/status?channel={self.channel}").get_json()

        self.assertEqual(body["mode"], "language_learning")
        self.assertFalse(body["translated_audio"]["peer-a"])

    def test_the_toggle_flips_one_participants_gate(self):
        self.start()

        response = self.client.post("/api/translation/audio", json={
            "channel": self.channel, "participant_id": "peer-a", "enabled": True
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["translated_audio_enabled"])

        body = self.client.get(f"/api/translation/status?channel={self.channel}").get_json()
        self.assertTrue(body["translated_audio"]["peer-a"])
        self.assertFalse(body["translated_audio"]["peer-b"])

    def test_the_toggle_requires_a_participant(self):
        self.start()
        response = self.client.post("/api/translation/audio", json={
            "channel": self.channel, "enabled": True
        })
        self.assertEqual(response.status_code, 400)

    def test_stop_closes_the_legs(self):
        self.start()

        response = self.client.post("/api/translation/stop", json={"channel": self.channel})

        self.assertEqual(response.status_code, 200)
        body = self.client.get(f"/api/translation/status?channel={self.channel}").get_json()
        self.assertEqual(body["legs"], {})

    def test_an_unavailable_translator_still_returns_200(self):
        """
        TASK-11.6 at the API boundary: no Gemini key configured is the ordinary local
        case, and it must read as a session without translated audio, not as a failure.
        """
        response = self.start()

        self.assertEqual(response.status_code, 200)
        self.assertIn("available", response.get_json())


if __name__ == "__main__":
    unittest.main()
