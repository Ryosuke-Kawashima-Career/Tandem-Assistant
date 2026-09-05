"""
Summary:
    test_agent.py provides unit and integration tests for EchoSphere's TeachingAgent
    orchestrator, prompt generators, and tri-lingual conversational mediation pipeline.
    It verifies prompt formatting, multi-speaker turn processing, code-switching analysis,
    cultural idiom annotations, silence breaker generation, and speaking balance tracking.
Example:
uv run python -c "
from src.audio.vad_processor import VoiceActivityDetector
from src.audio.stt_transcriber import STTTranscriber
from src.agent.orchestrator import TeachingAgent
from src.audio.tts_synthesizer import TTSSynthesizer
from src.rtc.agora_client import AgoraVoiceChannelClient
from tests.test_audio import generate_sine_wave_pcm, generate_silence_pcm

# 1. Initialize Pipeline
rtc = AgoraVoiceChannelClient(channel_name='tandem-room-01')
agent = TeachingAgent(engine='mock')
stt = STTTranscriber(engine='mock')
tts = TTSSynthesizer()
rtc.join_channel()

# 2. Register RTC Data Stream Listener
rtc.register_on_data_stream(lambda pkt: print(f'\n📡 [RTC Stream Event Received: {pkt.event_type}]\nPayload: {pkt.payload}'))

# 3. Simulate Student Speech ('一期一会')
audio_in = generate_sine_wave_pcm(duration_ms=300)
transcription = stt.transcribe(audio_in, language='ja', speaker_id='Kenji')
print(f'🎙️ Transcribed: [{transcription.speaker_id} ({transcription.language})]: {transcription.text}')

# 4. Agent Mediates & Broadcasts Live Subtitles + Idiom Card
result = agent.process_turn(transcription.speaker_id, transcription.text, transcription.language)
rtc.send_data_stream_message('subtitles', result['subtitles'])
rtc.send_data_stream_message('idiom_card', result['idiom_card'])

# 5. Synthesize AI Voice Response
spoken_audio = tts.synthesize(result['spoken_response'], language=result['spoken_language'])
rtc.publish_audio_frame(spoken_audio)
print(f'🔊 AI Spoken Response Synthesized: {len(spoken_audio)} bytes PCM')
"

"""

import os
import threading
import unittest
import json
from unittest.mock import patch, MagicMock
from src.agent.orchestrator import (
    TeachingAgent,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    looks_like_camera_question,
)
from src.agent.tools.camera_stream import CameraFrameBuffer
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.vision import CameraVisionTool
from src.sessions.models import SessionRecord
from src.agent.prompts import (
    create_teaching_prompt,
    create_tutor_prompt,
    create_tutor_voice_prompt,
    create_silence_breaker_prompt,
    SYSTEM_PROMPT_TANDEM_TEACHER,
    SYSTEM_PROMPT_TANDEM_TUTOR,
    SYSTEM_PROMPT_TANDEM_TUTOR_VOICE
)
from src.audio.tts_synthesizer import TTSSynthesizer


class TestAgent(unittest.TestCase):
    """
    Test suite verifying conversational AI co-teacher behavior, prompt composition,
    and tri-lingual peer mediation capabilities.
    """

    def setUp(self):
        """Initialize the TeachingAgent instance and test fixtures."""
        self.agent = TeachingAgent(
            engine="mock",
            target_language="Japanese",
            native_language="English"
        )
        self.tts = TTSSynthesizer(sample_rate=16000)

    def test_teaching_prompt(self):
        """
        Verify that create_teaching_prompt builds a well-structured prompt containing
        dialogue context, speaker statistics, and language pairing.
        
        Algorithm:
        1. Invoke create_teaching_prompt with sample history context and speaker percentages.
        2. Assert that target and native languages are present in the output string.
        3. Assert that speaker balance metrics and dialogue context are incorporated.
        """
        context = "[Kenji (ja)]: 一期一会ですね\n[Aarav (en)]: Yes, every meeting is special!"
        speaker_stats = {"Kenji": 60, "Aarav": 40}
        
        prompt = create_teaching_prompt(
            recent_context=context,
            speaker_stats=speaker_stats,
            target_language="Japanese",
            native_language="English",
            topic="Traditional Japanese Concepts"
        )

        self.assertIsInstance(prompt, str)
        self.assertIn("Target Language: Japanese", prompt)
        self.assertIn("Native Language: English", prompt)
        self.assertIn("Kenji: 60%", prompt)
        self.assertIn("Aarav: 40%", prompt)
        self.assertIn("一期一会ですね", prompt)
        self.assertIn("Traditional Japanese Concepts", prompt)

    def test_processing(self):
        """
        Verify that TeachingAgent.process_turn handles student conversational input,
        records dialogue history, and yields structured pedagogical feedback.
        
        Algorithm:
        1. Process a turn from a student speaking Japanese ('一期一会').
        2. Verify that the turn history is recorded in the agent.
        3. Verify the output dictionary contains all required contract keys:
           spoken_response, subtitles, idiom_card, quiz, teacher_alert.
        4. Validate that subtitles include Romaji transliteration and translations.
        5. Validate that the idiom card is detected and contains cultural notes.
        """
        speaker_id = "Kenji"
        text = "一期一会ですね！"
        detected_language = "ja"

        # Step 1: Process conversational turn
        response = self.agent.process_turn(
            speaker_id=speaker_id,
            text=text,
            detected_language=detected_language,
            topic="Philosophy & Idioms"
        )

        # Step 2: Verify history retention
        self.assertEqual(len(self.agent.turn_history), 1)
        self.assertEqual(self.agent.turn_history[0]["speaker"], "Kenji")
        self.assertEqual(self.agent.turn_history[0]["text"], "一期一会ですね！")

        # Step 3: Verify response contract
        self.assertIn("spoken_response", response)
        self.assertIn("subtitles", response)
        self.assertIn("idiom_card", response)
        self.assertIn("quiz", response)
        self.assertIn("teacher_alert", response)

        # Step 4: Verify subtitles & transliteration
        subtitles = response["subtitles"]
        self.assertEqual(subtitles["speaker"], "Kenji")
        self.assertIn("Ichigo ichie", subtitles["transliteration"])
        self.assertIn("Treasure every encounter", subtitles["translation_en"])
        self.assertIn("हर मुलाकात अनमोल", subtitles["translation_hi"])

        # Step 5: Verify idiom card
        idiom_card = response["idiom_card"]
        self.assertTrue(idiom_card["detected"])
        self.assertIn("一期一会", idiom_card["phrase"])
        self.assertIn("tea ceremony", idiom_card["cultural_note"].lower())

    def test_hindi_turn_processing(self):
        """
        Verify that TeachingAgent processes Hindi turns with Devanagari transliteration and cultural context.
        """
        speaker_id = "Aarav"
        text = "नमस्ते, आप कैसे हैं?"
        response = self.agent.process_turn(
            speaker_id=speaker_id,
            text=text,
            detected_language="hi"
        )

        self.assertIn("spoken_response", response)
        self.assertEqual(response["spoken_language"], "hi")
        
        # Verify Hindi subtitles and transliteration
        subtitles = response["subtitles"]
        self.assertIn("Namaste, aap kaise hain?", subtitles["transliteration"])
        self.assertIn("Hello, how are you?", subtitles["translation_en"])
        
        # Verify Hindi idiom & register quiz
        quiz = response["quiz"]
        self.assertTrue(quiz["active"])
        self.assertIn("आप (Aap)", quiz["options"])

    def test_silence_breaker(self):
        """
        Verify that generate_silence_breaker produces engaging discussion prompts
        when a conversation stall or awkward silence is detected.
        
        Algorithm:
        1. Invoke agent.generate_silence_breaker for an inactive learner.
        2. Verify that spoken intervention is generated and tailored to the topic.
        3. Verify teacher_alert is flagged to notify human instructor dashboard.
        """
        response = self.agent.generate_silence_breaker(
            topic="Festival Celebrations in Japan and India",
            inactive_speaker="Aarav"
        )

        self.assertIn("spoken_response", response)
        self.assertIn("Festival Celebrations", response["spoken_response"])
        self.assertIn("Aarav", response["spoken_response"])
        
        # Verify teacher alert is registered
        self.assertTrue(response["teacher_alert"]["alert_required"])
        self.assertIn("Silence breaker triggered", response["teacher_alert"]["message"])

    def test_speaker_balance_tracking(self):
        """
        Verify that update_speaker_time accurately computes peer speaking time percentages.
        """
        self.agent.update_speaker_time("Kenji", 60000)  # 60s (75%)
        self.agent.update_speaker_time("Aarav", 20000)  # 20s (25%)

        balance = self.agent.get_speaker_balance_percentages()
        self.assertEqual(balance["Kenji"], 75)
        self.assertEqual(balance["Aarav"], 25)

    def test_tutor_prompt_addresses_exactly_one_learner(self):
        """
        Verify the 1:1 tutor prompt carries no peer-mediation framing (REQ-LLM-03).

        Applied to a Convo AI session, the mediation prompt makes a real model faithfully
        address a second learner who is not in the channel - the same defect visible in
        the mediation mock's "What do both of you think about this topic?".
        """
        prompt = create_tutor_prompt(
            recent_context="[Kenji (ja)]: 一期一会ですね",
            target_language="Japanese",
            native_language="English",
            learner_name="Kenji",
            topic="Traditional Japanese Concepts"
        )
        combined = (prompt + SYSTEM_PROMPT_TANDEM_TUTOR).lower()

        for banned in [
            "both of you",
            "both learners",
            "speaker balance",
            "speaking balance",
            "speaker stats",
            "the other learner",
            "two learners",
            "peer",
        ]:
            self.assertNotIn(banned, combined, f"tutor prompt must not mention {banned!r}")

        # Still carries the context a 1:1 reply needs
        self.assertIn("Target Language: Japanese", prompt)
        self.assertIn("Kenji", prompt)
        self.assertIn("一期一会ですね", prompt)

    def test_mediation_prompt_is_unchanged_by_tutor_mode(self):
        """
        Verify the ambient peer-mediation prompt still carries speaker-balance framing,
        so REQ-04's pipeline is untouched by the Convo AI work.
        """
        self.assertIn("MULTI-SPEAKER BALANCE", SYSTEM_PROMPT_TANDEM_TEACHER)
        prompt = create_teaching_prompt(
            recent_context="[Kenji (ja)]: はい",
            speaker_stats={"Kenji": 70, "Aarav": 30}
        )
        self.assertIn("Speaker Balance Metrics", prompt)

    def test_tutor_mode_reply_does_not_address_a_second_learner(self):
        """
        Verify mode="tutor" produces a 1:1 offline reply, so the demo path stops saying
        "What do both of you think about this topic?" into a private conversation.
        """
        response = self.agent.process_turn(
            speaker_id="Kenji",
            text="I visited Kyoto last spring.",
            detected_language="en",
            mode="tutor"
        )

        spoken = response["spoken_response"].lower()
        self.assertTrue(spoken)
        self.assertNotIn("both of you", spoken)
        self.assertNotIn("both", spoken)

        # Same JSON contract as mediation mode, so downstream parsing is unchanged
        for key in ["spoken_response", "spoken_language", "subtitles", "idiom_card", "quiz", "teacher_alert"]:
            self.assertIn(key, response)

    def test_mediation_mode_is_the_default(self):
        """
        Verify process_turn still defaults to mediation, so every existing ambient-pipeline
        caller keeps its previous behaviour without passing `mode`.
        """
        response = self.agent.process_turn(
            speaker_id="Kenji",
            text="A regular English sentence.",
            detected_language="en"
        )
        self.assertIn("both of you", response["spoken_response"].lower())

    def test_reset_state_clears_conversation_history(self):
        """
        Verify reset_state clears per-session buffers (REQ-LLM-05) without rebuilding the
        agent, so provider SDK clients are not re-created per session.
        """
        self.agent.process_turn(speaker_id="Kenji", text="Hello", detected_language="en")
        self.agent.update_speaker_time("Kenji", 5000)
        self.assertTrue(self.agent.turn_history)

        self.agent.reset_state()

        self.assertEqual(self.agent.turn_history, [])
        self.assertEqual(self.agent.speaker_durations_ms, {})

    def test_unusable_engine_downgrades_to_mock_with_a_warning(self):
        """
        Verify a requested engine without credentials degrades to mock and says so
        (REQ-LLM-01), rather than silently looking identical to a hardcoded stub.
        """
        # Blank the key explicitly: a developer machine may hold a real GEMINI_API_KEY
        # in its shell environment, which would make this a live-credential test.
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertLogs("echosphere.agent.orchestrator", level="WARNING") as captured:
                agent = TeachingAgent(engine="gemini")

        self.assertEqual(agent.engine, "mock")
        self.assertTrue(
            any("Falling back to engine 'mock'" in line for line in captured.output),
            f"expected a downgrade warning, got: {captured.output}"
        )

    def test_tts_synthesizer_pcm_generation(self):
        """
        Verify that TTSSynthesizer produces valid non-empty 16kHz PCM audio
        for the AI co-teacher's spoken interventions.
        """
        speech_text = "素晴らしい会話ですね！ (Wonderful conversation!)"
        pcm_bytes = self.tts.synthesize(speech_text, language="ja")

        self.assertIsInstance(pcm_bytes, bytes)
        self.assertGreater(len(pcm_bytes), 0)


class TestLatencyFastPath(unittest.TestCase):
    """
    Test suite for the low-latency voice reply path (dev/tasks/task_specs/latency_improvement.md).

    Covers REQ-LAT-02 (voice reply decoupled from the structured scaffolding payload),
    REQ-LAT-03 (provider tokens forwarded as produced), and REQ-LAT-04 (model tier is
    environment-configurable).
    """

    def setUp(self):
        """Initialize a mock-engine agent for the fast-path tests."""
        self.agent = TeachingAgent(
            engine="mock",
            target_language="Japanese",
            native_language="English"
        )

    # -- REQ-LAT-02: decoupled fast-path reply -------------------------------------

    def test_voice_prompt_asks_for_plain_text_not_json(self):
        """
        Verify the voice-critical prompt does not request the structured JSON payload.

        This is what makes REQ-LAT-03 safe: a partial JSON object cannot be spoken by
        TTS mid-token, so the streamed call must produce bare reply text.
        """
        self.assertNotIn("JSON", SYSTEM_PROMPT_TANDEM_TUTOR_VOICE.upper())
        for field in ("subtitles", "idiom_card", "quiz", "teacher_alert"):
            self.assertNotIn(field, SYSTEM_PROMPT_TANDEM_TUTOR_VOICE)

    def test_voice_prompt_addresses_one_learner(self):
        """
        Verify the fast path keeps the 1:1 framing REQ-LLM-03 established, so shrinking
        the payload does not reintroduce peer-mediation language into a private call.
        """
        prompt = create_tutor_voice_prompt(
            recent_context="[Kenji (ja)]: こんにちは",
            latest_utterance="こんにちは",
            target_language="Japanese",
            native_language="English",
            learner_name="Kenji"
        )
        lowered = prompt.lower()
        self.assertIn("kenji", lowered)
        self.assertNotIn("both", lowered)
        self.assertNotIn("speaker balance", lowered)

    def test_generate_spoken_reply_returns_plain_text(self):
        """
        Verify the fast path yields speakable text, never a JSON document (REQ-LAT-02).
        """
        reply = "".join(
            self.agent.generate_spoken_reply(
                speaker_id="Kenji", text="Hello there", detected_language="en"
            )
        )

        self.assertTrue(reply.strip())
        self.assertFalse(reply.lstrip().startswith("{"))
        with self.assertRaises(json.JSONDecodeError):
            json.loads(reply)

    def test_generate_spoken_reply_yields_incremental_deltas(self):
        """
        Verify the fast path is a stream of deltas rather than one finished blob, so the
        bridge has something to forward before generation completes (REQ-LAT-03).
        """
        deltas = list(
            self.agent.generate_spoken_reply(
                speaker_id="Kenji", text="Tell me about festivals", detected_language="en"
            )
        )
        self.assertGreater(len(deltas), 1)

    def test_generate_spoken_reply_records_the_learner_turn(self):
        """
        Verify the fast path records conversation history itself.

        The scaffolding call is now asynchronous, so if it owned history the next turn's
        prompt could be built before the previous turn was recorded.
        """
        self.agent.generate_spoken_reply(
            speaker_id="Kenji", text="こんにちは", detected_language="ja"
        )
        self.assertEqual(len(self.agent.turn_history), 1)
        self.assertEqual(self.agent.turn_history[0]["text"], "こんにちは")

    def test_process_turn_can_skip_recording_history(self):
        """
        Verify process_turn(record_turn=False) leaves history untouched (REQ-LAT-02).

        The scaffolding call runs after the fast path already recorded the turn; without
        this flag the same utterance would appear twice in the model's context.
        """
        self.agent.generate_spoken_reply(
            speaker_id="Kenji", text="Hello", detected_language="en"
        )
        self.assertEqual(len(self.agent.turn_history), 1)

        self.agent.process_turn(
            speaker_id="Kenji", text="Hello", detected_language="en",
            mode="tutor", record_turn=False
        )
        self.assertEqual(len(self.agent.turn_history), 1)

    def test_process_turn_still_records_by_default(self):
        """Verify the mediation pipeline's history behaviour is unchanged (REQ-LAT-02 non-goal)."""
        self.agent.process_turn(speaker_id="Kenji", text="Hello", detected_language="en")
        self.assertEqual(len(self.agent.turn_history), 1)

    # -- REQ-LAT-03: true token-level streaming ------------------------------------

    def test_openai_fast_path_forwards_provider_deltas(self):
        """
        Verify OpenAI deltas are forwarded one by one rather than joined and re-split.
        """
        agent = TeachingAgent(engine="openai", openai_api_key="test-key")
        agent.engine = "openai"

        def fake_stream(*args, **kwargs):
            self.assertTrue(kwargs.get("stream"), "fast path must request a streaming response")
            for piece in ["Hello", " there", ", Kenji!"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=piece))]
                yield chunk

        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.side_effect = fake_stream

        deltas = list(
            agent.generate_spoken_reply(
                speaker_id="Kenji", text="Hi", detected_language="en"
            )
        )
        self.assertEqual(deltas, ["Hello", " there", ", Kenji!"])

    def test_gemini_fast_path_forwards_provider_deltas(self):
        """
        Verify Gemini deltas are forwarded incrementally through the streaming API.
        """
        agent = TeachingAgent(engine="gemini", gemini_api_key="test-key")
        agent.engine = "gemini"

        def fake_stream(*args, **kwargs):
            for piece in ["こんにちは", "、ケンジさん"]:
                chunk = MagicMock()
                chunk.text = piece
                yield chunk

        agent._gemini_client = MagicMock()
        agent._gemini_client.models.generate_content_stream.side_effect = fake_stream

        deltas = list(
            agent.generate_spoken_reply(
                speaker_id="Kenji", text="やあ", detected_language="ja"
            )
        )
        self.assertEqual(deltas, ["こんにちは", "、ケンジさん"])

    def test_fast_path_falls_back_to_mock_text_on_provider_error(self):
        """
        Verify a provider failure still yields speakable text.

        Silence is not neutral on this path: the Convo AI Engine terminates an agent on
        idle_timeout, so a raised exception mid-stream would end the conversation.
        """
        agent = TeachingAgent(engine="openai", openai_api_key="test-key")
        agent.engine = "openai"
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.side_effect = RuntimeError("provider down")

        with self.assertLogs("echosphere.agent.orchestrator", level="WARNING"):
            reply = "".join(
                agent.generate_spoken_reply(
                    speaker_id="Kenji", text="Hi", detected_language="en"
                )
            )
        self.assertTrue(reply.strip())

    # -- REQ-LAT-04: lightweight model configuration -------------------------------

    def test_openai_model_is_environment_configurable(self):
        """Verify the OpenAI model id is read from ECHOSPHERE_OPENAI_MODEL (REQ-LAT-04)."""
        with patch.dict(os.environ, {"ECHOSPHERE_OPENAI_MODEL": "gpt-test-tiny"}):
            agent = TeachingAgent(engine="openai", openai_api_key="test-key")
        self.assertEqual(agent.openai_model, "gpt-test-tiny")

    def test_gemini_model_is_environment_configurable(self):
        """Verify the Gemini model id is read from ECHOSPHERE_GEMINI_MODEL (REQ-LAT-04)."""
        with patch.dict(os.environ, {"ECHOSPHERE_GEMINI_MODEL": "gemini-test-tiny"}):
            agent = TeachingAgent(engine="gemini", gemini_api_key="test-key")
        self.assertEqual(agent.gemini_model, "gemini-test-tiny")

    def test_default_models_are_the_lightweight_tiers(self):
        """
        Verify the defaults are the documented low-latency tiers, not the previously
        hardcoded gpt-4o / gemini-2.5-flash (D-LAT-2).
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ECHOSPHERE_OPENAI_MODEL", None)
            os.environ.pop("ECHOSPHERE_GEMINI_MODEL", None)
            agent = TeachingAgent(engine="mock")

        self.assertEqual(agent.openai_model, DEFAULT_OPENAI_MODEL)
        self.assertEqual(agent.gemini_model, DEFAULT_GEMINI_MODEL)
        self.assertNotEqual(agent.openai_model, "gpt-4o")
        self.assertNotEqual(agent.gemini_model, "gemini-2.5-flash")

    def test_scaffolding_uses_a_separate_model_from_the_voice_path(self):
        """
        Verify the structured JSON call can run on a stronger model than the voice path.

        Observed live on 2026-08-30: the lightweight tier chosen for latency produces
        malformed JSON for the scaffolding contract often enough to matter ("Expecting
        property name enclosed in double quotes"). Scaffolding is no longer
        latency-critical after REQ-LAT-02, so it does not have to pay the fast tier's
        accuracy cost - this is the Risk 3 mitigation from the implementation plan.
        """
        agent = TeachingAgent(engine="mock")
        self.assertNotEqual(agent.gemini_scaffolding_model, agent.gemini_model)
        self.assertNotEqual(agent.openai_scaffolding_model, agent.openai_model)

    def test_scaffolding_models_are_environment_configurable(self):
        """Verify both scaffolding model ids are independently overridable."""
        with patch.dict(os.environ, {
            "ECHOSPHERE_OPENAI_SCAFFOLDING_MODEL": "gpt-test-big",
            "ECHOSPHERE_GEMINI_SCAFFOLDING_MODEL": "gemini-test-big",
        }):
            agent = TeachingAgent(engine="mock")
        self.assertEqual(agent.openai_scaffolding_model, "gpt-test-big")
        self.assertEqual(agent.gemini_scaffolding_model, "gemini-test-big")

    def test_structured_call_uses_the_scaffolding_model(self):
        """
        Verify the JSON path sends the scaffolding model, not the fast voice model.
        """
        agent = TeachingAgent(engine="gemini", gemini_api_key="test-key")
        agent.engine = "gemini"
        agent.gemini_model = "gemini-fast"
        agent.gemini_scaffolding_model = "gemini-strong"
        agent._gemini_client = MagicMock()
        agent._gemini_client.models.generate_content.return_value = MagicMock(
            text='{"spoken_response": "hi"}'
        )

        agent.process_turn(
            speaker_id="Kenji", text="Hello", detected_language="en", mode="tutor"
        )

        kwargs = agent._gemini_client.models.generate_content.call_args[1]
        self.assertEqual(kwargs["model"], "gemini-strong")

    def test_configured_model_reaches_the_provider_call(self):
        """Verify the configured model id is what is actually sent to the provider."""
        agent = TeachingAgent(engine="openai", openai_api_key="test-key")
        agent.engine = "openai"
        agent.openai_model = "gpt-test-tiny"
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.return_value = iter(())

        list(agent.generate_spoken_reply(
            speaker_id="Kenji", text="Hi", detected_language="en"
        ))

        kwargs = agent._openai_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs["model"], "gpt-test-tiny")


class TestTutorSessionLanguage(unittest.TestCase):
    """
    Test suite for the language the 1:1 tutor path actually teaches.

    A live Phase 8.2 smoke against real credentials showed a Hindi Convo AI session
    answering "Hindi isn't my specialty" in English: the session language reached
    `process_turn`/`generate_spoken_reply` as `detected_language`, but both prompts were
    built from the constructor default `target_language` ("Japanese"), so the model was
    told to teach a language the learner had not chosen. Only the tutor path is covered
    here - in mediation mode `detected_language` is a per-utterance detection, not the
    session's target language, and must not redirect the lesson.
    """

    def setUp(self):
        """Initialize a mock-engine agent whose default target is deliberately wrong."""
        self.agent = TeachingAgent(
            engine="mock",
            target_language="Japanese",
            native_language="English"
        )

    def test_tutor_prompt_follows_the_session_language(self):
        """Verify a Hindi tutor turn asks the model to teach Hindi, not Japanese."""
        prompt = self.agent.build_tutor_prompt(
            speaker_id="Kenji", detected_language="hi"
        )

        self.assertIn("Target Language: Hindi", prompt)
        self.assertNotIn("Target Language: Japanese", prompt)

    def test_voice_prompt_follows_the_session_language(self):
        """Verify the voice-critical path carries the same language as the session."""
        prompt = self.agent.build_tutor_voice_prompt(
            speaker_id="Kenji", text="नमस्ते", detected_language="hi"
        )

        self.assertIn("Target Language: Hindi", prompt)
        self.assertNotIn("Target Language: Japanese", prompt)

    def test_unknown_language_code_keeps_the_configured_target(self):
        """Verify an unmapped code falls back rather than naming a bogus language."""
        prompt = self.agent.build_tutor_prompt(
            speaker_id="Kenji", detected_language="zz"
        )

        self.assertIn("Target Language: Japanese", prompt)

    def test_mediation_prompt_ignores_the_detected_language(self):
        """
        Verify ambient mediation still teaches the configured target language: there,
        `detected_language` reports which language an utterance happened to be in.
        """
        with patch("src.agent.orchestrator.create_teaching_prompt") as create_prompt:
            create_prompt.return_value = "prompt"
            self.agent.process_turn(
                speaker_id="Kenji", text="Namaste", detected_language="hi"
            )

        self.assertEqual(
            create_prompt.call_args[1]["target_language"], "Japanese"
        )


class TestCameraQuestionDetection(unittest.TestCase):
    """
    Test suite for recognizing a camera-directed question (REQ-CAM-02, Phase 2).

    This predicate runs on every single voice turn, so it is deliberately a regex list
    rather than a model call - and deliberately narrow. Its job is not to be certain, it
    is to be free: a false positive with the camera off costs nothing, because the second
    gate (a fresh buffered frame) is what actually authorizes a vendor call.
    """

    CAMERA_QUESTIONS = [
        "What is this?",
        "what's this",
        "What is that?",
        "Hey, what are these?",
        "What am I looking at?",
        "what am i holding",
        "Can you see this?",
        "Could you read this for me?",
        "Do you see that?",
        "What does this say?",
        "what do these mean",
        "Look at this!",
        "take a look at that",
        "What is in front of me?",
        "What is written here?",
        "What kind of flower is this?",
    ]

    NOT_CAMERA_QUESTIONS = [
        "",
        "What is this word?",
        "What is that expression you just used?",
        "I went to the market yesterday.",
        "Can you explain the difference between these two verbs I just said?",
        "How do you say hello in Japanese?",
        "That was hard to pronounce.",
    ]

    def test_camera_directed_phrasings_are_recognized(self):
        """Verify the phrase list covers how a learner actually points at something."""
        for text in self.CAMERA_QUESTIONS:
            with self.subTest(text=text):
                self.assertTrue(looks_like_camera_question(text))

    def test_utterances_about_language_are_not_camera_questions(self):
        """
        Verify a question about a word is not read as a question about the room.

        "What is this word?" is the near-miss that matters: it is asking about something
        the learner just heard, and answering it by describing their desk would be an
        answer to a question nobody asked.
        """
        for text in self.NOT_CAMERA_QUESTIONS:
            with self.subTest(text=text):
                self.assertFalse(looks_like_camera_question(text))

    def test_detection_ignores_case_spacing_and_smart_quotes(self):
        """Verify transcribed speech is matched however the ASR punctuated it."""
        self.assertTrue(looks_like_camera_question("  WHAT'S   THIS?  "))
        self.assertTrue(looks_like_camera_question("What’s this?"))


class TestSpokenReplyCameraGrounding(unittest.TestCase):
    """
    Test suite for grounding a spoken reply in the live camera (REQ-CAM-03, Phase 3).

    The contract under test is a latency contract as much as a correctness one: the
    lookup happens only when the utterance points at something, only when a fresh frame
    is buffered, and never for longer than its own bound - because everything here runs
    while the learner is sitting in silence (REQ-LAT-02).
    """

    VISION_RESPONSE = {
        "candidates": [{
            "content": {"parts": [{
                "text": (
                    "A folding paper crane\n"
                    "An origami crane in red paper, held up close to the lens."
                )
            }]}
        }]
    }

    def setUp(self):
        """An OpenAI-shaped agent whose prompt can be captured, over a live buffer."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.buffer = CameraFrameBuffer()
        self.transport = MagicMock(return_value=self.VISION_RESPONSE)
        self.tools = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key="test-key", transport=self.transport),
            camera_buffer=self.buffer
        )
        self.session = SessionRecord.create(
            channel="camera-voice", mode="language_learning",
            languages=["ja"], participants=["Kenji"]
        )

        self.agent = TeachingAgent(engine="openai", openai_api_key="test-key", tools=self.tools)
        self.agent.engine = "openai"
        self.agent._openai_client = MagicMock()
        self.agent._openai_client.chat.completions.create.side_effect = self.fake_stream

    def fake_stream(self, *args, **kwargs):
        """Replays two deltas so the reply completes without a real provider."""
        for piece in ["I can see ", "it."]:
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=piece))]
            yield chunk

    def spoken_prompt(self):
        """Returns the user prompt the provider was actually asked to answer."""
        kwargs = self.agent._openai_client.chat.completions.create.call_args[1]
        return kwargs["messages"][-1]["content"]

    def reply(self, text="What is this?"):
        """Runs one spoken turn over the live session and drains the stream."""
        return "".join(self.agent.generate_spoken_reply(
            speaker_id="Kenji", text=text, detected_language="ja",
            session=self.session, channel="camera-voice"
        ))

    def cards(self):
        """Returns the reference cards published during the turn."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == "reference.card"
        ]

    def test_a_camera_question_over_a_fresh_frame_grounds_the_prompt(self):
        """
        Verify what the camera saw reaches the model, and the card is drawn once.

        One vendor call serves both: describing the frame twice - once to speak from,
        once to draw - would double the cost of the one thing the learner waits on.
        """
        self.buffer.put("camera-voice", b"a frame")

        self.assertTrue(self.reply().strip())

        self.assertIn("origami crane", self.spoken_prompt())
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(len(self.cards()), 1)
        self.assertEqual(self.cards()[0]["card"]["requested_by"], "voice")

    def test_without_a_buffered_frame_the_agent_is_told_it_cannot_see(self):
        """
        Verify the camera being off produces honesty, not a guess - and costs nothing.

        No frame means no vendor call at all: this is the gate that keeps a camera-shaped
        phrase from spending money on a camera nobody turned on (Risk 2).
        """
        self.assertTrue(self.reply().strip())

        self.assertIn("no camera view", self.spoken_prompt())
        self.assertEqual(self.transport.call_count, 0)
        self.assertEqual(self.cards(), [])

    def test_an_utterance_that_is_not_about_the_camera_is_left_alone(self):
        """Verify an ordinary turn carries no camera context and pays nothing."""
        self.buffer.put("camera-voice", b"a frame")

        self.assertTrue(self.reply("I went to the market yesterday.").strip())

        prompt = self.spoken_prompt()
        self.assertNotIn("origami crane", prompt)
        self.assertNotIn("no camera view", prompt)
        self.assertEqual(self.transport.call_count, 0)

    def test_a_lookup_that_times_out_still_produces_a_reply(self):
        """
        Verify a slow vendor degrades to a normal spoken reply (Risk 1).

        The learner is waiting in silence and the Convo AI Engine's idle_timeout is
        watching: a lookup that cannot finish in time has to be abandoned, not awaited.
        """
        released = threading.Event()
        self.addCleanup(released.set)

        def stalls(*args, **kwargs):
            released.wait(30.0)
            return self.VISION_RESPONSE

        self.transport.side_effect = stalls
        self.tools.camera_lookup_timeout = 0.3
        self.buffer.put("camera-voice", b"a frame")

        with self.assertLogs("echosphere.agent.tools.dispatch", level="WARNING"):
            reply = self.reply()

        self.assertTrue(reply.strip())
        self.assertIn("no camera view", self.spoken_prompt())
        self.assertEqual(self.cards(), [])

    def test_a_lookup_that_raises_still_produces_a_reply(self):
        """Verify a broken vision path never takes the conversation down with it."""
        self.transport.side_effect = RuntimeError("vendor down")
        self.buffer.put("camera-voice", b"a frame")

        with self.assertLogs("echosphere.agent.tools.dispatch", level="WARNING"):
            reply = self.reply()

        self.assertTrue(reply.strip())
        self.assertIn("no camera view", self.spoken_prompt())

    def test_a_turn_without_a_channel_never_looks(self):
        """
        Verify the pre-REQ-CAM-03 callers are unaffected.

        The ambient pipeline has no channel to look through, and must keep producing
        exactly the reply it did before this feature existed.
        """
        self.buffer.put("camera-voice", b"a frame")

        reply = "".join(self.agent.generate_spoken_reply(
            speaker_id="Kenji", text="What is this?", detected_language="ja"
        ))

        self.assertTrue(reply.strip())
        self.assertNotIn("no camera view", self.spoken_prompt())
        self.assertEqual(self.transport.call_count, 0)

    def test_the_scaffolding_pass_reuses_the_observation_without_a_second_card(self):
        """
        Verify the notes describe what was seen, not the pronoun (Task 3.3).

        The scaffolding call runs after the spoken reply, against the same buffered
        frame: it must reuse that description rather than pay for a second one, and must
        not draw a duplicate of the card already on screen.
        """
        self.buffer.put("camera-voice", b"a frame")
        self.reply()

        context = self.agent.observe_live_camera(
            self.session, "camera-voice", "Kenji", "What is this?", announce=False
        )

        self.assertIn("origami crane", context)
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(len(self.cards()), 1)


if __name__ == '__main__':
    unittest.main()
