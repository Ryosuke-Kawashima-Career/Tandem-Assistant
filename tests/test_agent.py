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

import unittest
import json
from src.agent.orchestrator import TeachingAgent
from src.agent.prompts import (
    create_teaching_prompt,
    create_silence_breaker_prompt,
    SYSTEM_PROMPT_TANDEM_TEACHER
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

    def test_tts_synthesizer_pcm_generation(self):
        """
        Verify that TTSSynthesizer produces valid non-empty 16kHz PCM audio
        for the AI co-teacher's spoken interventions.
        """
        speech_text = "素晴らしい会話ですね！ (Wonderful conversation!)"
        pcm_bytes = self.tts.synthesize(speech_text, language="ja")
        
        self.assertIsInstance(pcm_bytes, bytes)
        self.assertGreater(len(pcm_bytes), 0)


if __name__ == '__main__':
    unittest.main()
