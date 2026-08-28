"""
Summary:
    test_rtc_datastream.py provides automated unit and integration tests for EchoSphere's
    Agora RTC Data Stream synchronization engine (DataStreamManager).
    It verifies packet serialization, broadcast dispatching, payload validation for
    subtitles, cultural idiom cards, topic prompts, quizzes, teacher alerts,
    and live speaker balance metrics.

Key Test Classes:
    - TestRTCDataStream: Main test suite testing DataStreamManager with mock and live Agora clients.
"""

import unittest
import json
from unittest.mock import MagicMock

from src.rtc.data_stream import DataStreamManager
from src.rtc.agora_client import AgoraVoiceChannelClient, RTCDataStreamPacket
from src.audio.stt_transcriber import STTTranscriber
from src.audio.tts_synthesizer import TTSSynthesizer


class TestRTCDataStream(unittest.TestCase):
    """
    Test suite verifying RTC Data Stream message formatting, broadcasting, and listener integration.
    """

    def setUp(self):
        """Initialize mock Agora client and DataStreamManager instance."""
        self.mock_agora_client = MagicMock(spec=AgoraVoiceChannelClient)
        self.data_stream = DataStreamManager(self.mock_agora_client)

    def test_rtc_data_stream_subtitle(self):
        """
        Verify that send_subtitle constructs valid multi-lingual subtitle payload
        and calls the Agora client's dispatch method.
        
        Algorithm:
        1. Call send_subtitle with speaker, text, and transliteration/translations.
        2. Assert Agora client send method was invoked once.
        3. Inspect captured event_type ('subtitles') and payload contents.
        """
        self.data_stream.send_subtitle(
            speaker="Kenji",
            text="一期一会ですね",
            transliteration="Ichigo ichie desu ne",
            translation_en="Treasure every encounter",
            translation_ja="一期一会ですね",
            translation_hi="हर मुलाकात अनमोल है"
        )
        
        # Verify call on client
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "subtitles")
        self.assertEqual(payload["speaker"], "Kenji")
        self.assertEqual(payload["transliteration"], "Ichigo ichie desu ne")
        self.assertEqual(payload["translation_en"], "Treasure every encounter")

    def test_send_idiom_card(self):
        """
        Verify that send_idiom_card dispatches structured cultural annotation card.
        
        Algorithm:
        1. Call send_idiom_card with phrase, romaji, meaning, and cultural note.
        2. Assert Agora client send method was invoked once.
        3. Verify event_type is 'idiom_card' and payload has detected=True.
        """
        self.data_stream.send_idiom_card(
            phrase="一期一会 (Ichigo Ichie)",
            romaji="Ichigo ichie",
            meaning="Once-in-a-lifetime encounter",
            cultural_note="Derived from traditional Japanese tea ceremony philosophy."
        )
        
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "idiom_card")
        self.assertTrue(payload["detected"])
        self.assertEqual(payload["phrase"], "一期一会 (Ichigo Ichie)")
        self.assertIn("tea ceremony", payload["cultural_note"].lower())

    def test_send_topic(self):
        """
        Verify that send_topic broadcasts conversational topic prompts to peer learners.
        """
        self.data_stream.send_topic(
            topic_title="Festivals & Food in India and Japan",
            prompt="What is your favorite festival dish?"
        )
        
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "topic_prompt")
        self.assertEqual(payload["topic_title"], "Festivals & Food in India and Japan")
        self.assertIn("favorite festival dish", payload["prompt"])

    def test_send_quiz(self):
        """
        Verify that send_quiz constructs interactive multiple-choice question payload.
        """
        self.data_stream.send_quiz(
            question="What does 'Namaste' literally mean in Sanskrit?",
            options=["I bow to the divine in you", "Have a great day", "Thank you very much"],
            correct_index=0,
            explanation="'Namaste' is derived from 'namas' (bow) and 'te' (to you)."
        )
        
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "quiz")
        self.assertTrue(payload["active"])
        self.assertEqual(payload["correct_index"], 0)
        self.assertEqual(len(payload["options"]), 3)

    def test_send_teacher_alert(self):
        """
        Verify that send_teacher_alert dispatches notifications for instructor dashboard.
        """
        self.data_stream.send_teacher_alert(
            message="Speaking imbalance detected in Room 3: Learner A (85%) vs Learner B (15%)",
            severity="warning"
        )
        
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "teacher_alert")
        self.assertTrue(payload["alert_required"])
        self.assertEqual(payload["severity"], "warning")

    def test_send_speaking_balance(self):
        """
        Verify that send_speaking_balance broadcasts balance percentage distribution.
        """
        stats = {"Kenji": 52, "Aarav": 48}
        self.data_stream.send_speaking_balance(stats)
        
        self.mock_agora_client.send_data_stream.assert_called_once()
        args, _ = self.mock_agora_client.send_data_stream.call_args
        event_type, payload = args
        self.assertEqual(event_type, "speaking_balance")
        self.assertEqual(payload["speaker_percentages"]["Kenji"], 52)
        self.assertEqual(payload["speaker_percentages"]["Aarav"], 48)

    def test_end_to_end_real_client_integration(self):
        """
        Integration test verifying DataStreamManager with an actual AgoraVoiceChannelClient
        and verifying local event subscriber callbacks.
        
        Algorithm:
        1. Initialize live AgoraVoiceChannelClient and connect.
        2. Register an event subscriber callback.
        3. Send subtitle and idiom card messages via DataStreamManager.
        4. Assert that subscriber callback received valid RTCDataStreamPacket instances.
        """
        real_client = AgoraVoiceChannelClient(channel_name="test-sync-room")
        real_client.join_channel()
        
        received_packets = []
        real_client.register_on_data_stream(lambda pkt: received_packets.append(pkt))
        
        live_stream_mgr = DataStreamManager(real_client)
        live_stream_mgr.send_subtitle("Aarav", "नमस्ते", translation_en="Hello")
        live_stream_mgr.send_idiom_card("नमस्ते", meaning="Formal greeting")
        
        self.assertEqual(len(received_packets), 2)
        self.assertEqual(received_packets[0].event_type, "subtitles")
        self.assertEqual(received_packets[0].payload["speaker"], "Aarav")
        self.assertEqual(received_packets[1].event_type, "idiom_card")
        self.assertEqual(received_packets[1].payload["phrase"], "नमस्ते")


if __name__ == '__main__':
    unittest.main()
