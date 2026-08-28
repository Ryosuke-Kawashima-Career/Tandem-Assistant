"""
Summary:
    test_audio.py provides automated unit test cases for EchoSphere's audio & RTC layer:
    - Audio STT Transcription (Mock engine, WAV conversion, language handling for hi, ja, en)
    - Agora RTC Client (Token generation, channel lifecycle, and Data Stream broadcasting)
    - Voice Activity Detection (RMS energy calculation, speech frame detection, silence hangover segmentation)

Key Test Classes:
    - AudioTests: Main test suite covering STTTranscriber, AgoraVoiceChannelClient, and VoiceActivityDetector.
"""

import unittest
import math
import struct
import json
import time

from src.audio.stt_transcriber import STTTranscriber, TranscriptionResult
from src.rtc.agora_client import AgoraVoiceChannelClient, RTCDataStreamPacket
from src.audio.vad_processor import VoiceActivityDetector


def generate_sine_wave_pcm(
    frequency_hz: float = 440.0,
    duration_ms: int = 100,
    sample_rate: int = 16000,
    amplitude: float = 12000.0
) -> bytes:
    """
    Generates synthetic 16-bit mono linear PCM audio representing a continuous sine tone.
    
    Algorithm:
    1. Calculate total samples based on sample_rate and duration_ms.
    2. Compute sine amplitude for each sample index.
    3. Pack sample as signed 16-bit little-endian integer (<h).
    """
    total_samples = int(sample_rate * (duration_ms / 1000.0))
    buffer = bytearray()
    for i in range(total_samples):
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * (i / sample_rate)))
        buffer.extend(struct.pack("<h", max(-32768, min(32767, sample))))
    return bytes(buffer)


def generate_silence_pcm(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Generates synthetic 16-bit mono silence (null bytes) for given duration."""
    total_samples = int(sample_rate * (duration_ms / 1000.0))
    return b"\x00\x00" * total_samples


class AudioTests(unittest.TestCase):
    """
    Comprehensive test suite verifying STT transcription, Agora RTC client,
    and Voice Activity Detection (VAD) audio pipeline.
    """

    def setUp(self):
        """Initialize test fixtures before each test method runs."""
        self.sample_rate = 16000
        self.stt = STTTranscriber(engine="mock", sample_rate=self.sample_rate)
        self.agora = AgoraVoiceChannelClient(
            app_id="test_app_id_12345",
            app_certificate="test_cert_abcde",
            channel_name="echosphere-test-room",
            uid=1001
        )
        self.vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            frame_duration_ms=20,
            silence_hangover_ms=200,
            min_speech_duration_ms=100,
            energy_threshold=350.0
        )

    def test_stt_transcriber(self):
        """
        Verify STTTranscriber with mock engine across English, Japanese, and Hindi.
        
        Algorithm:
        1. Generate synthetic PCM audio data.
        2. Transcribe with 'en' language target and verify result fields.
        3. Transcribe with 'ja' language target and verify Japanese text response.
        4. Transcribe with 'hi' language target and verify Hindi text response.
        5. Verify RIFF/WAV header generation via pcm_to_wav.
        """
        # Step 1: Generate synthetic PCM audio
        audio_data = generate_sine_wave_pcm(duration_ms=150, sample_rate=self.sample_rate)

        # Step 2: Test English transcription
        res_en = self.stt.transcribe(audio_data, language="en", speaker_id="user_en", is_raw_pcm=True)
        self.assertIsInstance(res_en, TranscriptionResult)
        self.assertEqual(res_en.language, "en")
        self.assertEqual(res_en.speaker_id, "user_en")
        self.assertTrue(res_en.is_final)
        self.assertIn("Hello", res_en.text)

        # Step 3: Test Japanese transcription
        res_ja = self.stt.transcribe(audio_data, language="ja", speaker_id="user_ja")
        self.assertEqual(res_ja.language, "ja")
        self.assertIn("こんにちは", res_ja.text)

        # Step 4: Test Hindi transcription
        res_hi = self.stt.transcribe(audio_data, language="hi", speaker_id="user_hi")
        self.assertEqual(res_hi.language, "hi")
        self.assertIn("नमस्ते", res_hi.text)

        # Step 5: Verify WAV encapsulation
        wav_bytes = self.stt.pcm_to_wav(audio_data)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes)
        self.assertGreater(len(wav_bytes), len(audio_data))

    def test_agora_client(self):
        """
        Verify AgoraVoiceChannelClient authentication, connection lifecycle, and RTC Data Stream broadcasting.
        
        Algorithm:
        1. Test dynamic RTC token generation.
        2. Join the channel and verify connection state and stream ID.
        3. Register a data stream event listener callback.
        4. Broadcast a structured UI payload (subtitles / idiom card).
        5. Verify received packet properties and roundtrip serialization.
        6. Publish an audio frame and verify audio hook callback.
        7. Leave channel and verify cleanup.
        """
        # Step 1: Token generation
        token = self.agora.generate_token(role="publisher")
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)
        self.assertEqual(self.agora.current_token, token)

        # Step 2: Channel join
        self.assertFalse(self.agora.is_connected)
        join_success = self.agora.join_channel()
        self.assertTrue(join_success)
        self.assertTrue(self.agora.is_connected)
        self.assertIsNotNone(self.agora.stream_id)

        # Step 3: Register callbacks
        received_data_packets = []
        received_audio_frames = []
        self.agora.register_on_data_stream(lambda pkt: received_data_packets.append(pkt))
        self.agora.register_on_audio_frame(lambda pcm, rate: received_audio_frames.append((pcm, rate)))

        # Step 4: Send Data Stream Message (Subtitles with transliteration & translation)
        payload = {
            "speaker": "Kenji",
            "japanese": "一期一会ですね",
            "romaji": "Ichigo ichie desu ne",
            "english": "Treasure every encounter",
            "hindi": "हर मुलाकात अनमोल है"
        }
        sent = self.agora.send_data_stream_message(event_type="subtitles", payload=payload)
        self.assertTrue(sent)
        self.assertEqual(len(received_data_packets), 1)
        self.assertEqual(received_data_packets[0].event_type, "subtitles")
        self.assertEqual(received_data_packets[0].payload["romaji"], "Ichigo ichie desu ne")

        # Step 5: Test RTC Data Stream Packet serialization roundtrip
        packet = RTCDataStreamPacket(event_type="quiz", payload={"question": "What does 'Namaste' mean?"})
        packet_bytes = packet.to_bytes()
        restored = RTCDataStreamPacket.from_bytes(packet_bytes)
        self.assertEqual(restored.event_type, "quiz")
        self.assertEqual(restored.payload["question"], "What does 'Namaste' mean?")

        # Step 6: Test audio frame publication
        pcm_frame = generate_sine_wave_pcm(duration_ms=20, sample_rate=self.sample_rate)
        pub_ok = self.agora.publish_audio_frame(pcm_frame, sample_rate=self.sample_rate)
        self.assertTrue(pub_ok)
        self.assertEqual(len(received_audio_frames), 1)
        self.assertEqual(received_audio_frames[0][0], pcm_frame)
        self.assertEqual(received_audio_frames[0][1], self.sample_rate)

        # Step 7: Channel leave
        leave_success = self.agora.leave_channel()
        self.assertTrue(leave_success)
        self.assertFalse(self.agora.is_connected)
        self.assertIsNone(self.agora.stream_id)

    def test_vad_processor(self):
        """
        Verify VoiceActivityDetector frame classification, silence hangover, and utterance segmentation.
        
        Algorithm:
        1. Test RMS energy computation for silence vs active sine wave.
        2. Verify is_speech() correctly discriminates between silent and vocal frames.
        3. Feed a sequence of [Silence (100ms) -> Speech (300ms) -> Silence (300ms)] into process_chunk.
        4. Verify that exactly 1 segmented utterance is yielded after silence hangover.
        5. Test detector reset and buffer clearance.
        """
        # Step 1: RMS energy check
        silence_frame = generate_silence_pcm(duration_ms=20, sample_rate=self.sample_rate)
        speech_frame = generate_sine_wave_pcm(duration_ms=20, sample_rate=self.sample_rate, amplitude=14000.0)

        rms_silence = self.vad.calculate_rms_energy(silence_frame)
        rms_speech = self.vad.calculate_rms_energy(speech_frame)
        self.assertAlmostEqual(rms_silence, 0.0, delta=1.0)
        self.assertGreater(rms_speech, 4000.0)

        # Step 2: Speech frame classification
        self.assertFalse(self.vad.is_speech(silence_frame))
        self.assertTrue(self.vad.is_speech(speech_frame))

        # Step 3 & 4: Utterance stream segmentation
        silence_pre = generate_silence_pcm(duration_ms=100, sample_rate=self.sample_rate)
        speech_segment = generate_sine_wave_pcm(duration_ms=300, sample_rate=self.sample_rate, amplitude=14000.0)
        silence_post = generate_silence_pcm(duration_ms=300, sample_rate=self.sample_rate)

        # Feed leading silence -> no utterance emitted, state is idle
        res_pre = self.vad.process_chunk(silence_pre)
        self.assertEqual(len(res_pre), 0)
        self.assertFalse(self.vad.is_speaking)

        # Feed speech body -> speaking triggers, speech frames accumulate
        res_speech = self.vad.process_chunk(speech_segment)
        self.assertEqual(len(res_speech), 0)
        self.assertTrue(self.vad.is_speaking)

        # Feed trailing silence -> silence exceeds hangover (200ms), complete utterance is emitted
        res_post = self.vad.process_chunk(silence_post)
        self.assertEqual(len(res_post), 1)
        self.assertFalse(self.vad.is_speaking)
        self.assertGreater(len(res_post[0]), 0)

        # Step 5: Test reset
        self.vad.reset()
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.active_speech_frames), 0)


if __name__ == "__main__":
    unittest.main()
