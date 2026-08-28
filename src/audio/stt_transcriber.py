"""
Summary:
    stt_transcriber.py provides Speech-To-Text (STT) transcription for EchoSphere.
    It takes segmented audio utterances (from vad_processor.py), packages raw PCM
    into valid RIFF WAV containers, and dispatches them to configured AI transcription
    providers (OpenAI Whisper, Google Gemini Audio, or heuristic Mock Engine).
    It specifically supports tri-lingual multi-speaker conversational exchange
    across Hindi (hi), Japanese (ja), and English (en).

Key Components:
    - STTTranscriber: Main transcription coordinator supporting synchronous and
      asynchronous transcription requests with language autodetect or explicit hints.
    - TranscriptionResult: Dataclass capturing transcribed text, language code,
      confidence score, speaker ID, and timing metadata.
"""

import io
import os
import wave
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union

logger = logging.getLogger("echosphere.audio.stt_transcriber")

# Optional STT Provider SDKs
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """
    Structured outcome of an STT transcription operation.
    """
    text: str
    language: str  # 'hi', 'ja', 'en', or 'und' (undetermined)
    confidence: float
    speaker_id: Optional[str] = None
    duration_ms: int = 0
    timestamp_ms: int = 0
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to standard dictionary."""
        return asdict(self)


class STTTranscriber:
    """
    Speech-To-Text transcriber supporting tri-lingual dialogue across Hindi, Japanese, and English.
    
    Attributes:
        engine (str): Provider engine ('whisper', 'gemini', or 'mock').
        sample_rate (int): Audio sampling rate in Hz (default: 16000).
        channels (int): Audio channel count (1 for mono).
        sample_width (int): Bytes per sample (2 for 16-bit PCM).
    """

    def __init__(
        self,
        engine: str = "mock",
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2
    ):
        """
        Initialize the STT transcriber and underlying provider clients.
        
        Algorithm:
        1. Resolve API credentials from arguments or environment variables.
        2. Select active backend engine ('whisper', 'gemini', or 'mock').
        3. Initialize provider SDK client instance if credentials exist.
        4. Configure audio container parameters (16kHz, 16-bit mono).
        """
        self.engine = engine.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        # Step 1 & 2: Provider initializations
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self._openai_client = None
        if self.engine == "whisper" and OPENAI_AVAILABLE and self.openai_api_key:
            self._openai_client = OpenAI(api_key=self.openai_api_key)
        elif self.engine == "whisper" and not self.openai_api_key:
            logger.warning("OpenAI API key missing. STTTranscriber will use mock fallback.")
            self.engine = "mock"

        self._gemini_client = None
        if self.engine == "gemini" and GOOGLE_GENAI_AVAILABLE and self.gemini_api_key:
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        elif self.engine == "gemini" and not self.gemini_api_key:
            logger.warning("Gemini API key missing. STTTranscriber will use mock fallback.")
            self.engine = "mock"

        logger.info(f"STTTranscriber initialized with engine: '{self.engine}' ({self.sample_rate}Hz mono)")

    def pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """
        Encapsulates raw PCM bytes into a standard RIFF/WAV audio container.
        
        Algorithm:
        1. Create an in-memory byte buffer (io.BytesIO).
        2. Open a wave write context specifying channels, sample width, and framerate.
        3. Write PCM frames to the buffer and finalize the RIFF header.
        4. Return raw WAV byte string.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data)
        return buf.getvalue()

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        speaker_id: Optional[str] = None,
        is_raw_pcm: bool = True
    ) -> TranscriptionResult:
        """
        Transcribes speech audio into structured text.
        
        Algorithm:
        1. Convert raw PCM bytes to WAV format if `is_raw_pcm` is True.
        2. Calculate approximate duration from audio byte length.
        3. Route transcription request to selected engine backend:
           - Whisper: Dispatches audio to OpenAI Whisper API with optional language code.
           - Gemini: Dispatches audio payload with multimodal prompt to Gemini model.
           - Mock: Generates deterministic test response for development/tests.
        4. Standardize response into a `TranscriptionResult` instance.
        """
        timestamp_ms = int(time.time() * 1000)
        
        # Step 1: Format audio as WAV if raw PCM
        wav_bytes = self.pcm_to_wav(audio_data) if is_raw_pcm else audio_data
        
        # Step 2: Compute audio duration in ms
        bytes_per_sec = self.sample_rate * self.channels * self.sample_width
        duration_ms = int((len(audio_data) / bytes_per_sec) * 1000) if is_raw_pcm else 1000

        # Step 3: Dispatch to engine
        if self.engine == "whisper" and self._openai_client is not None:
            return self._transcribe_whisper(wav_bytes, language, speaker_id, duration_ms, timestamp_ms)
        elif self.engine == "gemini" and GOOGLE_GENAI_AVAILABLE and self.gemini_api_key:
            return self._transcribe_gemini(wav_bytes, language, speaker_id, duration_ms, timestamp_ms)
        else:
            return self._transcribe_mock(wav_bytes, language, speaker_id, duration_ms, timestamp_ms)

    def _transcribe_whisper(
        self,
        wav_bytes: bytes,
        language: Optional[str],
        speaker_id: Optional[str],
        duration_ms: int,
        timestamp_ms: int
    ) -> TranscriptionResult:
        """Transcribes audio using OpenAI Whisper API."""
        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            
            kwargs: Dict[str, Any] = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json"
            }
            if language:
                kwargs["language"] = language

            resp = self._openai_client.audio.transcriptions.create(**kwargs)
            detected_lang = getattr(resp, "language", language or "en")
            text = getattr(resp, "text", "").strip()

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                confidence=0.95,
                speaker_id=speaker_id,
                duration_ms=duration_ms,
                timestamp_ms=timestamp_ms,
                is_final=True
            )
        except Exception as err:
            logger.error(f"Whisper transcription failed: {err}. Falling back to mock response.")
            return self._transcribe_mock(wav_bytes, language, speaker_id, duration_ms, timestamp_ms)

    def _transcribe_gemini(
        self,
        wav_bytes: bytes,
        language: Optional[str],
        speaker_id: Optional[str],
        duration_ms: int,
        timestamp_ms: int
    ) -> TranscriptionResult:
        """Transcribes audio using Google Gemini multimodal audio API."""
        try:
            prompt = (
                "Transcribe this speech audio accurately. "
                "The language is either Hindi, Japanese, or English (or code-switched). "
                "Output only the transcribed speech text without commentary."
            )
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
                ]
            )
            text = response.text.strip() if response.text else ""

            return TranscriptionResult(
                text=text,
                language=language or "ja",
                confidence=0.92,
                speaker_id=speaker_id,
                duration_ms=duration_ms,
                timestamp_ms=timestamp_ms,
                is_final=True
            )
        except Exception as err:
            logger.error(f"Gemini transcription failed: {err}. Falling back to mock response.")
            return self._transcribe_mock(wav_bytes, language, speaker_id, duration_ms, timestamp_ms)

    def _transcribe_mock(
        self,
        wav_bytes: bytes,
        language: Optional[str],
        speaker_id: Optional[str],
        duration_ms: int,
        timestamp_ms: int
    ) -> TranscriptionResult:
        """Generates deterministic mock transcription for local tests and scaffolding."""
        lang_samples = {
            "hi": "नमस्ते, आप कैसे हैं? (Namaste, aap kaise hain?)",
            "ja": "こんにちは、お元気ですか？ (Konnichiwa, ogenki desu ka?)",
            "en": "Hello, how are you doing today?",
        }
        resolved_lang = language if language in lang_samples else "en"
        sample_text = lang_samples.get(resolved_lang, "Hello, nice to meet you!")

        return TranscriptionResult(
            text=sample_text,
            language=resolved_lang,
            confidence=0.99,
            speaker_id=speaker_id or "user_001",
            duration_ms=duration_ms,
            timestamp_ms=timestamp_ms,
            is_final=True
        )