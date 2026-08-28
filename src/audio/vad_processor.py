"""
Summary:
    vad_processor.py provides real-time Voice Activity Detection (VAD) and speech
    boundary segmentation for EchoSphere.
    It monitors incoming continuous raw PCM audio streams from Agora RTC, classifies
    each 10ms/20ms/30ms frame as speech or non-speech (silence/noise), maintains a
    pre-speech ring buffer, and slices full utterances when a silence boundary
    (hangover) is detected.

Key Components:
    - VoiceActivityDetector: State-machine processor that ingests audio frames,
      detects speech boundaries, and yields segmented utterance audio buffers.
    - AudioFrame: Lightweight container for raw PCM frames with timing metadata.
"""

import math
import struct
import logging
from collections import deque
from typing import Optional, List, Generator, Tuple

logger = logging.getLogger("echosphere.audio.vad_processor")

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logger.warning("webrtcvad module not found. Falling back to RMS energy-based VAD.")


class AudioFrame:
    """Represents a single discrete audio frame with raw PCM bytes and timestamp."""
    def __init__(self, pcm_bytes: bytes, timestamp_ms: float, duration_ms: float):
        self.pcm_bytes = pcm_bytes
        self.timestamp_ms = timestamp_ms
        self.duration_ms = duration_ms


class VoiceActivityDetector:
    """
    Real-time VAD processor and speech utterance segmenter.
    
    Attributes:
        sample_rate (int): Audio sampling rate in Hz (e.g., 16000).
        frame_duration_ms (int): Duration of each analysis frame in ms (10, 20, or 30).
        vad_mode (int): webrtcvad aggressiveness level (0 to 3, 3 is most aggressive).
        silence_hangover_ms (int): Milliseconds of consecutive silence before ending utterance.
        min_speech_duration_ms (int): Minimum valid speech duration to discard accidental clicks.
        energy_threshold (float): Fallback RMS threshold when webrtcvad is unavailable.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        vad_mode: int = 2,
        silence_hangover_ms: int = 600,
        min_speech_duration_ms: int = 250,
        energy_threshold: float = 450.0
    ):
        """
        Initialize the VAD state machine and frame parameters.
        
        Algorithm:
        1. Validate sample_rate and frame_duration_ms compliance (WebRTC VAD constraints: 8k/16k/32k/48k Hz).
        2. Calculate bytes per frame (sample_rate * 2 bytes/sample * (duration_ms / 1000)).
        3. Initialize WebRTC VAD instance or RMS energy fallback mode.
        4. Configure ring buffers for pre-speech capture and active speech accumulation.
        """
        # Step 1: Validate parameters
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate: {sample_rate}. Must be 8000, 16000, 32000, or 48000 Hz.")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"Unsupported frame duration: {frame_duration_ms}. Must be 10, 20, or 30 ms.")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_mode = vad_mode
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.energy_threshold = energy_threshold

        # Step 2: Compute frame byte size (16-bit mono = 2 bytes per sample)
        self.frame_bytes_size = int(self.sample_rate * 2 * (self.frame_duration_ms / 1000.0))

        # Step 3: Initialize VAD engine
        self._vad = None
        if WEBRTC_VAD_AVAILABLE:
            try:
                self._vad = webrtcvad.Vad(self.vad_mode)
            except Exception as err:
                logger.warning(f"Error initializing webrtcvad ({err}). Using RMS fallback.")

        # Step 4: Setup buffers and state tracking
        self.pre_speech_buffer_length = int(200 / self.frame_duration_ms)  # ~200ms pre-speech margin
        self.pre_speech_ring_buffer = deque(maxlen=self.pre_speech_buffer_length)
        self.active_speech_frames: List[bytes] = []
        
        self.is_speaking: bool = False
        self.consecutive_silence_ms: int = 0
        self.current_speech_duration_ms: int = 0
        self.total_processed_ms: float = 0.0

    def calculate_rms_energy(self, pcm_bytes: bytes) -> float:
        """
        Calculates Root-Mean-Square (RMS) audio signal amplitude for fallback VAD.
        
        Algorithm:
        1. Unpack 16-bit signed PCM samples from byte array.
        2. Compute sum of squared sample amplitudes.
        3. Divide by sample count and take square root.
        """
        count = len(pcm_bytes) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", pcm_bytes)
        sum_squares = sum(s * s for s in shorts)
        return math.sqrt(sum_squares / count)

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Determines whether a single audio frame contains voice activity.
        
        Algorithm:
        1. Pad or truncate incoming frame to exact frame_bytes_size.
        2. Compute RMS signal energy. If energy is below energy_threshold, classify as silence immediately.
        3. If energy exceeds threshold and webrtcvad is active, consult webrtcvad.is_speech().
        4. If webrtcvad is unavailable or fails, rely on energy threshold.
        """
        if len(pcm_bytes) != self.frame_bytes_size:
            # Pad or truncate if needed for exact frame evaluation
            if len(pcm_bytes) < self.frame_bytes_size:
                pcm_bytes = pcm_bytes.ljust(self.frame_bytes_size, b"\x00")
            else:
                pcm_bytes = pcm_bytes[:self.frame_bytes_size]

        # Step 2: Energy Gate - Low amplitude/silent audio frames are immediately silence
        rms = self.calculate_rms_energy(pcm_bytes)
        if rms < self.energy_threshold:
            return False

        # Step 3: Consult WebRTC VAD if available
        if self._vad is not None:
            try:
                return self._vad.is_speech(pcm_bytes, self.sample_rate)
            except Exception:
                pass

        # Step 4: Fallback based on energy threshold
        return True

    def process_chunk(self, raw_audio_chunk: bytes) -> List[bytes]:
        """
        Ingests a continuous chunk of raw PCM audio and returns any completed utterances.
        
        Algorithm:
        1. Slice incoming audio chunk into exact discrete frames of duration `frame_duration_ms`.
        2. Classify each frame as speech or non-speech via `is_speech()`.
        3. State Transition - Inactive to Active (Speech Start):
           - When speech frame arrives, transition to `is_speaking = True`.
           - Prepend frames stored in `pre_speech_ring_buffer` to avoid clipping word onsets.
        4. State Transition - Active (Speaking):
           - Append incoming frames to `active_speech_frames`.
           - If frame is silence, increment `consecutive_silence_ms`.
           - If frame is speech, reset `consecutive_silence_ms` to 0.
        5. State Transition - Active to Inactive (Silence Hangover / Utterance End):
           - When `consecutive_silence_ms >= silence_hangover_ms`:
             - Check if total utterance duration >= `min_speech_duration_ms`.
             - If valid, concatenate accumulated frames and push to completed utterances list.
             - Reset active speech accumulator and transition `is_speaking = False`.
        6. Return list of completed utterance byte buffers.
        """
        completed_utterances: List[bytes] = []
        offset = 0

        while offset + self.frame_bytes_size <= len(raw_audio_chunk):
            frame_bytes = raw_audio_chunk[offset : offset + self.frame_bytes_size]
            offset += self.frame_bytes_size
            self.total_processed_ms += self.frame_duration_ms

            has_speech = self.is_speech(frame_bytes)

            if not self.is_speaking:
                # Idle / Silence state
                if has_speech:
                    # Speech detected: Start accumulating new utterance
                    self.is_speaking = True
                    self.consecutive_silence_ms = 0
                    self.current_speech_duration_ms = 0
                    
                    # Prepend pre-speech buffer (recovers onset consonants)
                    self.active_speech_frames = list(self.pre_speech_ring_buffer)
                    self.active_speech_frames.append(frame_bytes)
                    self.pre_speech_ring_buffer.clear()
                else:
                    # Maintain rolling pre-speech ring buffer
                    self.pre_speech_ring_buffer.append(frame_bytes)
            else:
                # Active speech state
                self.active_speech_frames.append(frame_bytes)
                self.current_speech_duration_ms += self.frame_duration_ms

                if not has_speech:
                    self.consecutive_silence_ms += self.frame_duration_ms
                    # Check if silence exceeds end-of-utterance threshold
                    if self.consecutive_silence_ms >= self.silence_hangover_ms:
                        # Check minimum duration filter
                        if self.current_speech_duration_ms >= self.min_speech_duration_ms:
                            utterance_bytes = b"".join(self.active_speech_frames)
                            completed_utterances.append(utterance_bytes)
                            logger.info(f"Utterance detected: {len(utterance_bytes)} bytes (~{self.current_speech_duration_ms}ms)")
                        
                        # Reset utterance tracking
                        self.is_speaking = False
                        self.active_speech_frames = []
                        self.consecutive_silence_ms = 0
                        self.current_speech_duration_ms = 0
                else:
                    # Reset silence hangover counter upon subsequent speech
                    self.consecutive_silence_ms = 0

        return completed_utterances

    def flush(self) -> Optional[bytes]:
        """
        Forces emission of any accumulated active speech frames (e.g. on stream close).
        
        Algorithm:
        1. Check if currently speaking with frames accumulated.
        2. If accumulated duration >= min_speech_duration_ms, concatenate and return.
        3. Reset internal buffers.
        """
        if self.is_speaking and self.active_speech_frames:
            if self.current_speech_duration_ms >= self.min_speech_duration_ms:
                utterance = b"".join(self.active_speech_frames)
                self.reset()
                return utterance
        self.reset()
        return None

    def reset(self):
        """Resets detector state and clears all ring and utterance buffers."""
        self.is_speaking = False
        self.active_speech_frames = []
        self.pre_speech_ring_buffer.clear()
        self.consecutive_silence_ms = 0
        self.current_speech_duration_ms = 0