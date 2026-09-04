"""
Summary:
    audio.py adapts PCM between Agora's capture/publish format and the Gemini Live
    Translate wire format (REQ-17, TASK-11.2).

    Two directions, two different contracts:
    - Into Gemini: raw little-endian PCM16, mono, 16 kHz, in ~100 ms chunks. Chunking is
      not cosmetic - the Live session expects a steady realtime stream, and a partial
      chunk sent early is a chunk boundary in the middle of a phoneme.
    - Out of Gemini: PCM16 mono at 24 kHz, which has to be resampled and re-framed to
      whatever rate the Agora publish path wants before it can become a track.

    Both encoders buffer across calls, because Agora frame sizes are set by the capture
    device and have no relationship to either chunk size.

Key Classes:
    - GeminiInputEncoder: Agora frames -> 100 ms PCM16 mono 16 kHz chunks.
    - AgoraPublishAdapter: Gemini 24 kHz PCM -> fixed-duration Agora publish frames.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger("echosphere.translation.audio")

# Gemini Live Translate input contract (spec 3.1).
INPUT_SAMPLE_RATE = 16000
INPUT_CHUNK_MS = 100
INPUT_CHUNK_BYTES = INPUT_SAMPLE_RATE * 2 * INPUT_CHUNK_MS // 1000

# Gemini Live Translate output contract: PCM16 mono at 24 kHz.
GEMINI_OUTPUT_SAMPLE_RATE = 24000

# Agora publish defaults. 48 kHz is the Web SDK's own capture rate, and 20 ms is the
# frame duration the RTC pipeline is happiest consuming.
DEFAULT_PUBLISH_SAMPLE_RATE = 48000
DEFAULT_PUBLISH_FRAME_MS = 20

# Little-endian signed 16-bit: the one sample format both sides of this module speak.
PCM16 = "<i2"


def downmix_to_mono(pcm: bytes, channels: int = 1) -> bytes:
    """
    Averages an interleaved multi-channel PCM16 buffer down to one channel.

    Algorithm:
    1. Return the buffer untouched when it is already mono - the common case, and one
       that must not pay for a numpy round trip.
    2. Reinterpret as int16, drop any trailing partial frame, and reshape to (frames, ch).
    3. Average across channels in int32 (int16 would overflow on two loud channels),
       then round and clip back into int16 range.
    """
    if channels <= 1:
        return pcm
    samples = np.frombuffer(pcm, dtype=PCM16)
    usable = samples.size - (samples.size % channels)
    if usable <= 0:
        return b""
    frames = samples[:usable].reshape(-1, channels).astype(np.int32)
    mono = np.round(frames.mean(axis=1))
    return np.clip(mono, -32768, 32767).astype(PCM16).tobytes()


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """
    Resamples mono PCM16 by linear interpolation.

    Algorithm:
    1. Short-circuit when the rates already match, or the buffer is empty.
    2. Compute the output length from the rate ratio, so duration is preserved exactly.
    3. Interpolate the source samples onto the output grid, then round and clip.

    Linear interpolation rather than a windowed-sinc filter is a deliberate trade: this
    runs per audio frame inside a live call, the material is speech destined for an ASR
    front-end rather than for critical listening, and the alternative pulls `scipy.signal`
    into the realtime path for quality nobody in this pipeline can hear.
    """
    if source_rate == target_rate or not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=PCM16)
    if samples.size == 0:
        return b""

    output_length = int(round(samples.size * float(target_rate) / float(source_rate)))
    if output_length <= 0:
        return b""

    source_grid = np.arange(samples.size, dtype=np.float64)
    target_grid = np.linspace(0.0, samples.size - 1, output_length)
    resampled = np.interp(target_grid, source_grid, samples.astype(np.float64))
    return np.clip(np.round(resampled), -32768, 32767).astype(PCM16).tobytes()


class GeminiInputEncoder:
    """
    Turns one participant's Agora audio into Gemini Live input chunks (TASK-11.2).

    One encoder per speaker: the buffer holds a partial chunk between calls, so sharing
    an encoder across speakers would splice two voices into one utterance.
    """

    def __init__(
        self,
        source_rate: int = DEFAULT_PUBLISH_SAMPLE_RATE,
        channels: int = 1,
        chunk_ms: int = INPUT_CHUNK_MS
    ):
        """
        Initialize the encoder for one source track format.

        Algorithm:
        1. Record the capture format this track arrives in.
        2. Derive the emitted chunk size from the requested duration at 16 kHz mono.
        3. Start with an empty carry-over buffer.
        """
        self.source_rate = source_rate
        self.channels = max(1, channels)
        self.chunk_ms = chunk_ms
        self.chunk_bytes = INPUT_SAMPLE_RATE * 2 * chunk_ms // 1000
        self._buffer = bytearray()

    @property
    def pending_bytes(self) -> int:
        """Bytes of a partial chunk currently held back."""
        return len(self._buffer)

    def push(self, frame: bytes) -> List[bytes]:
        """
        Accepts one Agora frame and returns whatever whole chunks it completed.

        Algorithm:
        1. Downmix to mono, then resample to 16 kHz.
        2. Append to the carry-over buffer.
        3. Slice off every whole chunk; keep the remainder for the next frame.
        """
        if not frame:
            return []

        mono = downmix_to_mono(frame, self.channels)
        resampled = resample_pcm16(mono, self.source_rate, INPUT_SAMPLE_RATE)
        self._buffer.extend(resampled)

        chunks: List[bytes] = []
        while len(self._buffer) >= self.chunk_bytes:
            chunks.append(bytes(self._buffer[:self.chunk_bytes]))
            del self._buffer[:self.chunk_bytes]
        return chunks

    def flush(self) -> bytes:
        """
        Returns and clears the partial chunk, for use when a speaker's track ends.

        The tail of the last utterance is otherwise lost, which is audible precisely
        where it hurts: the final word of a turn.
        """
        remainder = bytes(self._buffer)
        self._buffer.clear()
        return remainder


class AgoraPublishAdapter:
    """
    Turns Gemini's 24 kHz translated PCM into Agora publish frames (TASK-11.2).

    One adapter per translated track, for the same buffering reason as the encoder.
    """

    def __init__(
        self,
        publish_rate: int = DEFAULT_PUBLISH_SAMPLE_RATE,
        frame_ms: int = DEFAULT_PUBLISH_FRAME_MS,
        source_rate: int = GEMINI_OUTPUT_SAMPLE_RATE
    ):
        """
        Initialize the adapter for one translated output track.

        Algorithm:
        1. Record the Gemini output rate and the Agora publish format.
        2. Derive the publish frame size in bytes.
        3. Start with an empty carry-over buffer.
        """
        self.publish_rate = publish_rate
        self.frame_ms = frame_ms
        self.source_rate = source_rate
        self.frame_bytes = publish_rate * 2 * frame_ms // 1000
        self._buffer = bytearray()

    def push(self, pcm: bytes) -> List[bytes]:
        """
        Accepts translated PCM and returns whatever whole publish frames it completed.

        Algorithm:
        1. Resample from the Gemini output rate to the Agora publish rate.
        2. Append to the carry-over buffer.
        3. Slice off every whole frame; keep the remainder.
        """
        if not pcm:
            return []

        resampled = resample_pcm16(pcm, self.source_rate, self.publish_rate)
        self._buffer.extend(resampled)

        frames: List[bytes] = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[:self.frame_bytes]))
            del self._buffer[:self.frame_bytes]
        return frames

    def flush(self) -> bytes:
        """Returns and clears the partial frame, for use when a leg closes."""
        remainder = bytes(self._buffer)
        self._buffer.clear()
        return remainder
