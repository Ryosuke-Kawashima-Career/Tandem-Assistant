"""
Summary:
    camera_stream.py holds the one frame the agent is allowed to look at right now
    (REQ-CAM-01).

    REQ-22's camera path is request/response: the participant points the device and
    presses capture, and one frame travels to the vendor and back. That cannot answer
    "what is this?" asked mid-sentence in a live voice turn, because by the time the
    agent decides it wants to look, there is nothing to look at - asking the client to
    capture first would add a browser round trip to the one path the learner waits on in
    silence (REQ-LAT-02/03).

    So the client pushes a small frame every few seconds while Camera Assist is open, and
    this buffer keeps the latest one per channel. The agent's lookup then reads memory
    instead of the network.

    Three properties are deliberate, and each one is a bound rather than a feature:

    - **One frame, replaced in place.** Not a queue and not a recording. Nothing here can
      answer "what did the camera see a minute ago", and a frame stops existing the
      moment a newer one arrives.
    - **It expires on its own.** Past `CAMERA_FRAME_TTL_SECONDS` a frame is reported as
      absent, so a learner who pointed the camera away and then asked a question is told
      the agent cannot see, rather than being told about what used to be in view.
    - **It only ever fills while the participant chose to show something.** The periodic
      push exists between `toggleCamera()` on and `stopCamera()` in the browser and
      nowhere else; closing the panel both stops the pushes and clears the entry. Nothing
      is captured, buffered, or billed without that explicit toggle - the same privacy
      posture `vision.py` states for the button flow, held over the toggle's duration
      instead of over one click.

    Frames live in memory only. Like `CameraVisionTool`, nothing here writes an image to
    disk, and the description it caches carries no copy of the capture.

Key Classes:
    - BufferedFrame: the latest still frame for one channel, plus its cached description.
    - CameraFrameBuffer: the per-channel, self-expiring store the agent looks through.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("echosphere.agent.tools.camera_stream")


def _float_env(name: str, default: float) -> float:
    """Reads a positive float setting, falling back rather than failing to start."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s.", name, raw, default)
        return default
    return value if value > 0 else default


# How long a pushed frame still counts as "what the camera is currently showing".
#
# Long enough to answer a question asked a beat after looking - a learner holds something
# up, then speaks - and short enough that a camera pointed elsewhere a few seconds ago is
# never described as if it were still in view. Past this, the agent behaves exactly as if
# the camera were off.
CAMERA_FRAME_TTL_SECONDS = _float_env("CAMERA_FRAME_TTL_SECONDS", 15.0)

# The whole budget for an agent-initiated lookup, vendor call included.
#
# This sits inside the spoken turn the learner waits on in silence, so it must stay well
# under `CONVOAI_LLM_TIMEOUT_SECONDS` (default 20): exceeding this abandons the lookup and
# the agent speaks a normal reply, which is a worse answer but never a lost turn.
CAMERA_LOOKUP_TIMEOUT_SECONDS = _float_env("CAMERA_LOOKUP_TIMEOUT_SECONDS", 6.0)


@dataclass
class BufferedFrame:
    """
    The latest still frame pushed for one channel.

    `description` is the REQ-CAM-04 cache: the frame is pushed on a timer whether or not
    anyone is asking, so the vendor is paid once per distinct frame rather than once per
    question. A new push builds a new `BufferedFrame`, which is what invalidates it.
    """

    frame: bytes
    mime_type: str
    captured_at: float
    description: Optional[Any] = field(default=None)


class CameraFrameBuffer:
    """
    Keeps the one frame per channel that an agent-initiated lookup may read.

    Guarded by a lock for the same reason `ToolDispatcher` guards its executor: pushes
    arrive on Flask request threads while lookups run on the tool executor, and the two
    touch the same entry.
    """

    def __init__(
        self,
        ttl_seconds: float = CAMERA_FRAME_TTL_SECONDS,
        clock: Optional[Callable[[], float]] = None
    ):
        """
        Initialize an empty buffer over a freshness window.

        The clock is injectable so expiry can be tested without sleeping through it.
        """
        self.ttl_seconds = ttl_seconds
        self._clock = clock or time.time
        self._frames: Dict[str, BufferedFrame] = {}
        self._lock = threading.Lock()

    def put(
        self,
        channel: str,
        frame: bytes,
        mime_type: str = "image/jpeg"
    ) -> Optional[BufferedFrame]:
        """
        Records the newest frame for a channel, replacing whatever it held.

        An empty frame is dropped rather than stored: a client that posted nothing must
        not leave the agent believing it can see.

        Returns:
            The stored entry, or `None` when there was nothing to store.
        """
        data = bytes(frame or b"")
        key = str(channel or "")
        if not data or not key:
            return None

        entry = BufferedFrame(
            frame=data,
            mime_type=(mime_type or "image/jpeg").strip() or "image/jpeg",
            captured_at=self._clock()
        )
        with self._lock:
            self._frames[key] = entry
        return entry

    def get(self, channel: str) -> Optional[BufferedFrame]:
        """
        Returns what the channel's camera is currently showing, or `None`.

        `None` covers every reason the agent should not claim to see: the camera was
        never opened, it was closed, or the last frame is older than the TTL. The caller
        does not distinguish them - all three mean "say you cannot see", not "guess".
        """
        key = str(channel or "")
        if not key:
            return None

        with self._lock:
            entry = self._frames.get(key)
            if entry is None:
                return None
            if self._clock() - entry.captured_at > self.ttl_seconds:
                # Dropped rather than merely hidden, so a camera left closed does not
                # keep its last frame in memory until the process restarts.
                self._frames.pop(key, None)
                return None
            return entry

    def remember_description(self, channel: str, entry: BufferedFrame, description: Any) -> None:
        """
        Caches one description against the exact frame it describes (REQ-CAM-04).

        Keyed on identity rather than on the channel: a frame pushed while the vendor was
        answering about the previous one must not inherit that answer.
        """
        key = str(channel or "")
        with self._lock:
            current = self._frames.get(key)
            if current is entry:
                current.description = description

    def clear(self, channel: str) -> None:
        """
        Forgets a channel's frame at once.

        Called when the participant closes Camera Assist and when their session ends, so
        opting out stops the agent seeing immediately instead of after the TTL.
        """
        with self._lock:
            self._frames.pop(str(channel or ""), None)

    def reset(self) -> None:
        """Empties every channel. Used by tests and by a full server reset."""
        with self._lock:
            self._frames.clear()
