"""
Summary:
    silence_monitor.py tracks how long each channel has been quiet, so a stalled
    conversation can be given a new topic without anyone having to ask (REQ-27).

    It exists because the only thing that has ever moved the "Active Conversation Topic"
    card is a human pressing a button - the teacher dock's "Break Silence" action, which
    is a frontend fixture, and Phase 15's simulation, which cycles a four-item local
    list. `TeachingAgent.generate_silence_breaker()` has been implemented and callable
    since Phase 3 with no caller at all. What was missing was not the generator but the
    *trip*: something on the server that notices the room went quiet.

    The signal is deliberately borrowed, not invented. "Activity" here is the same
    measured VAD/turn boundary REQ-02 produces and REQ-23 counts - the monitor holds a
    timestamp and a window, nothing else. That is the same discipline REQ-23 established
    when it refused to derive speaking time from text length: a facilitator that fires on
    a signal nobody produced is indistinguishable from one on a timer.

    Kept beside `SpeakingBalanceTracker` for the same reason that one is: a session's
    identity is decided once, while this is updated from whichever thread noticed a
    speech boundary.

Key Classes:
    - SilenceMonitor: per-channel quiet-window timer with one-shot claiming.
"""

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("echosphere.sessions.silence_monitor")

# How long a watched channel may stay quiet before the co-teacher offers a new topic.
# Long enough that a thinking pause, a slow sentence, or somebody looking up a word is
# not mistaken for a stalled conversation; short enough that a genuine stall does not
# outlast the participants' willingness to sit through it.
DEFAULT_SILENCE_SECONDS = 45.0

# A window of zero would fire on every poll tick, and a negative one is meaningless.
MIN_SILENCE_SECONDS = 5.0


class SilenceMonitor:
    """
    Per-channel silence timer for the automatic topic rotation (REQ-27).

    Thread-safe by construction: activity arrives from a request thread or a
    transcription worker while a background poller reads the same table.
    """

    def __init__(
        self,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        clock: Callable[[], float] = time.monotonic
    ):
        """
        Initialize an empty monitor.

        Algorithm:
        1. Record the quiet window and the clock that measures it.
        2. Create the channel -> last-activity table and the lock guarding it.

        `clock` is injected so a 45-second window costs a test no wall time, and so the
        default can be monotonic: a wall clock adjusted mid-session would otherwise trip
        the rotation, or suppress it, for reasons that have nothing to do with the room.
        """
        self.silence_seconds = max(MIN_SILENCE_SECONDS, float(silence_seconds))
        self.clock = clock
        self._last_activity: Dict[str, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SilenceMonitor":
        """
        Builds a monitor from `SILENCE_TOPIC_ROTATION_SECONDS` (REQ-27).

        A typo falls back to the default rather than to zero: an unparseable window that
        became `0` would rotate the topic on every poll tick, which is a louder failure
        than the misconfiguration that caused it.
        """
        raw = os.getenv("SILENCE_TOPIC_ROTATION_SECONDS", "")
        try:
            seconds = float(raw) if raw else DEFAULT_SILENCE_SECONDS
        except (TypeError, ValueError):
            logger.warning(
                f"SILENCE_TOPIC_ROTATION_SECONDS={raw!r} is not a number. "
                f"Using the default {DEFAULT_SILENCE_SECONDS}s."
            )
            seconds = DEFAULT_SILENCE_SECONDS
        return cls(silence_seconds=seconds)

    def start(self, channel: str) -> None:
        """
        Begins watching a channel, timing from now.

        Timing from the session's start rather than from zero is what stops a brand-new
        session from being declared stalled before anyone has had a chance to speak.
        """
        with self._lock:
            self._last_activity[str(channel)] = self.clock()

    def stop(self, channel: str) -> None:
        """Stops watching a channel, so an ended session is no longer facilitated."""
        with self._lock:
            self._last_activity.pop(str(channel), None)

    def note_activity(self, channel: str) -> None:
        """
        Restarts the window for a channel that is already being watched.

        Deliberately does not begin watching an unwatched channel: speech measured
        outside a live session must not conjure a facilitation timer for it.
        """
        key = str(channel)
        with self._lock:
            if key in self._last_activity:
                self._last_activity[key] = self.clock()

    def is_silent(self, channel: str) -> bool:
        """Returns whether a watched channel has been quiet for the full window."""
        with self._lock:
            last = self._last_activity.get(str(channel))
        if last is None:
            return False
        return (self.clock() - last) >= self.silence_seconds

    def claim(self, channel: str) -> bool:
        """
        Consumes one silence trip, returning whether the caller may act on it (REQ-27).

        Algorithm:
        1. A channel nobody is watching never trips.
        2. Still inside the window: nothing to claim.
        3. Otherwise restart the window and report the trip.

        Step 3 is what makes this one-shot. A poller ticks far more often than the window
        elapses, so a plain `is_silent` check would generate a new topic every few
        seconds for as long as the room stayed quiet - which is the same room being
        talked at rather than facilitated.
        """
        key = str(channel)
        now = self.clock()
        with self._lock:
            last = self._last_activity.get(key)
            if last is None or (now - last) < self.silence_seconds:
                return False
            self._last_activity[key] = now
        return True

    def watched_channels(self) -> List[str]:
        """Returns the channels currently being watched, for a poller to iterate."""
        with self._lock:
            return list(self._last_activity)

    def reset(self, channel: Optional[str] = None) -> None:
        """Stops watching one channel, or every channel when none is named."""
        with self._lock:
            if channel is None:
                self._last_activity.clear()
            else:
                self._last_activity.pop(str(channel), None)
