"""
Summary:
    speaking_balance.py accumulates how long each participant actually spoke, per channel
    (REQ-23).

    It exists because the only speaking-balance numbers this system ever showed were
    invented: `runSimulationDemo()` in app.js dispatched 70/30 and then 52/48 on a timer,
    with nothing behind them. REQ-23 asks for the real thing, and a gauge that claims to
    measure participation has to be built on measurements - so this module accepts
    durations that were observed (a transcribed audio segment, a client-side VAD
    boundary) and refuses to manufacture the rest. Silence reports nothing rather than a
    plausible-looking split.

    Kept beside `SessionService` rather than inside it: the service owns a session's
    identity and mode, which are decided once, while this is a running tally updated from
    whichever thread noticed a speech boundary.

Key Classes:
    - SpeakingBalanceTracker: per-channel speaking-time accumulator and share calculator.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("echosphere.sessions.speaking_balance")


class SpeakingBalanceTracker:
    """
    Accumulates measured speaking time per channel and derives each speaker's share.

    Thread-safe by construction: speech boundaries arrive from a transcription worker, a
    request thread, or a background scaffolding task, and a read-modify-write without a
    lock loses segments under exactly the load that makes the gauge worth watching.
    """

    def __init__(self):
        """
        Initialize an empty tracker.

        Algorithm:
        1. Create the channel -> {speaker -> milliseconds} table.
        2. Create the lock guarding it against concurrent recording.
        """
        self._durations: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

    def record(self, channel: str, speaker_id: str, duration_ms: int) -> Dict[str, int]:
        """
        Adds one measured speech segment and returns the updated shares (REQ-23).

        Algorithm:
        1. Drop anything that is not a positive, attributable segment.
        2. Add the duration to that speaker's running total on this channel.
        3. Return the recalculated percentages, so the caller can publish what it just
           recorded without a second lookup.

        Step 1 is not defensive tidiness: a VAD boundary that fires twice on one instant,
        or a client sending a negative delta after a clock adjustment, would otherwise
        register a participant at 0% and dilute everyone else's real share.
        """
        speaker = str(speaker_id or "").strip()
        try:
            milliseconds = int(duration_ms)
        except (TypeError, ValueError):
            milliseconds = 0

        if not speaker or milliseconds <= 0:
            return self.percentages(channel)

        with self._lock:
            channel_durations = self._durations.setdefault(str(channel), {})
            channel_durations[speaker] = channel_durations.get(speaker, 0) + milliseconds

        return self.percentages(channel)

    def durations(self, channel: str) -> Dict[str, int]:
        """Returns the raw milliseconds recorded per speaker, for absolute-time views."""
        with self._lock:
            return dict(self._durations.get(str(channel), {}))

    def total_ms(self, channel: str) -> int:
        """Returns the total measured speaking time on a channel."""
        return sum(self.durations(channel).values())

    def percentages(self, channel: str) -> Dict[str, int]:
        """
        Returns each speaker's integer share of the measured speaking time (REQ-23).

        Algorithm:
        1. A channel where nobody has been measured yet returns nothing at all - not an
           even split. "No data yet" and "you two are perfectly balanced" are different
           statements, and only one of them is true before anyone speaks.
        2. Floor each speaker's exact share.
        3. Hand the leftover points to the largest fractional remainders, so the shares
           total exactly 100.

        Step 3 is what the previous per-speaker `int()` truncation got wrong: three equal
        speakers rendered as 33/33/33, and a balance gauge that visibly fails to fill
        reads as a broken measurement rather than as rounding.
        """
        durations = self.durations(channel)
        total = sum(durations.values())
        if total <= 0:
            return {}

        shares: Dict[str, int] = {}
        remainders: List[Tuple[float, int, str]] = []
        assigned = 0

        for speaker, milliseconds in durations.items():
            exact = milliseconds * 100 / total
            floor = int(exact)
            shares[speaker] = floor
            assigned += floor
            remainders.append((exact - floor, milliseconds, speaker))

        # Ties break toward the longer speaker, then toward whoever was seen first
        # (`sorted` is stable), so the same durations always produce the same gauge.
        remainders.sort(key=lambda item: (-item[0], -item[1]))
        for _, _, speaker in remainders[:max(0, 100 - assigned)]:
            shares[speaker] += 1

        return shares

    def reset(self, channel: Optional[str] = None) -> None:
        """
        Clears one channel's tally, or every channel when none is named.

        A channel outlives the sessions held on it, so a new session must start from
        silence: inheriting the previous pair's ratio would show a new learner a gauge
        describing somebody else's conversation (the same failure `reset_state` exists
        for on `TeachingAgent`).
        """
        with self._lock:
            if channel is None:
                self._durations.clear()
            else:
                self._durations.pop(str(channel), None)
