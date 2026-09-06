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
import os
import threading
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("echosphere.sessions.speaking_balance")

# Share below which a participant counts as being left out of the conversation (REQ-26).
# A floor rather than a trigger: at 25% a pair is merely uneven, and a trio is exactly
# even, so the nudge is reserved for someone measurably further behind than that.
DEFAULT_NUDGE_THRESHOLD_PCT = 25

# How much speech must be on record before a share means anything (REQ-26). Ten seconds
# into a conversation somebody is always "behind", because somebody has to speak first;
# nudging on that measures who opened the call, not who is being left out.
DEFAULT_NUDGE_MIN_TOTAL_MS = 60000


class SpeakingBalanceTracker:
    """
    Accumulates measured speaking time per channel and derives each speaker's share.

    Thread-safe by construction: speech boundaries arrive from a transcription worker, a
    request thread, or a background scaffolding task, and a read-modify-write without a
    lock loses segments under exactly the load that makes the gauge worth watching.
    """

    def __init__(
        self,
        nudge_threshold_pct: int = DEFAULT_NUDGE_THRESHOLD_PCT,
        nudge_min_total_ms: int = DEFAULT_NUDGE_MIN_TOTAL_MS
    ):
        """
        Initialize an empty tracker.

        Algorithm:
        1. Create the channel -> {speaker -> milliseconds} table.
        2. Create the lock guarding it against concurrent recording.
        3. Record the REQ-26 nudge thresholds and the per-channel suppression set that
           keeps an automatic nudge from repeating on every recorded segment.
        """
        self._durations: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self.nudge_threshold_pct = int(nudge_threshold_pct)
        self.nudge_min_total_ms = int(nudge_min_total_ms)
        # channel -> speakers already nudged and not yet recovered (REQ-26).
        self._nudged: Dict[str, Set[str]] = {}

    @classmethod
    def from_env(cls) -> "SpeakingBalanceTracker":
        """
        Builds a tracker whose REQ-26 thresholds come from the environment.

        `SPEAKING_BALANCE_NUDGE_THRESHOLD_PCT` and `SPEAKING_BALANCE_NUDGE_MIN_MS` are
        the documented tuning surface. A typo falls back to the default rather than to
        zero: an unparseable minimum window that became `0` would nudge somebody two
        seconds into their first conversation, which is a worse failure than the
        misconfiguration behind it.
        """
        def read(name: str, default: int) -> int:
            raw = os.getenv(name, "")
            try:
                return int(raw) if raw else default
            except (TypeError, ValueError):
                logger.warning(
                    f"{name}={raw!r} is not an integer. Using the default {default}."
                )
                return default

        return cls(
            nudge_threshold_pct=read(
                "SPEAKING_BALANCE_NUDGE_THRESHOLD_PCT", DEFAULT_NUDGE_THRESHOLD_PCT
            ),
            nudge_min_total_ms=read(
                "SPEAKING_BALANCE_NUDGE_MIN_MS", DEFAULT_NUDGE_MIN_TOTAL_MS
            )
        )

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

    def quietest_speaker(self, channel: str) -> Optional[str]:
        """
        Returns who is measurably being left out, without consuming anything (REQ-26).

        Same judgement as `nudge_candidate` - the threshold, the minimum measured window,
        and the refusal to call a lone speaker an imbalance - but as a pure read. Two
        callers need this answer for different reasons: the nudge fires once and must
        suppress itself, while REQ-27's topic generator only wants to know who to address
        and must not silently eat the other feature's one-shot.
        """
        percentages = self.percentages(str(channel))
        if self.total_ms(str(channel)) < self.nudge_min_total_ms or len(percentages) < 2:
            return None

        speaker, share = min(percentages.items(), key=lambda item: (item[1], item[0]))
        return speaker if share < self.nudge_threshold_pct else None

    def nudge_candidate(self, channel: str) -> Optional[str]:
        """
        Nominates the participant who should be nudged to speak, once (REQ-26).

        Algorithm:
        1. Refuse to judge a conversation that has not been measured long enough.
        2. Refuse to judge a channel with a single measured speaker.
        3. Take the lowest share; if it is below the threshold and that speaker is not
           already under suppression, nominate them and suppress further nominations.
        4. Whether or not anyone is nominated, release suppression for everybody now back
           at or above the threshold, so the cue re-arms when it has actually worked.

        Steps 1-2 are the difference between a facilitation cue and an accusation. With
        one speaker on record there is no share to be low *relative to* - the other
        person may not have joined yet - and inside the first minute the "quiet" one is
        usually just the person who did not open the call.

        Step 3's suppression is why this is a method rather than an inline comparison in
        the server: `record()` runs on every measured segment, so an unsuppressed check
        would publish a `teacher_alert` several times a minute at exactly the moment
        somebody is already struggling to get a word in.
        """
        key = str(channel)
        # Both reads happen before the lock is taken: they acquire it themselves, and
        # `threading.Lock` is not reentrant.
        percentages = self.percentages(key)
        candidate = self.quietest_speaker(key)

        with self._lock:
            nudged = self._nudged.setdefault(key, set())

            # Step 4 first: recovery is re-armed even in the runs that nominate nobody,
            # since the usual reason nobody is nominated is that everybody recovered.
            for speaker, share in percentages.items():
                if share >= self.nudge_threshold_pct:
                    nudged.discard(speaker)

            if candidate is None or candidate in nudged:
                return None

            nudged.add(candidate)
            return candidate

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
                self._nudged.clear()
            else:
                self._durations.pop(str(channel), None)
                # The suppression goes with the tally: a new session on this channel may
                # nudge, even about the same person the last one already nudged.
                self._nudged.pop(str(channel), None)
