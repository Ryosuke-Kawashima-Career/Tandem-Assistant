"""
Summary:
    test_speaking_balance.py is the executable specification for REQ-23: each
    participant's share of the speaking time is computed by the server from real
    speech-activity data and broadcast to every participant, not just the teacher dock.

    The requirement's sharp edge is the word *real*. The only `speaking_balance` events
    this system has ever emitted came from `runSimulationDemo()`'s hardcoded fixture in
    app.js - 70/30, then 52/48, on a timer, regardless of who said anything. A gauge that
    moves whether or not you speak is worse than no gauge: it is a motivation cue
    (REQ-23) built on a number nobody earned. So these tests assert both halves - that
    measured speech produces the right share, and that unmeasured speech produces
    *nothing* rather than a plausible-looking guess.

Covers:
    - REQ-23 accumulation: real durations, per channel, thread-safe, no fabrication.
    - REQ-23 percentages: exact 100% totals via largest remainder.
    - REQ-23 publication: `speaking_balance` reaches every participant over RTC data.
    - REQ-23 API: `POST /api/session/speaking`, governed like every session endpoint.
    - REQ-16: a non-participant cannot report speech into someone else's session.
"""

import threading
import unittest

from src.server import app, server_instance
from src.sessions.speaking_balance import SpeakingBalanceTracker

CHANNEL = "balance-channel"


class TestSpeakingBalanceTracker(unittest.TestCase):
    """Test suite for the accumulator itself (REQ-23, TASK-13.3)."""

    def setUp(self):
        """Each test starts from an empty tracker."""
        self.tracker = SpeakingBalanceTracker()

    def test_an_unspoken_session_reports_nothing_rather_than_a_plausible_split(self):
        """
        Verify silence produces no percentages at all.

        The predecessor of this tracker returned a synthetic 50/50 whenever the total was
        zero. That is exactly the failure REQ-23 names: a balance nobody produced,
        rendered identically to one somebody did. An empty mapping makes the UI say "no
        data yet", which is the truth.
        """
        self.assertEqual(self.tracker.percentages(CHANNEL), {})
        self.assertEqual(self.tracker.durations(CHANNEL), {})
        self.assertEqual(self.tracker.total_ms(CHANNEL), 0)

    def test_measured_speech_produces_the_real_share(self):
        """Verify the ratio is the measured ratio, not an approximation of one."""
        self.tracker.record(CHANNEL, "Kenji", 6000)
        self.tracker.record(CHANNEL, "Aarav", 4000)

        self.assertEqual(self.tracker.percentages(CHANNEL), {"Kenji": 60, "Aarav": 40})
        self.assertEqual(self.tracker.total_ms(CHANNEL), 10000)

    def test_durations_accumulate_across_turns(self):
        """Verify a speaker's segments add up rather than replacing each other."""
        self.tracker.record(CHANNEL, "Kenji", 1000)
        self.tracker.record(CHANNEL, "Kenji", 3000)
        self.tracker.record(CHANNEL, "Aarav", 4000)

        self.assertEqual(self.tracker.durations(CHANNEL), {"Kenji": 4000, "Aarav": 4000})
        self.assertEqual(self.tracker.percentages(CHANNEL), {"Kenji": 50, "Aarav": 50})

    def test_percentages_always_total_one_hundred(self):
        """
        Verify three equal speakers total 100%, not 99%.

        Truncating each share independently (the old `int(dur / total * 100)`) loses a
        point per speaker on any ratio that does not divide cleanly, and a balance gauge
        that visibly fails to fill is read as a bug in the measurement.
        """
        for speaker in ("Kenji", "Aarav", "Priya"):
            self.tracker.record(CHANNEL, speaker, 1000)

        percentages = self.tracker.percentages(CHANNEL)

        self.assertEqual(sum(percentages.values()), 100)
        self.assertEqual(len(percentages), 3)

    def test_a_rounding_remainder_goes_to_the_longest_speaker(self):
        """Verify the spare point lands on the largest remainder, not on whoever is first."""
        self.tracker.record(CHANNEL, "Quiet", 1000)
        self.tracker.record(CHANNEL, "Talkative", 2000)

        percentages = self.tracker.percentages(CHANNEL)

        self.assertEqual(sum(percentages.values()), 100)
        self.assertGreater(percentages["Talkative"], percentages["Quiet"])

    def test_a_zero_or_negative_duration_registers_nobody(self):
        """
        Verify a non-positive segment is ignored entirely.

        A VAD boundary that fires twice on the same instant, or a client sending a
        negative delta from a clock adjustment, must not invent a participant at 0% -
        which would then dilute everyone else's share.
        """
        self.tracker.record(CHANNEL, "Ghost", 0)
        self.tracker.record(CHANNEL, "Ghost", -500)

        self.assertEqual(self.tracker.percentages(CHANNEL), {})

    def test_an_anonymous_segment_is_ignored(self):
        """Verify speech attributed to nobody is dropped rather than filed under ''."""
        self.tracker.record(CHANNEL, "", 5000)

        self.assertEqual(self.tracker.percentages(CHANNEL), {})

    def test_channels_do_not_share_a_balance(self):
        """Verify two concurrent rooms are measured independently."""
        self.tracker.record(CHANNEL, "Kenji", 5000)
        self.tracker.record("another-room", "Priya", 5000)

        self.assertEqual(list(self.tracker.percentages(CHANNEL)), ["Kenji"])
        self.assertEqual(list(self.tracker.percentages("another-room")), ["Priya"])

    def test_reset_clears_one_channel_and_leaves_the_others(self):
        """Verify a new session on a channel starts from silence, not from history."""
        self.tracker.record(CHANNEL, "Kenji", 5000)
        self.tracker.record("another-room", "Priya", 5000)

        self.tracker.reset(CHANNEL)

        self.assertEqual(self.tracker.percentages(CHANNEL), {})
        self.assertEqual(self.tracker.percentages("another-room"), {"Priya": 100})

    def test_concurrent_segments_are_not_lost(self):
        """
        Verify parallel recording sums exactly.

        Speech boundaries arrive from whichever thread noticed them - a VAD callback, a
        transcription worker, a request thread - so a read-modify-write without a lock
        loses segments under exactly the load that makes the gauge worth watching.
        """
        def record_many(speaker):
            for _ in range(100):
                self.tracker.record(CHANNEL, speaker, 10)

        threads = [
            threading.Thread(target=record_many, args=(name,))
            for name in ("Kenji", "Aarav", "Priya", "Mei")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(self.tracker.total_ms(CHANNEL), 4000)
        self.assertEqual(self.tracker.durations(CHANNEL)["Kenji"], 1000)

    def test_record_returns_the_updated_percentages(self):
        """Verify the caller can publish what it just recorded without a second lookup."""
        self.tracker.record(CHANNEL, "Kenji", 1000)
        percentages = self.tracker.record(CHANNEL, "Aarav", 1000)

        self.assertEqual(percentages, {"Kenji": 50, "Aarav": 50})


class SpeakingBalanceServerTestCase(unittest.TestCase):
    """Shared setup: a live two-participant session on a clean server."""

    def setUp(self):
        """Start from an empty registry, store, balance tracker, and packet history."""
        self.client = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.server.speaking_balance.reset()
        self.server.data_stream.packet_history.clear()
        self.client.post("/api/session/start", json={
            "channel": CHANNEL,
            "mode": "language_learning",
            "participants": ["Kenji", "Aarav"],
            "languages": ["ja", "hi"]
        })

    def published(self, event_type="speaking_balance"):
        """Returns the payloads broadcast under one event type."""
        return [
            packet.payload for packet in self.server.data_stream.packet_history
            if packet.event_type == event_type
        ]


class TestSpeakingBalanceApi(SpeakingBalanceServerTestCase):
    """Test suite for reporting measured speech to the server (REQ-23, TASK-13.3)."""

    def test_a_reported_segment_publishes_the_balance_to_every_participant(self):
        """
        Verify the computed share is broadcast, not merely returned to the reporter.

        REQ-23 extends REQ-07's teacher-only metric to the participant UI: the gauge is a
        motivation cue for the people speaking, so it has to reach both of them - which
        means the data stream, not the HTTP response of whoever happened to report.
        """
        response = self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Kenji", "speaker_id": "Kenji", "duration_ms": 6000
        })
        self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Aarav", "speaker_id": "Aarav", "duration_ms": 4000
        })

        self.assertEqual(response.status_code, 200)
        events = self.published()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["speaker_percentages"], {"Kenji": 60, "Aarav": 40})

    def test_the_response_carries_the_current_percentages(self):
        """Verify a reporting client can draw immediately without waiting for its echo."""
        body = self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Kenji", "speaker_id": "Kenji", "duration_ms": 1000
        }).get_json()

        self.assertTrue(body["success"])
        self.assertEqual(body["speaker_percentages"], {"Kenji": 100})
        self.assertEqual(body["total_ms"], 1000)

    def test_a_missing_duration_is_rejected(self):
        """Verify the server refuses to invent a duration it was not given."""
        response = self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Kenji", "speaker_id": "Kenji"
        })

        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.published())

    def test_an_unparseable_duration_is_rejected(self):
        """Verify a malformed value is a 400 rather than a silently dropped segment."""
        response = self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Kenji",
            "speaker_id": "Kenji", "duration_ms": "a while"
        })

        self.assertEqual(response.status_code, 400)

    def test_a_non_participant_cannot_report_speech(self):
        """
        Verify the endpoint is governed like every other session endpoint (REQ-16).

        Speaking time is a claim about who was talking in someone else's conversation;
        an ungoverned endpoint lets a stranger rewrite that record.
        """
        response = self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Stranger",
            "speaker_id": "Kenji", "duration_ms": 5000
        })

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.published())

    def test_an_unknown_session_is_a_404(self):
        """Verify speech cannot be reported into a channel that has no session."""
        response = self.client.post("/api/session/speaking", json={
            "channel": "no-such-channel", "speaker_id": "Kenji", "duration_ms": 5000
        })

        self.assertEqual(response.status_code, 404)


class TestSpeakingBalanceFromTurns(SpeakingBalanceServerTestCase):
    """Test suite for balance measured from the turn pipeline itself (REQ-23)."""

    def test_an_audio_turn_records_its_real_transcribed_duration(self):
        """
        Verify PCM audio contributes the duration it actually holds.

        `TranscriptionResult.duration_ms` is derived from the byte length at a known
        sample rate (REQ-02), which makes it a measurement rather than an estimate - the
        one signal on the ambient path that REQ-23 may legitimately count.
        """
        # 16 kHz mono PCM16 => 32 bytes per millisecond; 32000 bytes is one second.
        self.server.process_turn(
            speaker_id="Kenji", text_or_audio=b"\x00\x01" * 16000,
            language="ja", channel=CHANNEL
        )

        self.assertEqual(self.server.speaking_balance.total_ms(CHANNEL), 1000)
        self.assertEqual(self.published()[-1]["speaker_percentages"], {"Kenji": 100})

    def test_a_text_turn_contributes_no_invented_duration(self):
        """
        Verify a typed turn adds nothing to the balance.

        There is no honest way to derive speaking time from a string. Estimating one from
        character count would put the gauge back on fabricated data - the exact thing
        REQ-23 replaces - so a text turn is simply not speech that was measured.
        """
        self.client.post("/api/session/turn", json={
            "channel": CHANNEL, "speaker_id": "Kenji", "text": "こんにちは", "language": "ja"
        })

        self.assertEqual(self.server.speaking_balance.total_ms(CHANNEL), 0)
        self.assertFalse(self.published())

    def test_a_new_session_on_the_channel_starts_from_silence(self):
        """
        Verify the previous conversation's balance does not carry into the next one.

        Same failure `TeachingAgent.reset_state` exists for (REQ-LLM-05): the channel is
        reused, the session is not, and inheriting the last pair's ratio would show a new
        learner a gauge describing somebody else.
        """
        self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": "Kenji", "speaker_id": "Kenji", "duration_ms": 9000
        })

        self.client.post("/api/session/start", json={
            "channel": CHANNEL, "mode": "language_learning",
            "participants": ["Mei", "Priya"], "languages": ["ja", "hi"]
        })

        self.assertEqual(self.server.speaking_balance.percentages(CHANNEL), {})


if __name__ == "__main__":
    unittest.main()
