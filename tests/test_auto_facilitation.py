"""
Summary:
    test_auto_facilitation.py is the executable specification for REQ-26 (an automatic
    speaking-balance nudge) and REQ-27 (an automatic topic rotation when the room goes
    quiet) - the first genuinely new *server-side* behavior of the Phase 15/16 UI track.

    Both requirements exist because the equivalents already visible in the UI are not
    real. The teacher dock's "Nudge Turn" and "Break Silence" buttons (app.js) dispatch
    hardcoded local messages through `streamManager.handleStreamMessage`; they never
    reach the backend, they name whoever the fixture names, and they fire only when a
    human clicks them. That is a demo of a facilitation feature, not one. So these tests
    assert the two properties that distinguish the real thing:

    - the nudge names the participant who is *measured* to be quiet (REQ-23's tracker is
      the only admissible source - nothing derived from turn count or text length), and
    - both fire on their own, from server-side detection, and each fires *once* per
      condition rather than repeating on every recomputation or every timer tick.

    The suppression half matters as much as the firing half: a facilitation cue that
    repeats every time a segment is recorded is noise a participant learns to ignore,
    which costs the feature exactly the attention it was built to earn.

Covers:
    - REQ-26 detection: threshold, minimum measured window, single-speaker abstention,
      one-shot firing, and re-arming only after the quiet participant recovers.
    - REQ-26 publication: exactly one `teacher_alert` naming the real participant.
    - REQ-27 timing: a per-channel silence window over the existing VAD/turn signal.
    - REQ-27 publication: `topic_prompt` + `teacher_alert` from the previously unused
      `TeachingAgent.generate_silence_breaker()`.
    - REQ-27 API: `POST /api/session/topic/generate`, governed like every session
      endpoint (REQ-16), so the manual and automatic paths share one generator.
"""

import unittest
from unittest.mock import patch

from src.server import app, server_instance
from src.sessions.speaking_balance import SpeakingBalanceTracker
from src.sessions.silence_monitor import SilenceMonitor

CHANNEL = "facilitation-channel"


class FakeClock:
    """A hand-advanced monotonic clock, so a silence window costs no wall time."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestSpeakingBalanceNudgeDetection(unittest.TestCase):
    """Test suite for deciding *whether* to nudge, and whom (REQ-26, TASK-16.8)."""

    def setUp(self):
        """A tracker that nudges under 25% once a minute of speech is on record."""
        self.tracker = SpeakingBalanceTracker(
            nudge_threshold_pct=25, nudge_min_total_ms=60000
        )

    def test_silence_nominates_nobody(self):
        """Verify an unmeasured channel produces no candidate rather than a default one."""
        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_a_short_conversation_is_not_yet_evidence_of_imbalance(self):
        """
        Verify the minimum window is respected.

        Ten seconds into a conversation one participant is always "behind", because
        somebody has to speak first. Nudging on that measures who opened the call, not
        who is being left out.
        """
        self.tracker.record(CHANNEL, "Kenji", 9000)
        self.tracker.record(CHANNEL, "Aarav", 1000)

        self.assertEqual(self.tracker.percentages(CHANNEL), {"Kenji": 90, "Aarav": 10})
        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_a_sustained_low_share_nominates_that_participant(self):
        """Verify the quiet participant is named once enough speech has been measured."""
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)

        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")

    def test_a_balanced_conversation_nominates_nobody(self):
        """Verify a conversation already being shared is left alone."""
        self.tracker.record(CHANNEL, "Kenji", 55000)
        self.tracker.record(CHANNEL, "Aarav", 45000)

        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_a_share_exactly_at_the_threshold_is_not_low(self):
        """Verify the threshold is a floor, not a trigger: 25% of a pair is fine."""
        self.tracker.record(CHANNEL, "Kenji", 75000)
        self.tracker.record(CHANNEL, "Aarav", 25000)

        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_a_lone_speaker_is_not_an_imbalance(self):
        """
        Verify a single measured speaker nominates nobody.

        With one participant on record there is no share to be low relative to - the
        other person may not have joined yet. Nudging the absent is how an automatic
        facilitator becomes something participants mute.
        """
        self.tracker.record(CHANNEL, "Kenji", 100000)

        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_the_quietest_of_several_low_shares_is_nominated(self):
        """Verify one nudge addresses one person - the one furthest behind."""
        self.tracker.record(CHANNEL, "Kenji", 80000)
        self.tracker.record(CHANNEL, "Aarav", 15000)
        self.tracker.record(CHANNEL, "Priya", 5000)

        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Priya")

    def test_a_participant_is_nominated_only_once(self):
        """
        Verify the nudge does not repeat on every recomputation.

        `record()` runs on every measured segment, so an unsuppressed check would fire a
        `teacher_alert` several times a minute at exactly the moment somebody is already
        struggling to get a word in.
        """
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)

        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")
        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

        self.tracker.record(CHANNEL, "Kenji", 10000)
        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

    def test_a_recovered_participant_can_be_nominated_again(self):
        """Verify the nudge re-arms once the cue has actually worked, and not before."""
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)
        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")

        # Aarav answers at length: back above the threshold, so the cue is re-armed.
        self.tracker.record(CHANNEL, "Aarav", 60000)
        self.assertIsNone(self.tracker.nudge_candidate(CHANNEL))

        # ...and then goes quiet again while Kenji talks on.
        self.tracker.record(CHANNEL, "Kenji", 400000)
        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")

    def test_channels_are_suppressed_independently(self):
        """Verify one room's nudge does not silence another room's."""
        for channel in (CHANNEL, "another-room"):
            self.tracker.record(channel, "Kenji", 90000)
            self.tracker.record(channel, "Aarav", 10000)

        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")
        self.assertEqual(self.tracker.nudge_candidate("another-room"), "Aarav")

    def test_reset_re_arms_the_channel(self):
        """Verify a new session on the channel may nudge, even about the same person."""
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)
        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")

        self.tracker.reset(CHANNEL)
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)

        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")

    def test_the_quietest_speaker_can_be_read_without_consuming_the_nudge(self):
        """
        Verify the pure read leaves REQ-26's one-shot alone.

        REQ-27's topic generator needs the same answer - who should this prompt be aimed
        at - but asking must not silently spend the nudge, or the two cues would compete
        for one trigger and whichever ran first would mute the other.
        """
        self.tracker.record(CHANNEL, "Kenji", 90000)
        self.tracker.record(CHANNEL, "Aarav", 10000)

        self.assertEqual(self.tracker.quietest_speaker(CHANNEL), "Aarav")
        self.assertEqual(self.tracker.quietest_speaker(CHANNEL), "Aarav")
        self.assertEqual(self.tracker.nudge_candidate(CHANNEL), "Aarav")
        self.assertEqual(self.tracker.quietest_speaker(CHANNEL), "Aarav")

    def test_the_quietest_speaker_abstains_on_the_same_terms_as_the_nudge(self):
        """Verify the pure read applies the window and single-speaker guards too."""
        self.tracker.record(CHANNEL, "Kenji", 9000)
        self.tracker.record(CHANNEL, "Aarav", 1000)
        self.assertIsNone(self.tracker.quietest_speaker(CHANNEL))

        self.tracker.reset(CHANNEL)
        self.tracker.record(CHANNEL, "Kenji", 100000)
        self.assertIsNone(self.tracker.quietest_speaker(CHANNEL))

    def test_the_thresholds_are_configurable(self):
        """Verify a deployment can tune how patient the facilitator is."""
        strict = SpeakingBalanceTracker(nudge_threshold_pct=40, nudge_min_total_ms=1000)
        strict.record(CHANNEL, "Kenji", 70000)
        strict.record(CHANNEL, "Aarav", 30000)

        self.assertEqual(strict.nudge_candidate(CHANNEL), "Aarav")

    def test_the_defaults_come_from_the_environment(self):
        """Verify the documented environment variables are the tuning surface."""
        with patch.dict("os.environ", {
            "SPEAKING_BALANCE_NUDGE_THRESHOLD_PCT": "40",
            "SPEAKING_BALANCE_NUDGE_MIN_MS": "1000"
        }):
            tracker = SpeakingBalanceTracker.from_env()

        self.assertEqual(tracker.nudge_threshold_pct, 40)
        self.assertEqual(tracker.nudge_min_total_ms, 1000)

    def test_an_unparseable_environment_value_falls_back_to_the_default(self):
        """Verify a typo in a deployment variable does not disable the facilitator."""
        with patch.dict("os.environ", {
            "SPEAKING_BALANCE_NUDGE_THRESHOLD_PCT": "a quarter",
            "SPEAKING_BALANCE_NUDGE_MIN_MS": ""
        }):
            tracker = SpeakingBalanceTracker.from_env()

        self.assertEqual(tracker.nudge_threshold_pct, 25)
        self.assertEqual(tracker.nudge_min_total_ms, 60000)


class TestSilenceMonitor(unittest.TestCase):
    """Test suite for the per-channel silence window (REQ-27, TASK-16.9)."""

    def setUp(self):
        """A 45-second window on a hand-advanced clock."""
        self.clock = FakeClock()
        self.monitor = SilenceMonitor(silence_seconds=45.0, clock=self.clock)

    def test_a_channel_nobody_is_watching_never_trips(self):
        """Verify silence is only meaningful inside a session that started."""
        self.clock.advance(600)

        self.assertFalse(self.monitor.is_silent(CHANNEL))
        self.assertFalse(self.monitor.claim(CHANNEL))

    def test_a_fresh_session_is_not_immediately_silent(self):
        """Verify the window is measured from the session's start, not from zero."""
        self.monitor.start(CHANNEL)

        self.assertFalse(self.monitor.is_silent(CHANNEL))

    def test_the_window_trips_after_the_configured_quiet(self):
        """Verify an unspoken session trips once the window elapses."""
        self.monitor.start(CHANNEL)
        self.clock.advance(45)

        self.assertTrue(self.monitor.is_silent(CHANNEL))

    def test_speech_restarts_the_window(self):
        """
        Verify measured activity is what keeps the room "alive".

        This is REQ-27's reuse discipline: the signal is the same VAD/turn boundary
        REQ-02 already produces and REQ-23 already counts, not a new measurement source
        invented for this feature.
        """
        self.monitor.start(CHANNEL)
        self.clock.advance(40)
        self.monitor.note_activity(CHANNEL)
        self.clock.advance(40)

        self.assertFalse(self.monitor.is_silent(CHANNEL))

        self.clock.advance(10)
        self.assertTrue(self.monitor.is_silent(CHANNEL))

    def test_a_claim_consumes_the_window(self):
        """Verify the trip fires once, then needs another full window to fire again."""
        self.monitor.start(CHANNEL)
        self.clock.advance(50)

        self.assertTrue(self.monitor.claim(CHANNEL))
        self.assertFalse(self.monitor.claim(CHANNEL))

        self.clock.advance(45)
        self.assertTrue(self.monitor.claim(CHANNEL))

    def test_activity_on_one_channel_does_not_revive_another(self):
        """Verify two concurrent rooms are timed independently."""
        self.monitor.start(CHANNEL)
        self.monitor.start("another-room")
        self.clock.advance(40)
        self.monitor.note_activity("another-room")
        self.clock.advance(10)

        self.assertTrue(self.monitor.is_silent(CHANNEL))
        self.assertFalse(self.monitor.is_silent("another-room"))

    def test_a_stopped_channel_stops_tripping(self):
        """Verify an ended session's channel is no longer facilitated."""
        self.monitor.start(CHANNEL)
        self.monitor.stop(CHANNEL)
        self.clock.advance(600)

        self.assertFalse(self.monitor.claim(CHANNEL))

    def test_watched_channels_are_enumerable(self):
        """Verify a poller can find the channels to check without a registry lookup."""
        self.monitor.start(CHANNEL)
        self.monitor.start("another-room")
        self.monitor.stop("another-room")

        self.assertEqual(self.monitor.watched_channels(), [CHANNEL])

    def test_the_window_comes_from_the_environment(self):
        """Verify the documented environment variable is the tuning surface."""
        with patch.dict("os.environ", {"SILENCE_TOPIC_ROTATION_SECONDS": "20"}):
            monitor = SilenceMonitor.from_env()

        self.assertEqual(monitor.silence_seconds, 20.0)

    def test_an_unparseable_window_falls_back_to_the_default(self):
        """Verify a typo does not turn the window into zero and fire continuously."""
        with patch.dict("os.environ", {"SILENCE_TOPIC_ROTATION_SECONDS": "soon"}):
            monitor = SilenceMonitor.from_env()

        self.assertEqual(monitor.silence_seconds, 45.0)


class AutoFacilitationServerTestCase(unittest.TestCase):
    """Shared setup: a live two-participant session on a clean server."""

    def setUp(self):
        """Start from an empty registry, store, tracker, monitor, and packet history."""
        self.client = app.test_client()
        self.server = server_instance
        self.clock = FakeClock()
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.server.speaking_balance.reset()
        self.server.silence_monitor.reset()
        self.server.silence_monitor.clock = self.clock
        self.client.post("/api/session/start", json={
            "channel": CHANNEL,
            "mode": "language_learning",
            "participants": ["Kenji", "Aarav"],
            "languages": ["ja", "hi"]
        })
        # Cleared *after* the start: `start_session()` broadcasts the default welcome
        # topic, which is pre-existing behavior and not what these tests are about.
        self.server.data_stream.packet_history.clear()

    def tearDown(self):
        """Put the shared monitor back on a real clock for the next test module."""
        self.server.silence_monitor.reset()

    def published(self, event_type):
        """Returns the payloads broadcast under one event type."""
        return [
            packet.payload for packet in self.server.data_stream.packet_history
            if packet.event_type == event_type
        ]

    def report(self, speaker_id, duration_ms):
        """Reports one measured speech segment as that speaker."""
        return self.client.post("/api/session/speaking", json={
            "channel": CHANNEL, "actor": speaker_id,
            "speaker_id": speaker_id, "duration_ms": duration_ms
        })


class TestAutomaticNudgePublication(AutoFacilitationServerTestCase):
    """Test suite for broadcasting the nudge the tracker nominated (REQ-26)."""

    def test_a_sustained_low_share_publishes_one_nudge_naming_that_participant(self):
        """
        Verify the alert reaches the room and names the person actually measured.

        The teacher dock's existing "Nudge Turn" button addresses a hardcoded name from a
        frontend fixture. REQ-26 is the opposite: server-side detection, over the real
        data stream, about whoever the measurement says is being left out.
        """
        self.report("Kenji", 90000)
        self.report("Aarav", 10000)

        alerts = self.published("teacher_alert")
        self.assertEqual(len(alerts), 1)
        self.assertIn("Aarav", alerts[0]["message"])
        self.assertTrue(alerts[0]["alert_required"])

    def test_a_balanced_conversation_publishes_no_nudge(self):
        """Verify a shared conversation is not interrupted to be told it is fine."""
        self.report("Kenji", 55000)
        self.report("Aarav", 45000)

        self.assertFalse(self.published("teacher_alert"))

    def test_a_short_conversation_publishes_no_nudge(self):
        """Verify the minimum measured window governs the published path too."""
        self.report("Kenji", 9000)
        self.report("Aarav", 1000)

        self.assertFalse(self.published("teacher_alert"))

    def test_the_nudge_does_not_repeat_on_every_later_segment(self):
        """Verify continued imbalance produces one alert, not one per segment."""
        self.report("Kenji", 90000)
        self.report("Aarav", 10000)
        for _ in range(5):
            self.report("Kenji", 20000)

        self.assertEqual(len(self.published("teacher_alert")), 1)

    def test_the_balance_is_still_published_alongside_the_nudge(self):
        """Verify REQ-23's gauge is unaffected by REQ-26 riding on the same call."""
        self.report("Kenji", 90000)
        self.report("Aarav", 10000)

        balances = self.published("speaking_balance")
        self.assertEqual(len(balances), 2)
        self.assertEqual(balances[-1]["speaker_percentages"], {"Kenji": 90, "Aarav": 10})


class TestAutomaticTopicRotation(AutoFacilitationServerTestCase):
    """Test suite for rotating the topic when the room goes quiet (REQ-27)."""

    def test_a_live_session_starts_being_watched(self):
        """Verify starting a session arms the silence window for its channel."""
        self.assertIn(CHANNEL, self.server.silence_monitor.watched_channels())

    def test_stopping_a_session_stops_watching_it(self):
        """Verify an ended session's channel is no longer facilitated."""
        self.client.post("/api/session/stop", json={"channel": CHANNEL})

        self.assertNotIn(CHANNEL, self.server.silence_monitor.watched_channels())

    def test_a_quiet_room_publishes_a_generated_topic_and_an_alert(self):
        """
        Verify the previously unused generator finally reaches the room.

        `TeachingAgent.generate_silence_breaker()` has existed since Phase 3 with no
        caller: the only thing that ever moved the topic card was a frontend fixture.
        REQ-27 makes the trip real - detected on the server, generated by the agent,
        published as the same `topic_prompt`/`teacher_alert` shapes the widgets already
        render.
        """
        self.clock.advance(60)
        result = self.server.check_silence(CHANNEL)

        self.assertIsNotNone(result)
        topics = self.published("topic_prompt")
        self.assertEqual(len(topics), 1)
        self.assertTrue(topics[0]["topic_title"])
        self.assertTrue(topics[0]["prompt"])
        self.assertEqual(len(self.published("teacher_alert")), 1)

    def test_a_room_still_inside_the_window_is_left_alone(self):
        """Verify a pause between sentences is not treated as a stalled conversation."""
        self.clock.advance(10)

        self.assertIsNone(self.server.check_silence(CHANNEL))
        self.assertFalse(self.published("topic_prompt"))

    def test_measured_speech_postpones_the_rotation(self):
        """Verify a talking room is never interrupted with a new topic."""
        self.clock.advance(40)
        self.report("Kenji", 5000)
        self.clock.advance(40)

        self.assertIsNone(self.server.check_silence(CHANNEL))

    def test_the_rotation_does_not_repeat_until_the_window_elapses_again(self):
        """Verify a still-quiet room gets one topic, not one per poll tick."""
        self.clock.advance(60)
        self.server.check_silence(CHANNEL)
        self.clock.advance(10)
        self.server.check_silence(CHANNEL)

        self.assertEqual(len(self.published("topic_prompt")), 1)

        self.clock.advance(45)
        self.server.check_silence(CHANNEL)
        self.assertEqual(len(self.published("topic_prompt")), 2)

    def test_the_prompt_addresses_the_quieter_participant(self):
        """
        Verify the generated prompt is aimed at whoever the measurement says is quiet.

        `generate_silence_breaker` already accepts an `inactive_speaker`; feeding it the
        tracker's own answer is what keeps REQ-27 on measured signal instead of a name
        chosen by the caller.
        """
        self.report("Kenji", 90000)
        self.report("Aarav", 10000)
        self.clock.advance(60)

        with patch.object(
            self.server.agent, "generate_silence_breaker",
            wraps=self.server.agent.generate_silence_breaker
        ) as generator:
            self.server.check_silence(CHANNEL)

        self.assertEqual(generator.call_args.kwargs.get("inactive_speaker"), "Aarav")

    def test_the_watcher_can_be_disabled_for_a_deployment(self):
        """
        Verify `ECHOSPHERE_SILENCE_WATCHER=0` starts no background thread.

        The escape hatch exists because the watcher is the one piece of this feature that
        runs without anybody asking it to; a deployment that wants the manual generator
        and nothing else must be able to say so, and the test suite itself relies on it.
        """
        self.assertFalse(self.server.start_silence_watcher())

    def test_an_unwatched_channel_generates_nothing(self):
        """Verify no session means no facilitation - and no LLM call."""
        self.clock.advance(600)

        self.assertIsNone(self.server.check_silence("no-such-channel"))


class TestTopicGenerationApi(AutoFacilitationServerTestCase):
    """Test suite for generating a topic on demand (REQ-27, TASK-16.9)."""

    def test_generating_a_topic_publishes_it_to_every_participant(self):
        """
        Verify the "Generate Topics" button and the silence trip share one generator.

        Before this endpoint the button only cycled a four-item frontend fixture, so the
        manual path and the automatic path would have disagreed about what a topic is.
        """
        response = self.client.post("/api/session/topic/generate", json={
            "channel": CHANNEL, "actor": "Kenji"
        })
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["success"])
        self.assertTrue(body["topic"]["topic_title"])
        self.assertTrue(body["topic"]["prompt"])
        self.assertEqual(len(self.published("topic_prompt")), 1)

    def test_generating_a_topic_postpones_the_silence_rotation(self):
        """Verify a topic just asked for is not immediately replaced by an automatic one."""
        self.clock.advance(44)
        self.client.post("/api/session/topic/generate", json={
            "channel": CHANNEL, "actor": "Kenji"
        })
        self.clock.advance(10)

        self.assertIsNone(self.server.check_silence(CHANNEL))
        self.assertEqual(len(self.published("topic_prompt")), 1)

    def test_a_non_participant_cannot_generate_a_topic(self):
        """Verify the endpoint is governed like every other session endpoint (REQ-16)."""
        response = self.client.post("/api/session/topic/generate", json={
            "channel": CHANNEL, "actor": "Stranger"
        })

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.published("topic_prompt"))

    def test_an_unknown_session_is_a_404(self):
        """Verify a topic cannot be pushed into a channel that has no session."""
        response = self.client.post("/api/session/topic/generate", json={
            "channel": "no-such-channel", "actor": "Kenji"
        })

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
