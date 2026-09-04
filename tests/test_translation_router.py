"""
Summary:
    test_translation_router.py is the executable specification for TASK-11.3 and
    TASK-11.4: the router that owns the Gemini Live leg topology, feeds each leg only
    original participant audio, publishes translated audio to the right recipients, and
    emits versioned status/transcript events.

    The topology is mode-dependent and that is the whole point of the module:
    `language_learning` runs two direct peer-pair legs (A -> B's language, B -> A's),
    while `international_work` runs one English-pivot leg per participant fanned out to
    everyone else, so leg count grows with people rather than with language pairs.

Covers:
    - REQ-17 mode-dependent leg topology and immutable original-track routing.
    - REQ-17 feedback prevention: a translated track never re-enters a Gemini leg.
    - REQ-17 independent leg degradation and recovery.
    - REQ-17 versioned `translation.status` / `.input_transcript` / `.output_transcript`
      events, and once-only ingestion of finalized input transcripts.
"""

import unittest
from unittest.mock import MagicMock

from src.rtc.data_stream import DataStreamManager
from src.sessions.models import SessionRecord
from src.translation.gemini_live import LegState, TranslationUnavailableError
from src.translation.router import (
    EVENT_INPUT_TRANSCRIPT,
    EVENT_OUTPUT_TRANSCRIPT,
    EVENT_STATUS,
    TRANSLATION_SCHEMA_VERSION,
    Participant,
    TranslationRouter,
)


PEER_A = Participant(participant_id="peer-a", language="hi")
PEER_B = Participant(participant_id="peer-b", language="ja")
PEER_C = Participant(participant_id="peer-c", language="en")


class FakeLegSession:
    """A stand-in Gemini leg that records the audio the router hands it."""

    def __init__(self, config, **kwargs):
        self.config = config
        self.chunks = []
        self.state = LegState.IDLE
        self.closed = False

    def connect(self):
        self.state = LegState.ACTIVE
        return True

    def send_audio(self, chunk):
        if self.state is not LegState.ACTIVE:
            return False
        self.chunks.append(chunk)
        return True

    def close(self):
        self.closed = True
        self.state = LegState.CLOSED


class RecordingFactory:
    """Builds FakeLegSessions and keeps them addressable by leg id."""

    def __init__(self):
        self.sessions = {}

    def __call__(self, config, **kwargs):
        session = FakeLegSession(config, **kwargs)
        self.sessions[config.leg_id] = session
        return session


def learning_session():
    """A two-peer tandem session: a Hindi speaker and a Japanese speaker."""
    return SessionRecord.create(channel="c-learn", mode="language_learning",
                                languages=["hi", "ja"])


def work_session():
    """A three-person international work call."""
    return SessionRecord.create(channel="c-work", mode="international_work",
                                languages=["hi", "ja", "en"])


def build_router(session, factory=None, data_stream=None, sink=None):
    """Builds a router wired to fakes, so no socket or RTC channel is required."""
    return TranslationRouter(
        session=session,
        data_stream=data_stream,
        session_factory=factory or RecordingFactory(),
        transcript_sink=sink,
    )


class TestLegTopology(unittest.TestCase):
    """TASK-11.3: the mode decides the leg graph (REQ-17)."""

    def test_language_learning_builds_two_direct_peer_pair_legs(self):
        router = build_router(learning_session())

        legs = router.plan_legs([PEER_A, PEER_B])

        self.assertEqual(len(legs), 2)
        by_speaker = {leg.speaker_id: leg for leg in legs}
        # A's audio is interpreted into B's language and delivered only to B.
        self.assertEqual(by_speaker["peer-a"].target_language, "ja-JP")
        self.assertEqual(by_speaker["peer-a"].recipients, ("peer-b",))
        self.assertEqual(by_speaker["peer-b"].target_language, "hi-IN")
        self.assertEqual(by_speaker["peer-b"].recipients, ("peer-a",))

    def test_language_learning_never_targets_the_speakers_own_language(self):
        router = build_router(learning_session())
        for leg in router.plan_legs([PEER_A, PEER_B]):
            speaker_language = {"peer-a": "hi-IN", "peer-b": "ja-JP"}[leg.speaker_id]
            self.assertNotEqual(leg.target_language, speaker_language)

    def test_international_work_builds_one_english_leg_per_participant(self):
        router = build_router(work_session())

        legs = router.plan_legs([PEER_A, PEER_B, PEER_C])

        self.assertEqual(len(legs), 3)
        for leg in legs:
            self.assertTrue(leg.target_language.startswith("en"))

    def test_international_work_fans_each_leg_out_to_every_other_participant(self):
        router = build_router(work_session())

        legs = {leg.speaker_id: leg for leg in router.plan_legs([PEER_A, PEER_B, PEER_C])}

        self.assertEqual(set(legs["peer-a"].recipients), {"peer-b", "peer-c"})
        self.assertNotIn("peer-a", legs["peer-a"].recipients)

    def test_leg_count_grows_with_people_not_with_language_pairs(self):
        """
        Four work participants are four legs, not the twelve ordered pairs the
        language_learning topology would produce. This is why work mode pivots.
        """
        router = build_router(work_session())
        participants = [PEER_A, PEER_B, PEER_C, Participant("peer-d", "ja")]

        self.assertEqual(len(router.plan_legs(participants)), 4)

    def test_leg_ids_are_stable_and_unique(self):
        router = build_router(learning_session())

        first = [leg.leg_id for leg in router.plan_legs([PEER_A, PEER_B])]
        second = [leg.leg_id for leg in router.plan_legs([PEER_A, PEER_B])]

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_a_lone_participant_gets_no_learning_leg(self):
        """Nobody to interpret for: a peer-pair leg needs a pair."""
        router = build_router(learning_session())
        self.assertEqual(router.plan_legs([PEER_A]), [])


class TestLegStartup(unittest.TestCase):
    """TASK-11.3/11.6: starting legs, and failing closed when Gemini is unreachable."""

    def test_start_connects_every_planned_leg(self):
        factory = RecordingFactory()
        router = build_router(learning_session(), factory=factory)

        states = router.start([PEER_A, PEER_B])

        self.assertEqual(len(states), 2)
        self.assertTrue(all(state == LegState.ACTIVE.value for state in states.values()))
        self.assertTrue(all(s.state is LegState.ACTIVE for s in factory.sessions.values()))

    def test_start_emits_a_status_event_per_leg(self):
        data_stream = MagicMock(spec=DataStreamManager)
        router = build_router(learning_session(), data_stream=data_stream)

        router.start([PEER_A, PEER_B])

        statuses = [c for c in data_stream.send_translation_event.call_args_list
                    if c.args[0] == EVENT_STATUS]
        self.assertEqual(len(statuses), 2)

    def test_status_events_carry_the_versioned_session_envelope(self):
        data_stream = MagicMock(spec=DataStreamManager)
        session = learning_session()
        router = build_router(session, data_stream=data_stream)

        router.start([PEER_A, PEER_B])

        _, payload = data_stream.send_translation_event.call_args_list[0].args
        self.assertEqual(payload["schema_version"], TRANSLATION_SCHEMA_VERSION)
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["mode"], session.mode.value)
        self.assertIn("leg_id", payload)
        self.assertIn("state", payload)

    def test_unreachable_gemini_fails_every_leg_closed_without_raising(self):
        """
        REQ-17: session start proceeds without translated audio rather than blocking.
        """
        def broken_factory(config, **kwargs):
            session = FakeLegSession(config)
            def connect():
                session.state = LegState.UNAVAILABLE
                raise TranslationUnavailableError("quota exhausted")
            session.connect = connect
            return session

        data_stream = MagicMock(spec=DataStreamManager)
        router = build_router(learning_session(), factory=broken_factory,
                              data_stream=data_stream)

        states = router.start([PEER_A, PEER_B])

        self.assertTrue(all(state == LegState.UNAVAILABLE.value for state in states.values()))
        for call in data_stream.send_translation_event.call_args_list:
            event_type, payload = call.args
            self.assertEqual(event_type, EVENT_STATUS)
            self.assertEqual(payload["state"], LegState.UNAVAILABLE.value)
            self.assertIn("quota exhausted", payload["reason"])

    def test_a_failed_leg_does_not_stop_the_other_one(self):
        """Legs degrade independently (REQ-17)."""
        factory = RecordingFactory()

        def half_broken(config, **kwargs):
            session = factory(config, **kwargs)
            if config.speaker_id == "peer-a":
                def connect():
                    session.state = LegState.UNAVAILABLE
                    raise TranslationUnavailableError("auth rejected")
                session.connect = connect
            return session

        router = build_router(learning_session(), factory=half_broken)
        states = router.start([PEER_A, PEER_B])

        self.assertEqual(
            sorted(states.values()),
            sorted([LegState.ACTIVE.value, LegState.UNAVAILABLE.value])
        )


class TestAudioRouting(unittest.TestCase):
    """TASK-11.3: original tracks in, translated tracks out, no feedback loop."""

    def setUp(self):
        self.factory = RecordingFactory()
        self.router = build_router(learning_session(), factory=self.factory)
        self.router.start([PEER_A, PEER_B])

    def test_speaker_audio_reaches_only_that_speakers_leg(self):
        """Routing is by immutable speaker identity, never by whoever spoke last."""
        # 100 ms of 16 kHz mono silence: exactly one Gemini chunk.
        self.router.route_audio("peer-a", b"\x00\x00" * 1600, sample_rate=16000)

        leg_a = self.router.leg_for_speaker("peer-a")
        leg_b = self.router.leg_for_speaker("peer-b")
        self.assertEqual(len(self.factory.sessions[leg_a.leg_id].chunks), 1)
        self.assertEqual(self.factory.sessions[leg_b.leg_id].chunks, [])

    def test_audio_is_chunked_to_the_gemini_input_contract(self):
        self.router.route_audio("peer-a", b"\x00\x00" * 4800, sample_rate=48000)

        leg_a = self.router.leg_for_speaker("peer-a")
        chunks = self.factory.sessions[leg_a.leg_id].chunks
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 3200)  # 100 ms of 16 kHz mono PCM16

    def test_translated_track_audio_is_never_fed_back_into_a_leg(self):
        """
        The single most damaging failure mode: a translated track routed as if it were a
        participant would have Gemini interpret its own output, forever.
        """
        leg_a = self.router.leg_for_speaker("peer-a")
        translated_track = self.router.translated_track_id(leg_a.leg_id)

        sent = self.router.route_audio(translated_track, b"\x00\x00" * 1600,
                                       sample_rate=16000)

        self.assertEqual(sent, 0)
        self.assertTrue(self.router.is_translated_track(translated_track))
        self.assertEqual(self.factory.sessions[leg_a.leg_id].chunks, [])

    def test_audio_fans_out_to_every_leg_the_speaker_owns(self):
        """
        A three-peer `language_learning` room gives each speaker one leg per listener.
        Routing to only the first of them leaves the remaining peers hearing silence -
        and silence is exactly what an absent translation looks like, so the failure is
        indistinguishable from nobody having spoken.
        """
        factory = RecordingFactory()
        router = build_router(learning_session(), factory=factory)
        router.start([PEER_A, PEER_B, PEER_C])

        router.route_audio("peer-a", b"\x00\x00" * 1600, sample_rate=16000)

        speaker_legs = [leg for leg in router.legs() if leg.speaker_id == "peer-a"]
        self.assertEqual(len(speaker_legs), 2)
        for leg in speaker_legs:
            self.assertEqual(len(factory.sessions[leg.leg_id].chunks), 1, leg.leg_id)

    def test_legs_for_speaker_lists_every_leg_that_speaker_owns(self):
        router = build_router(learning_session(), factory=RecordingFactory())
        router.start([PEER_A, PEER_B, PEER_C])

        self.assertEqual(
            {leg.leg_id for leg in router.legs_for_speaker("peer-a")},
            {"leg-peer-a-to-peer-b", "leg-peer-a-to-peer-c"}
        )

    def test_one_degraded_leg_does_not_starve_the_speakers_other_leg(self):
        """A per-listener failure must not mute that speaker for everyone else."""
        factory = RecordingFactory()
        router = build_router(learning_session(), factory=factory)
        router.start([PEER_A, PEER_B, PEER_C])
        router.degrade_leg("leg-peer-a-to-peer-b", reason="websocket closed")

        router.route_audio("peer-a", b"\x00\x00" * 1600, sample_rate=16000)

        self.assertEqual(factory.sessions["leg-peer-a-to-peer-b"].chunks, [])
        self.assertEqual(len(factory.sessions["leg-peer-a-to-peer-c"].chunks), 1)

    def test_audio_from_an_unknown_speaker_is_ignored(self):
        self.assertEqual(
            self.router.route_audio("stranger", b"\x00\x00" * 1600, sample_rate=16000),
            0
        )

    def test_a_degraded_leg_stops_consuming_audio_but_leaves_the_other_running(self):
        leg_a = self.router.leg_for_speaker("peer-a")
        self.router.degrade_leg(leg_a.leg_id, reason="websocket closed")

        self.router.route_audio("peer-a", b"\x00\x00" * 1600, sample_rate=16000)
        self.router.route_audio("peer-b", b"\x00\x00" * 1600, sample_rate=16000)

        leg_b = self.router.leg_for_speaker("peer-b")
        self.assertEqual(self.factory.sessions[leg_a.leg_id].chunks, [])
        self.assertEqual(len(self.factory.sessions[leg_b.leg_id].chunks), 1)
        self.assertEqual(self.router.leg_states()[leg_a.leg_id], LegState.DEGRADED.value)

    def test_a_degraded_leg_can_recover(self):
        leg_a = self.router.leg_for_speaker("peer-a")
        self.router.degrade_leg(leg_a.leg_id, reason="websocket closed")

        self.router.recover_leg(leg_a.leg_id)

        self.assertEqual(self.router.leg_states()[leg_a.leg_id], LegState.ACTIVE.value)
        self.router.route_audio("peer-a", b"\x00\x00" * 1600, sample_rate=16000)
        self.assertEqual(len(self.factory.sessions[leg_a.leg_id].chunks), 1)


class TestTranslatedAudioPublication(unittest.TestCase):
    """TASK-11.4: translated audio is published as a recipient-specific track."""

    def test_learning_mode_publishes_only_to_the_paired_recipient(self):
        publisher = MagicMock()
        router = build_router(learning_session())
        router.audio_publisher = publisher
        router.start([PEER_A, PEER_B])
        router.set_translated_audio_enabled("peer-b", True)

        leg_a = router.leg_for_speaker("peer-a")
        recipients = router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)

        self.assertEqual(recipients, ["peer-b"])
        self.assertTrue(publisher.called)
        self.assertEqual(publisher.call_args.kwargs["recipient_id"], "peer-b")
        self.assertEqual(
            publisher.call_args.kwargs["track_id"],
            router.translated_track_id(leg_a.leg_id)
        )

    def test_work_mode_broadcasts_to_every_other_participant(self):
        router = build_router(work_session())
        router.start([PEER_A, PEER_B, PEER_C])

        leg_a = router.leg_for_speaker("peer-a")
        recipients = router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)

        self.assertEqual(sorted(recipients), ["peer-b", "peer-c"])

    def test_published_frames_match_the_agora_publish_rate(self):
        publisher = MagicMock()
        router = build_router(work_session())
        router.audio_publisher = publisher
        router.start([PEER_A, PEER_B])

        leg_a = router.leg_for_speaker("peer-a")
        router.handle_translated_audio(leg_a.leg_id, b"\x00\x00" * 2400)  # 100 ms @ 24k

        frames = publisher.call_args.kwargs["frames"]
        self.assertEqual(len(frames), 5)  # 5 x 20 ms

    def test_publication_on_an_unknown_leg_is_a_no_op(self):
        router = build_router(work_session())
        router.start([PEER_A, PEER_B])
        self.assertEqual(router.handle_translated_audio("no-such-leg", b"\x00\x00"), [])


class TestTranscriptEvents(unittest.TestCase):
    """TASK-11.4: versioned transcript events and once-only downstream ingestion."""

    def setUp(self):
        self.data_stream = MagicMock(spec=DataStreamManager)
        self.sink = MagicMock()
        self.session = work_session()
        self.router = build_router(self.session, data_stream=self.data_stream,
                                   sink=self.sink)
        self.router.start([PEER_A, PEER_B])
        self.leg_a = self.router.leg_for_speaker("peer-a")
        self.data_stream.reset_mock()

    def transcript_events(self, event_type):
        return [c.args[1] for c in self.data_stream.send_translation_event.call_args_list
                if c.args[0] == event_type]

    def test_input_transcript_event_carries_the_required_fields(self):
        self.router.handle_input_transcript(self.leg_a.leg_id, "namaste", is_final=True)

        payload = self.transcript_events(EVENT_INPUT_TRANSCRIPT)[0]
        for field in ("schema_version", "session_id", "leg_id", "speaker_id",
                      "source_language", "target_language", "sequence", "timestamp",
                      "text", "is_final"):
            self.assertIn(field, payload)
        self.assertEqual(payload["speaker_id"], "peer-a")
        self.assertEqual(payload["text"], "namaste")

    def test_output_transcript_event_is_published_separately(self):
        self.router.handle_output_transcript(self.leg_a.leg_id, "hello", is_final=True)

        payload = self.transcript_events(EVENT_OUTPUT_TRANSCRIPT)[0]
        self.assertEqual(payload["text"], "hello")
        self.assertTrue(payload["target_language"].startswith("en"))

    def test_sequence_increases_per_leg(self):
        self.router.handle_input_transcript(self.leg_a.leg_id, "one", is_final=True)
        self.router.handle_input_transcript(self.leg_a.leg_id, "two", is_final=True)

        sequences = [p["sequence"] for p in self.transcript_events(EVENT_INPUT_TRANSCRIPT)]
        self.assertEqual(sequences, sorted(sequences))
        self.assertNotEqual(sequences[0], sequences[1])

    def test_transcripts_carry_plain_text_only(self):
        """
        Transliteration and register-aware phrasing stay with the ASR -> TeachingAgent
        subtitle pipeline; a Gemini leg must not start emitting a second, divergent one.
        """
        self.router.handle_input_transcript(self.leg_a.leg_id, "namaste", is_final=True)

        payload = self.transcript_events(EVENT_INPUT_TRANSCRIPT)[0]
        for forbidden in ("transliteration", "romaji", "devanagari"):
            self.assertNotIn(forbidden, payload)

    def test_finalized_input_transcript_is_ingested_exactly_once(self):
        self.router.handle_input_transcript(self.leg_a.leg_id, "namaste", is_final=True)
        self.router.handle_input_transcript(self.leg_a.leg_id, "namaste", is_final=True)

        self.assertEqual(self.sink.call_count, 1)

    def test_interim_transcripts_publish_but_are_not_ingested(self):
        """Interim text is display-only (spec section 5); only finalized turns persist."""
        self.router.handle_input_transcript(self.leg_a.leg_id, "nam", is_final=False)

        self.assertEqual(len(self.transcript_events(EVENT_INPUT_TRANSCRIPT)), 1)
        self.sink.assert_not_called()

    def test_output_transcripts_are_not_ingested_as_speaker_turns(self):
        """The interpreter's own voice is not a participant turn."""
        self.router.handle_output_transcript(self.leg_a.leg_id, "hello", is_final=True)
        self.sink.assert_not_called()


class TestShutdown(unittest.TestCase):
    """Legs close on participant leave and on session stop (REQ-17)."""

    def test_stop_closes_every_leg(self):
        factory = RecordingFactory()
        router = build_router(learning_session(), factory=factory)
        router.start([PEER_A, PEER_B])

        router.stop()

        self.assertTrue(all(s.closed for s in factory.sessions.values()))
        self.assertEqual(router.leg_states(), {})

    def test_removing_a_participant_closes_only_the_legs_that_involve_them(self):
        factory = RecordingFactory()
        router = build_router(work_session(), factory=factory)
        router.start([PEER_A, PEER_B, PEER_C])
        leg_a = router.leg_for_speaker("peer-a")
        leg_b = router.leg_for_speaker("peer-b")

        router.remove_participant("peer-a")

        self.assertTrue(factory.sessions[leg_a.leg_id].closed)
        self.assertFalse(factory.sessions[leg_b.leg_id].closed)
        self.assertNotIn("peer-a", router.leg_for_speaker("peer-b").recipients)


if __name__ == "__main__":
    unittest.main()
