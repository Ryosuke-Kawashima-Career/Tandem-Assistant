"""
Summary:
    test_translation_english_pivot.py is the executable specification for TASK-11.7: in
    `international_work`, every leg targets English regardless of who is speaking.

    English is the shared working language, so the pivot is unconditional. A participant
    who is already speaking English still runs a leg, because they may code-switch into
    their own language for a precision term mid-sentence, and a leg that only existed for
    non-English speakers would miss exactly that. `echoTargetLanguage=false` is what keeps
    the already-English case from being redundantly re-synthesized back into the room.

Covers:
    - REQ-17 English target on every `international_work` leg, including English speakers.
    - REQ-17 `echoTargetLanguage=false` on the pivot legs.
    - REQ-17 an English speaker's translated track never returns to its own speaker.
    - REQ-17 the pivot is work-mode only; language_learning keeps direct peer-pair legs.
"""

import json
import unittest
from unittest.mock import MagicMock

from src.sessions.models import SessionRecord
from src.translation.gemini_live import GeminiLiveTranslateSession, LegState
from src.translation.router import Participant, TranslationRouter

HINDI = Participant(participant_id="peer-a", language="hi")
JAPANESE = Participant(participant_id="peer-b", language="ja")
ENGLISH = Participant(participant_id="peer-c", language="en")


class StubLeg:
    """A leg that connects successfully and records the audio it is given."""

    def __init__(self, config, **kwargs):
        self.config = config
        self.chunks = []
        self.state = LegState.IDLE
        self.closed = False

    def connect(self):
        self.state = LegState.ACTIVE
        return True

    def send_audio(self, chunk):
        self.chunks.append(chunk)
        return True

    def close(self):
        self.closed = True
        self.state = LegState.CLOSED


def work_router(participants, factory=None):
    """Builds a started `international_work` router over the given participants."""
    session = SessionRecord.create(
        channel="c-work", mode="international_work",
        languages=sorted({p.language for p in participants}),
    )
    router = TranslationRouter(session=session, session_factory=factory or StubLeg)
    router.start(participants)
    return router


class TestEveryLegTargetsEnglish(unittest.TestCase):
    """TASK-11.7: one target language on every leg, whoever is speaking."""

    def test_non_english_speakers_target_english(self):
        router = work_router([HINDI, JAPANESE])

        for leg in router.legs():
            self.assertTrue(leg.target_language.startswith("en"), leg.target_language)

    def test_an_english_speaker_still_gets_a_leg(self):
        """
        They may code-switch mid-sentence for a precision term. Skipping their leg would
        drop precisely the utterance the pivot exists to carry.
        """
        router = work_router([HINDI, ENGLISH])

        self.assertIsNotNone(router.leg_for_speaker("peer-c"))
        self.assertTrue(router.leg_for_speaker("peer-c").target_language.startswith("en"))

    def test_target_is_a_region_qualified_english_tag(self):
        router = work_router([HINDI, ENGLISH])

        for leg in router.legs():
            self.assertIn(leg.target_language, ("en-IN", "en-US"))

    def test_the_pivot_holds_as_participants_are_added(self):
        router = work_router([HINDI, JAPANESE, ENGLISH])

        targets = {leg.target_language for leg in router.legs()}

        self.assertEqual(len(router.legs()), 3)
        self.assertEqual(len(targets), 1)


class TestEchoTargetLanguage(unittest.TestCase):
    """TASK-11.7: `echoTargetLanguage=false` on the wire, for every pivot leg."""

    def test_leg_config_defaults_echo_off(self):
        router = work_router([HINDI, ENGLISH])

        for leg in router.legs():
            self.assertFalse(leg.echo_target_language)

    def test_the_setup_message_sent_to_gemini_disables_echo(self):
        """The property has to survive into the actual setup frame, not just the config."""
        router = work_router([ENGLISH])
        leg = router.leg_for_speaker("peer-c")

        session = GeminiLiveTranslateSession(config=leg, api_key="k")
        setup = json.loads(json.dumps(session.build_setup_message()))["setup"]
        translation = setup["generationConfig"]["translationConfig"]

        self.assertIs(translation["echoTargetLanguage"], False)
        self.assertTrue(translation["targetLanguageCode"].startswith("en"))


class TestNoSelfDelivery(unittest.TestCase):
    """TASK-11.7: an already-English utterance never re-enters its own track."""

    def test_a_speakers_translated_track_excludes_that_speaker(self):
        router = work_router([HINDI, ENGLISH])
        english_leg = router.leg_for_speaker("peer-c")

        self.assertNotIn("peer-c", english_leg.recipients)
        self.assertEqual(english_leg.recipients, ("peer-a",))

    def test_english_translated_audio_is_delivered_only_to_the_others(self):
        publisher = MagicMock()
        router = work_router([HINDI, JAPANESE, ENGLISH])
        router.audio_publisher = publisher

        recipients = router.handle_translated_audio(
            router.leg_for_speaker("peer-c").leg_id, b"\x00\x00" * 2400
        )

        self.assertEqual(sorted(recipients), ["peer-a", "peer-b"])

    def test_the_english_translated_track_is_not_routed_back_into_any_leg(self):
        factory_sessions = {}

        def factory(config, **kwargs):
            leg = StubLeg(config)
            factory_sessions[config.leg_id] = leg
            return leg

        router = work_router([HINDI, ENGLISH], factory=factory)
        english_leg = router.leg_for_speaker("peer-c")
        track = router.translated_track_id(english_leg.leg_id)

        self.assertEqual(router.route_audio(track, b"\x00\x00" * 1600, sample_rate=16000), 0)
        self.assertTrue(all(not leg.chunks for leg in factory_sessions.values()))


class TestPivotIsWorkModeOnly(unittest.TestCase):
    """The pivot must not leak into language_learning, which interprets directly."""

    def test_language_learning_targets_each_peers_own_language(self):
        session = SessionRecord.create(channel="c-learn", mode="language_learning",
                                       languages=["hi", "ja"])
        router = TranslationRouter(session=session, session_factory=StubLeg)
        router.start([HINDI, JAPANESE])

        targets = {leg.speaker_id: leg.target_language for leg in router.legs()}

        self.assertEqual(targets["peer-a"], "ja-JP")
        self.assertEqual(targets["peer-b"], "hi-IN")
        self.assertFalse(any(t.startswith("en") for t in targets.values()))


if __name__ == "__main__":
    unittest.main()
