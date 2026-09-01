"""
Summary:
    test_quiz_generation.py is the executable specification for REQ-13: an eligible
    learning point produces a source-linked quiz, asynchronously, with a stable id - and
    `international_work` produces no graded quiz unless one is explicitly requested.

Covers:
    - REQ-13 quiz shape (prompt, answer type/answer, explanation, difficulty, language).
    - Stable ids: regenerating from the same turn must not create a second quiz.
    - `quiz.created` event envelope carrying session id, mode, and an idempotent event id.
    - Mode gating: no unrequested quizzes in a work session.
"""

import unittest
from unittest.mock import MagicMock

from src.artifacts.generator import ArtifactGenerator
from src.artifacts.models import ARTIFACT_SCHEMA_VERSION, QuizItem
from src.artifacts.repository import InMemoryArtifactRepository
from src.rtc.data_stream import DataStreamManager
from src.sessions.models import SessionRecord


def learning_session():
    """Builds a language-learning session record for the generator under test."""
    return SessionRecord.create(channel="c1", mode="language_learning", languages=["ja"])


def work_session():
    """Builds an international-work session record for the generator under test."""
    return SessionRecord.create(channel="c2", mode="international_work", languages=["en"])


TURN_WITH_QUIZ = {
    "spoken_response": "Well done!",
    "spoken_language": "ja",
    "quiz": {
        "active": True,
        "question": "Which particle marks the topic of a Japanese sentence?",
        "options": ["は", "を", "に"],
        "correct_index": 0,
        "explanation": "は marks the topic; を marks the direct object.",
        "difficulty": "beginner",
    },
}


class TestQuizGeneration(unittest.TestCase):
    """Test suite for building a QuizItem out of a finalized turn (REQ-13)."""

    def setUp(self):
        """Use a fresh generator and repository per test."""
        self.repository = InMemoryArtifactRepository()
        self.generator = ArtifactGenerator(repository=self.repository)

    def test_eligible_learning_point_produces_a_source_linked_quiz(self):
        """Verify the quiz carries every field REQ-13 requires, including its source."""
        session = learning_session()

        quiz = self.generator.build_quiz(
            turn_result=TURN_WITH_QUIZ, session=session, source_turn_ids=["turn-1"]
        )

        self.assertIsInstance(quiz, QuizItem)
        self.assertEqual(quiz.prompt, TURN_WITH_QUIZ["quiz"]["question"])
        self.assertEqual(quiz.answer_type, "multiple_choice")
        self.assertEqual(quiz.expected_answer, "は")
        self.assertEqual(quiz.explanation, TURN_WITH_QUIZ["quiz"]["explanation"])
        self.assertEqual(quiz.source_turn_ids, ["turn-1"])
        self.assertEqual(quiz.difficulty, "beginner")
        self.assertEqual(quiz.target_language, "ja")
        self.assertEqual(quiz.session_id, session.session_id)
        self.assertEqual(quiz.schema_version, ARTIFACT_SCHEMA_VERSION)

    def test_free_text_quiz_when_no_options_are_offered(self):
        """Verify an open question is typed as free text rather than faking options."""
        turn = {"quiz": {"active": True, "question": "How would you greet a teacher?",
                         "options": [], "explanation": "Use the polite form."}}

        quiz = self.generator.build_quiz(
            turn_result=turn, session=learning_session(), source_turn_ids=["turn-1"]
        )

        self.assertEqual(quiz.answer_type, "free_text")

    def test_inactive_quiz_block_produces_nothing(self):
        """Verify a turn with no learning point does not manufacture a quiz."""
        turn = {"quiz": {"active": False, "question": "", "options": []}}

        quiz = self.generator.build_quiz(
            turn_result=turn, session=learning_session(), source_turn_ids=["turn-1"]
        )

        self.assertIsNone(quiz)

    def test_quiz_id_is_stable_across_regeneration(self):
        """
        Verify the same turn yields the same quiz id, so the asynchronous generation
        path retrying (or running twice) cannot double a quiz in the learner's UI.
        """
        session = learning_session()

        first = self.generator.build_quiz(TURN_WITH_QUIZ, session, ["turn-1"])
        second = self.generator.build_quiz(TURN_WITH_QUIZ, session, ["turn-1"])

        self.assertEqual(first.id, second.id)

    def test_different_source_turns_produce_different_quizzes(self):
        """Verify the id is derived from the source, not a constant."""
        session = learning_session()

        first = self.generator.build_quiz(TURN_WITH_QUIZ, session, ["turn-1"])
        second = self.generator.build_quiz(TURN_WITH_QUIZ, session, ["turn-2"])

        self.assertNotEqual(first.id, second.id)


class TestQuizModeGating(unittest.TestCase):
    """Test suite for REQ-13's mode rule: no graded quiz in work mode unless asked."""

    def setUp(self):
        """Use a fresh generator and repository per test."""
        self.repository = InMemoryArtifactRepository()
        self.generator = ArtifactGenerator(repository=self.repository)

    def test_work_mode_produces_no_unrequested_quiz(self):
        """Verify colleagues in a work call are not quizzed by default."""
        quiz = self.generator.build_quiz(
            turn_result=TURN_WITH_QUIZ, session=work_session(), source_turn_ids=["turn-1"]
        )

        self.assertIsNone(quiz)

    def test_work_mode_honors_an_explicit_request(self):
        """Verify a work session may still ask for a quiz on demand (REQ-13)."""
        quiz = self.generator.build_quiz(
            turn_result=TURN_WITH_QUIZ,
            session=work_session(),
            source_turn_ids=["turn-1"],
            requested=True
        )

        self.assertIsNotNone(quiz)


class TestQuizPersistenceAndEvents(unittest.TestCase):
    """Test suite for storing a quiz once and announcing it (REQ-13)."""

    def setUp(self):
        """Attach a spy data stream so emitted events can be inspected."""
        self.repository = InMemoryArtifactRepository()
        self.data_stream = DataStreamManager()
        self.data_stream._dispatch = MagicMock(return_value=True)
        self.generator = ArtifactGenerator(
            repository=self.repository, data_stream=self.data_stream
        )

    def test_generate_emits_quiz_created_once_per_quiz(self):
        """
        Verify a regenerated turn does not re-announce the same quiz: the scaffolding
        path is asynchronous and may run again for the same utterance.
        """
        session = learning_session()

        self.generator.generate_quiz(TURN_WITH_QUIZ, session, ["turn-1"])
        self.generator.generate_quiz(TURN_WITH_QUIZ, session, ["turn-1"])

        created_events = [
            call for call in self.data_stream._dispatch.call_args_list
            if call[0][0] == "quiz.created"
        ]
        self.assertEqual(len(created_events), 1)
        self.assertEqual(len(self.repository.list_quizzes(session.session_id)), 1)

    def test_quiz_created_event_carries_the_session_envelope(self):
        """Verify the event identifies its session, mode, version, and event id."""
        session = learning_session()

        self.generator.generate_quiz(TURN_WITH_QUIZ, session, ["turn-1"])

        event_type, payload = self.data_stream._dispatch.call_args[0]
        self.assertEqual(event_type, "quiz.created")
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["mode"], "language_learning")
        self.assertEqual(payload["schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertTrue(payload["event_id"])
        self.assertEqual(payload["quiz"]["prompt"], TURN_WITH_QUIZ["quiz"]["question"])

    def test_event_id_is_idempotent_for_the_same_entity_revision(self):
        """Verify a receiver can discard a duplicate event by id alone."""
        session = learning_session()
        quiz = self.generator.build_quiz(TURN_WITH_QUIZ, session, ["turn-1"])

        first = self.generator.event_id_for(quiz.id, revision=1)
        second = self.generator.event_id_for(quiz.id, revision=1)
        third = self.generator.event_id_for(quiz.id, revision=2)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class TestQuizApiIntegration(unittest.TestCase):
    """Test suite for quizzes produced by a real turn through the server (REQ-13)."""

    def setUp(self):
        """Start each test from a clean session registry and artifact store."""
        from src.server import app, server_instance
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()

    def test_a_learning_turn_produces_a_retrievable_quiz(self):
        """Verify the asynchronous scaffolding path files a quiz against the session."""
        self.app.post("/api/convoai/start", json={
            "channel": "quiz-channel", "language": "hi", "mode": "language_learning"
        })

        self.server.process_convoai_turn(
            speaker_id="Priya", text="नमस्ते",
            language="hi", channel="quiz-channel"
        )

        response = self.app.get("/api/session/quizzes?channel=quiz-channel&actor=Learner")
        self.assertEqual(response.status_code, 200)
        quizzes = response.get_json()["quizzes"]
        self.assertEqual(len(quizzes), 1)
        self.assertTrue(quizzes[0]["source_turn_ids"])
        self.assertEqual(quizzes[0]["mode"], "language_learning")

    def test_a_work_turn_produces_no_quiz(self):
        """Verify REQ-13's mode gate holds end to end, not just in the generator."""
        self.app.post("/api/convoai/start", json={
            "channel": "work-quiz-channel", "language": "en", "mode": "international_work"
        })

        self.server.process_convoai_turn(
            speaker_id="Priya", text="Namaste", language="hi", channel="work-quiz-channel"
        )

        response = self.app.get("/api/session/quizzes?channel=work-quiz-channel&actor=Learner")
        self.assertEqual(response.get_json()["quizzes"], [])


if __name__ == '__main__':
    unittest.main()
