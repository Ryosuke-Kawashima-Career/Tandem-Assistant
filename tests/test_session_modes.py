"""
Summary:
    test_session_modes.py is the executable specification for REQ-12: a session is
    created with an immutable mode of `language_learning` or `international_work`, and
    that mode propagates into prompt selection, RTC events, and artifacts.

    The two modes replace the previous `student` / `teacher` view roles: those described
    who was looking at the screen, not what the session is for, so nothing downstream
    (prompts, quizzes, notes) could branch on them.

Covers:
    - REQ-12 mode parsing, immutability, and the 400 contract on the creation APIs.
    - REQ-04 mode-specific assistance: `international_work` must not run the language
      tutor prompt, which grades and corrects the speaker.
"""

import unittest
from unittest.mock import patch

from src.sessions.models import (
    SessionMode,
    SessionRecord,
    InvalidSessionModeError,
    SessionModeImmutableError,
    SESSION_SCHEMA_VERSION,
)
from src.sessions.service import SessionService, SessionNotFoundError
from src.server import app, server_instance


class TestSessionMode(unittest.TestCase):
    """Test suite for the mode value object itself (REQ-12)."""

    def test_the_two_supported_modes(self):
        """Verify the mode vocabulary is exactly the two the spec defines."""
        self.assertEqual(
            {mode.value for mode in SessionMode},
            {"language_learning", "international_work"}
        )

    def test_parse_accepts_either_mode(self):
        """Verify both spec values parse to their enum member."""
        self.assertEqual(SessionMode.parse("language_learning"), SessionMode.LANGUAGE_LEARNING)
        self.assertEqual(SessionMode.parse("international_work"), SessionMode.INTERNATIONAL_WORK)

    def test_parse_rejects_the_retired_view_roles(self):
        """
        Verify the replaced `student`/`teacher` values are rejected rather than silently
        mapped: a stale client sending one must be told, not quietly given a default.
        """
        for retired in ("student", "teacher"):
            with self.assertRaises(InvalidSessionModeError):
                SessionMode.parse(retired)

    def test_parse_rejects_missing_and_invalid_values(self):
        """Verify missing/invalid input raises rather than defaulting (REQ-12)."""
        for value in (None, "", "  ", 7):
            with self.assertRaises(InvalidSessionModeError):
                SessionMode.parse(value)

    def test_mode_policy_differs_between_modes(self):
        """
        Verify each mode carries its own assistance policy, so callers branch on the
        mode object rather than re-deriving string comparisons everywhere.
        """
        learning = SessionMode.LANGUAGE_LEARNING
        work = SessionMode.INTERNATIONAL_WORK

        self.assertTrue(learning.grades_language)
        self.assertFalse(work.grades_language)
        self.assertTrue(learning.quizzes_by_default)
        self.assertFalse(work.quizzes_by_default)

    def test_note_types_are_mode_specific(self):
        """Verify REQ-14's two note vocabularies are owned by the mode."""
        learning = set(SessionMode.LANGUAGE_LEARNING.note_types)
        work = set(SessionMode.INTERNATIONAL_WORK.note_types)

        self.assertIn("vocabulary", learning)
        self.assertIn("grammar", learning)
        self.assertIn("culture", learning)
        self.assertIn("decision", work)
        self.assertIn("action", work)
        self.assertIn("risk", work)
        self.assertFalse(learning & work)


class TestSessionRecordAndService(unittest.TestCase):
    """Test suite for session creation and mode immutability (REQ-12)."""

    def setUp(self):
        """Start each test from an empty session registry."""
        self.service = SessionService()

    def test_create_returns_a_versioned_record(self):
        """Verify a created session carries a stable id, mode, and schema version."""
        record = self.service.create_session(
            channel="tokyo-mumbai-101",
            mode="language_learning",
            languages=["ja", "hi"],
            participants=["Kenji", "Priya"]
        )

        self.assertIsInstance(record, SessionRecord)
        self.assertEqual(record.mode, SessionMode.LANGUAGE_LEARNING)
        self.assertEqual(record.channel, "tokyo-mumbai-101")
        self.assertEqual(record.schema_version, SESSION_SCHEMA_VERSION)
        self.assertTrue(record.session_id)
        self.assertEqual(record.to_dict()["mode"], "language_learning")

    def test_create_rejects_an_invalid_mode(self):
        """Verify creation refuses an unusable mode rather than defaulting (REQ-12)."""
        with self.assertRaises(InvalidSessionModeError):
            self.service.create_session(channel="c1", mode="teacher")

    def test_mode_is_immutable_for_the_life_of_the_session(self):
        """
        Verify the mode cannot be switched mid-session. Notes, quizzes, and prompts are
        all mode-shaped, so a mid-session flip would leave one session holding artifacts
        generated under two different contracts.
        """
        self.service.create_session(channel="c1", mode="language_learning")

        with self.assertRaises(SessionModeImmutableError):
            self.service.set_mode("c1", "international_work")

        self.assertEqual(self.service.mode_for("c1"), SessionMode.LANGUAGE_LEARNING)

    def test_setting_the_same_mode_is_a_no_op(self):
        """Verify an idempotent re-send of the current mode is not an error."""
        self.service.create_session(channel="c1", mode="international_work")

        self.service.set_mode("c1", "international_work")

        self.assertEqual(self.service.mode_for("c1"), SessionMode.INTERNATIONAL_WORK)

    def test_recreating_a_channel_starts_a_new_session(self):
        """
        Verify a channel reused after its session ended gets a fresh session id and may
        take a different mode - immutability binds a session, not a channel forever.
        """
        first = self.service.create_session(channel="c1", mode="language_learning")
        self.service.end_session("c1")
        second = self.service.create_session(channel="c1", mode="international_work")

        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(second.mode, SessionMode.INTERNATIONAL_WORK)

    def test_unknown_channel_raises_rather_than_guessing(self):
        """Verify a missing session is an explicit error for callers that require one."""
        with self.assertRaises(SessionNotFoundError):
            self.service.require_session("never-created")

    def test_event_context_carries_mode_for_propagation(self):
        """
        Verify the service exposes the envelope that events and artifacts stamp
        themselves with, so mode propagation has exactly one source (REQ-12).
        """
        record = self.service.create_session(channel="c1", mode="international_work")

        context = self.service.event_context("c1")

        self.assertEqual(context["mode"], "international_work")
        self.assertEqual(context["session_id"], record.session_id)
        self.assertEqual(context["schema_version"], SESSION_SCHEMA_VERSION)


class TestSessionModeApi(unittest.TestCase):
    """Test suite for the mode contract on the creation endpoints (REQ-12)."""

    def setUp(self):
        """Use a Flask test client against a clean session registry."""
        self.app = app.test_client()
        server_instance.sessions.reset()

    def test_session_start_requires_a_mode(self):
        """Verify a missing mode is rejected with 400 rather than defaulted."""
        response = self.app.post("/api/session/start", json={})

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_session_start_rejects_an_invalid_mode(self):
        """Verify the retired `student` role is a 400 on the live API."""
        response = self.app.post("/api/session/start", json={"mode": "student"})

        self.assertEqual(response.status_code, 400)

    def test_session_start_accepts_either_mode(self):
        """Verify both spec modes create a session and are echoed back."""
        for mode in ("language_learning", "international_work"):
            server_instance.sessions.reset()
            response = self.app.post("/api/session/start", json={"mode": mode})

            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertTrue(body["success"])
            self.assertEqual(body["mode"], mode)
            self.assertTrue(body["session_id"])

    def test_convoai_start_requires_a_mode(self):
        """Verify the Convo AI creation path enforces the same contract (spec section 4)."""
        response = self.app.post("/api/convoai/start", json={"language": "ja"})

        self.assertEqual(response.status_code, 400)

    def test_convoai_start_registers_the_session_mode(self):
        """Verify a Convo AI session is retrievable by mode for downstream branching."""
        response = self.app.post(
            "/api/convoai/start",
            json={"channel": "mode-channel", "language": "ja", "mode": "international_work"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            server_instance.sessions.mode_for("mode-channel"),
            SessionMode.INTERNATIONAL_WORK
        )

    def test_status_reports_the_session_mode(self):
        """Verify the mode is observable without re-reading the creation response."""
        self.app.post(
            "/api/session/start",
            json={"channel": "status-channel", "mode": "language_learning"}
        )

        response = self.app.get("/api/session/status?channel=status-channel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["mode"], "language_learning")


class TestModePropagatesToPrompts(unittest.TestCase):
    """Test suite for REQ-04 mode-specific assistance in the reasoning path."""

    def setUp(self):
        """Use the Flask client with a clean registry and a mock-engine agent."""
        self.app = app.test_client()
        server_instance.sessions.reset()

    def test_work_mode_uses_the_work_prompt_not_the_tutor_prompt(self):
        """
        Verify an `international_work` turn is not graded as language practice: the
        work prompt clarifies terms and captures commitments instead.
        """
        self.app.post(
            "/api/convoai/start",
            json={"channel": "work-channel", "language": "en", "mode": "international_work"}
        )

        with patch("src.agent.orchestrator.create_work_prompt") as work_prompt:
            work_prompt.return_value = "work prompt"
            server_instance.process_convoai_turn(
                speaker_id="Priya",
                text="Let us ship the pilot by Friday.",
                language="en",
                channel="work-channel"
            )

        work_prompt.assert_called_once()

    def test_learning_mode_still_uses_the_tutor_prompt(self):
        """Verify the existing 1:1 tutor path is unchanged for language sessions."""
        self.app.post(
            "/api/convoai/start",
            json={"channel": "learn-channel", "language": "ja", "mode": "language_learning"}
        )

        with patch("src.agent.orchestrator.create_tutor_prompt") as tutor_prompt:
            tutor_prompt.return_value = "tutor prompt"
            server_instance.process_convoai_turn(
                speaker_id="Kenji",
                text="こんにちは",
                language="ja",
                channel="learn-channel"
            )

        tutor_prompt.assert_called_once()


if __name__ == '__main__':
    unittest.main()
