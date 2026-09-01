"""
Summary:
    test_artifact_repository.py is the executable specification for REQ-15's storage
    half: a session's transcript turns, notes, and quizzes survive a process restart,
    assemble into one versioned `SessionArtifact`, and render to Markdown.

    The Phase 9 in-memory semantics (dedupe on source+type, user edits pinned against
    regeneration, soft delete) must hold identically once the store is on disk - that is
    the whole point of keeping them in the base class rather than in the storage layer.

Covers:
    - REQ-15 durable local repository as the MVP source of truth.
    - Versioned `SessionArtifact` envelope and Markdown export.
    - Retention: artifacts older than the configured window are purged.
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from src.artifacts.export import render_markdown
from src.artifacts.models import (
    ARTIFACT_SCHEMA_VERSION,
    NoteItem,
    NoteStatus,
    QuizItem,
    SessionArtifact,
    TranscriptTurn,
)
from src.artifacts.repository import InMemoryArtifactRepository, LocalArtifactRepository
from src.sessions.models import SessionRecord


def build_session(mode="language_learning"):
    """Builds a session record for the repository under test."""
    return SessionRecord.create(
        channel="c1", mode=mode, languages=["ja", "en"], participants=["Kenji", "Priya"]
    )


def build_note(session, note_type="vocabulary", text="一期一会 - treasure this meeting"):
    """Builds one note belonging to a session."""
    return NoteItem(
        id=f"note-{session.session_id}-{note_type}",
        type=note_type,
        text=text,
        source_turn_ids=["turn-1"],
        confidence=0.9,
        status=NoteStatus.CAPTURED,
        session_id=session.session_id,
        mode=session.mode.value
    )


def build_quiz(session):
    """Builds one quiz belonging to a session."""
    return QuizItem(
        id=f"quiz-{session.session_id}",
        prompt="Which particle marks the topic?",
        answer_type="multiple_choice",
        expected_answer="は",
        explanation="は marks the topic.",
        options=["は", "を"],
        source_turn_ids=["turn-1"],
        difficulty="beginner",
        target_language="ja",
        session_id=session.session_id,
        mode=session.mode.value
    )


class TestTranscriptTurns(unittest.TestCase):
    """Test suite for storing the transcript a session's artifacts point back to."""

    def setUp(self):
        """Use a fresh in-memory repository per test."""
        self.repository = InMemoryArtifactRepository()
        self.session = build_session()

    def test_a_turn_is_stored_once_per_utterance(self):
        """
        Verify re-recording the same utterance does not duplicate the transcript: the
        scaffolding path may process one utterance twice, and both passes record it.
        """
        turn = TranscriptTurn.create(self.session.session_id, "Kenji", "こんにちは", "ja")

        first, created_first = self.repository.add_turn(turn)
        second, created_second = self.repository.add_turn(turn)

        self.assertTrue(created_first)
        self.assertFalse(created_second)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.repository.list_turns(self.session.session_id)), 1)

    def test_turn_ids_are_derived_from_their_content(self):
        """Verify a note's `source_turn_ids` can address a turn without a lookup table."""
        turn = TranscriptTurn.create(self.session.session_id, "Kenji", "hello", "en")
        same = TranscriptTurn.create(self.session.session_id, "Kenji", "hello", "en")
        other = TranscriptTurn.create(self.session.session_id, "Kenji", "goodbye", "en")

        self.assertEqual(turn.id, same.id)
        self.assertNotEqual(turn.id, other.id)


class TestSessionArtifactAssembly(unittest.TestCase):
    """Test suite for the versioned artifact envelope (REQ-15)."""

    def setUp(self):
        """Populate a repository with one session's transcript, note, and quiz."""
        self.repository = InMemoryArtifactRepository()
        self.session = build_session()
        self.repository.save_session(self.session)
        self.repository.add_turn(
            TranscriptTurn.create(self.session.session_id, "Kenji", "こんにちは", "ja")
        )
        self.repository.upsert_note(build_note(self.session))
        self.repository.add_quiz(build_quiz(self.session))

    def test_artifact_gathers_everything_the_session_produced(self):
        """Verify one envelope carries transcript, notes, quizzes, mode, and version."""
        artifact = self.repository.build_artifact(self.session.session_id)

        self.assertIsInstance(artifact, SessionArtifact)
        self.assertEqual(artifact.session_id, self.session.session_id)
        self.assertEqual(artifact.mode, "language_learning")
        self.assertEqual(artifact.schema_version, ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(len(artifact.transcript_turns), 1)
        self.assertEqual(len(artifact.notes), 1)
        self.assertEqual(len(artifact.quizzes), 1)
        self.assertEqual(artifact.participants, ["Kenji", "Priya"])

    def test_artifact_revision_advances_with_its_contents(self):
        """
        Verify the envelope's revision moves when the session produces something new, so
        a consumer can tell a re-read apart from a changed artifact without diffing it.
        """
        before = self.repository.build_artifact(self.session.session_id).revision

        self.repository.upsert_note(build_note(self.session, note_type="culture", text="Tea ceremony origin."))
        after = self.repository.build_artifact(self.session.session_id).revision

        self.assertGreater(after, before)

    def test_unknown_session_has_no_artifact(self):
        """Verify a session that produced nothing yields None rather than an empty shell."""
        self.assertIsNone(self.repository.build_artifact("sess-never-existed"))

    def test_artifact_serializes_to_plain_json(self):
        """Verify the envelope round-trips through JSON for storage and API responses."""
        artifact = self.repository.build_artifact(self.session.session_id)

        payload = json.loads(json.dumps(artifact.to_dict()))

        self.assertEqual(payload["session_id"], self.session.session_id)
        self.assertEqual(payload["notes"][0]["type"], "vocabulary")


class TestMarkdownExport(unittest.TestCase):
    """Test suite for REQ-15's Markdown export."""

    def setUp(self):
        """Assemble one artifact per mode to render."""
        self.repository = InMemoryArtifactRepository()
        self.session = build_session()
        self.repository.save_session(self.session)
        self.repository.add_turn(
            TranscriptTurn.create(self.session.session_id, "Kenji", "こんにちは", "ja")
        )
        self.repository.upsert_note(build_note(self.session))
        self.repository.add_quiz(build_quiz(self.session))

    def test_markdown_contains_the_sessions_content(self):
        """Verify the export is readable on its own - notes, quizzes, and transcript."""
        markdown = render_markdown(self.repository.build_artifact(self.session.session_id))

        self.assertIn("# ", markdown)
        self.assertIn("Language Learning", markdown)
        self.assertIn("treasure this meeting", markdown)
        self.assertIn("Which particle marks the topic?", markdown)
        self.assertIn("Kenji", markdown)

    def test_markdown_flags_unconfirmed_notes(self):
        """
        Verify an unconfirmed note is marked in the export too. Exported into a document
        the team acts on, an inferred commitment that reads as agreed is the failure
        REQ-14 exists to prevent, and the export is exactly where it would slip through.
        """
        unconfirmed = NoteItem(
            id="note-unconfirmed", type="goal", text="Maybe practise daily.",
            source_turn_ids=["turn-1"], confidence=0.3,
            status=NoteStatus.NEEDS_CONFIRMATION,
            session_id=self.session.session_id, mode=self.session.mode.value
        )
        self.repository.upsert_note(unconfirmed)

        markdown = render_markdown(self.repository.build_artifact(self.session.session_id))

        self.assertIn("needs confirmation", markdown.lower())

    def test_work_mode_export_uses_work_headings(self):
        """Verify the export reads as meeting minutes for a work session, not a lesson."""
        repository = InMemoryArtifactRepository()
        session = build_session(mode="international_work")
        repository.save_session(session)
        repository.upsert_note(NoteItem(
            id="note-decision", type="decision", text="Ship the pilot first.",
            source_turn_ids=["turn-1"], confidence=0.9, status=NoteStatus.CAPTURED,
            session_id=session.session_id, mode=session.mode.value
        ))

        markdown = render_markdown(repository.build_artifact(session.session_id))

        self.assertIn("International Work", markdown)
        self.assertIn("Ship the pilot first.", markdown)


class TestLocalArtifactRepository(unittest.TestCase):
    """Test suite for the durable local store (REQ-15)."""

    def setUp(self):
        """Give each test its own data directory."""
        self.data_dir = Path(tempfile.mkdtemp(prefix="echosphere-artifacts-"))
        self.session = build_session()

    def tearDown(self):
        """Remove the temporary data directory."""
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def open_repository(self, retention_days=None):
        """Opens a repository over this test's data directory."""
        return LocalArtifactRepository(data_dir=self.data_dir, retention_days=retention_days)

    def test_artifacts_survive_a_restart(self):
        """
        Verify a new process sees what the previous one stored - the local repository is
        the MVP source of truth, so losing it on restart loses the session.
        """
        writer = self.open_repository()
        writer.save_session(self.session)
        writer.add_turn(TranscriptTurn.create(self.session.session_id, "Kenji", "hello", "en"))
        writer.upsert_note(build_note(self.session))
        writer.add_quiz(build_quiz(self.session))

        reader = self.open_repository()

        artifact = reader.build_artifact(self.session.session_id)
        self.assertIsNotNone(artifact)
        self.assertEqual(len(artifact.notes), 1)
        self.assertEqual(len(artifact.quizzes), 1)
        self.assertEqual(len(artifact.transcript_turns), 1)
        self.assertEqual(artifact.mode, "language_learning")

    def test_persisted_notes_keep_their_status_and_revision(self):
        """Verify a user edit survives the restart, not just the note's text."""
        writer = self.open_repository()
        writer.save_session(self.session)
        note = build_note(self.session)
        writer.upsert_note(note)
        writer.edit_note(note.id, text="my own wording", actor="Kenji")

        reloaded = self.open_repository().get_note(note.id)

        self.assertEqual(reloaded.text, "my own wording")
        self.assertEqual(reloaded.status, NoteStatus.EDITED)
        self.assertEqual(reloaded.revision, 2)

    def test_regeneration_after_a_restart_still_respects_the_edit(self):
        """
        Verify the Phase 9 rule holds across processes: a model pass in a new process
        must not undo a wording the user chose in the previous one.
        """
        writer = self.open_repository()
        writer.save_session(self.session)
        note = build_note(self.session)
        writer.upsert_note(note)
        writer.edit_note(note.id, text="my own wording", actor="Kenji")

        reader = self.open_repository()
        reader.upsert_note(build_note(self.session, text="model wording again"))

        self.assertEqual(reader.get_note(note.id).text, "my own wording")

    def test_deleted_notes_stay_deleted_across_a_restart(self):
        """Verify the tombstone persists, so a late regeneration cannot resurrect it."""
        writer = self.open_repository()
        writer.save_session(self.session)
        note = build_note(self.session)
        writer.upsert_note(note)
        writer.delete_note(note.id)

        reader = self.open_repository()
        reader.upsert_note(build_note(self.session))

        self.assertEqual(reader.list_notes(self.session.session_id), [])

    def test_purging_a_session_removes_its_file(self):
        """Verify complete deletion leaves nothing behind on disk (REQ-16)."""
        writer = self.open_repository()
        writer.save_session(self.session)
        writer.upsert_note(build_note(self.session))

        writer.purge_session(self.session.session_id)

        self.assertEqual(list(self.data_dir.glob("*.json")), [])
        self.assertIsNone(self.open_repository().build_artifact(self.session.session_id))

    def test_retention_purges_artifacts_past_the_window(self):
        """
        Verify a configured retention window is enforced (REQ-16). Recordings of real
        conversations should not accumulate indefinitely by default.
        """
        writer = self.open_repository(retention_days=7)
        writer.save_session(self.session)
        writer.upsert_note(build_note(self.session))

        purged = writer.purge_expired(now=time.time() + 8 * 86400)

        self.assertEqual(purged, 1)
        self.assertIsNone(writer.build_artifact(self.session.session_id))

    def test_retention_keeps_artifacts_inside_the_window(self):
        """Verify a session younger than the window is untouched."""
        writer = self.open_repository(retention_days=7)
        writer.save_session(self.session)
        writer.upsert_note(build_note(self.session))

        purged = writer.purge_expired(now=time.time() + 6 * 86400)

        self.assertEqual(purged, 0)
        self.assertIsNotNone(writer.build_artifact(self.session.session_id))

    def test_no_retention_configured_keeps_everything(self):
        """Verify retention is opt-in: an unset window never deletes a user's data."""
        writer = self.open_repository(retention_days=None)
        writer.save_session(self.session)
        writer.upsert_note(build_note(self.session))

        purged = writer.purge_expired(now=time.time() + 3650 * 86400)

        self.assertEqual(purged, 0)

    def test_a_corrupt_file_does_not_take_down_the_store(self):
        """
        Verify one unreadable session file is skipped with a warning rather than making
        every other session unreadable - a crash mid-write must not cost the whole store.
        """
        writer = self.open_repository()
        writer.save_session(self.session)
        writer.upsert_note(build_note(self.session))
        (self.data_dir / "sess-corrupt.json").write_text("{not json", encoding="utf-8")

        reader = self.open_repository()

        self.assertIsNotNone(reader.build_artifact(self.session.session_id))


if __name__ == '__main__':
    unittest.main()
