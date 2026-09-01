"""
Summary:
    test_notes.py is the executable specification for REQ-14: every finalized turn
    upserts source-linked, mode-specific notes; the same source and type deduplicate;
    a user's edit survives regeneration; and anything the model only inferred is held at
    `needs_confirmation` rather than presented as fact.

Covers:
    - REQ-14 note vocabularies per mode, and rejection of a note typed for the other mode.
    - Idempotent upsert keyed on source turn + type, and `note.upserted` / `note.deleted`.
    - Ambiguity handling: low confidence, or a commitment missing its owner or date.
"""

import unittest
from unittest.mock import MagicMock

from src.artifacts.generator import ArtifactGenerator, NEEDS_CONFIRMATION_THRESHOLD
from src.artifacts.models import NoteItem, NoteStatus
from src.artifacts.repository import InMemoryArtifactRepository
from src.rtc.data_stream import DataStreamManager
from src.sessions.models import SessionRecord


def learning_session():
    """Builds a language-learning session record for the generator under test."""
    return SessionRecord.create(channel="c1", mode="language_learning", languages=["ja"])


def work_session():
    """Builds an international-work session record for the generator under test."""
    return SessionRecord.create(channel="c2", mode="international_work", languages=["en"])


LEARNING_TURN = {
    "spoken_language": "ja",
    "notes": [
        {"type": "vocabulary", "text": "一期一会 - a once-in-a-lifetime encounter",
         "confidence": 0.95},
        {"type": "culture", "text": "Used to mark the value of a single meeting.",
         "confidence": 0.9},
    ],
}

WORK_TURN = {
    "spoken_language": "en",
    "notes": [
        {"type": "decision", "text": "Ship the pilot to the Mumbai team first.",
         "confidence": 0.92},
        {"type": "action", "text": "Send the migration plan.", "owner": "Priya",
         "due_at": "2026-09-04", "confidence": 0.88},
    ],
}


class TestNoteGeneration(unittest.TestCase):
    """Test suite for turning a finalized turn into typed notes (REQ-14)."""

    def setUp(self):
        """Use a fresh generator and repository per test."""
        self.repository = InMemoryArtifactRepository()
        self.generator = ArtifactGenerator(repository=self.repository)

    def test_learning_notes_are_source_linked_and_typed(self):
        """Verify a learning turn yields notes from that mode's vocabulary."""
        session = learning_session()

        notes = self.generator.build_notes(LEARNING_TURN, session, ["turn-1"])

        self.assertEqual(len(notes), 2)
        for note in notes:
            self.assertIsInstance(note, NoteItem)
            self.assertEqual(note.source_turn_ids, ["turn-1"])
            self.assertEqual(note.session_id, session.session_id)
            self.assertIn(note.type, session.mode.note_types)
            self.assertEqual(note.status, NoteStatus.CAPTURED)

    def test_work_notes_capture_commitments(self):
        """Verify a work turn yields decisions and actions with owner and date."""
        notes = self.generator.build_notes(WORK_TURN, work_session(), ["turn-9"])

        by_type = {note.type: note for note in notes}
        self.assertIn("decision", by_type)
        self.assertEqual(by_type["action"].owner, "Priya")
        self.assertEqual(by_type["action"].due_at, "2026-09-04")

    def test_a_note_typed_for_the_other_mode_is_rejected(self):
        """
        Verify a work-typed note cannot land in a learning session. The two vocabularies
        are disjoint, so this is the check that keeps one session's artifact from
        holding items generated under the other mode's contract.
        """
        turn = {"notes": [{"type": "decision", "text": "Ship on Friday.", "confidence": 0.9}]}

        notes = self.generator.build_notes(turn, learning_session(), ["turn-1"])

        self.assertEqual(notes, [])

    def test_note_id_is_stable_for_the_same_source_and_type(self):
        """Verify regenerating the same turn addresses the same note (REQ-14)."""
        session = learning_session()

        first = self.generator.build_notes(LEARNING_TURN, session, ["turn-1"])
        second = self.generator.build_notes(LEARNING_TURN, session, ["turn-1"])

        self.assertEqual([n.id for n in first], [n.id for n in second])

    def test_falls_back_to_the_idiom_card_when_no_notes_block_is_emitted(self):
        """
        Verify a learning turn still produces a note when the model emitted only the
        older scaffolding blocks - the mock engine and older prompts do exactly that.
        """
        turn = {
            "spoken_language": "ja",
            "idiom_card": {"detected": True, "phrase": "一期一会",
                           "meaning": "once-in-a-lifetime encounter",
                           "cultural_note": "Rooted in tea ceremony."},
        }

        notes = self.generator.build_notes(turn, learning_session(), ["turn-1"])

        self.assertTrue(notes)
        self.assertIn("vocabulary", {note.type for note in notes})


class TestNoteAmbiguity(unittest.TestCase):
    """Test suite for REQ-14's ambiguity rule."""

    def setUp(self):
        """Use a fresh generator and repository per test."""
        self.repository = InMemoryArtifactRepository()
        self.generator = ArtifactGenerator(repository=self.repository)

    def test_low_confidence_note_needs_confirmation(self):
        """Verify an uncertain note is held rather than asserted."""
        turn = {"notes": [{"type": "decision", "text": "Maybe move the deadline.",
                           "confidence": NEEDS_CONFIRMATION_THRESHOLD - 0.1}]}

        note = self.generator.build_notes(turn, work_session(), ["turn-1"])[0]

        self.assertEqual(note.status, NoteStatus.NEEDS_CONFIRMATION)

    def test_action_without_an_owner_needs_confirmation(self):
        """
        Verify a commitment missing its owner is flagged, not completed by guesswork:
        an action item assigned to nobody is the ambiguity REQ-14 exists to surface.
        """
        turn = {"notes": [{"type": "action", "text": "Send the migration plan.",
                           "confidence": 0.95}]}

        note = self.generator.build_notes(turn, work_session(), ["turn-1"])[0]

        self.assertEqual(note.status, NoteStatus.NEEDS_CONFIRMATION)

    def test_confident_complete_action_is_captured(self):
        """Verify a fully stated commitment is not held back."""
        turn = {"notes": [{"type": "action", "text": "Send the migration plan.",
                           "owner": "Priya", "due_at": "2026-09-04", "confidence": 0.95}]}

        note = self.generator.build_notes(turn, work_session(), ["turn-1"])[0]

        self.assertEqual(note.status, NoteStatus.CAPTURED)


class TestNoteUpsertAndEvents(unittest.TestCase):
    """Test suite for idempotent persistence and note events (REQ-14)."""

    def setUp(self):
        """Attach a spy data stream so emitted events can be inspected."""
        self.repository = InMemoryArtifactRepository()
        self.data_stream = DataStreamManager()
        self.data_stream._dispatch = MagicMock(return_value=True)
        self.generator = ArtifactGenerator(
            repository=self.repository, data_stream=self.data_stream
        )

    def emitted(self, event_type):
        """Returns the payloads dispatched under one event type."""
        return [
            call[0][1] for call in self.data_stream._dispatch.call_args_list
            if call[0][0] == event_type
        ]

    def test_same_source_and_type_deduplicates(self):
        """Verify re-running a turn updates one note instead of appending a second."""
        session = learning_session()

        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])
        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])

        self.assertEqual(len(self.repository.list_notes(session.session_id)), 2)

    def test_unchanged_note_is_not_re_announced(self):
        """Verify an identical regeneration is silent, so the UI does not flicker."""
        session = learning_session()

        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])
        first_round = len(self.emitted("note.upserted"))
        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])

        self.assertEqual(len(self.emitted("note.upserted")), first_round)

    def test_revised_text_emits_a_new_upsert(self):
        """Verify a genuinely changed note is announced again with a higher revision."""
        session = learning_session()
        revised = {"notes": [dict(LEARNING_TURN["notes"][0], text="一期一会 - treasure this meeting")]}

        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])
        self.generator.generate_notes(revised, session, ["turn-1"])

        upserts = self.emitted("note.upserted")
        self.assertGreaterEqual(len(upserts), 3)
        self.assertEqual(upserts[-1]["note"]["revision"], 2)

    def test_user_edits_survive_regeneration(self):
        """
        Verify a note the user edited is not overwritten by the next model pass - the
        edit is the authoritative text from that point on (REQ-14).
        """
        session = learning_session()
        stored = self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])[0]
        self.repository.edit_note(stored.id, text="my own wording")

        self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])

        kept = self.repository.get_note(stored.id)
        self.assertEqual(kept.text, "my own wording")
        self.assertEqual(kept.status, NoteStatus.EDITED)

    def test_delete_marks_the_note_and_emits_note_deleted(self):
        """Verify deletion is observable to every client (REQ-14)."""
        session = learning_session()
        stored = self.generator.generate_notes(LEARNING_TURN, session, ["turn-1"])[0]

        deleted = self.generator.delete_note(stored.id, session)

        self.assertEqual(deleted.status, NoteStatus.DELETED)
        self.assertEqual(len(self.emitted("note.deleted")), 1)
        self.assertNotIn(stored.id, [n.id for n in self.repository.list_notes(session.session_id)])

    def test_note_upserted_event_carries_the_session_envelope(self):
        """Verify note events identify their session and mode like quiz events do."""
        session = work_session()

        self.generator.generate_notes(WORK_TURN, session, ["turn-9"])

        payload = self.emitted("note.upserted")[0]
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["mode"], "international_work")
        self.assertTrue(payload["event_id"])
        self.assertIn("note", payload)


class TestNotesApiIntegration(unittest.TestCase):
    """Test suite for notes produced and deleted through the server (REQ-14)."""

    def setUp(self):
        """Start each test from a clean session registry and artifact store."""
        from src.server import app, server_instance
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()

    def start_learning_turn(self):
        """Runs one finalized learning turn and returns the notes it produced."""
        self.app.post("/api/convoai/start", json={
            "channel": "note-channel", "language": "hi", "mode": "language_learning"
        })
        self.server.process_convoai_turn(
            speaker_id="Priya", text="नमस्ते",
            language="hi", channel="note-channel"
        )
        return self.app.get("/api/session/notes?channel=note-channel&actor=Learner").get_json()["notes"]

    def test_a_finalized_turn_upserts_retrievable_notes(self):
        """Verify every finalized turn leaves a retrievable, source-linked note."""
        notes = self.start_learning_turn()

        self.assertTrue(notes)
        for note in notes:
            self.assertTrue(note["source_turn_ids"])
            self.assertEqual(note["mode"], "language_learning")

    def test_deleting_a_note_removes_it_from_retrieval(self):
        """Verify deletion is honored through the API, not only in the repository."""
        note_id = self.start_learning_turn()[0]["id"]

        deleted = self.app.delete(f"/api/session/notes/{note_id}?channel=note-channel&actor=Learner")

        self.assertEqual(deleted.status_code, 200)
        remaining = self.app.get("/api/session/notes?channel=note-channel&actor=Learner").get_json()["notes"]
        self.assertNotIn(note_id, [note["id"] for note in remaining])

    def test_the_ambient_rest_turn_path_also_captures_notes(self):
        """
        Verify REQ-14 covers every finalized turn, not only the Convo AI ones: the
        ambient peer pipeline posts through /api/session/turn and its turns are just as
        final.
        """
        self.app.post("/api/session/start", json={
            "channel": "tokyo-mumbai-101", "mode": "language_learning"
        })

        self.app.post("/api/session/turn", json={
            "speaker_id": "Priya", "text": "नमस्ते", "language": "hi"
        })

        notes = self.app.get(
            "/api/session/notes?channel=tokyo-mumbai-101&actor=Priya"
        ).get_json()["notes"]
        self.assertTrue(notes)

    def test_rest_turn_files_notes_against_the_channel_it_names(self):
        """
        Verify a turn posted for a specific channel files its notes under that channel's
        session rather than the server's default one - otherwise a work session's notes
        land in whatever session happens to hold the default channel.
        """
        self.app.post("/api/session/start", json={
            "channel": "ambient-channel", "mode": "language_learning"
        })

        self.app.post("/api/session/turn", json={
            "channel": "ambient-channel", "speaker_id": "Priya",
            "text": "नमस्ते", "language": "hi"
        })

        notes = self.app.get(
            "/api/session/notes?channel=ambient-channel&actor=Priya"
        ).get_json()["notes"]
        self.assertTrue(notes)

    def test_deleting_an_unknown_note_is_a_404(self):
        """Verify a stale client deleting twice gets a miss rather than a second event."""
        self.start_learning_turn()

        response = self.app.delete("/api/session/notes/note-does-not-exist?channel=note-channel&actor=Learner")

        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
