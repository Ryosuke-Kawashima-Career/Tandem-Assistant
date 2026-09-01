"""
Summary:
    test_session_artifacts_api.py is the executable specification for REQ-15's API half:
    a finalized turn is persisted without any approval gate, the whole session is
    retrievable as one versioned artifact, a note can be edited afterwards with
    provenance, and the artifact exports as Markdown.

Covers:
    - REQ-15 automatic upsert (no approval), retrieval, later edits, Markdown export.
    - The transcript is persisted alongside notes and quizzes, so a note's
      `source_turn_ids` still resolve after the conversation has ended.
"""

import unittest

from src.artifacts.models import NoteStatus
from src.server import app, server_instance


class ArtifactApiTestCase(unittest.TestCase):
    """Shared setup: a clean registry and store, plus one finalized learning turn."""

    def setUp(self):
        """Start each test from an empty session registry and artifact store."""
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()

    def start_session(self, channel="artifact-channel", mode="language_learning",
                      participants=None):
        """Creates a session through the API and returns its id."""
        response = self.app.post("/api/session/start", json={
            "channel": channel,
            "mode": mode,
            "participants": participants or ["Kenji"],
            "languages": ["ja"]
        })
        return response.get_json()["session_id"]

    def run_turn(self, channel="artifact-channel", text="नमस्ते",
                 language="hi", speaker_id="Kenji"):
        """Posts one finalized turn through the ambient REST path."""
        return self.app.post("/api/session/turn", json={
            "channel": channel, "speaker_id": speaker_id, "text": text, "language": language
        })


class TestArtifactRetrieval(ArtifactApiTestCase):
    """Test suite for retrieving a whole session artifact (REQ-15)."""

    def test_a_finalized_turn_is_persisted_without_an_approval_gate(self):
        """
        Verify the turn lands in the artifact immediately. REQ-15 is explicit that this
        happens automatically: a queue of pending artifacts awaiting a click is a queue
        nobody clears, and the session's value evaporates with it.
        """
        self.start_session()
        self.run_turn()

        artifact = self.app.get("/api/session/artifact?channel=artifact-channel&actor=Kenji").get_json()

        self.assertTrue(artifact["success"])
        self.assertTrue(artifact["artifact"]["notes"])
        self.assertEqual(artifact["artifact"]["mode"], "language_learning")

    def test_the_transcript_is_persisted_with_the_notes(self):
        """
        Verify the turn a note cites is stored too - otherwise `source_turn_ids` points
        at something that no longer exists once the conversation ends.
        """
        self.start_session()
        self.run_turn()

        artifact = self.app.get(
            "/api/session/artifact?channel=artifact-channel&actor=Kenji"
        ).get_json()["artifact"]

        turn_ids = {turn["id"] for turn in artifact["transcript_turns"]}
        self.assertTrue(turn_ids)
        for note in artifact["notes"]:
            self.assertTrue(set(note["source_turn_ids"]) <= turn_ids)

    def test_the_artifact_outlives_the_session(self):
        """
        Verify a stopped session is still retrievable by its id. The conversation ending
        is exactly when someone wants the notes.
        """
        session_id = self.start_session()
        self.run_turn()
        self.app.post("/api/session/stop", json={"channel": "artifact-channel"})

        response = self.app.get(f"/api/session/artifact?session_id={session_id}&actor=Kenji")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["artifact"]["session_id"], session_id)

    def test_an_unknown_session_is_a_404(self):
        """Verify a session that never existed is a miss, not an empty artifact."""
        response = self.app.get("/api/session/artifact?session_id=sess-nope&actor=Kenji")

        self.assertEqual(response.status_code, 404)


class TestNoteEditing(ArtifactApiTestCase):
    """Test suite for editing a stored note after the fact (REQ-15 / REQ-16)."""

    def first_note_id(self):
        """Returns the id of the first note this session produced."""
        notes = self.app.get(
            "/api/session/notes?channel=artifact-channel&actor=Kenji"
        ).get_json()["notes"]
        return notes[0]["id"]

    def test_editing_a_note_records_who_changed_it(self):
        """Verify an edit carries provenance: the previous wording, the actor, the time."""
        self.start_session()
        self.run_turn()
        note_id = self.first_note_id()

        response = self.app.patch(
            f"/api/session/notes/{note_id}?channel=artifact-channel&actor=Kenji",
            json={"text": "my own wording", "actor": "Kenji"}
        )

        self.assertEqual(response.status_code, 200)
        note = response.get_json()["note"]
        self.assertEqual(note["text"], "my own wording")
        self.assertEqual(note["status"], NoteStatus.EDITED)
        self.assertEqual(note["updated_by"], "Kenji")
        self.assertEqual(len(note["edit_history"]), 1)
        self.assertIn("previous_text", note["edit_history"][0])

    def test_an_edit_survives_the_next_generation_pass(self):
        """Verify REQ-14's rule holds through the API, not only in the repository."""
        self.start_session()
        self.run_turn()
        note_id = self.first_note_id()
        self.app.patch(
            f"/api/session/notes/{note_id}?channel=artifact-channel&actor=Kenji",
            json={"text": "my own wording", "actor": "Kenji"}
        )

        self.run_turn()

        notes = self.app.get(
            "/api/session/notes?channel=artifact-channel&actor=Kenji"
        ).get_json()["notes"]
        edited = next(note for note in notes if note["id"] == note_id)
        self.assertEqual(edited["text"], "my own wording")

    def test_editing_an_unknown_note_is_a_404(self):
        """Verify a stale client editing a purged note gets a miss."""
        self.start_session()
        self.run_turn()

        response = self.app.patch(
            "/api/session/notes/note-nope?channel=artifact-channel&actor=Kenji",
            json={"text": "anything", "actor": "Kenji"}
        )

        self.assertEqual(response.status_code, 404)

    def test_an_edit_without_text_is_rejected(self):
        """Verify an empty edit is a 400 rather than a note silently blanked."""
        self.start_session()
        self.run_turn()
        note_id = self.first_note_id()

        response = self.app.patch(
            f"/api/session/notes/{note_id}?channel=artifact-channel&actor=Kenji",
            json={"actor": "Kenji"}
        )

        self.assertEqual(response.status_code, 400)


class TestArtifactExport(ArtifactApiTestCase):
    """Test suite for exporting a session (REQ-15)."""

    def test_markdown_export_returns_a_readable_document(self):
        """Verify the export is Markdown the user can paste anywhere."""
        self.start_session()
        self.run_turn()

        response = self.app.get(
            "/api/session/artifact/export?channel=artifact-channel&format=markdown&actor=Kenji"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/markdown", response.headers["Content-Type"])
        body = response.data.decode("utf-8")
        self.assertIn("# EchoSphere Session", body)
        self.assertIn("Language Learning", body)

    def test_an_unsupported_format_is_rejected(self):
        """Verify an unknown format is a 400 that names what is supported."""
        self.start_session()
        self.run_turn()

        response = self.app.get(
            "/api/session/artifact/export?channel=artifact-channel&format=pdf&actor=Kenji"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("markdown", str(response.get_json()).lower())

    def test_notion_export_reports_when_it_is_not_configured(self):
        """
        Verify the optional Notion adapter fails loudly and harmlessly when no
        credentials are set - it is an optional export, so an unconfigured server must
        say so rather than pretend the export happened.
        """
        self.start_session()
        self.run_turn()

        response = self.app.get(
            "/api/session/artifact/export?channel=artifact-channel&format=notion&actor=Kenji"
        )

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])


if __name__ == '__main__':
    unittest.main()
