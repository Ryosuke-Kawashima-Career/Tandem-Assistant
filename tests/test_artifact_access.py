"""
Summary:
    test_artifact_access.py is the executable specification for REQ-16: a session's
    artifacts are reachable only by its participants, credentials stay server-side, every
    edit carries provenance, retention is configurable, and deletion is complete.

Covers:
    - REQ-16 access control on retrieval, export, edit, and delete.
    - Complete deletion: transcript, notes, and quizzes all become unavailable.
    - Edit provenance surviving in the stored artifact.
    - Server-only credentials: no artifact response carries one.
"""

import unittest

from src.artifacts.access import AccessDeniedError, can_access, require_access
from src.server import app, server_instance


class AccessTestCase(unittest.TestCase):
    """Shared setup: a two-participant session with one finalized turn."""

    def setUp(self):
        """Start each test from an empty session registry and artifact store."""
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()

        self.app.post("/api/session/start", json={
            "channel": "governed-channel",
            "mode": "language_learning",
            "participants": ["Kenji", "Priya"],
            "languages": ["ja", "hi"]
        })
        self.app.post("/api/session/turn", json={
            "channel": "governed-channel", "speaker_id": "Kenji",
            "text": "नमस्ते", "language": "hi"
        })

    def note_id(self):
        """Returns the id of a note this session produced."""
        notes = self.app.get(
            "/api/session/notes?channel=governed-channel&actor=Kenji"
        ).get_json()["notes"]
        return notes[0]["id"]


class TestAccessPolicy(unittest.TestCase):
    """Test suite for the access rule itself (REQ-16)."""

    def test_a_participant_may_access_the_session(self):
        """Verify the people in the conversation can read what it produced."""
        meta = {"session_id": "s1", "participants": ["Kenji", "Priya"]}

        self.assertTrue(can_access(meta, "Priya"))

    def test_a_stranger_may_not(self):
        """Verify someone who was not in the room is refused."""
        meta = {"session_id": "s1", "participants": ["Kenji", "Priya"]}

        self.assertFalse(can_access(meta, "Someone Else"))

    def test_a_missing_actor_is_refused_when_participants_are_known(self):
        """Verify an unidentified caller cannot read a session that has an owner."""
        meta = {"session_id": "s1", "participants": ["Kenji"]}

        self.assertFalse(can_access(meta, None))
        self.assertFalse(can_access(meta, ""))

    def test_a_session_with_no_recorded_participants_is_open(self):
        """
        Verify a session that never recorded who was in it stays reachable. There is
        nobody to protect it from, and locking out the only person who could have asked
        for it would make the demo path unusable rather than safer.
        """
        meta = {"session_id": "s1", "participants": []}

        self.assertTrue(can_access(meta, None))

    def test_require_access_raises_for_a_stranger(self):
        """Verify the enforcing helper raises rather than returning a falsy value."""
        with self.assertRaises(AccessDeniedError):
            require_access({"session_id": "s1", "participants": ["Kenji"]}, "Mallory")


class TestApiAccessControl(AccessTestCase):
    """Test suite for the access rule on every artifact endpoint (REQ-16)."""

    def test_a_participant_can_retrieve_the_artifact(self):
        """Verify the normal path still works for someone who was in the session."""
        response = self.app.get(
            "/api/session/artifact?channel=governed-channel&actor=Kenji"
        )

        self.assertEqual(response.status_code, 200)

    def test_a_stranger_cannot_retrieve_the_artifact(self):
        """Verify a non-participant is refused with 403, not given the transcript."""
        response = self.app.get(
            "/api/session/artifact?channel=governed-channel&actor=Mallory"
        )

        self.assertEqual(response.status_code, 403)

    def test_a_stranger_cannot_export_the_artifact(self):
        """Verify export is not a way around retrieval's access check."""
        response = self.app.get(
            "/api/session/artifact/export?channel=governed-channel&format=markdown&actor=Mallory"
        )

        self.assertEqual(response.status_code, 403)

    def test_a_stranger_cannot_read_the_notes(self):
        """Verify the notes listing enforces the same rule as the whole artifact."""
        response = self.app.get(
            "/api/session/notes?channel=governed-channel&actor=Mallory"
        )

        self.assertEqual(response.status_code, 403)

    def test_a_stranger_cannot_edit_a_note(self):
        """Verify writes are governed too, not only reads."""
        note_id = self.note_id()

        response = self.app.patch(
            f"/api/session/notes/{note_id}?channel=governed-channel",
            json={"text": "rewritten by a stranger", "actor": "Mallory"}
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotEqual(
            self.server.artifacts.get_note(note_id).text, "rewritten by a stranger"
        )

    def test_the_actor_may_be_supplied_as_a_header(self):
        """
        Verify identity can travel in a header rather than the query string, so it stays
        out of URLs, server logs, and browser history.
        """
        response = self.app.get(
            "/api/session/artifact?channel=governed-channel",
            headers={"X-EchoSphere-Actor": "Priya"}
        )

        self.assertEqual(response.status_code, 200)


class TestEditProvenance(AccessTestCase):
    """Test suite for REQ-16 edit provenance."""

    def test_the_stored_artifact_records_who_edited_what(self):
        """Verify an audit of the artifact can recover the wording that was replaced."""
        note_id = self.note_id()
        original = self.server.artifacts.get_note(note_id).text

        self.app.patch(
            f"/api/session/notes/{note_id}?channel=governed-channel",
            json={"text": "corrected wording", "actor": "Priya"}
        )

        artifact = self.app.get(
            "/api/session/artifact?channel=governed-channel&actor=Kenji"
        ).get_json()["artifact"]
        note = next(item for item in artifact["notes"] if item["id"] == note_id)
        self.assertEqual(note["updated_by"], "Priya")
        self.assertEqual(note["edit_history"][0]["previous_text"], original)
        self.assertEqual(note["edit_history"][0]["by"], "Priya")


class TestCompleteDeletion(AccessTestCase):
    """Test suite for REQ-16 complete deletion."""

    def test_deleting_a_session_removes_transcript_notes_and_quizzes(self):
        """
        Verify deletion is complete rather than a hidden note. The acceptance criterion
        is that the transcript becomes unavailable too - a soft-deleted note over a
        retained transcript still holds everything that was said.
        """
        before = self.app.get(
            "/api/session/artifact?channel=governed-channel&actor=Kenji"
        ).get_json()["artifact"]
        self.assertTrue(before["transcript_turns"])

        deleted = self.app.delete(
            "/api/session/artifact?channel=governed-channel&actor=Kenji"
        )

        self.assertEqual(deleted.status_code, 200)
        self.assertGreater(deleted.get_json()["deleted_entities"], 0)
        after = self.app.get(
            f"/api/session/artifact?session_id={before['session_id']}&actor=Kenji"
        )
        self.assertEqual(after.status_code, 404)
        self.assertEqual(
            self.server.artifacts.list_turns(before["session_id"]), []
        )

    def test_a_stranger_cannot_delete_a_session(self):
        """Verify deletion is governed by the same rule as everything else."""
        response = self.app.delete(
            "/api/session/artifact?channel=governed-channel&actor=Mallory"
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            self.app.get(
                "/api/session/artifact?channel=governed-channel&actor=Kenji"
            ).status_code, 200
        )


class TestServerOnlyCredentials(AccessTestCase):
    """Test suite for REQ-16's server-only credential rule."""

    def test_no_artifact_response_carries_a_credential(self):
        """
        Verify the artifact surface never leaks server-side secrets. The artifact is the
        broadest payload the client receives, so it is the most likely place for a
        credential to be picked up by accident.
        """
        body = self.app.get(
            "/api/session/artifact?channel=governed-channel&actor=Kenji"
        ).data.decode("utf-8").lower()

        for forbidden in (
            "app_certificate", "customer_secret", "api_key", "notion_api_key",
            "tts_key", "authorization"
        ):
            self.assertNotIn(forbidden, body)


if __name__ == '__main__':
    unittest.main()
