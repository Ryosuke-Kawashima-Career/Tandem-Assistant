"""
Summary:
    test_notion_database_export.py is the executable specification for the REQ-15
    amendment carried by TASK-12.5: besides the existing page export, a session can be
    filed as a row in a Notion database, with its vocabulary and knowledge checks written
    as toggle questions a learner can revise from.

Covers:
    - REQ-15 database target: parent selection, title property, and toggle children.
    - REQ-15 question toggles: the answer stays hidden behind the question.
    - REQ-15 regression: the page export and the unconfigured 503 are unchanged.
"""

import unittest

from src.artifacts.adapters import ExportNotConfiguredError, NotionExportAdapter
from src.artifacts.models import (
    ARTIFACT_SCHEMA_VERSION,
    NoteItem,
    NoteStatus,
    QuizItem,
    SessionArtifact,
    TranscriptTurn,
)
from src.server import app, server_instance

DATABASE_ID = "2aeeddbc2b9b809d8253d2a01f64c053"
PARENT_PAGE_ID = "page-abc123"


class RecordingTransport:
    """A stand-in for the HTTP POST to the Notion API."""

    def __init__(self, response=None, error=None):
        """Store the response to replay, or the error to raise instead."""
        self.response = response if response is not None else {
            "id": "notion-page-1",
            "url": "https://notion.so/notion-page-1",
        }
        self.error = error
        self.calls = []

    def __call__(self, url, payload, headers=None):
        """Record the request and replay the canned outcome."""
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        if self.error is not None:
            raise self.error
        return self.response


def artifact_with_learning_content():
    """A finished learning session holding one vocabulary note and one quiz."""
    return SessionArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        session_id="sess-notion",
        mode="language_learning",
        started_at=1756000000.0,
        ended_at=1756003600.0,
        participants=["Kenji", "Priya"],
        languages=["ja", "hi"],
        transcript_turns=[TranscriptTurn.create("sess-notion", "Kenji", "一期一会", "ja")],
        notes=[NoteItem(
            id="note-1",
            type="vocabulary",
            text="一期一会 - a once-in-a-lifetime encounter",
            source_turn_ids=["turn-1"],
            confidence=0.95,
            status=NoteStatus.CAPTURED,
            session_id="sess-notion",
            mode="language_learning"
        )],
        quizzes=[QuizItem(
            id="quiz-1",
            prompt="What does 一期一会 convey?",
            answer_type="multiple_choice",
            expected_answer="Treasure this one meeting",
            explanation="It comes from the tea ceremony tradition.",
            options=["Treasure this one meeting", "See you tomorrow"],
            source_turn_ids=["turn-1"],
            difficulty="medium",
            target_language="ja",
            session_id="sess-notion",
            mode="language_learning"
        )]
    )


class TestNotionTargetSelection(unittest.TestCase):
    """Test suite for choosing between the page and database targets (TASK-12.5)."""

    def test_an_adapter_with_neither_target_is_unconfigured(self):
        """Verify a token alone is not enough: the export needs somewhere to write."""
        adapter = NotionExportAdapter(api_key="secret", parent_page_id="", database_id="")

        self.assertFalse(adapter.is_configured)
        with self.assertRaises(ExportNotConfiguredError):
            adapter.export(artifact_with_learning_content())

    def test_a_database_id_alone_configures_the_adapter(self):
        """
        Verify `NOTION_DATABASE_ID` is a complete configuration on its own.

        This is the variable already sitting in `.env`; before TASK-12.5 it configured
        nothing, and an export against it answered 503.
        """
        adapter = NotionExportAdapter(
            api_key="secret", parent_page_id="", database_id=DATABASE_ID
        )

        self.assertTrue(adapter.is_configured)
        self.assertEqual(adapter.default_target, "database")

    def test_a_page_id_alone_still_selects_the_page_export(self):
        """Verify the pre-existing configuration keeps its pre-existing behavior."""
        adapter = NotionExportAdapter(
            api_key="secret", parent_page_id=PARENT_PAGE_ID, database_id=""
        )

        self.assertEqual(adapter.default_target, "page")

    def test_an_unknown_target_is_rejected(self):
        """Verify a typo names an error rather than silently exporting somewhere else."""
        adapter = NotionExportAdapter(
            api_key="secret", database_id=DATABASE_ID, transport=RecordingTransport()
        )

        with self.assertRaises(ValueError):
            adapter.export(artifact_with_learning_content(), target="workspace")


class TestNotionDatabaseExport(unittest.TestCase):
    """Test suite for the database row an exported session becomes (TASK-12.5)."""

    def setUp(self):
        """A database-configured adapter over a recording transport."""
        self.transport = RecordingTransport()
        self.adapter = NotionExportAdapter(
            api_key="secret", database_id=DATABASE_ID, transport=self.transport
        )

    def test_the_export_creates_a_row_in_the_configured_database(self):
        """Verify the parent is the database, not a page."""
        result = self.adapter.export(artifact_with_learning_content())

        payload = self.transport.calls[0]["payload"]
        self.assertEqual(payload["parent"], {"database_id": DATABASE_ID})
        self.assertEqual(result["page_id"], "notion-page-1")
        self.assertEqual(result["target"], "database")

    def test_the_row_is_titled_by_the_property_id_every_database_has(self):
        """
        Verify the title is addressed as `title` rather than by a guessed column name.

        A database's title column is named by whoever created it ("Name", "Session",
        anything); Notion accepts the property id `title` for it in every database, and
        guessing the name 400s against half of them.
        """
        self.adapter.export(artifact_with_learning_content())

        properties = self.transport.calls[0]["payload"]["properties"]
        self.assertIn("title", properties)
        text = properties["title"]["title"][0]["text"]["content"]
        self.assertIn("Language Learning", text)

    def test_a_quiz_becomes_a_toggle_that_hides_its_answer(self):
        """
        Verify the question is visible and the answer is inside the toggle.

        This is the point of the toggle format in the user story: a learner revising the
        session has to attempt the answer before seeing it, or the page is a transcript
        with extra steps.
        """
        self.adapter.export(artifact_with_learning_content())

        toggles = [
            block for block in self.transport.calls[0]["payload"]["children"]
            if block["type"] == "toggle"
        ]
        quiz_toggle = next(
            block for block in toggles
            if "一期一会" in block["toggle"]["rich_text"][0]["text"]["content"]
            and "?" in block["toggle"]["rich_text"][0]["text"]["content"]
        )

        hidden = str(quiz_toggle["toggle"]["children"])
        self.assertIn("Treasure this one meeting", hidden)
        self.assertNotIn(
            "Treasure this one meeting",
            quiz_toggle["toggle"]["rich_text"][0]["text"]["content"]
        )

    def test_a_vocabulary_note_becomes_a_toggle_of_term_and_meaning(self):
        """Verify vocabulary is revisable the same way, term in front of meaning."""
        self.adapter.export(artifact_with_learning_content())

        toggles = [
            block for block in self.transport.calls[0]["payload"]["children"]
            if block["type"] == "toggle"
        ]
        vocab_toggle = next(
            block for block in toggles
            if block["toggle"]["rich_text"][0]["text"]["content"].strip() == "一期一会"
        )

        self.assertIn(
            "a once-in-a-lifetime encounter", str(vocab_toggle["toggle"]["children"])
        )

    def test_the_row_still_carries_the_full_session_document(self):
        """
        Verify the toggles supplement the Markdown rendering rather than replacing it.

        The page and database exports must say the same thing about a session; dropping
        the transcript from one of them is how the two drift apart.
        """
        self.adapter.export(artifact_with_learning_content())

        children = self.transport.calls[0]["payload"]["children"]
        rendered = str(children)
        self.assertIn("Transcript", rendered)
        self.assertIn("Kenji", rendered)


class TestNotionPageExportUnchanged(unittest.TestCase):
    """Regression suite: TASK-12.5 must not alter the shipped page export (REQ-15)."""

    def test_a_page_export_still_writes_paragraphs_under_the_parent_page(self):
        """Verify the existing export path is untouched by the new target."""
        transport = RecordingTransport()
        adapter = NotionExportAdapter(
            api_key="secret", parent_page_id=PARENT_PAGE_ID, transport=transport
        )

        result = adapter.export(artifact_with_learning_content())

        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["parent"], {"page_id": PARENT_PAGE_ID})
        self.assertTrue(
            all(block["type"] == "paragraph" for block in payload["children"])
        )
        self.assertEqual(result["target"], "page")

    def test_a_rejected_export_raises_rather_than_reporting_success(self):
        """Verify a vendor refusal is still an error (REQ-15's no-silent-success rule)."""
        adapter = NotionExportAdapter(
            api_key="secret", parent_page_id=PARENT_PAGE_ID,
            transport=RecordingTransport(error=RuntimeError("Notion rejected the export"))
        )

        with self.assertRaises(RuntimeError):
            adapter.export(artifact_with_learning_content())


class TestNotionExportEndpoint(unittest.TestCase):
    """Test suite for selecting the target over HTTP (REQ-15)."""

    def setUp(self):
        """One finished learning session to export."""
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.app.post("/api/session/start", json={
            "channel": "notion-channel", "mode": "language_learning",
            "participants": ["Kenji"], "languages": ["ja"]
        })
        self.app.post("/api/session/turn", json={
            "channel": "notion-channel", "speaker_id": "Kenji",
            "text": "一期一会", "language": "ja"
        })

    def test_an_unconfigured_notion_export_still_answers_503(self):
        """Verify the unconfigured path is unchanged for both targets."""
        response = self.app.get(
            "/api/session/artifact/export"
            "?channel=notion-channel&format=notion&target=database&actor=Kenji"
        )

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])

    def test_an_unknown_target_is_a_400(self):
        """Verify an unsupported target is rejected before any credential check."""
        response = self.app.get(
            "/api/session/artifact/export"
            "?channel=notion-channel&format=notion&target=workspace&actor=Kenji"
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
