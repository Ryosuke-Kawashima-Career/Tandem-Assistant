"""
Summary:
    test_agent_tools_anki.py is the executable specification for REQ-19: a session's
    captured vocabulary and terminology leaves EchoSphere as Anki cards for spaced
    repetition, on request only, and never at the cost of the exports that always work.

Covers:
    - REQ-19 card building from stored notes, including which note types are exportable.
    - REQ-19 MCP call shape, vendor error handling, and the unconfigured server.
    - REQ-19 dispatch: `anki.exported` on success, `tool.status` when unavailable.
    - REQ-16: the export endpoint is governed like every other artifact endpoint.
"""

import unittest
from unittest.mock import MagicMock

from src.agent.tools.anki_mcp import AnkiMCPTool
from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    ToolState,
)
from src.agent.tools.dispatch import ToolDispatcher
from src.artifacts.models import NoteItem, NoteStatus
from src.sessions.models import SessionRecord
from src.server import app, server_instance

ENDPOINT = "https://anki-mcp.test/mcp"


def note(note_type="vocabulary", text="一期一会 - a once-in-a-lifetime encounter",
         status=NoteStatus.CAPTURED, note_id="note-1", mode="language_learning"):
    """Builds one stored note for the exporter under test."""
    return NoteItem(
        id=note_id,
        type=note_type,
        text=text,
        source_turn_ids=["turn-1"],
        confidence=0.95,
        status=status,
        session_id="sess-anki",
        mode=mode
    )


def learning_session():
    """A language-learning session whose vocabulary is worth revising later."""
    return SessionRecord.create(
        channel="anki-channel", mode="language_learning",
        languages=["ja"], participants=["Kenji"]
    )


class RecordingTransport:
    """A stand-in for the HTTP POST to the MCP server."""

    def __init__(self, response=None, error=None):
        """Store the JSON-RPC response to replay, or the error to raise."""
        self.response = response if response is not None else {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "{\"created\": 2}"}],
                       "isError": False},
        }
        self.error = error
        self.calls = []

    def __call__(self, url, payload, headers=None):
        """Record the request and replay the canned outcome."""
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        if self.error is not None:
            raise self.error
        return self.response


class TestAnkiCardBuilding(unittest.TestCase):
    """Test suite for turning stored notes into cards (REQ-19, TASK-12.3)."""

    def setUp(self):
        """A configured tool with a recording transport."""
        self.transport = RecordingTransport()
        self.tool = AnkiMCPTool(endpoint=ENDPOINT, deck="EchoSphere", transport=self.transport)

    def test_a_vocabulary_note_splits_into_front_and_back(self):
        """
        Verify the generator's own "term - meaning" wording becomes a two-sided card.

        A card whose front already contains the answer is not a memory test, so the
        separator the note generator writes is the one the exporter has to honor.
        """
        cards = self.tool.build_cards([note()])

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["front"], "一期一会")
        self.assertEqual(cards[0]["back"], "a once-in-a-lifetime encounter")

    def test_a_card_links_back_to_the_note_and_turn_it_came_from(self):
        """Verify provenance survives the export (REQ-19 source linking)."""
        cards = self.tool.build_cards([note()])

        self.assertEqual(cards[0]["source_note_id"], "note-1")
        self.assertEqual(cards[0]["source_turn_ids"], ["turn-1"])
        self.assertIn("sess-anki", cards[0]["tags"])

    def test_a_note_without_a_separator_becomes_a_single_sided_card(self):
        """Verify an unstructured note still exports rather than being dropped."""
        cards = self.tool.build_cards([note(text="敬語 is used with customers")])

        self.assertEqual(cards[0]["front"], "敬語 is used with customers")
        self.assertEqual(cards[0]["back"], "")

    def test_work_terminology_is_exportable_too(self):
        """
        Verify `term` and `glossary` notes export, not only learning vocabulary.

        REQ-19 is about vocabulary *and terminology*: an onboarding colleague drilling
        the team's acronyms is the same spaced-repetition problem as a learner drilling
        idioms.
        """
        cards = self.tool.build_cards([
            note(note_type="term", text="SD-RTN - Agora's software-defined network",
                 mode="international_work", note_id="note-2"),
            note(note_type="glossary", text="AEC - acoustic echo cancellation",
                 mode="international_work", note_id="note-3"),
        ])

        self.assertEqual(len(cards), 2)

    def test_notes_of_other_types_and_deleted_notes_are_skipped(self):
        """
        Verify a decision, an action, and a tombstone never become flashcards.

        Drilling "Ship the pilot to Mumbai first" as vocabulary is noise, and a deleted
        note is something a participant explicitly removed - re-exporting it would
        resurrect it in another application entirely.
        """
        cards = self.tool.build_cards([
            note(note_type="decision", text="Ship the pilot first.",
                 mode="international_work", note_id="note-4"),
            note(status=NoteStatus.DELETED, note_id="note-5"),
        ])

        self.assertEqual(cards, [])


class TestAnkiMCPCall(unittest.TestCase):
    """Test suite for the MCP wire call (REQ-19, TASK-12.3)."""

    def test_an_unconfigured_tool_refuses_rather_than_guessing_an_endpoint(self):
        """Verify a server with no MCP endpoint reports unconfigured."""
        tool = AnkiMCPTool(endpoint="")

        self.assertFalse(tool.is_configured)
        with self.assertRaises(ToolNotConfiguredError):
            tool.export_notes([note()])

    def test_the_export_is_a_single_json_rpc_tools_call(self):
        """Verify the MCP envelope names the tool and carries deck and cards."""
        transport = RecordingTransport()
        tool = AnkiMCPTool(
            endpoint=ENDPOINT, deck="EchoSphere", tool_name="add_notes", transport=transport
        )

        result = tool.export_notes([note()])

        self.assertEqual(len(transport.calls), 1)
        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["method"], "tools/call")
        self.assertEqual(payload["params"]["name"], "add_notes")
        self.assertEqual(payload["params"]["arguments"]["deck"], "EchoSphere")
        self.assertEqual(len(payload["params"]["arguments"]["notes"]), 1)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["deck"], "EchoSphere")

    def test_nothing_exportable_makes_no_call_at_all(self):
        """
        Verify an empty selection short-circuits instead of posting an empty batch.

        A session that produced no vocabulary should report zero, not create an empty
        deck entry in someone's collection.
        """
        transport = RecordingTransport()
        tool = AnkiMCPTool(endpoint=ENDPOINT, transport=transport)

        result = tool.export_notes([note(note_type="decision", mode="international_work")])

        self.assertEqual(result["exported"], 0)
        self.assertEqual(transport.calls, [])

    def test_an_mcp_tool_error_is_raised_rather_than_reported_as_success(self):
        """
        Verify `isError` is honored (MCP reports tool failures inside a 200 response).

        This is the failure that would otherwise be invisible: the HTTP call succeeded,
        so only the body says the cards were rejected.
        """
        tool = AnkiMCPTool(endpoint=ENDPOINT, transport=RecordingTransport(response={
            "jsonrpc": "2.0", "id": 1,
            "result": {"content": [{"type": "text", "text": "deck is locked"}],
                       "isError": True},
        }))

        with self.assertRaises(ToolInvocationError):
            tool.export_notes([note()])

    def test_a_json_rpc_error_is_raised(self):
        """Verify a protocol-level error is not mistaken for an export."""
        tool = AnkiMCPTool(endpoint=ENDPOINT, transport=RecordingTransport(response={
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }))

        with self.assertRaises(ToolInvocationError):
            tool.export_notes([note()])


class TestAnkiDispatch(unittest.TestCase):
    """Test suite for announcing an export over RTC (REQ-19, TASK-12.3)."""

    def setUp(self):
        """A recording data stream and a configured Anki tool."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            anki=AnkiMCPTool(endpoint=ENDPOINT, deck="EchoSphere",
                             transport=RecordingTransport())
        )

    def published(self, event_type):
        """Returns the payloads published under one event type."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == event_type
        ]

    def test_a_successful_export_announces_the_count_and_deck(self):
        """Verify `anki.exported` carries what was sent and where it went."""
        session = learning_session()

        result = self.dispatcher.export_vocabulary(session, [note()])

        self.assertTrue(result.ok)
        payload = self.published("anki.exported")[0]
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["export"]["exported"], 1)
        self.assertEqual(payload["export"]["deck"], "EchoSphere")

    def test_an_unconfigured_server_reports_unavailable(self):
        """Verify a missing MCP endpoint degrades to a status event, not an exception."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream, anki=AnkiMCPTool(endpoint="")
        )

        result = dispatcher.export_vocabulary(learning_session(), [note()])

        self.assertEqual(result.state, ToolState.UNAVAILABLE)
        self.assertEqual(self.published("tool.status")[0]["tool"], "anki")
        self.assertFalse(self.published("anki.exported"))


class TestAnkiExportEndpoint(unittest.TestCase):
    """Test suite for the on-demand export endpoint (REQ-19, REQ-16)."""

    def setUp(self):
        """Start from an empty registry, with the server's Anki tool unconfigured."""
        self.app = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.original_anki = self.server.tools.anki
        self.app.post("/api/session/start", json={
            "channel": "anki-channel", "mode": "language_learning",
            "participants": ["Kenji"], "languages": ["ja"]
        })
        self.app.post("/api/session/turn", json={
            "channel": "anki-channel", "speaker_id": "Kenji",
            "text": "一期一会", "language": "ja"
        })

    def tearDown(self):
        """Restore the server's own tool so tests do not leak configuration."""
        self.server.tools.anki = self.original_anki

    def test_an_unconfigured_export_answers_503_rather_than_pretending(self):
        """
        Verify the endpoint refuses plainly, matching the Notion export's precedent.

        REQ-19 keeps Markdown and Notion export available regardless, so this is a
        missing convenience - but it must never look like a successful export.
        """
        self.server.tools.anki = AnkiMCPTool(endpoint="")

        response = self.app.post("/api/tools/anki/export", json={
            "channel": "anki-channel", "actor": "Kenji"
        })

        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["state"], "unavailable")

    def test_a_configured_export_sends_the_sessions_vocabulary(self):
        """Verify the endpoint exports the notes the session actually captured."""
        transport = RecordingTransport()
        self.server.tools.anki = AnkiMCPTool(
            endpoint=ENDPOINT, deck="EchoSphere", transport=transport
        )

        response = self.app.post("/api/tools/anki/export", json={
            "channel": "anki-channel", "actor": "Kenji"
        })

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertGreaterEqual(body["export"]["exported"], 1)
        sent = transport.calls[0]["payload"]["params"]["arguments"]["notes"]
        self.assertTrue(all("front" in card for card in sent))

    def test_a_non_participant_is_refused(self):
        """
        Verify the export is governed like every other artifact endpoint (REQ-16).

        The cards carry the conversation's content, so shipping them to a stranger's
        collection is the same disclosure as handing over the transcript.
        """
        self.server.tools.anki = AnkiMCPTool(endpoint=ENDPOINT, transport=RecordingTransport())

        response = self.app.post("/api/tools/anki/export", json={
            "channel": "anki-channel", "actor": "Stranger"
        })

        self.assertEqual(response.status_code, 403)

    def test_markdown_export_still_works_when_anki_is_unavailable(self):
        """Verify REQ-19's explicit guarantee: the always-available export is untouched."""
        self.server.tools.anki = AnkiMCPTool(endpoint="")
        self.app.post("/api/tools/anki/export", json={
            "channel": "anki-channel", "actor": "Kenji"
        })

        response = self.app.get(
            "/api/session/artifact/export?channel=anki-channel&format=markdown&actor=Kenji"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("EchoSphere", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
