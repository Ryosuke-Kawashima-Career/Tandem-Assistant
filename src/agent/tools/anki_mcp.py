"""
Summary:
    anki_mcp.py exports a session's vocabulary and terminology to Anki over the Model
    Context Protocol (REQ-19), so what was learned in one conversation is revised on a
    spaced-repetition schedule afterwards.

    Only vocabulary-shaped notes leave: a decision or an action item is a commitment, not
    something to drill, and a note a participant deleted must not reappear in another
    application. The card front/back split follows the separator the note generator
    itself writes ("term - meaning"), because a flashcard whose front already contains
    the answer tests nothing.

Key Classes:
    - AnkiMCPTool: the configured MCP client.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    call_transport,
    http_post_json,
)

logger = logging.getLogger("echosphere.agent.tools.anki")

# The MCP method that invokes a server-side tool, and the tool this one calls by default.
# The tool *name* varies between Anki MCP servers, so it is configurable rather than
# frozen here - check the server's own tool listing before overriding it.
MCP_METHOD = "tools/call"
DEFAULT_TOOL_NAME = "add_notes"
DEFAULT_DECK = "EchoSphere"

# Note types worth drilling. Learning mode contributes `vocabulary`; work mode
# contributes the terminology an onboarding colleague has to absorb (REQ-19).
EXPORTABLE_NOTE_TYPES = ("vocabulary", "term", "glossary")

# Separators the note generator and the prompt families use between a term and its
# meaning, most specific first so an em dash is not mistaken for a hyphen.
CARD_SEPARATORS = (" — ", " – ", " - ", ": ")

# A deleted note is one a participant explicitly removed from the session.
DELETED_STATUS = "deleted"


class AnkiMCPTool:
    """
    Pushes a session's vocabulary to an Anki MCP server (REQ-19).

    On-demand only: nothing here runs from the turn path. A learner asks for the export
    when the conversation is over, which is also the only moment the vocabulary list is
    complete.
    """

    name = "anki"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        deck: Optional[str] = None,
        tool_name: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the client from explicit values or the environment.

        Algorithm:
        1. Resolve the MCP endpoint, the destination deck, and the tool name to call.
        2. Bind the transport - injected in tests, a plain JSON POST otherwise.
        3. Leave the tool unconfigured, not failed, when no endpoint is set.
        """
        self.endpoint = endpoint if endpoint is not None else os.getenv("ANKI_MCP_URL", "")
        self.deck = deck if deck is not None else os.getenv("ANKI_MCP_DECK", DEFAULT_DECK)
        self.tool_name = (
            tool_name if tool_name is not None
            else os.getenv("ANKI_MCP_TOOL_NAME", DEFAULT_TOOL_NAME)
        )
        self.api_key = api_key if api_key is not None else os.getenv("ANKI_MCP_API_KEY", "")
        self._transport = transport or http_post_json
        self._request_id = 0

    @property
    def is_configured(self) -> bool:
        """Whether an MCP server has been named for this deployment."""
        return bool(self.endpoint)

    def build_cards(self, notes: Sequence[Any]) -> List[Dict[str, Any]]:
        """
        Turns stored notes into Anki cards (REQ-19).

        Algorithm:
        1. Drop deleted notes and any type that is not vocabulary or terminology.
        2. Split "term - meaning" into the two sides of a card; a note with no separator
           becomes a single-sided card rather than being dropped.
        3. Carry the note and turn ids so a card can be traced back to what was said.
        """
        cards: List[Dict[str, Any]] = []

        for note in notes:
            note_type = str(getattr(note, "type", "")).strip().lower()
            status = str(getattr(note, "status", "")).strip().lower()
            text = str(getattr(note, "text", "")).strip()

            if not text or status == DELETED_STATUS or note_type not in EXPORTABLE_NOTE_TYPES:
                continue

            front, back = self._split(text)
            cards.append({
                "front": front,
                "back": back,
                "tags": [
                    "echosphere",
                    str(getattr(note, "session_id", "")),
                    note_type,
                ],
                "source_note_id": str(getattr(note, "id", "")),
                "source_turn_ids": list(getattr(note, "source_turn_ids", []) or []),
            })

        return cards

    @staticmethod
    def _split(text: str) -> tuple:
        """Splits a note into card front and back on the first separator it contains."""
        for separator in CARD_SEPARATORS:
            if separator in text:
                front, back = text.split(separator, 1)
                return front.strip(), back.strip()
        return text, ""

    def export_notes(
        self,
        notes: Sequence[Any],
        deck: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exports a session's vocabulary as cards, returning what was sent (REQ-19).

        Algorithm:
        1. Refuse when unconfigured.
        2. Build the cards; an empty selection returns zero without calling anything, so
           a session that produced no vocabulary does not create an empty deck entry.
        3. Send one MCP `tools/call`, and treat both a JSON-RPC error and the protocol's
           own `isError` flag as failures - the second arrives inside a 200 response, so
           only the body says the cards were refused.

        Raises:
            ToolNotConfiguredError: no MCP endpoint on this server.
            ToolInvocationError: the server could not be reached, or refused the cards.
        """
        if not self.is_configured:
            raise ToolNotConfiguredError(
                "Anki export is not configured. Set ANKI_MCP_URL to the Anki MCP "
                "server, or export the session as Markdown instead."
            )

        target_deck = (deck or self.deck or DEFAULT_DECK).strip()
        cards = self.build_cards(notes)
        if not cards:
            return {"exported": 0, "deck": target_deck, "terms": []}

        self._request_id += 1
        headers = {
            "Content-Type": "application/json",
            # Streamable-HTTP MCP servers negotiate on Accept; this client reads JSON
            # replies only, so it asks for JSON explicitly rather than an SSE stream.
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = call_transport(self._transport, self.endpoint, {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": MCP_METHOD,
            "params": {
                "name": self.tool_name,
                "arguments": {"deck": target_deck, "notes": cards},
            },
        }, headers)

        self._raise_for_mcp_error(body, target_deck)

        logger.info("Exported %d card(s) to Anki deck %r.", len(cards), target_deck)
        return {
            "exported": len(cards),
            "deck": target_deck,
            "terms": [card["front"] for card in cards],
        }

    @staticmethod
    def _raise_for_mcp_error(body: Any, deck: str) -> None:
        """Raises when the MCP server reported a protocol or tool-level failure."""
        if not isinstance(body, dict):
            raise ToolInvocationError("The Anki MCP server returned an unreadable response.")

        error = body.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else error
            raise ToolInvocationError(f"Anki MCP refused the call: {message}")

        result = body.get("result")
        if isinstance(result, dict) and result.get("isError"):
            raise ToolInvocationError(
                f"Anki MCP could not add the cards to deck {deck!r}: "
                f"{_first_text(result)}"
            )


def _first_text(result: Dict[str, Any]) -> str:
    """Reads the first text block out of an MCP tool result, for an error message."""
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            return str(block["text"])[:200]
    return "no detail supplied"
