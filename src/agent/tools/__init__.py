"""
Summary:
    `src/agent/tools/` holds the external tools the `TeachingAgent` may call during a
    session (REQ-18–20): Google Search for reference and task material, an Anki MCP
    server for spaced-repetition export, and Google Calendar for follow-up meetings.

    Every tool in here is optional. A server holding none of these credentials runs a
    complete session; the tools add reference cards, flashcards, and meetings on top of
    it. That is why nothing in this package raises into the session path: an absent or
    failing tool becomes a `tool.status` event, never an aborted turn.

Key Modules:
    - base: shared result/state vocabulary, error types, and the JSON HTTP helpers.
    - google_search: REQ-18 research client.
    - anki_mcp: REQ-19 vocabulary export over the Model Context Protocol.
    - google_calendar / email: REQ-20 scheduling and its supplementary confirmation mail.
    - dispatch: the single entry point the orchestrator and the API call through.
"""

from src.agent.tools.base import (
    EVENT_ANKI_EXPORTED,
    EVENT_MEETING_SCHEDULED,
    EVENT_REFERENCE_CARD,
    EVENT_TOOL_STATUS,
    TOOL_SCHEMA_VERSION,
    ToolInvocationError,
    ToolNotConfiguredError,
    ToolResult,
    ToolState,
)
from src.agent.tools.anki_mcp import AnkiMCPTool
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.email import ResendEmailSender
from src.agent.tools.google_calendar import GoogleCalendarTool
from src.agent.tools.google_search import GoogleSearchTool, SearchResult

__all__ = [
    "AnkiMCPTool",
    "EVENT_ANKI_EXPORTED",
    "EVENT_MEETING_SCHEDULED",
    "EVENT_REFERENCE_CARD",
    "EVENT_TOOL_STATUS",
    "GoogleCalendarTool",
    "GoogleSearchTool",
    "ResendEmailSender",
    "SearchResult",
    "TOOL_SCHEMA_VERSION",
    "ToolDispatcher",
    "ToolInvocationError",
    "ToolNotConfiguredError",
    "ToolResult",
    "ToolState",
]
