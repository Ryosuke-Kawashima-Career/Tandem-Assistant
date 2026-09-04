"""
Summary:
    dispatch.py is the one door the agent's tools are called through (REQ-18–20).

    It exists so the rule that matters is enforced in one place rather than at every call
    site: a tool that is missing, unreachable, or failing becomes a `tool.status` event
    and a returned result, never an exception into the session. A learner's turn must not
    be lost because a search key expired, and someone who clicked "schedule" must never
    be left with a silent no-op (REQ-20).

    Success is announced as its own domain event - `reference.card`, `anki.exported`,
    `meeting.scheduled` - rather than as a status, so a client draws the thing that
    happened instead of interpreting a state machine.

Key Classes:
    - ToolDispatcher: builds, guards, and announces every agent tool call.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent.tools.anki_mcp import AnkiMCPTool
from src.agent.tools.base import (
    EVENT_ANKI_EXPORTED,
    EVENT_MEETING_SCHEDULED,
    EVENT_REFERENCE_CARD,
    EVENT_TOOL_STATUS,
    TOOL_SCHEMA_VERSION,
    ToolNotConfiguredError,
    ToolResult,
    ToolState,
)
from src.agent.tools.email import ResendEmailSender
from src.agent.tools.google_calendar import GoogleCalendarTool
from src.agent.tools.google_search import GoogleSearchTool
from src.artifacts.models import stable_entity_id

logger = logging.getLogger("echosphere.agent.tools.dispatch")

# Two workers: agent-initiated lookups are occasional and a backlog behind a hung vendor
# should stay small enough to notice rather than large enough to hide.
ASYNC_WORKERS = 2


class ToolDispatcher:
    """
    Runs the agent's external tools and publishes what they produced.

    Every tool is optional and injected, so a session can run - and be reasoned about -
    with any subset of them absent or unconfigured.
    """

    def __init__(
        self,
        data_stream: Optional[Any] = None,
        search: Optional[GoogleSearchTool] = None,
        calendar: Optional[GoogleCalendarTool] = None,
        anki: Optional[AnkiMCPTool] = None,
        email: Optional[ResendEmailSender] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize the dispatcher over its (optional) tools and data stream.

        Algorithm:
        1. Bind the data stream used to publish tool events.
        2. Bind each tool; `None` is treated exactly like an unconfigured one.
        3. Bind the executor for agent-initiated calls, created lazily when first needed.
        """
        self.data_stream = data_stream
        self.search = search
        self.calendar = calendar
        self.anki = anki
        self.email = email

        self._executor = executor
        self._executor_lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        data_stream: Optional[Any] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> "ToolDispatcher":
        """Builds a dispatcher with every tool resolved from the environment."""
        return cls(
            data_stream=data_stream,
            search=GoogleSearchTool(),
            calendar=GoogleCalendarTool(),
            anki=AnkiMCPTool(),
            email=ResendEmailSender(),
            executor=executor
        )

    def status(self) -> Dict[str, bool]:
        """
        Reports which tools this server can actually run.

        The frontend uses this to avoid offering a control whose only possible outcome is
        a 503; it deliberately says nothing about the credentials behind each answer.
        """
        return {
            "search": self._is_configured(self.search),
            "anki": self._is_configured(self.anki),
            "calendar": self._is_configured(self.calendar),
            "email": self._is_configured(self.email),
        }

    @staticmethod
    def _is_configured(tool: Optional[Any]) -> bool:
        """Whether a tool exists and holds credentials. A missing tool is unconfigured."""
        return bool(tool is not None and getattr(tool, "is_configured", False))

    # -- Google Search (REQ-18) ------------------------------------------------------

    def search_reference(
        self,
        session: Any,
        query: str,
        materials: Optional[bool] = None,
        requested_by: str = ""
    ) -> ToolResult:
        """
        Looks a topic up and publishes the reference card for it (REQ-18).

        Algorithm:
        1. Refuse early - and audibly - when the tool is unconfigured.
        2. Scope the query to the session: its language in `language_learning`, the task
           material in `international_work`.
        3. Publish one `reference.card` carrying the topic as asked and the results.

        Returns a `ToolResult` in every case; failures are reported, never raised.
        """
        topic = (query or "").strip()
        if not topic:
            return self._status_event(
                "search", session, ToolState.FAILED, "A lookup needs something to look up."
            )
        if not self._is_configured(self.search):
            return self._status_event(
                "search", session, ToolState.UNAVAILABLE,
                "Google Search is not configured on this server."
            )

        want_materials = (
            bool(materials) if materials is not None
            else not _grades_language(session)
        )
        language = _first_language(session)

        try:
            scoped = self.search.build_query(topic, language=language, materials=want_materials)
            results = self.search.search(scoped)
        except ToolNotConfiguredError as exc:
            return self._status_event("search", session, ToolState.UNAVAILABLE, str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning("Reference lookup for %r failed: %s", topic, exc)
            return self._status_event("search", session, ToolState.FAILED, str(exc))

        card = {
            "query": topic,
            "scoped_query": scoped,
            "materials": want_materials,
            "language": language,
            "requested_by": requested_by,
            "results": [result.to_dict() for result in results],
        }
        self._emit(EVENT_REFERENCE_CARD, session, "search", "card", card, topic)
        return ToolResult(tool="search", state=ToolState.OK, payload=card)

    def search_reference_async(self, session: Any, query: str, **kwargs: Any) -> Future:
        """
        Runs `search_reference` off the caller's thread (REQ-18).

        The agent-initiated caller is the turn path, and REQ-18 forbids a lookup from
        delaying a spoken reply - so this has to return before the vendor does.
        """
        return self._submit(self.search_reference, session, query, **kwargs)

    # -- Anki MCP (REQ-19) -----------------------------------------------------------

    def export_vocabulary(
        self,
        session: Any,
        notes: Sequence[Any],
        deck: Optional[str] = None
    ) -> ToolResult:
        """
        Exports a session's vocabulary to Anki and announces it (REQ-19).

        On-demand only: nothing calls this from the turn path. Markdown and Notion export
        remain available regardless of what happens here.
        """
        if not self._is_configured(self.anki):
            return self._status_event(
                "anki", session, ToolState.UNAVAILABLE,
                "No Anki MCP server is configured. Export as Markdown instead."
            )

        try:
            export = self.anki.export_notes(notes, deck=deck)
        except ToolNotConfiguredError as exc:
            return self._status_event("anki", session, ToolState.UNAVAILABLE, str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning("Anki export failed: %s", exc)
            return self._status_event("anki", session, ToolState.FAILED, str(exc))

        self._emit(
            EVENT_ANKI_EXPORTED, session, "anki", "export", export, export.get("deck", "")
        )
        return ToolResult(tool="anki", state=ToolState.OK, payload=export)

    # -- Google Calendar (REQ-20) ----------------------------------------------------

    def schedule_meeting(
        self,
        session: Any,
        summary: str = "",
        start_time: Any = None,
        duration_minutes: int = 30,
        attendees: Sequence[str] = (),
        description: str = "",
        requested_by: str = ""
    ) -> ToolResult:
        """
        Books a follow-up meeting and announces it (REQ-20).

        Algorithm:
        1. Refuse audibly when unconfigured - a silent no-op leaves someone believing a
           meeting exists.
        2. Create the event, letting Google deliver the invitations.
        3. Send the supplementary confirmation email only after that succeeded, and treat
           its failure as cosmetic: the meeting is already in everyone's calendar, and
           reporting failure here invites a second, duplicate booking.
        """
        if not self._is_configured(self.calendar):
            return self._status_event(
                "calendar", session, ToolState.UNAVAILABLE,
                "Google Calendar is not configured on this server."
            )

        title = summary.strip() if summary else _default_summary(session)

        try:
            meeting = self.calendar.schedule(
                summary=title,
                start_time=start_time,
                duration_minutes=duration_minutes,
                attendees=attendees,
                description=description or _default_description(session)
            )
        except ToolNotConfiguredError as exc:
            return self._status_event("calendar", session, ToolState.UNAVAILABLE, str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning("Scheduling a follow-up failed: %s", exc)
            return self._status_event("calendar", session, ToolState.FAILED, str(exc))

        meeting["summary"] = title
        meeting["requested_by"] = requested_by
        meeting["confirmation_email_sent"] = self._send_confirmation(meeting, title)

        self._emit(
            EVENT_MEETING_SCHEDULED, session, "calendar", "meeting", meeting,
            meeting.get("event_id", "")
        )
        return ToolResult(tool="calendar", state=ToolState.OK, payload=meeting)

    def _send_confirmation(self, meeting: Dict[str, Any], title: str) -> bool:
        """
        Sends the courtesy copy of an invitation Google has already delivered (REQ-20).

        Never raises: this runs after the meeting exists, so its failure is a missing
        email, not a missing meeting.
        """
        recipients = list(meeting.get("attendees") or [])
        if not recipients or not self._is_configured(self.email):
            return False

        try:
            self.email.send(
                to=recipients,
                subject=f"EchoSphere: {title}",
                text=(
                    f"{title}\n\n"
                    f"When: {meeting.get('start_time')} - {meeting.get('end_time')}\n"
                    f"Calendar invitation: {meeting.get('html_link') or 'sent by Google Calendar'}\n\n"
                    "This is a copy of the calendar invitation you have already received."
                )
            )
            return True
        except Exception as exc:  # noqa: BLE001 - cosmetic by contract
            logger.info("Confirmation email not sent: %s", exc)
            return False

    # -- Publishing ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        session: Any,
        tool: str,
        key: str,
        entity: Dict[str, Any],
        identity: Any = ""
    ) -> bool:
        """
        Publishes one successful tool outcome under the shared envelope.

        The envelope matches the artifact and translation events - schema version,
        idempotent event id, session, mode - so a client routes and deduplicates tool
        events by shape alone rather than knowing each type.
        """
        if self.data_stream is None:
            return False

        payload = {
            "schema_version": TOOL_SCHEMA_VERSION,
            "event_id": stable_entity_id(
                "evt", _session_id(session), event_type, identity
            ),
            "session_id": _session_id(session),
            "mode": _mode(session),
            "tool": tool,
            "timestamp": int(time.time() * 1000),
            key: entity,
        }
        return bool(self.data_stream.send_tool_event(event_type, payload))

    def _status_event(
        self,
        tool: str,
        session: Any,
        state: str,
        reason: str
    ) -> ToolResult:
        """Publishes one `tool.status` and returns the matching result."""
        if self.data_stream is not None:
            self.data_stream.send_tool_event(EVENT_TOOL_STATUS, {
                "schema_version": TOOL_SCHEMA_VERSION,
                "session_id": _session_id(session),
                "mode": _mode(session),
                "tool": tool,
                "state": state,
                "reason": reason,
                "timestamp": int(time.time() * 1000),
            })

        logger.info("Tool %s reported %s: %s", tool, state, reason)
        return ToolResult(tool=tool, state=state, reason=reason)

    def _submit(self, function: Callable[..., ToolResult], *args: Any, **kwargs: Any) -> Future:
        """Runs one dispatch call on the shared executor, creating it on first use."""
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=ASYNC_WORKERS, thread_name_prefix="agent-tools"
                )
        return self._executor.submit(function, *args, **kwargs)


# -- Session helpers -----------------------------------------------------------------
# Duck-typed rather than importing SessionRecord: the dispatcher only ever needs an id,
# a mode, and a language, and keeping it structural is what lets a caller pass the
# rebuilt record an ended session leaves behind in its stored metadata.

def _session_id(session: Any) -> str:
    """Reads a session's id, tolerating the absence of a session entirely."""
    return str(getattr(session, "session_id", "") or "")


def _mode(session: Any) -> str:
    """Reads a session's mode as its wire value."""
    mode = getattr(session, "mode", None)
    return str(getattr(mode, "value", mode) or "")


def _grades_language(session: Any) -> bool:
    """Whether this is a language-learning session (decides the search's scoping)."""
    mode = getattr(session, "mode", None)
    return bool(getattr(mode, "grades_language", False))


def _first_language(session: Any) -> str:
    """The session's primary language, used to scope a learning lookup."""
    languages: List[str] = list(getattr(session, "languages", []) or [])
    return str(languages[0]) if languages else ""


def _default_summary(session: Any) -> str:
    """Names a follow-up meeting after the session that asked for it."""
    return (
        "EchoSphere practice follow-up" if _grades_language(session)
        else "EchoSphere work follow-up"
    )


def _default_description(session: Any) -> str:
    """Explains where a meeting invitation came from, for whoever receives it."""
    return (
        "Follow-up session scheduled from EchoSphere "
        f"(session {_session_id(session)}, mode {_mode(session)})."
    )
