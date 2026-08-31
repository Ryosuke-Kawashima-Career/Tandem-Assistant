"""
Summary:
    service.py owns session lifecycle and mode enforcement (REQ-12).

    It is the single place that answers "what mode is this channel in?", so prompt
    selection, RTC events, quizzes, and notes all branch on one source rather than each
    carrying its own copy of the mode. `server.py` composes it; nothing else stores mode.

Key Classes:
    - SessionService: create / retrieve / end sessions, with an immutable mode.
    - SessionNotFoundError: raised by callers that require an existing session.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from src.sessions.models import (
    SESSION_SCHEMA_VERSION,
    InvalidSessionModeError,
    SessionMode,
    SessionModeImmutableError,
    SessionRecord,
)

logger = logging.getLogger("echosphere.sessions.service")


class SessionNotFoundError(LookupError):
    """Raised when a channel has no active session but the caller requires one."""


class SessionService:
    """
    In-memory registry of active sessions, keyed by RTC channel.

    In-memory is sufficient and deliberate at this stage: a session lives exactly as long
    as the RTC channel it describes, and durable storage of what a session *produced* is
    Phase 10's artifact repository, not this. The lock is real, though - the Convo AI
    scaffolding path resolves the mode from a background executor thread while the
    request thread may be creating or ending the session.
    """

    def __init__(self):
        """
        Initialize an empty session registry.

        Algorithm:
        1. Create the channel -> SessionRecord map.
        2. Create the lock guarding it against the background scaffolding threads.
        """
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        logger.info("SessionService initialized.")

    def create_session(
        self,
        channel: str,
        mode: Any,
        languages: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> SessionRecord:
        """
        Creates a session with an immutable mode (REQ-12).

        Algorithm:
        1. Build and validate the record - an invalid or missing mode raises here.
        2. Replace any prior session on the channel: a channel that starts again is a new
           session, so it gets a new id and may legitimately take a different mode.
        3. Register and return it.

        Raises:
            InvalidSessionModeError: mode missing or not one of the two supported modes.
        """
        record = SessionRecord.create(
            channel=channel,
            mode=mode,
            languages=languages,
            participants=participants,
            session_id=session_id
        )

        with self._lock:
            self._sessions[channel] = record

        logger.info(
            "Session %s created on channel '%s' in %s mode.",
            record.session_id, channel, record.mode.value
        )
        return record

    def get_session(self, channel: str) -> Optional[SessionRecord]:
        """Returns the active session for a channel, or None."""
        with self._lock:
            return self._sessions.get(channel)

    def require_session(self, channel: str) -> SessionRecord:
        """
        Returns the active session, raising when the channel has none.

        Used by paths that cannot proceed without a mode - generating a note or a quiz
        under a guessed mode produces artifacts under the wrong contract.
        """
        record = self.get_session(channel)
        if record is None:
            raise SessionNotFoundError(f"No active session on channel '{channel}'.")
        return record

    def mode_for(self, channel: str, default: Optional[SessionMode] = None) -> Optional[SessionMode]:
        """Returns the channel's mode, or `default` when no session is registered."""
        record = self.get_session(channel)
        return record.mode if record else default

    def set_mode(self, channel: str, mode: Any) -> SessionRecord:
        """
        Re-asserts the session's mode, enforcing immutability (REQ-12).

        Setting the mode a session already has is a deliberate no-op rather than an
        error: a client that resends its creation payload (a retry, a reconnect) must not
        be punished for repeating itself. Setting a *different* mode raises, because the
        session already holds prompts, notes, and quizzes generated under the first one.

        Raises:
            SessionNotFoundError: the channel has no session.
            InvalidSessionModeError: the supplied mode is unusable.
            SessionModeImmutableError: the supplied mode differs from the current one.
        """
        record = self.require_session(channel)
        requested = SessionMode.parse(mode)

        if requested is not record.mode:
            raise SessionModeImmutableError(
                f"Session {record.session_id} on channel '{channel}' is fixed in "
                f"{record.mode.value} mode and cannot switch to {requested.value}. "
                "End the session and start a new one to change mode."
            )
        return record

    def end_session(self, channel: str) -> Optional[SessionRecord]:
        """
        Ends the session on a channel and returns the closed record.

        The record is stamped with `ended_at` and removed from the active registry; the
        channel is then free to host a new session under any mode.
        """
        with self._lock:
            record = self._sessions.pop(channel, None)

        if record is None:
            return None

        closed = SessionRecord(
            session_id=record.session_id,
            mode=record.mode,
            channel=record.channel,
            languages=record.languages,
            participants=record.participants,
            created_at=record.created_at,
            ended_at=time.time(),
            schema_version=record.schema_version
        )
        logger.info("Session %s on channel '%s' ended.", closed.session_id, channel)
        return closed

    def reset(self) -> None:
        """Drops every registered session. Used by tests and by a full server restart."""
        with self._lock:
            self._sessions.clear()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns every active session as a serializable dict, for status endpoints."""
        with self._lock:
            return [record.to_dict() for record in self._sessions.values()]

    def event_context(self, channel: str) -> Dict[str, Any]:
        """
        Returns the envelope that RTC events and artifacts stamp themselves with.

        This is how the mode propagates (REQ-12): one function, one shape, so an event
        and the artifact generated from the same turn can never disagree about which
        session or mode they belong to. A channel with no session yields a context whose
        `mode` is None rather than a guessed default - the caller decides what an
        unmoded event means for it.
        """
        record = self.get_session(channel)
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": record.session_id if record else None,
            "mode": record.mode.value if record else None,
            "channel": channel,
        }
