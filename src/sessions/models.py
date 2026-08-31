"""
Summary:
    models.py defines the session vocabulary EchoSphere branches on: the two operating
    modes (REQ-12) and the immutable record created when a session starts.

    `language_learning` and `international_work` replace the earlier `student` /
    `teacher` view roles. Those roles described which UI a person was looking at; they
    said nothing about what the session was for, so prompts, quizzes, and notes had
    nothing to branch on. The mode does, and it is fixed for the life of a session
    because every artifact generated under it carries a mode-shaped contract.

Key Classes:
    - SessionMode: the two supported modes plus their assistance policy.
    - SessionRecord: an immutable, versioned description of one session.
    - InvalidSessionModeError / SessionModeImmutableError: the two REQ-12 failure modes.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Bumped when the persisted shape of a session record changes. Artifacts and RTC events
# stamp themselves with it so a stored session can be read back by a later revision.
SESSION_SCHEMA_VERSION = "1.0"


class InvalidSessionModeError(ValueError):
    """Raised when a mode is missing, malformed, or not one of the two supported modes."""


class SessionModeImmutableError(ValueError):
    """Raised on an attempt to change the mode of a session that already has one."""


class SessionMode(str, Enum):
    """
    The operating mode of a session (REQ-12).

    Subclasses `str` so a mode serializes as its own wire value in JSON payloads and RTC
    events without a conversion step at every boundary.
    """

    LANGUAGE_LEARNING = "language_learning"
    INTERNATIONAL_WORK = "international_work"

    @classmethod
    def parse(cls, value: Any) -> "SessionMode":
        """
        Resolves a wire value to a mode, raising rather than defaulting (REQ-12).

        Algorithm:
        1. Reject anything that is not a non-empty string, including None.
        2. Normalize surrounding whitespace and case.
        3. Look up the enum member, raising InvalidSessionModeError when absent.

        Deliberately strict about the retired `student` / `teacher` values: mapping them
        onto a mode would let a stale client silently open a session under a contract its
        user never chose, and a wrong mode is invisible until the artifacts come out
        wrong. A 400 is legible immediately.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise InvalidSessionModeError(
                f"Session mode is required and must be one of {cls.values()}; got {value!r}."
            )

        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            raise InvalidSessionModeError(
                f"Unsupported session mode {value!r}; expected one of {cls.values()}."
            ) from None

    @classmethod
    def values(cls) -> List[str]:
        """Returns the supported wire values, for error messages and API validation."""
        return [member.value for member in cls]

    @property
    def grades_language(self) -> bool:
        """
        Whether the assistant corrects and grades the speaker's language (REQ-13).

        False for `international_work`: colleagues in a work call are exchanging
        decisions, not practising, and unrequested grading is an interruption.
        """
        return self is SessionMode.LANGUAGE_LEARNING

    @property
    def quizzes_by_default(self) -> bool:
        """Whether eligible turns produce a quiz without being asked to (REQ-13)."""
        return self is SessionMode.LANGUAGE_LEARNING

    @property
    def note_types(self) -> Tuple[str, ...]:
        """
        The note vocabulary this mode may emit (REQ-14).

        The two vocabularies are disjoint on purpose: a note's type alone identifies
        which mode produced it, so a mixed-mode artifact is detectable rather than
        merely wrong.
        """
        if self is SessionMode.LANGUAGE_LEARNING:
            return ("vocabulary", "correction", "grammar", "culture", "example", "goal")
        return ("term", "decision", "action", "risk", "open_question", "glossary")

    @property
    def label(self) -> str:
        """Human-readable name for logs and UI copy."""
        return "Language Learning" if self.grades_language else "International Work"


@dataclass(frozen=True)
class SessionRecord:
    """
    One session: its mode, the channel it runs on, and who/what is in it (REQ-12).

    Frozen because the mode must not be reassigned in place - `SessionService` replaces
    the whole record when mutable fields (participants, languages) change, which keeps
    the immutability rule enforceable by the type rather than by convention.
    """

    session_id: str
    mode: SessionMode
    channel: str
    languages: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    schema_version: str = SESSION_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        channel: str,
        mode: Any,
        languages: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> "SessionRecord":
        """
        Builds a validated record, generating a stable session id when none is supplied.

        Algorithm:
        1. Parse and validate the mode (REQ-12).
        2. Assign a session id - caller-supplied ids are honored so a client can
           correlate a session it already named.
        3. Freeze the record.
        """
        return cls(
            session_id=session_id or f"sess-{uuid.uuid4().hex[:12]}",
            mode=SessionMode.parse(mode),
            channel=channel,
            languages=list(languages or []),
            participants=list(participants or [])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the record for API responses, RTC events, and artifacts."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "mode": self.mode.value,
            "channel": self.channel,
            "languages": list(self.languages),
            "participants": list(self.participants),
            "created_at": self.created_at,
            "ended_at": self.ended_at,
        }
