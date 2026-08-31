"""
Summary:
    models.py defines the two artifacts a session produces from its finalized turns:
    quizzes (REQ-13) and notes (REQ-14), plus the stable-id helper both rely on.

    Every item is source-linked (`source_turn_ids`) and version-stamped so a stored
    session remains readable by a later revision, and every id is derived from the turn
    that produced it rather than generated randomly - the scaffolding path is
    asynchronous and may run twice for the same utterance, so a random id would put the
    same learning point on the learner's screen twice.

Key Classes:
    - QuizItem: one knowledge check, linked to the turn that motivated it.
    - NoteItem: one captured point, typed from its session mode's vocabulary.
    - NoteStatus: the lifecycle states a note moves through.
"""

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

# Bumped when the persisted shape of a quiz or a note changes.
ARTIFACT_SCHEMA_VERSION = "1.0"


class NoteStatus:
    """
    Lifecycle states of a note (REQ-14).

    A plain constant holder rather than an Enum: these values travel in RTC event
    payloads and stored artifacts as bare strings, and the comparison sites read more
    directly against the string than against `.value` everywhere.
    """

    CAPTURED = "captured"
    NEEDS_CONFIRMATION = "needs_confirmation"
    EDITED = "edited"
    DELETED = "deleted"


def stable_entity_id(prefix: str, *parts: Any) -> str:
    """
    Derives a deterministic, collision-resistant id from the parts that define an entity.

    The same session, source turns, and content therefore always address the same
    entity - which is what makes the asynchronous generation path safe to retry, and
    what lets a client discard a duplicate without server coordination.
    """
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8"))
    return f"{prefix}-{digest.hexdigest()[:16]}"


@dataclass(frozen=True)
class QuizItem:
    """One source-linked knowledge check generated from a finalized turn (REQ-13)."""

    id: str
    prompt: str
    answer_type: str
    expected_answer: str
    explanation: str
    source_turn_ids: List[str]
    difficulty: str
    target_language: str
    session_id: str
    mode: str
    options: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the quiz for RTC events, API responses, and artifacts."""
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "prompt": self.prompt,
            "answer_type": self.answer_type,
            "expected_answer": self.expected_answer,
            "explanation": self.explanation,
            "options": list(self.options),
            "source_turn_ids": list(self.source_turn_ids),
            "difficulty": self.difficulty,
            "target_language": self.target_language,
            "session_id": self.session_id,
            "mode": self.mode,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class NoteItem:
    """
    One source-linked note captured from a finalized turn (REQ-14).

    Frozen, with `revision` advanced by producing a replacement: a note's history is what
    distinguishes a genuine revision (announce it) from an identical regeneration (stay
    quiet), and in-place mutation would erase that distinction.
    """

    id: str
    type: str
    text: str
    source_turn_ids: List[str]
    confidence: float
    status: str
    session_id: str
    mode: str
    owner: Optional[str] = None
    due_at: Optional[str] = None
    created_by: str = "assistant"
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def with_revision(self, **changes: Any) -> "NoteItem":
        """Returns the next revision of this note with the given fields changed."""
        return replace(
            self,
            revision=self.revision + 1,
            updated_at=time.time(),
            **changes
        )

    def same_content_as(self, other: "NoteItem") -> bool:
        """
        Whether a regenerated note says the same thing as the stored one.

        Compares only the fields a reader would notice; `confidence` and timestamps drift
        between model passes without changing what the note asserts, and re-announcing on
        that drift would make the notes panel flicker on every turn.
        """
        return (
            self.text == other.text
            and self.type == other.type
            and self.owner == other.owner
            and self.due_at == other.due_at
            and self.status == other.status
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the note for RTC events, API responses, and artifacts."""
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "source_turn_ids": list(self.source_turn_ids),
            "confidence": self.confidence,
            "status": self.status,
            "owner": self.owner,
            "due_at": self.due_at,
            "created_by": self.created_by,
            "revision": self.revision,
            "session_id": self.session_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
