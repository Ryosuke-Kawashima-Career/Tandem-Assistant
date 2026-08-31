"""
Summary:
    repository.py stores the quizzes and notes a session produces, with the idempotent
    upsert semantics REQ-13 and REQ-14 require.

    Deliberately in-memory for now: Phase 9 owns *producing* correct, deduplicated,
    edit-preserving artifacts, and Phase 10 owns persisting them. Keeping the semantics
    here - and the storage medium behind this one class - means Phase 10 replaces the
    backing store without revisiting the rules.

Key Classes:
    - InMemoryArtifactRepository: per-session quiz and note storage with dedupe,
      user-edit preservation, and soft deletion.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

from src.artifacts.models import NoteItem, NoteStatus, QuizItem

logger = logging.getLogger("echosphere.artifacts.repository")


class InMemoryArtifactRepository:
    """
    Stores notes and quizzes keyed by their stable ids.

    Locked because artifact generation runs on the background scaffolding executor while
    a request thread may be reading the same session's notes for an API response.
    """

    def __init__(self):
        """
        Initialize empty note and quiz stores.

        Algorithm:
        1. Create the id -> entity maps.
        2. Create the lock guarding them against the scaffolding executor.
        """
        self._notes: Dict[str, NoteItem] = {}
        self._quizzes: Dict[str, QuizItem] = {}
        self._lock = threading.Lock()

    # -- Quizzes (REQ-13) ------------------------------------------------------------

    def add_quiz(self, quiz: QuizItem) -> Tuple[QuizItem, bool]:
        """
        Stores a quiz, returning it with whether this call was the one that created it.

        A repeat of an id already held is not an error and not an update: quiz ids are
        derived from their source turn, so a second arrival is the asynchronous path
        having run twice for one utterance.
        """
        with self._lock:
            existing = self._quizzes.get(quiz.id)
            if existing is not None:
                return existing, False
            self._quizzes[quiz.id] = quiz
            return quiz, True

    def get_quiz(self, quiz_id: str) -> Optional[QuizItem]:
        """Returns one quiz by id, or None."""
        with self._lock:
            return self._quizzes.get(quiz_id)

    def list_quizzes(self, session_id: str) -> List[QuizItem]:
        """Returns every quiz belonging to a session, oldest first."""
        with self._lock:
            items = [q for q in self._quizzes.values() if q.session_id == session_id]
        return sorted(items, key=lambda q: q.created_at)

    # -- Notes (REQ-14) --------------------------------------------------------------

    def upsert_note(self, note: NoteItem) -> Tuple[NoteItem, bool]:
        """
        Inserts or revises a note, returning it with whether anything actually changed.

        Algorithm:
        1. No stored note with this id => insert it and report a change.
        2. Stored note was edited by a user => keep the user's text and report no change.
        3. Regenerated content identical to stored => report no change.
        4. Otherwise store the next revision and report a change.

        Step 2 is REQ-14's "user edits survive revisions": once a person has rewritten a
        note, the model's later passes over the same turn must not silently undo their
        wording. Step 3 is what keeps an unchanged turn from re-announcing itself on
        every regeneration.
        """
        with self._lock:
            existing = self._notes.get(note.id)

            if existing is None:
                self._notes[note.id] = note
                return note, True

            if existing.status == NoteStatus.EDITED:
                return existing, False

            if existing.same_content_as(note):
                return existing, False

            revised = existing.with_revision(
                text=note.text,
                owner=note.owner,
                due_at=note.due_at,
                confidence=note.confidence,
                status=note.status
            )
            self._notes[note.id] = revised
            return revised, True

    def edit_note(self, note_id: str, text: str, owner: Optional[str] = None,
                  due_at: Optional[str] = None) -> Optional[NoteItem]:
        """
        Applies a human edit, which pins the note's text against later regeneration.

        Returns None when the note does not exist, so a stale client editing a deleted
        note gets an explicit miss rather than resurrecting it.
        """
        with self._lock:
            existing = self._notes.get(note_id)
            if existing is None:
                return None

            edited = existing.with_revision(
                text=text,
                owner=owner if owner is not None else existing.owner,
                due_at=due_at if due_at is not None else existing.due_at,
                status=NoteStatus.EDITED,
                created_by="user"
            )
            self._notes[note_id] = edited
            return edited

    def get_note(self, note_id: str) -> Optional[NoteItem]:
        """Returns one note by id, including a deleted one, or None."""
        with self._lock:
            return self._notes.get(note_id)

    def delete_note(self, note_id: str) -> Optional[NoteItem]:
        """
        Marks a note deleted and returns it.

        A soft delete: the tombstone is what a late-arriving regeneration of the same
        turn collides with, so the note stays deleted instead of quietly reappearing.
        `list_notes` excludes it.
        """
        with self._lock:
            existing = self._notes.get(note_id)
            if existing is None:
                return None

            deleted = existing.with_revision(status=NoteStatus.DELETED)
            self._notes[note_id] = deleted
            return deleted

    def list_notes(self, session_id: str, include_deleted: bool = False) -> List[NoteItem]:
        """Returns a session's live notes, oldest first."""
        with self._lock:
            items = [
                note for note in self._notes.values()
                if note.session_id == session_id
                and (include_deleted or note.status != NoteStatus.DELETED)
            ]
        return sorted(items, key=lambda note: note.created_at)

    def purge_session(self, session_id: str) -> int:
        """
        Removes every artifact belonging to a session and returns how many were dropped.

        Used when a session's data is discarded outright; Phase 10.4 builds the retention
        and access rules on top of this.
        """
        with self._lock:
            note_ids = [k for k, v in self._notes.items() if v.session_id == session_id]
            quiz_ids = [k for k, v in self._quizzes.items() if v.session_id == session_id]
            for key in note_ids:
                del self._notes[key]
            for key in quiz_ids:
                del self._quizzes[key]
        return len(note_ids) + len(quiz_ids)

    def reset(self) -> None:
        """Drops every stored artifact. Used by tests and by a full server restart."""
        with self._lock:
            self._notes.clear()
            self._quizzes.clear()
