"""
Summary:
    repository.py stores what a session produced - transcript turns, notes, and quizzes -
    with the idempotent upsert semantics REQ-13 and REQ-14 require, and assembles them
    into the versioned `SessionArtifact` envelope REQ-15 defines.

    The semantics live in the base class and the storage medium in the subclass on
    purpose: dedupe on source+type, pinning a user's edit against regeneration, and the
    delete tombstone are rules about what the data means, not about where it is written,
    and they must behave identically in memory and on disk.

Key Classes:
    - InMemoryArtifactRepository: the rules, over an in-process store.
    - LocalArtifactRepository: the same rules, durable on the local filesystem (REQ-15's
      MVP source of truth), with configurable retention (REQ-16).
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.artifacts.models import (
    ARTIFACT_SCHEMA_VERSION,
    NoteItem,
    NoteStatus,
    QuizItem,
    SessionArtifact,
    TranscriptTurn,
)

logger = logging.getLogger("echosphere.artifacts.repository")

# Where the local repository writes its session files, and how long it keeps them.
# Retention is opt-in: an unset window never deletes a user's data, because silently
# discarding a record of a real conversation is worse than an oversized directory.
DEFAULT_DATA_DIR = os.getenv("ECHOSPHERE_DATA_DIR", "data/artifacts")
DEFAULT_RETENTION_DAYS = os.getenv("ECHOSPHERE_ARTIFACT_RETENTION_DAYS", "")


def configured_retention_days() -> Optional[int]:
    """
    Reads the retention window from the environment, or None when it is unset.

    An unparseable value is treated as unset and logged rather than raising: a typo in a
    retention setting must not be the thing that stops the server from starting.
    """
    raw = (DEFAULT_RETENTION_DAYS or "").strip()
    if not raw:
        return None
    try:
        days = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring ECHOSPHERE_ARTIFACT_RETENTION_DAYS=%r: not an integer. "
            "Artifacts will be kept indefinitely.", raw
        )
        return None
    return days if days > 0 else None


class InMemoryArtifactRepository:
    """
    Stores a session's turns, notes, and quizzes by their stable ids.

    Locked because artifact generation runs on the background scaffolding executor while
    a request thread may be reading the same session for an API response. The lock is
    reentrant so a mutation can flush a consistent snapshot without releasing it first.
    """

    def __init__(self):
        """
        Initialize empty stores.

        Algorithm:
        1. Create the id -> entity maps for turns, notes, and quizzes.
        2. Create the session-metadata map and the per-session revision counter.
        3. Create the reentrant lock guarding them.
        """
        self._turns: Dict[str, TranscriptTurn] = {}
        self._notes: Dict[str, NoteItem] = {}
        self._quizzes: Dict[str, QuizItem] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._revisions: Dict[str, int] = {}
        self._lock = threading.RLock()

    # -- Storage hook ----------------------------------------------------------------

    def _flush(self, session_id: Optional[str]) -> None:
        """
        Called after every mutation. A no-op in memory; the local store writes here.

        Taking the hook at this level is what keeps the durable subclass from having to
        restate any of the rules above it.
        """

    def _bump_revision(self, session_id: Optional[str]) -> None:
        """Advances a session's revision so a consumer can detect a changed artifact."""
        if session_id:
            self._revisions[session_id] = self._revisions.get(session_id, 1) + 1

    # -- Sessions --------------------------------------------------------------------

    def save_session(self, session: Any) -> Dict[str, Any]:
        """
        Records the session metadata an artifact needs after the session has ended.

        `SessionService` drops a session from its active registry when the channel
        closes, but the artifact outlives the conversation - so mode, participants, and
        start time are captured here rather than looked up later from something that no
        longer exists.
        """
        payload = session.to_dict() if hasattr(session, "to_dict") else dict(session)
        session_id = payload.get("session_id")

        with self._lock:
            existing = self._sessions.get(session_id, {})
            merged = {**existing, **payload}
            self._sessions[session_id] = merged
            self._revisions.setdefault(session_id, 1)
            self._flush(session_id)
        return merged

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Returns the stored session metadata, or None."""
        with self._lock:
            meta = self._sessions.get(session_id)
            return dict(meta) if meta else None

    def find_session_by_channel(self, channel: str) -> Optional[Dict[str, Any]]:
        """
        Returns the most recently created stored session for a channel, or None.

        A channel is reused across sessions, so "the artifact for this channel" can only
        mean the latest one - and the caller that wants an older session addresses it by
        its session id instead.
        """
        with self._lock:
            candidates = [
                dict(meta) for meta in self._sessions.values()
                if meta.get("channel") == channel
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda meta: meta.get("created_at") or 0)

    def list_session_ids(self) -> List[str]:
        """Returns every session id the store holds anything for."""
        with self._lock:
            ids = set(self._sessions)
            ids.update(turn.session_id for turn in self._turns.values())
            ids.update(note.session_id for note in self._notes.values())
            ids.update(quiz.session_id for quiz in self._quizzes.values())
        return sorted(ids)

    # -- Transcript turns ------------------------------------------------------------

    def add_turn(self, turn: TranscriptTurn) -> Tuple[TranscriptTurn, bool]:
        """
        Stores one finalized turn, returning it with whether this call created it.

        A repeat is not an error: turn ids are derived from their content, so a second
        arrival is the same utterance reaching the store twice, not a new one.
        """
        with self._lock:
            existing = self._turns.get(turn.id)
            if existing is not None:
                return existing, False

            self._turns[turn.id] = turn
            self._bump_revision(turn.session_id)
            self._flush(turn.session_id)
            return turn, True

    def list_turns(self, session_id: str) -> List[TranscriptTurn]:
        """Returns a session's transcript, oldest first."""
        with self._lock:
            items = [turn for turn in self._turns.values() if turn.session_id == session_id]
        return sorted(items, key=lambda turn: turn.created_at)

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
            self._bump_revision(quiz.session_id)
            self._flush(quiz.session_id)
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
        2. Stored note was edited or deleted by a user => leave it alone.
        3. Regenerated content identical to stored => report no change.
        4. Otherwise store the next revision and report a change.

        Step 2 is REQ-14's "user edits survive revisions", and its delete half: once a
        person has rewritten or removed a note, a later model pass over the same turn must
        not undo their decision.
        """
        with self._lock:
            existing = self._notes.get(note.id)

            if existing is None:
                self._notes[note.id] = note
                self._bump_revision(note.session_id)
                self._flush(note.session_id)
                return note, True

            if existing.status in (NoteStatus.EDITED, NoteStatus.DELETED):
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
            self._bump_revision(note.session_id)
            self._flush(note.session_id)
            return revised, True

    def edit_note(
        self,
        note_id: str,
        text: str,
        owner: Optional[str] = None,
        due_at: Optional[str] = None,
        actor: str = "user"
    ) -> Optional[NoteItem]:
        """
        Applies a human edit, which pins the note against later regeneration (REQ-15).

        The previous wording, who replaced it, and when are appended to `edit_history`:
        REQ-16 asks for edit provenance, and a note that quietly changed is one nobody
        can audit back to what the speaker actually said.

        Returns None when the note does not exist, so a stale client editing a purged
        note gets an explicit miss rather than resurrecting it.
        """
        with self._lock:
            existing = self._notes.get(note_id)
            if existing is None:
                return None

            history = list(existing.edit_history) + [{
                "at": time.time(),
                "by": actor,
                "previous_text": existing.text,
                "previous_status": existing.status,
            }]

            edited = existing.with_revision(
                text=text,
                owner=owner if owner is not None else existing.owner,
                due_at=due_at if due_at is not None else existing.due_at,
                status=NoteStatus.EDITED,
                updated_by=actor,
                edit_history=history
            )
            self._notes[note_id] = edited
            self._bump_revision(existing.session_id)
            self._flush(existing.session_id)
            return edited

    def get_note(self, note_id: str) -> Optional[NoteItem]:
        """Returns one note by id, including a deleted one, or None."""
        with self._lock:
            return self._notes.get(note_id)

    def delete_note(self, note_id: str, actor: str = "user") -> Optional[NoteItem]:
        """
        Marks a note deleted and returns it.

        A soft delete: the tombstone is what a late-arriving regeneration of the same
        turn collides with, so the note stays deleted instead of quietly reappearing.
        `list_notes` excludes it; `purge_session` removes it outright.
        """
        with self._lock:
            existing = self._notes.get(note_id)
            if existing is None:
                return None

            deleted = existing.with_revision(status=NoteStatus.DELETED, updated_by=actor)
            self._notes[note_id] = deleted
            self._bump_revision(existing.session_id)
            self._flush(existing.session_id)
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

    # -- Artifact assembly (REQ-15) ---------------------------------------------------

    def build_artifact(self, session_id: str) -> Optional[SessionArtifact]:
        """
        Assembles everything a session produced into one versioned envelope.

        Returns None when the store holds nothing for the session, so a caller can tell
        "this session produced nothing" apart from "this session does not exist" without
        inspecting an empty shell.
        """
        with self._lock:
            meta = dict(self._sessions.get(session_id, {}))
            revision = self._revisions.get(session_id, 1)

        turns = self.list_turns(session_id)
        notes = self.list_notes(session_id)
        quizzes = self.list_quizzes(session_id)

        if not meta and not turns and not notes and not quizzes:
            return None

        timestamps = [item.created_at for item in (*turns, *notes, *quizzes)]
        started_at = meta.get("created_at") or (min(timestamps) if timestamps else time.time())
        updated_at = max(
            [*(note.updated_at for note in notes), *timestamps, started_at]
        )

        return SessionArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            session_id=session_id,
            mode=meta.get("mode") or (notes[0].mode if notes else "language_learning"),
            started_at=started_at,
            ended_at=meta.get("ended_at"),
            participants=list(meta.get("participants") or []),
            languages=list(meta.get("languages") or []),
            transcript_turns=turns,
            notes=notes,
            quizzes=quizzes,
            summary=meta.get("summary", ""),
            revision=revision,
            created_at=started_at,
            updated_at=updated_at
        )

    # -- Deletion and retention (REQ-16) ----------------------------------------------

    def purge_session(self, session_id: str) -> int:
        """
        Removes every trace of a session and returns how many entities were dropped.

        A hard delete, unlike `delete_note`: REQ-16's acceptance is that deletion makes
        the transcript, notes, and quizzes unavailable, which a tombstone does not
        satisfy for the transcript.
        """
        with self._lock:
            turn_ids = [k for k, v in self._turns.items() if v.session_id == session_id]
            note_ids = [k for k, v in self._notes.items() if v.session_id == session_id]
            quiz_ids = [k for k, v in self._quizzes.items() if v.session_id == session_id]

            for key in turn_ids:
                del self._turns[key]
            for key in note_ids:
                del self._notes[key]
            for key in quiz_ids:
                del self._quizzes[key]
            self._sessions.pop(session_id, None)
            self._revisions.pop(session_id, None)

            self._purge_storage(session_id)

        return len(turn_ids) + len(note_ids) + len(quiz_ids)

    def _purge_storage(self, session_id: str) -> None:
        """Removes a session from durable storage. A no-op in memory."""

    def purge_expired(self, now: Optional[float] = None) -> int:
        """
        Purges sessions whose retention window has passed, returning how many were purged.

        Counts sessions, not entities: the operator's question is how many conversations
        aged out, and one long session dropping 200 notes is still one conversation.
        """
        window_days = getattr(self, "retention_days", None)
        if not window_days:
            return 0

        cutoff = (now or time.time()) - (window_days * 86400)
        purged = 0

        for session_id in self.list_session_ids():
            artifact = self.build_artifact(session_id)
            if artifact is None:
                continue
            last_touched = max(artifact.updated_at, artifact.started_at)
            if last_touched < cutoff:
                self.purge_session(session_id)
                purged += 1
                logger.info(
                    "Purged session %s: last activity was beyond the %s-day retention window.",
                    session_id, window_days
                )

        return purged

    def reset(self) -> None:
        """Drops every stored artifact from memory. Used by tests and a full restart."""
        with self._lock:
            self._turns.clear()
            self._notes.clear()
            self._quizzes.clear()
            self._sessions.clear()
            self._revisions.clear()


class LocalArtifactRepository(InMemoryArtifactRepository):
    """
    The same rules, written through to the local filesystem (REQ-15).

    One JSON file per session, named by session id: a session is the unit that is read,
    exported, and deleted whole, so a per-session file makes deletion a file removal
    rather than a rewrite of a shared database, and a corrupt file costs one conversation
    instead of all of them.
    """

    def __init__(
        self,
        data_dir: Any = None,
        retention_days: Optional[int] = None,
        load: bool = True
    ):
        """
        Initialize the store over a data directory.

        Algorithm:
        1. Resolve and create the data directory.
        2. Initialize the in-memory rules layer.
        3. Load every session file already on disk.
        """
        super().__init__()
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.retention_days = retention_days
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._loading = False

        if load:
            self.load()

    def _session_path(self, session_id: str) -> Path:
        """Returns the file one session is stored in."""
        safe = "".join(char for char in session_id if char.isalnum() or char in "-_")
        return self.data_dir / f"{safe}.json"

    def load(self) -> int:
        """
        Loads every session file into memory, returning how many sessions were read.

        A file that cannot be parsed is skipped with a warning rather than raising: a
        crash mid-write leaves one truncated file, and refusing to start over it would
        turn a single lost session into a lost store.
        """
        loaded = 0
        self._loading = True
        try:
            for path in sorted(self.data_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping unreadable artifact file %s: %s", path, exc)
                    continue

                session_id = payload.get("session_id") or path.stem
                with self._lock:
                    if payload.get("session"):
                        self._sessions[session_id] = payload["session"]
                    self._revisions[session_id] = int(payload.get("revision", 1))
                    for raw in payload.get("transcript_turns", []):
                        turn = TranscriptTurn.from_dict(raw)
                        self._turns[turn.id] = turn
                    for raw in payload.get("notes", []):
                        note = NoteItem.from_dict(raw)
                        self._notes[note.id] = note
                    for raw in payload.get("quizzes", []):
                        quiz = QuizItem.from_dict(raw)
                        self._quizzes[quiz.id] = quiz
                loaded += 1
        finally:
            self._loading = False

        if loaded:
            logger.info("Loaded %d stored session artifact(s) from %s.", loaded, self.data_dir)
        return loaded

    def _flush(self, session_id: Optional[str]) -> None:
        """
        Writes one session's whole file.

        Whole-file rather than append: the file is small, and a partial append is exactly
        the corruption `load` then has to skip. Written to a temporary file and moved into
        place so a crash mid-write leaves the previous good file rather than a truncated
        one. Never raises - losing durability must not take down the conversation that
        produced the data.
        """
        if session_id is None or self._loading:
            return

        with self._lock:
            payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "session_id": session_id,
                "session": self._sessions.get(session_id, {}),
                "revision": self._revisions.get(session_id, 1),
                "transcript_turns": [
                    turn.to_dict() for turn in self._turns.values()
                    if turn.session_id == session_id
                ],
                "notes": [
                    note.to_dict() for note in self._notes.values()
                    if note.session_id == session_id
                ],
                "quizzes": [
                    quiz.to_dict() for quiz in self._quizzes.values()
                    if quiz.session_id == session_id
                ],
            }

        path = self._session_path(session_id)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)
        except OSError as exc:
            logger.warning("Could not persist session %s to %s: %s", session_id, path, exc)

    def _purge_storage(self, session_id: str) -> None:
        """Removes a session's file from disk (REQ-16 complete deletion)."""
        path = self._session_path(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete artifact file %s: %s", path, exc)
