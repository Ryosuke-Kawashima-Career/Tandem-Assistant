"""
Summary:
    generator.py turns one finalized turn into the artifacts its session mode calls for:
    quizzes in `language_learning` (REQ-13) and typed notes in either mode (REQ-14).

    It is the only place that decides what a turn is worth keeping, so the mode rules
    live here rather than being re-derived at each call site: a work session is not
    quizzed unless it asks, a note typed for the other mode is dropped rather than
    stored, and anything the model only inferred is held at `needs_confirmation`.

Key Classes:
    - ArtifactGenerator: builds, stores, and announces quizzes and notes.
"""

import logging
from typing import Any, Dict, List, Optional

from src.artifacts.models import (
    ARTIFACT_SCHEMA_VERSION,
    NoteItem,
    NoteStatus,
    QuizItem,
    stable_entity_id,
)
from src.artifacts.repository import InMemoryArtifactRepository
from src.sessions.models import SessionRecord

logger = logging.getLogger("echosphere.artifacts.generator")

# Below this confidence a note is presented as unconfirmed rather than as fact (REQ-14).
# 0.6 is deliberately permissive: the cost of an unnecessary confirmation prompt is one
# click, while the cost of a wrongly asserted decision is a team acting on something
# nobody agreed to.
NEEDS_CONFIRMATION_THRESHOLD = 0.6

# Note types that are meaningless without a named owner. An action item assigned to
# nobody is precisely the ambiguity REQ-14 exists to surface, so it is never captured
# outright even when the model is confident about the wording.
OWNER_REQUIRED_NOTE_TYPES = ("action",)


class ArtifactGenerator:
    """
    Builds quizzes and notes from finalized turns and announces them over RTC.

    The data stream is optional so the generator can be exercised - and used - without a
    live RTC channel; a session running offline still accumulates its artifacts.
    """

    def __init__(
        self,
        repository: Optional[InMemoryArtifactRepository] = None,
        data_stream: Optional[Any] = None
    ):
        """
        Initialize the generator over a repository and an optional RTC data stream.

        Algorithm:
        1. Bind (or create) the artifact repository.
        2. Bind the data stream used to publish `quiz.created` / `note.*` events.
        """
        self.repository = repository or InMemoryArtifactRepository()
        self.data_stream = data_stream

    # -- Event helpers ---------------------------------------------------------------

    @staticmethod
    def event_id_for(entity_id: str, revision: int = 1) -> str:
        """
        Derives the idempotent id for an event about one revision of one entity.

        A receiver that has already applied this id can discard the duplicate without
        asking the server anything, which is what makes re-delivery of an RTC data event
        safe (spec section 5).
        """
        return stable_entity_id("evt", entity_id, revision)

    def _emit(self, event_type: str, session: SessionRecord, key: str,
              entity: Dict[str, Any], revision: int) -> bool:
        """
        Publishes one artifact event wrapped in the session envelope.

        Every artifact event carries the same envelope - schema version, session id,
        mode, idempotent event id - so a client can route and deduplicate by shape alone
        rather than knowing each event type.
        """
        if self.data_stream is None:
            return False

        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event_id": self.event_id_for(entity.get("id", ""), revision),
            "session_id": session.session_id,
            "mode": session.mode.value,
            key: entity,
        }
        return bool(self.data_stream.send_artifact_event(event_type, payload))

    # -- Quizzes (REQ-13) ------------------------------------------------------------

    def build_quiz(
        self,
        turn_result: Dict[str, Any],
        session: SessionRecord,
        source_turn_ids: List[str],
        requested: bool = False
    ) -> Optional[QuizItem]:
        """
        Builds the quiz a turn earns, or None when it earns none (REQ-13).

        Algorithm:
        1. Apply the mode gate: work sessions produce no quiz unless one was requested.
        2. Read the quiz block; an inactive or empty block yields nothing.
        3. Type the answer - multiple choice when options were offered, else free text.
        4. Derive a stable id from the session, source turns, and prompt.

        Step 1 is REQ-13's mode rule: grading colleagues who are not practising turns an
        assistant into an interruption.
        """
        if not session.mode.quizzes_by_default and not requested:
            return None

        block = turn_result.get("quiz") or {}
        prompt = (block.get("question") or "").strip()
        if not block.get("active", bool(prompt)) or not prompt:
            return None

        options = [str(option) for option in (block.get("options") or []) if str(option).strip()]
        if options:
            answer_type = "multiple_choice"
            index = block.get("correct_index", 0)
            expected = options[index] if 0 <= index < len(options) else options[0]
        else:
            answer_type = "free_text"
            expected = str(block.get("expected_answer", "")).strip()

        return QuizItem(
            id=stable_entity_id("quiz", session.session_id, tuple(source_turn_ids), prompt),
            prompt=prompt,
            answer_type=answer_type,
            expected_answer=expected,
            explanation=str(block.get("explanation", "")),
            options=options,
            source_turn_ids=list(source_turn_ids),
            difficulty=str(block.get("difficulty", "medium")),
            target_language=str(turn_result.get("spoken_language")
                                or (session.languages[0] if session.languages else "en")),
            session_id=session.session_id,
            mode=session.mode.value
        )

    def generate_quiz(
        self,
        turn_result: Dict[str, Any],
        session: SessionRecord,
        source_turn_ids: List[str],
        requested: bool = False
    ) -> Optional[QuizItem]:
        """
        Builds, stores, and announces a quiz, announcing it at most once (REQ-13).

        Returns the stored quiz - which may be one produced by an earlier run of the same
        turn - or None when the turn produced no quiz.
        """
        quiz = self.build_quiz(turn_result, session, source_turn_ids, requested=requested)
        if quiz is None:
            return None

        stored, created = self.repository.add_quiz(quiz)
        if created:
            self._emit("quiz.created", session, "quiz", stored.to_dict(), revision=1)
            logger.info(
                "Quiz %s created for session %s from turns %s.",
                stored.id, session.session_id, source_turn_ids
            )
        return stored

    # -- Notes (REQ-14) --------------------------------------------------------------

    def build_notes(
        self,
        turn_result: Dict[str, Any],
        session: SessionRecord,
        source_turn_ids: List[str]
    ) -> List[NoteItem]:
        """
        Builds the typed notes a turn produces (REQ-14).

        Algorithm:
        1. Read the model's `notes` block; fall back to the older scaffolding blocks when
           it is absent, so the mock engine and pre-REQ-14 prompts still produce notes.
        2. Drop any note whose type does not belong to this session's mode vocabulary.
        3. Resolve each note's status from its confidence and completeness.
        4. Derive a stable id from the session, source turns, and note type.

        Step 2 is what keeps one session's artifact from holding items generated under
        the other mode's contract - the two vocabularies are disjoint, so a stray
        `decision` in a learning session is a defect, not a note.
        """
        raw_notes = turn_result.get("notes")
        if not isinstance(raw_notes, list) or not raw_notes:
            raw_notes = self._derive_notes_from_scaffolding(turn_result, session)

        notes: List[NoteItem] = []
        for raw in raw_notes:
            if not isinstance(raw, dict):
                continue

            note_type = str(raw.get("type", "")).strip().lower()
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            if note_type not in session.mode.note_types:
                logger.debug(
                    "Dropped note of type %r: not in the %s vocabulary.",
                    note_type, session.mode.value
                )
                continue

            confidence = self._coerce_confidence(raw.get("confidence"))
            owner = raw.get("owner") or None
            due_at = raw.get("due_at") or None

            notes.append(NoteItem(
                id=stable_entity_id("note", session.session_id, tuple(source_turn_ids), note_type),
                type=note_type,
                text=text,
                source_turn_ids=list(source_turn_ids),
                confidence=confidence,
                status=self._resolve_status(note_type, confidence, owner),
                session_id=session.session_id,
                mode=session.mode.value,
                owner=owner,
                due_at=due_at
            ))

        return notes

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Reads a model-supplied confidence, defaulting to fully confident."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _resolve_status(note_type: str, confidence: float, owner: Optional[str]) -> str:
        """
        Decides whether a note is captured or held for confirmation (REQ-14).

        Two independent reasons to hold: the model was unsure of what was said, or the
        commitment is structurally incomplete - an action nobody owns. Either way the
        note is surfaced rather than dropped; the ambiguity is the point.
        """
        if confidence < NEEDS_CONFIRMATION_THRESHOLD:
            return NoteStatus.NEEDS_CONFIRMATION
        if note_type in OWNER_REQUIRED_NOTE_TYPES and not owner:
            return NoteStatus.NEEDS_CONFIRMATION
        return NoteStatus.CAPTURED

    @staticmethod
    def _derive_notes_from_scaffolding(
        turn_result: Dict[str, Any],
        session: SessionRecord
    ) -> List[Dict[str, Any]]:
        """
        Falls back to the pre-REQ-14 scaffolding blocks when no `notes` block was emitted.

        The idiom card already carries exactly what a vocabulary and a culture note hold,
        and the mock engine emits it, so a session running offline or against an older
        prompt still accumulates notes instead of silently producing none. Work sessions
        get nothing here: their vocabulary has no counterpart among the learning-shaped
        scaffolding blocks, and inventing one would fabricate commitments.
        """
        if not session.mode.grades_language:
            return []

        card = turn_result.get("idiom_card") or {}
        if not card.get("detected") or not card.get("phrase"):
            return []

        derived = [{
            "type": "vocabulary",
            "text": f"{card.get('phrase')} - {card.get('meaning', '')}".strip(" -"),
            "confidence": 0.9,
        }]
        if card.get("cultural_note"):
            derived.append({
                "type": "culture",
                "text": str(card["cultural_note"]),
                "confidence": 0.9,
            })
        return derived

    def generate_notes(
        self,
        turn_result: Dict[str, Any],
        session: SessionRecord,
        source_turn_ids: List[str]
    ) -> List[NoteItem]:
        """
        Builds, upserts, and announces the notes for one turn (REQ-14).

        Only notes that actually changed are announced, so a turn regenerated by the
        asynchronous scaffolding path does not make every client redraw its notes panel.
        """
        stored_notes: List[NoteItem] = []

        for note in self.build_notes(turn_result, session, source_turn_ids):
            stored, changed = self.repository.upsert_note(note)
            stored_notes.append(stored)
            if changed:
                self._emit(
                    "note.upserted", session, "note", stored.to_dict(), stored.revision
                )

        return stored_notes

    def delete_note(self, note_id: str, session: SessionRecord) -> Optional[NoteItem]:
        """
        Deletes a note and announces it, returning the tombstone (REQ-14).

        Returns None when no such note exists, so a stale client deleting twice gets a
        miss rather than a second `note.deleted` event.
        """
        deleted = self.repository.delete_note(note_id)
        if deleted is None:
            return None

        self._emit("note.deleted", session, "note", deleted.to_dict(), deleted.revision)
        logger.info("Note %s deleted from session %s.", note_id, session.session_id)
        return deleted
