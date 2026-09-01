"""
Summary:
    export.py renders a stored `SessionArtifact` as Markdown (REQ-15).

    Markdown is the export format the repository owes on its own: it is readable without
    EchoSphere, pastes into whatever the team already uses, and needs no account
    anywhere. The Notion adapter in `adapters.py` is layered on top of this and is
    optional by design.

Key Functions:
    - render_markdown: the whole session as one self-contained document.
"""

from typing import Dict, List

from src.artifacts.models import NoteItem, NoteStatus, SessionArtifact

# Headings per note type, per mode. The two modes produce different documents from the
# same machinery: a lesson summary and a set of meeting minutes are not the same artifact
# with different words, and grouping by type is what makes each read like its own kind.
LEARNING_SECTIONS = [
    ("vocabulary", "Vocabulary"),
    ("correction", "Corrections"),
    ("grammar", "Grammar"),
    ("culture", "Cultural Notes"),
    ("example", "Examples"),
    ("goal", "Goals"),
]

WORK_SECTIONS = [
    ("decision", "Decisions"),
    ("action", "Action Items"),
    ("risk", "Risks"),
    ("open_question", "Open Questions"),
    ("term", "Terms"),
    ("glossary", "Glossary"),
]


def _format_timestamp(value) -> str:
    """Formats a POSIX timestamp as a readable UTC date, or an em dash when absent."""
    if not value:
        return "—"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _format_note(note: NoteItem) -> str:
    """
    Renders one note as a bullet, carrying its owner, date, and confirmation state.

    The `needs confirmation` marker is not decoration: an exported document is read
    without the app around it, so an inferred commitment that reads as agreed is exactly
    how a note nobody confirmed becomes something a team acts on.
    """
    parts: List[str] = [note.text.strip()]

    details = []
    if note.owner:
        details.append(f"owner: {note.owner}")
    if note.due_at:
        details.append(f"due: {note.due_at}")
    if note.status == NoteStatus.NEEDS_CONFIRMATION:
        details.append("**needs confirmation**")
    if note.status == NoteStatus.EDITED:
        details.append(f"edited by {note.updated_by}")

    if details:
        parts.append(f"({'; '.join(details)})")

    return f"- {' '.join(parts)}"


def render_markdown(artifact: SessionArtifact) -> str:
    """
    Renders a whole session artifact as a self-contained Markdown document (REQ-15).

    Algorithm:
    1. Header: mode, participants, languages, timing, and revision.
    2. Notes, grouped under the headings that mode uses.
    3. Quizzes, with their expected answers and explanations.
    4. The finalized transcript, which every note above links back to.

    Returns an empty-but-valid document when the session produced nothing, so a caller
    never has to special-case a session that happened but yielded no artifacts.
    """
    if artifact is None:
        return ""

    lines: List[str] = [
        f"# EchoSphere Session — {artifact.mode_label}",
        "",
        f"- **Session:** `{artifact.session_id}`",
        f"- **Mode:** {artifact.mode_label} (`{artifact.mode}`)",
        f"- **Participants:** {', '.join(artifact.participants) or '—'}",
        f"- **Languages:** {', '.join(artifact.languages) or '—'}",
        f"- **Started:** {_format_timestamp(artifact.started_at)}",
        f"- **Ended:** {_format_timestamp(artifact.ended_at)}",
        f"- **Revision:** {artifact.revision} (schema {artifact.schema_version})",
        "",
    ]

    if artifact.summary:
        lines += ["## Summary", "", artifact.summary, ""]

    sections = LEARNING_SECTIONS if artifact.mode == "language_learning" else WORK_SECTIONS
    grouped: Dict[str, List[NoteItem]] = {}
    for note in artifact.notes:
        grouped.setdefault(note.type, []).append(note)

    if artifact.notes:
        lines += ["## Notes", ""]
        for note_type, heading in sections:
            items = grouped.get(note_type)
            if not items:
                continue
            lines += [f"### {heading}", ""]
            lines += [_format_note(note) for note in items]
            lines.append("")

        # Any type outside this mode's ordered list still gets rendered rather than
        # silently dropped - losing a note in the export is worse than an odd heading.
        for note_type, items in grouped.items():
            if note_type in dict(sections):
                continue
            lines += [f"### {note_type.replace('_', ' ').title()}", ""]
            lines += [_format_note(note) for note in items]
            lines.append("")

    if artifact.quizzes:
        lines += ["## Knowledge Checks", ""]
        for index, quiz in enumerate(artifact.quizzes, start=1):
            lines.append(f"{index}. **{quiz.prompt}**")
            for option in quiz.options:
                lines.append(f"   - {option}")
            lines.append(f"   - _Answer:_ {quiz.expected_answer or '—'}")
            if quiz.explanation:
                lines.append(f"   - _Why:_ {quiz.explanation}")
            lines.append("")

    if artifact.transcript_turns:
        lines += ["## Transcript", ""]
        for turn in artifact.transcript_turns:
            lines.append(f"- **{turn.speaker_id}** ({turn.language}): {turn.text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
