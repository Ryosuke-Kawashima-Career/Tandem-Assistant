"""
Summary:
    adapters.py holds the optional external export targets for a session artifact.

    Optional is the operative word (REQ-15): the local repository is the source of truth
    and Markdown is the export that always works. Notion is a convenience for teams who
    already live there, so an unconfigured server reports that plainly rather than
    failing a session or pretending an export happened.

    Two Notion targets are supported. A *page* export files the session as a child page,
    which is what a reader opens once. A *database* export files it as a row, and adds
    the session's vocabulary and knowledge checks as toggle questions - the shape the
    user story asks for, where the answer stays hidden until the learner has attempted
    it. Both are rendered from the same Markdown, so the two cannot describe the same
    session differently.

Key Classes:
    - NotionExportAdapter: pushes an artifact into a Notion page or database row.
    - ExportNotConfiguredError: raised when the adapter has no credentials.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from src.artifacts.export import render_markdown
from src.artifacts.models import NoteItem, SessionArtifact

logger = logging.getLogger("echosphere.artifacts.adapters")

# Notion's API version header is required and pinned by the caller, not negotiated.
# Their API changes behavior by this header, so it is configurable rather than frozen
# here - check Notion's current version before changing it.
NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2022-06-28")
NOTION_API_URL = "https://api.notion.com/v1/pages"

# Notion rejects a text block longer than this, so long content is split across blocks.
NOTION_TEXT_LIMIT = 1900

# The two supported destinations for an export.
TARGET_PAGE = "page"
TARGET_DATABASE = "database"
SUPPORTED_TARGETS = (TARGET_PAGE, TARGET_DATABASE)

# Note types that revise well as a question: a term on the front, its meaning behind the
# toggle. A decision or an action item is a record, not a thing to be quizzed on.
TOGGLE_NOTE_TYPES = ("vocabulary", "term", "glossary")

# Separators between a term and its meaning, matching the note generator's own wording.
TOGGLE_SEPARATORS = (" — ", " – ", " - ", ": ")


class ExportNotConfiguredError(RuntimeError):
    """Raised when an optional export target has no credentials."""


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Posts one JSON document to Notion and returns the decoded response.

    The default transport, injectable so the adapter can be exercised without a network.
    A rejection is raised with Notion's own message: their errors name the actual
    problem ("Could not find database with ID"), and that text is the only place it
    appears.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Notion export needs the 'requests' package, which is not installed."
        ) from exc

    response = requests.post(url, json=payload, headers=headers or {}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Notion rejected the export ({response.status_code}): {response.text[:300]}"
        )
    return response.json()


class NotionExportAdapter:
    """
    Exports a session artifact to Notion (REQ-15, optional).

    Credentials are read from the environment and stay server-side, like every other
    credential in this system (REQ-16) - the browser never sees the integration token.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        parent_page_id: Optional[str] = None,
        database_id: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None
    ):
        """
        Initialize the adapter from explicit values or the environment.

        Algorithm:
        1. Resolve the integration token and both possible destinations.
        2. Bind the transport - injected in tests, a plain JSON POST otherwise.
        3. Leave the adapter unconfigured - not failed - when the token or every
           destination is missing.
        """
        self.api_key = api_key if api_key is not None else os.getenv("NOTION_API_KEY", "")
        self.parent_page_id = (
            parent_page_id if parent_page_id is not None
            else os.getenv("NOTION_PARENT_PAGE_ID", "")
        )
        self.database_id = (
            database_id if database_id is not None
            else os.getenv("NOTION_DATABASE_ID", "")
        )
        self._transport = transport or _post_json

    @property
    def is_configured(self) -> bool:
        """Whether this adapter has a token and somewhere to write."""
        return bool(self.api_key and (self.parent_page_id or self.database_id))

    @property
    def default_target(self) -> str:
        """
        The destination used when a caller does not name one.

        A configured page wins, because that is the target this adapter shipped with and
        an existing deployment's exports must keep landing where they always have.
        """
        if self.parent_page_id:
            return TARGET_PAGE
        return TARGET_DATABASE if self.database_id else TARGET_PAGE

    # -- Block builders ---------------------------------------------------------------

    @staticmethod
    def _text_block(block_type: str, content: str) -> Dict[str, Any]:
        """Builds one Notion block of the given type from plain text."""
        return {
            "object": "block",
            "type": block_type,
            block_type: {"rich_text": [{"type": "text", "text": {"content": content}}]},
        }

    def build_blocks(self, artifact: SessionArtifact) -> List[Dict[str, Any]]:
        """
        Renders the artifact as Notion paragraph blocks.

        Deliberately built from the Markdown export rather than from the artifact
        directly: one rendering decides what a session looks like when it leaves
        EchoSphere, so the Notion page and the Markdown file cannot drift apart.
        """
        markdown = render_markdown(artifact)
        blocks: List[Dict[str, Any]] = []

        for line in markdown.splitlines():
            if not line.strip():
                continue
            for start in range(0, len(line), NOTION_TEXT_LIMIT):
                blocks.append(self._text_block("paragraph", line[start:start + NOTION_TEXT_LIMIT]))
        return blocks

    def build_toggle_blocks(self, artifact: SessionArtifact) -> List[Dict[str, Any]]:
        """
        Renders the session's revisable content as question toggles (REQ-15 amendment).

        Algorithm:
        1. One toggle per knowledge check: the question outside, the answer and its
           explanation inside.
        2. One toggle per vocabulary or terminology note: the term outside, its meaning
           inside.

        The split is the point. A learner revising this page has to attempt the answer
        before revealing it; a page that shows both at once is a transcript with extra
        formatting.
        """
        blocks: List[Dict[str, Any]] = []

        if artifact.quizzes:
            blocks.append(self._text_block("heading_2", "Knowledge Checks"))
            for quiz in artifact.quizzes:
                answer: List[Dict[str, Any]] = [
                    self._text_block("paragraph", f"Answer: {quiz.expected_answer or '—'}")
                ]
                if quiz.explanation:
                    answer.append(self._text_block("paragraph", quiz.explanation))
                blocks.append(self._toggle(quiz.prompt, answer))

        vocabulary = [note for note in artifact.notes if note.type in TOGGLE_NOTE_TYPES]
        if vocabulary:
            blocks.append(self._text_block("heading_2", "Vocabulary & Terminology"))
            for note in vocabulary:
                front, back = self._split_note(note)
                blocks.append(self._toggle(front, [self._text_block("paragraph", back or "—")]))

        return blocks

    def _toggle(self, question: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Builds one toggle block whose children stay hidden until it is opened."""
        block = self._text_block("toggle", question)
        block["toggle"]["children"] = children
        return block

    @staticmethod
    def _split_note(note: NoteItem) -> tuple:
        """Splits a note into the term to recall and the meaning hidden behind it."""
        text = (note.text or "").strip()
        for separator in TOGGLE_SEPARATORS:
            if separator in text:
                front, back = text.split(separator, 1)
                return front.strip(), back.strip()
        return text, ""

    # -- Export -----------------------------------------------------------------------

    def export(
        self,
        artifact: SessionArtifact,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a Notion page for the artifact and returns the API's response summary.

        Algorithm:
        1. Resolve and validate the target, so a typo names an error rather than
           silently exporting somewhere else.
        2. Refuse early when unconfigured, so the caller can report it as a 503 rather
           than surfacing a vendor auth error.
        3. Build the payload for that target: a titled child page, or a database row
           whose body leads with the session's question toggles.
        4. POST it, and return the created page's id and url.

        Raises:
            ValueError: the requested target is not one this adapter supports.
            ExportNotConfiguredError: no API key, or no destination of that kind.
            RuntimeError: the `requests` dependency is unavailable, or Notion refused.
        """
        resolved = (target or self.default_target).strip().lower()
        if resolved not in SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported Notion export target {target!r}; "
                f"expected one of {list(SUPPORTED_TARGETS)}."
            )

        if not self.is_configured:
            raise ExportNotConfiguredError(
                "Notion export is not configured. Set NOTION_API_KEY and either "
                "NOTION_PARENT_PAGE_ID or NOTION_DATABASE_ID, or export as markdown "
                "instead."
            )

        title = f"EchoSphere Session — {artifact.mode_label}"

        if resolved == TARGET_DATABASE:
            if not self.database_id:
                raise ExportNotConfiguredError(
                    "Notion database export is not configured. Set NOTION_DATABASE_ID, "
                    "or export to the configured page instead."
                )
            parent = {"database_id": self.database_id}
            # Addressed by the property *id* `title` rather than by a column name: every
            # database has exactly one title property, but its name is whatever its
            # creator typed ("Name", "Session", ...), and guessing wrong is a 400.
            properties = {"title": {"title": [{"type": "text", "text": {"content": title}}]}}
            children = self.build_toggle_blocks(artifact) + self.build_blocks(artifact)
        else:
            if not self.parent_page_id:
                raise ExportNotConfiguredError(
                    "Notion page export is not configured. Set NOTION_PARENT_PAGE_ID, "
                    "or export to the configured database instead."
                )
            parent = {"page_id": self.parent_page_id}
            properties = {"title": [{"type": "text", "text": {"content": title}}]}
            children = self.build_blocks(artifact)

        body = self._transport(
            NOTION_API_URL,
            {"parent": parent, "properties": properties, "children": children},
            {
                "Authorization": f"Bearer {self.api_key}",
                "Notion-Version": NOTION_API_VERSION,
                "Content-Type": "application/json",
            }
        )

        logger.info(
            "Exported session %s to Notion %s %s.",
            artifact.session_id, resolved, (body or {}).get("id")
        )
        return {
            "page_id": (body or {}).get("id"),
            "url": (body or {}).get("url"),
            "target": resolved,
        }
