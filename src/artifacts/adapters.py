"""
Summary:
    adapters.py holds the optional external export targets for a session artifact.

    Optional is the operative word (REQ-15): the local repository is the source of truth
    and Markdown is the export that always works. Notion is a convenience for teams who
    already live there, so an unconfigured server reports that plainly rather than
    failing a session or pretending an export happened.

Key Classes:
    - NotionExportAdapter: pushes an artifact into a Notion page.
    - ExportNotConfiguredError: raised when the adapter has no credentials.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from src.artifacts.export import render_markdown
from src.artifacts.models import SessionArtifact

logger = logging.getLogger("echosphere.artifacts.adapters")

# Notion's API version header is required and pinned by the caller, not negotiated.
# Their API changes behavior by this header, so it is configurable rather than frozen
# here - check Notion's current version before changing it.
NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2022-06-28")
NOTION_API_URL = "https://api.notion.com/v1/pages"

# Notion rejects a text block longer than this, so long content is split across blocks.
NOTION_TEXT_LIMIT = 1900


class ExportNotConfiguredError(RuntimeError):
    """Raised when an optional export target has no credentials configured."""


class NotionExportAdapter:
    """
    Exports a session artifact to Notion as a child page (REQ-15, optional).

    Credentials are read from the environment and stay server-side, like every other
    credential in this system (REQ-16) - the browser never sees the integration token.
    """

    def __init__(self, api_key: Optional[str] = None, parent_page_id: Optional[str] = None):
        """
        Initialize the adapter from explicit values or the environment.

        Algorithm:
        1. Resolve the integration token and the parent page it writes children under.
        2. Leave the adapter unconfigured - not failed - when either is missing.
        """
        self.api_key = api_key if api_key is not None else os.getenv("NOTION_API_KEY", "")
        self.parent_page_id = (
            parent_page_id if parent_page_id is not None
            else os.getenv("NOTION_PARENT_PAGE_ID", "")
        )

    @property
    def is_configured(self) -> bool:
        """Whether this adapter has everything it needs to export."""
        return bool(self.api_key and self.parent_page_id)

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
                chunk = line[start:start + NOTION_TEXT_LIMIT]
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    }
                })
        return blocks

    def export(self, artifact: SessionArtifact) -> Dict[str, Any]:
        """
        Creates a Notion page for the artifact and returns the API's response summary.

        Algorithm:
        1. Refuse early when unconfigured, so the caller can report it as a 503 rather
           than surfacing a vendor auth error.
        2. Build the page payload - a titled child page of the configured parent.
        3. POST it, and return the created page's id and url.

        Raises:
            ExportNotConfiguredError: no API key or parent page configured.
            RuntimeError: the `requests` dependency is unavailable, or Notion refused.
        """
        if not self.is_configured:
            raise ExportNotConfiguredError(
                "Notion export is not configured. Set NOTION_API_KEY and "
                "NOTION_PARENT_PAGE_ID, or export as markdown instead."
            )

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Notion export needs the 'requests' package, which is not installed."
            ) from exc

        payload = {
            "parent": {"page_id": self.parent_page_id},
            "properties": {
                "title": [{
                    "type": "text",
                    "text": {"content": f"EchoSphere Session — {artifact.mode_label}"}
                }]
            },
            "children": self.build_blocks(artifact),
        }

        response = requests.post(
            NOTION_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Notion-Version": NOTION_API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Notion rejected the export ({response.status_code}): {response.text[:300]}"
            )

        body = response.json()
        logger.info(
            "Exported session %s to Notion page %s.", artifact.session_id, body.get("id")
        )
        return {"page_id": body.get("id"), "url": body.get("url")}
