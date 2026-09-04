"""
Summary:
    google_search.py is the REQ-18 research client: it looks up a word, a cultural
    reference, or a piece of task material and returns what the participant UI renders as
    a reference card.

    The client owns the *scoping* of the query as well as the call. A bare "meaning of
    一期一会" returns SEO filler; naming the language, and in work mode naming the
    material, is what makes the first result usable as an aside in a live conversation -
    and the caller (a turn handler, or a button) is not the right place to know that.

Key Classes:
    - SearchResult: one result, in the shape the reference card renders.
    - GoogleSearchTool: the configured client.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    call_transport,
    http_get_json,
)

logger = logging.getLogger("echosphere.agent.tools.search")

# Google's Custom Search JSON API. Endpoint-configurable so a deployment can point at a
# proxy without a code change; the key and the engine id are always server-side (REQ-08).
DEFAULT_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# Three results is what a reference card can show without becoming a search page. The API
# accepts 1-10; anything past the first few is scrolled past in a live conversation.
DEFAULT_MAX_RESULTS = 3
MAX_SUPPORTED_RESULTS = 10

# Spoken names for the languages this system supports. A query reading "in Japanese"
# retrieves explanations aimed at learners; a query reading "in ja" retrieves nothing.
LANGUAGE_NAMES = {"ja": "Japanese", "hi": "Hindi", "en": "English"}


@dataclass(frozen=True)
class SearchResult:
    """One search hit, reduced to what a reference card shows."""

    title: str
    snippet: str
    url: str
    image_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result for the `reference.card` event and API responses."""
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "image_url": self.image_url,
        }


class GoogleSearchTool:
    """
    Looks up reference material for a session (REQ-18).

    Credentials are read from the environment and stay there: the results travel to the
    browser, the key never does.
    """

    name = "search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        max_results: int = DEFAULT_MAX_RESULTS
    ):
        """
        Initialize the client from explicit values or the environment.

        Algorithm:
        1. Resolve the API key and the programmable-search engine id.
        2. Bind the transport - injected in tests, a plain JSON GET otherwise.
        3. Leave the tool unconfigured, not failed, when either credential is missing.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self.cse_id = cse_id if cse_id is not None else os.getenv("GOOGLE_SEARCH_CSE_ID", "")
        self.endpoint = endpoint
        self.max_results = max_results
        self._transport = transport or http_get_json

    @property
    def is_configured(self) -> bool:
        """Whether this server can actually run a search."""
        return bool(self.api_key and self.cse_id)

    def build_query(
        self,
        topic: str,
        language: Optional[str] = None,
        materials: bool = False
    ) -> str:
        """
        Scopes a raw topic into the query actually sent (REQ-18).

        Algorithm:
        1. Work-mode material lookups ask for the document or the explanation itself.
        2. Otherwise anchor the topic in the session's language, which is what turns a
           generic word lookup into a language-learning answer.
        3. A topic with no language context is sent as written.

        The two branches are deliberately different in kind: a colleague asking about
        "the kanban migration plan" wants the plan, while a learner asking about 一期一会
        wants the meaning and the culture behind it.
        """
        topic = (topic or "").strip()
        if not topic:
            return ""

        if materials:
            return f"{topic} reference material explanation"

        name = LANGUAGE_NAMES.get((language or "").strip().lower())
        if name:
            return f"{topic} meaning and usage in {name}"
        return topic

    def search(self, query: str, count: Optional[int] = None) -> List[SearchResult]:
        """
        Runs one search and returns its results (REQ-18).

        Algorithm:
        1. Refuse when unconfigured, so the dispatcher can report `unavailable` rather
           than surfacing a vendor 403 as a reference card.
        2. Call the API with the query as given - scoping is `build_query`'s job, and the
           caller decides whether a topic needs it.
        3. Map each item to a `SearchResult`.

        Raises:
            ToolNotConfiguredError: no API key or engine id on this server.
            ToolInvocationError: the vendor could not be reached, or refused.
        """
        if not self.is_configured:
            raise ToolNotConfiguredError(
                "Google Search is not configured. Set GOOGLE_SEARCH_API_KEY and "
                "GOOGLE_SEARCH_CSE_ID to enable reference lookups."
            )

        text = (query or "").strip()
        if not text:
            return []

        num = max(1, min(int(count or self.max_results), MAX_SUPPORTED_RESULTS))
        body = call_transport(self._transport, self.endpoint, {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": text,
            "num": num,
        })

        if not isinstance(body, dict):
            raise ToolInvocationError("Google Search returned an unreadable response.")

        results = [
            self._to_result(item) for item in (body.get("items") or [])
            if isinstance(item, dict)
        ]
        logger.info("Search for %r returned %d result(s).", text, len(results))
        return results[:num]

    @staticmethod
    def _to_result(item: Dict[str, Any]) -> SearchResult:
        """
        Maps one API item to a result, including the thumbnail when the page has one.

        The image is what makes a work-mode material card show the material rather than
        describe it; its absence is normal and never an error.
        """
        images = ((item.get("pagemap") or {}).get("cse_image") or [])
        image_url = ""
        if images and isinstance(images[0], dict):
            image_url = str(images[0].get("src") or "")

        return SearchResult(
            title=str(item.get("title") or ""),
            snippet=str(item.get("snippet") or ""),
            url=str(item.get("link") or ""),
            image_url=image_url
        )
