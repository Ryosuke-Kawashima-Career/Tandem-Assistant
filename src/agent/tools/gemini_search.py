"""
Summary:
    gemini_search.py is the REQ-18 research client, adopted in place of Google's Custom
    Search JSON API after that API proved persistently unreachable on this deployment
    (403 "This project does not have the access to Custom Search JSON API", confirmed to
    survive project/billing/enablement/key-restriction checks and a freshly generated
    key - see the uiux_debugging task plan's Phase 1.5/1.6 troubleshooting record). Gemini's
    built-in Search grounding runs the same underlying Google Search, but through the
    `google-genai` credential (`GEMINI_API_KEY`) this deployment already has working -
    no separate Cloud project, Custom Search Engine id, or billing link to fight.

    The trade-off this client absorbs so the rest of the app never sees it: grounding
    does not return a clean list of (title, snippet, url) hits the way Custom Search did.
    It returns one generated answer plus a `grounding_metadata` block of *citations* -
    `grounding_chunks` (one per source, each just a domain-level title and a Google
    redirect URL - the API does not expose the source's real title or full URL) and
    `grounding_supports` (which spans of the generated answer each citation backs). This
    module reconstructs a `SearchResult` per citation, with `snippet` built from the
    spans of the generated answer that citation actually supports - so a card still
    reads as a paraphrasable claim tied to a source, the same shape `ReferenceCard.js`
    already renders, rather than a bare title-plus-link.

    A second trade-off, confirmed live rather than assumed: grounding is the model's own
    per-call decision, not a guarantee - the identical query returned 3 citations on one
    call and 0 on an immediate retry, most often when the model judges its training data
    already answers a well-established fact. An empty result in that case would
    non-deterministically reproduce the exact "Search does nothing" complaint this tool
    was adopted to fix, so an ungrounded call still returns the model's own answer as one
    plainly-labeled, unsourced result rather than nothing at all.

Key Classes:
    - GeminiGroundedSearchTool: drop-in replacement for `GoogleSearchTool` behind
      `ToolDispatcher` - same `name`, `is_configured`, `build_query`, and `search`
      surface, so nothing above this module (dispatch, the API route, the frontend)
      needed to change.
"""

import logging
import os
from typing import Any, Callable, List, Optional

from src.agent.tools.base import ToolInvocationError, ToolNotConfiguredError
from src.agent.tools.google_search import LANGUAGE_NAMES, SearchResult

logger = logging.getLogger("echosphere.agent.tools.search")

try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Matches this project's own default conversational tier (`DEFAULT_GEMINI_MODEL` in
# `src/agent/orchestrator.py`) rather than inventing a separate one; confirmed live to
# support Search grounding. Independently overridable because a search call has a
# different cost/latency profile than a turn reply.
DEFAULT_SEARCH_MODEL = "gemini-3.5-flash"

# Three results, matching `GoogleSearchTool.DEFAULT_MAX_RESULTS` - a reference card is
# read in a live conversation, not scrolled like a search results page.
DEFAULT_MAX_RESULTS = 3
MAX_SUPPORTED_RESULTS = 10

# How much of the answer text a citation's own snippet is allowed to carry. Several
# grounding_supports can back the same chunk; joined without a cap they can run to the
# length of the whole generated answer, which reads as a wall of text under one link.
MAX_SNIPPET_CHARS = 320

# Longer than a per-citation snippet's cap: this is the entire answer standing in for a
# result set the model chose not to ground, not one paraphrase excerpt among several.
UNGROUNDED_ANSWER_MAX_CHARS = 600


class GeminiGroundedSearchTool:
    """
    Looks up reference material via Gemini's built-in Google Search grounding (REQ-18).

    The API key is the same `GEMINI_API_KEY` every other Gemini-engine feature in this
    app already uses - there is no separate credential to configure or leak.
    """

    name = "search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        max_results: int = DEFAULT_MAX_RESULTS
    ):
        """
        Initialize the client from explicit values or the environment.

        Algorithm:
        1. Resolve the API key - shared with every other Gemini call in this process.
        2. Bind the model - overridable independently of the conversational tier.
        3. Accept an injected client (tests); build the real `genai.Client` lazily
           otherwise, so a server holding no key never imports or touches the SDK.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("ECHOSPHERE_GEMINI_SEARCH_MODEL", DEFAULT_SEARCH_MODEL)
        self.max_results = max_results
        self._client = client

    @property
    def is_configured(self) -> bool:
        """Whether this server can actually run a grounded search."""
        return bool(self.api_key) and GOOGLE_GENAI_AVAILABLE

    def build_query(
        self,
        topic: str,
        language: Optional[str] = None,
        materials: bool = False
    ) -> str:
        """
        Scopes a raw topic into the query actually sent (REQ-18).

        Identical scoping to `GoogleSearchTool.build_query` - the underlying search
        vendor changed, not what makes a lookup usable in a live conversation.
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
        Runs one grounded search and returns its results (REQ-18).

        Algorithm:
        1. Refuse when unconfigured, so the dispatcher reports `unavailable` rather than
           surfacing an SDK import error or a missing-key exception as a reference card.
        2. Ask Gemini to answer the query with Search grounding enabled.
        3. Reconstruct one `SearchResult` per citation the model actually used.

        Raises:
            ToolNotConfiguredError: no `GEMINI_API_KEY`, or the `google-genai` package
                is not installed.
            ToolInvocationError: the vendor could not be reached, or refused.
        """
        if not self.is_configured:
            raise ToolNotConfiguredError(
                "Google Search is not configured. Set GEMINI_API_KEY to enable "
                "Gemini-grounded reference lookups."
            )

        text = (query or "").strip()
        if not text:
            return []

        num = max(1, min(int(count or self.max_results), MAX_SUPPORTED_RESULTS))

        try:
            response = self._client_or_default().models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
        except (ToolInvocationError, ToolNotConfiguredError):
            raise
        except Exception as exc:  # noqa: BLE001 - every SDK failure is one outcome here
            raise ToolInvocationError(f"Gemini grounded search failed: {exc}") from exc

        results = self._to_results(response, num)
        logger.info("Grounded search for %r returned %d result(s).", text, len(results))
        return results

    def _client_or_default(self) -> Any:
        """Returns the injected client, or builds the real one on first use."""
        if self._client is not None:
            return self._client
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _to_results(response: Any, num: int) -> List[SearchResult]:
        """
        Reconstructs a `SearchResult` per grounding citation, in citation order.

        Algorithm:
        1. Bail out to an empty list when the model grounded on nothing - a real
           possibility for an obscure or malformed query, and not itself a failure.
        2. Map each citation's supporting answer spans to it, so its snippet is a real
           paraphrase excerpt rather than a bare title-and-link.
        3. Fall back to a slice of the whole answer for a citation no span names
           directly - rare, but a citation with no snippet reads as broken to a viewer
           who has no way to know the vendor simply didn't attribute a span to it.

        The redirect URL Google returns (`vertexaisearch.cloud.google.com/...`) is used
        as-is: it resolves to the real source when followed, and grounding responses do
        not expose the source's direct URL at all, so there is nothing more direct to
        substitute in.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = list(getattr(metadata, "grounding_chunks", None) or []) if metadata else []
        if not chunks:
            # Search grounding is the model's own per-call decision, not a guarantee: the
            # identical query can come back grounded on live sources one call and not the
            # next, most often when the model judges its training data already answers a
            # well-established fact. Confirmed live - this exact query returned 3 citations
            # on one call and 0 on a retry. An unsourced result set showing nothing here
            # would non-deterministically reproduce the exact "Search does nothing"
            # complaint this tool was adopted to fix, so the model's own answer is
            # returned as one plainly-labeled, unsourced result instead of a silent empty
            # list - honest about not being source-linked (REQ-18 asks for source-linked
            # material), but a fixed vendor bug and a variable model decision call for
            # different responses, and only one of the two is something a person can act
            # on by simply asking again.
            answer = (getattr(response, "text", "") or "").strip()
            if not answer:
                return []
            return [SearchResult(
                title="Gemini's answer (no live source found this time)",
                snippet=answer[:UNGROUNDED_ANSWER_MAX_CHARS],
                url="",
                image_url=""
            )]

        supports = list(getattr(metadata, "grounding_supports", None) or [])
        snippets_by_index: dict = {}
        for support in supports:
            segment = getattr(support, "segment", None)
            segment_text = (getattr(segment, "text", "") or "").strip()
            if not segment_text:
                continue
            for index in getattr(support, "grounding_chunk_indices", None) or []:
                snippets_by_index.setdefault(index, []).append(segment_text)

        fallback_snippet = (getattr(response, "text", "") or "").strip()[:MAX_SNIPPET_CHARS]

        results: List[SearchResult] = []
        for index, chunk in enumerate(chunks[:num]):
            web = getattr(chunk, "web", None)
            title = (getattr(web, "title", "") or getattr(web, "domain", "") or "").strip()
            url = (getattr(web, "uri", "") or "").strip()
            if not url:
                continue

            own_spans = snippets_by_index.get(index) or []
            # Deduplicated while preserving order: the same short claim can legitimately
            # back a citation twice across separate `grounding_supports` entries.
            seen = set()
            deduped = [s for s in own_spans if not (s in seen or seen.add(s))]
            snippet = " ".join(deduped)[:MAX_SNIPPET_CHARS] if deduped else fallback_snippet

            results.append(SearchResult(
                title=title or url,
                snippet=snippet,
                url=url,
                image_url=""
            ))

        return results
