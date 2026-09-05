"""
Summary:
    test_agent_tools_gemini_search.py is the executable specification for REQ-18's
    current search client, `GeminiGroundedSearchTool` - adopted after Google's Custom
    Search JSON API proved persistently unreachable on this deployment (see the
    uiux_debugging task plan's Phase 1.5/1.6 troubleshooting record). It runs through
    Gemini's built-in Search grounding instead, using the `GEMINI_API_KEY` credential
    every other Gemini-engine feature in this app already depends on.

Covers:
    - Configuration gate: no `GEMINI_API_KEY` (or no `google-genai` package) means
      unconfigured, exactly like the tool it replaced.
    - Query scoping: identical behavior to `GoogleSearchTool.build_query` - the vendor
      changed, not what makes a lookup usable in a live conversation.
    - Citation reconstruction: a `SearchResult` per grounding chunk, with its snippet
      built from the answer spans that actually cite it - the part this client has to
      do that the old one didn't, because grounding returns citations for a generated
      answer rather than a page of independent hits.
    - Failure handling: an SDK/vendor error becomes this package's own `ToolInvocationError`,
      exactly like every other tool in this dispatcher.
    - `ToolDispatcher.from_env()` wires this tool in as the default `search`.
"""

import unittest
from unittest.mock import MagicMock

from src.agent.tools.base import ToolInvocationError, ToolNotConfiguredError, ToolState
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.gemini_search import GeminiGroundedSearchTool
from src.sessions.models import SessionRecord

API_KEY = "test-gemini-key"


def learning_session():
    """A language-learning session, whose searches are scoped to the target language."""
    return SessionRecord.create(
        channel="tools-learning", mode="language_learning",
        languages=["ja"], participants=["Kenji"]
    )


# -- Fakes shaped like the real `google-genai` response schema ------------------------
# Confirmed live against the real API (see the task plan): `GroundingChunkWeb` carries
# `title`/`uri`/`domain`; `GroundingSupport` carries `segment.text` and
# `grounding_chunk_indices`. Plain attribute holders, not the real SDK's pydantic
# models, so these tests exercise this module's own attribute access (`getattr(...,
# None)`-guarded) without depending on the SDK being installed a particular way.

class FakeWeb:
    def __init__(self, uri, title=""):
        self.uri = uri
        self.title = title


class FakeChunk:
    def __init__(self, uri, title=""):
        self.web = FakeWeb(uri, title)


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeSupport:
    def __init__(self, text, indices):
        self.segment = FakeSegment(text)
        self.grounding_chunk_indices = indices


class FakeGroundingMetadata:
    def __init__(self, chunks, supports=None):
        self.grounding_chunks = chunks
        self.grounding_supports = supports or []


class FakeCandidate:
    def __init__(self, grounding_metadata):
        self.grounding_metadata = grounding_metadata


class FakeResponse:
    def __init__(self, text="", candidates=None):
        self.text = text
        self.candidates = candidates if candidates is not None else []


class RecordingClient:
    """Stands in for `genai.Client`, recording the call and replaying a canned response."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else FakeResponse()
        self.error = error
        self.calls = []
        self.models = self  # `client.models.generate_content(...)` - see gemini_search.py

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


class TestGeminiGroundedSearchTool(unittest.TestCase):
    """Test suite for the search client itself (REQ-18)."""

    def test_a_tool_without_a_key_is_unconfigured_and_refuses_to_run(self):
        """
        Verify an unconfigured tool says so and raises rather than reaching the network.

        Same contract `GoogleSearchTool` held: the server is where a missing credential
        has to be caught, or a vendor failure leaks into a learner's reference card.
        """
        tool = GeminiGroundedSearchTool(api_key="", client=RecordingClient())

        self.assertFalse(tool.is_configured)
        with self.assertRaises(ToolNotConfiguredError):
            tool.search("keigo")

    def test_a_search_reconstructs_results_from_citations(self):
        """
        Verify each grounding chunk becomes a result, snippeted from its own citations.

        A citation backed by two supports gets both, joined; a citation nothing cites
        directly still gets a usable answer excerpt rather than an empty snippet.
        """
        response = FakeResponse(
            text="Ichigo ichie means treasuring a once-in-a-lifetime encounter.",
            candidates=[FakeCandidate(FakeGroundingMetadata(
                chunks=[
                    FakeChunk("https://redirect.example/a", "example.com"),
                    FakeChunk("https://redirect.example/b", "other.example"),
                ],
                supports=[
                    FakeSupport("Ichigo ichie means treasuring a once-in-a-lifetime", [0]),
                    FakeSupport("encounter.", [0]),
                ]
            ))]
        )
        client = RecordingClient(response=response)
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=client)

        results = tool.search("ichigo ichie meaning", count=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "example.com")
        self.assertEqual(results[0].url, "https://redirect.example/a")
        self.assertEqual(
            results[0].snippet,
            "Ichigo ichie means treasuring a once-in-a-lifetime encounter."
        )
        self.assertEqual(results[0].image_url, "")
        # Second chunk has no support naming it -> falls back to the answer text.
        self.assertEqual(results[1].title, "other.example")
        self.assertEqual(results[1].snippet, response.text)

        call = client.calls[0]
        self.assertEqual(call["contents"], "ichigo ichie meaning")

    def test_results_are_capped_at_count(self):
        """Verify more citations than asked for are truncated, not just left unbounded."""
        chunks = [FakeChunk(f"https://redirect.example/{i}") for i in range(5)]
        response = FakeResponse(text="answer", candidates=[
            FakeCandidate(FakeGroundingMetadata(chunks=chunks))
        ])
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient(response=response))

        self.assertEqual(len(tool.search("topic", count=2)), 2)

    def test_query_scoping_names_the_target_language_for_a_learning_session(self):
        """Verify the query carries the session's language, same as the prior client."""
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient())

        query = tool.build_query("一期一会", language="ja")

        self.assertIn("一期一会", query)
        self.assertIn("Japanese", query)

    def test_query_scoping_asks_for_materials_in_work_mode(self):
        """Verify the work-mode query targets task material, same as the prior client."""
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient())

        query = tool.build_query("kanban migration plan", language="en", materials=True)

        self.assertIn("kanban migration plan", query)
        self.assertNotIn("culture", query.lower())

    def test_a_vendor_failure_becomes_a_tool_invocation_error(self):
        """
        Verify an SDK/vendor failure surfaces as this module's own error type.

        The dispatcher turns exactly this into a `tool.status` event; letting a raw
        `google-genai` exception escape would make that translation depend on which
        exception type the SDK happens to raise for a given failure.
        """
        tool = GeminiGroundedSearchTool(
            api_key=API_KEY,
            client=RecordingClient(error=RuntimeError("503 UNAVAILABLE"))
        )

        with self.assertRaises(ToolInvocationError):
            tool.search("keigo")

    def test_an_ungrounded_answer_returns_one_plainly_labeled_result(self):
        """
        Verify a call the model chose not to ground still surfaces its answer.

        Confirmed live: the identical query can return citations on one call and none
        on a retry - the model decides whether to search per call, not this client.
        Showing nothing here would non-deterministically reproduce the exact "Search
        does nothing" complaint this tool was adopted to fix, so the model's own
        answer stands in, clearly not claiming to be source-linked.
        """
        response = FakeResponse(text="Keigo is the Japanese system of honorific speech.", candidates=[
            FakeCandidate(FakeGroundingMetadata(chunks=[]))
        ])
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient(response=response))

        results = tool.search("keigo")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].snippet, response.text)
        self.assertEqual(results[0].url, "")

    def test_no_answer_and_no_grounding_returns_nothing(self):
        """Verify a response with neither text nor citations returns no results, not a crash."""
        response = FakeResponse(text="", candidates=[
            FakeCandidate(FakeGroundingMetadata(chunks=[]))
        ])
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient(response=response))

        self.assertEqual(tool.search("a phrase nobody has written about"), [])

    def test_an_empty_query_returns_nothing_without_calling_the_vendor(self):
        """Verify a blank query is handled locally rather than spent on a vendor call."""
        client = RecordingClient()
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=client)

        self.assertEqual(tool.search("   "), [])
        self.assertEqual(client.calls, [])


class TestGeminiSearchDispatch(unittest.TestCase):
    """
    Test suite confirming `GeminiGroundedSearchTool` interoperates with the dispatcher
    exactly like the tool it replaced (REQ-18) - the integration point that actually
    matters, since `search_reference`'s own event-publishing logic is unchanged and
    already covered by `test_agent_tools_search.py`.
    """

    def test_search_reference_publishes_a_card_built_from_citations(self):
        response = FakeResponse(
            text="Keigo is the Japanese system of honorific speech.",
            candidates=[FakeCandidate(FakeGroundingMetadata(
                chunks=[FakeChunk("https://redirect.example/keigo", "example.com")],
                supports=[FakeSupport(
                    "Keigo is the Japanese system of honorific speech.", [0]
                )]
            ))]
        )
        data_stream = MagicMock()
        data_stream.send_tool_event.return_value = True
        tool = GeminiGroundedSearchTool(api_key=API_KEY, client=RecordingClient(response=response))
        dispatcher = ToolDispatcher(data_stream=data_stream, search=tool)

        result = dispatcher.search_reference(learning_session(), "keigo")

        self.assertEqual(result.state, ToolState.OK)
        self.assertEqual(len(result.payload["results"]), 1)
        self.assertEqual(result.payload["results"][0]["url"], "https://redirect.example/keigo")
        data_stream.send_tool_event.assert_called_once()

    def test_from_env_wires_this_tool_in_as_the_default_search_client(self):
        """
        Verify `ToolDispatcher.from_env()` picks Gemini grounding, not Custom Search.

        This is the actual behavior change this adoption makes: nothing above the
        dispatcher (the API route, the frontend) had to change, because both search
        clients share the same duck-typed surface - only which one gets constructed did.
        """
        dispatcher = ToolDispatcher.from_env()

        self.assertIsInstance(dispatcher.search, GeminiGroundedSearchTool)


if __name__ == "__main__":
    unittest.main()
