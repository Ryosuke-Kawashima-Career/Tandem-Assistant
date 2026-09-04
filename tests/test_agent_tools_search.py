"""
Summary:
    test_agent_tools_search.py is the executable specification for REQ-18: the agent
    researches a target-language, culture, or task-material question with Google Search
    and surfaces the answer as a source-linked reference card - without ever blocking the
    voice path, and without the API key leaving the server.

Covers:
    - REQ-18 search client: configuration gate, query scoping, result parsing, failures.
    - REQ-18 dispatch: `reference.card` on success, `tool.status` on unavailable/failed.
    - REQ-18 asynchrony: an agent-initiated lookup is dispatched off the caller's thread.
    - REQ-08: the API key travels server-side only and never enters an event payload.
"""

import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock

from src.agent.orchestrator import TeachingAgent
from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    ToolState,
)
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.google_search import GoogleSearchTool
from src.sessions.models import SessionRecord

API_KEY = "test-search-key"
CSE_ID = "test-cse-id"

# Shaped like one page of the Google Custom Search JSON API, including the pagemap image
# a work-mode material card renders.
SEARCH_RESPONSE = {
    "items": [
        {
            "title": "Ichigo ichie - a once-in-a-lifetime encounter",
            "snippet": "A Japanese four-character idiom about treasuring a meeting.",
            "link": "https://example.com/ichigo-ichie",
            "pagemap": {"cse_image": [{"src": "https://example.com/ichigo.png"}]},
        },
        {
            "title": "Tea ceremony origins",
            "snippet": "The phrase comes from the tea ceremony tradition.",
            "link": "https://example.com/tea",
        },
    ]
}


def learning_session():
    """A language-learning session, whose searches are scoped to the target language."""
    return SessionRecord.create(
        channel="tools-learning", mode="language_learning",
        languages=["ja"], participants=["Kenji"]
    )


def work_session():
    """An international-work session, whose searches are scoped to task materials."""
    return SessionRecord.create(
        channel="tools-work", mode="international_work",
        languages=["en"], participants=["Priya"]
    )


class RecordingTransport:
    """A stand-in for the HTTP GET, recording its call and returning a canned page."""

    def __init__(self, response=None, error=None):
        """Store the response to replay, or the error to raise instead."""
        self.response = response if response is not None else SEARCH_RESPONSE
        self.error = error
        self.calls = []

    def __call__(self, url, params):
        """Record the request and replay the canned outcome."""
        self.calls.append({"url": url, "params": params})
        if self.error is not None:
            raise self.error
        return self.response


class TestGoogleSearchTool(unittest.TestCase):
    """Test suite for the search client itself (REQ-18, TASK-12.1)."""

    def test_a_tool_without_credentials_is_unconfigured_and_refuses_to_run(self):
        """
        Verify an unconfigured tool says so and raises rather than reaching the network.

        REQ-18 keeps the key server-side, which means the server is where the absence of
        one has to be detected - a request built without it would leak a 403 from Google
        into a learner's reference card.
        """
        tool = GoogleSearchTool(api_key="", cse_id="")

        self.assertFalse(tool.is_configured)
        with self.assertRaises(ToolNotConfiguredError):
            tool.search("keigo")

    def test_a_search_sends_the_credentials_and_parses_the_results(self):
        """Verify the client authenticates the request and maps items to results."""
        transport = RecordingTransport()
        tool = GoogleSearchTool(api_key=API_KEY, cse_id=CSE_ID, transport=transport)

        results = tool.search("一期一会 meaning", count=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Ichigo ichie - a once-in-a-lifetime encounter")
        self.assertEqual(results[0].url, "https://example.com/ichigo-ichie")
        self.assertEqual(results[0].image_url, "https://example.com/ichigo.png")
        self.assertEqual(results[1].image_url, "")

        params = transport.calls[0]["params"]
        self.assertEqual(params["key"], API_KEY)
        self.assertEqual(params["cx"], CSE_ID)
        self.assertEqual(params["num"], 2)

    def test_query_scoping_names_the_target_language_for_a_learning_session(self):
        """
        Verify the query carries the session's language (REQ-18 scoped query builder).

        A bare "meaning of ganbatte" returns English-language SEO filler; the language
        and the word "meaning" are what make the top result usable as a lesson aside.
        """
        tool = GoogleSearchTool(api_key=API_KEY, cse_id=CSE_ID, transport=RecordingTransport())

        query = tool.build_query("一期一会", language="ja")

        self.assertIn("一期一会", query)
        self.assertIn("Japanese", query)

    def test_query_scoping_asks_for_materials_in_work_mode(self):
        """Verify the work-mode query targets task material rather than language study."""
        tool = GoogleSearchTool(api_key=API_KEY, cse_id=CSE_ID, transport=RecordingTransport())

        query = tool.build_query("kanban migration plan", language="en", materials=True)

        self.assertIn("kanban migration plan", query)
        self.assertNotIn("culture", query.lower())

    def test_a_transport_failure_becomes_a_tool_invocation_error(self):
        """
        Verify a network or vendor failure surfaces as this module's own error type.

        The dispatcher turns exactly this into a `tool.status` event; letting a raw
        `requests` exception escape would make that translation depend on which HTTP
        library happens to be installed.
        """
        tool = GoogleSearchTool(
            api_key=API_KEY, cse_id=CSE_ID,
            transport=RecordingTransport(error=RuntimeError("connection reset"))
        )

        with self.assertRaises(ToolInvocationError):
            tool.search("keigo")

    def test_an_empty_result_page_is_not_an_error(self):
        """Verify a query with no hits returns nothing rather than raising."""
        tool = GoogleSearchTool(
            api_key=API_KEY, cse_id=CSE_ID, transport=RecordingTransport(response={})
        )

        self.assertEqual(tool.search("a phrase nobody has written about"), [])


class TestSearchDispatch(unittest.TestCase):
    """Test suite for publishing a reference card over RTC (REQ-18, TASK-12.2)."""

    def setUp(self):
        """Use a recording data stream and a configured search tool per test."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.transport = RecordingTransport()
        self.tool = GoogleSearchTool(
            api_key=API_KEY, cse_id=CSE_ID, transport=self.transport
        )
        self.dispatcher = ToolDispatcher(data_stream=self.data_stream, search=self.tool)

    def published(self, event_type):
        """Returns the payloads published under one event type."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == event_type
        ]

    def test_a_successful_search_publishes_a_reference_card(self):
        """Verify the card carries title, snippet, and source URL under the envelope."""
        session = learning_session()

        result = self.dispatcher.search_reference(session, "一期一会")

        self.assertTrue(result.ok)
        cards = self.published("reference.card")
        self.assertEqual(len(cards), 1)
        payload = cards[0]
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["mode"], "language_learning")
        self.assertTrue(payload["event_id"])
        card = payload["card"]
        self.assertEqual(card["query"], "一期一会")
        self.assertEqual(card["results"][0]["url"], "https://example.com/ichigo-ichie")
        self.assertTrue(card["results"][0]["snippet"])

    def test_a_work_session_card_carries_the_material_image(self):
        """
        Verify work mode surfaces the image REQ-18 promises for task materials.

        Onboarding on a document nobody can see is the gap this closes: the detail and
        the picture of the material are what make a term concrete to a new colleague.
        """
        self.dispatcher.search_reference(work_session(), "kanban migration plan")

        card = self.published("reference.card")[0]["card"]
        self.assertTrue(card["materials"])
        self.assertEqual(card["results"][0]["image_url"], "https://example.com/ichigo.png")

    def test_the_api_key_never_reaches_the_event_payload(self):
        """
        Verify no published payload carries the credential (REQ-08).

        The reference card is broadcast to every participant over the RTC data stream, so
        anything embedded in it has effectively been handed to the browser.
        """
        self.dispatcher.search_reference(learning_session(), "keigo")

        for call in self.data_stream.send_tool_event.call_args_list:
            self.assertNotIn(API_KEY, str(call.args[1]))

    def test_an_unconfigured_tool_reports_unavailable_instead_of_raising(self):
        """
        Verify a server with no search key degrades to a status event.

        REQ-18's invariant: a missing optional tool is a smaller session, not a broken
        one. Raising here would abort whatever turn asked for the lookup.
        """
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream, search=GoogleSearchTool(api_key="", cse_id="")
        )

        result = dispatcher.search_reference(learning_session(), "keigo")

        self.assertEqual(result.state, ToolState.UNAVAILABLE)
        self.assertFalse(self.published("reference.card"))
        status = self.published("tool.status")[0]
        self.assertEqual(status["tool"], "search")
        self.assertEqual(status["state"], "unavailable")
        self.assertTrue(status["reason"])

    def test_a_failing_search_reports_failed_and_publishes_no_card(self):
        """Verify a vendor error is announced as failed rather than as an empty card."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            search=GoogleSearchTool(
                api_key=API_KEY, cse_id=CSE_ID,
                transport=RecordingTransport(error=RuntimeError("quota exceeded"))
            )
        )

        result = dispatcher.search_reference(learning_session(), "keigo")

        self.assertEqual(result.state, ToolState.FAILED)
        self.assertFalse(self.published("reference.card"))
        self.assertEqual(self.published("tool.status")[0]["state"], "failed")

    def test_an_agent_initiated_lookup_runs_off_the_calling_thread(self):
        """
        Verify the async form hands back a future rather than a completed result.

        REQ-18 forbids the lookup from blocking the voice fast path, and the caller for
        an agent-initiated search *is* the turn path - so the call has to return before
        the vendor does.
        """
        pending = self.dispatcher.search_reference_async(learning_session(), "一期一会")

        self.assertIsInstance(pending, Future)
        self.assertTrue(pending.result(timeout=5).ok)
        self.assertTrue(self.published("reference.card"))


class TestAgentInitiatedResearch(unittest.TestCase):
    """Test suite for the agent deciding a turn needs a lookup (REQ-18)."""

    def setUp(self):
        """A mock-engine agent, which needs no provider credentials."""
        self.agent = TeachingAgent(engine="mock")

    def test_an_uncertain_cultural_point_produces_a_research_query(self):
        """
        Verify a low-confidence note is what triggers an agent-initiated lookup.

        REQ-18's trigger is "lacks a confident answer" - the model's own uncertainty,
        not every passing noun, which would put the session on a search engine's budget.
        """
        turn_result = {
            "notes": [
                {"type": "culture", "text": "一期一会 may relate to tea ceremony",
                 "confidence": 0.3}
            ]
        }

        query = self.agent.resolve_research_query(turn_result, learning_session())

        self.assertIsNotNone(query)
        self.assertIn("一期一会", query)

    def test_an_explicit_model_research_request_wins(self):
        """Verify the model may ask for a lookup outright."""
        turn_result = {"research_query": "keigo in business email"}

        self.assertEqual(
            self.agent.resolve_research_query(turn_result, learning_session()),
            "keigo in business email"
        )

    def test_a_confident_turn_asks_for_nothing(self):
        """Verify a turn the model answered confidently triggers no search."""
        turn_result = {
            "notes": [{"type": "vocabulary", "text": "ありがとう - thank you",
                       "confidence": 0.97}]
        }

        self.assertIsNone(self.agent.resolve_research_query(turn_result, learning_session()))


class TestResearchOnTheTurnPath(unittest.TestCase):
    """Test suite for the agent-initiated lookup inside a real turn (REQ-18)."""

    def setUp(self):
        """A started session, with the server's search tool replaced by a fake."""
        from src.server import app, server_instance

        self.client = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.original_search = self.server.tools.search
        self.server.data_stream.packet_history.clear()
        self.client.post("/api/session/start", json={
            "channel": "research-channel", "mode": "language_learning",
            "participants": ["Kenji"], "languages": ["ja"]
        })

    def tearDown(self):
        """Restore the server's own tool so tests do not leak configuration."""
        self.server.tools.search = self.original_search

    def published_types(self):
        """Event types broadcast over the data stream so far."""
        return [packet.event_type for packet in self.server.data_stream.packet_history]

    def run_turn(self, turn_result):
        """Files one finalized turn's artifacts and joins any background work."""
        self.server.tools.search = GoogleSearchTool(
            api_key=API_KEY, cse_id=CSE_ID, transport=RecordingTransport()
        )
        self.server.generate_turn_artifacts(
            turn_result, "Kenji", "一期一会って何ですか", "research-channel", "ja"
        )
        self.server.wait_for_convoai_scaffolding(timeout=5)

    def test_an_uncertain_turn_publishes_a_reference_card(self):
        """Verify the turn path actually dispatches the lookup it resolved."""
        self.run_turn({
            "notes": [{"type": "culture", "text": "一期一会 may relate to tea ceremony",
                       "confidence": 0.3}]
        })

        self.assertIn("reference.card", self.published_types())

    def test_a_confident_turn_publishes_no_card(self):
        """Verify a confident turn spends nothing on the search API."""
        self.run_turn({
            "notes": [{"type": "vocabulary", "text": "ありがとう - thank you",
                       "confidence": 0.97}]
        })

        self.assertNotIn("reference.card", self.published_types())


if __name__ == "__main__":
    unittest.main()
