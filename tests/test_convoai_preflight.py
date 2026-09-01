"""
Summary:
    test_convoai_preflight.py is the executable specification for the Custom LLM bridge
    preflight: before a live Convo AI agent is started, the backend verifies that the
    Engine can actually reach `CONVOAI_LLM_BASE_URL`, and refuses with an actionable
    error when it cannot.

    This exists because of a real incident (2026-09-01): the ngrok tunnel named by
    `CONVOAI_LLM_BASE_URL` was down, the agent still joined and reported RUNNING, the
    server log stayed clean, and the learner got silence with no diagnosis anywhere in
    the system. See dev/tasks/task_plans/implementation_plan_bridge_reachability.md.

Covers:
    - A loopback bridge URL is unreachable *by definition* in live mode: the Engine calls
      from Agora's network, where localhost can never resolve to this machine.
    - An unreachable public URL refuses the start, names the URL, and starts no agent.
    - A reachable URL leaves the happy path exactly as it was.
    - Simulated mode never preflights, so offline demos keep working.
    - The escape hatch for deliberate offline work.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.rtc.convoai_client import BridgeCheck, ConvoAIClient
from src.server import app, server_instance


def live_client(llm_base_url="https://tunnel.example.dev"):
    """Builds a client whose credentials put it in live mode."""
    return ConvoAIClient(
        app_id="a" * 32,
        app_certificate="c" * 32,
        customer_id="customer-id",
        customer_secret="customer-secret",
        llm_base_url=llm_base_url
    )


class TestBridgeCheck(unittest.TestCase):
    """Test suite for the reachability probe itself."""

    def test_a_loopback_url_is_never_reachable_for_the_engine(self):
        """
        Verify localhost is rejected without a network call. The Convo AI Engine calls
        the bridge from Agora's infrastructure, so a loopback URL resolves to Agora's own
        machine - a probe from here would succeed and prove nothing.
        """
        for url in ("http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"):
            check = live_client(url).check_llm_bridge()

            self.assertFalse(check.reachable, url)
            self.assertIn("public", check.detail.lower())

    def test_a_reachable_bridge_reports_reachable(self):
        """Verify a 200 from the bridge's own health endpoint passes the check."""
        client = live_client()

        with patch("src.rtc.convoai_client.requests.get") as get:
            get.return_value = MagicMock(status_code=200)
            check = client.check_llm_bridge()

        self.assertTrue(check.reachable)
        self.assertEqual(check.status_code, 200)
        self.assertIn("/health", get.call_args[0][0])

    def test_an_offline_tunnel_reports_unreachable(self):
        """
        Verify ngrok's own 404 page is treated as unreachable. A dead tunnel answers with
        an HTTP response, so 'the request completed' is not the question - 'did the
        EchoSphere backend answer' is.
        """
        client = live_client()

        with patch("src.rtc.convoai_client.requests.get") as get:
            get.return_value = MagicMock(status_code=404)
            check = client.check_llm_bridge()

        self.assertFalse(check.reachable)
        self.assertEqual(check.status_code, 404)

    def test_a_connection_failure_reports_unreachable_with_the_reason(self):
        """Verify a transport error is carried into the detail rather than raising."""
        client = live_client()

        with patch("src.rtc.convoai_client.requests.get", side_effect=OSError("no route")):
            check = client.check_llm_bridge()

        self.assertFalse(check.reachable)
        self.assertIn("no route", check.detail)

    def test_the_check_names_the_url_it_tried(self):
        """Verify the result is actionable on its own, without reading the config."""
        client = live_client("https://stale.example.dev")

        with patch("src.rtc.convoai_client.requests.get", side_effect=OSError("boom")):
            check = client.check_llm_bridge()

        self.assertIsInstance(check, BridgeCheck)
        self.assertEqual(check.url, "https://stale.example.dev")

    def test_the_probe_skips_the_ngrok_browser_interstitial(self):
        """
        Verify the probe sends the header that suppresses ngrok's warning page. Without
        it a free tunnel answers a browser-shaped request with an interstitial, and a
        working bridge would be reported as broken.
        """
        client = live_client()

        with patch("src.rtc.convoai_client.requests.get") as get:
            get.return_value = MagicMock(status_code=200)
            client.check_llm_bridge()

        headers = get.call_args[1]["headers"]
        self.assertIn("ngrok-skip-browser-warning", headers)


class TestStartPreflight(unittest.TestCase):
    """Test suite for the preflight on POST /api/convoai/start."""

    def setUp(self):
        """Use a Flask test client against a clean session registry."""
        self.app = app.test_client()
        server_instance.sessions.reset()
        server_instance.artifacts.reset()

    def start(self, channel="preflight-channel"):
        """Posts a well-formed start request."""
        return self.app.post("/api/convoai/start", json={
            "channel": channel, "language": "en", "mode": "language_learning"
        })

    @staticmethod
    def accepted_agent():
        """
        Stubs a successful Agora /join.

        Needed wherever a test forces live mode: the real client would then POST to
        Agora's API with this suite's mock credentials and get a 401, which says nothing
        about the preflight under test.
        """
        session = MagicMock()
        session.to_dict.return_value = {
            "agent_id": "A44CTEST", "status": "RUNNING", "simulated": False
        }
        return session

    def test_simulated_mode_never_preflights(self):
        """
        Verify the offline demo path is untouched. Without Agora credentials no Engine
        will ever call the bridge, so its reachability is irrelevant - and blocking the
        demo over it would be a fault, not a safeguard.
        """
        with patch.object(server_instance.convoai, "check_llm_bridge") as check:
            response = self.start()

        check.assert_not_called()
        self.assertEqual(response.status_code, 200)

    def test_an_unreachable_bridge_refuses_the_start(self):
        """Verify no agent is spent when the Engine could not possibly call back."""
        unreachable = BridgeCheck(
            url="https://stale.example.dev", reachable=False,
            status_code=404, detail="tunnel offline"
        )

        with patch.object(server_instance.convoai, "is_live_mode", return_value=True), \
             patch.object(server_instance.convoai, "check_llm_bridge", return_value=unreachable), \
             patch.object(server_instance.convoai, "start_agent") as start_agent:
            response = self.start()

        self.assertEqual(response.status_code, 502)
        start_agent.assert_not_called()

    def test_the_refusal_names_the_url_and_the_fix(self):
        """
        Verify the error a learner sees is actionable: which URL failed, and the two
        things that repair it. The whole point of this phase is that the failure explains
        itself instead of presenting as an agent that simply never speaks.
        """
        unreachable = BridgeCheck(
            url="https://stale.example.dev", reachable=False,
            status_code=404, detail="tunnel offline"
        )

        with patch.object(server_instance.convoai, "is_live_mode", return_value=True), \
             patch.object(server_instance.convoai, "check_llm_bridge", return_value=unreachable):
            body = self.start().get_json()

        self.assertFalse(body["success"])
        self.assertIn("stale.example.dev", body["error"])
        self.assertIn("CONVOAI_LLM_BASE_URL", body["error"])
        self.assertEqual(body["bridge"]["reachable"], False)

    def test_a_refused_start_leaves_no_session_behind(self):
        """
        Verify the refusal is clean. Registering a session for an agent that was never
        started would leave the channel holding a mode and a context that no
        conversation belongs to.
        """
        unreachable = BridgeCheck(
            url="https://stale.example.dev", reachable=False, status_code=None, detail="down"
        )

        with patch.object(server_instance.convoai, "is_live_mode", return_value=True), \
             patch.object(server_instance.convoai, "check_llm_bridge", return_value=unreachable):
            self.start()

        self.assertIsNone(server_instance.sessions.get_session("preflight-channel"))

    def test_a_reachable_bridge_starts_the_agent_as_before(self):
        """Verify the happy path is unchanged when the bridge answers."""
        reachable = BridgeCheck(
            url="https://tunnel.example.dev", reachable=True, status_code=200, detail="ok"
        )

        with patch.object(server_instance.convoai, "is_live_mode", return_value=True), \
             patch.object(server_instance.convoai, "check_llm_bridge", return_value=reachable), \
             patch.object(server_instance.convoai, "start_agent",
                          return_value=self.accepted_agent()):
            response = self.start()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])

    def test_the_escape_hatch_skips_the_preflight(self):
        """Verify deliberate offline work can bypass the check (documented, opt-in)."""
        with patch.dict("os.environ", {"CONVOAI_SKIP_BRIDGE_PREFLIGHT": "1"}), \
             patch.object(server_instance.convoai, "is_live_mode", return_value=True), \
             patch.object(server_instance.convoai, "start_agent",
                          return_value=self.accepted_agent()), \
             patch.object(server_instance.convoai, "check_llm_bridge") as check:
            response = self.start()

        check.assert_not_called()
        self.assertEqual(response.status_code, 200)


class TestStaleTunnelDetection(unittest.TestCase):
    """Test suite for the stale-URL warning - the trap that caused the incident."""

    def test_a_running_tunnel_with_a_different_url_is_reported(self):
        """
        Verify a mismatch between ngrok's live URL and the configured one is detected.
        A free ngrok domain changes on restart, so a stale `.env` is the expected steady
        state on a developer machine, not an edge case.
        """
        client = live_client("https://stale.example.dev")
        payload = {"tunnels": [{"public_url": "https://fresh.ngrok-free.dev",
                                "proto": "https"}]}

        with patch("src.rtc.convoai_client.requests.get") as get:
            get.return_value = MagicMock(status_code=200, json=lambda: payload)
            detected = client.detect_local_tunnel_url()

        self.assertEqual(detected, "https://fresh.ngrok-free.dev")

    def test_no_local_tunnel_agent_reports_nothing(self):
        """Verify the absence of ngrok is silent, not an error - it is optional."""
        client = live_client()

        with patch("src.rtc.convoai_client.requests.get", side_effect=OSError("refused")):
            self.assertIsNone(client.detect_local_tunnel_url())


if __name__ == '__main__':
    unittest.main()
