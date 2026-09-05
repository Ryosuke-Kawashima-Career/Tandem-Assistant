"""
Summary:
    test_convoai.py provides automated unit and integration tests for EchoSphere's
    Agora Conversational AI (Convo AI) integration.
    It verifies join payload construction against the documented Convo AI field
    contract, tri-lingual ASR configuration for Hindi/Japanese/English, agent session
    lifecycle (start / query / stop), the REST endpoints exposed by server.py, and the
    OpenAI-compatible streaming Custom LLM bridge.

Key Test Classes:
    - TestConvoAIClient: Join payload contract and agent session lifecycle.
    - TestConvoAIEndpoints: Flask REST surface and SSE Custom LLM bridge.
"""

import json
import time
import unittest
from unittest.mock import patch, MagicMock

from src.rtc.convoai_client import (
    ConvoAIClient,
    ConvoAIAgentSession,
    LANGUAGE_PROFILES,
    AGENT_STATUS_RUNNING,
    AGENT_STATUS_STOPPED,
)
from src.rtc.agora_client import is_usable_credential
from src.server import app, server_instance, FALLBACK_REPLIES

# Realistic Agora credential format: 32 hexadecimal characters.
VALID_APP_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
VALID_CERTIFICATE = "0f9e8d7c6b5a49382716f5e4d3c2b1a0"


class TestConvoAIClient(unittest.TestCase):
    """
    Test suite verifying Convo AI join payload construction and session lifecycle.
    """

    def setUp(self):
        """Initialize a ConvoAI client without live credentials (simulated mode)."""
        self.client = ConvoAIClient(
            app_id="mock_app_id",
            app_certificate="mock_certificate",
            llm_base_url="http://localhost:8000",
        )

    def test_simulated_mode_when_credentials_absent(self):
        """
        Verify the client degrades to simulated mode without real Agora credentials,
        matching the mock-first behaviour of AgoraVoiceChannelClient.
        """
        self.assertFalse(self.client.is_live_mode())

    def test_unfilled_env_placeholders_stay_simulated(self):
        """
        Verify placeholder credentials from an uncustomised `.env` do not count as live.

        Copying `.env.example` to `.env` without filling it in leaves non-empty but
        unusable values. Treating those as live makes the browser attempt a join that
        fails with 'invalid vendor key', so they must resolve to simulated mode.
        """
        placeholder_client = ConvoAIClient(
            app_id="your_agora_app_id_here",
            app_certificate="your_agora_app_certificate_here",
            customer_id="your_agora_customer_id_here",
            customer_secret="your_agora_customer_secret_here",
        )
        self.assertFalse(placeholder_client.is_live_mode())

    def test_credential_format_validation(self):
        """
        Verify only 32-character hexadecimal values are accepted as real credentials.
        """
        self.assertTrue(is_usable_credential(VALID_APP_ID))
        self.assertTrue(is_usable_credential(VALID_CERTIFICATE))

        for rejected in [
            None,
            "",
            "mock_app_id",
            "mock_certificate",
            "your_agora_app_id_here",
            "a1b2c3",                              # too short
            "a1b2c3d4e5f60718293a4b5c6d7e8f90ab",  # too long
            "z1b2c3d4e5f60718293a4b5c6d7e8f90",    # non-hexadecimal
        ]:
            self.assertFalse(is_usable_credential(rejected), f"should reject {rejected!r}")

    def test_join_payload_field_contract(self):
        """
        Verify the join payload matches the documented Convo AI field contract.

        Algorithm:
        1. Build a payload for a Japanese tandem session.
        2. Assert agent_rtc_uid is a string and remote_rtc_uids is an array of strings.
        3. Assert the required channel/token/llm/tts blocks are populated.
        """
        payload = self.client.build_join_payload(channel="tokyo-mumbai-101", language="ja")
        props = payload["properties"]

        # Field types that the API rejects when sent as the wrong type
        self.assertIsInstance(props["agent_rtc_uid"], str)
        self.assertIsInstance(props["remote_rtc_uids"], list)
        self.assertTrue(all(isinstance(uid, str) for uid in props["remote_rtc_uids"]))
        self.assertEqual(props["remote_rtc_uids"], ["*"])

        # Required blocks
        self.assertEqual(props["channel"], "tokyo-mumbai-101")
        self.assertTrue(props["token"])
        self.assertTrue(payload["name"])
        self.assertIn("vendor", props["tts"])
        self.assertIn("url", props["llm"])

    def test_microsoft_tts_params_include_key_and_region(self):
        """
        Verify the default 'microsoft' TTS vendor gets `key` and `region` in its
        params object, not just `voice_name`.

        The Convo AI join endpoint requires vendor-specific params (Azure Speech
        needs a resource key + region to authenticate), so shipping only voice_name
        would be rejected or silently misconfigured against a live Agora project.
        """
        with patch.dict("os.environ", {
            "CONVOAI_TTS_KEY": "azure-key-123",
            "CONVOAI_TTS_REGION": "eastus",
            "CONVOAI_TTS_VOICE_JA": "ja-JP-NanamiNeural",
        }):
            payload = self.client.build_join_payload(channel="test-room", language="ja")

        tts_params = payload["properties"]["tts"]["params"]
        self.assertEqual(tts_params["key"], "azure-key-123")
        self.assertEqual(tts_params["region"], "eastus")
        self.assertEqual(tts_params["voice_name"], "ja-JP-NanamiNeural")

    def test_non_microsoft_tts_vendor_only_sends_voice_name(self):
        """Verify a non-default vendor does not get Microsoft-specific key/region."""
        with patch.dict("os.environ", {
            "CONVOAI_TTS_VENDOR": "elevenlabs",
            "CONVOAI_TTS_VOICE_EN": "some-voice-id",
        }):
            payload = self.client.build_join_payload(channel="test-room", language="en")

        tts_params = payload["properties"]["tts"]["params"]
        self.assertNotIn("key", tts_params)
        self.assertNotIn("region", tts_params)
        self.assertEqual(tts_params["voice_name"], "some-voice-id")

    def test_join_payload_targets_custom_llm_bridge(self):
        """
        Verify the llm.url points at EchoSphere's own /chat/completions bridge so the
        TeachingAgent orchestrator is reused rather than an external vendor LLM.
        """
        payload = self.client.build_join_payload(channel="test-room", language="en")
        self.assertEqual(
            payload["properties"]["llm"]["url"],
            "http://localhost:8000/chat/completions"
        )

    def test_tri_lingual_asr_language_mapping(self):
        """
        Verify each in-scope language (hi / ja / en) maps to its ASR language code.
        """
        expected = {"en": "en-US", "ja": "ja-JP", "hi": "hi-IN"}
        for lang, asr_code in expected.items():
            payload = self.client.build_join_payload(channel="test-room", language=lang)
            self.assertEqual(payload["properties"]["asr"]["language"], asr_code)
            self.assertIn(lang, LANGUAGE_PROFILES)

    def test_unknown_language_falls_back_to_english(self):
        """Verify an unsupported language code degrades to the English profile."""
        payload = self.client.build_join_payload(channel="test-room", language="fr")
        self.assertEqual(payload["properties"]["asr"]["language"], "en-US")

    def test_join_payload_configures_end_of_speech_detection(self):
        """
        Verify turn-end detection is configured rather than left at the vendor default
        (REQ-LAT-05).

        The Engine's default is 640ms of silence before it decides the learner has
        stopped talking - spent before the LLM is called at all, so it comes straight
        out of the sub-one-second budget no backend optimisation can recover.
        """
        payload = self.client.build_join_payload(channel="test-room", language="en")
        vad_config = (
            payload["properties"]["turn_detection"]["config"]["end_of_speech"]["vad_config"]
        )

        self.assertLess(vad_config["silence_duration_ms"], 640)
        # Documented range is [120, 2000]; below it the request is rejected.
        self.assertGreaterEqual(vad_config["silence_duration_ms"], 120)

    def test_end_of_speech_silence_is_environment_configurable(self):
        """
        Verify the silence threshold is tunable without a code change (REQ-LAT-05).

        Language learners hesitate mid-sentence far more than native speakers, so the
        speed/clipping trade-off has to be tunable per deployment rather than frozen at
        whatever value suits a demo.
        """
        import os
        with patch.dict(os.environ, {"CONVOAI_END_OF_SPEECH_SILENCE_MS": "300"}):
            payload = self.client.build_join_payload(channel="test-room", language="en")

        vad_config = (
            payload["properties"]["turn_detection"]["config"]["end_of_speech"]["vad_config"]
        )
        self.assertEqual(vad_config["silence_duration_ms"], 300)

    def test_end_of_speech_silence_is_clamped_to_the_documented_range(self):
        """
        Verify an out-of-range value is clamped rather than sent to the API.

        The documented range is [120, 2000]; a typo like 50 would otherwise fail the
        whole /join call, taking down the conversation to save 70ms.
        """
        import os
        with patch.dict(os.environ, {"CONVOAI_END_OF_SPEECH_SILENCE_MS": "10"}):
            low = self.client.build_join_payload(channel="test-room", language="en")
        with patch.dict(os.environ, {"CONVOAI_END_OF_SPEECH_SILENCE_MS": "99999"}):
            high = self.client.build_join_payload(channel="test-room", language="en")

        def silence(payload):
            return (
                payload["properties"]["turn_detection"]["config"]
                ["end_of_speech"]["vad_config"]["silence_duration_ms"]
            )

        self.assertEqual(silence(low), 120)
        self.assertEqual(silence(high), 2000)

    def test_agent_names_are_unique_per_session(self):
        """
        Verify generated agent names differ across sessions, since duplicate names
        are rejected by the Convo AI Engine with HTTP 409.
        """
        names = {self.client._generate_agent_name("same-channel") for _ in range(10)}
        self.assertEqual(len(names), 10)

    def test_agent_session_lifecycle(self):
        """
        Verify start -> query -> stop transitions and session registry bookkeeping.
        """
        session = self.client.start_agent(channel="tokyo-mumbai-101", language="ja")

        self.assertIsInstance(session, ConvoAIAgentSession)
        self.assertTrue(session.agent_id)
        self.assertEqual(session.status, AGENT_STATUS_RUNNING)
        self.assertEqual(session.language, "ja")
        self.assertIn("tokyo-mumbai-101", self.client.active_sessions)

        queried = self.client.query_agent(channel="tokyo-mumbai-101")
        self.assertEqual(queried["agent_id"], session.agent_id)

        self.assertTrue(self.client.stop_agent(channel="tokyo-mumbai-101"))
        self.assertEqual(session.status, AGENT_STATUS_STOPPED)
        self.assertNotIn("tokyo-mumbai-101", self.client.active_sessions)

    def test_stop_unknown_session_returns_false(self):
        """Verify stopping a non-existent agent reports failure instead of raising."""
        self.assertFalse(self.client.stop_agent(channel="never-started"))

    def test_live_mode_posts_join_request(self):
        """
        Verify that with real credentials the client POSTs to the documented /join URL
        and parses agent_id from the response.

        Algorithm:
        1. Construct a client with live credentials.
        2. Patch requests.post to return a stubbed Convo AI response.
        3. Assert the request URL, auth header, and parsed session fields.
        """
        live_client = ConvoAIClient(
            app_id=VALID_APP_ID,
            app_certificate=VALID_CERTIFICATE,
            customer_id="cust_id",
            customer_secret="cust_secret",
        )
        self.assertTrue(live_client.is_live_mode())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "agent-abc-123",
            "create_ts": 1735000000,
            "status": "STARTING",
        }

        with patch("src.rtc.convoai_client.requests.post", return_value=mock_response) as mock_post:
            session = live_client.start_agent(channel="live-room", language="hi")

        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        called_headers = mock_post.call_args[1]["headers"]

        self.assertTrue(called_url.endswith(f"/{VALID_APP_ID}/join"))
        self.assertTrue(called_headers["Authorization"].startswith("Basic "))
        self.assertEqual(session.agent_id, "agent-abc-123")
        self.assertEqual(session.status, "STARTING")
        self.assertFalse(session.simulated)

    def test_live_mode_stop_treats_404_as_already_stopped(self):
        """
        Verify stop_agent() succeeds when Agora returns 404 on /leave, matching a
        real-world sequence: the Convo AI Engine auto-terminates an agent on
        idle_timeout (default 30s) before the client calls stop, so the agent_id is
        already gone. The desired end state (no agent running) is already true, so
        this must not surface as a failure.
        """
        live_client = ConvoAIClient(
            app_id=VALID_APP_ID,
            app_certificate=VALID_CERTIFICATE,
            customer_id="cust_id",
            customer_secret="cust_secret",
        )

        join_response = MagicMock(status_code=200)
        join_response.json.return_value = {"agent_id": "agent-expired-1", "status": "STARTING"}
        with patch("src.rtc.convoai_client.requests.post", return_value=join_response):
            live_client.start_agent(channel="idle-room", language="en")

        leave_404 = MagicMock(status_code=404)
        leave_404.raise_for_status.side_effect = AssertionError(
            "raise_for_status() must not be called on a 404 leave response"
        )
        with patch("src.rtc.convoai_client.requests.post", return_value=leave_404):
            result = live_client.stop_agent(channel="idle-room")

        self.assertTrue(result)
        self.assertNotIn("idle-room", live_client.active_sessions)

    def test_live_mode_stop_still_raises_on_other_errors(self):
        """Verify a non-404 failure (e.g. 500) on /leave still surfaces as an error."""
        import requests as requests_module

        live_client = ConvoAIClient(
            app_id=VALID_APP_ID,
            app_certificate=VALID_CERTIFICATE,
            customer_id="cust_id",
            customer_secret="cust_secret",
        )

        join_response = MagicMock(status_code=200)
        join_response.json.return_value = {"agent_id": "agent-2", "status": "STARTING"}
        with patch("src.rtc.convoai_client.requests.post", return_value=join_response):
            live_client.start_agent(channel="broken-room", language="en")

        leave_500 = MagicMock(status_code=500)
        leave_500.raise_for_status.side_effect = requests_module.HTTPError("500 Server Error")
        with patch("src.rtc.convoai_client.requests.post", return_value=leave_500):
            with self.assertRaises(requests_module.HTTPError):
                live_client.stop_agent(channel="broken-room")

    def test_live_mode_retries_once_on_name_collision(self):
        """
        Verify an HTTP 409 agent-name collision triggers exactly one retry with a
        freshly generated name.
        """
        live_client = ConvoAIClient(
            app_id=VALID_APP_ID,
            app_certificate=VALID_CERTIFICATE,
            customer_id="cust_id",
            customer_secret="cust_secret",
        )

        conflict = MagicMock(status_code=409)
        success = MagicMock(status_code=200)
        success.json.return_value = {"agent_id": "agent-retry-1", "status": "STARTING"}

        with patch("src.rtc.convoai_client.requests.post", side_effect=[conflict, success]) as mock_post:
            session = live_client.start_agent(channel="busy-room", language="en")

        self.assertEqual(mock_post.call_count, 2)
        first_name = mock_post.call_args_list[0][1]["json"]["name"]
        second_name = mock_post.call_args_list[1][1]["json"]["name"]
        self.assertNotEqual(first_name, second_name)
        self.assertEqual(session.agent_id, "agent-retry-1")


class TestConvoAIEndpoints(unittest.TestCase):
    """
    Test suite verifying the Convo AI REST surface and Custom LLM streaming bridge.
    """

    def setUp(self):
        """Initialize the Flask test client and clear any residual agent sessions."""
        self.app = app.test_client()
        server_instance.convoai.active_sessions.clear()
        # Session context now outlives a single request (REQ-LLM-02), so it must be
        # cleared between tests or a language captured by one test leaks into the next.
        server_instance.convoai_session_context.clear()
        server_instance._convoai_last_channel = None
        server_instance.agent.reset_state()

    def tearDown(self):
        """
        Drains any scaffolding future this test scheduled (REQ-LAT-02 test isolation).

        Since scaffolding runs asynchronously on a shared executor, a test that posts to
        /chat/completions without joining can leave a background task running past its
        own return. Left undrained, that task can complete during a *later* test and
        corrupt whichever mock happens to be active then - observed as
        test_broadcast_still_runs_on_ambient_path's send_subtitle mock receiving two
        different, genuinely LLM-generated replies from a test that made one direct call.
        """
        server_instance.wait_for_convoai_scaffolding(timeout=15)

    def test_module_singleton_never_hits_real_agora_api(self):
        """
        Safety invariant: the shared server_instance.convoai must stay in simulated
        mode during the test suite, even on a machine whose real .env holds working
        Agora credentials (tests/conftest.py enforces this).

        If this ever fails, /api/convoai/start in the tests above would silently
        start issuing live POST requests to Agora's REST API on every test run.
        """
        self.assertFalse(server_instance.convoai.is_live_mode())

    def test_convoai_session_start_and_stop(self):
        """
        Verify the /api/convoai/start and /api/convoai/stop lifecycle endpoints.
        """
        start_res = self.app.post("/api/convoai/start", json={"language": "ja", "mode": "language_learning"})
        self.assertEqual(start_res.status_code, 200)
        start_data = start_res.get_json()
        self.assertTrue(start_data["success"])
        self.assertTrue(start_data["agent"]["agent_id"])
        self.assertEqual(start_data["agent"]["language"], "ja")

        stop_res = self.app.post("/api/convoai/stop", json={})
        self.assertEqual(stop_res.status_code, 200)
        self.assertTrue(stop_res.get_json()["success"])

    def test_convoai_status_endpoint(self):
        """
        Verify /api/convoai/status reports the active agent for a channel.
        """
        self.app.post("/api/convoai/start", json={"language": "hi", "mode": "language_learning"})
        status_res = self.app.get("/api/convoai/status")

        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.get_json()
        self.assertTrue(status_data["success"])
        self.assertEqual(status_data["agent"]["language"], "hi")
        self.assertEqual(len(status_data["active_sessions"]), 1)

    def test_stop_without_active_agent_returns_404(self):
        """Verify stopping when no agent is running reports 404 rather than 200."""
        stop_res = self.app.post("/api/convoai/stop", json={})
        self.assertEqual(stop_res.status_code, 404)
        self.assertFalse(stop_res.get_json()["success"])

    def test_rtc_token_endpoint_supplies_browser_credentials(self):
        """
        Verify /api/rtc/token returns the App ID, channel, and token the browser needs
        to join the same channel as the agent, and never leaks the App Certificate.
        """
        res = self.app.get("/api/rtc/token?channel=tokyo-mumbai-101&uid=4242")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["channel"], "tokyo-mumbai-101")
        self.assertEqual(data["uid"], 4242)
        self.assertTrue(data["token"])
        self.assertIn("app_id", data)

        # The App Certificate must never be serialized to the client
        self.assertNotIn("app_certificate", data)
        self.assertNotIn("mock_certificate", res.get_data(as_text=True))

    def test_rtc_token_includes_rtm_credentials(self):
        """
        Verify the token endpoint also returns RTM credentials.

        The Convo AI Engine publishes transcripts over RTM, and the RTM identity must
        equal the value the client logs in with - str(uid) - or RTM auth fails in ways
        that surface as generic conversation-start errors rather than a clear 401.
        """
        res = self.app.get("/api/rtc/token?channel=tokyo-mumbai-101&uid=4242")
        data = res.get_json()

        self.assertTrue(data["rtm_token"])
        self.assertEqual(data["rtm_user_id"], "4242")
        self.assertEqual(data["rtm_user_id"], str(data["uid"]))
        # RTC and RTM tokens are distinct credentials, not the same string reused
        self.assertNotEqual(data["rtm_token"], data["token"])

    def test_rtc_token_includes_a_separate_identity_for_session_events(self):
        """
        Verify the session-event subscription (D-UIUX-2) gets its own RTM identity.

        It must not be `str(uid)`: RTM permits one live login per identity, so reusing
        it would kick the Convo AI transcript client off whenever both the ambient and
        tutor paths are active - which they are meant to be simultaneously.
        """
        res = self.app.get("/api/rtc/token?channel=tokyo-mumbai-101&uid=4242")
        data = res.get_json()

        self.assertTrue(data["events_rtm_token"])
        self.assertTrue(data["events_rtm_user_id"].startswith("4242-events"))
        self.assertNotEqual(data["events_rtm_user_id"], data["rtm_user_id"])
        self.assertNotEqual(data["events_rtm_token"], data["rtm_token"])

    def test_session_event_identity_is_unique_per_request(self):
        """
        Verify two token requests for the same uid get different event identities.

        RTM frees an identity a short while after logout, not immediately: with a
        fixed identity, a participant who left and rejoined inside that window was
        refused with "-10027 user ID already in use" and silently lost live delivery
        for the rest of the session. Found on the first live leave/rejoin test.
        """
        first = self.app.get("/api/rtc/token?channel=c&uid=4242").get_json()
        second = self.app.get("/api/rtc/token?channel=c&uid=4242").get_json()

        self.assertNotEqual(
            first["events_rtm_user_id"], second["events_rtm_user_id"]
        )

    def test_broadcast_skipped_when_backend_rtc_disconnected(self):
        """
        Verify turn payloads are not dispatched when the backend RTC client is not in
        the channel, which is the normal state on the Convo AI path.

        Previously every Convo AI turn logged one warning per payload type. Transcripts
        now reach the browser over RTM, so skipping here is correct rather than lossy.
        """
        server_instance.rtc_client.is_connected = False

        with patch.object(server_instance.data_stream, "send_subtitle") as mock_sub, \
                patch.object(server_instance.data_stream, "send_idiom_card") as mock_card:
            server_instance.process_convoai_turn(
                speaker_id="Learner", text="一期一会ですね！", language="ja"
            )

        mock_sub.assert_not_called()
        mock_card.assert_not_called()

    def test_broadcast_still_runs_on_ambient_path(self):
        """
        Verify the ambient tandem mediation path still broadcasts when the backend RTC
        client IS connected - the skip above must not disable peer-session scaffolding.
        """
        server_instance.rtc_client.is_connected = True
        try:
            with patch.object(server_instance.data_stream, "send_subtitle") as mock_sub:
                server_instance.process_convoai_turn(
                    speaker_id="Learner", text="一期一会ですね！", language="ja"
                )
            mock_sub.assert_called_once()
        finally:
            server_instance.rtc_client.is_connected = False

    def test_rtc_token_flags_simulated_mode(self):
        """
        Verify the endpoint advertises simulated mode when no real App ID is set, so
        the client knows to stay in offline demo mode rather than attempting a join.
        """
        res = self.app.get("/api/rtc/token")
        self.assertTrue(res.get_json()["simulated"])

    def test_rtc_token_rejects_non_integer_uid(self):
        """Verify a malformed uid is rejected with 400 rather than raising a 500."""
        res = self.app.get("/api/rtc/token?uid=not-a-number")
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["success"])

    def test_custom_llm_bridge_streams_sse_contract(self):
        """
        Verify the Custom LLM bridge returns the SSE format the Convo AI Engine
        requires: chat.completion.chunk objects terminated by a [DONE] sentinel.

        Algorithm:
        1. POST an OpenAI-style messages array to /chat/completions.
        2. Assert the response content type is text/event-stream.
        3. Parse the streamed chunks and assert object type and finish_reason.
        4. Assert the terminating [DONE] sentinel is present.
        """
        response = self.app.post(
            "/chat/completions",
            json={
                "model": "echosphere-teaching-agent",
                "stream": True,
                "language": "ja",
                "messages": [
                    {"role": "system", "content": "You are the tandem co-teacher."},
                    {"role": "user", "content": "一期一会ですね！"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.content_type)

        body = response.get_data(as_text=True)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

        chunks = [
            json.loads(line[len("data: "):])
            for line in body.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["choices"][0]["index"], 0)

        # First chunk carries assistant content, final chunk closes with finish_reason
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertTrue(chunks[0]["choices"][0]["delta"]["content"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_custom_llm_bridge_reuses_teaching_agent(self):
        """
        Verify the bridge routes through TeachingAgent (REQ-11) rather than a separate
        reasoning path, by asserting the Japanese idiom scaffolding is produced.

        Since REQ-LAT-02 the scaffolding call runs off the voice-critical path, so this
        joins on it rather than assuming it completed before the response did.
        """
        with patch.object(
            server_instance.agent,
            "process_turn",
            wraps=server_instance.agent.process_turn
        ) as spy:
            self.app.post(
                "/chat/completions",
                json={
                    "language": "ja",
                    "messages": [{"role": "user", "content": "一期一会ですね！"}],
                },
            )
            server_instance.wait_for_convoai_scaffolding(timeout=10)

        spy.assert_called_once()
        self.assertEqual(spy.call_args[1]["text"], "一期一会ですね！")
        self.assertEqual(spy.call_args[1]["detected_language"], "ja")

    def test_session_language_reaches_the_agent_without_a_request_field(self):
        """
        Verify the language chosen at /api/convoai/start reaches process_turn (REQ-LLM-02).

        Agora's Custom LLM request body is OpenAI-shaped and carries no `language` field,
        so a bridge reading data.get("language", "en") always resolved to English - the
        learner's choice never reached the model. The session context must supply it.
        """
        self.app.post("/api/convoai/start", json={"language": "ja", "mode": "language_learning"})

        with patch.object(
            server_instance.agent,
            "process_turn",
            wraps=server_instance.agent.process_turn
        ) as spy:
            self.app.post(
                "/chat/completions",
                json={
                    "model": "echosphere-teaching-agent",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello there"}],
                },
            )
            server_instance.wait_for_convoai_scaffolding(timeout=10)

        spy.assert_called_once()
        self.assertEqual(spy.call_args[1]["detected_language"], "ja")

    def test_request_language_overrides_session_context(self):
        """
        Verify an explicit request field still wins, so direct/manual calls can drive the
        bridge without a live Convo AI session.
        """
        self.app.post("/api/convoai/start", json={"language": "ja", "mode": "language_learning"})

        with patch.object(
            server_instance.agent,
            "process_turn",
            wraps=server_instance.agent.process_turn
        ) as spy:
            self.app.post(
                "/chat/completions",
                json={
                    "language": "hi",
                    "messages": [{"role": "user", "content": "नमस्ते"}],
                },
            )

            server_instance.wait_for_convoai_scaffolding(timeout=10)

        self.assertEqual(spy.call_args[1]["detected_language"], "hi")

    def test_session_speaker_id_reaches_the_agent(self):
        """Verify speaker_id resolves from session context too, not a permanent default."""
        self.app.post("/api/convoai/start", json={"language": "en", "speaker_id": "Kenji", "mode": "language_learning"})

        with patch.object(
            server_instance.agent,
            "process_turn",
            wraps=server_instance.agent.process_turn
        ) as spy:
            self.app.post(
                "/chat/completions",
                json={"messages": [{"role": "user", "content": "Good morning"}]},
            )

            server_instance.wait_for_convoai_scaffolding(timeout=10)

        self.assertEqual(spy.call_args[1]["speaker_id"], "Kenji")

    def test_bridge_runs_the_agent_in_tutor_mode(self):
        """
        Verify the Convo AI path selects the 1:1 tutor prompt mode (REQ-LLM-03) rather
        than the peer-mediation prompt, which addresses a second learner who is absent.
        """
        with patch.object(
            server_instance.agent,
            "process_turn",
            wraps=server_instance.agent.process_turn
        ) as spy:
            self.app.post(
                "/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
            server_instance.wait_for_convoai_scaffolding(timeout=10)

        self.assertEqual(spy.call_args[1]["mode"], "tutor")

    def test_second_session_on_same_channel_starts_from_clean_history(self):
        """
        Verify conversation state does not leak between sessions (REQ-LLM-05).

        server_instance.agent is a single process-wide TeachingAgent whose turn_history
        previously accumulated across every session and channel, so a second learner
        inherited the first learner's dialogue as model context.
        """
        self.app.post("/api/convoai/start", json={"language": "ja", "mode": "language_learning"})
        self.app.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "一期一会ですね！"}]},
        )
        self.assertGreater(len(server_instance.agent.turn_history), 0)

        self.app.post("/api/convoai/stop", json={})
        self.app.post("/api/convoai/start", json={"language": "ja", "mode": "language_learning"})

        self.assertEqual(server_instance.agent.turn_history, [])
        self.assertEqual(server_instance.agent.speaker_durations_ms, {})

    def test_bridge_streams_reply_in_multiple_content_chunks(self):
        """
        Verify reply text is emitted incrementally (REQ-LLM-06) rather than as one blob
        after the whole model response is assembled, so time-to-first-chunk stays low.
        """
        response = self.app.post(
            "/chat/completions",
            json={
                "language": "en",
                "messages": [{"role": "user", "content": "Tell me about festivals"}],
            },
        )

        body = response.get_data(as_text=True)
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        content_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("content")
        ]

        # More than one content delta means text is going out as it becomes available.
        self.assertGreater(len(content_chunks), 1)
        # Concatenated deltas must reconstruct the full reply the engine will speak.
        joined = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        self.assertTrue(joined.strip())
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_bridge_speaks_a_fallback_reply_when_the_engine_stalls(self):
        """
        Verify a slow or failing reasoning engine still produces speech (Task 5.2).

        Silence here is not neutral: the Convo AI Engine terminates an agent on
        idle_timeout (default 30s), so a hung model call ends the whole conversation.

        Targets `stream_convoai_reply` rather than `process_convoai_turn`: since
        REQ-LAT-02 the spoken reply comes from the streaming fast path, and a failure in
        the (now asynchronous) scaffolding call no longer affects what is spoken.
        """
        with patch.object(
            server_instance,
            "stream_convoai_reply",
            side_effect=RuntimeError("provider exploded")
        ):
            response = self.app.post(
                "/chat/completions",
                json={
                    "language": "en",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

        # Reassemble the deltas: the fallback line is split across SSE chunks like
        # any other reply, so it is not findable as one substring in the raw stream.
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        joined = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        self.assertEqual(joined, FALLBACK_REPLIES["en"])

    def test_custom_llm_bridge_handles_content_parts(self):
        """
        Verify the bridge accepts OpenAI content-part arrays as well as plain strings.
        """
        response = self.app.post(
            "/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "नमस्ते"}]}
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("data: [DONE]", response.get_data(as_text=True))


class TestConvoAIObservability(unittest.TestCase):
    """
    Test suite for agent-side observability (REQ-LLM-09 / REQ-LLM-10).

    Exists because a silent agent was indistinguishable from a healthy one in the
    server log: agent errors reached the browser console only, so the operator
    debugging from the terminal could not tell which module (asr / llm / tts) failed.
    """

    def setUp(self):
        """Initialize the Flask test client."""
        app.config["TESTING"] = True
        self.app = app.test_client()

    def test_agent_error_event_is_logged_server_side(self):
        """
        Verify a relayed AGENT_ERROR reaches the server log at ERROR level.

        The failing module is the single most valuable field: it distinguishes a TTS
        vendor rejection from an ASR failure from an LLM bridge problem, which is
        otherwise only inferable by elimination.
        """
        with self.assertLogs("echosphere.server", level="ERROR") as captured:
            res = self.app.post("/api/convoai/event", json={
                "channel": "tokyo-mumbai-101",
                "type": "agent_error",
                "payload": {"module": "tts", "code": 1301, "message": "vendor rejected"},
            })

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])
        joined = " ".join(captured.output)
        self.assertIn("tts", joined)
        self.assertIn("vendor rejected", joined)

    def test_agent_state_event_is_logged_server_side(self):
        """Verify agent state transitions are visible in the terminal (REQ-LLM-09)."""
        with self.assertLogs("echosphere.server", level="INFO") as captured:
            res = self.app.post("/api/convoai/event", json={
                "channel": "tokyo-mumbai-101",
                "type": "agent_state",
                "payload": {"state": "SPEAKING"},
            })

        self.assertEqual(res.status_code, 200)
        self.assertTrue(any("SPEAKING" in line for line in captured.output))

    def test_start_logs_the_agent_status_after_join(self):
        """
        Verify agent lifecycle status is polled and logged after /join (REQ-LLM-09).

        query_agent() already existed and was already exposed at /api/convoai/status,
        but nothing called it during a session, so a FAILED agent looked exactly like a
        healthy one from the terminal.
        """
        with self.assertLogs("echosphere.server", level="INFO") as captured:
            self.app.post("/api/convoai/start", json={"language": "en", "mode": "language_learning"})
            server_instance.log_convoai_agent_health("tokyo-mumbai-101")

        self.assertTrue(
            any("agent health" in line.lower() for line in captured.output),
            f"expected an agent health line, got: {captured.output}"
        )

    def test_agent_health_logging_survives_a_query_failure(self):
        """
        Verify a failing status query does not raise into the caller.

        This runs alongside session start; an exception here would turn a diagnostic
        into an outage.
        """
        with patch.object(
            server_instance.convoai, "query_agent", side_effect=RuntimeError("api down")
        ):
            with self.assertLogs("echosphere.server", level="WARNING"):
                server_instance.log_convoai_agent_health("tokyo-mumbai-101")

    def test_event_endpoint_tolerates_a_malformed_body(self):
        """
        Verify a diagnostic endpoint never becomes a new failure source.

        This is called from a client error handler; raising here would replace the
        original problem with a confusing second one.
        """
        res = self.app.post("/api/convoai/event", json={})
        self.assertEqual(res.status_code, 200)


class TestConvoAILatency(unittest.TestCase):
    """
    Test suite for Convo AI reply latency
    (dev/tasks/task_specs/latency_improvement.md).

    Covers REQ-LAT-01 (per-turn timing logs), REQ-LAT-02 (the spoken reply is not gated
    on scaffolding generation), and REQ-LAT-03 (provider deltas are forwarded as they
    arrive rather than assembled and re-split).
    """

    def setUp(self):
        """Initialize the Flask test client and clear any leftover session context."""
        app.config["TESTING"] = True
        self.app = app.test_client()
        server_instance.agent.reset_state()

    def tearDown(self):
        """Drains any scaffolding future this test scheduled - see TestConvoAIEndpoints.tearDown."""
        server_instance.wait_for_convoai_scaffolding(timeout=15)

    @staticmethod
    def _content_deltas(body: str) -> list:
        """Reassembles the content deltas from a raw SSE body."""
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        return [
            c["choices"][0]["delta"]["content"]
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]

    # -- REQ-LAT-02: the voice reply is not blocked by scaffolding -------------------

    def test_spoken_reply_is_not_blocked_by_slow_scaffolding(self):
        """
        Verify a slow structured-scaffolding call does not delay the spoken reply.

        This is D-LAT-1: the bridge previously computed the whole JSON payload -
        subtitles, idiom card, quiz, teacher alert - before emitting a single SSE chunk,
        so the learner waited out the full generation in silence.
        """
        slow_scaffolding_seconds = 2.0

        def slow_scaffolding(*args, **kwargs):
            time.sleep(slow_scaffolding_seconds)
            return {"spoken_response": "ignored"}

        with patch.object(
            server_instance, "process_convoai_turn", side_effect=slow_scaffolding
        ):
            started = time.time()
            response = self.app.post(
                "/chat/completions",
                json={
                    "language": "en",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            body = response.get_data(as_text=True)
            elapsed = time.time() - started

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        self.assertTrue("".join(self._content_deltas(body)).strip())
        self.assertLess(
            elapsed, slow_scaffolding_seconds,
            "the SSE reply must complete without waiting for scaffolding generation"
        )

    def test_scaffolding_still_reaches_the_data_stream(self):
        """
        Verify decoupling does not silently drop subtitles/idiom cards (criterion 6).

        The scaffolding call moved off the voice-critical path; it must still run and
        still broadcast, just without gating the spoken reply.
        """
        server_instance.rtc_client.is_connected = True
        try:
            with patch.object(server_instance.data_stream, "send_subtitle") as mock_sub:
                self.app.post(
                    "/chat/completions",
                    json={
                        "language": "ja",
                        "messages": [{"role": "user", "content": "一期一会ですね！"}],
                    },
                )
                server_instance.wait_for_convoai_scaffolding(timeout=10)
            mock_sub.assert_called_once()
        finally:
            server_instance.rtc_client.is_connected = False

    def test_scaffolding_failure_is_logged_and_does_not_break_the_reply(self):
        """
        Verify a failing background scaffolding task warns rather than vanishing.

        A fire-and-forget future swallows exceptions by default, which would make a
        broken scaffolding path invisible in the log.
        """
        with patch.object(
            server_instance,
            "process_convoai_turn",
            side_effect=RuntimeError("scaffolding exploded")
        ):
            with self.assertLogs("echosphere.server", level="WARNING") as captured:
                response = self.app.post(
                    "/chat/completions",
                    json={
                        "language": "en",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                body = response.get_data(as_text=True)
                server_instance.wait_for_convoai_scaffolding(timeout=10)

        self.assertEqual(response.status_code, 200)
        self.assertTrue("".join(self._content_deltas(body)).strip())
        self.assertTrue(
            any("scaffolding" in line.lower() for line in captured.output),
            f"expected a scaffolding failure warning, got: {captured.output}"
        )

    def test_learner_turn_is_recorded_exactly_once(self):
        """
        Verify the utterance is not double-recorded now that two calls touch the agent.

        The fast path records the turn; the scaffolding call must not record it again or
        the model would see the learner say everything twice.
        """
        self.app.post("/api/convoai/start", json={"language": "en", "mode": "language_learning"})
        self.app.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "Only once please"}]},
        )
        server_instance.wait_for_convoai_scaffolding(timeout=10)

        matching = [
            t for t in server_instance.agent.turn_history
            if t["text"] == "Only once please"
        ]
        self.assertEqual(len(matching), 1)

    # -- REQ-LAT-03: true token-level streaming ------------------------------------

    def test_bridge_forwards_provider_deltas_as_they_arrive(self):
        """
        Verify provider deltas reach the wire one-for-one (REQ-LAT-03).

        Previously the bridge assembled the whole reply and re-sliced it at fixed width,
        so 'streaming' only paced delivery of an already-finished string.
        """
        provider_deltas = ["Konnichiwa", " Kenji", ", how are you?"]

        with patch.object(
            server_instance, "stream_convoai_reply", return_value=iter(provider_deltas)
        ):
            response = self.app.post(
                "/chat/completions",
                json={
                    "language": "en",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            body = response.get_data(as_text=True)

        self.assertEqual(self._content_deltas(body), provider_deltas)

    def test_bridge_preserves_the_sse_wire_contract_while_streaming(self):
        """
        Verify the Convo AI wire contract is unchanged by the streaming rewrite
        (acceptance criterion 5): assistant role first, finish_reason stop, [DONE] last.
        """
        with patch.object(
            server_instance, "stream_convoai_reply", return_value=iter(["a", "b", "c"])
        ):
            response = self.app.post(
                "/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
            body = response.get_data(as_text=True)

        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.strip().split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "a")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_bridge_falls_back_when_the_first_token_never_arrives(self):
        """
        Verify a stalled provider still produces speech (Task 3.5).

        The timeout now bounds the wait for the *first* token rather than the whole
        blocking call, since the reply is streamed.
        """
        def never_yields():
            time.sleep(30)
            yield "too late"

        with patch.object(
            server_instance, "stream_convoai_reply", return_value=never_yields()
        ):
            with patch("src.server.CONVOAI_LLM_TIMEOUT_SECONDS", 0.3):
                response = self.app.post(
                    "/chat/completions",
                    json={
                        "language": "en",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                body = response.get_data(as_text=True)

        self.assertEqual("".join(self._content_deltas(body)), FALLBACK_REPLIES["en"])
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    # -- REQ-LAT-01: latency instrumentation ---------------------------------------

    def test_bridge_logs_time_to_first_chunk_and_total_duration(self):
        """
        Verify every turn reports its own latency (REQ-LAT-01).

        Without this there is no baseline: 'it feels faster' is not a verification
        strategy, and acceptance criterion 3 is measured from exactly this log line.
        """
        with self.assertLogs("echosphere.server", level="INFO") as captured:
            response = self.app.post(
                "/chat/completions",
                json={
                    "language": "en",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            # The timing line is emitted after the final SSE event, so the streamed body
            # must be drained inside this block for the log to have been written.
            response.get_data(as_text=True)

        timing_lines = [
            line for line in captured.output
            if "first_chunk" in line and "total" in line
        ]
        self.assertTrue(
            timing_lines,
            f"expected a per-turn latency line, got: {captured.output}"
        )


if __name__ == '__main__':
    unittest.main()
