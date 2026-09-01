"""
Summary:
    convoai_client.py manages the server-side interface to the Agora Conversational AI
    (Convo AI) Engine for EchoSphere.
    It encapsulates agent session lifecycle over the Convo AI REST API (start / stop /
    query), builds the tri-lingual ASR and TTS join configuration for Hindi, Japanese,
    and English, and wires the agent's LLM backend to EchoSphere's own Custom LLM bridge
    so the TeachingAgent scaffolding logic is reused rather than duplicated.

    This satisfies REQ-09 (Direct AI Audio Conversation) and REQ-10 (Convo AI Session
    Management) from dev/specs/spec_tandem.md section 4.

Key Components:
    - ConvoAIAgentSession: Structured dataclass describing one running agent session.
    - ConvoAIClient: REST client managing Convo AI agent join/leave/query lifecycle.
    - LANGUAGE_PROFILES: Tri-lingual ASR language mapping for `hi`, `ja`, and `en`.
"""

import os
import uuid
import base64
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict

from src.rtc.agora_client import AgoraVoiceChannelClient, is_usable_credential

# Configure module-level logger
logger = logging.getLogger("echosphere.rtc.convoai_client")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. Convo AI client will run in simulated mode.")


# Agora Conversational AI Engine REST base URL.
CONVOAI_BASE_URL = "https://api.agora.io/api/conversational-ai-agent/v2/projects"

# Agent lifecycle states reported by the Convo AI Engine (spec_tandem.md 4.3).
AGENT_STATUS_IDLE = "IDLE"
AGENT_STATUS_STARTING = "STARTING"
AGENT_STATUS_RUNNING = "RUNNING"
AGENT_STATUS_STOPPING = "STOPPING"
AGENT_STATUS_STOPPED = "STOPPED"
AGENT_STATUS_RECOVERING = "RECOVERING"
AGENT_STATUS_FAILED = "FAILED"

# Tri-language scope from spec_tandem.md 1.2 / 4.4.
# Only the ASR *language codes* are pinned here: they are stable BCP-47 identifiers.
# Vendor and voice names are deliberately NOT hardcoded - they change independently of
# this codebase and are resolved from environment configuration per REQ-08.
# Turn-end detection bounds documented by the Convo AI join schema (REQ-LAT-05).
# Values outside [120, 2000] are rejected by the API, so they are clamped rather than
# sent: failing the whole /join call to save a few milliseconds is a bad trade.
END_OF_SPEECH_SILENCE_MIN_MS = 120
END_OF_SPEECH_SILENCE_MAX_MS = 2000

# The Engine's own default is 640ms. That silence is spent *before* the Custom LLM
# bridge is called at all, so it is charged directly against the sub-one-second reply
# budget and no amount of backend streaming can win it back.
#
# 480ms is a deliberate compromise, not the fastest setting available. Language learners
# hesitate mid-sentence far more than native speakers do - dropping to the 120ms floor
# would cut them off mid-thought, which is a worse product than a slightly slower reply.
# Tune per deployment via CONVOAI_END_OF_SPEECH_SILENCE_MS.
DEFAULT_END_OF_SPEECH_SILENCE_MS = 480

LANGUAGE_PROFILES: Dict[str, Dict[str, str]] = {
    "en": {"asr_language": "en-US", "voice_env": "CONVOAI_TTS_VOICE_EN", "label": "English"},
    "ja": {"asr_language": "ja-JP", "voice_env": "CONVOAI_TTS_VOICE_JA", "label": "Japanese"},
    "hi": {"asr_language": "hi-IN", "voice_env": "CONVOAI_TTS_VOICE_HI", "label": "Hindi"},
}


@dataclass
class ConvoAIAgentSession:
    """
    Data model describing a single running Convo AI agent session.
    Mirrors the fields returned by the Convo AI `/join` endpoint.
    """
    agent_id: str
    channel: str
    language: str = "en"
    status: str = AGENT_STATUS_STARTING
    create_ts: int = 0
    agent_name: str = ""
    simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the session into a JSON-safe dictionary for REST responses."""
        return asdict(self)


# Hosts that can never be reached from Agora's network. The Convo AI Engine calls the
# Custom LLM bridge from Agora's own infrastructure, so a loopback URL there resolves to
# Agora's machine, not to this one - a probe from here would succeed and prove nothing.
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")

# ngrok's local agent exposes the tunnel it is currently serving. Used to tell a *stale*
# CONVOAI_LLM_BASE_URL from a missing tunnel: a free ngrok domain changes on restart, so
# a stale value in .env is the expected steady state on a developer machine.
NGROK_LOCAL_API = "http://127.0.0.1:4040/api/tunnels"

# How long to wait for the bridge to answer. Short on purpose: this runs in front of a
# button press, and a bridge that needs more than a couple of seconds to serve /health
# will not survive a live turn either.
BRIDGE_CHECK_TIMEOUT_SECONDS = 3.0


@dataclass
class BridgeCheck:
    """
    The result of probing the Custom LLM bridge (REQ-11).

    Carries the URL it tried so the outcome is actionable on its own - a caller
    reporting this to a person should never have to re-read the configuration to say
    which address failed.
    """

    url: str
    reachable: bool
    status_code: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the check for API responses and logs."""
        return {
            "url": self.url,
            "reachable": self.reachable,
            "status_code": self.status_code,
            "detail": self.detail,
        }


class ConvoAIClient:
    """
    Manages Agora Conversational AI agent sessions over the Convo AI REST API.

    The client is credential-aware in the same way as AgoraVoiceChannelClient: when real
    Agora credentials are absent it degrades to a simulated session so the tandem
    classroom remains runnable in local development and test environments.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_certificate: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_secret: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        base_url: str = CONVOAI_BASE_URL,
        request_timeout_seconds: int = 15,
    ):
        """
        Initialize Convo AI credentials, LLM bridge target, and session registry.

        Algorithm:
        1. Resolve Agora project credentials from parameters or environment variables.
        2. Resolve RESTful Customer ID/Secret used for Basic authentication.
        3. Resolve the publicly reachable base URL of EchoSphere's Custom LLM bridge.
        4. Initialize the in-memory registry of active agent sessions keyed by channel.
        """
        # Step 1: Agora project credentials
        self.app_id = app_id or os.getenv("AGORA_APP_ID", "mock_app_id")
        self.app_certificate = app_certificate or os.getenv("AGORA_APP_CERTIFICATE", "mock_certificate")

        # Step 2: RESTful API credentials
        self.customer_id = customer_id or os.getenv("AGORA_CUSTOMER_ID", "")
        self.customer_secret = customer_secret or os.getenv("AGORA_CUSTOMER_SECRET", "")

        # Step 3: Custom LLM bridge (Task 7.2) that the Convo AI Engine calls per turn
        self.llm_base_url = llm_base_url or os.getenv("CONVOAI_LLM_BASE_URL", "http://localhost:8000")

        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds

        # Step 4: Active session registry
        self.active_sessions: Dict[str, ConvoAIAgentSession] = {}

        logger.info(
            f"ConvoAIClient initialized (app_id: {self.app_id[:6]}..., "
            f"llm_bridge: {self.llm_base_url}, live_mode: {self.is_live_mode()})"
        )

    def is_live_mode(self) -> bool:
        """
        Determines whether real REST calls should be issued to the Convo AI Engine.

        Live mode requires the requests library, a real (non-placeholder) App ID, and
        RESTful API credentials. Anything less falls back to a simulated session.
        """
        return bool(
            REQUESTS_AVAILABLE
            and is_usable_credential(self.app_id)
            and self.customer_id
            and self.customer_secret
            and not self.customer_id.startswith("your_")
            and not self.customer_secret.startswith("your_")
        )

    def check_llm_bridge(
        self,
        timeout: float = BRIDGE_CHECK_TIMEOUT_SECONDS
    ) -> BridgeCheck:
        """
        Probes whether the Convo AI Engine could reach this backend's Custom LLM bridge.

        Algorithm:
        1. Reject a loopback URL outright - it is unreachable from Agora's network by
           definition, and no probe from this machine can demonstrate otherwise.
        2. GET `<llm_base_url>/health`, suppressing ngrok's browser interstitial.
        3. Treat any non-200 as unreachable: a dead tunnel still answers with an HTTP
           response (ngrok serves its own 404 page), so "the request completed" is not
           the question - "did the EchoSphere backend answer" is.

        Never raises: the caller is deciding whether to start an agent, and a probe that
        throws would take down the very path it exists to protect.
        """
        url = self.llm_base_url.rstrip("/")
        host = (urlparse(url).hostname or "").lower()

        if host in LOOPBACK_HOSTS:
            return BridgeCheck(
                url=url,
                reachable=False,
                detail=(
                    f"'{url}' is a loopback address. The Convo AI Engine calls this "
                    f"backend from Agora's network, so the bridge must be a public URL "
                    f"(for example an ngrok tunnel)."
                )
            )

        if not REQUESTS_AVAILABLE:
            return BridgeCheck(
                url=url, reachable=False,
                detail="The 'requests' package is unavailable, so the bridge cannot be probed."
            )

        try:
            response = requests.get(
                f"{url}/health",
                timeout=timeout,
                headers={"ngrok-skip-browser-warning": "1"}
            )
        except Exception as exc:
            return BridgeCheck(
                url=url, reachable=False,
                detail=f"{type(exc).__name__}: {exc}"
            )

        if response.status_code != 200:
            return BridgeCheck(
                url=url, reachable=False, status_code=response.status_code,
                detail=(
                    f"{url}/health answered {response.status_code}. The address is "
                    f"resolving to something other than this backend - most often a "
                    f"tunnel that is no longer running."
                )
            )

        return BridgeCheck(
            url=url, reachable=True, status_code=200,
            detail="The Custom LLM bridge is publicly reachable."
        )

    def detect_local_tunnel_url(self, timeout: float = 1.0) -> Optional[str]:
        """
        Returns the public URL of a tunnel running locally, or None.

        Used to distinguish a *stale* `CONVOAI_LLM_BASE_URL` from a missing tunnel, which
        are the same symptom with different fixes. Silent when no local agent is running:
        the tunnel is optional infrastructure, not a dependency.
        """
        if not REQUESTS_AVAILABLE:
            return None

        try:
            response = requests.get(NGROK_LOCAL_API, timeout=timeout)
            if response.status_code != 200:
                return None
            tunnels = response.json().get("tunnels") or []
        except Exception:
            return None

        for tunnel in tunnels:
            public_url = tunnel.get("public_url", "")
            if public_url.startswith("https://"):
                return public_url

        return tunnels[0].get("public_url") if tunnels else None

    def describe_bridge_problem(self, check: BridgeCheck) -> str:
        """
        Turns a failed check into the message a person can act on.

        Names the address that failed, what was observed, and the two repairs - start the
        tunnel, or point the configuration at the one that is already running. The second
        is included only when a local tunnel really is serving a different URL, so the
        advice is never speculative.
        """
        lines = [
            f"The Convo AI Engine cannot reach this backend's Custom LLM bridge at "
            f"{check.url}. {check.detail}"
        ]

        live_url = self.detect_local_tunnel_url()
        if live_url and live_url.rstrip("/") != check.url.rstrip("/"):
            lines.append(
                f"A tunnel is running locally on {live_url} - set "
                f"CONVOAI_LLM_BASE_URL to that value and restart the server."
            )
        else:
            lines.append(
                "Start the tunnel (ngrok http 8000) and set CONVOAI_LLM_BASE_URL to its "
                "public URL, then restart the server."
            )

        lines.append(
            "Set CONVOAI_SKIP_BRIDGE_PREFLIGHT=1 to start an agent anyway; it will join "
            "the channel but stay silent."
        )
        return " ".join(lines)

    def _auth_header(self) -> str:
        """
        Builds the Authorization header value for Convo AI REST requests.

        Algorithm:
        1. Concatenate Customer ID and Customer Secret.
        2. Base64-encode the credential pair.
        3. Return an HTTP Basic Authorization value.
        """
        raw = f"{self.customer_id}:{self.customer_secret}".encode("utf-8")
        return f"Basic {base64.b64encode(raw).decode('utf-8')}"

    def _generate_agent_name(self, channel: str) -> str:
        """
        Builds a per-session agent name that is unique within the Agora project.

        Agent names collide with HTTP 409 when reused, so a short UUID suffix is
        appended to the channel name for every start attempt (spec_tandem.md 4.3).
        """
        return f"echosphere_{channel}_{uuid.uuid4().hex[:8]}"

    def _generate_channel_token(self, channel: str, agent_uid: int = 0) -> str:
        """
        Mints the RTC channel token the Convo AI agent uses to join the voice channel.

        Reuses AgoraVoiceChannelClient.generate_token so token construction lives in
        exactly one place across the backend.
        """
        token_client = AgoraVoiceChannelClient(
            app_id=self.app_id,
            app_certificate=self.app_certificate,
            channel_name=channel,
            uid=agent_uid,
        )
        return token_client.generate_token(role="publisher")

    def _build_tts_params(self, vendor: str, voice_name: str) -> Dict[str, Any]:
        """
        Builds the `tts.params` object for the selected TTS vendor.

        Each vendor has a different required params shape (Microsoft Azure needs a
        resource key + region, others do not); this only implements the default
        `microsoft` vendor since that is what CONVOAI_TTS_VENDOR defaults to. Adding
        another vendor means adding its params shape here after checking the current
        Agora TTS vendor docs.
        """
        if vendor == "microsoft":
            params: Dict[str, Any] = {
                "key": os.getenv("CONVOAI_TTS_KEY", ""),
                "region": os.getenv("CONVOAI_TTS_REGION", ""),
            }
            if voice_name:
                params["voice_name"] = voice_name
            return params

        return {"voice_name": voice_name} if voice_name else {}

    def _resolve_end_of_speech_silence_ms(self) -> int:
        """
        Resolves the end-of-utterance silence threshold, clamped to the documented range.

        Algorithm:
        1. Read CONVOAI_END_OF_SPEECH_SILENCE_MS, falling back to the tuned default.
        2. Treat an unparseable value as the default rather than raising - a typo in
           `.env` must not stop an agent from starting.
        3. Clamp into [120, 2000], the range the join schema accepts.
        """
        raw = os.getenv("CONVOAI_END_OF_SPEECH_SILENCE_MS", "")
        try:
            value = int(raw) if raw else DEFAULT_END_OF_SPEECH_SILENCE_MS
        except ValueError:
            logger.warning(
                f"CONVOAI_END_OF_SPEECH_SILENCE_MS={raw!r} is not an integer. "
                f"Using the default {DEFAULT_END_OF_SPEECH_SILENCE_MS}ms."
            )
            value = DEFAULT_END_OF_SPEECH_SILENCE_MS

        clamped = max(END_OF_SPEECH_SILENCE_MIN_MS, min(END_OF_SPEECH_SILENCE_MAX_MS, value))
        if clamped != value:
            logger.warning(
                f"CONVOAI_END_OF_SPEECH_SILENCE_MS={value} is outside the documented "
                f"range [{END_OF_SPEECH_SILENCE_MIN_MS}, {END_OF_SPEECH_SILENCE_MAX_MS}]; "
                f"clamped to {clamped}ms."
            )
        return clamped

    def build_join_payload(
        self,
        channel: str,
        language: str = "en",
        agent_name: Optional[str] = None,
        agent_rtc_uid: str = "0",
        greeting: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Constructs the Convo AI `/join` request body for a tri-lingual tandem session.

        Algorithm:
        1. Resolve the language profile (ASR language code and TTS voice env key).
        2. Mint the RTC channel token for the agent participant.
        3. Point the `llm` block at EchoSphere's Custom LLM bridge (Task 7.2).
        4. Resolve ASR/TTS vendor and voice identifiers from environment config.
        5. Assemble the documented `properties` object.

        Field types follow the Convo AI API contract exactly:
        `agent_rtc_uid` is a string, and `remote_rtc_uids` is an array of strings.
        """
        # Step 1: Language profile
        profile = LANGUAGE_PROFILES.get(language, LANGUAGE_PROFILES["en"])

        # Step 2: Channel token for the agent
        token = self._generate_channel_token(channel)

        # Step 3 & 4: Bridge target and vendor configuration
        llm_url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
        tts_vendor = os.getenv("CONVOAI_TTS_VENDOR", "microsoft")
        tts_voice = os.getenv(profile["voice_env"], "")
        asr_vendor = os.getenv("CONVOAI_ASR_VENDOR", "ares")
        tts_params = self._build_tts_params(tts_vendor, tts_voice)

        default_greeting = (
            f"Hello! I am your EchoSphere tandem co-teacher. "
            f"Let's practice {profile['label']} together."
        )
        default_system_prompt = (
            f"You are the EchoSphere tandem co-teacher. Converse naturally in "
            f"{profile['label']} with a language learner, keep replies short and "
            f"encouraging, and gently correct mistakes."
        )

        # Step 5: Assemble documented properties object
        payload: Dict[str, Any] = {
            "name": agent_name or self._generate_agent_name(channel),
            "properties": {
                "channel": channel,
                "token": token,
                "agent_rtc_uid": agent_rtc_uid,
                "remote_rtc_uids": ["*"],
                "idle_timeout": 30,
                "asr": {
                    "language": profile["asr_language"],
                    "vendor": asr_vendor,
                    "params": {},
                },
                # REQ-LAT-05. This is the first thing in the whole reply path: the Engine
                # waits this long in silence before it even decides the learner has
                # finished, so it is charged against the reply budget before any of this
                # backend's streaming work begins.
                "turn_detection": {
                    "config": {
                        "end_of_speech": {
                            "mode": "vad",
                            "vad_config": {
                                "silence_duration_ms": self._resolve_end_of_speech_silence_ms(),
                            },
                        },
                    },
                },
                "llm": {
                    "url": llm_url,
                    "system_messages": [
                        {"role": "system", "content": system_prompt or default_system_prompt}
                    ],
                    "greeting_message": greeting or default_greeting,
                    "max_history": 32,
                    "params": {"model": "echosphere-teaching-agent"},
                },
                "tts": {
                    "vendor": tts_vendor,
                    "params": tts_params,
                },
                "advanced_features": {
                    "enable_rtm": True,
                },
                "parameters": {
                    "data_channel": "rtm",
                    "enable_metrics": True,
                    "enable_error_message": True,
                },
            },
        }
        return payload

    def start_agent(
        self,
        channel: str,
        language: str = "en",
        greeting: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ConvoAIAgentSession:
        """
        Starts a Convo AI agent that joins the RTC channel and converses by audio.

        Algorithm:
        1. Build the documented `/join` payload for the requested language.
        2. POST to the Convo AI Engine when live credentials are configured.
        3. On HTTP 409 (duplicate agent name), retry once with a freshly generated name.
        4. Register and return the resulting ConvoAIAgentSession.

        A successful response means the request was accepted, not that the agent is
        already audible: clients must wait for the RTC `user-joined` event before
        expecting agent audio (spec_tandem.md 4.3).
        """
        agent_name = self._generate_agent_name(channel)
        payload = self.build_join_payload(
            channel=channel,
            language=language,
            agent_name=agent_name,
            greeting=greeting,
            system_prompt=system_prompt,
        )

        # Step 2: Simulated path for local development and tests
        if not self.is_live_mode():
            session = ConvoAIAgentSession(
                agent_id=f"sim_agent_{uuid.uuid4().hex[:12]}",
                channel=channel,
                language=language,
                status=AGENT_STATUS_RUNNING,
                agent_name=agent_name,
                simulated=True,
            )
            self.active_sessions[channel] = session
            logger.info(f"[SIMULATED] Convo AI agent '{agent_name}' joined channel '{channel}' ({language}).")
            return session

        url = f"{self.base_url}/{self.app_id}/join"
        headers = {"Content-Type": "application/json", "Authorization": self._auth_header()}

        response = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout_seconds)

        # Step 3: Retry once on agent name collision with a freshly built payload.
        # A new payload object is built rather than mutating the original so the
        # request body sent on each attempt stays independent.
        if response.status_code == 409:
            logger.warning(f"Agent name '{agent_name}' collided (HTTP 409). Retrying with a new name.")
            agent_name = self._generate_agent_name(channel)
            payload = self.build_join_payload(
                channel=channel,
                language=language,
                agent_name=agent_name,
                greeting=greeting,
                system_prompt=system_prompt,
            )
            response = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout_seconds)

        response.raise_for_status()
        data = response.json()

        # Step 4: Register session
        session = ConvoAIAgentSession(
            agent_id=data.get("agent_id", ""),
            channel=channel,
            language=language,
            status=data.get("status", AGENT_STATUS_STARTING),
            create_ts=data.get("create_ts", 0),
            agent_name=payload["name"],
            simulated=False,
        )
        self.active_sessions[channel] = session
        logger.info(f"Convo AI agent '{session.agent_id}' starting on channel '{channel}' ({language}).")
        return session

    def stop_agent(self, agent_id: Optional[str] = None, channel: Optional[str] = None) -> bool:
        """
        Stops a running Convo AI agent and removes it from the session registry.

        Algorithm:
        1. Resolve the target session by agent_id or by channel.
        2. POST to `/agents/{agent_id}/leave` when live credentials are configured.
        3. Treat a 404 as "already stopped" rather than a failure - the Convo AI
           Engine auto-terminates an agent on `idle_timeout` (default 30s) with no
           detected activity, so the agent can legitimately be gone before the
           client ever clicks Stop. The desired end state (no agent running) is
           already achieved in that case.
        4. Deregister the session and report success.
        """
        session = self._resolve_session(agent_id=agent_id, channel=channel)
        if session is None:
            logger.warning(f"No active Convo AI session found (agent_id={agent_id}, channel={channel}).")
            return False

        if session.simulated or not self.is_live_mode():
            session.status = AGENT_STATUS_STOPPED
            self.active_sessions.pop(session.channel, None)
            logger.info(f"[SIMULATED] Convo AI agent '{session.agent_id}' left channel '{session.channel}'.")
            return True

        url = f"{self.base_url}/{self.app_id}/agents/{session.agent_id}/leave"
        headers = {"Content-Type": "application/json", "Authorization": self._auth_header()}

        response = requests.post(url, json={}, headers=headers, timeout=self.request_timeout_seconds)

        # Step 3: A 404 means the Engine already removed the agent (e.g. idle_timeout
        # expiry) - the caller wanted it gone, and it already is.
        if response.status_code != 404:
            response.raise_for_status()
        else:
            logger.info(
                f"Convo AI agent '{session.agent_id}' was already gone on the Engine "
                f"(likely idle_timeout expiry) - treating as stopped."
            )

        session.status = AGENT_STATUS_STOPPED
        self.active_sessions.pop(session.channel, None)
        logger.info(f"Convo AI agent '{session.agent_id}' stopped on channel '{session.channel}'.")
        return True

    def query_agent(self, agent_id: Optional[str] = None, channel: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Queries the current lifecycle status of a Convo AI agent session.

        Algorithm:
        1. Resolve the target session by agent_id or by channel.
        2. GET `/agents/{agent_id}` when live credentials are configured.
        3. Refresh and return the cached session state.
        """
        session = self._resolve_session(agent_id=agent_id, channel=channel)
        if session is None:
            return None

        if session.simulated or not self.is_live_mode():
            return session.to_dict()

        url = f"{self.base_url}/{self.app_id}/agents/{session.agent_id}"
        headers = {"Authorization": self._auth_header()}

        response = requests.get(url, headers=headers, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        data = response.json()

        session.status = data.get("status", session.status)
        return session.to_dict()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all locally tracked Convo AI agent sessions."""
        return [s.to_dict() for s in self.active_sessions.values()]

    def _resolve_session(
        self,
        agent_id: Optional[str] = None,
        channel: Optional[str] = None
    ) -> Optional[ConvoAIAgentSession]:
        """
        Looks up an active session by channel name first, then by agent identifier.
        """
        if channel and channel in self.active_sessions:
            return self.active_sessions[channel]
        if agent_id:
            for session in self.active_sessions.values():
                if session.agent_id == agent_id:
                    return session
        return None
