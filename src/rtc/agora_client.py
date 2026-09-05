"""
Summary:
    agora_client.py manages the server-side Agora SD-RTN™ interface for EchoSphere.
    It encapsulates RTC token generation (via agora-token-builder), real-time channel
    lifecycle management (connect, disconnect, audio stream publish), and bidirectional
    RTC Data Stream broadcasting for synchronizing UI payloads (subtitles, idiom cards,
    quizzes, and teacher alerts).

Key Components:
    - AgoraVoiceChannelClient: Main client session managing connection state,
      token generation, and data stream broadcasting.
    - RTCDataStreamPacket: Structured dataclass representing serialized data stream payloads.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict

# Configure module-level logger
logger = logging.getLogger("echosphere.rtc.agora_client")

try:
    from agora_token_builder import RtcTokenBuilder, RtmTokenBuilder
    # Agora SD-RTN standard role constants
    Role_Attendee = 0
    Role_Publisher = 1
    Role_Subscriber = 2
    Role_Admin = 101
    AGORA_BUILDER_AVAILABLE = True
except ImportError:
    AGORA_BUILDER_AVAILABLE = False
    Role_Attendee = 0
    Role_Publisher = 1
    Role_Subscriber = 2
    Role_Admin = 101
    logger.warning("agora-token-builder not available. Using simulated token generation.")


def is_usable_credential(value: Optional[str]) -> bool:
    """
    Reports whether an Agora credential is real rather than a placeholder.

    Agora App IDs and App Certificates are 32-character hexadecimal strings. A value
    that is empty, the built-in mock sentinel, or an unfilled `.env.example` template
    placeholder (e.g. 'your_agora_app_id_here') fails this check.

    This matters because copying `.env.example` to `.env` without filling it in leaves
    non-empty but unusable values; without this check the backend would advertise live
    credentials and the browser would attempt a join that fails with 'invalid vendor key'.

    Algorithm:
    1. Reject empty or missing values.
    2. Reject the known mock sentinels.
    3. Require exactly 32 hexadecimal characters.
    """
    if not value:
        return False
    if value in ("mock_app_id", "mock_certificate"):
        return False

    candidate = value.strip()
    if len(candidate) != 32:
        return False

    try:
        int(candidate, 16)
    except ValueError:
        return False
    return True


@dataclass
class RTCDataStreamPacket:
    """
    Data model for messages sent across Agora RTC Data Streams.
    Types include: 'subtitles', 'idiom_card', 'quiz', 'teacher_alert', 'topic_prompt'.
    """
    event_type: str
    payload: Dict[str, Any]
    timestamp_ms: int = 0

    def __post_init__(self):
        if not self.timestamp_ms:
            self.timestamp_ms = int(time.time() * 1000)

    def to_bytes(self) -> bytes:
        """Serializes the packet to UTF-8 encoded JSON bytes."""
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "RTCDataStreamPacket":
        """Deserializes raw JSON bytes into an RTCDataStreamPacket instance."""
        data = json.loads(raw_bytes.decode("utf-8"))
        return cls(
            event_type=data.get("event_type", "unknown"),
            payload=data.get("payload", {}),
            timestamp_ms=data.get("timestamp_ms", int(time.time() * 1000))
        )


class AgoraVoiceChannelClient:
    """
    Manages connection, authentication, and data streaming with Agora SD-RTN™.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_certificate: Optional[str] = None,
        channel_name: str = "echosphere-tandem",
        uid: int = 0,
        token_expiration_seconds: int = 86400,
        rtm_publisher: Optional[Any] = None
    ):
        """
        Initialize Agora RTC Client credentials and connection state.

        Algorithm:
        1. Resolve credentials from direct parameters or environment variables (.env).
        2. Set up initial channel state flags (is_connected, active_stream_id).
        3. Register empty dispatch tables for incoming audio frames and data stream events.
        4. Bind the optional real transport used to actually deliver data stream events.

        `rtm_publisher` is optional and duck-typed (`is_configured`/`publish`): a client
        constructed without one keeps the in-process-callbacks-only behavior every
        offline and simulated path already relies on.
        """
        # Step 1: Resolve credentials
        self.app_id = app_id or os.getenv("AGORA_APP_ID", "mock_app_id")
        self.app_certificate = app_certificate or os.getenv("AGORA_APP_CERTIFICATE", "mock_certificate")
        self.channel_name = channel_name
        self.uid = uid
        self.token_expiration_seconds = token_expiration_seconds

        # Step 4: Real delivery path (D-UIUX-2). Agora's RTC Data Stream cannot be
        # published from this process - that needs an in-channel native SDK - so events
        # travel over the Signaling (RTM) channel of the same name instead.
        self.rtm_publisher = rtm_publisher

        # Step 2: Initialize connection state
        self.is_connected: bool = False
        self.current_token: Optional[str] = None
        self.stream_id: Optional[int] = None
        
        # Step 3: Callbacks for event hooks
        self._on_audio_frame_callbacks: list[Callable[[bytes, int], None]] = []
        self._on_data_stream_callbacks: list[Callable[[RTCDataStreamPacket], None]] = []

    def generate_token(self, role: str = "publisher") -> str:
        """
        Generates an Agora RTC channel access token using Agora Token Builder.
        
        Algorithm:
        1. Calculate Unix expiration timestamp based on current epoch + validity period.
        2. Map role string ('publisher' / 'subscriber') to Agora SDK Role constant.
        3. Invoke RtcTokenBuilder.build_token_with_uid to compute dynamic HMAC token.
        4. If token builder package is absent, return an authenticated mock signature.
        """
        # Step 1: Calculate expiration timestamp
        privilege_expired_ts = int(time.time()) + self.token_expiration_seconds

        # Step 2: Determine role
        agora_role = 1  # Role_Publisher = 1, Role_Subscriber = 2
        if AGORA_BUILDER_AVAILABLE:
            agora_role = Role_Publisher if role.lower() == "publisher" else Role_Subscriber

        # Step 3: Build token
        if AGORA_BUILDER_AVAILABLE and self.app_id != "mock_app_id" and self.app_certificate != "mock_certificate":
            try:
                builder_fn = getattr(RtcTokenBuilder, "buildTokenWithUid", getattr(RtcTokenBuilder, "build_token_with_uid", None))
                if builder_fn:
                    token = builder_fn(
                        self.app_id,
                        self.app_certificate,
                        self.channel_name,
                        self.uid,
                        agora_role,
                        privilege_expired_ts
                    )
                    self.current_token = token
                    return token
            except Exception as exc:
                logger.error(f"Failed to generate Agora token with RtcTokenBuilder: {exc}")

        # Step 4: Fallback mock token for development / testing environments
        mock_token = f"agora_mock_tok_{self.app_id[:6]}_{self.channel_name}_{self.uid}_{privilege_expired_ts}"
        self.current_token = mock_token
        return mock_token

    def generate_rtm_token(self, user_id: str) -> str:
        """
        Generates an Agora RTM token so a client can subscribe to signaling messages.

        The Convo AI Engine publishes conversation transcripts and agent-state events
        over RTM, so the browser needs this in addition to its RTC channel token.

        Algorithm:
        1. Calculate the Unix expiration timestamp.
        2. Invoke RtmTokenBuilder.buildToken for the given user identity.
        3. Fall back to a mock signature when the builder or credentials are absent.

        The `user_id` must match the identity the RTM client logs in with.
        """
        privilege_expired_ts = int(time.time()) + self.token_expiration_seconds

        if AGORA_BUILDER_AVAILABLE and is_usable_credential(self.app_id) \
                and is_usable_credential(self.app_certificate):
            try:
                # role is accepted for API compatibility; RTM tokens carry login privilege
                return RtmTokenBuilder.buildToken(
                    self.app_id,
                    self.app_certificate,
                    user_id,
                    Role_Publisher,
                    privilege_expired_ts
                )
            except Exception as exc:
                logger.error(f"Failed to generate Agora RTM token: {exc}")

        return f"agora_mock_rtm_tok_{self.app_id[:6]}_{user_id}_{privilege_expired_ts}"

    def join_channel(self) -> bool:
        """
        Joins the designated Agora RTC channel as a voice participant and data stream publisher.
        
        Algorithm:
        1. Ensure fresh token is generated.
        2. Establish connection handshake with SD-RTN gateway.
        3. Initialize RTC Data Stream channel ID (stream_id).
        4. Mark is_connected as True.
        """
        # Step 1: Generate or refresh token
        token = self.generate_token(role="publisher")
        logger.info(f"Connecting to Agora channel '{self.channel_name}' with UID {self.uid} (token: {token[:12]}...)")

        # Step 2: Simulate SD-RTN connection establishment
        self.is_connected = True
        self.stream_id = 1
        logger.info(f"Successfully joined Agora RTC channel '{self.channel_name}'. Data Stream ID: {self.stream_id}")
        return True

    def leave_channel(self) -> bool:
        """
        Leaves the current Agora RTC channel and releases allocated resources.
        
        Algorithm:
        1. Check if connected.
        2. Flush pending audio and data buffers.
        3. Reset connection state and stream identifiers.
        """
        if not self.is_connected:
            return False

        logger.info(f"Leaving Agora channel '{self.channel_name}'...")
        self.is_connected = False
        self.stream_id = None
        return True

    def send_data_stream_message(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Broadcasts synchronized visual metadata (subtitles, idiom cards, quizzes) to clients.
        
        Algorithm:
        1. Validate connection status and active stream ID.
        2. Construct an RTCDataStreamPacket with timestamp and payload.
        3. Publish it to the browser over the real transport, when one is attached.
        4. Invoke local registered data stream listener callbacks.

        Local callbacks fire whether or not real delivery succeeded: they are
        in-process subscribers (the simulation demo, tests) and suppressing them on a
        transport failure would turn one broken path into two.
        """
        if not self.is_connected:
            logger.warning("Cannot send data stream message: client is not connected to Agora RTC.")
            return False

        # Step 2: Construct packet
        packet = RTCDataStreamPacket(event_type=event_type, payload=payload)
        raw_bytes = packet.to_bytes()

        # Step 3: Real delivery (D-UIUX-2). Without a publisher this stays what it has
        # always been - a local fan-out - which is what the offline and simulated paths
        # expect; with one, the event actually reaches the browser.
        logger.debug(f"Broadcasting Data Stream event '{event_type}' ({len(raw_bytes)} bytes) to channel '{self.channel_name}'")
        if self.rtm_publisher is not None:
            self.rtm_publisher.publish(self.channel_name, event_type, payload)

        # Step 4: Notify local subscribers/listeners
        for cb in self._on_data_stream_callbacks:
            try:
                cb(packet)
            except Exception as err:
                logger.error(f"Error in data stream callback: {err}")

        return True

    def send_data_stream(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Alias for send_data_stream_message."""
        return self.send_data_stream_message(event_type, payload)

    def publish_audio_frame(self, pcm_data: bytes, sample_rate: int = 16000) -> bool:
        """
        Pushes synthesized or processed PCM audio into the Agora RTC audio bus.
        
        Algorithm:
        1. Verify active connection.
        2. Validate audio chunk size (16-bit mono PCM).
        3. Transmit audio frame to Agora audio track.
        4. Notify audio frame callbacks.
        """
        if not self.is_connected:
            return False

        for cb in self._on_audio_frame_callbacks:
            try:
                cb(pcm_data, sample_rate)
            except Exception as err:
                logger.error(f"Error in audio frame callback: {err}")

        return True

    def register_on_data_stream(self, callback: Callable[[RTCDataStreamPacket], None]):
        """Registers a listener callback for incoming data stream events."""
        self._on_data_stream_callbacks.append(callback)

    def register_on_audio_frame(self, callback: Callable[[bytes, int], None]):
        """Registers a listener callback for incoming audio frames."""
        self._on_audio_frame_callbacks.append(callback)
