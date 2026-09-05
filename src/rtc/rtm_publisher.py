"""
Summary:
    rtm_publisher.py is the real server-to-browser transport for everything EchoSphere
    generates during a session (REQ-03, D-UIUX-2).

    Before this module, `AgoraVoiceChannelClient.send_data_stream_message` invoked local
    Python callbacks and nothing else: subtitles, idiom cards, quizzes, notes, reference
    cards and tool status were all produced correctly and delivered nowhere, while every
    layer above reported success. Agora's RTC Data Stream cannot be published from this
    backend - that needs an in-channel native SDK, the same Linux-only constraint that
    deferred TASK-11.9's audio path - so delivery goes over Agora's Signaling (RTM)
    channel instead, via its REST API, which needs no SDK and no channel membership.

    The browser side needs no new parsing: the message body is the exact
    `{event_type, payload, timestamp_ms}` envelope `AgoraStreamManager.handleStreamMessage`
    already decodes, so every existing widget keeps working unchanged.

    Verified live before adoption rather than inferred from the docs, which do not
    state it: a message published through this REST endpoint - whose path is Agora's
    legacy `/dev/v2/` Signaling namespace - *is* received by an RTM 2.x SDK subscriber
    (`agora-rtm` v2, what this app's frontend bundles). That was the one unknown that
    decided this design, and a "success" response alone would not have proven it: an
    accepted publish that no client receives is precisely the failure this module was
    written to end.

Key Classes:
    - RtmRestPublisher: publishes one enveloped event to an RTM channel over REST.
"""

import base64
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("echosphere.rtc.rtm_publisher")

# Agora's Signaling REST base. The `/dev/v2/` path is Agora's own versioning of the
# REST surface, not a beta marker; it is the documented endpoint for sending a channel
# message from a server, and its delivery to RTM 2.x subscribers is confirmed live.
RTM_REST_BASE_URL = "https://api.agora.io/dev/v2/project"

# Agora rejects a channel message larger than 32 KB. Checked here so an oversized
# payload fails with a readable local reason instead of an opaque vendor error.
RTM_MESSAGE_MAX_BYTES = 32 * 1024

# The publisher identity these messages arrive under. Deliberately not a participant's
# id: a receiver should be able to tell an event the server generated from one a peer
# sent, and RTM surfaces the publisher on every message.
DEFAULT_SENDER_ID = "echosphere-server"

DEFAULT_TIMEOUT_SECONDS = 5


class RtmRestPublisher:
    """
    Publishes session events into an Agora Signaling (RTM) channel from the server.

    Credentials are the `AGORA_CUSTOMER_ID`/`AGORA_CUSTOMER_SECRET` pair this project
    already uses for Convo AI control-plane calls - this transport adds no new secret.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_secret: Optional[str] = None,
        sender_id: str = DEFAULT_SENDER_ID,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        """
        Initialize the publisher from explicit values or the environment.

        Algorithm:
        1. Resolve the App ID and the RESTful customer credential pair.
        2. Bind the publisher identity and request timeout.
        """
        self.app_id = app_id if app_id is not None else os.getenv("AGORA_APP_ID", "")
        self.customer_id = (
            customer_id if customer_id is not None else os.getenv("AGORA_CUSTOMER_ID", "")
        )
        self.customer_secret = (
            customer_secret if customer_secret is not None
            else os.getenv("AGORA_CUSTOMER_SECRET", "")
        )
        self.sender_id = sender_id
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        """Whether this server holds everything needed to publish."""
        return bool(self.app_id and self.customer_id and self.customer_secret)

    def publish(self, channel: str, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Publishes one event to a channel, reporting whether it was actually delivered.

        Algorithm:
        1. Refuse when unconfigured - reporting success here is the original defect.
        2. Build the `{event_type, payload, timestamp_ms}` envelope the browser parses.
        3. Refuse a message Agora would reject for size, with a readable local reason.
        4. POST it, treating any non-2xx or transport failure as "not delivered".

        Never raises: this runs on the turn path, where an undeliverable subtitle must
        not cost the learner their answer (REQ-20's no-silent-no-op rule applies to the
        *reporting*, not to letting the failure propagate).
        """
        if not self.is_configured:
            logger.warning(
                "Cannot publish '%s': Agora RTM REST credentials are not configured.",
                event_type,
            )
            return False

        body = json.dumps(
            {
                "event_type": event_type,
                "payload": payload,
                "timestamp_ms": int(time.time() * 1000),
            },
            ensure_ascii=False,
        )

        encoded_size = len(body.encode("utf-8"))
        if encoded_size > RTM_MESSAGE_MAX_BYTES:
            logger.warning(
                "Refusing to publish '%s': %d bytes exceeds Agora's %d byte channel "
                "message limit.",
                event_type, encoded_size, RTM_MESSAGE_MAX_BYTES,
            )
            return False

        url = (
            f"{RTM_REST_BASE_URL}/{self.app_id}/rtm/users/{self.sender_id}"
            f"/channel_messages"
        )

        try:
            response = requests.post(
                url,
                json={"channel_name": channel, "payload": body},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._auth_header(),
                },
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never raised into a turn
            logger.warning("Publishing '%s' to channel '%s' failed: %s", event_type, channel, exc)
            return False

        if response.status_code // 100 != 2:
            logger.warning(
                "Agora refused '%s' on channel '%s': HTTP %s %s",
                event_type, channel, response.status_code, response.text[:200],
            )
            return False

        logger.debug("Published '%s' (%d bytes) to channel '%s'.", event_type, encoded_size, channel)
        return True

    def _auth_header(self) -> str:
        """Builds the HTTP Basic value from the RESTful customer credential pair."""
        raw = f"{self.customer_id}:{self.customer_secret}".encode("utf-8")
        return f"Basic {base64.b64encode(raw).decode('utf-8')}"
