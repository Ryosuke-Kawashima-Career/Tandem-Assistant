"""
Summary:
    email.py sends the supplementary confirmation mail for a scheduled meeting (REQ-20),
    over Resend.

    Supplementary is the whole contract: the invitation itself is the Google Calendar
    invite, and this message is a copy for people who read mail faster than they read
    their calendar. REQ-20 forbids it from ever being the only channel, so the dispatcher
    sends it after Calendar has accepted the event and never instead of it - and a
    bounced copy leaves the booked meeting booked.

Key Classes:
    - ResendEmailSender: the configured Resend client.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    call_transport,
    http_post_json,
)

logger = logging.getLogger("echosphere.agent.tools.email")

DEFAULT_ENDPOINT = "https://api.resend.com/emails"


class ResendEmailSender:
    """
    Sends transactional mail for the agent's tools (REQ-20, supplementary channel only).
    """

    name = "email"

    def __init__(
        self,
        api_key: Optional[str] = None,
        sender: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None,
        endpoint: str = DEFAULT_ENDPOINT
    ):
        """
        Initialize the sender from explicit values or the environment.

        `RESEND_FROM_EMAIL` falls back to `REPORT_EMAIL`: the second is the address this
        deployment already records as its own, and Resend rejects a `from` outside a
        verified domain, so guessing a different one only produces a 403 at send time.
        """
        self.api_key = api_key if api_key is not None else os.getenv("RESEND_API_KEY", "")
        self.sender = (
            sender if sender is not None
            else (os.getenv("RESEND_FROM_EMAIL") or os.getenv("REPORT_EMAIL", ""))
        ).strip().strip('"')
        self.endpoint = endpoint
        self._transport = transport or http_post_json

    @property
    def is_configured(self) -> bool:
        """Whether this server can send mail at all."""
        return bool(self.api_key and self.sender)

    def send(
        self,
        to: Sequence[str],
        subject: str,
        text: str
    ) -> Dict[str, Any]:
        """
        Sends one plain-text message and returns the provider's id.

        Raises:
            ToolNotConfiguredError: no API key or verified sender on this server.
            ToolInvocationError: the provider could not be reached, or refused.
        """
        if not self.is_configured:
            raise ToolNotConfiguredError(
                "Email is not configured. Set RESEND_API_KEY and RESEND_FROM_EMAIL "
                "(or REPORT_EMAIL) to send confirmation mail."
            )

        recipients: List[str] = [str(address).strip() for address in to or () if str(address).strip()]
        if not recipients:
            raise ToolInvocationError("An email needs at least one recipient.")

        body = call_transport(self._transport, self.endpoint, {
            "from": self.sender,
            "to": recipients,
            "subject": subject,
            "text": text,
        }, {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        message_id = str((body or {}).get("id") or "")
        logger.info("Sent confirmation mail to %d recipient(s).", len(recipients))
        return {"message_id": message_id, "recipients": recipients}
