"""
Summary:
    google_calendar.py books the follow-up session (REQ-20): it creates a Calendar event
    and asks Google to deliver the invitations to the participants.

    Google's own invitation is the delivery channel, not a courtesy email from this
    server: an event that exists only in the organizer's calendar is precisely the
    "meeting" nobody else turns up to. That is why the request sets `sendUpdates=all`,
    and why an unusable attendee address is dropped rather than sent - Calendar rejects
    the whole request over one malformed address, which would turn a typo in one
    person's name into no meeting for anybody.

Key Classes:
    - GoogleCalendarTool: the configured Calendar client.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    call_transport,
    http_post_json,
)

logger = logging.getLogger("echosphere.agent.tools.calendar")

DEFAULT_ENDPOINT = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
DEFAULT_CALENDAR_ID = "primary"
DEFAULT_DURATION_MINUTES = 30
DEFAULT_TIMEZONE = "UTC"

# The narrowest scope that can create an event and invite people to it.
CALENDAR_SCOPES = ("https://www.googleapis.com/auth/calendar.events",)


class GoogleCalendarTool:
    """
    Creates a follow-up meeting and invites the session's participants (REQ-20).

    Credentials never leave the server. Two forms are supported because they fail
    differently: a pinned `GOOGLE_CALENDAR_ACCESS_TOKEN` is trivial to set for a demo but
    expires within the hour, while a service account keeps working - and needs the Google
    auth SDK installed.
    """

    name = "calendar"

    def __init__(
        self,
        access_token: Optional[str] = None,
        credentials_json: Optional[str] = None,
        calendar_id: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None,
        token_provider: Optional[Callable[[], str]] = None,
        endpoint: str = DEFAULT_ENDPOINT
    ):
        """
        Initialize the client from explicit values or the environment.

        Algorithm:
        1. Resolve a static access token, a service-account credentials path, and the
           calendar to write to.
        2. Bind the transport and the optional token provider (both injected in tests).
        3. Leave the tool unconfigured, not failed, when no credential of any kind exists.
        """
        self.access_token = (
            access_token if access_token is not None
            else os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "")
        )
        self.credentials_json = (
            credentials_json if credentials_json is not None
            else os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON", "")
        )
        self.calendar_id = (
            calendar_id if calendar_id is not None
            else os.getenv("GOOGLE_CALENDAR_ID", DEFAULT_CALENDAR_ID)
        )
        self.endpoint = endpoint
        self._transport = transport or http_post_json
        self._token_provider = token_provider

    @property
    def is_configured(self) -> bool:
        """Whether some credential exists for this server to authenticate with."""
        return bool(self.access_token or self._token_provider or self.credentials_json)

    def _resolve_token(self) -> str:
        """
        Returns the bearer token for one request (REQ-20, server-side only).

        Algorithm:
        1. A pinned token wins - it is the explicit override.
        2. An injected provider is next; this is the seam a service account plugs into.
        3. Otherwise mint one from the service-account file, which needs the Google auth
           SDK - its absence is a configuration problem, reported as one.

        Raises:
            ToolNotConfiguredError: no usable credential on this server.
        """
        if self.access_token:
            return self.access_token
        if self._token_provider is not None:
            return str(self._token_provider())
        if self.credentials_json:
            return self._mint_service_account_token()

        raise ToolNotConfiguredError(
            "Google Calendar is not configured. Set GOOGLE_CALENDAR_ACCESS_TOKEN or "
            "GOOGLE_CALENDAR_CREDENTIALS_JSON to enable meeting scheduling."
        )

    def _mint_service_account_token(self) -> str:
        """
        Mints an access token from the configured service-account credentials.

        The Google auth SDK is imported lazily: a deployment that schedules nothing must
        not carry the dependency, and a missing SDK is reported as an unconfigured tool
        rather than an import error at server start.
        """
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise ToolNotConfiguredError(
                "GOOGLE_CALENDAR_CREDENTIALS_JSON is set but the Google auth SDK is not "
                "installed. Install `google-auth`, or set GOOGLE_CALENDAR_ACCESS_TOKEN."
            ) from exc

        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_json, scopes=list(CALENDAR_SCOPES)
            )
            credentials.refresh(Request())
        except Exception as exc:  # noqa: BLE001 - any auth failure is one outcome here
            raise ToolInvocationError(f"Could not mint a Calendar token: {exc}") from exc

        return str(credentials.token)

    def schedule(
        self,
        summary: str,
        start_time: Any,
        duration_minutes: int = DEFAULT_DURATION_MINUTES,
        attendees: Sequence[str] = (),
        description: str = "",
        timezone_name: str = DEFAULT_TIMEZONE
    ) -> Dict[str, Any]:
        """
        Creates one event and invites its attendees (REQ-20).

        Algorithm:
        1. Resolve the credential first, so an unconfigured server fails before it has
           built anything.
        2. Derive the end time from the duration, and keep only usable addresses.
        3. POST the event with `sendUpdates=all` so Google delivers the invitations.
        4. Return the event id and link - the link is what a participant is actually
           given, and a booking nobody can open is not much of a booking.

        Raises:
            ToolNotConfiguredError: no Calendar credential on this server.
            ToolInvocationError: the API could not be reached, or refused the event.
        """
        token = self._resolve_token()

        start = _parse_datetime(start_time)
        minutes = max(1, int(duration_minutes or DEFAULT_DURATION_MINUTES))
        end = start + timedelta(minutes=minutes)
        invitees = [address for address in attendees or () if _is_email(address)]

        url = self.endpoint.format(calendar_id=self.calendar_id) + "?sendUpdates=all"
        payload = {
            "summary": summary or "EchoSphere follow-up session",
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": timezone_name},
            "end": {"dateTime": end.isoformat(), "timeZone": timezone_name},
            "attendees": [{"email": address} for address in invitees],
        }

        body = call_transport(self._transport, url, payload, {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

        if not isinstance(body, dict) or not body.get("id"):
            raise ToolInvocationError(
                "Google Calendar accepted the request but returned no event id."
            )

        logger.info(
            "Scheduled Calendar event %s for %d attendee(s).", body["id"], len(invitees)
        )
        return {
            "event_id": str(body["id"]),
            "html_link": str(body.get("htmlLink") or ""),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "attendees": invitees,
            "calendar_id": self.calendar_id,
        }


def _parse_datetime(value: Any) -> datetime:
    """
    Reads a start time from a datetime, an epoch timestamp, or an ISO-8601 string.

    A naive value is read as UTC rather than as local time: the server, the two
    participants, and Google are routinely in four different zones, and "9:00 with no
    zone" is the one input where guessing the server's zone is silently wrong for
    everybody else.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        text = str(value or "").strip()
        if not text:
            raise ToolInvocationError("A meeting needs a start time.")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ToolInvocationError(
                f"Could not read {text!r} as an ISO-8601 start time."
            ) from exc

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_email(value: Any) -> bool:
    """Whether a value is usable as an attendee address (REQ-20)."""
    text = str(value or "").strip()
    if not text or " " in text or text.count("@") != 1:
        return False
    _, _, domain = text.partition("@")
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")
