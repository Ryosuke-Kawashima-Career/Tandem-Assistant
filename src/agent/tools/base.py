"""
Summary:
    base.py holds what every agent tool shares (REQ-18–20): the outcome vocabulary the
    dispatcher publishes, the two error types a tool may raise, and the JSON HTTP helpers
    the vendor clients are built on.

    The two error types exist so a caller can tell the difference that matters
    operationally: `ToolNotConfiguredError` means this deployment never had the
    credential (report it once and move on), while `ToolInvocationError` means the tool
    is configured but the call failed (worth a retry, and worth an alert if it keeps
    happening). Collapsing them into one exception loses that distinction at exactly the
    moment somebody is trying to work out why a button did nothing.

Key Classes:
    - ToolState / ToolResult: the outcome of one tool call.
    - ToolNotConfiguredError / ToolInvocationError: the two failure modes.
    - http_get_json / http_post_json: the default transports, injectable in tests.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("echosphere.agent.tools")

# Bumped when the shape of a tool event changes. Receivers route and deduplicate by
# envelope shape alone, exactly as they do for artifact and translation events.
TOOL_SCHEMA_VERSION = "1.0"

EVENT_TOOL_STATUS = "tool.status"
EVENT_REFERENCE_CARD = "reference.card"
EVENT_ANKI_EXPORTED = "anki.exported"
EVENT_MEETING_SCHEDULED = "meeting.scheduled"

# Every tool call is made while a conversation is running, so a hung vendor is a hung
# background thread rather than a hung session - but an unbounded wait would still leak
# one thread per click for the life of the process.
REQUEST_TIMEOUT_SECONDS = 20


class ToolState:
    """
    The three outcomes of a tool call.

    A plain constant holder rather than an Enum: these values travel in RTC event
    payloads and API responses as bare strings, matching `NoteStatus` and the translation
    leg states.
    """

    OK = "ok"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ToolNotConfiguredError(RuntimeError):
    """Raised when a tool has no credentials on this server."""


class ToolInvocationError(RuntimeError):
    """Raised when a configured tool was reached but the call did not succeed."""


@dataclass(frozen=True)
class ToolResult:
    """One tool call's outcome, as returned to the caller and reported over HTTP."""

    tool: str
    state: str
    reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the call actually did what it was asked to do."""
        return self.state == ToolState.OK

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result for API responses."""
        return {
            "tool": self.tool,
            "state": self.state,
            "reason": self.reason,
            "payload": dict(self.payload),
        }


def call_transport(transport: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Runs a tool's transport and normalizes whatever it raises.

    Every vendor client goes through here so a failure is this package's own error type
    regardless of which transport produced it - the default JSON helpers, an injected
    fake, or a future replacement that raises its own library's exceptions. The
    dispatcher turns exactly one exception type into a `tool.status` event, and it must
    not have to know what is underneath.
    """
    try:
        return transport(*args, **kwargs)
    except (ToolInvocationError, ToolNotConfiguredError):
        raise
    except Exception as exc:  # noqa: BLE001 - every transport failure is one outcome here
        raise ToolInvocationError(str(exc)) from exc


def _requests():
    """
    Returns the `requests` module, or explains its absence as a tool failure.

    Imported lazily and per call rather than at module import: every tool here is
    optional, and a server that uses none of them must not fail to start over a
    dependency it never reaches.
    """
    try:
        import requests
        return requests
    except ImportError as exc:
        raise ToolInvocationError(
            "Agent tools need the 'requests' package, which is not installed."
        ) from exc


def _json_or_raise(response, url: str) -> Dict[str, Any]:
    """
    Turns a vendor response into JSON, or into this package's own error type.

    A vendor's error body is included in the message because it is usually the only
    place the actual reason appears ("API key not valid", "quota exceeded"); truncated
    because some of them return an HTML page.
    """
    if response.status_code >= 400:
        raise ToolInvocationError(
            f"{url} answered {response.status_code}: {response.text[:300]}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ToolInvocationError(f"{url} returned a non-JSON response.") from exc


def http_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Performs a GET and returns the decoded JSON body."""
    requests = _requests()
    try:
        response = requests.get(
            url, params=params or {}, headers=headers or {}, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - every transport failure is one outcome here
        raise ToolInvocationError(f"GET {url} failed: {exc}") from exc
    return _json_or_raise(response, url)


def http_post_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Performs a JSON POST and returns the decoded JSON body."""
    requests = _requests()
    try:
        response = requests.post(
            url, json=payload or {}, headers=headers or {}, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - every transport failure is one outcome here
        raise ToolInvocationError(f"POST {url} failed: {exc}") from exc
    return _json_or_raise(response, url)
