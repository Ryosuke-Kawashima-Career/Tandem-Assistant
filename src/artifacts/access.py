"""
Summary:
    access.py decides who may read, edit, export, or delete a session's artifacts
    (REQ-16).

    One rule, in one place: a session belongs to the people who were in it. Retrieval,
    export, editing, and deletion all route through here, so a new endpoint cannot
    accidentally ship without the check that the others enforce.

Key Functions:
    - can_access: the predicate.
    - require_access: the enforcing form, for endpoints.
    - resolve_actor: reads the caller's identity off a request.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("echosphere.artifacts.access")

# Header form of the caller's identity. Preferred over a query parameter because a URL
# ends up in server logs, proxy logs, and browser history, and a participant name is
# personal data (REQ-16) that has no business in any of them.
ACTOR_HEADER = "X-EchoSphere-Actor"


class AccessDeniedError(PermissionError):
    """Raised when an actor may not touch a session's artifacts."""


def _participants(session_meta: Dict[str, Any]) -> list:
    """Returns the recorded participants of a session, normalized to a list."""
    return [str(name) for name in (session_meta.get("participants") or []) if str(name).strip()]


def can_access(session_meta: Optional[Dict[str, Any]], actor: Optional[str]) -> bool:
    """
    Whether `actor` may act on the session described by `session_meta` (REQ-16).

    Algorithm:
    1. No session => no access; there is nothing to authorize against.
    2. No recorded participants => open. A session that never captured who was in it has
       nobody to be protected from, and refusing everyone would lock out the only person
       who could have asked for it. This is the single-operator demo path, not a hole in
       a multi-tenant deployment - a deployment that cares must record participants.
    3. Otherwise the actor must be one of the participants, matched case-insensitively
       so a display name typed with different capitalization is not treated as a stranger.
    """
    if not session_meta:
        return False

    participants = _participants(session_meta)
    if not participants:
        return True

    if not actor or not str(actor).strip():
        return False

    return str(actor).strip().casefold() in {name.casefold() for name in participants}


def require_access(session_meta: Optional[Dict[str, Any]], actor: Optional[str]) -> None:
    """
    Enforces `can_access`, raising `AccessDeniedError` when the actor may not proceed.

    Raises rather than returning a boolean so an endpoint cannot forget to check the
    result - the failure mode of a forgotten `if` here is handing a stranger a
    conversation transcript.
    """
    if can_access(session_meta, actor):
        return

    session_id = (session_meta or {}).get("session_id", "unknown")
    logger.warning(
        "Denied artifact access to session %s for actor %r.", session_id, actor
    )
    raise AccessDeniedError(
        f"Actor {actor!r} is not a participant of session {session_id}."
    )


def resolve_actor(request: Any) -> Optional[str]:
    """
    Reads the caller's identity from a Flask request.

    Checks the header first, then a query parameter, then a JSON body field, so a browser
    fetch, a link, and an API client can all identify themselves without a session
    cookie - this is an MVP identity, not an authentication system, and it is deliberately
    the narrowest thing that satisfies REQ-16's access rule.
    """
    actor = request.headers.get(ACTOR_HEADER) or request.args.get("actor")
    if actor:
        return actor.strip()

    body = request.get_json(silent=True) or {}
    actor = body.get("actor") if isinstance(body, dict) else None
    return str(actor).strip() if actor else None
