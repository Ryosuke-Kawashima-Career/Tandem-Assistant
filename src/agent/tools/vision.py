"""
Summary:
    vision.py is the REQ-22 camera client: a participant points their device camera at
    what they are actually looking at - a page of kanji, a whiteboard, a handover
    document - and the agent explains it as a material card.

    One still frame per request, never a stream. That is the whole design decision: a
    live video leg would be a second continuous transport to reason about (and to keep
    off the REQ-17 audio path), while the question being asked - "what is this in front
    of me?" - is answered by a single capture. It also bounds the cost and the privacy
    exposure to the instant the participant chose to capture, rather than to however long
    they forgot the camera was on.

    Credentials are the REQ-17 `GEMINI_API_KEY`, deliberately reused: a deployment that
    already has Gemini for translation gets camera assist without sourcing a new secret,
    and one that has neither reports `unavailable` for both.

Key Classes:
    - VisionDescription: one explained frame, in the shape a material card renders.
    - CameraVisionTool: the configured client.
"""

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    call_transport,
    http_post_json,
)

logger = logging.getLogger("echosphere.agent.tools.vision")

# Gemini's `generateContent` REST surface. Used rather than the `google-genai` SDK so the
# transport stays a plain injectable JSON POST, exactly like the other tools in this
# package - and so the server does not need a second Gemini dependency to answer one
# request per camera capture.
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Environment-selected so a deployment can move to a newer vision model without a code
# change; the default names a current multimodal Gemini tier.
MODEL_ENV = "GEMINI_VISION_MODEL"
DEFAULT_VISION_MODEL = "gemini-3.5-flash"

# A still frame from a modern phone camera runs to several megabytes. Bounded here so a
# mis-set capture resolution becomes an immediate, explainable refusal rather than a slow
# upload, a large bill, and a vendor-side rejection that reads as an outage.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# A title has to fit one line of a card; the rest of the answer is the body.
MAX_TITLE_CHARS = 80

# Spoken names for the languages this system supports, so the prompt can ask for an
# answer the participant can actually read (REQ-22 answers in the working language).
LANGUAGE_NAMES = {"ja": "Japanese", "hi": "Hindi", "en": "English"}

DEFAULT_QUESTION = "What is in this image?"


@dataclass(frozen=True)
class VisionDescription:
    """One explained camera frame, reduced to what a material card shows."""

    title: str
    description: str
    language: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the description for the `reference.card` event and API responses."""
        return {
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "model": self.model,
        }


class CameraVisionTool:
    """
    Explains a single camera frame for a session (REQ-22).

    The frame is held only for the duration of the call: it is base64-encoded into the
    request and dropped. Nothing here writes an image to disk, and the description it
    returns carries no copy of the capture.
    """

    name = "vision"

    def __init__(
        self,
        api_key: Optional[str] = None,
        transport: Optional[Callable[..., Dict[str, Any]]] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        model: Optional[str] = None,
        max_bytes: int = MAX_IMAGE_BYTES
    ):
        """
        Initialize the client from explicit values or the environment.

        Algorithm:
        1. Resolve the Gemini API key (shared with REQ-17's translation legs).
        2. Resolve the vision model, environment-selected.
        3. Bind the transport - injected in tests, a plain JSON POST otherwise.
        4. Leave the tool unconfigured, not failed, when the key is missing.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv(MODEL_ENV, DEFAULT_VISION_MODEL)
        self.endpoint = endpoint
        self.max_bytes = max_bytes
        self._transport = transport or http_post_json

    @property
    def is_configured(self) -> bool:
        """Whether this server can actually explain a camera frame."""
        return bool(self.api_key)

    def build_prompt(
        self,
        question: str = "",
        language: str = "",
        materials: bool = False
    ) -> str:
        """
        Builds the instruction sent alongside the frame (REQ-22).

        Algorithm:
        1. Lead with what the participant asked, or a plain description request.
        2. Ask for a one-line heading and then the explanation, which is the shape the
           existing reference card renders.
        3. Name the session's language so the answer comes back readable to the person
           holding the camera, and say which kind of help is wanted: reading and
           understanding the material in a learning session, the task material itself in
           a work session.
        """
        asked = (question or "").strip() or DEFAULT_QUESTION
        name = LANGUAGE_NAMES.get((language or "").strip().lower())

        lines = [
            "You are looking at one still frame from a participant's device camera "
            "during a live language-exchange or international-work session.",
            f"The participant asks: {asked}",
            "Answer with a short heading on the first line, then a brief explanation of "
            "what is visible and what it means.",
        ]

        if materials:
            lines.append(
                "Focus on the task material itself: what this document, diagram, or "
                "object is, and what a colleague new to it needs to know."
            )
        else:
            lines.append(
                "Focus on helping the participant read and understand what is visible, "
                "including any text, and explain unfamiliar wording."
            )

        if name:
            lines.append(
                f"Explain in English, and quote any {name} text as it appears with a "
                "short reading and gloss."
            )

        lines.append("Describe only what is actually visible. Do not guess at what is not.")
        return "\n".join(lines)

    def describe(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        question: str = "",
        language: str = "",
        materials: bool = False
    ) -> VisionDescription:
        """
        Explains one camera frame (REQ-22).

        Algorithm:
        1. Refuse when unconfigured, so the dispatcher reports `unavailable` rather than
           surfacing a vendor 403 as a material card.
        2. Refuse an empty or oversized frame locally, before spending a request on it.
        3. Send the frame inline as base64 alongside the prompt.
        4. Parse the answer into a heading and a body.

        Raises:
            ToolNotConfiguredError: no Gemini API key on this server.
            ToolInvocationError: unusable frame, unreachable vendor, or an empty answer.
        """
        if not self.is_configured:
            raise ToolNotConfiguredError(
                "Camera vision is not configured. Set GEMINI_API_KEY to enable it."
            )

        frame = bytes(image_bytes or b"")
        if not frame:
            raise ToolInvocationError("The camera produced an empty frame.")
        if len(frame) > self.max_bytes:
            raise ToolInvocationError(
                f"The captured frame is {len(frame)} bytes, over the "
                f"{self.max_bytes}-byte limit. Capture at a lower resolution."
            )

        payload = {
            "contents": [{
                "parts": [
                    {"text": self.build_prompt(question, language, materials)},
                    {
                        "inline_data": {
                            "mime_type": (mime_type or "image/jpeg").strip(),
                            "data": base64.b64encode(frame).decode("ascii"),
                        }
                    },
                ]
            }]
        }

        # The key travels in a header, never in the URL: a query-string credential ends up
        # in every proxy log and error report between here and Google (REQ-08).
        body = call_transport(
            self._transport,
            f"{self.endpoint}/{self.model}:generateContent",
            payload,
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )

        text = self._answer_text(body)
        if not text:
            raise ToolInvocationError(
                "Gemini returned no description for the captured frame."
            )

        logger.info("Camera frame described in %d character(s).", len(text))
        return self._to_description(text, language)

    @staticmethod
    def _answer_text(body: Any) -> str:
        """
        Extracts the model's answer from a `generateContent` response.

        Concatenates every text part rather than taking the first: a multimodal reply may
        be split across parts, and reading only part one silently truncates the answer.
        """
        if not isinstance(body, dict):
            return ""

        candidates = body.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            return ""

        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        chunks: List[str] = [
            str(part.get("text") or "") for part in parts if isinstance(part, dict)
        ]
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    def _to_description(self, text: str, language: str) -> VisionDescription:
        """
        Splits the answer into the heading and body a material card renders.

        A model that answers in one sentence still has to produce a usable card, so the
        single line becomes both a (truncated) heading and the body - rather than a card
        with an empty title, which renders as an anonymous block of text.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        heading = lines[0] if lines else text.strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text.strip()

        return VisionDescription(
            title=heading[:MAX_TITLE_CHARS],
            description=body,
            language=(language or "").strip(),
            model=self.model
        )
