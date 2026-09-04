"""
Summary:
    test_agent_tools_vision.py is the executable specification for REQ-22: a participant
    points their camera at what they are looking at - a page of kanji, a whiteboard, a
    handover document - and the agent explains it as a material card, without the frame
    being stored and without the API key leaving the server.

    The tool is deliberately built as a fourth `agent/tools/` integration rather than as a
    new pipeline: it inherits Phase 12's rule that a tool call never raises into a session
    and never blocks a turn, which is the same contract REQ-22 states in its own words.

Covers:
    - REQ-22 client: configuration gate, inline frame upload, response parsing, failures.
    - REQ-22 dispatch: a material card on success, `tool.status {tool: 'vision'}` otherwise.
    - REQ-22 API: `POST /api/tools/vision` as JSON base64 and as a multipart upload.
    - REQ-22 retention: the frame is not persisted anywhere by describing it.
    - REQ-08/REQ-16: server-side key only; governed like every other session endpoint.
"""

import base64
import io
import unittest
from unittest.mock import MagicMock

from src.agent.tools.base import (
    ToolInvocationError,
    ToolNotConfiguredError,
    ToolState,
)
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.vision import CameraVisionTool
from src.sessions.models import SessionRecord

API_KEY = "test-vision-key"
FRAME = b"\xff\xd8\xff\xe0 not a real JPEG, but bytes are bytes"

# Shaped like one Gemini `generateContent` response over an inline image part.
VISION_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": (
                            "Kanban board with three columns\n"
                            "A whiteboard divided into To Do, Doing and Done. "
                            "The sticky notes are written in Japanese."
                        )
                    }
                ]
            }
        }
    ]
}


def learning_session():
    """A language-learning session; a camera lookup explains what the learner sees."""
    return SessionRecord.create(
        channel="vision-learning", mode="language_learning",
        languages=["ja"], participants=["Kenji"]
    )


def work_session():
    """An international-work session; a camera lookup explains the task material."""
    return SessionRecord.create(
        channel="vision-work", mode="international_work",
        languages=["en"], participants=["Priya"]
    )


class RecordingTransport:
    """A stand-in for the JSON POST, recording its call and replaying a canned answer."""

    def __init__(self, response=None, error=None):
        """Store the response to replay, or the error to raise instead."""
        self.response = response if response is not None else VISION_RESPONSE
        self.error = error
        self.calls = []

    def __call__(self, url, payload=None, headers=None):
        """Record the request and replay the canned outcome."""
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        if self.error is not None:
            raise self.error
        return self.response


class TestCameraVisionTool(unittest.TestCase):
    """Test suite for the vision client itself (REQ-22, TASK-13.2)."""

    def test_a_tool_without_a_key_is_unconfigured_and_never_reaches_the_network(self):
        """
        Verify the absence of a credential is detected here, before any request.

        REQ-22 reuses the REQ-17 `GEMINI_API_KEY`, so a deployment that has translation
        gets vision for free - and one that has neither must say "not configured" rather
        than forward a 403 from Google into a participant's material card.
        """
        transport = RecordingTransport()
        tool = CameraVisionTool(api_key="", transport=transport)

        self.assertFalse(tool.is_configured)
        with self.assertRaises(ToolNotConfiguredError):
            tool.describe(FRAME)
        self.assertEqual(transport.calls, [])

    def test_a_frame_is_uploaded_inline_and_the_answer_is_parsed(self):
        """Verify the image travels as base64 inline data and the reply becomes a card."""
        transport = RecordingTransport()
        tool = CameraVisionTool(api_key=API_KEY, transport=transport)

        description = tool.describe(FRAME, mime_type="image/jpeg", question="What is this?")

        self.assertEqual(description.title, "Kanban board with three columns")
        self.assertIn("To Do", description.description)

        parts = transport.calls[0]["payload"]["contents"][0]["parts"]
        inline = [part["inline_data"] for part in parts if "inline_data" in part][0]
        self.assertEqual(inline["mime_type"], "image/jpeg")
        self.assertEqual(base64.b64decode(inline["data"]), FRAME)
        self.assertTrue(any("What is this?" in str(part.get("text", "")) for part in parts))

    def test_the_key_travels_in_a_header_and_never_in_the_url(self):
        """
        Verify the credential stays out of the request line (REQ-08).

        A key in a query string is a key in every proxy log and error report between here
        and Google; the same reasoning already keeps the actor name out of artifact URLs.
        """
        transport = RecordingTransport()
        CameraVisionTool(api_key=API_KEY, transport=transport).describe(FRAME)

        call = transport.calls[0]
        self.assertNotIn(API_KEY, call["url"])
        self.assertEqual(call["headers"].get("x-goog-api-key"), API_KEY)

    def test_the_prompt_asks_for_the_answer_in_the_session_language(self):
        """Verify a learner reading a Japanese page can be answered about it in Japanese."""
        transport = RecordingTransport()
        tool = CameraVisionTool(api_key=API_KEY, transport=transport)

        tool.describe(FRAME, language="ja")

        prompt = str(transport.calls[0]["payload"]["contents"][0]["parts"])
        self.assertIn("Japanese", prompt)

    def test_an_empty_frame_fails_without_calling_the_vendor(self):
        """Verify a camera that produced nothing is not billed as a request."""
        transport = RecordingTransport()
        tool = CameraVisionTool(api_key=API_KEY, transport=transport)

        with self.assertRaises(ToolInvocationError):
            tool.describe(b"")
        self.assertEqual(transport.calls, [])

    def test_an_oversized_frame_is_refused_locally(self):
        """
        Verify a frame beyond the limit is rejected before the upload starts.

        A still frame from a modern camera can be several megabytes; sending an
        unbounded one turns a mis-set capture resolution into a slow request, a large
        bill, and a vendor-side rejection that reads as an outage.
        """
        transport = RecordingTransport()
        tool = CameraVisionTool(api_key=API_KEY, transport=transport, max_bytes=16)

        with self.assertRaises(ToolInvocationError):
            tool.describe(b"x" * 17)
        self.assertEqual(transport.calls, [])

    def test_a_transport_failure_becomes_this_package_s_error_type(self):
        """Verify a vendor or network failure surfaces as `ToolInvocationError`."""
        tool = CameraVisionTool(
            api_key=API_KEY, transport=RecordingTransport(error=RuntimeError("timed out"))
        )

        with self.assertRaises(ToolInvocationError):
            tool.describe(FRAME)

    def test_an_answerless_response_is_a_failure_not_an_empty_card(self):
        """
        Verify a reply with no candidates is reported rather than rendered.

        A blank material card is indistinguishable from "the model saw nothing worth
        saying", and the participant is left holding the camera up to no purpose.
        """
        tool = CameraVisionTool(
            api_key=API_KEY, transport=RecordingTransport(response={"candidates": []})
        )

        with self.assertRaises(ToolInvocationError):
            tool.describe(FRAME)

    def test_a_single_line_answer_still_yields_a_title_and_a_body(self):
        """Verify the card shape survives a model that answers in one sentence."""
        tool = CameraVisionTool(
            api_key=API_KEY,
            transport=RecordingTransport(response={
                "candidates": [{"content": {"parts": [{"text": "A page of hiragana practice."}]}}]
            })
        )

        description = tool.describe(FRAME)

        self.assertTrue(description.title)
        self.assertIn("hiragana", description.description)


class TestVisionDispatch(unittest.TestCase):
    """Test suite for publishing a camera material card (REQ-22, TASK-13.2)."""

    def setUp(self):
        """Use a recording data stream and a configured vision tool per test."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.tool = CameraVisionTool(api_key=API_KEY, transport=RecordingTransport())
        self.dispatcher = ToolDispatcher(data_stream=self.data_stream, vision=self.tool)

    def published(self, event_type):
        """Returns the payloads published under one event type."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == event_type
        ]

    def test_a_described_frame_publishes_a_camera_material_card(self):
        """
        Verify the answer arrives as a reference card marked as coming from the camera.

        It reuses the REQ-18 card shape deliberately: a participant asking "what is this"
        by typing and by pointing a camera is asking one question, and two differently
        shaped answers on the same column would be two things to learn instead of one.
        """
        session = work_session()

        result = self.dispatcher.describe_camera_frame(
            session, FRAME, question="What does this board say?", requested_by="Priya"
        )

        self.assertTrue(result.ok)
        cards = self.published("reference.card")
        self.assertEqual(len(cards), 1)
        card = cards[0]["card"]
        self.assertEqual(cards[0]["tool"], "vision")
        self.assertEqual(card["source"], "camera")
        self.assertEqual(card["query"], "What does this board say?")
        self.assertEqual(card["results"][0]["title"], "Kanban board with three columns")
        self.assertIn("To Do", card["results"][0]["snippet"])

    def test_the_published_card_carries_no_image_data(self):
        """
        Verify the frame itself is not broadcast to the channel (REQ-22 retention).

        The card is delivered to every participant, so embedding the capture would
        publish whatever was behind the person holding the camera to the whole room -
        and store it in every client's event log.
        """
        self.dispatcher.describe_camera_frame(learning_session(), FRAME)

        published = str(self.data_stream.send_tool_event.call_args_list)
        self.assertNotIn(base64.b64encode(FRAME).decode("ascii"), published)
        self.assertNotIn(API_KEY, published)

    def test_an_unconfigured_tool_reports_unavailable_instead_of_raising(self):
        """Verify a server without a Gemini key degrades to a status event."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream, vision=CameraVisionTool(api_key="")
        )

        result = dispatcher.describe_camera_frame(learning_session(), FRAME)

        self.assertEqual(result.state, ToolState.UNAVAILABLE)
        self.assertFalse(self.published("reference.card"))
        status = self.published("tool.status")[0]
        self.assertEqual(status["tool"], "vision")
        self.assertEqual(status["state"], "unavailable")
        self.assertTrue(status["reason"])

    def test_a_missing_vision_tool_behaves_exactly_like_an_unconfigured_one(self):
        """Verify a dispatcher built without vision at all still answers, never raises."""
        dispatcher = ToolDispatcher(data_stream=self.data_stream)

        result = dispatcher.describe_camera_frame(learning_session(), FRAME)

        self.assertEqual(result.state, ToolState.UNAVAILABLE)

    def test_a_failing_vision_call_reports_failed_and_publishes_no_card(self):
        """Verify a vendor error is announced as failed rather than as a blank card."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(
                api_key=API_KEY, transport=RecordingTransport(error=RuntimeError("quota"))
            )
        )

        result = dispatcher.describe_camera_frame(learning_session(), FRAME)

        self.assertEqual(result.state, ToolState.FAILED)
        self.assertFalse(self.published("reference.card"))
        self.assertEqual(self.published("tool.status")[0]["state"], "failed")

    def test_tool_availability_reports_vision(self):
        """Verify the UI can tell whether to offer the camera control at all."""
        self.assertTrue(self.dispatcher.status()["vision"])
        self.assertFalse(ToolDispatcher().status()["vision"])


class TestVisionApi(unittest.TestCase):
    """Test suite for `POST /api/tools/vision` (REQ-22, TASK-13.2)."""

    def setUp(self):
        """A started session, with the server's vision tool swapped for a fake."""
        from src.server import app, server_instance

        self.client = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.original_vision = self.server.tools.vision
        self.server.data_stream.packet_history.clear()
        self.client.post("/api/session/start", json={
            "channel": "vision-channel", "mode": "international_work",
            "participants": ["Priya"], "languages": ["en"]
        })

    def tearDown(self):
        """Restore the server's own tool so tests do not leak configuration."""
        self.server.tools.vision = self.original_vision

    def configure(self, **kwargs):
        """Gives the server a working vision tool."""
        self.server.tools.vision = CameraVisionTool(
            api_key=API_KEY, transport=RecordingTransport(**kwargs)
        )

    def post_frame(self, **body):
        """Posts one base64 frame as JSON."""
        payload = {
            "channel": "vision-channel",
            "actor": "Priya",
            "image_base64": base64.b64encode(FRAME).decode("ascii"),
            "mime_type": "image/jpeg",
            "question": "What is on this page?",
        }
        payload.update(body)
        return self.client.post("/api/tools/vision", json=payload)

    def test_a_described_frame_answers_with_the_card(self):
        """Verify the requester gets the answer directly, not only over the stream."""
        self.configure()

        response = self.post_frame()

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["tool"], "vision")
        self.assertEqual(body["card"]["source"], "camera")
        self.assertTrue(body["card"]["results"][0]["snippet"])

    def test_a_multipart_upload_is_accepted(self):
        """
        Verify the browser can post the capture as a file rather than as base64 JSON.

        `canvas.toBlob` is what a camera capture naturally produces, and base64 inflates
        it by a third on a link that is already carrying a live conversation.
        """
        self.configure()

        response = self.client.post(
            "/api/tools/vision?channel=vision-channel&actor=Priya",
            data={
                "image": (io.BytesIO(FRAME), "frame.jpg"),
                "question": "What is on this page?",
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["card"]["results"])

    def test_a_request_without_an_image_is_a_400(self):
        """Verify a missing frame is a client error, not an empty vendor call."""
        self.configure()

        response = self.client.post("/api/tools/vision", json={
            "channel": "vision-channel", "actor": "Priya", "question": "What is this?"
        })

        self.assertEqual(response.status_code, 400)

    def test_an_unconfigured_server_answers_503(self):
        """
        Verify the endpoint follows the Notion and Anki precedent when it cannot run.

        503 rather than 500: the deployment simply has no Gemini key, which is a
        configuration fact about this server, not a fault in the request.
        """
        self.server.tools.vision = CameraVisionTool(api_key="")

        response = self.post_frame()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["state"], "unavailable")

    def test_a_failing_vendor_answers_502(self):
        """Verify a reachable-but-broken tool is distinguishable from an absent one."""
        self.configure(error=RuntimeError("upstream error"))

        response = self.post_frame()

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["state"], "failed")

    def test_a_non_participant_is_refused(self):
        """Verify the camera endpoint is governed like every other one (REQ-16)."""
        self.configure()

        response = self.post_frame(actor="Stranger")

        self.assertEqual(response.status_code, 403)

    def test_an_unknown_session_is_a_404(self):
        """Verify a frame cannot be described into a session that does not exist."""
        self.configure()

        response = self.post_frame(channel="no-such-channel")

        self.assertEqual(response.status_code, 404)

    def test_describing_a_frame_stores_nothing(self):
        """
        Verify the capture leaves no trace in the session's artifacts (REQ-22).

        The requirement is explicit: no image persists past generating the card unless
        the participant saves it as a note themselves. A camera pointed at a desk sees
        more than the document on it.
        """
        self.configure()
        self.post_frame()

        artifact = self.client.get(
            "/api/session/artifact?channel=vision-channel&actor=Priya"
        ).get_json()

        self.assertFalse(artifact.get("artifact", {}).get("transcript_turns"))
        self.assertFalse(artifact.get("artifact", {}).get("notes"))


if __name__ == "__main__":
    unittest.main()
