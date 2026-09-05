"""
Summary:
    test_agent_tools_camera_stream.py is the executable specification for REQ-CAM-01 and
    REQ-CAM-04: the co-teacher can look through the participant's camera *during* a live
    voice turn, without the participant first pressing a capture button - and doing so
    costs one bounded vendor call per distinct frame, never one per question and never
    one that outlives the turn it was meant to answer.

    The buffer is deliberately the smallest thing that makes an agent-initiated lookup
    possible: one still frame per channel, replaced in place, expiring on its own. It is
    not a recording, and nothing here can answer "what did the camera see a minute ago".

Covers:
    - REQ-CAM-01 buffer: put/get, per-channel isolation, replacement, TTL expiry.
    - REQ-CAM-01 API: `POST /api/session/camera/stream` ingests without a vendor call.
    - REQ-CAM-03 lookup: a buffered frame becomes a described card; an empty buffer
      returns `None` without reaching the vendor.
    - REQ-CAM-04 bounds: a stalled vendor call is abandoned at the timeout, and a second
      question against the same frame reuses the first description.
"""

import base64
import io
import threading
import time
import unittest
from unittest.mock import MagicMock

from src.agent.tools.camera_stream import (
    CAMERA_FRAME_TTL_SECONDS,
    BufferedFrame,
    CameraFrameBuffer,
)
from src.agent.tools.dispatch import ToolDispatcher
from src.agent.tools.vision import CameraVisionTool
from src.sessions.models import SessionRecord

API_KEY = "test-vision-key"
FRAME = b"\xff\xd8\xff\xe0 not a real JPEG, but bytes are bytes"
OTHER_FRAME = b"\xff\xd8\xff\xe0 a different moment entirely"

VISION_RESPONSE = {
    "candidates": [{
        "content": {"parts": [{
            "text": (
                "A ceramic teacup\n"
                "A small unglazed cup with a hand-painted rim, held up to the camera."
            )
        }]}
    }]
}


def learning_session(channel="camera-stream-learning"):
    """A language-learning session; the camera explains what the learner is holding."""
    return SessionRecord.create(
        channel=channel, mode="language_learning",
        languages=["ja"], participants=["Kenji"]
    )


class SlowTransport:
    """A vendor transport that never answers in time, to exercise the lookup bound."""

    def __init__(self, delay=30.0):
        """Store how long the vendor will hang for."""
        self.delay = delay
        self.calls = 0
        self.released = threading.Event()

    def __call__(self, url, payload=None, headers=None):
        """Block past any sane timeout, then answer nobody."""
        self.calls += 1
        self.released.wait(self.delay)
        return VISION_RESPONSE


class CountingTransport:
    """A transport that answers normally and counts how often the vendor was paid."""

    def __init__(self):
        """Start with no calls recorded."""
        self.calls = 0

    def __call__(self, url, payload=None, headers=None):
        """Record the call and replay the canned answer."""
        self.calls += 1
        return VISION_RESPONSE


class TestCameraFrameBuffer(unittest.TestCase):
    """Test suite for the per-channel live frame buffer (REQ-CAM-01, Task 1.1)."""

    def test_a_frame_put_is_immediately_available(self):
        """Verify the agent can look the instant a frame arrives, with no round trip."""
        buffer = CameraFrameBuffer()

        buffer.put("room-a", FRAME, "image/jpeg")
        entry = buffer.get("room-a")

        self.assertIsInstance(entry, BufferedFrame)
        self.assertEqual(entry.frame, FRAME)
        self.assertEqual(entry.mime_type, "image/jpeg")

    def test_a_channel_without_a_frame_has_nothing_to_look_at(self):
        """Verify an untouched channel reports no frame rather than raising."""
        self.assertIsNone(CameraFrameBuffer().get("room-nobody-opened"))

    def test_a_second_frame_replaces_the_first_rather_than_queueing(self):
        """
        Verify the buffer holds the present, not a history.

        A queue would let the agent describe something the participant has already moved
        the camera away from, and would keep captured frames alive past the moment they
        were taken - both of which this feature is explicitly bounded against.
        """
        buffer = CameraFrameBuffer()

        buffer.put("room-a", FRAME)
        buffer.put("room-a", OTHER_FRAME)

        self.assertEqual(buffer.get("room-a").frame, OTHER_FRAME)

    def test_frames_do_not_leak_between_channels(self):
        """Verify one session's camera is never described into another's conversation."""
        buffer = CameraFrameBuffer()

        buffer.put("room-a", FRAME)

        self.assertIsNone(buffer.get("room-b"))

    def test_a_frame_older_than_the_ttl_is_no_longer_current(self):
        """
        Verify a stale frame is treated exactly like no frame at all (Risk 3).

        A learner who pointed the camera away ten seconds ago and then asks "what is
        this?" must not be told about what used to be in view.
        """
        clock = {"now": 1000.0}
        buffer = CameraFrameBuffer(clock=lambda: clock["now"])

        buffer.put("room-a", FRAME)
        clock["now"] += CAMERA_FRAME_TTL_SECONDS + 0.1

        self.assertIsNone(buffer.get("room-a"))

    def test_a_frame_inside_the_ttl_is_still_current(self):
        """Verify a question asked a beat after looking still finds the frame."""
        clock = {"now": 1000.0}
        buffer = CameraFrameBuffer(clock=lambda: clock["now"])

        buffer.put("room-a", FRAME)
        clock["now"] += CAMERA_FRAME_TTL_SECONDS - 0.1

        self.assertIsNotNone(buffer.get("room-a"))

    def test_clearing_a_channel_empties_it_at_once(self):
        """Verify closing the camera panel need not wait out the TTL to stop lookups."""
        buffer = CameraFrameBuffer()

        buffer.put("room-a", FRAME)
        buffer.clear("room-a")

        self.assertIsNone(buffer.get("room-a"))

    def test_an_empty_frame_is_not_buffered(self):
        """Verify a client posting nothing cannot make the agent think it can see."""
        buffer = CameraFrameBuffer()

        buffer.put("room-a", b"")

        self.assertIsNone(buffer.get("room-a"))


class TestLiveFrameLookup(unittest.TestCase):
    """Test suite for the bounded, agent-initiated lookup (REQ-CAM-03/04, Task 3.1)."""

    def setUp(self):
        """A dispatcher over a live buffer and a configured, counting vision tool."""
        self.data_stream = MagicMock()
        self.data_stream.send_tool_event.return_value = True
        self.buffer = CameraFrameBuffer()
        self.transport = CountingTransport()
        self.dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key=API_KEY, transport=self.transport),
            camera_buffer=self.buffer
        )

    def published(self, event_type):
        """Returns the payloads published under one event type."""
        return [
            call.args[1] for call in self.data_stream.send_tool_event.call_args_list
            if call.args[0] == event_type
        ]

    def test_a_buffered_frame_is_described_and_carded_in_one_vendor_call(self):
        """
        Verify one Gemini call serves both the spoken reply and the on-screen card.

        Describing the frame twice - once to speak from, once to draw - would double the
        cost and the latency of the one thing the learner is waiting on.
        """
        self.buffer.put("room-a", FRAME)

        result = self.dispatcher.describe_live_frame(
            learning_session(), "room-a", question="What is this?", requested_by="voice"
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(self.transport.calls, 1)
        cards = self.published("reference.card")
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["card"]["source"], "camera")
        self.assertEqual(cards[0]["card"]["requested_by"], "voice")
        self.assertIn("teacup", cards[0]["card"]["results"][0]["title"].lower())

    def test_an_empty_buffer_returns_nothing_without_paying_a_vendor(self):
        """
        Verify the camera being off costs nothing at all (Risk 1, Risk 2).

        This is what keeps a camera-shaped phrase - "what is this word?" - from turning
        into a vendor round trip in the middle of a voice turn.
        """
        result = self.dispatcher.describe_live_frame(
            learning_session(), "room-a", question="What is this?"
        )

        self.assertIsNone(result)
        self.assertEqual(self.transport.calls, 0)
        self.assertEqual(self.published("reference.card"), [])

    def test_a_stale_buffered_frame_is_treated_as_no_frame(self):
        """Verify the TTL decides what the agent may claim to currently see."""
        clock = {"now": 1000.0}
        buffer = CameraFrameBuffer(clock=lambda: clock["now"])
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key=API_KEY, transport=self.transport),
            camera_buffer=buffer
        )
        buffer.put("room-a", FRAME)
        clock["now"] += CAMERA_FRAME_TTL_SECONDS + 1

        self.assertIsNone(dispatcher.describe_live_frame(learning_session(), "room-a"))
        self.assertEqual(self.transport.calls, 0)

    def test_a_second_question_about_the_same_frame_reuses_the_description(self):
        """
        Verify the per-frame cache bounds cost (REQ-CAM-04, Task 4.2).

        The client pushes a frame every few seconds whether or not anyone is asking; the
        vendor must only be paid once per distinct frame, not once per question.
        """
        session = learning_session()
        self.buffer.put("room-a", FRAME)

        first = self.dispatcher.describe_live_frame(session, "room-a", question="What is this?")
        second = self.dispatcher.describe_live_frame(session, "room-a", question="And this?")

        self.assertEqual(self.transport.calls, 1)
        self.assertEqual(
            first.payload["results"][0]["snippet"],
            second.payload["results"][0]["snippet"]
        )

    def test_a_new_frame_invalidates_the_cached_description(self):
        """Verify moving the camera makes the agent look again rather than repeat itself."""
        session = learning_session()

        self.buffer.put("room-a", FRAME)
        self.dispatcher.describe_live_frame(session, "room-a")
        self.buffer.put("room-a", OTHER_FRAME)
        self.dispatcher.describe_live_frame(session, "room-a")

        self.assertEqual(self.transport.calls, 2)

    def test_a_stalled_vendor_call_is_abandoned_at_the_timeout(self):
        """
        Verify a slow vision call degrades to a normal spoken reply (Risk 1).

        The bound has to hold in wall-clock terms, not just in intent: the learner is
        sitting in silence, and the Convo AI Engine's own idle_timeout is watching.
        """
        transport = SlowTransport()
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key=API_KEY, transport=transport),
            camera_buffer=self.buffer,
            camera_lookup_timeout=0.3
        )
        self.buffer.put("room-a", FRAME)

        started = time.time()
        try:
            result = dispatcher.describe_live_frame(learning_session(), "room-a")
            elapsed = time.time() - started

            self.assertIsNone(result)
            self.assertLess(elapsed, 3.0)
            self.assertEqual(self.published("reference.card"), [])
        finally:
            transport.released.set()

    def test_a_dispatcher_without_a_camera_buffer_simply_never_looks(self):
        """Verify the live path is optional, exactly like every other tool here."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key=API_KEY, transport=self.transport)
        )

        self.assertIsNone(dispatcher.describe_live_frame(learning_session(), "room-a"))
        self.assertEqual(self.transport.calls, 0)

    def test_an_unconfigured_vision_tool_looks_at_nothing(self):
        """Verify a server without a Gemini key degrades quietly, not into an exception."""
        dispatcher = ToolDispatcher(
            data_stream=self.data_stream,
            vision=CameraVisionTool(api_key="", transport=self.transport),
            camera_buffer=self.buffer
        )
        self.buffer.put("room-a", FRAME)

        self.assertIsNone(dispatcher.describe_live_frame(learning_session(), "room-a"))
        self.assertEqual(self.transport.calls, 0)


class TestCameraStreamApi(unittest.TestCase):
    """Test suite for `POST /api/session/camera/stream` (REQ-CAM-01, Task 1.2)."""

    def setUp(self):
        """A started session, with the server's own camera buffer emptied."""
        from src.server import app, server_instance

        self.client = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.original_vision = self.server.tools.vision
        self.server.camera_buffer.clear("camera-stream-channel")
        self.client.post("/api/session/start", json={
            "channel": "camera-stream-channel", "mode": "language_learning",
            "participants": ["Kenji"], "languages": ["ja"]
        })

    def tearDown(self):
        """Restore the server's tool and leave no buffered frame behind."""
        self.server.tools.vision = self.original_vision
        self.server.camera_buffer.clear("camera-stream-channel")

    def post_frame(self, **body):
        """Pushes one base64 frame as JSON."""
        payload = {
            "channel": "camera-stream-channel",
            "actor": "Kenji",
            "image_base64": base64.b64encode(FRAME).decode("ascii"),
            "mime_type": "image/jpeg",
        }
        payload.update(body)
        return self.client.post("/api/session/camera/stream", json=payload)

    def test_a_pushed_frame_is_buffered_for_the_channel(self):
        """Verify the periodic push is what gives the agent something to look at."""
        response = self.post_frame()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        entry = self.server.camera_buffer.get("camera-stream-channel")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.frame, FRAME)

    def test_pushing_a_frame_never_calls_the_vendor(self):
        """
        Verify ingestion is cheap (Task 1.2).

        This endpoint runs every few seconds for as long as Camera Assist is open; a
        vendor call here would bill the participant for looking rather than for asking.
        """
        transport = CountingTransport()
        self.server.tools.vision = CameraVisionTool(api_key=API_KEY, transport=transport)

        self.post_frame()

        self.assertEqual(transport.calls, 0)

    def test_a_multipart_push_is_accepted(self):
        """Verify what `canvas.toBlob` produces natively is uploadable as-is."""
        response = self.client.post(
            "/api/session/camera/stream?channel=camera-stream-channel&actor=Kenji",
            data={"image": (io.BytesIO(FRAME), "frame.jpg")},
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.server.camera_buffer.get("camera-stream-channel").frame, FRAME
        )

    def test_a_push_without_an_image_is_a_400(self):
        """Verify an empty push is rejected rather than silently buffering nothing."""
        response = self.post_frame(image_base64="")

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.server.camera_buffer.get("camera-stream-channel"))

    def test_a_non_participant_cannot_push_frames(self):
        """Verify the ingestion path is governed exactly like every other endpoint."""
        response = self.post_frame(actor="Stranger")

        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.server.camera_buffer.get("camera-stream-channel"))

    def test_an_unknown_session_is_a_404(self):
        """Verify frames cannot be pushed at a channel that has no session."""
        response = self.post_frame(channel="no-such-channel")

        self.assertEqual(response.status_code, 404)

    def test_closing_the_camera_stops_the_agent_from_seeing(self):
        """
        Verify the participant's own opt-out empties the buffer at once (Risk 6).

        Turning Camera Assist off must stop the agent seeing immediately, rather than
        leaving one last frame describable until the TTL runs out.
        """
        self.post_frame()

        response = self.client.post("/api/session/camera/stream", json={
            "channel": "camera-stream-channel", "actor": "Kenji", "active": False
        })

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.server.camera_buffer.get("camera-stream-channel"))


if __name__ == "__main__":
    unittest.main()
