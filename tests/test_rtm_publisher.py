"""
Summary:
    test_rtm_publisher.py covers the real server-to-browser event transport adopted in
    Phase 2 of the uiux_debugging plan (D-UIUX-2, REQ-03).

    Until this transport existed, `AgoraVoiceChannelClient.send_data_stream_message`
    only invoked local Python callbacks: every event the backend generated -
    subtitles, idiom cards, quizzes, notes, reference cards, tool status - was built
    correctly and then delivered nowhere. These tests pin the two properties that
    failure mode came down to: an event must actually be handed to a transport, and a
    transport that cannot deliver must say so rather than report success.

    The wire shape is asserted deliberately and in detail. The browser parses these
    messages with `AgoraStreamManager.handleStreamMessage`, which already expects
    `{event_type, payload, timestamp_ms}`; if the publisher's envelope drifts from
    that, every UI widget silently stops rendering while every server-side log still
    reads "sent" - which is exactly the bug this phase exists to fix.

Key Test Classes:
    - TestRtmRestPublisherConfiguration: credential gating and the refusal contract.
    - TestRtmRestPublisherWireFormat: URL, auth, and the browser-facing envelope.
    - TestRtmRestPublisherFailureHandling: vendor and network failures stay non-fatal.
    - TestAgoraClientRealDelivery: the simulation stub now really publishes.
    - TestDataStreamManagerDelivery: every event type reaches the transport.
"""

import base64
import json
import unittest
from unittest.mock import MagicMock, patch

from src.rtc.agora_client import AgoraVoiceChannelClient
from src.rtc.data_stream import DataStreamManager
from src.rtc.rtm_publisher import (
    RTM_MESSAGE_MAX_BYTES,
    RtmRestPublisher,
)


def _configured_publisher(**kwargs) -> RtmRestPublisher:
    """Builds a publisher with credentials supplied explicitly, never from the env."""
    defaults = {
        "app_id": "a" * 32,
        "customer_id": "test-customer-id",
        "customer_secret": "test-customer-secret",
    }
    defaults.update(kwargs)
    return RtmRestPublisher(**defaults)


class TestRtmRestPublisherConfiguration(unittest.TestCase):
    """A publisher without complete credentials must refuse, visibly and early."""

    def test_is_configured_true_with_all_three_credentials(self):
        self.assertTrue(_configured_publisher().is_configured)

    def test_is_configured_false_when_any_credential_missing(self):
        for missing in ("app_id", "customer_id", "customer_secret"):
            with self.subTest(missing=missing):
                publisher = _configured_publisher(**{missing: ""})
                self.assertFalse(publisher.is_configured)

    def test_unconfigured_publish_returns_false_without_calling_the_vendor(self):
        """
        An unconfigured transport must not report success.

        This is the D-UIUX-2 contract in miniature: the old stub returned True while
        delivering nothing, so nothing downstream could tell a delivered event from a
        discarded one.
        """
        publisher = _configured_publisher(customer_secret="")

        with patch("src.rtc.rtm_publisher.requests.post") as post:
            delivered = publisher.publish("tokyo-mumbai-101", "subtitles", {"text": "hi"})

        self.assertFalse(delivered)
        post.assert_not_called()


class TestRtmRestPublisherWireFormat(unittest.TestCase):
    """The request Agora receives, and the envelope the browser has to parse."""

    def setUp(self):
        self.publisher = _configured_publisher()
        self.response = MagicMock(status_code=200, text='{"result":"success"}')

    def _publish(self, event_type="subtitles", payload=None, channel="tokyo-mumbai-101"):
        with patch("src.rtc.rtm_publisher.requests.post", return_value=self.response) as post:
            delivered = self.publisher.publish(channel, event_type, payload or {"a": 1})
        return delivered, post

    def test_publish_posts_to_the_channel_messages_endpoint_for_this_app(self):
        _, post = self._publish()

        url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
        self.assertIn(f"/project/{'a' * 32}/rtm/users/", url)
        self.assertTrue(url.endswith("/channel_messages"))

    def test_publish_authenticates_with_basic_customer_credentials(self):
        """Reuses the Basic-auth pair the Convo AI control plane already uses."""
        _, post = self._publish()

        header = post.call_args.kwargs["headers"]["Authorization"]
        self.assertTrue(header.startswith("Basic "))
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        self.assertEqual(decoded, "test-customer-id:test-customer-secret")

    def test_publish_targets_the_requested_channel(self):
        _, post = self._publish(channel="osaka-delhi-202")

        self.assertEqual(post.call_args.kwargs["json"]["channel_name"], "osaka-delhi-202")

    def test_published_payload_is_the_envelope_the_browser_already_parses(self):
        """
        `AgoraStreamManager.handleStreamMessage` reads `event_type` / `payload` /
        `timestamp_ms`. Sending anything else would render nothing while still
        logging a successful send.
        """
        _, post = self._publish(event_type="quiz", payload={"question": "Which?"})

        envelope = json.loads(post.call_args.kwargs["json"]["payload"])
        self.assertEqual(envelope["event_type"], "quiz")
        self.assertEqual(envelope["payload"], {"question": "Which?"})
        self.assertIsInstance(envelope["timestamp_ms"], int)
        self.assertGreater(envelope["timestamp_ms"], 0)

    def test_non_ascii_payload_survives_the_round_trip(self):
        """A Japanese/Hindi tandem app must not mangle its own subtitles."""
        _, post = self._publish(payload={"original_text": "一期一会ですね"})

        envelope = json.loads(post.call_args.kwargs["json"]["payload"])
        self.assertEqual(envelope["payload"]["original_text"], "一期一会ですね")

    def test_publish_returns_true_when_the_vendor_accepts_the_message(self):
        delivered, _ = self._publish()
        self.assertTrue(delivered)


class TestRtmRestPublisherFailureHandling(unittest.TestCase):
    """A failing transport must degrade the UI, never the conversation."""

    def setUp(self):
        self.publisher = _configured_publisher()

    def test_vendor_rejection_returns_false_rather_than_raising(self):
        response = MagicMock(status_code=401, text='{"result":"failed"}')

        with patch("src.rtc.rtm_publisher.requests.post", return_value=response):
            delivered = self.publisher.publish("c", "subtitles", {"text": "hi"})

        self.assertFalse(delivered)

    def test_network_failure_returns_false_rather_than_raising(self):
        """
        This runs on the turn path. An unreachable Agora must cost a subtitle, not
        the learner's answer.
        """
        with patch("src.rtc.rtm_publisher.requests.post", side_effect=OSError("boom")):
            delivered = self.publisher.publish("c", "subtitles", {"text": "hi"})

        self.assertFalse(delivered)

    def test_oversized_message_is_refused_locally(self):
        """
        Agora caps a channel message at 32 KB. Refusing here keeps the reason
        legible instead of surfacing it as an opaque vendor rejection.
        """
        payload = {"text": "x" * (RTM_MESSAGE_MAX_BYTES + 1024)}

        with patch("src.rtc.rtm_publisher.requests.post") as post:
            delivered = self.publisher.publish("c", "subtitles", payload)

        self.assertFalse(delivered)
        post.assert_not_called()


class TestAgoraClientRealDelivery(unittest.TestCase):
    """
    D-UIUX-2 itself: `send_data_stream_message` has to reach a real transport.
    """

    def setUp(self):
        self.publisher = MagicMock(spec=RtmRestPublisher)
        self.publisher.is_configured = True
        self.publisher.publish.return_value = True
        self.client = AgoraVoiceChannelClient(
            channel_name="tokyo-mumbai-101", rtm_publisher=self.publisher
        )
        self.client.join_channel()

    def test_send_data_stream_message_publishes_to_the_real_transport(self):
        self.client.send_data_stream_message("subtitles", {"original_text": "hello"})

        self.publisher.publish.assert_called_once_with(
            "tokyo-mumbai-101", "subtitles", {"original_text": "hello"}
        )

    def test_local_callbacks_still_fire_alongside_real_delivery(self):
        """
        The in-process listeners predate this transport and the simulation demo still
        depends on them; adding delivery must not remove them.
        """
        seen = []
        self.client.register_on_data_stream(lambda packet: seen.append(packet.event_type))

        self.client.send_data_stream_message("quiz", {"question": "Which?"})

        self.assertEqual(seen, ["quiz"])

    def test_a_client_without_a_publisher_keeps_its_previous_behavior(self):
        """
        Every existing test and the offline/simulated path construct this client with
        no publisher at all; that has to keep working exactly as before.
        """
        client = AgoraVoiceChannelClient(channel_name="c")
        client.join_channel()
        seen = []
        client.register_on_data_stream(lambda packet: seen.append(packet.event_type))

        self.assertTrue(client.send_data_stream_message("subtitles", {"text": "hi"}))
        self.assertEqual(seen, ["subtitles"])

    def test_nothing_is_published_when_the_client_is_not_in_a_channel(self):
        client = AgoraVoiceChannelClient(channel_name="c", rtm_publisher=self.publisher)

        self.assertFalse(client.send_data_stream_message("subtitles", {"text": "hi"}))
        self.publisher.publish.assert_not_called()


class TestDataStreamManagerDelivery(unittest.TestCase):
    """Every event family the manager exposes must reach the browser transport."""

    def setUp(self):
        self.publisher = MagicMock(spec=RtmRestPublisher)
        self.publisher.is_configured = True
        self.publisher.publish.return_value = True
        client = AgoraVoiceChannelClient(
            channel_name="tokyo-mumbai-101", rtm_publisher=self.publisher
        )
        client.join_channel()
        self.manager = DataStreamManager(client)

    def test_each_event_family_is_delivered_under_its_own_event_type(self):
        """
        Subtitles and quizzes were the reported symptoms (BUG-3/BUG-4), but tool and
        artifact events travelled the same dead path and are equally part of the fix.
        """
        self.manager.send_subtitle(speaker="Kenji", text="一期一会")
        self.manager.send_idiom_card(phrase="一期一会")
        self.manager.send_quiz(question="Which?", options=["a", "b"])
        self.manager.send_artifact_event("note.upserted", {"note": {"id": "n1"}})
        self.manager.send_tool_event("reference.card", {"card": {"query": "q"}})
        self.manager.send_translation_event("translation.output_transcript", {"leg": "1"})

        delivered = [call.args[1] for call in self.publisher.publish.call_args_list]
        self.assertEqual(
            delivered,
            [
                "subtitles",
                "idiom_card",
                "quiz",
                "note.upserted",
                "reference.card",
                "translation.output_transcript",
            ],
        )


if __name__ == "__main__":
    unittest.main()
