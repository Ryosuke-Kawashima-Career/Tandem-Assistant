"""
Summary:
    test_translation_streaming.py is the executable specification for TASK-11.8: a
    Gemini Live Translate leg is full-duplex and continuous, never a blocking
    send-then-wait request/response (REQ-17, spec 1.11.0).

    Before this task, `GeminiLiveTranslateSession` exposed `send_audio()` and
    `receive()` as two separate methods nothing interleaved - a caller had to poll
    `receive()` by hand, and nothing did, so no translated audio or transcript could
    ever reach a listener no matter how correct the wire mapping was. This suite
    verifies the concurrent reader that closes that gap: it drains the socket on its
    own thread from the moment a leg connects, dispatches events as they arrive without
    waiting for a full turn, keeps running independently of `send_audio()`, and resumes
    itself after a reconnect rather than leaving a leg silently but permanently
    degraded after one transient drop.

Covers:
    - REQ-17 continuous dispatch: events reach a listener without polling.
    - REQ-17 ordering: an interim transcript arrives before the matching final one.
    - Full duplex: sending is not blocked by a reader parked on the same socket.
    - `close()` releases a reader blocked in a live read, rather than hanging forever.
    - A leg that reconnects resumes its reader on the new socket automatically.
    - The router wires every connected leg's reader to its own event handlers, and
      keeps its leg-health bookkeeping in sync with reader-detected failure/recovery.
"""

import json
import queue
import threading
import unittest
from unittest.mock import MagicMock

from src.rtc.data_stream import DataStreamManager
from src.sessions.models import SessionRecord
from src.translation.gemini_live import GeminiLiveTranslateSession, LegConfig, LegState
from src.translation.router import EVENT_STATUS, Participant, TranslationRouter

# Every wait in this suite is bounded: a hang here means the production code is
# deadlocked or leaking a thread, and the test must fail fast rather than block CI.
WAIT_SECONDS = 2.0


def leg(leg_id="leg-a-to-b", speaker_id="peer-a", target_language="ja"):
    return LegConfig(leg_id=leg_id, speaker_id=speaker_id, target_language=target_language,
                     recipients=("peer-b",))


class QueueConnection:
    """
    A fake WebSocket whose `.recv()` genuinely blocks, like a real socket's.

    `queue.Queue.get()` blocks until an item exists, which is what makes the duplex and
    close-releases-a-blocked-reader tests meaningful rather than trivially true: a fake
    that just returns immediately would pass those tests even if `send_audio` secretly
    waited for a read to complete first.
    """

    _CLOSED = object()

    def __init__(self):
        self.inbox: "queue.Queue" = queue.Queue()
        self.sent = []
        self.closed = False

    def send(self, message):
        if self.closed:
            raise ConnectionError("cannot send on a closed connection")
        self.sent.append(message)

    def recv(self):
        item = self.inbox.get()
        if item is QueueConnection._CLOSED:
            raise ConnectionError("connection closed")
        return item

    def close(self):
        self.closed = True
        self.inbox.put(QueueConnection._CLOSED)

    def push(self, message: str) -> None:
        """Test helper: makes the next `.recv()` return this message."""
        self.inbox.put(message)


def input_transcript(text: str, final: bool) -> str:
    key = "inputTranscription" if final else "interimInputTranscription"
    return json.dumps({"serverContent": {key: {"text": text}}})


class RecordingConnector:
    """Hands out pre-built `QueueConnection`s, one per connect call, in order."""

    def __init__(self, connections=None):
        self.connections = list(connections) if connections else [QueueConnection()]
        self._index = 0

    def __call__(self, url, **kwargs):
        connection = self.connections[min(self._index, len(self.connections) - 1)]
        self._index += 1
        return connection


def collector():
    """
    A thread-safe event sink with two waiters bounded by WAIT_SECONDS.

    `wait_for(count)` waits for *at least* `count` events total - it says nothing about
    *which* ones, so a test racing against an incidental event (a `leg_state_changed`
    landing before the transcript it actually cares about) must use `wait_until` with a
    predicate instead, or it can return early having counted the wrong events.
    """
    import time

    events = []
    ready = threading.Event()
    lock = threading.Lock()

    def on_event(event):
        with lock:
            events.append(event)
        ready.set()

    def wait_for(count, timeout=WAIT_SECONDS):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with lock:
                if len(events) >= count:
                    return list(events)
            ready.wait(timeout=0.05)
            ready.clear()
        with lock:
            return list(events)

    def wait_until(predicate, timeout=WAIT_SECONDS):
        """Waits until any collected event satisfies `predicate`, or times out."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with lock:
                if any(predicate(e) for e in events):
                    return list(events)
            ready.wait(timeout=0.05)
            ready.clear()
        with lock:
            return list(events)

    return on_event, events, wait_for, wait_until


class TestReaderDispatchesContinuously(unittest.TestCase):
    """The reader drains the socket on its own, with no caller polling it."""

    def test_a_pushed_message_reaches_the_callback_without_polling(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        on_event, _events, wait_for, wait_until = collector()
        session.start_reader(on_event)

        connection.push(input_transcript("ohayou", final=True))

        events = wait_for(1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "input_transcript")
        self.assertEqual(events[0]["text"], "ohayou")

    def test_interim_transcript_reaches_the_callback_before_the_final_one(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        on_event, _events, wait_for, wait_until = collector()
        session.start_reader(on_event)

        connection.push(input_transcript("ohay", final=False))
        connection.push(input_transcript("ohayou", final=True))

        events = wait_for(2)
        self.assertEqual([e["is_final"] for e in events], [False, True])
        self.assertEqual([e["text"] for e in events], ["ohay", "ohayou"])

    def test_start_reader_refuses_on_an_inactive_leg(self):
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector())
        self.assertFalse(session.start_reader(lambda event: None))

    def test_start_reader_refuses_a_second_reader_on_the_same_leg(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        self.assertTrue(session.start_reader(lambda event: None))
        self.assertFalse(session.start_reader(lambda event: None))
        session.close()

    def test_one_bad_handler_call_does_not_kill_the_reader(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        calls = []

        def flaky(event):
            calls.append(event)
            if len(calls) == 1:
                raise ValueError("boom")

        session.start_reader(flaky)
        connection.push(input_transcript("one", final=True))
        connection.push(input_transcript("two", final=True))

        deadline_events = None
        import time
        start = time.monotonic()
        while time.monotonic() - start < WAIT_SECONDS and len(calls) < 2:
            time.sleep(0.02)

        self.assertEqual(len(calls), 2)
        session.close()


class TestFullDuplex(unittest.TestCase):
    """Sending and reading run independently over the same socket."""

    def test_send_audio_does_not_block_on_a_reader_parked_in_recv(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        session.start_reader(lambda event: None)  # immediately blocks on the empty queue

        import time
        start = time.monotonic()
        sent = session.send_audio(b"\x00\x00" * 1600)
        elapsed = time.monotonic() - start

        self.assertTrue(sent)
        self.assertLess(elapsed, 0.5, "send_audio waited on the reader instead of "
                                      "using the socket's independent send half")
        session.close()

    def test_close_returns_promptly_while_the_reader_is_blocked_on_recv(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        session.start_reader(lambda event: None)

        import time
        start = time.monotonic()
        session.close()
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, WAIT_SECONDS, "close() hung instead of releasing the "
                                               "blocked reader")
        self.assertEqual(session.state, LegState.CLOSED)

    def test_the_reader_thread_has_exited_after_close_returns(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        session.start_reader(lambda event: None)

        session.close()

        self.assertIsNone(session._reader_thread)


class TestReconnectResumesTheReader(unittest.TestCase):
    """A transient drop must not permanently end a leg's duplex stream."""

    def test_reader_resumes_on_the_new_socket_after_a_reconnect(self):
        dead = QueueConnection()
        replacement = QueueConnection()
        session = GeminiLiveTranslateSession(
            config=leg(), api_key="k",
            connector=RecordingConnector([dead, replacement])
        )
        session.connect()
        on_event, _events, wait_for, wait_until = collector()
        session.start_reader(on_event)

        # Kill the first socket; the reader (parked in dead.recv()) exits on its own,
        # and send_audio's own retry path re-establishes on `replacement`.
        dead.closed = True
        self.assertTrue(session.send_audio(b"\x00\x00" * 1600))

        # The reader must be running again, now against the new connection. A
        # `leg_state_changed` event or two may land first (the old reader's own exit,
        # the reconnect's recovery notice) - wait for the transcript specifically
        # rather than for a raw count that an incidental event could satisfy early.
        replacement.push(input_transcript("namaste", final=True))
        all_events = wait_until(lambda e: e.get("type") == "input_transcript")
        transcripts = [e for e in all_events if e.get("type") == "input_transcript"]
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(transcripts[0]["text"], "namaste")
        session.close()

    def test_reconnect_emits_a_recovered_leg_state_event(self):
        dead = QueueConnection()
        replacement = QueueConnection()
        session = GeminiLiveTranslateSession(
            config=leg(), api_key="k",
            connector=RecordingConnector([dead, replacement])
        )
        session.connect()
        on_event, _events, wait_for, wait_until = collector()
        session.start_reader(on_event)

        dead.closed = True
        session.send_audio(b"\x00\x00" * 1600)

        # The old reader's own organic-degrade notice may also land here (its exit and
        # the reconnect race independently) - assert the recovery notice arrived, not
        # that it was the only `leg_state_changed` event this sequence ever produces.
        all_events = wait_until(
            lambda e: e.get("type") == "leg_state_changed"
            and e.get("state") == LegState.ACTIVE.value
        )
        recovered = [e for e in all_events if e.get("type") == "leg_state_changed"
                    and e.get("state") == LegState.ACTIVE.value]
        self.assertEqual(len(recovered), 1)
        session.close()


class TestOrganicFailureNotifiesTheCallback(unittest.TestCase):
    """A leg that dies on its own (not via close()) must say so."""

    def test_a_receive_failure_emits_a_leg_state_changed_event(self):
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        on_event, _events, wait_for, wait_until = collector()
        session.start_reader(on_event)

        connection.closed = True
        connection.inbox.put(QueueConnection._CLOSED)

        events = wait_for(1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "leg_state_changed")
        self.assertEqual(events[0]["state"], LegState.DEGRADED.value)

    def test_an_explicit_close_does_not_emit_a_leg_state_changed_event(self):
        """close() means the router already knows and is about to forget this leg."""
        connection = QueueConnection()
        session = GeminiLiveTranslateSession(config=leg(), api_key="k",
                                             connector=RecordingConnector([connection]))
        session.connect()
        on_event, events, _wait_for, _wait_until = collector()
        session.start_reader(on_event)

        session.close()

        self.assertEqual(events, [])


class FakeLegSession:
    """
    A router-facing leg double whose reader is fired manually by the test.

    Deliberately synchronous: the threading behaviour above is this file's own
    responsibility to verify at the `GeminiLiveTranslateSession` level, so the router
    tests below only need to confirm the *wiring* - that `start()` registers a reader
    and that the callback it hands over dispatches to the correct router method.
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.state = LegState.IDLE
        self.closed = False
        self.on_event = None

    def connect(self):
        self.state = LegState.ACTIVE
        return True

    def send_audio(self, chunk):
        return True

    def start_reader(self, on_event):
        self.on_event = on_event
        return True

    def stop_reader(self, timeout=2.0):
        self.on_event = None

    def close(self):
        self.closed = True
        self.state = LegState.CLOSED

    def fire(self, event):
        assert self.on_event is not None, "router never started this leg's reader"
        self.on_event(event)


class RecordingFactory:
    def __init__(self):
        self.sessions = {}

    def __call__(self, config, **kwargs):
        session = FakeLegSession(config, **kwargs)
        self.sessions[config.leg_id] = session
        return session


PEER_A = Participant(participant_id="peer-a", language="hi")
PEER_B = Participant(participant_id="peer-b", language="ja")


def learning_session():
    return SessionRecord.create(channel="c1", mode="language_learning",
                                languages=["hi", "ja"])


class TestRouterWiresTheReader(unittest.TestCase):
    """TASK-11.8: the router starts and dispatches every connected leg's reader."""

    def test_start_registers_a_reader_for_every_connected_leg(self):
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), session_factory=factory)

        router.start([PEER_A, PEER_B])

        self.assertTrue(all(s.on_event is not None for s in factory.sessions.values()))

    def test_an_audio_event_is_dispatched_to_handle_translated_audio(self):
        publisher = MagicMock()
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), session_factory=factory)
        router.audio_publisher = publisher
        router.start([PEER_A, PEER_B])
        # language_learning defaults the translated-audio gate off (REQ-17); open it so
        # this test observes the dispatch wiring rather than the (separately tested) gate.
        router.set_translated_audio_enabled("peer-b", True)
        leg_a = router.leg_for_speaker("peer-a")

        factory.sessions[leg_a.leg_id].fire({"type": "audio", "audio": b"\x00\x00" * 2400})

        publisher.assert_called()

    def test_a_transcript_event_is_dispatched_to_the_input_handler(self):
        data_stream = MagicMock(spec=DataStreamManager)
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), data_stream=data_stream,
                                   session_factory=factory)
        router.start([PEER_A, PEER_B])
        leg_a = router.leg_for_speaker("peer-a")
        data_stream.reset_mock()

        factory.sessions[leg_a.leg_id].fire(
            {"type": "input_transcript", "text": "namaste", "is_final": True}
        )

        published = [c.args[0] for c in data_stream.send_translation_event.call_args_list]
        self.assertIn("translation.input_transcript", published)

    def test_an_output_transcript_event_is_dispatched_to_the_output_handler(self):
        data_stream = MagicMock(spec=DataStreamManager)
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), data_stream=data_stream,
                                   session_factory=factory)
        router.start([PEER_A, PEER_B])
        leg_a = router.leg_for_speaker("peer-a")
        data_stream.reset_mock()

        factory.sessions[leg_a.leg_id].fire(
            {"type": "output_transcript", "text": "hello", "is_final": True}
        )

        published = [c.args[0] for c in data_stream.send_translation_event.call_args_list]
        self.assertIn("translation.output_transcript", published)

    def test_a_leg_state_changed_degraded_event_marks_the_leg_degraded(self):
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), session_factory=factory)
        router.start([PEER_A, PEER_B])
        leg_a = router.leg_for_speaker("peer-a")

        factory.sessions[leg_a.leg_id].fire(
            {"type": "leg_state_changed", "state": "degraded", "reason": "read failed"}
        )

        self.assertEqual(router.leg_states()[leg_a.leg_id], LegState.DEGRADED.value)

    def test_a_leg_state_changed_active_event_recovers_the_leg(self):
        factory = RecordingFactory()
        router = TranslationRouter(session=learning_session(), session_factory=factory)
        router.start([PEER_A, PEER_B])
        leg_a = router.leg_for_speaker("peer-a")
        router.degrade_leg(leg_a.leg_id, reason="read failed")

        factory.sessions[leg_a.leg_id].fire(
            {"type": "leg_state_changed", "state": "active", "reason": "reconnected"}
        )

        self.assertEqual(router.leg_states()[leg_a.leg_id], LegState.ACTIVE.value)

    def test_no_reader_is_started_for_an_unavailable_leg(self):
        from src.translation.gemini_live import TranslationUnavailableError

        made = {}

        def unavailable_factory(config, **kwargs):
            session = FakeLegSession(config)
            made[config.leg_id] = session

            def connect():
                session.state = LegState.UNAVAILABLE
                raise TranslationUnavailableError("quota exhausted")

            session.connect = connect
            return session

        router = TranslationRouter(session=learning_session(),
                                   session_factory=unavailable_factory)
        router.start([PEER_A, PEER_B])

        self.assertTrue(made)
        self.assertTrue(all(session.on_event is None for session in made.values()))


if __name__ == "__main__":
    unittest.main()
