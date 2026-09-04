"""
Summary:
    gemini_live.py is EchoSphere's server-side Gemini Live Translate leg (REQ-17,
    TASK-11.1): one stateful WebSocket carrying one speaker's audio in and one target
    language's translated audio plus transcripts out.

    Three properties of the setup contract are load-bearing and easy to get wrong:

    1. No source language is configured anywhere. Gemini detects it per utterance, which
       is exactly what a Hinglish or Japanglish speaker needs - a pinned source would
       break the moment they code-switch mid-sentence, which in this product is not an
       edge case but the material.
    2. `echoTargetLanguage=false`. Without it, speech already in the target language is
       re-synthesized straight back into the translated track. That is the normal case
       for the `international_work` English pivot, where every leg targets English and
       participants routinely already speak it.
    3. `GEMINI_API_KEY` lives in the connection URL and nowhere else. Nothing in the setup
       message, and therefore nothing reachable from a client, ever carries it (REQ-08).

    Failures fail closed rather than loudly: an unreachable Live endpoint must degrade
    this leg, not take down the voice call it is attached to (TASK-11.6).

Key Classes:
    - GeminiLiveTranslateSession: one leg's WebSocket lifecycle.
    - LegConfig / LegState: what a leg is, and where it currently stands.
    - TranslationUnavailableError: the fail-closed signal the router catches.
"""

import base64
import binascii
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("echosphere.translation.gemini_live")

# Live Translate model. Environment-configured on purpose: the Live preview lineup moves
# independently of this repository, so freezing a name here would date the whole module.
DEFAULT_LIVE_TRANSLATE_MODEL = "gemini-3.5-live-translate-preview"
LIVE_TRANSLATE_MODEL_ENV = "GEMINI_LIVE_TRANSLATE_MODEL"

# The v1beta bidirectional endpoint. The key travels as a query parameter because that is
# the only auth the Live WebSocket accepts; it never appears in any message body.
LIVE_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# BCP-47 tags for the three supported languages (spec section 1). `en-US` is the default
# English tag; `en-IN` is equally valid on a leg whose participants are Indian English
# speakers, so a caller that already holds a qualified tag keeps it.
BCP47_BY_LANGUAGE: Dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "hi": "hi-IN",
}

# How many times one leg re-establishes its socket before reporting itself degraded.
# One is deliberate: a single retry covers the ordinary transient drop, while a longer
# retry ladder would keep a live speaker waiting on a leg that is not coming back.
DEFAULT_MAX_RECONNECT_ATTEMPTS = 1


class TranslationUnavailableError(RuntimeError):
    """
    Raised when a leg cannot be established (auth, quota, or outage).

    Distinct from a generic error because the router treats it as a state rather than a
    fault: the session proceeds without translated audio (REQ-17).
    """


class LegState(str, Enum):
    """
    Where one translation leg currently stands.

    Subclasses `str` so a state serializes as its own wire value into
    `translation.status` events with no conversion at the boundary - the same reason
    `SessionMode` does.
    """

    IDLE = "idle"
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


@dataclass(frozen=True)
class LegConfig:
    """
    One translation leg: whose audio it carries, into what language, and for whom.

    Frozen because a leg's identity is what the router routes by. Changing a speaker or a
    target language mid-flight is a different leg, and the router closes and replaces it
    rather than mutating this in place.
    """

    leg_id: str
    speaker_id: str
    target_language: str
    recipients: Tuple[str, ...] = ()
    echo_target_language: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def to_bcp47(language: str) -> str:
    """
    Resolves a language code to the BCP-47 tag the Live setup message requires.

    Algorithm:
    1. Pass through an already region-qualified tag whose primary subtag is supported.
    2. Map a bare supported code to its default region.
    3. Raise on anything else rather than guessing - a wrong target language is a whole
       session translated into a language nobody in the room reads.
    """
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"Unsupported translation language: {language!r}")

    normalized = language.strip()
    primary = normalized.split("-", 1)[0].lower()
    if primary not in BCP47_BY_LANGUAGE:
        raise ValueError(
            f"Unsupported translation language {language!r}; "
            f"expected one of {sorted(BCP47_BY_LANGUAGE)}."
        )

    if "-" in normalized:
        return normalized
    return BCP47_BY_LANGUAGE[primary]


def resolve_model() -> str:
    """Returns the configured Live Translate model, or the current default."""
    return os.getenv(LIVE_TRANSLATE_MODEL_ENV) or DEFAULT_LIVE_TRANSLATE_MODEL


def parse_server_message(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalizes one Live server message into the events the router acts on.

    Algorithm:
    1. Decode bytes, then parse JSON; a malformed frame yields no events rather than
       raising, since one bad frame must not tear down a live leg.
    2. Emit `input_transcript` / `output_transcript` for the transcription streams.
    3. Emit `audio` for each inline PCM part, decoded to raw bytes.
    4. Emit `turn_complete` when the server closes the turn.

    Finality is carried by *which field arrived*, not by a flag: the transcription object
    holds only `text` and `languageCode`. `inputTranscription` is already the finalized
    transcript, while `interimInputTranscription` is the low-latency partial that updates
    while the speaker is still talking. Reading finality from a non-existent `finished`
    key leaves every transcript interim forever, and the once-only ingestion into
    `TeachingAgent` and the artifacts then never fires at all.

    `languageCode` is passed through because it is what Gemini actually detected. On a leg
    with no configured source - which is every leg here - that is the only true source
    language, as opposed to the language we merely expected this speaker to use.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return []

    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        logger.debug("Discarding unparseable Gemini Live frame.")
        return []

    if not isinstance(message, dict):
        return []

    content = message.get("serverContent") or {}
    if not isinstance(content, dict):
        return []

    turn_complete = bool(content.get("turnComplete"))
    events: List[Dict[str, Any]] = []

    for key, event_type, is_final in (
        ("interimInputTranscription", "input_transcript", False),
        ("inputTranscription", "input_transcript", True),
        ("interimOutputTranscription", "output_transcript", False),
        ("outputTranscription", "output_transcript", True),
    ):
        block = content.get(key)
        if isinstance(block, dict) and block.get("text"):
            events.append({
                "type": event_type,
                "text": block["text"],
                "language_code": block.get("languageCode", ""),
                "is_final": is_final,
            })

    parts = ((content.get("modelTurn") or {}).get("parts") or []
             if isinstance(content.get("modelTurn"), dict) else [])
    for part in parts:
        inline = part.get("inlineData") if isinstance(part, dict) else None
        if not isinstance(inline, dict) or not inline.get("data"):
            continue
        try:
            audio = base64.b64decode(inline["data"])
        except (binascii.Error, ValueError):
            logger.warning("Discarding an undecodable Gemini Live audio part.")
            continue
        events.append({
            "type": "audio",
            "audio": audio,
            "mime_type": inline.get("mimeType", ""),
            "is_final": turn_complete,
        })

    if turn_complete:
        events.append({"type": "turn_complete", "text": "", "is_final": True})

    return events


def _default_connector(url: str, **kwargs) -> Any:
    """
    Opens a real Live WebSocket.

    Imported lazily so the module - and every test that injects a fake connector - stays
    usable without a network stack or an installed `websockets` build.
    """
    from websockets.sync.client import connect  # noqa: WPS433 (deliberately lazy)

    return connect(url, **kwargs)


class GeminiLiveTranslateSession:
    """
    One Gemini Live Translate leg: connect, stream audio in, read translation out.

    The socket is injected via `connector` rather than constructed here so the wire
    contract can be exercised without a network, and so a future async transport can be
    swapped in without touching the setup or reconnect logic.
    """

    def __init__(
        self,
        config: LegConfig,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        connector: Optional[Callable[..., Any]] = None,
        endpoint: str = LIVE_ENDPOINT,
        max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS
    ):
        """
        Initialize one leg without opening anything.

        Algorithm:
        1. Bind the leg configuration and resolve the model and credential.
        2. Bind the socket factory, defaulting to the real WebSocket client.
        3. Start in `idle`: nothing is open until `connect()` is called.
        """
        self.config = config
        self.model = model or resolve_model()
        self.endpoint = endpoint
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self._connector = connector or _default_connector
        self.max_reconnect_attempts = max_reconnect_attempts

        self._connection: Optional[Any] = None
        self.state = LegState.IDLE
        self.unavailable_reason = ""
        self.reconnect_count = 0

        # Full-duplex reader (REQ-17, TASK-11.8). `_on_event` is retained across a
        # reconnect specifically so `_ensure_reader_running` can resume streaming on the
        # new socket without the caller having to call `start_reader` a second time.
        self._on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

    # -- Wire contract -------------------------------------------------------------

    def build_setup_message(self) -> Dict[str, Any]:
        """
        Builds the Live setup frame for this leg (spec 3.1).

        Algorithm:
        1. Name the resolved model, normalizing it to the `models/...` resource form.
        2. Request AUDIO-only responses; text would be a second, divergent subtitle.
        3. Enable both transcriptions - they are the REQ-17 transcript events.
        4. Set the BCP-47 target and disable target-language echo.

        Every one of those blocks belongs inside `generationConfig`. Hoisting any of them
        to the top level of `setup` is accepted and ignored, so the leg connects, streams,
        and answers - untranslated, untranscribed, and without an error anywhere.

        Carries no source language and no credential, deliberately: see the module
        summary for why each of those absences matters.
        """
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        return {
            "setup": {
                "model": model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                    "translationConfig": {
                        "targetLanguageCode": to_bcp47(self.config.target_language),
                        "echoTargetLanguage": bool(self.config.echo_target_language),
                    },
                },
            }
        }

    def build_audio_message(self, chunk: bytes) -> Dict[str, Any]:
        """Wraps one PCM16 mono 16 kHz chunk as a realtime input frame."""
        from src.translation.audio import INPUT_SAMPLE_RATE

        return {
            "realtimeInput": {
                "audio": {
                    "mimeType": f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            }
        }

    @property
    def connection_url(self) -> str:
        """The authenticated endpoint URL. Server-side only (REQ-08)."""
        return f"{self.endpoint}?key={self.api_key}"

    # -- Lifecycle -----------------------------------------------------------------

    def connect(self) -> bool:
        """
        Opens the socket and sends the setup frame.

        Algorithm:
        1. Refuse without a credential - `unavailable`, not a crash deep in the socket.
        2. Open the socket through the injected connector.
        3. Send the setup frame first; the Live session is not usable before it lands.
        4. On any failure, record `unavailable` with the reason and raise
           TranslationUnavailableError for the router to convert into a status event.

        Raises:
            TranslationUnavailableError: the leg could not be established.
        """
        if not self.api_key:
            return self._mark_unavailable("GEMINI_API_KEY is not configured")

        try:
            self._connection = self._connector(self.connection_url)
            self._connection.send(json.dumps(self.build_setup_message()))
        except Exception as error:  # noqa: BLE001 - any transport failure fails closed
            self._connection = None
            return self._mark_unavailable(str(error) or error.__class__.__name__)

        self.state = LegState.ACTIVE
        self.unavailable_reason = ""
        logger.info(
            "Translation leg '%s' active (%s -> %s).",
            self.config.leg_id, self.config.speaker_id, self.config.target_language
        )
        return True

    def send_audio(self, chunk: bytes) -> bool:
        """
        Sends one 100 ms input chunk, reconnecting once if the socket has dropped.

        Algorithm:
        1. Refuse when the leg is closed or unavailable, or nothing is open.
        2. Attempt the send.
        3. On a transport error, re-establish the socket and retry once; if that fails
           too, mark the leg degraded and return False. The router keeps the other legs
           and the original Agora audio running either way.
        4. On success, resume the reader if a reconnect just replaced the socket out from
           under it - `_reconnect()` closes the old connection, which is exactly what
           unblocks a reader thread parked in `receive()` on it, so that thread has
           already exited by the time step 4 runs. Without this, one transient drop
           would permanently end the leg's translated audio and transcripts even though
           sending itself recovered (TASK-11.8).

        Runs independently of the reader: this method only ever touches the socket's
        send half, so a leg accepts new audio while its own reader is mid-`recv()` on
        the same socket - the send and receive halves of a WebSocket are independent.
        """
        if self.state in (LegState.CLOSED, LegState.UNAVAILABLE) or self._connection is None:
            return False

        message = json.dumps(self.build_audio_message(chunk))
        reconnected = False
        for attempt in range(self.max_reconnect_attempts + 1):
            try:
                self._connection.send(message)
                self.state = LegState.ACTIVE
                if reconnected:
                    self._resume_reader_after_reconnect()
                    self._notify_event({
                        "type": "leg_state_changed",
                        "state": LegState.ACTIVE.value,
                        "reason": "reconnected",
                    })
                return True
            except Exception as error:  # noqa: BLE001 - transport-level, recoverable
                logger.warning(
                    "Translation leg '%s' send failed (%s); attempt %d.",
                    self.config.leg_id, error, attempt + 1
                )
                if attempt >= self.max_reconnect_attempts or not self._reconnect():
                    self.state = LegState.DEGRADED
                    return False
                reconnected = True
        return False

    def receive(self) -> List[Dict[str, Any]]:
        """
        Reads one server frame and returns its normalized events.

        Returns an empty list rather than blocking forever or raising when the socket is
        not readable: the caller is a routing loop, not an error handler.

        Snapshots `self._connection` into a local before the blocking call: this method
        runs on the reader thread while `close()`/`_reconnect()` run on whatever thread
        called `send_audio()`, and either can null out or replace `self._connection`
        while this call is parked in `.recv()`. Reading the attribute twice would let
        that swap turn a blocking call into an `AttributeError` on `None` instead of the
        clean "socket closed" exception the local reference still raises.
        """
        connection = self._connection
        if connection is None or self.state is not LegState.ACTIVE:
            return []
        try:
            return parse_server_message(connection.recv())
        except Exception as error:  # noqa: BLE001 - a read failure degrades this leg only
            logger.warning("Translation leg '%s' receive failed: %s",
                           self.config.leg_id, error)
            self.state = LegState.DEGRADED
            return []

    def start_reader(self, on_event: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Starts this leg's full-duplex reader (REQ-17, TASK-11.8).

        Algorithm:
        1. Refuse when the leg is not active, or a reader is already running - one
           leg runs one reader.
        2. Remember `on_event` so a later reconnect can restart the reader on the leg's
           behalf without the caller calling this again.
        3. Spawn a daemon thread draining `receive()` continuously and forwarding every
           event to `on_event` as it arrives.

        This is what makes the leg full-duplex in practice: `send_audio()` and this
        reader run on independent threads over the same socket, so a translated
        response can surface while the caller is still feeding this leg new audio -
        never a blocking send-then-wait request/response (spec 1.11.0).
        """
        if self.state is not LegState.ACTIVE:
            return False
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return False

        self._on_event = on_event
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"gemini-leg-reader-{self.config.leg_id}",
            daemon=True
        )
        self._reader_thread.start()
        return True

    def stop_reader(self, timeout: float = 2.0) -> None:
        """Signals the reader to stop and waits briefly for it to exit."""
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)
            self._reader_thread = None

    def close(self) -> None:
        """
        Closes the socket. Idempotent: a leg is closed on participant leave, on language
        change, and on session stop, and those can legitimately coincide.

        Algorithm:
        1. Signal the reader to stop before touching the socket, so a normal exit never
           races the "did the socket just die on its own" path below.
        2. Close the connection - this is what unblocks a reader thread parked in
           `receive()`'s blocking read, not the stop signal alone.
        3. Join the reader so a caller that awaits `close()` knows the thread is gone,
           not still mid-callback against a leg the router is about to forget.
        """
        self._reader_stop.set()

        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as error:  # noqa: BLE001 - already-dead sockets are fine
                logger.debug("Translation leg '%s' close raised: %s",
                             self.config.leg_id, error)
            self._connection = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        self.state = LegState.CLOSED

    # -- Internals -----------------------------------------------------------------

    def _reader_loop(self) -> None:
        """
        The reader thread body: drains events until stopped or this leg leaves `active`.

        Algorithm:
        1. Loop while neither `stop_reader()` was called nor `receive()` moved the leg
           out of `active` (a read failure sets `degraded` internally, which this same
           condition catches without a separate check).
        2. Forward every event from one `receive()` call before blocking on the next -
           `receive()` already returns a whole decoded message's events together, so
           this preserves the order Gemini emitted them in.
        3. On exit, tell the router only when the leg died on its own: an explicit
           `close()`/`stop_reader()` already means the router knows and is about to
           forget this leg, so a status event at that point would just be noise about a
           leg nobody is tracking any more.
        """
        while not self._reader_stop.is_set() and self.state is LegState.ACTIVE:
            for event in self.receive():
                if self._reader_stop.is_set():
                    break
                self._notify_event(event)

        if not self._reader_stop.is_set():
            self._notify_event({
                "type": "leg_state_changed",
                "state": self.state.value,
                "reason": self.unavailable_reason or "read failed",
            })

    def _resume_reader_after_reconnect(self) -> None:
        """
        Restarts the reader on the new socket after `_reconnect()` replaced it
        (TASK-11.8).

        `_reconnect()` closes the old connection, and closing it is what unblocks a
        reader thread parked in `receive()` on that connection - but that unblocking
        happens on the reader's own thread, asynchronously with this one. A bare
        `is_alive()` check here would race it: this method can run before the old
        reader has actually finished exiting, and would then silently skip starting a
        new one, leaving the leg with no reader at all despite `send_audio` reporting
        success. Joining briefly (bounded, since the old connection is already closed
        and the reader is already unblocking) closes that race.

        Only called from the reconnect branch of `send_audio` - an ordinary send whose
        reader is already running must never pay for a join here.

        The join is also what makes `self.state` safe to reassert below: the dying
        reader's own `receive()` sets it to `degraded` on its way out, and that write
        happens on a different thread with no lock around `self.state`, so joining
        first guarantees that write has already landed before this method's own
        `ACTIVE` write - rather than racing it and losing.
        """
        if self._on_event is None:
            return
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        # `send_audio` already confirmed the new connection is healthy; reassert ACTIVE
        # so `start_reader`'s own-state check below does not refuse on the `degraded`
        # the old reader just wrote while exiting.
        self.state = LegState.ACTIVE
        self.start_reader(self._on_event)

    def _notify_event(self, event: Dict[str, Any]) -> None:
        """Forwards one event to the registered callback, isolating its failures."""
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception as error:  # noqa: BLE001 - one bad handler must not kill the reader
            logger.error("Translation leg '%s' event handler failed: %s",
                         self.config.leg_id, error)

    def _reconnect(self) -> bool:
        """Re-establishes the socket in place, preserving the leg's identity."""
        try:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:  # noqa: BLE001 - the socket is already gone
                    pass
            self._connection = self._connector(self.connection_url)
            self._connection.send(json.dumps(self.build_setup_message()))
        except Exception as error:  # noqa: BLE001 - reconnection is best-effort
            logger.warning("Translation leg '%s' could not reconnect: %s",
                           self.config.leg_id, error)
            self._connection = None
            return False

        self.reconnect_count += 1
        return True

    def _mark_unavailable(self, reason: str) -> bool:
        """Records the fail-closed state and raises for the router to publish."""
        self.state = LegState.UNAVAILABLE
        self.unavailable_reason = reason
        logger.warning("Translation leg '%s' unavailable: %s", self.config.leg_id, reason)
        raise TranslationUnavailableError(reason)
