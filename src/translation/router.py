"""
Summary:
    router.py owns the Gemini Live Translate topology for one session (REQ-17,
    TASK-11.3-11.7): which legs exist, which participant's audio reaches each one, whose
    ears the translated result lands in, and what the UI is told about all of it.

    The topology is mode-dependent, and that difference is the module's reason to exist:

    - `language_learning`: two direct legs per peer pair, `A -> B's language` and
      `B -> A's language`. Peers hear each other interpreted into their own language.
    - `international_work`: one leg per participant, always targeting English, fanned out
      to everyone else. Leg count then grows with the number of people rather than with
      the number of ordered language pairs, which is what makes a five-person work call
      tractable when a five-way `language_learning` topology would need twenty legs.

    Two invariants are enforced here rather than trusted:

    1. Only original participant tracks are routed into a leg, keyed by immutable speaker
       identity. A translated track fed back in would have Gemini interpreting its own
       output indefinitely, and it would sound plausible while doing it.
    2. A leg fails and recovers alone. Original Agora audio, the AI voice, quizzes, and
       notes are all unaffected by any state this module can reach, including total
       unavailability at session start.

Key Classes:
    - TranslationRouter: the per-session leg registry, audio router, and event publisher.
    - Participant: who is in the session and what language they speak.
"""

import functools
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.sessions.models import SessionMode, SessionRecord
from src.translation.audio import (
    DEFAULT_PUBLISH_FRAME_MS,
    DEFAULT_PUBLISH_SAMPLE_RATE,
    AgoraPublishAdapter,
    GeminiInputEncoder,
)
from src.translation.gemini_live import (
    GeminiLiveTranslateSession,
    LegConfig,
    LegState,
    TranslationUnavailableError,
    to_bcp47,
)

logger = logging.getLogger("echosphere.translation.router")

# Bumped when the shape of a translation event changes. Receivers route and deduplicate
# by envelope shape alone, exactly as they do for artifact events (spec section 5).
TRANSLATION_SCHEMA_VERSION = "1.0"

EVENT_STATUS = "translation.status"
EVENT_INPUT_TRANSCRIPT = "translation.input_transcript"
EVENT_OUTPUT_TRANSCRIPT = "translation.output_transcript"

# Prefix marking a track as this router's own output. Routing decisions are made on the
# track identity rather than on audio content, because by the time a feedback loop is
# audible it has already been running for several seconds.
TRANSLATED_TRACK_PREFIX = "xlat-"

# The shared working language every `international_work` leg pivots through.
WORK_PIVOT_LANGUAGE = "en"


@dataclass(frozen=True)
class Participant:
    """One person in the session, and the language their audio is expected to be in."""

    participant_id: str
    language: str = "en"


class TranslationRouter:
    """
    Owns every Gemini Live leg for one session.

    Collaborators are injected rather than constructed: `session_factory` builds the legs,
    `data_stream` publishes events, `transcript_sink` receives finalized input transcripts
    for `TeachingAgent` and artifacts, and `audio_publisher` puts translated frames on an
    Agora track. All four are optional, so a session can run - and be reasoned about -
    with any subset of them absent.
    """

    def __init__(
        self,
        session: SessionRecord,
        data_stream: Optional[Any] = None,
        session_factory: Optional[Callable[..., Any]] = None,
        transcript_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
        audio_publisher: Optional[Callable[..., Any]] = None,
        publish_rate: int = DEFAULT_PUBLISH_SAMPLE_RATE,
        frame_ms: int = DEFAULT_PUBLISH_FRAME_MS
    ):
        """
        Initialize an empty router for one session.

        Algorithm:
        1. Bind the session (its mode decides the topology and the audio-gate default).
        2. Bind the injected collaborators.
        3. Create the leg registry, per-speaker encoders, per-leg publish adapters, the
           per-participant audio gate, and the ingestion-deduplication set.
        """
        self.session = session
        self.data_stream = data_stream
        self.session_factory = session_factory or GeminiLiveTranslateSession
        self.transcript_sink = transcript_sink
        self.audio_publisher = audio_publisher
        self.publish_rate = publish_rate
        self.frame_ms = frame_ms

        self._participants: List[Participant] = []
        self._legs: Dict[str, LegConfig] = {}
        self._sessions: Dict[str, Any] = {}
        self._states: Dict[str, LegState] = {}
        self._encoders: Dict[Tuple[str, int, int], GeminiInputEncoder] = {}
        self._publishers: Dict[str, AgoraPublishAdapter] = {}
        self._sequences: Dict[str, int] = {}
        self._ingested: Set[Tuple[str, str]] = set()
        self._audio_gate: Dict[str, bool] = {}
        self._lock = threading.Lock()

        logger.info(
            "TranslationRouter initialized for session %s (%s).",
            session.session_id, session.mode.value
        )

    # -- Topology (TASK-11.3 / 11.7) -------------------------------------------------

    def plan_legs(self, participants: Sequence[Participant]) -> List[LegConfig]:
        """
        Builds the leg graph this session's mode calls for (REQ-17).

        Algorithm:
        1. `international_work`: one leg per participant targeting English, with every
           other participant as a recipient. Participants already speaking English still
           get a leg - they code-switch for precision terms, and that utterance is
           precisely what the pivot exists to carry. `echoTargetLanguage=false` keeps an
           already-English utterance from being re-synthesized back into the room.
        2. `language_learning`: one leg per ordered pair, targeting the recipient's own
           language and delivered only to that recipient.
        3. Leg ids are derived from participant ids, so replanning the same room yields
           the same ids and a client's leg-keyed UI state survives.
        """
        if self.session.mode is SessionMode.INTERNATIONAL_WORK:
            return self._plan_work_legs(participants)
        return self._plan_learning_legs(participants)

    def _plan_work_legs(self, participants: Sequence[Participant]) -> List[LegConfig]:
        """One English-pivot leg per participant, broadcast to everyone else."""
        target = to_bcp47(WORK_PIVOT_LANGUAGE)
        legs: List[LegConfig] = []
        for speaker in participants:
            recipients = tuple(
                other.participant_id for other in participants
                if other.participant_id != speaker.participant_id
            )
            legs.append(LegConfig(
                leg_id=f"leg-{speaker.participant_id}-to-en",
                speaker_id=speaker.participant_id,
                target_language=target,
                recipients=recipients,
                echo_target_language=False,
                metadata={"speaker_language": speaker.language, "pivot": "english"},
            ))
        return legs

    def _plan_learning_legs(self, participants: Sequence[Participant]) -> List[LegConfig]:
        """Direct peer-pair legs, each targeting the recipient's own language."""
        legs: List[LegConfig] = []
        for speaker in participants:
            for recipient in participants:
                if recipient.participant_id == speaker.participant_id:
                    continue
                legs.append(LegConfig(
                    leg_id=f"leg-{speaker.participant_id}-to-{recipient.participant_id}",
                    speaker_id=speaker.participant_id,
                    target_language=to_bcp47(recipient.language),
                    recipients=(recipient.participant_id,),
                    echo_target_language=False,
                    metadata={"speaker_language": speaker.language,
                              "recipient_language": recipient.language},
                ))
        return legs

    # -- Lifecycle (TASK-11.3 / 11.6) ------------------------------------------------

    def start(self, participants: Sequence[Participant]) -> Dict[str, str]:
        """
        Plans and connects every leg, failing closed leg by leg (REQ-17).

        Algorithm:
        1. Plan the topology for the current participant list.
        2. Build and connect each leg, catching TranslationUnavailableError - and any
           other transport failure - per leg rather than for the group.
        3. Publish one `translation.status` event per leg, including `unavailable` ones.
        4. Return the leg states. This never raises: a session whose translator is down
           still runs, it simply runs without translated audio.
        """
        self._participants = list(participants)
        self._reset_leg_state()

        for config in self.plan_legs(self._participants):
            self._legs[config.leg_id] = config
            self._publishers[config.leg_id] = AgoraPublishAdapter(
                publish_rate=self.publish_rate, frame_ms=self.frame_ms
            )
            reason = ""
            try:
                leg_session = self.session_factory(config)
                leg_session.connect()
                state = getattr(leg_session, "state", LegState.ACTIVE)
                self._sessions[config.leg_id] = leg_session
            except TranslationUnavailableError as error:
                state, reason = LegState.UNAVAILABLE, str(error)
            except Exception as error:  # noqa: BLE001 - any startup failure fails closed
                state, reason = LegState.UNAVAILABLE, str(error) or error.__class__.__name__

            self._states[config.leg_id] = state if isinstance(state, LegState) else LegState.ACTIVE
            self._emit_status(config.leg_id, reason=reason)

            # TASK-11.8: a connected leg starts producing translated audio and
            # transcripts only once something drains its socket. Without this, the wire
            # contract fixed in v1.13.1 has nothing to deliver it - every event Gemini
            # sends would sit unread until the leg's own send-side timeout gave up on it.
            if self._states[config.leg_id] is LegState.ACTIVE:
                start_reader = getattr(self._sessions.get(config.leg_id), "start_reader", None)
                if start_reader is not None:
                    start_reader(functools.partial(self._on_leg_event, config.leg_id))

        return self.leg_states()

    def stop(self) -> None:
        """Closes every leg and clears the registry (session stop, REQ-17)."""
        for leg_session in list(self._sessions.values()):
            self._close_quietly(leg_session)
        self._reset_leg_state()
        logger.info("TranslationRouter stopped for session %s.", self.session.session_id)

    def remove_participant(self, participant_id: str) -> None:
        """
        Closes the legs a departing participant owns and drops them as a recipient.

        Algorithm:
        1. Close and forget every leg whose speaker is this participant.
        2. Rewrite the remaining legs' recipient tuples so nothing is published to a
           participant who is no longer in the channel.

        The surviving legs keep their sockets: re-establishing them would interrupt live
        speech for everyone still in the room over someone else's departure.
        """
        for leg_id, config in list(self._legs.items()):
            if config.speaker_id == participant_id:
                self._close_quietly(self._sessions.pop(leg_id, None))
                self._legs.pop(leg_id, None)
                self._states.pop(leg_id, None)
                self._publishers.pop(leg_id, None)
                self._emit_status(leg_id, reason="participant left", state=LegState.CLOSED)
            elif participant_id in config.recipients:
                self._legs[leg_id] = replace(
                    config,
                    recipients=tuple(r for r in config.recipients if r != participant_id)
                )

        self._participants = [p for p in self._participants
                              if p.participant_id != participant_id]

    def degrade_leg(self, leg_id: str, reason: str = "") -> None:
        """Marks one leg degraded; the others and the original audio keep running."""
        if leg_id in self._legs:
            self._states[leg_id] = LegState.DEGRADED
            self._emit_status(leg_id, reason=reason)

    def recover_leg(self, leg_id: str) -> None:
        """Returns a degraded leg to service once its transport is healthy again."""
        if leg_id in self._legs:
            self._states[leg_id] = LegState.ACTIVE
            self._emit_status(leg_id)

    # -- Introspection ---------------------------------------------------------------

    def legs(self) -> List[LegConfig]:
        """Every currently registered leg configuration."""
        return list(self._legs.values())

    def leg_states(self) -> Dict[str, str]:
        """Leg id -> wire state, for `/api/translation/status` and the UI."""
        return {leg_id: state.value for leg_id, state in self._states.items()}

    def legs_for_speaker(self, speaker_id: str) -> List[LegConfig]:
        """
        Every leg carrying this speaker's audio.

        A speaker owns more than one leg whenever `language_learning` runs with more than
        two people: one leg per listener, each targeting that listener's own language.
        Audio must reach all of them, so routing goes through this rather than through
        the singular accessor below.
        """
        return [config for config in self._legs.values()
                if config.speaker_id == speaker_id]

    def leg_for_speaker(self, speaker_id: str) -> Optional[LegConfig]:
        """
        The first leg carrying this speaker's audio, or None.

        Unambiguous in `international_work` (one leg per participant) and in a two-person
        tandem pair. With three or more peers in `language_learning` a speaker has
        several, so use `legs_for_speaker` for anything that must reach all of them.
        """
        legs = self.legs_for_speaker(speaker_id)
        return legs[0] if legs else None

    @property
    def is_available(self) -> bool:
        """
        Whether any leg can still carry audio.

        Partial availability counts: one working leg is a participant who can be
        understood, which is materially different from none.
        """
        return any(state in (LegState.ACTIVE, LegState.DEGRADED)
                   for state in self._states.values())

    def translated_track_id(self, leg_id: str) -> str:
        """The Agora track id this leg's translated audio is published on."""
        return f"{TRANSLATED_TRACK_PREFIX}{leg_id}"

    def is_translated_track(self, track_id: str) -> bool:
        """Whether a track is this router's own output rather than a participant's."""
        return isinstance(track_id, str) and track_id.startswith(TRANSLATED_TRACK_PREFIX)

    # -- Audio routing (TASK-11.3) ---------------------------------------------------

    def route_audio(
        self,
        speaker_id: str,
        pcm: bytes,
        sample_rate: int = DEFAULT_PUBLISH_SAMPLE_RATE,
        channels: int = 1
    ) -> int:
        """
        Feeds one participant's original audio into their leg (REQ-17).

        Algorithm:
        1. Refuse anything that is one of our own translated tracks. This is the feedback
           guard, and it is checked first precisely because every later step would happily
           accept the audio.
        2. Resolve every leg this speaker owns, by immutable identity; unknown speakers
           are ignored. A `language_learning` room of three or more gives one speaker a
           leg per listener, and all of them carry the same audio.
        3. Encode once into 100 ms PCM16 mono 16 kHz chunks. Encoding per leg instead
           would run one buffer per listener over the same utterance, and they would
           drift apart at every partial chunk.
        4. Send each chunk to every active leg. Degraded, unavailable, and closed legs
           consume nothing; a leg that fails mid-send is degraded on its own and the
           speaker's remaining listeners keep hearing them.

        Returns:
            The number of chunk sends that succeeded, across every leg.
        """
        if self.is_translated_track(speaker_id):
            logger.debug("Refusing to route translated track '%s' back into a leg.",
                         speaker_id)
            return 0

        active = [
            (config, self._sessions[config.leg_id])
            for config in self.legs_for_speaker(speaker_id)
            if self._states.get(config.leg_id) is LegState.ACTIVE
            and config.leg_id in self._sessions
        ]
        if not active:
            return 0

        encoder = self._encoder_for(speaker_id, sample_rate, channels)
        chunks = encoder.push(pcm)
        sent = 0
        for chunk in chunks:
            for config, leg_session in active:
                if self._states.get(config.leg_id) is not LegState.ACTIVE:
                    continue
                if leg_session.send_audio(chunk):
                    sent += 1
                else:
                    self.degrade_leg(config.leg_id, reason="audio send failed")
        return sent

    def handle_translated_audio(self, leg_id: str, pcm: bytes) -> List[str]:
        """
        Publishes a leg's translated audio to its gated recipients (TASK-11.4/11.5).

        Algorithm:
        1. Ignore unknown or inactive legs.
        2. Filter the leg's recipients through the per-participant audio gate. A recipient
           who has the gate off simply is not published to; the leg keeps running, because
           its transcripts are still wanted.
        3. Adapt the 24 kHz PCM to Agora publish frames and hand them to the publisher,
           once per recipient, on this leg's own translated track.

        Returns:
            The recipients actually published to.
        """
        config = self._legs.get(leg_id)
        if config is None or self._states.get(leg_id) is not LegState.ACTIVE:
            return []

        recipients = [r for r in config.recipients if self.translated_audio_enabled(r)]
        if not recipients:
            return []

        adapter = self._publishers.setdefault(
            leg_id, AgoraPublishAdapter(publish_rate=self.publish_rate,
                                        frame_ms=self.frame_ms)
        )
        frames = adapter.push(pcm)
        if frames and self.audio_publisher is not None:
            for recipient in recipients:
                self.audio_publisher(
                    leg_id=leg_id,
                    recipient_id=recipient,
                    track_id=self.translated_track_id(leg_id),
                    frames=frames,
                    sample_rate=self.publish_rate,
                )
        return recipients

    # -- The translated-audio gate (TASK-11.5) ---------------------------------------

    def translated_audio_enabled(self, participant_id: str) -> bool:
        """
        Whether this participant currently hears translated audio.

        The default comes from the session mode, never from the client: work calls want
        it on immediately, and a tandem lesson wants it off so the learner still has to
        produce and parse the target language themselves.
        """
        return self._audio_gate.get(
            participant_id, self.session.mode.translated_audio_default
        )

    def set_translated_audio_enabled(self, participant_id: str, enabled: bool) -> bool:
        """
        Flips one participant's audio gate and announces it (REQ-06 controls, REQ-17).

        Publishing the new state matters because the gate is per participant while the
        leg is shared: without the event, two clients on the same leg disagree about
        whether anyone is hearing it.
        """
        self._audio_gate[participant_id] = bool(enabled)
        self._emit_gate_status(participant_id)
        return self._audio_gate[participant_id]

    def audio_gate_states(self) -> Dict[str, bool]:
        """Every known participant's gate state, resolved through the mode default."""
        return {
            participant.participant_id:
                self.translated_audio_enabled(participant.participant_id)
            for participant in self._participants
        }

    # -- Transcript events (TASK-11.4) -----------------------------------------------

    def handle_input_transcript(
        self,
        leg_id: str,
        text: str,
        is_final: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Publishes what the speaker said, and hands finalized text downstream once.

        Algorithm:
        1. Build and publish the versioned `translation.input_transcript` event.
        2. When the text is finalized, forward it to the transcript sink exactly once,
           keyed on leg plus normalized text. Re-delivery of a finalized transcript is
           expected on this transport, and a duplicate turn reaches `TeachingAgent` as
           the learner saying the same sentence twice.

        Interim text publishes but is never ingested: it is display-only (spec section 5).
        """
        payload = self._transcript_payload(leg_id, text, is_final, direction="input")
        if payload is None:
            return None

        self._publish(EVENT_INPUT_TRANSCRIPT, payload)

        if is_final and self.transcript_sink is not None:
            key = (leg_id, " ".join(str(text).split()).lower())
            with self._lock:
                if key in self._ingested:
                    return payload
                self._ingested.add(key)
            self.transcript_sink(payload)

        return payload

    def handle_output_transcript(
        self,
        leg_id: str,
        text: str,
        is_final: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Publishes what the interpreter said.

        Never ingested downstream: the interpreter's voice is a rendering of a turn that
        has already been recorded, and counting it again would double every utterance in
        the transcript and in the speaking-balance metrics.
        """
        payload = self._transcript_payload(leg_id, text, is_final, direction="output")
        if payload is None:
            return None
        self._publish(EVENT_OUTPUT_TRANSCRIPT, payload)
        return payload

    # -- Internals -------------------------------------------------------------------

    def _transcript_payload(
        self,
        leg_id: str,
        text: str,
        is_final: bool,
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """
        Builds one versioned transcript envelope.

        Carries source/target plain text only. Transliteration and register-aware
        phrasing stay with the ASR -> `TeachingAgent` subtitle pipeline; a second,
        divergent source of them here is how two subtitles end up on screen disagreeing.
        """
        config = self._legs.get(leg_id)
        if config is None:
            return None

        with self._lock:
            sequence = self._sequences.get(leg_id, 0) + 1
            self._sequences[leg_id] = sequence

        return {
            "schema_version": TRANSLATION_SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "mode": self.session.mode.value,
            "leg_id": leg_id,
            "speaker_id": config.speaker_id,
            "direction": direction,
            "source_language": config.metadata.get("speaker_language", ""),
            "target_language": config.target_language,
            "sequence": sequence,
            "timestamp": int(time.time() * 1000),
            "text": text,
            "is_final": bool(is_final),
        }

    def _on_leg_event(self, leg_id: str, event: Dict[str, Any]) -> None:
        """
        Dispatches one event a leg's reader produced to the matching handler (TASK-11.8).

        Algorithm:
        1. `audio` -> `handle_translated_audio`, gated by the recipient audio toggle.
        2. `input_transcript` / `output_transcript` -> the matching transcript handler,
           carrying `is_final` through so once-only ingestion still applies.
        3. `leg_state_changed` -> reconcile the router's own leg-health bookkeeping with
           what the leg discovered on its own: `active` recovers it, anything else
           degrades it. Without this, a leg whose reader died from a read failure - or
           came back after a reconnect - would leave the router's `_states` entry stale,
           silently gating routing/publication off (or on) forever after.

        Runs on whichever leg's reader thread produced the event; every handler this
        calls already synchronizes its own shared state (`self._lock`), so this method
        itself holds nothing.
        """
        event_type = event.get("type")
        if event_type == "audio":
            self.handle_translated_audio(leg_id, event.get("audio", b""))
        elif event_type == "input_transcript":
            self.handle_input_transcript(leg_id, event.get("text", ""),
                                         is_final=bool(event.get("is_final")))
        elif event_type == "output_transcript":
            self.handle_output_transcript(leg_id, event.get("text", ""),
                                          is_final=bool(event.get("is_final")))
        elif event_type == "leg_state_changed":
            if event.get("state") == LegState.ACTIVE.value:
                self.recover_leg(leg_id)
            else:
                self.degrade_leg(leg_id, reason=event.get("reason", ""))

    def _emit_status(
        self,
        leg_id: str,
        reason: str = "",
        state: Optional[LegState] = None
    ) -> None:
        """Publishes one `translation.status` event for a leg."""
        config = self._legs.get(leg_id)
        resolved = state or self._states.get(leg_id, LegState.IDLE)
        self._publish(EVENT_STATUS, {
            "schema_version": TRANSLATION_SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "mode": self.session.mode.value,
            "leg_id": leg_id,
            "speaker_id": config.speaker_id if config else "",
            "recipients": list(config.recipients) if config else [],
            "target_language": config.target_language if config else "",
            "state": resolved.value,
            "reason": reason,
            "timestamp": int(time.time() * 1000),
        })

    def _emit_gate_status(self, participant_id: str) -> None:
        """Publishes one participant's translated-audio gate state."""
        self._publish(EVENT_STATUS, {
            "schema_version": TRANSLATION_SCHEMA_VERSION,
            "session_id": self.session.session_id,
            "mode": self.session.mode.value,
            "participant_id": participant_id,
            "translated_audio_enabled": self.translated_audio_enabled(participant_id),
            "state": "audio_gate",
            "reason": "",
            "timestamp": int(time.time() * 1000),
        })

    def _publish(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Sends one translation event, tolerating a session with no data stream."""
        if self.data_stream is None:
            return False
        return bool(self.data_stream.send_translation_event(event_type, payload))

    def _encoder_for(
        self,
        speaker_id: str,
        sample_rate: int,
        channels: int
    ) -> GeminiInputEncoder:
        """
        Returns this speaker's input encoder, creating one per capture format.

        Keyed on the format as well as the speaker: an encoder holds a partial chunk in
        the format it was built for, so a mid-session format change needs a new one
        rather than a reinterpretation of the bytes already buffered.
        """
        key = (speaker_id, sample_rate, channels)
        if key not in self._encoders:
            self._encoders[key] = GeminiInputEncoder(
                source_rate=sample_rate, channels=channels
            )
        return self._encoders[key]

    def _close_quietly(self, leg_session: Optional[Any]) -> None:
        """Closes a leg session without letting a dead socket propagate."""
        if leg_session is None:
            return
        try:
            leg_session.close()
        except Exception as error:  # noqa: BLE001 - shutdown is best-effort
            logger.debug("Closing a translation leg raised: %s", error)

    def _reset_leg_state(self) -> None:
        """Clears every per-leg structure, leaving the audio gate intact."""
        self._legs.clear()
        self._sessions.clear()
        self._states.clear()
        self._encoders.clear()
        self._publishers.clear()
        self._sequences.clear()
        self._ingested.clear()
