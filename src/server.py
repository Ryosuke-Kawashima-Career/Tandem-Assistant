"""
Summary:
    server.py serves as the central backend API and lifecycle manager for
    the EchoSphere Tandem Co-Teacher platform.
    It exposes HTTP endpoints for health checks, room management, and conversational
    turn orchestration, bridging the Agora SD-RTN™ voice channels, LLM TeachingAgent,
    real-time STT/TTS audio engines, and RTC Data Stream broadcasters.

Key Classes and Objects:
    - app: Flask web application instance providing REST endpoints and test client support.
    - EchoSphereServer: Backend lifecycle manager coordinating RTC audio channels,
      AI co-teacher agent, STT transcription, TTS synthesis, and data stream synchronization.
"""

import os
import base64
import binascii
import json
import time
import queue
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict, Any, Iterator, List
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context

# Must run before any credential-reading object (AgoraVoiceChannelClient,
# ConvoAIClient) is constructed below. Flask's own dev server also auto-loads
# .env, but only inside app.run() - too late for the module-level server_instance
# below, and it never happens at all under a production WSGI runner (uvicorn,
# gunicorn) that only imports `app` and never calls `.run()`.
#
# override=True: .env is this project's single source of truth for these
# variables (see .env.example). Without it, load_dotenv() leaves a stale
# same-name variable already present in the shell untouched - e.g. a
# `$env:CONVOAI_LLM_BASE_URL` set during ngrok setup silently outliving an
# edit to .env, with no error or warning anywhere.
load_dotenv(override=True)

from src.rtc.agora_client import AgoraVoiceChannelClient, is_usable_credential
from src.rtc.data_stream import DataStreamManager
from src.rtc.rtm_publisher import RtmRestPublisher
from src.artifacts.access import AccessDeniedError, require_access, resolve_actor
from src.artifacts.adapters import (
    SUPPORTED_TARGETS as SUPPORTED_NOTION_TARGETS,
    ExportNotConfiguredError,
    NotionExportAdapter,
)
from src.artifacts.export import render_markdown
from src.artifacts.generator import ArtifactGenerator
from src.artifacts.models import TranscriptTurn
from src.artifacts.repository import LocalArtifactRepository, configured_retention_days
from src.sessions.models import InvalidSessionModeError, SessionMode, SessionRecord
from src.sessions.service import SessionNotFoundError, SessionService
from src.sessions.speaking_balance import SpeakingBalanceTracker
from src.translation.router import Participant, TranslationRouter
from src.rtc.convoai_client import (
    ConvoAIClient,
    AGENT_STATUS_FAILED,
    AGENT_STATUS_RECOVERING,
)
from src.agent.orchestrator import TeachingAgent
from src.agent.tools.base import ToolState
from src.agent.tools.camera_stream import CameraFrameBuffer
from src.agent.tools.dispatch import ToolDispatcher
from src.audio.vad_processor import VoiceActivityDetector
from src.audio.stt_transcriber import STTTranscriber
from src.audio.tts_synthesizer import TTSSynthesizer

# Setup logging. LOG_LEVEL is honoured (see .env.example) so the Custom LLM bridge's
# debug request dump (REQ-LLM-04) can be enabled without a code change.
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
logger = logging.getLogger("echosphere.server")

# Upper bound on how long the bridge waits for the reasoning engine before speaking a
# fallback line (Task 5.2). The Convo AI Engine expires an agent on `idle_timeout`
# (default 30s) with no activity, so silence here costs the whole session.
CONVOAI_LLM_TIMEOUT_SECONDS = float(os.getenv("CONVOAI_LLM_TIMEOUT_SECONDS", "20"))

# Size of each SSE content delta. The engine concatenates deltas, so this only controls
# how early audio synthesis can begin, not the final text.
SSE_CHUNK_CHARS = 24

# Delay before the post-join agent health poll (REQ-LLM-09). Long enough for the Engine
# to move the agent past STARTING, short enough to see a failure before the learner has
# given up waiting for a voice.
# Seconds to wait after startup before probing the Custom LLM bridge. Long enough for
# Flask to be listening, so a tunnel that forwards to this process is not reported dead
# while the process is still binding its socket.
BRIDGE_BOOT_CHECK_DELAY_SECONDS = 3.0

CONVOAI_HEALTH_POLL_DELAY_SECONDS = float(os.getenv("CONVOAI_HEALTH_POLL_DELAY_SECONDS", "5"))

# Shared worker pool for Custom LLM turns. Module-level and never shut down: a
# per-request executor would have to be joined, re-blocking on a hung provider call.
_TURN_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="convoai-turn")

# Spoken when the engine is too slow or errors outright, per language.
FALLBACK_REPLIES: Dict[str, str] = {
    "en": "Sorry, I need a moment to think. Could you say that again?",
    "ja": "すみません、少し考えさせてください。もう一度言ってもらえますか？",
    "hi": "क्षमा करें, मुझे एक पल चाहिए। क्या आप दोबारा कह सकते हैं?",
}


class EchoSphereServer:
    """
    Coordinates backend service lifecycles and real-time tandem co-teaching pipelines.
    """

    def __init__(
        self,
        channel_name: str = "tokyo-mumbai-101",
        engine: str = "mock"
    ):
        """
        Initialize the EchoSphere backend server lifecycle components.

        Algorithm:
        1. Store room channel name and operational engine mode.
        2. Instantiate Agora RTC voice channel client and Data Stream manager.
        3. Instantiate AI TeachingAgent, VAD detector, STT transcriber, and TTS synthesizer.
        4. Instantiate the Convo AI client for direct spoken AI conversations.
        5. Track active session state and per-channel Convo AI session context.
        """
        self.channel_name = channel_name
        self.engine = engine

        # Step 2: RTC and Data Stream. The publisher is what actually carries an event
        # to the browser (D-UIUX-2): without it the data stream fans out to in-process
        # callbacks only, which is why every generated subtitle, quiz, note and card was
        # correct server-side and invisible in the UI.
        self.rtm_publisher = RtmRestPublisher()
        if not self.rtm_publisher.is_configured:
            logger.warning(
                "Agora RTM REST credentials absent: live session events will not reach "
                "the browser. Set AGORA_APP_ID, AGORA_CUSTOMER_ID and "
                "AGORA_CUSTOMER_SECRET to enable real-time delivery."
            )
        self.rtc_client = AgoraVoiceChannelClient(
            channel_name=channel_name, rtm_publisher=self.rtm_publisher
        )
        self.data_stream = DataStreamManager(self.rtc_client)

        # Step 3: AI and Audio Pipeline. The agent's external tools (REQ-18-20) are built
        # here rather than inside the agent so they publish over this server's own data
        # stream; each one is unconfigured - and reports itself so - until its credentials
        # are present in the environment.
        # REQ-CAM-01: the live camera frame the agent may look at mid-turn. Held on the
        # server rather than inside the dispatcher so the ingestion endpoint and the
        # lookup share one store, and so ending a session can empty it.
        self.camera_buffer = CameraFrameBuffer()
        self.tools = ToolDispatcher.from_env(
            data_stream=self.data_stream, camera_buffer=self.camera_buffer
        )
        self.agent = TeachingAgent(engine=engine, tools=self.tools)
        self.vad = VoiceActivityDetector()
        self.stt = STTTranscriber(engine=engine)
        self.tts = TTSSynthesizer()

        # Step 4: Convo AI direct audio conversation engine (REQ-09 / REQ-10)
        self.convoai = ConvoAIClient()

        # Session modes (REQ-12). One registry owns the mode for every channel, so
        # prompts, RTC events, quizzes, and notes all branch on the same answer.
        self.sessions = SessionService()

        # Measured speaking time per channel (REQ-23). Separate from the session record
        # because it is a running tally updated from whichever thread noticed a speech
        # boundary, while a session's identity and mode are decided once.
        self.speaking_balance = SpeakingBalanceTracker()

        # Session artifacts (REQ-13 / REQ-14). Generation is driven from the finalized
        # turn, off the voice-critical path, and publishes quiz.created / note.upserted
        # over the same RTC data stream the scaffolding widgets already use.
        # REQ-15: the local repository is the MVP source of truth, so it is durable -
        # a session's notes must outlive both the conversation and the process. Retention
        # (REQ-16) is opt-in and applied at startup.
        self.artifacts = LocalArtifactRepository(retention_days=configured_retention_days())
        self.artifact_generator = ArtifactGenerator(
            repository=self.artifacts, data_stream=self.data_stream
        )
        purged = self.artifacts.purge_expired()
        if purged:
            logger.info("Retention purge removed %d stored session artifact(s).", purged)

        # Gemini Live Translate legs, one router per channel (REQ-17). Kept per channel
        # rather than on the server because the topology is derived from a session's mode
        # and its participant list, both of which are per channel.
        self.translation_routers: Dict[str, TranslationRouter] = {}

        self.is_active = False

        # Step 5: Per-channel Convo AI session context (REQ-LLM-02).
        # Agora's Custom LLM request body is OpenAI-shaped (model / messages / stream)
        # and carries no language or speaker field, so the learner's chosen language is
        # only ever known here - captured when /api/convoai/start is called.
        self.convoai_session_context: Dict[str, Dict[str, Any]] = {}
        self._convoai_last_channel: Optional[str] = None

        # Background scaffolding tasks (REQ-LAT-02). Retained only so tests and
        # diagnostics can join on them; the request path never waits.
        self._scaffolding_futures: List[Future] = []

        logger.info(
            f"EchoSphereServer initialized for channel '{channel_name}' "
            f"(engine: {engine}, agent engine: {self.agent.engine})."
        )

    # -- Gemini Live Translate legs (REQ-17) ---------------------------------------

    def start_translation(
        self,
        channel: str,
        participants: List[Dict[str, Any]]
    ) -> TranslationRouter:
        """
        Starts (or restarts) the translation legs for a channel (TASK-11.3 / 11.6).

        Algorithm:
        1. Require the channel's session - the mode decides the whole leg topology.
        2. Close any router already on the channel, so a participant list change
           replaces the legs rather than layering a second set on top of them.
        3. Build a router wired to this server's data stream, and start it. Startup never
           raises: legs that cannot reach Gemini report `unavailable` and the session
           carries on without translated audio.

        Raises:
            SessionNotFoundError: the channel has no session to derive a mode from.
        """
        session = self.sessions.require_session(channel)
        self.stop_translation(channel)

        router = TranslationRouter(
            session=session,
            data_stream=self.data_stream,
            transcript_sink=self._ingest_translation_transcript,
        )
        router.start([
            Participant(
                participant_id=str(entry.get("participant_id") or entry.get("uid") or ""),
                language=str(entry.get("language") or "en"),
            )
            for entry in participants
            if entry.get("participant_id") or entry.get("uid")
        ])

        self.translation_routers[channel] = router
        return router

    def stop_translation(self, channel: str) -> bool:
        """Closes and forgets a channel's translation legs. Safe on an unknown channel."""
        router = self.translation_routers.pop(channel, None)
        if router is None:
            return False
        router.stop()
        return True

    def _ingest_translation_transcript(self, payload: Dict[str, Any]) -> None:
        """
        Feeds one finalized translation input transcript into the artifact pipeline.

        Deliberately narrow: the router has already deduplicated, so this must not
        re-check, and it must not block - it runs on whatever thread delivered the
        transcript, which in a live session is the leg's read loop.
        """
        text = (payload.get("text") or "").strip()
        if not text:
            return
        logger.debug(
            "Finalized translation transcript on leg %s (%s).",
            payload.get("leg_id"), payload.get("speaker_id")
        )

    def register_convoai_session(
        self,
        channel: str,
        language: str = "en",
        speaker_id: str = "Learner",
        mode: Any = None
    ) -> Dict[str, Any]:
        """
        Records the session context for a starting Convo AI conversation (REQ-LLM-02).

        Also resets the shared TeachingAgent's conversation state (REQ-LLM-05): its
        turn_history and speaker_durations_ms otherwise persist for the lifetime of the
        process, so a new learner would inherit the previous one's dialogue as context.
        """
        session = self.sessions.create_session(
            channel=channel,
            mode=mode,
            languages=[language],
            participants=[speaker_id]
        )
        self.artifacts.save_session(session)
        context = {
            "channel": channel,
            "language": language,
            "speaker_id": speaker_id,
            "mode": session.mode.value,
            "session_id": session.session_id,
            "started_at": int(time.time()),
        }
        self.convoai_session_context[channel] = context
        self._convoai_last_channel = channel
        self.agent.reset_state()
        # REQ-23: for the same reason the agent's state is reset - the channel is reused,
        # the session is not, and a new learner must not be shown the previous one's share.
        self.speaking_balance.reset(channel)
        logger.info(
            f"Convo AI session context registered for channel '{channel}' "
            f"(language: {language}, speaker: {speaker_id}, mode: {session.mode.value})."
        )
        return context

    def clear_convoai_session(self, channel: str) -> None:
        """
        Drops the session context for a channel and resets conversation state (REQ-LLM-05).
        """
        self.convoai_session_context.pop(channel, None)
        closed = self.sessions.end_session(channel)
        if closed is not None:
            # Stamp the stored artifact with the end time; it outlives the session.
            self.artifacts.save_session(closed)
        if self._convoai_last_channel == channel:
            self._convoai_last_channel = None
        self.agent.reset_state()
        logger.info(f"Convo AI session context cleared for channel '{channel}'.")

    def resolve_convoai_context(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolves the session context a Custom LLM request belongs to.

        The Convo AI Engine POSTs a plain OpenAI-format body with no channel identifier,
        so an exact channel match is only possible when the caller supplies one (direct
        or manual calls). Otherwise the most recently started session is used - correct
        for the single-active-conversation deployment this backend targets, and an
        explicit thing to revisit before running concurrent channels.
        """
        if channel and channel in self.convoai_session_context:
            return self.convoai_session_context[channel]
        if self._convoai_last_channel:
            return self.convoai_session_context.get(self._convoai_last_channel, {})
        return {}

    def announce_note_change(
        self,
        note: Any,
        meta: Dict[str, Any],
        event_type: str = "note.upserted"
    ) -> bool:
        """
        Publishes a note event for a session identified only by its stored metadata.

        The generator's event path takes a live `SessionRecord`, but an edit can arrive
        after the session ended and its registry entry is gone - so the record is rebuilt
        from what the artifact store kept. Never raises: an unannounced edit is a stale
        panel, while a raised exception here loses the edit itself.
        """
        if note is None:
            return False

        try:
            session = _session_from_meta(meta)
            return self.artifact_generator._emit(
                event_type, session, "note", note.to_dict(), note.revision
            )
        except Exception as exc:
            logger.warning("Could not announce %s for note %s: %s", event_type, note.id, exc)
            return False

    def resolve_artifact_session(
        self,
        channel: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolves the session an artifact request is about, live or already ended.

        An explicit `session_id` wins; otherwise the channel's active session is used,
        falling back to the most recent stored session on that channel. The fallback is
        the point: someone asks for the notes *after* hanging up, when the live session
        registry no longer holds anything.
        """
        if session_id:
            return self.artifacts.get_session_meta(session_id)

        if channel:
            live = self.sessions.get_session(channel)
            if live is not None:
                return live.to_dict()
            return self.artifacts.find_session_by_channel(channel)

        return None

    def session_mode_for(self, channel: Optional[str] = None) -> SessionMode:
        """
        Resolves the session mode that a turn on this channel belongs to (REQ-12).

        Falls back to the Convo AI session context - and finally to language learning -
        because the Custom LLM bridge is called without a channel, so a turn can arrive
        before the caller has told us which channel it is on. The creation endpoints
        reject a missing mode, so this fallback covers resolution, never validation.
        """
        mode = self.sessions.mode_for(channel) if channel else None
        if mode is not None:
            return mode

        context = self.resolve_convoai_context(channel)
        resolved = self.sessions.mode_for(context.get("channel")) if context else None
        return resolved or SessionMode.LANGUAGE_LEARNING

    def start_session(self) -> bool:
        """
        Starts the tandem learning breakout session.

        Algorithm:
        1. Connect RTC client to the Agora voice channel.
        2. Broadcast initial welcome topic over RTC Data Stream.
        3. Mark session active.
        """
        success = self.rtc_client.join_channel()
        if success:
            self.is_active = True
            # Broadcast initial topic
            self.data_stream.send_topic(
                topic_title="Festivals & Food in India and Japan",
                prompt="What traditional dish do you prepare for Diwali or Oshogatsu?"
            )
            logger.info("Tandem breakout session started.")
        return success

    def stop_session(self) -> bool:
        """
        Stops the active tandem learning session cleanly.
        """
        if self.is_active:
            self.rtc_client.leave_channel()
            self.is_active = False
            logger.info("Tandem breakout session stopped.")
        return True

    def process_turn(
        self,
        speaker_id: str,
        text_or_audio: Any,
        language: str = "en",
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a single conversational turn from a student learner.

        Algorithm:
        1. If input is raw PCM audio bytes, transcribe speech via STTTranscriber.
        2. Forward transcription to TeachingAgent for mediation, transliteration, and cultural idiom analysis.
        3. Broadcast live subtitles and detected idiom cards over Agora RTC Data Stream,
           and file the notes and quiz this turn produced (REQ-13 / REQ-14).
        4. Synthesize AI voice response via TTSSynthesizer and publish to RTC channel if needed.
        5. Return full structured turn payload.
        """
        # Step 1: Speech-to-text if audio bytes provided
        if isinstance(text_or_audio, (bytes, bytearray)):
            transcription = self.stt.transcribe(text_or_audio, language=language, speaker_id=speaker_id)
            spoken_text = transcription.text
            detected_lang = transcription.language
            # REQ-23: the transcription's duration is derived from the byte length at a
            # known sample rate, so it is a measurement rather than an estimate - the one
            # signal on the ambient path this balance may legitimately count. A text turn
            # deliberately contributes nothing: there is no honest way to derive speaking
            # time from a string, and guessing one puts the gauge back on invented data.
            self.record_speaking_time(
                channel or self.channel_name, speaker_id, transcription.duration_ms
            )
        else:
            spoken_text = str(text_or_audio)
            detected_lang = language

        # Step 2: AI Agent Mediation, under this channel's session mode (REQ-12)
        turn_result = self.agent.process_turn(
            speaker_id=speaker_id,
            text=spoken_text,
            detected_language=detected_lang,
            session_mode=self.session_mode_for(channel or self.channel_name)
        )

        # Step 3: Broadcast over RTC Data Stream, then file the turn's artifacts
        # (REQ-13 / REQ-14) - an ambient peer turn is as finalized as a Convo AI one.
        self._broadcast_turn_payloads(turn_result, speaker_id, spoken_text)
        self.generate_turn_artifacts(
            turn_result, speaker_id, spoken_text, channel or self.channel_name, detected_lang
        )

        # Step 4: AI Speech Synthesis & RTC Audio Track
        spoken_response = turn_result.get("spoken_response", "")
        if spoken_response:
            ai_audio = self.tts.synthesize(spoken_response, language=turn_result.get("spoken_language", "en"))
            self.rtc_client.publish_audio_frame(ai_audio)

        return turn_result

    def record_speaking_time(
        self,
        channel: str,
        speaker_id: str,
        duration_ms: Any
    ) -> Dict[str, int]:
        """
        Records one measured speech segment and broadcasts the new balance (REQ-23).

        Algorithm:
        1. Accumulate the segment against this channel's tally.
        2. Broadcast the recalculated shares to every participant, not only to whoever
           reported the segment - REQ-23 extends REQ-07's teacher-only metric to the
           participant UI, and a motivation cue that only the reporter can see is not one.
        3. Return the shares so the reporting caller can draw immediately.

        Whether the segment counted is read from the tally rather than re-checked here:
        the tracker owns the rule about what a usable segment is (REQ-23), and a second
        copy of that rule in this method is a second thing to keep in step with it. A
        segment that changed nothing publishes nothing, so the stream carries no events
        that redraw an identical gauge.
        """
        recorded_before = self.speaking_balance.total_ms(channel)
        percentages = self.speaking_balance.record(channel, speaker_id, duration_ms)

        if self.speaking_balance.total_ms(channel) > recorded_before:
            self.data_stream.send_speaking_balance(percentages)
        return percentages

    def _broadcast_turn_payloads(
        self,
        turn_result: Dict[str, Any],
        speaker_id: str,
        spoken_text: str
    ) -> None:
        """
        Broadcasts the visual scaffolding produced by a turn over the RTC Data Stream.

        Shared by the ambient tandem mediation pipeline and the Convo AI direct
        conversation path so subtitle, idiom card, and quiz payloads are serialized in
        exactly one place.

        Algorithm:
        0. Skip entirely when the backend RTC client is not in the channel.
        1. Broadcast tri-lingual subtitles when the agent returned a subtitle block.
        2. Broadcast the cultural idiom card when one was detected.
        3. Broadcast the comprehension quiz widget when one is active.

        Step 0 matters for the Convo AI path: that flow never calls /api/session/start,
        so this client is not connected and every dispatch would log a warning per
        payload. Live transcripts for Convo AI conversations reach the browser over
        RTM directly from the Convo AI Engine instead (see
        app/src/services/convoaiTranscript.js), so nothing is lost by skipping here.

        Known gap: idiom cards and quizzes generated on the Convo AI path have no
        delivery route to the browser yet - RTM carries transcripts only.
        """
        if not self.rtc_client.is_connected:
            logger.debug(
                "Skipping data stream broadcast: backend RTC client not in channel "
                "(expected on the Convo AI path; transcripts arrive via RTM)."
            )
            return

        if "subtitles" in turn_result:
            sub = turn_result["subtitles"]
            self.data_stream.send_subtitle(
                speaker=sub.get("speaker", speaker_id),
                text=sub.get("original_text", spoken_text),
                transliteration=sub.get("transliteration", ""),
                translation_en=sub.get("translation_en", ""),
                translation_ja=sub.get("translation_ja", ""),
                translation_hi=sub.get("translation_hi", "")
            )

        if turn_result.get("idiom_card", {}).get("detected"):
            card = turn_result["idiom_card"]
            self.data_stream.send_idiom_card(
                phrase=card.get("phrase", ""),
                romaji=card.get("romaji", ""),
                meaning=card.get("meaning", ""),
                cultural_note=card.get("cultural_note", "")
            )

        if turn_result.get("quiz", {}).get("active"):
            q = turn_result["quiz"]
            self.data_stream.send_quiz(
                question=q.get("question", ""),
                options=q.get("options", []),
                correct_index=q.get("correct_index", 0),
                explanation=q.get("explanation", "")
            )

    def process_convoai_turn(
        self,
        speaker_id: str,
        text: str,
        language: str = "en",
        record_turn: bool = True,
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes one conversational turn arriving from the Convo AI Engine (REQ-11).

        Differs from process_turn in exactly one respect: speech synthesis is owned by
        the Convo AI Engine's TTS vendor, so this path returns reply text and never
        calls the local TTS synthesizer or publishes an audio frame. Transcription is
        also skipped because the Convo AI Engine performs ASR upstream.

        Algorithm:
        1. Forward the transcribed text to the shared TeachingAgent orchestrator in
           "tutor" mode.
        2. Broadcast subtitles, idiom cards, and quizzes over the RTC Data Stream.
        3. Return the structured turn payload for the SSE bridge to stream back.

        Runs in mode="tutor" (REQ-LLM-03): exactly one learner is in the channel, so the
        peer-mediation prompt - which instructs the model to balance two speakers - would
        make it address someone who is not there.

        Since REQ-LAT-02 this generates scaffolding only - the spoken reply comes from
        `stream_convoai_reply`, which already recorded the turn, hence record_turn=False
        from the scheduled caller.
        """
        session_mode = self.session_mode_for(channel)

        # REQ-CAM-03: the scaffolding reasons about the same observation the spoken reply
        # did, so a note written about "this" records the object rather than the pronoun.
        # `announce=False` because the spoken turn already published that card, and the
        # per-frame cache means this normally costs no vendor call at all.
        camera_context = self.agent.observe_live_camera(
            self.sessions.get_session(channel) if channel else None,
            channel or "", speaker_id, text, announce=False
        )

        turn_result = self.agent.process_turn(
            speaker_id=speaker_id,
            text=text,
            detected_language=language,
            mode="tutor",
            record_turn=record_turn,
            session_mode=session_mode,
            camera_context=camera_context
        )

        self._broadcast_turn_payloads(turn_result, speaker_id, text)
        self.generate_turn_artifacts(turn_result, speaker_id, text, channel, language)
        return turn_result

    def answer_direct_query(
        self,
        channel: Optional[str],
        question: str,
        speaker_id: str = "Learner",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Answers one participant's own question about the session (REQ-21).

        Deliberately thin: it resolves the mode the answer must obey and delegates. What
        matters is everything it does *not* do - no `generate_turn_artifacts`, no
        `_broadcast_turn_payloads`, no recorded turn. A direct query is an aside beside
        the conversation, so the only thing that leaves this method is the answer, back to
        the one person who asked for it.

        Not broadcast for the same reason: subtitles and cards describe the conversation
        both peers are in, and a private question - "what did she just say?" - is not
        that. Publishing it to the room would make the feature socially unusable in
        exactly the moment somebody needs it.
        """
        return self.agent.answer_query(
            question,
            speaker_id=speaker_id,
            detected_language=language,
            session_mode=self.session_mode_for(channel)
        )

    def generate_turn_artifacts(
        self,
        turn_result: Dict[str, Any],
        speaker_id: str,
        text: str,
        channel: Optional[str] = None,
        language: str = ""
    ) -> Dict[str, Any]:
        """
        Files the quizzes and notes a finalized turn produced (REQ-13 / REQ-14).

        Algorithm:
        1. Resolve the session this turn belongs to; without one there is no mode, and
           artifacts generated under a guessed mode are worse than none.
        2. Derive the turn's stable id so every artifact is source-linked to it.
        3. Generate notes and (mode permitting) a quiz, which stores and announces them.

        Runs on whichever thread called it - in the Convo AI path that is the background
        scaffolding executor, already off the voice-critical path (REQ-LAT-02). Failures
        are swallowed with a warning for the same reason the scaffolding call is: losing
        a note must not take down the conversation that produced it.
        """
        resolved_channel = channel or self.resolve_convoai_context(channel).get("channel")
        session = self.sessions.get_session(resolved_channel) if resolved_channel else None
        if session is None:
            return {"notes": [], "quiz": None}

        # REQ-15: the turn is persisted first, so a note's `source_turn_ids` still
        # resolve to something after the conversation has ended.
        turn, _ = self.artifacts.add_turn(
            TranscriptTurn.create(session.session_id, speaker_id, text, language)
        )
        turn_id = turn.id

        try:
            notes = self.artifact_generator.generate_notes(turn_result, session, [turn_id])
            quiz = self.artifact_generator.generate_quiz(turn_result, session, [turn_id])
        except Exception as exc:
            logger.warning(
                "Artifact generation failed for session %s: %s. The conversation is "
                "unaffected; this turn produced no notes or quiz.",
                session.session_id, exc
            )
            return {"notes": [], "quiz": None}

        self.research_turn(turn_result, session)
        return {"notes": notes, "quiz": quiz}

    def research_turn(self, turn_result: Dict[str, Any], session: SessionRecord) -> None:
        """
        Looks up a point this turn left uncertain, if there is one (REQ-18).

        Dispatched asynchronously even here: this method already runs off the voice path,
        but the vendor call is the slowest thing in the turn's tail, and holding notes and
        quizzes behind a search would put a working feature behind an optional one. The
        future is retained on the scaffolding list so diagnostics can join it.
        """
        query = self.agent.resolve_research_query(turn_result, session)
        if not query:
            return

        logger.info(
            "Turn in session %s left %r uncertain; researching it.",
            session.session_id, query
        )
        self._scaffolding_futures.append(
            self.tools.search_reference_async(session, query, requested_by="assistant")
        )

    def log_convoai_agent_health(self, channel: str) -> Optional[Dict[str, Any]]:
        """
        Polls and logs the Convo AI agent's lifecycle status (REQ-LLM-09).

        `query_agent()` and `/api/convoai/status` already existed, but nothing called
        them during a session, so an agent in RECOVERING or FAILED was indistinguishable
        in the terminal from a healthy one waiting for the learner to speak.

        Never raises: this is diagnostic, and a status-query failure must not take down
        the session start it runs alongside.
        """
        try:
            agent = self.convoai.query_agent(channel=channel)
        except Exception as exc:
            logger.warning(
                f"Could not query Convo AI agent health on channel '{channel}': {exc}"
            )
            return None

        if not agent:
            logger.warning(f"Convo AI agent health: no active agent on channel '{channel}'.")
            return None

        status = agent.get("status")
        message = (
            f"Convo AI agent health on channel '{channel}': status={status} "
            f"(agent_id={agent.get('agent_id')}, simulated={agent.get('simulated')})."
        )
        if status in (AGENT_STATUS_FAILED, AGENT_STATUS_RECOVERING):
            logger.error(f"{message} The agent is not healthy - it will not speak.")
        else:
            logger.info(message)
        return agent

    def stream_convoai_reply(
        self,
        speaker_id: str,
        text: str,
        language: str = "en",
        channel: Optional[str] = None
    ) -> Iterator[str]:
        """
        Streams the spoken reply for one Convo AI turn (REQ-LAT-02 / REQ-LAT-03).

        This is the only work the learner waits on. The structured scaffolding that used
        to be computed first - subtitles, idiom card, quiz, teacher alert - is scheduled
        separately by `schedule_convoai_scaffolding` so it no longer sits between the
        learner finishing a sentence and the agent starting to speak (D-LAT-1).

        The learner's turn is recorded here, so the scaffolding call must run with
        `record_turn=False` to avoid double-recording the same utterance.

        The session and channel travel with the turn so the agent can look through the
        learner's camera when they ask about something in front of them (REQ-CAM-03).
        That lookup is bounded and conditional - see `TeachingAgent.observe_live_camera`
        - so this stays the low-latency path it was.
        """
        return self.agent.generate_spoken_reply(
            speaker_id=speaker_id,
            text=text,
            detected_language=language,
            session_mode=self.session_mode_for(channel),
            session=self.sessions.get_session(channel) if channel else None,
            channel=channel or ""
        )

    def schedule_convoai_scaffolding(
        self,
        speaker_id: str,
        text: str,
        language: str = "en",
        channel: Optional[str] = None
    ) -> Future:
        """
        Runs the structured scaffolding generation off the voice-critical path (REQ-LAT-02).

        Returns the Future so tests (and any future caller that needs to sequence on it)
        can wait deterministically; nothing on the request path awaits it.

        Failures are logged rather than swallowed: a bare fire-and-forget Future discards
        its exception, which would make a permanently broken scaffolding path invisible
        while the voice reply kept working.
        """
        def run_scaffolding() -> Dict[str, Any]:
            return self.process_convoai_turn(
                speaker_id=speaker_id,
                text=text,
                language=language,
                record_turn=False,
                channel=channel
            )

        future = _TURN_EXECUTOR.submit(run_scaffolding)

        def report_failure(completed: Future) -> None:
            error = completed.exception()
            if error is not None:
                logger.warning(
                    f"Convo AI scaffolding generation failed for '{speaker_id}': {error}. "
                    f"The spoken reply was unaffected; subtitles and cards are missing "
                    f"for this turn."
                )

        future.add_done_callback(report_failure)
        self._scaffolding_futures.append(future)
        return future

    def wait_for_convoai_scaffolding(self, timeout: float = 10.0) -> None:
        """
        Blocks until scheduled scaffolding tasks finish. Test and diagnostic helper only -
        the request path must never call this, since not waiting is the entire point.
        """
        pending = list(self._scaffolding_futures)
        self._scaffolding_futures.clear()
        for future in pending:
            try:
                future.result(timeout=timeout)
            except Exception:
                # Already reported by the done-callback; swallowed so one failing turn
                # does not mask the remaining futures.
                pass


# ==============================================================================
# Flask Application & HTTP REST Endpoints
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "app"))
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")

app = Flask(
    __name__,
    static_folder=DIST_DIR if os.path.exists(os.path.join(DIST_DIR, "index.html")) else FRONTEND_DIR,
    static_url_path=""
)
# load_dotenv(override=True) has already run above, so .env is authoritative here.
# ECHOSPHERE_ENGINE selects the reasoning engine (REQ-LLM-01); an unset variable, a
# missing key, or an uninstalled SDK falls back to "mock" with a warning from
# TeachingAgent._downgrade_to_mock rather than a crash.
server_instance = EchoSphereServer(engine=os.getenv("ECHOSPHERE_ENGINE", "mock"))


def log_llm_bridge_status() -> None:
    """
    Reports the Custom LLM bridge's reachability once, shortly after startup.

    A broken tunnel is then visible in the first lines of the log rather than after a
    failed conversation - the incident this exists to prevent presented as an agent that
    joined, reported RUNNING, and said nothing, with no server-side signal at all.
    """
    if not server_instance.convoai.is_live_mode():
        return

    check = server_instance.convoai.check_llm_bridge()
    if check.reachable:
        logger.info("Custom LLM bridge reachable at %s.", check.url)
        return

    logger.error(
        "Custom LLM bridge NOT reachable. %s",
        server_instance.convoai.describe_bridge_problem(check)
    )


def schedule_llm_bridge_check(delay_seconds: float = BRIDGE_BOOT_CHECK_DELAY_SECONDS) -> None:
    """
    Runs the boot-time bridge check once the server is actually accepting connections.

    Deliberately deferred rather than run inline at import: the bridge URL normally
    points at a tunnel that forwards *to this process*, so probing it before the socket
    is listening reports a 502 every single start. A check that cries wolf on every boot
    is worse than no check - it trains the reader to skip the line that matters.
    """
    timer = threading.Timer(delay_seconds, log_llm_bridge_status)
    timer.daemon = True
    timer.start()


schedule_llm_bridge_check()


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning system status and component readiness.
    """
    return jsonify({
        "status": "online",
        "service": "EchoSphere Tandem Co-Teacher",
        "version": "1.1.0",
        "rtc_channel": server_instance.channel_name,
        "is_active": server_instance.is_active
    }), 200


@app.route("/api/session/start", methods=["POST"])
def api_start_session():
    """
    Starts the Agora RTC session under an explicit, immutable mode (REQ-12).

    `mode` is required: every downstream artifact - prompt selection, quizzes, notes -
    is mode-shaped, so a defaulted mode produces a session whose output silently does not
    match what the user asked for. Invalid or missing => 400.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)

    try:
        session = server_instance.sessions.create_session(
            channel=channel,
            mode=data.get("mode"),
            languages=data.get("languages") or [],
            participants=data.get("participants") or []
        )
    except InvalidSessionModeError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "supported_modes": SessionMode.values()
        }), 400

    # REQ-15: register the session with the artifact store now, so what it produces is
    # retrievable by session id after the channel has moved on.
    server_instance.artifacts.save_session(session)

    # REQ-23: a channel outlives the sessions held on it, so the new conversation starts
    # from silence rather than inheriting the previous pair's speaking balance.
    server_instance.speaking_balance.reset(channel)

    success = server_instance.start_session()
    return jsonify({
        "success": success,
        "channel": channel,
        "mode": session.mode.value,
        "session_id": session.session_id
    }), 200


@app.route("/api/session/stop", methods=["POST"])
def api_stop_session():
    """
    Stops the active session and closes its mode registration (REQ-12).
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)

    # REQ-17: the legs belong to the session, so they close with it. Left open, each one
    # holds a Gemini WebSocket for a channel nobody is in any more.
    server_instance.stop_translation(channel)

    # REQ-CAM-01: a buffered frame belongs to the conversation it was pushed during, and
    # nothing may look through a camera whose session has ended.
    server_instance.camera_buffer.clear(channel)

    closed = server_instance.sessions.end_session(channel)
    if closed is not None:
        # Stamp `ended_at` on the stored artifact; the artifact outlives the session.
        server_instance.artifacts.save_session(closed)
    success = server_instance.stop_session()
    return jsonify({
        "success": success,
        "channel": channel,
        "session_id": closed.session_id if closed else None
    }), 200


@app.route("/api/agent/query", methods=["POST"])
def api_agent_query():
    """
    Answers one participant's direct question to the AI co-teacher (REQ-21).

    The text path lives here rather than on the Convo AI voice pipeline because a typed
    question has no audio leg to drive: routing it through the spoken path would start a
    voice agent to answer something nobody asked out loud. A spoken question needs no new
    endpoint - it is already the REQ-09 direct-AI conversation.

    Governed like every other session endpoint (REQ-16). The answer is returned to the
    caller and to nobody else; see `answer_direct_query` for why it is not broadcast.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    question = (body.get("text") or body.get("query") or "").strip()
    if not question:
        return jsonify({
            "success": False,
            "error": "A direct query needs a `text` question to answer."
        }), 400

    answer = server_instance.answer_direct_query(
        meta.get("channel"),
        question,
        speaker_id=str(body.get("speaker_id") or resolve_actor(request) or "Learner"),
        language=str(body.get("language") or "en")
    )
    return jsonify({
        "success": True,
        "channel": meta.get("channel"),
        "session_id": meta["session_id"],
        "answer": answer
    }), 200


@app.route("/api/session/speaking", methods=["POST"])
def api_report_speaking_time():
    """
    Records one measured speech segment and broadcasts the balance (REQ-23).

    The browser is the reporter because it is where a real measurement exists: the local
    microphone track's own voice-activity boundaries. The server-side alternative would
    need raw per-participant PCM, which is exactly what TASK-11.9 deferred - so reporting
    a duration the client actually measured is the honest path to a real gauge, and it is
    still the server that accumulates, computes, and publishes it.

    Governed like every other session endpoint (REQ-16): speaking time is a claim about
    who was talking in a conversation, and an ungoverned endpoint lets a stranger rewrite
    somebody else's record of it.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    if "duration_ms" not in body:
        return jsonify({
            "success": False,
            "error": "A speech segment needs a measured `duration_ms`."
        }), 400

    try:
        duration_ms = int(body["duration_ms"])
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "`duration_ms` must be a whole number of milliseconds."
        }), 400

    channel = meta.get("channel") or server_instance.channel_name
    speaker_id = str(body.get("speaker_id") or resolve_actor(request) or "").strip()

    percentages = server_instance.record_speaking_time(channel, speaker_id, duration_ms)
    return jsonify({
        "success": True,
        "channel": channel,
        "session_id": meta["session_id"],
        "speaker_percentages": percentages,
        "total_ms": server_instance.speaking_balance.total_ms(channel)
    }), 200


@app.route("/api/session/status", methods=["GET"])
def api_session_status():
    """
    Reports the mode and identity of the session on a channel (REQ-12).

    Exists so a reconnecting client can recover the mode it must keep sending without
    re-creating the session - and so an operator can see which contract a channel is
    running under without reading the creation response back out of a log.
    """
    channel = request.args.get("channel", server_instance.channel_name)
    session = server_instance.sessions.get_session(channel)

    if session is None:
        return jsonify({
            "success": False,
            "channel": channel,
            "mode": None,
            "error": f"No active session on channel '{channel}'."
        }), 404

    body = session.to_dict()
    body["success"] = True
    return jsonify(body), 200


@app.route("/api/session/turn", methods=["POST"])
def api_process_turn():
    """
    Processes a student conversational turn via REST.
    """
    data = request.get_json() or {}
    speaker_id = data.get("speaker_id", "Learner")
    text = data.get("text", "")
    language = data.get("language", "en")

    result = server_instance.process_turn(
        speaker_id=speaker_id,
        text_or_audio=text,
        language=language,
        channel=data.get("channel")
    )
    return jsonify({"success": True, "result": result}), 200


# ------------------------------------------------------------------------------
# Gemini Live Bidirectional Translation (REQ-17)
# ------------------------------------------------------------------------------
@app.route("/api/translation/start", methods=["POST"])
def api_translation_start():
    """
    Starts the channel's Gemini Live Translate legs (TASK-11.3 / 11.6).

    Always 200 once the session exists, even when every leg is unavailable: translation
    being down is a degraded session, not a failed one, and returning an error here would
    hand the client a reason to abandon a voice call that is working perfectly well.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)

    try:
        router = server_instance.start_translation(
            channel=channel,
            participants=data.get("participants") or []
        )
    except SessionNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404

    return jsonify({
        "success": True,
        "channel": channel,
        "mode": router.session.mode.value,
        "available": router.is_available,
        "legs": router.leg_states(),
        "translated_audio": router.audio_gate_states(),
    }), 200


@app.route("/api/translation/stop", methods=["POST"])
def api_translation_stop():
    """Closes the channel's translation legs (session stop or participant departure)."""
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)

    stopped = server_instance.stop_translation(channel)
    return jsonify({"success": True, "channel": channel, "stopped": stopped}), 200


@app.route("/api/translation/status", methods=["GET"])
def api_translation_status():
    """
    Reports leg states and the per-participant translated-audio gate (REQ-17).

    A channel with no router reports empty rather than 404: "no translation running" is a
    legitimate steady state, and the REQ-06 control needs an answer either way.
    """
    channel = request.args.get("channel", server_instance.channel_name)
    router = server_instance.translation_routers.get(channel)

    if router is None:
        session = server_instance.sessions.get_session(channel)
        return jsonify({
            "success": True,
            "channel": channel,
            "mode": session.mode.value if session else None,
            "available": False,
            "legs": {},
            "translated_audio": {},
        }), 200

    return jsonify({
        "success": True,
        "channel": channel,
        "mode": router.session.mode.value,
        "available": router.is_available,
        "legs": router.leg_states(),
        "translated_audio": router.audio_gate_states(),
    }), 200


@app.route("/api/translation/audio", methods=["POST"])
def api_translation_audio_gate():
    """
    Flips one participant's translated-audio gate (TASK-11.5, REQ-06 controls).

    The gate is server-side because the router decides who is published to. A client-side
    mute would keep paying for the track and would leave two clients on the same leg
    disagreeing about whether anyone is hearing it.

    Transcripts are deliberately outside this gate: they still feed `TeachingAgent` and
    the artifacts, and stay available as an on-demand subtitle.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)
    participant_id = (data.get("participant_id") or "").strip()

    if not participant_id:
        return jsonify({
            "success": False,
            "error": "participant_id is required to set the translated-audio gate."
        }), 400

    router = server_instance.translation_routers.get(channel)
    if router is None:
        return jsonify({
            "success": False,
            "error": f"No translation legs running on channel '{channel}'."
        }), 404

    enabled = router.set_translated_audio_enabled(
        participant_id, bool(data.get("enabled", False))
    )
    return jsonify({
        "success": True,
        "channel": channel,
        "participant_id": participant_id,
        "translated_audio_enabled": enabled,
    }), 200


@app.route("/api/rtc/token", methods=["GET"])
def api_rtc_token():
    """
    Issues an Agora RTC channel token so the browser can join the voice channel.

    The student and the Convo AI agent must occupy the same channel for a spoken
    conversation to occur, so the frontend needs both the App ID and a channel token.
    The App Certificate never leaves the server.

    Returns `simulated: true` when the backend holds no real Agora credentials, which
    tells the client to stay in offline demo mode instead of attempting a live join.
    """
    channel = request.args.get("channel", server_instance.channel_name)
    try:
        uid = int(request.args.get("uid", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "uid must be an integer"}), 400

    token_client = AgoraVoiceChannelClient(channel_name=channel, uid=uid)
    token = token_client.generate_token(role="publisher")
    simulated = not (
        is_usable_credential(token_client.app_id)
        and is_usable_credential(token_client.app_certificate)
    )

    # The Convo AI Engine publishes live transcripts and agent-state events over RTM
    # on a channel named after the RTC channel. The client needs a separate RTM token
    # to subscribe. Its identity MUST be str(uid) - the same value the client logs in
    # with - or RTM auth fails in ways that surface as generic startup errors.
    rtm_token = token_client.generate_rtm_token(str(uid))

    # A second RTM identity for the session-event subscription (D-UIUX-2). It cannot
    # reuse the one above: RTM allows one live login per identity, so subscribing twice
    # under str(uid) would kick the Convo AI transcript client off whenever both are
    # running - the ambient and tutor paths are meant to coexist.
    #
    # Unique per request, not per uid, because RTM frees an identity a moment after
    # logout rather than immediately: a participant who left and rejoined within that
    # window was refused with "-10027 user ID already in use" and silently lost live
    # delivery for the rest of the session. Observed live on the first leave/rejoin
    # test, not anticipated. A receive-only identity has no reason to be stable, so
    # making it unique removes the race rather than retrying around it.
    events_rtm_user_id = f"{uid}-events-{uuid.uuid4().hex[:8]}"
    events_rtm_token = token_client.generate_rtm_token(events_rtm_user_id)

    return jsonify({
        "success": True,
        "app_id": token_client.app_id,
        "channel": channel,
        "uid": uid,
        "token": token,
        "rtm_token": rtm_token,
        "rtm_user_id": str(uid),
        "events_rtm_token": events_rtm_token,
        "events_rtm_user_id": events_rtm_user_id,
        "simulated": simulated
    }), 200


# ------------------------------------------------------------------------------
# Convo AI: Direct Spoken Conversation With The AI Co-Teacher (REQ-09 / REQ-10)
# ------------------------------------------------------------------------------
def _preflight_llm_bridge():
    """
    Refuses a live agent start when the Convo AI Engine could not call this backend.

    Returns a `(response, 502)` tuple to return to the client, or None to proceed.

    Skipped entirely in simulated mode - without Agora credentials no Engine will ever
    call the bridge, so its reachability is irrelevant and blocking the offline demo over
    it would be a fault rather than a safeguard. `CONVOAI_SKIP_BRIDGE_PREFLIGHT=1` is the
    documented escape hatch for deliberately starting an agent against a dead bridge.
    """
    if not server_instance.convoai.is_live_mode():
        return None

    if os.getenv("CONVOAI_SKIP_BRIDGE_PREFLIGHT", "").strip() in ("1", "true", "True"):
        logger.warning(
            "CONVOAI_SKIP_BRIDGE_PREFLIGHT is set: starting the agent without verifying "
            "that the Custom LLM bridge is reachable."
        )
        return None

    check = server_instance.convoai.check_llm_bridge()
    if check.reachable:
        return None

    message = server_instance.convoai.describe_bridge_problem(check)
    logger.error("Refusing to start a Convo AI agent. %s", message)
    return jsonify({
        "success": False,
        "error": message,
        "bridge": check.to_dict()
    }), 502


@app.route("/api/convoai/start", methods=["POST"])
def api_convoai_start():
    """
    Starts a Convo AI agent that joins the RTC channel and converses by audio.

    Returns the runtime agent_id used for all subsequent lifecycle calls. A 200 here
    means the request was accepted; the client must still wait for the RTC
    'user-joined' event before expecting agent audio.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)
    language = data.get("language", "en")

    # REQ-LLM-02 / REQ-LLM-05: this is the only point at which the learner's chosen
    # language is known, so capture it here for the Custom LLM bridge to read back, and
    # start the conversation from clean agent state.
    # Preflight the Custom LLM bridge before anything else is spent (D-BR-1). An agent
    # started against an unreachable bridge joins, reports RUNNING, and never speaks -
    # the failure this backend used to produce with a completely clean log.
    bridge_error = _preflight_llm_bridge()
    if bridge_error:
        return bridge_error

    # REQ-12: the mode is required here too, and creating the session is what registers
    # it. An unusable mode must fail before an agent is started, not after.
    try:
        server_instance.register_convoai_session(
            channel=channel,
            language=language,
            speaker_id=data.get("speaker_id", "Learner"),
            mode=data.get("mode")
        )
    except InvalidSessionModeError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "supported_modes": SessionMode.values()
        }), 400

    try:
        session = server_instance.convoai.start_agent(
            channel=channel,
            language=language,
            greeting=data.get("greeting"),
            system_prompt=data.get("system_prompt")
        )
    except Exception as exc:
        logger.error(f"Failed to start Convo AI agent on channel '{channel}': {exc}")
        return jsonify({"success": False, "error": str(exc)}), 502

    # REQ-LLM-09: a 200 from /join means accepted, not healthy. Poll shortly after so a
    # FAILED or RECOVERING agent is visible in the terminal rather than presenting as
    # an agent that simply never speaks.
    # Deliberately a timer thread, not _TURN_EXECUTOR: this task spends its whole life
    # sleeping, and the turn executor is the pool the voice path depends on. Parking a
    # sleeping worker there would starve exactly the requests this is meant to diagnose.
    health_timer = threading.Timer(
        CONVOAI_HEALTH_POLL_DELAY_SECONDS,
        server_instance.log_convoai_agent_health,
        args=(channel,)
    )
    health_timer.daemon = True
    health_timer.start()

    return jsonify({"success": True, "agent": session.to_dict()}), 200


@app.route("/api/convoai/stop", methods=["POST"])
def api_convoai_stop():
    """
    Stops the Convo AI agent currently attached to a channel.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", server_instance.channel_name)
    agent_id = data.get("agent_id")

    try:
        success = server_instance.convoai.stop_agent(agent_id=agent_id, channel=channel)
    except Exception as exc:
        logger.error(f"Failed to stop Convo AI agent on channel '{channel}': {exc}")
        return jsonify({"success": False, "error": str(exc)}), 502
    finally:
        # Cleared even on a failed stop: the conversation is over either way, and
        # leaking its history into the next session is the defect this prevents.
        server_instance.clear_convoai_session(channel)

    status_code = 200 if success else 404
    return jsonify({"success": success, "channel": channel}), status_code


@app.route("/api/convoai/event", methods=["POST"])
def api_convoai_event():
    """
    Receives agent-side lifecycle and error events relayed by the client (REQ-LLM-10).

    The Convo AI Engine reports agent state and module failures over RTM, which only
    the browser subscribes to. Without this relay an agent that joins and then fails
    inside its ASR, LLM, or TTS module is indistinguishable in the server log from a
    healthy one waiting for speech - the operator sees silence either way.

    Deliberately forgiving: this is called from a client error handler, so a malformed
    body must never raise. Losing a diagnostic is bad; replacing the original fault
    with a second one is worse.
    """
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", "unknown")
    event_type = str(data.get("type", "unknown"))
    payload = data.get("payload")

    if event_type == "agent_error":
        details = payload if isinstance(payload, dict) else {"message": payload}
        logger.error(
            "Convo AI AGENT ERROR on channel '%s': module=%r code=%r message=%r. "
            "The failing module names the layer to investigate: 'tts' points at the "
            "voice vendor, 'asr' at transcription, 'llm' at the Custom LLM bridge.",
            channel,
            details.get("module"), details.get("code"), details.get("message"),
        )
    elif event_type == "agent_state":
        state = payload.get("state") if isinstance(payload, dict) else payload
        logger.info("Convo AI agent state on channel '%s': %s", channel, state)
    else:
        logger.info(
            "Convo AI client event on channel '%s': type=%r payload=%r",
            channel, event_type, payload
        )

    return jsonify({"success": True}), 200


@app.route("/api/convoai/status", methods=["GET"])
def api_convoai_status():
    """
    Reports the lifecycle status of the Convo AI agent attached to a channel.
    """
    channel = request.args.get("channel", server_instance.channel_name)
    agent_id = request.args.get("agent_id")

    agent = server_instance.convoai.query_agent(agent_id=agent_id, channel=channel)
    return jsonify({
        "success": agent is not None,
        "agent": agent,
        "active_sessions": server_instance.convoai.list_sessions()
    }), 200


def _session_from_meta(meta: Dict[str, Any]) -> SessionRecord:
    """
    Rebuilds a session record from the metadata the artifact store kept.

    An artifact request can arrive after the session ended and its registry entry is
    gone, so the record every downstream collaborator expects - the artifact generator's
    events, the tool dispatcher's envelopes - is reconstructed from what was persisted
    rather than looked up in a registry that no longer holds it.
    """
    return SessionRecord(
        session_id=meta["session_id"],
        mode=SessionMode.parse(meta.get("mode") or SessionMode.LANGUAGE_LEARNING),
        channel=meta.get("channel") or "",
        languages=list(meta.get("languages") or []),
        participants=list(meta.get("participants") or [])
    )


def _resolve_governed_session(require_artifact: bool = False):
    """
    Resolves the session an artifact request addresses and authorizes the caller.

    Returns `(meta, artifact, None)` on success, or `(None, None, response)` carrying the
    404/403 the endpoint should return. Every artifact endpoint goes through here so the
    REQ-16 access check is structurally impossible to skip in one of them.

    A multipart request (REQ-22's camera capture) carries no JSON body, so the form is
    read as a fallback: without it, the one endpoint that has to upload a file would be
    the one endpoint that could not name its session.
    """
    body = request.get_json(silent=True) or {}
    if not body and request.form:
        body = request.form.to_dict()
    meta = server_instance.resolve_artifact_session(
        channel=request.args.get("channel") or body.get("channel"),
        session_id=request.args.get("session_id") or body.get("session_id")
    )
    if meta is None:
        return None, None, (jsonify({"success": False, "error": "No such session."}), 404)

    try:
        require_access(meta, resolve_actor(request))
    except AccessDeniedError as exc:
        return None, None, (jsonify({"success": False, "error": str(exc)}), 403)

    artifact = server_instance.artifacts.build_artifact(meta["session_id"])
    if require_artifact and artifact is None:
        return None, None, (jsonify({
            "success": False,
            "session_id": meta["session_id"],
            "error": "This session has produced no artifacts."
        }), 404)

    return meta, artifact, None


@app.route("/api/session/notes", methods=["GET"])
def api_session_notes():
    """
    Returns the notes captured for the session on a channel (REQ-14).
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    notes = server_instance.artifacts.list_notes(meta["session_id"])
    return jsonify({
        "success": True,
        "channel": meta.get("channel"),
        "session_id": meta["session_id"],
        "mode": meta.get("mode"),
        "notes": [note.to_dict() for note in notes]
    }), 200


@app.route("/api/session/notes/<note_id>", methods=["DELETE"])
def api_delete_session_note(note_id: str):
    """
    Deletes one note and announces it over RTC (REQ-14).

    A missing note is a 404 rather than a success: a stale client deleting twice must not
    produce a second `note.deleted` event for a note nobody else still shows.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    stored = server_instance.artifacts.get_note(note_id)
    if stored is None or stored.session_id != meta["session_id"]:
        return jsonify({"success": False, "note_id": note_id, "error": "No such note."}), 404

    deleted = server_instance.artifacts.delete_note(
        note_id, actor=resolve_actor(request) or "user"
    )
    server_instance.announce_note_change(deleted, meta, event_type="note.deleted")
    return jsonify({"success": True, "note": deleted.to_dict()}), 200


@app.route("/api/session/artifact", methods=["DELETE"])
def api_delete_session_artifact():
    """
    Deletes everything a session produced - transcript, notes, and quizzes (REQ-16).

    A hard delete, not a tombstone: the acceptance criterion is that the transcript
    becomes unavailable too, and a retained transcript still holds everything that was
    said no matter how the notes over it are flagged.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    session_id = meta["session_id"]
    deleted_entities = server_instance.artifacts.purge_session(session_id)
    if meta.get("channel"):
        server_instance.sessions.end_session(meta["channel"])

    logger.info(
        "Session %s deleted by %r: %d entities removed.",
        session_id, resolve_actor(request), deleted_entities
    )
    return jsonify({
        "success": True,
        "session_id": session_id,
        "deleted_entities": deleted_entities
    }), 200


@app.route("/api/session/artifact", methods=["GET"])
def api_session_artifact():
    """
    Returns everything one session produced, as the versioned envelope (REQ-15).

    Addressable by `channel` (the latest session on it) or by `session_id` - the second
    form is what a client uses after the conversation ended and the channel moved on.
    """
    _, artifact, error = _resolve_governed_session(require_artifact=True)
    if error:
        return error

    return jsonify({"success": True, "artifact": artifact.to_dict()}), 200


@app.route("/api/session/artifact/export", methods=["GET"])
def api_export_session_artifact():
    """
    Exports a session artifact (REQ-15).

    `markdown` always works and is served inline. `notion` is optional and answers 503
    when the server holds no Notion credentials - an unconfigured optional export must
    say so rather than appear to have succeeded. `target=page|database` selects which
    Notion destination to write to (TASK-12.5); omitted, the adapter uses whichever one
    this server has configured.
    """
    export_format = (request.args.get("format") or "markdown").strip().lower()
    target = (request.args.get("target") or "").strip().lower() or None
    if target is not None and target not in SUPPORTED_NOTION_TARGETS:
        return jsonify({
            "success": False,
            "error": f"Unsupported Notion export target {target!r}.",
            "supported_targets": list(SUPPORTED_NOTION_TARGETS)
        }), 400

    _, artifact, error = _resolve_governed_session(require_artifact=True)
    if error:
        return error

    if export_format == "markdown":
        return Response(
            render_markdown(artifact),
            mimetype="text/markdown",
            headers={
                "Content-Disposition":
                    f'inline; filename="echosphere-{artifact.session_id}.md"'
            }
        )

    if export_format == "notion":
        try:
            result = NotionExportAdapter().export(artifact, target=target)
        except ExportNotConfiguredError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        except Exception as exc:
            logger.error("Notion export failed for %s: %s", artifact.session_id, exc)
            return jsonify({"success": False, "error": str(exc)}), 502
        return jsonify({"success": True, "notion": result}), 200

    return jsonify({
        "success": False,
        "error": f"Unsupported export format {export_format!r}.",
        "supported_formats": ["markdown", "notion"]
    }), 400


@app.route("/api/session/notes/<note_id>", methods=["PATCH"])
def api_edit_session_note(note_id: str):
    """
    Edits one stored note, recording who changed it (REQ-15 / REQ-16).

    Deliberately has no approval gate on the way in - the note was already persisted the
    moment the turn finalized (REQ-15) - so this is a correction of stored content, not
    an acceptance step. The edit pins the note against later regeneration.
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({
            "success": False,
            "error": "An edit must supply non-empty 'text'."
        }), 400

    meta, _, error = _resolve_governed_session()
    if error:
        return error

    stored = server_instance.artifacts.get_note(note_id)
    if stored is None or stored.session_id != meta["session_id"]:
        return jsonify({"success": False, "note_id": note_id, "error": "No such note."}), 404

    edited = server_instance.artifacts.edit_note(
        note_id,
        text=text,
        owner=data.get("owner"),
        due_at=data.get("due_at"),
        actor=str(data.get("actor") or "user")
    )
    server_instance.announce_note_change(edited, meta)
    return jsonify({"success": True, "note": edited.to_dict()}), 200


@app.route("/api/session/quizzes", methods=["GET"])
def api_session_quizzes():
    """
    Returns the quizzes generated for the session on a channel (REQ-13).
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    quizzes = server_instance.artifacts.list_quizzes(meta["session_id"])
    return jsonify({
        "success": True,
        "channel": meta.get("channel"),
        "session_id": meta["session_id"],
        "mode": meta.get("mode"),
        "quizzes": [quiz.to_dict() for quiz in quizzes]
    }), 200


# ------------------------------------------------------------------------------
# Agent tools: Google Search, Anki MCP, Google Calendar (REQ-18 / REQ-19 / REQ-20)
# ------------------------------------------------------------------------------
# Status codes follow the Notion export's precedent: an unconfigured optional integration
# answers 503 and a failed call answers 502, so a client can tell "this server never had
# that tool" from "that tool is having a bad day" - and neither ever looks like success.
TOOL_STATUS_CODES = {ToolState.OK: 200, ToolState.UNAVAILABLE: 503, ToolState.FAILED: 502}


def _tool_response(result, key: str, extra: Optional[Dict[str, Any]] = None):
    """Renders one `ToolResult` as this API's JSON response."""
    body: Dict[str, Any] = {
        "success": result.ok,
        "tool": result.tool,
        "state": result.state,
        key: result.payload,
    }
    if result.reason:
        body["error"] = result.reason
    body.update(extra or {})
    return jsonify(body), TOOL_STATUS_CODES.get(result.state, 502)


@app.route("/api/tools/status", methods=["GET"])
def api_tools_status():
    """
    Reports which agent tools this server can run (REQ-18-20).

    The frontend needs this to avoid offering a control whose only possible outcome is a
    503; it deliberately carries no credential material, only a boolean per tool.
    """
    return jsonify({"success": True, "tools": server_instance.tools.status()}), 200


@app.route("/api/tools/search", methods=["POST"])
def api_tools_search():
    """
    Looks up reference or task material for a session and publishes the card (REQ-18).
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"success": False, "error": "A lookup needs a `query`."}), 400

    materials = body.get("materials")
    result = server_instance.tools.search_reference(
        _session_from_meta(meta),
        query,
        materials=None if materials is None else bool(materials),
        requested_by=resolve_actor(request) or ""
    )
    return _tool_response(result, "card", {"session_id": meta["session_id"]})


@app.route("/api/tools/anki/export", methods=["POST"])
def api_tools_anki_export():
    """
    Exports a session's vocabulary and terminology to Anki (REQ-19).

    Governed like every other artifact endpoint: the cards carry what was said, so
    sending them to a stranger's collection is the same disclosure as the transcript.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    notes = server_instance.artifacts.list_notes(meta["session_id"])
    result = server_instance.tools.export_vocabulary(
        _session_from_meta(meta), notes, deck=(body.get("deck") or None)
    )
    return _tool_response(result, "export", {"session_id": meta["session_id"]})


def _read_uploaded_frame(body, missing_message: str):
    """
    Reads one camera frame out of the request, however the client chose to send it.

    Accepts the capture two ways because both are natural in a browser: a multipart
    `image` file, which is what `canvas.toBlob` produces, or a base64 `image_base64`
    field for a client that already holds a data URL. Base64 inflates the payload by a
    third on a link that is already carrying a live conversation, so the file form is
    preferred - but requiring it would push that encoding decision onto every caller.

    Shared by the REQ-22 button capture and the REQ-CAM-01 periodic push so the two
    cannot drift into accepting different things.

    Returns:
        `(frame, mime_type, None)` on success, or `(None, "", response)` carrying the 400
        the endpoint should return.
    """
    upload = request.files.get("image")
    if upload is not None:
        frame = upload.read()
        mime_type = upload.mimetype or "image/jpeg"
    else:
        encoded = (body.get("image_base64") or "").strip()
        # A data URL ("data:image/jpeg;base64,...") is what a canvas capture hands a
        # client, so accept it rather than making the browser strip its own prefix.
        if "," in encoded and encoded.lower().startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            frame = base64.b64decode(encoded, validate=True) if encoded else b""
        except (ValueError, binascii.Error):
            return None, "", (jsonify({
                "success": False,
                "error": "`image_base64` is not valid base64 data."
            }), 400)
        mime_type = (body.get("mime_type") or "image/jpeg").strip()

    if not frame:
        return None, "", (jsonify({"success": False, "error": missing_message}), 400)

    return frame, mime_type, None


@app.route("/api/session/camera/stream", methods=["POST"])
def api_session_camera_stream():
    """
    Buffers the frame the camera is currently showing, for the agent to look at (REQ-CAM-01).

    Deliberately the cheapest endpoint in this file: it stores one frame per channel and
    returns. No vendor call, no event, no artifact. It runs every few seconds for as long
    as Camera Assist is open, so anything it did per request would be paid for by a
    participant who has not asked a question yet - which is exactly the cost REQ-CAM-04
    bounds.

    `active: false` is the opt-out: closing the camera panel clears the buffer at once
    rather than leaving one last frame describable until its TTL runs out. Nothing here
    persists a frame - the buffer is memory-only and self-expiring (see `camera_stream`).
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    if not body and request.form:
        body = request.form.to_dict()

    channel = meta.get("channel") or ""

    # A client that has just turned the camera off sends no frame; it is asking the
    # server to forget the last one, which is the participant's own opt-out.
    active = body.get("active")
    if active in (False, "false", "False", 0, "0"):
        server_instance.camera_buffer.clear(channel)
        return jsonify({
            "success": True, "session_id": meta["session_id"], "buffered": False
        })

    frame, mime_type, error = _read_uploaded_frame(
        body, "A camera frame push needs an `image` file or `image_base64` data."
    )
    if error:
        return error

    server_instance.camera_buffer.put(channel, frame, mime_type=mime_type)
    return jsonify({"success": True, "session_id": meta["session_id"], "buffered": True})


@app.route("/api/tools/vision", methods=["POST"])
def api_tools_vision():
    """
    Explains one captured camera frame as a material card (REQ-22).

    The capture arrives as a multipart file or as base64 (see `_read_uploaded_frame`).
    This is the participant-triggered path and it is unchanged by REQ-CAM-03's
    voice-initiated one: the two share only the underlying `CameraVisionTool`.

    The frame is passed to the tool and dropped: nothing here stores it, and REQ-22 is
    explicit that no image persists past generating the card unless the participant
    chooses to save the result as a note.
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    if not body and request.form:
        body = request.form.to_dict()

    frame, mime_type, error = _read_uploaded_frame(
        body, "A camera lookup needs an `image` file or `image_base64` data."
    )
    if error:
        return error

    result = server_instance.tools.describe_camera_frame(
        _session_from_meta(meta),
        frame,
        mime_type=mime_type,
        question=(body.get("question") or ""),
        requested_by=resolve_actor(request) or ""
    )
    return _tool_response(result, "card", {"session_id": meta["session_id"]})


@app.route("/api/tools/calendar/schedule", methods=["POST"])
def api_tools_schedule_meeting():
    """
    Books a follow-up meeting for a session and invites its attendees (REQ-20).

    Attendee addresses travel in the request rather than being read from the session:
    a `SessionRecord` records participants by display name, and this system has no
    registry mapping those to email addresses yet (see the plan's TASK-12.4 note).
    """
    meta, _, error = _resolve_governed_session()
    if error:
        return error

    body = request.get_json(silent=True) or {}
    start_time = body.get("start_time")
    if not start_time:
        return jsonify({
            "success": False,
            "error": "A meeting needs a `start_time` (ISO-8601, e.g. 2026-09-10T09:00:00Z)."
        }), 400

    result = server_instance.tools.schedule_meeting(
        _session_from_meta(meta),
        summary=(body.get("summary") or ""),
        start_time=start_time,
        duration_minutes=int(body.get("duration_minutes") or 30),
        attendees=body.get("attendees") or [],
        description=(body.get("description") or ""),
        requested_by=resolve_actor(request) or ""
    )
    return _tool_response(result, "meeting", {"session_id": meta["session_id"]})


def _split_for_streaming(text: str, size: int = SSE_CHUNK_CHARS) -> List[str]:
    """
    Splits a already-complete reply into incremental SSE deltas.

    Fixed-width slices rather than word boundaries: Japanese and Hindi replies contain
    long runs with no spaces, and the Convo AI Engine concatenates deltas before
    synthesis, so the split point does not affect the spoken output.

    Since REQ-LAT-03 this is only used for text that was never streamed in the first
    place - the canned fallback line. A live provider reply is forwarded delta by delta
    as the model produces it, which is the point: re-splitting a finished string paces
    delivery but cannot make the first token arrive any sooner.
    """
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


@app.route("/chat/completions", methods=["POST"])
def convoai_custom_llm_bridge():
    """
    Custom LLM backend consumed by the Convo AI Engine (REQ-11).

    Exposes the existing TeachingAgent orchestrator behind the OpenAI-compatible
    streaming contract the Convo AI Engine requires, so tri-lingual scaffolding logic
    is reused rather than reimplemented for spoken AI conversations.

    Algorithm:
    1. Extract the most recent user message from the OpenAI-style messages array.
    2. Resolve language and speaker from the session context captured at session start,
       with the request body as an override for direct/manual calls.
    3. Log the inbound request shape at debug level (REQ-LLM-04).
    4. Run the shared TeachingAgent turn pipeline inside the SSE generator, under a
       bounded timeout, and stream chunks out as reply text becomes available.

    Step 4 runs inside the generator on purpose (REQ-LLM-06): computing the whole reply
    before returning the Response means the client waits the full model latency in
    silence, and the Convo AI Engine expires an idle agent in `idle_timeout` seconds.
    """
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    model = data.get("model", "echosphere-teaching-agent")

    # Step 1: Locate the latest user utterance transcribed by the Convo AI Engine
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            # Content may arrive as a plain string or as OpenAI content parts
            if isinstance(content, list):
                user_text = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
            else:
                user_text = str(content)
            break

    # Step 2: Resolve session context (REQ-LLM-02 / REQ-LLM-03).
    # Agora's Custom LLM body carries no `language` or `speaker_id` field, so
    # data.get("language", "en") always resolved to English and a real model never
    # learned the target language. The session context captured by /api/convoai/start is
    # therefore authoritative; an explicit request field still wins so direct and manual
    # calls (including the test suite) can drive the bridge without a live session.
    session_context = server_instance.resolve_convoai_context(data.get("channel"))
    if "language" in data:
        language = data["language"]
        language_source = "request override"
    elif session_context.get("language"):
        language = session_context["language"]
        language_source = f"session context (channel '{session_context.get('channel')}')"
    else:
        language = "en"
        language_source = "default"

    speaker_id = data.get("speaker_id") or session_context.get("speaker_id") or "Learner"

    # Step 3: Inbound request shape (REQ-LLM-04). Settles empirically which fields Agora
    # actually sends, which the documented contract alone does not.
    logger.debug(
        "Convo AI Custom LLM request: fields=%s, messages=%d, speaker_id=%r, "
        "language=%r (source: %s), user_text=%r",
        sorted(data.keys()), len(messages), speaker_id, language, language_source, user_text
    )

    # Step 4: Stream the reply as SSE chunks, forwarding provider deltas as they arrive
    def generate_sse() -> Iterator[str]:
        completion_id = f"chatcmpl-echosphere-{int(time.time() * 1000)}"
        created = int(time.time())

        def chunk(delta: Dict[str, Any], finish_reason: Optional[str]) -> str:
            body = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
            }
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        started = time.time()
        first_chunk_at: Optional[float] = None

        # A worker thread drains the provider stream into a queue while this generator
        # forwards whatever has arrived. A queue rather than `future.result(timeout=...)`
        # because the reply is now a stream: the bound that matters is the wait for the
        # *next* delta, not for one finished blob (Task 3.5). A stall at any point -
        # including before the first token - therefore falls back to speech instead of
        # leaving the agent silent until the Engine's idle_timeout kills the session.
        deltas: "queue.Queue[tuple]" = queue.Queue()

        def drain_provider_stream() -> None:
            try:
                for delta in server_instance.stream_convoai_reply(
                    speaker_id=speaker_id, text=user_text, language=language,
                    channel=session_context.get("channel")
                ):
                    deltas.put(("delta", delta))
            except Exception as exc:
                deltas.put(("error", exc))
            finally:
                deltas.put(("done", None))

        _TURN_EXECUTOR.submit(drain_provider_stream)

        # Scaffolding runs in parallel with the spoken reply, never before it (REQ-LAT-02).
        server_instance.schedule_convoai_scaffolding(
            speaker_id=speaker_id, text=user_text, language=language,
            channel=session_context.get("channel")
        )

        emitted_any = False
        while True:
            try:
                kind, payload = deltas.get(timeout=CONVOAI_LLM_TIMEOUT_SECONDS)
            except queue.Empty:
                logger.warning(
                    f"Reasoning engine produced no output for "
                    f"{CONVOAI_LLM_TIMEOUT_SECONDS}s; speaking the '{language}' fallback."
                )
                break

            if kind == "error":
                logger.error(f"Convo AI turn failed: {payload}. Speaking the fallback reply.")
                break
            if kind == "done":
                break

            if not payload:
                continue

            if not emitted_any:
                emitted_any = True
                first_chunk_at = time.time()
                yield chunk({"role": "assistant", "content": payload}, None)
            else:
                yield chunk({"content": payload}, None)

        if not emitted_any:
            # Nothing was spoken: emit the canned line so the turn still produces audio.
            fallback = FALLBACK_REPLIES.get(language, FALLBACK_REPLIES["en"])
            pieces = _split_for_streaming(fallback)
            first_chunk_at = time.time()
            yield chunk({"role": "assistant", "content": pieces[0]}, None)
            for piece in pieces[1:]:
                yield chunk({"content": piece}, None)

        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

        # REQ-LAT-01: the measurement acceptance criterion 3 is read from. Logged at INFO
        # so a latency regression is visible without enabling debug logging.
        time_to_first = (first_chunk_at - started) if first_chunk_at else float("nan")
        logger.info(
            "Convo AI turn latency: first_chunk=%.3fs, total=%.3fs (engine=%s, language=%s).",
            time_to_first, time.time() - started, server_instance.agent.engine, language
        )

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/assets/<path:path>")
def serve_assets(path):
    """
    Serves compiled static assets from app/dist/assets.
    """
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(os.path.join(assets_dir, path)):
        return send_from_directory(assets_dir, path)
    return "Asset not found", 404


@app.route("/src/<path:path>")
def serve_src(path):
    """
    Serves source assets from app/src.
    """
    src_dir = os.path.join(FRONTEND_DIR, "src")
    if os.path.exists(os.path.join(src_dir, path)):
        return send_from_directory(src_dir, path)
    return "Source not found", 404


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """
    Serves the EchoSphere Tandem Co-Teacher frontend single-page application.
    Checks app/dist/index.html first, then falls back to app/index.html.
    """
    # If a specific static file exists in dist or app, serve it
    if path:
        if os.path.exists(os.path.join(DIST_DIR, path)):
            return send_from_directory(DIST_DIR, path)
        if os.path.exists(os.path.join(FRONTEND_DIR, path)):
            return send_from_directory(FRONTEND_DIR, path)

    # Default to index.html
    if os.path.exists(os.path.join(DIST_DIR, "index.html")):
        return send_from_directory(DIST_DIR, "index.html")
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")

    return "<h1>EchoSphere Tandem Co-Teacher</h1><p>Frontend assets initializing...</p>", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
