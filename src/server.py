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
import json
import time
import logging
from typing import Optional, Dict, Any, Iterator
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
from src.rtc.convoai_client import ConvoAIClient
from src.agent.orchestrator import TeachingAgent
from src.audio.vad_processor import VoiceActivityDetector
from src.audio.stt_transcriber import STTTranscriber
from src.audio.tts_synthesizer import TTSSynthesizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echosphere.server")


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
        5. Track active session state.
        """
        self.channel_name = channel_name
        self.engine = engine

        # Step 2: RTC and Data Stream
        self.rtc_client = AgoraVoiceChannelClient(channel_name=channel_name)
        self.data_stream = DataStreamManager(self.rtc_client)

        # Step 3: AI and Audio Pipeline
        self.agent = TeachingAgent(engine=engine)
        self.vad = VoiceActivityDetector()
        self.stt = STTTranscriber(engine=engine)
        self.tts = TTSSynthesizer()

        # Step 4: Convo AI direct audio conversation engine (REQ-09 / REQ-10)
        self.convoai = ConvoAIClient()

        self.is_active = False
        logger.info(f"EchoSphereServer initialized for channel '{channel_name}' (engine: {engine}).")

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
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Processes a single conversational turn from a student learner.

        Algorithm:
        1. If input is raw PCM audio bytes, transcribe speech via STTTranscriber.
        2. Forward transcription to TeachingAgent for mediation, transliteration, and cultural idiom analysis.
        3. Broadcast live subtitles and detected idiom cards over Agora RTC Data Stream.
        4. Synthesize AI voice response via TTSSynthesizer and publish to RTC channel if needed.
        5. Return full structured turn payload.
        """
        # Step 1: Speech-to-text if audio bytes provided
        if isinstance(text_or_audio, (bytes, bytearray)):
            transcription = self.stt.transcribe(text_or_audio, language=language, speaker_id=speaker_id)
            spoken_text = transcription.text
            detected_lang = transcription.language
        else:
            spoken_text = str(text_or_audio)
            detected_lang = language

        # Step 2: AI Agent Mediation
        turn_result = self.agent.process_turn(
            speaker_id=speaker_id,
            text=spoken_text,
            detected_language=detected_lang
        )

        # Step 3: Broadcast over RTC Data Stream
        self._broadcast_turn_payloads(turn_result, speaker_id, spoken_text)

        # Step 4: AI Speech Synthesis & RTC Audio Track
        spoken_response = turn_result.get("spoken_response", "")
        if spoken_response:
            ai_audio = self.tts.synthesize(spoken_response, language=turn_result.get("spoken_language", "en"))
            self.rtc_client.publish_audio_frame(ai_audio)

        return turn_result

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
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Processes one conversational turn arriving from the Convo AI Engine (REQ-11).

        Differs from process_turn in exactly one respect: speech synthesis is owned by
        the Convo AI Engine's TTS vendor, so this path returns reply text and never
        calls the local TTS synthesizer or publishes an audio frame. Transcription is
        also skipped because the Convo AI Engine performs ASR upstream.

        Algorithm:
        1. Forward the transcribed text to the shared TeachingAgent orchestrator.
        2. Broadcast subtitles, idiom cards, and quizzes over the RTC Data Stream.
        3. Return the structured turn payload for the SSE bridge to stream back.
        """
        turn_result = self.agent.process_turn(
            speaker_id=speaker_id,
            text=text,
            detected_language=language
        )

        self._broadcast_turn_payloads(turn_result, speaker_id, text)
        return turn_result


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
server_instance = EchoSphereServer()


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
    Starts the Agora RTC and AI co-teacher session.
    """
    success = server_instance.start_session()
    return jsonify({"success": success, "channel": server_instance.channel_name}), 200


@app.route("/api/session/stop", methods=["POST"])
def api_stop_session():
    """
    Stops the active session.
    """
    success = server_instance.stop_session()
    return jsonify({"success": success}), 200


@app.route("/api/session/turn", methods=["POST"])
def api_process_turn():
    """
    Processes a student conversational turn via REST.
    """
    data = request.get_json() or {}
    speaker_id = data.get("speaker_id", "Learner")
    text = data.get("text", "")
    language = data.get("language", "en")

    result = server_instance.process_turn(speaker_id=speaker_id, text_or_audio=text, language=language)
    return jsonify({"success": True, "result": result}), 200


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

    return jsonify({
        "success": True,
        "app_id": token_client.app_id,
        "channel": channel,
        "uid": uid,
        "token": token,
        "rtm_token": rtm_token,
        "rtm_user_id": str(uid),
        "simulated": simulated
    }), 200


# ------------------------------------------------------------------------------
# Convo AI: Direct Spoken Conversation With The AI Co-Teacher (REQ-09 / REQ-10)
# ------------------------------------------------------------------------------
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

    status_code = 200 if success else 404
    return jsonify({"success": success, "channel": channel}), status_code


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


@app.route("/chat/completions", methods=["POST"])
def convoai_custom_llm_bridge():
    """
    Custom LLM backend consumed by the Convo AI Engine (REQ-11).

    Exposes the existing TeachingAgent orchestrator behind the OpenAI-compatible
    streaming contract the Convo AI Engine requires, so tri-lingual scaffolding logic
    is reused rather than reimplemented for spoken AI conversations.

    Algorithm:
    1. Extract the most recent user message from the OpenAI-style messages array.
    2. Run the shared TeachingAgent turn pipeline (also broadcasting visual payloads).
    3. Stream the reply back as Server-Sent Events terminated by the [DONE] sentinel.
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

    # Step 2: Reuse the shared orchestrator (no local TTS on this path)
    language = data.get("language", "en")
    turn_result = server_instance.process_convoai_turn(
        speaker_id=data.get("speaker_id", "Learner"),
        text=user_text,
        language=language
    )
    reply_text = turn_result.get("spoken_response", "")

    # Step 3: Stream the reply as SSE chunks
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

        yield chunk({"role": "assistant", "content": reply_text}, None)
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

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
