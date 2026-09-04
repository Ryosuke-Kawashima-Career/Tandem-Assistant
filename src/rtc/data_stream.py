"""
Summary:
    data_stream.py provides high-level message serialization and dispatching for
    Agora SD-RTN™ RTC Data Streams in EchoSphere.
    It structures and synchronizes live UI payloads across all connected peers and teacher
    dashboards, including real-time tri-lingual subtitles, Romaji/transliterations,
    cultural idiom cards, conversation prompts, quizzes, speaking balance metrics, and
    the session artifact events (`quiz.created`, `note.upserted`, `note.deleted`).

Key Classes:
    - DataStreamManager: Coordinates serializing, throttling, and broadcasting structured
      JSON event payloads via the underlying AgoraVoiceChannelClient.
"""

import time
import json
import logging
from typing import Optional, Dict, Any, List
from src.rtc.agora_client import AgoraVoiceChannelClient, RTCDataStreamPacket

logger = logging.getLogger("echosphere.rtc.data_stream")


class DataStreamManager:
    """
    Manages structured event serialization and broadcasting over Agora RTC Data Streams.
    """

    def __init__(self, agora_client: Optional[AgoraVoiceChannelClient] = None):
        """
        Initialize the DataStreamManager with an active Agora RTC client bridge.
        
        Algorithm:
        1. Store reference to the Agora RTC client.
        2. Initialize transmission history log for auditing and debugging.
        """
        self.agora_client = agora_client
        self.packet_history: List[RTCDataStreamPacket] = []
        logger.info("DataStreamManager initialized.")

    def _dispatch(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Internal dispatch helper that encapsulates payload in RTCDataStreamPacket and transmits.
        
        Algorithm:
        1. Create an RTCDataStreamPacket timestamped to the current millisecond.
        2. Record packet in local transmission history.
        3. If agora_client is attached, invoke send_data_stream / send_data_stream_message.
        4. Return success status boolean.
        """
        packet = RTCDataStreamPacket(event_type=event_type, payload=payload)
        self.packet_history.append(packet)

        if self.agora_client is not None:
            # Check for send_data_stream or send_data_stream_message methods
            if hasattr(self.agora_client, "send_data_stream"):
                return bool(self.agora_client.send_data_stream(event_type, payload))
            elif hasattr(self.agora_client, "send_data_stream_message"):
                return bool(self.agora_client.send_data_stream_message(event_type, payload))
            else:
                logger.warning("Attached Agora client does not support data stream dispatch.")
                return False

        logger.debug(f"DataStreamManager dispatched offline packet: {event_type}")
        return True

    def send_subtitle(
        self,
        speaker: str,
        text: str,
        transliteration: str = "",
        translation_en: str = "",
        translation_ja: str = "",
        translation_hi: str = ""
    ) -> bool:
        """
        Broadcasts a live multi-lingual subtitle payload to student and teacher interfaces.
        
        Algorithm:
        1. Structure subtitle payload with speaker identification and tri-lingual translations.
        2. Forward payload to _dispatch with event_type 'subtitles'.
        """
        payload = {
            "speaker": speaker,
            "original_text": text,
            "transliteration": transliteration,
            "translation_en": translation_en,
            "translation_ja": translation_ja,
            "translation_hi": translation_hi
        }
        return self._dispatch("subtitles", payload)

    def send_idiom_card(
        self,
        phrase: str,
        romaji: str = "",
        meaning: str = "",
        cultural_note: str = ""
    ) -> bool:
        """
        Broadcasts a cultural idiom or vocabulary annotation card.
        
        Algorithm:
        1. Construct idiom card payload containing phrase, romaji, meaning, and cultural note.
        2. Forward payload to _dispatch with event_type 'idiom_card'.
        """
        payload = {
            "detected": True,
            "phrase": phrase,
            "romaji": romaji,
            "meaning": meaning,
            "cultural_note": cultural_note
        }
        return self._dispatch("idiom_card", payload)

    def send_topic(self, topic_title: str, prompt: str = "") -> bool:
        """
        Broadcasts a new conversational topic or peer prompt.
        
        Algorithm:
        1. Construct topic payload with title and discussion prompt.
        2. Forward payload to _dispatch with event_type 'topic_prompt'.
        """
        payload = {
            "topic_title": topic_title,
            "prompt": prompt
        }
        return self._dispatch("topic_prompt", payload)

    def send_quiz(
        self,
        question: str,
        options: Optional[List[str]] = None,
        correct_index: int = 0,
        explanation: str = ""
    ) -> bool:
        """
        Broadcasts an interactive comprehension or register quiz widget.
        
        Algorithm:
        1. Construct quiz payload with question, selectable options, correct index, and explanation.
        2. Forward payload to _dispatch with event_type 'quiz'.
        """
        payload = {
            "active": True,
            "question": question,
            "options": options or [],
            "correct_index": correct_index,
            "explanation": explanation
        }
        return self._dispatch("quiz", payload)

    def send_teacher_alert(self, message: str, severity: str = "info") -> bool:
        """
        Broadcasts an oversight notification specifically for the human teacher dashboard.
        
        Algorithm:
        1. Construct alert payload with message content and severity level.
        2. Forward payload to _dispatch with event_type 'teacher_alert'.
        """
        payload = {
            "alert_required": True,
            "message": message,
            "severity": severity
        }
        return self._dispatch("teacher_alert", payload)

    def send_artifact_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Broadcasts a session artifact event: `quiz.created`, `note.upserted`, `note.deleted`.

        Algorithm:
        1. Accept an already-enveloped payload (schema version, event id, session, mode).
        2. Forward it unchanged to _dispatch under the given event type.

        Unlike the widget events above, the envelope is built by the artifact generator
        rather than here: those events describe what to draw right now, while these
        describe a stored entity, and the entity's id, revision, and session are what a
        receiver deduplicates and files it by (spec section 5).
        """
        return self._dispatch(event_type, payload)

    def send_translation_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Broadcasts a Gemini Live Translate event (REQ-17): `translation.status`,
        `translation.input_transcript`, `translation.output_transcript`.

        Algorithm:
        1. Accept an already-enveloped payload (schema version, session, mode, leg id).
        2. Forward it unchanged to _dispatch under the given event type.

        Enveloped by `TranslationRouter` rather than here for the same reason artifact
        events are: the leg id, sequence, and session are what a receiver orders and
        deduplicates by, and only the router knows them.
        """
        return self._dispatch(event_type, payload)

    def send_tool_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Broadcasts an agent tool event (REQ-18–20): `tool.status`, `reference.card`,
        `anki.exported`, `meeting.scheduled`.

        Algorithm:
        1. Accept an already-enveloped payload (schema version, session, mode, tool).
        2. Forward it unchanged to _dispatch under the given event type.

        Enveloped by `ToolDispatcher` rather than here, for the same reason artifact and
        translation events are: only the dispatcher knows which session and which tool
        produced the outcome, and those are what a receiver files and deduplicates by.
        """
        return self._dispatch(event_type, payload)

    def send_speaking_balance(self, speaker_stats: Dict[str, int]) -> bool:
        """
        Broadcasts real-time speaking time percentage metrics to the classroom UI.
        
        Algorithm:
        1. Construct speaking balance payload from speaker statistics mapping.
        2. Forward payload to _dispatch with event_type 'speaking_balance'.
        """
        payload = {
            "speaker_percentages": speaker_stats
        }
        return self._dispatch("speaking_balance", payload)
