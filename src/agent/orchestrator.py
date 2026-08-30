"""
Summary:
    orchestrator.py defines the TeachingAgent class, the core AI intelligence orchestrator
    in EchoSphere.
    It tracks real-time multi-speaker conversation turns, maintains balance metrics,
    detects code-switching across Hindi, Japanese, and English, generates structured
    pedagogical scaffolding (subtitles, transliterations, idiom cards, and quizzes),
    and produces spoken AI co-teacher interventions.

Key Classes:
    - TeachingAgent: Main conversational agent coordinating LLM reasoning, turn analysis,
      and real-time mediation across tri-lingual tandem classrooms.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Iterator
from src.agent.prompts import (
    SYSTEM_PROMPT_TANDEM_TEACHER,
    SYSTEM_PROMPT_TANDEM_TUTOR,
    SYSTEM_PROMPT_TANDEM_TUTOR_VOICE,
    create_teaching_prompt,
    create_tutor_prompt,
    create_tutor_voice_prompt,
    create_silence_breaker_prompt
)

logger = logging.getLogger("echosphere.agent.orchestrator")

# Default reasoning models (REQ-LAT-04). Both are the vendors' current lowest-latency
# conversational tiers, chosen because the student is waiting in a live voice call:
# the previously hardcoded gpt-4o / gemini-2.5-flash are stronger but measurably slower
# to first token, and time-to-first-token is what this path is optimised for (D-LAT-2).
#
# These are overridable via ECHOSPHERE_OPENAI_MODEL / ECHOSPHERE_GEMINI_MODEL precisely
# because model names and their relative speed change independently of this repo - check
# the current provider matrix rather than trusting these constants indefinitely.
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Models for the structured scaffolding call (subtitles, idiom card, quiz). Deliberately
# stronger than the voice tiers above: since REQ-LAT-02 this call no longer blocks
# speech, so it does not have to pay the fast tier's accuracy cost.
#
# This is not hypothetical. A live run on 2026-08-30 with gemini-3.5-flash-lite failed
# to parse the scaffolding contract - "Expecting property name enclosed in double
# quotes" - and fell back to canned cards. The fast tiers are tuned for conversational
# text, not for emitting a 5-block JSON schema without drift.
DEFAULT_OPENAI_SCAFFOLDING_MODEL = "gpt-5.4"
DEFAULT_GEMINI_SCAFFOLDING_MODEL = "gemini-3.5-flash"

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False


class TeachingAgent:
    """
    Tandem Co-Teacher AI Agent mediating peer language exchanges in real-time.
    """

    def __init__(
        self,
        engine: str = "mock",
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        target_language: str = "Japanese",
        native_language: str = "English"
    ):
        """
        Initialize the TeachingAgent with configuration and client credentials.
        
        Algorithm:
        1. Resolve API credentials from arguments or environment variables.
        2. Initialize provider SDK clients (OpenAI or Google Gemini).
        3. Initialize turn history buffer and speaker duration tracking tables.
        4. Configure default target and native languages.
        """
        self.engine = engine.lower()
        self.target_language = target_language
        self.native_language = native_language
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Model tier per provider (REQ-LAT-04). Resolved once here rather than read at
        # each call site, which is where the old hardcoded "gpt-4o" / "gemini-2.5-flash"
        # literals lived and could not be tuned without a code change.
        self.openai_model = os.getenv("ECHOSPHERE_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.gemini_model = os.getenv("ECHOSPHERE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

        # Structured-output models. Separate from the voice tiers because the two calls
        # optimise for different things: the voice path for time-to-first-token, this
        # one for emitting a strict JSON schema correctly.
        self.openai_scaffolding_model = os.getenv(
            "ECHOSPHERE_OPENAI_SCAFFOLDING_MODEL", DEFAULT_OPENAI_SCAFFOLDING_MODEL
        )
        self.gemini_scaffolding_model = os.getenv(
            "ECHOSPHERE_GEMINI_SCAFFOLDING_MODEL", DEFAULT_GEMINI_SCAFFOLDING_MODEL
        )

        # Step 2: Initialize Provider Clients
        requested_engine = self.engine

        self._openai_client = None
        if self.engine == "whisper" or self.engine == "openai":
            if OPENAI_AVAILABLE and self.openai_api_key:
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            else:
                self._downgrade_to_mock(
                    requested_engine,
                    "the `openai` package is not installed" if not OPENAI_AVAILABLE
                    else "OPENAI_API_KEY is not set"
                )

        self._gemini_client = None
        if self.engine == "gemini":
            if GOOGLE_GENAI_AVAILABLE and self.gemini_api_key:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            else:
                self._downgrade_to_mock(
                    requested_engine,
                    "the `google-genai` package is not installed" if not GOOGLE_GENAI_AVAILABLE
                    else "GEMINI_API_KEY is not set"
                )

        # Step 3: Initialize State Buffers
        self.turn_history: List[Dict[str, str]] = []
        self.speaker_durations_ms: Dict[str, int] = {}

        logger.info(f"TeachingAgent initialized with engine: '{self.engine}' (Target: {self.target_language}, Native: {self.native_language})")

    def _downgrade_to_mock(self, requested_engine: str, missing: str) -> None:
        """
        Switches the agent to the mock engine and makes the reason visible (REQ-LLM-01).

        Without this the downgrade was silent, so a misconfigured API key produced
        exactly the symptom of a hardcoded stub - three canned replies - with nothing
        in the log to tell the two apart.
        """
        self.engine = "mock"
        logger.warning(
            f"TeachingAgent requested engine '{requested_engine}' but {missing}. "
            f"Falling back to engine 'mock': replies will be canned, not generated."
        )

    def _fallback_to_mock_response(
        self,
        reason: str,
        mode: str,
        speaker_id: str,
        text: str,
        lang: str
    ) -> Dict[str, Any]:
        """
        Returns a canned response after a live model call failed, and says so in the log.

        A provider error or a non-conforming (unparseable) JSON reply silently degrades
        to the same three canned strings the mock engine emits. Logging at warning level
        keeps that regression distinguishable from a genuine mock-mode run.
        """
        logger.warning(
            f"TeachingAgent falling back to a canned '{mode}' response: {reason}"
        )
        if mode == "tutor":
            return self._mock_tutor_response(speaker_id, text, lang)
        return self._mock_mediation_response(speaker_id, text, lang)

    def reset_state(self) -> None:
        """
        Clears per-session conversation state (REQ-LLM-05).

        `turn_history` and `speaker_durations_ms` otherwise live for the lifetime of the
        process, so a second Convo AI session on the same channel inherits the previous
        learner's dialogue. Harmless against a stub; wrong against a real model, which
        reads that history as context.

        Clears in place rather than rebuilding the agent so provider SDK clients (and
        their connection pools) are not re-created per session.
        """
        self.turn_history.clear()
        self.speaker_durations_ms.clear()
        logger.info("TeachingAgent conversation state reset for a new session.")

    def update_speaker_time(self, speaker_id: str, duration_ms: int):
        """Records speaking duration for multi-speaker balance monitoring."""
        self.speaker_durations_ms[speaker_id] = self.speaker_durations_ms.get(speaker_id, 0) + duration_ms

    def get_speaker_balance_percentages(self) -> Dict[str, int]:
        """
        Calculates percentage of total speaking time for each participant.
        
        Algorithm:
        1. Sum durations across all registered speakers.
        2. If total is zero, return equal split (50/50).
        3. Compute integer percentage for each speaker.
        """
        total = sum(self.speaker_durations_ms.values())
        if total == 0:
            return {spk: 50 for spk in self.speaker_durations_ms} if self.speaker_durations_ms else {}
        return {spk: int((dur / total) * 100) for spk, dur in self.speaker_durations_ms.items()}

    def format_history_context(self, max_turns: int = 6) -> str:
        """Formats the most recent dialogue turns into a plain text transcript."""
        recent = self.turn_history[-max_turns:]
        return "\n".join(f"[{item['speaker']} ({item['lang']})]: {item['text']}" for item in recent)

    def process_turn(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "ja",
        topic: Optional[str] = None,
        mode: str = "mediation",
        record_turn: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a newly transcribed student turn and generates real-time pedagogical scaffolding.

        Algorithm:
        1. Append the turn to the internal history buffer (unless already recorded).
        2. Retrieve current speaker balance metrics.
        3. Construct the prompt for the requested mode.
        4. Route to LLM backend (OpenAI, Gemini, or Mock).
        5. Parse JSON output and validate required contract fields.
        6. Return standardized result dictionary.

        `mode` selects the prompt pair (REQ-LLM-03) and defaults to "mediation" so every
        existing ambient-pipeline caller is unaffected:
        - "mediation": two peers in a breakout, where speaking balance matters.
        - "tutor": a 1:1 Convo AI conversation with exactly one learner present.
        Both modes return the identical JSON contract, so downstream parsing is shared.

        `record_turn=False` is used by the Convo AI scaffolding call (REQ-LAT-02), which
        runs *after* `generate_spoken_reply` already recorded the same utterance. Without
        it the learner would appear in the model's context saying everything twice.
        """
        # Step 1: Record turn
        if record_turn:
            self.turn_history.append({
                "speaker": speaker_id,
                "text": text,
                "lang": detected_language
            })

        # Step 2: Compute stats
        stats = self.get_speaker_balance_percentages()
        context_str = self.format_history_context()

        # Step 3: Build prompt for the requested mode
        if mode == "tutor":
            prompt = create_tutor_prompt(
                recent_context=context_str,
                target_language=self.target_language,
                native_language=self.native_language,
                learner_name=speaker_id,
                topic=topic
            )
            system_prompt = SYSTEM_PROMPT_TANDEM_TUTOR
        else:
            prompt = create_teaching_prompt(
                recent_context=context_str,
                speaker_stats=stats,
                target_language=self.target_language,
                native_language=self.native_language,
                topic=topic
            )
            system_prompt = SYSTEM_PROMPT_TANDEM_TEACHER

        # Step 4: Dispatch to LLM
        if self.engine in ("openai", "whisper") and self._openai_client:
            return self._call_openai(
                prompt, system_prompt=system_prompt, mode=mode,
                speaker_id=speaker_id, text=text, lang=detected_language
            )
        elif self.engine == "gemini" and self._gemini_client:
            return self._call_gemini(
                prompt, system_prompt=system_prompt, mode=mode,
                speaker_id=speaker_id, text=text, lang=detected_language
            )
        elif mode == "tutor":
            return self._mock_tutor_response(speaker_id, text, detected_language)
        else:
            return self._mock_mediation_response(speaker_id, text, detected_language)

    def generate_spoken_reply(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "ja",
        topic: Optional[str] = None
    ) -> Iterator[str]:
        """
        Streams the spoken 1:1 reply as plain text deltas (REQ-LAT-02 / REQ-LAT-03).

        This is the voice-critical path: everything the student waits on before hearing
        anything happens here, and nothing else does. The structured scaffolding
        (subtitles, idiom card, quiz, teacher alert) is generated by a separate
        `process_turn(mode="tutor", record_turn=False)` call that the caller runs off
        this path, so a large JSON generation no longer sits between the student
        finishing a sentence and the agent starting to speak (D-LAT-1, D-LAT-3).

        Algorithm:
        1. Record the turn in history (this path owns it - see `record_turn`).
        2. Build the plain-text voice prompt from the conversation so far.
        3. Stream deltas from the configured provider, or emit the mock reply offline.
        4. On any provider failure, fall back to speakable canned text rather than
           raising - silence ends the session via the Engine's idle_timeout.

        Returns:
            An iterator of successive text fragments. Concatenated, they form the full
            spoken reply.

        Steps 1 and 2 run eagerly, before any iteration: this is a plain method
        returning a generator rather than a generator function itself, so recording the
        turn does not depend on the caller consuming the stream.
        """
        # Step 1: This path owns history so the next turn's context is correct even
        # though the scaffolding call is asynchronous and may not have finished.
        self.turn_history.append({
            "speaker": speaker_id,
            "text": text,
            "lang": detected_language
        })

        # Step 2: Build the voice prompt from history *excluding* the utterance itself,
        # which is passed separately so the model sees clearly what to answer.
        prompt = create_tutor_voice_prompt(
            recent_context=self.format_history_context(),
            latest_utterance=text,
            target_language=self.target_language,
            native_language=self.native_language,
            learner_name=speaker_id,
            topic=topic
        )

        def _stream() -> Iterator[str]:
            started = time.time()
            emitted_any = False

            # Step 3: Dispatch to the streaming provider leg
            try:
                if self.engine in ("openai", "whisper") and self._openai_client:
                    stream = self._stream_openai(prompt, SYSTEM_PROMPT_TANDEM_TUTOR_VOICE)
                elif self.engine == "gemini" and self._gemini_client:
                    stream = self._stream_gemini(prompt, SYSTEM_PROMPT_TANDEM_TUTOR_VOICE)
                else:
                    stream = self._mock_spoken_reply_stream(speaker_id, text, detected_language)

                for delta in stream:
                    if not delta:
                        continue
                    if not emitted_any:
                        emitted_any = True
                        logger.debug(
                            "Spoken reply first token in %.3fs (engine=%s, model=%s).",
                            time.time() - started, self.engine, self._active_model()
                        )
                    yield delta
            except Exception as err:
                # Step 4: A mid-stream provider failure must still end in speech.
                logger.warning(
                    f"Spoken reply stream failed ({err}). "
                    f"Falling back to a canned reply so the agent does not go silent."
                )
                if not emitted_any:
                    yield self._mock_spoken_reply(speaker_id, text, detected_language)
                return

            if not emitted_any:
                logger.warning(
                    "Reasoning engine produced an empty spoken reply; speaking canned text."
                )
                yield self._mock_spoken_reply(speaker_id, text, detected_language)
                return

            logger.debug("Spoken reply stream completed in %.3fs.", time.time() - started)

        return _stream()

    def _active_model(self) -> str:
        """Returns the model id in use for the active engine, for log lines."""
        if self.engine in ("openai", "whisper"):
            return self.openai_model
        if self.engine == "gemini":
            return self.gemini_model
        return "mock"

    def _stream_openai(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """
        Streams plain-text deltas from OpenAI (REQ-LAT-03).

        Separate from `_call_openai` rather than a flag on it: that method is the
        mediation pipeline's JSON path (REQ-04) and must keep its blocking,
        `response_format`-constrained behaviour untouched.
        """
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content

    def _stream_gemini(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """
        Streams plain-text deltas from Google Gemini (REQ-LAT-03).

        See `_stream_openai` for why this does not reuse `_call_gemini`. No
        `response_mime_type` is set here: this call must return speech, not JSON.
        """
        full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}"
        response = self._gemini_client.models.generate_content_stream(
            model=self.gemini_model,
            contents=full_prompt
        )
        for chunk in response:
            content = getattr(chunk, "text", None)
            if content:
                yield content

    def _mock_spoken_reply(self, speaker_id: str, text: str, lang: str) -> str:
        """
        Returns the offline spoken reply for the fast path.

        Reuses the tutor mock's `spoken_response` so offline demos say the same thing on
        the fast path as they did on the old blocking path.
        """
        return self._mock_tutor_response(speaker_id, text, lang).get("spoken_response", "")

    def _mock_spoken_reply_stream(
        self,
        speaker_id: str,
        text: str,
        lang: str,
        chunk_size: int = 24
    ) -> Iterator[str]:
        """
        Emits the offline reply as several deltas so the mock path exercises the same
        incremental contract as a live provider stream.
        """
        reply = self._mock_spoken_reply(speaker_id, text, lang)
        for index in range(0, len(reply), chunk_size):
            yield reply[index:index + chunk_size]

    def generate_silence_breaker(
        self,
        topic: str = "Daily Life & Cultural Exchange",
        inactive_speaker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an engaging spoken conversation prompt when dialogue stalls.
        
        Algorithm:
        1. Build prompt using `create_silence_breaker_prompt`.
        2. Route to LLM engine.
        3. Return structured JSON payload with spoken intervention and topic card.
        """
        prompt = create_silence_breaker_prompt(
            topic=topic,
            target_language=self.target_language,
            native_language=self.native_language,
            inactive_speaker=inactive_speaker
        )

        if self.engine in ("openai", "whisper") and self._openai_client:
            return self._call_openai(prompt)
        elif self.engine == "gemini" and self._gemini_client:
            return self._call_gemini(prompt)
        else:
            return self._mock_silence_breaker(topic, inactive_speaker)

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT_TANDEM_TEACHER,
        mode: str = "mediation",
        speaker_id: str = "Student",
        text: str = "",
        lang: str = "ja"
    ) -> Dict[str, Any]:
        """
        Invokes OpenAI API and parses JSON response.

        `system_prompt` is a parameter rather than a hardcoded constant so the Convo AI
        tutor mode actually reaches the model. An API error or a non-conforming (
        unparseable) reply routes through `_fallback_to_mock_response`, which logs the
        downgrade instead of silently returning canned text.
        """
        started = time.time()
        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_scaffolding_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            # REQ-LAT-01: attributes a slow turn to model think-time rather than to
            # bridge overhead or the network, which the bridge-level timing cannot.
            logger.debug(
                "OpenAI structured call (%s) took %.3fs.",
                self.openai_scaffolding_model, time.time() - started
            )
            return result
        except Exception as err:
            logger.error(f"OpenAI agent invocation failed after {time.time() - started:.3f}s: {err}")
            return self._fallback_to_mock_response(
                f"OpenAI call or JSON parse failed ({err})", mode, speaker_id, text, lang
            )

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT_TANDEM_TEACHER,
        mode: str = "mediation",
        speaker_id: str = "Student",
        text: str = "",
        lang: str = "ja"
    ) -> Dict[str, Any]:
        """
        Invokes Google Gemini API with JSON mode.

        See `_call_openai` for why the system prompt is a parameter and why failures
        route through `_fallback_to_mock_response`.
        """
        started = time.time()
        try:
            full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}"
            response = self._gemini_client.models.generate_content(
                model=self.gemini_scaffolding_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            # REQ-LAT-01: see the matching line in _call_openai.
            logger.debug(
                "Gemini structured call (%s) took %.3fs.",
                self.gemini_scaffolding_model, time.time() - started
            )
            return result
        except Exception as err:
            logger.error(f"Gemini agent invocation failed after {time.time() - started:.3f}s: {err}")
            return self._fallback_to_mock_response(
                f"Gemini call or JSON parse failed ({err})", mode, speaker_id, text, lang
            )

    def _mock_mediation_response(self, speaker_id: str, text: str, lang: str) -> Dict[str, Any]:
        """Generates rich, realistic structured mock response across Hindi, Japanese, and English."""
        # Detect Japanese idioms or phrases
        if "一期一会" in text or "ichigo" in text.lower() or lang == "ja":
            return {
                "spoken_response": "素晴らしい表現ですね！Aaravさん、この意味を知っていますか？ (Wonderful expression! Aarav, do you know what this means?)",
                "spoken_language": "ja",
                "subtitles": {
                    "speaker": speaker_id,
                    "original_text": text,
                    "transliteration": "Ichigo ichie desu ne",
                    "translation_en": "Treasure every encounter, for it may never recur.",
                    "translation_ja": "一期一会ですね",
                    "translation_hi": "हर मुलाकात अनमोल और अनोखी होती है।"
                },
                "idiom_card": {
                    "detected": True,
                    "phrase": "一期一会 (Ichigo Ichie)",
                    "romaji": "Ichigo ichie",
                    "meaning": "Once-in-a-lifetime encounter / Treasure every meeting.",
                    "cultural_note": "A classic Japanese proverb derived from tea ceremony philosophy, emphasizing mindfulness."
                },
                "quiz": {
                    "active": True,
                    "question": "What is the cultural origin of '一期一会' (Ichigo Ichie)?",
                    "options": ["Tea Ceremony Philosophy", "Samurai Martial Code", "Modern Manga Slang"],
                    "correct_index": 0,
                    "explanation": "It originated from the traditional Japanese tea ceremony philosophy taught by Sen no Rikyu."
                },
                "teacher_alert": {
                    "alert_required": False,
                    "message": "Learners engaging well with cultural idioms."
                }
            }
        elif "नमस्ते" in text or "namaste" in text.lower() or lang == "hi":
            return {
                "spoken_response": "बहुत अच्छा! Kenji-san, do you know how to greet someone politely in Hindi? (Bahut achha!)",
                "spoken_language": "hi",
                "subtitles": {
                    "speaker": speaker_id,
                    "original_text": text,
                    "transliteration": "Namaste, aap kaise hain?",
                    "translation_en": "Hello, how are you?",
                    "translation_ja": "こんにちは、お元気ですか？",
                    "translation_hi": "नमस्ते, आप कैसे हैं?"
                },
                "idiom_card": {
                    "detected": True,
                    "phrase": "नमस्ते (Namaste)",
                    "romaji": "Namaste",
                    "meaning": "I bow to the divine in you / Formal greeting.",
                    "cultural_note": "Accompanied by joining palms (Añjali Mudrā), used for respectful greetings."
                },
                "quiz": {
                    "active": True,
                    "question": "Which pronoun represents the highest level of formal respect in Hindi?",
                    "options": ["आप (Aap)", "तुम (Tum)", "तू (Tu)"],
                    "correct_index": 0,
                    "explanation": "'Aap' is the most respectful formal second-person pronoun in Hindi."
                },
                "teacher_alert": {
                    "alert_required": False,
                    "message": "Hindi formal register used correctly."
                }
            }
        else:
            return {
                "spoken_response": "Great conversation! What do both of you think about this topic?",
                "spoken_language": "en",
                "subtitles": {
                    "speaker": speaker_id,
                    "original_text": text,
                    "transliteration": text,
                    "translation_en": text,
                    "translation_ja": "素晴らしい会話ですね！",
                    "translation_hi": "शानदार बातचीत!"
                },
                "idiom_card": {
                    "detected": False,
                    "phrase": "",
                    "romaji": "",
                    "meaning": "",
                    "cultural_note": ""
                },
                "quiz": {
                    "active": False,
                    "question": "",
                    "options": [],
                    "correct_index": 0,
                    "explanation": ""
                },
                "teacher_alert": {
                    "alert_required": False,
                    "message": ""
                }
            }

    def _mock_tutor_response(self, speaker_id: str, text: str, lang: str) -> Dict[str, Any]:
        """
        Generates a structured 1:1 offline response for the Convo AI path (REQ-LLM-03).

        Mirrors `_mock_mediation_response`'s JSON contract exactly, but addresses one
        learner: the mediation mock's "What do both of you think about this topic?" is a
        peer-breakout line that makes no sense spoken into a private 1:1 call.

        Still canned - this is the offline demo path. Genuinely varied, context-aware
        replies require a real engine via ECHOSPHERE_ENGINE (REQ-LLM-01).
        """
        turn_index = len(self.turn_history)

        if "一期一会" in text or "ichigo" in text.lower() or lang == "ja":
            return {
                "spoken_response": "いい表現ですね！どこでその言葉を覚えましたか？ (Nice expression! Where did you learn it?)",
                "spoken_language": "ja",
                "subtitles": {
                    "speaker": speaker_id,
                    "original_text": text,
                    "transliteration": "Ichigo ichie desu ne",
                    "translation_en": "Treasure every encounter, for it may never recur.",
                    "translation_ja": "一期一会ですね",
                    "translation_hi": "हर मुलाकात अनमोल और अनोखी होती है।"
                },
                "idiom_card": {
                    "detected": True,
                    "phrase": "一期一会 (Ichigo Ichie)",
                    "romaji": "Ichigo ichie",
                    "meaning": "Once-in-a-lifetime encounter / Treasure every meeting.",
                    "cultural_note": "A classic Japanese proverb derived from tea ceremony philosophy, emphasizing mindfulness."
                },
                "quiz": {
                    "active": True,
                    "question": "What is the cultural origin of '一期一会' (Ichigo Ichie)?",
                    "options": ["Tea Ceremony Philosophy", "Samurai Martial Code", "Modern Manga Slang"],
                    "correct_index": 0,
                    "explanation": "It originated from the traditional Japanese tea ceremony philosophy taught by Sen no Rikyu."
                },
                "teacher_alert": {
                    "alert_required": False,
                    "message": f"1:1 tutor session with {speaker_id}: turn {turn_index} in Japanese."
                }
            }

        if "नमस्ते" in text or "namaste" in text.lower() or lang == "hi":
            return {
                "spoken_response": "बहुत अच्छा! आप आज क्या कर रहे हैं? (Very good! What are you doing today?)",
                "spoken_language": "hi",
                "subtitles": {
                    "speaker": speaker_id,
                    "original_text": text,
                    "transliteration": "Namaste, aap kaise hain?",
                    "translation_en": "Hello, how are you?",
                    "translation_ja": "こんにちは、お元気ですか？",
                    "translation_hi": "नमस्ते, आप कैसे हैं?"
                },
                "idiom_card": {
                    "detected": True,
                    "phrase": "नमस्ते (Namaste)",
                    "romaji": "Namaste",
                    "meaning": "I bow to the divine in you / Formal greeting.",
                    "cultural_note": "Accompanied by joining palms (Añjali Mudrā), used for respectful greetings."
                },
                "quiz": {
                    "active": True,
                    "question": "Which pronoun represents the highest level of formal respect in Hindi?",
                    "options": ["आप (Aap)", "तुम (Tum)", "तू (Tu)"],
                    "correct_index": 0,
                    "explanation": "'Aap' is the most respectful formal second-person pronoun in Hindi."
                },
                "teacher_alert": {
                    "alert_required": False,
                    "message": f"1:1 tutor session with {speaker_id}: turn {turn_index} in Hindi."
                }
            }

        return {
            "spoken_response": f"Thanks for sharing that, {speaker_id}. Can you tell me a little more about it?",
            "spoken_language": "en",
            "subtitles": {
                "speaker": speaker_id,
                "original_text": text,
                "transliteration": text,
                "translation_en": text,
                "translation_ja": "もう少し詳しく教えてください。",
                "translation_hi": "कृपया इसके बारे में और बताइए।"
            },
            "idiom_card": {
                "detected": False,
                "phrase": "",
                "romaji": "",
                "meaning": "",
                "cultural_note": ""
            },
            "quiz": {
                "active": False,
                "question": "",
                "options": [],
                "correct_index": 0,
                "explanation": ""
            },
            "teacher_alert": {
                "alert_required": False,
                "message": f"1:1 tutor session with {speaker_id}: turn {turn_index}."
            }
        }

    def _mock_silence_breaker(self, topic: str, inactive_speaker: Optional[str]) -> Dict[str, Any]:
        """Generates structured mock silence breaker payload."""
        speaker_target = inactive_speaker or "everyone"
        return {
            "spoken_response": f"Let's explore {topic}! {speaker_target}, what is your favorite tradition related to this?",
            "spoken_language": "en",
            "subtitles": {
                "speaker": "EchoSphere AI",
                "original_text": f"Let's explore {topic}! {speaker_target}, what is your favorite tradition?",
                "transliteration": "",
                "translation_en": f"Let's explore {topic}!",
                "translation_ja": f"{topic}について話してみましょう！",
                "translation_hi": f"आइए {topic} के बारे में चर्चा करें!"
            },
            "idiom_card": {
                "detected": False,
                "phrase": "",
                "romaji": "",
                "meaning": "",
                "cultural_note": ""
            },
            "quiz": {
                "active": False,
                "question": "",
                "options": [],
                "correct_index": 0,
                "explanation": ""
            },
            "teacher_alert": {
                "alert_required": True,
                "message": f"Silence breaker triggered for {speaker_target} on topic: {topic}"
            }
        }
