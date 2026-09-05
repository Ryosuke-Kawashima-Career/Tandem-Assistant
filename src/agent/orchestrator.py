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
import re
import time
import logging
from typing import Optional, Dict, Any, List, Iterator
from src.agent.prompts import (
    SYSTEM_PROMPT_TANDEM_TEACHER,
    SYSTEM_PROMPT_TANDEM_TUTOR,
    SYSTEM_PROMPT_TANDEM_TUTOR_VOICE,
    SYSTEM_PROMPT_INTERNATIONAL_WORK,
    SYSTEM_PROMPT_INTERNATIONAL_WORK_VOICE,
    SYSTEM_PROMPT_DIRECT_QUERY,
    SYSTEM_PROMPT_DIRECT_QUERY_WORK,
    create_query_prompt,
    create_teaching_prompt,
    create_tutor_prompt,
    create_tutor_voice_prompt,
    create_work_prompt,
    create_work_voice_prompt,
    create_silence_breaker_prompt
)
from src.agent.tools.dispatch import ToolDispatcher
from src.sessions.models import SessionMode

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

# Below this, a scaffolding point counts as something the model was not sure about, and
# is worth looking up rather than asserting (REQ-18, agent-initiated research).
#
# Deliberately the same bar REQ-14 holds a note at `needs_confirmation`: the two
# questions - "should this be shown as unconfirmed?" and "should this be checked?" - are
# the same judgement about the same sentence, and answering them differently would put
# a note on screen as uncertain while researching nothing about it.
RESEARCH_CONFIDENCE_THRESHOLD = 0.6

# Note types a lookup can actually answer. An action item or a decision is a commitment
# somebody made in the room; searching the web for it returns somebody else's.
RESEARCHABLE_NOTE_TYPES = ("vocabulary", "culture", "term", "glossary")

# Separators between a term and its meaning, matching the note generator's own wording.
# A query is the term, not the whole explanatory sentence around it.
RESEARCH_SEPARATORS = (" — ", " – ", " - ", ": ")

# Long enough for a phrase and its context, short enough that a runaway model sentence
# does not become the search query.
MAX_RESEARCH_QUERY_CHARS = 120

# Human-readable names for the language codes the tutor path carries (REQ-LLM-02). The
# Convo AI session language arrives as an ISO code, but the prompts name the language in
# words, and the model has to be told which language the lesson is in: a live Phase 8.2
# smoke on a "hi" session answered "Hindi isn't my specialty" because the prompt still
# said "Target Language: Japanese", the agent's constructor default.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ja": "Japanese",
    "hi": "Hindi",
}

# Phrases that mean "look through my camera" rather than "answer my question"
# (REQ-CAM-02). Kept as a documented regex list rather than a model call because this is
# evaluated on *every* voice turn, and the one thing the voice path may not spend is a
# round trip to decide whether to spend a round trip.
#
# The deictic family - "what is this", "what's that" - only counts when it ends the
# clause. "What is this?" is a person holding something up; "what is this word" is a
# question about what they just heard, and describing their desk to them would be an
# answer to a question nobody asked. The rest of the list names the camera explicitly
# enough that no such guard is needed.
#
# This check is one of two gates, never the only one: `generate_spoken_reply` also
# requires a fresh buffered frame, so a false positive here with the camera off costs
# nothing at all (Risk 2).
CAMERA_QUESTION_PATTERNS = (
    r"\bwhat(?:'s| is| are)\s+th(?:is|at|ese|ose)\s*(?=[?!.,]|$)",
    r"\bwhat am i (?:looking at|holding|showing|pointing at)",
    r"\bwhat(?:'s| is)\s+(?:in front of|before) me",
    r"\b(?:can|could) you (?:see|read) th(?:is|at)",
    r"\bdo you see th(?:is|at)",
    r"\bwhat (?:does|do) th(?:is|ese) (?:say|mean)",
    r"\b(?:look|take a look) at th(?:is|at)",
    r"\bwhat is written (?:here|on th(?:is|at))",
    r"\bwhat kind of .{0,20}\bis th(?:is|at)\s*(?=[?!.,]|$)",
)

_CAMERA_QUESTION_RE = re.compile("|".join(CAMERA_QUESTION_PATTERNS), re.IGNORECASE)


# How a live camera lookup reaches the voice prompt. Phrased as a statement of fact
# rather than as an instruction: the model is being told what is in front of the student,
# and answers the question it was already going to answer, now with that in hand.
CAMERA_SEEN_TEMPLATE = (
    "What you can currently see through {learner}'s camera right now: {title} - "
    "{description}\n"
    "Answer using what is actually visible. Do not describe anything beyond it."
)

# The honest alternative to guessing. A camera-shaped question with no camera view is
# common - the student says "what is this?" about a word they just heard - and the worst
# possible reply is a confident description of a desk nobody is pointing at.
CAMERA_BLIND_NOTE = (
    "The student may be asking about something they are looking at, but you have no "
    "camera view right now - Camera Assist is off, or it has sent nothing recently. If "
    "they are asking about something physical, say briefly that you cannot see it and "
    "that they can turn Camera Assist on. Never guess at what they are holding."
)


def looks_like_camera_question(text: str) -> bool:
    """
    Whether an utterance is asking about what the camera can see (REQ-CAM-02).

    Costs nothing: a normalized string and one compiled alternation, run before the voice
    prompt is built. A model call here would put a vendor round trip in front of every
    spoken reply in order to avoid a vendor round trip, which is the wrong trade on the
    one path a learner waits on in silence (REQ-LAT-02).

    Returns:
        True when the phrasing points at something in view, False otherwise.
    """
    normalized = " ".join((text or "").replace("’", "'").lower().split())
    if not normalized:
        return False
    return bool(_CAMERA_QUESTION_RE.search(normalized))


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
        native_language: str = "English",
        primary_language: str = "English",
        peer_target_languages: Optional[Dict[str, str]] = None,
        tools: Optional[ToolDispatcher] = None
    ):
        """
        Initialize the TeachingAgent with configuration and client credentials.

        Algorithm:
        1. Resolve API credentials from arguments or environment variables.
        2. Initialize provider SDK clients (OpenAI or Google Gemini).
        3. Initialize turn history buffer and speaker duration tracking tables.
        4. Configure default target and native languages.
        5. Configure the REQ-17 language roles: one primary language for scaffolding,
           plus each peer's own target language as a complementary language.

        `peer_target_languages` maps a speaker id to the language that speaker is
        learning. A tandem pair holds two different ones - a Hindi speaker learning
        Japanese alongside a Japanese speaker learning Hindi - which the single
        `target_language` above cannot express, and picking either one of them makes the
        agent's own explanations unintelligible to the other peer (REQ-17, TASK-3.4).
        """
        self.engine = engine.lower()
        self.target_language = target_language
        self.native_language = native_language
        self.primary_language = primary_language
        self.peer_target_languages: Dict[str, str] = dict(peer_target_languages or {})

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

        # Step 4: External tools (REQ-18-20). Injected by the server so tool events reach
        # the live RTC data stream; the default dispatcher holds no tools at all, which is
        # exactly how an agent constructed for a unit test or an offline run behaves -
        # every tool call reports `unavailable` instead of reaching a vendor.
        self.tools = tools if tools is not None else ToolDispatcher()

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

    def set_peer_target_language(self, speaker_id: str, language: str) -> None:
        """
        Records which language one peer is learning (REQ-17, TASK-3.4).

        Available per turn because a pairing is often only known once both peers have
        actually joined and spoken; requiring it at construction time would mean tearing
        down and rebuilding the agent - and its conversation history - mid-session.
        """
        if speaker_id and language:
            self.peer_target_languages[speaker_id] = language

    def complementary_languages(self) -> List[str]:
        """
        Returns each peer's own target language, in first-seen order, without duplicates.

        Falls back to the single configured `target_language` so an agent constructed the
        old way still has exactly one complementary language and behaves as it did.
        """
        languages: List[str] = []
        for language in self.peer_target_languages.values():
            if language and language not in languages:
                languages.append(language)
        return languages or [self.target_language]

    def build_mediation_prompt(self, topic: Optional[str] = None) -> str:
        """
        Builds the ambient peer-mediation prompt under the REQ-17 language roles.

        Extracted from `process_turn` so the roles are applied in exactly one place: the
        prompt is also built by the Convo AI scaffolding path, and two call sites
        assembling it independently is how one of them ends up without the roles.
        """
        return create_teaching_prompt(
            recent_context=self.format_history_context(),
            speaker_stats=self.get_speaker_balance_percentages(),
            target_language=self.target_language,
            native_language=self.native_language,
            topic=topic,
            primary_language=self.primary_language,
            complementary_languages=self.complementary_languages()
        )

    def resolve_tutor_target_language(self, detected_language: str) -> str:
        """
        Resolves the language a 1:1 tutor turn should teach.

        In the Convo AI path `detected_language` is the language the learner selected
        when the session started (REQ-LLM-02), so it - not the constructor default -
        decides the lesson. An unmapped code keeps the configured target rather than
        inventing a language name for the model to act on.

        Deliberately not applied to mediation: there `detected_language` reports which
        language one utterance happened to be in, and a single code-switch must not
        redirect the lesson.
        """
        return LANGUAGE_NAMES.get(
            (detected_language or "").lower(), self.target_language
        )

    def build_tutor_prompt(
        self,
        speaker_id: str,
        detected_language: str = "ja",
        topic: Optional[str] = None
    ) -> str:
        """Builds the structured 1:1 scaffolding prompt for the session's language."""
        target_language = self.peer_target_languages.get(
            speaker_id, self.resolve_tutor_target_language(detected_language)
        )
        return create_tutor_prompt(
            recent_context=self.format_history_context(),
            target_language=target_language,
            native_language=self.native_language,
            learner_name=speaker_id,
            topic=topic,
            primary_language=self.primary_language,
            complementary_languages=[target_language]
        )

    def build_tutor_voice_prompt(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "ja",
        topic: Optional[str] = None,
        extra_context: str = ""
    ) -> str:
        """
        Builds the voice-critical 1:1 prompt for the session's language.

        The recent context deliberately excludes `text` itself, which is passed
        separately so the model sees clearly which utterance it is answering.

        `extra_context` carries what the agent looked at during this turn - what the
        camera is showing, or that it cannot currently see (REQ-CAM-03).
        """
        return create_tutor_voice_prompt(
            recent_context=self.format_history_context(),
            latest_utterance=text,
            target_language=self.resolve_tutor_target_language(detected_language),
            native_language=self.native_language,
            learner_name=speaker_id,
            topic=topic,
            extra_context=extra_context
        )

    @staticmethod
    def resolve_session_mode(session_mode: Any) -> SessionMode:
        """
        Normalizes a session mode, defaulting to language learning (REQ-12).

        The default exists for the ambient pipeline and the older REST callers, which
        predate session modes; the creation APIs themselves reject a missing mode rather
        than reaching this default.
        """
        if isinstance(session_mode, SessionMode):
            return session_mode
        try:
            return SessionMode.parse(session_mode)
        except Exception:
            return SessionMode.LANGUAGE_LEARNING

    def build_work_prompt(
        self,
        speaker_id: str,
        detected_language: str = "en",
        topic: Optional[str] = None
    ) -> str:
        """Builds the structured `international_work` prompt for one turn (REQ-12)."""
        return create_work_prompt(
            recent_context=self.format_history_context(),
            working_language=self.native_language,
            speaker_languages=self.observed_languages(),
            speaker_name=speaker_id,
            topic=topic
        )

    def build_work_voice_prompt(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "en",
        topic: Optional[str] = None,
        extra_context: str = ""
    ) -> str:
        """Builds the voice-critical `international_work` prompt (REQ-LAT-02)."""
        return create_work_voice_prompt(
            recent_context=self.format_history_context(),
            latest_utterance=text,
            working_language=self.native_language,
            speaker_name=speaker_id,
            topic=topic,
            extra_context=extra_context
        )

    def observed_languages(self) -> List[str]:
        """
        Returns the distinct languages heard so far, most recent first.

        Work sessions have no configured target language - the room's language mix is
        whatever the participants actually used - so it is read off the history rather
        than from constructor configuration.
        """
        seen: List[str] = []
        for item in reversed(self.turn_history):
            lang = item.get("lang")
            if lang and lang not in seen:
                seen.append(lang)
        return seen

    def resolve_research_query(
        self,
        turn_result: Dict[str, Any],
        session: Any = None
    ) -> Optional[str]:
        """
        Decides whether a turn is worth a reference lookup, and what to look up (REQ-18).

        Algorithm:
        1. An explicit `research_query` from the model wins - it asked outright.
        2. Otherwise take the first term or cultural point the model itself flagged as
           uncertain, and query the term rather than the sentence around it.
        3. Return None when the model was confident, which is the common case.

        Step 3 is the whole point of the gate: researching every noun would put a live
        conversation on a search engine's budget and bury the one card that mattered
        under four that did not. The trigger is the model's own uncertainty (REQ-18:
        "lacks a confident answer"), not the presence of a foreign word.
        """
        if not isinstance(turn_result, dict):
            return None

        explicit = str(turn_result.get("research_query") or "").strip()
        if explicit:
            return explicit[:MAX_RESEARCH_QUERY_CHARS]

        for note in turn_result.get("notes") or []:
            if not isinstance(note, dict):
                continue
            if str(note.get("type", "")).strip().lower() not in RESEARCHABLE_NOTE_TYPES:
                continue

            try:
                confidence = float(note.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            if confidence >= RESEARCH_CONFIDENCE_THRESHOLD:
                continue

            text = str(note.get("text", "")).strip()
            if not text:
                continue
            for separator in RESEARCH_SEPARATORS:
                if separator in text:
                    text = text.split(separator, 1)[0].strip()
                    break
            return text[:MAX_RESEARCH_QUERY_CHARS]

        return None

    def build_query_prompt(
        self,
        question: str,
        speaker_id: str = "the participant",
        detected_language: str = "en",
        session_mode: Any = None
    ) -> str:
        """
        Builds the prompt for one direct participant question (REQ-21).

        The asker's own target language leads the complementary list, so an example
        sentence comes back in the language *they* are practising rather than in their
        partner's - which is the whole difference between an answer they can use and one
        aimed at the other side of the pair.
        """
        mode = self.resolve_session_mode(session_mode)
        complementary = self.complementary_languages()
        asker_language = self.peer_target_languages.get(speaker_id)
        if asker_language:
            complementary = [asker_language] + [
                language for language in complementary if language != asker_language
            ]

        return create_query_prompt(
            question=question,
            recent_context=self.format_history_context(),
            speaker_name=speaker_id or "the participant",
            primary_language=self.primary_language,
            complementary_languages=complementary,
            materials=(mode is SessionMode.INTERNATIONAL_WORK)
        )

    def answer_query(
        self,
        question: str,
        speaker_id: str = "Learner",
        detected_language: str = "en",
        session_mode: Any = None
    ) -> Dict[str, Any]:
        """
        Answers one question a participant asked the assistant directly (REQ-21).

        Algorithm:
        1. Refuse an empty question rather than asking a model to guess at one.
        2. Build the mode-appropriate prompt and system prompt.
        3. Answer as plain text, reusing the streaming provider paths and joining them.
        4. Return the answer alone - no notes, no quiz, no recorded turn.

        Step 4 is the requirement, not an optimization. `turn_history` is the model's
        context for mediating the *peer* conversation: a learner's private lookup landing
        in it makes the co-teacher reply to something the other peer never heard, and
        REQ-15's approval-free upsert would then export that aside as part of the shared
        record of the session.

        The streaming helpers are reused rather than `_call_openai` / `_call_gemini`
        because those are the JSON-contract paths (REQ-04); an answer to "what does this
        word mean" is prose, and constraining it to the artifact schema would produce a
        note-shaped object where a sentence was wanted.
        """
        asked = (question or "").strip()
        if not asked:
            raise ValueError("A direct query needs a question to answer.")

        mode = self.resolve_session_mode(session_mode)
        prompt = self.build_query_prompt(
            asked, speaker_id=speaker_id,
            detected_language=detected_language, session_mode=mode
        )
        system_prompt = (
            SYSTEM_PROMPT_DIRECT_QUERY_WORK
            if mode is SessionMode.INTERNATIONAL_WORK
            else SYSTEM_PROMPT_DIRECT_QUERY
        )

        answer = ""
        try:
            if self.engine in ("openai", "whisper") and self._openai_client:
                answer = "".join(self._stream_openai(prompt, system_prompt))
            elif self.engine == "gemini" and self._gemini_client:
                answer = "".join(self._stream_gemini(prompt, system_prompt))
        except Exception as exc:  # noqa: BLE001 - a failed lookup must not raise into a session
            logger.warning("Direct query answer failed, falling back to offline: %s", exc)
            answer = ""

        if not answer.strip():
            answer = self._mock_query_answer(asked, mode)

        return {
            "query": asked,
            "answer": answer.strip(),
            "language": self.primary_language,
            "speaker_id": speaker_id,
            "mode": mode.value,
        }

    def _mock_query_answer(self, question: str, mode: SessionMode) -> str:
        """
        Returns the offline answer for a direct query.

        Says outright that it is offline: a canned sentence that reads like a real answer
        is the one failure mode that matters here, because a participant writes down what
        the assistant tells them about a word.
        """
        if mode is SessionMode.INTERNATIONAL_WORK:
            return (
                f"(offline mode) I cannot look up \"{question}\" without a configured "
                "model. Set ECHOSPHERE_ENGINE and a provider key to get a real answer."
            )
        return (
            f"(offline mode) You asked about \"{question}\". With a configured model I "
            "would explain it in English and give an example sentence in your target "
            "language. Set ECHOSPHERE_ENGINE and a provider key to enable that."
        )

    def process_turn(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "ja",
        topic: Optional[str] = None,
        mode: str = "mediation",
        record_turn: bool = True,
        session_mode: Any = None,
        camera_context: str = ""
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

        `camera_context` carries what the spoken turn actually saw (REQ-CAM-03), so a note
        written about "this" records the object that was described rather than the
        pronoun. It is supplied by the caller rather than looked up here: this call runs
        after the spoken reply, and the two must reason about the same observation.
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

        # Step 3: Build prompt for the requested mode.
        # The session mode (REQ-12) outranks the prompt mode: an `international_work`
        # session never runs a tutor or mediation prompt, because both grade the
        # speaker's language and a work call has nobody practising.
        if self.resolve_session_mode(session_mode) is SessionMode.INTERNATIONAL_WORK:
            prompt = self.build_work_prompt(
                speaker_id=speaker_id,
                detected_language=detected_language,
                topic=topic
            )
            system_prompt = SYSTEM_PROMPT_INTERNATIONAL_WORK
        elif mode == "tutor":
            prompt = self.build_tutor_prompt(
                speaker_id=speaker_id,
                detected_language=detected_language,
                topic=topic
            )
            system_prompt = SYSTEM_PROMPT_TANDEM_TUTOR
        else:
            prompt = self.build_mediation_prompt(topic=topic)
            system_prompt = SYSTEM_PROMPT_TANDEM_TEACHER

        # Prefixed rather than threaded through all three prompt builders: what the camera
        # saw is one extra fact about this turn, identical in every mode. It goes in front
        # so the JSON contract each builder ends with stays the last thing the model reads
        # - the scaffolding call's output is parsed, and that ordering is what keeps it
        # parseable.
        if camera_context and camera_context.strip():
            prompt = f"{camera_context.strip()}\n\n{prompt}"

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

    def observe_live_camera(
        self,
        session: Any,
        channel: str,
        speaker_id: str,
        text: str,
        announce: bool = True
    ) -> str:
        """
        Looks through the student's camera when the utterance points at it (REQ-CAM-03).

        Two gates, both cheap before anything is spent:
        1. The phrasing has to point at something in view (`looks_like_camera_question`),
           which costs one regex and no network.
        2. A fresh frame has to actually be buffered, which the dispatcher checks in
           memory. With the camera off there is no vendor call at all - which is what
           keeps a false-positive phrase from costing anything (Risk 1, Risk 2).

        Every failure - no camera, no tool, a slow vendor, a raised exception - lands on
        the same answer: the "I cannot see" note. Never an exception, and never silence:
        this runs inside the turn the learner is waiting on, and the reply has to happen.

        `announce=False` is for the scaffolding pass, which reads the same observation to
        write the turn's notes and must not publish a second copy of the card the spoken
        turn already drew.

        Returns:
            The context block for the voice prompt, or "" when the camera is irrelevant
            to what was said.
        """
        if not channel or not looks_like_camera_question(text):
            return ""

        result = None
        if self.tools is not None:
            try:
                result = self.tools.describe_live_frame(
                    session, channel, question=text, requested_by="voice",
                    emit_card=announce
                )
            except Exception as exc:  # noqa: BLE001 - the turn continues regardless
                logger.warning(
                    f"A live camera lookup raised ({exc}). Replying without it."
                )
                result = None

        if result is None or not getattr(result, "ok", False):
            return CAMERA_BLIND_NOTE

        results = (result.payload or {}).get("results") or [{}]
        return CAMERA_SEEN_TEMPLATE.format(
            learner=speaker_id or "the student",
            title=results[0].get("title", ""),
            description=results[0].get("snippet", "")
        )

    def generate_spoken_reply(
        self,
        speaker_id: str,
        text: str,
        detected_language: str = "ja",
        topic: Optional[str] = None,
        session_mode: Any = None,
        session: Any = None,
        channel: str = ""
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
        2. Look through the student's camera, but only when the utterance points at
           something in view and a fresh frame is buffered (REQ-CAM-03).
        3. Build the plain-text voice prompt from the conversation so far.
        4. Stream deltas from the configured provider, or emit the mock reply offline.
        5. On any provider failure, fall back to speakable canned text rather than
           raising - silence ends the session via the Engine's idle_timeout.

        `session` and `channel` are what make step 2 possible and are both optional: a
        caller that passes neither gets exactly the pre-REQ-CAM-03 behaviour, which is
        what the ambient pipeline and the older tests rely on.

        Returns:
            An iterator of successive text fragments. Concatenated, they form the full
            spoken reply.

        Steps 1 to 3 run eagerly, before any iteration: this is a plain method returning
        a generator rather than a generator function itself, so recording the turn does
        not depend on the caller consuming the stream.
        """
        # Step 1: This path owns history so the next turn's context is correct even
        # though the scaffolding call is asynchronous and may not have finished.
        self.turn_history.append({
            "speaker": speaker_id,
            "text": text,
            "lang": detected_language
        })

        # Step 2: Look through the camera, but only when the utterance points at it and
        # only for as long as the bound allows (REQ-CAM-03). Runs before the prompt is
        # built because its result is part of the prompt; it is free when the phrasing is
        # not camera-shaped, which is almost every turn.
        camera_context = self.observe_live_camera(session, channel, speaker_id, text)

        # Step 3: Build the voice prompt from history *excluding* the utterance itself,
        # which is passed separately so the model sees clearly what to answer.
        resolved_mode = self.resolve_session_mode(session_mode)
        if resolved_mode is SessionMode.INTERNATIONAL_WORK:
            prompt = self.build_work_voice_prompt(
                speaker_id=speaker_id,
                text=text,
                detected_language=detected_language,
                topic=topic,
                extra_context=camera_context
            )
            voice_system_prompt = SYSTEM_PROMPT_INTERNATIONAL_WORK_VOICE
        else:
            prompt = self.build_tutor_voice_prompt(
                speaker_id=speaker_id,
                text=text,
                detected_language=detected_language,
                topic=topic,
                extra_context=camera_context
            )
            voice_system_prompt = SYSTEM_PROMPT_TANDEM_TUTOR_VOICE

        def _stream() -> Iterator[str]:
            started = time.time()
            emitted_any = False

            # Step 4: Dispatch to the streaming provider leg
            try:
                if self.engine in ("openai", "whisper") and self._openai_client:
                    stream = self._stream_openai(prompt, voice_system_prompt)
                elif self.engine == "gemini" and self._gemini_client:
                    stream = self._stream_gemini(prompt, voice_system_prompt)
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
                # Step 5: A mid-stream provider failure must still end in speech.
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
