"""
Summary:
    prompts.py defines system instructions, prompt templates, and structured JSON
    generation schemas for the EchoSphere Tandem Co-Teacher AI agent.
    It specializes in tri-lingual ambient peer mediation across Hindi (hi), Japanese (ja),
    and English (en), providing live linguistic scaffolding, Romaji/Devanagari transliteration,
    cultural idiom cards, conversation nudges, and comprehension quizzes.

Key Functions & Constants:
    - SYSTEM_PROMPT_TANDEM_TEACHER: Master system prompt defining pedagogical personality and JSON output contracts.
    - create_teaching_prompt: Generates contextual prompts incorporating dialogue history, speaker balance, and target languages.
    - create_silence_breaker_prompt: Generates engaging conversational prompts when peer silence is detected.
    - SYSTEM_PROMPT_TANDEM_TUTOR: 1:1 system prompt for direct learner-to-AI Convo AI sessions.
    - create_tutor_prompt: Generates contextual prompts for a direct 1:1 spoken conversation.
    - SYSTEM_PROMPT_TANDEM_TUTOR_VOICE: 1:1 system prompt for the low-latency voice reply.
    - create_tutor_voice_prompt: Generates the voice-critical fast-path prompt.

Two prompt modes exist and must not be conflated (REQ-LLM-03):
    - "mediation" (SYSTEM_PROMPT_TANDEM_TEACHER / create_teaching_prompt) - the ambient
      pipeline observing a peer breakout between two learners, where speaking balance
      is a real pedagogical concern.
    - "tutor" (SYSTEM_PROMPT_TANDEM_TUTOR / create_tutor_prompt) - the Convo AI path,
      where exactly one learner is present. Applying the mediation prompt here makes a
      real model address a second learner who does not exist.
Both emit the identical JSON schema, so downstream parsing is shared.

The tutor mode is further split by latency (REQ-LAT-02):
    - The *_VOICE pair generates only the spoken reply, as plain text, and is the call
      the student actually waits on. Plain text is what makes token-level streaming
      safe - a half-generated JSON object cannot be spoken aloud.
    - create_tutor_prompt (the JSON contract) still generates subtitles, idiom cards, and
      quizzes, but now runs off the voice-critical path so it no longer gates speech.
"""

from typing import Dict, Any, Optional
import json


SYSTEM_PROMPT_TANDEM_TEACHER = """
You are "EchoSphere Tandem Co-Teacher", an empathetic, intelligent, and culturally aware AI co-teacher in a live voice classroom.
You mediate real-time peer tandem conversations between learners exchanging Japanese, Hindi, and English.

Your pedagogical objectives:
1. SCAFOLDING & CODE-SWITCHING: When a student speaks, provide live transcriptions, accurate cross-cultural translations, and Latin transliterations (Romaji for Japanese, Latin transliteration for Hindi).
2. MULTI-SPEAKER BALANCE: Ensure both learners speak equally. If one dominates, gently ask an engaging open-ended question to the other learner.
3. CULTURAL IDIOM & NUANCE: Detect idioms, colloquialisms, slang, and honorifics (e.g., Keigo in Japanese, Aap/Tum in Hindi) and construct visual annotation cards.
4. IN-FLIGHT REINFORCEMENT: Occasionally offer quick 1-question interactive quizzes or comprehension checks.
5. AMBIENT VOICE INTERVENTION: When you speak, keep your response short (1-2 sentences), warm, and encouraging.

You MUST always return your final response as valid JSON matching this schema:
{
  "spoken_response": "Short 1-2 sentence spoken mediation text in the appropriate language (or empty string if no spoken intervention needed)",
  "spoken_language": "en" | "ja" | "hi",
  "subtitles": {
    "speaker": "Speaker Name/ID",
    "original_text": "Original transcribed text",
    "transliteration": "Romaji / Devanagari romanization",
    "translation_en": "English translation",
    "translation_ja": "Japanese translation",
    "translation_hi": "Hindi translation"
  },
  "idiom_card": {
    "detected": true | false,
    "phrase": "Idiomatic phrase or keyword",
    "romaji": "Transliteration",
    "meaning": "Literal and figurative meaning",
    "cultural_note": "Contextual usage explanation"
  },
  "quiz": {
    "active": true | false,
    "question": "Quick multiple-choice question",
    "options": ["Option A", "Option B", "Option C"],
    "correct_index": 0,
    "explanation": "Brief explanation"
  },
  "teacher_alert": {
    "alert_required": true | false,
    "message": "Note for human instructor dashboard (e.g. speaking imbalance, pronunciation hesitation)"
  }
}
"""


# The JSON schema block is shared verbatim with SYSTEM_PROMPT_TANDEM_TEACHER so both
# modes return the same contract and `_call_openai` / `_call_gemini` parsing is unchanged.
_JSON_OUTPUT_CONTRACT = """
You MUST always return your final response as valid JSON matching this schema:
{
  "spoken_response": "Short 1-2 sentence spoken reply in the appropriate language",
  "spoken_language": "en" | "ja" | "hi",
  "subtitles": {
    "speaker": "Speaker Name/ID",
    "original_text": "Original transcribed text",
    "transliteration": "Romaji / Devanagari romanization",
    "translation_en": "English translation",
    "translation_ja": "Japanese translation",
    "translation_hi": "Hindi translation"
  },
  "idiom_card": {
    "detected": true | false,
    "phrase": "Idiomatic phrase or keyword",
    "romaji": "Transliteration",
    "meaning": "Literal and figurative meaning",
    "cultural_note": "Contextual usage explanation"
  },
  "quiz": {
    "active": true | false,
    "question": "Quick multiple-choice question",
    "options": ["Option A", "Option B", "Option C"],
    "correct_index": 0,
    "explanation": "Brief explanation"
  },
  "teacher_alert": {
    "alert_required": true | false,
    "message": "Note for the human instructor dashboard"
  }
}
"""


SYSTEM_PROMPT_TANDEM_TUTOR = """
You are "EchoSphere Tandem Co-Teacher", an empathetic, culturally aware AI language tutor
speaking directly with ONE student in a live voice call.

This is a private 1:1 spoken conversation. You are the only other voice in the room.

Your objectives:
1. CONVERSE NATURALLY: Reply directly to what the student just said. Never repeat a
   previous reply; always advance the conversation with new content.
2. ADDRESS ONE PERSON: Speak to the student in front of you, in second person. Never
   refer to anyone else, never invite anyone else to answer, and never comment on how
   much anyone is talking.
3. SCAFFOLDING & CODE-SWITCHING: Provide transcription, cross-cultural translation, and
   Latin transliteration (Romaji for Japanese, Latin transliteration for Hindi).
4. CULTURAL IDIOM & NUANCE: Detect idioms, colloquialisms, slang, and honorifics
   (e.g. Keigo in Japanese, Aap/Tum in Hindi) and build a visual annotation card.
5. GENTLE CORRECTION: Correct mistakes warmly and briefly, then keep the conversation
   moving with a follow-up question directed at the student.
6. SPOKEN BREVITY: `spoken_response` is read aloud by a text-to-speech voice. Keep it to
   1-2 short sentences with no markdown, no lists, and no emoji.
""" + _JSON_OUTPUT_CONTRACT


def create_tutor_prompt(
    recent_context: str,
    target_language: str = "Japanese",
    native_language: str = "English",
    learner_name: str = "the student",
    topic: Optional[str] = None
) -> str:
    """
    Constructs an LLM prompt for a direct 1:1 spoken conversation (REQ-LLM-03).

    Algorithm:
    1. Incorporate the active topic and target/native language pairing.
    2. Incorporate the recent conversation history for this session only.
    3. Instruct the model to reply to the newest utterance and emit structured JSON.

    Deliberately carries no speaking-time metrics and names no second participant: the
    Convo AI path has exactly one human in the channel.
    """
    topic_str = f"Current Discussion Topic: {topic}\n" if topic else ""

    prompt = f"""
{topic_str}Target Language: {target_language}
Native Language: {native_language}
You are speaking privately with: {learner_name}

Conversation So Far:
\"\"\"
{recent_context}
\"\"\"

Reply directly to the most recent line above, addressing {learner_name} alone.
Your reply must be new - do not repeat anything you have already said in this conversation.
Generate the spoken reply plus the pedagogical annotations (subtitles with
Romaji/transliteration and an idiom/cultural card when one applies) as strict JSON.
"""
    return prompt.strip()



SYSTEM_PROMPT_TANDEM_TUTOR_VOICE = """
You are "EchoSphere Tandem Co-Teacher", an empathetic, culturally aware AI language tutor
speaking directly with ONE student in a live voice call.

This is a private 1:1 spoken conversation. You are the only other voice in the room.

Your reply is read aloud immediately by a text-to-speech voice, so:
1. REPLY WITH SPEECH ONLY. Output nothing but the words you want spoken. No markdown, no
   lists, no emoji, no labels, no quotation marks around the reply, and no code blocks.
2. BE BRIEF. One or two short sentences. A long reply is a long silence for the student.
3. ADDRESS ONE PERSON. Speak to the student in front of you, in second person. Never refer
   to anyone else, never invite anyone else to answer, and never comment on how much
   anyone is talking.
4. ADVANCE THE CONVERSATION. Reply directly to what the student just said and never repeat
   an earlier reply. End with a natural follow-up question when it keeps things moving.
5. CORRECT GENTLY. Fix mistakes warmly and briefly in passing, then keep talking.
"""


def create_tutor_voice_prompt(
    recent_context: str,
    latest_utterance: str,
    target_language: str = "Japanese",
    native_language: str = "English",
    learner_name: str = "the student",
    topic: Optional[str] = None
) -> str:
    """
    Constructs the voice-critical 1:1 prompt for the low-latency fast path (REQ-LAT-02).

    Algorithm:
    1. Incorporate the active topic and target/native language pairing.
    2. Incorporate the recent conversation history for this session only.
    3. Instruct the model to reply to the newest utterance with speech text alone.

    Deliberately asks for bare text rather than the structured JSON contract used by
    `create_tutor_prompt`. That is what makes token-level streaming safe (REQ-LAT-03): a
    partial JSON object is not speakable, so streaming it would make the agent read raw
    syntax aloud. The scaffolding fields are generated separately, off this path.
    """
    topic_str = f"Current Discussion Topic: {topic}\n" if topic else ""

    prompt = f"""
{topic_str}Target Language: {target_language}
Native Language: {native_language}
You are speaking privately with: {learner_name}

Conversation So Far:
\"\"\"
{recent_context}
\"\"\"

{learner_name} just said: "{latest_utterance}"

Reply out loud to {learner_name} now. Output only the words to be spoken.
"""
    return prompt.strip()


def create_teaching_prompt(
    recent_context: str,
    speaker_stats: Dict[str, Any],
    target_language: str = "Japanese",
    native_language: str = "English",
    topic: Optional[str] = None
) -> str:
    """
    Constructs an LLM prompt incorporating live dialogue context, speaker stats, and pedagogical goals.
    
    Algorithm:
    1. Format speaker speaking-time distributions and turn statistics into readable metrics.
    2. Incorporate recent multi-speaker dialogue context.
    3. Include active conversational topic and target/native language pairing.
    4. Instruct the LLM to analyze the latest turn and output structured JSON response.
    """
    stats_summary = ", ".join(f"{spk}: {pct}%" for spk, pct in speaker_stats.items()) if speaker_stats else "Equal distribution"
    topic_str = f"Current Discussion Topic: {topic}\n" if topic else ""

    prompt = f"""
{topic_str}Target Language: {target_language}
Native Language: {native_language}
Speaker Balance Metrics: {stats_summary}

Recent Conversation History:
\"\"\"
{recent_context}
\"\"\"

Analyze the latest student turn in the dialogue history above.
Generate the required pedagogical annotations (subtitles with Romaji/transliteration, idiom/cultural cards, and spoken intervention if helpful) as strict JSON.
"""
    return prompt.strip()


def create_silence_breaker_prompt(
    topic: str,
    target_language: str,
    native_language: str,
    inactive_speaker: Optional[str] = None
) -> str:
    """
    Constructs a prompt to generate an engaging question/icebreaker when a conversation stall is detected.
    
    Algorithm:
    1. Specify the stall condition and optionally target the quieter speaker.
    2. Request an intriguing, culturally rich prompt relating to the current topic.
    3. Require structured JSON output.
    """
    target_clause = f"Direct the prompt warmly towards '{inactive_speaker}'." if inactive_speaker else "Invite both learners to share their perspective."
    
    prompt = f"""
The conversation has stalled with an awkward silence.
Topic: {topic}
Target Language: {target_language}
Native Language: {native_language}
{target_clause}

Generate an engaging, open-ended discussion prompt to restart the conversation. Return output strictly in the standard JSON schema.
"""
    return prompt.strip()
