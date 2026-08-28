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
