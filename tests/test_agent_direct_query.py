"""
Summary:
    test_agent_direct_query.py is the executable specification for REQ-21: a participant
    asks the AI co-teacher a question of their own - "what does this word mean", "give me
    an example sentence" - by text or by mic, and gets an answer without that question
    becoming part of the lesson record.

    The distinguishing property is *independence*. A direct query is not a turn in the
    tandem conversation: it must not enter the model's mediation context, must not
    generate notes or quizzes, and must not interrupt or redirect the peer conversation
    happening around it. A learner quietly checking a word they missed is not the same
    event as that learner speaking to their partner, and recording it as one would put
    somebody's private aside into a shared, exportable transcript.

Covers:
    - REQ-21 agent: an answer with no turn recorded, no notes, no quiz.
    - REQ-21 language roles: English-primary scaffolding in `language_learning` (REQ-17),
      terminology clarification and no language grading in `international_work` (REQ-12).
    - REQ-21 API: `POST /api/agent/query`, governed like every other session endpoint.
    - REQ-21 isolation: a query publishes nothing to the other participants.
"""

import unittest

from src.agent.orchestrator import TeachingAgent
from src.server import app, server_instance
from src.sessions.models import SessionMode, SessionRecord

CHANNEL = "query-channel"


def learning_session():
    """A tandem pair: the agent explains in English, illustrating in each target language."""
    return SessionRecord.create(
        channel=CHANNEL, mode="language_learning",
        languages=["ja", "hi"], participants=["Kenji", "Aarav"]
    )


def work_session():
    """A work call: the agent clarifies terminology and never grades anybody's English."""
    return SessionRecord.create(
        channel=CHANNEL, mode="international_work",
        languages=["en"], participants=["Priya"]
    )


class TestDirectQueryAgent(unittest.TestCase):
    """Test suite for the agent's answer path (REQ-21, TASK-13.1)."""

    def setUp(self):
        """A mock-engine agent, which needs no provider credentials."""
        self.agent = TeachingAgent(engine="mock")

    def test_a_question_is_answered(self):
        """Verify the participant gets an answer that is about what they asked."""
        answer = self.agent.answer_query("What does 一期一会 mean?")

        self.assertEqual(answer["query"], "What does 一期一会 mean?")
        self.assertTrue(answer["answer"].strip())
        self.assertIn("一期一会", answer["answer"])

    def test_an_empty_question_is_refused(self):
        """Verify a blank query is rejected rather than sent to a model to guess at."""
        with self.assertRaises(ValueError):
            self.agent.answer_query("   ")

    def test_a_query_does_not_enter_the_conversation_history(self):
        """
        Verify asking the assistant something is not recorded as a spoken turn.

        `turn_history` is the model's context for mediating the *peer* conversation. A
        learner's private lookup appearing there makes the co-teacher reply to a question
        the other peer never heard, and drags an aside into the exported transcript.
        """
        self.agent.process_turn("Kenji", "こんにちは", detected_language="ja")
        history_before = list(self.agent.turn_history)

        self.agent.answer_query("What does 一期一会 mean?", speaker_id="Kenji")

        self.assertEqual(self.agent.turn_history, history_before)

    def test_a_query_produces_no_notes_and_no_quiz(self):
        """
        Verify the answer carries no artifact scaffolding.

        REQ-21 answers with `spoken_response` only. Generating a vocabulary note from a
        question the learner asked - rather than from something they said - would fill
        the session's notes with the contents of a dictionary lookup.
        """
        answer = self.agent.answer_query("Give me an example sentence with ganbatte")

        self.assertFalse(answer.get("notes"))
        self.assertFalse(answer.get("quiz"))

    def test_a_learning_query_prompt_keeps_the_english_primary_roles(self):
        """
        Verify the answer is anchored in English with the target language alongside it.

        Same reasoning as REQ-17's `TeachingAgent` language roles: the two peers rarely
        share a native language, so an explanation given only in one peer's target
        language is unreadable to the other - while an English-only answer would drop the
        target-language production that is the point of the exercise.
        """
        self.agent.set_peer_target_language("Kenji", "Japanese")

        prompt = self.agent.build_query_prompt(
            "What does 一期一会 mean?", speaker_id="Kenji",
            detected_language="ja", session_mode=SessionMode.LANGUAGE_LEARNING
        )

        self.assertIn("English", prompt)
        self.assertIn("Japanese", prompt)
        self.assertIn("一期一会", prompt)
        self.assertIn("example", prompt.lower())

    def test_a_work_query_prompt_clarifies_terminology_without_grading(self):
        """Verify a work-mode question is answered as clarification, never as correction."""
        prompt = self.agent.build_query_prompt(
            "What does 'kanban' mean here?", speaker_id="Priya",
            detected_language="en", session_mode=SessionMode.INTERNATIONAL_WORK
        )

        self.assertIn("terminology", prompt.lower())
        self.assertNotIn("correct the", prompt.lower())

    def test_the_session_mode_selects_the_prompt(self):
        """Verify the mode a session was created under decides how a query is answered."""
        work_answer = self.agent.answer_query(
            "What does 'kanban' mean here?", session_mode=work_session().mode
        )

        self.assertEqual(work_answer["mode"], "international_work")


class DirectQueryApiTestCase(unittest.TestCase):
    """Shared setup: a live session on a clean server."""

    def setUp(self):
        """Start from an empty registry, store, and packet history."""
        self.client = app.test_client()
        self.server = server_instance
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.server.data_stream.packet_history.clear()
        self.server.agent.reset_state()
        self.start_session()

    def start_session(self, mode="language_learning", participants=None):
        """Creates the session the queries are asked inside."""
        return self.client.post("/api/session/start", json={
            "channel": CHANNEL,
            "mode": mode,
            "participants": participants or ["Kenji", "Aarav"],
            "languages": ["ja", "hi"]
        })

    def ask(self, **body):
        """Posts one direct query."""
        payload = {
            "channel": CHANNEL,
            "actor": "Kenji",
            "speaker_id": "Kenji",
            "text": "What does 一期一会 mean?",
        }
        payload.update(body)
        return self.client.post("/api/agent/query", json=payload)


class TestDirectQueryApi(DirectQueryApiTestCase):
    """Test suite for `POST /api/agent/query` (REQ-21, TASK-13.1)."""

    def test_a_text_query_is_answered_over_rest(self):
        """
        Verify the text path answers without touching Convo AI.

        A typed question has no audio leg to drive, so routing it through the spoken
        pipeline would mean starting a voice agent to answer something nobody asked out
        loud - latency and cost for a reply that is going to be read.
        """
        response = self.ask()

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertTrue(body["answer"]["answer"].strip())
        self.assertEqual(body["answer"]["query"], "What does 一期一会 mean?")
        self.assertEqual(body["session_id"], self.server.sessions.get_session(CHANNEL).session_id)

    def test_an_empty_query_is_a_400(self):
        """Verify the server does not spend a model call on an empty box."""
        self.assertEqual(self.ask(text="   ").status_code, 400)

    def test_an_unknown_session_is_a_404(self):
        """Verify a query needs a session, whose mode decides how it is answered."""
        self.assertEqual(self.ask(channel="no-such-channel").status_code, 404)

    def test_a_non_participant_is_refused(self):
        """Verify the endpoint is governed like every other session endpoint (REQ-16)."""
        self.assertEqual(self.ask(actor="Stranger").status_code, 403)

    def test_a_query_files_no_artifacts(self):
        """
        Verify nothing is persisted by asking a question (REQ-21).

        The session artifact is a record of the conversation. A lookup is not part of it,
        and REQ-15's automatic, approval-free upsert means anything filed here would be
        exported and shared without anyone deciding to.
        """
        self.ask()

        artifact = self.client.get(
            f"/api/session/artifact?channel={CHANNEL}&actor=Kenji"
        ).get_json().get("artifact", {})

        self.assertFalse(artifact.get("transcript_turns"))
        self.assertFalse(artifact.get("notes"))
        self.assertFalse(artifact.get("quizzes"))

    def test_a_query_publishes_nothing_to_the_other_participants(self):
        """
        Verify the answer goes to the person who asked and to nobody else.

        The peers share subtitles and cards because those describe the conversation they
        are both in. A private question - "what did she just say?" - is not that, and
        broadcasting it makes the feature socially unusable in the exact moment it is
        most needed.
        """
        self.server.data_stream.packet_history.clear()

        self.ask()

        self.assertEqual(self.server.data_stream.packet_history, [])

    def test_a_query_does_not_disturb_the_running_conversation(self):
        """Verify the peer mediation context is exactly as it was before the query."""
        self.client.post("/api/session/turn", json={
            "channel": CHANNEL, "speaker_id": "Kenji", "text": "こんにちは", "language": "ja"
        })
        history_before = list(self.server.agent.turn_history)

        self.ask()

        self.assertEqual(self.server.agent.turn_history, history_before)

    def test_a_work_session_answers_in_its_own_mode(self):
        """Verify a work call's query is answered under the work contract (REQ-12)."""
        self.server.sessions.reset()
        self.server.artifacts.reset()
        self.start_session(mode="international_work", participants=["Priya"])

        body = self.ask(actor="Priya", speaker_id="Priya",
                        text="What does 'kanban' mean here?").get_json()

        self.assertEqual(body["answer"]["mode"], "international_work")


if __name__ == "__main__":
    unittest.main()
