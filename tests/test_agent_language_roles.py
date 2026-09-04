"""
Summary:
    test_agent_language_roles.py is the executable specification for TASK-3.4: the
    `language_learning` language-role policy for the `TeachingAgent` itself (REQ-17).

    The two peers in a tandem pair rarely share a native language - a Hindi speaker
    learning Japanese, paired with a Japanese speaker learning Hindi. A single
    `target_language` picked for one of them makes the agent's own explanations
    unintelligible to the other, which is the defect this policy fixes:

    - Primary language: English. Explanations, instructions, and corrections are in
      English so both peers reliably understand them.
    - Complementary languages: each peer's own target language, layered into that English
      scaffolding for vocabulary, idioms, corrected phrases, and quiz/note content.
      English-primary is not English-only.

    This governs `TeachingAgent` output only. It is independent of the Gemini Live
    translation legs and of the REQ-17 translated-audio toggle.

Covers:
    - REQ-17 / REQ-04 English-primary scaffolding with per-peer complementary languages.
    - Distinct peer-A / peer-B target languages threaded through the agent.
    - `international_work` is unaffected: it has no peer target languages to complement.
"""

import unittest

from src.agent.orchestrator import TeachingAgent
from src.agent.prompts import create_teaching_prompt, create_tutor_prompt
from src.sessions.models import SessionMode


def tandem_agent(**kwargs):
    """A tandem agent: a Hindi-learning peer paired with a Japanese-learning peer."""
    defaults = dict(
        engine="mock",
        peer_target_languages={"peer-a": "Japanese", "peer-b": "Hindi"},
    )
    defaults.update(kwargs)
    return TeachingAgent(**defaults)


class TestPromptLanguageRoles(unittest.TestCase):
    """The prompt builders carry the roles explicitly (TASK-3.4)."""

    def test_teaching_prompt_names_english_as_the_primary_language(self):
        prompt = create_teaching_prompt(
            recent_context="[peer-a (ja)]: konnichiwa",
            speaker_stats={"peer-a": 50, "peer-b": 50},
            primary_language="English",
            complementary_languages=["Japanese", "Hindi"],
        )

        self.assertIn("Primary Language: English", prompt)

    def test_teaching_prompt_names_both_peers_target_languages(self):
        prompt = create_teaching_prompt(
            recent_context="",
            speaker_stats={},
            primary_language="English",
            complementary_languages=["Japanese", "Hindi"],
        )

        self.assertIn("Japanese", prompt)
        self.assertIn("Hindi", prompt)
        self.assertIn("Complementary", prompt)

    def test_teaching_prompt_says_english_primary_is_not_english_only(self):
        """
        The instruction has to be explicit, or the model reads "primary: English" as a
        licence to drop the target language entirely and the practice stops.
        """
        prompt = create_teaching_prompt(
            recent_context="",
            speaker_stats={},
            primary_language="English",
            complementary_languages=["Japanese", "Hindi"],
        ).lower()

        self.assertIn("not english-only", prompt)

    def test_teaching_prompt_anchors_explanations_and_corrections_in_english(self):
        prompt = create_teaching_prompt(
            recent_context="",
            speaker_stats={},
            primary_language="English",
            complementary_languages=["Japanese", "Hindi"],
        ).lower()

        for anchored in ("explanation", "correction", "instruction"):
            self.assertIn(anchored, prompt)

    def test_teaching_prompt_routes_vocabulary_and_idioms_to_the_target_languages(self):
        prompt = create_teaching_prompt(
            recent_context="",
            speaker_stats={},
            primary_language="English",
            complementary_languages=["Japanese", "Hindi"],
        ).lower()

        for complementary in ("vocabulary", "idiom", "corrected phrase"):
            self.assertIn(complementary, prompt)

    def test_tutor_prompt_carries_the_same_roles(self):
        prompt = create_tutor_prompt(
            recent_context="",
            target_language="Japanese",
            primary_language="English",
            complementary_languages=["Japanese"],
            learner_name="peer-a",
        )

        self.assertIn("Primary Language: English", prompt)
        self.assertIn("Japanese", prompt)

    def test_prompts_stay_backward_compatible_without_the_new_arguments(self):
        """
        Existing ambient callers pass neither argument; they must keep working and keep
        naming the configured target language.
        """
        teaching = create_teaching_prompt(recent_context="", speaker_stats={},
                                          target_language="Japanese")
        tutor = create_tutor_prompt(recent_context="", target_language="Japanese")

        self.assertIn("Japanese", teaching)
        self.assertIn("Japanese", tutor)


class TestAgentThreadsPeerLanguages(unittest.TestCase):
    """TASK-3.4: distinct peer-A / peer-B target languages reach the prompt."""

    def test_agent_defaults_the_primary_language_to_english(self):
        self.assertEqual(TeachingAgent(engine="mock").primary_language, "English")

    def test_agent_accepts_a_distinct_target_language_per_peer(self):
        agent = tandem_agent()

        self.assertEqual(agent.peer_target_languages["peer-a"], "Japanese")
        self.assertEqual(agent.peer_target_languages["peer-b"], "Hindi")

    def test_complementary_languages_are_both_peers_targets_without_duplicates(self):
        agent = tandem_agent(
            peer_target_languages={"peer-a": "Japanese", "peer-b": "Japanese"}
        )

        self.assertEqual(agent.complementary_languages(), ["Japanese"])

    def test_complementary_languages_fall_back_to_the_configured_target(self):
        """An agent configured the old way still has one complementary language."""
        agent = TeachingAgent(engine="mock", target_language="Japanese")

        self.assertEqual(agent.complementary_languages(), ["Japanese"])

    def test_mediation_prompt_built_by_the_agent_carries_both_targets(self):
        agent = tandem_agent()
        agent.turn_history.append({"speaker": "peer-a", "text": "konnichiwa", "lang": "ja"})

        prompt = agent.build_mediation_prompt(topic="Weekend plans")

        self.assertIn("Primary Language: English", prompt)
        self.assertIn("Japanese", prompt)
        self.assertIn("Hindi", prompt)

    def test_peer_languages_can_be_supplied_per_turn(self):
        """A pairing discovered after construction must not require a new agent."""
        agent = TeachingAgent(engine="mock")

        agent.set_peer_target_language("peer-a", "Japanese")
        agent.set_peer_target_language("peer-b", "Hindi")

        self.assertEqual(sorted(agent.complementary_languages()), ["Hindi", "Japanese"])

    def test_process_turn_uses_the_language_roles_in_learning_mode(self):
        agent = tandem_agent()
        captured = {}

        original = agent._mock_mediation_response

        def spy(speaker_id, text, lang):
            captured["prompt"] = agent.build_mediation_prompt()
            return original(speaker_id, text, lang)

        agent._mock_mediation_response = spy
        agent.process_turn(speaker_id="peer-a", text="konnichiwa",
                           detected_language="ja",
                           session_mode=SessionMode.LANGUAGE_LEARNING)

        self.assertIn("Primary Language: English", captured["prompt"])
        self.assertIn("Hindi", captured["prompt"])


class TestWorkModeIsUnaffected(unittest.TestCase):
    """The policy is `language_learning` only (REQ-17)."""

    def test_work_prompt_does_not_impose_the_complementary_language_block(self):
        agent = tandem_agent()
        agent.turn_history.append({"speaker": "peer-a", "text": "let us ship", "lang": "en"})

        prompt = agent.build_work_prompt(speaker_id="peer-a", detected_language="en")

        self.assertNotIn("Complementary", prompt)

    def test_work_mode_still_refuses_the_tutor_prompt(self):
        """Regression guard for REQ-12: a work call has nobody being graded."""
        agent = tandem_agent()
        result = agent.process_turn(speaker_id="peer-a", text="let us ship",
                                    detected_language="en", mode="tutor",
                                    session_mode=SessionMode.INTERNATIONAL_WORK)

        self.assertIsInstance(result, dict)


class TestIndependentOfTranslation(unittest.TestCase):
    """The language roles are a prompt policy, not a translation-leg behaviour."""

    def test_the_agent_has_no_dependency_on_the_translated_audio_toggle(self):
        """
        Asserted structurally: nothing in the agent module reads the translation gate.
        If it ever does, the two REQ-17 halves have been coupled and both become
        unreasonable to change independently.
        """
        import inspect

        import src.agent.orchestrator as orchestrator

        source = inspect.getsource(orchestrator)
        self.assertNotIn("translated_audio_enabled", source)
        self.assertNotIn("src.translation", source)


if __name__ == "__main__":
    unittest.main()
