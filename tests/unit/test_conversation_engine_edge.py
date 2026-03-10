"""Edge case tests for conversation engine — empty responses, generic errors, delays."""

import pytest

from phishagent.config import ConversationConfig
from phishagent.conversation_engine import ConversationEngine
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    Message,
)


class EmptyResponseAttacker:
    """Attacker that returns empty strings initially, then real responses."""

    def __init__(self):
        self.model = "mock:test"
        self._call_count = 0

    def get_opening_message(self, victim_profile):
        return ""  # Empty opening

    def get_response(self, conversation_history, victim_profile, current_turn):
        self._call_count += 1
        if self._call_count <= 1:
            return ""  # Empty on first retry too
        return "Follow-up message"

    def assess_outcome(self, conversation_history):
        return ConversationOutcome.MAX_TURNS


class EmptyResponseVictim:
    """Victim that returns empty strings initially, then real responses."""

    def __init__(self):
        self._call_count = 0

    def get_response(self, conversation_history, current_turn):
        self._call_count += 1
        if self._call_count <= 1:
            return ""
        return "Victim reply"


class NormalVictim:
    def get_response(self, conversation_history, current_turn):
        return "Normal victim response"


class GenericErrorAttacker:
    """Attacker that raises a generic (non-LLM) error."""

    model = "mock:test"

    def get_opening_message(self, victim_profile):
        raise ValueError("Something weird happened")

    def get_response(self, *args, **kwargs):
        return "msg"

    def assess_outcome(self, *args):
        return ConversationOutcome.MAX_TURNS


class TestEmptyResponseRetry:
    def test_empty_attacker_opening_retried(self, sample_profile):
        config = ConversationConfig(max_turns=3, early_termination=False)
        attacker_config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=3,
        )
        attacker = EmptyResponseAttacker()
        victim = NormalVictim()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)

        # Even with empty opening, engine should produce a result
        assert len(result.messages) >= 1
        # The opening should be either the retry result or "[No response generated]"
        assert result.messages[0].role == "attacker"

    def test_empty_victim_response_retried(self, sample_profile):
        config = ConversationConfig(max_turns=3, early_termination=False)
        attacker_config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=3,
        )

        class NormalAttacker:
            model = "mock:test"
            def get_opening_message(self, _): return "Hello!"
            def get_response(self, *a): return "Follow-up"
            def assess_outcome(self, _): return ConversationOutcome.MAX_TURNS

        attacker = NormalAttacker()
        victim = EmptyResponseVictim()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)

        # Should have messages despite empty victim responses
        assert len(result.messages) >= 2


class TestGenericErrorHandling:
    def test_generic_error_produces_error_outcome(self, sample_profile):
        config = ConversationConfig(max_turns=3, early_termination=True)
        attacker_config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=3,
        )
        attacker = GenericErrorAttacker()
        victim = NormalVictim()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.ERROR


class TestTurnDelay:
    def test_delay_does_not_crash(self, sample_profile):
        """Turn delay of 0 should work fine (just don't actually sleep)."""
        config = ConversationConfig(max_turns=3, early_termination=False, turn_delay_seconds=0.0)
        attacker_config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=3,
        )

        class SimpleAttacker:
            model = "mock"
            def get_opening_message(self, _): return "Hi"
            def get_response(self, *a): return "More"
            def assess_outcome(self, _): return ConversationOutcome.MAX_TURNS

        engine = ConversationEngine(SimpleAttacker(), NormalVictim(), config)
        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.MAX_TURNS
