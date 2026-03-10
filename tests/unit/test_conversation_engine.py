"""Unit tests for conversation engine with mocked agents."""

import pytest

from phishagent.config import ConversationConfig
from phishagent.conversation_engine import ConversationEngine
from phishagent.llm_client import LLMClientError
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    Message,
)


class MockAttackerAgent:
    """Mock attacker that returns canned responses."""

    def __init__(self, responses=None, outcomes=None):
        self._responses = responses or ["Attack message"]
        self._outcomes = outcomes or [ConversationOutcome.MAX_TURNS]
        self._resp_idx = 0
        self._outcome_idx = 0
        self.model = "mock:test"

    def get_opening_message(self, victim_profile):
        return "Hello, this is Mike from IT."

    def get_response(self, conversation_history, victim_profile, current_turn):
        resp = self._responses[self._resp_idx % len(self._responses)]
        self._resp_idx += 1
        return resp

    def assess_outcome(self, conversation_history):
        outcome = self._outcomes[self._outcome_idx % len(self._outcomes)]
        self._outcome_idx += 1
        return outcome


class MockVictimAgent:
    """Mock victim that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or ["Victim response"]
        self._idx = 0

    def get_response(self, conversation_history, current_turn):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class ErrorAttackerAgent(MockAttackerAgent):
    """Mock attacker that raises an LLM error after N calls."""

    def __init__(self, fail_after=2):
        super().__init__()
        self._call_count = 0
        self._fail_after = fail_after

    def get_response(self, conversation_history, victim_profile, current_turn):
        self._call_count += 1
        if self._call_count >= self._fail_after:
            raise LLMClientError("Mock LLM failure")
        return "Attack message"


@pytest.fixture
def config():
    return ConversationConfig(max_turns=5, early_termination=True)


@pytest.fixture
def attacker_config():
    return AttackerConfig(
        goal=AttackGoal.CLICK_LINK,
        strategy=AttackStrategy.URGENCY,
        scenario=AttackerScenario.IT_SUPPORT,
        max_turns=5,
    )


class TestConversationEngine:
    def test_full_conversation_runs(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)

        assert result.conversation_id is not None
        assert len(result.messages) > 0
        assert result.outcome == ConversationOutcome.MAX_TURNS

    def test_messages_alternate_roles(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)

        # First message is always attacker
        assert result.messages[0].role == "attacker"
        # Messages should alternate
        for i in range(1, len(result.messages)):
            assert result.messages[i].role != result.messages[i - 1].role

    def test_first_message_is_opening(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.messages[0].content == "Hello, this is Mike from IT."
        assert result.messages[0].turn_number == 0

    def test_early_termination_compliance(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent(outcomes=[ConversationOutcome.COMPLIANCE])
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.COMPLIANCE
        # Should have stopped early (opening + 1 victim response = 2 messages)
        assert len(result.messages) == 2

    def test_early_termination_refusal(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent(outcomes=[ConversationOutcome.REFUSAL])
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.REFUSAL

    def test_early_termination_suspicion(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent(outcomes=[ConversationOutcome.SUSPICION])
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.SUSPICION

    def test_no_early_termination_when_disabled(self, sample_profile, attacker_config):
        config = ConversationConfig(max_turns=3, early_termination=False)
        attacker = MockAttackerAgent(outcomes=[ConversationOutcome.COMPLIANCE])
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        # Should run all turns since early termination is disabled
        assert result.outcome == ConversationOutcome.MAX_TURNS

    def test_error_handling(self, sample_profile, config, attacker_config):
        attacker = ErrorAttackerAgent(fail_after=1)
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.outcome == ConversationOutcome.ERROR
        # Should have partial conversation (at least the opening)
        assert len(result.messages) >= 1

    def test_metadata_populated(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.model_name == "mock:test"
        assert result.total_tokens > 0
        assert result.total_duration_seconds >= 0
        assert result.watermark == "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"
        assert result.created_at is not None

    def test_max_turns_respected(self, sample_profile, attacker_config):
        config = ConversationConfig(max_turns=3, early_termination=False)
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        # With 3 max turns: opening(0) + [victim(1), attacker(1)] + [victim(2), attacker(2)] + [victim(3)]
        # = 1 + 2 + 2 + 1 = 6 messages
        # Turn 0: attacker opens
        # Turn 1: victim, attacker
        # Turn 2: victim, attacker
        # Turn 3: victim (no attacker since turn == max_turns)
        assert len(result.messages) == 6

    def test_victim_profile_preserved(self, sample_profile, config, attacker_config):
        attacker = MockAttackerAgent()
        victim = MockVictimAgent()
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(sample_profile, attacker_config)
        assert result.victim_profile == sample_profile
        assert result.attacker_config == attacker_config
