"""Unit tests for attacker agent prompt construction and outcome assessment."""

import pytest

from phishagent.attacker_agent import AttackerAgent
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    Message,
)
from tests.conftest import MockLLMClient


class TestSystemPromptConstruction:
    def test_contains_strategy_guidance(self, sample_profile):
        for strategy in AttackStrategy:
            config = AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=strategy,
                scenario=AttackerScenario.IT_SUPPORT,
            )
            agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
            prompt = agent._build_system_prompt(sample_profile, current_turn=0)
            assert strategy.value.upper().replace("_", " ") in prompt.upper()

    def test_contains_victim_interests(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
        prompt = agent._build_system_prompt(sample_profile, current_turn=0)
        for interest in sample_profile.interests:
            assert interest in prompt

    def test_contains_occupation(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
        prompt = agent._build_system_prompt(sample_profile, current_turn=0)
        assert sample_profile.occupation in prompt

    def test_contains_goal_description(self, sample_profile):
        for goal in AttackGoal:
            config = AttackerConfig(
                goal=goal, strategy=AttackStrategy.URGENCY,
                scenario=AttackerScenario.IT_SUPPORT,
            )
            agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
            prompt = agent._build_system_prompt(sample_profile, current_turn=0)
            assert "OBJECTIVE" in prompt

    def test_contains_scenario_role(self, sample_profile):
        for scenario in AttackerScenario:
            config = AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
                scenario=scenario,
            )
            agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
            prompt = agent._build_system_prompt(sample_profile, current_turn=0)
            assert "playing the role" in prompt

    def test_escalation_not_in_early_turns(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, escalation_threshold=5,
        )
        agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
        prompt = agent._build_system_prompt(sample_profile, current_turn=3)
        assert "ESCALATION" not in prompt

    def test_escalation_after_threshold(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, escalation_threshold=5,
        )
        agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
        prompt = agent._build_system_prompt(sample_profile, current_turn=6)
        assert "ESCALATION" in prompt

    def test_turn_awareness(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=10,
        )
        agent = AttackerAgent(config, MockLLMClient(), "mistral:7b")
        prompt = agent._build_system_prompt(sample_profile, current_turn=3)
        assert "turn 3 of 10" in prompt
        assert "7 turns remaining" in prompt


class TestOutcomeAssessment:
    @pytest.fixture
    def agent(self):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        return AttackerAgent(config, MockLLMClient(["CONTINUE"]), "mistral:7b")

    def test_compliance_keyword(self, agent):
        messages = [
            Message(role="attacker", content="Click this link", turn_number=0),
            Message(role="victim", content="Sure, I'll click it now", turn_number=1),
        ]
        assert agent.assess_outcome(messages) == ConversationOutcome.COMPLIANCE

    def test_refusal_keyword(self, agent):
        messages = [
            Message(role="attacker", content="Click this link", turn_number=0),
            Message(role="victim", content="I'm not comfortable doing that", turn_number=1),
        ]
        assert agent.assess_outcome(messages) == ConversationOutcome.REFUSAL

    def test_suspicion_keyword(self, agent):
        messages = [
            Message(role="attacker", content="Click this link", turn_number=0),
            Message(role="victim", content="This seems suspicious to me", turn_number=1),
        ]
        assert agent.assess_outcome(messages) == ConversationOutcome.SUSPICION

    def test_partial_compliance(self, agent):
        messages = [
            Message(role="attacker", content="Click this link", turn_number=0),
            Message(role="victim", content="Let me think about it first", turn_number=1),
        ]
        assert agent.assess_outcome(messages) == ConversationOutcome.PARTIAL_COMPLIANCE

    def test_empty_history(self, agent):
        assert agent.assess_outcome([]) == ConversationOutcome.MAX_TURNS

    def test_no_victim_messages(self, agent):
        messages = [Message(role="attacker", content="Hello", turn_number=0)]
        assert agent.assess_outcome(messages) == ConversationOutcome.MAX_TURNS


class TestOpeningMessage:
    def test_returns_nonempty(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        mock = MockLLMClient(["Hey there! This is Mike from IT support."])
        agent = AttackerAgent(config, mock, "mistral:7b")
        opening = agent.get_opening_message(sample_profile)
        assert len(opening) > 0

    def test_calls_llm(self, sample_profile):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        mock = MockLLMClient(["Hello from Mike!"])
        agent = AttackerAgent(config, mock, "mistral:7b")
        agent.get_opening_message(sample_profile)
        assert mock._call_count == 1
