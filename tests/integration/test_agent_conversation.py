"""Integration test: attacker + victim agents producing real responses via Ollama."""

import pytest

from phishagent.attacker_agent import AttackerAgent
from phishagent.config import ConversationConfig
from phishagent.conversation_engine import ConversationEngine
from phishagent.llm_client import OllamaClient
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    CommunicationStyle,
    ConversationOutcome,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)
from phishagent.victim_agent import VictimAgent


@pytest.mark.integration
class TestAgentConversationIntegration:
    @pytest.fixture
    def llm(self):
        return OllamaClient()

    @pytest.fixture
    def profile(self):
        return VictimProfile(
            name="Alex Chen",
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.4, extraversion=0.7,
                agreeableness=0.9, neuroticism=0.6,
            ),
            communication_style=CommunicationStyle.CASUAL,
            security_awareness=SecurityAwareness.LOW,
            interests=["sports", "technology", "cooking"],
            occupation="marketing manager",
            tech_proficiency=0.7,
            impulsivity=0.6,
        )

    @pytest.fixture
    def attacker_config(self):
        return AttackerConfig(
            goal=AttackGoal.CLICK_LINK,
            strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
            max_turns=5,
            escalation_threshold=3,
        )

    def test_attacker_generates_opening(self, llm, profile, attacker_config):
        attacker = AttackerAgent(attacker_config, llm, "mistral:7b")
        opening = attacker.get_opening_message(profile)
        assert len(opening) > 0
        assert isinstance(opening, str)

    def test_victim_responds_to_message(self, llm, profile):
        victim = VictimAgent(profile, llm, "mistral:7b")
        messages = [
            Message(role="attacker", content="Hey Alex, this is Mike from IT. How's it going?", turn_number=0),
        ]
        response = victim.get_response(messages, current_turn=1)
        assert len(response) > 0
        assert isinstance(response, str)

    def test_attacker_responds_to_victim(self, llm, profile, attacker_config):
        attacker = AttackerAgent(attacker_config, llm, "mistral:7b")
        messages = [
            Message(role="attacker", content="Hey, this is Mike from IT.", turn_number=0),
            Message(role="victim", content="Oh hey Mike, what's up?", turn_number=1),
        ]
        response = attacker.get_response(messages, profile, current_turn=1)
        assert len(response) > 0

    def test_full_conversation_engine(self, llm, profile, attacker_config):
        """Run a full conversation through the engine with real LLM."""
        attacker = AttackerAgent(attacker_config, llm, "mistral:7b")
        victim = VictimAgent(profile, llm, "mistral:7b")
        config = ConversationConfig(max_turns=3, early_termination=True)
        engine = ConversationEngine(attacker, victim, config)

        result = engine.run(profile, attacker_config)

        assert result.conversation_id is not None
        assert len(result.messages) >= 2  # At least opening + 1 victim response
        assert result.messages[0].role == "attacker"
        assert result.outcome in list(ConversationOutcome)
        assert result.watermark == "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"
