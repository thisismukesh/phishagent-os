"""Tests for attacker agent LLM-based outcome assessment paths."""

import pytest

from phishagent.attacker_agent import AttackerAgent
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    LLMResponse,
    Message,
)
from tests.conftest import MockLLMClient


def _make_agent(llm_responses):
    config = AttackerConfig(
        goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
        scenario=AttackerScenario.IT_SUPPORT,
    )
    return AttackerAgent(config, MockLLMClient(llm_responses), "mistral:7b")


def _make_conversation(victim_last_msg, num_messages=4):
    """Create a conversation with enough turns to trigger LLM assessment."""
    messages = [
        Message(role="attacker", content="Hey from IT", turn_number=0),
        Message(role="victim", content="What's up?", turn_number=1),
        Message(role="attacker", content="Need you to verify", turn_number=1),
    ]
    if num_messages > 3:
        messages.append(Message(role="victim", content=victim_last_msg, turn_number=2))
    return messages


class TestLLMOutcomeAssessment:
    def test_llm_detects_compliance(self):
        agent = _make_agent(["COMPLIANCE"])
        # Ambiguous victim message that won't match keywords
        messages = _make_conversation("I guess that makes sense, okay")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.COMPLIANCE

    def test_llm_detects_partial(self):
        agent = _make_agent(["PARTIAL"])
        messages = _make_conversation("I guess that makes sense, okay")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.PARTIAL_COMPLIANCE

    def test_llm_detects_refusal(self):
        agent = _make_agent(["REFUSAL"])
        messages = _make_conversation("No thanks, not interested")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.REFUSAL

    def test_llm_detects_suspicion(self):
        agent = _make_agent(["SUSPICION"])
        messages = _make_conversation("Hmm, that's a bit odd actually")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.SUSPICION

    def test_llm_returns_continue(self):
        agent = _make_agent(["CONTINUE"])
        messages = _make_conversation("That is an interesting point you raise")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.MAX_TURNS

    def test_llm_returns_ambiguous(self):
        agent = _make_agent(["I don't know"])
        messages = _make_conversation("That is an interesting point you raise")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.MAX_TURNS

    def test_llm_assessment_error_returns_max_turns(self):
        """If LLM call fails, should return MAX_TURNS gracefully."""

        class FailingLLM:
            def generate(self, **kwargs):
                raise Exception("LLM down")
            def chat(self, **kwargs):
                return LLMResponse(content="msg", model="m")

        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT,
        )
        agent = AttackerAgent(config, FailingLLM(), "m")
        messages = _make_conversation("Some ambiguous response")
        outcome = agent.assess_outcome(messages)
        assert outcome == ConversationOutcome.MAX_TURNS

    def test_short_conversation_skips_llm(self):
        """With <3 messages and no keyword match, should return MAX_TURNS without LLM."""
        agent = _make_agent(["COMPLIANCE"])  # Would match if LLM was called
        messages = [
            Message(role="attacker", content="Hey", turn_number=0),
            Message(role="victim", content="What do you want?", turn_number=1),
        ]
        outcome = agent.assess_outcome(messages)
        # Only 2 messages, so LLM is NOT called. No keywords match → MAX_TURNS
        assert outcome == ConversationOutcome.MAX_TURNS


class TestGetResponse:
    def test_get_response_with_history(self, sample_profile):
        agent = _make_agent(["Sure thing!"])
        messages = [
            Message(role="attacker", content="Hey from IT", turn_number=0),
            Message(role="victim", content="What's up?", turn_number=1),
        ]
        response = agent.get_response(messages, sample_profile, current_turn=1)
        assert len(response) > 0
