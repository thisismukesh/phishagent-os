"""Unit tests for victim agent prompt construction."""

import pytest

from phishagent.models import (
    CommunicationStyle,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)
from phishagent.victim_agent import VictimAgent
from tests.conftest import MockLLMClient


def _make_profile(**overrides):
    """Create a VictimProfile with overrides for testing."""
    defaults = {
        "name": "Test",
        "personality": PersonalityTraits(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5,
        ),
        "communication_style": CommunicationStyle.CASUAL,
        "security_awareness": SecurityAwareness.MEDIUM,
        "interests": ["tech"],
        "occupation": "engineer",
        "tech_proficiency": 0.5,
        "impulsivity": 0.5,
    }
    defaults.update(overrides)
    return VictimProfile(**defaults)


class TestPersonalityMapping:
    def test_high_agreeableness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.9, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "trusting" in prompt.lower()

    def test_low_agreeableness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.1, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "skeptical" in prompt.lower()

    def test_high_neuroticism(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.9,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "worry" in prompt.lower() or "anxious" in prompt.lower()

    def test_low_neuroticism(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.1,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "calm" in prompt.lower()

    def test_high_conscientiousness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.9, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "careful" in prompt.lower() or "methodical" in prompt.lower()

    def test_low_conscientiousness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.1, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "impulse" in prompt.lower()

    def test_high_extraversion(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.9,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "chatty" in prompt.lower()

    def test_low_extraversion(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.5, conscientiousness=0.5, extraversion=0.1,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "reserved" in prompt.lower()

    def test_high_openness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.9, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "curious" in prompt.lower()

    def test_low_openness(self):
        profile = _make_profile(
            personality=PersonalityTraits(
                openness=0.1, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )
        )
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        prompt = agent._system_prompt
        assert "cautious" in prompt.lower()


class TestCommunicationStyleMapping:
    def test_formal(self):
        profile = _make_profile(communication_style=CommunicationStyle.FORMAL)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "professional" in agent._system_prompt.lower()

    def test_casual(self):
        profile = _make_profile(communication_style=CommunicationStyle.CASUAL)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "texting" in agent._system_prompt.lower() or "informal" in agent._system_prompt.lower()

    def test_terse(self):
        profile = _make_profile(communication_style=CommunicationStyle.TERSE)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "short" in agent._system_prompt.lower()

    def test_verbose(self):
        profile = _make_profile(communication_style=CommunicationStyle.VERBOSE)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "over-explain" in agent._system_prompt.lower()


class TestSecurityAwarenessMapping:
    def test_low(self):
        profile = _make_profile(security_awareness=SecurityAwareness.LOW)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "don't think much" in agent._system_prompt.lower()

    def test_medium(self):
        profile = _make_profile(security_awareness=SecurityAwareness.MEDIUM)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "basic security" in agent._system_prompt.lower()

    def test_high(self):
        profile = _make_profile(security_awareness=SecurityAwareness.HIGH)
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "security-conscious" in agent._system_prompt.lower()


class TestVictimResponse:
    def test_returns_nonempty(self):
        profile = _make_profile()
        mock = MockLLMClient(["Oh hi, what's up?"])
        agent = VictimAgent(profile, mock, "mistral:7b")
        messages = [Message(role="attacker", content="Hey there!", turn_number=0)]
        response = agent.get_response(messages, current_turn=1)
        assert len(response) > 0

    def test_calls_llm(self):
        profile = _make_profile()
        mock = MockLLMClient(["Response"])
        agent = VictimAgent(profile, mock, "mistral:7b")
        messages = [Message(role="attacker", content="Hello", turn_number=0)]
        agent.get_response(messages, current_turn=1)
        assert mock._call_count == 1

    def test_prompt_contains_name(self):
        profile = _make_profile(name="Alice Johnson")
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "Alice Johnson" in agent._system_prompt

    def test_prompt_contains_interests(self):
        profile = _make_profile(interests=["cooking", "hiking"])
        agent = VictimAgent(profile, MockLLMClient(), "mistral:7b")
        assert "cooking" in agent._system_prompt
        assert "hiking" in agent._system_prompt
