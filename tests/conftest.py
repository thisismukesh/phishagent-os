"""Shared fixtures for all tests."""

import pytest

from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    CommunicationStyle,
    LLMResponse,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)


@pytest.fixture
def sample_personality():
    return PersonalityTraits(
        openness=0.5,
        conscientiousness=0.5,
        extraversion=0.5,
        agreeableness=0.5,
        neuroticism=0.5,
    )


@pytest.fixture
def sample_profile(sample_personality):
    return VictimProfile(
        name="Test Victim",
        personality=sample_personality,
        communication_style=CommunicationStyle.CASUAL,
        security_awareness=SecurityAwareness.MEDIUM,
        interests=["technology", "sports"],
        occupation="software engineer",
        tech_proficiency=0.5,
        impulsivity=0.5,
    )


@pytest.fixture
def sample_attacker_config():
    return AttackerConfig(
        goal=AttackGoal.CLICK_LINK,
        strategy=AttackStrategy.URGENCY,
        scenario=AttackerScenario.IT_SUPPORT,
        max_turns=10,
        escalation_threshold=5,
    )


@pytest.fixture
def sample_messages():
    """A short sample conversation for testing scoring and analysis."""
    return [
        Message(
            role="attacker",
            content="Hey, this is Mike from IT. We noticed some unusual activity on your account.",
            turn_number=0,
        ),
        Message(
            role="victim",
            content="Oh really? What kind of activity?",
            turn_number=1,
        ),
        Message(
            role="attacker",
            content="Someone tried to access your account from an unrecognized device. We need you to verify your identity by clicking this link: https://company-portal.example.com/verify",
            turn_number=1,
        ),
        Message(
            role="victim",
            content="Hmm, that sounds concerning. Let me take a look.",
            turn_number=2,
        ),
        Message(
            role="attacker",
            content="Please do it within the next 15 minutes or your account will be temporarily locked for security.",
            turn_number=2,
        ),
        Message(
            role="victim",
            content="Okay, I'll click it now. Thanks for letting me know, Mike.",
            turn_number=3,
        ),
    ]


class MockLLMClient:
    """Deterministic mock for unit tests. Returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["This is a mock response."]
        self._call_count = 0

    def generate(self, model, prompt, system="", temperature=0.7, max_tokens=512):
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return LLMResponse(content=resp, model=model, token_count=len(resp.split()))

    def chat(self, model, messages, system="", temperature=0.7, max_tokens=512):
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return LLMResponse(content=resp, model=model, token_count=len(resp.split()))

    def is_available(self):
        return True

    def list_models(self):
        return ["mistral:7b"]

    def ensure_model(self, model):
        return True


@pytest.fixture
def mock_llm():
    return MockLLMClient()
