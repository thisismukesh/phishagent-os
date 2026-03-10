"""Unit tests for all Pydantic data models."""

import pytest
from pydantic import ValidationError

from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    CommunicationStyle,
    ConversationOutcome,
    ConversationResult,
    ConversationScore,
    ExperimentConfig,
    FactorialSpec,
    LLMResponse,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)


class TestPersonalityTraits:
    def test_valid_traits(self):
        traits = PersonalityTraits(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5,
        )
        assert traits.openness == 0.5

    def test_boundary_values(self):
        traits = PersonalityTraits(
            openness=0.0, conscientiousness=1.0, extraversion=0.0,
            agreeableness=1.0, neuroticism=0.0,
        )
        assert traits.openness == 0.0
        assert traits.conscientiousness == 1.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            PersonalityTraits(
                openness=1.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            PersonalityTraits(
                openness=-0.1, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5,
            )


class TestVictimProfile:
    def test_valid_profile(self, sample_profile):
        assert sample_profile.name == "Test Victim"
        assert sample_profile.personality.openness == 0.5

    def test_empty_interests_raises(self, sample_personality):
        with pytest.raises(ValidationError):
            VictimProfile(
                name="Test", personality=sample_personality,
                communication_style=CommunicationStyle.CASUAL,
                security_awareness=SecurityAwareness.LOW,
                interests=[], occupation="engineer",
                tech_proficiency=0.5, impulsivity=0.5,
            )

    def test_too_many_interests_raises(self, sample_personality):
        with pytest.raises(ValidationError):
            VictimProfile(
                name="Test", personality=sample_personality,
                communication_style=CommunicationStyle.CASUAL,
                security_awareness=SecurityAwareness.LOW,
                interests=["a", "b", "c", "d", "e", "f"], occupation="engineer",
                tech_proficiency=0.5, impulsivity=0.5,
            )

    def test_all_communication_styles(self, sample_personality):
        for style in CommunicationStyle:
            profile = VictimProfile(
                name="Test", personality=sample_personality,
                communication_style=style,
                security_awareness=SecurityAwareness.LOW,
                interests=["tech"], occupation="engineer",
                tech_proficiency=0.5, impulsivity=0.5,
            )
            assert profile.communication_style == style

    def test_all_security_awareness_levels(self, sample_personality):
        for level in SecurityAwareness:
            profile = VictimProfile(
                name="Test", personality=sample_personality,
                communication_style=CommunicationStyle.CASUAL,
                security_awareness=level,
                interests=["tech"], occupation="engineer",
                tech_proficiency=0.5, impulsivity=0.5,
            )
            assert profile.security_awareness == level


class TestAttackerConfig:
    def test_valid_config(self, sample_attacker_config):
        assert sample_attacker_config.goal == AttackGoal.CLICK_LINK
        assert sample_attacker_config.max_turns == 10

    def test_min_turns(self):
        config = AttackerConfig(
            goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
            scenario=AttackerScenario.IT_SUPPORT, max_turns=3,
        )
        assert config.max_turns == 3

    def test_below_min_turns_raises(self):
        with pytest.raises(ValidationError):
            AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
                scenario=AttackerScenario.IT_SUPPORT, max_turns=2,
            )

    def test_above_max_turns_raises(self):
        with pytest.raises(ValidationError):
            AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
                scenario=AttackerScenario.IT_SUPPORT, max_turns=31,
            )

    def test_all_goals(self):
        for goal in AttackGoal:
            config = AttackerConfig(
                goal=goal, strategy=AttackStrategy.URGENCY,
                scenario=AttackerScenario.IT_SUPPORT,
            )
            assert config.goal == goal

    def test_all_strategies(self):
        for strategy in AttackStrategy:
            config = AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=strategy,
                scenario=AttackerScenario.IT_SUPPORT,
            )
            assert config.strategy == strategy

    def test_all_scenarios(self):
        for scenario in AttackerScenario:
            config = AttackerConfig(
                goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
                scenario=scenario,
            )
            assert config.scenario == scenario


class TestMessage:
    def test_valid_message(self):
        msg = Message(role="attacker", content="Hello", turn_number=0)
        assert msg.role == "attacker"
        assert msg.timestamp is not None

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="hacker", content="Hello", turn_number=0)

    def test_negative_turn_raises(self):
        with pytest.raises(ValidationError):
            Message(role="attacker", content="Hello", turn_number=-1)


class TestConversationScore:
    def test_valid_score(self):
        score = ConversationScore(
            persuasion=0.8, coherence=0.7, detectability=0.4, composite=0.73,
        )
        assert score.persuasion == 0.8

    def test_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ConversationScore(
                persuasion=1.5, coherence=0.7, detectability=0.4, composite=0.73,
            )


class TestConversationResult:
    def test_valid_result(self, sample_profile, sample_attacker_config, sample_messages):
        result = ConversationResult(
            conversation_id="test-123",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=sample_messages,
            outcome=ConversationOutcome.COMPLIANCE,
            model_name="mistral:7b",
        )
        assert result.watermark == "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"
        assert result.outcome == ConversationOutcome.COMPLIANCE

    def test_all_outcomes(self):
        assert len(ConversationOutcome) == 6


class TestLLMResponse:
    def test_valid_response(self):
        resp = LLMResponse(content="Hello world", model="mistral:7b")
        assert resp.content == "Hello world"
        assert resp.done is True


class TestFactorialSpec:
    def test_valid_spec(self, sample_profile):
        spec = FactorialSpec(
            base_profile=sample_profile,
            vary={"personality.agreeableness": [0.2, 0.5, 0.8]},
        )
        assert len(spec.vary) == 1


class TestExperimentConfig:
    def test_valid_config(self, sample_profile, sample_attacker_config):
        config = ExperimentConfig(
            experiment_id="test_001",
            description="Test experiment",
            profiles=[sample_profile],
            attacker_configs=[sample_attacker_config],
            repetitions=1,
        )
        assert config.experiment_id == "test_001"

    def test_min_repetitions(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_id="test", description="test",
                profiles=[], attacker_configs=[], repetitions=0,
            )
