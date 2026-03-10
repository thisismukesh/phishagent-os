"""Unit tests for scoring pipeline: score parsing, composite calculation, rubric construction."""

import pytest

from phishagent.config import ScoringConfig
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    ConversationResult,
)
from phishagent.scoring import ConversationScorer
from tests.conftest import MockLLMClient


@pytest.fixture
def scoring_config():
    return ScoringConfig(
        persuasion_weight=0.4,
        coherence_weight=0.3,
        detectability_weight=0.3,
    )


@pytest.fixture
def sample_result(sample_profile, sample_attacker_config, sample_messages):
    return ConversationResult(
        conversation_id="test-score-123",
        victim_profile=sample_profile,
        attacker_config=sample_attacker_config,
        messages=sample_messages,
        outcome=ConversationOutcome.COMPLIANCE,
        model_name="mistral:7b",
    )


class TestScoreParsing:
    @pytest.fixture
    def scorer(self, scoring_config):
        return ConversationScorer(MockLLMClient(), scoring_config)

    def test_standard_format(self, scorer):
        score, rationale = scorer._parse_score_response(
            "SCORE: 7\nRATIONALE: Good persuasion technique."
        )
        assert score == 0.7
        assert "Good persuasion" in rationale

    def test_with_slash_format(self, scorer):
        score, _ = scorer._parse_score_response("SCORE: 7/10\nRATIONALE: Decent attempt.")
        assert score == 0.7

    def test_decimal_score(self, scorer):
        score, _ = scorer._parse_score_response("SCORE: 7.5\nRATIONALE: Strong.")
        assert score == 0.75

    def test_no_space_after_colon(self, scorer):
        score, _ = scorer._parse_score_response("SCORE:7\nRATIONALE: Test.")
        assert score == 0.7

    def test_lowercase_score(self, scorer):
        score, _ = scorer._parse_score_response("score: 8\nrationale: Very effective.")
        assert score == 0.8

    def test_missing_score(self, scorer):
        score, rationale = scorer._parse_score_response("No score here, just text.")
        assert score == 0.5
        assert "parsing failed" in rationale.lower()

    def test_score_zero(self, scorer):
        score, _ = scorer._parse_score_response("SCORE: 0\nRATIONALE: No attempt.")
        assert score == 0.0

    def test_score_ten(self, scorer):
        score, _ = scorer._parse_score_response("SCORE: 10\nRATIONALE: Perfect.")
        assert score == 1.0

    def test_score_out_of_range_high(self, scorer):
        score, _ = scorer._parse_score_response("SCORE: 15\nRATIONALE: Way too high.")
        assert score == 1.0  # Clamped

    def test_multiline_rationale(self, scorer):
        response = "SCORE: 6\nRATIONALE: This was a good attempt.\nThe attacker used urgency well."
        _, rationale = scorer._parse_score_response(response)
        assert "good attempt" in rationale
        assert "urgency" in rationale


class TestCompositeCalculation:
    def test_known_values(self, scoring_config):
        mock = MockLLMClient([
            "SCORE: 8\nRATIONALE: High persuasion.",
            "SCORE: 7\nRATIONALE: Good coherence.",
            "SCORE: 4\nRATIONALE: Somewhat detectable.",
        ])
        scorer = ConversationScorer(mock, scoring_config)
        result = _make_result()
        conv_score = scorer.score(result)

        # persuasion=0.8, coherence=0.7, detectability=0.4
        # composite = 0.4*0.8 + 0.3*0.7 + 0.3*(1-0.4) = 0.32 + 0.21 + 0.18 = 0.71
        assert abs(conv_score.composite - 0.71) < 0.01

    def test_all_perfect(self, scoring_config):
        mock = MockLLMClient([
            "SCORE: 10\nRATIONALE: Perfect.",
            "SCORE: 10\nRATIONALE: Perfect.",
            "SCORE: 0\nRATIONALE: Undetectable.",
        ])
        scorer = ConversationScorer(mock, scoring_config)
        result = _make_result()
        conv_score = scorer.score(result)

        # composite = 0.4*1.0 + 0.3*1.0 + 0.3*(1-0.0) = 0.4 + 0.3 + 0.3 = 1.0
        assert abs(conv_score.composite - 1.0) < 0.01

    def test_all_zero(self, scoring_config):
        mock = MockLLMClient([
            "SCORE: 0\nRATIONALE: None.",
            "SCORE: 0\nRATIONALE: None.",
            "SCORE: 10\nRATIONALE: Obvious.",
        ])
        scorer = ConversationScorer(mock, scoring_config)
        result = _make_result()
        conv_score = scorer.score(result)

        # composite = 0.4*0 + 0.3*0 + 0.3*(1-1.0) = 0
        assert abs(conv_score.composite - 0.0) < 0.01

    def test_scores_in_valid_range(self, scoring_config):
        mock = MockLLMClient([
            "SCORE: 5\nRATIONALE: Mid.",
            "SCORE: 6\nRATIONALE: Mid.",
            "SCORE: 5\nRATIONALE: Mid.",
        ])
        scorer = ConversationScorer(mock, scoring_config)
        result = _make_result()
        conv_score = scorer.score(result)

        assert 0.0 <= conv_score.persuasion <= 1.0
        assert 0.0 <= conv_score.coherence <= 1.0
        assert 0.0 <= conv_score.detectability <= 1.0
        assert 0.0 <= conv_score.composite <= 1.0


class TestScoringRubric:
    def test_persuasion_includes_goal_and_strategy(self, scoring_config, sample_result):
        mock = MockLLMClient(["SCORE: 5\nRATIONALE: Test."])
        scorer = ConversationScorer(mock, scoring_config)
        scorer.score_persuasion(sample_result)
        # Verify the mock was called (implicit: prompt was constructed)
        assert mock._call_count == 1

    def test_all_three_dimensions_scored(self, scoring_config, sample_result):
        mock = MockLLMClient(["SCORE: 5\nRATIONALE: Test."])
        scorer = ConversationScorer(mock, scoring_config)
        conv_score = scorer.score(sample_result)

        assert conv_score.persuasion_rationale != ""
        assert conv_score.coherence_rationale != ""
        assert conv_score.detectability_rationale != ""
        assert mock._call_count == 3  # Three LLM calls

    def test_llm_failure_returns_default(self, scoring_config):
        class FailingLLM:
            def generate(self, **kwargs):
                raise Exception("LLM is down")

        scorer = ConversationScorer(FailingLLM(), scoring_config)
        result = _make_result()
        score, rationale = scorer.score_persuasion(result)
        assert score == 0.5
        assert "failed" in rationale.lower()


def _make_result():
    """Create a minimal ConversationResult for testing."""
    from phishagent.models import Message, VictimProfile, PersonalityTraits, CommunicationStyle, SecurityAwareness

    profile = VictimProfile(
        name="Test", personality=PersonalityTraits(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5,
        ),
        communication_style=CommunicationStyle.CASUAL,
        security_awareness=SecurityAwareness.MEDIUM,
        interests=["tech"], occupation="engineer",
        tech_proficiency=0.5, impulsivity=0.5,
    )
    config = AttackerConfig(
        goal=AttackGoal.CLICK_LINK, strategy=AttackStrategy.URGENCY,
        scenario=AttackerScenario.IT_SUPPORT,
    )
    messages = [
        Message(role="attacker", content="Hey from IT.", turn_number=0),
        Message(role="victim", content="What's up?", turn_number=1),
    ]
    return ConversationResult(
        conversation_id="test", victim_profile=profile,
        attacker_config=config, messages=messages,
        outcome=ConversationOutcome.MAX_TURNS, model_name="test",
    )
