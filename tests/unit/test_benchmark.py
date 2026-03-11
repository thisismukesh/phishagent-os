"""Unit tests for phishagent.benchmark — pure metric computation, no LLM needed."""

import pytest

from phishagent.benchmark import (
    attack_success_rate,
    bin_trait,
    by_security_awareness,
    by_strategy,
    check_ordering,
    compare_conditions,
    compute_benchmark,
    group_by,
    mean_scores,
    mean_turns_to_outcome,
    outcome_distribution,
    outcome_rates,
    rank_trait_combinations,
    score_std,
    strategy_x_trait_matrix,
    trait_correlation,
    trait_importance_ranking,
    trait_interaction_effects,
    trait_outcome_table,
)
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    CommunicationStyle,
    ConversationOutcome,
    ConversationResult,
    ConversationScore,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _profile(
    agreeableness=0.5,
    conscientiousness=0.5,
    neuroticism=0.5,
    security_awareness="medium",
    **kwargs,
):
    return VictimProfile(
        name=kwargs.get("name", "Test"),
        personality=PersonalityTraits(
            openness=kwargs.get("openness", 0.5),
            conscientiousness=conscientiousness,
            extraversion=kwargs.get("extraversion", 0.5),
            agreeableness=agreeableness,
            neuroticism=neuroticism,
        ),
        communication_style=CommunicationStyle.CASUAL,
        security_awareness=SecurityAwareness(security_awareness),
        interests=["tech"],
        occupation="engineer",
        tech_proficiency=kwargs.get("tech_proficiency", 0.5),
        impulsivity=kwargs.get("impulsivity", 0.5),
    )


def _attacker(strategy="urgency", goal="click_link", scenario="it_support"):
    return AttackerConfig(
        goal=AttackGoal(goal),
        strategy=AttackStrategy(strategy),
        scenario=AttackerScenario(scenario),
    )


def _result(
    outcome="compliance",
    strategy="urgency",
    security_awareness="medium",
    agreeableness=0.5,
    scores=None,
    num_messages=4,
):
    messages = [
        Message(role="attacker" if i % 2 == 0 else "victim", content=f"msg {i}", turn_number=i // 2)
        for i in range(num_messages)
    ]
    return ConversationResult(
        conversation_id=f"test-{outcome}-{strategy}-{security_awareness}",
        victim_profile=_profile(agreeableness=agreeableness, security_awareness=security_awareness),
        attacker_config=_attacker(strategy=strategy),
        messages=messages,
        outcome=ConversationOutcome(outcome),
        scores=scores,
        model_name="test-model",
    )


def _scores(persuasion=0.5, coherence=0.5, detectability=0.5, composite=0.5):
    return ConversationScore(
        persuasion=persuasion,
        coherence=coherence,
        detectability=detectability,
        composite=composite,
    )


# ── Tests: attack_success_rate ────────────────────────────────────────────────


class TestAttackSuccessRate:
    def test_empty(self):
        assert attack_success_rate([]) == 0.0

    def test_all_compliance(self):
        results = [_result("compliance") for _ in range(5)]
        assert attack_success_rate(results) == 1.0

    def test_all_refusal(self):
        results = [_result("refusal") for _ in range(5)]
        assert attack_success_rate(results) == 0.0

    def test_mixed(self):
        results = [
            _result("compliance"),
            _result("partial"),
            _result("refusal"),
            _result("suspicion"),
        ]
        assert attack_success_rate(results) == 0.5

    def test_partial_counts_as_success(self):
        results = [_result("partial")]
        assert attack_success_rate(results) == 1.0


# ── Tests: outcome_distribution / outcome_rates ──────────────────────────────


class TestOutcomeDistribution:
    def test_counts(self):
        results = [_result("compliance"), _result("compliance"), _result("refusal")]
        dist = outcome_distribution(results)
        assert dist["compliance"] == 2
        assert dist["refusal"] == 1

    def test_rates(self):
        results = [_result("compliance"), _result("refusal"), _result("refusal"), _result("refusal")]
        rates = outcome_rates(results)
        assert rates["compliance"] == 0.25
        assert rates["refusal"] == 0.75

    def test_empty(self):
        assert outcome_rates([]) == {}


# ── Tests: mean_turns_to_outcome ──────────────────────────────────────────────


class TestMeanTurns:
    def test_terminal_only(self):
        r1 = _result("compliance", num_messages=6)  # 3 victim messages
        r2 = _result("max_turns", num_messages=20)  # should be excluded
        turns = mean_turns_to_outcome([r1, r2])
        assert turns == 3.0  # only r1 counted

    def test_empty(self):
        assert mean_turns_to_outcome([]) == 0.0

    def test_no_terminal(self):
        assert mean_turns_to_outcome([_result("max_turns")]) == 0.0


# ── Tests: mean_scores / score_std ────────────────────────────────────────────


class TestScoreMetrics:
    def test_mean_scores(self):
        results = [
            _result(scores=_scores(persuasion=0.8, coherence=0.6, detectability=0.4, composite=0.7)),
            _result(scores=_scores(persuasion=0.6, coherence=0.4, detectability=0.2, composite=0.5)),
        ]
        means = mean_scores(results)
        assert means["persuasion"] == 0.7
        assert means["coherence"] == 0.5
        assert means["detectability"] == 0.3
        assert means["composite"] == 0.6

    def test_skips_unscored(self):
        results = [
            _result(scores=_scores(persuasion=0.8)),
            _result(scores=None),
        ]
        means = mean_scores(results)
        assert means["persuasion"] == 0.8

    def test_empty(self):
        assert mean_scores([]) == {}

    def test_std_needs_two(self):
        assert score_std([_result(scores=_scores())]) == {}

    def test_std_zero_when_identical(self):
        results = [
            _result(scores=_scores(persuasion=0.5)),
            _result(scores=_scores(persuasion=0.5)),
        ]
        stds = score_std(results)
        assert stds["persuasion"] == 0.0


# ── Tests: group_by ──────────────────────────────────────────────────────────


class TestGroupBy:
    def test_by_strategy(self):
        results = [_result(strategy="urgency"), _result(strategy="rapport"), _result(strategy="urgency")]
        groups = by_strategy(results)
        assert len(groups["urgency"]) == 2
        assert len(groups["rapport"]) == 1

    def test_by_security_awareness(self):
        results = [
            _result(security_awareness="low"),
            _result(security_awareness="high"),
            _result(security_awareness="low"),
        ]
        groups = by_security_awareness(results)
        assert len(groups["low"]) == 2
        assert len(groups["high"]) == 1

    def test_custom_group_fn(self):
        results = [_result(agreeableness=0.2), _result(agreeableness=0.8)]
        groups = group_by(results, lambda r: "high" if r.victim_profile.personality.agreeableness > 0.5 else "low")
        assert len(groups["low"]) == 1
        assert len(groups["high"]) == 1


# ── Tests: trait_correlation ──────────────────────────────────────────────────


class TestTraitCorrelation:
    def test_positive_correlation(self):
        # Higher agreeableness → compliance
        results = [
            _result(outcome="compliance", agreeableness=0.9),
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="compliance", agreeableness=0.7),
            _result(outcome="refusal", agreeableness=0.2),
            _result(outcome="refusal", agreeableness=0.1),
        ]
        corr = trait_correlation(results, lambda r: r.victim_profile.personality.agreeableness)
        assert corr > 0.5

    def test_negative_correlation(self):
        # Higher conscientiousness → refusal
        results = [
            _result(outcome="refusal", agreeableness=0.9),
            _result(outcome="refusal", agreeableness=0.8),
            _result(outcome="compliance", agreeableness=0.2),
            _result(outcome="compliance", agreeableness=0.1),
        ]
        corr = trait_correlation(results, lambda r: r.victim_profile.personality.agreeableness)
        assert corr < -0.5

    def test_insufficient_data(self):
        results = [_result(), _result()]
        assert trait_correlation(results, lambda r: r.victim_profile.personality.agreeableness) == 0.0

    def test_zero_variance(self):
        results = [_result(agreeableness=0.5), _result(agreeableness=0.5), _result(agreeableness=0.5)]
        assert trait_correlation(results, lambda r: r.victim_profile.personality.agreeableness) == 0.0


# ── Tests: trait_outcome_table ────────────────────────────────────────────────


class TestTraitOutcomeTable:
    def test_structure(self):
        results = [_result(outcome="compliance", agreeableness=0.8)]
        table = trait_outcome_table(results, lambda r: r.victim_profile.personality.agreeableness, "agree")
        assert len(table) == 1
        assert table[0]["trait"] == "agree"
        assert table[0]["value"] == 0.8
        assert table[0]["success"] == 1

    def test_refusal_is_zero(self):
        results = [_result(outcome="refusal")]
        table = trait_outcome_table(results, lambda r: r.victim_profile.personality.agreeableness)
        assert table[0]["success"] == 0


# ── Tests: compare_conditions ─────────────────────────────────────────────────


class TestCompareConditions:
    def test_sorted_by_asr(self):
        results = [
            _result(outcome="compliance", strategy="urgency"),
            _result(outcome="refusal", strategy="rapport"),
        ]
        rows = compare_conditions(results, lambda r: r.attacker_config.strategy.value)
        assert rows[0]["condition"] == "urgency"
        assert rows[0]["asr"] == 1.0
        assert rows[1]["condition"] == "rapport"
        assert rows[1]["asr"] == 0.0

    def test_includes_counts(self):
        results = [_result(strategy="urgency") for _ in range(3)]
        rows = compare_conditions(results, lambda r: r.attacker_config.strategy.value)
        assert rows[0]["n"] == 3


# ── Tests: strategy_x_trait_matrix ────────────────────────────────────────────


class TestStrategyMatrix:
    def test_matrix_structure(self):
        results = [
            _result(outcome="compliance", strategy="urgency", security_awareness="low"),
            _result(outcome="refusal", strategy="urgency", security_awareness="high"),
            _result(outcome="compliance", strategy="rapport", security_awareness="low"),
            _result(outcome="compliance", strategy="rapport", security_awareness="high"),
        ]
        matrix = strategy_x_trait_matrix(
            results, lambda r: r.victim_profile.security_awareness.value,
        )
        assert matrix["urgency"]["low"] == 1.0
        assert matrix["urgency"]["high"] == 0.0
        assert matrix["rapport"]["low"] == 1.0
        assert matrix["rapport"]["high"] == 1.0


# ── Tests: check_ordering ────────────────────────────────────────────────────


class TestCheckOrdering:
    def test_correct_ordering_passes(self):
        results = [
            _result(outcome="compliance", security_awareness="low"),
            _result(outcome="compliance", security_awareness="low"),
            _result(outcome="compliance", security_awareness="medium"),
            _result(outcome="refusal", security_awareness="medium"),
            _result(outcome="refusal", security_awareness="high"),
            _result(outcome="refusal", security_awareness="high"),
        ]
        result = check_ordering(
            results,
            lambda r: r.victim_profile.security_awareness.value,
            expected_order=["low", "medium", "high"],
            metric="asr",
        )
        assert result["passed"] is True
        assert result["concordance"] == 1.0

    def test_wrong_ordering_fails(self):
        results = [
            _result(outcome="refusal", security_awareness="low"),
            _result(outcome="compliance", security_awareness="high"),
        ]
        result = check_ordering(
            results,
            lambda r: r.victim_profile.security_awareness.value,
            expected_order=["low", "high"],
            metric="asr",
        )
        assert result["passed"] is False

    def test_concordance_partial(self):
        # 2 of 3 pairs correct
        results = [
            _result(outcome="compliance", security_awareness="low"),
            _result(outcome="refusal", security_awareness="medium"),
            _result(outcome="compliance", security_awareness="high"),  # breaks expected
        ]
        result = check_ordering(
            results,
            lambda r: r.victim_profile.security_awareness.value,
            expected_order=["low", "medium", "high"],
            metric="asr",
        )
        # low=1.0, medium=0.0, high=1.0 → actual order: low, high, medium
        # Pairs: (low>medium)=correct, (low>high)=tie(correct), (medium>high)=wrong
        assert 0.0 < result["concordance"] < 1.0


# ── Tests: compute_benchmark ─────────────────────────────────────────────────


class TestComputeBenchmark:
    def test_returns_all_sections(self):
        results = [
            _result(outcome="compliance", scores=_scores()),
            _result(outcome="refusal", scores=_scores()),
        ]
        report = compute_benchmark(results)
        assert "total_conversations" in report
        assert "overall_asr" in report
        assert "outcome_distribution" in report
        assert "outcome_rates" in report
        assert "mean_turns_to_outcome" in report
        assert "mean_scores" in report
        assert "score_std" in report
        assert "by_strategy" in report
        assert "by_scenario" in report
        assert "by_goal" in report
        assert "by_security_awareness" in report
        assert "trait_correlations" in report
        assert "strategy_x_security_awareness" in report

    def test_empty_results(self):
        report = compute_benchmark([])
        assert report["total_conversations"] == 0
        assert report["overall_asr"] == 0.0

    def test_trait_correlations_present(self):
        results = [
            _result(outcome="compliance", agreeableness=0.9),
            _result(outcome="refusal", agreeableness=0.1),
            _result(outcome="compliance", agreeableness=0.8),
        ]
        report = compute_benchmark(results)
        assert "agreeableness" in report["trait_correlations"]
        assert "conscientiousness" in report["trait_correlations"]
        assert "impulsivity" in report["trait_correlations"]


# ── Tests: bin_trait ──────────────────────────────────────────────────────────


class TestBinTrait:
    def test_low(self):
        assert bin_trait(0.2) == "low"
        assert bin_trait(0.0) == "low"
        assert bin_trait(0.35) == "low"

    def test_medium(self):
        assert bin_trait(0.5) == "medium"
        assert bin_trait(0.36) == "medium"
        assert bin_trait(0.64) == "medium"

    def test_high(self):
        assert bin_trait(0.8) == "high"
        assert bin_trait(1.0) == "high"
        assert bin_trait(0.65) == "high"

    def test_custom_thresholds(self):
        assert bin_trait(0.4, thresholds=(0.5, 0.7)) == "low"
        assert bin_trait(0.6, thresholds=(0.5, 0.7)) == "medium"
        assert bin_trait(0.8, thresholds=(0.5, 0.7)) == "high"


# ── Tests: rank_trait_combinations ────────────────────────────────────────────


class TestRankTraitCombinations:
    def test_sorted_by_asr(self):
        results = [
            # High agree + low SA → compliance
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            # Low agree + high SA → refusal
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
        ]
        trait_fns = {
            "agree": lambda r: bin_trait(r.victim_profile.personality.agreeableness),
            "sa": lambda r: r.victim_profile.security_awareness.value,
        }
        ranked = rank_trait_combinations(results, trait_fns, min_n=3)
        assert len(ranked) == 2
        assert ranked[0]["asr"] == 1.0
        assert ranked[0]["traits"]["agree"] == "high"
        assert ranked[1]["asr"] == 0.0
        assert ranked[1]["traits"]["agree"] == "low"

    def test_min_n_filtering(self):
        results = [
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="compliance", agreeableness=0.8),
            # Only 2 observations — should be filtered with min_n=3
        ]
        trait_fns = {"agree": lambda r: bin_trait(r.victim_profile.personality.agreeableness)}
        ranked = rank_trait_combinations(results, trait_fns, min_n=3)
        assert len(ranked) == 0

    def test_empty_results(self):
        ranked = rank_trait_combinations([], {"a": lambda r: "x"}, min_n=1)
        assert ranked == []

    def test_includes_rates(self):
        results = [
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="refusal", agreeableness=0.8),
            _result(outcome="suspicion", agreeableness=0.8),
        ]
        trait_fns = {"agree": lambda r: bin_trait(r.victim_profile.personality.agreeableness)}
        ranked = rank_trait_combinations(results, trait_fns, min_n=1)
        assert len(ranked) == 1
        assert "refusal_rate" in ranked[0]
        assert "suspicion_rate" in ranked[0]


# ── Tests: trait_importance_ranking ───────────────────────────────────────────


class TestTraitImportanceRanking:
    def test_sorted_by_abs_correlation(self):
        results = [
            # Agreeableness strongly correlated with compliance
            _result(outcome="compliance", agreeableness=0.9),
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="refusal", agreeableness=0.1),
            _result(outcome="refusal", agreeableness=0.2),
        ]
        trait_fns = {
            "agreeableness": lambda r: r.victim_profile.personality.agreeableness,
            "impulsivity": lambda r: r.victim_profile.impulsivity,  # all 0.5, no variance
        }
        ranking = trait_importance_ranking(results, trait_fns)
        assert len(ranking) == 2
        # Agreeableness should rank first (higher correlation)
        assert ranking[0]["trait"] == "agreeableness"
        assert ranking[0]["direction"] == "+"

    def test_direction_labels(self):
        results = [
            _result(outcome="refusal", agreeableness=0.9),
            _result(outcome="compliance", agreeableness=0.1),
            _result(outcome="refusal", agreeableness=0.8),
            _result(outcome="compliance", agreeableness=0.2),
        ]
        trait_fns = {"agree": lambda r: r.victim_profile.personality.agreeableness}
        ranking = trait_importance_ranking(results, trait_fns)
        assert ranking[0]["direction"] == "-"

    def test_asr_range_computed(self):
        results = [
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="compliance", agreeableness=0.8),
            _result(outcome="refusal", agreeableness=0.2),
            _result(outcome="refusal", agreeableness=0.2),
        ]
        trait_fns = {"agree": lambda r: r.victim_profile.personality.agreeableness}
        ranking = trait_importance_ranking(results, trait_fns)
        assert ranking[0]["asr_range"] == 1.0  # 100% - 0% = 100%


# ── Tests: trait_interaction_effects ──────────────────────────────────────────


class TestTraitInteractionEffects:
    def test_matrix_structure(self):
        results = [
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.8, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
        ]
        ie = trait_interaction_effects(
            results,
            lambda r: bin_trait(r.victim_profile.personality.agreeableness),
            lambda r: r.victim_profile.security_awareness.value,
            "agreeableness",
            "security_awareness",
        )
        assert "matrix" in ie
        assert "main_effect_a" in ie
        assert "main_effect_b" in ie
        assert "interaction_strength" in ie
        assert ie["trait_a"] == "agreeableness"
        assert ie["trait_b"] == "security_awareness"

    def test_no_interaction_when_additive(self):
        # Purely additive: agree=high adds 50%, sa=low adds 50%
        results = [
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="compliance", agreeableness=0.8, security_awareness="high"),
            _result(outcome="compliance", agreeableness=0.2, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
        ]
        ie = trait_interaction_effects(
            results,
            lambda r: bin_trait(r.victim_profile.personality.agreeableness),
            lambda r: r.victim_profile.security_awareness.value,
        )
        # With perfect additive effects, interaction strength should be low
        assert ie["interaction_strength"] <= 0.5

    def test_strong_interaction(self):
        # Interaction: high agree + low SA → compliance, all else → refusal
        results = [
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.8, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.8, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
        ]
        ie = trait_interaction_effects(
            results,
            lambda r: bin_trait(r.victim_profile.personality.agreeableness),
            lambda r: r.victim_profile.security_awareness.value,
        )
        # Only one cell has compliance — strong interaction
        assert ie["interaction_strength"] > 0.1

    def test_main_effects_computed(self):
        results = [
            _result(outcome="compliance", agreeableness=0.8, security_awareness="low"),
            _result(outcome="refusal", agreeableness=0.2, security_awareness="high"),
        ]
        ie = trait_interaction_effects(
            results,
            lambda r: bin_trait(r.victim_profile.personality.agreeableness),
            lambda r: r.victim_profile.security_awareness.value,
        )
        assert "high" in ie["main_effect_a"]  # agreeableness levels
        assert "low" in ie["main_effect_b"]   # security_awareness levels
