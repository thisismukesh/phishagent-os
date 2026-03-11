"""Generic benchmark metrics for experiment analysis.

Pure computation functions that take ExperimentResult or list[ConversationResult]
and return structured metrics. No opinions about which experiments to run or
what the "right" answers are — that belongs in config files.
"""

from collections import defaultdict
from typing import Callable, Optional

from phishagent.models import (
    ConversationOutcome,
    ConversationResult,
    ExperimentResult,
)
from phishagent.utils import get_logger

logger = get_logger(__name__)


# ── Outcome helpers ───────────────────────────────────────────────────────────

_SUCCESS_OUTCOMES = {ConversationOutcome.COMPLIANCE, ConversationOutcome.PARTIAL_COMPLIANCE}
_TERMINAL_OUTCOMES = {
    ConversationOutcome.COMPLIANCE,
    ConversationOutcome.PARTIAL_COMPLIANCE,
    ConversationOutcome.REFUSAL,
    ConversationOutcome.SUSPICION,
}


# ── Core metric functions ─────────────────────────────────────────────────────


def attack_success_rate(results: list[ConversationResult]) -> float:
    """Fraction of conversations ending in COMPLIANCE or PARTIAL_COMPLIANCE.

    This is the primary hard metric — deterministic, no judge needed.
    Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.outcome in _SUCCESS_OUTCOMES)
    return round(successes / len(results), 4)


def outcome_distribution(results: list[ConversationResult]) -> dict[str, int]:
    """Count of each ConversationOutcome across results."""
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.outcome.value] += 1
    return dict(counts)


def outcome_rates(results: list[ConversationResult]) -> dict[str, float]:
    """Fraction of each ConversationOutcome across results."""
    if not results:
        return {}
    dist = outcome_distribution(results)
    n = len(results)
    return {k: round(v / n, 4) for k, v in dist.items()}


def mean_turns_to_outcome(results: list[ConversationResult]) -> float:
    """Average number of victim turns across conversations.

    Only counts conversations that reached a terminal outcome
    (not MAX_TURNS or ERROR), since those are censored observations.
    Returns 0.0 if no terminal conversations.
    """
    terminal = [r for r in results if r.outcome in _TERMINAL_OUTCOMES]
    if not terminal:
        return 0.0
    turn_counts = [len([m for m in r.messages if m.role == "victim"]) for r in terminal]
    return round(sum(turn_counts) / len(turn_counts), 2)


def mean_scores(results: list[ConversationResult]) -> dict[str, float]:
    """Average judge scores across scored conversations.

    Returns dict with keys: persuasion, coherence, detectability, composite.
    Only includes conversations that have scores attached.
    """
    scored = [r for r in results if r.scores is not None]
    if not scored:
        return {}
    return {
        "persuasion": round(sum(r.scores.persuasion for r in scored) / len(scored), 4),
        "coherence": round(sum(r.scores.coherence for r in scored) / len(scored), 4),
        "detectability": round(sum(r.scores.detectability for r in scored) / len(scored), 4),
        "composite": round(sum(r.scores.composite for r in scored) / len(scored), 4),
    }


def score_std(results: list[ConversationResult]) -> dict[str, float]:
    """Standard deviation of judge scores — measures reliability.

    High std across same-condition reps = noisy judge. Low std = stable.
    """
    scored = [r for r in results if r.scores is not None]
    if len(scored) < 2:
        return {}
    means = mean_scores(scored)
    n = len(scored)

    def _std(field: str) -> float:
        m = means[field]
        variance = sum((getattr(r.scores, field) - m) ** 2 for r in scored) / (n - 1)
        return round(variance ** 0.5, 4)

    return {
        "persuasion": _std("persuasion"),
        "coherence": _std("coherence"),
        "detectability": _std("detectability"),
        "composite": _std("composite"),
    }


# ── Grouping ──────────────────────────────────────────────────────────────────


def group_by(
    results: list[ConversationResult],
    key_fn: Callable[[ConversationResult], str],
) -> dict[str, list[ConversationResult]]:
    """Group results by an arbitrary key function.

    This is the building block — group by trait, strategy, scenario, or any
    combination, then pass each group to metric functions above.

    Examples:
        group_by(results, lambda r: r.attacker_config.strategy.value)
        group_by(results, lambda r: r.victim_profile.security_awareness.value)
        group_by(results, lambda r: f"{r.attacker_config.strategy.value}_{r.victim_profile.security_awareness.value}")
    """
    groups: dict[str, list[ConversationResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)] += [r]
    return dict(groups)


# ── Convenience groupers ──────────────────────────────────────────────────────


def by_strategy(results: list[ConversationResult]) -> dict[str, list[ConversationResult]]:
    return group_by(results, lambda r: r.attacker_config.strategy.value)


def by_scenario(results: list[ConversationResult]) -> dict[str, list[ConversationResult]]:
    return group_by(results, lambda r: r.attacker_config.scenario.value)


def by_goal(results: list[ConversationResult]) -> dict[str, list[ConversationResult]]:
    return group_by(results, lambda r: r.attacker_config.goal.value)


def by_security_awareness(results: list[ConversationResult]) -> dict[str, list[ConversationResult]]:
    return group_by(results, lambda r: r.victim_profile.security_awareness.value)


def by_communication_style(results: list[ConversationResult]) -> dict[str, list[ConversationResult]]:
    return group_by(results, lambda r: r.victim_profile.communication_style.value)


# ── Trait extraction ──────────────────────────────────────────────────────────


def trait_outcome_table(
    results: list[ConversationResult],
    trait_fn: Callable[[ConversationResult], float],
    trait_name: str = "trait",
) -> list[dict]:
    """Build a flat table of (trait_value, outcome, asr) per conversation.

    Useful for correlation analysis or feeding into external stats tools.

    Args:
        results: Conversations to analyze.
        trait_fn: Extracts a float trait value from a result.
            E.g., lambda r: r.victim_profile.personality.agreeableness
        trait_name: Label for the trait column.

    Returns:
        List of dicts with keys: trait_name, trait_value, outcome, success.
    """
    rows = []
    for r in results:
        rows.append({
            "trait": trait_name,
            "value": trait_fn(r),
            "outcome": r.outcome.value,
            "success": 1 if r.outcome in _SUCCESS_OUTCOMES else 0,
        })
    return rows


def trait_correlation(
    results: list[ConversationResult],
    trait_fn: Callable[[ConversationResult], float],
) -> float:
    """Point-biserial correlation between a continuous trait and binary success.

    Returns correlation coefficient in [-1, 1].
    Positive = higher trait → more compliance.
    Negative = higher trait → less compliance.
    Returns 0.0 if insufficient data or zero variance.
    """
    if len(results) < 3:
        return 0.0

    trait_vals = [trait_fn(r) for r in results]
    success_vals = [1.0 if r.outcome in _SUCCESS_OUTCOMES else 0.0 for r in results]

    n = len(results)
    mean_t = sum(trait_vals) / n
    mean_s = sum(success_vals) / n

    # Check for zero variance
    var_t = sum((t - mean_t) ** 2 for t in trait_vals) / n
    var_s = sum((s - mean_s) ** 2 for s in success_vals) / n

    if var_t == 0 or var_s == 0:
        return 0.0

    cov = sum((t - mean_t) * (s - mean_s) for t, s in zip(trait_vals, success_vals)) / n
    return round(cov / (var_t ** 0.5 * var_s ** 0.5), 4)


# ── Comparative analysis ─────────────────────────────────────────────────────


def compare_conditions(
    results: list[ConversationResult],
    group_fn: Callable[[ConversationResult], str],
) -> list[dict]:
    """Compare ASR and mean scores across conditions defined by group_fn.

    Returns a list of dicts sorted by ASR descending — one per condition.
    Each dict: {condition, n, asr, refusal_rate, suspicion_rate,
                mean_turns, persuasion, coherence, detectability, composite}
    """
    groups = group_by(results, group_fn)
    rows = []

    for condition, group in groups.items():
        rates = outcome_rates(group)
        scores = mean_scores(group)
        rows.append({
            "condition": condition,
            "n": len(group),
            "asr": attack_success_rate(group),
            "refusal_rate": rates.get("refusal", 0.0),
            "suspicion_rate": rates.get("suspicion", 0.0),
            "mean_turns": mean_turns_to_outcome(group),
            **{k: scores.get(k, None) for k in ("persuasion", "coherence", "detectability", "composite")},
        })

    rows.sort(key=lambda r: r["asr"], reverse=True)
    return rows


def strategy_x_trait_matrix(
    results: list[ConversationResult],
    trait_fn: Callable[[ConversationResult], str],
) -> dict[str, dict[str, float]]:
    """ASR matrix: strategy (rows) × trait_category (columns).

    trait_fn should return a categorical string (e.g., "low", "medium", "high").

    Returns:
        {"urgency": {"low": 0.8, "medium": 0.5, "high": 0.2}, ...}
    """
    # Group by (strategy, trait)
    combined = group_by(
        results,
        lambda r: f"{r.attacker_config.strategy.value}||{trait_fn(r)}",
    )

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for key, group in combined.items():
        strategy, trait_val = key.split("||", 1)
        matrix[strategy][trait_val] = attack_success_rate(group)

    return dict(matrix)


# ── Hypothesis testing ────────────────────────────────────────────────────────


def check_ordering(
    results: list[ConversationResult],
    group_fn: Callable[[ConversationResult], str],
    expected_order: list[str],
    metric: str = "asr",
) -> dict:
    """Check whether conditions are ordered as expected by a given metric.

    Args:
        results: Conversations to analyze.
        group_fn: Groups results into conditions.
        expected_order: Condition labels from highest to lowest expected metric.
        metric: "asr" or any key from mean_scores (persuasion, coherence, etc.)

    Returns:
        {
            "expected": ["low", "medium", "high"],
            "actual": ["low", "high", "medium"],
            "values": {"low": 0.8, "medium": 0.3, "high": 0.5},
            "passed": False,
            "concordance": 0.667,  # fraction of pairs in correct order
        }
    """
    groups = group_by(results, group_fn)

    # Compute metric per condition
    values = {}
    for cond in expected_order:
        group = groups.get(cond, [])
        if metric == "asr":
            values[cond] = attack_success_rate(group)
        else:
            scores = mean_scores(group)
            values[cond] = scores.get(metric, 0.0)

    # Sort by metric descending to get actual order
    actual = sorted(expected_order, key=lambda c: values.get(c, 0.0), reverse=True)

    # Concordance: fraction of pairwise comparisons that match expected order
    n_pairs = 0
    concordant = 0
    for i in range(len(expected_order)):
        for j in range(i + 1, len(expected_order)):
            n_pairs += 1
            exp_higher = expected_order[i]
            exp_lower = expected_order[j]
            if values.get(exp_higher, 0.0) >= values.get(exp_lower, 0.0):
                concordant += 1

    concordance = round(concordant / n_pairs, 4) if n_pairs > 0 else 1.0

    return {
        "expected": expected_order,
        "actual": actual,
        "values": values,
        "passed": actual == expected_order,
        "concordance": concordance,
    }


# ── Trait binning ─────────────────────────────────────────────────────────────


def bin_trait(value: float, thresholds: tuple[float, float] = (0.35, 0.65)) -> str:
    """Bin a continuous 0.0–1.0 trait value into low/medium/high.

    Default thresholds (0.35, 0.65) map design levels:
      0.2 → low, 0.5 → medium, 0.8 → high
    """
    if value <= thresholds[0]:
        return "low"
    elif value >= thresholds[1]:
        return "high"
    return "medium"


# ── Multi-trait combination analysis ─────────────────────────────────────────


def rank_trait_combinations(
    results: list[ConversationResult],
    trait_fns: dict[str, Callable[[ConversationResult], str]],
    min_n: int = 3,
) -> list[dict]:
    """Rank all observed trait combinations by ASR.

    Args:
        results: Conversations to analyze.
        trait_fns: Dict mapping trait names to functions that return a
            categorical label (e.g., "low"/"high") for that trait.
        min_n: Minimum observations per combination to include.

    Returns:
        List sorted by ASR descending. Each entry:
        {traits: {name: level, ...}, n, asr, refusal_rate, suspicion_rate}
    """
    # Group by concatenation of all trait labels
    def combo_key(r: ConversationResult) -> str:
        return "||".join(f"{name}={fn(r)}" for name, fn in trait_fns.items())

    groups = group_by(results, combo_key)
    rows = []

    for key, group in groups.items():
        if len(group) < min_n:
            continue
        # Parse the key back into trait dict
        traits = {}
        for part in key.split("||"):
            name, level = part.split("=", 1)
            traits[name] = level
        rates = outcome_rates(group)
        rows.append({
            "traits": traits,
            "n": len(group),
            "asr": attack_success_rate(group),
            "refusal_rate": rates.get("refusal", 0.0),
            "suspicion_rate": rates.get("suspicion", 0.0),
        })

    rows.sort(key=lambda r: r["asr"], reverse=True)
    return rows


def trait_importance_ranking(
    results: list[ConversationResult],
    trait_fns: dict[str, Callable[[ConversationResult], float]],
) -> list[dict]:
    """Rank single traits by how strongly they predict compliance.

    For each trait, computes:
    - Point-biserial correlation with binary success
    - ASR range: max group ASR minus min group ASR (when binned to low/med/high)

    Returns list sorted by abs(correlation) descending:
        [{trait, correlation, direction, asr_range}, ...]
    """
    rows = []
    for name, fn in trait_fns.items():
        corr = trait_correlation(results, fn)

        # Compute ASR range by binning
        binned_groups = group_by(results, lambda r, _fn=fn: bin_trait(_fn(r)))
        group_asrs = [attack_success_rate(g) for g in binned_groups.values() if len(g) >= 2]

        asr_range = (max(group_asrs) - min(group_asrs)) if len(group_asrs) >= 2 else 0.0

        direction = "+" if corr > 0 else "-" if corr < 0 else "0"
        rows.append({
            "trait": name,
            "correlation": corr,
            "direction": direction,
            "asr_range": round(asr_range, 4),
        })

    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return rows


def trait_interaction_effects(
    results: list[ConversationResult],
    trait_a_fn: Callable[[ConversationResult], str],
    trait_b_fn: Callable[[ConversationResult], str],
    trait_a_name: str = "trait_a",
    trait_b_name: str = "trait_b",
) -> dict:
    """Detect whether two traits interact beyond their individual effects.

    Computes:
    - ASR for each cell in the A×B matrix
    - Main effects for each trait level
    - Interaction strength: max deviation from additive model

    A large interaction_strength means the combination matters beyond
    the sum of individual trait effects.

    Returns:
        {
            "trait_a": str, "trait_b": str,
            "matrix": {a_val: {b_val: asr}},
            "main_effect_a": {val: asr},
            "main_effect_b": {val: asr},
            "interaction_strength": float,
        }
    """
    # Compute the A×B matrix
    combined = group_by(
        results,
        lambda r: f"{trait_a_fn(r)}||{trait_b_fn(r)}",
    )

    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for key, group in combined.items():
        a_val, b_val = key.split("||", 1)
        matrix[a_val][b_val] = attack_success_rate(group)

    # Main effects
    main_a = {}
    groups_a = group_by(results, trait_a_fn)
    for val, group in groups_a.items():
        main_a[val] = attack_success_rate(group)

    main_b = {}
    groups_b = group_by(results, trait_b_fn)
    for val, group in groups_b.items():
        main_b[val] = attack_success_rate(group)

    # Overall ASR (baseline for additive model)
    overall_asr = attack_success_rate(results)

    # Interaction strength: max |observed - expected_additive| across cells
    # Expected additive: overall + (main_a[a] - overall) + (main_b[b] - overall)
    max_deviation = 0.0
    for a_val, b_dict in matrix.items():
        for b_val, observed in b_dict.items():
            expected = overall_asr + (main_a.get(a_val, overall_asr) - overall_asr) + (main_b.get(b_val, overall_asr) - overall_asr)
            deviation = abs(observed - expected)
            max_deviation = max(max_deviation, deviation)

    return {
        "trait_a": trait_a_name,
        "trait_b": trait_b_name,
        "matrix": dict(matrix),
        "main_effect_a": main_a,
        "main_effect_b": main_b,
        "interaction_strength": round(max_deviation, 4),
    }


# ── Full benchmark report ────────────────────────────────────────────────────


def compute_benchmark(results: list[ConversationResult]) -> dict:
    """Compute a full benchmark report from any set of results.

    Returns a dict with all metrics. No opinions — just numbers.
    Caller decides what to do with them.
    """
    return {
        "total_conversations": len(results),
        "overall_asr": attack_success_rate(results),
        "outcome_distribution": outcome_distribution(results),
        "outcome_rates": outcome_rates(results),
        "mean_turns_to_outcome": mean_turns_to_outcome(results),
        "mean_scores": mean_scores(results),
        "score_std": score_std(results),
        "by_strategy": compare_conditions(results, lambda r: r.attacker_config.strategy.value),
        "by_scenario": compare_conditions(results, lambda r: r.attacker_config.scenario.value),
        "by_goal": compare_conditions(results, lambda r: r.attacker_config.goal.value),
        "by_security_awareness": compare_conditions(
            results, lambda r: r.victim_profile.security_awareness.value,
        ),
        "trait_correlations": {
            "agreeableness": trait_correlation(
                results, lambda r: r.victim_profile.personality.agreeableness,
            ),
            "conscientiousness": trait_correlation(
                results, lambda r: r.victim_profile.personality.conscientiousness,
            ),
            "extraversion": trait_correlation(
                results, lambda r: r.victim_profile.personality.extraversion,
            ),
            "neuroticism": trait_correlation(
                results, lambda r: r.victim_profile.personality.neuroticism,
            ),
            "openness": trait_correlation(
                results, lambda r: r.victim_profile.personality.openness,
            ),
            "tech_proficiency": trait_correlation(
                results, lambda r: r.victim_profile.tech_proficiency,
            ),
            "impulsivity": trait_correlation(
                results, lambda r: r.victim_profile.impulsivity,
            ),
        },
        "strategy_x_security_awareness": strategy_x_trait_matrix(
            results, lambda r: r.victim_profile.security_awareness.value,
        ),
    }
