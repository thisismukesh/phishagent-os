#!/usr/bin/env python3
"""Analysis script for PhishAgent-OS experiment results.

Reads experiment CSV and produces summary statistics:
- Descriptive statistics by personality trait level, strategy, scenario
- Correlation matrix: Big Five traits vs. persuasion score
- Top predictors of attack success
- Comparison across attack strategies

Usage:
    python scripts/analyze.py output/experiments/personality_sweep_001.csv
"""

import csv
import math
import statistics
import sys
from collections import defaultdict


def load_results(path: str) -> list[dict]:
    """Load experiment results from CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # Convert numeric fields
    numeric_fields = [
        "openness", "conscientiousness", "extraversion", "agreeableness",
        "neuroticism", "tech_proficiency", "impulsivity", "num_turns",
        "persuasion", "coherence", "detectability", "composite",
        "total_tokens", "duration_seconds",
    ]
    for row in rows:
        for field in numeric_fields:
            if row.get(field) and row[field] != "":
                try:
                    row[field] = float(row[field])
                except ValueError:
                    row[field] = None
            else:
                row[field] = None
    return rows


def safe_mean(values: list[float]) -> float:
    """Mean that handles empty lists."""
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    """Standard deviation that handles lists with <2 items."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def pearson_r(x: list[float], y: list[float]) -> float:
    """Simple Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.stdev(x), statistics.stdev(y)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)


def p_value_approx(r: float, n: int) -> float:
    """Approximate two-tailed p-value for Pearson r using t-distribution approximation."""
    if n < 4 or abs(r) >= 1.0:
        return 1.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # Approximate using normal distribution for large n
    df = n - 2
    p = 2 * (1 - _normal_cdf(abs(t) * math.sqrt(df / (df + t * t))))
    return min(1.0, max(0.0, p))


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def format_p(p: float) -> str:
    """Format p-value with significance stars."""
    if p < 0.001:
        return "p<0.001 ***"
    elif p < 0.01:
        return f"p={p:.3f} **"
    elif p < 0.05:
        return f"p={p:.2f} *"
    else:
        return f"p={p:.2f}"


def group_by(rows: list[dict], field: str) -> dict[str, list[dict]]:
    """Group rows by a field value."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return dict(groups)


def analyze(path: str) -> None:
    """Run full analysis on experiment CSV."""
    rows = load_results(path)

    if not rows:
        print("No data found in CSV.")
        return

    # Filter rows with valid persuasion scores
    scored = [r for r in rows if r.get("persuasion") is not None]
    total = len(rows)
    completed = len(scored)
    failed = total - completed

    # Outcome counts
    outcomes = defaultdict(int)
    for r in rows:
        outcomes[r.get("outcome", "unknown")] += 1

    # ── Header ──
    exp_name = path.rsplit("/", 1)[-1].replace(".csv", "")
    print("=" * 60)
    print(f"PhishAgent-OS Experiment Analysis: {exp_name}")
    print("=" * 60)
    print()

    # ── Overview ──
    print("OVERVIEW")
    print(f"  Total conversations: {total}")
    print(f"  Completed: {completed} | Failed: {failed}")
    compliance_rate = outcomes.get("compliance", 0) / total * 100 if total else 0
    refusal_rate = outcomes.get("refusal", 0) / total * 100 if total else 0
    suspicion_rate = outcomes.get("suspicion", 0) / total * 100 if total else 0
    print(
        f"  Compliance rate: {compliance_rate:.1f}% | "
        f"Refusal rate: {refusal_rate:.1f}% | "
        f"Suspicion rate: {suspicion_rate:.1f}%"
    )
    print()

    if not scored:
        print("No scored conversations to analyze.")
        return

    # ── Persuasion by personality trait levels ──
    trait_fields = {
        "agreeableness": "PERSUASION BY AGREEABLENESS LEVEL",
        "conscientiousness": "PERSUASION BY CONSCIENTIOUSNESS LEVEL",
        "neuroticism": "PERSUASION BY NEUROTICISM LEVEL",
        "openness": "PERSUASION BY OPENNESS LEVEL",
        "extraversion": "PERSUASION BY EXTRAVERSION LEVEL",
    }

    for trait, header in trait_fields.items():
        # Group by unique trait values
        trait_groups = defaultdict(list)
        for r in scored:
            if r.get(trait) is not None:
                trait_groups[r[trait]].append(r["persuasion"])

        if len(trait_groups) > 1:
            print(header)
            for level in sorted(trait_groups.keys()):
                values = trait_groups[level]
                m = safe_mean(values)
                sd = safe_stdev(values)
                print(f"  {trait.capitalize()}={level}: M={m:.2f}, SD={sd:.2f} (n={len(values)})")
            print()

    # ── Persuasion by security awareness ──
    sa_groups = group_by(scored, "security_awareness")
    if sa_groups:
        print("PERSUASION BY SECURITY AWARENESS")
        for level in ["low", "medium", "high"]:
            if level in sa_groups:
                values = [r["persuasion"] for r in sa_groups[level] if r["persuasion"] is not None]
                if values:
                    m = safe_mean(values)
                    sd = safe_stdev(values)
                    print(f"  {level.capitalize():8s}: M={m:.2f}, SD={sd:.2f} (n={len(values)})")
        print()

    # ── Persuasion by attack strategy ──
    strat_groups = group_by(scored, "strategy")
    if strat_groups:
        print("PERSUASION BY ATTACK STRATEGY")
        for strat, group in sorted(strat_groups.items()):
            values = [r["persuasion"] for r in group if r["persuasion"] is not None]
            if values:
                m = safe_mean(values)
                sd = safe_stdev(values)
                print(f"  {strat.capitalize():15s}: M={m:.2f}, SD={sd:.2f} (n={len(values)})")
        print()

    # ── Correlation matrix ──
    big_five = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    print("CORRELATION MATRIX (Big Five vs. Persuasion)")
    correlations = {}
    for trait in big_five:
        x = [r[trait] for r in scored if r.get(trait) is not None and r.get("persuasion") is not None]
        y = [r["persuasion"] for r in scored if r.get(trait) is not None and r.get("persuasion") is not None]
        if len(x) >= 3:
            r = pearson_r(x, y)
            p = p_value_approx(r, len(x))
            correlations[trait] = (r, p, len(x))
            print(f"  {trait.capitalize():20s}: r={r:+.2f}, {format_p(p)}")
        else:
            print(f"  {trait.capitalize():20s}: insufficient data")
    print()

    # ── Top predictors ──
    print("TOP PREDICTORS (by absolute correlation with persuasion)")
    sorted_corr = sorted(correlations.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
    for i, (trait, (r, p, n)) in enumerate(sorted_corr, 1):
        print(f"  {i}. {trait.capitalize()} (r={r:+.2f}, {format_p(p)})")
    print()

    # ── Mean scores summary ──
    print("OVERALL SCORE SUMMARY")
    for metric in ["persuasion", "coherence", "detectability", "composite"]:
        values = [r[metric] for r in scored if r.get(metric) is not None]
        if values:
            print(f"  {metric.capitalize():15s}: M={safe_mean(values):.2f}, SD={safe_stdev(values):.2f}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze.py <path_to_experiment_csv>")
        sys.exit(1)
    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
