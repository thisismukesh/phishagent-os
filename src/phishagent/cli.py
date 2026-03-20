"""Click-based CLI entry point for PhishAgent-OS.

Three main commands: run (single conversation), experiment (batch), status (system check).
"""

import sys

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from phishagent.attacker_agent import AttackerAgent
from phishagent.config import AppConfig, ConversationConfig, load_config
from phishagent.conversation_engine import ConversationEngine
from phishagent.llm_client import OllamaClient, detect_cuda
from phishagent.victim_agent import VictimAgent
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationResult,
    ExperimentConfig,
    FactorialSpec,
    VictimProfile,
)
from phishagent.profile_manager import ProfileManager
from phishagent.scoring import ConversationScorer
from phishagent.experiment_runner import ExperimentRunner
from phishagent.utils import format_conversation_for_display, get_logger, safe_json_dump

logger = get_logger(__name__)
console = Console()


@click.group(invoke_without_command=True)
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.pass_context
def main(ctx, config_path):
    """PhishAgent-OS: Open-Source Social Engineering Simulation Pipeline

    Run without a subcommand to launch the interactive guided mode.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)

    # Launch interactive mode when no subcommand is given
    if ctx.invoked_subcommand is None:
        from phishagent.interactive import run_interactive_app
        run_interactive_app(ctx.obj["config"])


@main.command()
@click.option("--profile", required=True, help="Path to victim profile YAML")
@click.option(
    "--strategy",
    type=click.Choice([s.value for s in AttackStrategy]),
    required=True,
)
@click.option(
    "--scenario",
    type=click.Choice([s.value for s in AttackerScenario]),
    required=True,
)
@click.option(
    "--goal",
    type=click.Choice([g.value for g in AttackGoal]),
    required=True,
)
@click.option("--model", default=None, help="Override model name")
@click.option("--turns", default=None, type=int, help="Override max turns")
@click.option("--output", default=None, help="Output directory")
@click.pass_context
def run(ctx, profile, strategy, scenario, goal, model, turns, output):
    """Run a single social engineering conversation simulation."""
    app_config: AppConfig = ctx.obj["config"]

    # Load profile
    pm = ProfileManager()
    try:
        victim_profile = pm.load_profile(profile)
    except Exception as e:
        console.print(f"[red]Error loading profile: {e}[/red]")
        sys.exit(1)

    # Build attacker config
    model_name = model or app_config.model.name
    max_turns = turns or app_config.conversation.max_turns

    attacker_config = AttackerConfig(
        goal=AttackGoal(goal),
        strategy=AttackStrategy(strategy),
        scenario=AttackerScenario(scenario),
        max_turns=max_turns,
    )

    # Create LLM client
    llm = OllamaClient(
        base_url=app_config.model.ollama_url,
        timeout=app_config.model.timeout_seconds,
        num_gpu=app_config.model.num_gpu,
    )

    if not llm.is_available():
        console.print("[red]Ollama is not reachable. Run 'ollama serve' first.[/red]")
        sys.exit(1)

    # Print header
    p = victim_profile.personality
    console.print("[bold][PhishAgent-OS] Starting conversation simulation[/bold]")
    console.print(
        f"[PhishAgent-OS] Model: {model_name} | Strategy: {strategy} | Scenario: {scenario}"
    )
    console.print(
        f"[PhishAgent-OS] Victim: {victim_profile.name} "
        f"(A={p.agreeableness}, C={p.conscientiousness}, E={p.extraversion}, "
        f"N={p.neuroticism}, O={p.openness})"
    )
    console.print("─" * 50)

    # Create agents and engine
    attacker = AttackerAgent(attacker_config, llm, model_name)
    victim = VictimAgent(victim_profile, llm, model_name)
    conv_config = ConversationConfig(
        max_turns=max_turns,
        early_termination=app_config.conversation.early_termination,
    )
    engine = ConversationEngine(attacker, victim, conv_config)

    # Run conversation
    with console.status("[bold green]Running conversation..."):
        result = engine.run(victim_profile, attacker_config)

    # Display conversation
    console.print(format_conversation_for_display(result.messages))
    console.print("─" * 50)

    # Score
    console.print("[bold]Scoring conversation...[/bold]")
    scorer = ConversationScorer(llm, app_config.scoring)
    try:
        scores = scorer.score(result)
        result.scores = scores

        console.print(
            f"[PhishAgent-OS] Outcome: {result.outcome.value.upper()} "
            f"(turn {len([m for m in result.messages if m.role == 'victim'])})"
        )
        console.print(
            f"[PhishAgent-OS] Scores: Persuasion={scores.persuasion:.2f} | "
            f"Coherence={scores.coherence:.2f} | Detectability={scores.detectability:.2f}"
        )
        console.print(f"[PhishAgent-OS] Composite: {scores.composite:.2f}")
    except Exception as e:
        console.print(f"[yellow]Scoring failed: {e}[/yellow]")

    # Save output
    output_dir = output or f"{app_config.output.base_dir}/conversations"
    output_path = f"{output_dir}/conv_{result.conversation_id}.json"
    safe_json_dump(result, output_path)
    console.print(f"[PhishAgent-OS] Saved to: {output_path}")


@main.command()
@click.option("--experiment-config", required=True, help="Path to factorial experiment YAML")
@click.option("--model", default=None, help="Override model name")
@click.option("--repetitions", default=None, type=int, help="Override repetitions")
@click.option("--output", default=None, help="Output directory")
@click.pass_context
def experiment(ctx, experiment_config, model, repetitions, output):
    """Run a batch factorial experiment."""
    app_config: AppConfig = ctx.obj["config"]

    # Load experiment config from YAML
    try:
        with open(experiment_config) as f:
            exp_data = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading experiment config: {e}[/red]")
        sys.exit(1)

    if not exp_data or not isinstance(exp_data, dict):
        console.print("[red]Experiment config file is empty or invalid.[/red]")
        sys.exit(1)

    for required_key in ("experiment_id", "base_profile"):
        if required_key not in exp_data:
            console.print(f"[red]Experiment config missing required key: '{required_key}'[/red]")
            sys.exit(1)

    # Parse factorial specification
    model_name = model or exp_data.get("model_name", app_config.model.name)
    reps = repetitions or exp_data.get("repetitions", 3)

    # Generate profiles from factorial spec
    pm = ProfileManager()
    try:
        base_profile = VictimProfile(**exp_data["base_profile"])
    except Exception as e:
        console.print(f"[red]Invalid base_profile in experiment config: {e}[/red]")
        sys.exit(1)
    vary = exp_data.get("vary", {})

    if vary:
        spec = FactorialSpec(base_profile=base_profile, vary=vary)
        profiles = pm.generate_factorial_profiles(spec)
    else:
        profiles = [base_profile]

    # Parse attacker configs
    attacker_configs = [
        AttackerConfig(**ac) for ac in exp_data.get("attacker_configs", [])
    ]

    if not attacker_configs:
        console.print("[red]No attacker configs found in experiment file.[/red]")
        sys.exit(1)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output or f"{app_config.output.base_dir}/experiments/{exp_data['experiment_id']}_{timestamp}"

    exp_config = ExperimentConfig(
        experiment_id=exp_data["experiment_id"],
        description=exp_data.get("description", ""),
        model_name=model_name,
        profiles=profiles,
        attacker_configs=attacker_configs,
        repetitions=reps,
        output_dir=output_dir,
    )

    total = len(profiles) * len(attacker_configs) * reps
    console.print(f"[bold][PhishAgent-OS] Starting batch experiment: {exp_config.experiment_id}[/bold]")
    console.print(
        f"[PhishAgent-OS] {len(profiles)} profiles x {len(attacker_configs)} configs x "
        f"{reps} reps = {total} conversations"
    )
    console.print(f"[PhishAgent-OS] Model: {model_name}")

    # Check Ollama
    llm = OllamaClient(
        base_url=app_config.model.ollama_url,
        timeout=app_config.model.timeout_seconds,
        num_gpu=app_config.model.num_gpu,
    )
    if not llm.is_available():
        console.print("[red]Ollama is not reachable. Run 'ollama serve' first.[/red]")
        sys.exit(1)

    # Run experiment with progress bar
    runner = ExperimentRunner(app_config, llm)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running conversations...", total=total)

        def on_progress(completed, total_count, last_result):
            progress.update(task, completed=completed)
            if last_result:
                outcome = last_result.outcome.value
                progress.update(
                    task,
                    description=f"Running conversations... (last: {outcome})",
                )

        result = runner.run(exp_config, progress_callback=on_progress)

    # Export CSV
    csv_path = f"{output_dir}/{exp_config.experiment_id}.csv"
    runner.export_csv(result, csv_path)

    # Summary
    console.print("─" * 50)
    console.print(f"[bold green]Experiment complete![/bold green]")
    console.print(f"  Completed: {result.completed_conversations}/{result.total_conversations}")
    console.print(f"  Failed: {result.failed_conversations}")
    console.print(f"  CSV: {csv_path}")
    console.print(f"  Conversations: {output_dir}/conversations/")


@main.command()
@click.pass_context
def status(ctx):
    """Check system status (Ollama connectivity, available models)."""
    app_config: AppConfig = ctx.obj["config"]
    client = OllamaClient(
        base_url=app_config.model.ollama_url,
        num_gpu=app_config.model.num_gpu,
    )

    console.print("[bold]PhishAgent-OS System Status[/bold]")
    console.print(f"  Config model: {app_config.model.name}")
    console.print(f"  Ollama URL: {app_config.model.ollama_url}")

    # GPU status
    cuda = client.cuda_info
    if cuda["available"]:
        console.print(f"  [green]✓ CUDA GPU detected ({cuda['count']} device(s))[/green]")
        for dev in cuda["devices"]:
            console.print(f"    • {dev}")
        console.print(f"  GPU layers (num_gpu): {client._num_gpu}")
    else:
        console.print("  [yellow]No CUDA GPU detected — CPU-only mode[/yellow]")
        console.print("  [dim]Set PHISHAGENT_NUM_GPU=0 to suppress GPU offload attempts[/dim]")

    if client.is_available():
        console.print("  [green]✓ Ollama is running[/green]")
        models = client.list_models()
        if models:
            console.print(f"  Available models ({len(models)}):")
            for m in models:
                console.print(f"    • {m}")
        else:
            console.print("  [yellow]No models found. Run 'ollama pull mistral:7b'[/yellow]")
    else:
        console.print("  [red]✗ Ollama is not reachable[/red]")
        console.print("  [dim]Run 'ollama serve' to start Ollama[/dim]")


@main.command()
@click.pass_context
def interactive(ctx):
    """Launch the guided interactive terminal mode (same as running with no subcommand)."""
    from phishagent.interactive import run_interactive_app
    run_interactive_app(ctx.obj["config"])


@main.command()
@click.option("--suite", default="config/benchmark_suite.yaml", help="Path to benchmark suite YAML")
@click.option("--model", default=None, help="Override model name")
@click.option("--output", default=None, help="Output directory")
@click.option("--skip-run", is_flag=True, help="Skip running experiments, analyze existing results")
@click.option("--results-dir", default=None, help="Directory of existing results (with --skip-run)")
@click.pass_context
def benchmark(ctx, suite, model, output, skip_run, results_dir):
    """Run a benchmark validation suite and report metrics."""
    import json
    from pathlib import Path

    from phishagent.benchmark import (
        attack_success_rate,
        bin_trait,
        check_ordering,
        compare_conditions,
        compute_benchmark,
        group_by,
        rank_trait_combinations,
        trait_correlation,
        trait_importance_ranking,
        trait_interaction_effects,
    )

    app_config: AppConfig = ctx.obj["config"]

    # Load suite config
    try:
        with open(suite) as f:
            suite_data = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading benchmark suite: {e}[/red]")
        sys.exit(1)

    experiment_id = suite_data.get("experiment_id", "benchmark")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output or f"{app_config.output.base_dir}/benchmarks/{experiment_id}_{timestamp}"

    if skip_run:
        # Load existing results
        load_dir = results_dir or f"{output_dir}/conversations"
        console.print(f"[bold]Loading existing results from {load_dir}...[/bold]")
        results = _load_results_from_dir(load_dir)
        if not results:
            console.print("[red]No results found. Run the benchmark first (without --skip-run).[/red]")
            sys.exit(1)
        console.print(f"  Loaded {len(results)} conversations")
    else:
        # Run the experiment
        model_name = model or suite_data.get("model_name", app_config.model.name)
        reps = suite_data.get("repetitions", 5)

        # Parse profiles
        pm = ProfileManager()
        base_profile = VictimProfile(**suite_data["base_profile"])
        vary = suite_data.get("vary", {})

        if vary:
            spec = FactorialSpec(base_profile=base_profile, vary=vary)
            profiles = pm.generate_factorial_profiles(spec)
        else:
            profiles = [base_profile]

        # Parse attacker configs
        attacker_configs = [
            AttackerConfig(**ac) for ac in suite_data.get("attacker_configs", [])
        ]
        if not attacker_configs:
            console.print("[red]No attacker configs in benchmark suite.[/red]")
            sys.exit(1)

        total = len(profiles) * len(attacker_configs) * reps

        exp_config = ExperimentConfig(
            experiment_id=experiment_id,
            description=suite_data.get("description", ""),
            model_name=model_name,
            profiles=profiles,
            attacker_configs=attacker_configs,
            repetitions=reps,
            output_dir=output_dir,
        )

        # Check Ollama
        llm = OllamaClient(
            base_url=app_config.model.ollama_url,
            timeout=app_config.model.timeout_seconds,
            num_gpu=app_config.model.num_gpu,
        )
        if not llm.is_available():
            console.print("[red]Ollama is not reachable. Run 'ollama serve' first.[/red]")
            sys.exit(1)

        console.print(f"[bold][PhishAgent-OS] Benchmark: {experiment_id}[/bold]")
        console.print(
            f"  {len(profiles)} profiles x {len(attacker_configs)} strategies x "
            f"{reps} reps = {total} conversations"
        )
        console.print(f"  Model: {model_name} | Judge: {app_config.scoring.judge_model}")

        runner = ExperimentRunner(app_config, llm)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=total)

            def on_progress(completed, total_count, last_result):
                progress.update(task, completed=completed)

            exp_result = runner.run(exp_config, progress_callback=on_progress)

        # Export CSV
        csv_path = f"{output_dir}/{experiment_id}.csv"
        runner.export_csv(exp_result, csv_path)
        results = exp_result.results

    # ── Compute and display benchmark report ──────────────────────────────
    console.print()
    console.print("[bold]Benchmark Report[/bold]")
    console.print("=" * 60)

    report = compute_benchmark(results)

    # Overview
    console.print(f"\n[bold]Overview[/bold]")
    console.print(f"  Conversations: {report['total_conversations']}")
    console.print(f"  Overall ASR:   {report['overall_asr']:.1%}")
    console.print(f"  Mean turns:    {report['mean_turns_to_outcome']}")

    # Outcome distribution
    console.print(f"\n[bold]Outcome Distribution[/bold]")
    for outcome, rate in sorted(report["outcome_rates"].items(), key=lambda x: -x[1]):
        count = report["outcome_distribution"][outcome]
        console.print(f"  {outcome:<20s} {rate:>6.1%}  ({count})")

    # Judge scores
    if report["mean_scores"]:
        console.print(f"\n[bold]Mean Judge Scores[/bold]")
        stds = report["score_std"]
        for dim in ("persuasion", "coherence", "detectability", "composite"):
            mean = report["mean_scores"][dim]
            std = stds.get(dim, 0.0)
            console.print(f"  {dim:<16s} {mean:.3f} +/- {std:.3f}")

    # By strategy
    console.print(f"\n[bold]By Strategy[/bold]")
    _print_condition_table(report["by_strategy"])

    # By security awareness
    console.print(f"\n[bold]By Security Awareness[/bold]")
    _print_condition_table(report["by_security_awareness"])

    # Trait correlations
    console.print(f"\n[bold]Trait-Vulnerability Correlations[/bold]")
    console.print("  (positive = higher trait → more compliance)")
    for trait, corr in sorted(
        report["trait_correlations"].items(), key=lambda x: -abs(x[1])
    ):
        if corr > 0:
            console.print(f"  {trait:<22s} [green]+{abs(corr):.3f}[/green]")
        elif corr < 0:
            console.print(f"  {trait:<22s} [red]-{abs(corr):.3f}[/red]")
        else:
            console.print(f"  {trait:<22s}  {abs(corr):.3f}")

    # Strategy x Security Awareness matrix
    matrix = report["strategy_x_security_awareness"]
    if matrix:
        console.print(f"\n[bold]Strategy x Security Awareness (ASR)[/bold]")
        sa_levels = ["low", "medium", "high"]
        header = f"  {'strategy':<16s}" + "".join(f"{sa:>10s}" for sa in sa_levels)
        console.print(header)
        for strategy in sorted(matrix.keys()):
            row = f"  {strategy:<16s}"
            for sa in sa_levels:
                val = matrix[strategy].get(sa)
                row += f"{val:>9.1%} " if val is not None else f"{'n/a':>10s}"
            console.print(row)

    # ── Trait combination analysis ────────────────────────────────────────
    # Build trait extraction functions for the 4 benchmark factors
    _trait_fns_continuous = {
        "agreeableness": lambda r: r.victim_profile.personality.agreeableness,
        "conscientiousness": lambda r: r.victim_profile.personality.conscientiousness,
        "impulsivity": lambda r: r.victim_profile.impulsivity,
    }
    _trait_fns_categorical = {
        "agreeableness": lambda r: bin_trait(r.victim_profile.personality.agreeableness),
        "conscientiousness": lambda r: bin_trait(r.victim_profile.personality.conscientiousness),
        "security_awareness": lambda r: r.victim_profile.security_awareness.value,
        "impulsivity": lambda r: bin_trait(r.victim_profile.impulsivity),
    }

    # Trait importance ranking
    importance = trait_importance_ranking(results, _trait_fns_continuous)
    if importance:
        console.print(f"\n[bold]Trait Importance Ranking[/bold]")
        console.print("  (which single trait matters most for predicting compliance)")
        console.print(f"  {'trait':<22s} {'corr':>7s} {'dir':>4s} {'ASR range':>10s}")
        for row in importance:
            d = "[green]+[/green]" if row["direction"] == "+" else "[red]-[/red]" if row["direction"] == "-" else " "
            console.print(
                f"  {row['trait']:<22s} {abs(row['correlation']):>6.3f} {d}   {row['asr_range']:>9.1%}"
            )

    # Top compliance / resistance combinations
    combos = rank_trait_combinations(results, _trait_fns_categorical, min_n=3)
    if combos:
        n_show = min(5, len(combos))
        console.print(f"\n[bold]Top {n_show} Most Vulnerable Trait Combinations[/bold]")
        _print_combo_table(combos[:n_show])

        console.print(f"\n[bold]Top {n_show} Most Resistant Trait Combinations[/bold]")
        _print_combo_table(combos[-n_show:][::-1])

    # Interaction effects for all trait pairs
    _pair_fns = list(_trait_fns_categorical.items())
    interactions = []
    for i in range(len(_pair_fns)):
        for j in range(i + 1, len(_pair_fns)):
            name_a, fn_a = _pair_fns[i]
            name_b, fn_b = _pair_fns[j]
            ie = trait_interaction_effects(results, fn_a, fn_b, name_a, name_b)
            interactions.append(ie)

    interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

    if interactions:
        n_show = min(3, len(interactions))
        console.print(f"\n[bold]Strongest Trait Interactions[/bold]")
        console.print("  (pairs where the combination effect differs from sum of parts)")
        for ie in interactions[:n_show]:
            console.print(
                f"\n  {ie['trait_a']} x {ie['trait_b']}  "
                f"(interaction strength: {ie['interaction_strength']:.1%})"
            )
            # Print the matrix
            a_vals = sorted(ie["matrix"].keys())
            b_vals = sorted(set(v for d in ie["matrix"].values() for v in d.keys()))
            header = f"    {'':>12s}" + "".join(f"{bv:>10s}" for bv in b_vals)
            console.print(header)
            for av in a_vals:
                row = f"    {av:>12s}"
                for bv in b_vals:
                    val = ie["matrix"].get(av, {}).get(bv)
                    row += f"{val:>9.1%} " if val is not None else f"{'n/a':>10s}"
                console.print(row)

    # ── Hypothesis validation ─────────────────────────────────────────────
    hypotheses = suite_data.get("hypotheses", [])
    if hypotheses:
        console.print(f"\n[bold]Hypothesis Validation[/bold]")
        console.print("-" * 60)

        passed_count = 0
        for hyp in hypotheses:
            name = hyp["name"]
            expected = hyp["expected_order"]
            metric = hyp.get("metric", "asr")
            group_key = hyp["group_by"]
            filter_spec = hyp.get("filter")

            # Apply filter if specified
            filtered = results
            if filter_spec:
                for field, value in filter_spec.items():
                    filtered = [
                        r for r in filtered
                        if _get_result_field(r, field) == value
                    ]

            # Build group function from the group_by string
            group_fn = _make_group_fn(group_key)
            result = check_ordering(filtered, group_fn, expected, metric)

            status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
            if result["passed"]:
                passed_count += 1

            console.print(f"\n  {status} {name}")
            console.print(f"    {hyp.get('description', '')}")
            console.print(f"    Expected: {' > '.join(expected)}")
            console.print(f"    Actual:   {' > '.join(result['actual'])}")
            vals_str = ", ".join(f"{k}={v:.1%}" for k, v in result["values"].items())
            console.print(f"    Values:   {vals_str}")
            console.print(f"    Concordance: {result['concordance']:.0%}")

        console.print(f"\n  Result: {passed_count}/{len(hypotheses)} hypotheses passed")

    # Save report JSON
    report_path = f"{output_dir}/benchmark_report.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n  Full report saved to: {report_path}")


def _print_condition_table(rows: list[dict]) -> None:
    """Print a compare_conditions table to the console."""
    if not rows:
        console.print("  (no data)")
        return
    header = f"  {'condition':<16s} {'n':>4s} {'ASR':>7s} {'refusal':>8s} {'suspicion':>9s} {'turns':>6s}"
    has_scores = rows[0].get("persuasion") is not None
    if has_scores:
        header += f" {'pers':>6s} {'coher':>6s} {'detect':>6s}"
    console.print(header)
    for r in rows:
        line = (
            f"  {r['condition']:<16s} {r['n']:>4d} {r['asr']:>6.1%} "
            f"{r['refusal_rate']:>7.1%} {r['suspicion_rate']:>8.1%} "
            f"{r['mean_turns']:>6.1f}"
        )
        if has_scores and r.get("persuasion") is not None:
            line += f" {r['persuasion']:>6.3f} {r['coherence']:>6.3f} {r['detectability']:>6.3f}"
        console.print(line)


def _print_combo_table(combos: list[dict]) -> None:
    """Print a rank_trait_combinations table to the console."""
    if not combos:
        console.print("  (no data)")
        return
    for i, combo in enumerate(combos, 1):
        traits_str = ", ".join(f"{k}={v}" for k, v in combo["traits"].items())
        console.print(
            f"  {i}. [{traits_str}]  "
            f"ASR={combo['asr']:.1%}  "
            f"refusal={combo['refusal_rate']:.1%}  "
            f"suspicion={combo['suspicion_rate']:.1%}  "
            f"(n={combo['n']})"
        )


def _get_result_field(r: ConversationResult, field: str) -> str:
    """Extract a field value from a ConversationResult by dot-path string."""
    if field == "security_awareness":
        return r.victim_profile.security_awareness.value
    elif field == "communication_style":
        return r.victim_profile.communication_style.value
    elif field == "strategy":
        return r.attacker_config.strategy.value
    elif field == "scenario":
        return r.attacker_config.scenario.value
    elif field == "goal":
        return r.attacker_config.goal.value
    elif field.startswith("personality."):
        trait = field.split(".", 1)[1]
        return str(getattr(r.victim_profile.personality, trait))
    else:
        return str(getattr(r.victim_profile, field, ""))


def _make_group_fn(group_key: str):
    """Build a grouping lambda from a field path string."""
    return lambda r: _get_result_field(r, group_key)


def _load_results_from_dir(dir_path: str) -> list[ConversationResult]:
    """Load ConversationResult objects from a directory of JSON files."""
    import json
    from pathlib import Path

    results = []
    p = Path(dir_path)
    if not p.exists():
        return results

    for f in sorted(p.glob("conv_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append(ConversationResult(**data))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    return results


if __name__ == "__main__":
    main()
