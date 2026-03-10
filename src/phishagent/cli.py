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
from phishagent.llm_client import OllamaClient
from phishagent.victim_agent import VictimAgent
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ExperimentConfig,
    FactorialSpec,
    VictimProfile,
)
from phishagent.profile_manager import ProfileManager
from phishagent.scoring import ConversationScorer
from phishagent.experiment_runner import ExperimentRunner
from phishagent.utils import format_conversation_for_display, safe_json_dump

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

    output_dir = output or f"{app_config.output.base_dir}/experiments/{exp_data['experiment_id']}"

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
    client = OllamaClient(base_url=app_config.model.ollama_url)

    console.print("[bold]PhishAgent-OS System Status[/bold]")
    console.print(f"  Config model: {app_config.model.name}")
    console.print(f"  Ollama URL: {app_config.model.ollama_url}")

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


if __name__ == "__main__":
    main()
