"""Interactive terminal mode for PhishAgent-OS.

Provides a guided, menu-driven interface so users can run simulations without
constructing long command-line flags. Calls into the same backend services used
by the standard CLI commands — no logic is duplicated here.

Entry point: run_interactive_app(app_config)
"""

import json
import statistics
from pathlib import Path
from typing import Optional

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from phishagent.attacker_agent import AttackerAgent
from phishagent.config import AppConfig, ConversationConfig
from phishagent.conversation_engine import ConversationEngine
from phishagent.experiment_runner import ExperimentRunner
from phishagent.llm_client import OllamaClient
from phishagent.models import (
    AttackGoal,
    AttackStrategy,
    AttackerConfig,
    AttackerScenario,
    CommunicationStyle,
    ConversationResult,
    ExperimentConfig,
    FactorialSpec,
    Message,
    PersonalityTraits,
    SecurityAwareness,
    VictimProfile,
)
from phishagent.profile_manager import ProfileManager
from phishagent.scoring import ConversationScorer
from phishagent.utils import safe_json_dump
from phishagent.victim_agent import VictimAgent

# One shared console for the whole interactive session
console = Console()

# Visual styles for attacker vs. victim messages
_ATTACKER_STYLE = "bold red"
_VICTIM_STYLE = "bold blue"


# ── Main Entry Point ───────────────────────────────────────────────────────────


def run_interactive_app(app_config: AppConfig) -> None:
    """Main interactive loop: show the menu and dispatch to sub-flows.

    Loops until the user selects Exit (option 5).
    """
    _print_welcome()

    while True:
        choice = _show_main_menu()

        if choice == 1:
            interactive_single_run(app_config)
        elif choice == 2:
            interactive_batch_run(app_config)
        elif choice == 3:
            interactive_status(app_config)
        elif choice == 4:
            interactive_browse_outputs(app_config)
        elif choice == 5:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break


# ── Welcome Banner & Main Menu ─────────────────────────────────────────────────


def _print_welcome() -> None:
    """Print the welcome banner."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]PhishAgent-OS[/bold cyan]\n"
            "[dim]Open-Source Social Engineering Simulation Pipeline[/dim]\n"
            "[dim]All conversations are synthetic — for defensive research only[/dim]",
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )
    console.print()


def _show_main_menu() -> int:
    """Display the main menu and return the user's numeric choice (1–5)."""
    console.print("[bold]Main Menu[/bold]")
    console.print("  [cyan]1.[/cyan] Run single conversation")
    console.print("  [cyan]2.[/cyan] Run batch experiment")
    console.print("  [cyan]3.[/cyan] Check model status")
    console.print("  [cyan]4.[/cyan] Browse saved outputs")
    console.print("  [cyan]5.[/cyan] Exit")
    console.print()
    return _prompt_int_range("Choice", 1, 5)


# ── Single Conversation Flow ───────────────────────────────────────────────────


def interactive_single_run(app_config: AppConfig) -> None:
    """Guide the user step-by-step through a single conversation simulation."""
    console.print()
    console.print(Panel("[bold]Run Single Conversation[/bold]", box=box.SIMPLE_HEAD))

    # Verify Ollama is available before asking a dozen questions
    llm = OllamaClient(
        base_url=app_config.model.ollama_url,
        timeout=app_config.model.timeout_seconds,
    )
    if not llm.is_available():
        console.print("[red]✗ Ollama is not reachable. Run 'ollama serve' first.[/red]")
        return

    available_models = llm.list_models()

    # ── Step 1: model ────────────────────────────────────────────────────────
    model_name = _prompt_model(available_models, app_config.model.name)

    # ── Step 2: attack parameters ────────────────────────────────────────────
    strategy = _prompt_enum_choice(
        "Attack Strategy",
        AttackStrategy,
        descriptions={
            AttackStrategy.AUTHORITY: "Pose as an authority figure (IT admin, manager)",
            AttackStrategy.RECIPROCITY: "Offer something first, then make the request",
            AttackStrategy.URGENCY: "Create time pressure and deadlines",
            AttackStrategy.SOCIAL_PROOF: "Claim others have already complied",
            AttackStrategy.RAPPORT: "Build trust over several turns before asking",
            AttackStrategy.FLATTERY: "Compliment the victim, appeal to ego",
        },
    )

    scenario = _prompt_enum_choice(
        "Attack Scenario",
        AttackerScenario,
        descriptions={
            AttackerScenario.IT_SUPPORT: "IT support technician",
            AttackerScenario.RECRUITER: "External job recruiter",
            AttackerScenario.COLLEAGUE: "Coworker from another team",
            AttackerScenario.VENDOR: "External vendor representative",
            AttackerScenario.EVENT_ORGANIZER: "Company event organizer",
        },
    )

    goal = _prompt_enum_choice(
        "Attack Goal",
        AttackGoal,
        descriptions={
            AttackGoal.CLICK_LINK: "Get the victim to click a link",
            AttackGoal.SHARE_PERSONAL_INFO: "Get the victim to share personal information",
            AttackGoal.DOWNLOAD_FILE: "Get the victim to download a file",
            AttackGoal.SHARE_CREDENTIALS: "Get the victim to share login credentials",
        },
    )

    max_turns = _prompt_int_range(
        "Max conversation turns (3–10)",
        3,
        10,
        default=app_config.conversation.max_turns,
    )

    # ── Step 3: victim profile ───────────────────────────────────────────────
    victim_profile = _prompt_victim_profile(app_config)
    if victim_profile is None:
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Step 4: confirm before running ──────────────────────────────────────
    attacker_config = AttackerConfig(
        goal=goal,
        strategy=strategy,
        scenario=scenario,
        max_turns=max_turns,
    )
    _display_run_config_summary(victim_profile, attacker_config, model_name)

    if not Confirm.ask("Proceed with this configuration?", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Step 5: run ──────────────────────────────────────────────────────────
    _execute_single_run(app_config, llm, model_name, victim_profile, attacker_config)


def _execute_single_run(
    app_config: AppConfig,
    llm: OllamaClient,
    model_name: str,
    victim_profile: VictimProfile,
    attacker_config: AttackerConfig,
) -> None:
    """Execute the conversation, display each turn live, score it, and save."""
    console.print()
    console.print(Panel("[bold green]Simulation Running[/bold green]", box=box.SIMPLE_HEAD))
    console.print("[dim]─[/dim]" * 40)

    attacker = AttackerAgent(attacker_config, llm, model_name)
    victim = VictimAgent(victim_profile, llm, model_name)
    conv_config = ConversationConfig(
        max_turns=attacker_config.max_turns,
        early_termination=app_config.conversation.early_termination,
    )
    engine = ConversationEngine(attacker, victim, conv_config)

    # turn_callback is called by the engine each time a message is appended,
    # so the conversation appears progressively in the terminal.
    def on_turn(message: Message) -> None:
        if message.role == "attacker":
            label = Text(f"[Turn {message.turn_number}] ATTACKER", style=_ATTACKER_STYLE)
        else:
            label = Text(f"[Turn {message.turn_number}] VICTIM   ", style=_VICTIM_STYLE)
        console.print(label, end="  ")
        console.print(message.content)
        console.print()

    result = engine.run(victim_profile, attacker_config, turn_callback=on_turn)
    console.print("[dim]─[/dim]" * 40)

    # Score the conversation
    scorer = ConversationScorer(llm, app_config.scoring)
    with console.status("[bold]Scoring conversation (this may take a moment)...[/bold]"):
        try:
            result.scores = scorer.score(result)
        except Exception as e:
            console.print(f"[yellow]Scoring failed: {e}[/yellow]")

    _display_result_summary(result)

    # Offer to save
    output_dir = Path(app_config.output.base_dir) / "conversations"
    output_path = output_dir / f"conv_{result.conversation_id}.json"
    console.print()
    if Confirm.ask(f"Save conversation to [cyan]{output_path}[/cyan]?", default=True):
        safe_json_dump(result, str(output_path))
        console.print(f"[green]✓ Saved to {output_path}[/green]")

    console.print()


# ── Batch Experiment Flow ──────────────────────────────────────────────────────


def interactive_batch_run(app_config: AppConfig) -> None:
    """Guide the user through selecting and running a batch factorial experiment."""
    console.print()
    console.print(Panel("[bold]Run Batch Experiment[/bold]", box=box.SIMPLE_HEAD))

    # Find available experiment configs
    exp_path = _pick_experiment_config()
    if not exp_path:
        return

    try:
        with open(exp_path) as f:
            exp_data = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        return

    # Check Ollama
    llm = OllamaClient(
        base_url=app_config.model.ollama_url,
        timeout=app_config.model.timeout_seconds,
    )
    if not llm.is_available():
        console.print("[red]✗ Ollama is not reachable. Run 'ollama serve' first.[/red]")
        return

    available_models = llm.list_models()

    # Optional overrides
    console.print("\n[bold]Overrides[/bold] [dim](press Enter to keep config defaults)[/dim]")

    default_model = exp_data.get("model_name", app_config.model.name)
    model_name = _prompt_model(available_models, default_model)

    default_reps = exp_data.get("repetitions", 3)
    reps_raw = Prompt.ask("  Repetitions", default=str(default_reps))
    try:
        repetitions = max(1, int(reps_raw))
    except (ValueError, TypeError):
        repetitions = default_reps

    # Build profiles and attacker configs from the spec
    pm = ProfileManager()
    if not exp_data or "base_profile" not in exp_data:
        console.print("[red]Experiment config is missing 'base_profile'.[/red]")
        return
    try:
        base_profile = VictimProfile(**exp_data["base_profile"])
    except Exception as e:
        console.print(f"[red]Invalid base_profile in experiment config: {e}[/red]")
        return
    vary = exp_data.get("vary", {})
    profiles = pm.generate_factorial_profiles(FactorialSpec(base_profile=base_profile, vary=vary)) if vary else [base_profile]
    attacker_configs = [AttackerConfig(**ac) for ac in exp_data.get("attacker_configs", [])]

    if not attacker_configs:
        console.print("[red]No attacker_configs found in experiment file.[/red]")
        return

    total = len(profiles) * len(attacker_configs) * repetitions

    # Summary table
    console.print()
    table = Table(title="Experiment Summary", box=box.ROUNDED)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Experiment ID", exp_data["experiment_id"])
    table.add_row("Description", exp_data.get("description", "—"))
    table.add_row("Model", model_name)
    table.add_row("Profiles", str(len(profiles)))
    table.add_row("Attacker Configs", str(len(attacker_configs)))
    table.add_row("Repetitions", str(repetitions))
    table.add_row("Total Conversations", f"[bold cyan]{total}[/bold cyan]")
    console.print(table)

    if not Confirm.ask("\nStart experiment?", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    output_dir = str(
        Path(app_config.output.base_dir) / "experiments" / exp_data["experiment_id"]
    )
    exp_config = ExperimentConfig(
        experiment_id=exp_data["experiment_id"],
        description=exp_data.get("description", ""),
        model_name=model_name,
        profiles=profiles,
        attacker_configs=attacker_configs,
        repetitions=repetitions,
        output_dir=output_dir,
    )

    # Run with a progress bar
    runner = ExperimentRunner(app_config, llm)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running conversations...", total=total)

        def on_progress(completed: int, total_count: int, last_result: Optional[ConversationResult]) -> None:
            progress.update(task, completed=completed)
            if last_result:
                progress.update(
                    task,
                    description=f"Running... last outcome: [bold]{last_result.outcome.value}[/bold]",
                )

        result = runner.run(exp_config, progress_callback=on_progress)

    csv_path = f"{output_dir}/{exp_config.experiment_id}.csv"
    runner.export_csv(result, csv_path)
    _display_experiment_summary(result, csv_path, output_dir)


def _pick_experiment_config() -> Optional[str]:
    """List known experiment configs and let the user pick one or enter a path."""
    config_dir = Path("config/profiles")
    experiment_files: list[tuple[Path, str, str]] = []

    if config_dir.exists():
        for fpath in sorted(config_dir.glob("*.yaml")):
            try:
                with open(fpath) as fh:
                    data = yaml.safe_load(fh)
                # Experiment files have both experiment_id and attacker_configs
                if "experiment_id" in data and "attacker_configs" in data:
                    experiment_files.append(
                        (fpath, data.get("experiment_id", ""), data.get("description", ""))
                    )
            except Exception:
                pass

    if experiment_files:
        console.print("\n[bold]Available experiment configs:[/bold]")
        for i, (fpath, exp_id, desc) in enumerate(experiment_files, 1):
            console.print(
                f"  [cyan]{i}.[/cyan] {fpath.name}  "
                f"[dim]{exp_id}{'  — ' + desc if desc else ''}[/dim]"
            )
        console.print(f"  [cyan]{len(experiment_files) + 1}.[/cyan] Enter a custom path")
        console.print(f"  [cyan]{len(experiment_files) + 2}.[/cyan] Cancel")
        console.print()

        choice = _prompt_int_range("Choice", 1, len(experiment_files) + 2)
        if choice == len(experiment_files) + 2:
            return None
        if choice <= len(experiment_files):
            return str(experiment_files[choice - 1][0])

    # Custom path
    path = _prompt_string("Experiment config path")
    return path if path else None


# ── System Status ──────────────────────────────────────────────────────────────


def interactive_status(app_config: AppConfig) -> None:
    """Show Ollama connectivity and available models."""
    console.print()
    console.print(Panel("[bold]System Status[/bold]", box=box.SIMPLE_HEAD))

    client = OllamaClient(base_url=app_config.model.ollama_url)

    table = Table(box=box.ROUNDED)
    table.add_column("Item", style="dim")
    table.add_column("Value")
    table.add_row("Config model", app_config.model.name)
    table.add_row("Ollama URL", app_config.model.ollama_url)

    if client.is_available():
        table.add_row("Ollama", "[green]✓ Running[/green]")
        models = client.list_models()
        if models:
            table.add_row("Available models", "\n".join(f"• {m}" for m in models))
            configured_available = app_config.model.name in models
            table.add_row(
                "Config model available",
                "[green]✓ Yes[/green]" if configured_available else "[yellow]✗ Not pulled[/yellow]",
            )
            if not configured_available:
                table.add_row("Fix", f"[dim]Run: ollama pull {app_config.model.name}[/dim]")
        else:
            table.add_row("Available models", "[yellow]None — run: ollama pull mistral:7b[/yellow]")
    else:
        table.add_row("Ollama", "[red]✗ Not reachable[/red]")
        table.add_row("Fix", "[dim]Run: ollama serve[/dim]")

    console.print(table)
    console.print()


# ── Browse Saved Outputs ───────────────────────────────────────────────────────


def interactive_browse_outputs(app_config: AppConfig) -> None:
    """Browse conversation JSON files and experiment CSV files."""
    console.print()
    console.print(Panel("[bold]Browse Saved Outputs[/bold]", box=box.SIMPLE_HEAD))

    base_dir = Path(app_config.output.base_dir)

    console.print("[bold]Browse:[/bold]")
    console.print("  [cyan]1.[/cyan] Recent conversations (JSON)")
    console.print("  [cyan]2.[/cyan] Experiment results (CSV)")
    console.print("  [cyan]3.[/cyan] Back")
    console.print()

    choice = _prompt_int_range("Choice", 1, 3)
    if choice == 1:
        _browse_conversations(base_dir / "conversations")
    elif choice == 2:
        _browse_experiments(base_dir / "experiments")


def _browse_conversations(conv_dir: Path) -> None:
    """List recent conversation JSON files and offer a detail view."""
    if not conv_dir.exists() or not list(conv_dir.glob("*.json")):
        console.print("[dim]No conversations found. Run a simulation first.[/dim]")
        return

    json_files = sorted(
        conv_dir.glob("conv_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    shown = json_files[:20]

    console.print(f"\n[bold]Recent conversations ({len(json_files)} total, showing {len(shown)}):[/bold]")
    for i, fpath in enumerate(shown, 1):
        try:
            with open(fpath) as f:
                data = json.load(f)
            outcome = data.get("outcome", "?")
            victim = data.get("victim_profile", {}).get("name", "?")
            strategy = data.get("attacker_config", {}).get("strategy", "?")
            created = (data.get("created_at") or "")[:10]
            console.print(
                f"  [cyan]{i:2}.[/cyan]  {fpath.name}  "
                f"[dim]{created}[/dim]  "
                f"victim=[bold]{victim}[/bold]  "
                f"strategy=[bold]{strategy}[/bold]  "
                f"outcome=[bold]{outcome}[/bold]"
            )
        except Exception:
            console.print(f"  [cyan]{i:2}.[/cyan]  {fpath.name}  [dim](unreadable)[/dim]")

    console.print()
    idx = _prompt_int_range("Enter number to view details (0 = back)", 0, len(shown))
    if idx > 0:
        _display_conversation_file(shown[idx - 1])


def _display_conversation_file(fpath: Path) -> None:
    """Show a formatted summary of a single saved conversation JSON."""
    try:
        with open(fpath) as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Could not read file: {e}[/red]")
        return

    console.print()
    console.print(Panel(f"[bold]{fpath.name}[/bold]", box=box.SIMPLE_HEAD))

    # Victim profile
    vp = data.get("victim_profile", {})
    p = vp.get("personality", {})
    pt = Table(title="Victim Profile", box=box.SIMPLE)
    pt.add_column("Trait", style="dim")
    pt.add_column("Value")
    pt.add_row("Name", vp.get("name", "?"))
    pt.add_row("Occupation", vp.get("occupation", "?"))
    pt.add_row("Security Awareness", vp.get("security_awareness", "?"))
    pt.add_row("Agreeableness", f"{p.get('agreeableness', 0):.2f}")
    pt.add_row("Conscientiousness", f"{p.get('conscientiousness', 0):.2f}")
    pt.add_row("Neuroticism", f"{p.get('neuroticism', 0):.2f}")
    console.print(pt)

    # Attack config
    ac = data.get("attacker_config", {})
    console.print(
        f"[dim]Strategy:[/dim] {ac.get('strategy', '?')}  "
        f"[dim]Scenario:[/dim] {ac.get('scenario', '?')}  "
        f"[dim]Goal:[/dim] {ac.get('goal', '?')}"
    )
    console.print()

    # Messages
    messages = data.get("messages", [])
    console.print(f"[bold]Conversation ({len(messages)} messages):[/bold]")
    console.print("[dim]─[/dim]" * 40)
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        turn = msg.get("turn_number", 0)
        if role == "attacker":
            label = Text(f"[Turn {turn}] ATTACKER", style=_ATTACKER_STYLE)
        else:
            label = Text(f"[Turn {turn}] VICTIM   ", style=_VICTIM_STYLE)
        console.print(label, end="  ")
        console.print(content)
        console.print()

    # Outcome and scores
    outcome = data.get("outcome", "?")
    console.print("[dim]─[/dim]" * 40)
    console.print(f"[bold]Outcome:[/bold] [cyan]{outcome.upper()}[/cyan]")
    scores = data.get("scores")
    if scores:
        _print_scores_table(scores)
    console.print()


def _browse_experiments(exp_dir: Path) -> None:
    """List experiment CSV files and show a high-level summary of the selected one."""
    if not exp_dir.exists():
        console.print("[dim]No experiments found. Run a batch experiment first.[/dim]")
        return

    csv_files = sorted(exp_dir.rglob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not csv_files:
        console.print("[dim]No CSV files found.[/dim]")
        return

    console.print(f"\n[bold]Experiment CSV files ({len(csv_files)}):[/bold]")
    for i, fpath in enumerate(csv_files, 1):
        size_kb = fpath.stat().st_size // 1024
        console.print(f"  [cyan]{i}.[/cyan] {fpath}  [dim]({size_kb} KB)[/dim]")

    console.print()
    choice = _prompt_int_range("Select a file to summarize (0 = back)", 0, len(csv_files))
    if choice > 0:
        _display_experiment_csv(csv_files[choice - 1])


def _display_experiment_csv(fpath: Path) -> None:
    """Print outcome distribution and score summary for an experiment CSV."""
    import csv as _csv

    try:
        with open(fpath) as f:
            rows = list(_csv.DictReader(f))
    except Exception as e:
        console.print(f"[red]Could not read CSV: {e}[/red]")
        return

    if not rows:
        console.print("[dim]CSV is empty.[/dim]")
        return

    console.print()
    console.print(Panel(f"[bold]{fpath.name}[/bold]", box=box.SIMPLE_HEAD))
    console.print(f"Total rows: [bold]{len(rows)}[/bold]\n")

    # Outcome distribution
    outcomes: dict[str, int] = {}
    for r in rows:
        key = r.get("outcome", "?")
        outcomes[key] = outcomes.get(key, 0) + 1

    ot = Table(title="Outcome Distribution", box=box.SIMPLE)
    ot.add_column("Outcome")
    ot.add_column("Count", justify="right")
    ot.add_column("Pct", justify="right")
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
        ot.add_row(outcome, str(count), f"{100 * count / len(rows):.1f}%")
    console.print(ot)

    # Score summary
    def _floats(col: str) -> list[float]:
        result = []
        for r in rows:
            val = r.get(col)
            if val:
                try:
                    result.append(float(val))
                except (ValueError, TypeError):
                    pass
        return result

    persuasion = _floats("persuasion")
    coherence = _floats("coherence")
    composite = _floats("composite")

    if persuasion:
        st = Table(title="Score Summary", box=box.SIMPLE)
        st.add_column("Metric")
        st.add_column("Mean", justify="right")
        st.add_column("Std", justify="right")
        st.add_column("Min", justify="right")
        st.add_column("Max", justify="right")
        for label, vals in [("Persuasion", persuasion), ("Coherence", coherence), ("Composite", composite)]:
            if vals:
                st.add_row(
                    label,
                    f"{statistics.mean(vals):.3f}",
                    f"{statistics.stdev(vals):.3f}" if len(vals) > 1 else "—",
                    f"{min(vals):.3f}",
                    f"{max(vals):.3f}",
                )
        console.print(st)

    console.print()


# ── Victim Profile Builder ─────────────────────────────────────────────────────


def _prompt_victim_profile(app_config: AppConfig) -> Optional[VictimProfile]:
    """Ask whether to load a profile from file or build one interactively."""
    console.print("\n[bold]Victim Profile[/bold]")
    console.print("  [cyan]1.[/cyan] Load from a saved YAML file")
    console.print("  [cyan]2.[/cyan] Build interactively")
    console.print("  [cyan]3.[/cyan] Cancel")
    console.print()

    choice = _prompt_int_range("Choice", 1, 3)
    if choice == 1:
        return _load_profile_interactive()
    if choice == 2:
        return build_profile_interactive()
    return None


def _load_profile_interactive() -> Optional[VictimProfile]:
    """Let the user pick a profile YAML from known locations or enter a custom path."""
    profile_dir = Path("config/profiles")
    profile_files: list[tuple[Path, str]] = []

    if profile_dir.exists():
        for fpath in sorted(profile_dir.glob("*.yaml")):
            try:
                with open(fpath) as fh:
                    data = yaml.safe_load(fh)
                # Only single-profile files have name + personality at the top level
                if "name" in data and "personality" in data:
                    profile_files.append((fpath, data.get("name", fpath.stem)))
            except Exception:
                pass

    if profile_files:
        console.print("\n[bold]Available profiles:[/bold]")
        for i, (fpath, name) in enumerate(profile_files, 1):
            console.print(f"  [cyan]{i}.[/cyan] {fpath.name}  [dim]({name})[/dim]")
        console.print(f"  [cyan]{len(profile_files) + 1}.[/cyan] Enter a custom path")
        console.print()

        choice = _prompt_int_range("Choose profile", 1, len(profile_files) + 1)
        path = str(profile_files[choice - 1][0]) if choice <= len(profile_files) else _prompt_string("Profile YAML path")
    else:
        path = _prompt_string("Profile YAML path")

    pm = ProfileManager()
    try:
        profile = pm.load_profile(path)
        console.print(f"[green]✓ Loaded: {profile.name}[/green]")
        return profile
    except Exception as e:
        console.print(f"[red]Failed to load profile: {e}[/red]")
        return None


def build_profile_interactive() -> Optional[VictimProfile]:
    """Interactively prompt for every victim profile field with full validation.

    Returns a VictimProfile on success, or None if validation fails.
    """
    console.print("\n[bold]Build Victim Profile[/bold]")
    console.print("[dim]Personality fields are 0.0 (low) – 1.0 (high).[/dim]\n")

    name = _prompt_string("Name", default="Alex Doe")
    occupation = _prompt_string("Occupation", default="software engineer")

    interests_raw = _prompt_string(
        "Interests (comma-separated, 1–5 items, e.g. technology,sports)",
        default="technology,sports",
        required=False,  # blank input falls through to the "technology" fallback below
    )
    interests = [s.strip() for s in interests_raw.split(",") if s.strip()][:5]
    if not interests:
        interests = ["technology"]

    comm_style = _prompt_enum_choice("Communication Style", CommunicationStyle)
    sec_awareness = _prompt_enum_choice("Security Awareness", SecurityAwareness)

    console.print("\n  [bold]Big Five Personality Traits[/bold] [dim](0.0 = low, 1.0 = high)[/dim]")
    console.print("  [dim]These shape how the victim responds to the attacker's social engineering attempts.[/dim]")
    openness = _prompt_float("  Openness (high = curious about novel requests, low = sticks to routine)", default=0.5)
    conscientiousness = _prompt_float("  Conscientiousness (high = verifies before acting, low = acts on impulse)", default=0.5)
    extraversion = _prompt_float("  Extraversion (high = chatty and shares freely, low = reserved)", default=0.5)
    agreeableness = _prompt_float("  Agreeableness (high = trusting and cooperative, low = skeptical)", default=0.5)
    neuroticism = _prompt_float("  Neuroticism (high = anxious, may rush to resolve pressure, low = calm under pressure)", default=0.5)

    console.print("\n  [bold]Additional Traits[/bold]")
    tech_proficiency = _prompt_float("  Tech proficiency (high = overconfident, paradoxically more vulnerable)", default=0.5)
    impulsivity = _prompt_float("  Impulsivity (high = responds without thinking, low = deliberate)", default=0.5)

    try:
        return VictimProfile(
            name=name,
            occupation=occupation,
            interests=interests,
            communication_style=comm_style,
            security_awareness=sec_awareness,
            personality=PersonalityTraits(
                openness=openness,
                conscientiousness=conscientiousness,
                extraversion=extraversion,
                agreeableness=agreeableness,
                neuroticism=neuroticism,
            ),
            tech_proficiency=tech_proficiency,
            impulsivity=impulsivity,
        )
    except Exception as e:
        console.print(f"[red]Profile validation failed: {e}[/red]")
        return None


# ── Display Helpers ────────────────────────────────────────────────────────────


def _display_run_config_summary(
    victim_profile: VictimProfile,
    attacker_config: AttackerConfig,
    model_name: str,
) -> None:
    """Print a summary table of the full run configuration before execution."""
    console.print()
    p = victim_profile.personality

    table = Table(title="Run Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Model", model_name)
    table.add_row("Strategy", attacker_config.strategy.value)
    table.add_row("Scenario", attacker_config.scenario.value)
    table.add_row("Goal", attacker_config.goal.value)
    table.add_row("Max Turns", str(attacker_config.max_turns))

    table.add_section()

    table.add_row("Victim", victim_profile.name)
    table.add_row("Occupation", victim_profile.occupation)
    table.add_row("Interests", ", ".join(victim_profile.interests))
    table.add_row("Comm. Style", victim_profile.communication_style.value)
    table.add_row("Security Awareness", victim_profile.security_awareness.value)
    table.add_row(
        "Big Five (O/C/E/A/N)",
        f"{p.openness:.1f} / {p.conscientiousness:.1f} / {p.extraversion:.1f}"
        f" / {p.agreeableness:.1f} / {p.neuroticism:.1f}",
    )

    console.print(table)
    console.print()


def _display_result_summary(result: ConversationResult) -> None:
    """Print outcome, scores, and metadata after a completed run."""
    console.print()
    victim_turns = len([m for m in result.messages if m.role == "victim"])
    outcome_color = {
        "compliance": "green",
        "partial": "yellow",
        "refusal": "red",
        "suspicion": "red",
        "max_turns": "yellow",
        "error": "red",
    }.get(result.outcome.value, "white")

    table = Table(title="Simulation Results", box=box.ROUNDED)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row(
        "Outcome",
        f"[{outcome_color}]{result.outcome.value.upper()}[/{outcome_color}]",
    )
    table.add_row("Total Turns", str(victim_turns))
    table.add_row("Total Tokens", str(result.total_tokens))
    table.add_row("Duration", f"{result.total_duration_seconds:.1f}s")

    if result.scores:
        table.add_section()
        table.add_row(
            "Persuasion",
            f"{result.scores.persuasion:.2f}  [dim]— how effectively the attacker convinced the victim (0=failed, 1=total success)[/dim]",
        )
        table.add_row(
            "Coherence",
            f"{result.scores.coherence:.2f}  [dim]— how natural and realistic the dialogue felt (0=nonsensical, 1=indistinguishable from real)[/dim]",
        )
        table.add_row(
            "Detectability",
            f"{result.scores.detectability:.2f}  [dim]— how obvious the attack was (0=completely stealthy, 1=blatantly malicious)[/dim]",
        )
        table.add_row(
            "Composite",
            f"[cyan]{result.scores.composite:.2f}[/cyan]  [dim]— weighted overall score (high persuasion + high coherence + LOW detectability = high composite)[/dim]",
        )

    console.print(table)

    # Print rationales if available
    if result.scores:
        if result.scores.persuasion_rationale:
            console.print(f"\n[dim]Persuasion rationale:[/dim] {result.scores.persuasion_rationale}")
        if result.scores.coherence_rationale:
            console.print(f"\n[dim]Coherence rationale:[/dim] {result.scores.coherence_rationale}")
        if result.scores.detectability_rationale:
            console.print(f"\n[dim]Detectability rationale:[/dim] {result.scores.detectability_rationale}")


def _print_scores_table(scores: dict) -> None:
    """Print scores from a raw dict (e.g. loaded from a JSON file)."""
    table = Table(title="Scores", box=box.SIMPLE)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    for key in ("persuasion", "coherence", "detectability", "composite"):
        val = scores.get(key)
        if val is not None:
            table.add_row(key.capitalize(), f"{float(val):.2f}")
    console.print(table)


def _display_experiment_summary(
    result, csv_path: str, output_dir: str
) -> None:
    """Print experiment completion summary."""
    console.print()
    table = Table(title="[bold green]Experiment Complete[/bold green]", box=box.ROUNDED)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Total Conversations", str(result.total_conversations))
    table.add_row("Completed", f"[green]{result.completed_conversations}[/green]")
    failed_fmt = f"[red]{result.failed_conversations}[/red]" if result.failed_conversations else "[green]0[/green]"
    table.add_row("Failed", failed_fmt)
    table.add_row("CSV Output", csv_path)
    table.add_row("Conversations Dir", f"{output_dir}/conversations/")
    console.print(table)
    console.print()


# ── Input Prompt Helpers ───────────────────────────────────────────────────────


def _prompt_int_range(label: str, min_val: int, max_val: int, default: Optional[int] = None) -> int:
    """Prompt for an integer in [min_val, max_val], re-prompting on bad input."""
    default_str = str(default) if default is not None else None
    while True:
        raw = Prompt.ask(f"  {label}", default=default_str)
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            console.print(f"  [red]Please enter a number between {min_val} and {max_val}.[/red]")
        except (ValueError, TypeError):
            console.print("  [red]Invalid input. Please enter a whole number.[/red]")


def _prompt_float(label: str, min_val: float = 0.0, max_val: float = 1.0, default: float = 0.5) -> float:
    """Prompt for a float in [min_val, max_val], re-prompting on bad input."""
    while True:
        raw = Prompt.ask(label, default=str(default))
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return round(val, 3)
            console.print(f"  [red]Value must be between {min_val:.1f} and {max_val:.1f}.[/red]")
        except (ValueError, TypeError):
            console.print("  [red]Invalid input. Please enter a decimal number (e.g. 0.7).[/red]")


def _prompt_string(label: str, default: Optional[str] = None, required: bool = True) -> str:
    """Prompt for a non-empty string, re-prompting if required and blank."""
    while True:
        raw = Prompt.ask(f"  {label}", default=default or "")
        val = (raw or "").strip()
        if val or not required:
            return val
        console.print("  [red]This field is required.[/red]")


def _prompt_enum_choice(title: str, enum_class, descriptions: Optional[dict] = None):
    """Present a numbered list of enum values and return the selected member."""
    values = list(enum_class)
    console.print(f"\n[bold]{title}:[/bold]")
    for i, v in enumerate(values, 1):
        desc = f"  [dim]{descriptions[v]}[/dim]" if descriptions and v in descriptions else ""
        console.print(f"  [cyan]{i}.[/cyan] {v.value}{desc}")
    console.print()
    choice = _prompt_int_range("Choice", 1, len(values))
    return values[choice - 1]


def _prompt_model(available_models: list[str], default: str) -> str:
    """Let the user pick from locally available Ollama models or enter a name."""
    if not available_models:
        console.print("[yellow]No local models found. Using default.[/yellow]")
        return default

    console.print("\n[bold]Select Model:[/bold]")
    for i, m in enumerate(available_models, 1):
        marker = " [dim](default)[/dim]" if m == default else ""
        console.print(f"  [cyan]{i}.[/cyan] {m}{marker}")
    console.print(f"  [cyan]{len(available_models) + 1}.[/cyan] Enter a custom model name")
    console.print()

    choice = _prompt_int_range("Choice", 1, len(available_models) + 1)
    if choice <= len(available_models):
        return available_models[choice - 1]
    return _prompt_string("Custom model name", default=default)
