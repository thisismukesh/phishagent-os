# PhishAgent-OS

Open-Source Social Engineering Simulation Pipeline for defensive cybersecurity research.

## Overview

PhishAgent-OS is a command-line research tool that simulates social engineering attacks in a fully sandboxed environment using only open-source, locally-running LLMs. No human subjects are involved. The system runs multi-turn attacker-vs-victim conversations between two LLM agents and scores results on persuasion effectiveness, dialogue coherence, and detection difficulty.

**Research Question:** Can an open-source LLM, embedded in an autonomous multi-stage pipeline, conduct effective personalized social engineering attacks in simulated multi-turn chat scenarios, and what profile features most strongly predict attack success?

## Ethics Statement

This project is **defensive research**. It produces no real attacks, contacts no real people, and exfiltrates no real data. All conversations occur between two LLM agents in a local sandbox. Every generated conversation is watermarked as synthetic research output. The goal is to understand and improve defenses against AI-powered social engineering.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- At least one model pulled (e.g., `mistral:7b`)

## Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model
ollama pull mistral:7b

# 3. Clone and install
git clone <repo_url>
cd phishagent-os
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 4. Verify
phishagent status
```

## Usage

### Interactive Mode (Recommended for Humans)

Launch the guided menu by running the tool with no arguments:

```bash
phishagent
```

Or explicitly:

```bash
phishagent interactive
```

You will see a menu like this:

```
╔══════════════════════════════════════════╗
║  PhishAgent-OS                           ║
║  Open-Source Social Engineering Sim...   ║
╚══════════════════════════════════════════╝

Main Menu
  1. Run single conversation
  2. Run batch experiment
  3. Check model status
  4. Browse saved outputs
  5. Exit
```

**Single conversation** guides you through:
- Selecting an Ollama model from locally available options
- Choosing attack strategy, scenario, and goal from numbered menus
- Setting max turns
- Loading a victim profile from a YAML file or building one interactively
- Confirming the full configuration before running
- Watching the conversation appear turn-by-turn as it executes
- Viewing outcome and persuasion/coherence/detectability scores
- Optionally saving the result to JSON

**Batch experiment** guides you through:
- Picking an experiment config file from `config/profiles/` (or entering a path)
- Optionally overriding model and repetition count
- Reviewing the factorial design before running
- Watching a live progress bar as conversations execute
- Seeing a summary table with CSV output path

**Browse saved outputs** lets you:
- List and inspect recent conversation JSON files
- View a formatted transcript with scores
- Summarize experiment CSV files (outcome distribution, score statistics)

### Command-Line Mode (Scripting and Automation)

The original flag-based CLI is fully preserved for scripting use.

#### Single Conversation

```bash
phishagent run \
  --profile config/profiles/high_agreeableness.yaml \
  --strategy urgency \
  --scenario it_support \
  --goal click_link
```

#### Batch Experiment

```bash
phishagent experiment \
  --experiment-config config/profiles/factorial_batch.yaml \
  --repetitions 3
```

#### Check System Status

```bash
phishagent status
```

#### Analyze Results

```bash
python scripts/analyze.py output/experiments/personality_sweep_001/personality_sweep_001.csv
```

## Project Structure

```
phishagent-os/
├── src/phishagent/
│   ├── models.py              # All Pydantic data models
│   ├── config.py              # Configuration loading
│   ├── llm_client.py          # Ollama HTTP client
│   ├── profile_manager.py     # Profile loading and factorial generation
│   ├── attacker_agent.py      # Attacker agent logic
│   ├── victim_agent.py        # Victim agent logic
│   ├── conversation_engine.py # Multi-turn orchestration
│   ├── scoring.py             # LLM-as-judge evaluation
│   ├── experiment_runner.py   # Batch execution
│   ├── interactive.py         # Guided interactive terminal mode
│   ├── cli.py                 # CLI entry point (flag-based + interactive)
│   └── utils.py               # Shared utilities
├── config/                    # Configuration and profile YAML files
├── tests/                     # Unit, integration, and E2E tests
├── scripts/                   # Analysis scripts
└── output/                    # Generated results (gitignored)
```

## Running Tests

```bash
# Unit tests only (no Ollama needed)
pytest tests/unit/ -v

# Integration tests (requires Ollama + mistral:7b)
pytest tests/integration/ -m integration -v

# All tests
pytest -v

# With coverage
pytest tests/unit/ --cov=phishagent --cov-report=term-missing
```

## Configuration

Default configuration is in `config/default.yaml`. Override with environment variables (`PHISHAGENT_MODEL_NAME`, `PHISHAGENT_OLLAMA_URL`, etc.) or CLI flags.

## License

Research use only. See CLAUDE.md for full ethical guidelines and research context.
