"""Batch experiment execution, factorial design, and CSV/JSON export.

Executes all conversations in a factorial experiment, manages progress,
collects results, and exports to CSV for statistical analysis.
"""

import csv
import itertools
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from phishagent.attacker_agent import AttackerAgent
from phishagent.config import AppConfig
from phishagent.conversation_engine import ConversationEngine
from phishagent.llm_client import OllamaClient
from phishagent.models import (
    ConversationOutcome,
    ConversationResult,
    ExperimentConfig,
    ExperimentResult,
)
from phishagent.scoring import ConversationScorer
from phishagent.utils import get_logger, safe_json_dump
from phishagent.victim_agent import VictimAgent

logger = get_logger(__name__)

# CSV column specification
CSV_COLUMNS = [
    "conversation_id", "model", "strategy", "scenario", "goal",
    "victim_name", "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism", "security_awareness", "tech_proficiency",
    "impulsivity", "communication_style", "num_turns", "outcome",
    "persuasion", "coherence", "detectability", "composite",
    "total_tokens", "duration_seconds",
]


class ExperimentRunner:
    def __init__(self, config: AppConfig, llm: Optional[OllamaClient] = None):
        """Initialize with app config. Creates LLM client, scorer."""
        self.config = config
        self.llm = llm or OllamaClient(
            base_url=config.model.ollama_url,
            timeout=config.model.timeout_seconds,
        )
        self.scorer = ConversationScorer(self.llm, config.scoring)

    def run(
        self,
        experiment: ExperimentConfig,
        progress_callback: Optional[Callable[[int, int, Optional[ConversationResult]], None]] = None,
    ) -> ExperimentResult:
        """Execute all conversations in the experiment.

        For each (profile, attacker_config) pair x repetitions:
          1. Create attacker and victim agents.
          2. Run conversation via ConversationEngine.
          3. Score the conversation.
          4. Save individual result to disk.
          5. Update progress.
        """
        started_at = datetime.now(timezone.utc)
        results: list[ConversationResult] = []
        failed = 0

        # Generate all combinations
        combinations = list(itertools.product(experiment.profiles, experiment.attacker_configs))
        random.shuffle(combinations)

        total = len(combinations) * experiment.repetitions
        completed = 0

        model_name = experiment.model_name

        for rep in range(experiment.repetitions):
            for profile, attacker_config in combinations:
                try:
                    # Create agents
                    attacker = AttackerAgent(attacker_config, self.llm, model_name)
                    victim = VictimAgent(profile, self.llm, model_name)

                    # Run conversation — use attacker's max_turns if specified
                    from phishagent.config import ConversationConfig
                    conv_config = ConversationConfig(
                        max_turns=attacker_config.max_turns,
                        turn_delay_seconds=self.config.conversation.turn_delay_seconds,
                        early_termination=self.config.conversation.early_termination,
                    )
                    engine = ConversationEngine(attacker, victim, conv_config)
                    result = engine.run(profile, attacker_config)

                    # Score conversation
                    try:
                        score = self.scorer.score(result)
                        result.scores = score
                    except Exception as e:
                        logger.error(f"Scoring failed for {result.conversation_id}: {e}")

                    results.append(result)

                    # Save individual result
                    if self.config.output.save_conversations:
                        out_dir = Path(experiment.output_dir) / "conversations"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        safe_json_dump(
                            result,
                            str(out_dir / f"conv_{result.conversation_id}.json"),
                        )

                except Exception as e:
                    logger.error(f"Conversation failed: {e}")
                    failed += 1

                completed += 1
                if progress_callback:
                    progress_callback(
                        completed,
                        total,
                        results[-1] if results else None,
                    )

        completed_at = datetime.now(timezone.utc)

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            total_conversations=total,
            completed_conversations=len(results),
            failed_conversations=failed,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )

    def export_csv(self, result: ExperimentResult, path: str) -> None:
        """Export experiment results to a flat CSV for statistical analysis."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for conv in result.results:
                row = {
                    "conversation_id": conv.conversation_id,
                    "model": conv.model_name,
                    "strategy": conv.attacker_config.strategy.value,
                    "scenario": conv.attacker_config.scenario.value,
                    "goal": conv.attacker_config.goal.value,
                    "victim_name": conv.victim_profile.name,
                    "openness": conv.victim_profile.personality.openness,
                    "conscientiousness": conv.victim_profile.personality.conscientiousness,
                    "extraversion": conv.victim_profile.personality.extraversion,
                    "agreeableness": conv.victim_profile.personality.agreeableness,
                    "neuroticism": conv.victim_profile.personality.neuroticism,
                    "security_awareness": conv.victim_profile.security_awareness.value,
                    "tech_proficiency": conv.victim_profile.tech_proficiency,
                    "impulsivity": conv.victim_profile.impulsivity,
                    "communication_style": conv.victim_profile.communication_style.value,
                    "num_turns": len([m for m in conv.messages if m.role == "victim"]),
                    "outcome": conv.outcome.value,
                    "persuasion": conv.scores.persuasion if conv.scores else "",
                    "coherence": conv.scores.coherence if conv.scores else "",
                    "detectability": conv.scores.detectability if conv.scores else "",
                    "composite": conv.scores.composite if conv.scores else "",
                    "total_tokens": conv.total_tokens,
                    "duration_seconds": conv.total_duration_seconds,
                }
                writer.writerow(row)

        logger.info(f"Exported {len(result.results)} results to {path}")

    def export_conversations(self, result: ExperimentResult, dir: str) -> None:
        """Save each conversation as a separate JSON file."""
        out_dir = Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for conv in result.results:
            safe_json_dump(conv, str(out_dir / f"conv_{conv.conversation_id}.json"))

        logger.info(f"Exported {len(result.results)} conversations to {dir}")
