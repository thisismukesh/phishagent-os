"""E2E test: full CLI batch experiment run."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.mark.e2e
class TestBatchRun:
    def test_cli_experiment_produces_csv(self, tmp_path):
        """Invoke CLI experiment command with a tiny config and verify CSV output."""
        # Create a minimal experiment config (2x1x1 = 2 conversations)
        exp_config = {
            "experiment_id": "e2e_test",
            "description": "E2E test experiment",
            "model_name": "mistral:7b",
            "repetitions": 1,
            "base_profile": {
                "name": "Victim",
                "personality": {
                    "openness": 0.5, "conscientiousness": 0.5,
                    "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5,
                },
                "communication_style": "casual",
                "security_awareness": "medium",
                "interests": ["technology"],
                "occupation": "engineer",
                "tech_proficiency": 0.5,
                "impulsivity": 0.5,
            },
            "vary": {
                "personality.agreeableness": [0.2, 0.8],
            },
            "attacker_configs": [
                {
                    "goal": "click_link",
                    "strategy": "urgency",
                    "scenario": "it_support",
                    "max_turns": 3,
                    "escalation_threshold": 2,
                },
            ],
        }

        config_path = tmp_path / "test_experiment.yaml"
        with open(config_path, "w") as f:
            yaml.dump(exp_config, f)

        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, "-m", "phishagent.cli",
                "experiment",
                "--experiment-config", str(config_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        # Verify CSV
        csv_path = output_dir / "e2e_test.csv"
        assert csv_path.exists(), f"CSV not found at {csv_path}"

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2  # 2 profiles × 1 config × 1 rep

        # Verify conversation JSONs
        conv_dir = output_dir / "conversations"
        json_files = list(conv_dir.glob("conv_*.json"))
        assert len(json_files) == 2
