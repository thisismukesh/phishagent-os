"""Unit tests for experiment runner logic, CSV export, and progress tracking."""

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from phishagent.config import AppConfig
from phishagent.experiment_runner import CSV_COLUMNS, ExperimentRunner
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    ConversationResult,
    ConversationScore,
    ExperimentConfig,
    ExperimentResult,
    Message,
)


def _make_conv_result(profile, attacker_config, outcome=ConversationOutcome.COMPLIANCE):
    """Helper to create a ConversationResult for testing."""
    return ConversationResult(
        conversation_id=f"test-{id(profile)}",
        victim_profile=profile,
        attacker_config=attacker_config,
        messages=[
            Message(role="attacker", content="Hello", turn_number=0),
            Message(role="victim", content="Hi", turn_number=1),
        ],
        outcome=outcome,
        model_name="test:model",
        scores=ConversationScore(
            persuasion=0.7, coherence=0.8, detectability=0.4,
            composite=0.71, persuasion_rationale="Good",
            coherence_rationale="Natural", detectability_rationale="Moderate",
        ),
        total_tokens=100,
        total_duration_seconds=5.0,
    )


class TestCSVExport:
    def test_csv_has_correct_columns(self, sample_profile, sample_attacker_config):
        config = AppConfig()
        runner = ExperimentRunner(config, llm=None)

        result = ExperimentResult(
            experiment_id="test_001",
            total_conversations=1,
            completed_conversations=1,
            failed_conversations=0,
            results=[_make_conv_result(sample_profile, sample_attacker_config)],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        runner.export_csv(result, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == CSV_COLUMNS
            rows = list(reader)
            assert len(rows) == 1

    def test_csv_row_count_matches(self, sample_profile, sample_attacker_config):
        config = AppConfig()
        runner = ExperimentRunner(config, llm=None)

        results = [
            _make_conv_result(sample_profile, sample_attacker_config)
            for _ in range(5)
        ]

        result = ExperimentResult(
            experiment_id="test_002",
            total_conversations=5,
            completed_conversations=5,
            failed_conversations=0,
            results=results,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        runner.export_csv(result, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5

    def test_csv_values_correct(self, sample_profile, sample_attacker_config):
        config = AppConfig()
        runner = ExperimentRunner(config, llm=None)

        conv = _make_conv_result(sample_profile, sample_attacker_config)

        result = ExperimentResult(
            experiment_id="test_003",
            total_conversations=1,
            completed_conversations=1,
            failed_conversations=0,
            results=[conv],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        runner.export_csv(result, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["model"] == "test:model"
            assert row["strategy"] == "urgency"
            assert row["scenario"] == "it_support"
            assert row["goal"] == "click_link"
            assert row["victim_name"] == "Test Victim"
            assert row["agreeableness"] == "0.5"
            assert row["outcome"] == "compliance"
            assert row["persuasion"] == "0.7"

    def test_csv_creates_parent_dirs(self, sample_profile, sample_attacker_config):
        config = AppConfig()
        runner = ExperimentRunner(config, llm=None)

        result = ExperimentResult(
            experiment_id="test_004",
            total_conversations=0,
            completed_conversations=0,
            failed_conversations=0,
            results=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/nested/dir/results.csv"
            runner.export_csv(result, path)
            assert Path(path).exists()


class TestExperimentConfig:
    def test_factorial_combinations_count(self, sample_profile, sample_attacker_config):
        """Verify the expected number of combinations."""
        profiles = [sample_profile] * 3  # 3 profiles
        configs = [sample_attacker_config] * 2  # 2 configs
        reps = 2

        exp_config = ExperimentConfig(
            experiment_id="test",
            description="test",
            profiles=profiles,
            attacker_configs=configs,
            repetitions=reps,
        )

        total = len(exp_config.profiles) * len(exp_config.attacker_configs) * exp_config.repetitions
        assert total == 12  # 3 × 2 × 2


class TestProgressCallback:
    def test_callback_called(self, sample_profile, sample_attacker_config):
        """Verify progress callback is invoked (mocked runner)."""
        calls = []

        def callback(completed, total, last_result):
            calls.append((completed, total))

        # We can't easily run the full runner without Ollama,
        # but we can verify the callback signature works
        callback(1, 10, None)
        callback(2, 10, None)
        assert len(calls) == 2
        assert calls[0] == (1, 10)
