"""Tests for ExperimentRunner.run() with mocked agents — covers lines 65-131."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from phishagent.config import AppConfig
from phishagent.experiment_runner import ExperimentRunner
from phishagent.models import (
    AttackGoal,
    AttackerConfig,
    AttackStrategy,
    AttackerScenario,
    ConversationOutcome,
    ConversationResult,
    ConversationScore,
    ExperimentConfig,
    LLMResponse,
    Message,
)


def _make_mock_llm():
    """Create a mock LLM that returns deterministic responses."""
    mock = MagicMock()
    mock.chat.return_value = LLMResponse(content="Mock chat response", model="test", token_count=5)
    mock.generate.return_value = LLMResponse(
        content="SCORE: 5\nRATIONALE: Average.", model="test", token_count=10
    )
    return mock


class TestExperimentRunnerRun:
    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_single_conversation(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        config = AppConfig()
        mock_llm = _make_mock_llm()
        runner = ExperimentRunner(config, llm=mock_llm)

        # Setup mock engine to return a result
        mock_result = ConversationResult(
            conversation_id="mock-001",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=[
                Message(role="attacker", content="Hi", turn_number=0),
                Message(role="victim", content="Hello", turn_number=1),
            ],
            outcome=ConversationOutcome.MAX_TURNS,
            model_name="test",
        )
        MockEngine.return_value.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_run",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=1,
                output_dir=tmpdir,
            )

            result = runner.run(exp_config)

        assert result.experiment_id == "test_run"
        assert result.total_conversations == 1
        assert result.completed_conversations == 1
        assert result.failed_conversations == 0
        assert len(result.results) == 1

    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_multiple_repetitions(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        config = AppConfig()
        mock_llm = _make_mock_llm()
        runner = ExperimentRunner(config, llm=mock_llm)

        mock_result = ConversationResult(
            conversation_id="mock-002",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=[Message(role="attacker", content="Hi", turn_number=0)],
            outcome=ConversationOutcome.MAX_TURNS,
            model_name="test",
        )
        MockEngine.return_value.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_reps",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=3,
                output_dir=tmpdir,
            )

            result = runner.run(exp_config)

        assert result.total_conversations == 3
        assert result.completed_conversations == 3

    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_with_progress_callback(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        config = AppConfig()
        mock_llm = _make_mock_llm()
        runner = ExperimentRunner(config, llm=mock_llm)

        mock_result = ConversationResult(
            conversation_id="mock-003",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=[Message(role="attacker", content="Hi", turn_number=0)],
            outcome=ConversationOutcome.MAX_TURNS,
            model_name="test",
        )
        MockEngine.return_value.run.return_value = mock_result

        callback_calls = []

        def track_progress(completed, total, last_result):
            callback_calls.append((completed, total, last_result))

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_cb",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=2,
                output_dir=tmpdir,
            )

            runner.run(exp_config, progress_callback=track_progress)

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1  # completed=1
        assert callback_calls[1][0] == 2  # completed=2

    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_handles_conversation_failure(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        config = AppConfig()
        mock_llm = _make_mock_llm()
        runner = ExperimentRunner(config, llm=mock_llm)

        # Engine raises exception
        MockEngine.return_value.run.side_effect = RuntimeError("LLM exploded")

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_fail",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=1,
                output_dir=tmpdir,
            )

            result = runner.run(exp_config)

        assert result.failed_conversations == 1
        assert result.completed_conversations == 0

    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_handles_scoring_failure(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        """Scoring failure should not crash the run — just skip scoring."""
        config = AppConfig()
        mock_llm = _make_mock_llm()
        # Make generate raise for scoring calls
        mock_llm.generate.side_effect = Exception("Scoring failed")
        runner = ExperimentRunner(config, llm=mock_llm)

        mock_result = ConversationResult(
            conversation_id="mock-score-fail",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=[Message(role="attacker", content="Hi", turn_number=0)],
            outcome=ConversationOutcome.MAX_TURNS,
            model_name="test",
        )
        MockEngine.return_value.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_score_fail",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=1,
                output_dir=tmpdir,
            )

            result = runner.run(exp_config)

        assert result.completed_conversations == 1
        # Scorer handles errors gracefully by returning default 0.5 scores
        assert result.results[0].scores is not None
        assert result.results[0].scores.persuasion == 0.5

    @patch("phishagent.experiment_runner.ConversationEngine")
    @patch("phishagent.experiment_runner.VictimAgent")
    @patch("phishagent.experiment_runner.AttackerAgent")
    def test_run_saves_conversations(self, MockAttacker, MockVictim, MockEngine, sample_profile, sample_attacker_config):
        config = AppConfig()
        mock_llm = _make_mock_llm()
        runner = ExperimentRunner(config, llm=mock_llm)

        mock_result = ConversationResult(
            conversation_id="mock-save",
            victim_profile=sample_profile,
            attacker_config=sample_attacker_config,
            messages=[Message(role="attacker", content="Hi", turn_number=0)],
            outcome=ConversationOutcome.MAX_TURNS,
            model_name="test",
        )
        MockEngine.return_value.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_id="test_save",
                description="test",
                model_name="test",
                profiles=[sample_profile],
                attacker_configs=[sample_attacker_config],
                repetitions=1,
                output_dir=tmpdir,
            )

            runner.run(exp_config)

            # Verify conversation JSON was saved
            conv_dir = Path(tmpdir) / "conversations"
            json_files = list(conv_dir.glob("conv_*.json"))
            assert len(json_files) == 1


class TestExportConversations:
    def test_export_conversations(self, sample_profile, sample_attacker_config):
        config = AppConfig()
        runner = ExperimentRunner(config, llm=None)

        from phishagent.models import ExperimentResult
        results = [
            ConversationResult(
                conversation_id=f"export-{i}",
                victim_profile=sample_profile,
                attacker_config=sample_attacker_config,
                messages=[Message(role="attacker", content="Hi", turn_number=0)],
                outcome=ConversationOutcome.MAX_TURNS,
                model_name="test",
            )
            for i in range(3)
        ]

        exp_result = ExperimentResult(
            experiment_id="test_export",
            total_conversations=3,
            completed_conversations=3,
            failed_conversations=0,
            results=results,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.export_conversations(exp_result, tmpdir)
            json_files = list(Path(tmpdir).glob("conv_*.json"))
            assert len(json_files) == 3
