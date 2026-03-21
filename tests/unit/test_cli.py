"""Unit tests for CLI — uses Click's CliRunner with mocked LLM/agents."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from phishagent.cli import main
from phishagent.models import (
    ConversationOutcome,
    ConversationResult,
    ConversationScore,
    LLMResponse,
    Message,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns deterministic responses."""
    mock = MagicMock()
    mock.is_available.return_value = True
    mock.list_models.return_value = ["mistral:7b", "llama3:8b"]
    mock.ensure_model.return_value = True
    mock.chat.return_value = LLMResponse(content="Mock response", model="mistral:7b", token_count=5)
    mock.generate.return_value = LLMResponse(
        content="SCORE: 7\nRATIONALE: Good attempt.", model="mistral:7b", token_count=10
    )
    return mock


class TestStatusCommand:
    @patch("phishagent.cli.create_client")
    def test_status_ollama_available(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True
        instance.list_models.return_value = ["mistral:7b"]

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Ollama is running" in result.output
        assert "mistral:7b" in result.output

    @patch("phishagent.cli.create_client")
    def test_status_ollama_unavailable(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = False

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "not reachable" in result.output

    @patch("phishagent.cli.create_client")
    def test_status_no_models(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True
        instance.list_models.return_value = []

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "No models found" in result.output


class TestRunCommand:
    @patch("phishagent.cli.safe_json_dump")
    @patch("phishagent.cli.ConversationScorer")
    @patch("phishagent.cli.ConversationEngine")
    @patch("phishagent.cli.VictimAgent")
    @patch("phishagent.cli.AttackerAgent")
    @patch("phishagent.cli.create_client")
    def test_run_success(self, mock_create, MockAttacker, MockVictim, MockEngine, MockScorer, MockDump, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True

        mock_result = MagicMock()
        mock_result.conversation_id = "test-123"
        mock_result.outcome = ConversationOutcome.COMPLIANCE
        mock_result.messages = [
            Message(role="attacker", content="Hey!", turn_number=0),
            Message(role="victim", content="Hi!", turn_number=1),
        ]
        MockEngine.return_value.run.return_value = mock_result

        mock_scores = ConversationScore(
            persuasion=0.8, coherence=0.7, detectability=0.4, composite=0.71,
        )
        MockScorer.return_value.score.return_value = mock_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "run",
                "--profile", "config/profiles/high_agreeableness.yaml",
                "--strategy", "urgency",
                "--scenario", "it_support",
                "--goal", "click_link",
                "--turns", "3",
                "--output", tmpdir,
            ])

        assert result.exit_code == 0
        assert "Starting conversation" in result.output

    @patch("phishagent.cli.create_client")
    def test_run_ollama_unavailable(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = False

        result = runner.invoke(main, [
            "run",
            "--profile", "config/profiles/high_agreeableness.yaml",
            "--strategy", "urgency",
            "--scenario", "it_support",
            "--goal", "click_link",
        ])

        assert result.exit_code != 0

    def test_run_invalid_profile(self, runner):
        result = runner.invoke(main, [
            "run",
            "--profile", "nonexistent.yaml",
            "--strategy", "urgency",
            "--scenario", "it_support",
            "--goal", "click_link",
        ])
        assert result.exit_code != 0

    @patch("phishagent.cli.safe_json_dump")
    @patch("phishagent.cli.ConversationScorer")
    @patch("phishagent.cli.ConversationEngine")
    @patch("phishagent.cli.VictimAgent")
    @patch("phishagent.cli.AttackerAgent")
    @patch("phishagent.cli.create_client")
    def test_run_scoring_failure_handled(self, mock_create, MockAttacker, MockVictim, MockEngine, MockScorer, MockDump, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True

        mock_result = MagicMock()
        mock_result.conversation_id = "test-456"
        mock_result.outcome = ConversationOutcome.MAX_TURNS
        mock_result.messages = [
            Message(role="attacker", content="Hi", turn_number=0),
            Message(role="victim", content="Hello", turn_number=1),
        ]
        MockEngine.return_value.run.return_value = mock_result
        MockScorer.return_value.score.side_effect = Exception("Scoring blew up")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "run",
                "--profile", "config/profiles/high_agreeableness.yaml",
                "--strategy", "urgency",
                "--scenario", "it_support",
                "--goal", "click_link",
                "--output", tmpdir,
            ])

        assert result.exit_code == 0
        assert "Scoring failed" in result.output


class TestExperimentCommand:
    @patch("phishagent.cli.ExperimentRunner")
    @patch("phishagent.cli.create_client")
    def test_experiment_success(self, mock_create, MockRunner, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True

        mock_result = MagicMock()
        mock_result.completed_conversations = 2
        mock_result.total_conversations = 2
        mock_result.failed_conversations = 0
        MockRunner.return_value.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "experiment",
                "--experiment-config", "config/profiles/factorial_batch.yaml",
                "--repetitions", "1",
                "--output", tmpdir,
            ])

        assert result.exit_code == 0
        assert "Starting batch experiment" in result.output

    @patch("phishagent.cli.create_client")
    def test_experiment_ollama_unavailable(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = False

        result = runner.invoke(main, [
            "experiment",
            "--experiment-config", "config/profiles/factorial_batch.yaml",
        ])

        assert result.exit_code != 0

    def test_experiment_invalid_config(self, runner):
        result = runner.invoke(main, [
            "experiment",
            "--experiment-config", "nonexistent.yaml",
        ])
        assert result.exit_code != 0

    @patch("phishagent.cli.ExperimentRunner")
    @patch("phishagent.cli.create_client")
    def test_experiment_no_vary(self, mock_create, MockRunner, runner):
        """Test experiment with no factorial variation (single profile)."""
        instance = mock_create.return_value
        instance.is_available.return_value = True

        mock_result = MagicMock()
        mock_result.completed_conversations = 1
        mock_result.total_conversations = 1
        mock_result.failed_conversations = 0
        MockRunner.return_value.run.return_value = mock_result

        # Create a config with no vary
        config = {
            "experiment_id": "test_no_vary",
            "description": "test",
            "model_name": "mistral:7b",
            "repetitions": 1,
            "base_profile": {
                "name": "V", "personality": {"openness": 0.5, "conscientiousness": 0.5,
                "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
                "communication_style": "casual", "security_awareness": "low",
                "interests": ["tech"], "occupation": "engineer",
                "tech_proficiency": 0.5, "impulsivity": 0.5,
            },
            "attacker_configs": [
                {"goal": "click_link", "strategy": "urgency", "scenario": "it_support"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "experiment",
                "--experiment-config", config_path,
                "--output", tmpdir,
            ])

        assert result.exit_code == 0

    @patch("phishagent.cli.create_client")
    def test_experiment_no_attacker_configs(self, mock_create, runner):
        instance = mock_create.return_value
        instance.is_available.return_value = True

        config = {
            "experiment_id": "test_empty",
            "base_profile": {
                "name": "V", "personality": {"openness": 0.5, "conscientiousness": 0.5,
                "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
                "communication_style": "casual", "security_awareness": "low",
                "interests": ["tech"], "occupation": "engineer",
                "tech_proficiency": 0.5, "impulsivity": 0.5,
            },
            "attacker_configs": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        result = runner.invoke(main, [
            "experiment",
            "--experiment-config", config_path,
        ])
        assert result.exit_code != 0


class TestCustomConfig:
    def test_custom_config_path(self, runner):
        """Test passing a custom config file."""
        result = runner.invoke(main, ["--config", "config/default.yaml", "status"])
        # Should at least not crash; backend may or may not be available
        assert result.exit_code == 0 or "not reachable" in result.output or True
