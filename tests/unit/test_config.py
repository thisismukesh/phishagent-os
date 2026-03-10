"""Unit tests for configuration loading."""

import os
import tempfile

import pytest
import yaml

from phishagent.config import AppConfig, load_config


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.model.name == "mistral:7b"
        assert config.model.ollama_url == "http://localhost:11434"
        assert config.conversation.max_turns == 10
        assert config.scoring.persuasion_weight == 0.4

    def test_load_from_yaml(self):
        config = load_config("config/default.yaml")
        assert config.model.name == "mistral:7b"
        assert config.scoring.judge_temperature == 0.1

    def test_missing_file_uses_defaults(self):
        config = load_config("nonexistent/path.yaml")
        assert config.model.name == "mistral:7b"

    def test_partial_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": {"name": "llama3:8b"}}, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.model.name == "llama3:8b"
        # Defaults for unspecified fields
        assert config.model.temperature == 0.7

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("PHISHAGENT_MODEL_NAME", "llama3:8b")
        config = load_config("config/default.yaml")
        assert config.model.name == "llama3:8b"

    def test_env_var_numeric_override(self, monkeypatch):
        monkeypatch.setenv("PHISHAGENT_MODEL_MAX_TOKENS", "1024")
        config = load_config("config/default.yaml")
        assert config.model.max_tokens == 1024

    def test_explicit_overrides(self):
        config = load_config(
            "config/default.yaml",
            overrides={"model.name": "phi3:mini", "conversation.max_turns": 20},
        )
        assert config.model.name == "phi3:mini"
        assert config.conversation.max_turns == 20

    def test_scoring_weights_sum_to_one(self):
        config = AppConfig()
        total = (
            config.scoring.persuasion_weight
            + config.scoring.coherence_weight
            + config.scoring.detectability_weight
        )
        assert abs(total - 1.0) < 1e-9

    def test_empty_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.model.name == "mistral:7b"
