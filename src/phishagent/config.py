"""Configuration loading and validation.

Resolution order: defaults → YAML file → environment variables (PHISHAGENT_*) → CLI overrides.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from phishagent.utils import get_logger

logger = get_logger(__name__)


class ModelConfig(BaseModel):
    name: str = "mistral:7b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: int = 120
    num_gpu: Optional[int] = None  # None = auto (all layers on GPU when CUDA available), 0 = CPU


class ConversationConfig(BaseModel):
    max_turns: int = 10
    turn_delay_seconds: float = 0.0
    early_termination: bool = True


class ScoringConfig(BaseModel):
    judge_model: str = "llama3.2"
    persuasion_weight: float = 0.4
    coherence_weight: float = 0.3
    detectability_weight: float = 0.3
    judge_temperature: float = 0.1


class OutputConfig(BaseModel):
    base_dir: str = "output"
    save_conversations: bool = True
    save_csv: bool = True
    conversation_format: str = "json"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class AppConfig(BaseModel):
    """Application-wide configuration. Loaded from YAML, overridable by env vars."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Environment variable mapping: PHISHAGENT_SECTION_FIELD → config.section.field
_ENV_MAP = {
    "PHISHAGENT_MODEL_NAME": ("model", "name"),
    "PHISHAGENT_OLLAMA_URL": ("model", "ollama_url"),
    "PHISHAGENT_MODEL_TEMPERATURE": ("model", "temperature"),
    "PHISHAGENT_MODEL_MAX_TOKENS": ("model", "max_tokens"),
    "PHISHAGENT_MODEL_TIMEOUT_SECONDS": ("model", "timeout_seconds"),
    "PHISHAGENT_NUM_GPU": ("model", "num_gpu"),
    "PHISHAGENT_CONVERSATION_MAX_TURNS": ("conversation", "max_turns"),
    "PHISHAGENT_SCORING_JUDGE_MODEL": ("scoring", "judge_model"),
    "PHISHAGENT_OUTPUT_BASE_DIR": ("output", "base_dir"),
    "PHISHAGENT_LOGGING_LEVEL": ("logging", "level"),
    "PHISHAGENT_LOGGING_FILE": ("logging", "file"),
}


def _apply_env_overrides(data: dict) -> dict:
    """Apply environment variable overrides to the config dictionary."""
    for env_var, (section, field) in _ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in data:
                data[section] = {}
            # Attempt type coercion for numeric fields
            try:
                if field in ("temperature", "judge_temperature", "turn_delay_seconds"):
                    data[section][field] = float(value)
                elif field in ("max_tokens", "timeout_seconds", "max_turns", "num_gpu"):
                    data[section][field] = int(value)
                elif field in ("early_termination", "save_conversations", "save_csv"):
                    data[section][field] = value.lower() in ("true", "1", "yes")
                else:
                    data[section][field] = value
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {env_var}={value!r}, skipping")
                continue
            logger.debug(f"Env override: {env_var} → {section}.{field} = {value}")
    return data


def load_config(path: str = "config/default.yaml", overrides: dict | None = None) -> AppConfig:
    """Load config from YAML file, apply env var overrides (PHISHAGENT_*), validate."""
    data: dict = {}

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
    else:
        logger.warning(f"Config file not found at {path}, using defaults")

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    # Apply explicit overrides (from CLI args, etc.)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                section, field = parts
                if section not in data:
                    data[section] = {}
                data[section][field] = value

    config = AppConfig(**data)

    # Configure root logger based on config
    import logging as _logging

    root_logger = _logging.getLogger("phishagent")
    root_logger.setLevel(getattr(_logging, config.logging.level.upper(), _logging.INFO))
    if config.logging.file:
        file_handler = _logging.FileHandler(config.logging.file)
        file_handler.setFormatter(_logging.Formatter(config.logging.format))
        root_logger.addHandler(file_handler)

    return config
