"""Shared utilities: logging setup, watermarking, formatting, and I/O helpers."""

import json
import logging
import uuid
from pathlib import Path

from pydantic import BaseModel

from phishagent.models import Message


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def generate_conversation_id() -> str:
    """Return a UUID4 string for conversation identification."""
    return str(uuid.uuid4())


def watermark_text() -> str:
    """Return the standard watermark string for synthetic data."""
    return "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"


def format_conversation_for_display(messages: list[Message]) -> str:
    """Pretty-print a conversation for terminal output."""
    lines = []
    for msg in messages:
        role_label = "ATTACKER" if msg.role == "attacker" else "VICTIM"
        lines.append(f"[Turn {msg.turn_number}] {role_label}: {msg.content}")
    return "\n".join(lines)


def format_conversation_for_judge(messages: list[Message]) -> str:
    """Format conversation transcript for LLM judge scoring prompts."""
    lines = []
    for msg in messages:
        role_label = "Attacker" if msg.role == "attacker" else "Victim"
        lines.append(f"{role_label}: {msg.content}")
    return "\n".join(lines)


def safe_json_dump(obj: BaseModel, path: str) -> None:
    """Write a Pydantic model to JSON file with error handling."""
    logger = get_logger("utils")
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        filepath.write_text(obj.model_dump_json(indent=2))
        logger.debug(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise
