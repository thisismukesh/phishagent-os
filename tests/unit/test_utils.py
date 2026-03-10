"""Unit tests for utils.py — logging, formatting, watermarking, JSON dump."""

import json
import tempfile
from pathlib import Path

import pytest

from phishagent.models import Message, PersonalityTraits, VictimProfile, CommunicationStyle, SecurityAwareness
from phishagent.utils import (
    format_conversation_for_display,
    format_conversation_for_judge,
    generate_conversation_id,
    get_logger,
    safe_json_dump,
    watermark_text,
)


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_has_handler(self):
        logger = get_logger("test_handler_check")
        assert len(logger.handlers) >= 1

    def test_idempotent(self):
        """Calling get_logger twice with same name doesn't duplicate handlers."""
        logger1 = get_logger("test_idempotent")
        count1 = len(logger1.handlers)
        logger2 = get_logger("test_idempotent")
        count2 = len(logger2.handlers)
        assert count1 == count2


class TestGenerateConversationId:
    def test_returns_string(self):
        cid = generate_conversation_id()
        assert isinstance(cid, str)

    def test_unique(self):
        ids = {generate_conversation_id() for _ in range(100)}
        assert len(ids) == 100

    def test_uuid_format(self):
        cid = generate_conversation_id()
        # UUID4 format: 8-4-4-4-12 hex characters
        parts = cid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8


class TestWatermarkText:
    def test_returns_expected(self):
        assert watermark_text() == "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"


class TestFormatConversationForDisplay:
    def test_basic_formatting(self):
        messages = [
            Message(role="attacker", content="Hey there!", turn_number=0),
            Message(role="victim", content="Hi, who is this?", turn_number=1),
        ]
        result = format_conversation_for_display(messages)
        assert "[Turn 0] ATTACKER: Hey there!" in result
        assert "[Turn 1] VICTIM: Hi, who is this?" in result

    def test_empty_messages(self):
        result = format_conversation_for_display([])
        assert result == ""

    def test_multiline_content(self):
        messages = [
            Message(role="attacker", content="Line 1\nLine 2", turn_number=0),
        ]
        result = format_conversation_for_display(messages)
        assert "Line 1\nLine 2" in result


class TestFormatConversationForJudge:
    def test_basic_formatting(self):
        messages = [
            Message(role="attacker", content="Click this link", turn_number=0),
            Message(role="victim", content="Why should I?", turn_number=1),
        ]
        result = format_conversation_for_judge(messages)
        assert "Attacker: Click this link" in result
        assert "Victim: Why should I?" in result

    def test_empty_messages(self):
        result = format_conversation_for_judge([])
        assert result == ""


class TestSafeJsonDump:
    def test_writes_valid_json(self):
        from phishagent.models import LLMResponse

        obj = LLMResponse(content="test", model="m")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.json"
            safe_json_dump(obj, path)
            with open(path) as f:
                data = json.load(f)
            assert data["content"] == "test"

    def test_creates_parent_dirs(self):
        from phishagent.models import LLMResponse

        obj = LLMResponse(content="test", model="m")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/nested/deep/test.json"
            safe_json_dump(obj, path)
            assert Path(path).exists()

    def test_raises_on_write_error(self):
        from phishagent.models import LLMResponse

        obj = LLMResponse(content="test", model="m")
        # Try to write to a path that can't exist
        with pytest.raises(Exception):
            safe_json_dump(obj, "/dev/null/impossible/path.json")
