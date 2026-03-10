"""Unit tests for the interactive terminal module.

Tests focus on the pure helper functions (input validation, display logic)
that can be exercised without an LLM or real terminal prompts.
"""

from unittest.mock import patch

import pytest

from phishagent.interactive import (
    _prompt_float,
    _prompt_int_range,
    _prompt_string,
    build_profile_interactive,
)
from phishagent.models import (
    CommunicationStyle,
    SecurityAwareness,
    VictimProfile,
)


# ── _prompt_int_range ──────────────────────────────────────────────────────────


class TestPromptIntRange:
    def test_valid_input_returned(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="3"):
            assert _prompt_int_range("X", 1, 5) == 3

    def test_boundary_min_accepted(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="1"):
            assert _prompt_int_range("X", 1, 5) == 1

    def test_boundary_max_accepted(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="5"):
            assert _prompt_int_range("X", 1, 5) == 5

    def test_out_of_range_retries_then_accepts(self):
        # First call returns out-of-range, second call returns valid
        with patch("phishagent.interactive.Prompt.ask", side_effect=["0", "2"]):
            assert _prompt_int_range("X", 1, 5) == 2

    def test_non_numeric_retries_then_accepts(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=["abc", "4"]):
            assert _prompt_int_range("X", 1, 5) == 4

    def test_default_used_when_no_input(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="3"):
            result = _prompt_int_range("X", 1, 5, default=3)
        assert result == 3


# ── _prompt_float ──────────────────────────────────────────────────────────────


class TestPromptFloat:
    def test_valid_float_returned(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="0.7"):
            assert _prompt_float("X") == 0.7

    def test_zero_accepted(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="0.0"):
            assert _prompt_float("X") == 0.0

    def test_one_accepted(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="1.0"):
            assert _prompt_float("X") == 1.0

    def test_above_max_retries(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=["1.5", "0.5"]):
            assert _prompt_float("X") == 0.5

    def test_below_min_retries(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=["-0.1", "0.3"]):
            assert _prompt_float("X") == 0.3

    def test_non_numeric_retries(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=["high", "0.8"]):
            assert _prompt_float("X") == 0.8

    def test_result_rounded_to_3dp(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="0.123456"):
            assert _prompt_float("X") == 0.123


# ── _prompt_string ─────────────────────────────────────────────────────────────


class TestPromptString:
    def test_valid_string_returned(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="hello"):
            assert _prompt_string("X") == "hello"

    def test_whitespace_only_retries_when_required(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=["   ", "ok"]):
            assert _prompt_string("X", required=True) == "ok"

    def test_blank_ok_when_not_required(self):
        with patch("phishagent.interactive.Prompt.ask", return_value=""):
            assert _prompt_string("X", required=False) == ""

    def test_strips_trailing_whitespace(self):
        with patch("phishagent.interactive.Prompt.ask", return_value="  value  "):
            assert _prompt_string("X") == "value"


# ── build_profile_interactive ──────────────────────────────────────────────────


class TestBuildProfileInteractive:
    """Test that build_profile_interactive assembles a valid VictimProfile
    when all prompts return sensible values."""

    def _make_side_effects(self):
        """Return a predictable sequence of Prompt.ask answers."""
        return [
            "Jordan Test",        # name
            "data analyst",       # occupation
            "gaming,music",       # interests
            # CommunicationStyle: 4 choices → pick 2 (casual)
            "2",
            # SecurityAwareness: 3 choices → pick 1 (low)
            "1",
            # Big Five (5 floats)
            "0.6", "0.4", "0.7", "0.8", "0.3",
            # tech_proficiency, impulsivity
            "0.9", "0.5",
        ]

    def test_returns_victim_profile(self):
        with patch("phishagent.interactive.Prompt.ask", side_effect=self._make_side_effects()):
            profile = build_profile_interactive()

        assert isinstance(profile, VictimProfile)
        assert profile.name == "Jordan Test"
        assert profile.occupation == "data analyst"
        assert profile.interests == ["gaming", "music"]
        assert profile.security_awareness == SecurityAwareness.LOW
        assert profile.communication_style == CommunicationStyle.CASUAL
        assert profile.personality.agreeableness == pytest.approx(0.8)
        assert profile.tech_proficiency == pytest.approx(0.9)

    def test_interests_capped_at_five(self):
        effects = self._make_side_effects()
        # Override interests input with 7 items
        effects[2] = "a,b,c,d,e,f,g"
        with patch("phishagent.interactive.Prompt.ask", side_effect=effects):
            profile = build_profile_interactive()

        assert len(profile.interests) == 5

    def test_empty_interests_defaults_to_technology(self):
        effects = self._make_side_effects()
        effects[2] = "   "  # blank interests
        with patch("phishagent.interactive.Prompt.ask", side_effect=effects):
            profile = build_profile_interactive()

        assert profile.interests == ["technology"]
