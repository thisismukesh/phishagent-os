"""E2E test: full CLI single-conversation run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestSingleRun:
    def test_cli_run_produces_output(self, tmp_path):
        """Invoke CLI run command and verify output JSON is created."""
        result = subprocess.run(
            [
                sys.executable, "-m", "phishagent.cli",
                "run",
                "--profile", "config/profiles/high_agreeableness.yaml",
                "--strategy", "urgency",
                "--scenario", "it_support",
                "--goal", "click_link",
                "--turns", "3",
                "--output", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        # Find the output JSON
        json_files = list(tmp_path.glob("conv_*.json"))
        assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"

        # Validate JSON structure
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "conversation_id" in data
        assert "messages" in data
        assert "watermark" in data
        assert data["watermark"] == "SYNTHETIC_RESEARCH_OUTPUT:PhishAgent-OS"
        assert len(data["messages"]) >= 2

    def test_cli_status(self):
        """Verify status command runs without error."""
        result = subprocess.run(
            [sys.executable, "-m", "phishagent.cli", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "PhishAgent-OS" in result.stdout
