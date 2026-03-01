"""
Integration test - requires GOOGLE_API_KEY or OPENAI_API_KEY set.
Run with: pytest tests/test_integration.py -v -s
"""
import os
import pytest
from pathlib import Path
from click.testing import CliRunner

from apo.cli.commands import cli


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="No API key available",
)
class TestIntegration:
    def test_evaluate_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--prompt", "Answer concisely: {input}",
            "--dataset", str(Path(__file__).parent.parent / "examples" / "sample_dataset.json"),
            "--model", "gemini/gemini-2.0-flash",
            "--eval-mode", "reference",
        ])
        assert result.exit_code == 0
        assert "Average score" in result.output

    def test_optimize_minimal(self):
        """Minimal optimization with small beam to verify the pipeline works."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "optimize",
            "--prompt", "Answer: {input}",
            "--dataset", str(Path(__file__).parent.parent / "examples" / "sample_dataset.json"),
            "--model", "gemini/gemini-2.0-flash",
            "--beam-width", "2",
            "--beam-rounds", "1",
            "--n-runners", "2",
            "--eval-mode", "reference",
        ])
        assert result.exit_code == 0
        assert "Optimization complete" in result.output
