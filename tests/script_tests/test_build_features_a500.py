"""Tests for build_features_a500 script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class TestBuildFeaturesA500Script:
    """Tests for the build_features_a500.py script."""

    def test_script_exists(self) -> None:
        """Test that the build script exists."""
        script_path = Path(__file__).parents[2] / "scripts" / "build_features_a500.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_script_is_importable(self) -> None:
        """Test that the script's main function can be imported."""
        # Import the script module
        import importlib.util

        script_path = Path(__file__).parents[2] / "scripts" / "build_features_a500.py"
        spec = importlib.util.spec_from_file_location("build_features_a500", script_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check main function exists
        assert hasattr(module, "main"), "Script should have a main() function"
        assert callable(module.main)

    def test_script_help_option(self) -> None:
        """Test that the script responds to --help."""
        script_path = Path(__file__).parents[2] / "scripts" / "build_features_a500.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "ticker" in result.stdout.lower(), "Help should mention --ticker"
        assert "raw-path" in result.stdout.lower(), "Help should mention --raw-path"
        assert "vix-path" in result.stdout.lower(), "Help should mention --vix-path"
        assert "output-path" in result.stdout.lower(), "Help should mention --output-path"
