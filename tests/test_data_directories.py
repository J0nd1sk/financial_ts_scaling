"""Tests for data directory structure.

These tests verify that the expected data directories exist.
"""

from pathlib import Path

import pytest


def test_data_directories_exist():
    """Test that required data directories exist."""
    project_root = Path(__file__).parent.parent

    # Required directories
    required_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "samples",
    ]

    for dir_path in required_dirs:
        assert dir_path.exists(), f"Directory does not exist: {dir_path}"
        assert dir_path.is_dir(), f"Path is not a directory: {dir_path}"
