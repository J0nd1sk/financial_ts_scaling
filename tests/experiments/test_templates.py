"""Tests for experiment script templates.

Tests template generation and validates generated scripts compile correctly.
Uses py_compile for syntax validation.
"""

from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.templates import (
    generate_final_training_script,
    generate_hpo_script,
    generate_training_script,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hpo_params():
    """Standard parameters for HPO script generation."""
    return {
        "experiment": "phase6a_2M_threshold_1pct",
        "phase": "phase6a",
        "budget": "2M",
        "task": "threshold_1pct",
        "horizon": 1,
        "timescale": "daily",
        "data_path": "data/processed/SPY_features_a20.parquet",
        "feature_columns": ["Open", "High", "Low", "Close", "Volume", "RSI_14"],
        "n_trials": 50,
        "timeout_hours": 4.0,
    }


@pytest.fixture
def training_params():
    """Standard parameters for training script generation."""
    return {
        "experiment": "phase6a_2M_threshold_1pct",
        "phase": "phase6a",
        "budget": "2M",
        "task": "threshold_1pct",
        "horizon": 1,
        "timescale": "daily",
        "data_path": "data/processed/SPY_features_a20.parquet",
        "feature_columns": ["Open", "High", "Low", "Close", "Volume", "RSI_14"],
        "hyperparameters": {"learning_rate": 0.001, "epochs": 50, "batch_size": 32},
        "borrowed_from": None,
    }


@pytest.fixture
def final_training_params():
    """Standard parameters for final training script generation."""
    return {
        "experiment": "phase6a_final_2M_h1",
        "phase": "phase6a",
        "budget": "2M",
        "horizon": 1,
        # Architecture (fixed from HPO)
        "d_model": 64,
        "n_layers": 48,
        "n_heads": 2,
        "d_ff": 256,  # 4 * d_model
        # Training params (fixed from HPO)
        "learning_rate": 0.0008,
        "dropout": 0.12,
        "weight_decay": 0.001,
        "warmup_steps": 100,
        "epochs": 50,
        # Data
        "data_path": "data/processed/v1/SPY_dataset_a20.parquet",
        "feature_columns": ["Open", "High", "Low", "Close", "Volume", "RSI_14"],
        # Optional
        "early_stopping_patience": 10,
    }


# =============================================================================
# Test: generate_hpo_script - Basic
# =============================================================================


class TestGenerateHpoScriptBasic:
    """Tests for basic generate_hpo_script functionality."""

    def test_generate_hpo_script_returns_string(self, hpo_params):
        """Test that generate_hpo_script returns a non-empty string."""
        result = generate_hpo_script(**hpo_params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_hpo_script_compiles(self, hpo_params):
        """Test that generated HPO script is valid Python syntax."""
        script = generate_hpo_script(**hpo_params)

        # Write to temp file and compile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            # py_compile.compile raises if syntax is invalid
            py_compile.compile(temp_path, doraise=True)
        finally:
            Path(temp_path).unlink()


class TestGenerateHpoScriptContent:
    """Tests for HPO script content."""

    def test_generate_hpo_script_contains_parameters(self, hpo_params):
        """Test that script contains visible experiment parameters."""
        script = generate_hpo_script(**hpo_params)

        assert 'BUDGET = "2M"' in script
        assert 'TASK = "threshold_1pct"' in script
        assert "EXPERIMENT = " in script
        assert "phase6a_2M_threshold_1pct" in script

    def test_generate_hpo_script_has_validation_function(self, hpo_params):
        """Test that script contains data validation function."""
        script = generate_hpo_script(**hpo_params)

        assert "def validate_data():" in script
        assert "pd.read_parquet" in script

    def test_generate_hpo_script_has_hpo_settings(self, hpo_params):
        """Test that script contains HPO-specific settings."""
        script = generate_hpo_script(**hpo_params)

        assert "N_TRIALS = 50" in script
        assert "TIMEOUT_HOURS = None" in script  # No timeout - experiments run to completion


# =============================================================================
# Test: generate_training_script - Basic
# =============================================================================


class TestGenerateTrainingScriptBasic:
    """Tests for basic generate_training_script functionality."""

    def test_generate_training_script_returns_string(self, training_params):
        """Test that generate_training_script returns a non-empty string."""
        result = generate_training_script(**training_params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_training_script_compiles(self, training_params):
        """Test that generated training script is valid Python syntax."""
        script = generate_training_script(**training_params)

        # Write to temp file and compile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            py_compile.compile(temp_path, doraise=True)
        finally:
            Path(temp_path).unlink()


class TestGenerateTrainingScriptContent:
    """Tests for training script content."""

    def test_generate_training_script_contains_hyperparameters(self, training_params):
        """Test that script contains hyperparameters dict."""
        script = generate_training_script(**training_params)

        assert "HYPERPARAMETERS = " in script
        assert "learning_rate" in script
        assert "0.001" in script

    def test_generate_training_script_has_borrowed_from(self, training_params):
        """Test that script contains BORROWED_FROM variable for 2% task support."""
        script = generate_training_script(**training_params)

        assert "BORROWED_FROM = " in script

    def test_generate_training_script_has_validation_function(self, training_params):
        """Test that training script also has data validation."""
        script = generate_training_script(**training_params)

        assert "def validate_data():" in script


# =============================================================================
# Test: generate_hpo_script - Architectural HPO
# =============================================================================


class TestGenerateHpoScriptArchitecture:
    """Tests for architectural HPO in generated scripts.

    Phase 6A requires HPO to search both architecture (d_model, n_layers, etc.)
    AND training parameters. These tests verify the template generates scripts
    that properly set up architectural search.
    """

    def test_hpo_script_imports_arch_grid(self, hpo_params):
        """Test that script imports get_architectures_for_budget from arch_grid."""
        script = generate_hpo_script(**hpo_params)

        assert "from src.models.arch_grid import get_architectures_for_budget" in script

    def test_hpo_script_imports_architectural_objective(self, hpo_params):
        """Test that script imports create_architectural_objective from hpo."""
        script = generate_hpo_script(**hpo_params)

        # Check for import from hpo module and the function name (may be multi-line)
        assert "from src.training.hpo import" in script
        assert "create_architectural_objective" in script

    def test_hpo_script_computes_architecture_grid(self, hpo_params):
        """Test that script computes architecture grid for the budget."""
        script = generate_hpo_script(**hpo_params)

        # Should call get_architectures_for_budget with budget
        assert "get_architectures_for_budget(" in script
        # Should store result in ARCHITECTURES constant
        assert "ARCHITECTURES = " in script

    def test_hpo_script_loads_architectural_search_config(self, hpo_params):
        """Test that script loads architectural search config."""
        script = generate_hpo_script(**hpo_params)

        assert "architectural_search.yaml" in script

    def test_hpo_script_uses_architectural_objective(self, hpo_params):
        """Test that script uses create_architectural_objective for HPO."""
        script = generate_hpo_script(**hpo_params)

        # Should call create_architectural_objective to create the objective
        assert "create_architectural_objective(" in script


# =============================================================================
# Test: generate_hpo_script - Thermal Callback (Task B)
# =============================================================================


class TestGenerateHpoScriptThermal:
    """Tests for thermal callback integration in generated HPO scripts.

    Task B: Generated HPO scripts should include ThermalCallback to monitor
    temperature between trials and pause/abort if thresholds exceeded.
    """

    def test_hpo_script_imports_thermal_callback(self, hpo_params):
        """Test that script imports ThermalCallback from thermal module."""
        script = generate_hpo_script(**hpo_params)

        assert "from src.training.thermal import ThermalCallback" in script

    def test_hpo_script_imports_time(self, hpo_params):
        """Test that script imports time module for thermal pause."""
        script = generate_hpo_script(**hpo_params)

        assert "import time" in script

    def test_hpo_script_creates_thermal_callback(self, hpo_params):
        """Test that script creates ThermalCallback instance."""
        script = generate_hpo_script(**hpo_params)

        # Should create ThermalCallback instance
        assert "thermal_callback = ThermalCallback()" in script

    def test_hpo_script_has_thermal_check_callback(self, hpo_params):
        """Test that script defines thermal_check_callback function."""
        script = generate_hpo_script(**hpo_params)

        # Should have callback function that checks thermal status
        assert "def thermal_check_callback(" in script
        assert "thermal_callback.check()" in script

    def test_hpo_script_passes_callbacks_to_optimize(self, hpo_params):
        """Test that script passes thermal and incremental logging callbacks to study.optimize()."""
        script = generate_hpo_script(**hpo_params)

        # Should pass both thermal and incremental logging callbacks to optimize
        assert "callbacks=[thermal_check_callback, incremental_logging_callback]" in script


# =============================================================================
# Test: generate_final_training_script - Basic
# =============================================================================


class TestGenerateFinalTrainingScriptBasic:
    """Tests for generate_final_training_script functionality.

    Final training scripts use fixed architecture and training parameters
    from HPO, with contiguous splits for production-realistic evaluation.
    """

    def test_generate_final_training_script_returns_string(self, final_training_params):
        """Test that generate_final_training_script returns a non-empty string."""
        result = generate_final_training_script(**final_training_params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_final_training_script_compiles(self, final_training_params):
        """Test that generated final training script is valid Python syntax."""
        script = generate_final_training_script(**final_training_params)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            py_compile.compile(temp_path, doraise=True)
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Test: generate_final_training_script - Content
# =============================================================================


class TestGenerateFinalTrainingScriptContent:
    """Tests for final training script content."""

    def test_generate_final_training_script_contains_experiment_name(
        self, final_training_params
    ):
        """Test that script contains experiment name as visible constant."""
        script = generate_final_training_script(**final_training_params)

        assert 'EXPERIMENT = "phase6a_final_2M_h1"' in script

    def test_generate_final_training_script_contains_architecture_params(
        self, final_training_params
    ):
        """Test that script contains fixed architecture parameters from HPO."""
        script = generate_final_training_script(**final_training_params)

        # Architecture params should be visible (either as kwargs or constants)
        assert "d_model=64" in script or "D_MODEL = 64" in script
        assert "n_layers=48" in script or "N_LAYERS = 48" in script
        assert "n_heads=2" in script or "N_HEADS = 2" in script

    def test_generate_final_training_script_uses_simple_splitter(
        self, final_training_params
    ):
        """Test that script uses SimpleSplitter with date-based splits for production-realistic evaluation."""
        script = generate_final_training_script(**final_training_params)

        # Should use SimpleSplitter instead of ChunkSplitter
        assert "SimpleSplitter" in script
        assert 'val_start="2023-01-01"' in script
        assert 'test_start="2025-01-01"' in script
