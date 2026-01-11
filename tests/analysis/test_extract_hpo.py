"""Tests for HPO analysis extraction script."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# === Unit Tests: Core Functions ===


class TestLoadSingleTrial:
    """Tests for load_trial function."""

    def test_load_single_trial_returns_dict(self, sample_trial_json):
        """Test that load_trial returns a dict with expected keys."""
        from scripts.extract_hpo_analysis import load_trial

        result = load_trial(sample_trial_json)

        assert isinstance(result, dict)
        assert "study" in result
        assert "budget" in result
        assert "horizon" in result
        assert "trial_num" in result
        assert "val_loss" in result

    def test_load_single_trial_extracts_architecture(self, sample_trial_json):
        """Test that architecture fields are extracted correctly."""
        from scripts.extract_hpo_analysis import load_trial

        result = load_trial(sample_trial_json)

        assert result["d_model"] == 64
        assert result["n_layers"] == 32
        assert result["n_heads"] == 8
        assert result["d_ff"] == 256
        assert result["param_count"] == 1617729

    def test_load_single_trial_extracts_training_params(self, sample_trial_json):
        """Test that training parameters are extracted correctly."""
        from scripts.extract_hpo_analysis import load_trial

        result = load_trial(sample_trial_json)

        assert "lr" in result
        assert "epochs_config" in result
        assert "weight_decay" in result
        assert "warmup_steps" in result
        assert "dropout" in result

    def test_load_single_trial_parses_study_name(self, sample_trial_json):
        """Test that study name is parsed into budget and horizon."""
        from scripts.extract_hpo_analysis import load_trial

        result = load_trial(sample_trial_json)

        assert result["budget"] == "2M"
        assert result["horizon"] == "h1"


class TestComputeDerivedFields:
    """Tests for derived field computation."""

    def test_compute_best_epoch_from_learning_curve(self):
        """Test that best_epoch is the epoch with minimum val_loss."""
        from scripts.extract_hpo_analysis import compute_best_epoch

        learning_curve = [
            {"epoch": 0, "val_loss": 0.5},
            {"epoch": 1, "val_loss": 0.4},
            {"epoch": 2, "val_loss": 0.35},  # best
            {"epoch": 3, "val_loss": 0.38},
            {"epoch": 4, "val_loss": 0.40},
        ]

        best_epoch, best_val_loss = compute_best_epoch(learning_curve)

        assert best_epoch == 2
        assert best_val_loss == 0.35

    def test_compute_best_epoch_empty_curve(self):
        """Test that empty learning curve returns None."""
        from scripts.extract_hpo_analysis import compute_best_epoch

        best_epoch, best_val_loss = compute_best_epoch([])

        assert best_epoch is None
        assert best_val_loss is None

    def test_compute_early_stopped_flag_true(self):
        """Test early_stopped is True when epochs_trained < epochs_config."""
        from scripts.extract_hpo_analysis import compute_early_stopped

        result = compute_early_stopped(epochs_trained=35, epochs_config=100)

        assert result is True

    def test_compute_early_stopped_flag_false(self):
        """Test early_stopped is False when all epochs completed."""
        from scripts.extract_hpo_analysis import compute_early_stopped

        result = compute_early_stopped(epochs_trained=100, epochs_config=100)

        assert result is False

    def test_diverged_flag_threshold(self):
        """Test diverged flag is True when val_loss >= 10.0."""
        from scripts.extract_hpo_analysis import is_diverged

        assert is_diverged(100.0) is True
        assert is_diverged(10.0) is True
        assert is_diverged(9.99) is False
        assert is_diverged(0.35) is False


class TestComputeRanks:
    """Tests for rank computation within studies."""

    def test_compute_rank_in_study(self):
        """Test that rank 1 is assigned to best trial (lowest val_loss)."""
        from scripts.extract_hpo_analysis import compute_ranks

        trials = [
            {"study": "phase6a_2M_h1", "val_loss": 0.35},
            {"study": "phase6a_2M_h1", "val_loss": 0.32},  # best
            {"study": "phase6a_2M_h1", "val_loss": 0.40},
        ]

        ranked = compute_ranks(trials)

        # Find the trial with val_loss 0.32
        best_trial = next(t for t in ranked if t["val_loss"] == 0.32)
        assert best_trial["rank_in_study"] == 1
        assert best_trial["is_best_in_study"] is True

    def test_compute_rank_multiple_studies(self):
        """Test that ranks are computed per study."""
        from scripts.extract_hpo_analysis import compute_ranks

        trials = [
            {"study": "phase6a_2M_h1", "val_loss": 0.35},
            {"study": "phase6a_2M_h1", "val_loss": 0.32},
            {"study": "phase6a_2M_h3", "val_loss": 0.28},
            {"study": "phase6a_2M_h3", "val_loss": 0.30},
        ]

        ranked = compute_ranks(trials)

        # Best in h1 study
        h1_best = next(t for t in ranked if t["study"] == "phase6a_2M_h1" and t["val_loss"] == 0.32)
        assert h1_best["rank_in_study"] == 1

        # Best in h3 study
        h3_best = next(t for t in ranked if t["study"] == "phase6a_2M_h3" and t["val_loss"] == 0.28)
        assert h3_best["rank_in_study"] == 1


# === Integration Tests ===


class TestLoadAllTrials:
    """Integration tests for loading all trials."""

    def test_load_all_trials_returns_600_rows(self):
        """Test that all 600 trials are loaded from outputs/hpo."""
        from scripts.extract_hpo_analysis import load_all_trials

        base_dir = Path("outputs/hpo")
        trials = load_all_trials(base_dir)

        assert len(trials) == 600

    def test_diverged_filter_returns_14_rows(self):
        """Test that exactly 14 diverged trials are identified."""
        from scripts.extract_hpo_analysis import load_all_trials

        base_dir = Path("outputs/hpo")
        trials = load_all_trials(base_dir)
        diverged = [t for t in trials if t["diverged"]]

        assert len(diverged) == 14


class TestOutputGeneration:
    """Tests for output file generation."""

    def test_csv_output_has_required_columns(self, sample_trials):
        """Test that CSV output has all required columns."""
        from scripts.extract_hpo_analysis import write_summary_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            write_summary_csv(sample_trials, output_path)
            df = pd.read_csv(output_path)

            required_columns = [
                "study", "budget", "horizon", "trial_num",
                "d_model", "n_layers", "n_heads", "d_ff", "param_count",
                "lr", "epochs_config", "weight_decay", "warmup_steps", "dropout",
                "val_loss", "train_loss",
                "best_epoch", "best_val_loss", "epochs_trained", "early_stopped",
                "diverged", "rank_in_study", "is_best_in_study",
            ]

            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
        finally:
            output_path.unlink(missing_ok=True)

    def test_json_output_valid_structure(self, sample_trials):
        """Test that JSON output has valid structure."""
        from scripts.extract_hpo_analysis import write_full_json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            write_full_json(sample_trials, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "trials" in data
            assert data["metadata"]["n_trials"] == len(sample_trials)
        finally:
            output_path.unlink(missing_ok=True)


# === Fixtures ===


@pytest.fixture
def sample_trial_json(tmp_path):
    """Create a sample trial JSON file for testing."""
    study_dir = tmp_path / "phase6a_2M_h1_threshold_1pct" / "trials"
    study_dir.mkdir(parents=True)

    trial_data = {
        "trial_number": 0,
        "value": 0.3634,
        "params": {
            "learning_rate": 0.0005,
            "epochs": 100,
            "weight_decay": 0.0003,
            "warmup_steps": 100,
            "dropout": 0.14
        },
        "state": "COMPLETE",
        "datetime_start": "2025-12-30T19:33:09",
        "datetime_complete": "2025-12-30T19:33:42",
        "duration_seconds": 33.08,
        "architecture": {
            "d_model": 64,
            "n_layers": 32,
            "n_heads": 8,
            "d_ff": 256,
            "param_count": 1617729
        },
        "user_attrs": {
            "arch_idx": 2,
            "learning_curve": [
                {"epoch": 0, "train_loss": 0.48, "val_loss": 0.49},
                {"epoch": 1, "train_loss": 0.46, "val_loss": 0.44},
            ],
            "val_accuracy": 0.89,
            "train_accuracy": 0.86
        }
    }

    trial_path = study_dir / "trial_0000.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f)

    return trial_path


@pytest.fixture
def sample_trials():
    """Create sample trial dicts for output testing."""
    return [
        {
            "study": "phase6a_2M_h1_threshold_1pct",
            "budget": "2M",
            "horizon": "h1",
            "trial_num": 0,
            "d_model": 64,
            "n_layers": 32,
            "n_heads": 8,
            "d_ff": 256,
            "param_count": 1617729,
            "arch_idx": 2,
            "lr": 0.0005,
            "epochs_config": 100,
            "weight_decay": 0.0003,
            "warmup_steps": 100,
            "dropout": 0.14,
            "val_loss": 0.3634,
            "train_loss": 0.35,
            "val_accuracy": 0.89,
            "best_epoch": 1,
            "best_val_loss": 0.44,
            "epochs_trained": 2,
            "early_stopped": True,
            "duration_sec": 33.08,
            "diverged": False,
            "rank_in_study": 1,
            "is_best_in_study": True,
            "learning_curve": [
                {"epoch": 0, "train_loss": 0.48, "val_loss": 0.49},
                {"epoch": 1, "train_loss": 0.46, "val_loss": 0.44},
            ],
        },
        {
            "study": "phase6a_2M_h1_threshold_1pct",
            "budget": "2M",
            "horizon": "h1",
            "trial_num": 1,
            "d_model": 64,
            "n_layers": 48,
            "n_heads": 2,
            "d_ff": 256,
            "param_count": 2400000,
            "arch_idx": 3,
            "lr": 0.0003,
            "epochs_config": 100,
            "weight_decay": 0.0001,
            "warmup_steps": 50,
            "dropout": 0.10,
            "val_loss": 0.3199,
            "train_loss": 0.31,
            "val_accuracy": 0.91,
            "best_epoch": 45,
            "best_val_loss": 0.3199,
            "epochs_trained": 50,
            "early_stopped": True,
            "duration_sec": 45.2,
            "diverged": False,
            "rank_in_study": 2,
            "is_best_in_study": False,
            "learning_curve": [],
        },
    ]
