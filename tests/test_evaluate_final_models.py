"""Tests for evaluate_final_models.py script.

TDD RED phase: These tests define the expected behavior of the evaluation
script before implementation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoadModelFromCheckpoint:
    """Tests for load_model_from_checkpoint function."""

    def test_load_model_from_checkpoint_returns_model(self, tmp_path):
        """Test that load_model_from_checkpoint returns a PatchTST model in eval mode."""
        from scripts.evaluate_final_models import load_model_from_checkpoint
        from src.models.patchtst import PatchTST, PatchTSTConfig

        # Create a mock checkpoint with known architecture
        config = PatchTSTConfig(
            num_features=25,
            context_length=60,
            patch_length=16,
            stride=8,
            d_model=64,
            n_heads=2,
            n_layers=48,
            d_ff=256,
            dropout=0.1,
            head_dropout=0.0,
        )
        model = PatchTST(config)

        checkpoint = {
            "epoch": 10,
            "val_loss": 0.25,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "experiment_config": {"task": "threshold_1pct"},
            "data_md5": "abc123",
        }

        # Save checkpoint
        checkpoint_path = tmp_path / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load model
        loaded_model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            budget="2M",
            horizon=1,
        )

        # Assertions
        assert isinstance(loaded_model, PatchTST)
        assert not loaded_model.training  # Should be in eval mode
        # Verify architecture matches expected for 2M/h1
        assert loaded_model.config.d_model == 64
        assert loaded_model.config.n_layers == 48
        assert loaded_model.config.n_heads == 2


class TestPrepareTestData:
    """Tests for prepare_test_data function."""

    def test_prepare_test_data_filters_to_2025(self):
        """Test that prepare_test_data filters to prediction_date >= 2025-01-01."""
        from scripts.evaluate_final_models import prepare_test_data

        # Create mock DataFrame spanning 2024-2025
        dates = pd.date_range("2024-01-01", "2025-12-31", freq="B")  # Business days
        n_rows = len(dates)

        df = pd.DataFrame({
            "Date": dates,
            "Open": np.random.randn(n_rows),
            "High": np.random.randn(n_rows),
            "Low": np.random.randn(n_rows),
            "Close": np.random.randn(n_rows) + 100,  # Ensure positive
            "Volume": np.abs(np.random.randn(n_rows)) * 1e6,
        })
        # Add 20 indicator columns
        for i in range(20):
            df[f"indicator_{i}"] = np.random.randn(n_rows)

        # Prepare test data for h1 (horizon=1)
        test_indices, test_dates = prepare_test_data(
            df=df,
            horizon=1,
            context_length=60,
            test_start_date="2025-01-01",
        )

        # All test dates should be >= 2025-01-01
        assert all(d >= pd.Timestamp("2025-01-01") for d in test_dates)

        # Should have ~250 samples (roughly one year of business days)
        assert len(test_indices) > 200
        assert len(test_indices) < 300


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_returns_all_metrics(self):
        """Test that compute_metrics returns accuracy, precision, recall, f1, auc_roc."""
        from scripts.evaluate_final_models import compute_metrics

        # Create known predictions and targets
        predictions = np.array([0.8, 0.6, 0.4, 0.3, 0.9, 0.2, 0.7, 0.5])
        targets = np.array([1, 1, 0, 0, 1, 0, 1, 0])

        metrics = compute_metrics(predictions, targets, threshold=0.5)

        # Check all expected keys exist
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics

        # Check values are reasonable (0-1 range)
        for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_compute_metrics_handles_single_class(self):
        """Test that compute_metrics handles all-zeros or all-ones targets gracefully."""
        from scripts.evaluate_final_models import compute_metrics

        # All positive targets
        predictions = np.array([0.8, 0.6, 0.7, 0.9])
        targets_all_ones = np.array([1, 1, 1, 1])

        metrics = compute_metrics(predictions, targets_all_ones, threshold=0.5)

        # Should not crash, should return valid metrics
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # All predicted as 1, all targets are 1

        # All negative targets
        targets_all_zeros = np.array([0, 0, 0, 0])
        predictions_low = np.array([0.2, 0.3, 0.1, 0.4])

        metrics = compute_metrics(predictions_low, targets_all_zeros, threshold=0.5)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # All predicted as 0, all targets are 0


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_model_returns_predictions_and_targets(self):
        """Test that evaluate_model returns numpy arrays of predictions and targets."""
        from scripts.evaluate_final_models import evaluate_model

        # Create mock model that returns fixed logits (pre-sigmoid)
        # Use logits that produce ~[0.7, 0.3, 0.8, 0.2] after sigmoid
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: torch.tensor([[0.85], [-0.85], [1.4], [-1.4]])
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        # Create mock dataloader
        mock_batch_x = torch.randn(4, 60, 25)
        mock_batch_y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        mock_dataloader = [(mock_batch_x, mock_batch_y)]

        predictions, targets = evaluate_model(
            model=mock_model,
            dataloader=mock_dataloader,
            device="cpu",
        )

        # Check return types
        assert isinstance(predictions, np.ndarray)
        assert isinstance(targets, np.ndarray)

        # Check shapes match
        assert predictions.shape == targets.shape
        assert len(predictions) == 4


class TestArchitectureTable:
    """Tests for architecture lookup table."""

    def test_architecture_table_matches_training_scripts(self):
        """Test that ARCHITECTURES table has all 16 entries matching training scripts."""
        from scripts.evaluate_final_models import ARCHITECTURES

        # Expected architectures from generate_final_training_scripts.py
        expected = {
            ("2M", 1): (64, 48, 2),
            ("2M", 2): (64, 32, 2),
            ("2M", 3): (64, 32, 2),
            ("2M", 5): (64, 64, 16),
            ("20M", 1): (128, 180, 16),
            ("20M", 2): (256, 32, 2),
            ("20M", 3): (256, 32, 2),
            ("20M", 5): (384, 12, 4),
            ("200M", 1): (384, 96, 4),
            ("200M", 2): (768, 24, 16),
            ("200M", 3): (768, 24, 16),
            ("200M", 5): (256, 256, 16),
            ("2B", 1): (1024, 128, 2),
            ("2B", 2): (768, 256, 32),
            ("2B", 3): (768, 256, 32),
            ("2B", 5): (1024, 180, 4),
        }

        # Check all 16 entries exist
        assert len(ARCHITECTURES) == 16

        # Check each entry matches
        for key, value in expected.items():
            assert key in ARCHITECTURES, f"Missing architecture for {key}"
            assert ARCHITECTURES[key] == value, f"Mismatch for {key}: expected {value}, got {ARCHITECTURES[key]}"
