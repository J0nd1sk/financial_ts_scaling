"""Tests for threshold sweep functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestComputeMetricsAtThreshold:
    """Tests for computing metrics at a specific threshold."""

    def test_basic_computation(self):
        """Test precision/recall computation at threshold with known values."""
        from scripts.threshold_sweep import compute_metrics_at_threshold

        # 10 samples: 5 positive labels, 5 negative labels
        # Predictions: 4 above 0.5, 6 below 0.5
        predictions = np.array([0.9, 0.8, 0.6, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1, 0.05])
        labels = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
        # At threshold 0.5:
        # Predicted positive (>=0.5): indices 0,1,2,3 -> labels 1,1,1,0
        # TP=3, FP=1, FN=2 (labels 4,7 are positive but predicted negative)
        # Precision = 3/4 = 0.75
        # Recall = 3/5 = 0.6

        result = compute_metrics_at_threshold(predictions, labels, threshold=0.5)

        assert result["precision"] == pytest.approx(0.75, rel=0.01)
        assert result["recall"] == pytest.approx(0.6, rel=0.01)
        assert result["n_positive_preds"] == 4
        assert result["n_samples"] == 10

    def test_edge_case_no_positive_predictions(self):
        """Test when threshold is above all predictions."""
        from scripts.threshold_sweep import compute_metrics_at_threshold

        predictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        labels = np.array([0, 1, 1, 0, 1])

        result = compute_metrics_at_threshold(predictions, labels, threshold=0.9)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["n_positive_preds"] == 0
        assert result["n_samples"] == 5

    def test_edge_case_all_positive_predictions(self):
        """Test when threshold is below all predictions."""
        from scripts.threshold_sweep import compute_metrics_at_threshold

        predictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        labels = np.array([0, 1, 1, 0, 1])
        # All predicted positive, 3 actual positive, 2 actual negative
        # TP=3, FP=2, FN=0
        # Precision = 3/5 = 0.6
        # Recall = 3/3 = 1.0

        result = compute_metrics_at_threshold(predictions, labels, threshold=0.1)

        assert result["precision"] == pytest.approx(0.6, rel=0.01)
        assert result["recall"] == pytest.approx(1.0, rel=0.01)
        assert result["n_positive_preds"] == 5
        assert result["n_samples"] == 5


class TestSweepThresholds:
    """Tests for sweeping across multiple thresholds."""

    def test_returns_dataframe_with_correct_columns(self):
        """Test that sweep returns DataFrame with expected columns."""
        from scripts.threshold_sweep import sweep_thresholds

        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        thresholds = [0.2, 0.4, 0.6, 0.8]

        result = sweep_thresholds(predictions, labels, thresholds)

        assert isinstance(result, pd.DataFrame)
        expected_columns = {"threshold", "precision", "recall", "f1", "n_positive_preds", "n_samples"}
        assert expected_columns.issubset(set(result.columns))
        assert len(result) == len(thresholds)

    def test_thresholds_are_in_output(self):
        """Test that each threshold appears in output."""
        from scripts.threshold_sweep import sweep_thresholds

        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        thresholds = [0.2, 0.4, 0.6, 0.8]

        result = sweep_thresholds(predictions, labels, thresholds)

        assert list(result["threshold"]) == thresholds
