#!/usr/bin/env python3
"""Tests for evaluation utilities in experiments/architectures/common.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root and experiments directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "architectures"))

from common import evaluate_forecasting_model


class TestDirectionAccuracy:
    """Tests for direction_accuracy calculation in evaluate_forecasting_model."""

    def test_direction_accuracy_classification_mode_uses_05_threshold(self):
        """Test that classification mode uses 0.5 threshold for predicted direction.

        For Bernoulli outputs [0, 1], direction should be positive if prediction > 0.5.
        """
        # Predictions: 3 above 0.5, 2 below 0.5
        predicted = np.array([0.8, 0.7, 0.6, 0.3, 0.2])
        # Actual returns: all positive (direction = 1)
        actual_returns = np.array([0.01, 0.02, 0.01, 0.015, 0.005])
        # Threshold targets (not used for direction accuracy)
        threshold_targets = np.array([1, 1, 1, 1, 0])

        metrics = evaluate_forecasting_model(
            predicted_returns=predicted,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # Predicted direction should be [1, 1, 1, 0, 0] (using 0.5 threshold)
        # Actual direction should be [1, 1, 1, 1, 1] (all positive returns)
        # Direction accuracy should be 3/5 = 0.6
        assert metrics["direction_accuracy"] == pytest.approx(0.6, abs=0.01), (
            f"Expected direction_accuracy=0.6 but got {metrics['direction_accuracy']}. "
            "Classification mode should use 0.5 threshold for predicted direction."
        )

    def test_direction_accuracy_regression_mode_uses_zero_threshold(self):
        """Test that regression mode uses 0 threshold for predicted direction.

        For return forecasts, direction is positive if prediction > 0.
        """
        # Predictions: 3 positive, 2 negative
        predicted = np.array([0.01, 0.005, 0.002, -0.003, -0.01])
        # Actual returns: same signs
        actual_returns = np.array([0.02, 0.01, 0.015, -0.005, -0.008])
        # Threshold targets (not used for direction accuracy)
        threshold_targets = np.array([1, 1, 1, 0, 0])

        metrics = evaluate_forecasting_model(
            predicted_returns=predicted,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=False,
        )

        # Predicted direction: [1, 1, 1, 0, 0] (using 0 threshold)
        # Actual direction: [1, 1, 1, 0, 0]
        # Direction accuracy should be 5/5 = 1.0
        assert metrics["direction_accuracy"] == pytest.approx(1.0, abs=0.01), (
            f"Expected direction_accuracy=1.0 but got {metrics['direction_accuracy']}. "
            "Regression mode should use 0 threshold for predicted direction."
        )

    def test_direction_accuracy_classification_all_above_05(self):
        """Test classification mode when all predictions are above 0.5."""
        # All predictions above 0.5
        predicted = np.array([0.9, 0.8, 0.7, 0.6, 0.55])
        # Half positive, half negative actual returns
        actual_returns = np.array([0.01, -0.01, 0.02, -0.015, -0.005])
        threshold_targets = np.array([1, 0, 1, 0, 0])

        metrics = evaluate_forecasting_model(
            predicted_returns=predicted,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # Predicted direction: [1, 1, 1, 1, 1] (all > 0.5)
        # Actual direction: [1, 0, 1, 0, 0]
        # Direction accuracy should be 2/5 = 0.4
        assert metrics["direction_accuracy"] == pytest.approx(0.4, abs=0.01), (
            f"Expected direction_accuracy=0.4 but got {metrics['direction_accuracy']}. "
            "All predictions above 0.5 should predict positive direction."
        )

    def test_direction_accuracy_classification_all_below_05(self):
        """Test classification mode when all predictions are below 0.5."""
        # All predictions below 0.5
        predicted = np.array([0.45, 0.3, 0.2, 0.1, 0.05])
        # Half positive, half negative actual returns
        actual_returns = np.array([0.01, -0.01, 0.02, -0.015, -0.005])
        threshold_targets = np.array([1, 0, 1, 0, 0])

        metrics = evaluate_forecasting_model(
            predicted_returns=predicted,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # Predicted direction: [0, 0, 0, 0, 0] (all < 0.5)
        # Actual direction: [1, 0, 1, 0, 0]
        # Direction accuracy should be 3/5 = 0.6
        assert metrics["direction_accuracy"] == pytest.approx(0.6, abs=0.01), (
            f"Expected direction_accuracy=0.6 but got {metrics['direction_accuracy']}. "
            "All predictions below 0.5 should predict negative direction."
        )

    def test_direction_accuracy_classification_bug_detection(self):
        """Test that detects the bug: using > 0 threshold for Bernoulli outputs.

        This test will FAIL if direction_accuracy uses (predictions > 0) for classification.
        For Bernoulli outputs in [0.1, 0.9], all predictions > 0, giving wrong direction.
        """
        # Bernoulli-like predictions all in (0, 1) range
        # 2 above 0.5 (should predict positive), 3 below 0.5 (should predict negative)
        predicted = np.array([0.8, 0.6, 0.4, 0.3, 0.1])
        # Actual returns: 2 positive, 3 negative
        actual_returns = np.array([0.01, 0.02, -0.01, -0.02, -0.015])
        threshold_targets = np.array([1, 1, 0, 0, 0])

        metrics = evaluate_forecasting_model(
            predicted_returns=predicted,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # Correct behavior (using 0.5 threshold):
        # Predicted direction: [1, 1, 0, 0, 0]
        # Actual direction: [1, 1, 0, 0, 0]
        # Direction accuracy should be 5/5 = 1.0

        # Bug behavior (using 0 threshold):
        # Predicted direction: [1, 1, 1, 1, 1] (all > 0!)
        # Actual direction: [1, 1, 0, 0, 0]
        # Direction accuracy would be 2/5 = 0.4

        assert metrics["direction_accuracy"] == pytest.approx(1.0, abs=0.01), (
            f"Expected direction_accuracy=1.0 but got {metrics['direction_accuracy']}. "
            "BUG DETECTED: Classification mode is using wrong threshold for predicted direction. "
            "Should use 0.5 for Bernoulli outputs, not 0."
        )
