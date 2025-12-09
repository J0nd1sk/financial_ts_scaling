"""Tests for FinancialDataset class.

Tests cover:
- Correct tensor shapes for inputs and targets
- Binary threshold label construction
- Horizon handling for target calculation
- Edge case handling (end of sequence, short sequences, NaN values)
- Warmup period exclusion
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import FinancialDataset


@pytest.fixture
def sample_features_df():
    """Create a minimal features DataFrame for testing."""
    n_rows = 100
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "feature_1": np.random.randn(n_rows),
            "feature_2": np.random.randn(n_rows),
            "feature_3": np.random.randn(n_rows),
        }
    )


@pytest.fixture
def sample_close_prices():
    """Create close prices that align with sample_features_df."""
    n_rows = 100
    # Start at 100, random walk with small changes
    np.random.seed(42)
    returns = np.random.randn(n_rows) * 0.01
    close = 100 * np.cumprod(1 + returns)
    return close


@pytest.fixture
def known_close_for_threshold_test():
    """Close prices with known threshold crossings.

    For threshold_1pct test at t=5 with horizon=5:
    - close[5] = 100.0
    - Need to check max(close[6:11]) >= 100.0 * 1.01 = 101.0
    - Set close[8] = 102.0 so label should be 1
    """
    close = np.array([
        98.0,   # t=0
        99.0,   # t=1
        99.5,   # t=2
        99.8,   # t=3
        99.9,   # t=4
        100.0,  # t=5 - test point
        100.2,  # t=6
        100.5,  # t=7
        102.0,  # t=8 - exceeds 1% threshold
        100.3,  # t=9
        100.1,  # t=10
        99.0,   # t=11
        98.5,   # t=12
        98.0,   # t=13
        97.5,   # t=14
    ])
    return close


class TestDatasetShapes:
    """Test that dataset returns correctly shaped tensors."""

    def test_dataset_returns_correct_shapes(
        self, sample_features_df, sample_close_prices
    ):
        """Input should be (context_length, n_features), target should be (1,)."""
        context_length = 10
        horizon = 5
        n_features = 3  # feature_1, feature_2, feature_3

        dataset = FinancialDataset(
            features_df=sample_features_df,
            close_prices=sample_close_prices,
            context_length=context_length,
            horizon=horizon,
            threshold=0.01,
        )

        x, y = dataset[0]

        assert x.shape == (context_length, n_features), (
            f"Expected input shape ({context_length}, {n_features}), got {x.shape}"
        )
        assert y.shape == (1,), f"Expected target shape (1,), got {y.shape}"
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


class TestTargetConstruction:
    """Test binary threshold label construction."""

    def test_dataset_binary_label_threshold_1pct(self, known_close_for_threshold_test):
        """Known sequence with 1% threshold crossing should produce label=1."""
        close = known_close_for_threshold_test
        n_rows = len(close)

        # Create matching features
        features_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "feature_1": np.ones(n_rows),
            }
        )

        context_length = 5
        horizon = 5
        threshold = 0.01  # 1%

        dataset = FinancialDataset(
            features_df=features_df,
            close_prices=close,
            context_length=context_length,
            horizon=horizon,
            threshold=threshold,
        )

        # Index 0 in dataset corresponds to t=context_length-1=4 in original data
        # We want to test t=5, which is dataset index 1
        # At t=5: close=100.0, future_max=max(close[6:11])=102.0
        # 102.0 >= 100.0 * 1.01 = 101.0 -> label=1
        _, y = dataset[1]

        assert y.item() == 1.0, (
            f"Expected label=1 (threshold exceeded), got {y.item()}"
        )

    def test_dataset_handles_horizon_correctly(self):
        """Horizon parameter should look exactly horizon steps ahead."""
        # Create data where only the last day of horizon exceeds threshold
        close = np.array([
            100.0,  # t=0
            100.0,  # t=1
            100.0,  # t=2
            100.0,  # t=3
            100.0,  # t=4 - context ends here for first sample
            100.0,  # t=5 - first prediction point
            100.0,  # t=6
            100.0,  # t=7
            100.0,  # t=8
            102.0,  # t=9 - only this exceeds 1% (horizon=5 means t+1 to t+5)
            100.0,  # t=10
        ])
        n_rows = len(close)
        features_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "feature_1": np.ones(n_rows),
            }
        )

        context_length = 5
        horizon = 5
        threshold = 0.01

        dataset = FinancialDataset(
            features_df=features_df,
            close_prices=close,
            context_length=context_length,
            horizon=horizon,
            threshold=threshold,
        )

        # First sample: t=4 (context_length-1), looks at t=5 to t=9
        # max(close[5:10]) = 102.0 >= 100.0 * 1.01 -> label=1
        _, y = dataset[0]
        assert y.item() == 1.0, "Should detect threshold crossing at end of horizon"


class TestDatasetLength:
    """Test dataset length calculations."""

    def test_dataset_excludes_samples_near_end(
        self, sample_features_df, sample_close_prices
    ):
        """Last horizon rows should be excluded (no future data for labels)."""
        context_length = 10
        horizon = 5
        n_rows = len(sample_features_df)

        dataset = FinancialDataset(
            features_df=sample_features_df,
            close_prices=sample_close_prices,
            context_length=context_length,
            horizon=horizon,
            threshold=0.01,
        )

        # Valid indices: context_length-1 to n_rows-horizon-1
        # Length = n_rows - context_length - horizon + 1
        expected_length = n_rows - context_length - horizon + 1
        assert len(dataset) == expected_length, (
            f"Expected length {expected_length}, got {len(dataset)}"
        )

    def test_dataset_length_matches_expected(self):
        """Total - context_length - horizon + 1 = usable samples."""
        n_rows = 50
        context_length = 10
        horizon = 5

        features_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "feature_1": np.random.randn(n_rows),
            }
        )
        close = np.random.randn(n_rows) * 10 + 100

        dataset = FinancialDataset(
            features_df=features_df,
            close_prices=close,
            context_length=context_length,
            horizon=horizon,
            threshold=0.01,
        )

        # Formula: n_rows - context_length - horizon + 1
        expected = n_rows - context_length - horizon + 1  # 50 - 10 - 5 + 1 = 36
        assert len(dataset) == expected

    def test_dataset_warmup_excludes_initial_rows(self):
        """First context_length-1 rows cannot be starting points."""
        n_rows = 30
        context_length = 10
        horizon = 3

        features_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "feature_1": np.arange(n_rows, dtype=float),  # Use indices as values
            }
        )
        close = np.arange(n_rows, dtype=float) + 100

        dataset = FinancialDataset(
            features_df=features_df,
            close_prices=close,
            context_length=context_length,
            horizon=horizon,
            threshold=0.01,
        )

        # First valid sample uses rows 0 to context_length-1 as input
        # The feature at position 0 in context should be row 0's feature
        x, _ = dataset[0]

        # x[0, 0] should be feature_1 at row 0 = 0.0
        assert x[0, 0].item() == 0.0, (
            f"First context element should be from row 0, got {x[0, 0].item()}"
        )
        # x[-1, 0] should be feature_1 at row context_length-1 = 9.0
        assert x[-1, 0].item() == context_length - 1, (
            f"Last context element should be from row {context_length-1}"
        )


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_dataset_raises_on_nan_in_close(self, sample_features_df):
        """NaN in close prices should raise ValueError."""
        close = np.array([100.0] * len(sample_features_df))
        close[50] = np.nan  # Insert NaN

        with pytest.raises(ValueError, match="NaN"):
            FinancialDataset(
                features_df=sample_features_df,
                close_prices=close,
                context_length=10,
                horizon=5,
                threshold=0.01,
            )

    def test_dataset_rejects_short_sequences(self):
        """Sequence shorter than context_length + horizon should raise ValueError."""
        n_rows = 10  # Too short for context_length=10, horizon=5
        features_df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "feature_1": np.random.randn(n_rows),
            }
        )
        close = np.random.randn(n_rows) * 10 + 100

        with pytest.raises(ValueError, match="too short|insufficient"):
            FinancialDataset(
                features_df=features_df,
                close_prices=close,
                context_length=10,
                horizon=5,
                threshold=0.01,
            )
