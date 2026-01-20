"""Tests for FinancialDataset and ChunkSplitter classes.

Tests cover:
- Correct tensor shapes for inputs and targets
- Binary threshold label construction
- Horizon handling for target calculation
- Edge case handling (end of sequence, short sequences, NaN values)
- Warmup period exclusion
- Hybrid chunk-based train/val/test splits
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    FinancialDataset,
    ChunkSplitter,
    SimpleSplitter,
    SplitIndices,
    compute_normalization_params,
    normalize_dataframe,
    BOUNDED_FEATURES,
)


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


# ============================================================
# ChunkSplitter Tests - Hybrid train/val/test splits
# ============================================================


class TestChunkSplitterReturnsDataclass:
    """Test that ChunkSplitter returns correct SplitIndices dataclass."""

    def test_chunk_splitter_returns_split_indices(self):
        """ChunkSplitter should return SplitIndices dataclass."""
        total_days = 1000
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon  # 61

        splitter = ChunkSplitter(
            total_days=total_days,
            context_length=context_length,
            horizon=horizon,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        assert isinstance(splits, SplitIndices), "Should return SplitIndices"
        assert hasattr(splits, "train_indices")
        assert hasattr(splits, "val_indices")
        assert hasattr(splits, "test_indices")
        assert hasattr(splits, "chunk_size")

    def test_split_indices_are_numpy_arrays(self):
        """All indices should be numpy arrays."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        assert isinstance(splits.train_indices, np.ndarray)
        assert isinstance(splits.val_indices, np.ndarray)
        assert isinstance(splits.test_indices, np.ndarray)


class TestChunkSplitterValTestNonOverlapping:
    """Test that val/test chunks are non-overlapping."""

    def test_val_chunks_are_non_overlapping(self):
        """Val chunks should not overlap with each other."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size

        # Check no two val chunks overlap
        val_starts = sorted(splits.val_indices)
        for i in range(len(val_starts) - 1):
            assert val_starts[i + 1] >= val_starts[i] + chunk_size, (
                f"Val chunks overlap: {val_starts[i]} and {val_starts[i + 1]}"
            )

    def test_test_chunks_are_non_overlapping(self):
        """Test chunks should not overlap with each other."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size

        # Check no two test chunks overlap
        test_starts = sorted(splits.test_indices)
        for i in range(len(test_starts) - 1):
            assert test_starts[i + 1] >= test_starts[i] + chunk_size, (
                f"Test chunks overlap: {test_starts[i]} and {test_starts[i + 1]}"
            )

    def test_val_and_test_chunks_dont_overlap(self):
        """Val and test chunks should not overlap with each other."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size

        # Create sets of all days in val and test chunks
        val_days = set()
        for start in splits.val_indices:
            val_days.update(range(start, start + chunk_size))

        test_days = set()
        for start in splits.test_indices:
            test_days.update(range(start, start + chunk_size))

        overlap = val_days & test_days
        assert len(overlap) == 0, f"Val and test chunks overlap on days: {overlap}"


class TestChunkSplitterTrainNoOverlap:
    """Test that train samples don't overlap with val/test regions."""

    def test_train_samples_dont_overlap_val_test(self):
        """No train sample's [t, t+context_length+horizon) can overlap val/test."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size
        context_length = 60
        horizon = 1
        sample_span = context_length + horizon  # Days covered by one sample

        # Create set of all val/test days
        blocked_days = set()
        for start in splits.val_indices:
            blocked_days.update(range(start, start + chunk_size))
        for start in splits.test_indices:
            blocked_days.update(range(start, start + chunk_size))

        # Check each train sample doesn't touch blocked days
        for train_start in splits.train_indices:
            sample_days = set(range(train_start, train_start + sample_span))
            overlap = sample_days & blocked_days
            assert len(overlap) == 0, (
                f"Train sample at {train_start} overlaps blocked days: {overlap}"
            )

    def test_train_uses_sliding_window(self):
        """Train should have more samples than val+test (sliding vs non-overlapping)."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        # Val/test are non-overlapping chunks, train is sliding
        # Train should have significantly more samples
        assert len(splits.train_indices) > len(splits.val_indices) + len(splits.test_indices), (
            f"Train ({len(splits.train_indices)}) should be > val+test "
            f"({len(splits.val_indices) + len(splits.test_indices)})"
        )


class TestChunkSplitterSplitRatios:
    """Test that split ratios are approximately correct."""

    def test_val_test_chunk_counts_approximate_ratio(self):
        """Val/test chunk counts should approximate the requested ratio."""
        total_days = 8000  # ~130 chunks of 61 days
        splitter = ChunkSplitter(
            total_days=total_days,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size

        total_chunks = total_days // chunk_size
        expected_val_chunks = int(total_chunks * 0.15)
        expected_test_chunks = int(total_chunks * 0.15)

        # Allow ±2 chunks tolerance
        assert abs(len(splits.val_indices) - expected_val_chunks) <= 2, (
            f"Val chunks {len(splits.val_indices)} not close to expected {expected_val_chunks}"
        )
        assert abs(len(splits.test_indices) - expected_test_chunks) <= 2, (
            f"Test chunks {len(splits.test_indices)} not close to expected {expected_test_chunks}"
        )


class TestChunkSplitterReproducibility:
    """Test that splits are reproducible with same seed."""

    def test_same_seed_produces_same_splits(self):
        """Same seed should produce identical splits."""
        kwargs = dict(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        splitter1 = ChunkSplitter(**kwargs)
        splits1 = splitter1.split()

        splitter2 = ChunkSplitter(**kwargs)
        splits2 = splitter2.split()

        np.testing.assert_array_equal(splits1.train_indices, splits2.train_indices)
        np.testing.assert_array_equal(splits1.val_indices, splits2.val_indices)
        np.testing.assert_array_equal(splits1.test_indices, splits2.test_indices)

    def test_different_seed_produces_different_splits(self):
        """Different seeds should produce different splits."""
        base_kwargs = dict(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        splits1 = ChunkSplitter(**base_kwargs, seed=42).split()
        splits2 = ChunkSplitter(**base_kwargs, seed=123).split()

        # At least one of the indices should be different
        val_different = not np.array_equal(splits1.val_indices, splits2.val_indices)
        test_different = not np.array_equal(splits1.test_indices, splits2.test_indices)
        assert val_different or test_different, "Different seeds should produce different splits"


class TestChunkSplitterEdgeCases:
    """Test edge cases and validation."""

    def test_raises_on_insufficient_data(self):
        """Should raise if total_days is too small for meaningful splits."""
        with pytest.raises(ValueError, match="insufficient|too short|small"):
            ChunkSplitter(
                total_days=100,  # Too small for context=60, horizon=1
                context_length=60,
                horizon=1,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
            )

    def test_raises_on_invalid_ratios(self):
        """Should raise if val_ratio + test_ratio >= 1.0."""
        with pytest.raises(ValueError, match="ratio"):
            ChunkSplitter(
                total_days=1000,
                context_length=60,
                horizon=1,
                val_ratio=0.5,
                test_ratio=0.6,  # Total > 1.0
                seed=42,
            )

    def test_handles_real_world_data_size(self):
        """Should handle ~8000 days (1993-2025 SPY data)."""
        splitter = ChunkSplitter(
            total_days=8073,  # Real SPY data size
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        # Should have reasonable number of samples
        assert len(splits.train_indices) > 1000, "Should have many train samples"
        assert len(splits.val_indices) >= 10, "Should have val samples"
        assert len(splits.test_indices) >= 10, "Should have test samples"


class TestChunkSplitterHPOSubset:
    """Test HPO subset functionality."""

    def test_get_hpo_subset_returns_subset(self):
        """get_hpo_subset should return a fraction of train indices."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        hpo_fraction = 0.3
        hpo_indices = splitter.get_hpo_subset(splits, fraction=hpo_fraction)

        expected_count = int(len(splits.train_indices) * hpo_fraction)
        # Allow ±5% tolerance
        assert abs(len(hpo_indices) - expected_count) <= expected_count * 0.05 + 1, (
            f"HPO subset size {len(hpo_indices)} not close to expected {expected_count}"
        )

    def test_hpo_subset_is_subset_of_train(self):
        """HPO subset should only contain train indices."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        hpo_indices = splitter.get_hpo_subset(splits, fraction=0.3)

        train_set = set(splits.train_indices)
        for idx in hpo_indices:
            assert idx in train_set, f"HPO index {idx} not in train indices"

    def test_hpo_subset_reproducible(self):
        """HPO subset should be reproducible with same seed."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        splits = splitter.split()

        hpo1 = splitter.get_hpo_subset(splits, fraction=0.3, seed=42)
        hpo2 = splitter.get_hpo_subset(splits, fraction=0.3, seed=42)

        np.testing.assert_array_equal(hpo1, hpo2)


class TestChunkSplitterContiguousMode:
    """Test contiguous split mode for production-realistic train/val/test."""

    def test_contiguous_mode_test_at_end(self):
        """When mode='contiguous', test should be contiguous block at dataset end."""
        total_days = 1000
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon  # 61

        splitter = ChunkSplitter(
            total_days=total_days,
            context_length=context_length,
            horizon=horizon,
            val_ratio=0.10,
            test_ratio=0.15,
            seed=42,
            mode="contiguous",
        )
        splits = splitter.split()

        # Test indices should be at the end of the dataset
        n_chunks = total_days // chunk_size
        n_test_chunks = max(1, int(n_chunks * 0.15))
        expected_test_start = (n_chunks - n_test_chunks) * chunk_size

        # All test indices should be >= expected_test_start
        assert all(idx >= expected_test_start for idx in splits.test_indices), (
            f"Test indices should start at {expected_test_start}, got {splits.test_indices}"
        )

        # Test indices should be contiguous (consecutive chunk starts)
        test_starts = sorted(splits.test_indices)
        for i in range(len(test_starts) - 1):
            assert test_starts[i + 1] == test_starts[i] + chunk_size, (
                f"Test chunks not contiguous: {test_starts[i]} and {test_starts[i + 1]}"
            )

    def test_contiguous_mode_val_before_test(self):
        """Val should be contiguous block immediately before test."""
        total_days = 1000
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon

        splitter = ChunkSplitter(
            total_days=total_days,
            context_length=context_length,
            horizon=horizon,
            val_ratio=0.10,
            test_ratio=0.15,
            seed=42,
            mode="contiguous",
        )
        splits = splitter.split()

        # Val should end where test begins
        val_end = max(splits.val_indices) + chunk_size
        test_start = min(splits.test_indices)
        assert val_end == test_start, (
            f"Val should end where test begins: val_end={val_end}, test_start={test_start}"
        )

        # Val indices should be contiguous
        val_starts = sorted(splits.val_indices)
        for i in range(len(val_starts) - 1):
            assert val_starts[i + 1] == val_starts[i] + chunk_size, (
                f"Val chunks not contiguous: {val_starts[i]} and {val_starts[i + 1]}"
            )

    def test_contiguous_mode_train_no_overlap(self):
        """Train samples shouldn't overlap val or test regions in contiguous mode."""
        splitter = ChunkSplitter(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.10,
            test_ratio=0.15,
            seed=42,
            mode="contiguous",
        )
        splits = splitter.split()
        chunk_size = splits.chunk_size

        # Create set of all val/test days
        blocked_days = set()
        for start in splits.val_indices:
            blocked_days.update(range(start, start + chunk_size))
        for start in splits.test_indices:
            blocked_days.update(range(start, start + chunk_size))

        # Check each train sample doesn't touch blocked days
        for train_start in splits.train_indices:
            sample_days = set(range(train_start, train_start + chunk_size))
            overlap = sample_days & blocked_days
            assert len(overlap) == 0, (
                f"Train sample at {train_start} overlaps blocked days: {overlap}"
            )

    def test_scattered_mode_is_default(self):
        """Default mode='scattered' produces same output as no mode specified."""
        kwargs = dict(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # Explicit scattered
        splitter_scattered = ChunkSplitter(**kwargs, mode="scattered")
        splits_scattered = splitter_scattered.split()

        # Implicit (no mode, should default to scattered)
        splitter_default = ChunkSplitter(**kwargs)
        splits_default = splitter_default.split()

        np.testing.assert_array_equal(splits_scattered.train_indices, splits_default.train_indices)
        np.testing.assert_array_equal(splits_scattered.val_indices, splits_default.val_indices)
        np.testing.assert_array_equal(splits_scattered.test_indices, splits_default.test_indices)

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode"):
            ChunkSplitter(
                total_days=1000,
                context_length=60,
                horizon=1,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
                mode="invalid",
            )

    def test_contiguous_mode_reproducible(self):
        """Contiguous mode should be deterministic (no randomness needed)."""
        kwargs = dict(
            total_days=1000,
            context_length=60,
            horizon=1,
            val_ratio=0.10,
            test_ratio=0.15,
            mode="contiguous",
        )

        # Different seeds should produce same splits (contiguous is deterministic)
        splits1 = ChunkSplitter(**kwargs, seed=42).split()
        splits2 = ChunkSplitter(**kwargs, seed=999).split()

        np.testing.assert_array_equal(splits1.val_indices, splits2.val_indices)
        np.testing.assert_array_equal(splits1.test_indices, splits2.test_indices)


# ============================================================
# SimpleSplitter Tests - Date-based contiguous splits
# ============================================================


class TestSimpleSplitterReturnsDataclass:
    """Test that SimpleSplitter returns correct SplitIndices dataclass."""

    def test_returns_split_indices(self):
        """SimpleSplitter should return SplitIndices dataclass."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")

        splitter = SimpleSplitter(
            dates=dates,
            context_length=60,
            horizon=1,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        assert isinstance(splits, SplitIndices)
        assert hasattr(splits, "train_indices")
        assert hasattr(splits, "val_indices")
        assert hasattr(splits, "test_indices")

    def test_indices_are_numpy_arrays(self):
        """All indices should be numpy arrays."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")

        splitter = SimpleSplitter(
            dates=dates,
            context_length=60,
            horizon=1,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        assert isinstance(splits.train_indices, np.ndarray)
        assert isinstance(splits.val_indices, np.ndarray)
        assert isinstance(splits.test_indices, np.ndarray)


class TestSimpleSplitterStrictContainment:
    """Test that samples are strictly contained within their date regions."""

    def test_train_samples_end_before_val_start(self):
        """All train samples should end before val_start date."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon

        splitter = SimpleSplitter(
            dates=dates,
            context_length=context_length,
            horizon=horizon,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        val_start_idx = dates.get_loc(pd.Timestamp("2020-10-01"))

        # Every train sample's end (start + chunk_size) must be <= val_start_idx
        for train_start in splits.train_indices:
            sample_end = train_start + chunk_size
            assert sample_end <= val_start_idx, (
                f"Train sample at {train_start} extends to {sample_end} "
                f"which is >= val_start_idx {val_start_idx}"
            )

    def test_val_samples_within_val_region(self):
        """All val samples should be strictly within [val_start, test_start)."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon

        splitter = SimpleSplitter(
            dates=dates,
            context_length=context_length,
            horizon=horizon,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        val_start_idx = dates.get_loc(pd.Timestamp("2020-10-01"))
        test_start_idx = dates.get_loc(pd.Timestamp("2020-12-01"))

        for val_start in splits.val_indices:
            assert val_start >= val_start_idx, (
                f"Val sample starts at {val_start} which is < val_start_idx {val_start_idx}"
            )
            sample_end = val_start + chunk_size
            assert sample_end <= test_start_idx, (
                f"Val sample at {val_start} extends to {sample_end} "
                f"which is >= test_start_idx {test_start_idx}"
            )

    def test_test_samples_within_test_region(self):
        """All test samples should start >= test_start and fit within data."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon

        splitter = SimpleSplitter(
            dates=dates,
            context_length=context_length,
            horizon=horizon,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        test_start_idx = dates.get_loc(pd.Timestamp("2020-12-01"))

        for test_start in splits.test_indices:
            assert test_start >= test_start_idx, (
                f"Test sample starts at {test_start} which is < test_start_idx {test_start_idx}"
            )
            sample_end = test_start + chunk_size
            assert sample_end <= len(dates), (
                f"Test sample at {test_start} extends beyond data length {len(dates)}"
            )


class TestSimpleSplitterSlidingWindow:
    """Test that SimpleSplitter uses sliding window for all splits."""

    def test_val_has_sliding_window_samples(self):
        """Val should have sliding window samples, not 1 per chunk."""
        # Create data similar to real scenario: 500 days
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        context_length = 60
        horizon = 1
        chunk_size = context_length + horizon  # 61

        splitter = SimpleSplitter(
            dates=dates,
            context_length=context_length,
            horizon=horizon,
            val_start="2020-10-01",  # ~274 days in
            test_start="2020-12-01",  # ~335 days in
        )
        splits = splitter.split()

        # Val region is ~61 days (Oct 1 to Dec 1)
        # With sliding window, we should get region_size - chunk_size + 1 samples
        val_region_days = 61  # Oct 1 to Dec 1 is ~61 days
        expected_min_val_samples = max(1, val_region_days - chunk_size + 1)

        assert len(splits.val_indices) >= expected_min_val_samples, (
            f"Val should have at least {expected_min_val_samples} samples with sliding window, "
            f"got {len(splits.val_indices)}"
        )

    def test_realistic_sample_counts(self):
        """Test with realistic data matching SPY dataset structure."""
        # Simulate ~8100 days like SPY dataset (1993 to 2016)
        dates = pd.date_range("1993-11-01", periods=8100, freq="D")
        context_length = 60
        horizon = 1

        # Use dates within the actual range (ends around 2016-01-04)
        splitter = SimpleSplitter(
            dates=dates,
            context_length=context_length,
            horizon=horizon,
            val_start="2014-01-01",  # ~2 years before end
            test_start="2015-06-01",  # ~6 months of test data
        )
        splits = splitter.split()

        # Val region is ~1.5 years = ~547 days
        # Expected val samples = 547 - 61 + 1 = 487 (approximately)
        # Should be WAY more than 19 (the ChunkSplitter bug)
        assert len(splits.val_indices) > 100, (
            f"Val should have >100 samples with sliding window, "
            f"got {len(splits.val_indices)} (ChunkSplitter bug gave only 19)"
        )

        # Train should have many more samples
        assert len(splits.train_indices) > len(splits.val_indices), (
            f"Train ({len(splits.train_indices)}) should have more samples than val ({len(splits.val_indices)})"
        )


class TestSimpleSplitterNoOverlap:
    """Test that there's no overlap between train, val, and test."""

    def test_no_index_overlap_between_splits(self):
        """No sample index should appear in multiple splits."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")

        splitter = SimpleSplitter(
            dates=dates,
            context_length=60,
            horizon=1,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )
        splits = splitter.split()

        train_set = set(splits.train_indices)
        val_set = set(splits.val_indices)
        test_set = set(splits.test_indices)

        assert len(train_set & val_set) == 0, "Train and val indices overlap"
        assert len(train_set & test_set) == 0, "Train and test indices overlap"
        assert len(val_set & test_set) == 0, "Val and test indices overlap"


class TestSimpleSplitterReproducibility:
    """Test that SimpleSplitter is deterministic."""

    def test_same_dates_produce_same_splits(self):
        """Same inputs should produce identical splits."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        kwargs = dict(
            dates=dates,
            context_length=60,
            horizon=1,
            val_start="2020-10-01",
            test_start="2020-12-01",
        )

        splits1 = SimpleSplitter(**kwargs).split()
        splits2 = SimpleSplitter(**kwargs).split()

        np.testing.assert_array_equal(splits1.train_indices, splits2.train_indices)
        np.testing.assert_array_equal(splits1.val_indices, splits2.val_indices)
        np.testing.assert_array_equal(splits1.test_indices, splits2.test_indices)


class TestSimpleSplitterEdgeCases:
    """Test edge cases for SimpleSplitter."""

    def test_raises_on_insufficient_val_region(self):
        """Should raise if val region is smaller than context+horizon."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")

        with pytest.raises(ValueError, match="val"):
            SimpleSplitter(
                dates=dates,
                context_length=60,
                horizon=1,
                val_start="2020-12-01",  # Only ~30 days before test_start
                test_start="2020-12-15",
            )

    def test_raises_on_insufficient_test_region(self):
        """Should raise if test region is smaller than context+horizon."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")  # 1 year

        with pytest.raises(ValueError, match="test"):
            SimpleSplitter(
                dates=dates,
                context_length=60,
                horizon=1,
                val_start="2020-10-01",
                test_start="2020-12-20",  # Only ~12 days of test data
            )

    def test_raises_on_date_not_in_data(self):
        """Should raise if val_start or test_start not in dates."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        with pytest.raises(ValueError, match="not found"):
            SimpleSplitter(
                dates=dates,
                context_length=60,
                horizon=1,
                val_start="2025-01-01",  # Not in data
                test_start="2025-06-01",
            )


# ============================================================
# Feature Normalization Tests - Z-score normalization
# ============================================================


class TestComputeNormalizationParams:
    """Test compute_normalization_params function."""

    def test_returns_dict_with_mean_std(self):
        """Should return dict mapping feature names to (mean, std) tuples."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "feature_1": np.random.randn(100) * 10 + 50,
            "feature_2": np.random.randn(100) * 5 + 100,
        })

        params = compute_normalization_params(df, train_end_row=70)

        assert isinstance(params, dict)
        assert "feature_1" in params
        assert "feature_2" in params

        # Each value should be (mean, std) tuple
        mean, std = params["feature_1"]
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_uses_only_train_rows(self):
        """Stats should be computed only from rows 0:train_end_row."""
        # Create data where first half has different distribution than second half
        n = 100
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feature_1": np.concatenate([
                np.ones(50) * 10,  # First 50 rows: mean=10
                np.ones(50) * 1000,  # Last 50 rows: mean=1000
            ]),
        })

        # Compute params using only first 50 rows
        params = compute_normalization_params(df, train_end_row=50)

        mean, std = params["feature_1"]
        # Mean should be ~10, not ~505
        assert abs(mean - 10.0) < 0.1, f"Mean should be ~10, got {mean}"

    def test_excludes_bounded_features(self):
        """Naturally bounded features (RSI, etc.) should be excluded."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "Close": np.random.randn(100) * 100 + 500,  # Should be included
            "rsi_daily": np.random.rand(100) * 100,  # Should be excluded
            "stochrsi_weekly": np.random.rand(100) * 100,  # Should be excluded
            "adx_14": np.random.rand(100) * 100,  # Should be excluded
            "bb_percent_b": np.random.rand(100),  # Should be excluded
        })

        params = compute_normalization_params(df, train_end_row=70)

        assert "Close" in params, "Close should be included"
        assert "rsi_daily" not in params, "RSI should be excluded"
        assert "stochrsi_weekly" not in params, "StochRSI should be excluded"
        assert "adx_14" not in params, "ADX should be excluded"
        assert "bb_percent_b" not in params, "BB %B should be excluded"

    def test_excludes_date_column(self):
        """Date column should never be in normalization params."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "feature_1": np.random.randn(100),
        })

        params = compute_normalization_params(df, train_end_row=70)

        assert "Date" not in params


class TestNormalizeDataframe:
    """Test normalize_dataframe function."""

    def test_applies_zscore_correctly(self):
        """Normalized values should have mean ~0, std ~1 within train portion."""
        n = 100
        train_end = 70

        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feature_1": np.random.randn(n) * 10 + 50,
        })

        params = compute_normalization_params(df, train_end_row=train_end)
        df_norm = normalize_dataframe(df, params)

        # Check normalized values in train portion
        train_values = df_norm["feature_1"].iloc[:train_end].values
        assert abs(train_values.mean()) < 0.1, f"Mean should be ~0, got {train_values.mean()}"
        assert abs(train_values.std() - 1.0) < 0.1, f"Std should be ~1, got {train_values.std()}"

    def test_preserves_excluded_columns(self):
        """Date and bounded features should remain unchanged."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        rsi_values = np.random.rand(n) * 100

        df = pd.DataFrame({
            "Date": dates,
            "feature_1": np.random.randn(n) * 100,
            "rsi_daily": rsi_values,
        })

        params = compute_normalization_params(df, train_end_row=70)
        df_norm = normalize_dataframe(df, params)

        # Date should be unchanged
        pd.testing.assert_series_equal(df_norm["Date"], df["Date"])

        # RSI should be unchanged (not in params)
        np.testing.assert_array_almost_equal(
            df_norm["rsi_daily"].values, rsi_values
        )

    def test_handles_zero_std(self):
        """Feature with std=0 should normalize to 0, not NaN/inf."""
        n = 100
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "constant_feature": np.ones(n) * 42,  # Constant value, std=0
            "normal_feature": np.random.randn(n) * 10,
        })

        params = compute_normalization_params(df, train_end_row=70)
        df_norm = normalize_dataframe(df, params)

        # Constant feature should be 0 (42 - 42) / (0 + eps) = 0
        assert not df_norm["constant_feature"].isna().any(), "Should not have NaN"
        assert not np.isinf(df_norm["constant_feature"]).any(), "Should not have inf"
        # All values should be 0 (since x - mean = 0 for constant)
        assert (df_norm["constant_feature"] == 0).all(), "Constant should normalize to 0"

    def test_preserves_dtypes(self):
        """Float32 features should remain float32 after normalization."""
        n = 100
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feature_f32": np.random.randn(n).astype(np.float32) * 10,
            "feature_f64": np.random.randn(n).astype(np.float64) * 10,
        })

        params = compute_normalization_params(df, train_end_row=70)
        df_norm = normalize_dataframe(df, params)

        # Should preserve float32
        assert df_norm["feature_f32"].dtype == np.float32, (
            f"Expected float32, got {df_norm['feature_f32'].dtype}"
        )
