"""Financial time-series dataset for PyTorch.

Provides FinancialDataset class that loads features and constructs
binary threshold targets for financial prediction tasks.

Also provides ChunkSplitter for hybrid train/val/test splits:
- Val/Test: Non-overlapping chunks (strict isolation)
- Train: Sliding window on remaining data (maximizes samples)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndices:
    """Container for train/val/test split indices.

    Attributes:
        train_indices: Array of valid training sample start indices (sliding window).
        val_indices: Array of validation chunk start indices (non-overlapping).
        test_indices: Array of test chunk start indices (non-overlapping).
        chunk_size: Size of each val/test chunk (context_length + horizon).
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    chunk_size: int


class ChunkSplitter:
    """Hybrid chunk-based train/val/test splitter for time-series data.

    Creates data splits that maximize training samples while ensuring
    strict isolation of validation and test sets:

    - Val/Test: Non-overlapping chunks of size (context_length + horizon)
    - Train: Sliding window samples that don't overlap any val/test chunk

    This prevents any data leakage between train and val/test sets while
    providing ~30x more training samples than pure non-overlapping splits.

    Example:
        >>> splitter = ChunkSplitter(
        ...     total_days=8073,
        ...     context_length=60,
        ...     horizon=1,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15,
        ...     seed=42,
        ... )
        >>> splits = splitter.split()
        >>> print(f"Train: {len(splits.train_indices)}, Val: {len(splits.val_indices)}")
    """

    def __init__(
        self,
        total_days: int,
        context_length: int,
        horizon: int,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initialize the chunk splitter.

        Args:
            total_days: Total number of days in the dataset.
            context_length: Number of days in input sequence (e.g., 60).
            horizon: Number of days to predict ahead (e.g., 1).
            val_ratio: Fraction of chunks to assign to validation (default 0.15).
            test_ratio: Fraction of chunks to assign to test (default 0.15).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If total_days is too small for meaningful splits.
            ValueError: If val_ratio + test_ratio >= 1.0.
        """
        self.total_days = total_days
        self.context_length = context_length
        self.horizon = horizon
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Chunk size = context + horizon (one complete sample's span)
        self.chunk_size = context_length + horizon

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Check ratio constraints
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError(
                f"val_ratio ({self.val_ratio}) + test_ratio ({self.test_ratio}) "
                f"must be < 1.0, got {self.val_ratio + self.test_ratio}"
            )

        # Check minimum data size
        # Need at least 3 chunks (1 train, 1 val, 1 test) + some margin
        min_chunks = 5
        min_days = min_chunks * self.chunk_size
        if self.total_days < min_days:
            raise ValueError(
                f"Insufficient data: {self.total_days} days is too small. "
                f"Need at least {min_days} days for context_length={self.context_length}, "
                f"horizon={self.horizon}"
            )

    def split(self) -> SplitIndices:
        """Compute train/val/test split indices.

        Returns:
            SplitIndices containing arrays of start indices for each split.
        """
        rng = np.random.default_rng(self.seed)

        # Calculate total number of non-overlapping chunks
        n_chunks = self.total_days // self.chunk_size

        # Calculate chunk counts for each split
        n_val_chunks = max(1, int(n_chunks * self.val_ratio))
        n_test_chunks = max(1, int(n_chunks * self.test_ratio))
        n_train_chunks = n_chunks - n_val_chunks - n_test_chunks

        if n_train_chunks < 1:
            raise ValueError(
                f"Not enough chunks for train set: {n_chunks} total chunks, "
                f"{n_val_chunks} val, {n_test_chunks} test"
            )

        # Create chunk indices and shuffle
        chunk_indices = np.arange(n_chunks)
        rng.shuffle(chunk_indices)

        # Assign chunks to splits
        val_chunk_ids = sorted(chunk_indices[:n_val_chunks])
        test_chunk_ids = sorted(chunk_indices[n_val_chunks : n_val_chunks + n_test_chunks])

        # Convert chunk IDs to start day indices
        val_indices = np.array([cid * self.chunk_size for cid in val_chunk_ids])
        test_indices = np.array([cid * self.chunk_size for cid in test_chunk_ids])

        # Compute blocked day ranges (val + test regions)
        blocked_ranges = []
        for start in val_indices:
            blocked_ranges.append((start, start + self.chunk_size))
        for start in test_indices:
            blocked_ranges.append((start, start + self.chunk_size))
        blocked_ranges.sort()

        # Compute valid train sample start indices (sliding window)
        # A sample at position t spans [t, t + chunk_size)
        # It's valid if none of those days fall in blocked ranges
        train_indices = self._compute_valid_train_indices(blocked_ranges)

        return SplitIndices(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            chunk_size=self.chunk_size,
        )

    def _compute_valid_train_indices(
        self, blocked_ranges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Compute valid training sample start indices.

        A training sample at position t is valid if its span [t, t + chunk_size)
        doesn't overlap any blocked (val/test) range.

        Args:
            blocked_ranges: List of (start, end) tuples for val/test regions.

        Returns:
            Array of valid training sample start indices.
        """
        # Maximum valid start position
        max_start = self.total_days - self.chunk_size

        # Build set of blocked days for fast lookup
        blocked_days = set()
        for start, end in blocked_ranges:
            blocked_days.update(range(start, end))

        # Find all valid train start positions
        valid_starts = []
        for t in range(max_start + 1):
            # Check if sample span [t, t + chunk_size) overlaps blocked days
            sample_days = range(t, t + self.chunk_size)
            if not any(day in blocked_days for day in sample_days):
                valid_starts.append(t)

        return np.array(valid_starts, dtype=np.int64)

    def get_hpo_subset(
        self,
        splits: SplitIndices,
        fraction: float = 0.3,
        seed: int | None = None,
    ) -> np.ndarray:
        """Get a subset of training indices for faster HPO.

        Args:
            splits: SplitIndices from split() method.
            fraction: Fraction of train indices to use (default 0.3 = 30%).
            seed: Random seed for subset selection (uses self.seed if None).

        Returns:
            Array of training indices for HPO.
        """
        if seed is None:
            seed = self.seed

        rng = np.random.default_rng(seed)
        n_samples = int(len(splits.train_indices) * fraction)
        n_samples = max(1, n_samples)  # At least 1 sample

        # Randomly select subset
        selected_indices = rng.choice(
            splits.train_indices, size=n_samples, replace=False
        )

        return np.sort(selected_indices)


class FinancialDataset(Dataset):
    """PyTorch Dataset for financial time-series with binary threshold targets.

    Loads feature data and close prices, constructs binary labels based on
    whether the maximum future price exceeds a threshold percentage gain.

    Target construction rule:
        future_max = max(close[t+1 : t+1+horizon])
        label = 1 if future_max >= close[t] * (1 + threshold) else 0

    Attributes:
        features: numpy array of shape (n_samples, context_length, n_features)
        labels: numpy array of shape (n_samples,) with binary labels
        context_length: number of time steps in each input sequence
        horizon: number of future time steps to consider for target
        threshold: percentage threshold for positive label (e.g., 0.01 for 1%)
    """

    # Columns to exclude from features (metadata only - OHLCV are valid features)
    EXCLUDED_COLUMNS = {"Date"}
    # Numeric dtypes that are valid for features
    NUMERIC_DTYPES = {"float64", "float32", "int64", "int32", "float16", "int16", "int8"}

    def __init__(
        self,
        features_df: pd.DataFrame,
        close_prices: np.ndarray,
        context_length: int,
        horizon: int,
        threshold: float,
        feature_columns: list[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            features_df: DataFrame with 'Date' column and feature columns.
            close_prices: Array of close prices aligned with features_df rows.
            context_length: Number of time steps in input sequence.
            horizon: Number of future time steps for target calculation.
            threshold: Percentage threshold for positive label (e.g., 0.01 = 1%).
            feature_columns: Optional explicit list of feature columns to use.
                If None, auto-discovers numeric columns (excluding only Date).

        Raises:
            ValueError: If close_prices contains NaN values.
            ValueError: If sequence is too short for context_length + horizon.
            ValueError: If specified feature_columns are missing or non-numeric.
        """
        self.context_length = context_length
        self.horizon = horizon
        self.threshold = threshold

        # Convert close_prices to numpy array if needed
        close = np.asarray(close_prices, dtype=np.float64)

        # Validate no NaN in close prices
        if np.any(np.isnan(close)):
            raise ValueError("close_prices contains NaN values")

        # Determine feature columns
        if feature_columns is not None:
            # Explicit feature columns - validate they exist and are numeric
            feature_cols = self._validate_explicit_features(features_df, feature_columns)
        else:
            # Auto-discover: exclude Date, keep only numeric
            feature_cols = self._discover_numeric_features(features_df)

        self.feature_columns = feature_cols
        features = features_df[feature_cols].values.astype(np.float32)

        n_rows = len(features)

        # Validate sequence length
        min_required = context_length + horizon
        if n_rows < min_required:
            raise ValueError(
                f"Sequence too short: got {n_rows} rows, "
                f"need at least {min_required} (context_length={context_length} + horizon={horizon})"
            )

        # Calculate number of valid samples
        # First valid sample: uses rows [0, context_length) as input, predicts from row context_length-1
        # Last valid sample: must have horizon rows after the prediction point
        # Valid prediction points: context_length-1 to n_rows-horizon-1 (inclusive)
        # Number of samples: (n_rows - horizon - 1) - (context_length - 1) + 1 = n_rows - context_length - horizon + 1
        self._n_samples = n_rows - context_length - horizon + 1

        # Store features
        self._features = features

        # Pre-compute all labels using vectorized operations
        self._labels = self._compute_labels(close)

    def _compute_labels(self, close: np.ndarray) -> np.ndarray:
        """Compute binary threshold labels for all valid samples.

        For each valid sample index i (0-indexed in dataset):
            - Prediction point t = i + context_length - 1 (in original data)
            - Future window: close[t+1 : t+1+horizon]
            - Label: 1 if max(future_window) >= close[t] * (1 + threshold), else 0

        Args:
            close: Array of close prices.

        Returns:
            Array of binary labels (0.0 or 1.0) for each sample.
        """
        labels = np.zeros(self._n_samples, dtype=np.float32)

        for i in range(self._n_samples):
            # Prediction point in original data
            t = i + self.context_length - 1

            # Current close price at prediction point
            current_close = close[t]

            # Future window: t+1 to t+horizon (inclusive), i.e., close[t+1 : t+1+horizon]
            future_start = t + 1
            future_end = t + 1 + self.horizon
            future_max = close[future_start:future_end].max()

            # Threshold comparison
            threshold_price = current_close * (1 + self.threshold)
            labels[i] = 1.0 if future_max >= threshold_price else 0.0

        return labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index (0 to len(dataset)-1).

        Returns:
            Tuple of (input_tensor, target_tensor) where:
                - input_tensor: shape (context_length, n_features), dtype float32
                - target_tensor: shape (1,), dtype float32
        """
        # Input sequence: rows [idx, idx + context_length)
        start = idx
        end = idx + self.context_length
        x = self._features[start:end]

        # Target label (pre-computed)
        y = self._labels[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

    def _validate_explicit_features(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> list[str]:
        """Validate explicitly specified feature columns.

        Args:
            df: DataFrame to validate against.
            feature_columns: List of column names to validate.

        Returns:
            Validated list of feature columns.

        Raises:
            ValueError: If any columns are missing or non-numeric.
        """
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in DataFrame: {missing}")

        non_numeric = [
            c for c in feature_columns
            if str(df[c].dtype) not in self.NUMERIC_DTYPES
        ]
        if non_numeric:
            raise ValueError(
                f"Feature columns must be numeric. Non-numeric columns: "
                f"{[(c, str(df[c].dtype)) for c in non_numeric]}"
            )

        return feature_columns

    def _discover_numeric_features(self, df: pd.DataFrame) -> list[str]:
        """Auto-discover numeric feature columns.

        Excludes Date column, keeps only numeric types (OHLCV included).

        Args:
            df: DataFrame to discover features from.

        Returns:
            List of numeric feature column names.
        """
        feature_cols = []
        skipped = []

        for col in df.columns:
            if col in self.EXCLUDED_COLUMNS:
                continue
            if str(df[col].dtype) in self.NUMERIC_DTYPES:
                feature_cols.append(col)
            else:
                skipped.append((col, str(df[col].dtype)))

        if skipped:
            import warnings
            warnings.warn(
                f"Skipped non-numeric columns during feature discovery: {skipped}",
                UserWarning,
            )

        if not feature_cols:
            raise ValueError(
                "No numeric feature columns found after excluding Date. "
                f"Columns in DataFrame: {list(df.columns)}"
            )

        return feature_cols
