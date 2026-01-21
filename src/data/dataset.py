"""Financial time-series dataset for PyTorch.

Provides FinancialDataset class that loads features and constructs
binary threshold targets for financial prediction tasks.

Also provides ChunkSplitter for hybrid train/val/test splits:
- Val/Test: Non-overlapping chunks (strict isolation)
- Train: Sliding window on remaining data (maximizes samples)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Features that are naturally bounded and don't need normalization
# RSI, StochRSI: 0-100 by construction
# ADX: 0-100 by construction
# Bollinger Band %B: typically 0-1 (can exceed in extreme moves)
BOUNDED_FEATURES = frozenset({
    "rsi_daily",
    "rsi_weekly",
    "stochrsi_daily",
    "stochrsi_weekly",
    "adx_14",
    "bb_percent_b",
})

# Columns to exclude from normalization (metadata, not features)
EXCLUDED_FROM_NORM = BOUNDED_FEATURES | {"Date"}


def compute_normalization_params(
    df: pd.DataFrame,
    train_end_row: int,
    exclude_features: set[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute mean/std for features using only training rows.

    Computes Z-score normalization parameters from the first train_end_row
    rows of the dataframe. Bounded features (RSI, etc.) and Date column
    are automatically excluded.

    Args:
        df: DataFrame with feature columns.
        train_end_row: Last row (exclusive) to use for computing stats.
            Only rows 0 to train_end_row-1 are used.
        exclude_features: Additional feature names to exclude from normalization.

    Returns:
        Dict mapping feature name to (mean, std) tuple.

    Example:
        >>> params = compute_normalization_params(df, train_end_row=5000)
        >>> # params = {"Close": (150.5, 45.2), "Volume": (1e8, 5e7), ...}
    """
    if exclude_features is None:
        exclude_features = set()

    # Combine all exclusions
    all_excluded = EXCLUDED_FROM_NORM | exclude_features

    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to columns that should be normalized
    cols_to_normalize = [c for c in numeric_cols if c not in all_excluded]

    # Compute stats from training portion only
    train_df = df.iloc[:train_end_row]

    params = {}
    for col in cols_to_normalize:
        mean_val = float(train_df[col].mean())
        std_val = float(train_df[col].std())

        # Warn if std is zero (constant feature)
        if std_val == 0 or np.isnan(std_val):
            warnings.warn(
                f"Feature '{col}' has zero std in training data. "
                "Will normalize to 0.",
                UserWarning,
            )
            std_val = 0.0

        params[col] = (mean_val, std_val)

    return params


def normalize_dataframe(
    df: pd.DataFrame,
    norm_params: dict[str, tuple[float, float]],
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Apply Z-score normalization using pre-computed params.

    Applies the transformation: (x - mean) / (std + epsilon) for each
    feature in norm_params. Features not in norm_params are unchanged.

    Args:
        df: DataFrame to normalize.
        norm_params: Dict from compute_normalization_params mapping
            feature name to (mean, std) tuple.
        epsilon: Small value added to std to prevent division by zero.

    Returns:
        New DataFrame with normalized features. Original df is not modified.

    Example:
        >>> params = compute_normalization_params(df, train_end_row=5000)
        >>> df_norm = normalize_dataframe(df, params)
    """
    # Create copy to avoid modifying original
    df_norm = df.copy()

    for col, (mean_val, std_val) in norm_params.items():
        if col not in df_norm.columns:
            continue

        # Get original dtype to preserve it
        original_dtype = df_norm[col].dtype

        # Apply Z-score: (x - mean) / (std + epsilon)
        df_norm[col] = (df_norm[col] - mean_val) / (std_val + epsilon)

        # Restore original dtype
        df_norm[col] = df_norm[col].astype(original_dtype)

    return df_norm


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

    Two modes are available:

    - "scattered" (default): Val and test chunks are randomly distributed
      throughout the dataset. Good for HPO where you want val to cover
      all market regimes.

    - "contiguous": Test chunks are at the END of the dataset (most recent),
      val chunks immediately BEFORE test. Good for production-realistic
      evaluation where you train on history and test on recent/future data.

    Example (scattered mode - default):
        >>> splitter = ChunkSplitter(
        ...     total_days=8073,
        ...     context_length=60,
        ...     horizon=1,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15,
        ...     seed=42,
        ... )
        >>> splits = splitter.split()

    Example (contiguous mode - production):
        >>> splitter = ChunkSplitter(
        ...     total_days=8073,
        ...     context_length=60,
        ...     horizon=1,
        ...     val_ratio=0.01,   # Small val, just for early stopping
        ...     test_ratio=0.03,  # ~250 days for backtest
        ...     mode="contiguous",
        ... )
        >>> splits = splitter.split()
    """

    def __init__(
        self,
        total_days: int,
        context_length: int,
        horizon: int,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        mode: Literal["scattered", "contiguous"] = "scattered",
    ) -> None:
        """Initialize the chunk splitter.

        Args:
            total_days: Total number of days in the dataset.
            context_length: Number of days in input sequence (e.g., 60).
            horizon: Number of days to predict ahead (e.g., 1).
            val_ratio: Fraction of chunks to assign to validation (default 0.15).
            test_ratio: Fraction of chunks to assign to test (default 0.15).
            seed: Random seed for reproducibility (only used in scattered mode).
            mode: Split mode - "scattered" (random chunks) or "contiguous"
                (test at end, val before test). Default: "scattered".

        Raises:
            ValueError: If total_days is too small for meaningful splits.
            ValueError: If val_ratio + test_ratio >= 1.0.
            ValueError: If mode is not "scattered" or "contiguous".
        """
        self.total_days = total_days
        self.context_length = context_length
        self.horizon = horizon
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.mode = mode

        # Chunk size = context + horizon (one complete sample's span)
        self.chunk_size = context_length + horizon

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Check mode
        if self.mode not in ("scattered", "contiguous"):
            raise ValueError(
                f"mode must be 'scattered' or 'contiguous', got {self.mode!r}"
            )

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

        if self.mode == "contiguous":
            # Contiguous mode: test at end, val immediately before test
            # Test: last n_test_chunks (most recent data)
            test_chunk_ids = list(range(n_chunks - n_test_chunks, n_chunks))
            # Val: n_val_chunks immediately before test
            val_start = n_chunks - n_test_chunks - n_val_chunks
            val_chunk_ids = list(range(val_start, val_start + n_val_chunks))
        else:
            # Scattered mode (default): shuffle all chunks, pick val then test
            chunk_indices = np.arange(n_chunks)
            rng.shuffle(chunk_indices)
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


class SimpleSplitter:
    """Simple date-based contiguous splitter with sliding window for all splits.

    Creates train/val/test splits based on date boundaries with sliding window
    sampling for ALL regions (not just train). This fixes the ChunkSplitter bug
    where val/test only got 1 sample per chunk.

    Key properties:
    - Train: all samples where entire span (context+horizon) < val_start
    - Val: all samples where entire span is within [val_start, test_start)
    - Test: all samples where entire span >= test_start and fits in data
    - Strict containment: sample included only if ENTIRE span within region

    Example:
        >>> dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        >>> splitter = SimpleSplitter(
        ...     dates=dates,
        ...     context_length=60,
        ...     horizon=1,
        ...     val_start="2020-10-01",
        ...     test_start="2020-12-01",
        ... )
        >>> splits = splitter.split()
        >>> len(splits.val_indices)  # Sliding window, not 1 per chunk!
        ~60
    """

    def __init__(
        self,
        dates: pd.Series | pd.DatetimeIndex,
        context_length: int,
        horizon: int,
        val_start: str,
        test_start: str,
    ) -> None:
        """Initialize the simple splitter.

        Args:
            dates: Date column from DataFrame (must be sorted chronologically).
            context_length: Number of days in input sequence (e.g., 60).
            horizon: Number of days to predict ahead (e.g., 1).
            val_start: Start date for validation region (e.g., "2023-01-01").
            test_start: Start date for test region (e.g., "2025-01-01").

        Raises:
            ValueError: If dates not found in data or regions too small.
        """
        # Convert to DatetimeIndex if needed
        if isinstance(dates, pd.Series):
            dates = pd.DatetimeIndex(dates)
        self.dates = dates
        self.context_length = context_length
        self.horizon = horizon
        self.chunk_size = context_length + horizon

        # Parse and validate date boundaries
        self.val_start_date = pd.Timestamp(val_start)
        self.test_start_date = pd.Timestamp(test_start)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Check that val_start is in data
        if self.val_start_date not in self.dates:
            # Try to find closest date
            if self.val_start_date < self.dates.min() or self.val_start_date > self.dates.max():
                raise ValueError(
                    f"val_start {self.val_start_date.date()} not found in data. "
                    f"Data range: {self.dates.min().date()} to {self.dates.max().date()}"
                )
            # Find first date >= val_start
            mask = self.dates >= self.val_start_date
            if not mask.any():
                raise ValueError(f"val_start {self.val_start_date.date()} not found in data")
            self.val_start_date = self.dates[mask][0]

        # Check that test_start is in data
        if self.test_start_date not in self.dates:
            if self.test_start_date < self.dates.min() or self.test_start_date > self.dates.max():
                raise ValueError(
                    f"test_start {self.test_start_date.date()} not found in data. "
                    f"Data range: {self.dates.min().date()} to {self.dates.max().date()}"
                )
            mask = self.dates >= self.test_start_date
            if not mask.any():
                raise ValueError(f"test_start {self.test_start_date.date()} not found in data")
            self.test_start_date = self.dates[mask][0]

        # Get index positions
        self.val_start_idx = self.dates.get_loc(self.val_start_date)
        self.test_start_idx = self.dates.get_loc(self.test_start_date)

        # Check region sizes
        val_region_size = self.test_start_idx - self.val_start_idx
        if val_region_size < self.chunk_size:
            raise ValueError(
                f"val region too small: {val_region_size} days, "
                f"need at least {self.chunk_size} (context={self.context_length} + horizon={self.horizon})"
            )

        test_region_size = len(self.dates) - self.test_start_idx
        if test_region_size < self.chunk_size:
            raise ValueError(
                f"test region too small: {test_region_size} days, "
                f"need at least {self.chunk_size} (context={self.context_length} + horizon={self.horizon})"
            )

        train_region_size = self.val_start_idx
        if train_region_size < self.chunk_size:
            raise ValueError(
                f"train region too small: {train_region_size} days, "
                f"need at least {self.chunk_size} (context={self.context_length} + horizon={self.horizon})"
            )

    def split(self) -> SplitIndices:
        """Compute train/val/test split indices with sliding window.

        Returns:
            SplitIndices containing arrays of start indices for each split.
        """
        # Train: samples where entire span < val_start_idx
        # A sample at index t spans [t, t + chunk_size)
        # For train: t + chunk_size <= val_start_idx, so t <= val_start_idx - chunk_size
        train_max_start = self.val_start_idx - self.chunk_size
        train_indices = np.arange(0, train_max_start + 1, dtype=np.int64)

        # Val: samples where entire span is in [val_start_idx, test_start_idx)
        # Start >= val_start_idx AND t + chunk_size <= test_start_idx
        val_max_start = self.test_start_idx - self.chunk_size
        val_indices = np.arange(self.val_start_idx, val_max_start + 1, dtype=np.int64)

        # Test: samples where start >= test_start_idx AND fits in data
        test_max_start = len(self.dates) - self.chunk_size
        test_indices = np.arange(self.test_start_idx, test_max_start + 1, dtype=np.int64)

        return SplitIndices(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            chunk_size=self.chunk_size,
        )


class FinancialDataset(Dataset):
    """PyTorch Dataset for financial time-series with binary threshold targets.

    Loads feature data and close/high prices, constructs binary labels based on
    whether the maximum future HIGH price exceeds a threshold percentage gain
    over the current CLOSE price.

    Target construction rule:
        future_max = max(high[t+1 : t+1+horizon])  # Uses High prices
        label = 1 if future_max >= close[t] * (1 + threshold) else 0

    If high_prices is not provided, falls back to using close prices for
    backward compatibility.

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
        high_prices: np.ndarray | None = None,
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
            high_prices: Optional array of high prices for target calculation.
                If provided, labels use max(high[future_window]) >= close[t] * threshold.
                If None, falls back to using close prices (backward compatible).

        Raises:
            ValueError: If close_prices contains NaN values.
            ValueError: If high_prices contains NaN values (when provided).
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

        # Handle high_prices for target calculation
        if high_prices is not None:
            high = np.asarray(high_prices, dtype=np.float64)
            if np.any(np.isnan(high)):
                raise ValueError("high_prices contains NaN values")
            if len(high) != len(close):
                raise ValueError(
                    f"high_prices length ({len(high)}) must match "
                    f"close_prices length ({len(close)})"
                )
        else:
            # Backward compatibility: use close for future max if high not provided
            high = close

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
        # Pass close for threshold comparison, high for future max calculation
        self._labels = self._compute_labels(close, high)

    def _compute_labels(self, close: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Compute binary threshold labels for all valid samples.

        For each valid sample index i (0-indexed in dataset):
            - Prediction point t = i + context_length - 1 (in original data)
            - Future window: high[t+1 : t+1+horizon]
            - Label: 1 if max(future_window) >= close[t] * (1 + threshold), else 0

        Args:
            close: Array of close prices (used for threshold comparison).
            high: Array of high prices (used for future max calculation).

        Returns:
            Array of binary labels (0.0 or 1.0) for each sample.
        """
        labels = np.zeros(self._n_samples, dtype=np.float32)

        for i in range(self._n_samples):
            # Prediction point in original data
            t = i + self.context_length - 1

            # Current close price at prediction point
            current_close = close[t]

            # Future window: t+1 to t+horizon (inclusive)
            # Use HIGH prices for max calculation
            future_start = t + 1
            future_end = t + 1 + self.horizon
            future_max = high[future_start:future_end].max()

            # Threshold comparison: future max HIGH >= current CLOSE * threshold
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
