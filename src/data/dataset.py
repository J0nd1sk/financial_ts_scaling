"""Financial time-series dataset for PyTorch.

Provides FinancialDataset class that loads features and constructs
binary threshold targets for financial prediction tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def __init__(
        self,
        features_df: pd.DataFrame,
        close_prices: np.ndarray,
        context_length: int,
        horizon: int,
        threshold: float,
    ) -> None:
        """Initialize the dataset.

        Args:
            features_df: DataFrame with 'Date' column and feature columns.
            close_prices: Array of close prices aligned with features_df rows.
            context_length: Number of time steps in input sequence.
            horizon: Number of future time steps for target calculation.
            threshold: Percentage threshold for positive label (e.g., 0.01 = 1%).

        Raises:
            ValueError: If close_prices contains NaN values.
            ValueError: If sequence is too short for context_length + horizon.
        """
        self.context_length = context_length
        self.horizon = horizon
        self.threshold = threshold

        # Convert close_prices to numpy array if needed
        close = np.asarray(close_prices, dtype=np.float64)

        # Validate no NaN in close prices
        if np.any(np.isnan(close)):
            raise ValueError("close_prices contains NaN values")

        # Extract feature columns (exclude 'Date')
        feature_cols = [c for c in features_df.columns if c != "Date"]
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
