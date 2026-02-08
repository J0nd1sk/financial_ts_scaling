"""Regime detection for financial time-series models.

Implements regime detection strategies that identify different market states
(e.g., high/low volatility, bull/bear/sideways) and condition training
accordingly.

Regime conditioning can:
- Weight loss by regime (emphasize certain market conditions)
- Add regime embedding to model input
- Use regime-specific model heads

Strategies:
- Volatility-based: High/medium/low volatility regimes
- Trend-based: Bull/bear/sideways using SMA or ADX
- Cluster-based: Learned regime discovery via clustering
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import pandas as pd


class RegimeDetector(ABC):
    """Base class for regime detection."""

    @property
    @abstractmethod
    def n_regimes(self) -> int:
        """Return number of regimes."""
        pass

    @abstractmethod
    def detect(self, features: np.ndarray, index: int) -> int:
        """Detect regime for a single sample.

        Args:
            features: Feature array for the context window.
                Shape: (context_length, n_features)
            index: Index in the original dataset (for accessing global info).

        Returns:
            Regime index (0 to n_regimes-1).
        """
        pass

    def detect_batch(
        self, features: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Detect regimes for a batch of samples.

        Args:
            features: Batch of feature arrays.
                Shape: (batch_size, context_length, n_features)
            indices: Batch of dataset indices.

        Returns:
            Array of regime indices, shape (batch_size,).
        """
        regimes = [
            self.detect(features[i], int(indices[i]))
            for i in range(len(features))
        ]
        return np.array(regimes, dtype=np.int64)


class VolatilityRegimeDetector(RegimeDetector):
    """Detect volatility regimes: low, medium, high.

    Uses rolling standard deviation of returns to classify
    the market regime for each sample.

    Args:
        volatility_thresholds: Tuple of (low_threshold, high_threshold).
            Volatility below low_threshold -> regime 0 (low)
            Volatility between thresholds -> regime 1 (medium)
            Volatility above high_threshold -> regime 2 (high)
            Default: (0.01, 0.02) for daily returns.
        close_col_idx: Index of Close price column. Default 0.

    Example:
        >>> detector = VolatilityRegimeDetector(
        ...     volatility_thresholds=(0.01, 0.02)
        ... )
        >>> regime = detector.detect(features, index)
    """

    def __init__(
        self,
        volatility_thresholds: tuple[float, float] = (0.01, 0.02),
        close_col_idx: int = 0,
    ) -> None:
        self.low_threshold, self.high_threshold = volatility_thresholds
        self.close_col_idx = close_col_idx

    @property
    def n_regimes(self) -> int:
        return 3

    def detect(self, features: np.ndarray, index: int) -> int:
        """Detect volatility regime for a sample.

        Args:
            features: Feature window, shape (context_length, n_features).
            index: Not used.

        Returns:
            Regime: 0 (low), 1 (medium), or 2 (high) volatility.
        """
        close = features[:, self.close_col_idx]

        # Compute returns
        returns = np.diff(close) / (np.abs(close[:-1]) + 1e-8)

        # Volatility = std of returns
        vol = np.std(returns)

        if vol < self.low_threshold:
            return 0  # Low volatility
        elif vol < self.high_threshold:
            return 1  # Medium volatility
        else:
            return 2  # High volatility


class TrendRegimeDetector(RegimeDetector):
    """Detect trend regimes: bear, sideways, bull.

    Uses the relationship between short-term and long-term moving averages
    to classify the market trend.

    Args:
        method: "sma" (SMA crossover) or "adx" (ADX-based). Default "sma".
        short_window: Short MA period. Default 10.
        long_window: Long MA period. Default 30.
        adx_threshold: ADX threshold for trending market (adx method). Default 25.
        close_col_idx: Index of Close price column. Default 0.

    Example:
        >>> detector = TrendRegimeDetector(method="sma", short_window=10)
        >>> regime = detector.detect(features, index)
    """

    def __init__(
        self,
        method: str = "sma",
        short_window: int = 10,
        long_window: int = 30,
        adx_threshold: float = 25.0,
        close_col_idx: int = 0,
    ) -> None:
        if method not in ("sma", "adx"):
            raise ValueError(f"method must be 'sma' or 'adx', got {method}")
        self.method = method
        self.short_window = short_window
        self.long_window = long_window
        self.adx_threshold = adx_threshold
        self.close_col_idx = close_col_idx

    @property
    def n_regimes(self) -> int:
        return 3

    def detect(self, features: np.ndarray, index: int) -> int:
        """Detect trend regime for a sample.

        Args:
            features: Feature window, shape (context_length, n_features).
            index: Not used.

        Returns:
            Regime: 0 (bear), 1 (sideways), or 2 (bull).
        """
        close = features[:, self.close_col_idx]

        if self.method == "sma":
            return self._detect_sma(close)
        else:
            return self._detect_adx(features)

    def _detect_sma(self, close: np.ndarray) -> int:
        """SMA-based trend detection."""
        if len(close) < self.long_window:
            return 1  # Default to sideways if not enough data

        # Compute SMAs
        short_ma = np.mean(close[-self.short_window:])
        long_ma = np.mean(close[-self.long_window:])

        # Trend strength based on divergence
        pct_diff = (short_ma - long_ma) / (long_ma + 1e-8)

        if pct_diff > 0.02:  # 2% above = bull
            return 2
        elif pct_diff < -0.02:  # 2% below = bear
            return 0
        else:
            return 1  # Sideways

    def _detect_adx(self, features: np.ndarray) -> int:
        """ADX-based trend detection.

        Note: Simplified ADX approximation. For proper ADX, use full
        high/low/close with proper smoothing.
        """
        close = features[:, self.close_col_idx]

        if len(close) < self.long_window:
            return 1  # Default to sideways

        # Approximate trend strength from price movement
        returns = np.diff(close) / (np.abs(close[:-1]) + 1e-8)

        # Directional movement: sum of positive vs negative returns
        pos_returns = np.sum(returns[returns > 0])
        neg_returns = np.abs(np.sum(returns[returns < 0]))

        total = pos_returns + neg_returns
        if total < 1e-8:
            return 1  # No movement = sideways

        # Approximate ADX-like metric
        dm_diff = abs(pos_returns - neg_returns)
        adx_approx = (dm_diff / total) * 100

        if adx_approx < self.adx_threshold:
            return 1  # Low ADX = sideways

        # Trending: determine direction
        if pos_returns > neg_returns:
            return 2  # Bull trend
        else:
            return 0  # Bear trend


class ClusterRegimeDetector(RegimeDetector):
    """Detect regimes using clustering on feature statistics.

    Learns regimes from data by clustering samples based on their
    statistical properties (mean, std, trend of features).

    Args:
        n_clusters: Number of regimes to detect. Default 3.
        feature_indices: Indices of features to use for clustering.
            Default None uses all features.
        seed: Random seed for clustering. Default 42.

    Example:
        >>> detector = ClusterRegimeDetector(n_clusters=4)
        >>> detector.fit(all_features)  # Pre-compute clusters
        >>> regime = detector.detect(features, index)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        feature_indices: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        self._n_clusters = n_clusters
        self.feature_indices = feature_indices
        self.seed = seed
        self._fitted = False
        self._cluster_centers: np.ndarray | None = None
        self._sample_regimes: np.ndarray | None = None

    @property
    def n_regimes(self) -> int:
        return self._n_clusters

    def fit(self, all_features: np.ndarray) -> None:
        """Fit clustering on all samples.

        Args:
            all_features: All feature windows.
                Shape: (n_samples, context_length, n_features)
        """
        from sklearn.cluster import KMeans

        # Extract statistical features for clustering
        cluster_features = self._extract_cluster_features(all_features)

        # Fit KMeans
        kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self.seed,
            n_init=10,
        )
        self._sample_regimes = kmeans.fit_predict(cluster_features)
        self._cluster_centers = kmeans.cluster_centers_
        self._fitted = True

    def _extract_cluster_features(self, features: np.ndarray) -> np.ndarray:
        """Extract statistical features for clustering.

        Args:
            features: Feature windows, shape (n_samples, seq_len, n_features)
                or (seq_len, n_features) for single sample.

        Returns:
            Statistical features for clustering.
        """
        if features.ndim == 2:
            features = features[np.newaxis, ...]

        n_samples = features.shape[0]

        # Select feature columns
        if self.feature_indices is not None:
            features = features[:, :, self.feature_indices]

        # Compute per-sample statistics: mean, std, trend for each feature
        means = np.mean(features, axis=1)  # (n_samples, n_features)
        stds = np.std(features, axis=1)  # (n_samples, n_features)

        # Trend: slope of linear fit
        seq_len = features.shape[1]
        t = np.arange(seq_len)
        trends = np.zeros((n_samples, features.shape[2]))
        for i in range(n_samples):
            for j in range(features.shape[2]):
                # Simple linear regression slope
                y = features[i, :, j]
                slope = np.polyfit(t, y, 1)[0]
                trends[i, j] = slope

        # Concatenate all statistics
        cluster_features = np.concatenate([means, stds, trends], axis=1)

        return cluster_features

    def detect(self, features: np.ndarray, index: int) -> int:
        """Detect regime using pre-computed clusters.

        If fitted, uses stored sample regime.
        If not fitted, assigns to nearest cluster center.

        Args:
            features: Feature window.
            index: Sample index in dataset.

        Returns:
            Regime index.
        """
        if self._fitted and self._sample_regimes is not None:
            if 0 <= index < len(self._sample_regimes):
                return int(self._sample_regimes[index])

        # Fallback: compute cluster assignment for this sample
        if self._cluster_centers is not None:
            cluster_features = self._extract_cluster_features(features)
            distances = np.linalg.norm(
                cluster_features - self._cluster_centers, axis=1
            )
            return int(np.argmin(distances))

        return 0  # Default regime


class RegimeLossWeighter:
    """Apply regime-dependent loss weighting.

    Weights the loss for each sample based on its regime, allowing
    the model to focus more on certain market conditions.

    Args:
        regime_detector: RegimeDetector instance.
        regime_weights: Dict mapping regime index to weight.
            Default: equal weights of 1.0 for all regimes.

    Example:
        >>> detector = VolatilityRegimeDetector()
        >>> weighter = RegimeLossWeighter(
        ...     detector,
        ...     regime_weights={0: 1.0, 1: 1.5, 2: 2.0}  # Weight high vol more
        ... )
        >>> weights = weighter.get_weights(features, indices)
        >>> loss = (criterion(pred, target) * weights).mean()
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        regime_weights: dict[int, float] | None = None,
    ) -> None:
        self.regime_detector = regime_detector
        self.regime_weights = regime_weights or {
            i: 1.0 for i in range(regime_detector.n_regimes)
        }

    def get_weights(
        self,
        features: np.ndarray,
        indices: np.ndarray,
    ) -> torch.Tensor:
        """Get loss weights for a batch based on regimes.

        Args:
            features: Batch features, shape (batch, seq_len, n_features).
            indices: Batch indices.

        Returns:
            Weight tensor, shape (batch,).
        """
        regimes = self.regime_detector.detect_batch(features, indices)
        weights = np.array([self.regime_weights.get(r, 1.0) for r in regimes])
        return torch.tensor(weights, dtype=torch.float32)


class RegimeEmbedding(nn.Module):
    """Learnable regime embedding added to model input.

    Adds a regime-specific embedding to each timestep, allowing the model
    to condition its predictions on the detected regime.

    Args:
        n_regimes: Number of regimes.
        embedding_dim: Dimension of embedding (should match model's feature dim).

    Example:
        >>> regime_embed = RegimeEmbedding(n_regimes=3, embedding_dim=128)
        >>> # In forward pass:
        >>> regime_ids = detector.detect_batch(x, indices)
        >>> x = x + regime_embed(regime_ids)
    """

    def __init__(self, n_regimes: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_regimes, embedding_dim)

    def forward(self, regime_ids: torch.Tensor) -> torch.Tensor:
        """Get regime embeddings.

        Args:
            regime_ids: Regime indices, shape (batch,).

        Returns:
            Embeddings, shape (batch, embedding_dim).
        """
        return self.embedding(regime_ids)


def get_regime_detector(
    regime_strategy: str | None,
    regime_params: dict[str, Any] | None = None,
) -> RegimeDetector | None:
    """Factory function to create regime detector from experiment spec.

    Args:
        regime_strategy: Strategy type ("volatility", "trend", "cluster").
            None returns None.
        regime_params: Parameters for the detector:
            - volatility: {"thresholds": (0.01, 0.02), "close_col_idx": 0}
            - trend: {"method": "sma", "short_window": 10, "long_window": 30}
            - cluster: {"n_clusters": 3, "feature_indices": None}

    Returns:
        RegimeDetector instance, or None if regime_strategy is None.

    Raises:
        ValueError: If regime_strategy is unknown.
    """
    if regime_strategy is None:
        return None

    params = regime_params or {}

    if regime_strategy == "volatility":
        thresholds = params.get("thresholds", (0.01, 0.02))
        return VolatilityRegimeDetector(
            volatility_thresholds=thresholds,
            close_col_idx=params.get("close_col_idx", 0),
        )

    elif regime_strategy == "trend":
        return TrendRegimeDetector(
            method=params.get("method", "sma"),
            short_window=params.get("short_window", 10),
            long_window=params.get("long_window", 30),
            adx_threshold=params.get("adx_threshold", 25.0),
            close_col_idx=params.get("close_col_idx", 0),
        )

    elif regime_strategy == "cluster":
        return ClusterRegimeDetector(
            n_clusters=params.get("n_clusters", 3),
            feature_indices=params.get("feature_indices"),
            seed=params.get("seed", 42),
        )

    else:
        raise ValueError(f"Unknown regime_strategy: {regime_strategy}")
