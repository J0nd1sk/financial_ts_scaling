"""Curriculum learning for financial time-series models.

Implements curriculum learning strategies that train on easier samples first,
gradually introducing harder samples as training progresses.

Curriculum strategies:
- Loss-based: Start with low-loss samples (easier for model to learn)
- Confidence-based: Start with high-confidence predictions
- Volatility-based: Start with low-volatility periods (financial-specific)
- Anti-curriculum: Start with hard samples (for comparison)

Reference: Bengio et al. "Curriculum Learning" (2009)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class CurriculumSampler(Sampler):
    """Sampler that implements curriculum learning.

    Gradually increases the pool of available samples during training,
    starting with "easier" samples based on a difficulty metric.

    Args:
        dataset: The dataset to sample from.
        difficulty_scores: Per-sample difficulty scores. Lower = easier.
            Shape: (len(dataset),)
        initial_pct: Initial percentage of easiest samples to use. Default 0.3.
        growth_rate: Percentage of samples to add each epoch. Default 0.1.
        seed: Random seed for reproducibility. Default 42.

    Example:
        >>> # With pre-computed difficulty scores
        >>> scores = compute_loss_difficulty(model, dataset)
        >>> sampler = CurriculumSampler(
        ...     dataset=dataset,
        ...     difficulty_scores=scores,
        ...     initial_pct=0.3,
        ...     growth_rate=0.1,
        ... )
        >>> for epoch in range(10):
        ...     sampler.set_epoch(epoch)
        ...     for batch in DataLoader(dataset, sampler=sampler):
        ...         train_step(batch)
    """

    def __init__(
        self,
        dataset: "Dataset",
        difficulty_scores: np.ndarray,
        initial_pct: float = 0.3,
        growth_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.difficulty_scores = np.asarray(difficulty_scores)
        self.initial_pct = initial_pct
        self.growth_rate = growth_rate
        self.seed = seed
        self._epoch = 0

        # Validate
        if len(self.difficulty_scores) != len(dataset):
            raise ValueError(
                f"difficulty_scores length ({len(self.difficulty_scores)}) "
                f"must match dataset length ({len(dataset)})"
            )

        # Sort indices by difficulty (easiest first)
        self._sorted_indices = np.argsort(self.difficulty_scores)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for curriculum progression.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        self._epoch = epoch

    def _get_current_pct(self) -> float:
        """Get percentage of samples to use for current epoch."""
        pct = self.initial_pct + self._epoch * self.growth_rate
        return min(1.0, pct)

    def __iter__(self) -> Iterator[int]:
        """Return iterator over sample indices for current epoch."""
        # Calculate how many samples to include
        current_pct = self._get_current_pct()
        n_samples = max(1, int(len(self.dataset) * current_pct))

        # Get the easiest n_samples based on difficulty
        available_indices = self._sorted_indices[:n_samples]

        # Shuffle the available indices for this epoch
        rng = np.random.default_rng(self.seed + self._epoch)
        shuffled = rng.permutation(available_indices)

        return iter(shuffled.tolist())

    def __len__(self) -> int:
        """Return number of samples for current epoch."""
        current_pct = self._get_current_pct()
        return max(1, int(len(self.dataset) * current_pct))


class BaseDifficultyScorer(ABC):
    """Base class for computing sample difficulty scores."""

    @abstractmethod
    def compute_scores(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> np.ndarray:
        """Compute difficulty scores for all samples.

        Args:
            model: The model to use for scoring.
            dataloader: DataLoader with the full dataset (no shuffle, batch=1).
            device: Device to run on.

        Returns:
            Array of difficulty scores, one per sample. Lower = easier.
        """
        pass


class LossDifficultyScorer(BaseDifficultyScorer):
    """Compute difficulty based on per-sample loss.

    Samples with higher loss are considered harder to learn.

    Args:
        criterion: Loss function to use. Default BCELoss.
    """

    def __init__(self, criterion: torch.nn.Module | None = None) -> None:
        self.criterion = criterion or torch.nn.BCELoss(reduction="none")

    def compute_scores(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> np.ndarray:
        """Compute loss-based difficulty scores.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader (should have shuffle=False, batch_size=1 or small).
            device: Device string.

        Returns:
            Array of losses, one per sample.
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = model(batch_x)
                loss = self.criterion(pred, batch_y)

                # Append per-sample losses
                scores.extend(loss.view(-1).cpu().numpy())

        return np.array(scores)


class ConfidenceDifficultyScorer(BaseDifficultyScorer):
    """Compute difficulty based on prediction confidence.

    Low-confidence predictions indicate harder samples.
    Confidence = max(p, 1-p) for binary classification.

    Difficulty = 1 - confidence (so low confidence = high difficulty)
    """

    def compute_scores(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> np.ndarray:
        """Compute confidence-based difficulty scores.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader.
            device: Device string.

        Returns:
            Array of (1 - confidence) scores, one per sample.
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)

                pred = model(batch_x).view(-1)

                # Confidence = max(p, 1-p)
                # Difficulty = 1 - confidence
                confidence = torch.max(pred, 1 - pred)
                difficulty = 1 - confidence

                scores.extend(difficulty.cpu().numpy())

        return np.array(scores)


class VolatilityDifficultyScorer(BaseDifficultyScorer):
    """Compute difficulty based on local volatility of features.

    High-volatility periods are considered harder to predict.
    This is a financial-specific curriculum strategy.

    Volatility is computed as the standard deviation of returns
    within the context window.

    Args:
        close_col_idx: Index of the Close price column in features.
            If None, uses the first column. Default None.
    """

    def __init__(self, close_col_idx: int | None = None) -> None:
        self.close_col_idx = close_col_idx or 0

    def compute_scores(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> np.ndarray:
        """Compute volatility-based difficulty scores.

        Note: model is not used here; difficulty is based on input features.

        Args:
            model: Not used (included for interface compatibility).
            dataloader: DataLoader.
            device: Not used.

        Returns:
            Array of volatility scores, one per sample.
        """
        scores = []

        for batch_x, _ in dataloader:
            # batch_x shape: (batch, seq_len, n_features)
            for sample in batch_x:
                # Get close prices for this sample
                close = sample[:, self.close_col_idx].numpy()

                # Compute returns
                returns = np.diff(close) / (close[:-1] + 1e-8)

                # Volatility = std of returns
                volatility = np.std(returns)
                scores.append(volatility)

        return np.array(scores)


class AntiCurriculumSampler(CurriculumSampler):
    """Anti-curriculum: start with hardest samples.

    This is the inverse of curriculum learning - starts with the hardest
    samples and gradually adds easier ones. Useful as a baseline to verify
    that curriculum learning actually helps.

    Args:
        dataset: The dataset to sample from.
        difficulty_scores: Per-sample difficulty scores. Lower = easier.
        initial_pct: Initial percentage of hardest samples to use. Default 0.3.
        growth_rate: Percentage of samples to add each epoch. Default 0.1.
        seed: Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        dataset: "Dataset",
        difficulty_scores: np.ndarray,
        initial_pct: float = 0.3,
        growth_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, difficulty_scores, initial_pct, growth_rate, seed)

        # Override: sort by difficulty descending (hardest first)
        self._sorted_indices = np.argsort(self.difficulty_scores)[::-1]


def get_curriculum_sampler(
    curriculum_strategy: str | None,
    curriculum_params: dict[str, Any] | None,
    dataset: "Dataset",
    model: torch.nn.Module | None = None,
    dataloader: DataLoader | None = None,
    device: str = "cpu",
) -> CurriculumSampler | None:
    """Factory function to create curriculum sampler from experiment spec.

    Args:
        curriculum_strategy: Strategy type ("loss", "confidence", "volatility",
            "anti"). None returns None.
        curriculum_params: Parameters for the sampler:
            - initial_pct: Initial percentage of samples (default 0.3)
            - growth_rate: Rate to add samples per epoch (default 0.1)
            - close_col_idx: For volatility, column index of Close price
        dataset: The dataset to sample from.
        model: Trained model for loss/confidence scoring. Required for loss/confidence.
        dataloader: DataLoader for computing scores. Required for loss/confidence.
        device: Device string for scoring.

    Returns:
        CurriculumSampler instance, or None if curriculum_strategy is None.

    Raises:
        ValueError: If curriculum_strategy is unknown or required args missing.
    """
    if curriculum_strategy is None:
        return None

    params = curriculum_params or {}
    initial_pct = params.get("initial_pct", 0.3)
    growth_rate = params.get("growth_rate", 0.1)

    if curriculum_strategy == "loss":
        if model is None or dataloader is None:
            raise ValueError("loss curriculum requires model and dataloader")

        scorer = LossDifficultyScorer()
        scores = scorer.compute_scores(model, dataloader, device)

        return CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=initial_pct,
            growth_rate=growth_rate,
        )

    elif curriculum_strategy == "confidence":
        if model is None or dataloader is None:
            raise ValueError("confidence curriculum requires model and dataloader")

        scorer = ConfidenceDifficultyScorer()
        scores = scorer.compute_scores(model, dataloader, device)

        return CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=initial_pct,
            growth_rate=growth_rate,
        )

    elif curriculum_strategy == "volatility":
        if dataloader is None:
            raise ValueError("volatility curriculum requires dataloader")

        close_col_idx = params.get("close_col_idx", 0)
        scorer = VolatilityDifficultyScorer(close_col_idx=close_col_idx)
        scores = scorer.compute_scores(None, dataloader, device)

        return CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=initial_pct,
            growth_rate=growth_rate,
        )

    elif curriculum_strategy == "anti":
        if model is None or dataloader is None:
            raise ValueError("anti-curriculum requires model and dataloader")

        # Use loss-based difficulty but sample hardest first
        scorer = LossDifficultyScorer()
        scores = scorer.compute_scores(model, dataloader, device)

        return AntiCurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=initial_pct,
            growth_rate=growth_rate,
        )

    else:
        raise ValueError(f"Unknown curriculum_strategy: {curriculum_strategy}")
