"""Calibration methods for probability predictions.

This module provides calibration techniques to improve the reliability of
probability predictions from classification models. Calibration maps model
outputs to well-calibrated probabilities where predicted probability matches
observed frequency.

Classes:
    PlattScaling: Logistic regression-based calibration
    IsotonicCalibration: Non-parametric isotonic regression calibration
    TemperatureScaling: Single-parameter scaling for neural networks

Functions:
    expected_calibration_error: Compute ECE metric for calibration quality
    reliability_diagram_data: Generate data for calibration plots
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


class PlattScaling:
    """Platt scaling calibration using logistic regression.

    Fits a logistic regression model to map uncalibrated probabilities
    to calibrated probabilities. This is a parametric approach that
    assumes a sigmoid relationship between raw scores and true probabilities.

    Reference:
        Platt, J. (1999). Probabilistic outputs for support vector machines.

    Example:
        >>> calibrator = PlattScaling()
        >>> calibrator.fit(val_probs, val_targets)
        >>> calibrated = calibrator.predict(test_probs)
    """

    def __init__(self):
        """Initialize PlattScaling calibrator."""
        self._model = None

    def fit(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> "PlattScaling":
        """Fit the calibration model.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)
            targets: Binary ground truth labels, shape (n_samples,)

        Returns:
            self for method chaining
        """
        probs = _to_numpy(probs)
        targets = _to_numpy(targets)

        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._model.fit(probs.reshape(-1, 1), targets)
        return self

    def predict(self, probs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Apply calibration to probabilities.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)

        Returns:
            Calibrated probabilities as numpy array, shape (n_samples,)
        """
        probs = _to_numpy(probs)
        return self._model.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibration:
    """Isotonic regression calibration.

    Fits a non-parametric isotonic regression model that learns a monotonically
    increasing mapping from uncalibrated to calibrated probabilities. More
    flexible than Platt scaling but requires more data.

    Reference:
        Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
        accurate multiclass probability estimates.

    Example:
        >>> calibrator = IsotonicCalibration()
        >>> calibrator.fit(val_probs, val_targets)
        >>> calibrated = calibrator.predict(test_probs)
    """

    def __init__(self):
        """Initialize IsotonicCalibration calibrator."""
        self._model = None

    def fit(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> "IsotonicCalibration":
        """Fit the calibration model.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)
            targets: Binary ground truth labels, shape (n_samples,)

        Returns:
            self for method chaining
        """
        probs = _to_numpy(probs)
        targets = _to_numpy(targets)

        self._model = IsotonicRegression(out_of_bounds='clip')
        self._model.fit(probs, targets)
        return self

    def predict(self, probs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Apply calibration to probabilities.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)

        Returns:
            Calibrated probabilities as numpy array, shape (n_samples,)
        """
        probs = _to_numpy(probs)
        return self._model.predict(probs)


class TemperatureScaling(nn.Module):
    """Temperature scaling calibration for neural networks.

    Learns a single temperature parameter T that scales the logits before
    applying sigmoid. This is the simplest post-hoc calibration method for
    neural networks and often works well in practice.

    calibrated_prob = sigmoid(logit / T)

    where logit = log(p / (1-p)) for original probability p.

    Reference:
        Guo, C., et al. (2017). On calibration of modern neural networks.

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.fit(val_probs, val_targets)
        >>> calibrated = calibrator.predict(test_probs)
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize TemperatureScaling calibrator.

        Args:
            temperature: Initial temperature value (default 1.0, identity)
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def fit(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> "TemperatureScaling":
        """Fit the temperature parameter using NLL minimization.

        Uses scipy L-BFGS-B to optimize the negative log-likelihood with
        respect to the temperature parameter.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)
            targets: Binary ground truth labels, shape (n_samples,)

        Returns:
            self for method chaining
        """
        probs = _to_numpy(probs)
        targets = _to_numpy(targets)

        eps = 1e-7

        def nll(T: np.ndarray) -> float:
            """Compute negative log-likelihood for temperature T."""
            # Convert probabilities to logits
            probs_clipped = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs_clipped / (1 - probs_clipped))

            # Apply temperature scaling
            scaled_logits = logits / T[0]
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)

            # Compute NLL
            loss = -np.mean(
                targets * np.log(scaled_probs)
                + (1 - targets) * np.log(1 - scaled_probs)
            )
            return loss

        # Optimize temperature
        result = minimize(
            nll,
            x0=[1.0],
            bounds=[(0.01, 10.0)],
            method='L-BFGS-B',
        )

        self.temperature = nn.Parameter(torch.tensor(result.x[0], dtype=torch.float32))
        return self

    def predict(self, probs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Apply temperature scaling to probabilities.

        Args:
            probs: Uncalibrated probability predictions, shape (n_samples,)

        Returns:
            Calibrated probabilities as numpy array, shape (n_samples,)
        """
        probs = _to_numpy(probs)
        eps = 1e-7

        # Convert to logits
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        # Apply temperature scaling
        T = self.temperature.item()
        scaled_logits = logits / T
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))

        return scaled_probs

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass for use in PyTorch training pipelines.

        Args:
            logits: Raw model logits, shape (batch_size,) or (batch_size, 1)

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature


def expected_calibration_error(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual
    accuracy across probability bins. Lower ECE indicates better calibration.

    ECE = sum_b (|B_b| / n) * |accuracy(B_b) - confidence(B_b)|

    where B_b is the set of samples in bin b.

    Args:
        probs: Predicted probabilities, shape (n_samples,)
        targets: Binary ground truth labels, shape (n_samples,)
        n_bins: Number of bins for grouping predictions
        strategy: Binning strategy, "uniform" for equal-width bins

    Returns:
        Expected calibration error as a float in [0, 1]

    Example:
        >>> ece = expected_calibration_error(predictions, labels)
        >>> print(f"ECE: {ece:.4f}")
    """
    probs = _to_numpy(probs)
    targets = _to_numpy(targets)

    n_samples = len(probs)
    if n_samples == 0:
        return 0.0

    # Create uniform bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        # Compute bin accuracy and confidence
        bin_accuracy = targets[mask].mean()
        bin_confidence = probs[mask].mean()

        # Weighted contribution to ECE
        ece += (bin_count / n_samples) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def reliability_diagram_data(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Generate data for plotting reliability diagrams.

    A reliability diagram plots predicted confidence against actual accuracy
    for each bin. A perfectly calibrated model lies on the diagonal.

    Args:
        probs: Predicted probabilities, shape (n_samples,)
        targets: Binary ground truth labels, shape (n_samples,)
        n_bins: Number of bins for grouping predictions
        strategy: Binning strategy, "uniform" for equal-width bins

    Returns:
        Dictionary containing:
            - bin_edges: Array of bin boundaries, shape (n_bins + 1,)
            - bin_accuracies: Actual accuracy per bin, shape (n_bins,)
            - bin_confidences: Mean predicted probability per bin, shape (n_bins,)
            - bin_counts: Number of samples per bin, shape (n_bins,)

    Example:
        >>> data = reliability_diagram_data(predictions, labels)
        >>> plt.plot(data['bin_confidences'], data['bin_accuracies'], 'o-')
        >>> plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
    """
    probs = _to_numpy(probs)
    targets = _to_numpy(targets)

    # Create uniform bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])

        bin_count = mask.sum()
        bin_counts[i] = bin_count

        if bin_count > 0:
            bin_accuracies[i] = targets[mask].mean()
            bin_confidences[i] = probs[mask].mean()
        else:
            # Use bin midpoint for empty bins
            bin_accuracies[i] = 0.0
            bin_confidences[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    return {
        'bin_edges': bin_edges,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }
