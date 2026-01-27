"""Tests for calibration module."""

import numpy as np
import pytest
import torch

from src.evaluation.calibration import (
    PlattScaling,
    IsotonicCalibration,
    TemperatureScaling,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestPlattScaling:
    """Tests for PlattScaling calibration."""

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        probs = np.random.uniform(0.3, 0.7, 100)
        targets = (probs > 0.5).astype(int)

        calibrator = PlattScaling()
        result = calibrator.fit(probs, targets)

        assert result is calibrator

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        probs = np.random.uniform(0.1, 0.9, 50)
        targets = np.random.randint(0, 2, 50)

        calibrator = PlattScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert calibrated.shape == probs.shape

    def test_output_in_valid_range(self):
        """Test that calibrated probabilities are in [0, 1]."""
        probs = np.random.uniform(0.1, 0.9, 100)
        targets = np.random.randint(0, 2, 100)

        calibrator = PlattScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_accepts_torch_tensor(self):
        """Test that PlattScaling accepts torch tensors."""
        probs = torch.rand(100) * 0.6 + 0.2  # Range [0.2, 0.8]
        targets = torch.randint(0, 2, (100,))

        calibrator = PlattScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert isinstance(calibrated, np.ndarray)
        assert calibrated.shape == (100,)


class TestIsotonicCalibration:
    """Tests for IsotonicCalibration."""

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        probs = np.random.uniform(0.1, 0.9, 100)
        targets = np.random.randint(0, 2, 100)

        calibrator = IsotonicCalibration()
        result = calibrator.fit(probs, targets)

        assert result is calibrator

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        probs = np.random.uniform(0.1, 0.9, 50)
        targets = np.random.randint(0, 2, 50)

        calibrator = IsotonicCalibration().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert calibrated.shape == probs.shape

    def test_output_in_valid_range(self):
        """Test that calibrated probabilities are in [0, 1]."""
        probs = np.random.uniform(0.1, 0.9, 100)
        targets = np.random.randint(0, 2, 100)

        calibrator = IsotonicCalibration().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_monotonic_output(self):
        """Test that isotonic regression produces monotonic output."""
        # Create monotonically increasing probabilities
        probs = np.linspace(0.1, 0.9, 100)
        # Create targets correlated with probs
        targets = (probs + np.random.normal(0, 0.1, 100) > 0.5).astype(int)

        calibrator = IsotonicCalibration().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        # Calibrated values should be monotonically non-decreasing
        assert np.all(np.diff(calibrated) >= -1e-10)

    def test_accepts_torch_tensor(self):
        """Test that IsotonicCalibration accepts torch tensors."""
        probs = torch.rand(100) * 0.6 + 0.2  # Range [0.2, 0.8]
        targets = torch.randint(0, 2, (100,))

        calibrator = IsotonicCalibration().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert isinstance(calibrated, np.ndarray)
        assert calibrated.shape == (100,)


class TestTemperatureScaling:
    """Tests for TemperatureScaling calibration."""

    def test_init_default_temperature_one(self):
        """Test that default temperature is 1.0."""
        calibrator = TemperatureScaling()

        assert hasattr(calibrator, 'temperature')
        assert calibrator.temperature.item() == pytest.approx(1.0)

    def test_temperature_one_is_identity(self):
        """Test that T=1.0 produces identity transformation."""
        probs = np.array([0.2, 0.5, 0.8])
        calibrator = TemperatureScaling()  # T=1.0

        calibrated = calibrator.predict(probs)

        np.testing.assert_array_almost_equal(calibrated, probs, decimal=5)

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        probs = np.random.uniform(0.3, 0.7, 100)
        targets = np.random.randint(0, 2, 100)

        calibrator = TemperatureScaling()
        result = calibrator.fit(probs, targets)

        assert result is calibrator

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        probs = np.random.uniform(0.3, 0.7, 50)
        targets = np.random.randint(0, 2, 50)

        calibrator = TemperatureScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert calibrated.shape == probs.shape

    def test_gradient_flows(self):
        """Test that temperature parameter has gradients."""
        calibrator = TemperatureScaling()

        assert isinstance(calibrator.temperature, torch.nn.Parameter)
        assert calibrator.temperature.requires_grad

    def test_predict_returns_numpy(self):
        """Test that predict always returns numpy array."""
        probs = np.random.uniform(0.3, 0.7, 50)
        targets = np.random.randint(0, 2, 50)

        calibrator = TemperatureScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert isinstance(calibrated, np.ndarray)

    def test_accepts_torch_tensor(self):
        """Test that TemperatureScaling accepts torch tensors."""
        probs = torch.rand(100) * 0.4 + 0.3  # Range [0.3, 0.7]
        targets = torch.randint(0, 2, (100,))

        calibrator = TemperatureScaling().fit(probs, targets)
        calibrated = calibrator.predict(probs)

        assert isinstance(calibrated, np.ndarray)
        assert calibrated.shape == (100,)


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error function."""

    def test_perfect_calibration_near_zero(self):
        """Test that perfectly calibrated predictions have ECE near 0."""
        # Create perfectly calibrated predictions
        n_samples = 1000
        probs = np.random.uniform(0.0, 1.0, n_samples)
        # For perfect calibration, targets should match probability
        targets = (np.random.uniform(0, 1, n_samples) < probs).astype(int)

        ece = expected_calibration_error(probs, targets, n_bins=10)

        # With enough samples, ECE should be reasonably low
        assert ece < 0.15  # Allow some variance

    def test_overconfident_predictions_high_ece(self):
        """Test that overconfident predictions have higher ECE."""
        n_samples = 500
        # Overconfident: all predictions near 0.9 but only 50% are actually positive
        probs = np.full(n_samples, 0.9)
        targets = np.random.randint(0, 2, n_samples)

        ece = expected_calibration_error(probs, targets, n_bins=10)

        # Should have significant calibration error (~0.4)
        assert ece > 0.3

    def test_returns_float(self):
        """Test that ECE returns a Python float."""
        probs = np.random.uniform(0.1, 0.9, 100)
        targets = np.random.randint(0, 2, 100)

        ece = expected_calibration_error(probs, targets)

        assert isinstance(ece, float)

    def test_ece_in_valid_range(self):
        """Test that ECE is in [0, 1]."""
        probs = np.random.uniform(0.0, 1.0, 200)
        targets = np.random.randint(0, 2, 200)

        ece = expected_calibration_error(probs, targets)

        assert 0.0 <= ece <= 1.0

    def test_accepts_torch_tensor(self):
        """Test that ECE accepts torch tensors."""
        probs = torch.rand(100)
        targets = torch.randint(0, 2, (100,))

        ece = expected_calibration_error(probs, targets)

        assert isinstance(ece, float)


class TestReliabilityDiagramData:
    """Tests for reliability_diagram_data function."""

    def test_returns_expected_keys(self):
        """Test that function returns dict with expected keys."""
        probs = np.random.uniform(0.0, 1.0, 100)
        targets = np.random.randint(0, 2, 100)

        data = reliability_diagram_data(probs, targets)

        expected_keys = {'bin_edges', 'bin_accuracies', 'bin_confidences', 'bin_counts'}
        assert set(data.keys()) == expected_keys

    def test_bin_counts_sum_to_samples(self):
        """Test that bin counts sum to total number of samples."""
        n_samples = 150
        probs = np.random.uniform(0.0, 1.0, n_samples)
        targets = np.random.randint(0, 2, n_samples)

        data = reliability_diagram_data(probs, targets, n_bins=10)

        assert sum(data['bin_counts']) == n_samples

    def test_shapes_consistent(self):
        """Test that output arrays have consistent shapes."""
        probs = np.random.uniform(0.0, 1.0, 100)
        targets = np.random.randint(0, 2, 100)
        n_bins = 10

        data = reliability_diagram_data(probs, targets, n_bins=n_bins)

        assert len(data['bin_edges']) == n_bins + 1
        assert len(data['bin_accuracies']) == n_bins
        assert len(data['bin_confidences']) == n_bins
        assert len(data['bin_counts']) == n_bins

    def test_perfect_calibration_diagonal(self):
        """Test that perfectly calibrated data shows diagonal pattern."""
        n_samples = 2000
        # Create probabilities spread across bins
        probs = np.random.uniform(0.0, 1.0, n_samples)
        # Perfect calibration: outcome matches probability
        targets = (np.random.uniform(0, 1, n_samples) < probs).astype(int)

        data = reliability_diagram_data(probs, targets, n_bins=10)

        # For bins with sufficient samples, accuracy should be close to confidence
        for acc, conf, count in zip(data['bin_accuracies'],
                                     data['bin_confidences'],
                                     data['bin_counts']):
            if count >= 50:  # Only check bins with enough samples
                # Allow some variance due to randomness
                assert abs(acc - conf) < 0.2

    def test_accepts_torch_tensor(self):
        """Test that function accepts torch tensors."""
        probs = torch.rand(100)
        targets = torch.randint(0, 2, (100,))

        data = reliability_diagram_data(probs, targets)

        assert isinstance(data, dict)
        assert 'bin_edges' in data
