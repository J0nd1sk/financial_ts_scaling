"""Tests for scaling curve analysis module.

Uses synthetic data to verify power law fitting and visualization.
No actual HPO results required - all tests use fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from src.analysis.scaling_curves import (
    fit_power_law,
    plot_scaling_curve,
    load_experiment_results,
    generate_scaling_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def perfect_power_law_data():
    """Generate perfect power law data: error = 1.5 * params^(-0.1)."""
    params = np.array([2e6, 2e7, 2e8, 2e9])
    alpha = 0.1
    a = 1.5
    errors = a * params ** (-alpha)
    return params, errors, alpha, a


@pytest.fixture
def noisy_power_law_data():
    """Generate noisy power law data with ~5% noise."""
    np.random.seed(42)
    params = np.array([2e6, 2e7, 2e8, 2e9])
    alpha = 0.15
    a = 2.0
    errors = a * params ** (-alpha)
    noise = 1 + 0.05 * np.random.randn(len(errors))
    return params, errors * noise, alpha, a


@pytest.fixture
def results_dataframe():
    """Create synthetic results DataFrame."""
    return pd.DataFrame({
        "budget": ["2M", "20M", "200M", "2B"],
        "params": [2_000_000, 20_000_000, 200_000_000, 2_000_000_000],
        "val_loss": [0.52, 0.48, 0.44, 0.41],
    })


@pytest.fixture
def hpo_results_dir(tmp_path):
    """Create temporary directory with synthetic HPO results."""
    experiment_dir = tmp_path / "outputs" / "hpo" / "test_experiment"
    experiment_dir.mkdir(parents=True)

    budgets_data = [
        ("2M", 2_000_000, 0.52),
        ("20M", 20_000_000, 0.48),
        ("200M", 200_000_000, 0.44),
        ("2B", 2_000_000_000, 0.41),
    ]

    for budget, params, val_loss in budgets_data:
        result = {
            "experiment": "test_experiment",
            "budget": budget,
            "best_params": {"learning_rate": 0.001},
            "best_value": val_loss,
            "params": params,
        }
        result_file = experiment_dir / f"{budget}_best.json"
        result_file.write_text(json.dumps(result))

    return tmp_path / "outputs" / "hpo"


# =============================================================================
# Tests for fit_power_law()
# =============================================================================

class TestFitPowerLawReturnType:
    """Test that fit_power_law returns correct type."""

    def test_fit_power_law_returns_tuple(self, perfect_power_law_data):
        """Test that fit_power_law returns a 3-tuple."""
        params, errors, _, _ = perfect_power_law_data
        result = fit_power_law(params, errors)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fit_power_law_returns_floats(self, perfect_power_law_data):
        """Test that all return values are floats."""
        params, errors, _, _ = perfect_power_law_data
        alpha, a, r_squared = fit_power_law(params, errors)

        assert isinstance(alpha, float)
        assert isinstance(a, float)
        assert isinstance(r_squared, float)


class TestFitPowerLawPerfectData:
    """Test fit_power_law with perfect synthetic data."""

    def test_fit_power_law_recovers_alpha(self, perfect_power_law_data):
        """Test that known alpha is recovered within tolerance."""
        params, errors, expected_alpha, _ = perfect_power_law_data
        alpha, _, _ = fit_power_law(params, errors)

        assert abs(alpha - expected_alpha) < 0.01, (
            f"Expected alpha={expected_alpha}, got {alpha}"
        )

    def test_fit_power_law_recovers_coefficient(self, perfect_power_law_data):
        """Test that coefficient a is recovered within tolerance."""
        params, errors, _, expected_a = perfect_power_law_data
        _, a, _ = fit_power_law(params, errors)

        assert abs(a - expected_a) / expected_a < 0.01, (
            f"Expected a={expected_a}, got {a}"
        )

    def test_fit_power_law_perfect_r_squared(self, perfect_power_law_data):
        """Test that R² ≈ 1.0 for perfect data."""
        params, errors, _, _ = perfect_power_law_data
        _, _, r_squared = fit_power_law(params, errors)

        assert r_squared > 0.99, f"Expected R²≈1.0, got {r_squared}"


class TestFitPowerLawNoisyData:
    """Test fit_power_law with noisy data."""

    def test_fit_power_law_noisy_reasonable_r_squared(self, noisy_power_law_data):
        """Test that R² is in reasonable range for noisy data."""
        params, errors, _, _ = noisy_power_law_data
        _, _, r_squared = fit_power_law(params, errors)

        assert 0 < r_squared < 1, f"R² should be in (0,1), got {r_squared}"
        assert r_squared > 0.9, f"R² should be high for 5% noise, got {r_squared}"

    def test_fit_power_law_noisy_alpha_reasonable(self, noisy_power_law_data):
        """Test that alpha is close to true value despite noise."""
        params, errors, expected_alpha, _ = noisy_power_law_data
        alpha, _, _ = fit_power_law(params, errors)

        # Allow 20% error for noisy data
        assert abs(alpha - expected_alpha) / expected_alpha < 0.2, (
            f"Expected alpha≈{expected_alpha}, got {alpha}"
        )


class TestFitPowerLawValidation:
    """Test input validation for fit_power_law."""

    def test_fit_power_law_rejects_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        params = np.array([1e6, 1e7, 1e8])
        errors = np.array([0.5, 0.4])  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            fit_power_law(params, errors)

    def test_fit_power_law_rejects_single_point(self):
        """Test that single data point raises ValueError."""
        params = np.array([1e6])
        errors = np.array([0.5])

        with pytest.raises(ValueError, match="at least 2"):
            fit_power_law(params, errors)

    def test_fit_power_law_rejects_negative_params(self):
        """Test that negative params raise ValueError."""
        params = np.array([-1e6, 1e7, 1e8])
        errors = np.array([0.5, 0.4, 0.3])

        with pytest.raises(ValueError, match="positive"):
            fit_power_law(params, errors)

    def test_fit_power_law_rejects_zero_errors(self):
        """Test that zero errors raise ValueError."""
        params = np.array([1e6, 1e7, 1e8])
        errors = np.array([0.5, 0.0, 0.3])

        with pytest.raises(ValueError, match="positive"):
            fit_power_law(params, errors)


# =============================================================================
# Tests for plot_scaling_curve()
# =============================================================================

class TestPlotScalingCurveReturnType:
    """Test that plot_scaling_curve returns correct type."""

    def test_plot_scaling_curve_returns_figure(self, results_dataframe):
        """Test that function returns matplotlib Figure."""
        fig = plot_scaling_curve(results_dataframe)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scaling_curve_has_axes(self, results_dataframe):
        """Test that returned figure has axes."""
        fig = plot_scaling_curve(results_dataframe)

        assert len(fig.axes) > 0
        plt.close(fig)


class TestPlotScalingCurveSavesFile:
    """Test file saving functionality."""

    def test_plot_scaling_curve_saves_png(self, results_dataframe, tmp_path):
        """Test that PNG is saved when path provided."""
        output_path = tmp_path / "test_scaling.png"

        fig = plot_scaling_curve(results_dataframe, output_path=str(output_path))
        plt.close(fig)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_scaling_curve_creates_parent_dirs(self, results_dataframe, tmp_path):
        """Test that parent directories are created if needed."""
        output_path = tmp_path / "nested" / "dirs" / "test_scaling.png"

        fig = plot_scaling_curve(results_dataframe, output_path=str(output_path))
        plt.close(fig)

        assert output_path.exists()


class TestPlotScalingCurveContent:
    """Test plot content and formatting."""

    def test_plot_scaling_curve_uses_log_scale(self, results_dataframe):
        """Test that both axes use log scale."""
        fig = plot_scaling_curve(results_dataframe)
        ax = fig.axes[0]

        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_plot_scaling_curve_custom_title(self, results_dataframe):
        """Test that custom title is applied."""
        custom_title = "My Custom Title"
        fig = plot_scaling_curve(results_dataframe, title=custom_title)
        ax = fig.axes[0]

        assert ax.get_title() == custom_title
        plt.close(fig)


# =============================================================================
# Tests for load_experiment_results()
# =============================================================================

class TestLoadExperimentResults:
    """Test loading HPO results from JSON files."""

    def test_load_experiment_results_returns_dataframe(self, hpo_results_dir):
        """Test that function returns DataFrame."""
        df = load_experiment_results(
            "test_experiment",
            hpo_dir=str(hpo_results_dir),
        )

        assert isinstance(df, pd.DataFrame)

    def test_load_experiment_results_has_expected_columns(self, hpo_results_dir):
        """Test that DataFrame has required columns."""
        df = load_experiment_results(
            "test_experiment",
            hpo_dir=str(hpo_results_dir),
        )

        assert "budget" in df.columns
        assert "params" in df.columns
        assert "best_value" in df.columns

    def test_load_experiment_results_correct_row_count(self, hpo_results_dir):
        """Test that all budgets are loaded."""
        df = load_experiment_results(
            "test_experiment",
            hpo_dir=str(hpo_results_dir),
        )

        assert len(df) == 4

    def test_load_experiment_results_filters_budgets(self, hpo_results_dir):
        """Test that budget filter works."""
        df = load_experiment_results(
            "test_experiment",
            budgets=["2M", "20M"],
            hpo_dir=str(hpo_results_dir),
        )

        assert len(df) == 2
        assert set(df["budget"]) == {"2M", "20M"}

    def test_load_experiment_results_missing_experiment_raises(self, hpo_results_dir):
        """Test that missing experiment raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_experiment_results(
                "nonexistent_experiment",
                hpo_dir=str(hpo_results_dir),
            )


# =============================================================================
# Tests for generate_scaling_report()
# =============================================================================

class TestGenerateScalingReport:
    """Test scaling report generation."""

    def test_generate_scaling_report_returns_dict(self, hpo_results_dir, tmp_path):
        """Test that function returns dictionary."""
        output_dir = tmp_path / "figures"

        result = generate_scaling_report(
            "test_experiment",
            output_dir=str(output_dir),
            hpo_dir=str(hpo_results_dir),
        )

        assert isinstance(result, dict)

    def test_generate_scaling_report_has_alpha(self, hpo_results_dir, tmp_path):
        """Test that result contains alpha."""
        output_dir = tmp_path / "figures"

        result = generate_scaling_report(
            "test_experiment",
            output_dir=str(output_dir),
            hpo_dir=str(hpo_results_dir),
        )

        assert "alpha" in result or "fit_results" in result

    def test_generate_scaling_report_creates_png(self, hpo_results_dir, tmp_path):
        """Test that PNG file is created."""
        output_dir = tmp_path / "figures"

        generate_scaling_report(
            "test_experiment",
            output_dir=str(output_dir),
            hpo_dir=str(hpo_results_dir),
        )

        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_generate_scaling_report_creates_json(self, hpo_results_dir, tmp_path):
        """Test that JSON file is created."""
        output_dir = tmp_path / "figures"

        generate_scaling_report(
            "test_experiment",
            output_dir=str(output_dir),
            hpo_dir=str(hpo_results_dir),
        )

        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) > 0
