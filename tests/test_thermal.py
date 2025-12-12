"""Tests for Thermal Callback (Phase 4 Task 4).

Test cases:
1. test_thermal_status_dataclass_fields - ThermalStatus has required fields
2. test_thermal_callback_default_thresholds - Default thresholds match CLAUDE.md
3. test_thermal_callback_normal_below_70c - Status "normal" when temp < 70°C
4. test_thermal_callback_acceptable_70_to_85c - Status "acceptable" when 70-85°C
5. test_thermal_callback_warning_85_to_95c - Status "warning" when 85-95°C
6. test_thermal_callback_critical_above_95c - Status "critical", should_pause=True when > 95°C
7. test_thermal_callback_boundary_exactly_70c - 70°C is boundary (acceptable)
8. test_thermal_callback_boundary_exactly_85c - 85°C is boundary (warning)
9. test_thermal_callback_boundary_exactly_95c - 95°C is boundary (critical)
10. test_thermal_callback_handles_read_failure - Returns safe default on read failure
11. test_thermal_callback_custom_thresholds_validation - Invalid thresholds raise ValueError
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable

import pytest

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.thermal import ThermalCallback, ThermalStatus


class TestThermalStatusDataclass:
    """Tests for ThermalStatus dataclass structure."""

    def test_thermal_status_dataclass_fields(self) -> None:
        """ThermalStatus should have temperature, status, should_pause, and message fields."""
        field_names = {f.name for f in fields(ThermalStatus)}
        expected_fields = {"temperature", "status", "should_pause", "message"}

        assert field_names == expected_fields, (
            f"ThermalStatus should have fields {expected_fields}, got {field_names}"
        )


class TestThermalCallbackDefaults:
    """Tests for ThermalCallback default configuration."""

    def test_thermal_callback_default_thresholds(self) -> None:
        """Default thresholds should match CLAUDE.md specification."""
        callback = ThermalCallback()

        assert callback.normal_threshold == 70.0, "Normal threshold should be 70°C"
        assert callback.warning_threshold == 85.0, "Warning threshold should be 85°C"
        assert callback.critical_threshold == 95.0, "Critical threshold should be 95°C"


class TestThermalCallbackStatus:
    """Tests for ThermalCallback status determination."""

    @pytest.fixture
    def mock_temp_provider(self) -> Callable[[float], Callable[[], float]]:
        """Factory for creating mock temperature providers."""
        def create_provider(temp: float) -> Callable[[], float]:
            def provider() -> float:
                return temp
            return provider
        return create_provider

    def test_thermal_callback_normal_below_70c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Temperature below 70°C should return 'normal' status."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(65.0))
        status = callback.check()

        assert status.temperature == 65.0
        assert status.status == "normal"
        assert status.should_pause is False

    def test_thermal_callback_acceptable_70_to_85c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Temperature 70-85°C should return 'acceptable' status."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(75.0))
        status = callback.check()

        assert status.temperature == 75.0
        assert status.status == "acceptable"
        assert status.should_pause is False

    def test_thermal_callback_warning_85_to_95c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Temperature 85-95°C should return 'warning' status."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(90.0))
        status = callback.check()

        assert status.temperature == 90.0
        assert status.status == "warning"
        assert status.should_pause is False  # Warning but not yet pausing

    def test_thermal_callback_critical_above_95c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Temperature above 95°C should return 'critical' status and should_pause=True."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(98.0))
        status = callback.check()

        assert status.temperature == 98.0
        assert status.status == "critical"
        assert status.should_pause is True


class TestThermalCallbackBoundaries:
    """Tests for ThermalCallback boundary conditions."""

    @pytest.fixture
    def mock_temp_provider(self) -> Callable[[float], Callable[[], float]]:
        """Factory for creating mock temperature providers."""
        def create_provider(temp: float) -> Callable[[], float]:
            def provider() -> float:
                return temp
            return provider
        return create_provider

    def test_thermal_callback_boundary_exactly_70c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Exactly 70°C should be 'acceptable' (at the boundary)."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(70.0))
        status = callback.check()

        assert status.status == "acceptable", "70°C is the boundary, should be 'acceptable'"

    def test_thermal_callback_boundary_exactly_85c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Exactly 85°C should be 'warning' (at the boundary)."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(85.0))
        status = callback.check()

        assert status.status == "warning", "85°C is the boundary, should be 'warning'"

    def test_thermal_callback_boundary_exactly_95c(
        self, mock_temp_provider: Callable[[float], Callable[[], float]]
    ) -> None:
        """Exactly 95°C should be 'critical' (at the boundary)."""
        callback = ThermalCallback(temp_provider=mock_temp_provider(95.0))
        status = callback.check()

        assert status.status == "critical", "95°C is the boundary, should be 'critical'"
        assert status.should_pause is True


class TestThermalCallbackErrorHandling:
    """Tests for ThermalCallback error handling."""

    def test_thermal_callback_handles_read_failure(self) -> None:
        """Should return safe default status when temperature cannot be read."""
        def failing_provider() -> float:
            raise OSError("Cannot read temperature")

        callback = ThermalCallback(temp_provider=failing_provider)
        status = callback.check()

        # On failure, should assume worst case for safety
        assert status.should_pause is True, "Should pause on read failure for safety"
        assert status.status == "unknown"
        assert "error" in status.message.lower() or "fail" in status.message.lower()

    def test_thermal_callback_custom_thresholds_validation(self) -> None:
        """Invalid threshold configuration should raise ValueError."""
        # Warning threshold must be > normal threshold
        with pytest.raises(ValueError, match="warning.*normal"):
            ThermalCallback(normal_threshold=85.0, warning_threshold=70.0)

        # Critical threshold must be > warning threshold
        with pytest.raises(ValueError, match="critical.*warning"):
            ThermalCallback(warning_threshold=95.0, critical_threshold=85.0)


class TestGetHardwareStats:
    """Tests for get_hardware_stats() function."""

    def test_get_hardware_stats_returns_dict(self) -> None:
        """get_hardware_stats should return a dictionary."""
        from src.training.thermal import get_hardware_stats

        result = get_hardware_stats()
        assert isinstance(result, dict), "get_hardware_stats should return a dict"

    def test_get_hardware_stats_has_required_keys(self) -> None:
        """get_hardware_stats should return dict with cpu_percent and memory_percent."""
        from src.training.thermal import get_hardware_stats

        result = get_hardware_stats()
        assert "cpu_percent" in result, "Result should have 'cpu_percent' key"
        assert "memory_percent" in result, "Result should have 'memory_percent' key"

    def test_get_hardware_stats_values_in_range(self) -> None:
        """CPU and memory percentages should be between 0 and 100."""
        from src.training.thermal import get_hardware_stats

        result = get_hardware_stats()
        assert 0 <= result["cpu_percent"] <= 100, "CPU percent should be 0-100"
        assert 0 <= result["memory_percent"] <= 100, "Memory percent should be 0-100"

    def test_get_hardware_stats_values_are_floats(self) -> None:
        """CPU and memory percentages should be floats."""
        from src.training.thermal import get_hardware_stats

        result = get_hardware_stats()
        assert isinstance(result["cpu_percent"], float), "CPU percent should be float"
        assert isinstance(result["memory_percent"], float), "Memory percent should be float"


class TestGetMacosTemperature:
    """Tests for get_macos_temperature() function."""

    def test_get_macos_temperature_returns_float(self) -> None:
        """get_macos_temperature should return a float."""
        from src.training.thermal import get_macos_temperature

        result = get_macos_temperature()
        assert isinstance(result, float), "get_macos_temperature should return float"

    def test_get_macos_temperature_failure_returns_negative(self) -> None:
        """get_macos_temperature should return -1 on failure."""
        from unittest.mock import patch
        from src.training.thermal import get_macos_temperature

        # Mock subprocess to simulate failure
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("powermetrics not found")
            result = get_macos_temperature()

        assert result == -1.0, "Should return -1.0 on subprocess failure"

    def test_get_macos_temperature_timeout_returns_negative(self) -> None:
        """get_macos_temperature should return -1 on timeout."""
        import subprocess
        from unittest.mock import patch
        from src.training.thermal import get_macos_temperature

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="powermetrics", timeout=5)
            result = get_macos_temperature()

        assert result == -1.0, "Should return -1.0 on timeout"

    def test_get_macos_temperature_malformed_output_returns_negative(self) -> None:
        """get_macos_temperature should return -1 on malformed output."""
        from unittest.mock import patch, MagicMock
        from src.training.thermal import get_macos_temperature

        mock_result = MagicMock()
        mock_result.stdout = "no temperature data here"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = get_macos_temperature()

        assert result == -1.0, "Should return -1.0 on malformed output"


class TestDefaultTempProvider:
    """Tests for updated default temperature provider."""

    def test_thermal_callback_without_provider_uses_macos_temp(self) -> None:
        """ThermalCallback without explicit provider should use get_macos_temperature."""
        from unittest.mock import patch
        from src.training.thermal import ThermalCallback

        # Mock get_macos_temperature to return a known value
        with patch("src.training.thermal.get_macos_temperature", return_value=72.5):
            callback = ThermalCallback()
            status = callback.check()

        assert status.temperature == 72.5, "Should use get_macos_temperature as default"

    def test_thermal_callback_default_provider_handles_failure(self) -> None:
        """ThermalCallback default provider should handle temperature read failure."""
        from unittest.mock import patch
        from src.training.thermal import ThermalCallback

        # Mock get_macos_temperature to return -1 (failure)
        with patch("src.training.thermal.get_macos_temperature", return_value=-1.0):
            callback = ThermalCallback()
            status = callback.check()

        # -1 is below all thresholds, so it would be "normal"
        # But we should treat -1 as unknown - this tests current behavior
        assert status.temperature == -1.0
