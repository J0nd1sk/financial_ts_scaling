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
