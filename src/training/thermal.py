"""Thermal monitoring callback for M4 MacBook Pro training.

Monitors CPU/GPU temperature during training and pauses if thresholds exceeded.

Thresholds (from CLAUDE.md):
- <70°C: Normal operation
- 70-85°C: Acceptable, monitor
- 85-95°C: Warning, consider pause
- >95°C: CRITICAL STOP
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable

import psutil


@dataclass
class ThermalStatus:
    """Status returned by thermal check.

    Attributes:
        temperature: Current temperature in Celsius (or -1 if read failed)
        status: One of "normal", "acceptable", "warning", "critical", "unknown"
        should_pause: Whether training should pause
        message: Human-readable status message
    """

    temperature: float
    status: str
    should_pause: bool
    message: str


def get_hardware_stats() -> dict[str, float]:
    """Get current CPU and memory utilization.

    Returns:
        Dictionary with 'cpu_percent' and 'memory_percent' keys.
        Values are floats between 0 and 100.
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    return {
        "cpu_percent": float(cpu_percent),
        "memory_percent": float(memory_percent),
    }


def get_macos_temperature() -> float:
    """Get CPU temperature on macOS using powermetrics.

    Requires sudo access. Returns -1.0 if temperature cannot be read.

    Note: On M4 MacBook Pro, this requires:
        sudo powermetrics --samplers thermal -n 1

    Returns:
        Temperature in Celsius, or -1.0 on any failure.
    """
    try:
        # Run powermetrics with thermal sampler, single sample
        # Timeout after 5 seconds to avoid hanging
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "thermal", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Parse output for temperature
        # Look for patterns like "CPU die temperature: 45.67 C"
        # or "GPU die temperature: 45.67 C"
        output = result.stdout
        temp_pattern = r"(?:CPU|GPU|die)\s+(?:die\s+)?temperature:\s*([\d.]+)\s*C"
        matches = re.findall(temp_pattern, output, re.IGNORECASE)

        if matches:
            # Return the highest temperature found (most relevant for throttling)
            temps = [float(t) for t in matches]
            return max(temps)

        # No temperature found in output
        return -1.0

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return -1.0


def _default_temp_provider() -> float:
    """Default temperature provider using get_macos_temperature.

    Returns temperature from powermetrics on macOS.
    Returns -1.0 if temperature cannot be read.
    """
    return get_macos_temperature()


class ThermalCallback:
    """Thermal monitoring callback for training loops.

    Monitors temperature and provides status for training loop control.
    Temperature reading is injectable for testing.

    Example:
        >>> def get_temp():
        ...     return 65.0  # Read from sensor
        >>> callback = ThermalCallback(temp_provider=get_temp)
        >>> status = callback.check()
        >>> if status.should_pause:
        ...     # Pause training
        ...     pass
    """

    def __init__(
        self,
        temp_provider: Callable[[], float] | None = None,
        normal_threshold: float = 70.0,
        warning_threshold: float = 85.0,
        critical_threshold: float = 95.0,
    ) -> None:
        """Initialize thermal callback.

        Args:
            temp_provider: Callable that returns current temperature in Celsius.
                          If None, uses default provider (requires sudo).
            normal_threshold: Temperature below this is "normal" (default: 70°C)
            warning_threshold: Temperature at/above this is "warning" (default: 85°C)
            critical_threshold: Temperature at/above this triggers pause (default: 95°C)

        Raises:
            ValueError: If thresholds are not in ascending order.
        """
        # Validate threshold ordering
        if warning_threshold <= normal_threshold:
            raise ValueError(
                f"warning_threshold ({warning_threshold}) must be greater than "
                f"normal_threshold ({normal_threshold})"
            )
        if critical_threshold <= warning_threshold:
            raise ValueError(
                f"critical_threshold ({critical_threshold}) must be greater than "
                f"warning_threshold ({warning_threshold})"
            )

        self.temp_provider = temp_provider or _default_temp_provider
        self.normal_threshold = normal_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check(self) -> ThermalStatus:
        """Check current temperature and return status.

        Returns:
            ThermalStatus with current temperature and recommended action.
            On read failure, returns status="unknown" with should_pause=True
            for safety.
        """
        try:
            temperature = self.temp_provider()
        except Exception as e:
            return ThermalStatus(
                temperature=-1.0,
                status="unknown",
                should_pause=True,
                message=f"Failed to read temperature: {e}",
            )

        # Determine status based on thresholds
        if temperature >= self.critical_threshold:
            return ThermalStatus(
                temperature=temperature,
                status="critical",
                should_pause=True,
                message=f"CRITICAL: {temperature:.1f}°C >= {self.critical_threshold}°C - STOP IMMEDIATELY",
            )
        elif temperature >= self.warning_threshold:
            return ThermalStatus(
                temperature=temperature,
                status="warning",
                should_pause=False,
                message=f"WARNING: {temperature:.1f}°C >= {self.warning_threshold}°C - consider pausing",
            )
        elif temperature >= self.normal_threshold:
            return ThermalStatus(
                temperature=temperature,
                status="acceptable",
                should_pause=False,
                message=f"Acceptable: {temperature:.1f}°C - monitoring",
            )
        else:
            return ThermalStatus(
                temperature=temperature,
                status="normal",
                should_pause=False,
                message=f"Normal: {temperature:.1f}°C",
            )
