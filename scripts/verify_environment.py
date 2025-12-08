"""Development environment verification utility."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass(frozen=True)
class CheckResult:
    """Represents a single verification check outcome."""

    name: str
    passed: bool
    message: str


def _success(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, passed=True, message=message)


def _failure(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, passed=False, message=message)


def check_python_version(expected_major: int = 3, expected_minor: int = 12) -> CheckResult:
    """Ensure the interpreter is Python 3.12.x."""
    major, minor = sys.version_info[:2]
    if (major, minor) == (expected_major, expected_minor):
        return _success("Python Version", f"Python {major}.{minor} detected")
    return _failure("Python Version", f"Python 3.12 required, found {major}.{minor}")


def check_pytorch_mps() -> CheckResult:
    """Confirm PyTorch is installed with MPS support."""
    name = "PyTorch MPS"
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        return _failure(name, f"torch not installed: {exc}")

    if not torch.backends.mps.is_available():
        return _failure(name, "MPS backend is unavailable")
    if not torch.backends.mps.is_built():
        return _failure(name, "PyTorch built without MPS support")

    return _success(name, f"torch {torch.__version__} with MPS OK")


def check_optuna() -> CheckResult:
    """Verify Optuna can be imported."""
    name = "Optuna"
    try:
        optuna = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        return _failure(name, f"optuna not installed: {exc}")
    return _success(name, f"optuna {optuna.__version__} OK")


def check_data_libraries() -> CheckResult:
    """Ensure core data libraries import correctly."""
    name = "Data Libraries"
    modules = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yfinance", "yfinance"),
        ("fredapi", "fredapi"),
    ]
    missing: list[str] = []
    versions: list[str] = []
    for label, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            versions.append(f"{label} {version}")
        except ModuleNotFoundError:
            missing.append(label)
    if missing:
        return _failure(name, f"Missing modules: {', '.join(missing)}")
    return _success(name, ", ".join(versions))


def check_indicator_libraries() -> CheckResult:
    """Ensure TA indicator libraries load."""
    name = "Indicator Libraries"
    modules = [("pandas_ta", "pandas_ta"), ("TA-Lib", "talib")]
    missing: list[str] = []
    versions: list[str] = []
    for label, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            versions.append(f"{label} {version}")
        except ModuleNotFoundError:
            missing.append(label)
    if missing:
        return _failure(name, f"Missing modules: {', '.join(missing)}")
    return _success(name, ", ".join(versions))


def check_tracking_libraries() -> CheckResult:
    """Ensure experiment tracking packages are available."""
    name = "Tracking Libraries"
    modules = [("wandb", "wandb"), ("mlflow", "mlflow")]
    missing: list[str] = []
    versions: list[str] = []
    for label, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            versions.append(f"{label} {version}")
        except ModuleNotFoundError:
            missing.append(label)
    if missing:
        return _failure(name, f"Missing modules: {', '.join(missing)}")
    return _success(name, ", ".join(versions))


CHECKS: List[Callable[[], CheckResult]] = [
    check_python_version,
    check_pytorch_mps,
    check_optuna,
    check_data_libraries,
    check_indicator_libraries,
    check_tracking_libraries,
]


def run_checks(checks: Iterable[Callable[[], CheckResult]] | None = None) -> list[CheckResult]:
    """Execute all verification checks."""
    selected = list(checks) if checks is not None else CHECKS
    return [check() for check in selected]


def summarize_results(results: Iterable[CheckResult]) -> None:
    """Print a human-readable summary."""
    for result in results:
        status = "✅" if result.passed else "❌"
        print(f"{status} {result.name}: {result.message}")


def main() -> int:
    """Entry point for CLI usage."""
    results = run_checks()
    summarize_results(results)
    if all(result.passed for result in results):
        print("Environment verification PASSED")
        return 0
    print("Environment verification FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

