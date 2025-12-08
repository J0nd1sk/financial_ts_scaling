"""
Tests for the environment verification utility.

All tests are written before the implementation exists to enforce TDD.
"""

from importlib import reload
from pathlib import Path
from types import ModuleType
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def import_verify_module() -> ModuleType:
    """Reload the module so monkeypatching CHECKS is reliable."""
    import scripts.verify_environment as verify_environment

    return reload(verify_environment)


def test_check_python_version_requires_312(monkeypatch: pytest.MonkeyPatch) -> None:
    """check_python_version should fail when interpreter is not Python 3.12."""
    module = import_verify_module()

    monkeypatch.setattr(module, "sys", type("Sys", (), {"version_info": (3, 11, 9)}))
    result = module.check_python_version()

    assert result.passed is False
    assert "Python 3.12" in result.message


def test_run_checks_collects_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_checks should aggregate failures and successes from all checks."""
    module = import_verify_module()

    success = module.CheckResult(name="ok", passed=True, message="fine")
    failure = module.CheckResult(name="bad", passed=False, message="broken")

    def fake_success() -> module.CheckResult:
        return success

    def fake_failure() -> module.CheckResult:
        return failure

    monkeypatch.setattr(module, "CHECKS", [fake_success, fake_failure])

    results = module.run_checks()

    assert success in results
    assert failure in results
    assert any(not r.passed for r in results), "Failure should be reported"

