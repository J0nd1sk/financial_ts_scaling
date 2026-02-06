"""Tests for hpo_neuralforecast.py CLI and search modes.

Tests for:
1. FocalLoss CLI argument parsing
2. Loss-only mode with fixed architecture
3. Extended search space with extreme values
4. Joint HPO mode

Note: We cannot directly import hpo_neuralforecast due to its internal dependencies
on experiments.architectures.common which requires neuralforecast. Instead, we
parse the script file to verify the constants are defined correctly.
"""

import re
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Path to the HPO script
HPO_SCRIPT_PATH = PROJECT_ROOT / "experiments" / "architectures" / "hpo_neuralforecast.py"


def read_hpo_script() -> str:
    """Read the HPO script content."""
    return HPO_SCRIPT_PATH.read_text()


def extract_dict_from_script(script_content: str, dict_name: str) -> dict | None:
    """Extract a dictionary definition from the script content.

    Args:
        script_content: The script source code
        dict_name: Name of the dictionary to extract

    Returns:
        The dictionary if found, None otherwise
    """
    # Pattern to match dict definition like: DICT_NAME = {...}
    # This is a simple extraction - may need adjustment for complex dicts
    pattern = rf'{dict_name}\s*=\s*\{{'
    match = re.search(pattern, script_content)
    if not match:
        return None

    # Find the matching closing brace
    start = match.end() - 1
    brace_count = 1
    pos = start + 1
    while pos < len(script_content) and brace_count > 0:
        if script_content[pos] == '{':
            brace_count += 1
        elif script_content[pos] == '}':
            brace_count -= 1
        pos += 1

    dict_str = script_content[start:pos]
    try:
        # Safe eval with restricted globals
        return eval(dict_str, {"__builtins__": {}})
    except Exception:
        return None


class TestFocalLossCLIArgs:
    """Tests for FocalLoss CLI argument parsing."""

    def test_focal_loss_type_in_choices(self):
        """Test --loss-type focal is defined in the script."""
        script = read_hpo_script()
        # Check that 'focal' is in the loss-type choices
        assert "'focal'" in script or '"focal"' in script
        assert "choices=" in script
        # Look for the specific pattern
        assert re.search(r'choices=\[.*["\']focal["\'].*\]', script)

    def test_focal_defaults_defined(self):
        """Test FOCAL_DEFAULTS dictionary is defined."""
        script = read_hpo_script()
        focal_defaults = extract_dict_from_script(script, "FOCAL_DEFAULTS")
        assert focal_defaults is not None, "FOCAL_DEFAULTS should be defined"

    def test_focal_gamma_default_value(self):
        """Test --focal-gamma default is 2.0."""
        script = read_hpo_script()
        focal_defaults = extract_dict_from_script(script, "FOCAL_DEFAULTS")
        assert focal_defaults is not None
        assert "gamma" in focal_defaults
        assert focal_defaults["gamma"] == 2.0

    def test_focal_alpha_default_value(self):
        """Test --focal-alpha default is between 0 and 1."""
        script = read_hpo_script()
        focal_defaults = extract_dict_from_script(script, "FOCAL_DEFAULTS")
        assert focal_defaults is not None
        assert "alpha" in focal_defaults
        assert 0.0 < focal_defaults["alpha"] < 1.0


class TestLossOnlyMode:
    """Tests for --loss-only mode with fixed architecture."""

    def test_loss_search_space_defined(self):
        """Test LOSS_SEARCH_SPACE is defined with gamma and alpha."""
        script = read_hpo_script()
        loss_space = extract_dict_from_script(script, "LOSS_SEARCH_SPACE")
        assert loss_space is not None, "LOSS_SEARCH_SPACE should be defined"
        assert "focal_gamma" in loss_space
        assert "focal_alpha" in loss_space

    def test_loss_search_gamma_values(self):
        """Test gamma search space contains expected values [0.0, 0.5, 1.0, 2.0]."""
        script = read_hpo_script()
        loss_space = extract_dict_from_script(script, "LOSS_SEARCH_SPACE")
        assert loss_space is not None
        expected_gammas = [0.0, 0.5, 1.0, 2.0]
        assert loss_space["focal_gamma"] == expected_gammas

    def test_loss_search_alpha_values(self):
        """Test alpha search space contains expected values."""
        script = read_hpo_script()
        loss_space = extract_dict_from_script(script, "LOSS_SEARCH_SPACE")
        assert loss_space is not None
        expected_alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert loss_space["focal_alpha"] == expected_alphas

    def test_loss_search_total_combinations(self):
        """Test total combinations is 4 gamma × 7 alpha = 28."""
        script = read_hpo_script()
        loss_space = extract_dict_from_script(script, "LOSS_SEARCH_SPACE")
        assert loss_space is not None
        n_gamma = len(loss_space["focal_gamma"])
        n_alpha = len(loss_space["focal_alpha"])
        assert n_gamma * n_alpha == 28, f"Expected 28 combinations, got {n_gamma * n_alpha}"

    def test_fixed_arch_for_loss_hpo_defined(self):
        """Test FIXED_ARCH_FOR_LOSS_HPO is defined."""
        script = read_hpo_script()
        fixed_arch = extract_dict_from_script(script, "FIXED_ARCH_FOR_LOSS_HPO")
        assert fixed_arch is not None, "FIXED_ARCH_FOR_LOSS_HPO should be defined"
        required_keys = {"hidden_size", "n_layers", "n_heads", "dropout", "learning_rate", "max_steps", "batch_size"}
        assert required_keys.issubset(set(fixed_arch.keys()))

    def test_fixed_arch_values_reasonable(self):
        """Test FIXED_ARCH_FOR_LOSS_HPO values are in reasonable ranges."""
        script = read_hpo_script()
        fixed_arch = extract_dict_from_script(script, "FIXED_ARCH_FOR_LOSS_HPO")
        assert fixed_arch is not None
        assert 32 <= fixed_arch["hidden_size"] <= 256
        assert 2 <= fixed_arch["n_layers"] <= 8
        assert fixed_arch["n_heads"] in [2, 4, 8, 16]
        assert 0.3 <= fixed_arch["dropout"] <= 0.6
        assert 1e-5 <= fixed_arch["learning_rate"] <= 1e-3


class TestExtendedSearchSpace:
    """Tests for SEARCH_SPACE_EXTENDED with extreme values."""

    def test_extended_search_space_defined(self):
        """Test SEARCH_SPACE_EXTENDED is defined."""
        script = read_hpo_script()
        extended = extract_dict_from_script(script, "SEARCH_SPACE_EXTENDED")
        assert extended is not None, "SEARCH_SPACE_EXTENDED should be defined"

    def test_extended_has_extreme_hidden_sizes(self):
        """Test extended search space includes 32 and 384 hidden sizes."""
        script = read_hpo_script()
        extended = extract_dict_from_script(script, "SEARCH_SPACE_EXTENDED")
        assert extended is not None
        hidden_sizes = extended["hidden_size"]
        assert 32 in hidden_sizes, "Extended should include small hidden_size=32"
        assert 384 in hidden_sizes, "Extended should include large hidden_size=384"

    def test_extended_has_extreme_layers(self):
        """Test extended search space includes up to 8 layers."""
        script = read_hpo_script()
        extended = extract_dict_from_script(script, "SEARCH_SPACE_EXTENDED")
        assert extended is not None
        n_layers = extended["n_layers"]
        assert 8 in n_layers, "Extended should include n_layers=8"
        assert 2 in n_layers, "Extended should include n_layers=2"

    def test_extended_has_high_dropout(self):
        """Test extended search space includes higher dropout values."""
        script = read_hpo_script()
        extended = extract_dict_from_script(script, "SEARCH_SPACE_EXTENDED")
        assert extended is not None
        dropout = extended["dropout"]
        assert 0.6 in dropout or 0.7 in dropout, "Extended should include high dropout"

    def test_extended_has_low_learning_rates(self):
        """Test extended search space includes lower learning rates."""
        script = read_hpo_script()
        extended = extract_dict_from_script(script, "SEARCH_SPACE_EXTENDED")
        assert extended is not None
        learning_rates = extended["learning_rate"]
        assert any(lr <= 2e-5 for lr in learning_rates), "Extended should include low learning rates"


class TestJointHPOMode:
    """Tests for joint architecture + loss HPO mode."""

    def test_joint_loss_search_defined(self):
        """Test JOINT_LOSS_SEARCH is defined for joint HPO mode."""
        script = read_hpo_script()
        joint_search = extract_dict_from_script(script, "JOINT_LOSS_SEARCH")
        assert joint_search is not None, "JOINT_LOSS_SEARCH should be defined"
        assert "focal_gamma" in joint_search
        assert "focal_alpha" in joint_search

    def test_joint_loss_search_has_expected_structure(self):
        """Test JOINT_LOSS_SEARCH has correct structure."""
        script = read_hpo_script()
        joint_search = extract_dict_from_script(script, "JOINT_LOSS_SEARCH")
        loss_search = extract_dict_from_script(script, "LOSS_SEARCH_SPACE")
        assert joint_search is not None
        assert loss_search is not None

        # Joint search should have equal or fewer options
        assert len(joint_search["focal_gamma"]) <= len(loss_search["focal_gamma"])
        assert len(joint_search["focal_alpha"]) <= len(loss_search["focal_alpha"])


class TestCLIArguments:
    """Tests for CLI argument definitions."""

    def test_loss_only_flag_defined(self):
        """Test --loss-only flag is defined."""
        script = read_hpo_script()
        assert "--loss-only" in script

    def test_joint_hpo_flag_defined(self):
        """Test --joint-hpo flag is defined."""
        script = read_hpo_script()
        assert "--joint-hpo" in script

    def test_extended_flag_defined(self):
        """Test --extended flag is defined."""
        script = read_hpo_script()
        assert "--extended" in script

    def test_focal_gamma_arg_defined(self):
        """Test --focal-gamma argument is defined."""
        script = read_hpo_script()
        assert "--focal-gamma" in script

    def test_focal_alpha_arg_defined(self):
        """Test --focal-alpha argument is defined."""
        script = read_hpo_script()
        assert "--focal-alpha" in script


class TestDataPathsInCommon:
    """Tests for data tier paths in common.py."""

    def test_data_path_a200_defined(self):
        """Test DATA_PATH_A200 is defined in common.py."""
        common_path = PROJECT_ROOT / "experiments" / "architectures" / "common.py"
        script = common_path.read_text()
        assert "DATA_PATH_A200" in script, "DATA_PATH_A200 should be defined in common.py"

    def test_data_paths_dict_defined(self):
        """Test DATA_PATHS dictionary is defined in common.py."""
        common_path = PROJECT_ROOT / "experiments" / "architectures" / "common.py"
        script = common_path.read_text()
        assert "DATA_PATHS" in script, "DATA_PATHS dict should be defined in common.py"

    def test_get_data_path_function_defined(self):
        """Test get_data_path function is defined in common.py."""
        common_path = PROJECT_ROOT / "experiments" / "architectures" / "common.py"
        script = common_path.read_text()
        assert "def get_data_path(" in script, "get_data_path function should be defined in common.py"


class TestGetSearchSpaceFunction:
    """Tests for get_search_space function definition."""

    def test_get_search_space_function_defined(self):
        """Test get_search_space function is defined."""
        script = read_hpo_script()
        assert "def get_search_space(" in script

    def test_get_search_space_has_extended_param(self):
        """Test get_search_space accepts extended parameter."""
        script = read_hpo_script()
        # Look for function signature with extended parameter
        assert re.search(r'def get_search_space\([^)]*extended[^)]*\)', script)

    def test_get_search_space_has_loss_only_param(self):
        """Test get_search_space accepts loss_only parameter."""
        script = read_hpo_script()
        # Look for function signature with loss_only parameter
        assert re.search(r'def get_search_space\([^)]*loss_only[^)]*\)', script)


class TestDataTierArgument:
    """Tests for --data-tier argument to select feature tier."""

    def test_data_tier_argument_defined(self):
        """Test --data-tier argument is in script."""
        script = read_hpo_script()
        assert "--data-tier" in script, "--data-tier argument should be defined"

    def test_data_tier_choices(self):
        """Test --data-tier has correct tier choices."""
        script = read_hpo_script()
        # Should have a20, a100, a200 as valid choices
        assert re.search(r'--data-tier.*choices=\[.*["\']a20["\']', script, re.DOTALL), \
            "--data-tier should have a20 in choices"
        assert re.search(r'--data-tier.*choices=\[.*["\']a200["\']', script, re.DOTALL), \
            "--data-tier should have a200 in choices"

    def test_data_tier_default_a20(self):
        """Test --data-tier defaults to a20 for backward compatibility."""
        script = read_hpo_script()
        # Look for default="a20" or default='a20' near --data-tier
        assert re.search(r'--data-tier.*default=["\']a20["\']', script, re.DOTALL), \
            "--data-tier should default to a20"

    def test_run_hpo_accepts_data_tier_param(self):
        """Test run_hpo function accepts data_tier parameter."""
        script = read_hpo_script()
        assert re.search(r'def run_hpo\([^)]*data_tier[^)]*\)', script, re.DOTALL), \
            "run_hpo should accept data_tier parameter"


class TestInputSizeFlag:
    """Tests for --input-size CLI flag for context length control."""

    def test_input_size_flag_defined(self):
        """Test --input-size flag is defined in CLI arguments."""
        script = read_hpo_script()
        assert "--input-size" in script, "--input-size flag should be defined"

    def test_input_size_default_is_80(self):
        """Test --input-size default value is 80."""
        script = read_hpo_script()
        # Look for default=80 near --input-size
        # Pattern: argument definition followed by default=80
        assert re.search(r'--input-size.*default=80', script, re.DOTALL), \
            "--input-size should have default=80"

    def test_run_hpo_accepts_input_size_param(self):
        """Test run_hpo function accepts input_size parameter."""
        script = read_hpo_script()
        # Look for input_size in run_hpo function signature
        assert re.search(r'def run_hpo\([^)]*input_size[^)]*\)', script, re.DOTALL), \
            "run_hpo should accept input_size parameter"

    def test_input_size_type_is_int(self):
        """Test --input-size argument type is int."""
        script = read_hpo_script()
        # Look for type=int near --input-size
        assert re.search(r'--input-size.*type=int', script, re.DOTALL), \
            "--input-size should have type=int"


class TestForcedExtremesFlags:
    """Tests for --forced-extremes and related budget-aware HPO flags."""

    def test_forced_extremes_flag_defined(self):
        """Test --forced-extremes flag is defined in CLI arguments."""
        script = read_hpo_script()
        assert "--forced-extremes" in script, "--forced-extremes flag should be defined"

    def test_budgets_flag_defined(self):
        """Test --budgets flag is defined for selecting parameter budgets."""
        script = read_hpo_script()
        assert "--budgets" in script, "--budgets flag should be defined"

    def test_early_stop_patience_flag_defined(self):
        """Test --early-stop-patience flag is defined."""
        script = read_hpo_script()
        assert "--early-stop-patience" in script, "--early-stop-patience flag should be defined"

    def test_early_stop_threshold_flag_defined(self):
        """Test --early-stop-threshold flag is defined."""
        script = read_hpo_script()
        assert "--early-stop-threshold" in script, "--early-stop-threshold flag should be defined"

    def test_run_hpo_accepts_forced_extremes_param(self):
        """Test run_hpo function accepts forced_extremes parameter."""
        script = read_hpo_script()
        assert re.search(r'def run_hpo\([^)]*forced_extremes[^)]*\)', script, re.DOTALL), \
            "run_hpo should accept forced_extremes parameter"


class TestSupplementaryModeFlags:
    """Tests for Phase 2 supplementary mode flags."""

    def test_supplementary_flag_defined(self):
        """Test --supplementary flag is defined for Phase 2 mode."""
        script = read_hpo_script()
        assert "--supplementary" in script, "--supplementary flag should be defined"

    def test_param_budget_flag_defined(self):
        """Test --param-budget flag is defined for focusing on specific budget."""
        script = read_hpo_script()
        assert "--param-budget" in script, "--param-budget flag should be defined"


class TestBudgetExtremesImport:
    """Tests for hpo_budget_extremes module imports."""

    def test_imports_budget_configs(self):
        """Test script imports BUDGET_CONFIGS from hpo_budget_extremes."""
        script = read_hpo_script()
        assert "from src.training.hpo_budget_extremes import" in script or \
               "import src.training.hpo_budget_extremes" in script, \
            "Should import from hpo_budget_extremes module"

    def test_imports_generate_forced_configs(self):
        """Test script imports generate_forced_configs function."""
        script = read_hpo_script()
        assert "generate_forced_configs" in script, \
            "Should use generate_forced_configs function"

    def test_imports_check_early_stopping(self):
        """Test script imports check_early_stopping_convergence function."""
        script = read_hpo_script()
        assert "check_early_stopping_convergence" in script, \
            "Should use check_early_stopping_convergence function"
