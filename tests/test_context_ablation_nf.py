"""Tests for context_ablation_nf.py - Context length ablation experiments.

Tests for:
1. Best config loading for each architecture
2. CLI argument parsing (--model, --context-length)
3. Dry-run mode functionality
4. Output JSON format validation
"""

import re
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Path to the context ablation script
ABLATION_SCRIPT_PATH = PROJECT_ROOT / "experiments" / "architectures" / "context_ablation_nf.py"


def read_ablation_script() -> str:
    """Read the context ablation script content."""
    if not ABLATION_SCRIPT_PATH.exists():
        pytest.skip("context_ablation_nf.py not yet created")
    return ABLATION_SCRIPT_PATH.read_text()


def extract_dict_from_script(script_content: str, dict_name: str) -> dict | None:
    """Extract a dictionary definition from the script content.

    Args:
        script_content: The script source code
        dict_name: Name of the dictionary to extract

    Returns:
        The dictionary if found, None otherwise
    """
    # Pattern to match dict definition like: DICT_NAME = {...}
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


class TestBestConfigs:
    """Tests for best hyperparameter configurations."""

    def test_itransformer_best_config_defined(self):
        """Test ITRANSFORMER_BEST config is defined."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "ITRANSFORMER_BEST")
        assert config is not None, "ITRANSFORMER_BEST should be defined"

    def test_informer_best_config_defined(self):
        """Test INFORMER_BEST config is defined."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "INFORMER_BEST")
        assert config is not None, "INFORMER_BEST should be defined"

    def test_itransformer_config_has_required_keys(self):
        """Test iTransformer config has all required hyperparameters."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "ITRANSFORMER_BEST")
        assert config is not None

        required_keys = {
            "hidden_size", "learning_rate", "max_steps", "dropout",
            "n_layers", "n_heads", "batch_size", "focal_gamma", "focal_alpha"
        }
        assert required_keys.issubset(set(config.keys())), \
            f"Missing keys: {required_keys - set(config.keys())}"

    def test_informer_config_has_required_keys(self):
        """Test Informer config has all required hyperparameters."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "INFORMER_BEST")
        assert config is not None

        required_keys = {
            "hidden_size", "learning_rate", "max_steps", "dropout",
            "n_layers", "n_heads", "batch_size", "focal_gamma", "focal_alpha"
        }
        assert required_keys.issubset(set(config.keys())), \
            f"Missing keys: {required_keys - set(config.keys())}"

    def test_itransformer_config_values_match_hpo_results(self):
        """Test iTransformer config values match documented HPO results."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "ITRANSFORMER_BEST")
        assert config is not None

        # From HPO results: hidden_size=32, lr=1e-5, max_steps=3000
        assert config["hidden_size"] == 32
        assert config["learning_rate"] == 1e-5
        assert config["max_steps"] == 3000
        assert config["dropout"] == 0.4
        assert config["n_layers"] == 6
        assert config["n_heads"] == 4

    def test_informer_config_values_match_hpo_results(self):
        """Test Informer config values match documented HPO results."""
        script = read_ablation_script()
        config = extract_dict_from_script(script, "INFORMER_BEST")
        assert config is not None

        # From HPO results: hidden_size=256, lr=1e-4, max_steps=1000
        assert config["hidden_size"] == 256
        assert config["learning_rate"] == 1e-4
        assert config["max_steps"] == 1000
        assert config["dropout"] == 0.4
        assert config["n_layers"] == 2
        assert config["n_heads"] == 2


class TestCLIArguments:
    """Tests for CLI argument definitions."""

    def test_model_argument_defined(self):
        """Test --model argument is defined."""
        script = read_ablation_script()
        assert "--model" in script

    def test_context_length_argument_defined(self):
        """Test --context-length argument is defined."""
        script = read_ablation_script()
        assert "--context-length" in script

    def test_dry_run_argument_defined(self):
        """Test --dry-run argument is defined."""
        script = read_ablation_script()
        assert "--dry-run" in script

    def test_model_choices_include_both_architectures(self):
        """Test --model choices include itransformer and informer."""
        script = read_ablation_script()
        # Look for choices with both models
        assert re.search(r'choices=\[.*["\']itransformer["\'].*\]', script)
        assert re.search(r'choices=\[.*["\']informer["\'].*\]', script)

    def test_context_length_default_is_80(self):
        """Test --context-length default value is 80."""
        script = read_ablation_script()
        # Look for default=80 near --context-length
        assert re.search(r'--context-length.*default=80', script, re.DOTALL), \
            "--context-length should have default=80"


class TestFunctions:
    """Tests for core function definitions."""

    def test_run_single_eval_function_defined(self):
        """Test run_single_eval function is defined."""
        script = read_ablation_script()
        assert "def run_single_eval(" in script

    def test_prepare_data_function_defined(self):
        """Test prepare_data or similar data preparation function exists."""
        script = read_ablation_script()
        # Either uses common.prepare_hpo_data or defines its own
        assert "prepare_hpo_data" in script or "def prepare_data(" in script

    def test_main_function_defined(self):
        """Test main function is defined."""
        script = read_ablation_script()
        assert "def main(" in script

    def test_run_single_eval_accepts_context_length(self):
        """Test run_single_eval accepts context_length parameter."""
        script = read_ablation_script()
        assert re.search(r'def run_single_eval\([^)]*context_length[^)]*\)', script, re.DOTALL)


class TestOutputFormat:
    """Tests for output directory and file format."""

    def test_output_directory_pattern(self):
        """Test output follows pattern: outputs/architectures/context_ablation/{model}/ctx{N}/"""
        script = read_ablation_script()
        # Look for output path construction with context_ablation
        assert "context_ablation" in script
        # Should include ctx prefix for context length
        assert re.search(r'ctx.*\d+|ctx.*context', script, re.IGNORECASE)

    def test_results_json_saved(self):
        """Test results.json is saved."""
        script = read_ablation_script()
        assert "results.json" in script


class TestScriptStructure:
    """Tests for overall script structure."""

    def test_script_is_executable(self):
        """Test script has proper entry point."""
        script = read_ablation_script()
        assert 'if __name__ == "__main__":' in script

    def test_imports_common_module(self):
        """Test script imports from common module."""
        script = read_ablation_script()
        assert "from experiments.architectures.common import" in script

    def test_imports_neuralforecast(self):
        """Test script imports NeuralForecast components."""
        script = read_ablation_script()
        assert "neuralforecast" in script.lower()

    def test_uses_focal_loss(self):
        """Test script uses focal loss (from HPO)."""
        script = read_ablation_script()
        # Either imports or defines focal loss
        assert "FocalLoss" in script or "focal" in script.lower()
