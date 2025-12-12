"""Script templates for experiment generation.

Generates self-contained, executable Python scripts for HPO and training experiments.
All parameters are visible inline for reproducibility and publication.
"""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent
from typing import Any


def generate_hpo_script(
    experiment: str,
    phase: str,
    budget: str,
    task: str,
    horizon: int,
    timescale: str,
    data_path: str,
    feature_columns: list[str],
    n_trials: int = 50,
    timeout_hours: float = 4.0,
) -> str:
    """Generate HPO experiment script from template.

    Generates self-contained scripts that perform architectural HPO,
    searching both model architecture (d_model, n_layers, etc.) AND
    training parameters (lr, epochs, batch_size).

    Args:
        experiment: Full experiment name (e.g., "phase6a_2M_threshold_1pct")
        phase: Phase name (e.g., "phase6a")
        budget: Parameter budget (e.g., "2M", "20M")
        task: Task name (e.g., "threshold_1pct")
        horizon: Prediction horizon in days
        timescale: Data timescale (e.g., "daily", "weekly")
        data_path: Path to data parquet file
        feature_columns: List of feature column names
        n_trials: Number of HPO trials
        timeout_hours: HPO timeout in hours

    Returns:
        Complete Python script as string.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    feature_list_str = repr(feature_columns)

    script = dedent(f'''\
        #!/usr/bin/env python3
        """
        {phase.upper()} Experiment: {budget} parameters, {task} task
        Type: HPO (Hyperparameter Optimization) with Architectural Search
        Generated: {timestamp}

        This script searches both model ARCHITECTURE (d_model, n_layers, n_heads, d_ff)
        and TRAINING parameters (lr, epochs, batch_size) to find optimal configuration.
        """
        import sys
        import time
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        import pandas as pd
        import yaml
        from src.models.arch_grid import get_architectures_for_budget
        from src.training.hpo import (
            create_architectural_objective,
            create_study,
            save_best_params,
        )
        from src.data.dataset import ChunkSplitter
        from src.experiments.runner import update_experiment_log

        # ============================================================
        # EXPERIMENT CONFIGURATION (all parameters visible)
        # ============================================================

        EXPERIMENT = "{experiment}"
        PHASE = "{phase}"
        BUDGET = "{budget}"
        TASK = "{task}"
        HORIZON = {horizon}
        TIMESCALE = "{timescale}"
        DATA_PATH = "{data_path}"
        FEATURE_COLUMNS = {feature_list_str}

        # HPO settings
        N_TRIALS = {n_trials}
        TIMEOUT_HOURS = {timeout_hours}
        SEARCH_SPACE_PATH = "configs/hpo/architectural_search.yaml"
        CONFIG_PATH = f"configs/experiments/{{TASK}}.yaml"

        # ============================================================
        # ARCHITECTURE GRID (pre-computed valid architectures for budget)
        # ============================================================

        ARCHITECTURES = get_architectures_for_budget(
            budget=BUDGET,
            num_features=len(FEATURE_COLUMNS),
        )
        print(f"✓ Architecture grid: {{len(ARCHITECTURES)}} valid configs for {{BUDGET}}")

        # ============================================================
        # DATA VALIDATION
        # ============================================================

        def validate_data():
            """Validate data file before running experiment."""
            df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
            assert len(df) > 1000, f"Insufficient data: {{len(df)}} rows"
            assert all(col in df.columns for col in FEATURE_COLUMNS), "Missing feature columns"
            assert not df[FEATURE_COLUMNS].isna().any().any(), "NaN values in features"
            print(f"✓ Data validated: {{len(df)}} rows, {{len(FEATURE_COLUMNS)}} features")
            return df

        # ============================================================
        # HPO CONFIGURATION
        # ============================================================

        def load_training_search_space():
            """Load training parameter search space from YAML."""
            config_path = PROJECT_ROOT / SEARCH_SPACE_PATH
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("training_search_space", {{}})

        def get_split_indices(df):
            """Get train/val/test split indices."""
            splitter = ChunkSplitter(
                total_days=len(df),
                context_length=60,  # PatchTST context window
                horizon=HORIZON,
                val_ratio=0.15,
                test_ratio=0.15,
            )
            return splitter.split()

        # ============================================================
        # MAIN
        # ============================================================

        if __name__ == "__main__":
            start_time = time.time()

            # Validate data and get splits
            df = validate_data()
            split_indices = get_split_indices(df)
            print(f"✓ Splits: train={{len(split_indices.train_indices)}}, val={{len(split_indices.val_indices)}}, test={{len(split_indices.test_indices)}}")

            # Load training search space
            training_search_space = load_training_search_space()
            print(f"✓ Training params: {{list(training_search_space.keys())}}")

            # Create Optuna study
            study = create_study(
                experiment_name=EXPERIMENT,
                budget=BUDGET,
                direction="minimize",
            )

            # Create architectural objective
            objective = create_architectural_objective(
                config_path=str(PROJECT_ROOT / CONFIG_PATH),
                budget=BUDGET,
                architectures=ARCHITECTURES,
                training_search_space=training_search_space,
                split_indices=split_indices,
                num_features=len(FEATURE_COLUMNS),
            )

            # Run optimization
            print(f"\\nStarting HPO: {{N_TRIALS}} trials, {{len(ARCHITECTURES)}} architectures...")
            study.optimize(
                objective,
                n_trials=N_TRIALS,
                timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None,
            )

            # Save best params (includes architecture info)
            output_dir = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT
            output_path = save_best_params(
                study=study,
                experiment_name=EXPERIMENT,
                budget=BUDGET,
                output_dir=output_dir,
                architectures=ARCHITECTURES,
            )

            duration = time.time() - start_time

            # Get best architecture info
            best_arch = ARCHITECTURES[study.best_params.get("arch_idx", 0)]

            # Prepare result for logging
            result = {{
                "experiment": EXPERIMENT,
                "phase": PHASE,
                "budget": BUDGET,
                "task": TASK,
                "horizon": HORIZON,
                "timescale": TIMESCALE,
                "script_path": __file__,
                "run_type": "hpo",
                "status": "success",
                "val_loss": study.best_value,
                "hyperparameters": study.best_params,
                "duration_seconds": duration,
                "d_model": best_arch["d_model"],
                "n_layers": best_arch["n_layers"],
                "n_heads": best_arch["n_heads"],
                "d_ff": best_arch["d_ff"],
                "param_count": best_arch["param_count"],
            }}

            # Log to experiment CSV
            update_experiment_log(result, PROJECT_ROOT / "docs" / "experiment_results.csv")

            print(f"\\n✓ HPO complete in {{duration/60:.1f}} min")
            print(f"  Best val_loss: {{study.best_value:.6f}}")
            print(f"  Best arch: d_model={{best_arch['d_model']}}, n_layers={{best_arch['n_layers']}}, params={{best_arch['param_count']:,}}")
            print(f"  Results saved to: {{output_path}}")
    ''')

    return script


def generate_training_script(
    experiment: str,
    phase: str,
    budget: str,
    task: str,
    horizon: int,
    timescale: str,
    data_path: str,
    feature_columns: list[str],
    hyperparameters: dict[str, Any],
    borrowed_from: str | None = None,
) -> str:
    """Generate training experiment script from template.

    Args:
        experiment: Full experiment name
        phase: Phase name
        budget: Parameter budget
        task: Task name
        horizon: Prediction horizon in days
        timescale: Data timescale
        data_path: Path to data parquet file
        feature_columns: List of feature column names
        hyperparameters: Dict of hyperparameters from HPO
        borrowed_from: Source experiment if borrowing params (for 2% task)

    Returns:
        Complete Python script as string.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    feature_list_str = repr(feature_columns)
    hyperparams_str = repr(hyperparameters)
    borrowed_str = repr(borrowed_from)

    script = dedent(f'''\
        #!/usr/bin/env python3
        """
        {phase.upper()} Experiment: {budget} parameters, {task} task
        Type: Training (Final Evaluation)
        Generated: {timestamp}
        """
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))

        import pandas as pd
        from src.experiments.runner import run_training_experiment, update_experiment_log

        # ============================================================
        # EXPERIMENT CONFIGURATION (all parameters visible)
        # ============================================================

        EXPERIMENT = "{experiment}"
        PHASE = "{phase}"
        BUDGET = "{budget}"
        TASK = "{task}"
        HORIZON = {horizon}
        TIMESCALE = "{timescale}"
        DATA_PATH = "{data_path}"
        FEATURE_COLUMNS = {feature_list_str}

        # Training settings
        HYPERPARAMETERS = {hyperparams_str}
        BORROWED_FROM = {borrowed_str}  # Source if params borrowed (e.g., for 2% task)

        # ============================================================
        # DATA VALIDATION
        # ============================================================

        def validate_data():
            """Validate data file before running experiment."""
            df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
            assert len(df) > 1000, f"Insufficient data: {{len(df)}} rows"
            assert all(col in df.columns for col in FEATURE_COLUMNS), "Missing feature columns"
            assert not df[FEATURE_COLUMNS].isna().any().any(), "NaN values in features"
            print(f"✓ Data validated: {{len(df)}} rows, {{len(FEATURE_COLUMNS)}} features")

        # ============================================================
        # MAIN
        # ============================================================

        if __name__ == "__main__":
            validate_data()

            result = run_training_experiment(
                experiment=EXPERIMENT,
                budget=BUDGET,
                task=TASK,
                data_path=PROJECT_ROOT / DATA_PATH,
                hyperparameters=HYPERPARAMETERS,
                output_dir=PROJECT_ROOT / "outputs" / "training" / EXPERIMENT,
            )

            # Log result
            result.update({{
                "experiment": EXPERIMENT,
                "phase": PHASE,
                "budget": BUDGET,
                "task": TASK,
                "horizon": HORIZON,
                "timescale": TIMESCALE,
                "script_path": __file__,
                "run_type": "training",
            }})
            update_experiment_log(result, PROJECT_ROOT / "docs" / "experiment_results.csv")

            print(f"Training complete: {{result['status']}}")
    ''')

    return script
