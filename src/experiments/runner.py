"""Experiment execution utilities with thermal monitoring and logging.

Provides functions to run HPO and training experiments with:
- Pre-flight thermal checks
- CSV logging of all results (success and failure)
- Markdown report generation
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.hpo import run_hpo
from src.training.thermal import ThermalCallback
from src.training.trainer import Trainer


# CSV schema columns (22 total: 17 core + 5 architecture)
EXPERIMENT_LOG_COLUMNS = [
    "timestamp",
    "experiment",
    "phase",
    "budget",
    "task",
    "horizon",
    "timescale",
    "script_path",
    "run_type",
    "status",
    "duration_seconds",
    "val_loss",
    "test_accuracy",
    "hyperparameters",
    "error_message",
    "thermal_max_temp",
    "data_md5",
    # Architecture columns (for architectural HPO)
    "d_model",
    "n_layers",
    "n_heads",
    "d_ff",
    "param_count",
]


def update_experiment_log(result: dict[str, Any], log_path: Path | str) -> Path:
    """Append experiment result to CSV log.

    Creates the log file with headers if it doesn't exist.
    Serializes hyperparameters dict to JSON string.

    Args:
        result: Dict with experiment result fields matching EXPERIMENT_LOG_COLUMNS.
        log_path: Path to CSV log file.

    Returns:
        Path to the log file.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare row - serialize hyperparameters to JSON
    row = result.copy()
    if "hyperparameters" in row and isinstance(row["hyperparameters"], dict):
        row["hyperparameters"] = json.dumps(row["hyperparameters"])

    # Ensure all columns present
    for col in EXPERIMENT_LOG_COLUMNS:
        if col not in row:
            row[col] = None

    # Create DataFrame with single row
    df_new = pd.DataFrame([row], columns=EXPERIMENT_LOG_COLUMNS)

    # Append or create
    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(log_path, index=False)
    else:
        df_new.to_csv(log_path, index=False)

    return log_path


def regenerate_results_report(log_path: Path | str, output_path: Path | str) -> Path:
    """Regenerate markdown report from experiment CSV log.

    Creates a summary report with:
    - Total experiments and success rate
    - Results grouped by phase
    - Individual experiment details

    Args:
        log_path: Path to experiment_log.csv
        output_path: Path for output markdown file

    Returns:
        Path to the generated report.
    """
    log_path = Path(log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read log
    df = pd.read_csv(log_path)

    # Generate report
    lines = ["# Experiment Results", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if len(df) == 0:
        lines.append("No experiments recorded yet.")
        lines.append("")
    else:
        # Summary stats
        total = len(df)
        success = len(df[df["status"] == "success"])
        success_rate = (success / total * 100) if total > 0 else 0

        lines.append(f"**Total Experiments:** {total}")
        lines.append(f"**Success Rate:** {success_rate:.1f}%")
        lines.append("")

        # Group by phase
        if "phase" in df.columns:
            for phase in df["phase"].dropna().unique():
                phase_df = df[df["phase"] == phase]
                lines.append(f"## {phase}")
                lines.append("")
                lines.append(f"| Budget | Task | Status | Val Loss |")
                lines.append("|--------|------|--------|----------|")
                for _, row in phase_df.iterrows():
                    val_loss = f"{row['val_loss']:.4f}" if pd.notna(row["val_loss"]) else "-"
                    lines.append(
                        f"| {row['budget']} | {row['task']} | {row['status']} | {val_loss} |"
                    )
                lines.append("")

    # Write report
    output_path.write_text("\n".join(lines))

    return output_path


def run_hpo_experiment(
    experiment: str,
    budget: str,
    task: str,
    data_path: Path | str,
    output_dir: Path | str,
    n_trials: int = 50,
    timeout_hours: float = 4.0,
    search_space_path: str = "configs/hpo/default_search.yaml",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run HPO experiment with thermal monitoring.

    Performs pre-flight thermal check before starting.
    Aborts immediately if temperature is critical.

    Args:
        experiment: Experiment name
        budget: Parameter budget (2M, 20M, etc.)
        task: Task name (threshold_1pct, etc.)
        data_path: Path to data parquet file
        output_dir: Directory for outputs
        n_trials: Number of HPO trials
        timeout_hours: Timeout in hours
        search_space_path: Path to search space YAML
        config_path: Optional experiment config path

    Returns:
        Dict with keys: status, val_loss, thermal_max_temp, etc.
    """
    output_dir = Path(output_dir)
    start_time = time.time()

    # Initialize thermal monitoring
    thermal = ThermalCallback()
    thermal_status = thermal.check()

    # Pre-flight thermal check
    if thermal_status.status == "critical":
        return {
            "status": "thermal_abort",
            "val_loss": None,
            "thermal_max_temp": thermal_status.temperature,
            "error_message": f"Thermal abort: {thermal_status.temperature}°C",
            "duration_seconds": time.time() - start_time,
        }

    max_temp = thermal_status.temperature

    try:
        # Run HPO
        hpo_result = run_hpo(
            config_path=config_path or f"configs/experiments/{task}.yaml",
            budget=budget,
            n_trials=n_trials,
            timeout_hours=timeout_hours,
            search_space_path=search_space_path,
            thermal_callback=thermal,
        )

        # Track max temp (simplified - actual would track during HPO)
        final_status = thermal.check()
        max_temp = max(max_temp, final_status.temperature)

        return {
            "status": "success",
            "val_loss": hpo_result.get("best_value"),
            "hyperparameters": hpo_result.get("best_params", {}),
            "thermal_max_temp": max_temp,
            "error_message": None,
            "duration_seconds": time.time() - start_time,
        }

    except Exception as e:
        return {
            "status": "failed",
            "val_loss": None,
            "thermal_max_temp": max_temp,
            "error_message": str(e),
            "duration_seconds": time.time() - start_time,
        }


def run_training_experiment(
    experiment: str,
    budget: str,
    task: str,
    data_path: Path | str,
    hyperparameters: dict[str, Any],
    output_dir: Path | str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run training experiment with thermal monitoring.

    Performs pre-flight thermal check before starting.
    Aborts immediately if temperature is critical.

    Args:
        experiment: Experiment name
        budget: Parameter budget (2M, 20M, etc.)
        task: Task name (threshold_1pct, etc.)
        data_path: Path to data parquet file
        hyperparameters: Dict of hyperparameters from HPO
        output_dir: Directory for outputs
        config_path: Optional experiment config path

    Returns:
        Dict with keys: status, val_loss, test_accuracy, thermal_max_temp, etc.
    """
    output_dir = Path(output_dir)
    start_time = time.time()

    # Initialize thermal monitoring
    thermal = ThermalCallback()
    thermal_status = thermal.check()

    # Pre-flight thermal check
    if thermal_status.status == "critical":
        return {
            "status": "thermal_abort",
            "val_loss": None,
            "test_accuracy": None,
            "thermal_max_temp": thermal_status.temperature,
            "error_message": f"Thermal abort: {thermal_status.temperature}°C",
            "duration_seconds": time.time() - start_time,
        }

    max_temp = thermal_status.temperature

    try:
        # Create trainer and run
        trainer = Trainer(
            config_path=config_path or f"configs/experiments/{task}.yaml",
            budget=budget,
            hyperparameters=hyperparameters,
            thermal_callback=thermal,
        )

        train_result = trainer.train()

        # Track max temp
        final_status = thermal.check()
        max_temp = max(max_temp, final_status.temperature)

        return {
            "status": "success",
            "val_loss": train_result.get("val_loss"),
            "test_accuracy": train_result.get("test_accuracy"),
            "hyperparameters": hyperparameters,
            "thermal_max_temp": max_temp,
            "error_message": None,
            "duration_seconds": time.time() - start_time,
        }

    except Exception as e:
        return {
            "status": "failed",
            "val_loss": None,
            "test_accuracy": None,
            "thermal_max_temp": max_temp,
            "error_message": str(e),
            "duration_seconds": time.time() - start_time,
        }
