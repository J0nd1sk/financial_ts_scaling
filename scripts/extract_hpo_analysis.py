#!/usr/bin/env python3
"""Extract HPO trial data into analysis-ready formats.

Reads all trial JSON files from outputs/hpo/phase6a_*/trials/
and generates:
- hpo_summary.csv: 600 rows, flattened for spreadsheet analysis
- hpo_full.json: Complete data with learning curves
- diverged_analysis.csv: 14 diverged trials with diagnostics
- diverged_analysis.json: Full diverged trial data
- README.md: Column definitions and usage examples
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_study_name(study_name: str) -> tuple[str, str]:
    """Parse study name into budget and horizon.

    Args:
        study_name: e.g., "phase6a_2M_h1_threshold_1pct"

    Returns:
        Tuple of (budget, horizon) e.g., ("2M", "h1")
    """
    match = re.match(r"phase6a_(\d+[MB])_h(\d+)_", study_name)
    if match:
        budget = match.group(1)
        horizon = f"h{match.group(2)}"
        return budget, horizon
    return "unknown", "unknown"


def compute_best_epoch(learning_curve: list[dict]) -> tuple[int | None, float | None]:
    """Find the epoch with minimum validation loss.

    Args:
        learning_curve: List of dicts with epoch, train_loss, val_loss

    Returns:
        Tuple of (best_epoch, best_val_loss) or (None, None) if empty
    """
    if not learning_curve:
        return None, None

    best_entry = min(learning_curve, key=lambda x: x["val_loss"])
    return best_entry["epoch"], best_entry["val_loss"]


def compute_early_stopped(epochs_trained: int, epochs_config: int) -> bool:
    """Determine if early stopping triggered.

    Args:
        epochs_trained: Actual epochs completed
        epochs_config: Configured max epochs

    Returns:
        True if training stopped early
    """
    return epochs_trained < epochs_config


def is_diverged(val_loss: float, threshold: float = 10.0) -> bool:
    """Check if trial diverged based on validation loss.

    Args:
        val_loss: Final validation loss
        threshold: Divergence threshold (default 10.0)

    Returns:
        True if val_loss >= threshold
    """
    return val_loss >= threshold


def load_trial(trial_path: Path) -> dict[str, Any]:
    """Load and parse a single trial JSON file.

    Args:
        trial_path: Path to trial JSON file

    Returns:
        Dict with extracted and derived fields
    """
    with open(trial_path) as f:
        data = json.load(f)

    # Parse study name from path
    study_name = trial_path.parent.parent.name
    budget, horizon = parse_study_name(study_name)

    # Extract architecture
    arch = data.get("architecture", {})

    # Extract training params
    params = data.get("params", {})
    user_attrs = data.get("user_attrs", {})
    training_params = user_attrs.get("training_params", params)

    # Get learning curve
    learning_curve = user_attrs.get("learning_curve", [])

    # Compute derived fields
    best_epoch, best_val_loss = compute_best_epoch(learning_curve)
    epochs_trained = len(learning_curve)
    epochs_config = training_params.get("epochs", params.get("epochs", 100))
    early_stopped = compute_early_stopped(epochs_trained, epochs_config)

    # Get arch_idx from either location
    arch_idx = user_attrs.get("arch_idx", params.get("arch_idx"))

    # Get final losses
    val_loss = data.get("value", 100.0)
    train_loss = learning_curve[-1]["train_loss"] if learning_curve else None

    return {
        # Identifiers
        "study": study_name,
        "budget": budget,
        "horizon": horizon,
        "trial_num": data.get("trial_number", 0),

        # Architecture
        "d_model": arch.get("d_model"),
        "n_layers": arch.get("n_layers"),
        "n_heads": arch.get("n_heads"),
        "d_ff": arch.get("d_ff"),
        "param_count": arch.get("param_count"),
        "arch_idx": arch_idx,

        # Training params
        "lr": training_params.get("learning_rate", params.get("learning_rate")),
        "epochs_config": epochs_config,
        "weight_decay": training_params.get("weight_decay", params.get("weight_decay")),
        "warmup_steps": training_params.get("warmup_steps", params.get("warmup_steps")),
        "dropout": training_params.get("dropout", params.get("dropout")),

        # Results
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": user_attrs.get("val_accuracy"),

        # Training dynamics
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
        "duration_sec": data.get("duration_seconds"),

        # Flags (rank computed later)
        "diverged": is_diverged(val_loss),
        "rank_in_study": None,  # Computed after all trials loaded
        "is_best_in_study": None,

        # Full learning curve (for JSON output)
        "learning_curve": learning_curve,
    }


def compute_ranks(trials: list[dict]) -> list[dict]:
    """Compute rank within each study based on val_loss.

    Args:
        trials: List of trial dicts

    Returns:
        Same list with rank_in_study and is_best_in_study populated
    """
    # Group by study
    from collections import defaultdict
    by_study: dict[str, list[dict]] = defaultdict(list)
    for trial in trials:
        by_study[trial["study"]].append(trial)

    # Rank within each study
    for study_trials in by_study.values():
        sorted_trials = sorted(study_trials, key=lambda x: x["val_loss"])
        for rank, trial in enumerate(sorted_trials, start=1):
            trial["rank_in_study"] = rank
            trial["is_best_in_study"] = (rank == 1)

    return trials


def load_all_trials(base_dir: Path) -> list[dict]:
    """Load all trials from HPO output directories.

    Args:
        base_dir: Path to outputs/hpo directory

    Returns:
        List of trial dicts with all fields populated
    """
    trials = []

    for study_dir in sorted(base_dir.glob("phase6a_*_threshold_1pct")):
        trials_dir = study_dir / "trials"
        if not trials_dir.exists():
            continue

        for trial_path in sorted(trials_dir.glob("trial_*.json")):
            trial = load_trial(trial_path)
            trials.append(trial)

    # Compute ranks after all trials loaded
    trials = compute_ranks(trials)

    return trials


def write_summary_csv(trials: list[dict], output_path: Path) -> None:
    """Write flattened trial data to CSV.

    Args:
        trials: List of trial dicts
        output_path: Path for output CSV
    """
    # Exclude learning_curve from CSV (too large)
    rows = [{k: v for k, v in t.items() if k != "learning_curve"} for t in trials]

    df = pd.DataFrame(rows)

    # Ensure column order
    column_order = [
        "study", "budget", "horizon", "trial_num",
        "d_model", "n_layers", "n_heads", "d_ff", "param_count", "arch_idx",
        "lr", "epochs_config", "weight_decay", "warmup_steps", "dropout",
        "val_loss", "train_loss", "val_accuracy",
        "best_epoch", "best_val_loss", "epochs_trained", "early_stopped", "duration_sec",
        "diverged", "rank_in_study", "is_best_in_study",
    ]

    # Only include columns that exist
    columns = [c for c in column_order if c in df.columns]
    df = df[columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_full_json(trials: list[dict], output_path: Path) -> None:
    """Write complete trial data to JSON.

    Args:
        trials: List of trial dicts
        output_path: Path for output JSON
    """
    n_diverged = sum(1 for t in trials if t["diverged"])

    # Count unique studies
    studies = set(t["study"] for t in trials)

    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "n_studies": len(studies),
            "n_trials": len(trials),
            "n_diverged": n_diverged,
        },
        "trials": trials,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def write_diverged_files(
    trials: list[dict],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Write diverged trial data to separate files.

    Args:
        trials: List of all trial dicts
        csv_path: Path for diverged CSV
        json_path: Path for diverged JSON
    """
    diverged = [t for t in trials if t["diverged"]]

    if not diverged:
        return

    # Add diagnostic columns for diverged trials
    for trial in diverged:
        curve = trial.get("learning_curve", [])

        # Find divergence point
        diverge_epoch = None
        last_good_loss = None
        loss_at_diverge = None

        for i, entry in enumerate(curve):
            if entry["val_loss"] >= 10.0:
                diverge_epoch = entry["epoch"]
                loss_at_diverge = entry["val_loss"]
                if i > 0:
                    last_good_loss = curve[i - 1]["val_loss"]
                break

        trial["diverge_epoch"] = diverge_epoch
        trial["last_good_loss"] = last_good_loss
        trial["loss_at_diverge"] = loss_at_diverge

        # Categorize architecture
        d_model = trial.get("d_model", 0)
        n_layers = trial.get("n_layers", 0)

        if n_layers > 100:
            trial["arch_category"] = "narrow-deep"
        elif d_model >= 1024:
            trial["arch_category"] = "wide-shallow"
        else:
            trial["arch_category"] = "balanced"

    # Write CSV (without learning curves)
    rows = [{k: v for k, v in t.items() if k != "learning_curve"} for t in diverged]
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Write JSON (with learning curves)
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "n_diverged": len(diverged),
            "description": "Diverged trials for failure analysis",
        },
        "trials": diverged,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)


def write_readme(output_path: Path, n_trials: int, n_diverged: int) -> None:
    """Write README with column definitions.

    Args:
        output_path: Path for README.md
        n_trials: Total number of trials
        n_diverged: Number of diverged trials
    """
    content = f"""# HPO Analysis Data

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary

- **Total trials**: {n_trials}
- **Studies**: 12 (4 budgets × 3 horizons)
- **Diverged trials**: {n_diverged}

## Files

| File | Description |
|------|-------------|
| `hpo_summary.csv` | All trials, flattened for spreadsheet analysis |
| `hpo_full.json` | Complete data with learning curves |
| `diverged_analysis.csv` | Diverged trials with diagnostic columns |
| `diverged_analysis.json` | Full diverged trial data |

## Column Definitions

### Identifiers
- `study`: Full study name (e.g., "phase6a_2M_h1_threshold_1pct")
- `budget`: Parameter budget (2M, 20M, 200M, 2B)
- `horizon`: Prediction horizon (h1, h3, h5)
- `trial_num`: Trial number within study (0-49)

### Architecture
- `d_model`: Model dimension
- `n_layers`: Number of transformer layers
- `n_heads`: Number of attention heads
- `d_ff`: Feed-forward dimension
- `param_count`: Total parameter count
- `arch_idx`: Architecture index in grid

### Training Parameters
- `lr`: Learning rate
- `epochs_config`: Configured max epochs
- `weight_decay`: Weight decay
- `warmup_steps`: Warmup steps
- `dropout`: Dropout rate

### Results
- `val_loss`: Final validation loss
- `train_loss`: Final training loss
- `val_accuracy`: Validation accuracy

### Training Dynamics
- `best_epoch`: Epoch with best val_loss
- `best_val_loss`: Best val_loss achieved
- `epochs_trained`: Actual epochs trained
- `early_stopped`: True if training stopped early
- `duration_sec`: Training duration in seconds

### Flags
- `diverged`: True if val_loss >= 10.0
- `rank_in_study`: Rank by val_loss within study (1=best)
- `is_best_in_study`: True if best trial in study

### Diverged-Only Columns
- `diverge_epoch`: First epoch where loss >= 10
- `last_good_loss`: Loss before divergence
- `loss_at_diverge`: First abnormal loss value
- `arch_category`: "narrow-deep", "wide-shallow", or "balanced"

## Example Queries

### pandas
```python
import pandas as pd

df = pd.read_csv("hpo_summary.csv")

# Best trial per study
best = df[df["is_best_in_study"] == True]

# All 2M trials
df_2m = df[df["budget"] == "2M"]

# Trials that early stopped
early = df[df["early_stopped"] == True]
```

### jq
```bash
# Best trial per study
jq '.trials | group_by(.study) | map(min_by(.val_loss))' hpo_full.json

# Count diverged by budget
jq '.trials | group_by(.budget) | map({{budget: .[0].budget, diverged: [.[] | select(.diverged)] | length}})' hpo_full.json
```
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract HPO trial data into analysis-ready formats"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/hpo"),
        help="Input directory containing HPO study folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    print(f"Loading trials from {args.input_dir}...")
    trials = load_all_trials(args.input_dir)
    print(f"Loaded {len(trials)} trials")

    n_diverged = sum(1 for t in trials if t["diverged"])
    print(f"Found {n_diverged} diverged trials")

    # Write outputs
    print(f"\nWriting outputs to {args.output_dir}/...")

    csv_path = args.output_dir / "hpo_summary.csv"
    write_summary_csv(trials, csv_path)
    print(f"  {csv_path}")

    json_path = args.output_dir / "hpo_full.json"
    write_full_json(trials, json_path)
    print(f"  {json_path}")

    div_csv = args.output_dir / "diverged_analysis.csv"
    div_json = args.output_dir / "diverged_analysis.json"
    write_diverged_files(trials, div_csv, div_json)
    print(f"  {div_csv}")
    print(f"  {div_json}")

    readme_path = args.output_dir / "README.md"
    write_readme(readme_path, len(trials), n_diverged)
    print(f"  {readme_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
