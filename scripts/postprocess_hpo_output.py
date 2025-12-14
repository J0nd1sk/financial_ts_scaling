#!/usr/bin/env python3
"""Post-process HPO output to add architecture info from trial files.

This script fixes HPO output files that were generated before the architecture
logging bug was fixed. It reads architecture info from individual trial JSON
files and regenerates _best.json and all_trials.json with the architecture data.

Usage:
    python scripts/postprocess_hpo_output.py outputs/hpo/phase6a_200M_h1_threshold_1pct

The script will:
1. Create backups of existing _best.json and all_trials.json
2. Read architecture from trial files in trials/ subdirectory
3. Regenerate both files with architecture info included
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_trial_files(trials_dir: Path) -> dict[int, dict]:
    """Load all trial JSON files from trials directory.

    Args:
        trials_dir: Path to trials/ subdirectory

    Returns:
        Dict mapping trial_number to trial data
    """
    trials = {}
    for trial_file in sorted(trials_dir.glob("trial_*.json")):
        try:
            with open(trial_file) as f:
                data = json.load(f)
            trial_num = data.get("trial_number", -1)
            if trial_num >= 0:
                trials[trial_num] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {trial_file}: {e}")
    return trials


def get_architecture_for_trial(trial_data: dict) -> dict | None:
    """Extract architecture info from trial data.

    Checks user_attrs.architecture first (for forced extreme trials),
    then falls back to params.arch_idx if architectures list provided.

    Args:
        trial_data: Trial JSON data

    Returns:
        Architecture dict or None if not found
    """
    user_attrs = trial_data.get("user_attrs", {})

    # Check user_attrs.architecture first (forced extreme trials store it here)
    if "architecture" in user_attrs:
        return user_attrs["architecture"]

    return None


def postprocess_hpo_output(output_dir: Path) -> None:
    """Post-process HPO output directory to add architecture info.

    Args:
        output_dir: Path to HPO output directory (contains trials/, *_best.json, etc.)
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    trials_dir = output_dir / "trials"
    if not trials_dir.exists():
        print(f"Warning: No trials/ directory found in {output_dir}")
        return

    # Load all trial files
    trials = load_trial_files(trials_dir)
    if not trials:
        print(f"Warning: No trial files found in {trials_dir}")
        return

    print(f"Loaded {len(trials)} trial files")

    # Find and process _best.json files
    best_files = list(output_dir.glob("*_best.json"))
    best_files = [f for f in best_files if not f.name.endswith(".bak")]

    for best_file in best_files:
        process_best_file(best_file, trials)

    # Find and process all_trials.json files
    all_trials_files = list(output_dir.glob("*_all_trials.json"))
    all_trials_files = [f for f in all_trials_files if not f.name.endswith(".bak")]

    for all_trials_file in all_trials_files:
        process_all_trials_file(all_trials_file, trials)


def process_best_file(best_file: Path, trials: dict[int, dict]) -> None:
    """Process a _best.json file to add architecture info.

    Args:
        best_file: Path to _best.json file
        trials: Dict mapping trial_number to trial data
    """
    # Create backup
    backup_path = best_file.with_suffix(".json.bak")
    shutil.copy2(best_file, backup_path)
    print(f"Created backup: {backup_path}")

    # Load existing data
    with open(best_file) as f:
        data = json.load(f)

    # Find best trial number
    best_trial_num = data.get("best_trial_number", 0)

    # Get architecture from trial file
    if best_trial_num in trials:
        arch = get_architecture_for_trial(trials[best_trial_num])
        if arch:
            data["architecture"] = arch
            print(f"Added architecture to {best_file.name} from trial {best_trial_num}")
        else:
            print(f"Warning: No architecture found for best trial {best_trial_num}")
    else:
        print(f"Warning: Best trial {best_trial_num} not found in trial files")

    # Update timestamp
    data["postprocessed_at"] = datetime.now(timezone.utc).isoformat()

    # Write updated file
    with open(best_file, "w") as f:
        json.dump(data, f, indent=2)


def process_all_trials_file(all_trials_file: Path, trials: dict[int, dict]) -> None:
    """Process an all_trials.json file to add architecture info.

    Args:
        all_trials_file: Path to all_trials.json file
        trials: Dict mapping trial_number to trial data
    """
    # Create backup
    backup_path = all_trials_file.with_suffix(".json.bak")
    shutil.copy2(all_trials_file, backup_path)
    print(f"Created backup: {backup_path}")

    # Load existing data
    with open(all_trials_file) as f:
        data = json.load(f)

    # Process each trial
    trials_updated = 0
    for trial_entry in data.get("trials", []):
        trial_num = trial_entry.get("trial_number", -1)
        if trial_num in trials:
            arch = get_architecture_for_trial(trials[trial_num])
            if arch:
                # Add architecture fields directly to trial entry
                trial_entry["d_model"] = arch.get("d_model")
                trial_entry["n_layers"] = arch.get("n_layers")
                trial_entry["n_heads"] = arch.get("n_heads")
                trial_entry["d_ff"] = arch.get("d_ff")
                trial_entry["param_count"] = arch.get("param_count")
                trials_updated += 1

    print(f"Added architecture to {trials_updated}/{len(data.get('trials', []))} trials in {all_trials_file.name}")

    # Update timestamp
    data["postprocessed_at"] = datetime.now(timezone.utc).isoformat()

    # Write updated file
    with open(all_trials_file, "w") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Post-process HPO output to add architecture info from trial files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to HPO output directory (e.g., outputs/hpo/phase6a_200M_h1_threshold_1pct)",
    )

    args = parser.parse_args()

    try:
        postprocess_hpo_output(args.output_dir)
        print("Post-processing complete!")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
