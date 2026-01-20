#!/usr/bin/env python3
"""
Context Length Ablation: Run All Experiments
Runs all context length experiments sequentially and summarizes results.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Scripts to run in order
# Note: ctx336 removed - test region (2025+) too small for 336-day context
EXPERIMENTS = [
    "train_ctx60.py",
    "train_ctx80.py",
    "train_ctx90.py",
    "train_ctx120.py",
    "train_ctx180.py",
    "train_ctx252.py",
]

def main():
    print("=" * 70)
    print("CONTEXT LENGTH ABLATION STUDY")
    print("Testing: 60, 80, 90, 120, 180, 252 days")
    print("=" * 70)

    results = []
    script_dir = Path(__file__).parent

    for script in EXPERIMENTS:
        script_path = script_dir / script
        print(f"\n{'='*70}")
        print(f"Running: {script}")
        print("=" * 70)

        # Run the experiment
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"ERROR: {script} failed with return code {result.returncode}")
            continue

        # Load results
        ctx_len = script.replace("train_ctx", "").replace(".py", "")
        results_path = PROJECT_ROOT / "outputs/context_length_ablation" / f"ctx_ablation_{ctx_len}" / "results.json"

        if results_path.exists():
            with open(results_path) as f:
                exp_results = json.load(f)
            results.append(exp_results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Context Length Ablation Results")
    print("=" * 70)
    print(f"\n{'Context':<10} {'Val AUC':<10} {'Best Epoch':<12} {'Train Time':<12}")
    print("-" * 50)

    for r in results:
        ctx = r["context_length"]
        auc = r["results"]["val_auc"]
        epoch = r["results"].get("best_epoch", "N/A")
        time_min = r["results"]["training_time_minutes"]
        print(f"{ctx:<10} {auc:<10.4f} {epoch:<12} {time_min:<12.1f}m")

    # Save summary
    summary = {
        "experiment": "context_length_ablation",
        "context_lengths_tested": [60, 80, 90, 120, 180, 252],
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = PROJECT_ROOT / "outputs/context_length_ablation/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Find best
    if results:
        best = max(results, key=lambda x: x["results"]["val_auc"])
        print(f"\nBEST: context_length={best['context_length']} with Val AUC={best['results']['val_auc']:.4f}")


if __name__ == "__main__":
    main()
