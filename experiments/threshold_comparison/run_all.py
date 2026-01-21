#!/usr/bin/env python3
"""
Run all threshold comparison experiments and compile results.

Tests: 0.5% (balanced), 1% (current), 2% (bigger moves)
All use: ctx=80, d=64, L=4, h=4, SimpleSplitter + RevIN
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/threshold_comparison"

EXPERIMENTS = [
    ("train_threshold_0.5pct.py", "threshold_0.5pct"),
    ("train_threshold_1pct.py", "threshold_1pct"),
    ("train_threshold_2pct.py", "threshold_2pct"),
]


def run_experiment(script_name: str, exp_name: str) -> dict | None:
    """Run a single experiment script."""
    script_path = EXPERIMENTS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"ERROR: {exp_name} failed with return code {result.returncode}")
        return None

    # Load results
    results_path = OUTPUT_DIR / exp_name / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 70)
    print("THRESHOLD COMPARISON EXPERIMENT")
    print("Testing: 0.5%, 1%, 2% thresholds")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for script_name, exp_name in EXPERIMENTS:
        result = run_experiment(script_name, exp_name)
        if result:
            all_results.append(result)

    # Compile summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Threshold':<15} {'Val AUC':<12} {'Val Loss':<12} {'Positive %':<12}")
    print("-" * 55)

    for r in all_results:
        task = r["task"]
        val_auc = r["results"]["val_auc"]
        val_loss = r["results"]["val_loss"]
        # Estimate positive rate from threshold
        thresh_map = {"threshold_0.5pct": "~50%", "threshold_1pct": "~24%", "threshold_2pct": "~6%"}
        pos_rate = thresh_map.get(task, "?")
        print(f"{task:<15} {val_auc:<12.4f} {val_loss:<12.4f} {pos_rate:<12}")

    # Save summary
    summary = {
        "experiment": "threshold_comparison",
        "thresholds_tested": ["0.5%", "1%", "2%"],
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Find winner
    if all_results:
        best = max(all_results, key=lambda x: x["results"]["val_auc"])
        print(f"\n🏆 Best AUC: {best['task']} with {best['results']['val_auc']:.4f}")


if __name__ == "__main__":
    main()
