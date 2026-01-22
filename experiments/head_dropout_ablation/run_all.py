#!/usr/bin/env python3
"""
Run all head dropout ablation experiments.

Experiments:
- 2M h=8: head_dropout = 0.05, 0.15, 0.30 (baseline: 0.0, AUC 0.713)
- 20M h=4: head_dropout = 0.05, 0.15, 0.30 (baseline: 0.0, AUC 0.712)

Usage:
    python experiments/head_dropout_ablation/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    # 2M scale
    "train_2M_h8_hd005.py",
    "train_2M_h8_hd015.py",
    "train_2M_h8_hd030.py",
    # 20M scale
    "train_20M_h4_hd005.py",
    "train_20M_h4_hd015.py",
    "train_20M_h4_hd030.py",
]

def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("HEAD DROPOUT ABLATION EXPERIMENTS")
    print("=" * 70)
    print(f"Running {len(SCRIPTS)} experiments")
    print("Baselines: 2M h=8 (AUC 0.713), 20M h=4 (AUC 0.712)")
    print("=" * 70)

    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] Running {script}...")
        script_path = script_dir / script

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"ERROR: {script} failed with return code {result.returncode}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("Results saved in outputs/head_dropout_ablation/*/results.json")


if __name__ == "__main__":
    main()
