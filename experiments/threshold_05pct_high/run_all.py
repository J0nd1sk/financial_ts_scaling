#!/usr/bin/env python3
"""
Run all n_heads experiments with correct HIGH-based targets.

Experiments:
  - h=8: train_20M_wide_h8.py
  - h=4: train_20M_wide_h4.py
  - h=2: train_20M_wide_h2.py

Run with: ./venv/bin/python experiments/threshold_05pct_high/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

EXPERIMENTS = [
    "train_20M_wide_h8.py",
    "train_20M_wide_h4.py",
    "train_20M_wide_h2.py",
]

def main():
    print("=" * 70)
    print("Running all n_heads experiments with HIGH-based 0.5% threshold")
    print("=" * 70)

    for i, script in enumerate(EXPERIMENTS, 1):
        script_path = SCRIPT_DIR / script
        print(f"\n[{i}/{len(EXPERIMENTS)}] Running {script}...")
        print("-" * 70)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent,
        )

        if result.returncode != 0:
            print(f"\n❌ {script} failed with return code {result.returncode}")
            sys.exit(1)

        print(f"\n✅ {script} completed")

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("Results in: outputs/threshold_05pct_high/")
    print("=" * 70)


if __name__ == "__main__":
    main()
