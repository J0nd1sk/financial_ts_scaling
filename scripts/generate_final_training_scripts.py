#!/usr/bin/env python3
"""Generate 16 final training scripts for Phase 6A.

Creates self-contained training scripts for all budget x horizon combinations
using the best architectures and training params from HPO.

Usage:
    python scripts/generate_final_training_scripts.py
"""

import os
import py_compile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.templates import generate_final_training_script


# Architecture table from HPO results (Appendix B.2 + H2 interpolation)
# Key: (budget, horizon) -> (d_model, n_layers, n_heads)
ARCHITECTURES = {
    # 2M budget
    ("2M", 1): (64, 48, 2),
    ("2M", 2): (64, 32, 2),    # H2 = H3 architecture
    ("2M", 3): (64, 32, 2),
    ("2M", 5): (64, 64, 16),
    # 20M budget
    ("20M", 1): (128, 180, 16),
    ("20M", 2): (256, 32, 2),   # H2 = H3 architecture
    ("20M", 3): (256, 32, 2),
    ("20M", 5): (384, 12, 4),
    # 200M budget
    ("200M", 1): (384, 96, 4),
    ("200M", 2): (768, 24, 16),  # H2 = H3 architecture
    ("200M", 3): (768, 24, 16),
    ("200M", 5): (256, 256, 16),
    # 2B budget
    ("2B", 1): (1024, 128, 2),
    ("2B", 2): (768, 256, 32),   # H2 = H3 architecture
    ("2B", 3): (768, 256, 32),
    ("2B", 5): (1024, 180, 4),
}

# Training params from HPO results (Appendix B.3)
# Key: budget -> (learning_rate, dropout, weight_decay, warmup_steps, epochs)
TRAINING_PARAMS = {
    "2M": (0.8e-3, 0.12, 1.0e-3, 100, 50),
    "20M": (0.55e-3, 0.20, 0.8e-3, 100, 50),
    "200M": (0.65e-3, 0.25, 0.3e-3, 200, 50),
    "2B": (0.25e-3, 0.22, 0.5e-3, 200, 50),
}

# Feature columns from SPY_dataset_a20.parquet (25 features)
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv", "adosc", "atr_14", "adx_14",
    "bb_percent_b", "vwap_20",
]

# Data path (relative to project root)
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "phase6a_final"


def main():
    """Generate all 16 final training scripts."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    budgets = ["2M", "20M", "200M", "2B"]
    horizons = [1, 2, 3, 5]

    generated = []
    errors = []

    for budget in budgets:
        lr, dropout, weight_decay, warmup, epochs = TRAINING_PARAMS[budget]

        for horizon in horizons:
            d_model, n_layers, n_heads = ARCHITECTURES[(budget, horizon)]
            d_ff = 4 * d_model  # Standard transformer ratio

            # Generate experiment name
            experiment = f"train_{budget}_h{horizon}"

            # Generate script content
            script_content = generate_final_training_script(
                experiment=experiment,
                phase="6A",
                budget=budget,
                horizon=horizon,
                # Architecture
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                # Training params
                learning_rate=lr,
                dropout=dropout,
                weight_decay=weight_decay,
                warmup_steps=warmup,
                epochs=epochs,
                # Data
                data_path=DATA_PATH,
                feature_columns=FEATURE_COLUMNS,
            )

            # Write script
            script_path = OUTPUT_DIR / f"{experiment}.py"
            script_path.write_text(script_content)

            # Verify it compiles
            try:
                py_compile.compile(str(script_path), doraise=True)
                generated.append(script_path.name)
                print(f"  {script_path.name} (d={d_model}, L={n_layers}, h={n_heads})")
            except py_compile.PyCompileError as e:
                errors.append((script_path.name, str(e)))
                print(f"  {script_path.name} - COMPILE ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated: {len(generated)}/16 scripts")
    print(f"Output directory: {OUTPUT_DIR}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll scripts compiled successfully!")
        sys.exit(0)


if __name__ == "__main__":
    print("Generating Phase 6A final training scripts...")
    print(f"Budgets: 2M, 20M, 200M, 2B")
    print(f"Horizons: h1, h2, h3, h5")
    print(f"Data: {DATA_PATH}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print()
    main()
