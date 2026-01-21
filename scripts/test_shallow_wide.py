#!/usr/bin/env python3
"""
Shallow + Wide Experiment
Hypothesis: Fewer layers with more capacity per layer may behave more like ensemble methods.
Depth causes early convergence, not total parameters.

Configurations:
- L1_d256: L=1, d=256, h=4, d_ff=1024 (~400K params)
- L1_d512: L=1, d=512, h=8, d_ff=2048 (~1.5M params)
- L1_d768: L=1, d=768, h=8, d_ff=3072 (~3.5M params)
- L2_d256: L=2, d=256, h=4, d_ff=1024 (~700K params)
- L2_d512: L=2, d=512, h=8, d_ff=2048 (~2.8M params)
- L2_d768: L=2, d=768, h=8, d_ff=3072 (~6.5M params)

Baseline: Current best 2M config (L=4, d=64) - AUC ~0.695
Target: RF baseline AUC 0.716
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer
from src.models.arch_grid import estimate_param_count

# ============================================================================
# CONFIGURATIONS TO TEST
# ============================================================================

CONFIGS = {
    # 1-layer (very shallow)
    "L1_d256": {"n_layers": 1, "d_model": 256, "n_heads": 4, "d_ff": 1024},
    "L1_d512": {"n_layers": 1, "d_model": 512, "n_heads": 8, "d_ff": 2048},
    "L1_d768": {"n_layers": 1, "d_model": 768, "n_heads": 8, "d_ff": 3072},
    # 2-layer (shallow)
    "L2_d256": {"n_layers": 2, "d_model": 256, "n_heads": 4, "d_ff": 1024},
    "L2_d512": {"n_layers": 2, "d_model": 512, "n_heads": 8, "d_ff": 2048},
    "L2_d768": {"n_layers": 2, "d_model": 768, "n_heads": 8, "d_ff": 3072},
    # Baseline for comparison
    "L4_d64_baseline": {"n_layers": 4, "d_model": 64, "n_heads": 4, "d_ff": 256},
}

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

CONTEXT_LENGTH = 80  # Optimal from ablation study
HORIZON = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 50
DROPOUT = 0.20
NUM_FEATURES = 20

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/shallow_wide_experiment"


def run_single_config(config_name: str, arch_config: dict, df: pd.DataFrame, device: str) -> dict:
    """Run a single architecture configuration."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")

    # Estimate param count
    est_params = estimate_param_count(
        d_model=arch_config["d_model"],
        n_layers=arch_config["n_layers"],
        n_heads=arch_config["n_heads"],
        d_ff=arch_config["d_ff"],
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
    )
    print(f"Architecture: L={arch_config['n_layers']}, d={arch_config['d_model']}, "
          f"h={arch_config['n_heads']}, d_ff={arch_config['d_ff']}")
    print(f"Estimated params: {est_params:,}")

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=arch_config["d_model"],
        n_heads=arch_config["n_heads"],
        n_layers=arch_config["n_layers"],
        d_ff=arch_config["d_ff"],
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    # Create splits
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"Train: {len(split_indices.train_indices)}, "
          f"Val: {len(split_indices.val_indices)}, "
          f"Test: {len(split_indices.test_indices)}")

    # Train
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
        )

        start_time = time.time()
        result = trainer.train(verbose=False)
        elapsed = time.time() - start_time

    val_auc = result.get("val_auc")
    val_loss = result.get("val_loss")
    stopped_early = result.get("stopped_early", False)

    print(f"\nResults:")
    print(f"  Val AUC: {val_auc:.4f}" if val_auc else "  Val AUC: N/A")
    print(f"  Val Loss: {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
    print(f"  Stopped Early: {stopped_early}")
    print(f"  Training Time: {elapsed/60:.1f} min")

    return {
        "config_name": config_name,
        "architecture": arch_config,
        "estimated_params": est_params,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "stopped_early": stopped_early,
        "training_time_min": elapsed / 60,
        "train_samples": len(split_indices.train_indices),
        "val_samples": len(split_indices.val_indices),
    }


def main():
    print("=" * 70)
    print("SHALLOW + WIDE EXPERIMENT")
    print("Hypothesis: Fewer layers with higher capacity behave like ensembles")
    print("Target: RF baseline AUC 0.716")
    print("=" * 70)

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all configs
    results = []
    for config_name, arch_config in CONFIGS.items():
        result = run_single_config(config_name, arch_config, df, device)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Shallow + Wide Experiment")
    print(f"{'='*70}")
    print(f"{'Config':<16} {'Params':<10} {'Layers':<8} {'d_model':<10} {'Val AUC':<10}")
    print("-" * 54)

    for r in sorted(results, key=lambda x: x["val_auc"] or 0, reverse=True):
        params_str = f"{r['estimated_params']/1000:.0f}K" if r['estimated_params'] < 1_000_000 else f"{r['estimated_params']/1_000_000:.1f}M"
        auc_str = f"{r['val_auc']:.4f}" if r["val_auc"] else "N/A"
        print(f"{r['config_name']:<16} {params_str:<10} {r['architecture']['n_layers']:<8} {r['architecture']['d_model']:<10} {auc_str:<10}")

    # Compare to RF baseline
    best_result = max(results, key=lambda x: x["val_auc"] or 0)
    rf_baseline = 0.716
    print(f"\nBest config: {best_result['config_name']} (AUC {best_result['val_auc']:.4f})")
    print(f"RF baseline: {rf_baseline:.4f}")
    if best_result["val_auc"]:
        gap = rf_baseline - best_result["val_auc"]
        print(f"Gap to RF: {gap:.4f} ({gap/rf_baseline*100:.1f}%)")

    # Analyze: Does width help at fixed depth?
    print(f"\n{'='*70}")
    print("ANALYSIS: Effect of width at fixed depth")
    print(f"{'='*70}")

    for n_layers in [1, 2]:
        layer_results = [r for r in results if r["architecture"]["n_layers"] == n_layers]
        if layer_results:
            layer_results.sort(key=lambda x: x["architecture"]["d_model"])
            print(f"\nL={n_layers}:")
            for r in layer_results:
                auc_str = f"{r['val_auc']:.4f}" if r["val_auc"] else "N/A"
                print(f"  d={r['architecture']['d_model']}: AUC {auc_str}")

    # Save results
    output = {
        "experiment": "shallow_wide",
        "hypothesis": "Fewer layers with higher capacity behave like ensembles",
        "target_auc": rf_baseline,
        "context_length": CONTEXT_LENGTH,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save CSV for easy comparison
    csv_path = OUTPUT_DIR / "results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
