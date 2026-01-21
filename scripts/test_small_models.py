#!/usr/bin/env python3
"""
Small Models Experiment (200K-500K params)
Hypothesis: Very small models cannot memorize noise and must learn generalizable patterns.
Current 2M model may be overparameterized for ~7K training samples.

Configurations:
- 200K_a: L=2, d=32, h=2, d_ff=128
- 200K_b: L=1, d=64, h=2, d_ff=256
- 500K_a: L=2, d=64, h=2, d_ff=256
- 500K_b: L=3, d=48, h=2, d_ff=192

Baseline: Current best 2M config (L=4, d=64) - AUC ~0.67
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
    "200K_a": {"n_layers": 2, "d_model": 32, "n_heads": 2, "d_ff": 128},
    "200K_b": {"n_layers": 1, "d_model": 64, "n_heads": 2, "d_ff": 256},
    "500K_a": {"n_layers": 2, "d_model": 64, "n_heads": 2, "d_ff": 256},
    "500K_b": {"n_layers": 3, "d_model": 48, "n_heads": 2, "d_ff": 192},
    # Baseline for comparison
    "2M_baseline": {"n_layers": 4, "d_model": 64, "n_heads": 4, "d_ff": 256},
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
OUTPUT_DIR = PROJECT_ROOT / "outputs/small_models_experiment"


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

        # Get prediction spread from validation
        val_probs = result.get("val_predictions", None)
        if val_probs is not None:
            pred_min = float(np.min(val_probs))
            pred_max = float(np.max(val_probs))
            pred_std = float(np.std(val_probs))
        else:
            pred_min = pred_max = pred_std = None

    val_auc = result.get("val_auc")
    val_loss = result.get("val_loss")
    stopped_early = result.get("stopped_early", False)
    best_epoch = result.get("best_epoch", EPOCHS)

    print(f"\nResults:")
    print(f"  Val AUC: {val_auc:.4f}" if val_auc else "  Val AUC: N/A")
    print(f"  Val Loss: {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Stopped Early: {stopped_early}")
    print(f"  Training Time: {elapsed/60:.1f} min")
    if pred_std is not None:
        print(f"  Prediction Range: [{pred_min:.3f}, {pred_max:.3f}]")
        print(f"  Prediction Std: {pred_std:.4f}")

    return {
        "config_name": config_name,
        "architecture": arch_config,
        "estimated_params": est_params,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "training_time_min": elapsed / 60,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_std": pred_std,
        "train_samples": len(split_indices.train_indices),
        "val_samples": len(split_indices.val_indices),
    }


def main():
    print("=" * 70)
    print("SMALL MODELS EXPERIMENT")
    print("Hypothesis: Smaller models (200K-500K) cannot memorize noise")
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
    print("SUMMARY: Small Models Experiment")
    print(f"{'='*70}")
    print(f"{'Config':<12} {'Params':<10} {'Val AUC':<10} {'Best Ep':<10} {'Pred Std':<10}")
    print("-" * 52)

    for r in sorted(results, key=lambda x: x["val_auc"] or 0, reverse=True):
        params_str = f"{r['estimated_params']/1000:.0f}K"
        auc_str = f"{r['val_auc']:.4f}" if r["val_auc"] else "N/A"
        pred_std_str = f"{r['pred_std']:.4f}" if r["pred_std"] else "N/A"
        print(f"{r['config_name']:<12} {params_str:<10} {auc_str:<10} {r['best_epoch']:<10} {pred_std_str:<10}")

    # Compare to RF baseline
    best_result = max(results, key=lambda x: x["val_auc"] or 0)
    rf_baseline = 0.716
    print(f"\nBest config: {best_result['config_name']} (AUC {best_result['val_auc']:.4f})")
    print(f"RF baseline: {rf_baseline:.4f}")
    if best_result["val_auc"]:
        gap = rf_baseline - best_result["val_auc"]
        print(f"Gap to RF: {gap:.4f} ({gap/rf_baseline*100:.1f}%)")

    # Save results
    output = {
        "experiment": "small_models",
        "hypothesis": "Smaller models cannot memorize noise",
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
