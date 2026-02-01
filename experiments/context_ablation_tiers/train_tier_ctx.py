#!/usr/bin/env python3
"""
Shared training function for multi-tier context length ablation.

This module provides a common training function that accepts tier (a50, a100, a200)
and context length parameters, using the best architecture from each tier's HPO.

Usage:
    from experiments.context_ablation_tiers.train_tier_ctx import train_with_context

    result = train_with_context(
        tier="a100",
        context_length=80,
        output_dir=Path("outputs/context_ablation_tiers/a100/ctx80"),
    )
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer


# ============================================================================
# TIER CONFIGURATIONS
# ============================================================================

# Data paths for each tier
TIER_DATA = {
    "a50": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet",
    "a100": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet",
    "a200": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet",
}

# Number of features for each tier (incl. OHLCV)
TIER_NUM_FEATURES = {
    "a50": 55,   # 50 indicators + 5 OHLCV
    "a100": 105,  # 100 indicators + 5 OHLCV
    "a200": 211,  # 206 indicators + 5 OHLCV
}

# Best architecture from HPO for each tier
# These are updated after HPO runs complete
TIER_BEST_ARCH = {
    "a50": {
        # From a50 HPO results
        "d_model": 128,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff_ratio": 4,
        "dropout": 0.3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    },
    "a100": {
        # From a100 HPO results
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff_ratio": 4,
        "dropout": 0.7,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    },
    "a200": {
        # Initial config from loss sweep (update after a200 HPO completes)
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff_ratio": 4,
        "dropout": 0.7,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    },
}

# Context lengths to test
# Note: max valid context = 235 (test region has 236 days, minus 1 for horizon)
CONTEXT_LENGTHS = [60, 80, 90, 120, 180, 220]

# Fixed parameters
EPOCHS = 50
HORIZON = 1


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_context(
    tier: str,
    context_length: int,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a model with specified tier and context length.

    Args:
        tier: Feature tier ('a50', 'a100', or 'a200')
        context_length: Context window size in days
        output_dir: Directory to save results and checkpoints
        verbose: If True, print progress

    Returns:
        Dictionary with training results and metrics
    """
    if tier not in TIER_DATA:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {list(TIER_DATA.keys())}")

    if verbose:
        print("=" * 70)
        print(f"CONTEXT ABLATION: {tier.upper()} @ {context_length}d context")
        print("=" * 70)

    # Get tier-specific configuration
    data_path = TIER_DATA[tier]
    num_features = TIER_NUM_FEATURES[tier]
    arch = TIER_BEST_ARCH[tier]

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_parquet(data_path)
    high_prices = df["High"].values

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Batch size based on model size
    d_model = arch["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    experiment_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=context_length,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=num_features,
        context_length=context_length,
        patch_length=16,
        stride=8,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_model"] * arch["d_ff_ratio"],
        dropout=arch["dropout"],
        head_dropout=0.0,
    )

    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=context_length,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=arch["learning_rate"],
        weight_decay=arch["weight_decay"],
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=output_dir,
        split_indices=split_indices,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric="val_auc",
        use_revin=True,
        high_prices=high_prices,
    )

    if verbose:
        print(f"\nTier: {tier}")
        print(f"Context length: {context_length}d")
        print(f"Architecture: d_model={arch['d_model']}, n_layers={arch['n_layers']}, n_heads={arch['n_heads']}")
        print(f"Dropout: {arch['dropout']}")
        print(f"Learning rate: {arch['learning_rate']}")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(split_indices.train_indices)}")
        print(f"Validation samples: {len(split_indices.val_indices)}")
        print()

    start_time = time.time()

    try:
        result = trainer.train(verbose=verbose)
        val_auc = result.get("val_auc", 0.5)
        val_precision = result.get("val_precision", 0.0)
        val_recall = result.get("val_recall", 0.0)
        training_time = (time.time() - start_time) / 60

        if verbose:
            print()
            print("=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"Val AUC: {val_auc:.4f}")
            print(f"Val Precision: {val_precision:.4f}")
            print(f"Val Recall: {val_recall:.4f}")
            print(f"Training time: {training_time:.1f} min")

        # Package results
        results = {
            "tier": tier,
            "context_length": context_length,
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "architecture": arch,
            "batch_size": batch_size,
            "epochs": EPOCHS,
            "training_time_min": training_time,
            "train_samples": len(split_indices.train_indices),
            "val_samples": len(split_indices.val_indices),
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"\nResults saved to {results_path}")

        return results

    except Exception as e:
        if verbose:
            print(f"\nError during training: {e}")
        return {
            "tier": tier,
            "context_length": context_length,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def update_tier_architecture(tier: str, arch: dict[str, Any]) -> None:
    """Update the best architecture for a tier after HPO completes.

    This should be called after HPO runs to update TIER_BEST_ARCH with
    the optimal architecture found.

    Args:
        tier: Feature tier ('a50', 'a100', or 'a200')
        arch: Dictionary with architecture parameters
    """
    if tier not in TIER_BEST_ARCH:
        raise ValueError(f"Unknown tier: {tier}")

    TIER_BEST_ARCH[tier].update(arch)
    print(f"Updated {tier} architecture: {TIER_BEST_ARCH[tier]}")


if __name__ == "__main__":
    # Test with default parameters
    import argparse

    parser = argparse.ArgumentParser(description="Train model with specified tier and context")
    parser.add_argument("--tier", type=str, default="a100", choices=["a50", "a100", "a200"])
    parser.add_argument("--context-length", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true", help="Just print config, don't train")

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / f"outputs/context_ablation_tiers/{args.tier}/ctx{args.context_length}"

    if args.dry_run:
        print(f"Would train {args.tier} with context_length={args.context_length}")
        print(f"Output: {output_dir}")
        print(f"Architecture: {TIER_BEST_ARCH[args.tier]}")
    else:
        result = train_with_context(
            tier=args.tier,
            context_length=args.context_length,
            output_dir=output_dir,
        )
        print(f"\nFinal AUC: {result.get('val_auc', 'N/A')}")
