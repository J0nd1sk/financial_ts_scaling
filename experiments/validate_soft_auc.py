#!/usr/bin/env python3
"""
Validation: SoftAUCLoss vs BCELoss prediction spread comparison
Quick validation to verify SoftAUCLoss produces better prediction calibration.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer
from src.training.losses import SoftAUCLoss

# Use 2M_h1 architecture
D_MODEL, N_LAYERS, N_HEADS, D_FF = 64, 48, 2, 256
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
HORIZON = 1
CONTEXT_LENGTH = 60
EPOCHS = 10  # Quick validation


def evaluate_spread(trainer):
    """Evaluate prediction spread on validation set."""
    trainer.model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in trainer.val_dataloader:
            batch_x = batch_x.to(trainer.device)
            preds = trainer.model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        auc = 0.5  # All same class

    return {
        "min": float(preds.min()),
        "max": float(preds.max()),
        "spread": float(preds.max() - preds.min()),
        "std": float(preds.std()),
        "mean": float(preds.mean()),
        "auc": auc,
        "n_samples": len(preds),
        "pos_rate": float(targets.mean()),
    }


if __name__ == "__main__":
    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create splits
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="contiguous",
    )
    split_indices = splitter.split()
    print(f"Splits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Configs
    experiment_config = ExperimentConfig(
        data_path=str(PROJECT_ROOT / DATA_PATH),
        task="threshold_1pct",
        timescale="daily",
        horizon=HORIZON,
        context_length=CONTEXT_LENGTH,
    )
    model_config = PatchTSTConfig(
        num_features=25,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=0.12,
        head_dropout=0.0,
    )

    # Train with SoftAUCLoss
    print("\n" + "=" * 50)
    print("Training with SoftAUCLoss (gamma=2.0)")
    print("=" * 50)

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=32,
        learning_rate=0.0008,
        epochs=EPOCHS,
        device="mps",
        checkpoint_dir=Path("/tmp/soft_auc_validation"),
        split_indices=split_indices,
        criterion=SoftAUCLoss(gamma=2.0),
    )

    result = trainer.train(verbose=True)
    spread_soft_auc = evaluate_spread(trainer)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\nSoftAUCLoss after {EPOCHS} epochs:")
    print(f"  Prediction range: [{spread_soft_auc['min']:.4f}, {spread_soft_auc['max']:.4f}]")
    print(f"  Spread: {spread_soft_auc['spread']:.4f}")
    print(f"  Std: {spread_soft_auc['std']:.4f}")
    print(f"  AUC-ROC: {spread_soft_auc['auc']:.4f}")
    print(f"  Val samples: {spread_soft_auc['n_samples']}, pos_rate: {spread_soft_auc['pos_rate']:.2%}")

    print("\n" + "-" * 50)
    print("COMPARISON (expected from prior investigation):")
    print("  BCE baseline spread: <0.01 (prior collapse)")
    print("  BCE baseline range: [0.518, 0.524]")
    print("-" * 50)

    if spread_soft_auc['spread'] > 0.1:
        print("\n SUCCESS: SoftAUCLoss produces meaningful spread (>0.1)")
    elif spread_soft_auc['spread'] > 0.05:
        print("\n PARTIAL: SoftAUCLoss shows improvement but spread <0.1")
    else:
        print("\n FAILED: SoftAUCLoss still showing prior collapse")
