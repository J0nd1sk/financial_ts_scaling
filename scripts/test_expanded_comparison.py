#!/usr/bin/env python3
"""
Expanded Comparison: BCE vs weighted_05

Tests whether weighted_05 (50% BCE + 50% SoftAUC) consistently outperforms BCE
across different horizons (h1, h3, h5) and model sizes (2M, 20M).

Based on seed comparison finding:
- BCE converges to identical accuracy (majority class) across all seeds
- weighted_05 shows marginal improvement (+0.23% accuracy, AUC ~same)

This expanded test checks if the pattern holds across horizons and scales.

12 experiments total: 2 losses x 3 horizons x 2 sizes
Estimated runtime: ~70 minutes

Usage:
  python scripts/test_expanded_comparison.py              # Run all (12 experiments)
  python scripts/test_expanded_comparison.py --size 2M    # Run only 2M (6 experiments)
  python scripts/test_expanded_comparison.py --size 20M   # Run only 20M (6 experiments)
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from src.config.experiment import ExperimentConfig
from src.data.dataset import SimpleSplitter
from src.models.patchtst import PatchTSTConfig
from src.training.trainer import Trainer
from src.training.losses import WeightedSumLoss

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
TASK = "threshold_1pct"
SEED = 42

# Horizons to test
HORIZONS = [1, 3, 5]

# Model sizes with architectures
MODEL_CONFIGS = {
    "2M": {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 2,
        "d_ff": 256,
    },
    "20M": {
        "d_model": 256,
        "n_layers": 32,
        "n_heads": 4,
        "d_ff": 1024,
    },
}

# Training settings
EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT = 0.2

# Batch sizes (20M needs smaller batch)
BATCH_SIZES = {
    "2M": 32,
    "20M": 16,
}

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv",
]

# ============================================================
# FUNCTIONS
# ============================================================


def load_data():
    """Load dataset and return dataframe with feature columns."""
    data_path = PROJECT_ROOT / DATA_PATH
    df = pd.read_parquet(data_path)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df, features


def get_split_indices(df, horizon: int):
    """Get train/val/test split indices for given horizon."""
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=60,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    return splitter.split()


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader, return AUC and accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5
    accuracy = accuracy_score(labels, (preds >= 0.5).astype(int))
    pred_spread = preds.max() - preds.min()
    return auc, accuracy, pred_spread


def run_experiment(
    model_size: str,
    horizon: int,
    criterion,
    loss_name: str,
    df,
    features,
    output_dir: Path,
) -> dict:
    """Run single experiment with given configuration."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    split_indices = get_split_indices(df, horizon)
    arch = MODEL_CONFIGS[model_size]
    batch_size = BATCH_SIZES[model_size]

    # Create temp data file
    temp_path = output_dir / f"temp_{model_size}_h{horizon}_{loss_name}.parquet"
    df.to_parquet(temp_path)

    model_config = PatchTSTConfig(
        num_features=len(features),
        context_length=60,
        patch_length=16,
        stride=8,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=DROPOUT,
        head_dropout=0.0,
        num_classes=1,
    )

    exp_config = ExperimentConfig(
        task=TASK,
        timescale="daily",
        data_path=str(temp_path),
        horizon=horizon,
        seed=SEED,
    )

    checkpoint_dir = output_dir / f"{model_size}_h{horizon}_{loss_name}"
    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=checkpoint_dir,
        split_indices=split_indices,
        use_revin=True,
        criterion=criterion,
    )

    start = time.time()
    result = trainer.train()
    train_time = time.time() - start
    val_loss = result.get("val_loss", result.get("train_loss"))

    auc, accuracy, pred_spread = evaluate(trainer.model, trainer.val_dataloader, trainer.device)

    # Cleanup temp file
    temp_path.unlink()

    return {
        "model_size": model_size,
        "horizon": horizon,
        "loss": loss_name,
        "val_loss": val_loss,
        "auc": auc,
        "accuracy": accuracy,
        "pred_spread": pred_spread,
        "train_time_s": train_time,
        "n_train": len(split_indices.train_indices),
        "n_val": len(split_indices.val_indices),
    }


def main():
    parser = argparse.ArgumentParser(description="BCE vs weighted_05 comparison")
    parser.add_argument("--size", choices=["2M", "20M"], help="Run only this model size")
    args = parser.parse_args()

    # Filter model configs if --size specified
    if args.size:
        model_configs = {args.size: MODEL_CONFIGS[args.size]}
    else:
        model_configs = MODEL_CONFIGS

    print("=" * 70)
    print("Expanded Comparison: BCE vs weighted_05")
    print("=" * 70)
    print(f"Model sizes: {list(model_configs.keys())}")
    print(f"Horizons: {HORIZONS}")
    print(f"Total experiments: {len(model_configs) * len(HORIZONS) * 2}")
    print()

    output_dir = PROJECT_ROOT / "outputs" / "expanded_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, features = load_data()
    print(f"Data loaded: {len(df)} rows, {len(features)} features")

    results = []
    total_experiments = len(model_configs) * len(HORIZONS) * 2
    exp_num = 0

    for model_size in model_configs:
        for horizon in HORIZONS:
            # BCE
            exp_num += 1
            print(f"\n[{exp_num}/{total_experiments}] {model_size} h{horizon} BCE...", end=" ", flush=True)
            r = run_experiment(model_size, horizon, None, "bce", df, features, output_dir)
            print(f"AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}, spread={r['pred_spread']:.3f}, {r['train_time_s']:.0f}s")
            results.append(r)

            # weighted_05
            exp_num += 1
            print(f"[{exp_num}/{total_experiments}] {model_size} h{horizon} weighted_05...", end=" ", flush=True)
            criterion = WeightedSumLoss(alpha=0.5, gamma=2.0)
            r = run_experiment(model_size, horizon, criterion, "weighted_05", df, features, output_dir)
            print(f"AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}, spread={r['pred_spread']:.3f}, {r['train_time_s']:.0f}s")
            results.append(r)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "expanded_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # ============================================================
    # ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("RESULTS BY CONFIGURATION")
    print("=" * 70)

    for model_size in model_configs:
        print(f"\n### {model_size} ###")
        for horizon in HORIZONS:
            bce = results_df[
                (results_df["model_size"] == model_size)
                & (results_df["horizon"] == horizon)
                & (results_df["loss"] == "bce")
            ].iloc[0]
            w05 = results_df[
                (results_df["model_size"] == model_size)
                & (results_df["horizon"] == horizon)
                & (results_df["loss"] == "weighted_05")
            ].iloc[0]
            auc_diff = w05["auc"] - bce["auc"]
            acc_diff = w05["accuracy"] - bce["accuracy"]
            print(f"  h{horizon}:")
            print(f"    BCE:         AUC={bce['auc']:.4f}, Acc={bce['accuracy']:.4f}")
            print(f"    weighted_05: AUC={w05['auc']:.4f}, Acc={w05['accuracy']:.4f}")
            print(f"    Diff:        AUC={auc_diff:+.4f}, Acc={acc_diff:+.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("PAIRED DIFFERENCES (weighted_05 - BCE)")
    print("=" * 70)

    paired_diffs = []
    for model_size in model_configs:
        for horizon in HORIZONS:
            bce = results_df[
                (results_df["model_size"] == model_size)
                & (results_df["horizon"] == horizon)
                & (results_df["loss"] == "bce")
            ].iloc[0]
            w05 = results_df[
                (results_df["model_size"] == model_size)
                & (results_df["horizon"] == horizon)
                & (results_df["loss"] == "weighted_05")
            ].iloc[0]
            paired_diffs.append({
                "model_size": model_size,
                "horizon": horizon,
                "auc_diff": w05["auc"] - bce["auc"],
                "acc_diff": w05["accuracy"] - bce["accuracy"],
            })

    paired_df = pd.DataFrame(paired_diffs)
    print(f"\nAll pairs:")
    for _, row in paired_df.iterrows():
        print(f"  {row['model_size']} h{row['horizon']}: AUC={row['auc_diff']:+.4f}, Acc={row['acc_diff']:+.4f}")

    print(f"\nSummary:")
    print(f"  AUC diff:  mean={paired_df['auc_diff'].mean():+.4f}, std={paired_df['auc_diff'].std():.4f}")
    print(f"  Acc diff:  mean={paired_df['acc_diff'].mean():+.4f}, std={paired_df['acc_diff'].std():.4f}")

    # Count wins/losses
    auc_wins = (paired_df["auc_diff"] > 0).sum()
    auc_losses = (paired_df["auc_diff"] < 0).sum()
    acc_wins = (paired_df["acc_diff"] > 0).sum()
    acc_losses = (paired_df["acc_diff"] < 0).sum()
    print(f"\n  AUC:  {auc_wins} wins, {auc_losses} losses, {6 - auc_wins - auc_losses} ties")
    print(f"  Acc:  {acc_wins} wins, {acc_losses} losses, {6 - acc_wins - acc_losses} ties")

    # By model size
    print("\nBy model size:")
    for model_size in model_configs:
        subset = paired_df[paired_df["model_size"] == model_size]
        print(f"  {model_size}: AUC={subset['auc_diff'].mean():+.4f}, Acc={subset['acc_diff'].mean():+.4f}")

    # By horizon
    print("\nBy horizon:")
    for horizon in HORIZONS:
        subset = paired_df[paired_df["horizon"] == horizon]
        print(f"  h{horizon}: AUC={subset['auc_diff'].mean():+.4f}, Acc={subset['acc_diff'].mean():+.4f}")

    # ============================================================
    # CONCLUSION
    # ============================================================

    print("\n" + "=" * 70)
    mean_auc_diff = paired_df["auc_diff"].mean()
    mean_acc_diff = paired_df["acc_diff"].mean()

    if mean_auc_diff > 0.01 and auc_wins >= 4:
        print("CONCLUSION: weighted_05 shows CONSISTENT AUC improvement")
        print("            RECOMMEND: Incorporate into HPO")
    elif mean_auc_diff > 0.005 and auc_wins >= 3:
        print("CONCLUSION: weighted_05 shows MARGINAL improvement")
        print("            RECOMMEND: Consider for specific horizons/sizes where it wins")
    elif mean_auc_diff < -0.01 and auc_losses >= 4:
        print("CONCLUSION: weighted_05 is CONSISTENTLY WORSE")
        print("            RECOMMEND: Stick with BCE")
    else:
        print("CONCLUSION: No consistent difference between BCE and weighted_05")
        print("            RECOMMEND: Stick with BCE (simpler)")

    print("=" * 70)


if __name__ == "__main__":
    main()
