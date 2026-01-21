#!/usr/bin/env python3
"""
Seed Comparison: BCE vs weighted_05

Tests whether weighted_05 consistently improves accuracy over BCE baseline
across multiple random seeds.

Baseline observation (seed=42):
- BCE: AUC 0.667, Accuracy 0.896
- weighted_05: AUC 0.665, Accuracy 0.900 (+0.45%)

Question: Is this improvement consistent or just noise?
"""
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
HORIZON = 1
TASK = "threshold_1pct"

# Seeds to test
SEEDS = [42, 123, 456, 789, 1001]

# Training settings
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# 2M architecture
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 2
D_FF = 256
DROPOUT = 0.2

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
    data_path = PROJECT_ROOT / DATA_PATH
    df = pd.read_parquet(data_path)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df, features


def get_split_indices(df):
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=60,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    return splitter.split()


def evaluate(model, dataloader, device):
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
    return auc, accuracy


def run_experiment(seed: int, criterion, df, features, split_indices, output_dir, name: str):
    """Run single experiment with given seed and criterion."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    temp_path = output_dir / f"temp_{name}_{seed}.parquet"
    df.to_parquet(temp_path)

    model_config = PatchTSTConfig(
        num_features=len(features),
        context_length=60,
        patch_length=16,
        stride=8,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
        num_classes=1,
    )

    exp_config = ExperimentConfig(
        task=TASK,
        timescale="daily",
        data_path=str(temp_path),
        horizon=HORIZON,
        seed=seed,
    )

    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=output_dir / f"{name}_{seed}",
        split_indices=split_indices,
        use_revin=True,
        criterion=criterion,
    )

    start = time.time()
    result = trainer.train()
    train_time = time.time() - start
    val_loss = result.get("val_loss", result.get("train_loss"))

    auc, accuracy = evaluate(trainer.model, trainer.val_dataloader, trainer.device)

    temp_path.unlink()

    return {
        "seed": seed,
        "loss": name,
        "val_loss": val_loss,
        "auc": auc,
        "accuracy": accuracy,
        "train_time_s": train_time,
    }


def main():
    print("=" * 60)
    print("Seed Comparison: BCE vs weighted_05")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")

    output_dir = PROJECT_ROOT / "outputs" / "seed_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, features = load_data()
    split_indices = get_split_indices(df)
    print(f"Split: {len(split_indices.train_indices)} train, {len(split_indices.val_indices)} val")

    results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # BCE
        print(f"  BCE...", end=" ", flush=True)
        r = run_experiment(seed, None, df, features, split_indices, output_dir, "bce")
        print(f"AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}")
        results.append(r)

        # weighted_05
        print(f"  weighted_05...", end=" ", flush=True)
        criterion = WeightedSumLoss(alpha=0.5, gamma=2.0)
        r = run_experiment(seed, criterion, df, features, split_indices, output_dir, "weighted_05")
        print(f"AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}")
        results.append(r)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "seed_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS BY SEED")
    print("=" * 60)

    for seed in SEEDS:
        bce = results_df[(results_df["seed"] == seed) & (results_df["loss"] == "bce")].iloc[0]
        w05 = results_df[(results_df["seed"] == seed) & (results_df["loss"] == "weighted_05")].iloc[0]
        auc_diff = w05["auc"] - bce["auc"]
        acc_diff = w05["accuracy"] - bce["accuracy"]
        print(f"Seed {seed}:")
        print(f"  BCE:         AUC={bce['auc']:.4f}, Acc={bce['accuracy']:.4f}")
        print(f"  weighted_05: AUC={w05['auc']:.4f}, Acc={w05['accuracy']:.4f}")
        print(f"  Diff:        AUC={auc_diff:+.4f}, Acc={acc_diff:+.4f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    bce_results = results_df[results_df["loss"] == "bce"]
    w05_results = results_df[results_df["loss"] == "weighted_05"]

    print(f"BCE:")
    print(f"  AUC:      mean={bce_results['auc'].mean():.4f}, std={bce_results['auc'].std():.4f}")
    print(f"  Accuracy: mean={bce_results['accuracy'].mean():.4f}, std={bce_results['accuracy'].std():.4f}")

    print(f"weighted_05:")
    print(f"  AUC:      mean={w05_results['auc'].mean():.4f}, std={w05_results['auc'].std():.4f}")
    print(f"  Accuracy: mean={w05_results['accuracy'].mean():.4f}, std={w05_results['accuracy'].std():.4f}")

    # Paired differences
    auc_diffs = []
    acc_diffs = []
    for seed in SEEDS:
        bce = results_df[(results_df["seed"] == seed) & (results_df["loss"] == "bce")].iloc[0]
        w05 = results_df[(results_df["seed"] == seed) & (results_df["loss"] == "weighted_05")].iloc[0]
        auc_diffs.append(w05["auc"] - bce["auc"])
        acc_diffs.append(w05["accuracy"] - bce["accuracy"])

    print(f"\nPaired Differences (weighted_05 - BCE):")
    print(f"  AUC:      mean={np.mean(auc_diffs):+.4f}, std={np.std(auc_diffs):.4f}")
    print(f"  Accuracy: mean={np.mean(acc_diffs):+.4f}, std={np.std(acc_diffs):.4f}")

    # Conclusion
    print("\n" + "=" * 60)
    if np.mean(acc_diffs) > 0.003 and np.mean(auc_diffs) > -0.01:
        print("CONCLUSION: weighted_05 shows consistent accuracy improvement")
        print("            Consider incorporating into HPO")
    elif np.mean(acc_diffs) > 0 and np.mean(auc_diffs) > -0.005:
        print("CONCLUSION: weighted_05 shows marginal improvement")
        print("            May be worth further testing across horizons/sizes")
    else:
        print("CONCLUSION: No consistent improvement from weighted_05")
        print("            Stick with BCE baseline")
    print("=" * 60)


if __name__ == "__main__":
    main()
