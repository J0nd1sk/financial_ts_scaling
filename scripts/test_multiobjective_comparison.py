#!/usr/bin/env python3
"""
Multi-Objective Loss Comparison Test

Compares 8 multi-objective loss strategies against BCE baseline (AUC 0.667):

Weighted Sum (3):
1. weighted_03: α=0.3 (70% SoftAUC emphasis)
2. weighted_05: α=0.5 (balanced)
3. weighted_07: α=0.7 (70% BCE emphasis)

Two-Phase Training (3):
4. twophase_5_5: 5 epochs BCE → 5 epochs SoftAUC (lr÷10)
5. twophase_7_3: 7 epochs BCE → 3 epochs SoftAUC (lr÷10)
6. twophase_5_5_lr5: 5+5 with lr÷5 (less aggressive LR drop)

Variations (2):
7. twophase_focal: 5 epochs Focal → 5 epochs SoftAUC
8. weighted_focal_05: 0.5×Focal + 0.5×SoftAUC

Uses established foundation:
- SimpleSplitter (442 val samples) for reliable validation
- RevIN only (no Z-score) based on previous comparison results

Reference: BCE baseline AUC = 0.667 (from test_loss_comparison.py)
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from src.config.experiment import ExperimentConfig
from src.data.dataset import SimpleSplitter, FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.trainer import Trainer
from src.training.losses import SoftAUCLoss, FocalLoss, WeightedSumLoss

# ============================================================
# CONFIGURATION
# ============================================================

# Data settings
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
HORIZON = 1  # 1-day ahead prediction
TASK = "threshold_1pct"
SEED = 42

# Training settings (fixed across all configs for fair comparison)
TOTAL_EPOCHS = 10  # Total epochs budget (split for two-phase)
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# 2M architecture (same as loss comparison baseline)
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 2
D_FF = 256
DROPOUT = 0.2

# Feature columns (a20 tier)
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv",
]

# Experiment configurations
# Format: (name, config_dict)
EXPERIMENT_CONFIGS = [
    # Weighted Sum (3 configs)
    ("weighted_03", {"type": "weighted", "alpha": 0.3}),
    ("weighted_05", {"type": "weighted", "alpha": 0.5}),
    ("weighted_07", {"type": "weighted", "alpha": 0.7}),
    # Two-Phase Training (3 configs)
    ("twophase_5_5", {"type": "twophase", "criterion1": "bce", "epochs1": 5, "epochs2": 5, "lr_factor": 0.1}),
    ("twophase_7_3", {"type": "twophase", "criterion1": "bce", "epochs1": 7, "epochs2": 3, "lr_factor": 0.1}),
    ("twophase_5_5_lr5", {"type": "twophase", "criterion1": "bce", "epochs1": 5, "epochs2": 5, "lr_factor": 0.2}),
    # Variations (2 configs)
    ("twophase_focal", {"type": "twophase", "criterion1": "focal", "epochs1": 5, "epochs2": 5, "lr_factor": 0.1}),
    ("weighted_focal_05", {"type": "weighted_focal", "alpha": 0.5}),
]

# Baseline for comparison
BCE_BASELINE_AUC = 0.667

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def load_data():
    """Load and validate data."""
    data_path = PROJECT_ROOT / DATA_PATH
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Filter to feature columns that exist
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"Using {len(available_features)} features")

    return df, available_features


def get_split_indices(df):
    """Create train/val/test split indices using SimpleSplitter."""
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=60,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    return splitter.split()


def create_model_config(num_features: int) -> PatchTSTConfig:
    """Create PatchTST config."""
    return PatchTSTConfig(
        num_features=num_features,
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


def create_experiment_config(data_path: str) -> ExperimentConfig:
    """Create experiment config."""
    return ExperimentConfig(
        task=TASK,
        timescale="daily",
        data_path=data_path,
        horizon=HORIZON,
        seed=SEED,
    )


def evaluate_predictions(model, dataloader, device):
    """Evaluate model and return predictions + labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5  # If only one class present

    binary_preds = (preds >= 0.5).astype(int)
    accuracy = accuracy_score(labels, binary_preds)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "pred_min": preds.min(),
        "pred_max": preds.max(),
        "pred_std": preds.std(),
        "pred_mean": preds.mean(),
        "pred_spread": preds.max() - preds.min(),
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================


def run_weighted_sum(
    config_name: str,
    alpha: float,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    output_dir: Path,
) -> dict:
    """Run weighted sum loss training."""
    print(f"\n{'='*60}")
    print(f"Running: {config_name} (WeightedSumLoss, alpha={alpha})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Save data to temp file
    temp_data_path = output_dir / f"temp_{config_name}.parquet"
    df.to_parquet(temp_data_path)

    # Create configs
    model_config = create_model_config(len(features))
    exp_config = create_experiment_config(str(temp_data_path))

    # Create criterion
    criterion = WeightedSumLoss(alpha=alpha, gamma=2.0)

    # Create trainer
    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=TOTAL_EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=output_dir / config_name,
        split_indices=split_indices,
        use_revin=True,
        criterion=criterion,
    )

    # Train
    start_time = time.time()
    try:
        result = trainer.train()
        train_time = time.time() - start_time
        val_loss = result.get("val_loss", result.get("train_loss"))
        status = "success"
    except Exception as e:
        print(f"  ERROR: {e}")
        train_time = time.time() - start_time
        val_loss = float("nan")
        status = f"error: {e}"

    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Val loss: {val_loss:.4f}" if not np.isnan(val_loss) else "  Val loss: NaN")

    # Evaluate
    if status == "success":
        preds, labels = evaluate_predictions(trainer.model, trainer.val_dataloader, trainer.device)
        metrics = compute_metrics(preds, labels)
        print(f"  AUC-ROC: {metrics['auc']:.4f}")
        print(f"  Predictions: min={metrics['pred_min']:.4f}, max={metrics['pred_max']:.4f}")
    else:
        metrics = {"auc": 0.5, "accuracy": 0.5, "pred_min": float("nan"), "pred_max": float("nan"),
                   "pred_std": float("nan"), "pred_mean": float("nan"), "pred_spread": float("nan")}

    # Cleanup
    temp_data_path.unlink()

    return {
        "config": config_name,
        "type": "weighted_sum",
        "val_loss": val_loss,
        **metrics,
        "train_time_s": train_time,
        "status": status,
    }


def run_weighted_focal(
    config_name: str,
    alpha: float,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    output_dir: Path,
) -> dict:
    """Run weighted Focal + SoftAUC loss training.

    This is a custom weighted loss combining Focal and SoftAUC instead of BCE and SoftAUC.
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name} (Focal + SoftAUC weighted, alpha={alpha})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Save data to temp file
    temp_data_path = output_dir / f"temp_{config_name}.parquet"
    df.to_parquet(temp_data_path)

    # Create configs
    model_config = create_model_config(len(features))
    exp_config = create_experiment_config(str(temp_data_path))

    # Create custom combined criterion
    class WeightedFocalSoftAUC(nn.Module):
        def __init__(self, alpha: float, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.focal = FocalLoss(gamma=2.0, alpha=0.25)
            self.softauc = SoftAUCLoss(gamma=gamma)

        def forward(self, predictions, targets):
            predictions = predictions.view(-1)
            targets = targets.view(-1)
            focal_loss = self.focal(predictions, targets)
            softauc_loss = self.softauc(predictions, targets)
            return self.alpha * focal_loss + (1 - self.alpha) * softauc_loss

    criterion = WeightedFocalSoftAUC(alpha=alpha)

    # Create trainer
    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=TOTAL_EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=output_dir / config_name,
        split_indices=split_indices,
        use_revin=True,
        criterion=criterion,
    )

    # Train
    start_time = time.time()
    try:
        result = trainer.train()
        train_time = time.time() - start_time
        val_loss = result.get("val_loss", result.get("train_loss"))
        status = "success"
    except Exception as e:
        print(f"  ERROR: {e}")
        train_time = time.time() - start_time
        val_loss = float("nan")
        status = f"error: {e}"

    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Val loss: {val_loss:.4f}" if not np.isnan(val_loss) else "  Val loss: NaN")

    # Evaluate
    if status == "success":
        preds, labels = evaluate_predictions(trainer.model, trainer.val_dataloader, trainer.device)
        metrics = compute_metrics(preds, labels)
        print(f"  AUC-ROC: {metrics['auc']:.4f}")
        print(f"  Predictions: min={metrics['pred_min']:.4f}, max={metrics['pred_max']:.4f}")
    else:
        metrics = {"auc": 0.5, "accuracy": 0.5, "pred_min": float("nan"), "pred_max": float("nan"),
                   "pred_std": float("nan"), "pred_mean": float("nan"), "pred_spread": float("nan")}

    # Cleanup
    temp_data_path.unlink()

    return {
        "config": config_name,
        "type": "weighted_focal",
        "val_loss": val_loss,
        **metrics,
        "train_time_s": train_time,
        "status": status,
    }


def run_twophase(
    config_name: str,
    criterion1_name: str,
    epochs1: int,
    epochs2: int,
    lr_factor: float,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    output_dir: Path,
) -> dict:
    """Run two-phase training: criterion1 → SoftAUC.

    Phase 1: Train with criterion1 (BCE or Focal)
    Phase 2: Continue training with SoftAUC at reduced learning rate
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  Phase 1: {criterion1_name.upper()} for {epochs1} epochs (lr={LEARNING_RATE})")
    print(f"  Phase 2: SoftAUC for {epochs2} epochs (lr={LEARNING_RATE * lr_factor})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Save data to temp file
    temp_data_path = output_dir / f"temp_{config_name}.parquet"
    df.to_parquet(temp_data_path)

    # Create configs
    model_config = create_model_config(len(features))
    exp_config = create_experiment_config(str(temp_data_path))

    # Select phase 1 criterion
    if criterion1_name == "bce":
        criterion1 = None  # Trainer default
    elif criterion1_name == "focal":
        criterion1 = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        raise ValueError(f"Unknown criterion: {criterion1_name}")

    # ========== PHASE 1 ==========
    print(f"\n  --- Phase 1: {criterion1_name.upper()} ---")

    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=epochs1,
        device=str(device),
        checkpoint_dir=output_dir / config_name / "phase1",
        split_indices=split_indices,
        use_revin=True,
        criterion=criterion1,
    )

    start_time = time.time()
    try:
        result1 = trainer.train()
        phase1_loss = result1.get("val_loss", result1.get("train_loss"))
        print(f"  Phase 1 val_loss: {phase1_loss:.4f}")
    except Exception as e:
        print(f"  Phase 1 ERROR: {e}")
        temp_data_path.unlink()
        return {
            "config": config_name,
            "type": "twophase",
            "val_loss": float("nan"),
            "auc": 0.5,
            "accuracy": 0.5,
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "pred_std": float("nan"),
            "pred_mean": float("nan"),
            "pred_spread": float("nan"),
            "train_time_s": time.time() - start_time,
            "status": f"phase1_error: {e}",
        }

    # ========== PHASE 2 ==========
    print(f"\n  --- Phase 2: SoftAUC ---")

    # Get model state from phase 1
    model = trainer.model

    # Create new optimizer with reduced learning rate
    phase2_lr = LEARNING_RATE * lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=phase2_lr, weight_decay=1e-5)

    # Create SoftAUC criterion
    criterion2 = SoftAUCLoss(gamma=2.0)

    # Manual training loop for phase 2
    model.train()
    train_dataloader = trainer.train_dataloader

    for epoch in range(epochs2):
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x).view(-1)
            loss = criterion2(outputs, batch_y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"    Epoch {epochs1 + epoch + 1}/{epochs1 + epochs2}: train_loss={avg_loss:.4f}")

    train_time = time.time() - start_time

    # Evaluate on validation set
    preds, labels = evaluate_predictions(model, trainer.val_dataloader, device)
    metrics = compute_metrics(preds, labels)

    # Compute final val loss with phase 2 criterion
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in trainer.val_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            outputs = model(batch_x).view(-1)
            loss = criterion2(outputs, batch_y.view(-1))
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)

    print(f"  Final val_loss: {val_loss:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  Predictions: min={metrics['pred_min']:.4f}, max={metrics['pred_max']:.4f}")

    # Cleanup
    temp_data_path.unlink()

    return {
        "config": config_name,
        "type": "twophase",
        "val_loss": val_loss,
        **metrics,
        "train_time_s": train_time,
        "status": "success",
    }


def run_experiment(
    config_name: str,
    config: dict,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    output_dir: Path,
) -> dict:
    """Route to appropriate training function based on config type."""
    config_type = config["type"]

    if config_type == "weighted":
        return run_weighted_sum(
            config_name=config_name,
            alpha=config["alpha"],
            df=df,
            features=features,
            split_indices=split_indices,
            output_dir=output_dir,
        )
    elif config_type == "weighted_focal":
        return run_weighted_focal(
            config_name=config_name,
            alpha=config["alpha"],
            df=df,
            features=features,
            split_indices=split_indices,
            output_dir=output_dir,
        )
    elif config_type == "twophase":
        return run_twophase(
            config_name=config_name,
            criterion1_name=config["criterion1"],
            epochs1=config["epochs1"],
            epochs2=config["epochs2"],
            lr_factor=config["lr_factor"],
            df=df,
            features=features,
            split_indices=split_indices,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("Multi-Objective Loss Comparison Test")
    print("=" * 60)
    print(f"Configs to test: {[c[0] for c in EXPERIMENT_CONFIGS]}")
    print(f"Foundation: SimpleSplitter + RevIN only")
    print(f"Baseline to beat: BCE AUC = {BCE_BASELINE_AUC}")

    # Setup
    output_dir = PROJECT_ROOT / "outputs" / "multiobjective_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, features = load_data()

    # Get split indices
    split_indices = get_split_indices(df)
    print(f"Split: {len(split_indices.train_indices)} train, "
          f"{len(split_indices.val_indices)} val samples")

    # Run all experiments
    results = []
    for config_name, config in EXPERIMENT_CONFIGS:
        result = run_experiment(
            config_name=config_name,
            config=config,
            df=df,
            features=features,
            split_indices=split_indices,
            output_dir=output_dir,
        )
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    display_cols = ["config", "type", "val_loss", "auc", "accuracy", "pred_spread", "train_time_s"]
    print(results_df[display_cols].to_string(index=False))

    # Compare to baseline
    print("\n" + "=" * 60)
    print(f"COMPARISON TO BCE BASELINE (AUC = {BCE_BASELINE_AUC})")
    print("=" * 60)

    successful = results_df[results_df["status"] == "success"]

    if len(successful) > 0:
        for _, row in successful.iterrows():
            auc_diff = row["auc"] - BCE_BASELINE_AUC
            auc_pct = (auc_diff / BCE_BASELINE_AUC) * 100
            if auc_diff > 0.01:
                indicator = f"  {row['config']}: AUC {row['auc']:.4f} (+{auc_pct:.1f}%)"
            elif auc_diff < -0.01:
                indicator = f"  {row['config']}: AUC {row['auc']:.4f} ({auc_pct:.1f}%)"
            else:
                indicator = f"  {row['config']}: AUC {row['auc']:.4f} (~same)"
            print(indicator)

        # Best result
        best_idx = successful["auc"].idxmax()
        best_config = successful.loc[best_idx, "config"]
        best_auc = successful.loc[best_idx, "auc"]
        best_diff = best_auc - BCE_BASELINE_AUC
        best_pct = (best_diff / BCE_BASELINE_AUC) * 100

        print(f"\n{'='*60}")
        if best_auc > BCE_BASELINE_AUC + 0.01:
            print(f" WINNER: {best_config} (AUC: {best_auc:.4f}, +{best_pct:.1f}% vs baseline)")
            print(f" Recommendation: Consider integrating into HPO")
        elif best_auc > BCE_BASELINE_AUC:
            print(f" MARGINAL: {best_config} (AUC: {best_auc:.4f}, +{best_pct:.1f}% vs baseline)")
            print(f" Recommendation: May not be worth the complexity")
        else:
            print(f" NO IMPROVEMENT: Best was {best_config} (AUC: {best_auc:.4f})")
            print(f" Recommendation: Try h3 horizon, more features, or context length")

    # Prediction spread analysis
    print(f"\nPrediction Spread Analysis:")
    for _, row in results_df.iterrows():
        spread = row["pred_spread"]
        if np.isnan(spread):
            status = " ERROR"
        elif spread > 0.3:
            status = " Good spread"
        elif spread > 0.1:
            status = " Moderate spread"
        else:
            status = " Low spread"
        print(f"  {status}: {row['config']} (spread={spread:.4f})")


if __name__ == "__main__":
    main()
