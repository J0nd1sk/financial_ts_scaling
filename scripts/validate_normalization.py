#!/usr/bin/env python3
"""Quick validation script for feature normalization fix.

Tests that:
1. Normalization params computed correctly from training data
2. Model trained with normalized features produces varied predictions
3. Predictions on recent data (2024-2025) are not collapsed to ~0.52

Usage:
    python scripts/validate_normalization.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    ChunkSplitter,
    FinancialDataset,
    compute_normalization_params,
    normalize_dataframe,
    BOUNDED_FEATURES,
)
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.config.experiment import ExperimentConfig


def main():
    print("=" * 60)
    print("NORMALIZATION VALIDATION")
    print("=" * 60)

    # Load data
    data_path = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a25.parquet"
    df = pd.read_parquet(data_path)
    print(f"\nLoaded data: {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Use first 70% for computing normalization stats
    train_end_row = int(len(df) * 0.70)
    print(f"\nComputing normalization params from rows 0-{train_end_row}")

    # Compute normalization params
    norm_params = compute_normalization_params(df, train_end_row)
    print(f"Normalized {len(norm_params)} features")
    print(f"Excluded (bounded): {BOUNDED_FEATURES}")

    # Show a few params
    print("\nSample normalization params:")
    for i, (feature, (mean, std)) in enumerate(list(norm_params.items())[:5]):
        print(f"  {feature}: mean={mean:.2f}, std={std:.2f}")

    # Apply normalization
    df_norm = normalize_dataframe(df, norm_params)

    # Verify normalization worked - check stats in train portion
    print("\nVerifying normalization (train portion stats):")
    for feature in ["Close", "Volume", "macd_line"]:
        if feature in norm_params:
            train_vals = df_norm[feature].iloc[:train_end_row]
            print(f"  {feature}: mean={train_vals.mean():.4f}, std={train_vals.std():.4f}")

    # Check bounded features unchanged
    print("\nBounded features (should be unchanged):")
    for feature in ["rsi_daily", "bb_percent_b"]:
        if feature in df.columns:
            orig = df[feature].iloc[0]
            norm = df_norm[feature].iloc[0]
            print(f"  {feature}: orig={orig:.4f}, norm={norm:.4f}, match={abs(orig-norm) < 0.001}")

    # Create a small model and train briefly
    print("\n" + "=" * 60)
    print("TRAINING SMALL MODEL (2M params, 5 epochs)")
    print("=" * 60)

    # Get feature columns
    feature_cols = [c for c in df_norm.columns if c != "Date"]
    num_features = len(feature_cols)

    # Create dataset from normalized data
    context_length = 60
    horizon = 1
    threshold = 0.01

    dataset = FinancialDataset(
        features_df=df_norm,
        close_prices=df_norm["Close"].values,
        context_length=context_length,
        horizon=horizon,
        threshold=threshold,
    )

    print(f"Dataset: {len(dataset)} samples, {num_features} features")

    # Simple train/test split by time
    train_size = int(len(dataset) * 0.8)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))

    # Create model (small 2M config)
    config = PatchTSTConfig(
        num_features=num_features,
        context_length=context_length,
        patch_length=10,
        stride=5,
        d_model=64,
        n_heads=2,
        n_layers=32,  # ~2M params
        d_ff=256,
        dropout=0.1,
        head_dropout=0.0,
        num_classes=1,
    )

    model = PatchTST(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    # Simple training loop
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    print(f"Device: {device}")
    print("Training...")

    batch_size = 32
    for epoch in range(5):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle train indices
        shuffled = np.random.permutation(train_indices)

        for i in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[i:i+batch_size]

            # Get batch
            x_batch = torch.stack([dataset[j][0] for j in batch_idx]).to(device)
            y_batch = torch.stack([dataset[j][1] for j in batch_idx]).to(device)

            # Forward
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Evaluate on test set (recent data)
    print("\n" + "=" * 60)
    print("EVALUATING ON RECENT DATA (2024-2025)")
    print("=" * 60)

    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for idx in test_indices[-500:]:  # Last 500 samples
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            pred = model(x).cpu().item()
            predictions.append(pred)
            labels.append(y.item())

    predictions = np.array(predictions)
    labels = np.array(labels)

    print(f"\nPredictions on recent data:")
    print(f"  Min:  {predictions.min():.4f}")
    print(f"  Max:  {predictions.max():.4f}")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std:  {predictions.std():.4f}")
    print(f"  Spread (max-min): {predictions.max() - predictions.min():.4f}")

    # Check if predictions are varied
    spread = predictions.max() - predictions.min()
    if spread > 0.05:
        print("\n✅ SUCCESS: Predictions show meaningful variation!")
        print("   (spread > 0.05, not collapsed to ~0.52)")
    else:
        print("\n⚠️ WARNING: Predictions still show low variation")
        print("   This may need further investigation")

    # Compare with actual labels
    print(f"\nActual labels: {labels.mean():.2%} positive rate")

    # Simple accuracy
    pred_binary = (predictions > 0.5).astype(float)
    accuracy = (pred_binary == labels).mean()
    print(f"Accuracy (threshold=0.5): {accuracy:.2%}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
