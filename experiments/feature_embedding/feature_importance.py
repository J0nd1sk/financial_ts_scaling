#!/usr/bin/env python3
"""
Feature Importance via Permutation

Computes feature importance by measuring how much precision/AUC drops when
each feature is shuffled. Higher drop = more important feature.

Usage:
    # Analyze a specific experiment
    ./venv/bin/python experiments/feature_embedding/feature_importance.py --exp-dir outputs/feature_embedding/a100_dembed_32

    # Analyze best model from tier
    ./venv/bin/python experiments/feature_embedding/feature_importance.py --tier a100

    # Quick analysis (fewer permutations)
    ./venv/bin/python experiments/feature_embedding/feature_importance.py --exp-dir ... --n-permutations 3

Design doc: docs/feature_embedding_experiments.md
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SimpleSplitter
from src.models.patchtst import PatchTST, PatchTSTConfig

# Data paths
DATA_PATHS = {
    "a100": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet",
    "a200": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet",
    "a500": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a500_combined.parquet",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/feature_embedding"


def load_model_from_experiment(exp_dir: Path) -> tuple[PatchTST, dict]:
    """Load model and config from experiment directory.

    Returns:
        Tuple of (model, results_dict)
    """
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json in {exp_dir}")

    with open(results_path) as f:
        results = json.load(f)

    # Reconstruct config
    arch = results["architecture"]
    config = PatchTSTConfig(
        num_features=results["num_features"],
        context_length=arch["context_length"],
        patch_length=arch["patch_length"],
        stride=arch["stride"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=results["hyperparameters"]["dropout"],
        head_dropout=0.0,
        d_embed=results.get("d_embed"),
    )

    # Create model
    model = PatchTST(config, use_revin=results["hyperparameters"]["use_revin"])

    # Load checkpoint if exists
    checkpoint_path = exp_dir / "best_model.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Using randomly initialized model (results will not be meaningful)")

    return model, results


def get_feature_names(tier: str) -> list[str]:
    """Get feature names for a tier."""
    data_path = DATA_PATHS[tier]
    df = pd.read_parquet(data_path)

    # Exclude non-feature columns
    exclude = {"Date", "Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in df.columns if c not in exclude]

    return feature_cols


def prepare_validation_data(
    tier: str,
    num_features: int,
    context_length: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Prepare validation data for permutation importance.

    Returns:
        Tuple of (X_val, y_val, feature_names)
    """
    data_path = DATA_PATHS[tier]
    df = pd.read_parquet(data_path)

    # Get feature columns
    exclude = {"Date", "Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in df.columns if c not in exclude][:num_features]

    # Create splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=context_length,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Get high prices for target calculation
    high_prices = df["High"].values
    close_prices = df["Close"].values

    # Calculate targets (1% threshold)
    threshold = 0.01
    targets = np.zeros(len(df))
    for i in range(len(df) - horizon):
        future_highs = high_prices[i + 1 : i + 1 + horizon]
        if len(future_highs) > 0:
            max_high = future_highs.max()
            targets[i] = 1 if max_high >= close_prices[i] * (1 + threshold) else 0

    # Get feature data
    features = df[feature_cols].values

    # Extract validation samples
    X_list = []
    y_list = []

    for idx in split_indices.val_indices:
        start_idx = idx - context_length + 1
        if start_idx >= 0:
            X_list.append(features[start_idx : idx + 1])
            y_list.append(targets[idx])

    X_val = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_list), dtype=torch.float32)

    return X_val, y_val, feature_cols


def compute_baseline_metrics(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Compute baseline metrics on unperturbed data."""
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_dev = X.to(device)
        preds = model(X_dev).cpu().numpy().flatten()

    binary_preds = (preds >= 0.5).astype(int)
    labels = y.numpy()

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None

    precision = precision_score(labels, binary_preds, zero_division=0)

    return {
        "precision": precision,
        "auc": auc,
    }


def permutation_importance(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: list[str],
    n_permutations: int = 5,
    device: str = "cpu",
    metric: str = "precision",
) -> pd.DataFrame:
    """Compute permutation importance for all features.

    Args:
        model: Trained model
        X: Validation features (n_samples, context_length, n_features)
        y: Validation labels
        feature_names: List of feature names
        n_permutations: Number of permutations per feature
        device: Device to run on
        metric: Metric to use ("precision" or "auc")

    Returns:
        DataFrame with feature importance scores
    """
    model.eval()
    model.to(device)

    # Compute baseline
    baseline = compute_baseline_metrics(model, X, y, device)
    baseline_score = baseline[metric]

    if baseline_score is None:
        print(f"WARNING: Baseline {metric} is None, falling back to precision")
        metric = "precision"
        baseline_score = baseline["precision"]

    print(f"Baseline {metric}: {baseline_score:.4f}")

    n_features = X.shape[2]
    importance_scores = []

    for feat_idx in range(n_features):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"

        drops = []
        for perm in range(n_permutations):
            # Create permuted copy
            X_perm = X.clone()

            # Shuffle this feature across all samples and timesteps
            # Get the feature values
            feat_values = X_perm[:, :, feat_idx].flatten()
            # Shuffle
            perm_idx = torch.randperm(len(feat_values))
            shuffled = feat_values[perm_idx].reshape(X_perm.shape[0], X_perm.shape[1])
            X_perm[:, :, feat_idx] = shuffled

            # Compute metrics with permuted feature
            with torch.no_grad():
                X_dev = X_perm.to(device)
                preds = model(X_dev).cpu().numpy().flatten()

            binary_preds = (preds >= 0.5).astype(int)
            labels = y.numpy()

            if metric == "auc":
                try:
                    score = roc_auc_score(labels, preds)
                except ValueError:
                    score = 0.5
            else:
                score = precision_score(labels, binary_preds, zero_division=0)

            drop = baseline_score - score
            drops.append(drop)

        mean_drop = np.mean(drops)
        std_drop = np.std(drops)

        importance_scores.append({
            "feature": feat_name,
            "feature_idx": feat_idx,
            "importance": mean_drop,
            "importance_std": std_drop,
            "baseline": baseline_score,
            "metric": metric,
        })

        if (feat_idx + 1) % 20 == 0:
            print(f"  Processed {feat_idx + 1}/{n_features} features...")

    # Create DataFrame and sort by importance
    df = pd.DataFrame(importance_scores)
    df = df.sort_values("importance", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df


def print_importance_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print summary of feature importance results."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RESULTS")
    print("=" * 70)

    metric = df["metric"].iloc[0]
    baseline = df["baseline"].iloc[0]
    print(f"\nMetric: {metric}")
    print(f"Baseline: {baseline:.4f}")

    print(f"\n## TOP {top_n} MOST IMPORTANT FEATURES\n")
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Std':<10}")
    print("-" * 70)

    for _, row in df.head(top_n).iterrows():
        print(f"{row['rank']:<6} {row['feature']:<40} {row['importance']:>+.4f}      {row['importance_std']:.4f}")

    # Show features with negative importance (shuffling improved performance)
    negative = df[df["importance"] < -0.01]
    if len(negative) > 0:
        print(f"\n## FEATURES WITH NEGATIVE IMPORTANCE (shuffling improved {metric})\n")
        for _, row in negative.iterrows():
            print(f"  {row['feature']}: {row['importance']:+.4f}")

    # Summary statistics
    print("\n## SUMMARY STATISTICS\n")
    print(f"Total features: {len(df)}")
    print(f"Features with positive importance: {len(df[df['importance'] > 0])}")
    print(f"Features with importance > 0.01: {len(df[df['importance'] > 0.01])}")
    print(f"Features with importance > 0.05: {len(df[df['importance'] > 0.05])}")

    # Top feature categories (if names follow patterns)
    print("\n## TOP FEATURE CATEGORIES\n")
    categories = {}
    for _, row in df.head(30).iterrows():
        name = row["feature"]
        # Extract category from feature name (e.g., "rsi_daily" -> "rsi")
        parts = name.split("_")
        if len(parts) > 0:
            cat = parts[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row["importance"])

    cat_importance = [(cat, np.mean(imps), len(imps)) for cat, imps in categories.items()]
    cat_importance.sort(key=lambda x: x[1], reverse=True)

    for cat, imp, count in cat_importance[:10]:
        print(f"  {cat}: avg importance {imp:+.4f} ({count} features in top 30)")


def find_best_experiment(tier: str) -> Path | None:
    """Find the best experiment for a tier based on precision."""
    best_precision = -1
    best_dir = None

    for exp_dir in OUTPUT_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith(tier):
            results_path = exp_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                precision = results.get("val_metrics", {}).get("precision", 0)
                if precision > best_precision:
                    best_precision = precision
                    best_dir = exp_dir

    return best_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compute feature importance via permutation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp-dir", type=str,
                        help="Experiment directory with results.json and checkpoint")
    parser.add_argument("--tier", type=str, choices=["a100", "a200", "a500"],
                        help="Find best experiment for tier")
    parser.add_argument("--n-permutations", type=int, default=5,
                        help="Number of permutations per feature (default: 5)")
    parser.add_argument("--metric", type=str, default="precision",
                        choices=["precision", "auc"],
                        help="Metric to use for importance (default: precision)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to display (default: 20)")

    args = parser.parse_args()

    # Find experiment directory
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if not exp_dir.is_absolute():
            exp_dir = PROJECT_ROOT / exp_dir
    elif args.tier:
        exp_dir = find_best_experiment(args.tier)
        if exp_dir is None:
            print(f"ERROR: No experiments found for tier {args.tier}")
            print("Run experiments first with: ./venv/bin/python experiments/feature_embedding/run_experiments.py")
            sys.exit(1)
        print(f"Using best experiment for {args.tier}: {exp_dir.name}")
    else:
        parser.print_help()
        print("\nError: Either --exp-dir or --tier required")
        sys.exit(1)

    if not exp_dir.exists():
        print(f"ERROR: Directory not found: {exp_dir}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {exp_dir}...")
    model, results = load_model_from_experiment(exp_dir)

    tier = results["tier"]
    num_features = results["num_features"]
    context_length = results["architecture"]["context_length"]
    horizon = 1  # Fixed for these experiments

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Prepare data
    print(f"\nPreparing validation data for {tier} ({num_features} features)...")
    X_val, y_val, feature_names = prepare_validation_data(
        tier=tier,
        num_features=num_features,
        context_length=context_length,
        horizon=horizon,
    )
    print(f"Validation samples: {len(X_val)}")
    print(f"Positive rate: {y_val.mean():.3f}")

    # Compute importance
    print(f"\nComputing permutation importance ({args.n_permutations} permutations per feature)...")
    print("This may take a few minutes...")

    importance_df = permutation_importance(
        model=model,
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        n_permutations=args.n_permutations,
        device=device,
        metric=args.metric,
    )

    # Print results
    print_importance_summary(importance_df, top_n=args.top_n)

    # Save results
    output_path = exp_dir / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nSaved feature importance to {output_path}")

    # Also save as JSON for easier programmatic access
    json_path = exp_dir / "feature_importance.json"
    importance_dict = {
        "tier": tier,
        "num_features": num_features,
        "d_embed": results.get("d_embed"),
        "metric": args.metric,
        "baseline": float(importance_df["baseline"].iloc[0]),
        "n_permutations": args.n_permutations,
        "top_features": importance_df.head(args.top_n)[["feature", "importance", "rank"]].to_dict(orient="records"),
        "all_features": importance_df[["feature", "importance", "rank"]].to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
