"""Feature importance analysis module.

Provides multiple methods for assessing feature importance and signal quality:
1. Target Correlation (Pearson/Spearman) - Linear/monotonic signal
2. Mutual Information - Non-linear signal
3. Redundancy Matrix - Identify duplicate/highly correlated features
4. Permutation Importance - Model-based importance (requires trained model)
5. Stability Analysis - Cross-seed variance (requires multiple model runs)

Combined Score formula:
    combined_score = (
        0.40 * precision_drop_normalized +      # Model says it matters
        0.30 * target_correlation_normalized +  # Statistical signal
        0.20 * mutual_information_normalized +  # Non-linear signal
        0.10 * (1 - redundancy_normalized)      # Unique information
    )

Note: When precision_drop is not available (no model), weights are redistributed:
    combined_score = (
        0.45 * target_correlation_normalized +
        0.35 * mutual_information_normalized +
        0.20 * (1 - redundancy_normalized)
    )
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif


def compute_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.Series:
    """Compute correlation between each feature and the target.

    Args:
        X: Feature DataFrame with columns as feature names.
        y: Target Series (binary or continuous).
        method: Correlation method - "pearson" or "spearman".

    Returns:
        Series with correlation values, indexed by feature name.
    """
    correlations = {}
    y_arr = y.values

    for col in X.columns:
        x_arr = X[col].values

        # Handle any NaN by dropping
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        if mask.sum() < 10:
            correlations[col] = 0.0
            continue

        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        if method == "pearson":
            corr, _ = pearsonr(x_clean, y_clean)
        else:
            corr, _ = spearmanr(x_clean, y_clean)

        correlations[col] = corr if not np.isnan(corr) else 0.0

    return pd.Series(correlations)


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Compute mutual information between each feature and the target.

    Uses sklearn's mutual_info_classif for binary classification targets.

    Args:
        X: Feature DataFrame with columns as feature names.
        y: Binary target Series.
        n_neighbors: Number of neighbors for MI estimation.
        random_state: Random state for reproducibility.

    Returns:
        Series with MI values, indexed by feature name.
    """
    # Handle any NaN by filling with column mean
    X_filled = X.fillna(X.mean())
    y_arr = y.values.astype(int)

    mi_values = mutual_info_classif(
        X_filled,
        y_arr,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    return pd.Series(mi_values, index=X.columns)


def compute_redundancy_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix between features.

    Uses absolute Pearson correlation to measure redundancy.

    Args:
        X: Feature DataFrame.

    Returns:
        Square DataFrame with absolute correlations between features.
    """
    corr_matrix = X.corr(method="pearson").abs()
    return corr_matrix


def get_max_redundancy(X: pd.DataFrame) -> pd.Series:
    """Get maximum redundancy (correlation) with any other feature.

    For each feature, returns its maximum absolute correlation with any
    other feature (excluding self-correlation).

    Args:
        X: Feature DataFrame.

    Returns:
        Series with max redundancy per feature.
    """
    corr_matrix = compute_redundancy_matrix(X)

    max_redundancy = {}
    for col in corr_matrix.columns:
        # Get correlations with other features (excluding self)
        other_corrs = corr_matrix[col].drop(col)
        max_redundancy[col] = other_corrs.max()

    return pd.Series(max_redundancy)


def normalize_to_01(scores: pd.Series) -> pd.Series:
    """Normalize scores to [0, 1] range using min-max scaling.

    Args:
        scores: Series of scores to normalize.

    Returns:
        Normalized Series in [0, 1] range.
    """
    min_val = scores.min()
    max_val = scores.max()

    if max_val == min_val:
        # All values same - return 0.5 for all
        return pd.Series(0.5, index=scores.index)

    return (scores - min_val) / (max_val - min_val)


def compute_combined_scores(
    X: pd.DataFrame,
    y: pd.Series,
    precision_drop: pd.Series | None = None,
) -> pd.Series:
    """Compute combined importance scores from multiple methods.

    When precision_drop is available (from model):
        combined = 0.40 * precision_drop + 0.30 * correlation + 0.20 * MI + 0.10 * (1 - redundancy)

    When precision_drop is not available:
        combined = 0.45 * correlation + 0.35 * MI + 0.20 * (1 - redundancy)

    Args:
        X: Feature DataFrame.
        y: Target Series.
        precision_drop: Optional Series with precision drop from permutation importance.

    Returns:
        Series with combined scores in [0, 1].
    """
    # Compute individual metrics
    correlation = compute_target_correlation(X, y, method="spearman")
    correlation_abs = correlation.abs()  # Use absolute correlation

    mi = compute_mutual_information(X, y)
    max_redundancy = get_max_redundancy(X)

    # Normalize all metrics to [0, 1]
    corr_norm = normalize_to_01(correlation_abs)
    mi_norm = normalize_to_01(mi)
    red_norm = normalize_to_01(max_redundancy)

    # Redundancy penalty: higher redundancy = lower score
    red_penalty = 1 - red_norm

    if precision_drop is not None:
        # Full formula with model-based importance
        prec_norm = normalize_to_01(precision_drop)
        combined = (
            0.40 * prec_norm +
            0.30 * corr_norm +
            0.20 * mi_norm +
            0.10 * red_penalty
        )
    else:
        # Redistributed weights without model
        combined = (
            0.45 * corr_norm +
            0.35 * mi_norm +
            0.20 * red_penalty
        )

    return combined


def generate_importance_report(
    X: pd.DataFrame,
    y: pd.Series,
    precision_drop: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate comprehensive importance report for all features.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        precision_drop: Optional Series with precision drop from permutation importance.

    Returns:
        DataFrame with columns:
        - target_correlation: Spearman correlation with target
        - mutual_information: MI with target
        - max_redundancy: Max correlation with any other feature
        - precision_drop: Precision drop from permutation (if provided)
        - combined_score: Weighted combined score

        Sorted by combined_score descending.
    """
    correlation = compute_target_correlation(X, y, method="spearman")
    mi = compute_mutual_information(X, y)
    max_redundancy = get_max_redundancy(X)
    combined = compute_combined_scores(X, y, precision_drop)

    report_data = {
        "target_correlation": correlation,
        "mutual_information": mi,
        "max_redundancy": max_redundancy,
        "combined_score": combined,
    }

    if precision_drop is not None:
        report_data["precision_drop"] = precision_drop

    report = pd.DataFrame(report_data)

    # Sort by combined score descending
    report = report.sort_values("combined_score", ascending=False)

    return report


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "precision",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Compute permutation importance for trained model.

    Shuffles each feature and measures drop in specified metric.

    Args:
        model: Trained model with predict method.
        X: Feature DataFrame (validation set).
        y: Target Series.
        metric: Metric to measure drop ("precision", "auc").
        n_repeats: Number of permutation repeats per feature.
        random_state: Random state for reproducibility.

    Returns:
        Series with mean metric drop per feature.
    """
    from sklearn.metrics import precision_score, roc_auc_score

    np.random.seed(random_state)

    # Get baseline metric
    y_pred = model.predict(X)

    if metric == "precision":
        y_pred_binary = (y_pred > 0.5).astype(int)
        baseline = precision_score(y, y_pred_binary, zero_division=0)
    elif metric == "auc":
        baseline = roc_auc_score(y, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    importance = {}

    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            y_pred_perm = model.predict(X_permuted)

            if metric == "precision":
                y_pred_binary = (y_pred_perm > 0.5).astype(int)
                score = precision_score(y, y_pred_binary, zero_division=0)
            else:
                score = roc_auc_score(y, y_pred_perm)

            drops.append(baseline - score)

        importance[col] = np.mean(drops)

    return pd.Series(importance)


def compute_stability_analysis(
    model_class,
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """Analyze feature importance stability across random seeds.

    Trains models with different seeds and measures variance in
    permutation importance.

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model initialization.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        n_seeds: Number of seeds to test.

    Returns:
        DataFrame with columns:
        - mean_importance: Mean permutation importance
        - std_importance: Std of importance across seeds
        - cv_importance: Coefficient of variation (std/mean)
    """
    all_importances = []

    for seed in range(n_seeds):
        # Set seed in model kwargs
        kwargs = model_kwargs.copy()
        if "random_state" in kwargs:
            kwargs["random_state"] = seed

        # Train model
        model = model_class(**kwargs)
        model.fit(X_train, y_train)

        # Get permutation importance
        imp = compute_permutation_importance(
            model, X_val, y_val, metric="precision", random_state=seed
        )
        all_importances.append(imp)

    # Combine into DataFrame
    imp_df = pd.DataFrame(all_importances)

    result = pd.DataFrame({
        "mean_importance": imp_df.mean(),
        "std_importance": imp_df.std(),
    })

    # CV = std / mean (handle mean=0)
    result["cv_importance"] = result["std_importance"] / result["mean_importance"].where(
        result["mean_importance"] != 0, 1.0
    )

    return result
