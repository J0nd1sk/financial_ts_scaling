#!/usr/bin/env python3
"""
Phase 6C Stage 2: Analyze exploration experiment results

Loads all results.json files and creates summary tables ranked by:
1. Precision (highest priority)
2. AUC
3. Recall

Usage: python experiments/phase6c/analyze_stage2.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def load_results(output_dir: Path) -> list[dict]:
    """Load all S2 results from output directory."""
    results = []

    for exp_dir in sorted(output_dir.glob("s2_*")):
        results_file = exp_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                data["exp_dir"] = exp_dir.name
                results.append(data)

    return results


def create_summary_table(results: list[dict]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    rows = []

    for r in results:
        vm = r.get("val_metrics", {})

        row = {
            "experiment": r.get("experiment", ""),
            "track": r.get("track", ""),
            "budget": r.get("budget", ""),
            "horizon": r.get("horizon", 1),
            "change": r.get("arch_change", r.get("train_change", "")),
            "auc": vm.get("auc"),
            "precision": vm.get("precision"),
            "recall": vm.get("recall"),
            "accuracy": vm.get("accuracy"),
            "pred_min": vm.get("pred_min"),
            "pred_max": vm.get("pred_max"),
            "pred_spread": vm.get("pred_max", 0) - vm.get("pred_min", 0) if vm else 0,
            "baseline_auc": r.get("baseline_auc"),
            "time_min": r.get("training", {}).get("training_time_min"),
        }

        # Calculate delta vs baseline
        if row["auc"] and row["baseline_auc"]:
            row["auc_delta_pct"] = (row["auc"] - row["baseline_auc"]) / row["baseline_auc"] * 100
        else:
            row["auc_delta_pct"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_track_summary(df: pd.DataFrame, track: str, title: str):
    """Print summary for a specific track."""
    track_df = df[df["track"] == track].copy()

    if len(track_df) == 0:
        print(f"\n{title}: No results found")
        return

    # Sort by precision (descending), then AUC
    track_df = track_df.sort_values(
        by=["precision", "auc"],
        ascending=[False, False]
    )

    print(f"\n{'='*80}")
    print(f"{title}")
    print("="*80)

    # Print table
    print(f"\n{'Experiment':<35} {'AUC':>6} {'Prec%':>6} {'Rec%':>6} {'Spread':>6} {'Δ%':>6}")
    print("-" * 80)

    for _, row in track_df.iterrows():
        auc_str = f"{row['auc']:.3f}" if row['auc'] else "N/A"
        prec_str = f"{row['precision']*100:.1f}" if row['precision'] else "N/A"
        rec_str = f"{row['recall']*100:.1f}" if row['recall'] else "N/A"
        spread_str = f"{row['pred_spread']:.2f}" if row['pred_spread'] else "N/A"
        delta_str = f"{row['auc_delta_pct']:+.1f}" if row['auc_delta_pct'] else "N/A"

        print(f"{row['experiment']:<35} {auc_str:>6} {prec_str:>6} {rec_str:>6} {spread_str:>6} {delta_str:>6}")


def main():
    print("=" * 80)
    print("PHASE 6C STAGE 2: EXPLORATION RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nAnalysis time: {datetime.now().isoformat()}")

    output_dir = PROJECT_ROOT / "outputs" / "phase6c"
    results = load_results(output_dir)

    if not results:
        print("\nNo Stage 2 results found in outputs/phase6c/s2_*/")
        print("Run experiments first: ./experiments/phase6c/run_stage2.sh")
        return

    print(f"\nFound {len(results)} experiment results")

    df = create_summary_table(results)

    # Stage 1 baselines for reference
    print("\n" + "=" * 80)
    print("STAGE 1 BASELINES (a50, H1)")
    print("=" * 80)
    print("\n  2M:   AUC=0.708, Precision=50.0%, Recall=6.6%")
    print("  20M:  AUC=0.722, Precision=42.3%, Recall=14.5%")
    print("  200M: AUC=0.699, Precision=42.4%, Recall=18.4% (REGRESSED)")
    print("\n  Phase 6A a20 200M: AUC=0.718 (target to match or beat)")

    # Print track summaries
    print_track_summary(df, "T1_horizon", "TRACK 1: HORIZON EXPERIMENTS (ranked by Precision)")
    print_track_summary(df, "T2_arch", "TRACK 2: ARCHITECTURE EXPERIMENTS (ranked by Precision)")
    print_track_summary(df, "T3_train", "TRACK 3: TRAINING PARAMETER EXPERIMENTS (ranked by Precision)")

    # Overall top performers
    print("\n" + "=" * 80)
    print("TOP 5 OVERALL (by Precision)")
    print("=" * 80)

    top5 = df.nlargest(5, "precision")
    print(f"\n{'#':>2} {'Experiment':<35} {'AUC':>6} {'Prec%':>6} {'Rec%':>6} {'Track':<12}")
    print("-" * 80)

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        auc_str = f"{row['auc']:.3f}" if row['auc'] else "N/A"
        prec_str = f"{row['precision']*100:.1f}" if row['precision'] else "N/A"
        rec_str = f"{row['recall']*100:.1f}" if row['recall'] else "N/A"
        print(f"{i:>2} {row['experiment']:<35} {auc_str:>6} {prec_str:>6} {rec_str:>6} {row['track']:<12}")

    # 200M regression analysis
    print("\n" + "=" * 80)
    print("200M REGRESSION ANALYSIS")
    print("=" * 80)

    df_200m = df[df["budget"] == "200M"].copy()
    if len(df_200m) > 0:
        # Find configs that beat 6A baseline (0.718)
        beat_6a = df_200m[df_200m["auc"] > 0.718]
        if len(beat_6a) > 0:
            print(f"\nConfigs that BEAT Phase 6A a20 baseline (0.718):")
            for _, row in beat_6a.iterrows():
                print(f"  - {row['experiment']}: AUC={row['auc']:.3f}, Precision={row['precision']*100:.1f}%")
        else:
            print("\nNo 200M configs beat Phase 6A a20 baseline (0.718)")

        # Best 200M config
        best_200m = df_200m.loc[df_200m["auc"].idxmax()]
        print(f"\nBest 200M config: {best_200m['experiment']}")
        print(f"  AUC: {best_200m['auc']:.3f} (Δ vs 6A: {(best_200m['auc']-0.718)/0.718*100:+.1f}%)")
        print(f"  Precision: {best_200m['precision']*100:.1f}%")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Horizon effect
    t1_results = df[df["track"] == "T1_horizon"]
    if len(t1_results) > 0:
        h2_mean = t1_results[t1_results["experiment"].str.contains("h2")]["auc"].mean()
        h5_mean = t1_results[t1_results["experiment"].str.contains("h5")]["auc"].mean()
        if h2_mean and h5_mean:
            print(f"\nHorizon effect: H2 avg AUC={h2_mean:.3f}, H5 avg AUC={h5_mean:.3f}")

    # Architecture effect for 200M
    t2_200m = df[(df["track"] == "T2_arch") & (df["budget"] == "200M")]
    if len(t2_200m) > 0:
        best_arch = t2_200m.loc[t2_200m["auc"].idxmax()]
        print(f"\nBest 200M architecture: {best_arch['change']}")
        print(f"  AUC={best_arch['auc']:.3f}, Precision={best_arch['precision']*100:.1f}%")

    # Training effect for 200M
    t3_200m = df[(df["track"] == "T3_train") & (df["budget"] == "200M")]
    if len(t3_200m) > 0:
        best_train = t3_200m.loc[t3_200m["auc"].idxmax()]
        print(f"\nBest 200M training config: {best_train['change']}")
        print(f"  AUC={best_train['auc']:.3f}, Precision={best_train['precision']*100:.1f}%")

    # Save to CSV
    csv_path = output_dir / "stage2_results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
