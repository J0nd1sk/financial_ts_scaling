"""Enhanced trial logging for HPO experiments.

Captures comprehensive metrics for each trial:
- Data split statistics
- Per-epoch learning curves (train_loss, val_loss)
- Final metrics breakdown (loss, accuracy, confusion matrix)
- Architecture and training parameter details

Also provides study-level summaries:
- All trials table with sortable metrics
- Architecture analysis (which configs work best)
- Training parameter sensitivity analysis
- Loss distribution across trials
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    """Statistics about train/val/test data splits."""

    n_train_samples: int
    n_val_samples: int
    n_test_samples: int = 0
    train_date_start: str = ""
    train_date_end: str = ""
    val_date_start: str = ""
    val_date_end: str = ""
    train_pos_ratio: float = 0.0  # Fraction of positive labels in train
    val_pos_ratio: float = 0.0  # Fraction of positive labels in val


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    learning_rate: float | None = None
    duration_seconds: float = 0.0


@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / max(total, 1)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / max(denom, 1)

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / max(denom, 1)

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-8)


@dataclass
class FinalMetrics:
    """Final metrics at end of training."""

    train_loss: float
    val_loss: float | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    train_confusion: ConfusionMatrix | None = None
    val_confusion: ConfusionMatrix | None = None


@dataclass
class TrialResult:
    """Complete result for a single HPO trial."""

    trial_number: int
    arch_idx: int
    architecture: dict[str, Any]
    training_params: dict[str, Any]
    split_stats: SplitStats | None = None
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    final_metrics: FinalMetrics | None = None
    duration_seconds: float = 0.0
    status: str = "completed"  # completed, failed, pruned
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "trial_number": self.trial_number,
            "arch_idx": self.arch_idx,
            "architecture": self.architecture,
            "training_params": self.training_params,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
        }

        if self.split_stats:
            result["split_stats"] = asdict(self.split_stats)

        if self.epoch_metrics:
            result["learning_curve"] = {
                "epochs": [e.epoch for e in self.epoch_metrics],
                "train_loss": [e.train_loss for e in self.epoch_metrics],
                "val_loss": [e.val_loss for e in self.epoch_metrics if e.val_loss is not None],
            }

        if self.final_metrics:
            result["final_metrics"] = {
                "train_loss": self.final_metrics.train_loss,
                "val_loss": self.final_metrics.val_loss,
                "train_accuracy": self.final_metrics.train_accuracy,
                "val_accuracy": self.final_metrics.val_accuracy,
            }
            if self.final_metrics.val_confusion:
                cm = self.final_metrics.val_confusion
                result["final_metrics"]["val_confusion_matrix"] = {
                    "tp": cm.true_positives,
                    "tn": cm.true_negatives,
                    "fp": cm.false_positives,
                    "fn": cm.false_negatives,
                    "accuracy": cm.accuracy,
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "f1": cm.f1_score,
                }

        if self.error_message:
            result["error_message"] = self.error_message

        return result


class TrialLogger:
    """Logger for capturing comprehensive trial metrics during HPO.

    Usage in HPO objective function:
        trial_logger = TrialLogger(output_dir)

        # Before training
        trial_logger.start_trial(trial.number, arch_idx, architecture, training_params)
        trial_logger.log_split_stats(split_stats)

        # During training (each epoch)
        trial_logger.log_epoch(epoch, train_loss, val_loss)

        # After training
        trial_logger.log_final_metrics(final_metrics)
        trial_logger.end_trial(status="completed")

        # After all trials
        trial_logger.generate_study_summary()
    """

    def __init__(self, output_dir: Path | str, experiment_name: str):
        """Initialize trial logger.

        Args:
            output_dir: Directory to save logs and summaries.
            experiment_name: Name of the experiment for file naming.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.trials: list[TrialResult] = []
        self.current_trial: TrialResult | None = None
        self._trial_start_time: float = 0.0

    def start_trial(
        self,
        trial_number: int,
        arch_idx: int,
        architecture: dict[str, Any],
        training_params: dict[str, Any],
    ) -> None:
        """Start logging a new trial."""
        import time

        self._trial_start_time = time.time()
        self.current_trial = TrialResult(
            trial_number=trial_number,
            arch_idx=arch_idx,
            architecture=architecture.copy(),
            training_params=training_params.copy(),
        )

    def log_split_stats(self, stats: SplitStats) -> None:
        """Log data split statistics."""
        if self.current_trial:
            self.current_trial.split_stats = stats

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        train_accuracy: float | None = None,
        val_accuracy: float | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics for an epoch."""
        if self.current_trial:
            self.current_trial.epoch_metrics.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_accuracy=train_accuracy,
                    val_accuracy=val_accuracy,
                    learning_rate=learning_rate,
                )
            )

    def log_final_metrics(self, metrics: FinalMetrics) -> None:
        """Log final training metrics."""
        if self.current_trial:
            self.current_trial.final_metrics = metrics

    def end_trial(self, status: str = "completed", error_message: str | None = None) -> None:
        """End the current trial and save results."""
        import time

        if self.current_trial:
            self.current_trial.duration_seconds = time.time() - self._trial_start_time
            self.current_trial.status = status
            self.current_trial.error_message = error_message
            self.trials.append(self.current_trial)

            # Save trial result to individual JSON
            trial_path = self.output_dir / f"trial_{self.current_trial.trial_number:03d}.json"
            with open(trial_path, "w") as f:
                json.dump(self.current_trial.to_dict(), f, indent=2)

            self.current_trial = None

    def generate_study_summary(
        self,
        study: Any = None,
        architectures: list[dict[str, Any]] | None = None,
        training_search_space: dict[str, Any] | None = None,
        split_indices: Any = None,
    ) -> dict[str, Path]:
        """Generate comprehensive study summary after all trials complete.

        Can work in two modes:
        1. If trials were logged via start_trial/end_trial, uses internal data
        2. If an Optuna study is provided, extracts data from trial.user_attrs

        Args:
            study: Optional Optuna study to extract trial data from.
            architectures: Optional list of architecture dicts (for Optuna mode).
            training_search_space: Optional search space config (for Optuna mode).
            split_indices: Optional split indices (for Optuna mode).

        Creates:
        - {experiment}_study_summary.json: Complete structured data
        - {experiment}_study_summary.md: Human-readable markdown report

        Returns:
            Dict with 'json' and 'markdown' paths.
        """
        # If Optuna study provided, extract trial data from user_attrs
        if study is not None:
            self._extract_from_optuna_study(study, architectures or [])

        if not self.trials:
            logger.warning("No trials to summarize")
            empty_md_path = self.output_dir / f"{self.experiment_name}_study_summary.md"
            empty_json_path = self.output_dir / f"{self.experiment_name}_study_summary.json"
            empty_md_path.write_text("# No trials completed\n")
            empty_json_path.write_text("{}")
            return {"json": empty_json_path, "markdown": empty_md_path}

        summary = self._build_summary()

        # Save JSON summary
        json_path = self.output_dir / f"{self.experiment_name}_study_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate markdown report
        md_path = self.output_dir / f"{self.experiment_name}_study_summary.md"
        md_content = self._generate_markdown_report(summary)
        with open(md_path, "w") as f:
            f.write(md_content)

        logger.info(f"Study summary saved to {md_path}")
        return {"json": json_path, "markdown": md_path}

    def _extract_from_optuna_study(
        self,
        study: Any,
        architectures: list[dict[str, Any]],
    ) -> None:
        """Extract trial data from Optuna study's user_attrs.

        This allows generating summaries from completed HPO runs where
        verbose=True was used during training.
        """
        import optuna

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Get architecture from user_attrs or from architectures list
            arch_idx = trial.params.get("arch_idx", 0)
            architecture = trial.user_attrs.get("architecture")
            if architecture is None and arch_idx < len(architectures):
                architecture = architectures[arch_idx]
            if architecture is None:
                architecture = {"d_model": 0, "n_layers": 0, "n_heads": 0, "d_ff": 0, "param_count": 0}

            # Get training params
            training_params = trial.user_attrs.get("training_params", {})
            if not training_params:
                # Extract from trial.params
                training_params = {
                    k: v for k, v in trial.params.items()
                    if k != "arch_idx"
                }

            # Create TrialResult
            trial_result = TrialResult(
                trial_number=trial.number,
                arch_idx=arch_idx,
                architecture=architecture,
                training_params=training_params,
                duration_seconds=trial.duration.total_seconds() if trial.duration else 0.0,
                status="completed",
            )

            # Extract split stats if available
            split_stats = trial.user_attrs.get("split_stats")
            if split_stats:
                trial_result.split_stats = SplitStats(
                    n_train_samples=split_stats.get("n_train", 0),
                    n_val_samples=split_stats.get("n_val", 0),
                    n_test_samples=split_stats.get("n_test", 0),
                )

            # Extract learning curve if available
            learning_curve = trial.user_attrs.get("learning_curve", [])
            for epoch_data in learning_curve:
                trial_result.epoch_metrics.append(
                    EpochMetrics(
                        epoch=epoch_data.get("epoch", 0),
                        train_loss=epoch_data.get("train_loss", 0.0),
                        val_loss=epoch_data.get("val_loss"),
                    )
                )

            # Build final metrics
            val_accuracy = trial.user_attrs.get("val_accuracy")
            train_accuracy = trial.user_attrs.get("train_accuracy")
            val_confusion = trial.user_attrs.get("val_confusion")

            val_cm = None
            if val_confusion:
                val_cm = ConfusionMatrix(
                    true_positives=val_confusion.get("tp", 0),
                    true_negatives=val_confusion.get("tn", 0),
                    false_positives=val_confusion.get("fp", 0),
                    false_negatives=val_confusion.get("fn", 0),
                )

            # Get train/val loss from learning curve or trial value
            train_loss = 0.0
            val_loss = trial.value
            if learning_curve:
                train_loss = learning_curve[-1].get("train_loss", 0.0)
                val_loss = learning_curve[-1].get("val_loss", trial.value)

            trial_result.final_metrics = FinalMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                val_confusion=val_cm,
            )

            self.trials.append(trial_result)

    def _build_summary(self) -> dict[str, Any]:
        """Build comprehensive summary dictionary."""
        completed_trials = [t for t in self.trials if t.status == "completed"]

        if not completed_trials:
            return {"error": "No completed trials"}

        # Sort by val_loss
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t.final_metrics.val_loss if t.final_metrics and t.final_metrics.val_loss else float("inf"),
        )

        best_trial = sorted_trials[0]

        # All trials table
        all_trials_table = []
        for t in sorted_trials:
            row = {
                "trial": t.trial_number,
                "arch_idx": t.arch_idx,
                "d_model": t.architecture.get("d_model"),
                "n_layers": t.architecture.get("n_layers"),
                "n_heads": t.architecture.get("n_heads"),
                "param_count": t.architecture.get("param_count"),
                "lr": t.training_params.get("learning_rate"),
                "epochs": t.training_params.get("epochs"),
                "batch_size": t.training_params.get("batch_size"),
                "val_loss": t.final_metrics.val_loss if t.final_metrics else None,
                "val_accuracy": t.final_metrics.val_accuracy if t.final_metrics else None,
                "duration_s": round(t.duration_seconds, 1),
            }
            all_trials_table.append(row)

        # Architecture analysis
        arch_stats = self._analyze_architectures(completed_trials)

        # Training param sensitivity
        param_sensitivity = self._analyze_param_sensitivity(completed_trials)

        # Loss distribution
        val_losses = [
            t.final_metrics.val_loss
            for t in completed_trials
            if t.final_metrics and t.final_metrics.val_loss is not None
        ]
        loss_distribution = {
            "min": min(val_losses) if val_losses else None,
            "max": max(val_losses) if val_losses else None,
            "mean": float(np.mean(val_losses)) if val_losses else None,
            "std": float(np.std(val_losses)) if val_losses else None,
            "median": float(np.median(val_losses)) if val_losses else None,
            "quartiles": {
                "q25": float(np.percentile(val_losses, 25)) if val_losses else None,
                "q50": float(np.percentile(val_losses, 50)) if val_losses else None,
                "q75": float(np.percentile(val_losses, 75)) if val_losses else None,
            },
        }

        return {
            "experiment": self.experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_trials_total": len(self.trials),
            "n_trials_completed": len(completed_trials),
            "n_trials_failed": len([t for t in self.trials if t.status == "failed"]),
            "best_trial": best_trial.to_dict(),
            "all_trials_table": all_trials_table,
            "architecture_analysis": arch_stats,
            "param_sensitivity": param_sensitivity,
            "loss_distribution": loss_distribution,
        }

    def _analyze_architectures(self, trials: list[TrialResult]) -> dict[str, Any]:
        """Analyze which architecture configurations perform best."""
        # Group by d_model
        by_d_model: dict[int, list[float]] = defaultdict(list)
        by_n_layers: dict[int, list[float]] = defaultdict(list)
        by_config: dict[str, list[float]] = defaultdict(list)

        for t in trials:
            if t.final_metrics and t.final_metrics.val_loss is not None:
                d_model = t.architecture.get("d_model", 0)
                n_layers = t.architecture.get("n_layers", 0)
                val_loss = t.final_metrics.val_loss

                by_d_model[d_model].append(val_loss)
                by_n_layers[n_layers].append(val_loss)
                by_config[f"d{d_model}_L{n_layers}"].append(val_loss)

        def summarize(groups: dict) -> list[dict]:
            result = []
            for key, losses in sorted(groups.items()):
                result.append({
                    "value": key,
                    "n_trials": len(losses),
                    "mean_val_loss": round(float(np.mean(losses)), 4),
                    "min_val_loss": round(min(losses), 4),
                    "std_val_loss": round(float(np.std(losses)), 4),
                })
            return sorted(result, key=lambda x: x["mean_val_loss"])

        return {
            "by_d_model": summarize(by_d_model),
            "by_n_layers": summarize(by_n_layers),
            "by_config": summarize(by_config)[:10],  # Top 10 configs
        }

    def _analyze_param_sensitivity(self, trials: list[TrialResult]) -> dict[str, Any]:
        """Analyze sensitivity of val_loss to training parameters."""
        params_to_analyze = ["learning_rate", "epochs", "batch_size", "weight_decay"]
        sensitivity = {}

        for param in params_to_analyze:
            values = []
            losses = []
            for t in trials:
                if t.final_metrics and t.final_metrics.val_loss is not None:
                    val = t.training_params.get(param)
                    if val is not None:
                        values.append(float(val))
                        losses.append(t.final_metrics.val_loss)

            if len(values) >= 5:
                # Compute correlation
                correlation = float(np.corrcoef(values, losses)[0, 1])
                sensitivity[param] = {
                    "correlation_with_val_loss": round(correlation, 3),
                    "interpretation": (
                        "lower is better" if correlation > 0.1 else
                        "higher is better" if correlation < -0.1 else
                        "weak effect"
                    ),
                }

        return sensitivity

    def _generate_markdown_report(self, summary: dict[str, Any]) -> str:
        """Generate human-readable markdown report."""
        lines = [
            f"# HPO Study Summary: {summary['experiment']}",
            "",
            f"**Generated:** {summary['timestamp']}",
            "",
            "## Overview",
            "",
            f"- **Total Trials:** {summary['n_trials_total']}",
            f"- **Completed:** {summary['n_trials_completed']}",
            f"- **Failed:** {summary['n_trials_failed']}",
            "",
        ]

        # Best trial
        best = summary.get("best_trial", {})
        if best:
            lines.extend([
                "## Best Trial",
                "",
                f"- **Trial:** {best.get('trial_number')}",
                f"- **Architecture:** d_model={best.get('architecture', {}).get('d_model')}, "
                f"n_layers={best.get('architecture', {}).get('n_layers')}, "
                f"params={best.get('architecture', {}).get('param_count'):,}",
                f"- **Val Loss:** {best.get('final_metrics', {}).get('val_loss'):.4f}",
                "",
            ])

            # Confusion matrix if available
            cm = best.get("final_metrics", {}).get("val_confusion_matrix")
            if cm:
                lines.extend([
                    "### Validation Confusion Matrix",
                    "",
                    f"| | Predicted + | Predicted - |",
                    f"|---|---|---|",
                    f"| **Actual +** | {cm['tp']} | {cm['fn']} |",
                    f"| **Actual -** | {cm['fp']} | {cm['tn']} |",
                    "",
                    f"- **Accuracy:** {cm['accuracy']:.3f}",
                    f"- **Precision:** {cm['precision']:.3f}",
                    f"- **Recall:** {cm['recall']:.3f}",
                    f"- **F1 Score:** {cm['f1']:.3f}",
                    "",
                ])

        # Loss distribution
        dist = summary.get("loss_distribution", {})
        if dist.get("mean"):
            lines.extend([
                "## Loss Distribution",
                "",
                f"- **Min:** {dist['min']:.4f}",
                f"- **Max:** {dist['max']:.4f}",
                f"- **Mean:** {dist['mean']:.4f} ± {dist['std']:.4f}",
                f"- **Median:** {dist['median']:.4f}",
                f"- **IQR:** [{dist['quartiles']['q25']:.4f}, {dist['quartiles']['q75']:.4f}]",
                "",
            ])

        # Architecture analysis
        arch = summary.get("architecture_analysis", {})
        if arch.get("by_d_model"):
            lines.extend([
                "## Architecture Analysis",
                "",
                "### By d_model (sorted by mean val_loss)",
                "",
                "| d_model | n_trials | mean_val_loss | min_val_loss |",
                "|---------|----------|---------------|--------------|",
            ])
            for row in arch["by_d_model"]:
                lines.append(
                    f"| {row['value']} | {row['n_trials']} | {row['mean_val_loss']:.4f} | {row['min_val_loss']:.4f} |"
                )
            lines.append("")

        if arch.get("by_n_layers"):
            lines.extend([
                "### By n_layers (sorted by mean val_loss)",
                "",
                "| n_layers | n_trials | mean_val_loss | min_val_loss |",
                "|----------|----------|---------------|--------------|",
            ])
            for row in arch["by_n_layers"]:
                lines.append(
                    f"| {row['value']} | {row['n_trials']} | {row['mean_val_loss']:.4f} | {row['min_val_loss']:.4f} |"
                )
            lines.append("")

        # Param sensitivity
        sens = summary.get("param_sensitivity", {})
        if sens:
            lines.extend([
                "## Training Parameter Sensitivity",
                "",
                "| Parameter | Correlation | Interpretation |",
                "|-----------|-------------|----------------|",
            ])
            for param, data in sens.items():
                lines.append(
                    f"| {param} | {data['correlation_with_val_loss']:.3f} | {data['interpretation']} |"
                )
            lines.append("")

        # All trials table (top 20)
        trials_table = summary.get("all_trials_table", [])
        if trials_table:
            lines.extend([
                "## All Trials (sorted by val_loss)",
                "",
                "| trial | d_model | n_layers | params | lr | epochs | batch | val_loss | duration |",
                "|-------|---------|----------|--------|-------|--------|-------|----------|----------|",
            ])
            for row in trials_table[:20]:  # Top 20
                lines.append(
                    f"| {row['trial']} | {row['d_model']} | {row['n_layers']} | "
                    f"{row['param_count']:,} | {row['lr']:.1e} | {row['epochs']} | "
                    f"{row['batch_size']} | {row['val_loss']:.4f} | {row['duration_s']:.0f}s |"
                )
            if len(trials_table) > 20:
                lines.append(f"*... and {len(trials_table) - 20} more trials*")
            lines.append("")

        return "\n".join(lines)


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> ConfusionMatrix:
    """Compute confusion matrix from predictions and targets.

    Args:
        predictions: Model output probabilities (0-1).
        targets: Ground truth binary labels.
        threshold: Classification threshold.

    Returns:
        ConfusionMatrix with counts.
    """
    pred_binary = (predictions >= threshold).astype(int)
    targets_binary = targets.astype(int)

    tp = int(np.sum((pred_binary == 1) & (targets_binary == 1)))
    tn = int(np.sum((pred_binary == 0) & (targets_binary == 0)))
    fp = int(np.sum((pred_binary == 1) & (targets_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (targets_binary == 1)))

    return ConfusionMatrix(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )
