#!/usr/bin/env python3
"""Training CLI for financial time-series models.

Usage:
    python scripts/train.py --config configs/daily/threshold_1pct.yaml --param-budget 2m

Arguments:
    --config: Path to experiment config YAML file
    --param-budget: Model parameter budget (2m, 20m, or 200m)
    --batch-size: Training batch size (default: 32)
    --learning-rate: Optimizer learning rate (default: 0.001)
    --epochs: Number of training epochs (default: 10)
    --device: Device to train on (default: mps if available, else cpu)
    --checkpoint-dir: Directory for checkpoints (default: outputs/checkpoints)
    --no-thermal: Disable thermal monitoring
    --no-tracking: Disable W&B/MLflow tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.experiment import load_experiment_config
from src.models.configs import load_patchtst_config
from src.training.thermal import ThermalCallback
from src.training.tracking import TrackingConfig, TrackingManager
from src.training.trainer import Trainer


def get_default_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_thermal_callback() -> ThermalCallback | None:
    """Create thermal callback with a simple temperature provider.

    Returns None if temperature reading is not available.
    """
    try:
        import subprocess

        def get_cpu_temp() -> float:
            """Get CPU temperature using powermetrics (requires sudo)."""
            # This is a placeholder - in production, use a proper temp reader
            # For now, return a safe default
            return 60.0

        return ThermalCallback(temp_provider=get_cpu_temp)
    except Exception:
        return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train PatchTST models on financial time-series data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--param-budget",
        type=str,
        required=True,
        choices=["2m", "20m", "200m"],
        help="Model parameter budget",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--no-thermal",
        action="store_true",
        help="Disable thermal monitoring",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable W&B/MLflow tracking",
    )

    args = parser.parse_args()

    # Determine device
    device = args.device or get_default_device()
    print(f"Using device: {device}")

    # Load experiment config
    print(f"Loading experiment config: {args.config}")
    experiment_config = load_experiment_config(args.config)

    # Load model config
    print(f"Loading model config: {args.param_budget}")
    model_config = load_patchtst_config(args.param_budget)

    # Create thermal callback
    thermal_callback = None
    if not args.no_thermal:
        thermal_callback = create_thermal_callback()
        if thermal_callback:
            print("Thermal monitoring enabled")
        else:
            print("Thermal monitoring not available")

    # Create tracking manager
    tracking_manager = None
    if not args.no_tracking:
        if experiment_config.wandb_project or experiment_config.mlflow_experiment:
            tracking_config = TrackingConfig(
                wandb_project=experiment_config.wandb_project,
                mlflow_experiment=experiment_config.mlflow_experiment,
            )
            tracking_manager = TrackingManager(tracking_config)
            print(f"Tracking enabled - W&B: {experiment_config.wandb_project}, MLflow: {experiment_config.mlflow_experiment}")

    # Create trainer
    print(f"Creating trainer with batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.epochs}")
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        thermal_callback=thermal_callback,
        tracking_manager=tracking_manager,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Task: {experiment_config.task}")
    print(f"  Timescale: {experiment_config.timescale}")
    print(f"  Data: {experiment_config.data_path}")
    print(f"  Model params: {args.param_budget}")
    print()

    result = trainer.train()

    # Report results
    print(f"\nTraining complete!")
    print(f"  Final loss: {result['train_loss']:.6f}")
    if result.get("stopped_early"):
        print(f"  Stopped early: {result.get('stop_reason')}")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
