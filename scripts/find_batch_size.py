#!/usr/bin/env python3
"""Batch size discovery CLI for finding optimal training batch size.

Discovers the largest viable batch size for a given parameter budget
by attempting forward+backward passes with progressively larger batch
sizes until memory is exhausted.

Usage:
    python scripts/find_batch_size.py --config configs/daily/threshold_1pct.yaml --param-budget 2m

Arguments:
    --config: Path to experiment config YAML file
    --param-budget: Model parameter budget (2m, 20m, or 200m)
    --device: Device to test on (default: mps if available, else cpu)
    --min-batch: Minimum batch size to try (default: 8)
    --max-batch: Maximum batch size to try (default: 512)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_batch_size(
    try_batch_fn: Callable[[int], bool],
    min_batch: int = 8,
    max_batch: int = 512,
) -> int:
    """Find largest viable batch size via exponential search.

    Algorithm:
    1. Start with min_batch
    2. Try forward + backward pass via try_batch_fn
    3. If success, double batch_size
    4. If RuntimeError (OOM), return previous successful value

    Args:
        try_batch_fn: Callable that takes batch_size and returns True on success.
                      Should raise RuntimeError on OOM.
        min_batch: Minimum batch size to start with (default: 8)
        max_batch: Maximum batch size to try (default: 512)

    Returns:
        Largest batch size that succeeded, or min_batch if all fail.

    Raises:
        Any non-RuntimeError exception from try_batch_fn.
    """
    batch_size = min_batch
    last_successful = min_batch  # Default to min if even first attempt fails

    while batch_size <= max_batch:
        try:
            try_batch_fn(batch_size)
            last_successful = batch_size
            batch_size *= 2
        except RuntimeError:
            # OOM - stop and return last successful
            break

    return last_successful


def create_try_batch_fn(
    model_config,
    experiment_config,
    device: str,
) -> Callable[[int], bool]:
    """Create a try_batch function for real batch size testing.

    Args:
        model_config: PatchTST model configuration
        experiment_config: Experiment configuration
        device: Device to test on

    Returns:
        Function that takes batch_size and returns True on success.
    """
    from src.models.patchtst import PatchTST

    def try_batch(batch_size: int) -> bool:
        """Attempt forward+backward pass with given batch size."""
        # Create model
        model = PatchTST(model_config).to(device)

        # Create synthetic input matching expected shape
        # Shape: (batch_size, context_length, num_features)
        x = torch.randn(
            batch_size,
            model_config.context_length,
            model_config.num_features,
            device=device,
        )

        # Create synthetic target
        # Shape: (batch_size, 1)
        y = torch.randint(0, 2, (batch_size, 1), device=device).float()

        # Forward pass
        output = model(x)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy(output, y)

        # Backward pass
        loss.backward()

        # Clear memory for next attempt
        del model, x, y, output, loss
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        return True

    return try_batch


def get_default_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find optimal batch size for PatchTST training.",
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
        "--device",
        type=str,
        default=None,
        help="Device to test on (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=8,
        help="Minimum batch size to try",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=512,
        help="Maximum batch size to try",
    )

    args = parser.parse_args()

    # Import here to avoid import errors when running tests
    from src.config.experiment import load_experiment_config
    from src.models.configs import load_patchtst_config

    # Determine device
    device = args.device or get_default_device()
    print(f"Using device: {device}")

    # Load configs
    print(f"Loading experiment config: {args.config}")
    experiment_config = load_experiment_config(args.config)

    print(f"Loading model config: {args.param_budget}")
    model_config = load_patchtst_config(args.param_budget)

    # Create try_batch function
    try_batch_fn = create_try_batch_fn(model_config, experiment_config, device)

    # Find optimal batch size
    print(f"\nFinding optimal batch size for {args.param_budget} model...")
    print(f"  Search range: {args.min_batch} to {args.max_batch}")
    print()

    # Track progress
    batch_size = args.min_batch
    while batch_size <= args.max_batch:
        try:
            print(f"  Trying batch_size={batch_size}...", end=" ", flush=True)
            try_batch_fn(batch_size)
            print("OK")
            batch_size *= 2
        except RuntimeError as e:
            print(f"FAIL ({e})")
            break

    # Get result using the algorithm
    result = find_batch_size(
        try_batch_fn=try_batch_fn,
        min_batch=args.min_batch,
        max_batch=args.max_batch,
    )

    print(f"\nRecommended batch size: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
