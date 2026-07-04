"""Training entry point for SeismicNet.

Supports multi-phase training with MLflow tracking, mixed precision,
and early stopping. Defined in docs/04_training_and_export.md § train.py.

CLI:
    python training/train.py \
        --model-config configs/model_config.json \
        --training-config configs/training_config.json \
        --preprocessing-config configs/preprocessing_config.json \
        --data-dir data/ \
        --output-dir checkpoints/ \
        --mlflow-uri http://localhost:5000 \
        --experiment-name seismic_risk \
        --run-name phase1_baseline \
        --phase 1 \
        [--resume checkpoints/run_abc/epoch_12.pt]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader


def parse_args() -> Any:
    """Parse command-line arguments for training.

    Returns:
        Parsed arguments namespace.
    """
    # TODO: implement argparse
    raise NotImplementedError


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets torch.manual_seed, numpy.random.seed, random.seed.

    Args:
        seed: Random seed value from training_config.json.
    """
    # TODO: implement


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    active_tasks: list[str],
    gradient_clip_norm: float,
    epoch: int,
    mlflow_run: Any,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model: SeismicNet model.
        train_loader: Training DataLoader.
        loss_fn: SeismicLoss instance.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        scaler: GradScaler for mixed precision.
        device: torch device (cuda).
        active_tasks: List of active task names.
        gradient_clip_norm: Max gradient norm for clipping.
        epoch: Current epoch number.
        mlflow_run: Active MLflow run (or None).

    Returns:
        Dict of mean training losses for the epoch.
    """
    # TODO: implement
    raise NotImplementedError


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    active_tasks: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Run validation and compute metrics.

    Args:
        model: SeismicNet model.
        val_loader: Validation DataLoader.
        loss_fn: SeismicLoss instance.
        device: torch device.
        active_tasks: List of active task names.

    Returns:
        Tuple of (loss_metrics, eval_metrics).
    """
    # TODO: implement
    raise NotImplementedError


def main() -> None:
    """Main training entry point."""
    # TODO: implement full training loop
    raise NotImplementedError


if __name__ == "__main__":
    main()
