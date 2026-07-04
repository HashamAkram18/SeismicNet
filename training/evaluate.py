"""Standalone evaluation on test split.

Loads a checkpoint and runs full evaluation, producing per-sample predictions
and confusion matrices. Defined in docs/04_training_and_export.md § evaluate.py.

CLI:
    python training/evaluate.py \
        --checkpoint checkpoints/run_abc/epoch_045.pt \
        --preprocessing-config configs/preprocessing_config.json \
        --data-dir data/ \
        --output-dir eval_results/ \
        --mlflow-uri http://localhost:5000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def parse_args() -> Any:
    """Parse command-line arguments for evaluation.

    Returns:
        Parsed arguments namespace.
    """
    # TODO: implement argparse
    raise NotImplementedError


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    active_tasks: list[str],
) -> dict[str, Any]:
    """Run full evaluation on test split.

    Args:
        model: SeismicNet model in eval mode.
        test_loader: Test DataLoader.
        device: torch device.
        active_tasks: List of active task names.

    Returns:
        Dict with metrics, per-sample predictions, and confusion matrix.
    """
    # TODO: implement
    raise NotImplementedError


def save_predictions_csv(predictions: list[dict], output_path: Path) -> None:
    """Save per-sample predictions to CSV.

    Args:
        predictions: List of prediction dicts.
        output_path: Path to output CSV file.
    """
    # TODO: implement


def save_confusion_matrix(confusion_matrix: list[list[int]], output_path: Path) -> None:
    """Save confusion matrix to JSON.

    Args:
        confusion_matrix: 4x4 confusion matrix as nested list.
        output_path: Path to output JSON file.
    """
    # TODO: implement


def main() -> None:
    """Main evaluation entry point."""
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
