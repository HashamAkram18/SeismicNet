"""Utility functions: checkpoint save/load, config loaders.

Every torch.save() for a checkpoint must also write preprocessing_config.json
into the same checkpoint directory (Invariant 2).
Defined in docs/04_training_and_export.md § utils.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    val_metrics: dict[str, float],
    phase_id: int,
    mlflow_run_id: str,
    config_versions: dict[str, str],
    checkpoint_dir: Path,
    keep_top_k: int = 3,
    metric_key: str = "detection_f1",
) -> Path:
    """Save model checkpoint and preprocessing config.

    Args:
        model: Model with state_dict.
        optimizer: Optimizer with state_dict.
        scheduler: LR scheduler with state_dict.
        scaler: GradScaler with state_dict (or None).
        epoch: Current epoch number.
        val_metrics: Validation metrics dict.
        phase_id: Current training phase ID.
        mlflow_run_id: MLflow run ID.
        config_versions: Dict with model/preprocessing/training config versions.
        checkpoint_dir: Directory to save checkpoint.
        keep_top_k: Number of top checkpoints to keep.
        metric_key: Metric key for ranking checkpoints.

    Returns:
        Path to the saved checkpoint.
    """
    # TODO: implement
    raise NotImplementedError


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Load checkpoint and validate preprocessing config version.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: Model to load weights into.
        preprocessing_config: Current preprocessing config for version validation.

    Returns:
        Dict with optimizer_state_dict, scheduler_state_dict, epoch, val_metrics.

    Raises:
        ValueError: If preprocessing_config_version in checkpoint does not match config.
    """
    # TODO: implement
    raise NotImplementedError


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a JSON config file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If JSON is malformed or missing required fields.
    """
    # TODO: implement
    raise NotImplementedError


def validate_config_version(config: dict[str, Any], expected_version: str = "1.0.0") -> None:
    """Validate that config schema_version matches expected value.

    Args:
        config: Configuration dictionary.
        expected_version: Expected schema version string.

    Raises:
        ValueError: If schema_version does not match.
    """
    # TODO: implement
