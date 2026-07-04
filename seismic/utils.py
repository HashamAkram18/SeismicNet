"""Utility functions: checkpoint save/load, config loaders.

Every torch.save() for a checkpoint must also write preprocessing_config.json
into the same checkpoint directory (Invariant 2).

load_checkpoint() must validate that preprocessing_config_version in the
checkpoint matches the current preprocessing_config.json. Raise hard error
on mismatch.

Defined in docs/04_training_and_export.md § utils.
"""
from __future__ import annotations

import json
import shutil
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

    Saves to checkpoints/run_{mlflow_run_id}/epoch_{epoch:03d}.pt.
    Copies preprocessing_config.json into the same directory.
    Keeps only the top-K checkpoints by metric_key.

    Args:
        model: Model with state_dict.
        optimizer: Optimizer with state_dict.
        scheduler: LR scheduler with state_dict (or None).
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "val_metrics": val_metrics,
        "model_config_version": config_versions["model"],
        "preprocessing_config_version": config_versions["preprocessing"],
        "training_config_version": config_versions["training"],
        "mlflow_run_id": mlflow_run_id,
        "phase_id": phase_id,
    }

    ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(ckpt, ckpt_path)

    # Copy preprocessing_config.json alongside checkpoint (Invariant 2)
    src_config = Path("configs/preprocessing_config.json")
    dst_config = checkpoint_dir / "preprocessing_config.json"
    if src_config.exists():
        shutil.copy2(src_config, dst_config)

    # Keep only top-K checkpoints
    _prune_checkpoints(checkpoint_dir, keep_top_k, metric_key)

    return ckpt_path


def _prune_checkpoints(checkpoint_dir: Path, keep_top_k: int, metric_key: str) -> None:
    """Remove old checkpoints, keeping only top-K by metric_key."""
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_top_k:
        return

    # Load metrics for each checkpoint
    scored = []
    for p in ckpts:
        try:
            data = torch.load(p, map_location="cpu", weights_only=False)
            score = data.get("val_metrics", {}).get(metric_key, 0.0)
            scored.append((p, score))
        except Exception:
            scored.append((p, 0.0))

    # Sort descending by score, keep top-K
    scored.sort(key=lambda x: x[1], reverse=True)
    for p, _ in scored[keep_top_k:]:
        p.unlink(missing_ok=True)


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
        Dict with keys: model_state_dict, optimizer_state_dict, scheduler_state_dict,
        scaler_state_dict, epoch, val_metrics, phase_id, mlflow_run_id.

    Raises:
        ValueError: If preprocessing_config_version in checkpoint does not match config.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate preprocessing config version (Invariant 2)
    ckpt_version = ckpt.get("preprocessing_config_version")
    current_version = preprocessing_config.get("schema_version")
    if ckpt_version != current_version:
        raise ValueError(
            f"Preprocessing config version mismatch: checkpoint has "
            f"'{ckpt_version}', current config has '{current_version}'"
        )

    return {
        "model_state_dict": ckpt["model_state_dict"],
        "optimizer_state_dict": ckpt["optimizer_state_dict"],
        "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
        "scaler_state_dict": ckpt.get("scaler_state_dict"),
        "epoch": ckpt["epoch"],
        "val_metrics": ckpt.get("val_metrics", {}),
        "phase_id": ckpt.get("phase_id"),
        "mlflow_run_id": ckpt.get("mlflow_run_id"),
    }


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a JSON config file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If JSON is malformed or missing schema_version field.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    if "schema_version" not in config:
        raise ValueError(
            f"Config file {config_path.name} is missing required 'schema_version' field"
        )

    return config


def validate_config_version(config: dict[str, Any], expected_version: str = "1.0.0") -> None:
    """Validate that config schema_version matches expected value.

    Args:
        config: Configuration dictionary.
        expected_version: Expected schema version string.

    Raises:
        ValueError: If schema_version does not match.
    """
    actual = config.get("schema_version")
    if actual != expected_version:
        raise ValueError(
            f"Config schema_version mismatch: expected '{expected_version}', got '{actual}'"
        )
