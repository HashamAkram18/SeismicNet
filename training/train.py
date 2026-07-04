"""Training entry point for SeismicNet.

Supports multi-phase training with MLflow tracking, mixed precision,
and early stopping. Defined in docs/04_training_and_export.md.

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

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from seismic.dataset import SeismicDataset, get_dataloader
from seismic.loss import SeismicLoss
from seismic.metrics import SeismicMetrics
from seismic.model.seismic_net import SeismicNet
from seismic.utils import load_checkpoint, load_config, save_checkpoint, validate_config_version

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train SeismicNet multi-task model")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment-name", type=str, default="seismic_risk")
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value from training_config.json.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """Create cosine schedule with linear warmup.

    Steps per gradient step, not per epoch.

    Args:
        optimizer: Optimizer instance.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr: Minimum learning rate at end of cosine decay.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: SeismicLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    active_tasks: list[str],
    gradient_clip_norm: float,
    epoch: int,
    mlflow_run: Any,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """Run one training epoch.

    Returns:
        Tuple of (mean_losses_dict, updated_global_step).
    """
    model.train()
    total_losses: dict[str, float] = {
        "detection": 0.0, "phase_pick": 0.0, "magnitude": 0.0, "risk": 0.0, "total": 0.0
    }
    n_batches = 0
    grad_norms: list[float] = []

    for batch in train_loader:
        optimizer.zero_grad()

        waveform = batch["waveform"].to(device)
        pgv = batch["pgv"].to(device)
        batch_on_device = {
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(waveform, pgv, active_tasks=active_tasks)
            losses = loss_fn(outputs, batch_on_device)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)

        # Compute grad norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        n_batches += 1
        global_step += 1

        # Log every 100 steps
        if global_step % 100 == 0 and mlflow_run is not None:
            mlflow.log_metrics({
                f"train/loss_{k}": v / n_batches
                for k, v in total_losses.items()
            }, step=global_step)
            mlflow.log_metric("train/grad_norm", np.mean(grad_norms[-100:]), step=global_step)
            mlflow.log_metric("train/lr", optimizer.param_groups[0]["lr"], step=global_step)

    # Average losses
    for key in total_losses:
        total_losses[key] /= max(n_batches, 1)

    # Log epoch-level train metrics
    if mlflow_run is not None:
        mlflow.log_metrics({f"train/loss_{k}": v for k, v in total_losses.items()}, step=global_step)
        mlflow.log_metric("train/grad_norm", float(np.mean(grad_norms)) if grad_norms else 0.0, step=global_step)

    return total_losses, global_step


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: SeismicLoss,
    device: torch.device,
    active_tasks: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run validation and compute metrics.

    Returns:
        Tuple of (loss_metrics, eval_metrics).
    """
    model.eval()
    metrics = SeismicMetrics()
    total_losses: dict[str, float] = {
        "detection": 0.0, "phase_pick": 0.0, "magnitude": 0.0, "risk": 0.0, "total": 0.0
    }
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            waveform = batch["waveform"].to(device)
            pgv = batch["pgv"].to(device)
            batch_on_device = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(waveform, pgv, active_tasks=active_tasks)
            losses = loss_fn(outputs, batch_on_device)

            for key in total_losses:
                total_losses[key] += losses[key].item()
            n_batches += 1

            metrics.update(outputs, batch_on_device)

    for key in total_losses:
        total_losses[key] /= max(n_batches, 1)

    eval_metrics = metrics.compute()
    return total_losses, eval_metrics


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Validate paths exist
    for path_attr in ["model_config", "training_config", "preprocessing_config"]:
        p = getattr(args, path_attr)
        if not p.exists():
            print(f"Error: {path_attr} not found: {p}", file=sys.stderr)
            sys.exit(1)

    # 1. Seed everything
    training_config = load_config(args.training_config)
    seed_everything(training_config["seed"])

    # 2. Load and validate all 3 configs
    model_config = load_config(args.model_config)
    preprocess_config = load_config(args.preprocessing_config)
    validate_config_version(model_config)
    validate_config_version(preprocess_config)
    validate_config_version(training_config)

    # Find phase config
    phase_config = None
    for pc in training_config["phases"]:
        if pc["phase_id"] == args.phase:
            phase_config = pc
            break
    if phase_config is None:
        print(f"Error: phase {args.phase} not found in training config", file=sys.stderr)
        sys.exit(1)

    # 3. Start MLflow run
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    run = mlflow.start_run(run_name=args.run_name)
    mlflow_run_id = run.info.run_id

    # Log configs as artifacts
    mlflow.log_artifact(str(args.model_config))
    mlflow.log_artifact(str(args.training_config))
    mlflow.log_artifact(str(args.preprocessing_config))
    mlflow.log_params({
        "phase_id": args.phase,
        "active_tasks": ",".join(phase_config["active_tasks"]),
        "seed": training_config["seed"],
    })

    # 4. Build DataLoaders
    hdf5_paths = sorted(str(p) for p in (args.data_dir / "raw").glob("*.hdf5"))
    metadata_csv = str(args.data_dir / "raw" / "metadata.csv")

    train_loader = get_dataloader(
        hdf5_paths=hdf5_paths,
        metadata_csv=metadata_csv,
        split_file=str(args.data_dir / "splits" / "train.txt"),
        preprocessing_config=preprocess_config,
        augment=True,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
    )
    val_loader = get_dataloader(
        hdf5_paths=hdf5_paths,
        metadata_csv=metadata_csv,
        split_file=str(args.data_dir / "splits" / "val.txt"),
        preprocessing_config=preprocess_config,
        augment=False,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
    )

    # 5. Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicNet(model_config).to(device)

    # 6. Resume if specified
    start_epoch = 0
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, preprocess_config)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        mlflow.log_param("resumed_from", str(args.resume))
        mlflow.log_param("resumed_epoch", start_epoch)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    # 7. Instantiate optimizer and scheduler (fresh per phase)
    opt_cfg = phase_config["optimizer"]
    scheduler_cfg = phase_config["scheduler"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # Estimate total steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * phase_config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=scheduler_cfg["warmup_steps"],
        total_steps=total_steps,
        min_lr=scheduler_cfg["min_lr"],
    )

    # 8. GradScaler + Loss
    scaler = GradScaler(enabled=training_config["mixed_precision"])
    loss_fn = SeismicLoss(training_config, phase_config)

    # 9. Training loop
    best_metric = 0.0
    patience = training_config["early_stopping_patience"]
    patience_counter = 0
    global_step = start_epoch * steps_per_epoch

    try:
        for epoch in range(start_epoch, phase_config["epochs"]):
            train_losses, global_step = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, scaler,
                device, phase_config["active_tasks"],
                training_config["gradient_clip_norm"], epoch,
                run, global_step,
            )

            val_losses, val_metrics = validate_epoch(
                model, val_loader, loss_fn, device, phase_config["active_tasks"]
            )

            # Log val metrics
            mlflow.log_metrics({f"val/loss_{k}": v for k, v in val_losses.items()}, step=global_step)
            mlflow.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

            # Save checkpoint if improved
            current_metric = val_metrics.get("detection_f1", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                ckpt_dir = args.output_dir / f"run_{mlflow_run_id}"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_metrics,
                    args.phase, mlflow_run_id,
                    {"model": model_config["schema_version"],
                     "preprocessing": preprocess_config["schema_version"],
                     "training": training_config["schema_version"]},
                    ckpt_dir,
                    keep_top_k=training_config["keep_top_k_checkpoints"],
                    metric_key=training_config["early_stopping_metric"],
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                mlflow.log_param("early_stop_epoch", epoch)
                break

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
