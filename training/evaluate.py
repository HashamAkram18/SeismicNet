"""Standalone evaluation on test split.

Loads a checkpoint and runs full evaluation, producing per-sample predictions
and confusion matrices. Defined in docs/04_training_and_export.md.

CLI:
    python training/evaluate.py \
        --checkpoint checkpoints/run_abc/epoch_045.pt \
        --preprocessing-config configs/preprocessing_config.json \
        --data-dir data/ \
        --output-dir eval_results/ \
        --mlflow-uri http://localhost:5000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from seismic.dataset import get_dataloader
from seismic.metrics import SeismicMetrics
from seismic.model.seismic_net import SeismicNet
from seismic.utils import load_checkpoint, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SeismicNet on test split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    active_tasks: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run full evaluation on test split.

    Args:
        model: SeismicNet model in eval mode.
        test_loader: Test DataLoader.
        device: torch device.
        active_tasks: List of active task names.

    Returns:
        Tuple of (metrics_dict, per_sample_predictions_list).
    """
    model.eval()
    metrics = SeismicMetrics()
    all_predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            waveform = batch["waveform"].to(device)
            pgv = batch["pgv"].to(device)
            batch_device = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(waveform, pgv, active_tasks=active_tasks)
            metrics.update(outputs, batch_device)

            # Collect per-sample predictions
            batch_size = waveform.shape[0]
            for i in range(batch_size):
                pred: dict[str, Any] = {
                    "event_id": batch["event_id"][i] if "event_id" in batch else f"sample_{len(all_predictions)}",
                }

                if "detection" in outputs:
                    det_prob = torch.sigmoid(outputs["detection"][i]).item()
                    pred["pred_detection"] = det_prob
                    pred["label_detection"] = batch["label_detection"][i].item()

                if "phase_pick" in outputs:
                    p_pick = outputs["phase_pick"][i, 0].item() * 3000
                    s_pick = outputs["phase_pick"][i, 1].item() * 3000
                    pred["pred_p_pick_sample"] = p_pick
                    pred["pred_s_pick_sample"] = s_pick
                    pred["label_p_pick_sample"] = batch["label_phase_pick"][i, 0].item() * 3000
                    pred["label_s_pick_sample"] = batch["label_phase_pick"][i, 1].item() * 3000

                if "magnitude" in outputs:
                    mag_log = outputs["magnitude"][i].item()
                    pred["pred_magnitude"] = 10 ** mag_log - 1
                    label_mag = batch["label_magnitude"][i].item()
                    pred["label_magnitude"] = 10 ** label_mag - 1 if not torch.isnan(torch.tensor(label_mag)) else float("nan")

                if "risk" in outputs:
                    risk_probs = torch.softmax(outputs["risk"][i], dim=-1)
                    pred["pred_risk_class"] = risk_probs.argmax().item()
                    pred["pred_risk_probs"] = risk_probs.tolist()
                    pred["label_risk"] = batch["label_risk"][i].item()

                all_predictions.append(pred)

    eval_metrics = metrics.compute()
    return eval_metrics, all_predictions


def save_predictions_csv(predictions: list[dict], output_path: Path) -> None:
    """Save per-sample predictions to CSV."""
    if not predictions:
        return

    fieldnames = sorted(predictions[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not args.preprocessing_config.exists():
        print(f"Error: preprocessing config not found: {args.preprocessing_config}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    preprocess_config = load_config(args.preprocessing_config)

    # Load model config from checkpoint or argument
    if args.model_config and args.model_config.exists():
        model_config = load_config(args.model_config)
    else:
        model_config = load_config(Path("configs/model_config.json"))

    # Load model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicNet(model_config).to(device)
    ckpt_data = load_checkpoint(args.checkpoint, model, preprocess_config)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    active_tasks = ["detection", "phase_pick", "magnitude", "risk"]

    # Build test DataLoader
    data_dir = args.data_dir
    hdf5_paths = sorted(str(p) for p in (data_dir / "raw").glob("*.hdf5"))
    metadata_csv = str(data_dir / "raw" / "metadata.csv")
    split_file = data_dir / "splits" / "test.txt"

    test_loader = get_dataloader(
        hdf5_paths=hdf5_paths,
        metadata_csv=metadata_csv,
        split_file=str(split_file),
        preprocessing_config=preprocess_config,
        augment=False,
        batch_size=256,
        num_workers=4,
        pin_memory=True,
    )

    # Run evaluation
    eval_metrics, predictions = evaluate(model, test_loader, device, active_tasks)

    # Save predictions CSV
    csv_path = args.output_dir / "predictions.csv"
    save_predictions_csv(predictions, csv_path)
    logger.info("Saved predictions to %s", csv_path)

    # Save confusion matrix
    cm = eval_metrics.get("risk_confusion_matrix", [[0] * 4 for _ in range(4)])
    cm_path = args.output_dir / "confusion_matrix.json"
    cm_path.write_text(json.dumps(cm, indent=2))
    logger.info("Saved confusion matrix to %s", cm_path)

    # Log to MLflow if URI provided
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        run_id = ckpt_data.get("mlflow_run_id")
        if run_id:
            mlflow.log_metrics(
                {f"test/{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))},
                run_id=run_id,
            )
            mlflow.log_artifact(str(csv_path), run_id=run_id)
            mlflow.log_artifact(str(cm_path), run_id=run_id)
            logger.info("Logged metrics to MLflow run %s", run_id)

    # Print summary
    print("\n=== Test Metrics ===")
    for k, v in sorted(eval_metrics.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print(f"\nPredictions saved to: {csv_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
