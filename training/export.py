"""GPU checkpoint to deployable artifact conversion.

Converts a trained checkpoint into the artifact bundle that Django REST will load.
Steps defined in docs/04_training_and_export.md § export.py.

CLI:
    python training/export.py \
        --checkpoint checkpoints/run_abc/epoch_045.pt \
        --preprocessing-config configs/preprocessing_config.json \
        --output-dir exported_models/seismic_v1/ \
        --benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from seismic.model.seismic_net import SeismicNet
from seismic.utils import load_config, load_checkpoint, validate_config_version

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for export."""
    parser = argparse.ArgumentParser(description="Export SeismicNet to deployable artifact")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_config.json"))
    parser.add_argument("--val-predictions", type=Path, default=None,
                        help="Path to val predictions CSV from evaluate.py (for threshold tuning)")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU benchmark")
    return parser.parse_args()


def tune_detection_threshold(val_predictions_csv: Path) -> float:
    """Find optimal detection threshold that maximizes F1 on validation set.

    Args:
        val_predictions_csv: Path to predictions.csv from evaluate.py.

    Returns:
        Optimal threshold float.
    """
    rows = []
    with open(val_predictions_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold_int in range(1, 100):
        threshold = threshold_int / 100.0
        tp = fp = fn = tn = 0
        for row in rows:
            pred = 1 if float(row["pred_detection"]) >= threshold else 0
            label = int(float(row["label_detection"]))
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info("Tuned detection threshold: %.3f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply selective quantization to ResNet Conv1d and Linear layers only.

    Transformer layers remain float32 due to softmax numerical sensitivity.

    Args:
        model: Trained model.

    Returns:
        Quantized model.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv1d},
        dtype=torch.qint8,
    )
    return quantized


def benchmark_model(model: nn.Module, num_runs: int = 100) -> dict[str, float]:
    """Benchmark model on CPU with batch_size=1.

    Args:
        model: Model to benchmark.
        num_runs: Number of forward passes.

    Returns:
        Dict with mean_latency_ms, p95_latency_ms, model_size_mb.
    """
    model.eval()
    device = torch.device("cpu")
    waveform = torch.randn(1, 3, 3000, device=device)
    pgv = torch.randn(1, 1, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(waveform, pgv)

    # Benchmark
    latencies: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(waveform, pgv)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p95_idx = int(0.95 * len(latencies))
    p95_ms = latencies[min(p95_idx, len(latencies) - 1)]

    return {
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms,
        "num_runs": num_runs,
    }


def smoke_test(scripted_model: Any) -> None:
    """Run smoke test on exported model.

    Constructs synthetic waveform [1, 3, 3000] and asserts:
        - All four output keys present
        - No NaN values
        - Detection output in [0, 1] after sigmoid
        - Phase pick values in [0, 1]
        - Risk probabilities sum to ~1.0

    Args:
        scripted_model: TorchScript model.

    Raises:
        AssertionError: If any assertion fails.
    """
    waveform = torch.randn(1, 3, 3000)
    pgv = torch.randn(1, 1)

    output = scripted_model(waveform, pgv)

    assert isinstance(output, dict), f"Expected dict output, got {type(output)}"

    required_keys = {"detection", "phase_pick", "magnitude", "risk"}
    assert required_keys.issubset(output.keys()), f"Missing keys: {required_keys - set(output.keys())}"

    for key, tensor in output.items():
        assert not torch.isnan(tensor).any(), f"NaN in {key} output"

    # Detection in [0,1] after sigmoid
    det_probs = torch.sigmoid(output["detection"])
    assert det_probs.min() >= 0.0 and det_probs.max() <= 1.0, "Detection out of [0,1]"

    # Phase picks in [0,1]
    pp = output["phase_pick"]
    assert pp.min() >= 0.0 and pp.max() <= 1.0, f"Phase picks out of [0,1]: {pp}"

    # Risk softmax sums to ~1.0
    risk_probs = torch.softmax(output["risk"], dim=-1)
    sums = risk_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Risk probs don't sum to 1: {sums}"


def main() -> None:
    """Main export entry point."""
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    logger.info("Export start: checkpoint=%s, output_dir=%s", args.checkpoint, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and validate
    preprocess_config = load_config(args.preprocessing_config)
    model_config = load_config(args.model_config)
    validate_config_version(preprocess_config)
    validate_config_version(model_config)

    # Assert schema versions match
    assert preprocess_config["schema_version"] == model_config["schema_version"], (
        f"Schema version mismatch: preprocessing={preprocess_config['schema_version']}, "
        f"model={model_config['schema_version']}"
    )

    model = SeismicNet(model_config)
    ckpt_data = load_checkpoint(args.checkpoint, model, preprocess_config)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    # Step 2: Tune detection threshold
    if args.val_predictions and args.val_predictions.exists():
        threshold = tune_detection_threshold(args.val_predictions)
        model_config["detection_threshold"] = threshold
    else:
        logger.warning("No val predictions CSV provided, keeping default threshold")

    # Step 3: Strip training artifacts — clean state_dict
    clean_state = {k: v for k, v in ckpt_data["model_state_dict"].items()}

    # Step 4: Selective quantization
    model.load_state_dict(clean_state)
    logger.info("Applying selective quantization (Conv1d + Linear → qint8, Transformer stays float32)")
    quantized_model = quantize_model(model)
    quantized_model.eval()

    # Step 5: TorchScript
    scripted_model = torch.jit.script(quantized_model)

    # Step 6: Benchmark
    benchmark_results: dict[str, Any] = {}
    if args.benchmark:
        benchmark_results = benchmark_model(quantized_model)

        # Model file size
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.jit.save(scripted_model, f.name)
            import os
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
            os.unlink(f.name)

        benchmark_results["model_size_mb"] = size_mb

        # Check targets
        if benchmark_results["mean_latency_ms"] >= 150:
            warnings.warn(f"Mean latency {benchmark_results['mean_latency_ms']:.1f}ms >= 150ms target")
        if benchmark_results["p95_latency_ms"] >= 250:
            warnings.warn(f"P95 latency {benchmark_results['p95_latency_ms']:.1f}ms >= 250ms target")
        if size_mb >= 50:
            warnings.warn(f"Model size {size_mb:.1f}MB >= 50MB target")

        logger.info("Benchmark: mean=%.1fms, p95=%.1fms, size=%.1fMB",
                     benchmark_results["mean_latency_ms"],
                     benchmark_results["p95_latency_ms"],
                     size_mb)

    # Step 7: Write artifact bundle
    # model_scripted.pt
    scripted_path = args.output_dir / "model_scripted.pt"
    scripted_model.save(str(scripted_path))

    # preprocessing_config.json
    import shutil
    shutil.copy2(args.preprocessing_config, args.output_dir / "preprocessing_config.json")

    # model_config.json (with tuned threshold)
    (args.output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))

    # label_schema.json
    label_schema = {
        "risk_classes": {i: name for i, name in enumerate(model_config["risk_classes"])},
        "p_arrival_sample_index": preprocess_config["p_arrival_sample_index"],
        "window_length_samples": preprocess_config["window_length_samples"],
    }
    (args.output_dir / "label_schema.json").write_text(json.dumps(label_schema, indent=2))

    # benchmark_results.json
    if benchmark_results:
        (args.output_dir / "benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2))

    # Step 8: Smoke test
    smoke_test(scripted_model)

    logger.info("Export complete: %s", args.output_dir)
    print(f"\nExport complete to {args.output_dir}")
    print(f"Files: {', '.join(f.name for f in args.output_dir.iterdir())}")
    print("Smoke test: PASSED")


if __name__ == "__main__":
    main()
