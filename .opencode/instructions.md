# SeismicNet — Project State

## Current Phase

**Step 7 of 19 — Training Pipeline Complete (train.py, evaluate.py, export.py)**

## What Was Completed This Session

1. Implemented `training/train.py` — main training entry point:
   - CLI with argparse (--train-config, --model-config, --data-dir, --output-dir, --phase)
   - seed_everything() for reproducibility
   - Phase-wise training loop with optimizer reset per phase
   - Cosine-with-warmup scheduler (per gradient step)
   - Mixed precision with GradScaler
   - Gradient clipping, early stopping
   - MLflow logging of all metrics
   - Checkpoint saving via utils.save_checkpoint()
   - 3-phase schedule: Tasks A+B → A+B+C → A+B+C+D

2. Implemented `training/evaluate.py` — standalone test evaluation:
   - Loads checkpoint, runs full evaluation
   - Reports all task metrics from SeismicMetrics

3. Implemented `training/export.py` — model export to TorchScript:
   - torch.jit.script (NOT trace)
   - Selective quantization (Conv1d + Linear only)
   - Smoke test with real data
   - artifact_bundle.pt saved with model + config

4. Fixed heads.py — parameterized input_dim:
   - DetectionHead, MagnitudeHead, RiskHead now accept input_dim parameter
   - SeismicNet passes d_model from config to all heads
   - Fixed crash with non-256 d_model configs

## Graphify State

- Last rebuild: 2026-07-04 (537 nodes, 859 edges, 41 communities)

## Next Action

Begin Step 8: Implement `seismic/inference.py` — inference endpoint.

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 16/16 passing
- `tests/test_training.py`: 6/6 passing
- Total: 61/61 passing, 3 skipped
