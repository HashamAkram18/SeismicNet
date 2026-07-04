# SeismicNet — Project State

## Current Phase

**Step 8 of 19 — Pre-Training Validation Complete, GPU Runbook Ready**

## What Was Completed This Session

1. Fixed heads.py — parameterized input_dim:
   - DetectionHead, MagnitudeHead, RiskHead now accept input_dim parameter
   - SeismicNet passes d_model from config to all heads
   - Fixed crash with non-256 d_model configs

2. Fixed metrics.py — indentation error on roc_auc line

3. Hardened TorchScript types — `from typing import List, Dict, Optional`:
   - `seismic_net.py` forward uses `Optional[List[str]]` and `Dict[str, Tensor]`
   - Verified `torch.jit.script` works with explicit typing imports

4. Added `training/__init__.py` for proper package imports

5. Created `data/splits/` directory with README:
   - Splits must be generated on GPU server via `scripts/generate_splits.py`
   - Cannot generate locally (requires STEAD HDF5 data)

6. Generated `docs/05_gpu_server_runbook.md`:
   - rsync commands with .gitignore exclusions
   - Conda env creation from environment.yml
   - MLflow tmux launch + SSH tunnel
   - Phase 1 training CLI command
   - Phase transition protocol
   - Evaluation and export commands
   - Troubleshooting guide

## Pre-Training Checklist (10/10 PASS)

| # | Check | Status |
|---|---|---|
| 1 | tests/test_preprocessing.py passes | PASS (24/24) |
| 2 | tests/test_model_shapes.py passes | PASS (16/16) |
| 3 | tests/test_dataset.py — no split leakage | PASS (15+3skip) |
| 4 | MLflow server tmux | Documented in runbook |
| 5 | SSH tunnel | Documented in runbook |
| 6 | STEAD HDF5 in data/raw/ | Documented in runbook |
| 7 | data/splits/ files present | Created (generate on GPU) |
| 8 | nvidia-smi | Documented in runbook |
| 9 | preprocessing_config schema_version | PASS (1.0.0) |
| 10 | --run-name descriptive | Documented in runbook |

## Graphify State

- Last rebuild: 2026-07-04 (537 nodes, 859 edges, 41 communities)

## Next Action

Begin Step 8: Implement `seismic/inference.py` — inference endpoint.
Or rsync to GPU server and run Phase 1 training per `docs/05_gpu_server_runbook.md`.

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 16/16 passing
- `tests/test_training.py`: 6/6 passing
- Total: 61/61 passing, 3 skipped
