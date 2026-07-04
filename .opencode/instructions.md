# SeismicNet — Project State

## Current Phase

**Step 5 of 19 — heads.py + seismic_net.py implemented**

## What Was Completed This Session

1. Implemented `seismic/model/heads.py` — all 4 task heads:
   - `DetectionHead`: Linear(256→64)→ReLU→Dropout→Linear(64→1), no sigmoid
   - `PhasePickHead`: attention pointer mechanism, separate P/S pointers, softmax over 375 positions × position_index, outputs [B,2] in [0,1]
   - `MagnitudeHead`: concat(pooled_features, pgv)→[B,257]→Linear→ReLU→Dropout→Linear(64→1)
   - `RiskHead`: Linear(256→64)→ReLU→Dropout→Linear(64→4), no softmax
2. Implemented `seismic/model/seismic_net.py` — assembler with fixed forward signature:
   - Encoder returns (sequence_features, pooled_features)
   - PhasePickHead ← sequence_features (verified by test)
   - Detection/Magnitude/Risk heads ← pooled_features
   - MagnitudeHead additionally receives pgv (verified by test)
   - active_tasks filter: only computes requested heads
3. Updated `tests/test_model_shapes.py` — 16 tests, all passing:
   - Encoder shape: 3 tests
   - Head shapes: 4 tests
   - SeismicNet forward: 9 tests including:
     - test_phase_pick_reads_sequence_features (stub encoder routing test)
     - test_pgv_routing (pgv only affects MagnitudeHead output)
     - test_active_tasks_filter (only requested keys)
     - test_no_nan_in_output (no NaN in any tensor)

## Graphify State

- Last rebuild: 2026-07-04 (534 nodes, 735 edges, 37 communities)

## Next Action

Begin Step 6: Implement `seismic/loss.py` — per-task losses + noise masking + weighted combiner.

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 16/16 passing ✅
