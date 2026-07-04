# SeismicNet — Project State

## Current Phase

**Step 6 of 19 — loss.py + metrics.py + utils.py implemented**

## What Was Completed This Session

1. Implemented `seismic/loss.py` — SeismicLoss class:
   - Task A: BCEWithLogitsLoss with pos_weight from config (default 10.0)
   - Task B: HuberLoss(delta from config, reduction=none) — noise-masked
   - Task C: MSELoss(reduction=none) — noise-masked
   - Task D: CrossEntropyLoss(reduction=none) — noise-masked
   - Noise mask: is_noise boolean masks Tasks B/C/D, division by n_seismic
   - Weighted total: w_detection × A + w_phase_pick × B + w_magnitude × C + w_risk × D
   - All weights from phase_config, not hardcoded

2. Implemented `seismic/metrics.py` — SeismicMetrics class:
   - Task A: precision, recall, f1, roc_auc, false_alarm_rate
   - Task B: p/s_pick_mae_samples, p/s_pick_within_0.1s
   - Task C: magnitude_rmse, magnitude_mae, magnitude_bias (back-transformed from log)
   - Task D: risk_accuracy, risk_f1_macro, risk_confusion_matrix (4×4), risk_critical_recall
   - update() accumulates per batch, compute() returns full dict

3. Implemented `seismic/utils.py` — checkpoint and config utilities:
   - save_checkpoint(): saves .pt + copies preprocessing_config.json, prunes top-K
   - load_checkpoint(): validates preprocessing_config_version, raises ValueError on mismatch
   - load_config(): validates schema_version field present
   - validate_config_version(): checks version matches expected

## Graphify State

- Last rebuild: 2026-07-04 (525 nodes, 729 edges, 37 communities)

## Next Action

Begin Step 7: Implement `training/train.py` — main training entry point.

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 16/16 passing ✅
- Total: 55/55 passing, 3 skipped
