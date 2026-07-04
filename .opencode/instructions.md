# SeismicNet — Project State

## Current Phase

**Step 4 of 19 — augmentation.py + encoder.py implemented**

## What Was Completed This Session

1. Implemented `seismic/augmentation.py` — all 4 augmentation types:
   - `time_shift()` — random circular shift ±50 samples, zero-pad vacated ends
   - `amplitude_scale()` — random Uniform(0.5, 2.0) scaling
   - `additive_noise()` — Gaussian noise at Uniform(0.5-2%) of per-channel std
   - `channel_dropout()` — zero one random channel (E/N/Z)
   - `augment_waveform()` — pipeline orchestrator with per-augmentation probability
2. Implemented `seismic/model/encoder.py` — full ResNet + Transformer encoder:
   - `ResBlock` — 1D residual block with optional stride and skip projection
   - `SeismicEncoder` — Stem → 4 stages → Transformer → pool
   - Shape verified: input [B,3,3000] → output ([B,256,375], [B,256])
   - Learned positional encoding (nn.Embedding, not sinusoidal)
   - Transformer: 2 layers, 8 heads, 512 FFN, gelu activation
   - Stage 4 stride=1 to preserve temporal resolution for phase picking
3. Updated `tests/test_model_shapes.py` — 3 encoder tests passing, head/net stubs remain

## Graphify State

- Last rebuild: 2026-07-04 (538 nodes, 686 edges, 36 communities)

## Next Action

Begin Step 5: Implement `seismic/model/heads.py` — all 4 task heads.

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 3/14 passing (encoder), 11 stubs (heads/net)
