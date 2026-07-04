# Seismic Risk ML — Model Architecture, Loss & Metrics

## Scope

This document specifies the exact architecture of `SeismicNet`, all four task heads, the loss combiner, and the evaluation metrics. The agent must implement these in `seismic/model/encoder.py`, `seismic/model/heads.py`, `seismic/model/seismic_net.py`, `seismic/loss.py`, and `seismic/metrics.py` respectively.

---

## Architecture Overview

`SeismicNet` is a multi-task model with one shared encoder and four task-specific heads. The encoder is a hybrid of 1D ResNet stages (for local feature extraction) followed by Transformer encoder layers (for long-range temporal context). The full model processes 3-channel seismic waveforms of shape `[B, 3, 3000]` and a PGV scalar `[B, 1]`.

### Why this architecture

- Pure 1D CNN (PhaseNet-style): fast but poor long-range context; multi-task heads are awkward
- Pure LSTM+Attention (EQTransformer): best accuracy but ~400ms CPU inference — too slow
- **1D ResNet + Transformer (chosen)**: ~80ms CPU inference, near-EQTransformer accuracy, natural multi-task design

---

## seismic/model/encoder.py

### ResNet Block (1D)

Each ResBlock contains:
1. `Conv1d(in_ch, out_ch, kernel_size, padding='same', stride=stride)`
2. `BatchNorm1d(out_ch)`
3. `ReLU`
4. `Conv1d(out_ch, out_ch, kernel_size, padding='same', stride=1)`
5. `BatchNorm1d(out_ch)`
6. Skip connection: if `in_ch != out_ch` or `stride != 1`, use `Conv1d(in_ch, out_ch, 1, stride=stride)` + `BatchNorm1d`
7. `ReLU` after addition

The agent must not use `padding='same'` with strided convolutions (PyTorch does not support this combination). Use explicit `padding = kernel_size // 2` and set `stride` only on the first conv of each stage.

### Encoder architecture — exact layer specification

```
Input: [B, 3, 3000]

Stem:
  Conv1d(3, 64, kernel_size=7, stride=2, padding=3)  → [B, 64, 1500]
  BatchNorm1d(64)
  ReLU

Stage 1 — 2× ResBlock(64, 64, kernel_size=7, stride=1 on 1st block):
  [B, 64, 1500] → [B, 64, 1500]
  Note: no stride in Stage 1 — the stem already halved temporal dim

Stage 2 — 2× ResBlock(64, 128, kernel_size=7):
  First block: stride=2 on first conv
  [B, 64, 1500] → [B, 128, 750]

Stage 3 — 2× ResBlock(128, 256, kernel_size=5):
  First block: stride=2 on first conv
  [B, 128, 750] → [B, 256, 375]

Stage 4 — 2× ResBlock(256, 256, kernel_size=5, stride=1):
  [B, 256, 375] → [B, 256, 375]
  Note: no stride — we preserve temporal resolution here for phase picking head

Transformer encoder:
  Permute: [B, 256, 375] → [B, 375, 256]   (seq_len=375, d_model=256)
  Learned positional encoding: nn.Embedding(375, 256) added to sequence
  2× TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512,
                              dropout=0.1, activation='gelu', batch_first=True)
  Permute back: [B, 375, 256] → [B, 256, 375]

Output of encoder:
  sequence_features: [B, 256, 375]    ← used by phase picking head
  pooled_features:   [B, 256]         ← used by detection, magnitude, risk heads
    (computed as: global average pool over temporal dim, then Dropout(0.2))
```

### Critical implementation notes for the agent

**Positional encoding must be learned, not sinusoidal.** Sinusoidal encodings assume a fixed positional structure that seismic signals do not have. Use `nn.Embedding(375, 256)` with a position index tensor `[0, 1, ..., 374]`.

**Transformer dropout.** Set `dropout=0.1` inside `TransformerEncoderLayer`. Do not add additional dropout around the transformer block.

**Do not use `nn.Transformer`** (the full encoder-decoder). Use `nn.TransformerEncoder` with `nn.TransformerEncoderLayer` only.

**The encoder's `forward()` must return both `sequence_features` and `pooled_features`** as a tuple. The `SeismicNet` assembler routes them to the correct heads.

---

## seismic/model/heads.py

All four heads are implemented as separate `nn.Module` subclasses. They receive their inputs from `SeismicNet` — the agent must not call the encoder inside any head.

### Task A — DetectionHead

```
Input: pooled_features [B, 256]
Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 1)
Output: logits [B, 1]   (sigmoid applied in loss, not in head)
```

At inference, apply `torch.sigmoid()` to logits. The classification threshold is **not** hardcoded to 0.5 — it is loaded from `model_config.json:detection_threshold` which is tuned on the validation set after training.

### Task B — PhasePickHead

**This head reads from `sequence_features [B, 256, 375]`, not from `pooled_features`.** This is the most important architectural constraint in the entire system — see `01_project_overview.md` for the reason.

```
Input: sequence_features [B, 256, 375]

Attention pointer mechanism:
  Query projection: Linear(256, 64) applied to each time step → [B, 375, 64]
  Score: Linear(64, 1) → [B, 375, 1] → squeeze → [B, 375]
  Softmax over temporal dim → attention weights [B, 375]

P-pick output:
  p_weights = Softmax(Linear(256→64→1) per timestep) [B, 375]
  p_pick = sum(p_weights * position_index_normalized) → [B, 1] scalar
  position_index_normalized = torch.linspace(0, 1, 375)

S-pick output:
  Separate identical attention pointer for S-wave
  s_pick → [B, 1] scalar

Output: [B, 2]   (p_pick_normalized, s_pick_normalized)
Both values are in [0, 1] representing position within the 3000-sample window.
```

This "soft argmax" / differentiable pointer design allows gradient flow through the pick position without a hard argmax. The agent must implement this pattern — not a simple linear regression from the pooled vector.

At inference, convert normalized pick back to sample index:
`sample_index = pick_normalized * config["window_length_samples"]`

### Task C — MagnitudeHead

```
Input: concat([pooled_features [B, 256], pgv [B, 1]]) → [B, 257]
Linear(257, 64) → ReLU → Dropout(0.1) → Linear(64, 1)
Output: magnitude_pred [B, 1]   (raw value — trained on log10(Ml + 1) scale)
```

At inference, convert back: `Ml = 10^(output) - 1`

The PGV scalar concatenation is mandatory. Without it the head must infer magnitude purely from waveform shape (which is z-score normalized and has no amplitude information). The PGV carries the only amplitude signal that survives normalization.

### Task D — RiskHead

```
Input: pooled_features [B, 256]
Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 4)
Output: logits [B, 4]   (softmax applied in loss)
Classes: {0: low, 1: moderate, 2: high, 3: critical}
```

At inference, apply `torch.softmax()` to get class probabilities. Return both the argmax class and the full probability vector — the API consumer needs confidence scores, not just the class label.

---

## seismic/model/seismic_net.py — Assembly

`SeismicNet` assembles the encoder and all four heads. Its `forward()` signature is fixed and must not change:

```python
def forward(
    self,
    waveform: Tensor,    # [B, 3, 3000]
    pgv: Tensor,         # [B, 1]
    active_tasks: list[str] = ["detection", "phase_pick", "magnitude", "risk"]
) -> dict:
```

The `active_tasks` argument controls which heads are executed. During Phase 1 training, pass `["detection", "phase_pick"]`. This avoids computing gradients for inactive heads and is more efficient than zero-weighting their losses.

Return dict keys (only keys for active tasks are populated):
```python
{
    "detection": Tensor[B, 1],        # logits
    "phase_pick": Tensor[B, 2],       # normalized pick positions
    "magnitude": Tensor[B, 1],        # log-scale magnitude
    "risk": Tensor[B, 4],             # class logits
}
```

---

## seismic/loss.py — Loss Functions

### Per-task loss functions

**Task A — Detection loss**
```
BCEWithLogitsLoss(pos_weight=tensor([10.0]))
```
`pos_weight=10.0` compensates for the ~50% noise rate in STEAD. This value must be treated as a hyperparameter in `training_config.json` and not hardcoded.

**Task B — Phase pick loss**
```
HuberLoss(delta=0.02, reduction='mean')
Applied only on seismic windows (is_noise == False)
Applied only to the p_pick output in Phase 1; both p and s picks from Phase 2 onward
```
`delta=0.02` corresponds to ~60 samples (0.6 seconds) at the normalized scale — anything larger than this is treated as a linear penalty rather than quadratic. Robust to catalog labeling errors.

**Task C — Magnitude loss**
```
MSELoss(reduction='mean')
Applied on log10(label_magnitude + 1) vs model output
Applied only on seismic windows (is_noise == False)
```

**Task D — Risk classification loss**
```
CrossEntropyLoss(weight=class_weights)
class_weights: inverse frequency weights computed from training split label distribution
Applied only on seismic windows (is_noise == False)
```

### Loss combiner

The agent must implement a `SeismicLoss` class that:

1. Computes each active task's loss
2. Applies the noise mask (`is_noise`) to Tasks B, C, D before computing loss
3. Multiplies each task loss by its phase-specific weight from `training_config.json`
4. Returns a dict of individual losses AND the weighted sum

```python
total_loss = (
    w_detection * loss_A +
    w_phase_pick * loss_B +
    w_magnitude * loss_C +
    w_risk * loss_D
)
```

Loss weights per training phase (from `training_config.json`):

| Task | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| A · Detection | 1.0 | 1.0 | 1.0 |
| B · Phase pick | 2.0 | 2.0 | 1.5 |
| C · Magnitude | 0.0 | 0.3 | 0.8 |
| D · Risk | 0.0 | 0.0 | 0.5 |

The loss weight values must come from config, not be hardcoded in `loss.py`. The agent must load them from `training_config.json` at the start of each training phase.

---

## seismic/metrics.py — Evaluation Metrics

The agent must implement a `SeismicMetrics` class with a `compute()` method that takes model outputs and labels for an entire val/test epoch and returns the following dict:

### Task A metrics
```
detection_precision    float
detection_recall       float
detection_f1           float
detection_roc_auc      float
detection_false_alarm_rate   float    # FP / (FP + TN) — key operational metric
```

### Task B metrics
```
p_pick_mae_samples     float    # mean absolute error in samples (multiply by 0.01 for seconds)
s_pick_mae_samples     float
p_pick_within_0.1s     float    # fraction of picks with MAE < 10 samples
s_pick_within_0.1s     float
```

### Task C metrics
```
magnitude_rmse         float    # on original Ml scale (after back-transforming log)
magnitude_mae          float
magnitude_bias         float    # mean(pred - true) — systematic over/under estimation
```

### Task D metrics
```
risk_accuracy          float
risk_f1_macro          float
risk_confusion_matrix  list[list[int]]    # 4×4, logged as MLflow artifact
risk_critical_recall   float              # recall on class 3 (critical) — most important operationally
```

All metrics except `risk_confusion_matrix` must be logged as scalar MLflow metrics at epoch end. The confusion matrix is logged as a JSON artifact.

---

## seismic/model/seismic_net.py — Model Config Schema

`model_config.json` controls all architecture dimensions. The agent must read all values from this file — no magic numbers in model code.

```json
{
  "schema_version": "1.0.0",
  "stem_channels": 64,
  "stage_channels": [64, 128, 256, 256],
  "stage_blocks": [2, 2, 2, 2],
  "stage_kernels": [7, 7, 5, 5],
  "transformer_d_model": 256,
  "transformer_nhead": 8,
  "transformer_layers": 2,
  "transformer_ffn_dim": 512,
  "transformer_dropout": 0.1,
  "head_hidden_dim": 64,
  "head_dropout": 0.1,
  "pooling_dropout": 0.2,
  "detection_threshold": 0.5,
  "risk_classes": ["low", "moderate", "high", "critical"]
}
```

`detection_threshold` defaults to 0.5 but must be overwritten after tuning on the validation set. The tuned value is logged as an MLflow parameter and saved back to `model_config.json` in the model export step.

---

## Model Shape Test — tests/test_model_shapes.py

The agent must implement shape assertion tests for every forward path:

| Test | Input | Expected output |
|---|---|---|
| `test_detection_shape` | waveform [4,3,3000], pgv [4,1] | detection [4,1] |
| `test_phase_pick_shape` | waveform [4,3,3000], pgv [4,1] | phase_pick [4,2] |
| `test_magnitude_shape` | waveform [4,3,3000], pgv [4,1] | magnitude [4,1] |
| `test_risk_shape` | waveform [4,3,3000], pgv [4,1] | risk [4,4] |
| `test_active_tasks_filter` | active_tasks=["detection"] | only "detection" key in output dict |
| `test_no_nan_in_output` | random valid input | no NaN in any output tensor |
| `test_phase_pick_range` | random valid input | all phase pick values in [0, 1] |

These tests run on CPU with a randomly initialized model and batch size 4. They must pass before any training script is executed on the GPU server.
