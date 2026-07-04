# Seismic Risk ML — Training Pipeline, GPU Setup & Model Export

## Scope

This document covers everything from launching a training run on the GPU server to producing a deployable inference artifact. It governs `training/train.py`, `training/evaluate.py`, `training/export.py`, the MLflow setup, and the SSH workflow.

---

## training_config.json — Schema

All training hyperparameters must be read from this file. The agent must never hardcode training values in `train.py`. This config is logged as an MLflow artifact at the start of every run so results are always reproducible.

```json
{
  "schema_version": "1.0.0",
  "seed": 42,
  "batch_size": 256,
  "num_workers": 8,
  "pin_memory": true,
  "phases": [
    {
      "phase_id": 1,
      "epochs": 20,
      "active_tasks": ["detection", "phase_pick"],
      "loss_weights": {
        "detection": 1.0,
        "phase_pick": 2.0,
        "magnitude": 0.0,
        "risk": 0.0
      },
      "optimizer": {
        "type": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4
      },
      "scheduler": {
        "type": "cosine_with_warmup",
        "warmup_steps": 500,
        "min_lr": 1e-6
      }
    },
    {
      "phase_id": 2,
      "epochs": 20,
      "active_tasks": ["detection", "phase_pick", "magnitude"],
      "loss_weights": {
        "detection": 1.0,
        "phase_pick": 2.0,
        "magnitude": 0.3,
        "risk": 0.0
      },
      "optimizer": {
        "type": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4
      },
      "scheduler": {
        "type": "cosine_with_warmup",
        "warmup_steps": 200,
        "min_lr": 1e-6
      }
    },
    {
      "phase_id": 3,
      "epochs": 40,
      "active_tasks": ["detection", "phase_pick", "magnitude", "risk"],
      "loss_weights": {
        "detection": 1.0,
        "phase_pick": 1.5,
        "magnitude": 0.8,
        "risk": 0.5
      },
      "optimizer": {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4
      },
      "scheduler": {
        "type": "cosine_with_warmup",
        "warmup_steps": 200,
        "min_lr": 1e-6
      }
    }
  ],
  "gradient_clip_norm": 1.0,
  "early_stopping_patience": 10,
  "early_stopping_metric": "detection_f1",
  "mixed_precision": true,
  "detection_pos_weight": 10.0,
  "huber_delta": 0.02,
  "keep_top_k_checkpoints": 3
}
```

---

## training/train.py — Entry Point

### Command-line interface

The agent must implement `train.py` with the following CLI arguments:

```
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
  --resume checkpoints/run_abc/epoch_12.pt   # optional
```

The `--phase` argument selects which phase config block from `training_config.json` to use. A new MLflow run is created per phase — they are linked by a `parent_run_id` tag on phases 2 and 3.

### train.py execution order

The agent must implement `train.py` to execute in exactly this order:

1. **Seed everything**: `torch.manual_seed`, `numpy.random.seed`, `random.seed` — all from `training_config.json:seed`
2. **Load and validate all three configs**: check schema versions match expected values; raise hard errors on mismatch
3. **Start MLflow run**: log all config files as artifacts immediately
4. **Build dataset and DataLoader**: training split with `augment=True`, val split with `augment=False`
5. **Instantiate model**: `SeismicNet` from `model_config.json`
6. **If `--resume`**: load checkpoint, restore model weights and optimizer state; log resumed epoch as MLflow param
7. **Instantiate optimizer and scheduler**: fresh instances at the start of every phase (even when resuming within a phase, reset the scheduler to warmup start)
8. **Instantiate `SeismicLoss`**: with phase-specific weights from config
9. **Training loop**: see below
10. **On loop completion**: run `evaluate.py` logic on test split; log final test metrics to MLflow

### Training loop — per epoch

```
for epoch in range(start_epoch, phase_config.epochs):

    model.train()
    for batch in train_loader:

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=training_config.mixed_precision):
            outputs = model(batch["waveform"].cuda(), batch["pgv"].cuda(),
                           active_tasks=phase_config.active_tasks)
            losses = loss_fn(outputs, batch)    # returns dict of per-task losses + total

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Log to MLflow every 100 steps:
        # losses["detection"], losses["phase_pick"], losses["magnitude"], losses["risk"], losses["total"]
        # grad_norm (compute before clip for monitoring)

    # End of epoch:
    model.eval()
    val_metrics = run_validation(model, val_loader, loss_fn)
    log_all_metrics_to_mlflow(val_metrics, epoch)
    maybe_save_checkpoint(model, optimizer, epoch, val_metrics)
    if early_stopping_triggered(val_metrics, patience): break
```

### Checkpoint saving

The agent must implement `utils.save_checkpoint()` that:

1. Saves to `checkpoints/run_{mlflow_run_id}/epoch_{epoch:03d}.pt`
2. The checkpoint file contains:
   ```python
   {
       "epoch": int,
       "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "scheduler_state_dict": scheduler.state_dict(),
       "val_metrics": dict,
       "model_config_version": str,
       "preprocessing_config_version": str,
       "training_config_version": str,
       "mlflow_run_id": str,
       "phase_id": int,
   }
   ```
3. Copies `preprocessing_config.json` into the same checkpoint directory
4. Keeps only the top-K checkpoints by `detection_f1` (K from `training_config.json:keep_top_k_checkpoints`)
5. Logs the checkpoint path as an MLflow artifact

Loading a checkpoint (`utils.load_checkpoint()`) must validate that `preprocessing_config_version` in the checkpoint matches the current `preprocessing_config.json`. Raise a hard error if they differ.

### Mixed precision — GradScaler

The agent must instantiate `torch.cuda.amp.GradScaler` at the start of training. If `--resume` is used, the scaler state must also be saved and restored from the checkpoint.

### Early stopping

Monitor `detection_f1` on the validation set. If it does not improve for `patience=10` consecutive epochs, stop the current phase. Log the stopping epoch to MLflow. Do not apply early stopping across phases — only within a phase.

---

## MLflow Setup on SSH GPU Server

### Server launch (run once, keep alive in tmux)

```bash
# On GPU server
tmux new-session -s mlflow
mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

### SSH tunnel (run from local machine)

```bash
ssh -N -L 5000:localhost:5000 user@gpu-server-address
```

Access MLflow UI at `http://localhost:5000` on the local machine. Use `autossh` to keep the tunnel alive if sessions drop.

### MLflow experiment structure

| Experiment | Purpose |
|---|---|
| `seismic_risk` | All SeismicNet training runs |
| `seismic_baselines` | PhaseNet and EQTransformer baseline comparison runs |
| `seismic_ablations` | Architecture ablation studies |

### What to log per run

**Parameters** (logged once at run start):
- All fields from all three config files, flattened with dot notation (e.g., `model.transformer_layers`, `training.phase1.lr`)
- `git_commit` (output of `git rev-parse HEAD`)
- `dataset_version` (hash of the split files)
- `phase_id`

**Metrics** (logged per epoch):
- `train/loss_total`, `train/loss_detection`, `train/loss_phase_pick`, `train/loss_magnitude`, `train/loss_risk`
- `val/loss_total` and same per-task
- All metrics from `seismic/metrics.py` under `val/` prefix
- `train/grad_norm` (mean over epoch)
- `train/lr` (current learning rate)
- `train/gpu_memory_mb`

**Artifacts** (logged once or at epoch end):
- All three config JSON files (at run start)
- Best checkpoint `.pt` file (when a new best is reached)
- `preprocessing_config.json` (at run start and alongside every checkpoint)
- `risk_confusion_matrix.json` (at each epoch)

### Model registry promotion

After Phase 3 training is complete:
1. Identify the run with the best `val/detection_f1` in the `seismic_risk` experiment
2. Register the best checkpoint as model version in the MLflow model registry under name `SeismicNet`
3. Promote to `Staging` after passing the export benchmark (see below)
4. Promote to `Production` only after the Django inference integration test passes

---

## training/evaluate.py — Standalone Evaluation

The agent must implement a standalone script for running evaluation on the test split:

```
python training/evaluate.py \
  --checkpoint checkpoints/run_abc/epoch_045.pt \
  --preprocessing-config configs/preprocessing_config.json \
  --data-dir data/ \
  --output-dir eval_results/ \
  --mlflow-uri http://localhost:5000
```

This script:
1. Loads checkpoint and validates preprocessing config version match
2. Runs full test split through the model in `eval()` mode, no augmentation
3. Computes all metrics from `seismic/metrics.py`
4. Saves per-sample predictions to `eval_results/predictions.csv` (event_id, all predictions, all labels)
5. Saves the confusion matrix to `eval_results/confusion_matrix.json`
6. Logs all metrics to the originating MLflow run (using the run ID from the checkpoint)

The predictions CSV enables offline analysis and error case inspection without re-running the model.

---

## training/export.py — GPU Checkpoint to Deployable Artifact

This script converts a trained checkpoint into the artifact bundle that Django REST will load.

```
python training/export.py \
  --checkpoint checkpoints/run_abc/epoch_045.pt \
  --preprocessing-config configs/preprocessing_config.json \
  --output-dir exported_models/seismic_v1/ \
  --benchmark
```

### Export steps — in order

**Step 1 — Load and validate**
Load checkpoint. Validate preprocessing config version. Instantiate `SeismicNet` from the checkpoint's `model_config_version`. Load weights. Set to `eval()` mode.

**Step 2 — Tune detection threshold**
Using the validation split predictions from `evaluate.py` output (must be run first), find the decision threshold that maximizes F1 on the val set. Update `model_config.json:detection_threshold` with this value.

**Step 3 — Strip training artifacts**
Create a clean `state_dict` containing only model weights. No optimizer state, no scheduler state, no epoch info.

**Step 4 — Selective quantization**
Apply `torch.quantization.quantize_dynamic` to the ResNet Conv1d and Linear layers only. The Transformer encoder layers must remain in float32 due to softmax numerical sensitivity.

Specifically:
```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear, nn.Conv1d},   # only these layer types
    dtype=torch.qint8
)
```

**Step 5 — TorchScript**
Apply `torch.jit.script(quantized_model)`. If scripting fails due to type annotations, fix the annotations in the model code — do not fall back to `torch.jit.trace`. Trace does not handle control flow (the `active_tasks` argument) correctly.

**Step 6 — Benchmark on CPU**
Run 100 forward passes on CPU with a batch of 1 (single waveform inference — the inference scenario). Measure:
- Mean latency (ms)
- P95 latency (ms)
- Memory footprint (MB)

Acceptable targets:
- Mean latency < 150ms
- P95 latency < 250ms
- Model file size < 50MB

If targets are not met, log the benchmark results and raise a warning — do not block the export, but flag for review.

**Step 7 — Write artifact bundle**
Write to `output-dir/`:
```
exported_models/seismic_v1/
├── model_scripted.pt              # TorchScript + quantized model
├── preprocessing_config.json     # Identical copy from training
├── model_config.json             # With tuned detection_threshold
├── label_schema.json             # Risk class names and indices
└── benchmark_results.json        # Latency and memory measurements
```

The version string in `preprocessing_config.json:schema_version` and `model_config.json:schema_version` must match. The export script must assert this before writing.

**Step 8 — Smoke test**
Construct a synthetic waveform (random noise, shape `[1, 3, 3000]`) and run it through the exported `model_scripted.pt`. Assert that:
- All four output keys are present
- No NaN values in any output
- Detection output is in [0, 1] after sigmoid
- Phase pick values are in [0, 1]
- Risk probabilities sum to ~1.0

If any assertion fails, the export is invalid. Fix the model or export process before proceeding.

---

## Data Augmentation During Training — Reminder

Augmentation is applied inside the DataLoader workers (CPU-side) via `augmentation.py`. It must not use GPU operations. The agent must confirm this by checking that no `augmentation.py` function calls `.cuda()` or `.to(device)`. This is enforced as a lint check in the test suite.

---

## GPU Server — Practical Checklist for the Agent

Before launching any training run on the SSH GPU server, the agent must confirm:

1. `tests/test_preprocessing.py` — all tests pass on CPU (local machine)
2. `tests/test_model_shapes.py` — all shape tests pass on CPU
3. `tests/test_dataset.py` — no split leakage detected
4. MLflow server is running in a tmux session on the GPU server
5. SSH tunnel is active — `curl http://localhost:5000` returns MLflow UI HTML
6. STEAD HDF5 files are present in `data/raw/` on the GPU server
7. `data/splits/train.txt`, `val.txt`, `test.txt` are present and committed
8. GPU is visible: `nvidia-smi` returns at least one GPU
9. `configs/preprocessing_config.json` schema_version matches what `train.py` expects
10. `--run-name` argument is descriptive (e.g., `phase1_resnet_transformer_baseline`) — not `test` or `run1`

The agent must not skip any item on this checklist. A failed run at epoch 15 because of a missing HDF5 file wastes GPU time that cannot be recovered.

---

## Phase Transition Protocol

When moving from Phase 1 to Phase 2 (or Phase 2 to Phase 3):

1. Confirm Phase 1's best checkpoint has `detection_f1 > 0.90` on val set before proceeding. If not, extend Phase 1 or debug before adding tasks.
2. Load the best Phase 1 checkpoint as the starting point for Phase 2 — do not train Phase 2 from scratch.
3. Create a new MLflow run for Phase 2, tagged with `parent_run_id = <phase1_run_id>`.
4. Reset the optimizer (fresh `AdamW` instance) — do not carry over Phase 1 optimizer momentum.
5. Reset the LR scheduler to warmup start — do not inherit Phase 1's decayed LR.
6. Set `active_tasks` to Phase 2 list in the `SeismicNet` forward call.
7. Update loss weights in `SeismicLoss` to Phase 2 values.

The same protocol applies at the Phase 2 → Phase 3 boundary, with the additional gate: `magnitude_rmse < 0.5` on val set before proceeding to Phase 3.
