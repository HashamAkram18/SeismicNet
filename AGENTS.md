# AGENTS.md — SeismicNet Project

Read this file completely before doing anything else in this project.
After reading this file, read the specific doc in `docs/` that governs the module you are about to work on.

---

## What This Project Is

A multi-task deep learning system for seismic risk analysis. It ingests 3-component seismic waveforms and simultaneously outputs: event detection, P/S phase picks, magnitude estimate, and structural damage risk class. One shared encoder, four task heads. Trained on GPU over SSH, served on CPU via Django REST.

**Primary dataset**: STEAD (Stanford Earthquake Dataset) — HDF5, 1.2M samples, 100 Hz, 3 components.

---

## docs/ Index — Read Before Touching Any Module

| Document | Governs | Read when you are about to... |
|---|---|---|
| `docs/01_project_overview.md` | Folder structure, environment, configs, invariants | Start a new session or scaffold the project |
| `docs/02_data_pipeline.md` | preprocessing.py, dataset.py, augmentation.py, splits | Touch any data or preprocessing code |
| `docs/03_model_architecture.md` | encoder.py, heads.py, seismic_net.py, loss.py, metrics.py | Touch any model, loss, or metric code |
| `docs/04_training_and_export.md` | train.py, evaluate.py, export.py, MLflow, checkpointing | Touch any training or export code |

If you are unsure which doc applies, read `docs/01_project_overview.md` first — it will route you.

---

## The 5 Invariants — Never Break These

These are repeated here from `docs/01_project_overview.md` because they must be checked without looking anything up.

**Invariant 1** — `seismic/preprocessing.py` is the only place signal processing logic lives. The inference endpoint and the training DataLoader both import and call the same functions from this module. Never duplicate or reimplement signal processing elsewhere.

**Invariant 2** — Every `torch.save()` for a checkpoint must also write `preprocessing_config.json` into the same checkpoint directory with a matching `schema_version`. Enforced in `utils.save_checkpoint()`.

**Invariant 3** — `SeismicNet.forward()` signature is fixed: `(waveform: Tensor[B,3,3000], pgv: Tensor[B,1], active_tasks: list[str])` → returns dict with keys `detection`, `phase_pick`, `magnitude`, `risk`. Do not change this signature.

**Invariant 4** — Noise windows (Task A label = 0) must have Tasks B, C, D losses masked to zero in `loss.py`. This is not optional and must be enforced with the `is_noise` batch key.

**Invariant 5** — All random seeds (torch, numpy, Python random) are set from `training_config.json:seed` at the start of `train.py`. Never hardcode a seed value in any module.

---

## Preprocessing Pipeline Order — Locked

This order must not change. It is frozen in `configs/preprocessing_config.json:schema_version`.

```
1. Validate sampling rate (assert == 100 Hz, raise ValueError if not)
2. Sub-window extraction (P-arrival at sample index 1000 of output window)
3. Detrend (scipy.signal.detrend, type='linear', per channel)
4. Taper (5% cosine taper, both ends, per channel)
5. Bandpass filter (1–45 Hz, Butterworth order 4, zero-phase sosfiltfilt)
6. Extract PGV scalar (max(abs(waveform)) across all channels, before normalization)
7. Per-trace z-score normalization (each of 3 channels independently)
```

Changing the order requires bumping `preprocessing_config.json:schema_version` and retraining from scratch.

---

## Key Architectural Facts — No Lookup Required

- Input shape: `[B, 3, 3000]` waveform + `[B, 1]` PGV scalar
- Encoder output: `sequence_features [B, 256, 375]` AND `pooled_features [B, 256]`
- Phase picking head reads from `sequence_features` — NOT from `pooled_features`
- Detection, magnitude, risk heads read from `pooled_features`
- Magnitude head receives `concat([pooled_features, pgv]) = [B, 257]` as input
- Training has 3 phases: Tasks A+B → Tasks A+B+C → Tasks A+B+C+D
- Optimizer resets (fresh AdamW + LR scheduler warmup) at every phase boundary
- Quantization: ResNet Conv1d + Linear layers only — Transformer stays float32

---

## What To Do at the Start of Every Session

1. Read `.opencode/instructions.md` — this is the live project state
2. Read the `docs/` file relevant to today's task
3. Run `/status` to confirm current phase, best metrics, and next action
4. Do not write code before confirming which module you are implementing and that you have read its governing doc

## What To Do at the End of Every Session

1. Update `.opencode/instructions.md` with:
   - What was implemented or changed
   - Current training phase and best metrics
   - Any open issues discovered
   - Next action for the following session
2. Run `/test [module]` for every module touched in this session
3. Confirm all tests pass before ending the session

---

## Hard Rules for the Agent

- Never reimplement signal processing outside `preprocessing.py`
- Never split data by random sample — always split by event ID
- Never apply augmentation to val or test splits
- Never commit HDF5 files, `.pt` checkpoints, or `exported_models/` to git
- Never hardcode loss weights, learning rates, or threshold values in source code — always read from config
- Never use `torch.jit.trace` for export — always use `torch.jit.script`
- Never skip the pre-training checklist in `docs/04_training_and_export.md` before a GPU run

---

## Graphify Knowledge Graph

Graphify is installed and integrated with OpenCode via `.opencode/plugins/graphify.js`.
It produces three files in `graphify-out/` — treat them as read-only build artifacts, never edit them manually.

### Two-source rule

The agent uses **two complementary sources** depending on the question type:

| Question type | Source to use |
|---|---|
| "What should this function do?" | `docs/` — prescriptive specs, read before implementing |
| "What already exists / what calls what?" | `graphify query "..."` — descriptive graph of actual code |
| "Broad architecture overview of current state" | `graphify-out/GRAPH_REPORT.md` |
| "Is this module imported anywhere?" | `graphify query "which modules import preprocessing"` |

Never grep raw source files for architecture questions when the graph exists and is fresh.

### When the graph is fresh vs stale

The graph is considered **stale** if any `.py` or `.md` file in the project is newer than `graphify-out/graph.json`.
Run `/regraph check` at the start of any session where the agent needs to navigate existing code.
Run `/regraph` (full rebuild) at these milestones:

1. After initial scaffold (skeleton files exist, no implementation yet)
2. After each implementation milestone: all of `seismic/` done, all of `training/` done
3. After any architectural deviation from the docs
4. Before starting a new ML domain (structural, seismic, orbital etc.)

### How to query the graph

The agent must prefer scoped queries over reading the full report:

```
graphify query "what does preprocessing.py export"
graphify query "which modules call SeismicNet"
graphify query "what is the call path from train.py to loss.py"
graphify query "which files define ResBlock"
graphify query "what imports augmentation"
```

For broad architectural orientation only (not for specific function lookup), read `graphify-out/GRAPH_REPORT.md`.

### After every /regraph rebuild

The agent must:
1. Read the "surprising connections" section of the new `GRAPH_REPORT.md`
2. Flag any connection that contradicts the docs (e.g. signal processing logic found outside `preprocessing.py`)
3. Report findings to the human before continuing with implementation
4. Log the rebuild timestamp in `.opencode/instructions.md`

### Hard rules for Graphify

- Never commit `graphify-out/` to git — it is in `.gitignore`
- Never edit `graph.json` or `GRAPH_REPORT.md` manually
- Never treat a stale graph as authoritative — run `/regraph check` first
- If `graphify query` returns empty or unexpected results, run `/regraph` before trusting the output