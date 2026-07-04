# Seismic Risk ML Platform — Project Overview & Structure

## Purpose

This document is the authoritative reference for the IDE agent responsible for scaffolding and maintaining the seismic risk ML project. Read this file first before touching any other file. Every structural and environment decision made here is final unless a later versioned `.md` file explicitly overrides it with a reason.

---

## What This Project Does

This is a **multi-task deep learning system** that ingests 3-component seismic waveform data and simultaneously performs:

- **Task A — Detection**: Binary classification — is this window a seismic event or noise?
- **Task B — Phase Picking**: Regression — at which sample does the P-wave and S-wave arrive?
- **Task C — Magnitude Estimation**: Regression — what is the local magnitude (Ml)?
- **Task D — Risk Classification**: Multi-class — what is the structural damage risk level (low / moderate / high / critical)?

All four tasks share a single encoder backbone. The system is trained on a GPU server over SSH and served for inference on CPU via a Django REST API (documented separately).

---

## Non-Negotiable Constraints

These constraints are fixed. The agent must not make decisions that violate them.

| Constraint | Value | Reason |
|---|---|---|
| Training hardware | SSH-accessible GPU server | User-owned, not cloud |
| Inference hardware | CPU only | Django REST deployment target |
| ML framework | PyTorch (standard, no custom CUDA) | Team skill level |
| Primary dataset | STEAD (Stanford Earthquake Dataset) | 1.2M labeled samples, HDF5, open access |
| Sampling rate | 100 Hz | STEAD standard — all preprocessing locked to this |
| Window length | 3000 samples (30 seconds) | Fixed at training time, frozen in config |
| P-arrival position | Sample index 1000 (10s into window) | Fixed convention — must not vary between train and inference |
| Preprocessing config | Must be versioned JSON artifact | Bundled with every model checkpoint |

---

## Project Folder Structure

The agent must create and maintain exactly this structure. Do not add top-level directories without documenting the reason in a commit message.

```
seismic_ml/
│
├── configs/
│   ├── preprocessing_config.json       # Frozen at first training run — never edited after
│   ├── model_config.json               # Architecture hyperparameters
│   └── training_config.json            # Optimizer, LR, loss weights per phase
│
├── data/
│   ├── raw/                            # Downloaded STEAD HDF5 files — never modified
│   ├── processed/                      # Preprocessed tensors cached to disk (optional)
│   └── splits/                         # event_id split files: train.txt, val.txt, test.txt
│
├── seismic/
│   ├── __init__.py
│   ├── dataset.py                      # PyTorch Dataset — reads STEAD HDF5
│   ├── preprocessing.py                # ALL signal processing lives here and only here
│   ├── augmentation.py                 # Waveform augmentation (time-shift, noise, amplitude)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py                  # ResNet stem + stages + Transformer layers
│   │   ├── heads.py                    # All 4 task heads
│   │   └── seismic_net.py              # Assembles encoder + heads into SeismicNet
│   ├── loss.py                         # Per-task losses + weighted combiner
│   ├── metrics.py                      # Per-task evaluation metrics
│   └── utils.py                        # Checkpoint save/load, config loaders
│
├── training/
│   ├── train.py                        # Main training entry point
│   ├── evaluate.py                     # Standalone evaluation on test split
│   └── export.py                       # GPU checkpoint → TorchScript + quantization
│
├── mlflow/
│   └── README.md                       # SSH tunnel instructions for MLflow UI access
│
├── tests/
│   ├── test_preprocessing.py           # Unit tests — preprocessing order and output shape
│   ├── test_model_shapes.py            # Forward pass shape assertions for every head
│   └── test_dataset.py                 # Dataset split integrity, no station leakage
│
├── requirements.txt
├── environment.yml                     # Conda env for GPU server
└── README.md
```

---

## Environment Setup

### GPU Server (training)

The agent must generate an `environment.yml` targeting this exact stack. Do not use pip-only installs for PyTorch on the GPU server — use the conda-forge + pytorch channel combination to ensure CUDA compatibility.

| Package | Version constraint | Purpose |
|---|---|---|
| Python | 3.10.x | Stable, widely supported |
| PyTorch | ≥ 2.1, with CUDA | Training backbone |
| torchaudio | Match PyTorch version | Waveform utilities |
| obspy | ≥ 1.4 | Seismic format I/O and signal processing |
| h5py | ≥ 3.9 | STEAD HDF5 reading |
| numpy | ≥ 1.24 | Array ops |
| scipy | ≥ 1.11 | Signal processing backup |
| mlflow | ≥ 2.9 | Experiment tracking |
| tqdm | any | Progress bars |
| pytest | any | Test suite |

### CPU Inference Server (Django REST)

A separate, lighter `requirements_inference.txt` must be maintained containing only:

- `torch` (CPU-only build, no CUDA)
- `numpy`
- `scipy`
- `obspy`

No training dependencies (MLflow, h5py, torchaudio) should be present on the inference server. The agent must keep these two dependency lists strictly separated.

---

## configs/preprocessing_config.json — Schema

This file is written once before the first training run and is treated as immutable thereafter. The agent must generate it with the following fields and must never change values mid-project without bumping `schema_version`.

```json
{
  "schema_version": "1.0.0",
  "sampling_rate_hz": 100,
  "window_length_samples": 3000,
  "p_arrival_sample_index": 1000,
  "bandpass_low_hz": 1.0,
  "bandpass_high_hz": 45.0,
  "bandpass_order": 4,
  "bandpass_type": "butter",
  "taper_fraction": 0.05,
  "taper_type": "cosine",
  "normalization_strategy": "per_trace_zscore",
  "pgv_extraction": "peak_abs_velocity_before_norm",
  "detrend_type": "linear",
  "channels": ["E", "N", "Z"],
  "label_schema_version": "1.0.0"
}
```

**The agent must validate this config at the start of every training run and every inference call.** If the loaded config does not match the expected schema version, raise a hard error — never silently continue with a mismatched config.

---

## data/splits/ — Event-Level Split Strategy

Splits must be generated by `scripts/generate_splits.py` (agent must create this) and must follow these rules:

- Split unit is **event ID**, not sample index.
- The same event must not appear in more than one split.
- The same seismic station must not dominate any single split — check station distribution across splits and log warnings if any station contributes more than 5% of a single split.
- Split ratios: **70% train / 15% val / 15% test**.
- Files are plain text, one event ID per line.
- Split files are committed to the repository and never regenerated mid-project.

---

## label_schema — Risk Classification Labels

Task D labels are not in STEAD directly. They must be derived during dataset construction from the following rule, which the agent must implement exactly in `dataset.py`:

| Magnitude (Ml) | Depth < 20km | Depth ≥ 20km |
|---|---|---|
| < 3.0 | low | low |
| 3.0 – 4.5 | moderate | low |
| 4.5 – 6.0 | high | moderate |
| ≥ 6.0 | critical | high |

This is a simplified proxy. The label derivation function must be unit-tested in `tests/test_dataset.py`.

---

## Key Invariants the Agent Must Enforce

These are checked by the test suite and must never be broken by a refactor:

1. `preprocessing.py` is the **only** place signal processing logic lives. The inference endpoint and the training DataLoader both call the same functions from this module — never duplicate or reimplement.
2. Every `torch.save()` call for a checkpoint must also write `preprocessing_config.json` into the same directory with a matching `schema_version`.
3. The `SeismicNet` forward pass must accept `(waveform: Tensor[B,3,3000], pgv: Tensor[B,1])` and return a dict with keys `detection`, `phase_pick`, `magnitude`, `risk` — this signature must not change.
4. Noise windows (Task A label = 0) must have their Task B, C, D losses masked to zero in `loss.py`. This is not optional.
5. All random seeds (torch, numpy, Python random) must be set from `training_config.json:seed` at the start of `train.py`.
