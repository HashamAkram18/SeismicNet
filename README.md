# SeismicNet

**Multi-task deep learning for seismic risk analysis.**

SeismicNet ingests 3-component seismic waveforms and simultaneously outputs event detection, P/S phase picks, magnitude estimate, and structural damage risk classification. A shared encoder (ResNet + Transformer) feeds four task-specific heads, trained on the STEAD dataset (1.2M samples) and served via a Django REST API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Evaluation and Export](#evaluation-and-export)
- [API Server](#api-server)
- [Configuration Reference](#configuration-reference)
- [Design Invariants](#design-invariants)
- [Testing](#testing)

---

## Overview

SeismicNet is a multi-task deep learning system designed for real-time seismic risk analysis. The system processes raw 3-component seismic waveforms and produces four simultaneous outputs from a single forward pass:

| Task | Output | Head Input |
|---|---|---|
| **Detection** | Event vs. noise classification | Pooled features |
| **Phase Picking** | P-wave and S-wave arrival times | Sequence features |
| **Magnitude Estimation** | Event magnitude (regression) | Pooled features + PGV |
| **Risk Classification** | Structural damage risk class | Pooled features |

The model is trained on GPU over SSH and served on CPU via a Django REST API with JWT authentication, asynchronous job processing, and Swagger/ReDoc documentation.

### Key Features

- Shared encoder with four task-specific heads in a single forward pass
- Event-level data splitting (70/15/15 train/val/test) to prevent station leakage
- Three-phase training schedule (A+B, then A+B+C, then A+B+C+D) with fresh optimizer resets
- Noise-aware loss masking (noise windows suppress phase picking, magnitude, and risk losses)
- TorchScript export with optional quantization for CPU inference
- Async REST API with Celery job queue and Redis broker

---

## Architecture

```
Input: waveform [B, 3, 3000] + PGV [B, 1]
                    |
         +----------v-----------+
         |    Shared Encoder     |
         |  ResNet (stem+4       |
         |  stages, skip conn.)  |
         |  + Transformer        |
         |  (d=256, 8 heads,     |
         |   2 layers)           |
         +----------+-----------+
                    |
         +----------+-----------+
         |                      |
   sequence_features       pooled_features
     [B, 256, 375]           [B, 256]
         |                      |
    +----+----+         +------+------+------+------+
    |         |         |      |      |      |      |
    v         v         v      v      v      v      v
 PhasePick  ...      Det.   Det.   Mag.   Risk   ...
Head reads            Head   Head   Head   Head
sequence_features     |      |      |      |
                      v      v      v      v
                  pooled_features (+ PGV for Magnitude)
```

**Encoder details:**
- ResNet stem followed by 4 stages with residual skip connections
- Transformer block: `d_model=256`, 8 attention heads, 2 layers
- Outputs both `sequence_features [B, 256, 375]` and `pooled_features [B, 256]`
- PhasePickHead reads from sequence features using soft-argmax attention pointer
- Detection, Magnitude, and Risk heads read from pooled features
- Magnitude head receives `concat([pooled_features, pgv]) = [B, 257]`

**Training phases:**
1. Tasks A + B (Detection + Phase Picking)
2. Tasks A + B + C (add Magnitude)
3. Tasks A + B + C + D (add Risk)

Each phase boundary triggers a fresh AdamW optimizer and LR scheduler warmup reset.

---

## Project Structure

```
seismic_ml/
├── configs/
│   ├── preprocessing_config.json   # Frozen 7-step preprocessing pipeline config
│   ├── model_config.json           # Architecture hyperparameters
│   └── training_config.json        # Optimizer, LR, loss weights, seed, phase config
├── data/
│   ├── raw/                        # Downloaded STEAD HDF5 files
│   ├── splits/                     # Event-level split files (train/val/test)
│   └── dummy_waveform.mseed        # Sample MiniSEED file for testing
├── seismic/
│   ├── preprocessing.py            # ALL signal processing (7-step frozen pipeline)
│   ├── dataset.py                  # PyTorch Dataset for STEAD HDF5
│   ├── augmentation.py             # Waveform augmentation (training only)
│   ├── model/
│   │   ├── encoder.py              # ResNet stem + stages + Transformer
│   │   ├── heads.py                # Detection, PhasePick, Magnitude, Risk heads
│   │   └── seismic_net.py          # Assembler (shared encoder + 4 heads)
│   ├── loss.py                     # Per-task losses with noise masking
│   ├── metrics.py                  # Per-task evaluation metrics
│   └── utils.py                    # Checkpoint save/load, config loaders
├── training/
│   ├── train.py                    # CLI training entry point
│   ├── evaluate.py                 # Standalone test-set evaluation
│   └── export.py                   # TorchScript export + quantization
├── api/
│   ├── api_config/                 # Django project (settings, urls, wsgi, celery)
│   ├── seismic_api/                # REST API app (views, inference, auth, tasks)
│   ├── templates/dashboard.html    # Waveform visualization frontend
│   └── pyproject.toml              # Dependencies for API server (uv/pip)
├── tests/                          # ML unit tests
├── scripts/                        # Utility scripts
├── docs/                           # Detailed module specifications
├── environment.yml                 # Conda environment for GPU server
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training) or CPU-only (for inference)
- conda or uv package manager
- Redis (for Celery job queue in API)

### Installation

**GPU training environment:**

```bash
conda env create -f environment.yml
conda activate seismicnet
```

**API server environment:**

```bash
cd api
pip install -e .
```

### Download the STEAD Dataset

Place the STEAD HDF5 files in `data/raw/`. The dataset contains 1.2M seismic samples at 100 Hz with 3 components (E, N, Z).

---

## Data Pipeline

### Preprocessing (7-step frozen pipeline)

All signal processing is centralized in `seismic/preprocessing.py`. This pipeline is frozen and changing the order requires bumping `schema_version` in the config and retraining from scratch.

1. **Validate sampling rate** -- assert 100 Hz, raise `ValueError` if not
2. **Sub-window extraction** -- P-arrival centered at sample index 1000
3. **Detrend** -- linear detrend per channel (`scipy.signal.detrend`)
4. **Taper** -- 5% cosine taper, both ends, per channel
5. **Bandpass filter** -- 1--45 Hz, Butterworth order 4, zero-phase (`sosfiltfilt`)
6. **Extract PGV** -- max absolute amplitude across all channels (before normalization)
7. **Z-score normalization** -- per-trace, per-channel independently

### Data Splits

Splits are performed by **event ID** (not by random sample) to prevent station leakage:

- Train: 70%
- Validation: 15%
- Test: 15%

Split files are stored in `data/splits/` and reference event IDs from the STEAD dataset.

### Risk Labels

Structural damage risk classes are derived from magnitude and depth:

| Class | Description |
|---|---|
| Low | Small magnitude, shallow depth |
| Moderate | Intermediate parameters |
| High | Large magnitude or deep events |
| Critical | Extreme magnitude/depth combinations |

---

## Training

### Run Training

```bash
python training/train.py \
  --config configs/training_config.json \
  --data-dir data/ \
  --splits-dir data/splits
```

### Training Configuration

Training is controlled by `configs/training_config.json`, which specifies:

- Optimizer parameters (AdamW)
- Learning rate schedule with warmup
- Loss weights per task per phase
- Random seed (applied to torch, numpy, Python random)
- Three-phase schedule: A+B, A+B+C, A+B+C+D

### MLflow Tracking

Experiments are tracked with MLflow. Metrics, parameters, and checkpoints are logged automatically during training.

### Checkpointing

Every checkpoint save also writes `preprocessing_config.json` into the same directory with a matching `schema_version`, ensuring reproducibility of the preprocessing pipeline used during training.

---

## Evaluation and Export

### Evaluate on Test Set

```bash
python training/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --data-dir data/ \
  --splits-dir data/splits
```

### Export to TorchScript

```bash
python training/export.py \
  --checkpoint checkpoints/best.pt \
  --output-dir exported_models/seismic_v1
```

Export produces a TorchScript model with optional quantization applied to ResNet Conv1d and Linear layers (Transformer remains float32).

**Important:** Export uses `torch.jit.script`, not `torch.jit.trace`.

---

## API Server

### Local Development

```bash
cd api
python manage.py runserver 8080
```

### Docker

```bash
cd api
docker-compose up
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/seismic/analyze/` | Submit waveform for async analysis (returns 202 + job_id) |
| GET | `/api/v1/seismic/jobs/{job_id}/` | Poll job status and result |
| GET | `/api/v1/seismic/jobs/{job_id}/result/` | Retrieve result only |
| GET | `/api/v1/health/` | Model load status |
| POST | `/api/v1/auth/register/` | Create account |
| POST | `/api/v1/auth/login/` | Obtain JWT tokens |
| GET | `/api/v1/auth/profile/` | User profile (JWT required) |
| GET | `/swagger/` | Swagger UI |
| GET | `/redoc/` | ReDoc UI |

### Authentication

JWT authentication is required for the `/auth/profile/` endpoint. Obtain tokens via `/auth/login/` and include the `Authorization: Bearer <token>` header.

---

## Configuration Reference

### `configs/preprocessing_config.json`

Controls the frozen preprocessing pipeline. Contains `schema_version` for reproducibility tracking. Changing pipeline order requires a version bump and full retraining.

### `configs/model_config.json`

Architecture hyperparameters: encoder depths, transformer dimensions, head configurations.

### `configs/training_config.json`

Training hyperparameters: optimizer settings, learning rate, loss weights per phase, random seed, phase definitions.

---

## Design Invariants

These constraints are enforced throughout the codebase and must not be violated:

1. **Single preprocessing module** -- All signal processing logic lives exclusively in `seismic/preprocessing.py`. The inference endpoint and training DataLoader import the same functions from this module.

2. **Checkpoint config pairing** -- Every `torch.save()` for a checkpoint also writes `preprocessing_config.json` into the same directory with a matching `schema_version`.

3. **Fixed forward signature** -- `SeismicNet.forward()` accepts `(waveform, pgv, active_tasks)` and returns a dict with keys `detection`, `phase_pick`, `magnitude`, `risk`. This signature is not subject to change.

4. **Noise-aware loss masking** -- When a batch contains noise windows (task A label = 0), losses for tasks B, C, and D are masked to zero via the `is_noise` batch key.

5. **Config-driven seeds** -- All random seeds (torch, numpy, Python random) are set from `training_config.json:seed` at the start of `train.py`. Seed values are never hardcoded in any module.

---

## Testing

```bash
pytest tests/ api/tests/ -v
```

Tests cover:

- ML pipeline unit tests (preprocessing, dataset, model forward pass, loss, metrics)
- API endpoint tests (authentication, inference, job management)

---

## License

MIT
