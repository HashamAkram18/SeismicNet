# Seismic Risk ML — Data Pipeline & Preprocessing

## Scope

This document governs everything between raw `.mseed` / STEAD HDF5 files and the model-ready tensor that enters `SeismicNet`. The agent must implement all logic described here in the exact modules specified. No signal processing logic may live outside `seismic/preprocessing.py`.

---

## Data Source — STEAD

**Primary dataset**: Stanford Earthquake Dataset (STEAD)  
**Download**: `https://github.com/smousavi05/STEAD`  
**Format**: Two HDF5 files — waveforms (`chunk1.hdf5`, `chunk2.hdf5`) and a CSV metadata file  
**Size**: ~202 GB total, ~1.2M waveform windows  
**Composition**: ~50% earthquake windows, ~50% noise windows — this imbalance is real and must not be artificially balanced

The agent must **not** download or modify raw HDF5 files. They live in `data/raw/` and are read-only. All derived artifacts go to `data/processed/` or `data/splits/`.

### HDF5 Structure (per waveform entry)
Each entry in STEAD provides:
- `waveforms`: shape `[3, 6000]` at 100 Hz — E, N, Z channels, 60 seconds
- Metadata (from CSV): `trace_start_time`, `p_arrival_sample`, `s_arrival_sample`, `source_magnitude`, `source_depth_km`, `receiver_code`, `trace_category` (`earthquake_local` or `noise`)

Note: STEAD windows are 60s (6000 samples). The agent must extract a **30s sub-window** of 3000 samples centered such that the P-arrival lands at sample index 1000 of the extracted window.

---

## seismic/preprocessing.py — Exact Pipeline

The agent must implement a `preprocess_waveform()` function in this module. The function is called identically from the training DataLoader and from the Django inference endpoint. It must never be duplicated.

### Function signature

```
preprocess_waveform(
    waveform: np.ndarray,          # shape [3, N] — raw counts or velocity
    sampling_rate: int,            # must equal config.sampling_rate_hz (100)
    config: dict,                  # loaded preprocessing_config.json
    p_arrival_sample: int,         # P-arrival sample index in the original waveform
    extract_pgv: bool = True       # compute PGV before normalization
) -> dict:
    # Returns:
    # {
    #   "waveform": np.ndarray [3, 3000],   # processed, normalized
    #   "pgv": float,                        # peak ground velocity scalar (pre-norm)
    # }
```

### Processing order — this order is locked and must not be changed

The agent must implement steps in exactly this sequence. Changing the order invalidates the preprocessing config version.

**Step 1 — Validate sampling rate**  
Assert `sampling_rate == config["sampling_rate_hz"]`. Raise `ValueError` if not. Never silently resample.

**Step 2 — Sub-window extraction**  
Extract 3000 samples from the raw waveform such that `p_arrival_sample` in the original array maps to sample index `config["p_arrival_sample_index"]` (1000) in the output window. If the required window extends before index 0 or after the array end, zero-pad symmetrically. Log a warning when padding is applied — do not silently pad.

**Step 3 — Detrend**  
Apply linear detrend along the time axis (axis=-1) per channel independently. Use `scipy.signal.detrend(type='linear')`.

**Step 4 — Taper**  
Apply a cosine taper of fraction `config["taper_fraction"]` (0.05) to both ends of each channel. Use ObsPy's `cosine_taper` or an equivalent implementation. Tapering reduces edge artifacts before filtering.

**Step 5 — Bandpass filter**  
Apply a zero-phase Butterworth bandpass filter:
- Low cutoff: `config["bandpass_low_hz"]` (1.0 Hz)
- High cutoff: `config["bandpass_high_hz"]` (45.0 Hz)
- Order: `config["bandpass_order"]` (4)
- Use `scipy.signal.sosfiltfilt` with `scipy.signal.butter(..., output='sos')` — zero-phase (filtfilt) is mandatory to avoid phase distortion that would corrupt phase picking

**Step 6 — Extract PGV scalar (before normalization)**  
If `extract_pgv=True`, compute peak ground velocity as `max(abs(waveform))` across all channels and all samples — a single scalar. Store as `pgv`. This is the amplitude feature that survives normalization and is passed to the magnitude head.

**Step 7 — Per-trace z-score normalization**  
For each of the 3 channels independently:
- Compute `mean` and `std` of that channel's 3000 samples
- If `std < 1e-10` (flat channel), set the channel to all zeros and log a warning
- Otherwise apply `(x - mean) / std`

**Step 8 — Return**  
Return `{"waveform": array [3, 3000] float32, "pgv": float32}`.

### What must NOT happen in preprocessing.py

- No model calls
- No file I/O (reading HDF5 is done in `dataset.py` before calling this function)
- No label derivation
- No augmentation (augmentation is applied in `augmentation.py` after preprocessing)
- No instrument response removal (STEAD waveforms are already instrument-corrected velocity — do not apply response removal to STEAD data; only apply when processing raw user-uploaded `.mseed` files from unknown stations)

---

## seismic/dataset.py — PyTorch Dataset

The agent must implement a `SeismicDataset(torch.utils.data.Dataset)` class with the following behavior.

### Constructor arguments

```
SeismicDataset(
    hdf5_paths: list[str],          # paths to chunk1.hdf5, chunk2.hdf5
    metadata_csv: str,              # path to STEAD metadata CSV
    split_file: str,                # path to data/splits/train.txt (or val/test)
    preprocessing_config: dict,     # loaded preprocessing_config.json
    label_schema_config: dict,      # loaded label_schema config
    augment: bool = False           # only True for training split
)
```

### __getitem__ behavior

1. Look up event ID from the split file index
2. Load waveform `[3, 6000]` from HDF5 by event ID
3. Load metadata row from CSV (p_arrival_sample, s_arrival_sample, magnitude, depth, station, category)
4. Call `preprocessing.preprocess_waveform()` → get `waveform [3, 3000]` and `pgv` scalar
5. If `augment=True`, call `augmentation.augment_waveform()` on the preprocessed waveform
6. Derive Task D risk label from magnitude + depth using the label table in `01_project_overview.md`
7. Compute normalized phase pick targets:
   - `p_pick_normalized = (config["p_arrival_sample_index"]) / config["window_length_samples"]` — this is always 1000/3000 = 0.333 by construction (P is always at index 1000)
   - `s_pick_normalized = (s_arrival_sample - (original_p - 1000)) / 3000` — S relative to the extracted window start; clamp to [0, 1]
8. Return a dict:

```python
{
    "waveform": Tensor[3, 3000],        # float32
    "pgv": Tensor[1],                   # float32 scalar
    "label_detection": Tensor[1],       # float32, 0.0 or 1.0
    "label_phase_pick": Tensor[2],      # float32, [p_norm, s_norm]
    "label_magnitude": Tensor[1],       # float32, Ml value (NaN for noise windows)
    "label_risk": Tensor[1],            # long, class index 0–3 (-1 for noise windows)
    "is_noise": Tensor[1],              # bool — used for loss masking
    "event_id": str,                    # for debugging only, not used in loss
}
```

### DataLoader configuration

The agent must configure `torch.utils.data.DataLoader` with:

```
DataLoader(
    dataset,
    batch_size=256,            # reduce to 128 if GPU OOM
    shuffle=True,              # training only; False for val/test
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    drop_last=True             # avoids batch norm issues with tiny final batches
)
```

---

## seismic/augmentation.py — Augmentation

Augmentation is applied **after preprocessing** and **only during training**. It operates on the preprocessed, normalized waveform tensor. It must not alter the `pgv` scalar (PGV was extracted from the raw signal before normalization).

### Augmentations to implement

**Time shift** (apply with probability 0.5)  
Shift the waveform along the time axis by a random offset drawn from `Uniform(-50, +50)` samples. Correspondingly shift the phase pick labels by the same offset. Clamp pick labels to [0, 1] after shift. Zero-pad the vacated end.

**Amplitude scaling** (apply with probability 0.5)  
Multiply the entire waveform by a scalar drawn from `Uniform(0.5, 2.0)`. This does not affect normalized waveform quality but simulates distance variation. Do not update `pgv` — it is already extracted.

**Additive Gaussian noise** (apply with probability 0.3)  
Add Gaussian noise with std drawn from `Uniform(0.005, 0.02)` of the waveform's per-channel std. Simulates instrument noise floor.

**Channel dropout** (apply with probability 0.1)  
Zero out one randomly selected channel (E, N, or Z). Simulates a dead sensor. Only apply to earthquake windows — never to noise windows (noise window augmentation is irrelevant and wastes compute).

### What augmentation must not do

- Must not alter `label_detection` or `label_risk`
- Must not augment val or test splits — `augment=False` on those DataLoader instances
- Must not be applied inside `preprocessing.py`

---

## Handling User-Uploaded .mseed Files (Inference Path)

When a user submits a `.mseed` file via the Django API, the preprocessing path differs from STEAD in one way: **instrument response removal must be applied** if the user provides a StationXML response file.

The agent must implement a separate `preprocess_mseed()` function in `preprocessing.py` that:

1. Reads the `.mseed` file using `obspy.read()`
2. Merges all traces into a 3-component stream
3. If a StationXML response file is provided, removes instrument response to velocity using `stream.remove_response(output='VEL')`
4. Resamples to 100 Hz if the input sampling rate differs (using `stream.resample(100)`)
5. Extracts the 3-component array `[3, N]`
6. Calls `preprocess_waveform()` with the known or estimated P-arrival sample

If no P-arrival is known (user hasn't picked it), the agent must default to centering the 30s window on the largest amplitude peak and logging a warning that phase picks may be less accurate.

---

## Preprocessing Unit Tests — tests/test_preprocessing.py

The agent must implement all of the following tests. They must pass before any training run is launched.

| Test | What it checks |
|---|---|
| `test_output_shape` | `preprocess_waveform()` always returns `[3, 3000]` regardless of input length |
| `test_p_arrival_position` | After preprocessing, a known P-arrival is at sample index 1000 |
| `test_zscore_mean` | Each output channel has mean ≈ 0.0 (tolerance 1e-5) |
| `test_zscore_std` | Each output channel has std ≈ 1.0 (tolerance 1e-3) |
| `test_flat_channel` | A flat (zero) input channel returns all zeros without raising |
| `test_filter_order` | Detrend is applied before filter (test by checking output differs from filter-first) |
| `test_pgv_positive` | PGV scalar is always ≥ 0.0 |
| `test_config_mismatch_raises` | Passing wrong sampling rate raises `ValueError` |
| `test_no_leakage_across_splits` | No event ID appears in more than one split file |
| `test_station_distribution` | No station contributes >5% of any single split |
