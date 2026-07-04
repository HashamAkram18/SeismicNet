# API Schema — SeismicNet REST Interface

## Scope

This document defines the Django REST API for serving SeismicNet inference.
The API loads the exported TorchScript artifact and exposes the four seismic
tasks (detection, phase picking, magnitude estimation, risk classification)
as REST endpoints.

Defined in `docs/05_api_schema.md`. Governs `api/seismic_api/` code.

---

## Base URL

```
http://<host>:<port>/api/v1/
```

---

## Endpoints

### POST /api/v1/seismic/analyze/

Submit a seismic waveform for analysis. Returns a `job_id` immediately
(async Celery task). Synchronous in local dev (`CELERY_TASK_ALWAYS_EAGER=True`).

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `mseed_file` | file | yes | MiniSEED waveform file (`.mseed`) |
| `stationxml_file` | file | no | StationXML metadata (`.xml`) — improves accuracy |
| `p_arrival_sample` | int | no | P-arrival sample index — if known, improves sub-window extraction |

**Response**: `202 Accepted`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "submitted_at": "2026-07-05T12:00:00Z"
}
```

**Error responses**:

| Status | Condition |
|---|---|
| 400 | Missing `mseed_file`, invalid file type, or `p_arrival_sample` out of range |
| 413 | File too large (>50MB) |
| 503 | Model not loaded (inference engine unavailable) |

---

### GET /api/v1/seismic/jobs/{job_id}/

Get job status and result (if complete).

**Response**: `200 OK`

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "submitted_at": "2026-07-05T12:00:00Z",
  "completed_at": "2026-07-05T12:00:01Z",
  "processing_time_ms": 142,
  "model_version": "1.0.0",
  "preprocessing_config_version": "1.0.0",
  "results": { ... },
  "warnings": []
}
```

**Status values**:

| Status | Description |
|---|---|
| `pending` | Job queued, waiting for Celery worker |
| `processing` | Inference in progress |
| `complete` | Results available in `results` field |
| `failed` | Error in `error_detail` field |

**Error responses**:

| Status | Condition |
|---|---|
| 404 | `job_id` not found |

---

### GET /api/v1/seismic/jobs/{job_id}/result/

Return only the result payload (no status wrapper).

**Response**: `200 OK` — the `results` object from the job.

**Error responses**:

| Status | Condition |
|---|---|
| 404 | `job_id` not found |
| 409 | Job not yet complete (status is `pending` or `processing`) |

---

### GET /api/v1/health/

Health check. Returns model load status and version info.

**Response**: `200 OK` (model loaded) or `503 Service Unavailable` (not loaded)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "preprocessing_config_version": "1.0.0",
  "risk_classes": ["low", "moderate", "high", "critical"]
}
```

---

## Result Payload Schema

The `results` field in job responses follows this structure:

```json
{
  "detection": {
    "is_seismic": true,
    "confidence": 0.97,
    "threshold_used": 0.5
  },
  "phase_picks": {
    "p_arrival_sample": 1000,
    "s_arrival_sample": 1450,
    "p_arrival_time_s": 10.0,
    "s_arrival_time_s": 14.5,
    "sp_interval_s": 4.5
  },
  "magnitude": {
    "ml": 3.2,
    "ml_log_raw": 0.623
  },
  "risk": {
    "level": "moderate",
    "class_index": 1,
    "probabilities": {
      "low": 0.12,
      "moderate": 0.71,
      "high": 0.14,
      "critical": 0.03
    }
  }
}
```

### Field descriptions

**detection**:
- `is_seismic`: `True` if `sigmoid(logit) >= threshold_used`
- `confidence`: Raw sigmoid output in [0, 1]
- `threshold_used`: Detection threshold from `model_config.json:detection_threshold`

**phase_picks**:
- `p_arrival_sample`: P-wave arrival sample index in the 3000-sample window
- `s_arrival_sample`: S-wave arrival sample index in the 3000-sample window
- `p_arrival_time_s`: P-arrival time in seconds (`sample / sampling_rate`)
- `s_arrival_time_s`: S-arrival time in seconds (`sample / sampling_rate`)
- `sp_interval_s`: S-P time interval in seconds (distance proxy)

**magnitude**:
- `ml`: Back-transformed local magnitude: `10 ** raw_output - 1`
- `ml_log_raw`: Raw model output (log10(Ml+1) scale)

**risk**:
- `level`: Human-readable risk class name
- `class_index`: Integer class index (0–3)
- `probabilities`: Softmax probabilities for all 4 classes

---

## Error Response Schema

```json
{
  "error": "invalid_file_type",
  "detail": "Expected .mseed file, got .csv",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Description |
|---|---|---|
| `error` | str | Machine-readable error code |
| `detail` | str | Human-readable explanation |
| `job_id` | str | Job ID if the error is associated with a job (optional) |

### Error codes

| Code | HTTP Status | Description |
|---|---|---|
| `missing_file` | 400 | No `mseed_file` in request |
| `invalid_file_type` | 400 | File is not MiniSEED format |
| `invalid_p_arrival` | 400 | `p_arrival_sample` is negative or exceeds waveform length |
| `file_too_large` | 413 | File exceeds 50MB limit |
| `job_not_found` | 404 | No job with given `job_id` |
| `job_not_complete` | 409 | Result requested before job finished |
| `model_not_loaded` | 503 | Inference engine not initialized |
| `inference_error` | 500 | Error during model forward pass |

---

## Data Flow

```
Client
  │
  ▼
POST /api/v1/seismic/analyze/
  │  multipart: mseed_file, stationxml_file?, p_arrival_sample?
  │
  ▼
views.py → validators.py → SeismicJob(status=pending)
  │
  ▼
tasks.py → Celery task (or sync in local dev)
  │
  ├─── inference.py: load_from_mseed(file_bytes, stationxml_bytes)
  │      → calls seismic.preprocessing.preprocess_waveform()  [Invariant 1]
  │      → returns (waveform_np, sampling_rate, p_arrival_sample, warnings)
  │
  ├─── inference.py: predict(waveform_np, sampling_rate, p_arrival_sample)
  │      → preprocess → tensor → model.forward() → post-process
  │      → returns structured result dict
  │
  ▼
SeismicJob(status=complete, result=...)
  │
  ▼
GET /api/v1/seismic/jobs/{job_id}/
  │
  ▼
Client receives full result payload
```

---

## Invariant Compliance

- **Invariant 1**: `inference.py` calls `seismic.preprocessing.preprocess_waveform()`.
  No signal processing logic is reimplemented in the API layer.
- **Invariant 2**: The exported artifact bundle includes `preprocessing_config.json`.
  `inference.py` validates `schema_version` on load.
- **Invariant 3**: The model forward signature is `(waveform, pgv, active_tasks)`.
  `inference.py` constructs tensors matching this signature.
