# SeismicNet — Project State

## Current Phase

**Step 8 of 19 — Django REST API Layer Complete**

## What Was Completed This Session

1. Wrote `docs/05_api_schema.md` — full API schema:
   - POST /api/v1/seismic/analyze/ (async, returns 202 + job_id)
   - GET /api/v1/seismic/jobs/{job_id}/ (status + result)
   - GET /api/v1/seismic/jobs/{job_id}/result/ (result only)
   - GET /api/v1/health/ (model load status, 200/503)
   - Full result payload schema with all 4 task outputs
   - Error response schema with error codes

2. Created Django project scaffold under `api/`:
   - api_config/settings/{base,local,production}.py
   - api_config/urls.py, wsgi.py, celery.py
   - seismic_api/ app with models, views, tasks, inference, validators, serializers

3. Implemented `seismic_api/models.py` — SeismicJob:
   - UUID primary key, status choices, JSONField for results
   - submitted_at, completed_at, processing_time_ms, error_detail

4. Implemented `seismic_api/inference.py` — SeismicInferenceEngine:
   - Singleton pattern with threading.Lock
   - Loads TorchScript model via torch.jit.load()
   - Validates preprocessing_config schema_version on load
   - predict() calls seismic.preprocessing.preprocess_waveform() (Invariant 1)
   - Post-processes all 4 heads: sigmoid, denormalize, back-transform, softmax
   - load_from_mseed() parses MiniSEED via obspy
   - get_engine() / reset_engine() for singleton management

5. Implemented views, serializers, tasks, validators:
   - AnalyzeView: multipart upload, validation, Celery task dispatch
   - JobStatusView: full job status with result
   - JobResultView: result-only endpoint
   - HealthView: model load check
   - run_seismic_analysis Celery task: processes job asynchronously
   - CELERY_TASK_ALWAYS_EAGER=True in local settings (no Redis needed)

6. Wrote tests:
   - test_inference.py: 7 tests (engine load, predict keys, detection range, phase picks, magnitude back-transform, risk probs, singleton)
   - test_views.py: 11 tests (health 200/503, analyze accept/reject/missing, job status pending/complete/not-found, result not-complete/complete)
   - All 20 API tests passing

## Graphify State

- Last rebuild: 2026-07-04 (765 nodes, 1160 edges, 65 communities)

## Next Action

Run the API locally: `cd api && python manage.py runserver`
Or deploy to production: `cd api && docker-compose up`

## Open Issues

- No open issues

## Test Status

- `tests/test_preprocessing.py`: 24/24 passing
- `tests/test_dataset.py`: 15/15 passing, 3 skipped (HDF5)
- `tests/test_model_shapes.py`: 16/16 passing
- `tests/test_training.py`: 6/6 passing
- `api/tests/test_inference.py`: 9/9 passing
- `api/tests/test_views.py`: 11/11 passing
- Total: 81/81 passing, 3 skipped
