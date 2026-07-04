"""Celery tasks for seismic analysis."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=1)
def run_seismic_analysis(self: Any, job_id: str) -> None:
    """Run seismic inference on a submitted job.

    This task is executed synchronously in local dev (CELERY_TASK_ALWAYS_EAGER=True)
    and asynchronously in production with Redis.

    Args:
        job_id: UUID string of the SeismicJob to process.
    """
    from seismic_api.models import SeismicJob

    logger.info("Task start: job_id=%s", job_id)

    job = SeismicJob.objects.get(job_id=job_id)
    job.status = "processing"
    job.save(update_fields=["status"])

    try:
        # Import engine — loaded once at startup via AppConfig.ready()
        from django.conf import settings
        from seismic_api.inference import get_engine

        engine = get_engine(settings.ARTIFACT_DIR)

        # Load waveform from stored file bytes
        # The view stores raw file bytes in the job model for the task to pick up
        # We need to read the file from a temporary location
        import json

        # Read the stored input metadata
        input_data = json.loads(job.result) if job.result else {}
        waveform_path = input_data.get("waveform_path")

        if waveform_path is None:
            raise ValueError("No waveform data stored for job")

        # Load the saved waveform
        import numpy as np

        waveform_data = np.load(waveform_path)
        sampling_rate = int(input_data.get("sampling_rate", 100))
        p_arrival_sample = input_data.get("p_arrival_sample")

        t0 = time.perf_counter()
        result = engine.predict(
            waveform_np=waveform_data,
            sampling_rate=sampling_rate,
            p_arrival_sample=p_arrival_sample,
        )
        processing_time_ms = int((time.perf_counter() - t0) * 1000)

        # Clean up temporary file
        import os
        if waveform_path and os.path.exists(waveform_path):
            os.remove(waveform_path)

        # Update job with results
        job.status = "complete"
        job.completed_at = datetime.now(timezone.utc)
        job.processing_time_ms = processing_time_ms
        # Store only the results dict, not the full predict output
        job.result = {
            "results": result["results"],
            "warnings": result["warnings"],
            "model_version": result["model_version"],
            "preprocessing_config_version": result["preprocessing_config_version"],
        }
        job.save(update_fields=[
            "status", "completed_at", "processing_time_ms", "result",
        ])

        logger.info("Task complete: job_id=%s, processing_time=%dms", job_id, processing_time_ms)

    except Exception as e:
        logger.error("Task failure: job_id=%s, error=%s", job_id, e, exc_info=True)
        job.status = "failed"
        job.error_detail = str(e)[:1024]
        job.completed_at = datetime.now(timezone.utc)
        job.save(update_fields=["status", "error_detail", "completed_at"])
