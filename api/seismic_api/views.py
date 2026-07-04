"""API views for seismic analysis endpoints."""
from __future__ import annotations

import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from seismic_api.inference import get_engine
from seismic_api.models import SeismicJob
from seismic_api.serializers import (
    ErrorResponseSerializer,
    HealthResponseSerializer,
    JobResponseSerializer,
    AnalyzeResponseSerializer,
)
from seismic_api.tasks import run_seismic_analysis
from seismic_api.validators import (
    validate_mseed_file,
    validate_p_arrival,
    validate_stationxml_file,
)

logger = logging.getLogger(__name__)


class HealthView(APIView):
    """GET /api/v1/health/ — Model health check."""

    def get(self, request: Request) -> Response:
        """Return model load status and version info."""
        try:
            engine = get_engine(settings.ARTIFACT_DIR)
            data = {
                "status": "healthy",
                "model_loaded": True,
                "model_version": engine.model_version,
                "preprocessing_config_version": engine.preprocessing_config_version,
                "risk_classes": engine.risk_classes,
            }
            return Response(data, status=status.HTTP_200_OK)
        except Exception:
            data = {
                "status": "unhealthy",
                "model_loaded": False,
                "model_version": None,
                "preprocessing_config_version": None,
                "risk_classes": [],
            }
            return Response(data, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class AnalyzeView(APIView):
    """POST /api/v1/seismic/analyze/ — Submit waveform for analysis."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: Request) -> Response:
        """Accept mseed upload and return job_id."""
        logger.info("Analyze request received: filename=%s, size=%s", request.FILES.get("mseed_file", "N/A"), getattr(request.FILES.get("mseed_file"), "size", "N/A"))

        # Validate mseed file
        mseed_file = request.FILES.get("mseed_file")
        if mseed_file is None:
            return Response(
                {"error": "missing_file", "detail": "mseed_file is required", "job_id": None},
                status=status.HTTP_400_BAD_REQUEST,
            )

        error = validate_mseed_file(mseed_file.name, mseed_file.size)
        if error is not None:
            logger.warning("Mseed validation failed: %s (file=%s)", error, mseed_file.name)
            return Response(
                {"error": error, "detail": f"File validation failed: {error}", "job_id": None},
                status=status.HTTP_400_BAD_REQUEST
                if error != "file_too_large"
                else status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        # Validate optional StationXML
        stationxml_file = request.FILES.get("stationxml_file")
        if stationxml_file is not None:
            sx_error = validate_stationxml_file(stationxml_file.name, stationxml_file.size)
            if sx_error is not None:
                return Response(
                    {"error": sx_error, "detail": f"StationXML validation failed: {sx_error}", "job_id": None},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Validate optional p_arrival_sample
        p_arrival_sample = request.data.get("p_arrival_sample")
        if p_arrival_sample is not None:
            try:
                p_arrival_sample = int(p_arrival_sample)
            except (TypeError, ValueError):
                return Response(
                    {"error": "invalid_p_arrival", "detail": "p_arrival_sample must be an integer", "job_id": None},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            pa_error = validate_p_arrival(p_arrival_sample)
            if pa_error is not None:
                return Response(
                    {"error": pa_error, "detail": f"p_arrival_sample out of range", "job_id": None},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Check model is loaded
        try:
            engine = get_engine(settings.ARTIFACT_DIR)
        except Exception:
            return Response(
                {"error": "model_not_loaded", "detail": "Inference engine not available", "job_id": None},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Parse mseed to get waveform info for the job record
        try:
            mseed_bytes = mseed_file.read()
            stationxml_bytes = stationxml_file.read() if stationxml_file else None
            waveform, sampling_rate, detected_p_arrival, warnings = engine.load_from_mseed(
                mseed_bytes, stationxml_bytes
            )
        except Exception as e:
            return Response(
                {"error": "invalid_file_type", "detail": f"Failed to parse MiniSEED: {e}", "job_id": None},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Use user-provided p_arrival_sample if given, else detected
        final_p_arrival = p_arrival_sample if p_arrival_sample is not None else detected_p_arrival
        if final_p_arrival is None:
            final_p_arrival = engine._preprocessing_config.get("p_arrival_sample_index", 1000)
            warnings.append("No P-arrival provided or detected; using default")

        # Create job
        job = SeismicJob(
            status="pending",
            input_filename=mseed_file.name,
            model_version=engine.model_version,
            preprocessing_config_version=engine.preprocessing_config_version,
        )
        job.save()
        logger.info("Job created: %s (file=%s)", job.job_id, mseed_file.name)

        # Save waveform to temp file for Celery task
        tmp_dir = Path(tempfile.gettempdir()) / "seismic_jobs"
        tmp_dir.mkdir(exist_ok=True)
        waveform_path = str(tmp_dir / f"{job.job_id}.npy")
        np.save(waveform_path, waveform)

        # Store metadata for the task
        input_metadata = {
            "waveform_path": waveform_path,
            "sampling_rate": sampling_rate,
            "p_arrival_sample": final_p_arrival,
        }
        job.result = json.dumps(input_metadata)
        job.save(update_fields=["result"])

        # Dispatch Celery task
        run_seismic_analysis.delay(str(job.job_id))

        logger.info("Job dispatched: %s", job.job_id)
        return Response(
            {
                "job_id": str(job.job_id),
                "status": "pending",
                "submitted_at": job.submitted_at.isoformat(),
            },
            status=status.HTTP_202_ACCEPTED,
        )


class JobStatusView(APIView):
    """GET /api/v1/seismic/jobs/{job_id}/ — Job status and result."""

    def get(self, request: Request, job_id: uuid.UUID) -> Response:
        """Return job status, including full result if complete."""
        try:
            job = SeismicJob.objects.get(job_id=job_id)
        except SeismicJob.DoesNotExist:
            return Response(
                {"error": "job_not_found", "detail": f"Job {job_id} not found", "job_id": str(job_id)},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Build response
        result_data = None
        warnings = []
        model_version = job.model_version
        preprocessing_config_version = job.preprocessing_config_version

        if job.status == "complete" and job.result:
            stored = job.result if isinstance(job.result, dict) else json.loads(job.result)
            result_data = stored.get("results")
            warnings = stored.get("warnings", [])
            model_version = stored.get("model_version", model_version)
            preprocessing_config_version = stored.get("preprocessing_config_version", preprocessing_config_version)

        data = {
            "job_id": str(job.job_id),
            "status": job.status,
            "submitted_at": job.submitted_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time_ms": job.processing_time_ms,
            "model_version": model_version,
            "preprocessing_config_version": preprocessing_config_version,
            "results": result_data,
            "warnings": warnings,
        }

        return Response(data, status=status.HTTP_200_OK)


class JobResultView(APIView):
    """GET /api/v1/seismic/jobs/{job_id}/result/ — Result payload only."""

    def get(self, request: Request, job_id: uuid.UUID) -> Response:
        """Return only the results dict if job is complete."""
        try:
            job = SeismicJob.objects.get(job_id=job_id)
        except SeismicJob.DoesNotExist:
            return Response(
                {"error": "job_not_found", "detail": f"Job {job_id} not found", "job_id": str(job_id)},
                status=status.HTTP_404_NOT_FOUND,
            )

        if job.status != "complete":
            return Response(
                {
                    "error": "job_not_complete",
                    "detail": f"Job status is '{job.status}', not 'complete'",
                    "job_id": str(job_id),
                },
                status=status.HTTP_409_CONFLICT,
            )

        if not job.result:
            return Response(
                {"error": "job_not_complete", "detail": "No result available", "job_id": str(job_id)},
                status=status.HTTP_409_CONFLICT,
            )

        stored = job.result if isinstance(job.result, dict) else json.loads(job.result)
        return Response(stored.get("results"), status=status.HTTP_200_OK)
