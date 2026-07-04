"""Tests for API views using Django test client.

Inference engine tests mock the model to avoid needing a real checkpoint.
View tests use Django's test client with Celery eager mode.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import Client, TestCase, override_settings

from seismic_api.models import SeismicJob
from seismic_api.inference import reset_engine
from tests.test_inference import _create_dummy_artifact


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestHealthEndpoint(TestCase):
    """Tests for GET /api/v1/health/."""

    def setUp(self) -> None:
        reset_engine()

    def tearDown(self) -> None:
        reset_engine()

    def test_health_endpoint_200_or_503(self) -> None:
        """Health returns 200 when model loaded, 503 when not."""
        client = Client()
        response = client.get("/api/v1/health/")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_endpoint_503_when_no_model(self) -> None:
        """Health returns 503 when model is not loaded."""
        client = Client()
        with override_settings(ARTIFACT_DIR="/nonexistent/path"):
            response = client.get("/api/v1/health/")
            assert response.status_code == 503
            data = response.json()
            assert data["model_loaded"] is False


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestAnalyzeEndpoint(TestCase):
    """Tests for POST /api/v1/seismic/analyze/."""

    def test_analyze_missing_file(self) -> None:
        """Analyze returns 400 when no mseed_file is provided."""
        client = Client()
        response = client.post(
            "/api/v1/seismic/analyze/",
            {},
            format="multipart",
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "missing_file"

    def test_analyze_rejects_non_mseed(self) -> None:
        """Analyze rejects non-mseed file uploads."""
        client = Client()
        fake_file = io.BytesIO(b"col1,col2\n1,2")
        fake_file.name = "data.csv"
        response = client.post(
            "/api/v1/seismic/analyze/",
            {"mseed_file": fake_file},
            format="multipart",
        )
        assert response.status_code == 400

    def test_analyze_accepts_valid_upload(self) -> None:
        """Analyze accepts valid upload and returns 202 + job_id."""
        client = Client()
        fake_mseed = io.BytesIO(b"\x00" * 100)
        fake_mseed.name = "test.mseed"

        with patch("seismic_api.views.get_engine") as mock_get_engine, \
             patch("seismic_api.views.run_seismic_analysis") as mock_task:
            mock_engine = MagicMock()
            mock_engine.model_version = "1.0.0"
            mock_engine.preprocessing_config_version = "1.0.0"
            mock_engine._preprocessing_config = {"p_arrival_sample_index": 1000}
            mock_engine.load_from_mseed.return_value = (
                np.random.randn(3, 3000).astype(np.float32),
                100,
                1000,
                [],
            )
            mock_get_engine.return_value = mock_engine
            mock_task.delay.return_value = None

            response = client.post(
                "/api/v1/seismic/analyze/",
                {"mseed_file": fake_mseed},
                format="multipart",
            )

            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "pending"

    def test_analyze_model_not_loaded(self) -> None:
        """Analyze returns 503 when model is not loaded."""
        client = Client()
        fake_mseed = io.BytesIO(b"\x00" * 100)
        fake_mseed.name = "test.mseed"

        with patch("seismic_api.views.get_engine", side_effect=FileNotFoundError):
            response = client.post(
                "/api/v1/seismic/analyze/",
                {"mseed_file": fake_mseed},
                format="multipart",
            )
            assert response.status_code == 503


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestJobStatusEndpoint(TestCase):
    """Tests for GET /api/v1/seismic/jobs/{job_id}/."""

    def test_job_status_pending(self) -> None:
        """Job status returns pending for newly created job."""
        job = SeismicJob(
            status="pending",
            input_filename="test.mseed",
            model_version="1.0.0",
            preprocessing_config_version="1.0.0",
        )
        job.save()

        client = Client()
        response = client.get(f"/api/v1/seismic/jobs/{job.job_id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["job_id"] == str(job.job_id)

    def test_job_status_complete_has_result(self) -> None:
        """Job status returns full result when complete."""
        result_payload = {
            "results": {
                "detection": {"is_seismic": True, "confidence": 0.95, "threshold_used": 0.5},
                "phase_picks": {
                    "p_arrival_sample": 1000,
                    "s_arrival_sample": 1450,
                    "p_arrival_time_s": 10.0,
                    "s_arrival_time_s": 14.5,
                    "sp_interval_s": 4.5,
                },
                "magnitude": {"ml": 3.2, "ml_log_raw": 0.623},
                "risk": {
                    "level": "moderate",
                    "class_index": 1,
                    "probabilities": {"low": 0.1, "moderate": 0.7, "high": 0.15, "critical": 0.05},
                },
            },
            "warnings": [],
            "model_version": "1.0.0",
            "preprocessing_config_version": "1.0.0",
        }

        job = SeismicJob(
            status="complete",
            input_filename="test.mseed",
            model_version="1.0.0",
            preprocessing_config_version="1.0.0",
            result=result_payload,
            processing_time_ms=150,
        )
        job.save()

        client = Client()
        response = client.get(f"/api/v1/seismic/jobs/{job.job_id}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["results"] is not None
        assert data["results"]["detection"]["is_seismic"] is True
        assert data["processing_time_ms"] == 150

    def test_job_not_found(self) -> None:
        """Job status returns 404 for nonexistent job."""
        client = Client()
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/seismic/jobs/{fake_id}/")
        assert response.status_code == 404


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestJobResultEndpoint(TestCase):
    """Tests for GET /api/v1/seismic/jobs/{job_id}/result/."""

    def test_result_not_complete(self) -> None:
        """Result endpoint returns 409 when job is not complete."""
        job = SeismicJob(
            status="pending",
            input_filename="test.mseed",
            model_version="1.0.0",
            preprocessing_config_version="1.0.0",
        )
        job.save()

        client = Client()
        response = client.get(f"/api/v1/seismic/jobs/{job.job_id}/result/")
        assert response.status_code == 409

    def test_result_complete(self) -> None:
        """Result endpoint returns results dict when job is complete."""
        result_payload = {
            "results": {
                "detection": {"is_seismic": False, "confidence": 0.1, "threshold_used": 0.5},
                "phase_picks": {
                    "p_arrival_sample": 900,
                    "s_arrival_sample": 1200,
                    "p_arrival_time_s": 9.0,
                    "s_arrival_time_s": 12.0,
                    "sp_interval_s": 3.0,
                },
                "magnitude": {"ml": 1.5, "ml_log_raw": 0.398},
                "risk": {
                    "level": "low",
                    "class_index": 0,
                    "probabilities": {"low": 0.8, "moderate": 0.15, "high": 0.04, "critical": 0.01},
                },
            },
            "warnings": [],
            "model_version": "1.0.0",
            "preprocessing_config_version": "1.0.0",
        }

        job = SeismicJob(
            status="complete",
            input_filename="test.mseed",
            model_version="1.0.0",
            preprocessing_config_version="1.0.0",
            result=result_payload,
        )
        job.save()

        client = Client()
        response = client.get(f"/api/v1/seismic/jobs/{job.job_id}/result/")
        assert response.status_code == 200
        data = response.json()
        assert "detection" in data
        assert "phase_picks" in data
        assert "magnitude" in data
        assert "risk" in data
