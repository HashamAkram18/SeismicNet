"""DRF serializers for seismic API requests and responses."""
from __future__ import annotations

from rest_framework import serializers


class AnalyzeResponseSerializer(serializers.Serializer):
    """Response for POST /api/v1/seismic/analyze/."""

    job_id = serializers.UUIDField()
    status = serializers.CharField()
    submitted_at = serializers.DateTimeField()


class DetectionResultSerializer(serializers.Serializer):
    is_seismic = serializers.BooleanField()
    confidence = serializers.FloatField()
    threshold_used = serializers.FloatField()


class PhasePicksResultSerializer(serializers.Serializer):
    p_arrival_sample = serializers.IntegerField()
    s_arrival_sample = serializers.IntegerField()
    p_arrival_time_s = serializers.FloatField()
    s_arrival_time_s = serializers.FloatField()
    sp_interval_s = serializers.FloatField()


class MagnitudeResultSerializer(serializers.Serializer):
    ml = serializers.FloatField()
    ml_log_raw = serializers.FloatField()


class RiskProbabilitiesSerializer(serializers.Serializer):
    low = serializers.FloatField()
    moderate = serializers.FloatField()
    high = serializers.FloatField()
    critical = serializers.FloatField()


class RiskResultSerializer(serializers.Serializer):
    level = serializers.CharField()
    class_index = serializers.IntegerField()
    probabilities = RiskProbabilitiesSerializer()


class ResultsPayloadSerializer(serializers.Serializer):
    """The results field inside a job response."""

    detection = DetectionResultSerializer()
    phase_picks = PhasePicksResultSerializer()
    magnitude = MagnitudeResultSerializer()
    risk = RiskResultSerializer()


class JobResponseSerializer(serializers.Serializer):
    """Response for GET /api/v1/seismic/jobs/{job_id}/."""

    job_id = serializers.UUIDField()
    status = serializers.CharField()
    submitted_at = serializers.DateTimeField()
    completed_at = serializers.DateTimeField(allow_null=True)
    processing_time_ms = serializers.IntegerField(allow_null=True)
    model_version = serializers.CharField()
    preprocessing_config_version = serializers.CharField()
    results = ResultsPayloadSerializer(allow_null=True)
    warnings = serializers.ListField(child=serializers.CharField())


class JobResultOnlySerializer(serializers.Serializer):
    """Response for GET /api/v1/seismic/jobs/{job_id}/result/."""

    results = ResultsPayloadSerializer()


class HealthResponseSerializer(serializers.Serializer):
    """Response for GET /api/v1/health/."""

    status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    model_version = serializers.CharField(allow_null=True)
    preprocessing_config_version = serializers.CharField(allow_null=True)
    risk_classes = serializers.ListField(child=serializers.CharField())


class ErrorResponseSerializer(serializers.Serializer):
    """Error response body."""

    error = serializers.CharField()
    detail = serializers.CharField()
    job_id = serializers.UUIDField(allow_null=True)
