"""Django models for seismic analysis jobs."""
import uuid

from django.db import models


class SeismicJob(models.Model):
    """Represents a seismic waveform analysis job."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("complete", "Complete"),
        ("failed", "Failed"),
    ]

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    submitted_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    processing_time_ms = models.IntegerField(null=True, blank=True)
    input_filename = models.CharField(max_length=255)
    result = models.JSONField(null=True, blank=True)
    error_detail = models.CharField(max_length=1024, null=True, blank=True)
    model_version = models.CharField(max_length=50)
    preprocessing_config_version = models.CharField(max_length=50)

    class Meta:
        ordering = ["-submitted_at"]

    def __str__(self) -> str:
        return f"Job {self.job_id} — {self.status}"
