"""URL configuration for seismic_api."""
from django.urls import path

from seismic_api import views

urlpatterns = [
    path("seismic/analyze/", views.AnalyzeView.as_view(), name="analyze"),
    path("seismic/jobs/<uuid:job_id>/", views.JobStatusView.as_view(), name="job-status"),
    path("seismic/jobs/<uuid:job_id>/result/", views.JobResultView.as_view(), name="job-result"),
    path("health/", views.HealthView.as_view(), name="health"),
]
