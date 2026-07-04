"""Celery application for SeismicNet API."""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api_config.settings.local")

app = Celery("seismic_api")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
