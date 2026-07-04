"""Local development settings — SQLite, no Redis required."""
from .base import *  # noqa: F401,F403

DEBUG = True

ALLOWED_HOSTS = ["*"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Celery runs tasks synchronously in local dev — no Redis needed
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Use local filesystem for artifacts
ARTIFACT_DIR = str(BASE_DIR.parent / "exported_models" / "seismic_v1")
