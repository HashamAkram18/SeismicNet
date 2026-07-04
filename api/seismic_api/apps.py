"""Django AppConfig for seismic_api — loads model on startup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class SeismicApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "seismic_api"
    verbose_name = "Seismic Analysis API"

    def ready(self) -> None:
        """Load inference engine when Django starts.

        Adds repo root to sys.path so we can import from the seismic package.
        The ML package (seismic/) and the Django app (api/) are sibling
        directories under the repo root.
        """
        # Add repo root to sys.path for seismic package imports
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Load the inference engine singleton
        from django.conf import settings

        artifact_dir = getattr(settings, "ARTIFACT_DIR", None)
        if artifact_dir and Path(artifact_dir).exists():
            try:
                from seismic_api.inference import get_engine

                engine = get_engine(artifact_dir)
                logger.info("Inference engine loaded successfully from %s", artifact_dir)
            except Exception as e:
                logger.error("Failed to load inference engine: %s", e)
        else:
            logger.warning(
                "ARTIFACT_DIR not set or does not exist: %s. "
                "Model not loaded. /api/v1/health/ will return 503.",
                artifact_dir,
            )
