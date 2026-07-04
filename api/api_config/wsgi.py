"""WSGI config for api_config project."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api_config.settings.local")

application = get_wsgi_application()
