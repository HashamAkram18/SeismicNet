"""pytest configuration for API tests — sets up Django before test collection."""
import os
import sys

# Ensure the api directory and repo root are on sys.path
api_dir = os.path.join(os.path.dirname(__file__), "..")
repo_root = os.path.join(api_dir, "..")
sys.path.insert(0, os.path.abspath(api_dir))
sys.path.insert(0, os.path.abspath(repo_root))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api_config.settings.local")

import django
django.setup()

# Run migrations to create test database tables
from django.core.management import call_command
call_command("migrate", "--run-syncdb", verbosity=0)
