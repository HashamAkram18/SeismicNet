"""URL configuration for api_config project."""
from django.contrib import admin
from django.urls import include, path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from seismic_api.dashboard_view import dashboard

schema_view = get_schema_view(
    openapi.Info(
        title="SeismicNet API",
        default_version="v1",
        description="Multi-task seismic waveform analysis REST API",
        terms_of_service="https://www.seismicnet.com/terms/",
        contact=openapi.Contact(email="admin@seismicnet.com"),
        license=openapi.License(name="MIT"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", dashboard, name="dashboard"),
    path("api/v1/", include("seismic_api.urls")),
    path("api/v1/auth/", include("seismic_api.auth_urls")),
    path(
        "swagger/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    path(
        "redoc/",
        schema_view.with_ui("redoc", cache_timeout=0),
        name="schema-redoc",
    ),
    path(
        "swagger.json",
        schema_view.without_ui(cache_timeout=0),
        name="schema-json",
    ),
]
