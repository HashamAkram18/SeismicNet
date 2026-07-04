"""Dashboard view — serves the waveform visualization frontend."""
from django.shortcuts import render


def dashboard(request):
    """Render the SeismicNet waveform analysis dashboard."""
    return render(request, "dashboard.html")
