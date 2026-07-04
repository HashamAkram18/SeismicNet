"""File upload validation for seismic API."""
from __future__ import annotations

from typing import Optional

# Max file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

ALLOWED_MSEED_EXTENSIONS = {".mseed", ".msd"}
ALLOWED_STATIONXML_EXTENSIONS = {".xml", ".stationxml"}


def validate_mseed_file(file_name: str, file_size: int) -> Optional[str]:
    """Validate a MiniSEED upload file.

    Args:
        file_name: Name of the uploaded file.
        file_size: Size of the uploaded file in bytes.

    Returns:
        Error message string if validation fails, None if OK.
    """
    if not file_name:
        return "missing_file"

    ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in ALLOWED_MSEED_EXTENSIONS:
        return "invalid_file_type"

    if file_size > MAX_FILE_SIZE:
        return "file_too_large"

    return None


def validate_stationxml_file(file_name: str, file_size: int) -> Optional[str]:
    """Validate a StationXML upload file.

    Args:
        file_name: Name of the uploaded file.
        file_size: Size of the uploaded file in bytes.

    Returns:
        Error message string if validation fails, None if OK.
    """
    if not file_name:
        return None  # StationXML is optional

    ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in ALLOWED_STATIONXML_EXTENSIONS:
        return "invalid_stationxml_type"

    if file_size > MAX_FILE_SIZE:
        return "file_too_large"

    return None


def validate_p_arrival(value: Optional[int], max_samples: int = 30000) -> Optional[str]:
    """Validate p_arrival_sample parameter.

    Args:
        value: P-arrival sample index (optional).
        max_samples: Maximum allowed sample index.

    Returns:
        Error message string if validation fails, None if OK.
    """
    if value is None:
        return None

    if not isinstance(value, int):
        return "invalid_p_arrival"

    if value < 0 or value > max_samples:
        return "invalid_p_arrival"

    return None
