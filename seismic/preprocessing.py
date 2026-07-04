"""Seismic waveform preprocessing.

All signal processing logic lives in this module and only here.
The inference endpoint and training DataLoader both call these functions.

Order (frozen — see docs/01_project_overview.md and docs/02_data_pipeline.md):
1. Validate sampling rate (assert == 100 Hz)
2. Sub-window extraction (P-arrival at sample index 1000)
3. Detrend (linear, per channel)
4. Taper (5% cosine taper, both ends, per channel)
5. Bandpass filter (1–45 Hz, Butterworth order 4, zero-phase)
6. Extract PGV scalar (max(abs(waveform)) across all channels, before normalization)
7. Per-trace z-score normalization (each of 3 channels independently)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt, detrend as sp_detrend

logger = logging.getLogger(__name__)


def validate_sampling_rate(sampling_rate: int, expected_hz: int = 100) -> None:
    """Validate that the sampling rate matches expected_hz.

    Args:
        sampling_rate: Sampling rate of the input waveform in Hz.
        expected_hz: Expected sampling rate in Hz.

    Raises:
        ValueError: If sampling rate does not match expected_hz.
    """
    if sampling_rate != expected_hz:
        raise ValueError(
            f"Sampling rate mismatch: expected {expected_hz} Hz, got {sampling_rate} Hz"
        )


def extract_subwindow(
    trace: NDArray[np.floating],
    p_arrival_sample: int,
    p_arrival_target: int = 1000,
    window_length: int = 3000,
) -> NDArray[np.floating]:
    """Extract a fixed-length sub-window such that p_arrival maps to p_arrival_target.

    The P-arrival in the original trace is placed at sample index p_arrival_target
    in the output window. Zero-padding is applied symmetrically if the window extends
    before index 0 or after the array end.

    Args:
        trace: Input waveform of shape [channels, samples].
        p_arrival_sample: Sample index of the P-arrival in the original trace.
        p_arrival_target: Target sample index for P-arrival in output (default 1000).
        window_length: Output window length in samples (default 3000).

    Returns:
        Sub-windowed waveform of shape [channels, window_length].
    """
    n_channels, n_samples = trace.shape

    # Compute the start index in the original array
    start = p_arrival_sample - p_arrival_target
    end = start + window_length

    # Determine padding needed
    pad_before = max(0, -start)
    pad_after = max(0, end - n_samples)

    if pad_before > 0 or pad_after > 0:
        logger.warning(
            "Zero-padding applied: pad_before=%d, pad_after=%d "
            "(original samples=%d, p_arrival=%d, window_length=%d)",
            pad_before,
            pad_after,
            n_samples,
            p_arrival_sample,
            window_length,
        )

    # Clip the valid range in the original array
    src_start = max(0, start)
    src_end = min(n_samples, end)
    valid_length = src_end - src_start

    # Build output array
    output = np.zeros((n_channels, window_length), dtype=trace.dtype)
    output[:, pad_before : pad_before + valid_length] = trace[:, src_start:src_end]

    return output


def detrend_trace(
    trace: NDArray[np.floating],
    detrend_type: str = "linear",
) -> NDArray[np.floating]:
    """Remove trend from each channel independently along the time axis.

    Args:
        trace: Input waveform of shape [channels, samples].
        detrend_type: Type of detrending (default: "linear").

    Returns:
        Detrended waveform of same shape.
    """
    return sp_detrend(trace, type=detrend_type, axis=-1).astype(trace.dtype)


def cosine_taper(
    trace: NDArray[np.floating],
    taper_fraction: float = 0.05,
) -> NDArray[np.floating]:
    """Apply cosine taper to both ends of each channel.

    Args:
        trace: Input waveform of shape [channels, samples].
        taper_fraction: Fraction of the trace to taper at each end.

    Returns:
        Tapered waveform of same shape.
    """
    n_channels, n_samples = trace.shape
    taper_length = int(n_samples * taper_fraction)

    # Build cosine taper window: 1 in the middle, cosine taper at edges
    taper = np.ones(n_samples, dtype=trace.dtype)
    for i in range(taper_length):
        frac = i / taper_length
        taper[i] = 0.5 * (1.0 - np.cos(np.pi * frac))
        taper[n_samples - 1 - i] = 0.5 * (1.0 - np.cos(np.pi * frac))

    return (trace * taper[np.newaxis, :]).astype(trace.dtype)


def bandpass_filter(
    trace: NDArray[np.floating],
    low_hz: float = 1.0,
    high_hz: float = 45.0,
    order: int = 4,
    sampling_rate_hz: int = 100,
) -> NDArray[np.floating]:
    """Apply zero-phase Butterworth bandpass filter to each channel.

    Uses scipy.signal.sosfiltfilt for zero-phase filtering (no phase distortion).

    Args:
        trace: Input waveform of shape [channels, samples].
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.
        sampling_rate_hz: Sampling rate of the signal in Hz.

    Returns:
        Filtered waveform of same shape.
    """
    nyquist = sampling_rate_hz / 2.0
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    filtered = np.zeros_like(trace)
    for ch in range(trace.shape[0]):
        filtered[ch] = sosfiltfilt(sos, trace[ch])
    return filtered.astype(trace.dtype)


def extract_pgv(trace: NDArray[np.floating]) -> np.floating:
    """Extract peak ground velocity scalar before normalization.

    Computes max absolute amplitude across all channels and all samples.
    This is the amplitude feature that survives normalization and is passed
    to the magnitude head.

    Args:
        trace: Input waveform of shape [channels, samples].

    Returns:
        Scalar PGV value (max absolute amplitude across all channels).
    """
    return np.float32(np.max(np.abs(trace)))


def normalize(trace: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply per-trace z-score normalization to each channel independently.

    For each of the 3 channels independently:
    - Compute mean and std of that channel's samples
    - If std < 1e-10 (flat channel), set the channel to all zeros and log a warning
    - Otherwise apply (x - mean) / std

    Args:
        trace: Input waveform of shape [channels, samples].

    Returns:
        Normalized waveform of same shape.
    """
    n_channels, n_samples = trace.shape
    output = np.zeros_like(trace, dtype=np.float32)

    for ch in range(n_channels):
        mean = np.mean(trace[ch])
        std = np.std(trace[ch])
        if std < 1e-10:
            logger.warning(
                "Channel %d has near-zero std (%.2e), setting to zeros", ch, std
            )
            output[ch] = 0.0
        else:
            output[ch] = (trace[ch] - mean) / std

    return output


def preprocess_waveform(
    waveform: NDArray[np.floating],
    sampling_rate: int,
    config: dict[str, Any],
    p_arrival_sample: int,
    extract_pgv_flag: bool = True,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline on a single waveform.

    This is the single entry point called by both the training DataLoader
    and the Django inference endpoint. It implements the frozen 7-step pipeline.

    Args:
        waveform: Raw waveform of shape [channels, samples].
        sampling_rate: Sampling rate of the input waveform in Hz.
        config: Preprocessing configuration dict from preprocessing_config.json.
        p_arrival_sample: P-arrival sample index in the original waveform.
        extract_pgv_flag: Whether to compute PGV before normalization.

    Returns:
        Dict with keys:
            "waveform": np.ndarray [3, 3000] float32, processed and normalized
            "pgv": float32 scalar, peak ground velocity (before normalization)

    Raises:
        ValueError: If sampling_rate does not match config sampling_rate_hz.
    """
    logger.info("Preprocessing entry: shape=%s, sampling_rate=%d Hz", waveform.shape, sampling_rate)

    # Step 1: Validate sampling rate
    validate_sampling_rate(sampling_rate, config["sampling_rate_hz"])
    logger.info("Step 1/7: validate_sampling_rate passed (%d Hz)", sampling_rate)

    # Step 2: Sub-window extraction
    trace = extract_subwindow(
        waveform,
        p_arrival_sample=p_arrival_sample,
        p_arrival_target=config["p_arrival_sample_index"],
        window_length=config["window_length_samples"],
    )
    logger.info("Step 2/7: extract_subwindow complete, shape=%s", trace.shape)

    # Step 3: Detrend
    trace = detrend_trace(trace, detrend_type=config["detrend_type"])
    logger.info("Step 3/7: detrend complete")

    # Step 4: Taper
    trace = cosine_taper(trace, taper_fraction=config["taper_fraction"])
    logger.info("Step 4/7: cosine_taper complete")

    # Step 5: Bandpass filter
    trace = bandpass_filter(
        trace,
        low_hz=config["bandpass_low_hz"],
        high_hz=config["bandpass_high_hz"],
        order=config["bandpass_order"],
        sampling_rate_hz=config["sampling_rate_hz"],
    )
    logger.info("Step 5/7: bandpass_filter complete (%s-%s Hz)", config["bandpass_low_hz"], config["bandpass_high_hz"])

    # Step 6: Extract PGV scalar (before normalization)
    pgv = extract_pgv(trace) if extract_pgv_flag else np.float32(0.0)
    logger.info("Step 6/7: extract_pgv complete, pgv=%.6f", pgv)

    # Step 7: Per-trace z-score normalization
    trace = normalize(trace)
    logger.info("Step 7/7: normalize complete")

    logger.info("Preprocessing done: output shape=%s", trace.shape)
    return {"waveform": trace.astype(np.float32), "pgv": np.float32(pgv)}
