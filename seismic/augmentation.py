"""Waveform augmentation for training.

Applies time-shift, noise injection, amplitude perturbation, and channel
dropout to seismic waveforms. All operations are CPU-only (no GPU tensors).

Augmentation is applied AFTER preprocessing and ONLY during training.
It operates on the preprocessed, normalized waveform tensor.
It must NOT alter the pgv scalar (PGV was extracted before normalization).

Spec: docs/02_data_pipeline.md § seismic/augmentation.py
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def augment_waveform(
    trace: NDArray[np.floating],
    pgv: float,
    max_shift: int = 50,
    noise_std_range: tuple[float, float] = (0.005, 0.02),
    amp_scale_range: tuple[float, float] = (0.5, 2.0),
    time_shift_prob: float = 0.5,
    amp_scale_prob: float = 0.5,
    noise_prob: float = 0.3,
    channel_dropout_prob: float = 0.1,
) -> tuple[NDArray[np.floating], float]:
    """Apply the full random augmentation pipeline to a waveform.

    Each augmentation is applied independently with its own probability.
    The order of application is: time_shift → amplitude_scale → noise → channel_dropout.

    Args:
        trace: Input waveform of shape [3, 3000] (preprocessed, normalized).
        pgv: Peak ground velocity scalar (NOT modified by augmentation).
        max_shift: Maximum random time shift in samples for time_shift augmentation.
        noise_std_range: (min, max) range for additive noise std as fraction of per-channel std.
        amp_scale_range: (min, max) range for random amplitude scaling.
        time_shift_prob: Probability of applying time_shift.
        amp_scale_prob: Probability of applying amplitude_scale.
        noise_prob: Probability of applying additive Gaussian noise.
        channel_dropout_prob: Probability of applying channel_dropout.

    Returns:
        Tuple of (augmented_trace, pgv_unchanged).
    """
    rng = np.random.RandomState()

    # Time shift (probability 0.5)
    if rng.random() < time_shift_prob:
        trace = time_shift(trace, max_shift=max_shift)

    # Amplitude scaling (probability 0.5)
    if rng.random() < amp_scale_prob:
        trace = amplitude_scale(trace, scale_range=amp_scale_range)

    # Additive Gaussian noise (probability 0.3)
    if rng.random() < noise_prob:
        trace = additive_noise(trace, std_range=noise_std_range)

    # Channel dropout (probability 0.1)
    if rng.random() < channel_dropout_prob:
        trace = channel_dropout(trace)

    return trace, pgv


def time_shift(
    trace: NDArray[np.floating],
    max_shift: int = 50,
) -> NDArray[np.floating]:
    """Apply random circular time shift to waveform.

    Shifts the waveform along the time axis by a random offset drawn from
    Uniform(-max_shift, +max_shift). Vacated positions are zero-padded.

    Args:
        trace: Input waveform of shape [channels, samples].
        max_shift: Maximum shift in samples (positive = forward, negative = backward).

    Returns:
        Time-shifted waveform of same shape.
    """
    n_channels, n_samples = trace.shape
    offset = np.random.randint(-max_shift, max_shift + 1)

    output = np.zeros_like(trace)
    if offset > 0:
        # Shift right: data moves forward, pad left
        output[:, offset:] = trace[:, : n_samples - offset]
    elif offset < 0:
        # Shift left: data moves backward, pad right
        output[:, : n_samples + offset] = trace[:, -offset:]
    else:
        output = trace.copy()

    return output


def amplitude_scale(
    trace: NDArray[np.floating],
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> tuple[NDArray[np.floating], float]:
    """Apply random amplitude scaling.

    Multiplies the entire waveform by a scalar drawn from
    Uniform(scale_range[0], scale_range[1]). Simulates distance variation.
    Does NOT update pgv — it is already extracted before normalization.

    Args:
        trace: Input waveform of shape [channels, samples].
        scale_range: (min_scale, max_scale) tuple.

    Returns:
        Tuple of (scaled_trace, scale_factor).
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (trace * scale).astype(trace.dtype), float(scale)


def additive_noise(
    trace: NDArray[np.floating],
    std_range: tuple[float, float] = (0.005, 0.02),
) -> NDArray[np.floating]:
    """Add Gaussian noise with std drawn as fraction of per-channel std.

    Simulates instrument noise floor. The noise std for each channel is
    drawn from Uniform(std_range[0], std_range[1]) × channel_std.

    Args:
        trace: Input waveform of shape [channels, samples].
        std_range: (min_fraction, max_fraction) of per-channel std.

    Returns:
        Noisy waveform of same shape.
    """
    n_channels, n_samples = trace.shape
    output = trace.copy()

    for ch in range(n_channels):
        ch_std = np.std(trace[ch])
        noise_std = np.random.uniform(std_range[0], std_range[1]) * ch_std
        output[ch] += np.random.normal(0, noise_std, n_samples).astype(trace.dtype)

    return output


def channel_dropout(
    trace: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Zero out one randomly selected channel (E, N, or Z).

    Simulates a dead sensor. Only applies to earthquake windows —
    never to noise windows (noise window augmentation is irrelevant).

    Args:
        trace: Input waveform of shape [3, samples].

    Returns:
        Waveform with one channel zeroed out.
    """
    output = trace.copy()
    ch = np.random.randint(0, 3)
    output[ch] = 0.0
    return output
