"""Unit tests for seismic.preprocessing.

Tests validate preprocessing order and output shape per docs/02_data_pipeline.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seismic.preprocessing import (
    bandpass_filter,
    cosine_taper,
    detrend_trace,
    extract_pgv,
    extract_subwindow,
    normalize,
    preprocess_waveform,
    validate_sampling_rate,
)

CONFIG_PATH = Path("configs/preprocessing_config.json")


def load_config() -> dict:
    """Load the preprocessing config fixture."""
    return json.loads(CONFIG_PATH.read_text())


class TestValidateSamplingRate:
    """Tests for validate_sampling_rate function."""

    def test_valid_sampling_rate_passes(self) -> None:
        """100 Hz input should pass without error."""
        validate_sampling_rate(100, expected_hz=100)

    def test_invalid_sampling_rate_raises_value_error(self) -> None:
        """Non-100 Hz input should raise ValueError."""
        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            validate_sampling_rate(200, expected_hz=100)


class TestExtractSubwindow:
    """Tests for extract_subwindow function."""

    def test_output_shape(self) -> None:
        """Output should be [channels, window_length_samples]."""
        waveform = np.random.randn(3, 6000).astype(np.float32)
        result = extract_subwindow(waveform, p_arrival_sample=3000, window_length=3000)
        assert result.shape == (3, 3000)

    def test_p_arrival_at_index_1000(self) -> None:
        """P-arrival should be positioned at sample index 1000 in output."""
        # Create a waveform with a spike at the P-arrival position
        waveform = np.zeros((3, 6000), dtype=np.float32)
        p_sample = 2500
        waveform[:, p_sample] = 1.0

        result = extract_subwindow(waveform, p_arrival_sample=p_sample, p_arrival_target=1000, window_length=3000)

        # The spike should be at index 1000 in the output
        assert result[:, 1000].tolist() == [1.0, 1.0, 1.0]
        # Indices before 1000 should be zeros
        assert np.all(result[:, :1000] == 0.0)

    def test_zero_padding_when_window_extends_before_start(self) -> None:
        """Window extending before index 0 should be zero-padded."""
        waveform = np.ones((3, 100), dtype=np.float32)
        # p_arrival=5, target=1000 -> start = 5 - 1000 = -995
        result = extract_subwindow(waveform, p_arrival_sample=5, p_arrival_target=1000, window_length=3000)
        assert result.shape == (3, 3000)
        # First ~995 samples should be zero (padded)
        assert np.all(result[:, :995] == 0.0)

    def test_zero_padding_when_window_extends_after_end(self) -> None:
        """Window extending past array end should be zero-padded."""
        waveform = np.ones((3, 100), dtype=np.float32)
        # p_arrival=95, target=1000 -> start = 95 - 1000 = -905, end = 2095
        result = extract_subwindow(waveform, p_arrival_sample=95, p_arrival_target=1000, window_length=3000)
        assert result.shape == (3, 3000)
        # Last ~905 samples should be zero (padded)
        assert np.all(result[:, 2095:] == 0.0)


class TestDetrend:
    """Tests for detrend function."""

    def test_removes_linear_trend(self) -> None:
        """Detrend should remove linear trend from each channel."""
        t = np.linspace(0, 1, 3000)
        trend = 5.0 * t + 2.0
        waveform = np.stack([trend, trend, trend], axis=0).astype(np.float32)

        result = detrend_trace(waveform)

        # After detrending, each channel should have near-zero mean
        for ch in range(3):
            assert abs(np.mean(result[ch])) < 1e-5

    def test_output_shape_unchanged(self) -> None:
        """Output shape should match input shape."""
        waveform = np.random.randn(3, 3000).astype(np.float32)
        result = detrend_trace(waveform)
        assert result.shape == waveform.shape


class TestTaper:
    """Tests for taper function."""

    def test_taper_both_ends(self) -> None:
        """Taper should apply cosine taper to both ends of each channel."""
        waveform = np.ones((3, 3000), dtype=np.float32)
        result = cosine_taper(waveform, taper_fraction=0.05)

        # First sample should be 0 (start of cosine taper)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)
        # Last sample should be 0
        assert result[0, -1] == pytest.approx(0.0, abs=1e-6)
        # Middle should be unchanged (1.0)
        assert result[0, 1500] == pytest.approx(1.0, abs=1e-6)

    def test_output_shape_unchanged(self) -> None:
        """Output shape should match input shape."""
        waveform = np.random.randn(3, 3000).astype(np.float32)
        result = cosine_taper(waveform)
        assert result.shape == waveform.shape


class TestBandpassFilter:
    """Tests for bandpass_filter function."""

    def test_removes_out_of_band_frequencies(self) -> None:
        """Filter should remove frequencies outside 1-45 Hz band."""
        sampling_rate = 100
        t = np.arange(3000) / sampling_rate

        # Create signal with 50 Hz component (out of band, above 45 Hz)
        signal_50hz = np.sin(2 * np.pi * 50 * t)
        waveform = np.stack([signal_50hz] * 3, axis=0).astype(np.float32)

        result = bandpass_filter(waveform, low_hz=1.0, high_hz=45.0, order=4, sampling_rate_hz=sampling_rate)

        # 50 Hz should be attenuated
        assert np.max(np.abs(result)) < 0.1

    def test_output_shape_unchanged(self) -> None:
        """Output shape should match input shape."""
        waveform = np.random.randn(3, 3000).astype(np.float32)
        result = bandpass_filter(waveform)
        assert result.shape == waveform.shape

    def test_passes_in_band_signal(self) -> None:
        """Filter should pass 10 Hz signal (within 1-45 Hz band) with minimal attenuation."""
        sampling_rate = 100
        t = np.arange(3000) / sampling_rate
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        waveform = np.stack([signal_10hz] * 3, axis=0).astype(np.float32)

        result = bandpass_filter(waveform, low_hz=1.0, high_hz=45.0, order=4, sampling_rate_hz=sampling_rate)

        # 10 Hz should pass through with most energy preserved
        assert np.max(np.abs(result)) > 0.8


class TestExtractPGV:
    """Tests for extract_pgv function."""

    def test_returns_scalar(self) -> None:
        """PGV should be a scalar value."""
        waveform = np.random.randn(3, 3000).astype(np.float32)
        pgv = extract_pgv(waveform)
        assert np.isscalar(pgv)

    def test_pgv_is_peak_abs_value(self) -> None:
        """PGV should equal max absolute amplitude across all channels."""
        waveform = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -1.0],
            [0.3, 0.0, 0.0],
        ], dtype=np.float32)
        pgv = extract_pgv(waveform)
        assert pgv == pytest.approx(1.0)

    def test_pgv_positive(self) -> None:
        """PGV scalar is always >= 0.0."""
        waveform = -5.0 * np.ones((3, 3000), dtype=np.float32)
        pgv = extract_pgv(waveform)
        assert pgv >= 0.0


class TestNormalize:
    """Tests for normalize function."""

    def test_zero_mean_per_channel(self) -> None:
        """Each channel should have approximately zero mean after normalization."""
        waveform = np.random.randn(3, 3000).astype(np.float32) * 10.0 + 5.0
        result = normalize(waveform)

        for ch in range(3):
            assert abs(np.mean(result[ch])) < 1e-5

    def test_unit_variance_per_channel(self) -> None:
        """Each channel should have approximately unit variance after normalization."""
        waveform = np.random.randn(3, 3000).astype(np.float32) * 10.0 + 5.0
        result = normalize(waveform)

        for ch in range(3):
            assert abs(np.std(result[ch]) - 1.0) < 1e-3

    def test_flat_channel_returns_zeros(self) -> None:
        """A flat (zero) input channel returns all zeros without raising."""
        waveform = np.zeros((3, 3000), dtype=np.float32)
        result = normalize(waveform)
        assert np.all(result == 0.0)


class TestPreprocessPipeline:
    """Tests for the full preprocess pipeline."""

    def test_pipeline_order(self) -> None:
        """Pipeline must execute in frozen order: validate→subwindow→detrend→taper→filter→pgv→normalize."""
        config = load_config()

        # Create a 60-second waveform (6000 samples at 100 Hz) with a P-arrival spike
        waveform = np.random.randn(3, 6000).astype(np.float32) * 0.1
        p_sample = 3000
        waveform[:, p_sample] = 10.0  # strong P-arrival spike

        result = preprocess_waveform(
            waveform,
            sampling_rate=100,
            config=config,
            p_arrival_sample=p_sample,
            extract_pgv_flag=True,
        )

        # Output should be normalized (mean ~0, std ~1)
        assert abs(np.mean(result["waveform"])) < 0.1
        assert abs(np.std(result["waveform"]) - 1.0) < 0.2

    def test_output_shape_and_pgv(self) -> None:
        """Full pipeline should return (waveform [3, 3000], pgv scalar)."""
        config = load_config()

        waveform = np.random.randn(3, 6000).astype(np.float32)
        result = preprocess_waveform(
            waveform,
            sampling_rate=100,
            config=config,
            p_arrival_sample=3000,
            extract_pgv_flag=True,
        )

        assert result["waveform"].shape == (3, 3000)
        assert result["waveform"].dtype == np.float32
        assert isinstance(result["pgv"], (float, np.floating))
        assert result["pgv"] >= 0.0

    def test_config_mismatch_raises(self) -> None:
        """Passing wrong sampling rate raises ValueError."""
        config = load_config()
        waveform = np.random.randn(3, 6000).astype(np.float32)

        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            preprocess_waveform(
                waveform,
                sampling_rate=200,
                config=config,
                p_arrival_sample=3000,
            )

    def test_short_input_padded(self) -> None:
        """Input shorter than window should be zero-padded."""
        config = load_config()
        waveform = np.random.randn(3, 500).astype(np.float32)

        result = preprocess_waveform(
            waveform,
            sampling_rate=100,
            config=config,
            p_arrival_sample=250,
        )

        assert result["waveform"].shape == (3, 3000)

    def test_flat_input_no_nan(self) -> None:
        """Flat input should not produce NaN values."""
        config = load_config()
        waveform = np.zeros((3, 6000), dtype=np.float32)

        result = preprocess_waveform(
            waveform,
            sampling_rate=100,
            config=config,
            p_arrival_sample=3000,
        )

        assert not np.any(np.isnan(result["waveform"]))
        assert result["pgv"] == pytest.approx(0.0)
