"""Tests for seismic inference engine.

All tests mock the actual model to avoid needing a real checkpoint.
Creates tiny dummy artifacts for engine loading tests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from seismic.model.seismic_net import SeismicNet


def _tiny_config() -> dict:
    """Create a tiny model config for fast testing."""
    return {
        "stem_channels": 8,
        "stage_channels": [8, 16, 32, 32],
        "stage_blocks": [2, 2, 2, 2],
        "stage_kernels": [7, 7, 5, 5],
        "transformer_d_model": 32,
        "transformer_nhead": 4,
        "transformer_layers": 1,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.1,
        "head_hidden_dim": 16,
        "head_dropout": 0.1,
        "pooling_dropout": 0.2,
        "detection_threshold": 0.5,
        "risk_classes": ["low", "moderate", "high", "critical"],
    }


def _create_dummy_artifact(artifact_dir: Path) -> None:
    """Create a minimal dummy artifact bundle for testing."""
    # Create and script a tiny model
    model_config = _tiny_config()
    model = SeismicNet(model_config)
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(str(artifact_dir / "model_scripted.pt"))

    # Write configs
    preprocess_config = {
        "schema_version": "1.0.0",
        "sampling_rate_hz": 100,
        "window_length_samples": 3000,
        "p_arrival_sample_index": 1000,
        "bandpass_low_hz": 1.0,
        "bandpass_high_hz": 45.0,
        "bandpass_order": 4,
        "bandpass_type": "butter",
        "taper_fraction": 0.05,
        "taper_type": "cosine",
        "normalization_strategy": "per_trace_zscore",
        "pgv_extraction": "peak_abs_velocity_before_norm",
        "detrend_type": "linear",
        "channels": ["E", "N", "Z"],
        "label_schema_version": "1.0.0",
    }
    with open(artifact_dir / "preprocessing_config.json", "w") as f:
        json.dump(preprocess_config, f)

    model_config_full = {
        "schema_version": "1.0.0",
        **model_config,
    }
    with open(artifact_dir / "model_config.json", "w") as f:
        json.dump(model_config_full, f)

    label_schema = {
        "risk_classes": ["low", "moderate", "high", "critical"],
        "risk_class_to_index": {"low": 0, "moderate": 1, "high": 2, "critical": 3},
    }
    with open(artifact_dir / "label_schema.json", "w") as f:
        json.dump(label_schema, f)


class TestEngineLoading:
    """Tests for inference engine initialization."""

    def test_engine_loads_with_dummy_artifact(self) -> None:
        """Engine loads successfully with a dummy artifact bundle."""
        from seismic_api.inference import SeismicInferenceEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            _create_dummy_artifact(artifact_dir)

            engine = SeismicInferenceEngine(str(artifact_dir))
            assert engine.model_version == "1.0.0"
            assert engine.preprocessing_config_version == "1.0.0"
            assert engine.risk_classes == ["low", "moderate", "high", "critical"]

    def test_engine_missing_artifact_dir(self) -> None:
        """Engine raises FileNotFoundError for missing directory."""
        from seismic_api.inference import SeismicInferenceEngine

        with pytest.raises(FileNotFoundError):
            SeismicInferenceEngine("/nonexistent/path")

    def test_engine_missing_model_file(self) -> None:
        """Engine raises FileNotFoundError for missing model_scripted.pt."""
        from seismic_api.inference import SeismicInferenceEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            # Write configs but not the model
            with open(artifact_dir / "preprocessing_config.json", "w") as f:
                json.dump({"schema_version": "1.0.0"}, f)
            with open(artifact_dir / "model_config.json", "w") as f:
                json.dump({"schema_version": "1.0.0", "detection_threshold": 0.5}, f)
            with open(artifact_dir / "label_schema.json", "w") as f:
                json.dump({"risk_classes": ["low"]}, f)

            with pytest.raises(FileNotFoundError, match="model_scripted.pt"):
                SeismicInferenceEngine(str(artifact_dir))


class TestPredict:
    """Tests for inference engine predict method."""

    def _get_engine(self) -> tuple:
        """Create a temporary artifact bundle and return engine + cleanup function."""
        from seismic_api.inference import SeismicInferenceEngine

        tmpdir = tempfile.TemporaryDirectory()
        artifact_dir = Path(tmpdir.name)
        _create_dummy_artifact(artifact_dir)
        engine = SeismicInferenceEngine(str(artifact_dir))
        return engine, tmpdir

    def test_predict_returns_all_keys(self) -> None:
        """Predict returns all required result keys."""
        engine, tmpdir = self._get_engine()
        try:
            waveform = np.random.randn(3, 3000).astype(np.float32)
            result = engine.predict(waveform, sampling_rate=100, p_arrival_sample=1000)

            assert "results" in result
            assert "detection" in result["results"]
            assert "phase_picks" in result["results"]
            assert "magnitude" in result["results"]
            assert "risk" in result["results"]
            assert "warnings" in result
            assert "model_version" in result
        finally:
            tmpdir.cleanup()

    def test_detection_confidence_in_range(self) -> None:
        """Detection confidence is always in [0, 1] (sigmoid output)."""
        engine, tmpdir = self._get_engine()
        try:
            waveform = np.random.randn(3, 3000).astype(np.float32)
            result = engine.predict(waveform, sampling_rate=100, p_arrival_sample=1000)

            conf = result["results"]["detection"]["confidence"]
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range [0, 1]"
        finally:
            tmpdir.cleanup()

    def test_phase_picks_positive(self) -> None:
        """Phase pick sample indices are always >= 0."""
        engine, tmpdir = self._get_engine()
        try:
            waveform = np.random.randn(3, 3000).astype(np.float32)
            result = engine.predict(waveform, sampling_rate=100, p_arrival_sample=1000)

            p_sample = result["results"]["phase_picks"]["p_arrival_sample"]
            s_sample = result["results"]["phase_picks"]["s_arrival_sample"]
            assert p_sample >= 0, f"P-sample {p_sample} is negative"
            assert s_sample >= 0, f"S-sample {s_sample} is negative"
        finally:
            tmpdir.cleanup()

    def test_magnitude_back_transform(self) -> None:
        """Magnitude is correctly back-transformed from log10(Ml+1)."""
        engine, tmpdir = self._get_engine()
        try:
            waveform = np.random.randn(3, 3000).astype(np.float32)
            result = engine.predict(waveform, sampling_rate=100, p_arrival_sample=1000)

            ml = result["results"]["magnitude"]["ml"]
            ml_log_raw = result["results"]["magnitude"]["ml_log_raw"]
            expected_ml = 10**ml_log_raw - 1
            assert abs(ml - expected_ml) < 1e-4, f"Magnitude mismatch: {ml} vs {expected_ml}"
        finally:
            tmpdir.cleanup()

    def test_risk_probs_sum_to_one(self) -> None:
        """Risk class probabilities sum to 1.0."""
        engine, tmpdir = self._get_engine()
        try:
            waveform = np.random.randn(3, 3000).astype(np.float32)
            result = engine.predict(waveform, sampling_rate=100, p_arrival_sample=1000)

            probs = result["results"]["risk"]["probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-5, f"Risk probs sum to {total}, expected 1.0"
            assert all(0.0 <= v <= 1.0 for v in probs.values()), "Risk prob out of [0,1]"
        finally:
            tmpdir.cleanup()

    def test_singleton_returns_same_instance(self) -> None:
        """Two get_engine() calls return identical object."""
        from seismic_api.inference import get_engine, reset_engine

        reset_engine()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir)
                _create_dummy_artifact(artifact_dir)

                engine1 = get_engine(str(artifact_dir))
                engine2 = get_engine(str(artifact_dir))
                assert engine1 is engine2, "Singleton not working"
        finally:
            reset_engine()
