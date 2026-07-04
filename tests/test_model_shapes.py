"""Unit tests for model shape assertions.

Tests verify forward pass shapes for every head per docs/03_model_architecture.md.
All tests run on CPU with randomly initialized model and batch_size=4.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from seismic.model.encoder import SeismicEncoder

MODEL_CONFIG_PATH = Path("configs/model_config.json")


def load_model_config() -> dict:
    """Load the model config fixture."""
    return json.loads(MODEL_CONFIG_PATH.read_text())


class TestEncoderShape:
    """Tests for SeismicEncoder output shapes."""

    def test_sequence_features_shape(self) -> None:
        """Sequence features should be [B, 256, 375]."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(4, 3, 3000)
        seq_feat, _ = encoder(x)
        assert seq_feat.shape == (4, 256, 375)

    def test_pooled_features_shape(self) -> None:
        """Pooled features should be [B, 256]."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(4, 3, 3000)
        _, pooled = encoder(x)
        assert pooled.shape == (4, 256)

    def test_no_nan_in_output(self) -> None:
        """No NaN values in encoder output."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(4, 3, 3000)
        seq_feat, pooled = encoder(x)
        assert not torch.isnan(seq_feat).any()
        assert not torch.isnan(pooled).any()


class TestDetectionHeadShape:
    """Tests for DetectionHead output shape."""

    def test_detection_shape(self) -> None:
        """Detection head should output [B, 1] logits."""
        assert False, "TODO: implement when heads.py is implemented"


class TestPhasePickHeadShape:
    """Tests for PhasePickHead output shape."""

    def test_phase_pick_shape(self) -> None:
        """Phase pick head should output [B, 2] normalized positions."""
        assert False, "TODO: implement when heads.py is implemented"

    def test_phase_pick_range(self) -> None:
        """All phase pick values should be in [0, 1]."""
        assert False, "TODO: implement when heads.py is implemented"


class TestMagnitudeHeadShape:
    """Tests for MagnitudeHead output shape."""

    def test_magnitude_shape(self) -> None:
        """Magnitude head should output [B, 1] log-scale prediction."""
        assert False, "TODO: implement when heads.py is implemented"


class TestRiskHeadShape:
    """Tests for RiskHead output shape."""

    def test_risk_shape(self) -> None:
        """Risk head should output [B, 4] logits."""
        assert False, "TODO: implement when heads.py is implemented"


class TestSeismicNetForward:
    """Tests for full SeismicNet forward pass."""

    def test_detection_output_shape(self) -> None:
        """Full model detection output should be [B, 1]."""
        assert False, "TODO: implement when seismic_net.py is implemented"

    def test_phase_pick_output_shape(self) -> None:
        """Full model phase_pick output should be [B, 2]."""
        assert False, "TODO: implement when seismic_net.py is implemented"

    def test_magnitude_output_shape(self) -> None:
        """Full model magnitude output should be [B, 1]."""
        assert False, "TODO: implement when seismic_net.py is implemented"

    def test_risk_output_shape(self) -> None:
        """Full model risk output should be [B, 4]."""
        assert False, "TODO: implement when seismic_net.py is implemented"

    def test_active_tasks_filter(self) -> None:
        """Only active_tasks keys should be in output dict."""
        assert False, "TODO: implement when seismic_net.py is implemented"

    def test_no_nan_in_output(self) -> None:
        """No NaN values in any output tensor with random valid input."""
        assert False, "TODO: implement when seismic_net.py is implemented"
