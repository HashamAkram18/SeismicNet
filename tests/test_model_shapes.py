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
from seismic.model.heads import DetectionHead, MagnitudeHead, PhasePickHead, RiskHead
from seismic.model.seismic_net import SeismicNet

MODEL_CONFIG_PATH = Path("configs/model_config.json")
B = 4


def load_model_config() -> dict:
    """Load the model config fixture."""
    return json.loads(MODEL_CONFIG_PATH.read_text())


class TestEncoderShape:
    """Tests for SeismicEncoder output shapes."""

    def test_sequence_features_shape(self) -> None:
        """Sequence features should be [B, 256, 375]."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(B, 3, 3000)
        seq_feat, _ = encoder(x)
        assert seq_feat.shape == (B, 256, 375)

    def test_pooled_features_shape(self) -> None:
        """Pooled features should be [B, 256]."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(B, 3, 3000)
        _, pooled = encoder(x)
        assert pooled.shape == (B, 256)

    def test_no_nan_in_encoder_output(self) -> None:
        """No NaN values in encoder output."""
        config = load_model_config()
        encoder = SeismicEncoder(config)
        x = torch.randn(B, 3, 3000)
        seq_feat, pooled = encoder(x)
        assert not torch.isnan(seq_feat).any()
        assert not torch.isnan(pooled).any()


class TestDetectionHeadShape:
    """Tests for DetectionHead output shape."""

    def test_detection_shape(self) -> None:
        """Detection head should output [B, 1] logits."""
        head = DetectionHead()
        pooled = torch.randn(B, 256)
        out = head(pooled)
        assert out.shape == (B, 1)


class TestPhasePickHeadShape:
    """Tests for PhasePickHead output shape."""

    def test_phase_pick_shape(self) -> None:
        """Phase pick head should output [B, 2] normalized positions."""
        head = PhasePickHead()
        seq_feat = torch.randn(B, 256, 375)
        out = head(seq_feat)
        assert out.shape == (B, 2)

    def test_phase_pick_range(self) -> None:
        """All phase pick values should be in [0, 1]."""
        head = PhasePickHead()
        seq_feat = torch.randn(B, 256, 375)
        out = head(seq_feat)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestMagnitudeHeadShape:
    """Tests for MagnitudeHead output shape."""

    def test_magnitude_shape(self) -> None:
        """Magnitude head should output [B, 1] log-scale prediction."""
        head = MagnitudeHead()
        pooled = torch.randn(B, 256)
        pgv = torch.randn(B, 1)
        out = head(pooled, pgv)
        assert out.shape == (B, 1)


class TestRiskHeadShape:
    """Tests for RiskHead output shape."""

    def test_risk_shape(self) -> None:
        """Risk head should output [B, 4] logits."""
        head = RiskHead()
        pooled = torch.randn(B, 256)
        out = head(pooled)
        assert out.shape == (B, 4)


class TestSeismicNetForward:
    """Tests for full SeismicNet forward pass."""

    def _make_model(self) -> SeismicNet:
        return SeismicNet(load_model_config())

    def test_detection_output_shape(self) -> None:
        """Full model detection output should be [B, 1]."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv, active_tasks=["detection"])
        assert out["detection"].shape == (B, 1)

    def test_phase_pick_output_shape(self) -> None:
        """Full model phase_pick output should be [B, 2]."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv, active_tasks=["phase_pick"])
        assert out["phase_pick"].shape == (B, 2)

    def test_magnitude_output_shape(self) -> None:
        """Full model magnitude output should be [B, 1]."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv, active_tasks=["magnitude"])
        assert out["magnitude"].shape == (B, 1)

    def test_risk_output_shape(self) -> None:
        """Full model risk output should be [B, 4]."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv, active_tasks=["risk"])
        assert out["risk"].shape == (B, 4)

    def test_active_tasks_filter(self) -> None:
        """Only active_tasks keys should be in output dict."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv, active_tasks=["detection", "risk"])
        assert set(out.keys()) == {"detection", "risk"}
        assert "phase_pick" not in out
        assert "magnitude" not in out

    def test_no_nan_in_output(self) -> None:
        """No NaN values in any output tensor with random valid input."""
        model = self._make_model()
        waveform = torch.randn(B, 3, 3000)
        pgv = torch.randn(B, 1)
        out = model(waveform, pgv)
        for key, tensor in out.items():
            assert not torch.isnan(tensor).any(), f"NaN in {key} output"

    def test_phase_pick_reads_sequence_features(self) -> None:
        """PhasePickHead must receive sequence_features, not pooled_features.

        Uses a stub encoder that returns fixed tensors with distinguishable values.
        If PhasePickHead received pooled_features (shape [B,256]) instead of
        sequence_features (shape [B,256,375]), it would crash with a shape error.
        """
        class StubEncoder(torch.nn.Module):
            """Returns fixed tensors to verify routing."""
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                seq = torch.ones(x.shape[0], 256, 375, device=x.device)
                pooled = torch.zeros(x.shape[0], 256, device=x.device)
                return seq, pooled

        config = load_model_config()
        model = SeismicNet(config)
        model.encoder = StubEncoder()
        model.eval()

        waveform = torch.randn(1, 3, 3000)
        pgv = torch.randn(1, 1)
        out = model(waveform, pgv, active_tasks=["phase_pick"])

        # PhasePickHead received torch.ones(1,256,375) — if it got pooled (1,256)
        # the Linear layers would crash with shape mismatch
        assert out["phase_pick"].shape == (1, 2)
        assert not torch.isnan(out["phase_pick"]).any()

    def test_pgv_routing(self) -> None:
        """MagnitudeHead receives pgv, other heads do not.

        Tests that MagnitudeHead is the only head whose forward receives
        the pgv tensor by verifying output changes when pgv changes.
        """
        model = self._make_model()
        model.eval()

        waveform = torch.randn(1, 3, 3000)
        pgv_a = torch.tensor([[1.0]])
        pgv_b = torch.tensor([[100.0]])

        out_a = model(waveform, pgv_a, active_tasks=["magnitude", "detection", "risk"])
        out_b = model(waveform, pgv_b, active_tasks=["magnitude", "detection", "risk"])

        # Magnitude output should differ (it receives pgv)
        assert not torch.allclose(out_a["magnitude"], out_b["magnitude"]), \
            "MagnitudeHead output should change when pgv changes"

        # Detection and risk outputs should be identical (they don't receive pgv)
        assert torch.allclose(out_a["detection"], out_b["detection"]), \
            "DetectionHead output should NOT change when pgv changes"
        assert torch.allclose(out_a["risk"], out_b["risk"]), \
            "RiskHead output should NOT change when pgv changes"
