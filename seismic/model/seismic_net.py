"""SeismicNet: assembles encoder and task heads.

Fixed forward signature (Invariant 3 — must not change):
    forward(waveform: Tensor[B,3,3000], pgv: Tensor[B,1],
            active_tasks: list[str]) -> dict

Defined in docs/03_model_architecture.md § seismic_net.py.

Routing rules (non-negotiable):
- encoder() returns (sequence_features, pooled_features) tuple
- PhasePickHead receives sequence_features [B, 256, 375]
- DetectionHead, RiskHead receive pooled_features [B, 256]
- MagnitudeHead receives pooled_features [B, 256] AND pgv [B, 1]
- Only compute heads listed in active_tasks — skip others entirely
- Return dict with only active task keys populated
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from seismic.model.encoder import SeismicEncoder
from seismic.model.heads import DetectionHead, MagnitudeHead, PhasePickHead, RiskHead


class SeismicNet(nn.Module):
    """Multi-task seismic waveform analysis model.

    One shared encoder backbone, four task-specific heads.
    Forward signature is fixed (Invariant 3) and must not change.

    Args:
        model_config: Model configuration dictionary from model_config.json.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = SeismicEncoder(model_config)
        hidden_dim = model_config["head_hidden_dim"]
        dropout = model_config["head_dropout"]
        d_model = model_config["transformer_d_model"]

        self.detection_head = DetectionHead(input_dim=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.phase_pick_head = PhasePickHead(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )
        self.magnitude_head = MagnitudeHead(input_dim=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.risk_head = RiskHead(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=len(model_config["risk_classes"]),
        )

    def forward(
        self,
        waveform: Tensor,
        pgv: Tensor,
        active_tasks: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            waveform: Input waveform tensor [B, 3, 3000].
            pgv: Peak ground velocity scalar [B, 1].
            active_tasks: List of active task names. Defaults to all tasks.

        Returns:
            Dict with keys "detection", "phase_pick", "magnitude", "risk".
            Only keys for active_tasks are populated.
        """
        if active_tasks is None:
            active_tasks = ["detection", "phase_pick", "magnitude", "risk"]

        # Encoder produces both sequence and pooled features
        sequence_features, pooled_features = self.encoder(waveform)

        outputs: dict[str, Tensor] = {}

        # DetectionHead ← pooled_features
        if "detection" in active_tasks:
            outputs["detection"] = self.detection_head(pooled_features)

        # PhasePickHead ← sequence_features (NOT pooled_features)
        if "phase_pick" in active_tasks:
            outputs["phase_pick"] = self.phase_pick_head(sequence_features)

        # MagnitudeHead ← pooled_features + pgv
        if "magnitude" in active_tasks:
            outputs["magnitude"] = self.magnitude_head(pooled_features, pgv)

        # RiskHead ← pooled_features
        if "risk" in active_tasks:
            outputs["risk"] = self.risk_head(pooled_features)

        return outputs
