"""SeismicNet: assembles encoder and task heads.

Fixed forward signature:
    forward(waveform: Tensor[B,3,3000], pgv: Tensor[B,1], active_tasks: list[str]) -> dict

Defined in docs/03_model_architecture.md § seismic_net.py.
"""
from __future__ import annotations

from typing import Any

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

        self.detection_head = DetectionHead(hidden_dim=hidden_dim, dropout=dropout)
        self.phase_pick_head = PhasePickHead(
            d_model=model_config["transformer_d_model"],
            hidden_dim=hidden_dim,
        )
        self.magnitude_head = MagnitudeHead(hidden_dim=hidden_dim, dropout=dropout)
        self.risk_head = RiskHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=len(model_config["risk_classes"]),
        )

    def forward(
        self,
        waveform: Tensor,
        pgv: Tensor,
        active_tasks: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            waveform: Input waveform tensor [B, 3, 3000].
            pgv: Peak ground velocity scalar [B, 1].
            active_tasks: List of active task names. Defaults to all tasks.

        Returns:
            Dict with keys "detection", "phase_pick", "magnitude", "risk".
            Only keys for active_tasks are populated.
        """
        # TODO: implement
        raise NotImplementedError
