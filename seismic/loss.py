"""Per-task losses and weighted combiner for SeismicNet.

Noise windows (Task A label = 0) must have Tasks B, C, D losses masked to zero.
This is not optional — enforced by the is_noise batch key (Invariant 4).

Loss weights come from training_config.json, never hardcoded.
Defined in docs/03_model_architecture.md § loss.py.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class SeismicLoss(nn.Module):
    """Multi-task loss combiner with noise masking.

    Computes per-task losses and returns weighted sum.
    Phase-specific weights are loaded from training_config.json.

    Per-task losses (docs/03_model_architecture.md § loss.py):
        Task A: BCEWithLogitsLoss(pos_weight from config)
        Task B: HuberLoss(delta from config) — noise-masked
        Task C: MSELoss — noise-masked
        Task D: CrossEntropyLoss — noise-masked

    Args:
        training_config: Full training configuration dict.
        phase_config: Phase-specific configuration dict (one entry from phases list).
    """

    def __init__(self, training_config: dict[str, Any], phase_config: dict[str, Any]) -> None:
        super().__init__()
        self.phase_config = phase_config
        weights = phase_config["loss_weights"]

        # Store weights as buffers (move with model to device)
        self.register_buffer("w_detection", torch.tensor(weights["detection"]))
        self.register_buffer("w_phase_pick", torch.tensor(weights["phase_pick"]))
        self.register_buffer("w_magnitude", torch.tensor(weights["magnitude"]))
        self.register_buffer("w_risk", torch.tensor(weights["risk"]))

        # Task A: Detection — BCEWithLogitsLoss with pos_weight
        pos_weight = torch.tensor([training_config["detection_pos_weight"]])
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Task B: Phase pick — HuberLoss
        self.huber_loss = nn.HuberLoss(
            delta=training_config["huber_delta"], reduction="none"
        )

        # Task C: Magnitude — MSELoss
        self.mse_loss = nn.MSELoss(reduction="none")

        # Task D: Risk — CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute per-task losses and weighted total.

        Args:
            outputs: Model output dict with keys matching active_tasks.
            batch: Batch dict with label tensors and "is_noise" flag.

        Returns:
            Dict with keys "detection", "phase_pick", "magnitude", "risk", "total".
            Inactive task losses are zero.
        """
        is_noise: Tensor = batch["is_noise"].squeeze()  # [B]
        n_seismic = (~is_noise).sum().clamp(min=1)  # avoid div-by-zero

        losses: dict[str, Tensor] = {}

        # Task A: Detection — applied to ALL windows (seismic + noise)
        if "detection" in outputs:
            det_loss = self.bce_loss(
                outputs["detection"].squeeze(-1),
                batch["label_detection"].squeeze(-1).float(),
            )
            losses["detection"] = det_loss
        else:
            losses["detection"] = torch.tensor(0.0, device=is_noise.device)

        # Task B: Phase pick — noise-masked
        if "phase_pick" in outputs:
            phase_pred = outputs["phase_pick"]  # [B, 2]
            phase_target = batch["label_phase_pick"]  # [B, 2]
            per_sample = self.huber_loss(phase_pred, phase_target).mean(dim=-1)  # [B]
            masked = per_sample * (~is_noise).float()
            losses["phase_pick"] = masked.sum() / n_seismic
        else:
            losses["phase_pick"] = torch.tensor(0.0, device=is_noise.device)

        # Task C: Magnitude — noise-masked
        if "magnitude" in outputs:
            mag_pred = outputs["magnitude"].squeeze(-1)  # [B]
            mag_target = batch["label_magnitude"].squeeze(-1)  # [B]
            per_sample = self.mse_loss(mag_pred, mag_target)  # [B]
            masked = per_sample * (~is_noise).float()
            losses["magnitude"] = masked.sum() / n_seismic
        else:
            losses["magnitude"] = torch.tensor(0.0, device=is_noise.device)

        # Task D: Risk — noise-masked
        if "risk" in outputs:
            risk_pred = outputs["risk"]  # [B, 4]
            risk_target = batch["label_risk"].squeeze(-1).long()  # [B]
            per_sample = self.ce_loss(risk_pred, risk_target)  # [B]
            masked = per_sample * (~is_noise).float()
            losses["risk"] = masked.sum() / n_seismic
        else:
            losses["risk"] = torch.tensor(0.0, device=is_noise.device)

        # Weighted total
        losses["total"] = (
            self.w_detection * losses["detection"]
            + self.w_phase_pick * losses["phase_pick"]
            + self.w_magnitude * losses["magnitude"]
            + self.w_risk * losses["risk"]
        )

        return losses
