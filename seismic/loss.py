"""Per-task losses and weighted combiner for SeismicNet.

Noise windows (Task A label = 0) must have Tasks B, C, D losses masked to zero.
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

    Args:
        training_config: Training configuration dict.
        phase_config: Phase-specific configuration dict.
    """

    def __init__(self, training_config: dict[str, Any], phase_config: dict[str, Any]) -> None:
        super().__init__()
        # TODO: initialize per-task loss functions
        pass

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
        # TODO: implement
        raise NotImplementedError
