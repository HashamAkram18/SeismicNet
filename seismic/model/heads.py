"""Task-specific heads for SeismicNet.

Four heads: detection, phase picking, magnitude estimation, risk classification.
Defined in docs/03_model_architecture.md § heads.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DetectionHead(nn.Module):
    """Binary seismic event detection head.

    Input: pooled_features [B, 256]
    Output: logits [B, 1] (sigmoid applied in loss, not here)

    Architecture: Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 1)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: implement
        pass

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].

        Returns:
            Detection logits [B, 1].
        """
        # TODO: implement
        raise NotImplementedError


class PhasePickHead(nn.Module):
    """Phase picking head with attention pointer mechanism.

    Input: sequence_features [B, 256, 375]
    Output: [B, 2] (p_pick_normalized, s_pick_normalized) in [0, 1]

    Uses soft argmax / differentiable pointer for gradient flow through pick position.
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()
        # TODO: implement attention pointer layers
        pass

    def forward(self, sequence_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            sequence_features: Sequence features from encoder [B, 256, 375].

        Returns:
            Phase pick positions [B, 2] in normalized [0, 1] range.
        """
        # TODO: implement
        raise NotImplementedError


class MagnitudeHead(nn.Module):
    """Magnitude estimation head.

    Input: concat([pooled_features [B, 256], pgv [B, 1]]) → [B, 257]
    Output: magnitude_pred [B, 1] (log10(Ml+1) scale)

    Architecture: Linear(257, 64) → ReLU → Dropout(0.1) → Linear(64, 1)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: implement
        pass

    def forward(self, pooled_features: Tensor, pgv: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].
            pgv: Peak ground velocity scalar [B, 1].

        Returns:
            Predicted magnitude [B, 1] in log10(Ml+1) scale.
        """
        # TODO: implement
        raise NotImplementedError


class RiskHead(nn.Module):
    """Structural damage risk classification head.

    Input: pooled_features [B, 256]
    Output: logits [B, 4] (softmax applied in loss, not here)
    Classes: {0: low, 1: moderate, 2: high, 3: critical}

    Architecture: Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 4)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1, num_classes: int = 4) -> None:
        super().__init__()
        # TODO: implement
        pass

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].

        Returns:
            Risk class logits [B, 4].
        """
        # TODO: implement
        raise NotImplementedError
