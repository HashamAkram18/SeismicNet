"""Task-specific heads for SeismicNet.

Four heads: detection, phase picking, magnitude estimation, risk classification.
Defined in docs/03_model_architecture.md § heads.py.

CRITICAL: PhasePickHead reads from sequence_features [B, 256, 375], NOT pooled_features.
This is the most important architectural constraint in the entire system.
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

    At inference, apply torch.sigmoid() to logits.
    Classification threshold loaded from model_config.json:detection_threshold.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].

        Returns:
            Detection logits [B, 1].
        """
        return self.net(pooled_features)


class PhasePickHead(nn.Module):
    """Phase picking head with attention pointer mechanism.

    Input: sequence_features [B, 256, 375] — NOT pooled_features
    Output: [B, 2] (p_pick_normalized, s_pick_normalized) in [0, 1]

    Uses soft argmax / differentiable pointer for gradient flow through pick position.
    Separate attention pointers for P and S picks.

    This head MUST receive sequence_features. If it reads pooled_features,
    the model will be physically incapable of learning accurate phase picks.
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()
        # P-pick attention pointer
        self.p_query = nn.Linear(d_model, hidden_dim)
        self.p_score = nn.Linear(hidden_dim, 1)

        # S-pick attention pointer (separate parameters)
        self.s_query = nn.Linear(d_model, hidden_dim)
        self.s_score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            sequence_features: Sequence features from encoder [B, 256, 375].

        Returns:
            Phase pick positions [B, 2] in normalized [0, 1] range.
            [p_pick_normalized, s_pick_normalized]
        """
        # Permute to [B, 375, 256] for linear layers
        x = sequence_features.permute(0, 2, 1)  # [B, 375, 256]

        # Position index: normalized [0, 1] across 375 time steps
        seq_len = x.shape[1]
        position_index = torch.linspace(0, 1, seq_len, device=x.device)  # [375]

        # P-pick: attention pointer
        p_hidden = torch.relu(self.p_query(x))     # [B, 375, 64]
        p_scores = self.p_score(p_hidden).squeeze(-1)  # [B, 375]
        p_weights = torch.softmax(p_scores, dim=-1)    # [B, 375]
        p_pick = torch.sum(p_weights * position_index, dim=-1, keepdim=True)  # [B, 1]

        # S-pick: separate attention pointer
        s_hidden = torch.relu(self.s_query(x))     # [B, 375, 64]
        s_scores = self.s_score(s_hidden).squeeze(-1)  # [B, 375]
        s_weights = torch.softmax(s_scores, dim=-1)    # [B, 375]
        s_pick = torch.sum(s_weights * position_index, dim=-1, keepdim=True)  # [B, 1]

        return torch.cat([p_pick, s_pick], dim=-1)  # [B, 2]


class MagnitudeHead(nn.Module):
    """Magnitude estimation head.

    Input: concat([pooled_features [B, 256], pgv [B, 1]]) → [B, 257]
    Output: magnitude_pred [B, 1] (log10(Ml+1) scale)

    Architecture: Linear(257, 64) → ReLU → Dropout(0.1) → Linear(64, 1)

    PGV concatenation is mandatory. Without it the head must infer magnitude
    purely from waveform shape (which is z-score normalized and has no amplitude
    information). The PGV carries the only amplitude signal that survives normalization.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + 1, hidden_dim),  # 256 pooled + 1 pgv = 257
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_features: Tensor, pgv: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].
            pgv: Peak ground velocity scalar [B, 1].

        Returns:
            Predicted magnitude [B, 1] in log10(Ml+1) scale.
        """
        x = torch.cat([pooled_features, pgv], dim=-1)  # [B, 257]
        return self.net(x)


class RiskHead(nn.Module):
    """Structural damage risk classification head.

    Input: pooled_features [B, 256]
    Output: logits [B, 4] (softmax applied in loss, not here)
    Classes: {0: low, 1: moderate, 2: high, 3: critical}

    Architecture: Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 4)

    At inference, apply torch.softmax() to get class probabilities.
    Return both the argmax class and the full probability vector.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pooled_features: Pooled encoder features [B, 256].

        Returns:
            Risk class logits [B, 4].
        """
        return self.net(pooled_features)
