"""ResNet + Transformer encoder backbone for SeismicNet.

Hybrid 1D ResNet stages followed by Transformer encoder layers.
Architecture defined in docs/03_model_architecture.md § encoder.py.

Input: [B, 3, 3000]
Output: (sequence_features [B, 256, 375], pooled_features [B, 256])
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    """1D Residual block with optional stride.

    Architecture (docs/03_model_architecture.md § ResNet Block):
        Conv1d(in_ch, out_ch, kernel_size, padding, stride)
        BatchNorm1d(out_ch)
        ReLU
        Conv1d(out_ch, out_ch, kernel_size, padding, stride=1)
        BatchNorm1d(out_ch)
        Skip connection (identity or projection)
        ReLU after addition

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size.
        stride: Stride for the first conv (default 1).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, stride=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # Skip connection: project if channels change or stride != 1
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, in_ch, T].

        Returns:
            Output tensor of shape [B, out_ch, T].
        """
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class SeismicEncoder(nn.Module):
    """Hybrid ResNet + Transformer encoder.

    Architecture (docs/03_model_architecture.md § Encoder architecture):

    Stem:
        Conv1d(3, 64, k=7, s=2, p=3) → [B, 64, 1500]
        BatchNorm1d(64), ReLU

    Stage 1 — 2× ResBlock(64, 64, k=7, s=1):
        [B, 64, 1500] → [B, 64, 1500]
        (no stride — stem already halved temporal dim)

    Stage 2 — 2× ResBlock(64, 128, k=7):
        First block: stride=2
        [B, 64, 1500] → [B, 128, 750]

    Stage 3 — 2× ResBlock(128, 256, k=5):
        First block: stride=2
        [B, 128, 750] → [B, 256, 375]

    Stage 4 — 2× ResBlock(256, 256, k=5, s=1):
        [B, 256, 375] → [B, 256, 375]
        (no stride — preserve temporal resolution for phase picking head)

    Transformer encoder:
        Permute: [B, 256, 375] → [B, 375, 256]
        Learned positional encoding: nn.Embedding(375, 256) added
        2× TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512,
                                    dropout=0.1, activation='gelu', batch_first=True)
        Permute back: [B, 375, 256] → [B, 256, 375]

    Returns:
        Tuple of (sequence_features, pooled_features):
            sequence_features: [B, 256, 375]
            pooled_features: [B, 256]
    """

    def __init__(self, config: dict) -> None:
        """Initialize encoder from model_config.json.

        Args:
            config: Model configuration dictionary containing:
                stem_channels, stage_channels, stage_blocks, stage_kernels,
                transformer_d_model, transformer_nhead, transformer_layers,
                transformer_ffn_dim, transformer_dropout, pooling_dropout.
        """
        super().__init__()
        stem_ch = config["stem_channels"]
        stage_ch = config["stage_channels"]
        stage_blocks = config["stage_blocks"]
        stage_kernels = config["stage_kernels"]
        d_model = config["transformer_d_model"]
        nhead = config["transformer_nhead"]
        n_layers = config["transformer_layers"]
        ffn_dim = config["transformer_ffn_dim"]
        dropout = config["transformer_dropout"]
        pool_dropout = config["pooling_dropout"]

        # Stem: Conv1d(3, 64, k=7, s=2, p=3) + BN + ReLU
        self.stem = nn.Sequential(
            nn.Conv1d(3, stem_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # Build stages
        # Stride pattern: stages 1→2→3 use stride=2, stage 4 uses stride=1
        # Stage 4 has no stride to preserve temporal resolution for phase picking
        stage_strides = [1, 2, 2, 1]
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        for i, (out_ch, n_blocks, k, s) in enumerate(
            zip(stage_ch, stage_blocks, stage_kernels, stage_strides)
        ):
            blocks = []
            for j in range(n_blocks):
                stride = s if j == 0 else 1  # stride only on first block of each stage
                blocks.append(ResBlock(in_ch, out_ch, kernel_size=k, stride=stride))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        # Transformer
        self.pos_embedding = nn.Embedding(375, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pooling: global average pool + dropout
        self.pool_dropout = nn.Dropout(pool_dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input waveform tensor [B, 3, 3000].

        Returns:
            Tuple of (sequence_features [B, 256, 375], pooled_features [B, 256]).
        """
        # Stem: [B, 3, 3000] → [B, 64, 1500]
        x = self.stem(x)

        # Stages: [B, 64, 1500] → [B, 256, 375]
        for stage in self.stages:
            x = stage(x)

        # x is now [B, 256, 375]
        sequence_features = x

        # Transformer: permute to [B, 375, 256]
        x = x.permute(0, 2, 1)  # [B, 375, 256]

        # Add learned positional encoding
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, 375]
        x = x + self.pos_embedding(positions)

        # Transformer layers
        x = self.transformer(x)  # [B, 375, 256]

        # Permute back: [B, 375, 256] → [B, 256, 375]
        sequence_features = x.permute(0, 2, 1)

        # Global average pooling: [B, 256, 375] → [B, 256]
        pooled_features = sequence_features.mean(dim=-1)
        pooled_features = self.pool_dropout(pooled_features)

        return sequence_features, pooled_features
