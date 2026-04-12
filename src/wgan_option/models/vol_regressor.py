"""Deterministic residual forecaster for fixed-grid vol surfaces."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgan_option.models.common import ResidualConvBlock, conv2d_out_size


class VolSurfaceRegressor(nn.Module):
    """Predict the future surface directly from the current surface and text embedding."""

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        surface_height: int,
        surface_width: int,
        base_channels: int = 32,
        res_blocks: int = 0,
        text_hidden_dim: int = 256,
        text_out_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.surface_height = surface_height
        self.surface_width = surface_width
        self.base_channels = base_channels
        self.res_blocks = max(0, int(res_blocks))

        encoder_layers = [
            nn.Conv2d(channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(self.res_blocks):
            encoder_layers.append(ResidualConvBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = conv2d_out_size(conv2d_out_size(surface_height))
        reduced_w = conv2d_out_size(conv2d_out_size(surface_width))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.surface_feat_dim + text_out_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, surface_height * surface_width),
        )

    def forward(self, current_surface: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = current_surface.size(0)
        surface_features = self.surface_encoder(current_surface).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        fused = torch.cat([surface_features, text_features], dim=1)
        delta = self.fusion(fused).view(batch_size, 1, self.surface_height, self.surface_width)
        return F.softplus(current_surface + delta) + 1e-4
