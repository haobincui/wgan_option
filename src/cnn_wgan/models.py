"""Standalone CNN WGAN models and reconstruction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

VOL_FLOOR = 1e-4


def _conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class _ResidualConvBlock(nn.Module):
    """Residual block for fixed-resolution CNN feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)


class CnnWGANGenerator(nn.Module):
    """CNN generator producing normalized future log-IV increments from condition and noise."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        noise_dim: int,
        base_channels: int,
        res_blocks: int,
        text_hidden_dim: int,
        text_out_dim: int,
        fusion_hidden_dim: int,
    ):
        super().__init__()
        self.surface_height = int(surface_height)
        self.surface_width = int(surface_width)
        self.surface_dim = int(surface_height * surface_width)
        self.embedding_dim = int(embedding_dim)
        self.noise_dim = int(noise_dim)

        encoder_layers = [
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(max(0, int(res_blocks))):
            encoder_layers.append(_ResidualConvBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = _conv2d_out_size(_conv2d_out_size(surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(surface_width))
        self.surface_feat_dim = int(base_channels * 4 * reduced_h * reduced_w)

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.surface_feat_dim + text_out_dim + noise_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fusion_hidden_dim, self.surface_dim),
        )

    def forward(self, current_surface: torch.Tensor, text_embedding: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = current_surface.size(0)
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=current_surface.device, dtype=torch.float32)
        surface_features = self.surface_encoder(current_surface).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        fused = torch.cat([surface_features, text_features, noise], dim=1)
        return self.fusion(fused)


class CnnWGANCritic(nn.Module):
    """CNN critic over current surface, candidate future surface, and text embedding."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        base_channels: int,
        res_blocks: int,
        text_hidden_dim: int,
        text_out_dim: int,
        fusion_hidden_dim: int,
    ):
        super().__init__()
        encoder_layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(max(0, int(res_blocks))):
            encoder_layers.append(_ResidualConvBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_height)))
        reduced_w = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_width)))
        self.surface_feat_dim = int(base_channels * 4 * reduced_h * reduced_w)

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.surface_feat_dim + text_out_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        future_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat([current_surface, future_surface], dim=1)
        surface_features = self.surface_encoder(stacked).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a log-IV increment."""

    current_log = torch.log(torch.clamp(current_surface_flat, min=VOL_FLOOR))
    future_log = current_log + delta
    return torch.exp(future_log).clamp_min(VOL_FLOOR)
