"""FiLM-conditioned WGAN models: text embedding modulates surface CNN features via Feature-wise Linear Modulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

VOL_FLOOR = 1e-4
VOL_CEIL = 5.0
_LOG_VOL_FLOOR = math.log(VOL_FLOOR)
_LOG_VOL_CEIL = math.log(VOL_CEIL)


def _conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class FiLMLayer(nn.Module):
    """Generate channel-wise affine parameters (gamma, beta) from a conditioning vector."""

    def __init__(self, conditioning_dim: int, num_channels: int):
        super().__init__()
        self.projection = nn.Linear(conditioning_dim, num_channels * 2)
        # With `(1 + gamma) * x + beta`, identity-preserving FiLM means `gamma=0`, not `gamma=1`.
        nn.init.zeros_(self.projection.weight[:num_channels])
        nn.init.zeros_(self.projection.weight[num_channels:])
        nn.init.zeros_(self.projection.bias[:num_channels])
        nn.init.zeros_(self.projection.bias[num_channels:])

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        params = self.projection(conditioning)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class _FiLMResidualConvBlock(nn.Module):
    """Residual block with FiLM conditioning applied after each convolution."""

    def __init__(self, channels: int, conditioning_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLMLayer(conditioning_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.film2 = FiLMLayer(conditioning_dim, channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.film1(out, conditioning)
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
        out = self.conv2(out)
        out = self.film2(out, conditioning)
        return torch.nn.functional.leaky_relu(residual + out, negative_slope=0.2)


class FilmWGANGenerator(nn.Module):
    """CNN generator with FiLM conditioning: text embedding modulates surface feature extraction."""

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
        self.num_res_blocks = max(0, int(res_blocks))

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        c = base_channels
        self.conv1 = nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLMLayer(text_out_dim, c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.film2 = FiLMLayer(text_out_dim, c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.film3 = FiLMLayer(text_out_dim, c * 4)

        self.res_blocks = nn.ModuleList(
            [_FiLMResidualConvBlock(c * 4, text_out_dim) for _ in range(self.num_res_blocks)]
        )

        reduced_h = _conv2d_out_size(_conv2d_out_size(surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(surface_width))
        self.surface_feat_dim = int(c * 4 * reduced_h * reduced_w)

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

        text_features = self.text_encoder(text_embedding)

        x = self.conv1(current_surface)
        x = self.film1(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = self.film2(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = self.film3(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        for res_block in self.res_blocks:
            x = res_block(x, text_features)

        surface_features = x.flatten(start_dim=1)
        fused = torch.cat([surface_features, text_features, noise], dim=1)
        return self.fusion(fused)


class FilmWGANCritic(nn.Module):
    """CNN critic with FiLM conditioning: text embedding modulates joint surface feature extraction."""

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
        self.num_res_blocks = max(0, int(res_blocks))

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        c = base_channels
        self.conv1 = nn.Conv2d(2, c, kernel_size=3, stride=2, padding=1)
        self.film1 = FiLMLayer(text_out_dim, c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.film2 = FiLMLayer(text_out_dim, c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.film3 = FiLMLayer(text_out_dim, c * 4)

        self.res_blocks_list = nn.ModuleList(
            [_FiLMResidualConvBlock(c * 4, text_out_dim) for _ in range(self.num_res_blocks)]
        )

        reduced_h = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_height)))
        reduced_w = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_width)))
        self.surface_feat_dim = int(c * 4 * reduced_h * reduced_w)

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
        text_features = self.text_encoder(text_embedding)

        stacked = torch.cat([current_surface, future_surface], dim=1)

        x = self.conv1(stacked)
        x = self.film1(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = self.film2(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = self.film3(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        for res_block in self.res_blocks_list:
            x = res_block(x, text_features)

        surface_features = x.flatten(start_dim=1)
        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a log-IV increment.

    Clamps future_log into [log(VOL_FLOOR), log(VOL_CEIL)] before exp to prevent overflow
    when the generator emits large deltas during early training.
    """

    current_log = torch.log(torch.clamp(current_surface_flat, min=VOL_FLOOR))
    future_log = torch.clamp(current_log + delta, min=_LOG_VOL_FLOOR, max=_LOG_VOL_CEIL)
    return torch.exp(future_log).clamp(min=VOL_FLOOR, max=VOL_CEIL)
