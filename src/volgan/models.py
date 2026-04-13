"""Standalone VolGAN models and reconstruction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

VOL_FLOOR = 1e-4


class VolGANGenerator(nn.Module):
    """MLP generator producing future log-IV increments from condition and noise."""

    def __init__(self, surface_dim: int, embedding_dim: int, noise_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = int(surface_dim) + int(embedding_dim) + int(noise_dim)
        hidden_dim = int(hidden_dim)
        self.surface_dim = int(surface_dim)
        self.embedding_dim = int(embedding_dim)
        self.noise_dim = int(noise_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Softplus(),
            nn.Linear(hidden_dim * 2, surface_dim),
        )

    def forward(
        self,
        current_surface_flat: torch.Tensor,
        text_embedding: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = current_surface_flat.shape[0]
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=current_surface_flat.device, dtype=torch.float32)
        inputs = torch.cat([current_surface_flat, text_embedding, noise], dim=1)
        return self.network(inputs)


class VolGANDiscriminator(nn.Module):
    """MLP discriminator over condition plus a candidate log-IV increment."""

    def __init__(self, surface_dim: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = int(surface_dim) * 2 + int(embedding_dim)
        hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current_surface_flat: torch.Tensor,
        text_embedding: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([current_surface_flat, text_embedding, candidate_delta], dim=1)
        return self.network(inputs)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a log-IV increment."""

    current_log = torch.log(torch.clamp(current_surface_flat, min=VOL_FLOOR))
    future_log = current_log + delta
    return torch.exp(future_log).clamp_min(VOL_FLOOR)
