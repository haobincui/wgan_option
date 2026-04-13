"""Standalone Transformer WGAN models and reconstruction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VOL_FLOOR = 1e-4


def _build_transformer_encoder(model_dim: int, num_heads: int, ffn_dim: int, dropout: float, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=int(model_dim),
        nhead=int(num_heads),
        dim_feedforward=int(ffn_dim),
        dropout=float(dropout),
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=max(1, int(layers)))


def _positional_tokens(
    row_embedding: nn.Parameter,
    col_embedding: nn.Parameter,
) -> torch.Tensor:
    rows = row_embedding.unsqueeze(1)
    cols = col_embedding.unsqueeze(0)
    return (rows + cols).reshape(-1, row_embedding.size(-1))


class TransformerWGANGenerator(nn.Module):
    """Transformer generator producing delta surfaces from condition and noise."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        noise_dim: int,
        model_dim: int,
        layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        text_hidden_dim: int,
        text_token_dim: int,
        noise_hidden_dim: int,
    ):
        super().__init__()
        self.surface_height = int(surface_height)
        self.surface_width = int(surface_width)
        self.surface_dim = int(surface_height * surface_width)
        self.embedding_dim = int(embedding_dim)
        self.noise_dim = int(noise_dim)
        self.model_dim = int(model_dim)

        self.surface_value_proj = nn.Linear(1, model_dim)
        self.row_embedding = nn.Parameter(torch.randn(surface_height, model_dim) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(surface_width, model_dim) * 0.02)
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, text_token_dim),
            nn.GELU(),
            nn.Linear(text_token_dim, model_dim),
        )
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, noise_hidden_dim),
            nn.GELU(),
            nn.Linear(noise_hidden_dim, model_dim),
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.transformer = _build_transformer_encoder(model_dim, num_heads, ffn_dim, dropout, layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_head = nn.Linear(model_dim, 1)

    def forward(self, current_surface: torch.Tensor, text_embedding: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = current_surface.size(0)
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=current_surface.device, dtype=torch.float32)
        surface_tokens = current_surface.flatten(start_dim=2).transpose(1, 2)
        surface_tokens = self.surface_value_proj(surface_tokens)
        surface_tokens = surface_tokens + _positional_tokens(self.row_embedding, self.col_embedding).unsqueeze(0)
        text_token = self.text_encoder(text_embedding).unsqueeze(1)
        noise_token = self.noise_encoder(noise).unsqueeze(1)
        tokens = torch.cat([text_token, noise_token, surface_tokens], dim=1)
        encoded = self.transformer(self.input_norm(tokens))
        surface_encoded = self.output_norm(encoded[:, 2:, :])
        return self.output_head(surface_encoded).squeeze(-1)


class TransformerWGANCritic(nn.Module):
    """Transformer critic over current surface, candidate future surface, and text embedding."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        model_dim: int,
        layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        text_hidden_dim: int,
        text_token_dim: int,
    ):
        super().__init__()
        self.surface_height = int(surface_height)
        self.surface_width = int(surface_width)
        self.surface_pair_proj = nn.Linear(2, model_dim)
        self.row_embedding = nn.Parameter(torch.randn(surface_height, model_dim) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(surface_width, model_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, text_token_dim),
            nn.GELU(),
            nn.Linear(text_token_dim, model_dim),
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.transformer = _build_transformer_encoder(model_dim, num_heads, ffn_dim, dropout, layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.score_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        future_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_surface.size(0)
        current_tokens = current_surface.flatten(start_dim=2).transpose(1, 2)
        future_tokens = future_surface.flatten(start_dim=2).transpose(1, 2)
        paired_tokens = torch.cat([current_tokens, future_tokens], dim=-1)
        surface_tokens = self.surface_pair_proj(paired_tokens)
        surface_tokens = surface_tokens + _positional_tokens(self.row_embedding, self.col_embedding).unsqueeze(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        text_token = self.text_encoder(text_embedding).unsqueeze(1)
        tokens = torch.cat([cls_token, text_token, surface_tokens], dim=1)
        encoded = self.transformer(self.input_norm(tokens))
        cls_output = self.output_norm(encoded[:, 0, :])
        return self.score_head(cls_output)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a level delta."""

    return F.softplus(current_surface_flat + delta) + VOL_FLOOR
