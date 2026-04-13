"""MLP regressor for paired SVI forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class SviRegressor(nn.Module):
    """Predict future padded SVI parameters and slice count from current SVI + text."""

    def __init__(
        self,
        current_input_dim: int,
        embedding_dim: int,
        regression_dim: int,
        count_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.current_encoder = nn.Sequential(
            nn.Linear(current_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.regression_head = nn.Linear(hidden_dim, regression_dim)
        self.count_head = nn.Linear(hidden_dim, count_classes)

    def forward(self, current_features: torch.Tensor, text_embedding: torch.Tensor):
        current_encoded = self.current_encoder(current_features)
        text_encoded = self.text_encoder(text_embedding)
        fused = torch.cat([current_encoded, text_encoded], dim=1)
        hidden = self.trunk(fused)
        return self.regression_head(hidden), self.count_head(hidden)
