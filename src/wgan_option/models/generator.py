import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConvBlock(nn.Module):
    """Small residual block used to widen the surface encoder without changing resolution."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)


class Generator(nn.Module):
    """
    Conditional generator:
    (current surface, news embedding, noise) -> next surface
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        noise_dim: int,
        surface_height: int,
        surface_width: int,
        base_channels: int = 32,
        res_blocks: int = 0,
        text_hidden_dim: int = 256,
        text_out_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.noise_dim = noise_dim
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
            encoder_layers.append(_ResidualConvBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = _conv2d_out_size(_conv2d_out_size(surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(surface_width))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        fusion_dim = self.surface_feat_dim + text_out_dim + noise_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, surface_height * surface_width),
        )

    def forward(self, current_surface: torch.Tensor, text_embedding: torch.Tensor, noise: torch.Tensor = None):
        batch_size = current_surface.size(0)
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=current_surface.device)

        surface_features = self.surface_encoder(current_surface).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        fused = torch.cat([surface_features, text_features, noise], dim=1)
        delta = self.fusion(fused).view(batch_size, 1, self.surface_height, self.surface_width)

        # Softplus guarantees strictly positive volatility values.
        next_surface = F.softplus(current_surface + delta) + 1e-4
        return next_surface
