"""Cross-Attention conditioned WGAN models: surface feature maps attend to text embedding tokens."""

from __future__ import annotations

import math

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


class SurfaceTextCrossAttention(nn.Module):
    """Multi-head cross-attention: surface spatial tokens attend to text virtual tokens."""

    def __init__(
        self,
        surface_channels: int,
        text_dim: int,
        attn_dim: int,
        num_heads: int,
        num_text_tokens: int,
    ):
        super().__init__()
        self.surface_channels = surface_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.num_text_tokens = num_text_tokens
        self.head_dim = attn_dim // num_heads
        assert self.head_dim * num_heads == attn_dim, "attn_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(surface_channels, attn_dim)
        self.text_to_tokens = nn.Linear(text_dim, num_text_tokens * attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, surface_channels)
        self.layer_norm = nn.LayerNorm(surface_channels)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, surface_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surface_features: (B, C, H, W)
            text_features: (B, text_dim)
        Returns:
            (B, C, H, W) with residual connection
        """
        B, C, H, W = surface_features.shape
        spatial_len = H * W

        # Surface tokens: (B, H*W, C)
        surface_tokens = surface_features.flatten(2).permute(0, 2, 1)

        # Text virtual tokens: (B, num_tokens, attn_dim)
        text_tokens = self.text_to_tokens(text_features).view(B, self.num_text_tokens, self.attn_dim)

        # Q from surface, K/V from text
        Q = self.q_proj(surface_tokens)  # (B, H*W, attn_dim)
        K = self.k_proj(text_tokens)      # (B, num_tokens, attn_dim)
        V = self.v_proj(text_tokens)      # (B, num_tokens, attn_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, spatial_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, H*W, head_dim)
        K = K.view(B, self.num_text_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, tokens, head_dim)
        V = V.view(B, self.num_text_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, tokens, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, heads, H*W, tokens)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, H*W, head_dim)

        # Merge heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, spatial_len, self.attn_dim)
        projected = self.out_proj(attn_output)  # (B, H*W, C)

        # Residual + LayerNorm
        surface_out = self.layer_norm(surface_tokens + projected)

        # Reshape back to (B, C, H, W)
        return surface_out.permute(0, 2, 1).view(B, C, H, W)


class CrossAttnWGANGenerator(nn.Module):
    """CNN generator with cross-attention: surface features attend to text embedding tokens."""

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
        num_attn_heads: int = 4,
        num_text_tokens: int = 4,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.surface_height = int(surface_height)
        self.surface_width = int(surface_width)
        self.surface_dim = int(surface_height * surface_width)
        self.embedding_dim = int(embedding_dim)
        self.noise_dim = int(noise_dim)

        # Surface CNN encoder (same as cnn_wgan)
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

        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Cross-attention: surface features attend to text
        self.cross_attention = SurfaceTextCrossAttention(
            surface_channels=base_channels * 4,
            text_dim=text_out_dim,
            attn_dim=attn_dim,
            num_heads=num_attn_heads,
            num_text_tokens=num_text_tokens,
        )

        # Fusion MLP
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

        # Encode text
        text_features = self.text_encoder(text_embedding)

        # Encode surface via CNN
        surface_feature_map = self.surface_encoder(current_surface)  # (B, 4C, H', W')

        # Cross-attention: surface attends to text
        attended_features = self.cross_attention(surface_feature_map, text_features)  # (B, 4C, H', W')

        # Flatten and fuse
        surface_features = attended_features.flatten(start_dim=1)
        fused = torch.cat([surface_features, text_features, noise], dim=1)
        return self.fusion(fused)


class CrossAttnWGANCritic(nn.Module):
    """CNN critic with cross-attention: joint surface features attend to text embedding tokens."""

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
        num_attn_heads: int = 4,
        num_text_tokens: int = 4,
        attn_dim: int = 128,
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

        self.cross_attention = SurfaceTextCrossAttention(
            surface_channels=base_channels * 4,
            text_dim=text_out_dim,
            attn_dim=attn_dim,
            num_heads=num_attn_heads,
            num_text_tokens=num_text_tokens,
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
        text_features = self.text_encoder(text_embedding)

        stacked = torch.cat([current_surface, future_surface], dim=1)
        surface_feature_map = self.surface_encoder(stacked)  # (B, 4C, H', W')

        attended_features = self.cross_attention(surface_feature_map, text_features)
        surface_features = attended_features.flatten(start_dim=1)

        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a log-IV increment."""

    current_log = torch.log(torch.clamp(current_surface_flat, min=VOL_FLOOR))
    future_log = current_log + delta
    return torch.exp(future_log).clamp_min(VOL_FLOOR)
