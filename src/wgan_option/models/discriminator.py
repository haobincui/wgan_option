import math

import torch
import torch.nn as nn

from wgan_option.models.common import (
    INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
    LEGACY_CRITIC_NORMALIZATION_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
    LP_PROJECTION_CRITIC_CONDITIONING_MODE,
    critic_conditioning_fingerprint,
    critic_normalization_fingerprint,
    normalize_critic_conditioning_mode,
    normalize_critic_normalization_mode,
)


def _conv2d_out_size(
    size: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    dilation: int = 1,
) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualCriticBlock(nn.Module):
    """Residual block for the critic encoder at fixed spatial resolution."""

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
        return torch.nn.functional.leaky_relu(
            x + self.block(x), negative_slope=0.2, inplace=False
        )


class Discriminator(nn.Module):
    """
    Conditional critic:
    score(real/fake next surface | current surface, text embedding)
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        surface_height: int,
        surface_width: int,
        base_channels: int = 32,
        res_blocks: int = 0,
        text_hidden_dim: int = 128,
        hidden_dim: int = 256,
        critic_normalization_mode: str = LEGACY_CRITIC_NORMALIZATION_MODE,
        critic_conditioning_mode: str = LP_CONCAT_CRITIC_CONDITIONING_MODE,
    ):
        super().__init__()
        self.surface_height = surface_height
        self.surface_width = surface_width
        self.critic_normalization_mode = normalize_critic_normalization_mode(
            critic_normalization_mode
        )
        self.critic_normalization_fingerprint = critic_normalization_fingerprint(
            self.critic_normalization_mode
        )
        self.critic_conditioning_mode = normalize_critic_conditioning_mode(
            critic_conditioning_mode
        )
        self.critic_conditioning_fingerprint = critic_conditioning_fingerprint(
            self.critic_conditioning_mode
        )
        tail_norm: nn.Module
        if (
            self.critic_normalization_mode
            == INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE
        ):
            tail_norm = nn.GroupNorm(1, base_channels * 4, affine=True)
        else:
            tail_norm = nn.InstanceNorm2d(base_channels * 4, affine=True)

        encoder_layers = [
            nn.Conv2d(channels * 2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
            ),
            tail_norm,
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(max(0, int(res_blocks))):
            encoder_layers.append(_ResidualCriticBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_height)))
        reduced_w = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_width)))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.critic_conditioning_mode == LP_PROJECTION_CRITIC_CONDITIONING_MODE:
            self.classifier = nn.Sequential(
                nn.Linear(self.surface_feat_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            self.text_projection = nn.Linear(text_hidden_dim, hidden_dim, bias=False)
            self.projection_scale = math.sqrt(float(hidden_dim))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.surface_feat_dim + text_hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        next_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat([current_surface, next_surface], dim=1)
        surface_features = self.surface_encoder(stacked).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        if self.critic_conditioning_mode == LP_PROJECTION_CRITIC_CONDITIONING_MODE:
            surface_hidden = self.classifier[1](self.classifier[0](surface_features))
            unconditional_score = self.classifier[2](surface_hidden)
            projected_text = self.text_projection(text_features)
            compatibility_score = (surface_hidden * projected_text).sum(
                dim=1, keepdim=True
            ) / self.projection_scale
            return unconditional_score + compatibility_score
        if (
            self.critic_conditioning_mode
            == LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE
        ):
            # Keep the complete text branch and classifier shape in the state
            # dict while making both its value and LP-related gradients exactly
            # zero for a controlled, parameter-count-preserving ablation.
            text_features = torch.zeros_like(text_features) + text_features * 0.0
        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)
