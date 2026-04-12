"""Shared building blocks for surface-based neural network models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    """Return the spatial dimension after a Conv2d with the given parameters."""
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def group_count(channels: int) -> int:
    """Pick the largest valid GroupNorm group count for *channels*."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small residual block used to widen the surface encoder without changing resolution."""

    def __init__(self, channels: int):
        super().__init__()
        groups = group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)
