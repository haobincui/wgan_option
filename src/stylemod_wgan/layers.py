"""Style-modulated convolution layers for the standalone StyleMod WGAN module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleProjection(nn.Module):
    """Project one global style vector into per-layer modulation coefficients."""

    def __init__(self, style_dim: int, channels: int, *, bias_init: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(int(style_dim), int(channels))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(bias_init))

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        return self.linear(style)


class ModulatedConv2d(nn.Module):
    """Fixed-resolution style-modulated convolution with optional demodulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        demodulate: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)
        self.demodulate = bool(demodulate)

        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.style_projection = StyleProjection(style_dim, self.in_channels, bias_init=1.0)
        self.scale = 1.0 / math.sqrt(float(self.in_channels * self.kernel_size * self.kernel_size))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")

        modulation = self.style_projection(style).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.weight * self.scale * modulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch_size * self.in_channels, height, width)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=batch_size)
        out = out.view(batch_size, self.out_channels, out.shape[-2], out.shape[-1])
        return out + self.bias.view(1, -1, 1, 1)


class StyledResidualConvBlock(nn.Module):
    """Residual block built from style-modulated convolutions."""

    def __init__(self, channels: int, style_dim: int, *, demodulate: bool = True):
        super().__init__()
        self.conv1 = ModulatedConv2d(channels, channels, 3, style_dim, demodulate=demodulate)
        self.conv2 = ModulatedConv2d(channels, channels, 3, style_dim, demodulate=demodulate)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.leaky_relu(self.conv1(x, style), negative_slope=0.2, inplace=False)
        out = self.conv2(out, style)
        return F.leaky_relu((residual + out) / math.sqrt(2.0), negative_slope=0.2, inplace=False)
