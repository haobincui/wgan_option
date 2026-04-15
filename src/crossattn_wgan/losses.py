"""Loss helpers for standalone Cross-Attention WGAN training."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import autograd


def critic_wgan_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return fake_scores.mean() - real_scores.mean()


def generator_wgan_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def gradient_penalty(
    *,
    critic,
    real_future_surface: torch.Tensor,
    fake_future_surface: torch.Tensor,
    current_surface: torch.Tensor,
    text_embedding: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    batch_size = real_future_surface.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_future_surface.device)
    interpolated = alpha * real_future_surface + (1.0 - alpha) * fake_future_surface
    interpolated.requires_grad_(True)

    interpolated_scores = critic(interpolated, current_surface, text_embedding)
    grad_outputs = torch.ones_like(interpolated_scores, device=real_future_surface.device)
    gradients = autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * float(lambda_gp)


def _difference_weights(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.ones(1, device=values.device, dtype=values.dtype)
    diffs = values[1:] - values[:-1]
    return 1.0 / torch.clamp(diffs, min=1e-6).pow(2)


def maturity_smoothness_penalty(future_log_surface: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    if future_log_surface.size(1) < 2:
        return torch.zeros(1, device=future_log_surface.device).squeeze()
    weights = _difference_weights(maturity_days_grid.to(future_log_surface.device)).view(1, -1, 1)
    diff = future_log_surface[:, 1:, :] - future_log_surface[:, :-1, :]
    return (weights * diff.pow(2)).mean()


def strike_smoothness_penalty(future_log_surface: torch.Tensor, strike_grid: torch.Tensor) -> torch.Tensor:
    if future_log_surface.size(2) < 2:
        return torch.zeros(1, device=future_log_surface.device).squeeze()
    weights = _difference_weights(strike_grid.to(future_log_surface.device)).view(1, 1, -1)
    diff = future_log_surface[:, :, 1:] - future_log_surface[:, :, :-1]
    return (weights * diff.pow(2)).mean()


def reconstruction_loss(fake_future_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
    """MAE between generated and real target surface, penalising lazy zero-delta generators."""
    return torch.nn.functional.l1_loss(fake_future_flat, target_flat)


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))
