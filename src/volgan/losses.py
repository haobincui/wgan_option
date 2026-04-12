"""Loss helpers for standalone VolGAN training."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .models import reconstruct_future_surface


def generator_bce_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))


def discriminator_bce_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
    return 0.5 * (real_loss + fake_loss)


def _difference_weights(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.ones(1, device=values.device, dtype=values.dtype)
    diffs = values[1:] - values[:-1]
    return 1.0 / torch.clamp(diffs, min=1e-6).pow(2)


def maturity_smoothness_penalty(
    future_log_surface: torch.Tensor,
    maturity_days_grid: torch.Tensor,
) -> torch.Tensor:
    if future_log_surface.size(1) < 2:
        return torch.zeros(1, device=future_log_surface.device).squeeze()
    weights = _difference_weights(maturity_days_grid.to(future_log_surface.device)).view(1, -1, 1)
    diff = future_log_surface[:, 1:, :] - future_log_surface[:, :-1, :]
    return (weights * diff.pow(2)).mean()


def strike_smoothness_penalty(
    future_log_surface: torch.Tensor,
    strike_grid: torch.Tensor,
) -> torch.Tensor:
    if future_log_surface.size(2) < 2:
        return torch.zeros(1, device=future_log_surface.device).squeeze()
    weights = _difference_weights(strike_grid.to(future_log_surface.device)).view(1, 1, -1)
    diff = future_log_surface[:, :, 1:] - future_log_surface[:, :, :-1]
    return (weights * diff.pow(2)).mean()


def _gradient_norm(parameters: Iterable[torch.nn.Parameter], loss: torch.Tensor) -> float:
    gradients = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
    sq_norm = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        sq_norm += float(torch.sum(gradient.detach() ** 2).cpu())
    return math.sqrt(max(sq_norm, 0.0))


def estimate_gradient_matching(
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    train_loader,
    device: torch.device,
    strike_grid: torch.Tensor,
    maturity_days_grid: torch.Tensor,
    noise_dim: int,
    epochs: int,
) -> tuple[float, float]:
    """Estimate penalty weights by matching BCE and smoothness gradient magnitudes."""

    bce_to_m: list[float] = []
    bce_to_t: list[float] = []
    max_epochs = max(1, int(epochs))
    parameters = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    iterator = list(train_loader)
    if not iterator:
        return 1.0, 1.0

    generator.train()
    discriminator.train()
    for _ in range(max_epochs):
        for current_flat, text_embedding, _, _target_flat in iterator:
            current_flat = current_flat.to(device)
            text_embedding = text_embedding.to(device)
            noise = torch.randn(current_flat.size(0), int(noise_dim), device=device, dtype=torch.float32)
            fake_delta = generator(current_flat, text_embedding, noise=noise)
            fake_scores = discriminator(current_flat, text_embedding, fake_delta)
            future_flat = reconstruct_future_surface(current_flat, fake_delta)
            future_log_surface = torch.log(torch.clamp(future_flat, min=1e-4)).view(
                current_flat.size(0),
                int(maturity_days_grid.numel()),
                int(strike_grid.numel()),
            )
            bce_loss = generator_bce_loss(fake_scores)
            m_loss = strike_smoothness_penalty(future_log_surface, strike_grid)
            t_loss = maturity_smoothness_penalty(future_log_surface, maturity_days_grid)

            bce_grad = _gradient_norm(parameters, bce_loss)
            m_grad = _gradient_norm(parameters, m_loss)
            t_grad = _gradient_norm(parameters, t_loss)
            if m_grad > 0:
                bce_to_m.append(bce_grad / m_grad)
            if t_grad > 0:
                bce_to_t.append(bce_grad / t_grad)
    alpha_m = float(np.mean(bce_to_m)) if bce_to_m else 1.0
    alpha_tau = float(np.mean(bce_to_t)) if bce_to_t else 1.0
    return alpha_m, alpha_tau
