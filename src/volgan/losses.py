"""Loss helpers for standalone VolGAN training."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .data import denormalize_tensor
from .models import reconstruct_future_surface


def generator_bce_loss(fake_scores: torch.Tensor, *, target_value: float = 1.0) -> torch.Tensor:
    targets = torch.full_like(fake_scores, float(target_value))
    return F.binary_cross_entropy(fake_scores, targets)


def discriminator_bce_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    *,
    real_label_value: float = 1.0,
    fake_label_value: float = 0.0,
) -> torch.Tensor:
    real_targets = torch.full_like(real_scores, float(real_label_value))
    fake_targets = torch.full_like(fake_scores, float(fake_label_value))
    real_loss = F.binary_cross_entropy(real_scores, real_targets)
    fake_loss = F.binary_cross_entropy(fake_scores, fake_targets)
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
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
    noise_dim: int,
    epochs: int,
    real_label_value: float,
    alpha_clip_min: float,
    alpha_clip_max: float,
    normalize_target_delta: bool,
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
        for current_features, text_embedding, _, current_flat, _target_flat in iterator:
            current_features = current_features.to(device)
            text_embedding = text_embedding.to(device)
            current_flat = current_flat.to(device)
            noise = torch.randn(current_features.size(0), int(noise_dim), device=device, dtype=torch.float32)
            fake_delta_norm = generator(current_features, text_embedding, noise=noise)
            fake_scores = discriminator(current_features, text_embedding, fake_delta_norm)
            fake_delta = (
                denormalize_tensor(fake_delta_norm, delta_mean, delta_std)
                if normalize_target_delta
                else fake_delta_norm
            )
            future_flat = reconstruct_future_surface(current_flat, fake_delta)
            future_log_surface = torch.log(torch.clamp(future_flat, min=1e-4)).view(
                current_features.size(0),
                int(maturity_days_grid.numel()),
                int(strike_grid.numel()),
            )
            bce_loss = generator_bce_loss(fake_scores, target_value=float(real_label_value))
            m_loss = strike_smoothness_penalty(future_log_surface, strike_grid)
            t_loss = maturity_smoothness_penalty(future_log_surface, maturity_days_grid)

            bce_grad = _gradient_norm(parameters, bce_loss)
            m_grad = _gradient_norm(parameters, m_loss)
            t_grad = _gradient_norm(parameters, t_loss)
            if m_grad > 0:
                bce_to_m.append(bce_grad / m_grad)
            if t_grad > 0:
                bce_to_t.append(bce_grad / t_grad)
    finite_m = np.asarray([value for value in bce_to_m if np.isfinite(value)], dtype=np.float64)
    finite_t = np.asarray([value for value in bce_to_t if np.isfinite(value)], dtype=np.float64)
    alpha_m = float(np.median(finite_m)) if finite_m.size > 0 else 1.0
    alpha_tau = float(np.median(finite_t)) if finite_t.size > 0 else 1.0
    alpha_m = float(np.clip(alpha_m, float(alpha_clip_min), float(alpha_clip_max)))
    alpha_tau = float(np.clip(alpha_tau, float(alpha_clip_min), float(alpha_clip_max)))
    return alpha_m, alpha_tau
