"""Loss helpers for standalone FiLM WGAN training."""

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
    has_text: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = real_future_surface.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_future_surface.device)
    interpolated = alpha * real_future_surface + (1.0 - alpha) * fake_future_surface
    interpolated.requires_grad_(True)

    if has_text is None:
        interpolated_scores = critic(interpolated, current_surface, text_embedding)
    else:
        interpolated_scores = critic(
            interpolated,
            current_surface,
            text_embedding,
            has_text=has_text,
        )
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


def _expand_weights_to_match(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = weights.to(device=values.device, dtype=values.dtype)
    if expanded.dim() > values.dim():
        raise ValueError(f"weights has more dimensions than values: {expanded.shape} vs {values.shape}")
    while expanded.dim() < values.dim():
        expanded = expanded.unsqueeze(0)
    return torch.broadcast_to(expanded, values.shape)


def build_reconstruction_weight_template(
    *,
    strike_grid: torch.Tensor,
    maturity_days_grid: torch.Tensor,
    mode: str,
    atm_range: float,
    short_end_max_days: float,
    atm_multiplier: float,
) -> torch.Tensor:
    normalized_mode = str(mode).strip().lower()
    strike_grid = strike_grid.reshape(-1)
    maturity_days_grid = maturity_days_grid.reshape(-1)
    weights = torch.ones(
        (maturity_days_grid.numel(), strike_grid.numel()),
        device=strike_grid.device,
        dtype=strike_grid.dtype,
    )
    if normalized_mode == "uniform":
        return weights
    if normalized_mode != "short_atm_band":
        raise ValueError(f"Unsupported recon_weight_mode: {mode}")
    if float(atm_multiplier) <= 0.0:
        raise ValueError(f"recon_atm_multiplier must be positive, got {atm_multiplier}")
    if float(atm_range) < 0.0:
        raise ValueError(f"recon_atm_range must be non-negative, got {atm_range}")

    strike_mask = torch.abs(strike_grid - 1.0) <= float(atm_range) + 1e-6
    maturity_mask = maturity_days_grid <= float(short_end_max_days) + 1e-6
    emphasis_mask = maturity_mask.view(-1, 1) & strike_mask.view(1, -1)
    if torch.any(emphasis_mask):
        weights = torch.where(emphasis_mask, torch.full_like(weights, float(atm_multiplier)), weights)

    normalizer = weights.mean()
    if not torch.isfinite(normalizer) or float(normalizer.item()) <= 0.0:
        raise ValueError("Reconstruction weight template must have a positive finite mean.")
    return weights / normalizer


def weighted_surface_mae(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target must share the same shape, got {predicted.shape} vs {target.shape}")
    has_explicit_batch = predicted.dim() == weights.dim() + 1
    expanded_weights = _expand_weights_to_match(predicted, weights)
    abs_error = torch.abs(predicted - target)
    if abs_error.dim() == 0:
        denom = torch.clamp(expanded_weights, min=1e-12)
        return abs_error * expanded_weights / denom
    if not has_explicit_batch:
        weighted_abs = abs_error * expanded_weights
        return weighted_abs.sum() / torch.clamp(expanded_weights.sum(), min=1e-12)
    weighted_abs = abs_error * expanded_weights
    per_sample = weighted_abs.reshape(abs_error.shape[0], -1).sum(dim=1) / torch.clamp(
        expanded_weights.reshape(abs_error.shape[0], -1).sum(dim=1),
        min=1e-12,
    )
    return per_sample.mean()


def reconstruction_loss(fake_future_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
    """MAE between generated and real target surface, penalising lazy zero-delta generators."""
    return torch.nn.functional.l1_loss(fake_future_flat, target_flat)


def build_atm_short_mask(
    *,
    strike_grid: torch.Tensor,
    maturity_days_grid: torch.Tensor,
    atm_range: float,
    max_days: float,
) -> torch.Tensor:
    if float(atm_range) < 0.0:
        raise ValueError(f"atm_short_range must be non-negative, got {atm_range}")
    if float(max_days) <= 0.0:
        raise ValueError(f"atm_short_max_days must be positive, got {max_days}")
    strike_grid = strike_grid.reshape(-1)
    maturity_days_grid = maturity_days_grid.reshape(-1)
    strike_mask = torch.abs(strike_grid - 1.0) <= float(atm_range) + 1e-6
    maturity_mask = maturity_days_grid <= float(max_days) + 1e-6
    mask = (maturity_mask.view(-1, 1) & strike_mask.view(1, -1)).to(dtype=strike_grid.dtype)
    return mask


def atm_short_pure_mae(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MAE over eligible short-maturity x ATM cells and samples."""
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target must share the same shape, got {predicted.shape} vs {target.shape}")
    expanded_mask = _expand_weights_to_match(predicted, mask)
    abs_error = torch.abs(predicted - target)
    if abs_error.dim() == 0:
        denom = torch.clamp(expanded_mask, min=1e-12)
        return abs_error * expanded_mask / denom
    has_explicit_batch = predicted.dim() == mask.dim() + 1
    masked = abs_error * expanded_mask
    if not has_explicit_batch:
        return masked.sum() / torch.clamp(expanded_mask.sum(), min=1e-12)
    per_sample_denominator = expanded_mask.reshape(abs_error.shape[0], -1).sum(
        dim=1
    )
    eligible = per_sample_denominator > 0.0
    if not bool(torch.any(eligible)):
        return masked.sum() * 0.0
    per_sample = masked.reshape(abs_error.shape[0], -1).sum(dim=1) / torch.clamp(
        per_sample_denominator,
        min=1e-12,
    )
    return per_sample[eligible].mean()


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))
