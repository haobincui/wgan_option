"""Loss and metrics helpers for standalone Transformer WGAN training."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd


def _math_sdpa_context():
    """Force math SDPA for double-backward paths like WGAN-GP."""

    attention_module = getattr(torch.nn, "attention", None)
    if attention_module is not None and hasattr(attention_module, "sdpa_kernel") and hasattr(attention_module, "SDPBackend"):
        return attention_module.sdpa_kernel(backends=[attention_module.SDPBackend.MATH])
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    return nullcontext()


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

    # WGAN-GP differentiates through the critic gradient, which requires
    # double-backward support. Force the math SDPA kernel here because
    # efficient/flash attention kernels can fail on second-order gradients.
    with _math_sdpa_context():
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


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relative_call_prices(surface: torch.Tensor, strike_grid: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(surface, min=1e-4)
    k = strike_grid.to(surface.device).view(1, 1, -1)
    tau = torch.clamp(maturity_days_grid.to(surface.device).view(1, -1, 1) / 365.0, min=1.0 / 365.0)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(1.0 / k) + 0.5 * sigma.pow(2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return _normal_cdf(d1) - k * _normal_cdf(d2)


def calendar_arbitrage_penalty(surface: torch.Tensor, strike_grid: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    call_prices = relative_call_prices(surface, strike_grid, maturity_days_grid)
    if call_prices.size(1) < 2:
        return torch.zeros(call_prices.size(0), device=call_prices.device)
    violations = torch.relu(call_prices[:, :-1, :] - call_prices[:, 1:, :])
    return violations.mean(dim=(1, 2))


def butterfly_arbitrage_penalty(surface: torch.Tensor, strike_grid: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    call_prices = relative_call_prices(surface, strike_grid, maturity_days_grid)
    if call_prices.size(2) < 3:
        return torch.zeros(call_prices.size(0), device=call_prices.device)
    second_diff = call_prices[:, :, 2:] - 2.0 * call_prices[:, :, 1:-1] + call_prices[:, :, :-2]
    return torch.relu(-second_diff).mean(dim=(1, 2))


def total_arbitrage_penalty(surface: torch.Tensor, strike_grid: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    return calendar_arbitrage_penalty(surface, strike_grid, maturity_days_grid) + butterfly_arbitrage_penalty(
        surface,
        strike_grid,
        maturity_days_grid,
    )


def smoothness_penalty(future_surface: torch.Tensor) -> torch.Tensor:
    penalty = torch.zeros(1, device=future_surface.device).squeeze()
    if future_surface.size(1) > 1:
        penalty = penalty + (future_surface[:, 1:, :] - future_surface[:, :-1, :]).pow(2).mean()
    if future_surface.size(2) > 1:
        penalty = penalty + (future_surface[:, :, 1:] - future_surface[:, :, :-1]).pow(2).mean()
    return penalty


def delta_shrink_penalty(fake_future: torch.Tensor, current_surface: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(fake_future - current_surface))


def assemble_generator_loss(
    *,
    adv_loss: torch.Tensor,
    recon_loss: torch.Tensor,
    calendar_penalty: torch.Tensor,
    butterfly_penalty: torch.Tensor,
    smooth_penalty_value: torch.Tensor,
    delta_shrink: torch.Tensor,
    pure_adversarial: bool,
    lambda_recon: float,
    lambda_calendar: float,
    lambda_butterfly: float,
    lambda_smooth: float,
    lambda_delta_shrink: float,
    use_calendar_constraint: bool,
    use_butterfly_constraint: bool,
    use_smooth_constraint: bool,
) -> torch.Tensor:
    total = adv_loss
    if not pure_adversarial:
        total = total + float(lambda_recon) * recon_loss
    if use_calendar_constraint:
        total = total + float(lambda_calendar) * calendar_penalty
    if use_butterfly_constraint:
        total = total + float(lambda_butterfly) * butterfly_penalty
    if use_smooth_constraint:
        total = total + float(lambda_smooth) * smooth_penalty_value
    if not pure_adversarial and float(lambda_delta_shrink) > 0.0:
        total = total + float(lambda_delta_shrink) * delta_shrink
    return total


def mean_abs_error(predicted_surface, target_surface) -> float:
    predicted = np.asarray(predicted_surface, dtype=np.float32)
    target = np.asarray(target_surface, dtype=np.float32)
    return float(np.mean(np.abs(predicted - target)))


def summarize_baseline_aware_metrics(
    recon_values: Iterable[float],
    current_recon_values: Iterable[float],
    *,
    baseline_penalty_weight: float,
) -> dict[str, float]:
    recon_array = np.asarray(list(recon_values), dtype=np.float32)
    current_array = np.asarray(list(current_recon_values), dtype=np.float32)
    val_recon = float(recon_array.mean()) if recon_array.size else 0.0
    val_current_recon = float(current_array.mean()) if current_array.size else 0.0
    val_baseline_gap = float(val_recon - val_current_recon)
    val_hybrid_score = float(val_recon + float(baseline_penalty_weight) * max(0.0, val_baseline_gap))
    return {
        "val_recon": val_recon,
        "val_current_recon": val_current_recon,
        "val_baseline_gap": val_baseline_gap,
        "val_hybrid_score": val_hybrid_score,
    }


def resolve_monitor_metric(metrics: Mapping[str, float], monitor_metric: str) -> float:
    metric_name = str(monitor_metric).strip()
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise ValueError(
            f"Requested best_checkpoint_metric='{metric_name}' is unavailable. Available metrics: {available}"
        )
    return float(metrics[metric_name])


def reweight_scenarios(
    penalties: np.ndarray,
    *,
    beta_mode: str,
    beta_value: float,
) -> tuple[np.ndarray, float]:
    penalties = np.asarray(penalties, dtype=np.float64).reshape(-1)
    if penalties.size <= 0:
        raise ValueError("reweight_scenarios requires at least one penalty.")
    normalized_mode = str(beta_mode).strip().lower()
    if normalized_mode == "fixed":
        effective_beta = float(beta_value)
    elif normalized_mode == "adaptive":
        max_penalty = float(np.max(np.abs(penalties)))
        effective_beta = 0.0 if max_penalty <= 0.0 else float(beta_value) / max_penalty
    else:
        raise ValueError(f"reweight_beta_mode must be one of ['fixed', 'adaptive'], got: {beta_mode}")
    logits = -effective_beta * penalties
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return np.full_like(weights, 1.0 / float(weights.size)), effective_beta
    return weights / weight_sum, effective_beta


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))
