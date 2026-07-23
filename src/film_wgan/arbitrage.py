"""Arbitrage penalties and scenario reweighting for standalone FiLM WGAN."""

from __future__ import annotations

import numpy as np
import torch


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


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


def calendar_arbitrage_violation_rate(
    surface: torch.Tensor,
    strike_grid: torch.Tensor,
    maturity_days_grid: torch.Tensor,
    *,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Return the fraction of adjacent maturity constraints that are violated."""

    call_prices = relative_call_prices(surface, strike_grid, maturity_days_grid)
    if call_prices.size(1) < 2:
        return torch.zeros(call_prices.size(0), device=call_prices.device)
    violations = call_prices[:, :-1, :] - call_prices[:, 1:, :]
    return (violations > float(tolerance)).to(dtype=surface.dtype).mean(dim=(1, 2))


def butterfly_arbitrage_violation_rate(
    surface: torch.Tensor,
    strike_grid: torch.Tensor,
    maturity_days_grid: torch.Tensor,
    *,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Return the fraction of discrete strike-convexity constraints violated."""

    call_prices = relative_call_prices(surface, strike_grid, maturity_days_grid)
    if call_prices.size(2) < 3:
        return torch.zeros(call_prices.size(0), device=call_prices.device)
    second_diff = call_prices[:, :, 2:] - 2.0 * call_prices[:, :, 1:-1] + call_prices[:, :, :-2]
    return (second_diff < -float(tolerance)).to(dtype=surface.dtype).mean(dim=(1, 2))


def total_arbitrage_penalty(surface: torch.Tensor, strike_grid: torch.Tensor, maturity_days_grid: torch.Tensor) -> torch.Tensor:
    return calendar_arbitrage_penalty(surface, strike_grid, maturity_days_grid) + butterfly_arbitrage_penalty(
        surface,
        strike_grid,
        maturity_days_grid,
    )


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
