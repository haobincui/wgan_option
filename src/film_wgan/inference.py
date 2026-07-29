"""Scenario sampling and arbitrage-weighted aggregation for standalone FiLM WGAN."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utils.training_paths import generate_result_dir, infer_run_dir_from_checkpoint
from .arbitrage import (
    butterfly_arbitrage_violation_rate,
    calendar_arbitrage_violation_rate,
    reweight_scenarios,
    total_arbitrage_penalty,
)
from .config import FilmWGANSampleConfig, FilmWGANTrainConfig
from .data import (
    FilmWGANNormalizationStats,
    FilmWGANSample,
    denormalize_tensor,
    load_film_wgan_samples,
    normalize_surface_tensor,
    normalize_tensor,
    split_samples,
)
from .io import load_checkpoint, write_csv, write_json
from .losses import (
    atm_short_pure_mae,
    build_atm_short_mask,
    build_reconstruction_weight_template,
    weighted_surface_mae,
)
from .models import FilmWGANGenerator, reconstruct_future_surface
from .plotting import extract_atm_short_value, plot_atm_vol_timeseries, plot_film_wgan_payload


@dataclass(frozen=True)
class TensorNormalizationStats:
    """Normalization statistics stored as broadcastable device tensors."""

    current_log_mean: torch.Tensor
    current_log_std: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor
    text_mean: torch.Tensor
    text_std: torch.Tensor


def _metadata_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _to_device_row(values: Any, device: torch.device) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return torch.tensor(array, dtype=torch.float32, device=device)


def normalization_stats_to_tensors(
    stats: FilmWGANNormalizationStats | Mapping[str, Any],
    device: torch.device,
) -> TensorNormalizationStats:
    if isinstance(stats, FilmWGANNormalizationStats):
        payload = {
            "current_log_mean": stats.current_log_mean,
            "current_log_std": stats.current_log_std,
            "delta_mean": stats.delta_mean,
            "delta_std": stats.delta_std,
            "text_mean": stats.text_mean,
            "text_std": stats.text_std,
        }
    else:
        payload = dict(stats)
    return TensorNormalizationStats(
        current_log_mean=_to_device_row(payload["current_log_mean"], device),
        current_log_std=_to_device_row(payload["current_log_std"], device),
        delta_mean=_to_device_row(payload["delta_mean"], device),
        delta_std=_to_device_row(payload["delta_std"], device),
        text_mean=_to_device_row(payload["text_mean"], device),
        text_std=_to_device_row(payload["text_std"], device),
    )


def prepare_condition_tensors(
    sample: FilmWGANSample,
    normalization: TensorNormalizationStats,
    device: torch.device,
    *,
    normalize_current_surface: bool,
    normalize_text_embedding: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = sample.surface_shape
    current_flat = torch.tensor(sample.current_surface.reshape(1, -1), dtype=torch.float32, device=device)
    text_embedding = torch.tensor(sample.text_embedding.reshape(1, -1), dtype=torch.float32, device=device)
    if normalize_current_surface:
        current_features_flat = normalize_surface_tensor(current_flat, normalization.current_log_mean, normalization.current_log_std)
    else:
        current_features_flat = torch.log(torch.clamp(current_flat, min=1e-4))
    current_support = torch.tensor(
        (
            np.asarray(sample.current_support_mask, dtype=np.float32)
            if sample.current_support_mask is not None
            else np.ones(sample.surface_shape, dtype=np.float32)
        ).reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )
    current_features_flat = torch.where(
        current_support > 0.0,
        current_features_flat,
        torch.zeros_like(current_features_flat),
    )
    if normalize_text_embedding:
        text_features = normalize_tensor(text_embedding, normalization.text_mean, normalization.text_std)
    else:
        text_features = text_embedding
    current_features = current_features_flat.view(1, 1, height, width)
    if sample.current_support_mask is not None:
        current_features = torch.cat(
            [current_features, current_support.view(1, 1, height, width)],
            dim=1,
        )
    return current_flat, current_features, text_features


def generate_surface_scenarios(
    *,
    generator: FilmWGANGenerator,
    sample: FilmWGANSample,
    normalization: TensorNormalizationStats,
    noise_dim: int,
    mc_samples: int,
    seed: int,
    device: torch.device,
    normalize_current_surface: bool,
    normalize_text_embedding: bool,
    normalize_target_delta: bool,
) -> np.ndarray:
    current_flat, current_features, text_features = prepare_condition_tensors(
        sample,
        normalization,
        device,
        normalize_current_surface=normalize_current_surface,
        normalize_text_embedding=normalize_text_embedding,
    )
    height, width = sample.surface_shape
    has_text = torch.tensor(
        [float(sample.metadata.get("has_text", 1.0))],
        dtype=torch.float32,
        device=device,
    )
    generated_surfaces: list[np.ndarray] = []
    with torch.no_grad():
        draw_count = 1 if int(noise_dim) <= 0 else max(1, int(mc_samples))
        for draw_idx in range(draw_count):
            generator_noise = torch.Generator(device="cpu")
            seed_payload = (
                f"{int(seed)}|{sample.surface_pair_id}|{int(draw_idx)}"
            ).encode("utf-8")
            draw_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8],
                byteorder="big",
                signed=False,
            ) % (2**63 - 1)
            generator_noise.manual_seed(draw_seed)
            noise = torch.randn((1, int(noise_dim)), generator=generator_noise, dtype=torch.float32).to(device)
            fake_delta_norm = generator(
                current_features,
                text_features,
                noise=noise,
                has_text=has_text,
            )
            fake_delta = (
                denormalize_tensor(fake_delta_norm, normalization.delta_mean, normalization.delta_std)
                if normalize_target_delta
                else fake_delta_norm
            )
            future_flat = reconstruct_future_surface(current_flat, fake_delta)
            generated_surfaces.append(future_flat.view(height, width).detach().cpu().numpy().astype(np.float32))
    return np.stack(generated_surfaces, axis=0).astype(np.float64)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, values, axes=(0, 0))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    output = np.zeros(values.shape[1], dtype=np.float64)
    for column_idx in range(values.shape[1]):
        order = np.argsort(values[:, column_idx])
        sorted_values = values[order, column_idx]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, float(quantile), side="left"))
        index = min(max(index, 0), len(sorted_values) - 1)
        output[column_idx] = sorted_values[index]
    return output


def _weight_entropy(weights: np.ndarray) -> float:
    normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
    if normalized.size <= 1:
        return 0.0
    safe = np.clip(normalized, 1e-12, None)
    entropy = float(-np.sum(safe * np.log(safe)))
    return entropy / math.log(float(normalized.size))


def energy_score(
    surface_stack: np.ndarray,
    target_surface: np.ndarray,
    weights: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> float:
    """Weighted multivariate energy score in per-grid-cell IV units."""

    scenarios = np.asarray(surface_stack, dtype=np.float64).reshape(len(surface_stack), -1)
    target = np.asarray(target_surface, dtype=np.float64).reshape(1, -1)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool).reshape(-1)
        scenarios = scenarios[:, mask]
        target = target[:, mask]
    if scenarios.shape[1] == 0:
        return float("nan")
    normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    scale = math.sqrt(float(scenarios.shape[1]))
    target_distances = np.linalg.norm(scenarios - target, axis=1) / scale
    squared_norms = np.sum(scenarios * scenarios, axis=1, keepdims=True)
    squared_distances = np.maximum(squared_norms + squared_norms.T - 2.0 * scenarios @ scenarios.T, 0.0)
    pair_distances = np.sqrt(squared_distances) / scale
    return float(normalized_weights @ target_distances - 0.5 * normalized_weights @ pair_distances @ normalized_weights)


def variogram_score(
    surface_stack: np.ndarray,
    target_surface: np.ndarray,
    weights: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    order: float = 0.5,
) -> float:
    """Local-neighbour variogram score for spatial dependence across the IV grid."""

    scenarios = np.asarray(surface_stack, dtype=np.float64)
    target = np.asarray(target_surface, dtype=np.float64)
    if scenarios.ndim != 3 or target.shape != scenarios.shape[1:]:
        raise ValueError("Variogram score expects scenarios [draw, maturity, strike].")
    mask = (
        np.asarray(support_mask, dtype=bool)
        if support_mask is not None
        else np.ones(target.shape, dtype=bool)
    )
    pair_terms: list[np.ndarray] = []
    target_terms: list[np.ndarray] = []
    if target.shape[1] > 1:
        valid = mask[:, 1:] & mask[:, :-1]
        if np.any(valid):
            pair_terms.append(np.abs(scenarios[:, :, 1:] - scenarios[:, :, :-1])[:, valid] ** float(order))
            target_terms.append(np.abs(target[:, 1:] - target[:, :-1])[valid] ** float(order))
    if target.shape[0] > 1:
        valid = mask[1:, :] & mask[:-1, :]
        if np.any(valid):
            pair_terms.append(np.abs(scenarios[:, 1:, :] - scenarios[:, :-1, :])[:, valid] ** float(order))
            target_terms.append(np.abs(target[1:, :] - target[:-1, :])[valid] ** float(order))
    if not pair_terms:
        return float("nan")
    scenario_differences = np.concatenate(pair_terms, axis=1)
    target_differences = np.concatenate(target_terms)
    expected_differences = np.asarray(weights, dtype=np.float64) @ scenario_differences
    return float(np.mean(np.square(target_differences - expected_differences)))


def _probabilistic_metrics(
    *,
    surface_stack: np.ndarray,
    target_surface: np.ndarray,
    weights: np.ndarray,
    calibration_levels: Sequence[float],
    support_mask: np.ndarray | None = None,
) -> dict[str, float]:
    flat_stack = np.asarray(surface_stack, dtype=np.float64).reshape(len(surface_stack), -1)
    flat_target = np.asarray(target_surface, dtype=np.float64).reshape(-1)
    flat_mask = (
        np.asarray(support_mask, dtype=bool).reshape(-1)
        if support_mask is not None
        else np.ones(flat_target.shape, dtype=bool)
    )
    flat_stack = flat_stack[:, flat_mask]
    flat_target = flat_target[flat_mask]
    if flat_target.size == 0:
        return {
            "energy_score": float("nan"),
            "variogram_score": float("nan"),
            "scenario_spread": float("nan"),
            "effective_scenario_count": float(1.0 / np.sum(np.square(weights))),
            "calibration_error": float("nan"),
            "mc_surface_mae_se": float("nan"),
        }
    weighted_mean = _weighted_mean(flat_stack, weights)
    scale = math.sqrt(float(flat_stack.shape[1]))
    spread = float(np.sum(weights * (np.linalg.norm(flat_stack - weighted_mean, axis=1) / scale)))
    metrics: dict[str, float] = {
        "energy_score": energy_score(flat_stack, flat_target, weights),
        "variogram_score": variogram_score(
            surface_stack,
            target_surface,
            weights,
            support_mask=support_mask,
        ),
        "scenario_spread": spread,
        "effective_scenario_count": float(1.0 / np.sum(np.square(weights))),
    }
    scenario_mae = np.mean(np.abs(flat_stack - flat_target.reshape(1, -1)), axis=1)
    metrics["mc_surface_mae_se"] = (
        float(np.std(scenario_mae, ddof=1) / math.sqrt(float(scenario_mae.size)))
        if scenario_mae.size > 1
        else 0.0
    )
    calibration_errors: list[float] = []
    for raw_level in calibration_levels:
        level = float(raw_level)
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        lower = _weighted_quantile(flat_stack, weights, lower_q)
        upper = _weighted_quantile(flat_stack, weights, upper_q)
        coverage = float(np.mean((flat_target >= lower) & (flat_target <= upper)))
        width = float(np.mean(upper - lower))
        suffix = str(int(round(level * 100.0)))
        metrics[f"coverage_{suffix}"] = coverage
        metrics[f"interval_width_{suffix}"] = width
        calibration_errors.append(abs(coverage - level))
    metrics["calibration_error"] = float(np.mean(calibration_errors)) if calibration_errors else 0.0
    return metrics


def _surface_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> dict[str, float]:
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if support_mask is not None:
        diff = diff[np.asarray(support_mask, dtype=bool)]
    if diff.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _apply_residual_blend(
    generated_surface: np.ndarray,
    current_surface: np.ndarray,
    *,
    residual_blend_alpha: float,
) -> np.ndarray:
    alpha = float(residual_blend_alpha)
    generated = np.asarray(generated_surface, dtype=np.float32)
    current = np.asarray(current_surface, dtype=np.float32)
    if math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return generated
    return (current + alpha * (generated - current)).astype(np.float32)


def _short_atm_metrics(
    *,
    generated_surface: np.ndarray,
    current_surface: np.ndarray,
    target_surface: np.ndarray,
    recon_weights_surface: torch.Tensor | None,
    atm_short_mask_surface: torch.Tensor | None,
    support_mask: np.ndarray | None = None,
) -> dict[str, float]:
    metric_device = (
        recon_weights_surface.device
        if recon_weights_surface is not None
        else (
            atm_short_mask_surface.device
            if atm_short_mask_surface is not None
            else torch.device("cpu")
        )
    )
    generated_tensor = torch.tensor(
        np.asarray(generated_surface, dtype=np.float32),
        dtype=torch.float32,
        device=metric_device,
    )
    current_tensor = torch.tensor(
        np.asarray(current_surface, dtype=np.float32),
        dtype=torch.float32,
        device=metric_device,
    )
    target_tensor = torch.tensor(
        np.asarray(target_surface, dtype=np.float32),
        dtype=torch.float32,
        device=metric_device,
    )
    support_tensor = torch.tensor(
        (
            np.asarray(support_mask, dtype=np.float32)
            if support_mask is not None
            else np.ones_like(target_surface, dtype=np.float32)
        ),
        dtype=torch.float32,
        device=metric_device,
    )

    metrics = {
        "short_atm_weighted_mae": 0.0,
        "current_short_atm_weighted_mae": 0.0,
        "short_atm_mae_gap_vs_current": 0.0,
        "short_atm_weighted_win_flag_vs_current": 0.0,
        "atm_short_pure_mae": 0.0,
        "current_atm_short_pure_mae": 0.0,
        "atm_short_pure_mae_gap_vs_current": 0.0,
        "atm_short_pure_win_flag_vs_current": 0.0,
    }
    if recon_weights_surface is not None:
        supported_weights = recon_weights_surface * support_tensor
        generated_weighted = float(
            weighted_surface_mae(generated_tensor, target_tensor, supported_weights).detach().cpu()
        )
        current_weighted = float(
            weighted_surface_mae(current_tensor, target_tensor, supported_weights).detach().cpu()
        )
        metrics.update(
            {
                "short_atm_weighted_mae": generated_weighted,
                "current_short_atm_weighted_mae": current_weighted,
                "short_atm_mae_gap_vs_current": generated_weighted - current_weighted,
                "short_atm_weighted_win_flag_vs_current": 1.0 if generated_weighted < current_weighted else 0.0,
            }
        )
    supported_atm_mask = (
        atm_short_mask_surface * support_tensor
        if atm_short_mask_surface is not None
        else None
    )
    if supported_atm_mask is not None and float(supported_atm_mask.sum().item()) > 0.0:
        generated_pure = float(
            atm_short_pure_mae(generated_tensor, target_tensor, supported_atm_mask).detach().cpu()
        )
        current_pure = float(
            atm_short_pure_mae(current_tensor, target_tensor, supported_atm_mask).detach().cpu()
        )
        metrics.update(
            {
                "atm_short_pure_mae": generated_pure,
                "current_atm_short_pure_mae": current_pure,
                "atm_short_pure_mae_gap_vs_current": generated_pure - current_pure,
                "atm_short_pure_win_flag_vs_current": 1.0 if generated_pure < current_pure else 0.0,
            }
        )
    elif support_mask is not None:
        metrics.update(
            {
                "atm_short_pure_mae": float("nan"),
                "current_atm_short_pure_mae": float("nan"),
                "atm_short_pure_mae_gap_vs_current": float("nan"),
                "atm_short_pure_win_flag_vs_current": float("nan"),
            }
        )
    return metrics


def summarize_surface_scenarios(
    *,
    surface_stack: np.ndarray,
    current_surface: np.ndarray,
    target_surface: np.ndarray,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    reweight_beta_mode: str,
    reweight_beta: float,
    aggregation_mode: str,
    quantiles: Sequence[float] = (),
    calibration_levels: Sequence[float] = (0.5, 0.8, 0.9),
    arbitrage_violation_tolerance: float = 1e-8,
    recon_weights_surface: torch.Tensor | None = None,
    atm_short_mask_surface: torch.Tensor | None = None,
    residual_blend_alpha: float = 1.0,
    support_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    surface_stack = np.asarray(surface_stack, dtype=np.float64)
    current_surface = np.asarray(current_surface, dtype=np.float32)
    target_surface = np.asarray(target_surface, dtype=np.float32)
    resolved_support = (
        np.asarray(support_mask, dtype=bool)
        if support_mask is not None
        else np.ones(target_surface.shape, dtype=bool)
    )
    if resolved_support.shape != target_surface.shape:
        raise ValueError(
            f"support_mask shape {resolved_support.shape} does not match target {target_surface.shape}."
        )
    full_rectangular_support = bool(np.all(resolved_support))
    if full_rectangular_support:
        penalties = total_arbitrage_penalty(
            torch.tensor(surface_stack, dtype=torch.float32),
            torch.tensor(np.asarray(strike_grid, dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(maturity_days_grid, dtype=np.float32), dtype=torch.float32),
        ).detach().cpu().numpy().astype(np.float64)
        weights, effective_beta = reweight_scenarios(
            penalties,
            beta_mode=reweight_beta_mode,
            beta_value=float(reweight_beta),
        )
    else:
        # Static-arbitrage penalties require a complete rectangular surface.
        penalties = np.zeros(surface_stack.shape[0], dtype=np.float64)
        weights = np.full(surface_stack.shape[0], 1.0 / float(surface_stack.shape[0]))
        effective_beta = 0.0
    normalized_aggregation = str(aggregation_mode).strip().lower()
    if normalized_aggregation != "weighted_mean":
        raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
    weighted_mean_surface = _weighted_mean(surface_stack, weights).astype(np.float32)
    flat_surface_stack = surface_stack.reshape(surface_stack.shape[0], -1)
    blended_surface = _apply_residual_blend(
        weighted_mean_surface,
        current_surface,
        residual_blend_alpha=float(residual_blend_alpha),
    )
    blended_surface_stack = np.stack(
        [
            _apply_residual_blend(
                scenario,
                current_surface,
                residual_blend_alpha=float(residual_blend_alpha),
            )
            for scenario in surface_stack
        ],
        axis=0,
    ).astype(np.float64)
    quantile_surfaces = {
        f"q_{float(quantile):.2f}": _apply_residual_blend(
            _weighted_quantile(flat_surface_stack, weights, float(quantile))
            .reshape(target_surface.shape)
            .astype(np.float32),
            current_surface,
            residual_blend_alpha=float(residual_blend_alpha),
        ).tolist()
        for quantile in quantiles
    }
    current_metrics = _surface_metrics(current_surface, target_surface, resolved_support)
    metrics = _surface_metrics(blended_surface, target_surface, resolved_support)
    generated_current_metrics = _surface_metrics(
        blended_surface,
        current_surface,
        resolved_support,
    )
    comparison_metrics = {
        "mae_gap_vs_current": metrics["mae"] - current_metrics["mae"],
        "rmse_gap_vs_current": metrics["rmse"] - current_metrics["rmse"],
        "max_abs_gap_vs_current": metrics["max_abs"] - current_metrics["max_abs"],
        "win_flag_vs_current": (
            1.0
            if np.isfinite(metrics["mae"])
            and np.isfinite(current_metrics["mae"])
            and metrics["mae"] < current_metrics["mae"]
            else 0.0
        ),
    }
    if full_rectangular_support:
        strike_tensor = torch.tensor(np.asarray(strike_grid, dtype=np.float32), dtype=torch.float32)
        maturity_tensor = torch.tensor(np.asarray(maturity_days_grid, dtype=np.float32), dtype=torch.float32)
        scenario_tensor = torch.tensor(blended_surface_stack, dtype=torch.float32)
        aggregate_tensor = torch.tensor(blended_surface[None, ...], dtype=torch.float32)
        scenario_calendar_rates = (
            calendar_arbitrage_violation_rate(
                scenario_tensor,
                strike_tensor,
                maturity_tensor,
                tolerance=float(arbitrage_violation_tolerance),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scenario_butterfly_rates = (
            butterfly_arbitrage_violation_rate(
                scenario_tensor,
                strike_tensor,
                maturity_tensor,
                tolerance=float(arbitrage_violation_tolerance),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        arbitrage_metrics = {
            "calendar_violation_rate": float(
                calendar_arbitrage_violation_rate(
                    aggregate_tensor,
                    strike_tensor,
                    maturity_tensor,
                    tolerance=float(arbitrage_violation_tolerance),
                )[0]
            ),
            "butterfly_violation_rate": float(
                butterfly_arbitrage_violation_rate(
                    aggregate_tensor,
                    strike_tensor,
                    maturity_tensor,
                    tolerance=float(arbitrage_violation_tolerance),
                )[0]
            ),
            "scenario_calendar_violation_rate": float(weights @ scenario_calendar_rates),
            "scenario_butterfly_violation_rate": float(weights @ scenario_butterfly_rates),
            "violation_tolerance": float(arbitrage_violation_tolerance),
            "status": "computed_full_rectangular_support",
        }
    else:
        arbitrage_metrics = {
            "calendar_violation_rate": float("nan"),
            "butterfly_violation_rate": float("nan"),
            "scenario_calendar_violation_rate": float("nan"),
            "scenario_butterfly_violation_rate": float("nan"),
            "violation_tolerance": float(arbitrage_violation_tolerance),
            "status": "not_computed_irregular_raw_support",
        }
    return {
        "generated_surface": blended_surface.astype(float).tolist(),
        "quantile_surfaces": quantile_surfaces,
        "penalties": penalties.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "effective_beta": float(effective_beta),
        "weight_entropy": float(_weight_entropy(weights)),
        "penalty_mean": float(np.mean(penalties)) if penalties.size else 0.0,
        "penalty_std": float(np.std(penalties)) if penalties.size else 0.0,
        "metrics": metrics,
        "current_metrics": current_metrics,
        "generated_current_metrics": generated_current_metrics,
        "comparison_metrics": comparison_metrics,
        "probabilistic_metrics": _probabilistic_metrics(
            surface_stack=blended_surface_stack,
            target_surface=target_surface,
            weights=weights,
            calibration_levels=calibration_levels,
            support_mask=resolved_support,
        ),
        "arbitrage_metrics": arbitrage_metrics,
        "short_atm_metrics": _short_atm_metrics(
            generated_surface=blended_surface,
            current_surface=current_surface,
            target_surface=target_surface,
            recon_weights_surface=recon_weights_surface,
            atm_short_mask_surface=atm_short_mask_surface,
            support_mask=resolved_support,
        ),
        "support_metrics": {
            "supported_cell_count": int(resolved_support.sum()),
            "supported_cell_fraction": float(resolved_support.mean()),
            "full_rectangular_support": full_rectangular_support,
        },
    }


def build_sample_payload(
    *,
    generator: FilmWGANGenerator,
    sample: FilmWGANSample,
    normalization: TensorNormalizationStats,
    noise_dim: int,
    mc_samples: int,
    seed: int,
    device: torch.device,
    reweight_beta_mode: str,
    reweight_beta: float,
    aggregation_mode: str,
    quantiles: Sequence[float],
    calibration_levels: Sequence[float],
    arbitrage_violation_tolerance: float,
    checkpoint_path: str,
    split: str,
    selection_mode: str,
    recon_weights_surface: torch.Tensor | None,
    atm_short_mask_surface: torch.Tensor | None,
    normalize_current_surface: bool,
    normalize_text_embedding: bool,
    normalize_target_delta: bool,
    residual_blend_alpha: float,
) -> dict[str, Any]:
    surface_stack = generate_surface_scenarios(
        generator=generator,
        sample=sample,
        normalization=normalization,
        noise_dim=noise_dim,
        mc_samples=mc_samples,
        seed=seed,
        device=device,
        normalize_current_surface=normalize_current_surface,
        normalize_text_embedding=normalize_text_embedding,
        normalize_target_delta=normalize_target_delta,
    )
    summary = summarize_surface_scenarios(
        surface_stack=surface_stack,
        current_surface=sample.current_surface,
        target_surface=sample.target_surface,
        strike_grid=sample.strike_grid,
        maturity_days_grid=sample.maturity_days_grid,
        reweight_beta_mode=reweight_beta_mode,
        reweight_beta=reweight_beta,
        aggregation_mode=aggregation_mode,
        quantiles=quantiles,
        calibration_levels=calibration_levels,
        arbitrage_violation_tolerance=float(arbitrage_violation_tolerance),
        recon_weights_surface=recon_weights_surface,
        atm_short_mask_surface=atm_short_mask_surface,
        residual_blend_alpha=float(residual_blend_alpha),
        support_mask=sample.evaluation_support_mask,
    )
    return {
        "sample_id": sample.sample_id,
        "mode": "film_wgan",
        "global_index": int(sample.global_index),
        "surface_pair_id": sample.surface_pair_id,
        "news_timestamp_utc": sample.timestamp,
        "current_snapshot_time_utc": sample.current_snapshot_time_utc,
        "target_snapshot_time_utc": sample.target_snapshot_time_utc,
        "strike_grid": sample.strike_grid.astype(float).tolist(),
        "maturity_days_grid": sample.maturity_days_grid.astype(float).tolist(),
        "current_surface": sample.current_surface.astype(float).tolist(),
        "generated_surface": summary["generated_surface"],
        "real_surface": sample.target_surface.astype(float).tolist(),
        "target_surface": sample.target_surface.astype(float).tolist(),
        "quantile_surfaces": summary["quantile_surfaces"],
        "penalties": summary["penalties"],
        "weights": summary["weights"],
        "effective_beta": summary["effective_beta"],
        "weight_entropy": summary["weight_entropy"],
        "penalty_mean": summary["penalty_mean"],
        "penalty_std": summary["penalty_std"],
        "metrics": summary["metrics"],
        "current_metrics": summary["current_metrics"],
        "generated_current_metrics": summary["generated_current_metrics"],
        "comparison_metrics": summary["comparison_metrics"],
        "probabilistic_metrics": summary["probabilistic_metrics"],
        "arbitrage_metrics": summary["arbitrage_metrics"],
        "short_atm_metrics": summary["short_atm_metrics"],
        "support_metrics": summary["support_metrics"],
        "current_support_mask": (
            np.asarray(sample.current_support_mask, dtype=bool).tolist()
            if sample.current_support_mask is not None
            else None
        ),
        "target_support_mask": (
            np.asarray(sample.target_support_mask, dtype=bool).tolist()
            if sample.target_support_mask is not None
            else None
        ),
        "evaluation_support_mask": sample.evaluation_support_mask.tolist(),
        "metadata": {
            "pair_quality_label": sample.metadata.get("pair_quality_label", ""),
            "publication_timestamp_utc": sample.metadata.get(
                "publication_timestamp_utc",
                "",
            ),
            "publication_availability_lag_minutes": int(
                sample.metadata.get("publication_availability_lag_minutes", 0)
            ),
            "news_alignment_mode": sample.metadata.get(
                "news_alignment_mode",
                "exact",
            ),
            "alignment_type": sample.metadata.get(
                "alignment_type",
                "exact",
            ),
            "origin_shift_minutes": _metadata_float(
                sample.metadata.get("origin_shift_minutes", 0.0)
            ),
            "origin_shift_minutes_min": _metadata_float(
                sample.metadata.get("origin_shift_minutes_min", 0.0)
            ),
            "origin_shift_minutes_max": _metadata_float(
                sample.metadata.get("origin_shift_minutes_max", 0.0)
            ),
            "origin_shift_minutes_mean": _metadata_float(
                sample.metadata.get("origin_shift_minutes_mean", 0.0)
            ),
            "source_alignment_types": sample.metadata.get(
                "source_alignment_types",
                [sample.metadata.get("alignment_type", "exact")],
            ),
            "alignment_type_counts": sample.metadata.get(
                "alignment_type_counts",
                {
                    sample.metadata.get("alignment_type", "exact"): int(
                        sample.metadata.get("news_count", 1)
                    )
                },
            ),
            "checkpoint_path": str(checkpoint_path),
            "mc_samples": int(mc_samples),
            "split": str(split),
            "selection_mode": str(selection_mode),
            "aggregation_mode": str(aggregation_mode),
            "residual_blend_alpha": float(residual_blend_alpha),
            "text_alignment_mode": sample.metadata.get("text_alignment_mode", "matched"),
            "text_source_sample_id": sample.metadata.get("text_source_sample_id", sample.sample_id),
            "text_source_surface_pair_id": sample.metadata.get("text_source_surface_pair_id", sample.surface_pair_id),
            "source_sample_ids": sample.metadata.get("source_sample_ids", [sample.sample_id]),
            "article_ids": sample.metadata.get("article_ids", []),
            "news_count": int(sample.metadata.get("news_count", 1)),
            "unique_embedding_count": int(sample.metadata.get("unique_embedding_count", 1)),
            "pooling_mode": sample.metadata.get("pooling_mode", ""),
            "has_text": float(sample.metadata.get("has_text", 1.0)),
            "surface_support_mode": sample.metadata.get("surface_support_mode", "full_grid"),
            "report_atm7_metric": bool(sample.metadata.get("report_atm7_metric", True)),
        },
    }


def _resolve_atm_vol_dir(*, checkpoint_path: str | Path, output_dir: str | Path) -> Path:
    run_dir = infer_run_dir_from_checkpoint(checkpoint_path)
    if run_dir is None:
        output_path = Path(output_dir)
        for candidate in (output_path, *output_path.parents):
            if candidate.name == "generate_result":
                run_dir = candidate.parent
                break
    if run_dir is None:
        raise ValueError(
            f"Could not infer the training run directory for ATM outputs from "
            f"checkpoint_path={checkpoint_path} output_dir={output_dir}"
        )
    target = generate_result_dir(run_dir) / "atm_vol"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _build_atm_vol_row(
    *,
    sample: FilmWGANSample,
    payload: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    report_atm7 = bool(sample.metadata.get("report_atm7_metric", True))
    current_atm = extract_atm_short_value(
        sample.current_surface,
        strike_grid=sample.strike_grid,
        maturity_days_grid=sample.maturity_days_grid,
    )
    generated_atm = extract_atm_short_value(
        payload["generated_surface"],
        strike_grid=sample.strike_grid,
        maturity_days_grid=sample.maturity_days_grid,
    )
    target_atm = extract_atm_short_value(
        sample.target_surface,
        strike_grid=sample.strike_grid,
        maturity_days_grid=sample.maturity_days_grid,
    )
    support_mask = sample.evaluation_support_mask
    supported_locations = np.argwhere(support_mask)
    supported_payload = {
        "supported_atm_strike": float("nan"),
        "supported_short_maturity_days": float("nan"),
        "supported_current_atm_vol": float("nan"),
        "supported_generated_atm_vol": float("nan"),
        "supported_target_atm_vol": float("nan"),
        "supported_generated_target_abs_error": float("nan"),
        "supported_current_target_abs_error": float("nan"),
    }
    if supported_locations.size:
        minimum_maturity_index = int(np.min(supported_locations[:, 0]))
        strike_candidates = np.flatnonzero(support_mask[minimum_maturity_index])
        strike_index = int(
            strike_candidates[
                np.argmin(np.abs(sample.strike_grid[strike_candidates] - 1.0))
            ]
        )
        current_value = float(sample.current_surface[minimum_maturity_index, strike_index])
        generated_value = float(
            np.asarray(payload["generated_surface"])[minimum_maturity_index, strike_index]
        )
        target_value = float(sample.target_surface[minimum_maturity_index, strike_index])
        supported_payload = {
            "supported_atm_strike": float(sample.strike_grid[strike_index]),
            "supported_short_maturity_days": float(
                sample.maturity_days_grid[minimum_maturity_index]
            ),
            "supported_current_atm_vol": current_value,
            "supported_generated_atm_vol": generated_value,
            "supported_target_atm_vol": target_value,
            "supported_generated_target_abs_error": abs(generated_value - target_value),
            "supported_current_target_abs_error": abs(current_value - target_value),
        }
    return {
        "sample_id": sample.sample_id,
        "global_index": int(sample.global_index),
        "surface_pair_id": sample.surface_pair_id,
        "news_timestamp_utc": sample.timestamp,
        "current_snapshot_time_utc": sample.current_snapshot_time_utc,
        "target_snapshot_time_utc": sample.target_snapshot_time_utc,
        "atm_strike": float(current_atm["atm_strike"]),
        "short_maturity_days": float(current_atm["short_maturity_days"]),
        "current_atm_vol": float(current_atm["value"]),
        "generated_atm_vol": float(generated_atm["value"]),
        "target_atm_vol": float(target_atm["value"]),
        "generated_target_abs_error": (
            float(abs(float(generated_atm["value"]) - float(target_atm["value"])))
            if report_atm7
            else float("nan")
        ),
        "current_target_abs_error": (
            float(abs(float(current_atm["value"]) - float(target_atm["value"])))
            if report_atm7
            else float("nan")
        ),
        "atm7_metric_status": (
            "reported_legacy_grid"
            if report_atm7
            else "not_reported_raw_maturity_extrapolation"
        ),
        **supported_payload,
        "checkpoint_path": str(checkpoint_path),
    }


class FilmWGANSampler:
    """Load a standalone FiLM WGAN checkpoint and generate arbitrage-weighted scenarios."""

    def __init__(self, config: FilmWGANSampleConfig):
        self.config = config
        self.run_dir = Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.run_dir / "samples"
        self.plots_dir = self.run_dir / "plots"
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

    def _load_generator(self) -> tuple[FilmWGANGenerator, dict[str, Any], FilmWGANTrainConfig, TensorNormalizationStats]:
        if not str(self.config.checkpoint_path).strip():
            raise ValueError("checkpoint_path must be set for standalone FiLM WGAN sampling.")
        checkpoint = load_checkpoint(self.config.checkpoint_path, self.device)
        train_config = FilmWGANTrainConfig(**checkpoint["config"])
        surface_shape = tuple(int(v) for v in checkpoint["surface_shape"])
        forecast_mode = str(checkpoint.get("forecast_mode", train_config.forecast_mode)).strip().lower()
        conditioning_mode = str(checkpoint.get("conditioning_mode", train_config.conditioning_mode)).strip().lower()
        generator = FilmWGANGenerator(
            surface_height=surface_shape[0],
            surface_width=surface_shape[1],
            embedding_dim=int(checkpoint["embedding_dim"]),
            noise_dim=0 if forecast_mode == "deterministic" else int(train_config.noise_dim),
            base_channels=int(train_config.gen_base_channels),
            res_blocks=int(train_config.gen_res_blocks),
            text_hidden_dim=int(train_config.text_hidden_dim),
            text_out_dim=int(train_config.text_out_dim),
            fusion_hidden_dim=int(train_config.fusion_hidden_dim),
            conditioning_mode=conditioning_mode,
            text_dropout=float(train_config.text_dropout),
            text_gate_initial_value=float(train_config.text_gate_initial_value),
            current_surface_channels=int(checkpoint.get("current_surface_channels", 1)),
        ).to(self.device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        normalization = normalization_stats_to_tensors(checkpoint["normalization_stats"], self.device)
        return generator, checkpoint, train_config, normalization

    def sample(self):
        generator, _checkpoint, train_config, normalization = self._load_generator()
        strike_grid = torch.tensor(_checkpoint["strike_grid"], dtype=torch.float32)
        maturity_days_grid = torch.tensor(_checkpoint["maturity_days_grid"], dtype=torch.float32)
        recon_weights_surface = build_reconstruction_weight_template(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            mode=str(train_config.recon_weight_mode),
            atm_range=float(train_config.recon_atm_range),
            short_end_max_days=float(train_config.recon_atm_short_end_max_days),
            atm_multiplier=float(train_config.recon_atm_multiplier),
        )
        atm_short_mask_surface = build_atm_short_mask(
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            atm_range=float(train_config.atm_short_range),
            max_days=float(train_config.atm_short_max_days),
        )
        atm_vol_dir = _resolve_atm_vol_dir(
            checkpoint_path=self.config.checkpoint_path,
            output_dir=self.config.output_dir,
        )

        all_samples = load_film_wgan_samples(self.config)
        selected_samples = split_samples(self.config, all_samples)
        if not selected_samples:
            raise ValueError("No standalone FiLM WGAN samples are available after split/selection.")

        summary_rows: list[dict[str, Any]] = []
        sample_atm_rows: list[dict[str, Any]] = []
        payload_cache: dict[tuple[int, str], dict[str, Any]] = {}

        def _cache_key(sample: FilmWGANSample) -> tuple[int, str]:
            return int(sample.global_index), str(sample.sample_id)

        def _payload_for_sample(sample: FilmWGANSample) -> dict[str, Any]:
            key = _cache_key(sample)
            if key not in payload_cache:
                payload_cache[key] = build_sample_payload(
                    generator=generator,
                    sample=sample,
                    normalization=normalization,
                    noise_dim=int(generator.noise_dim),
                    mc_samples=int(self.config.mc_samples),
                    seed=int(self.config.seed),
                    device=self.device,
                    reweight_beta_mode=self.config.reweight_beta_mode,
                    reweight_beta=float(self.config.reweight_beta),
                    aggregation_mode=self.config.aggregation_mode,
                    quantiles=self.config.quantiles,
                    calibration_levels=self.config.calibration_levels,
                    arbitrage_violation_tolerance=float(self.config.arbitrage_violation_tolerance),
                    checkpoint_path=str(self.config.checkpoint_path),
                    split=str(self.config.split),
                    selection_mode=str(self.config.selection_mode),
                    recon_weights_surface=recon_weights_surface,
                    atm_short_mask_surface=atm_short_mask_surface,
                    normalize_current_surface=bool(train_config.normalize_current_surface),
                    normalize_text_embedding=bool(train_config.normalize_text_embedding),
                    normalize_target_delta=bool(train_config.normalize_target_delta),
                    residual_blend_alpha=float(self.config.residual_blend_alpha),
                )
                payload_cache[key]["metadata"].update(
                    {
                        "forecast_mode": str(train_config.forecast_mode),
                        "conditioning_mode": str(train_config.conditioning_mode),
                    }
                )
            return payload_cache[key]

        for sample in selected_samples:
            payload = _payload_for_sample(sample)
            json_path = self.samples_dir / f"{sample.global_index:04d}_{sample.sample_id}.json"
            if bool(self.config.save_json):
                write_json(json_path, payload)
            if bool(self.config.save_plots):
                plot_film_wgan_payload(payload, self.plots_dir / f"{sample.global_index:04d}_{sample.sample_id}.png")
            atm_row = _build_atm_vol_row(
                sample=sample,
                payload=payload,
                checkpoint_path=self.config.checkpoint_path,
            )
            sample_atm_rows.append(atm_row)
            summary_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "global_index": int(sample.global_index),
                    "surface_pair_id": sample.surface_pair_id,
                    "split": str(self.config.split),
                    "news_timestamp_utc": sample.timestamp,
                    "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                    "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                    "publication_timestamp_utc": sample.metadata.get(
                        "publication_timestamp_utc",
                        "",
                    ),
                    "publication_availability_lag_minutes": int(
                        sample.metadata.get("publication_availability_lag_minutes", 0)
                    ),
                    "news_alignment_mode": sample.metadata.get(
                        "news_alignment_mode",
                        "exact",
                    ),
                    "alignment_type": sample.metadata.get(
                        "alignment_type",
                        "exact",
                    ),
                    "origin_shift_minutes": _metadata_float(
                        sample.metadata.get("origin_shift_minutes", 0.0)
                    ),
                    "origin_shift_minutes_min": _metadata_float(
                        sample.metadata.get("origin_shift_minutes_min", 0.0)
                    ),
                    "origin_shift_minutes_max": _metadata_float(
                        sample.metadata.get("origin_shift_minutes_max", 0.0)
                    ),
                    "origin_shift_minutes_mean": _metadata_float(
                        sample.metadata.get("origin_shift_minutes_mean", 0.0)
                    ),
                    "source_alignment_types": json.dumps(
                        sample.metadata.get(
                            "source_alignment_types",
                            [sample.metadata.get("alignment_type", "exact")],
                        ),
                        ensure_ascii=True,
                    ),
                    "alignment_type_counts": json.dumps(
                        sample.metadata.get(
                            "alignment_type_counts",
                            {
                                sample.metadata.get(
                                    "alignment_type",
                                    "exact",
                                ): int(sample.metadata.get("news_count", 1))
                            },
                        ),
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    "event_group": sample.metadata.get("event_group", ""),
                    "has_news": sample.metadata.get("has_news", ""),
                    "news_cluster_id": sample.metadata.get("news_cluster_id", ""),
                    "quiet_buffer_minutes": sample.metadata.get("quiet_buffer_minutes", ""),
                    "quiet_grid_minutes": sample.metadata.get("quiet_grid_minutes", ""),
                    "source_sample_ids": json.dumps(
                        sample.metadata.get("source_sample_ids", [sample.sample_id]),
                        ensure_ascii=True,
                    ),
                    "article_ids": json.dumps(
                        sample.metadata.get("article_ids", []),
                        ensure_ascii=True,
                    ),
                    "news_count": int(sample.metadata.get("news_count", 1)),
                    "unique_embedding_count": int(sample.metadata.get("unique_embedding_count", 1)),
                    "pooling_mode": sample.metadata.get("pooling_mode", ""),
                    "has_text_condition": float(sample.metadata.get("has_text", 1.0)),
                    "effective_beta": float(payload["effective_beta"]),
                    "weight_entropy": float(payload["weight_entropy"]),
                    "penalty_mean": float(payload["penalty_mean"]),
                    "penalty_std": float(payload["penalty_std"]),
                    "residual_blend_alpha": float(payload["metadata"]["residual_blend_alpha"]),
                    "mae": float(payload["metrics"]["mae"]),
                    "surface_mae": float(payload["metrics"]["mae"]),
                    "rmse": float(payload["metrics"]["rmse"]),
                    "max_abs": float(payload["metrics"]["max_abs"]),
                    "current_mae": float(payload["current_metrics"]["mae"]),
                    "current_rmse": float(payload["current_metrics"]["rmse"]),
                    "current_max_abs": float(payload["current_metrics"]["max_abs"]),
                    "mae_gap_vs_current": float(payload["comparison_metrics"]["mae_gap_vs_current"]),
                    "rmse_gap_vs_current": float(payload["comparison_metrics"]["rmse_gap_vs_current"]),
                    "max_abs_gap_vs_current": float(payload["comparison_metrics"]["max_abs_gap_vs_current"]),
                    "win_flag_vs_current": float(payload["comparison_metrics"]["win_flag_vs_current"]),
                    "short_atm_weighted_mae": float(payload["short_atm_metrics"]["short_atm_weighted_mae"]),
                    "current_short_atm_weighted_mae": float(
                        payload["short_atm_metrics"]["current_short_atm_weighted_mae"]
                    ),
                    "short_atm_mae_gap_vs_current": float(payload["short_atm_metrics"]["short_atm_mae_gap_vs_current"]),
                    "short_atm_weighted_win_flag_vs_current": float(
                        payload["short_atm_metrics"]["short_atm_weighted_win_flag_vs_current"]
                    ),
                    "atm_short_pure_mae": float(payload["short_atm_metrics"]["atm_short_pure_mae"]),
                    "short_atm_mae": float(payload["short_atm_metrics"]["atm_short_pure_mae"]),
                    "current_atm_short_pure_mae": float(payload["short_atm_metrics"]["current_atm_short_pure_mae"]),
                    "atm_short_pure_mae_gap_vs_current": float(
                        payload["short_atm_metrics"]["atm_short_pure_mae_gap_vs_current"]
                    ),
                    "atm_short_pure_win_flag_vs_current": float(
                        payload["short_atm_metrics"]["atm_short_pure_win_flag_vs_current"]
                    ),
                    "generated_current_mae": float(payload["generated_current_metrics"]["mae"]),
                    "atm7_abs_err": float(atm_row["generated_target_abs_error"]),
                    "current_atm7_abs_err": float(atm_row["current_target_abs_error"]),
                    "atm7_metric_status": atm_row["atm7_metric_status"],
                    "supported_shortest_atm_abs_err": float(
                        atm_row["supported_generated_target_abs_error"]
                    ),
                    "current_supported_shortest_atm_abs_err": float(
                        atm_row["supported_current_target_abs_error"]
                    ),
                    "supported_shortest_maturity_days": float(
                        atm_row["supported_short_maturity_days"]
                    ),
                    "energy_score": float(payload["probabilistic_metrics"]["energy_score"]),
                    "variogram_score": float(
                        payload["probabilistic_metrics"]["variogram_score"]
                    ),
                    "scenario_spread": float(payload["probabilistic_metrics"]["scenario_spread"]),
                    "mc_surface_mae_se": float(
                        payload["probabilistic_metrics"]["mc_surface_mae_se"]
                    ),
                    "effective_scenario_count": float(
                        payload["probabilistic_metrics"]["effective_scenario_count"]
                    ),
                    "coverage_50": float(payload["probabilistic_metrics"].get("coverage_50", 0.0)),
                    "coverage_80": float(payload["probabilistic_metrics"].get("coverage_80", 0.0)),
                    "coverage_90": float(payload["probabilistic_metrics"].get("coverage_90", 0.0)),
                    "interval_width_50": float(
                        payload["probabilistic_metrics"].get("interval_width_50", 0.0)
                    ),
                    "interval_width_80": float(
                        payload["probabilistic_metrics"].get("interval_width_80", 0.0)
                    ),
                    "interval_width_90": float(
                        payload["probabilistic_metrics"].get("interval_width_90", 0.0)
                    ),
                    "calibration_error": float(payload["probabilistic_metrics"]["calibration_error"]),
                    "calendar_violation_rate": float(payload["arbitrage_metrics"]["calendar_violation_rate"]),
                    "butterfly_violation_rate": float(payload["arbitrage_metrics"]["butterfly_violation_rate"]),
                    "scenario_calendar_violation_rate": float(
                        payload["arbitrage_metrics"]["scenario_calendar_violation_rate"]
                    ),
                    "scenario_butterfly_violation_rate": float(
                        payload["arbitrage_metrics"]["scenario_butterfly_violation_rate"]
                    ),
                    "arbitrage_metric_status": payload["arbitrage_metrics"]["status"],
                    "supported_cell_count": int(
                        payload["support_metrics"]["supported_cell_count"]
                    ),
                    "supported_cell_fraction": float(
                        payload["support_metrics"]["supported_cell_fraction"]
                    ),
                    "text_alignment_mode": payload["metadata"]["text_alignment_mode"],
                    "text_source_sample_id": payload["metadata"]["text_source_sample_id"],
                    "forecast_mode": payload["metadata"]["forecast_mode"],
                    "conditioning_mode": payload["metadata"]["conditioning_mode"],
                }
            )
        write_csv(self.run_dir / "summary.csv", summary_rows)

        def _ordered_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                rows,
                key=lambda row: (str(row.get("news_timestamp_utc", "")), int(row.get("global_index", -1))),
            )

        ordered_sample_rows = _ordered_rows(sample_atm_rows)
        write_csv(atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.csv", ordered_sample_rows)
        ordered_full_rows: list[dict[str, Any]] = []
        if bool(self.config.save_full_atm_timeseries):
            full_atm_rows = [
                _build_atm_vol_row(
                    sample=sample,
                    payload=_payload_for_sample(sample),
                    checkpoint_path=self.config.checkpoint_path,
                )
                for sample in all_samples
            ]
            ordered_full_rows = _ordered_rows(full_atm_rows)
            write_csv(atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.csv", ordered_full_rows)
        if bool(self.config.save_plots):
            plot_atm_vol_timeseries(
                ordered_sample_rows,
                atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.png",
                series_scope="sample",
            )
            if ordered_full_rows:
                plot_atm_vol_timeseries(
                    ordered_full_rows,
                    atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.png",
                    series_scope="full",
                )
        write_json(
            self.run_dir / "run_metadata.json",
            {
                "checkpoint_path": str(self.config.checkpoint_path),
                "selected_samples": len(summary_rows),
                "full_samples": len(ordered_full_rows),
                "available_samples": len(all_samples),
                "config": asdict(self.config),
            },
        )
        return self.run_dir
