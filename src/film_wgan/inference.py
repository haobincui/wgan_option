"""Scenario sampling and arbitrage-weighted aggregation for standalone FiLM WGAN."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utils.training_paths import generate_result_dir, infer_run_dir_from_checkpoint
from .arbitrage import reweight_scenarios, total_arbitrage_penalty
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
    if normalize_text_embedding:
        text_features = normalize_tensor(text_embedding, normalization.text_mean, normalization.text_std)
    else:
        text_features = text_embedding
    current_features = current_features_flat.view(1, 1, height, width)
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
    generated_surfaces: list[np.ndarray] = []
    with torch.no_grad():
        for draw_idx in range(max(1, int(mc_samples))):
            generator_noise = torch.Generator(device="cpu")
            generator_noise.manual_seed(int(seed) + int(sample.global_index) + draw_idx * 1000003)
            noise = torch.randn((1, int(noise_dim)), generator=generator_noise, dtype=torch.float32).to(device)
            fake_delta_norm = generator(current_features, text_features, noise=noise)
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


def _surface_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
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
) -> dict[str, float]:
    generated_tensor = torch.tensor(np.asarray(generated_surface, dtype=np.float32), dtype=torch.float32)
    current_tensor = torch.tensor(np.asarray(current_surface, dtype=np.float32), dtype=torch.float32)
    target_tensor = torch.tensor(np.asarray(target_surface, dtype=np.float32), dtype=torch.float32)

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
        generated_weighted = float(
            weighted_surface_mae(generated_tensor, target_tensor, recon_weights_surface).detach().cpu()
        )
        current_weighted = float(
            weighted_surface_mae(current_tensor, target_tensor, recon_weights_surface).detach().cpu()
        )
        metrics.update(
            {
                "short_atm_weighted_mae": generated_weighted,
                "current_short_atm_weighted_mae": current_weighted,
                "short_atm_mae_gap_vs_current": generated_weighted - current_weighted,
                "short_atm_weighted_win_flag_vs_current": 1.0 if generated_weighted < current_weighted else 0.0,
            }
        )
    if atm_short_mask_surface is not None and float(atm_short_mask_surface.sum().item()) > 0.0:
        generated_pure = float(
            atm_short_pure_mae(generated_tensor, target_tensor, atm_short_mask_surface).detach().cpu()
        )
        current_pure = float(
            atm_short_pure_mae(current_tensor, target_tensor, atm_short_mask_surface).detach().cpu()
        )
        metrics.update(
            {
                "atm_short_pure_mae": generated_pure,
                "current_atm_short_pure_mae": current_pure,
                "atm_short_pure_mae_gap_vs_current": generated_pure - current_pure,
                "atm_short_pure_win_flag_vs_current": 1.0 if generated_pure < current_pure else 0.0,
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
    recon_weights_surface: torch.Tensor | None = None,
    atm_short_mask_surface: torch.Tensor | None = None,
    residual_blend_alpha: float = 1.0,
) -> dict[str, Any]:
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
    normalized_aggregation = str(aggregation_mode).strip().lower()
    if normalized_aggregation != "weighted_mean":
        raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
    weighted_mean_surface = _weighted_mean(surface_stack, weights).astype(np.float32)
    flat_surface_stack = surface_stack.reshape(surface_stack.shape[0], -1)
    current_surface = np.asarray(current_surface, dtype=np.float32)
    target_surface = np.asarray(target_surface, dtype=np.float32)
    blended_surface = _apply_residual_blend(
        weighted_mean_surface,
        current_surface,
        residual_blend_alpha=float(residual_blend_alpha),
    )
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
    current_metrics = _surface_metrics(current_surface, target_surface)
    metrics = _surface_metrics(blended_surface, target_surface)
    generated_current_metrics = _surface_metrics(blended_surface, current_surface)
    comparison_metrics = {
        "mae_gap_vs_current": metrics["mae"] - current_metrics["mae"],
        "rmse_gap_vs_current": metrics["rmse"] - current_metrics["rmse"],
        "max_abs_gap_vs_current": metrics["max_abs"] - current_metrics["max_abs"],
        "win_flag_vs_current": 1.0 if metrics["mae"] < current_metrics["mae"] else 0.0,
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
        "short_atm_metrics": _short_atm_metrics(
            generated_surface=blended_surface,
            current_surface=current_surface,
            target_surface=target_surface,
            recon_weights_surface=recon_weights_surface,
            atm_short_mask_surface=atm_short_mask_surface,
        ),
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
        recon_weights_surface=recon_weights_surface,
        atm_short_mask_surface=atm_short_mask_surface,
        residual_blend_alpha=float(residual_blend_alpha),
    )
    return {
        "sample_id": sample.sample_id,
        "mode": "film_wgan",
        "global_index": int(sample.global_index),
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
        "short_atm_metrics": summary["short_atm_metrics"],
        "metadata": {
            "pair_quality_label": sample.metadata.get("pair_quality_label", ""),
            "checkpoint_path": str(checkpoint_path),
            "mc_samples": int(mc_samples),
            "split": str(split),
            "selection_mode": str(selection_mode),
            "aggregation_mode": str(aggregation_mode),
            "residual_blend_alpha": float(residual_blend_alpha),
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
    return {
        "sample_id": sample.sample_id,
        "global_index": int(sample.global_index),
        "news_timestamp_utc": sample.timestamp,
        "current_snapshot_time_utc": sample.current_snapshot_time_utc,
        "target_snapshot_time_utc": sample.target_snapshot_time_utc,
        "atm_strike": float(current_atm["atm_strike"]),
        "short_maturity_days": float(current_atm["short_maturity_days"]),
        "current_atm_vol": float(current_atm["value"]),
        "generated_atm_vol": float(generated_atm["value"]),
        "target_atm_vol": float(target_atm["value"]),
        "generated_target_abs_error": float(abs(float(generated_atm["value"]) - float(target_atm["value"]))),
        "current_target_abs_error": float(abs(float(current_atm["value"]) - float(target_atm["value"]))),
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
        generator = FilmWGANGenerator(
            surface_height=surface_shape[0],
            surface_width=surface_shape[1],
            embedding_dim=int(checkpoint["embedding_dim"]),
            noise_dim=int(train_config.noise_dim),
            base_channels=int(train_config.gen_base_channels),
            res_blocks=int(train_config.gen_res_blocks),
            text_hidden_dim=int(train_config.text_hidden_dim),
            text_out_dim=int(train_config.text_out_dim),
            fusion_hidden_dim=int(train_config.fusion_hidden_dim),
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
                    noise_dim=int(train_config.noise_dim),
                    mc_samples=int(self.config.mc_samples),
                    seed=int(self.config.seed),
                    device=self.device,
                    reweight_beta_mode=self.config.reweight_beta_mode,
                    reweight_beta=float(self.config.reweight_beta),
                    aggregation_mode=self.config.aggregation_mode,
                    quantiles=self.config.quantiles,
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
            return payload_cache[key]

        for sample in selected_samples:
            payload = _payload_for_sample(sample)
            json_path = self.samples_dir / f"{sample.global_index:04d}_{sample.sample_id}.json"
            if bool(self.config.save_json):
                write_json(json_path, payload)
            if bool(self.config.save_plots):
                plot_film_wgan_payload(payload, self.plots_dir / f"{sample.global_index:04d}_{sample.sample_id}.png")
            sample_atm_rows.append(
                _build_atm_vol_row(
                    sample=sample,
                    payload=payload,
                    checkpoint_path=self.config.checkpoint_path,
                )
            )
            summary_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "global_index": int(sample.global_index),
                    "news_timestamp_utc": sample.timestamp,
                    "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                    "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                    "event_group": sample.metadata.get("event_group", ""),
                    "has_news": sample.metadata.get("has_news", ""),
                    "news_cluster_id": sample.metadata.get("news_cluster_id", ""),
                    "quiet_buffer_minutes": sample.metadata.get("quiet_buffer_minutes", ""),
                    "quiet_grid_minutes": sample.metadata.get("quiet_grid_minutes", ""),
                    "effective_beta": float(payload["effective_beta"]),
                    "weight_entropy": float(payload["weight_entropy"]),
                    "penalty_mean": float(payload["penalty_mean"]),
                    "penalty_std": float(payload["penalty_std"]),
                    "residual_blend_alpha": float(payload["metadata"]["residual_blend_alpha"]),
                    "mae": float(payload["metrics"]["mae"]),
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
                    "current_atm_short_pure_mae": float(payload["short_atm_metrics"]["current_atm_short_pure_mae"]),
                    "atm_short_pure_mae_gap_vs_current": float(
                        payload["short_atm_metrics"]["atm_short_pure_mae_gap_vs_current"]
                    ),
                    "atm_short_pure_win_flag_vs_current": float(
                        payload["short_atm_metrics"]["atm_short_pure_win_flag_vs_current"]
                    ),
                    "generated_current_mae": float(payload["generated_current_metrics"]["mae"]),
                }
            )
        write_csv(self.run_dir / "summary.csv", summary_rows)

        def _ordered_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                rows,
                key=lambda row: (str(row.get("news_timestamp_utc", "")), int(row.get("global_index", -1))),
            )

        full_atm_rows = [
            _build_atm_vol_row(
                sample=sample,
                payload=_payload_for_sample(sample),
                checkpoint_path=self.config.checkpoint_path,
            )
            for sample in all_samples
        ]
        ordered_sample_rows = _ordered_rows(sample_atm_rows)
        ordered_full_rows = _ordered_rows(full_atm_rows)
        write_csv(atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.csv", ordered_sample_rows)
        write_csv(atm_vol_dir / "film_wgan_best_atm_vol_timeseries_full.csv", ordered_full_rows)
        if bool(self.config.save_plots):
            plot_atm_vol_timeseries(
                ordered_sample_rows,
                atm_vol_dir / "film_wgan_best_atm_vol_timeseries_sample.png",
                series_scope="sample",
            )
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
                "config": asdict(self.config),
            },
        )
        return self.run_dir
