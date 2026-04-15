"""Scenario sampling and arbitrage-weighted aggregation for standalone Cross-Attention WGAN."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .arbitrage import reweight_scenarios, total_arbitrage_penalty
from .config import CrossAttnWGANSampleConfig, CrossAttnWGANTrainConfig
from .data import (
    CrossAttnWGANNormalizationStats,
    CrossAttnWGANSample,
    denormalize_tensor,
    load_crossattn_wgan_samples,
    normalize_surface_tensor,
    normalize_tensor,
    split_samples,
)
from .io import load_checkpoint, write_csv, write_json
from .models import CrossAttnWGANGenerator, reconstruct_future_surface
from .plotting import plot_crossattn_wgan_payload


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
    stats: CrossAttnWGANNormalizationStats | Mapping[str, Any],
    device: torch.device,
) -> TensorNormalizationStats:
    if isinstance(stats, CrossAttnWGANNormalizationStats):
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
    sample: CrossAttnWGANSample,
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
    generator: CrossAttnWGANGenerator,
    sample: CrossAttnWGANSample,
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
    quantile_surfaces = {
        f"q_{float(quantile):.2f}": _weighted_quantile(flat_surface_stack, weights, float(quantile))
        .reshape(target_surface.shape)
        .astype(np.float32)
        .tolist()
        for quantile in quantiles
    }
    current_surface = np.asarray(current_surface, dtype=np.float32)
    target_surface = np.asarray(target_surface, dtype=np.float32)
    current_metrics = _surface_metrics(current_surface, target_surface)
    metrics = _surface_metrics(weighted_mean_surface, target_surface)
    generated_current_metrics = _surface_metrics(weighted_mean_surface, current_surface)
    return {
        "generated_surface": weighted_mean_surface.astype(float).tolist(),
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
    }


def build_sample_payload(
    *,
    generator: CrossAttnWGANGenerator,
    sample: CrossAttnWGANSample,
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
    normalize_current_surface: bool,
    normalize_text_embedding: bool,
    normalize_target_delta: bool,
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
    )
    return {
        "sample_id": sample.sample_id,
        "mode": "crossattn_wgan",
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
        "metadata": {
            "pair_quality_label": sample.metadata.get("pair_quality_label", ""),
            "checkpoint_path": str(checkpoint_path),
            "mc_samples": int(mc_samples),
            "split": str(split),
            "selection_mode": str(selection_mode),
            "aggregation_mode": str(aggregation_mode),
        },
    }


class CrossAttnWGANSampler:
    """Load a standalone Cross-Attention WGAN checkpoint and generate arbitrage-weighted scenarios."""

    def __init__(self, config: CrossAttnWGANSampleConfig):
        self.config = config
        self.run_dir = Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.run_dir / "samples"
        self.plots_dir = self.run_dir / "plots"
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

    def _load_generator(self) -> tuple[CrossAttnWGANGenerator, dict[str, Any], CrossAttnWGANTrainConfig, TensorNormalizationStats]:
        if not str(self.config.checkpoint_path).strip():
            raise ValueError("checkpoint_path must be set for standalone Cross-Attention WGAN sampling.")
        checkpoint = load_checkpoint(self.config.checkpoint_path, self.device)
        train_config = CrossAttnWGANTrainConfig(**checkpoint["config"])
        surface_shape = tuple(int(v) for v in checkpoint["surface_shape"])
        generator = CrossAttnWGANGenerator(
            surface_height=surface_shape[0],
            surface_width=surface_shape[1],
            embedding_dim=int(checkpoint["embedding_dim"]),
            noise_dim=int(train_config.noise_dim),
            base_channels=int(train_config.gen_base_channels),
            res_blocks=int(train_config.gen_res_blocks),
            text_hidden_dim=int(train_config.text_hidden_dim),
            text_out_dim=int(train_config.text_out_dim),
            fusion_hidden_dim=int(train_config.fusion_hidden_dim),
            num_attn_heads=int(train_config.num_attn_heads),
            num_text_tokens=int(train_config.num_text_tokens),
            attn_dim=int(train_config.attn_dim),
        ).to(self.device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        normalization = normalization_stats_to_tensors(checkpoint["normalization_stats"], self.device)
        return generator, checkpoint, train_config, normalization

    def sample(self):
        generator, _checkpoint, train_config, normalization = self._load_generator()

        all_samples = load_crossattn_wgan_samples(self.config)
        selected_samples = split_samples(self.config, all_samples)
        if not selected_samples:
            raise ValueError("No standalone Cross-Attention WGAN samples are available after split/selection.")

        summary_rows: list[dict[str, Any]] = []
        for sample in selected_samples:
            payload = build_sample_payload(
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
                normalize_current_surface=bool(train_config.normalize_current_surface),
                normalize_text_embedding=bool(train_config.normalize_text_embedding),
                normalize_target_delta=bool(train_config.normalize_target_delta),
            )
            json_path = self.samples_dir / f"{sample.global_index:04d}_{sample.sample_id}.json"
            if bool(self.config.save_json):
                write_json(json_path, payload)
            if bool(self.config.save_plots):
                plot_crossattn_wgan_payload(payload, self.plots_dir / f"{sample.global_index:04d}_{sample.sample_id}.png")
            summary_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "global_index": int(sample.global_index),
                    "news_timestamp_utc": sample.timestamp,
                    "effective_beta": float(payload["effective_beta"]),
                    "weight_entropy": float(payload["weight_entropy"]),
                    "penalty_mean": float(payload["penalty_mean"]),
                    "penalty_std": float(payload["penalty_std"]),
                    "mae": float(payload["metrics"]["mae"]),
                    "rmse": float(payload["metrics"]["rmse"]),
                    "max_abs": float(payload["metrics"]["max_abs"]),
                    "current_mae": float(payload["current_metrics"]["mae"]),
                    "current_rmse": float(payload["current_metrics"]["rmse"]),
                    "current_max_abs": float(payload["current_metrics"]["max_abs"]),
                    "generated_current_mae": float(payload["generated_current_metrics"]["mae"]),
                }
            )
        write_csv(self.run_dir / "summary.csv", summary_rows)
        write_json(
            self.run_dir / "run_metadata.json",
            {
                "checkpoint_path": str(self.config.checkpoint_path),
                "selected_samples": len(summary_rows),
                "config": asdict(self.config),
            },
        )
        return self.run_dir
