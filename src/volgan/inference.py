"""Scenario sampling and arbitrage-weighted aggregation for standalone VolGAN."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arbitrage import reweight_scenarios, total_arbitrage_penalty
from .config import VolGANSampleConfig, VolGANTrainConfig, save_config_yaml
from .data import VolSurfaceSample, load_vol_surface_samples, split_samples
from .io import load_checkpoint, prepare_run_dir, write_csv, write_json
from .models import VolGANGenerator, reconstruct_future_surface
from .plotting import plot_volgan_payload


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


def _surface_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
    }


class VolGANSampler:
    """Load a standalone VolGAN checkpoint and generate arbitrage-weighted scenarios."""

    def __init__(self, config: VolGANSampleConfig):
        self.config = config
        self.run_dir = prepare_run_dir(config.output_dir)
        self.samples_dir = self.run_dir / "samples"
        self.plots_dir = self.run_dir / "plots"
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

    def _load_generator(self) -> tuple[VolGANGenerator, dict[str, Any], VolGANTrainConfig]:
        if not str(self.config.checkpoint_path).strip():
            raise ValueError("checkpoint_path must be set for standalone VolGAN sampling.")
        checkpoint = load_checkpoint(self.config.checkpoint_path, self.device)
        train_config = VolGANTrainConfig(**checkpoint["config"])
        surface_shape = tuple(int(v) for v in checkpoint["surface_shape"])
        surface_dim = int(surface_shape[0] * surface_shape[1])
        generator = VolGANGenerator(
            surface_dim=surface_dim,
            embedding_dim=int(checkpoint["embedding_dim"]),
            noise_dim=int(train_config.noise_dim),
            hidden_dim=int(train_config.hidden_dim),
        ).to(self.device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        return generator, checkpoint, train_config

    def _sample_one(
        self,
        *,
        generator: VolGANGenerator,
        checkpoint: dict[str, Any],
        train_config: VolGANTrainConfig,
        sample: VolSurfaceSample,
    ) -> dict[str, Any]:
        strike_grid = np.asarray(checkpoint["strike_grid"], dtype=np.float32)
        maturity_days_grid = np.asarray(checkpoint["maturity_days_grid"], dtype=np.float32)
        height, width = tuple(int(v) for v in checkpoint["surface_shape"])

        current_flat = torch.tensor(sample.current_surface.reshape(1, -1), dtype=torch.float32, device=self.device)
        text_embedding = torch.tensor(sample.text_embedding.reshape(1, -1), dtype=torch.float32, device=self.device)

        generated_surfaces: list[np.ndarray] = []
        penalties: list[float] = []
        with torch.no_grad():
            for draw_idx in range(max(1, int(self.config.mc_samples))):
                generator_noise = torch.Generator(device="cpu")
                generator_noise.manual_seed(int(self.config.seed) + int(sample.global_index) + draw_idx * 1000003)
                noise = torch.randn((1, int(train_config.noise_dim)), generator=generator_noise, dtype=torch.float32).to(
                    self.device
                )
                fake_delta = generator(current_flat, text_embedding, noise=noise)
                future_flat = reconstruct_future_surface(current_flat, fake_delta)
                future_surface = future_flat.view(height, width).detach().cpu().numpy().astype(np.float32)
                generated_surfaces.append(future_surface)

        generated_tensor = torch.tensor(np.stack(generated_surfaces, axis=0), dtype=torch.float32, device=self.device)
        penalty_tensor = total_arbitrage_penalty(
            generated_tensor,
            torch.tensor(strike_grid, dtype=torch.float32, device=self.device),
            torch.tensor(maturity_days_grid, dtype=torch.float32, device=self.device),
        )
        penalties = penalty_tensor.detach().cpu().numpy().astype(np.float64)
        weights, effective_beta = reweight_scenarios(
            penalties,
            beta_mode=self.config.reweight_beta_mode,
            beta_value=float(self.config.reweight_beta),
        )

        surface_stack = np.stack(generated_surfaces, axis=0).astype(np.float64)
        weighted_mean_surface = _weighted_mean(surface_stack, weights).astype(np.float32)
        flat_surface_stack = surface_stack.reshape(surface_stack.shape[0], -1)
        quantile_surfaces = {
            f"q_{float(quantile):.2f}": _weighted_quantile(flat_surface_stack, weights, float(quantile))
            .reshape(height, width)
            .astype(np.float32)
            .tolist()
            for quantile in self.config.quantiles
        }
        target_surface = sample.target_surface.astype(np.float32)
        metrics = _surface_metrics(weighted_mean_surface, target_surface)
        current_metrics = _surface_metrics(sample.current_surface.astype(np.float32), target_surface)
        return {
            "sample_id": sample.sample_id,
            "mode": "volgan",
            "global_index": int(sample.global_index),
            "news_timestamp_utc": sample.timestamp,
            "current_snapshot_time_utc": sample.current_snapshot_time_utc,
            "target_snapshot_time_utc": sample.target_snapshot_time_utc,
            "strike_grid": strike_grid.astype(float).tolist(),
            "maturity_days_grid": maturity_days_grid.astype(float).tolist(),
            "current_surface": sample.current_surface.astype(float).tolist(),
            "generated_surface": weighted_mean_surface.astype(float).tolist(),
            "real_surface": target_surface.astype(float).tolist(),
            "target_surface": target_surface.astype(float).tolist(),
            "quantile_surfaces": quantile_surfaces,
            "penalties": penalties.astype(float).tolist(),
            "weights": weights.astype(float).tolist(),
            "effective_beta": float(effective_beta),
            "metrics": metrics,
            "current_metrics": current_metrics,
            "metadata": {
                "pair_quality_label": sample.metadata.get("pair_quality_label", ""),
                "checkpoint_path": str(self.config.checkpoint_path),
                "mc_samples": int(self.config.mc_samples),
                "split": str(self.config.split),
                "selection_mode": str(self.config.selection_mode),
            },
        }

    def sample(self) -> Path:
        generator, checkpoint, train_config = self._load_generator()
        save_config_yaml(self.config, self.run_dir / "resolved_config.yaml")

        all_samples = load_vol_surface_samples(self.config)
        selected_samples = split_samples(self.config, all_samples)
        if not selected_samples:
            raise ValueError("No standalone VolGAN samples are available after split/selection.")

        summary_rows: list[dict[str, Any]] = []
        for sample in selected_samples:
            payload = self._sample_one(
                generator=generator,
                checkpoint=checkpoint,
                train_config=train_config,
                sample=sample,
            )
            json_path = self.samples_dir / f"{sample.global_index:04d}_{sample.sample_id}.json"
            if bool(self.config.save_json):
                write_json(json_path, payload)
            if bool(self.config.save_plots):
                plot_volgan_payload(payload, self.plots_dir / f"{sample.global_index:04d}_{sample.sample_id}.png")
            summary_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "global_index": int(sample.global_index),
                    "news_timestamp_utc": sample.timestamp,
                    "effective_beta": float(payload["effective_beta"]),
                    "mae": float(payload["metrics"]["mae"]),
                    "rmse": float(payload["metrics"]["rmse"]),
                    "max_abs": float(payload["metrics"]["max_abs"]),
                    "current_mae": float(payload["current_metrics"]["mae"]),
                    "current_rmse": float(payload["current_metrics"]["rmse"]),
                    "current_max_abs": float(payload["current_metrics"]["max_abs"]),
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
