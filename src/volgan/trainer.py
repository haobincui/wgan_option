"""Standalone VolGAN training loop and checkpoint management."""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import RMSprop
from trainer import BaseTrainer
from utils.output_paths import find_best_checkpoint, prepare_run_dir
from utils.training_paths import checkpoint_named_dir, generate_result_dir, resolve_existing_run_dir

from .arbitrage import butterfly_arbitrage_penalty, calendar_arbitrage_penalty
from .config import (
    VolGANSampleConfig,
    VolGANTrainConfig,
    build_sample_config_from_train_config,
    default_train_output_root,
    load_sample_config,
)
from .data import VolSurfaceDataBundle, create_train_val_bundle, denormalize_tensor
from .inference import VolGANSampler, build_sample_payload, normalization_stats_to_tensors
from .io import config_payload, save_checkpoint, write_csv, write_json
from .losses import (
    discriminator_bce_loss,
    estimate_gradient_matching,
    generator_bce_loss,
    maturity_smoothness_penalty,
    strike_smoothness_penalty,
)
from .models import VolGANDiscriminator, VolGANGenerator, reconstruct_future_surface
from .training_plots import plot_training_curves


class VolGANTrainer(BaseTrainer):
    """Train the standalone VolGAN model on `merged_vol.xlsx` rows."""

    trainer_id = "volgan"
    logger_name = "volgan.trainer"

    def __init__(self, config: VolGANTrainConfig, *, config_path: str | None = None):
        super().__init__(config, config_path=config_path)
        self.checkpoints_dir = Path(".")
        self.metrics_dir = Path(".")
        self.samples_dir = Path(".")
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

        self.bundle: Optional[VolSurfaceDataBundle] = None
        self.generator: Optional[VolGANGenerator] = None
        self.discriminator: Optional[VolGANDiscriminator] = None
        self.generator_optimizer: Optional[RMSprop] = None
        self.discriminator_optimizer: Optional[RMSprop] = None
        self.alpha_m = float(config.alpha_m)
        self.alpha_tau = float(config.alpha_tau)
        self.normalization = None

    def _prepare_runtime_config(self, config: VolGANTrainConfig) -> tuple[VolGANTrainConfig, Path]:
        output_root = str(config.output_root).strip() or default_train_output_root(config.data_path)
        run_dir = prepare_run_dir(output_root, create=False)
        resolved_config = replace(
            config,
            output_root=str(output_root),
            models_path=str(run_dir / "checkpoints"),
            metrics_path=str(run_dir / "metrics"),
            samples_path=str(run_dir / "samples"),
        )
        self.checkpoints_dir = run_dir / "checkpoints"
        self.metrics_dir = run_dir / "metrics"
        self.samples_dir = run_dir / "samples"
        return resolved_config, run_dir

    @property
    def logger(self) -> logging.Logger:
        return self._get_or_create_logger()

    def _set_seed(self) -> None:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))

    def setup(self) -> None:
        self._set_seed()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Standalone VolGAN outputs: %s", self.run_dir)
        self.logger.info("Loading merged-vol workbook from %s (%s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_train_val_bundle(self.config)
        self.normalization = normalization_stats_to_tensors(self.bundle.normalization_stats, self.device)
        surface_height, surface_width = self.bundle.surface_shape
        surface_dim = int(surface_height * surface_width)
        self.generator = VolGANGenerator(
            surface_dim=surface_dim,
            embedding_dim=self.bundle.embedding_dim,
            noise_dim=self.config.noise_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.discriminator = VolGANDiscriminator(
            surface_dim=surface_dim,
            embedding_dim=self.bundle.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.generator_optimizer = RMSprop(
            self.generator.parameters(),
            lr=float(self.config.generator_learning_rate),
        )
        self.discriminator_optimizer = RMSprop(
            self.discriminator.parameters(),
            lr=float(self.config.discriminator_learning_rate),
        )

        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, surface_shape=%s, embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            self.bundle.surface_shape,
            self.bundle.embedding_dim,
        )
        self.logger.info(
            "Model initialized: G params=%s, D params=%s",
            sum(parameter.numel() for parameter in self.generator.parameters()),
            sum(parameter.numel() for parameter in self.discriminator.parameters()),
        )

        if self.config.use_gradient_matching:
            strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
            maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)
            self.alpha_m, self.alpha_tau = estimate_gradient_matching(
                generator=self.generator,
                discriminator=self.discriminator,
                train_loader=self.bundle.train_loader,
                device=self.device,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
                delta_mean=self.normalization.delta_mean,
                delta_std=self.normalization.delta_std,
                noise_dim=self.config.noise_dim,
                epochs=self.config.gradient_match_epochs,
                real_label_value=float(self.config.real_label_value),
                alpha_clip_min=float(self.config.alpha_clip_min),
                alpha_clip_max=float(self.config.alpha_clip_max),
                normalize_target_delta=bool(self.config.normalize_target_delta),
            )
            self.logger.info("Gradient matching selected alpha_m=%.6f alpha_tau=%.6f", self.alpha_m, self.alpha_tau)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _surface_tensor(self, flat_surface: torch.Tensor) -> torch.Tensor:
        assert self.bundle is not None
        height, width = self.bundle.surface_shape
        return flat_surface.view(flat_surface.size(0), height, width)

    def _discriminator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        real_delta_norm: torch.Tensor,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.discriminator_optimizer is not None

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_features.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            fake_delta_norm = self.generator(current_features, text_features, noise=noise)
        fake_scores = self.discriminator(current_features, text_features, fake_delta_norm)
        real_scores = self.discriminator(current_features, text_features, real_delta_norm)
        disc_loss = discriminator_bce_loss(
            real_scores,
            fake_scores,
            real_label_value=float(self.config.real_label_value),
            fake_label_value=float(self.config.fake_label_value),
        )
        disc_loss.backward()
        clip_grad_norm_(self.discriminator.parameters(), float(self.config.discriminator_grad_clip))
        self.discriminator_optimizer.step()
        return {
            "d_total": float(disc_loss.detach().cpu()),
            "d_real": float(real_scores.mean().detach().cpu()),
            "d_fake": float(fake_scores.mean().detach().cpu()),
        }

    def _generator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        current_flat: torch.Tensor,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.generator_optimizer is not None
        assert self.bundle is not None
        assert self.normalization is not None

        strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)

        self.generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_features.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        fake_delta_norm = self.generator(current_features, text_features, noise=noise)
        fake_scores = self.discriminator(current_features, text_features, fake_delta_norm)
        fake_delta = (
            denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
            if self.config.normalize_target_delta
            else fake_delta_norm
        )
        future_flat = reconstruct_future_surface(current_flat, fake_delta)
        future_log_surface = torch.log(torch.clamp(future_flat, min=1e-4)).view(
            current_features.size(0),
            self.bundle.surface_shape[0],
            self.bundle.surface_shape[1],
        )

        adv_loss = generator_bce_loss(fake_scores, target_value=float(self.config.real_label_value))
        smooth_m = strike_smoothness_penalty(future_log_surface, strike_grid)
        smooth_t = maturity_smoothness_penalty(future_log_surface, maturity_days_grid)
        total_loss = adv_loss + self.alpha_m * smooth_m + self.alpha_tau * smooth_t
        total_loss.backward()
        clip_grad_norm_(self.generator.parameters(), float(self.config.generator_grad_clip))
        self.generator_optimizer.step()
        return {
            "g_total": float(total_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_smooth_m": float(smooth_m.detach().cpu()),
            "g_smooth_t": float(smooth_t.detach().cpu()),
        }

    def _evaluate(self) -> dict[str, float]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.normalization is not None

        if not self.bundle.val_items:
            return {}

        strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)

        self.generator.eval()
        mae: list[float] = []
        rmse: list[float] = []
        current_mae: list[float] = []
        current_rmse: list[float] = []
        win_flags: list[float] = []
        generated_current_mae: list[float] = []
        real_current_mae: list[float] = []
        calendar: list[float] = []
        butterfly: list[float] = []
        penalty_mean: list[float] = []
        penalty_std: list[float] = []
        weight_entropy: list[float] = []

        for sample in self.bundle.val_items:
            payload = build_sample_payload(
                generator=self.generator,
                sample=sample,
                normalization=self.normalization,
                noise_dim=int(self.config.noise_dim),
                mc_samples=int(self.config.eval_mc_samples),
                seed=int(self.config.seed),
                device=self.device,
                reweight_beta_mode=self.config.eval_reweight_beta_mode,
                reweight_beta=float(self.config.eval_reweight_beta),
                aggregation_mode=self.config.eval_aggregation_mode,
                quantiles=(),
                checkpoint_path="",
                split="val",
                selection_mode="all",
                normalize_current_surface=bool(self.config.normalize_current_surface),
                normalize_text_embedding=bool(self.config.normalize_text_embedding),
                normalize_target_delta=bool(self.config.normalize_target_delta),
            )
            mae.append(float(payload["metrics"]["mae"]))
            rmse.append(float(payload["metrics"]["rmse"]))
            current_mae.append(float(payload["current_metrics"]["mae"]))
            current_rmse.append(float(payload["current_metrics"]["rmse"]))
            generated_current_mae.append(float(payload["generated_current_metrics"]["mae"]))
            real_current_mae.append(float(payload["current_metrics"]["mae"]))
            win_flags.append(1.0 if float(payload["metrics"]["mae"]) < float(payload["current_metrics"]["mae"]) else 0.0)
            penalty_mean.append(float(payload["penalty_mean"]))
            penalty_std.append(float(payload["penalty_std"]))
            weight_entropy.append(float(payload["weight_entropy"]))

            weighted_surface = torch.tensor(payload["generated_surface"], dtype=torch.float32, device=self.device).unsqueeze(0)
            calendar.extend(
                calendar_arbitrage_penalty(weighted_surface, strike_grid, maturity_days_grid).detach().cpu().tolist()
            )
            butterfly.extend(
                butterfly_arbitrage_penalty(weighted_surface, strike_grid, maturity_days_grid).detach().cpu().tolist()
            )

        self.generator.train()
        val_mae = float(np.mean(mae)) if mae else 0.0
        val_current_mae = float(np.mean(current_mae)) if current_mae else 0.0
        val_generated_current_mae = float(np.mean(generated_current_mae)) if generated_current_mae else 0.0
        val_real_current_mae = float(np.mean(real_current_mae)) if real_current_mae else 0.0
        mean_reverting_regime = (
            1.0
            if val_mae > val_current_mae and val_generated_current_mae < 0.75 * max(val_real_current_mae, 1e-12)
            else 0.0
        )
        return {
            "val_mae": val_mae,
            "val_rmse": float(np.mean(rmse)) if rmse else 0.0,
            "val_current_mae": val_current_mae,
            "val_current_rmse": float(np.mean(current_rmse)) if current_rmse else 0.0,
            "val_mae_gap_vs_current": val_mae - val_current_mae,
            "val_win_rate_vs_current": float(np.mean(win_flags)) if win_flags else 0.0,
            "val_generated_current_mae": val_generated_current_mae,
            "val_real_current_mae": val_real_current_mae,
            "val_calendar": float(np.mean(calendar)) if calendar else 0.0,
            "val_butterfly": float(np.mean(butterfly)) if butterfly else 0.0,
            "val_penalty_mean": float(np.mean(penalty_mean)) if penalty_mean else 0.0,
            "val_penalty_std": float(np.mean(penalty_std)) if penalty_std else 0.0,
            "val_weight_entropy": float(np.mean(weight_entropy)) if weight_entropy else 0.0,
            "mean_reverting_regime": mean_reverting_regime,
        }

    def _checkpoint_payload(self) -> dict[str, object]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.discriminator is not None
        return {
            "config": config_payload(self.config),
            "alpha_m": float(self.alpha_m),
            "alpha_tau": float(self.alpha_tau),
            "surface_shape": list(self.bundle.surface_shape),
            "strike_grid": self.bundle.strike_grid.astype(float).tolist(),
            "maturity_days_grid": self.bundle.maturity_days_grid.astype(float).tolist(),
            "embedding_dim": int(self.bundle.embedding_dim),
            "normalization_stats": {
                "current_log_mean": self.bundle.normalization_stats.current_log_mean.astype(float).tolist(),
                "current_log_std": self.bundle.normalization_stats.current_log_std.astype(float).tolist(),
                "delta_mean": self.bundle.normalization_stats.delta_mean.astype(float).tolist(),
                "delta_std": self.bundle.normalization_stats.delta_std.astype(float).tolist(),
                "text_mean": self.bundle.normalization_stats.text_mean.astype(float).tolist(),
                "text_std": self.bundle.normalization_stats.text_std.astype(float).tolist(),
            },
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }

    def _save_loss_curves(self, metrics_rows: list[dict[str, float]]) -> None:
        output_path = self.metrics_dir / "loss_curves.png"
        plot_training_curves(metrics_rows, output_path=output_path, title="Standalone VolGAN Training Curves")

    def _train_impl(self) -> Path:
        assert self.bundle is not None

        metrics_rows: list[dict[str, float]] = []
        best_metric = float("inf")
        best_epoch = 0

        for epoch in range(1, int(self.config.num_epochs) + 1):
            running: dict[str, list[float]] = {
                "d_total": [],
                "d_real": [],
                "d_fake": [],
                "g_total": [],
                "g_adv": [],
                "g_smooth_m": [],
                "g_smooth_t": [],
            }
            for current_features, text_features, real_delta_norm, current_flat, _target_flat in self.bundle.train_loader:
                current_features = self._to_device(current_features)
                text_features = self._to_device(text_features)
                real_delta_norm = self._to_device(real_delta_norm)
                current_flat = self._to_device(current_flat)

                for _ in range(max(1, int(self.config.disc_steps_per_batch))):
                    d_metrics = self._discriminator_step(current_features, text_features, real_delta_norm)
                    for key, value in d_metrics.items():
                        running[key].append(float(value))
                for _ in range(max(1, int(self.config.gen_steps_per_batch))):
                    g_metrics = self._generator_step(current_features, text_features, current_flat)
                    for key, value in g_metrics.items():
                        running[key].append(float(value))

            row = {
                "epoch": float(epoch),
                "d_total": float(np.mean(running["d_total"])) if running["d_total"] else 0.0,
                "d_real": float(np.mean(running["d_real"])) if running["d_real"] else 0.0,
                "d_fake": float(np.mean(running["d_fake"])) if running["d_fake"] else 0.0,
                "g_total": float(np.mean(running["g_total"])) if running["g_total"] else 0.0,
                "g_adv": float(np.mean(running["g_adv"])) if running["g_adv"] else 0.0,
                "g_smooth_m": float(np.mean(running["g_smooth_m"])) if running["g_smooth_m"] else 0.0,
                "g_smooth_t": float(np.mean(running["g_smooth_t"])) if running["g_smooth_t"] else 0.0,
                "alpha_m": float(self.alpha_m),
                "alpha_tau": float(self.alpha_tau),
            }
            row.update(self._evaluate())
            metrics_rows.append(row)
            write_json(self.metrics_dir / "training_metrics.json", metrics_rows)
            write_csv(self.metrics_dir / "training_metrics.csv", metrics_rows)
            self._save_loss_curves(metrics_rows)

            monitor_name = str(self.config.checkpoint_metric).strip() or "val_mae_gap_vs_current"
            monitor_value = float(row.get(monitor_name, row.get("val_mae", row["g_total"])))
            if monitor_value < best_metric:
                best_metric = monitor_value
                best_epoch = epoch
                save_checkpoint(self.checkpoints_dir / "volgan_best.pt", self._checkpoint_payload())

            if epoch % int(self.config.save_every) == 0:
                save_checkpoint(self.checkpoints_dir / f"volgan_epoch_{epoch:04d}.pt", self._checkpoint_payload())

            if float(row.get("mean_reverting_regime", 0.0)) > 0.0:
                self.logger.warning(
                    "epoch=%s entered mean-reverting regime: val_mae=%.6f current_mae=%.6f generated_current_mae=%.6f",
                    epoch,
                    row.get("val_mae", 0.0),
                    row.get("val_current_mae", 0.0),
                    row.get("val_generated_current_mae", 0.0),
                )

            self.logger.info(
                "epoch=%s g_total=%.6f d_total=%.6f val_mae=%.6f val_current_mae=%.6f gap=%.6f win_rate=%.3f",
                epoch,
                row["g_total"],
                row["d_total"],
                row.get("val_mae", 0.0),
                row.get("val_current_mae", 0.0),
                row.get("val_mae_gap_vs_current", 0.0),
                row.get("val_win_rate_vs_current", 0.0),
            )

        save_checkpoint(self.checkpoints_dir / "volgan_final.pt", self._checkpoint_payload())
        write_json(
            self.metrics_dir / "best_checkpoint.json",
            {
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "checkpoint_metric": str(self.config.checkpoint_metric),
                "checkpoint_path": str(self.checkpoints_dir / "volgan_best.pt"),
            },
        )
        self.logger.info("Standalone VolGAN training complete. Best epoch=%s best_metric=%.6f", best_epoch, best_metric)
        return self.run_dir

    def _prepare_generate_result(
        self,
        generate_config: VolGANSampleConfig | None = None,
        *,
        overrides: Mapping[str, object] | None = None,
        config_path: str | None = None,
    ) -> tuple[VolGANSampleConfig, Path]:
        output_root = str(getattr(self.raw_config, "output_root", "")).strip() or default_train_output_root(
            self.raw_config.data_path
        )
        checkpoint_override = None if not overrides else overrides.get("checkpoint_path")
        run_dir = self.run_dir or resolve_existing_run_dir(
            output_root=output_root,
            checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
        )
        if generate_config is None:
            if config_path:
                generate_config = load_sample_config(
                    config_path,
                    run_dir=run_dir,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                )
            elif self.config_path:
                generate_config = build_sample_config_from_train_config(
                    self.config_path,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                    run_dir=run_dir,
                )
            else:
                generate_config = VolGANSampleConfig(
                    data_path=self.raw_config.data_path,
                    sheet_name=self.raw_config.sheet_name,
                    text_embedding_mode=self.raw_config.text_embedding_mode,
                    train_ratio=self.raw_config.train_ratio,
                    checkpoint_path="",
                    seed=int(self.raw_config.seed),
                    cuda=bool(self.raw_config.cuda),
                    mc_samples=int(self.raw_config.eval_mc_samples),
                    reweight_beta_mode=str(self.raw_config.eval_reweight_beta_mode),
                    reweight_beta=float(self.raw_config.eval_reweight_beta),
                    aggregation_mode=str(self.raw_config.eval_aggregation_mode),
                    output_dir="",
                )
        if overrides:
            generate_config = replace(generate_config, **dict(overrides))
        resolved_checkpoint_path = (
            Path(generate_config.checkpoint_path)
            if str(generate_config.checkpoint_path).strip()
            else find_best_checkpoint(run_dir, filename="volgan_best.pt")
        )
        if str(generate_config.output_dir).strip():
            resolved_generate_dir = generate_result_dir(run_dir, generate_config.output_dir)
        else:
            resolved_generate_dir = checkpoint_named_dir(generate_result_dir(run_dir), resolved_checkpoint_path)
        resolved_config = replace(
            generate_config,
            checkpoint_path=str(resolved_checkpoint_path),
            output_dir=str(resolved_generate_dir),
        )
        return resolved_config, resolved_generate_dir

    def _generate_result_impl(self, generate_config: VolGANSampleConfig, generate_dir: Path) -> Path:
        del generate_dir
        sampler = VolGANSampler(generate_config)
        return sampler.sample()
