"""Standalone VolGAN training loop and checkpoint management."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import RMSprop

from .arbitrage import butterfly_arbitrage_penalty, calendar_arbitrage_penalty
from .config import VolGANTrainConfig, save_config_yaml
from .data import VolSurfaceDataBundle, create_train_val_bundle
from .io import config_payload, prepare_run_dir, save_checkpoint, write_csv, write_json
from .losses import (
    discriminator_bce_loss,
    estimate_gradient_matching,
    generator_bce_loss,
    maturity_smoothness_penalty,
    strike_smoothness_penalty,
)
from .models import VolGANDiscriminator, VolGANGenerator, reconstruct_future_surface
from .training_plots import plot_training_curves


class VolGANTrainer:
    """Train the standalone VolGAN model on `merged_vol.xlsx` rows."""

    def __init__(self, config: VolGANTrainConfig):
        self.config = config
        self.run_dir = prepare_run_dir(config.output_root)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.samples_dir = self.run_dir / "samples"
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")
        self._logger: Optional[logging.Logger] = None

        self.bundle: Optional[VolSurfaceDataBundle] = None
        self.generator: Optional[VolGANGenerator] = None
        self.discriminator: Optional[VolGANDiscriminator] = None
        self.generator_optimizer: Optional[RMSprop] = None
        self.discriminator_optimizer: Optional[RMSprop] = None
        self.alpha_m = float(config.alpha_m)
        self.alpha_tau = float(config.alpha_tau)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger("volgan.trainer")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(handler)
            self._logger = logger
        return self._logger

    def _set_seed(self) -> None:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))

    def _save_resolved_config(self) -> None:
        save_config_yaml(self.config, self.metrics_dir / "resolved_config.yaml")

    def setup(self) -> None:
        self._set_seed()
        self.logger.info("Standalone VolGAN outputs: %s", self.run_dir)
        self.logger.info("Loading merged-vol workbook from %s (%s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_train_val_bundle(self.config)
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
        self.generator_optimizer = RMSprop(self.generator.parameters(), lr=float(self.config.learning_rate))
        self.discriminator_optimizer = RMSprop(self.discriminator.parameters(), lr=float(self.config.learning_rate))
        self._save_resolved_config()

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
                noise_dim=self.config.noise_dim,
                epochs=self.config.gradient_match_epochs,
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
        current_flat: torch.Tensor,
        text_embedding: torch.Tensor,
        real_delta: torch.Tensor,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.discriminator_optimizer is not None

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_flat.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            fake_delta = self.generator(current_flat, text_embedding, noise=noise)
        fake_scores = self.discriminator(current_flat, text_embedding, fake_delta)
        real_scores = self.discriminator(current_flat, text_embedding, real_delta)
        disc_loss = discriminator_bce_loss(real_scores, fake_scores)
        disc_loss.backward()
        self.discriminator_optimizer.step()
        return {
            "d_total": float(disc_loss.detach().cpu()),
            "d_real": float(real_scores.mean().detach().cpu()),
            "d_fake": float(fake_scores.mean().detach().cpu()),
        }

    def _generator_step(
        self,
        current_flat: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.generator_optimizer is not None
        assert self.bundle is not None

        strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)

        self.generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_flat.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        fake_delta = self.generator(current_flat, text_embedding, noise=noise)
        fake_scores = self.discriminator(current_flat, text_embedding, fake_delta)
        future_flat = reconstruct_future_surface(current_flat, fake_delta)
        future_log_surface = torch.log(torch.clamp(future_flat, min=1e-4)).view(
            current_flat.size(0),
            self.bundle.surface_shape[0],
            self.bundle.surface_shape[1],
        )

        adv_loss = generator_bce_loss(fake_scores)
        smooth_m = strike_smoothness_penalty(future_log_surface, strike_grid)
        smooth_t = maturity_smoothness_penalty(future_log_surface, maturity_days_grid)
        total_loss = adv_loss + self.alpha_m * smooth_m + self.alpha_tau * smooth_t
        total_loss.backward()
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
        strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)

        if self.bundle.val_loader is None:
            return {}

        self.generator.eval()
        mae: list[float] = []
        rmse: list[float] = []
        calendar: list[float] = []
        butterfly: list[float] = []
        with torch.no_grad():
            for current_flat, text_embedding, _real_delta, target_flat in self.bundle.val_loader:
                current_flat = self._to_device(current_flat)
                text_embedding = self._to_device(text_embedding)
                target_flat = self._to_device(target_flat)
                zero_noise = torch.zeros(
                    current_flat.size(0),
                    int(self.config.noise_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                predicted_delta = self.generator(current_flat, text_embedding, noise=zero_noise)
                predicted_flat = reconstruct_future_surface(current_flat, predicted_delta)
                diff = predicted_flat - target_flat
                mae.append(float(torch.mean(torch.abs(diff)).cpu()))
                rmse.append(float(torch.sqrt(torch.mean(diff.pow(2))).cpu()))
                predicted_surface = self._surface_tensor(predicted_flat)
                calendar.extend(
                    calendar_arbitrage_penalty(predicted_surface, strike_grid, maturity_days_grid).detach().cpu().tolist()
                )
                butterfly.extend(
                    butterfly_arbitrage_penalty(predicted_surface, strike_grid, maturity_days_grid)
                    .detach()
                    .cpu()
                    .tolist()
                )
        self.generator.train()
        return {
            "val_mae": float(np.mean(mae)) if mae else 0.0,
            "val_rmse": float(np.mean(rmse)) if rmse else 0.0,
            "val_calendar": float(np.mean(calendar)) if calendar else 0.0,
            "val_butterfly": float(np.mean(butterfly)) if butterfly else 0.0,
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
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }

    def _save_loss_curves(self, metrics_rows: list[dict[str, float]]) -> None:
        output_path = self.metrics_dir / "loss_curves.png"
        plot_training_curves(metrics_rows, output_path=output_path, title="Standalone VolGAN Training Curves")

    def train(self) -> Path:
        self.setup()
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
            for current_flat, text_embedding, real_delta, _target_flat in self.bundle.train_loader:
                current_flat = self._to_device(current_flat)
                text_embedding = self._to_device(text_embedding)
                real_delta = self._to_device(real_delta)
                d_metrics = self._discriminator_step(current_flat, text_embedding, real_delta)
                g_metrics = self._generator_step(current_flat, text_embedding)
                for key, value in {**d_metrics, **g_metrics}.items():
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

            monitor_value = float(row.get("val_mae", row["g_total"]))
            if monitor_value < best_metric:
                best_metric = monitor_value
                best_epoch = epoch
                save_checkpoint(self.checkpoints_dir / "volgan_best.pt", self._checkpoint_payload())

            if epoch % int(self.config.save_every) == 0:
                save_checkpoint(self.checkpoints_dir / f"volgan_epoch_{epoch:04d}.pt", self._checkpoint_payload())

            self.logger.info(
                "epoch=%s g_total=%.6f d_total=%.6f val_mae=%.6f val_calendar=%.6f",
                epoch,
                row["g_total"],
                row["d_total"],
                row.get("val_mae", 0.0),
                row.get("val_calendar", 0.0),
            )

        save_checkpoint(self.checkpoints_dir / "volgan_final.pt", self._checkpoint_payload())
        write_json(
            self.metrics_dir / "best_checkpoint.json",
            {
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "checkpoint_path": str(self.checkpoints_dir / "volgan_best.pt"),
            },
        )
        self.logger.info("Standalone VolGAN training complete. Best epoch=%s best_metric=%.6f", best_epoch, best_metric)
        return self.run_dir
