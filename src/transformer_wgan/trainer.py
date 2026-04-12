"""Standalone Transformer WGAN training loop and checkpoint management."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TransformerWGANTrainConfig, save_config_yaml
from .data import TransformerWGANDataBundle, create_train_val_bundle, denormalize_tensor, normalize_tensor
from .inference import normalization_stats_to_tensors
from .io import config_payload, prepare_run_dir, save_checkpoint, write_csv, write_json
from .losses import (
    assemble_generator_loss,
    butterfly_arbitrage_penalty,
    calendar_arbitrage_penalty,
    critic_wgan_loss,
    delta_shrink_penalty,
    generator_wgan_loss,
    gradient_penalty,
    parameter_count,
    resolve_monitor_metric,
    smoothness_penalty,
    summarize_baseline_aware_metrics,
)
from .models import TransformerWGANCritic, TransformerWGANGenerator, reconstruct_future_surface
from .training_plots import plot_training_curves


class TransformerWGANTrainer:
    """Train the standalone Transformer WGAN model on `merged_vol.xlsx` rows."""

    def __init__(self, config: TransformerWGANTrainConfig):
        self.config = config
        self.run_dir = prepare_run_dir(config.output_root)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")
        self._logger: Optional[logging.Logger] = None

        self.bundle: Optional[TransformerWGANDataBundle] = None
        self.generator: Optional[TransformerWGANGenerator] = None
        self.critic: Optional[TransformerWGANCritic] = None
        self.generator_optimizer: Optional[Adam] = None
        self.critic_optimizer: Optional[Adam] = None
        self.generator_scheduler: Optional[ReduceLROnPlateau] = None
        self.critic_scheduler: Optional[ReduceLROnPlateau] = None
        self.normalization = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger("transformer_wgan.trainer")
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

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _normalize_surface_flat(self, surface_flat: torch.Tensor) -> torch.Tensor:
        assert self.bundle is not None
        assert self.normalization is not None
        height, width = self.bundle.surface_shape
        normalized = (
            normalize_tensor(surface_flat, self.normalization.current_mean, self.normalization.current_std)
            if self.config.normalize_current_surface
            else surface_flat
        )
        return normalized.view(surface_flat.size(0), 1, height, width)

    def _constraint_switches_for_epoch(self, epoch: int) -> tuple[bool, bool, bool]:
        if int(epoch) <= max(0, int(self.config.constraint_warmup_epochs)):
            return False, False, False
        return (
            bool(self.config.use_calendar_constraint),
            bool(self.config.use_butterfly_constraint),
            bool(self.config.use_smooth_constraint),
        )

    @staticmethod
    def _optimizer_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _create_plateau_scheduler(self, optimizer) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.reduce_lr_factor),
            patience=int(self.config.reduce_lr_patience),
            min_lr=float(self.config.reduce_lr_min_lr),
        )

    def _step_plateau_scheduler(
        self,
        *,
        scheduler: Optional[ReduceLROnPlateau],
        optimizer,
        metric_name: str,
        metric_value: float,
        label: str,
    ) -> None:
        if scheduler is None:
            return
        old_lr = self._optimizer_lr(optimizer)
        scheduler.step(metric_value)
        new_lr = self._optimizer_lr(optimizer)
        if not np.isclose(old_lr, new_lr):
            self.logger.info(
                "%s ReduceLROnPlateau lowered LR from %.6g to %.6g using %s=%.6f",
                label,
                old_lr,
                new_lr,
                metric_name,
                metric_value,
            )

    def setup(self) -> None:
        self._set_seed()
        self.logger.info("Standalone Transformer WGAN outputs: %s", self.run_dir)
        self.logger.info("Loading merged-vol workbook from %s (%s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_train_val_bundle(self.config)
        self.normalization = normalization_stats_to_tensors(self.bundle.normalization_stats, self.device)
        surface_height, surface_width = self.bundle.surface_shape
        self.generator = TransformerWGANGenerator(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=self.bundle.embedding_dim,
            noise_dim=self.config.noise_dim,
            model_dim=self.config.model_dim,
            layers=self.config.gen_layers,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            text_hidden_dim=self.config.text_hidden_dim,
            text_token_dim=self.config.text_token_dim,
            noise_hidden_dim=self.config.noise_hidden_dim,
        ).to(self.device)
        self.critic = TransformerWGANCritic(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=self.bundle.embedding_dim,
            model_dim=self.config.model_dim,
            layers=self.config.disc_layers,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            text_hidden_dim=self.config.text_hidden_dim,
            text_token_dim=self.config.text_token_dim,
        ).to(self.device)
        self.generator_optimizer = Adam(
            self.generator.parameters(),
            lr=float(self.config.generator_learning_rate),
            betas=(float(self.config.beta_1), float(self.config.beta_2)),
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=float(self.config.discriminator_learning_rate),
            betas=(float(self.config.beta_1), float(self.config.beta_2)),
        )
        if bool(self.config.use_reduce_lr_on_plateau):
            self.generator_scheduler = self._create_plateau_scheduler(self.generator_optimizer)
            self.critic_scheduler = self._create_plateau_scheduler(self.critic_optimizer)
        self._save_resolved_config()
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, surface_shape=%s, embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            self.bundle.surface_shape,
            self.bundle.embedding_dim,
        )
        self.logger.info(
            "Model initialized: G params=%s, C params=%s",
            parameter_count(self.generator.parameters()),
            parameter_count(self.critic.parameters()),
        )
        self.logger.info("Pure adversarial mode: %s", bool(self.config.pure_adversarial))

    def _discriminator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        current_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.critic is not None
        assert self.critic_optimizer is not None
        assert self.normalization is not None

        self.critic_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_features.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            fake_delta_norm = self.generator(current_features, text_features, noise=noise)
            fake_delta = (
                denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
                if self.config.normalize_target_delta
                else fake_delta_norm
            )
            fake_future_flat = reconstruct_future_surface(current_flat, fake_delta)
        fake_future_surface = self._normalize_surface_flat(fake_future_flat)
        real_future_surface = self._normalize_surface_flat(target_flat)

        fake_scores = self.critic(fake_future_surface, current_features, text_features)
        real_scores = self.critic(real_future_surface, current_features, text_features)
        gp = gradient_penalty(
            critic=self.critic,
            real_future_surface=real_future_surface,
            fake_future_surface=fake_future_surface,
            current_surface=current_features,
            text_embedding=text_features,
            lambda_gp=float(self.config.lambda_gp),
        )
        disc_loss = critic_wgan_loss(real_scores, fake_scores) + gp
        disc_loss.backward()
        self.critic_optimizer.step()
        return {
            "d_total": float(disc_loss.detach().cpu()),
            "d_real": float(real_scores.mean().detach().cpu()),
            "d_fake": float(fake_scores.mean().detach().cpu()),
            "gp": float(gp.detach().cpu()),
        }

    def _generator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        current_flat: torch.Tensor,
        target_flat: torch.Tensor,
        *,
        epoch: int,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.critic is not None
        assert self.generator_optimizer is not None
        assert self.normalization is not None

        self.generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_features.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
        fake_delta_norm = self.generator(current_features, text_features, noise=noise)
        fake_delta = (
            denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
            if self.config.normalize_target_delta
            else fake_delta_norm
        )
        fake_future_flat = reconstruct_future_surface(current_flat, fake_delta)
        fake_future_surface = self._normalize_surface_flat(fake_future_flat)
        fake_scores = self.critic(fake_future_surface, current_features, text_features)

        future_surface_level = fake_future_flat.view(current_features.size(0), self.bundle.surface_shape[0], self.bundle.surface_shape[1])
        strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)
        adv_loss = generator_wgan_loss(fake_scores)
        recon_loss = F.l1_loss(fake_future_flat, target_flat)
        calendar_penalty = calendar_arbitrage_penalty(future_surface_level, strike_grid, maturity_days_grid).mean()
        butterfly_penalty = butterfly_arbitrage_penalty(future_surface_level, strike_grid, maturity_days_grid).mean()
        smooth_penalty_value = smoothness_penalty(future_surface_level)
        delta_shrink = delta_shrink_penalty(fake_future_flat, current_flat)
        use_calendar_constraint, use_butterfly_constraint, use_smooth_constraint = self._constraint_switches_for_epoch(epoch)

        total_loss = assemble_generator_loss(
            adv_loss=adv_loss,
            recon_loss=recon_loss,
            calendar_penalty=calendar_penalty,
            butterfly_penalty=butterfly_penalty,
            smooth_penalty_value=smooth_penalty_value,
            delta_shrink=delta_shrink,
            pure_adversarial=bool(self.config.pure_adversarial),
            lambda_recon=float(self.config.lambda_recon),
            lambda_calendar=float(self.config.lambda_calendar),
            lambda_butterfly=float(self.config.lambda_butterfly),
            lambda_smooth=float(self.config.lambda_smooth),
            lambda_delta_shrink=float(self.config.lambda_delta_shrink),
            use_calendar_constraint=use_calendar_constraint,
            use_butterfly_constraint=use_butterfly_constraint,
            use_smooth_constraint=use_smooth_constraint,
        )
        total_loss.backward()
        self.generator_optimizer.step()
        return {
            "g_total": float(total_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_recon": float(recon_loss.detach().cpu()),
            "g_calendar": float(calendar_penalty.detach().cpu()),
            "g_butterfly": float(butterfly_penalty.detach().cpu()),
            "g_smooth": float(smooth_penalty_value.detach().cpu()),
            "g_delta_shrink": float(delta_shrink.detach().cpu()),
        }

    def _evaluate(self) -> dict[str, float]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.normalization is not None

        if self.bundle.val_loader is None:
            return {}

        self.generator.eval()
        recon: list[float] = []
        current_recon: list[float] = []
        calendar: list[float] = []
        butterfly: list[float] = []
        delta_shrink_values: list[float] = []
        with torch.no_grad():
            for current_features, text_features, _real_delta_norm, current_flat, real_future in self.bundle.val_loader:
                current_features = self._to_device(current_features)
                text_features = self._to_device(text_features)
                current_flat = self._to_device(current_flat)
                real_future = self._to_device(real_future)
                zero_noise = torch.zeros(current_features.size(0), int(self.config.noise_dim), device=self.device, dtype=torch.float32)
                fake_delta_norm = self.generator(current_features, text_features, noise=zero_noise)
                fake_delta = (
                    denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
                    if self.config.normalize_target_delta
                    else fake_delta_norm
                )
                fake_future = reconstruct_future_surface(current_flat, fake_delta)
                future_surface_level = fake_future.view(current_features.size(0), self.bundle.surface_shape[0], self.bundle.surface_shape[1])
                strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
                maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)
                recon.append(float(F.l1_loss(fake_future, real_future).detach().cpu()))
                current_recon.append(float(F.l1_loss(current_flat, real_future).detach().cpu()))
                calendar.append(float(calendar_arbitrage_penalty(future_surface_level, strike_grid, maturity_days_grid).mean().detach().cpu()))
                butterfly.append(float(butterfly_arbitrage_penalty(future_surface_level, strike_grid, maturity_days_grid).mean().detach().cpu()))
                delta_shrink_values.append(float(delta_shrink_penalty(fake_future, current_flat).detach().cpu()))
        self.generator.train()
        metrics = summarize_baseline_aware_metrics(
            recon,
            current_recon,
            baseline_penalty_weight=float(self.config.baseline_penalty_weight),
        )
        metrics.update(
            {
                "val_calendar": float(np.mean(calendar)) if calendar else 0.0,
                "val_butterfly": float(np.mean(butterfly)) if butterfly else 0.0,
                "val_delta_shrink": float(np.mean(delta_shrink_values)) if delta_shrink_values else 0.0,
            }
        )
        return metrics

    def _checkpoint_payload(self) -> dict[str, object]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.critic is not None
        return {
            "config": config_payload(self.config),
            "surface_shape": list(self.bundle.surface_shape),
            "strike_grid": self.bundle.strike_grid.astype(float).tolist(),
            "maturity_days_grid": self.bundle.maturity_days_grid.astype(float).tolist(),
            "embedding_dim": int(self.bundle.embedding_dim),
            "normalization_stats": {
                "current_mean": self.bundle.normalization_stats.current_mean.astype(float).tolist(),
                "current_std": self.bundle.normalization_stats.current_std.astype(float).tolist(),
                "delta_mean": self.bundle.normalization_stats.delta_mean.astype(float).tolist(),
                "delta_std": self.bundle.normalization_stats.delta_std.astype(float).tolist(),
                "text_mean": self.bundle.normalization_stats.text_mean.astype(float).tolist(),
                "text_std": self.bundle.normalization_stats.text_std.astype(float).tolist(),
            },
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }

    def _save_loss_curves(self, metrics_rows: list[dict[str, float]]) -> None:
        plot_training_curves(
            metrics_rows,
            output_path=self.metrics_dir / "loss_curves.png",
            title="Standalone Transformer WGAN Training Curves",
        )

    def train(self) -> Path:
        self.setup()
        assert self.bundle is not None
        assert self.generator_optimizer is not None
        assert self.critic_optimizer is not None

        metrics_rows: list[dict[str, float]] = []
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, int(self.config.num_epochs) + 1):
            running: dict[str, list[float]] = {
                "d_total": [],
                "d_real": [],
                "d_fake": [],
                "gp": [],
                "g_total": [],
                "g_adv": [],
                "g_recon": [],
                "g_calendar": [],
                "g_butterfly": [],
                "g_smooth": [],
                "g_delta_shrink": [],
            }
            for current_features, text_features, _real_delta_norm, current_flat, target_flat in self.bundle.train_loader:
                current_features = self._to_device(current_features)
                text_features = self._to_device(text_features)
                current_flat = self._to_device(current_flat)
                target_flat = self._to_device(target_flat)

                for _ in range(max(1, int(self.config.critic_iter))):
                    d_metrics = self._discriminator_step(current_features, text_features, current_flat, target_flat)
                    for key, value in d_metrics.items():
                        running[key].append(float(value))

                g_metrics = self._generator_step(
                    current_features,
                    text_features,
                    current_flat,
                    target_flat,
                    epoch=epoch,
                )
                for key, value in g_metrics.items():
                    running[key].append(float(value))

            row = {
                "epoch": float(epoch),
                "d_total": float(np.mean(running["d_total"])) if running["d_total"] else 0.0,
                "d_real": float(np.mean(running["d_real"])) if running["d_real"] else 0.0,
                "d_fake": float(np.mean(running["d_fake"])) if running["d_fake"] else 0.0,
                "gp": float(np.mean(running["gp"])) if running["gp"] else 0.0,
                "g_total": float(np.mean(running["g_total"])) if running["g_total"] else 0.0,
                "g_adv": float(np.mean(running["g_adv"])) if running["g_adv"] else 0.0,
                "g_recon": float(np.mean(running["g_recon"])) if running["g_recon"] else 0.0,
                "g_calendar": float(np.mean(running["g_calendar"])) if running["g_calendar"] else 0.0,
                "g_butterfly": float(np.mean(running["g_butterfly"])) if running["g_butterfly"] else 0.0,
                "g_smooth": float(np.mean(running["g_smooth"])) if running["g_smooth"] else 0.0,
                "g_delta_shrink": float(np.mean(running["g_delta_shrink"])) if running["g_delta_shrink"] else 0.0,
            }
            row.update(self._evaluate())
            metrics_rows.append(row)
            write_json(self.metrics_dir / "training_metrics.json", metrics_rows)
            write_csv(self.metrics_dir / "training_metrics.csv", metrics_rows)
            self._save_loss_curves(metrics_rows)

            monitor_name = str(self.config.best_checkpoint_metric).strip() or "val_recon"
            monitor_value = resolve_monitor_metric(row, monitor_name)
            improved = monitor_value < (best_metric - float(self.config.early_stopping_min_delta))
            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(self.checkpoints_dir / "transformer_wgan_best.pt", self._checkpoint_payload())
            else:
                epochs_without_improvement += 1

            if epoch % int(self.config.save_every) == 0:
                save_checkpoint(self.checkpoints_dir / f"transformer_wgan_epoch_{epoch:04d}.pt", self._checkpoint_payload())

            self._step_plateau_scheduler(
                scheduler=self.generator_scheduler,
                optimizer=self.generator_optimizer,
                metric_name=monitor_name,
                metric_value=monitor_value,
                label="Generator",
            )
            self._step_plateau_scheduler(
                scheduler=self.critic_scheduler,
                optimizer=self.critic_optimizer,
                metric_name=monitor_name,
                metric_value=monitor_value,
                label="Critic",
            )

            self.logger.info(
                "epoch=%s g_total=%.6f d_total=%.6f val_recon=%.6f val_current_recon=%.6f gap=%.6f pure_adv=%s",
                epoch,
                row["g_total"],
                row["d_total"],
                row.get("val_recon", 0.0),
                row.get("val_current_recon", 0.0),
                row.get("val_baseline_gap", 0.0),
                bool(self.config.pure_adversarial),
            )

            if bool(self.config.use_early_stopping) and epochs_without_improvement >= int(self.config.early_stopping_patience):
                self.logger.info(
                    "Early stopping triggered at epoch=%s after %s epochs without improvement.",
                    epoch,
                    epochs_without_improvement,
                )
                break

        save_checkpoint(self.checkpoints_dir / "transformer_wgan_final.pt", self._checkpoint_payload())
        write_json(
            self.metrics_dir / "best_checkpoint.json",
            {
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "checkpoint_metric": str(self.config.best_checkpoint_metric),
                "checkpoint_path": str(self.checkpoints_dir / "transformer_wgan_best.pt"),
            },
        )
        self.logger.info("Standalone Transformer WGAN training complete. Best epoch=%s best_metric=%.6f", best_epoch, best_metric)
        return self.run_dir
