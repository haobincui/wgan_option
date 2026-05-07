"""Standalone FiLM WGAN training loop and checkpoint management."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainer import BaseTrainer
from utils.output_paths import find_best_checkpoint, prepare_run_dir
from utils.training_paths import checkpoint_named_dir, generate_result_dir, infer_training_output_root, resolve_existing_run_dir
from .arbitrage import butterfly_arbitrage_penalty, calendar_arbitrage_penalty
from .config import (
    FilmWGANSampleConfig,
    FilmWGANTrainConfig,
    build_sample_config,
    build_sample_config_from_train_config,
    config_to_dict,
)
from .data import FilmWGANDataBundle, create_train_val_bundle, denormalize_tensor, normalize_surface_tensor
from .inference import FilmWGANSampler, build_sample_payload, normalization_stats_to_tensors
from .io import config_payload, save_checkpoint, write_csv, write_json
from .losses import (
    atm_short_pure_mae,
    build_atm_short_mask,
    build_reconstruction_weight_template,
    critic_wgan_loss,
    generator_wgan_loss,
    gradient_penalty,
    maturity_smoothness_penalty,
    parameter_count,
    reconstruction_loss,
    strike_smoothness_penalty,
    weighted_surface_mae,
)
from .models import FilmWGANCritic, FilmWGANGenerator, reconstruct_future_surface
from .training_plots import plot_training_curves
from wgan_option.config_parsing import load_yaml_mapping


def _load_generate_result_section(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    _, payload = load_yaml_mapping(config_path)
    section = payload.get("generate_result") or {}
    if section and not isinstance(section, dict):
        raise ValueError(f"Config section 'generate_result' in {config_path} must contain a YAML mapping.")
    return dict(section)


class FilmWGANTrainer(BaseTrainer):
    """Train the standalone FiLM WGAN model on `merged_vol.xlsx` rows."""

    trainer_id = "film_wgan"
    logger_name = "film_wgan.trainer"

    def __init__(self, config: FilmWGANTrainConfig, *, config_path: str | None = None):
        super().__init__(config, config_path=config_path)
        self.checkpoints_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

        self.bundle: Optional[FilmWGANDataBundle] = None
        self.generator: Optional[FilmWGANGenerator] = None
        self.critic: Optional[FilmWGANCritic] = None
        self.generator_optimizer: Optional[Adam] = None
        self.critic_optimizer: Optional[Adam] = None
        self.generator_scheduler: Optional[CosineAnnealingLR] = None
        self.critic_scheduler: Optional[CosineAnnealingLR] = None
        self.normalization = None
        self._strike_grid: Optional[torch.Tensor] = None
        self._maturity_days_grid: Optional[torch.Tensor] = None
        self._recon_weights_surface: Optional[torch.Tensor] = None
        self._recon_weights_flat: Optional[torch.Tensor] = None
        self._atm_short_mask_surface: Optional[torch.Tensor] = None
        self._atm_short_mask_flat: Optional[torch.Tensor] = None
        self._current_epoch: int = 0
        self._disc_step_count: int = 0
        self._gp_warmup_steps: int = 200
        self._grad_clip: float = 5.0
        self._lr_min_ratio: float = 0.1

    def _set_seed(self) -> None:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))

    def _prepare_runtime_config(self, config: FilmWGANTrainConfig) -> tuple[FilmWGANTrainConfig, Path]:
        output_root = infer_training_output_root(config, trainer_id=self.trainer_id)
        run_dir = prepare_run_dir(output_root, create=False)
        resolved_config = replace(
            config,
            output_root=str(output_root),
            checkpoints_path=str(run_dir / "checkpoints"),
            metrics_path=str(run_dir / "metrics"),
        )
        self.checkpoints_dir = Path(resolved_config.checkpoints_path)
        self.metrics_dir = Path(resolved_config.metrics_path)
        return resolved_config, run_dir

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _normalize_surface_flat(self, surface_flat: torch.Tensor) -> torch.Tensor:
        assert self.bundle is not None
        assert self.normalization is not None
        height, width = self.bundle.surface_shape
        normalized = normalize_surface_tensor(surface_flat, self.normalization.current_log_mean, self.normalization.current_log_std)
        return normalized.view(surface_flat.size(0), 1, height, width)

    def setup(self) -> None:
        self._set_seed()
        assert self.checkpoints_dir is not None
        assert self.metrics_dir is not None
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Standalone FiLM WGAN outputs: %s", self.run_dir)
        self.logger.info("Loading merged-vol workbook from %s (%s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_train_val_bundle(self.config)
        self.normalization = normalization_stats_to_tensors(self.bundle.normalization_stats, self.device)
        self._strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        self._maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)
        self._recon_weights_surface = build_reconstruction_weight_template(
            strike_grid=self._strike_grid,
            maturity_days_grid=self._maturity_days_grid,
            mode=self.config.recon_weight_mode,
            atm_range=float(self.config.recon_atm_range),
            short_end_max_days=float(self.config.recon_atm_short_end_max_days),
            atm_multiplier=float(self.config.recon_atm_multiplier),
        )
        self._recon_weights_flat = self._recon_weights_surface.reshape(-1)
        self._atm_short_mask_surface = build_atm_short_mask(
            strike_grid=self._strike_grid,
            maturity_days_grid=self._maturity_days_grid,
            atm_range=float(self.config.atm_short_range),
            max_days=float(self.config.atm_short_max_days),
        )
        self._atm_short_mask_flat = self._atm_short_mask_surface.reshape(-1)
        surface_height, surface_width = self.bundle.surface_shape
        self.generator = FilmWGANGenerator(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=self.bundle.embedding_dim,
            noise_dim=self.config.noise_dim,
            base_channels=self.config.gen_base_channels,
            res_blocks=self.config.gen_res_blocks,
            text_hidden_dim=self.config.text_hidden_dim,
            text_out_dim=self.config.text_out_dim,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
        ).to(self.device)
        self.critic = FilmWGANCritic(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=self.bundle.embedding_dim,
            base_channels=self.config.disc_base_channels,
            res_blocks=self.config.disc_res_blocks,
            text_hidden_dim=self.config.text_hidden_dim,
            text_out_dim=self.config.text_out_dim,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
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
        total_epochs = max(1, int(self.config.num_epochs))
        self.generator_scheduler = CosineAnnealingLR(
            self.generator_optimizer,
            T_max=total_epochs,
            eta_min=float(self.config.generator_learning_rate) * self._lr_min_ratio,
        )
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optimizer,
            T_max=total_epochs,
            eta_min=float(self.config.discriminator_learning_rate) * self._lr_min_ratio,
        )
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
        self.logger.info(
            "Reconstruction weighting: mode=%s atm_range=%.4f short_end_max_days=%.1f atm_multiplier=%.3f",
            str(self.config.recon_weight_mode),
            float(self.config.recon_atm_range),
            float(self.config.recon_atm_short_end_max_days),
            float(self.config.recon_atm_multiplier),
        )
        atm_short_cells = int(self._atm_short_mask_surface.sum().item()) if self._atm_short_mask_surface is not None else 0
        self.logger.info(
            "ATM-short pure loss: enabled=%s lambda=%.4f range=%.4f max_days=%.1f cells=%d/%d",
            bool(self.config.use_atm_short_loss),
            float(self.config.lambda_atm_short),
            float(self.config.atm_short_range),
            float(self.config.atm_short_max_days),
            atm_short_cells,
            surface_height * surface_width,
        )
        self.logger.info(
            "Adversarial weighting: lambda_adv=%.4f adv_warmup_epochs=%d",
            float(self.config.lambda_adv),
            int(self.config.adv_warmup_epochs),
        )

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
        self._disc_step_count += 1
        warmup_factor = min(1.0, self._disc_step_count / max(1, self._gp_warmup_steps))
        effective_lambda_gp = float(self.config.lambda_gp) * warmup_factor
        gp = gradient_penalty(
            critic=self.critic,
            real_future_surface=real_future_surface,
            fake_future_surface=fake_future_surface,
            current_surface=current_features,
            text_embedding=text_features,
            lambda_gp=effective_lambda_gp,
        )
        disc_loss = critic_wgan_loss(real_scores, fake_scores) + gp
        disc_loss.backward()
        if torch.isfinite(disc_loss):
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self._grad_clip)
            self.critic_optimizer.step()
        else:
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.logger.warning("Non-finite disc_loss detected at step %s; skipping critic update.", self._disc_step_count)
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
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.critic is not None
        assert self.generator_optimizer is not None
        assert self.bundle is not None
        assert self.normalization is not None
        assert self._strike_grid is not None
        assert self._maturity_days_grid is not None
        assert self._recon_weights_flat is not None

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
        future_log_surface = torch.log(torch.clamp(future_surface_level, min=1e-4))
        adv_loss = generator_wgan_loss(fake_scores)
        calendar_penalty = calendar_arbitrage_penalty(future_surface_level, self._strike_grid, self._maturity_days_grid).mean()
        butterfly_penalty = butterfly_arbitrage_penalty(future_surface_level, self._strike_grid, self._maturity_days_grid).mean()
        smooth_penalty = strike_smoothness_penalty(future_log_surface, self._strike_grid) + maturity_smoothness_penalty(
            future_log_surface,
            self._maturity_days_grid,
        )
        recon_penalty = reconstruction_loss(fake_future_flat, target_flat)
        recon_penalty_weighted = weighted_surface_mae(fake_future_flat, target_flat, self._recon_weights_flat)
        if self.config.use_atm_short_loss and self._atm_short_mask_flat is not None:
            atm_short_penalty = atm_short_pure_mae(fake_future_flat, target_flat, self._atm_short_mask_flat)
        else:
            atm_short_penalty = torch.zeros((), device=fake_future_flat.device, dtype=fake_future_flat.dtype)

        adv_warmup_epochs = max(0, int(self.config.adv_warmup_epochs))
        if self._current_epoch <= adv_warmup_epochs:
            effective_lambda_adv = 0.0
        else:
            effective_lambda_adv = float(self.config.lambda_adv)

        total_loss = effective_lambda_adv * adv_loss
        if self.config.use_calendar_constraint:
            total_loss = total_loss + float(self.config.lambda_calendar) * calendar_penalty
        if self.config.use_butterfly_constraint:
            total_loss = total_loss + float(self.config.lambda_butterfly) * butterfly_penalty
        if self.config.use_smooth_constraint:
            total_loss = total_loss + float(self.config.lambda_smooth) * smooth_penalty
        if self.config.use_recon_constraint:
            total_loss = total_loss + float(self.config.lambda_recon) * recon_penalty_weighted
        if self.config.use_atm_short_loss:
            total_loss = total_loss + float(self.config.lambda_atm_short) * atm_short_penalty
        total_loss.backward()
        if torch.isfinite(total_loss):
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self._grad_clip)
            self.generator_optimizer.step()
        else:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.logger.warning("Non-finite generator total_loss detected; skipping generator update.")
        return {
            "g_total": float(total_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_adv_effective_lambda": float(effective_lambda_adv),
            "g_calendar": float(calendar_penalty.detach().cpu()),
            "g_butterfly": float(butterfly_penalty.detach().cpu()),
            "g_smooth": float(smooth_penalty.detach().cpu()),
            "g_recon": float(recon_penalty.detach().cpu()),
            "g_recon_weighted": float(recon_penalty_weighted.detach().cpu()),
            "g_atm_short": float(atm_short_penalty.detach().cpu()),
        }

    def _evaluate(self) -> dict[str, float]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.normalization is not None
        assert self._strike_grid is not None
        assert self._maturity_days_grid is not None
        assert self._recon_weights_surface is not None

        if not self.bundle.val_items:
            return {}

        self.generator.eval()
        mae: list[float] = []
        rmse: list[float] = []
        current_mae: list[float] = []
        current_rmse: list[float] = []
        short_atm_weighted_mae: list[float] = []
        current_short_atm_weighted_mae: list[float] = []
        atm_short_pure_list: list[float] = []
        current_atm_short_pure_list: list[float] = []
        atm_short_win_flags: list[float] = []
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
                recon_weights_surface=self._recon_weights_surface,
                atm_short_mask_surface=self._atm_short_mask_surface,
                normalize_current_surface=bool(self.config.normalize_current_surface),
                normalize_text_embedding=bool(self.config.normalize_text_embedding),
                normalize_target_delta=bool(self.config.normalize_target_delta),
                residual_blend_alpha=1.0,
            )
            mae.append(float(payload["metrics"]["mae"]))
            rmse.append(float(payload["metrics"]["rmse"]))
            current_mae.append(float(payload["current_metrics"]["mae"]))
            current_rmse.append(float(payload["current_metrics"]["rmse"]))
            generated_current_mae.append(float(payload["generated_current_metrics"]["mae"]))
            real_current_mae.append(float(payload["current_metrics"]["mae"]))
            generated_surface = torch.tensor(payload["generated_surface"], dtype=torch.float32, device=self.device)
            current_surface = torch.tensor(payload["current_surface"], dtype=torch.float32, device=self.device)
            target_surface = torch.tensor(payload["target_surface"], dtype=torch.float32, device=self.device)
            short_atm_weighted_mae.append(
                float(weighted_surface_mae(generated_surface, target_surface, self._recon_weights_surface).detach().cpu())
            )
            current_short_atm_weighted_mae.append(
                float(weighted_surface_mae(current_surface, target_surface, self._recon_weights_surface).detach().cpu())
            )
            if self._atm_short_mask_surface is not None and float(self._atm_short_mask_surface.sum().item()) > 0.0:
                gen_atm_pure = float(
                    atm_short_pure_mae(generated_surface, target_surface, self._atm_short_mask_surface).detach().cpu()
                )
                cur_atm_pure = float(
                    atm_short_pure_mae(current_surface, target_surface, self._atm_short_mask_surface).detach().cpu()
                )
                atm_short_pure_list.append(gen_atm_pure)
                current_atm_short_pure_list.append(cur_atm_pure)
                atm_short_win_flags.append(1.0 if gen_atm_pure < cur_atm_pure else 0.0)
            win_flags.append(1.0 if float(payload["metrics"]["mae"]) < float(payload["current_metrics"]["mae"]) else 0.0)
            penalty_mean.append(float(payload["penalty_mean"]))
            penalty_std.append(float(payload["penalty_std"]))
            weight_entropy.append(float(payload["weight_entropy"]))

            weighted_surface = generated_surface.unsqueeze(0)
            calendar.extend(
                calendar_arbitrage_penalty(weighted_surface, self._strike_grid, self._maturity_days_grid).detach().cpu().tolist()
            )
            butterfly.extend(
                butterfly_arbitrage_penalty(weighted_surface, self._strike_grid, self._maturity_days_grid).detach().cpu().tolist()
            )

        self.generator.train()
        val_mae = float(np.mean(mae)) if mae else 0.0
        val_current_mae = float(np.mean(current_mae)) if current_mae else 0.0
        val_short_atm_weighted_mae = float(np.mean(short_atm_weighted_mae)) if short_atm_weighted_mae else 0.0
        val_current_short_atm_weighted_mae = (
            float(np.mean(current_short_atm_weighted_mae)) if current_short_atm_weighted_mae else 0.0
        )
        val_atm_short_pure_mae = float(np.mean(atm_short_pure_list)) if atm_short_pure_list else 0.0
        val_current_atm_short_pure_mae = (
            float(np.mean(current_atm_short_pure_list)) if current_atm_short_pure_list else 0.0
        )
        return {
            "val_mae": val_mae,
            "val_rmse": float(np.mean(rmse)) if rmse else 0.0,
            "val_current_mae": val_current_mae,
            "val_current_rmse": float(np.mean(current_rmse)) if current_rmse else 0.0,
            "val_short_atm_weighted_mae": val_short_atm_weighted_mae,
            "val_current_short_atm_weighted_mae": val_current_short_atm_weighted_mae,
            "val_short_atm_mae_gap_vs_current": val_short_atm_weighted_mae - val_current_short_atm_weighted_mae,
            "val_atm_short_pure_mae": val_atm_short_pure_mae,
            "val_current_atm_short_pure_mae": val_current_atm_short_pure_mae,
            "val_atm_short_pure_mae_gap_vs_current": val_atm_short_pure_mae - val_current_atm_short_pure_mae,
            "val_atm_short_win_rate_vs_current": float(np.mean(atm_short_win_flags)) if atm_short_win_flags else 0.0,
            "val_mae_gap_vs_current": val_mae - val_current_mae,
            "val_win_rate_vs_current": float(np.mean(win_flags)) if win_flags else 0.0,
            "val_generated_current_mae": float(np.mean(generated_current_mae)) if generated_current_mae else 0.0,
            "val_real_current_mae": float(np.mean(real_current_mae)) if real_current_mae else 0.0,
            "val_calendar": float(np.mean(calendar)) if calendar else 0.0,
            "val_butterfly": float(np.mean(butterfly)) if butterfly else 0.0,
            "val_penalty_mean": float(np.mean(penalty_mean)) if penalty_mean else 0.0,
            "val_penalty_std": float(np.mean(penalty_std)) if penalty_std else 0.0,
            "val_weight_entropy": float(np.mean(weight_entropy)) if weight_entropy else 0.0,
        }

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
                "current_log_mean": self.bundle.normalization_stats.current_log_mean.astype(float).tolist(),
                "current_log_std": self.bundle.normalization_stats.current_log_std.astype(float).tolist(),
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
            title="Standalone FiLM WGAN Training Curves",
        )

    def _train_impl(self) -> Path:
        assert self.bundle is not None
        assert self.checkpoints_dir is not None
        assert self.metrics_dir is not None

        primary_metric = str(self.config.checkpoint_metric).strip() or "val_mae_gap_vs_current"
        configured_extra_metrics: list[str] = []
        seen_metrics = {primary_metric}
        for raw_metric in getattr(self.config, "extra_checkpoint_metrics", ()) or ():
            metric_name = str(raw_metric).strip()
            if not metric_name or metric_name in seen_metrics:
                continue
            configured_extra_metrics.append(metric_name)
            seen_metrics.add(metric_name)

        summary_metrics: list[str] = []
        for metric_name in [
            primary_metric,
            "val_mae_gap_vs_current",
            "val_short_atm_mae_gap_vs_current",
            "val_atm_short_pure_mae_gap_vs_current",
            *configured_extra_metrics,
        ]:
            if metric_name and metric_name not in summary_metrics:
                summary_metrics.append(metric_name)

        extra_checkpoint_paths = {
            metric_name: self.checkpoints_dir / f"film_wgan_best_{metric_name}.pt"
            for metric_name in configured_extra_metrics
        }
        metric_summary: dict[str, dict[str, object]] = {}
        for metric_name in summary_metrics:
            tracking_mode = "summary_only"
            if metric_name == primary_metric:
                tracking_mode = "primary"
            elif metric_name in configured_extra_metrics:
                tracking_mode = "extra"
            metric_summary[metric_name] = {
                "best_epoch": 0,
                "best_value": None,
                "checkpoint_path": "",
                "tracking_mode": tracking_mode,
                "is_primary": metric_name == primary_metric,
                "is_extra": metric_name in configured_extra_metrics,
            }

        metrics_rows: list[dict[str, float]] = []
        best_metric = float("inf")
        best_epoch = 0
        fallback_metric = float("inf")
        fallback_epoch = 0
        checkpoint_warmup_epochs = max(0, int(getattr(self.config, "checkpoint_warmup_epochs", 0)))
        selection_start_epoch = checkpoint_warmup_epochs + 1
        fallback_checkpoint_path = self.checkpoints_dir / "film_wgan_best_warmup_fallback.pt"
        fallback_used = False

        early_stopping_enabled = bool(getattr(self.config, "use_early_stopping", False))
        early_stopping_patience = max(1, int(getattr(self.config, "early_stopping_patience", 10)))
        early_stopping_min_delta = float(getattr(self.config, "early_stopping_min_delta", 0.0))
        epochs_without_improvement = 0
        early_stopped = False

        if checkpoint_warmup_epochs > 0:
            self.logger.info(
                "Best checkpoint selection warmup enabled: skipping epochs <= %s; selection begins at epoch %s.",
                checkpoint_warmup_epochs,
                selection_start_epoch,
            )
        if early_stopping_enabled:
            self.logger.info(
                "Early stopping enabled (patience=%d, min_delta=%.6f) on %s.",
                early_stopping_patience,
                early_stopping_min_delta,
                primary_metric,
            )

        for epoch in range(1, int(self.config.num_epochs) + 1):
            self._current_epoch = epoch
            running: dict[str, list[float]] = {
                "d_total": [],
                "d_real": [],
                "d_fake": [],
                "gp": [],
                "g_total": [],
                "g_adv": [],
                "g_adv_effective_lambda": [],
                "g_calendar": [],
                "g_butterfly": [],
                "g_smooth": [],
                "g_recon": [],
                "g_recon_weighted": [],
                "g_atm_short": [],
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

                g_metrics = self._generator_step(current_features, text_features, current_flat, target_flat)
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
                "g_adv_effective_lambda": float(np.mean(running["g_adv_effective_lambda"]))
                if running["g_adv_effective_lambda"]
                else 0.0,
                "g_calendar": float(np.mean(running["g_calendar"])) if running["g_calendar"] else 0.0,
                "g_butterfly": float(np.mean(running["g_butterfly"])) if running["g_butterfly"] else 0.0,
                "g_smooth": float(np.mean(running["g_smooth"])) if running["g_smooth"] else 0.0,
                "g_recon": float(np.mean(running["g_recon"])) if running["g_recon"] else 0.0,
                "g_recon_weighted": float(np.mean(running["g_recon_weighted"])) if running["g_recon_weighted"] else 0.0,
                "g_atm_short": float(np.mean(running["g_atm_short"])) if running["g_atm_short"] else 0.0,
            }
            row.update(self._evaluate())
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
            row["lr_generator"] = float(self.generator_optimizer.param_groups[0]["lr"]) if self.generator_optimizer else 0.0
            row["lr_critic"] = float(self.critic_optimizer.param_groups[0]["lr"]) if self.critic_optimizer else 0.0
            metrics_rows.append(row)
            write_json(self.metrics_dir / "training_metrics.json", metrics_rows)
            write_csv(self.metrics_dir / "training_metrics.csv", metrics_rows)
            self._save_loss_curves(metrics_rows)

            monitor_name = primary_metric
            monitor_value = float(row.get(monitor_name, row.get("val_mae", row["g_total"])))
            if monitor_value < fallback_metric:
                fallback_metric = monitor_value
                fallback_epoch = epoch
                save_checkpoint(fallback_checkpoint_path, self._checkpoint_payload())
            improved = False
            if epoch > checkpoint_warmup_epochs and monitor_value < (best_metric - early_stopping_min_delta):
                best_metric = monitor_value
                best_epoch = epoch
                improved = True
                epochs_without_improvement = 0
                save_checkpoint(self.checkpoints_dir / "film_wgan_best.pt", self._checkpoint_payload())
            elif epoch > checkpoint_warmup_epochs and early_stopping_enabled:
                epochs_without_improvement += 1
                self.logger.info(
                    "Early stopping patience %d/%d without %s improvement (current=%.6f, best=%.6f at epoch %d)",
                    epochs_without_improvement,
                    early_stopping_patience,
                    monitor_name,
                    monitor_value,
                    best_metric,
                    best_epoch,
                )
            if epoch > checkpoint_warmup_epochs:
                for metric_name, summary in metric_summary.items():
                    if metric_name not in row:
                        continue
                    metric_value = float(row[metric_name])
                    current_best = summary["best_value"]
                    if current_best is None or metric_value < float(current_best):
                        summary["best_epoch"] = int(epoch)
                        summary["best_value"] = float(metric_value)
                        if summary["tracking_mode"] == "extra":
                            save_checkpoint(extra_checkpoint_paths[metric_name], self._checkpoint_payload())
            if epoch % int(self.config.save_every) == 0:
                save_checkpoint(self.checkpoints_dir / f"film_wgan_epoch_{epoch:04d}.pt", self._checkpoint_payload())

            self.logger.info(
                "epoch=%s g_total=%.6f d_total=%.6f val_mae=%.6f val_current_mae=%.6f gap=%.6f short_atm_gap=%.6f atm_short_pure_gap=%.6f win_rate=%.3f",
                epoch,
                row["g_total"],
                row["d_total"],
                row.get("val_mae", 0.0),
                row.get("val_current_mae", 0.0),
                row.get("val_mae_gap_vs_current", 0.0),
                row.get("val_short_atm_mae_gap_vs_current", 0.0),
                row.get("val_atm_short_pure_mae_gap_vs_current", 0.0),
                row.get("val_win_rate_vs_current", 0.0),
            )

            if (
                early_stopping_enabled
                and epoch > checkpoint_warmup_epochs
                and best_epoch > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                self.logger.info(
                    "Early stopping triggered at epoch %d. Best %s=%.6f at epoch %d.",
                    epoch,
                    monitor_name,
                    best_metric,
                    best_epoch,
                )
                early_stopped = True
                break

        if best_epoch == 0 and fallback_epoch > 0:
            fallback_used = True
            best_metric = fallback_metric
            best_epoch = fallback_epoch
            fallback_checkpoint_path.replace(self.checkpoints_dir / "film_wgan_best.pt")
            self.logger.warning(
                "No epoch exceeded checkpoint_warmup_epochs=%s during training; falling back to overall best epoch=%s.",
                checkpoint_warmup_epochs,
                best_epoch,
            )
        elif fallback_checkpoint_path.exists():
            fallback_checkpoint_path.unlink()

        save_checkpoint(self.checkpoints_dir / "film_wgan_final.pt", self._checkpoint_payload())
        write_json(
            self.metrics_dir / "best_checkpoint.json",
            {
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "checkpoint_metric": str(self.config.checkpoint_metric),
                "checkpoint_warmup_epochs": int(checkpoint_warmup_epochs),
                "selection_start_epoch": int(selection_start_epoch),
                "fallback_used": bool(fallback_used),
                "early_stopped": bool(early_stopped),
                "checkpoint_path": str(self.checkpoints_dir / "film_wgan_best.pt"),
            },
        )
        primary_summary = metric_summary[primary_metric]
        primary_summary["best_epoch"] = int(best_epoch)
        primary_summary["best_value"] = float(best_metric)
        primary_summary["checkpoint_path"] = str(self.checkpoints_dir / "film_wgan_best.pt")
        for metric_name, summary in metric_summary.items():
            if summary["tracking_mode"] == "extra" and int(summary["best_epoch"]) > 0:
                summary["checkpoint_path"] = str(extra_checkpoint_paths[metric_name])
        write_json(
            self.metrics_dir / "best_metrics_summary.json",
            {
                "primary_metric": primary_metric,
                "checkpoint_warmup_epochs": int(checkpoint_warmup_epochs),
                "selection_start_epoch": int(selection_start_epoch),
                "metrics": metric_summary,
            },
        )
        self.logger.info("Standalone FiLM WGAN training complete. Best epoch=%s best_metric=%.6f", best_epoch, best_metric)
        return self.run_dir

    def _prepare_generate_result(
        self,
        generate_config: FilmWGANSampleConfig | None = None,
        *,
        overrides: Mapping[str, object] | None = None,
        config_path: str | None = None,
    ) -> tuple[FilmWGANSampleConfig, Path]:
        base_training_config = self.config if self._runtime_prepared else self.raw_config
        output_root = infer_training_output_root(base_training_config, trainer_id=self.trainer_id)
        checkpoint_override = None if not overrides else overrides.get("checkpoint_path")

        if generate_config is None:
            if config_path:
                generate_config = build_sample_config_from_train_config(
                    config_path,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                    overrides=overrides,
                )
            elif self.config_path:
                generate_config = build_sample_config_from_train_config(
                    self.config_path,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                    overrides=overrides,
                )
            else:
                generate_config = build_sample_config(
                    training_values=config_to_dict(base_training_config),
                    generate_values=_load_generate_result_section(config_path),
                    overrides=overrides,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                )
        elif overrides:
            generate_config = replace(generate_config, **dict(overrides))

        run_dir = self.run_dir or resolve_existing_run_dir(
            output_root=output_root,
            checkpoint_path=generate_config.checkpoint_path or None,
        )
        resolved_checkpoint_path = (
            Path(generate_config.checkpoint_path)
            if str(generate_config.checkpoint_path).strip()
            else find_best_checkpoint(run_dir, filename="film_wgan_best.pt")
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

    def _generate_result_impl(self, generate_config: FilmWGANSampleConfig, generate_dir: Path) -> Path:
        del generate_dir
        sampler = FilmWGANSampler(generate_config)
        return sampler.sample()
