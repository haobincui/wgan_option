"""Training wrapper for deterministic merged vol-surface regression."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import BaseTrainer
from utils.generate_result_runtime import (
    generate_vol_regression_result,
    validate_result_config,
)
from utils.postprocess_runtime import resolve_checkpoint_path
from utils.result_config import GenerateResultConfig, build_generate_result_config
from utils.training_paths import (
    checkpoint_named_dir,
    generate_result_dir,
    infer_training_output_root,
    resolve_existing_run_dir,
)
from wgan_option.config import (
    Config,
    config_to_dict,
    default_config,
    label_reliability_lineage,
)
from wgan_option.config_parsing import load_yaml_mapping
from wgan_option.models.vol_regressor import VolSurfaceRegressor
from wgan_option.utils.merged_xlsx import (
    VolSurfaceXlsxBundle,
    create_configured_vol_surface_dataloaders,
)
from wgan_option.utils.reproducibility import seed_everything
from wgan_option.utils.training_artifacts import (
    write_best_checkpoint,
    write_metrics_csv,
    write_metrics_json,
)
from wgan_option.utils.training_run_paths import prepare_timestamped_training_config
from wgan_option.utils.visualization import plot_training_curves
from wgan_option.utils.vol_forecast_metrics import resolve_monitor_metric
from wgan_option.utils.weighted_training import (
    masked_mean_per_sample,
    reconstruction_training_weights,
    training_weighted_mean,
    unpack_vol_training_batch,
    validated_sample_weights,
    validated_label_reliability_weights,
    validated_surface_mask,
    weighted_mean,
)


def _load_generate_result_section(config_path: str | None) -> dict:
    if not config_path:
        return {}
    _, payload = load_yaml_mapping(config_path)
    section = payload.get("generate_result") or {}
    if section and not isinstance(section, dict):
        raise ValueError(
            f"Config section 'generate_result' in {config_path} must contain a YAML mapping."
        )
    return dict(section)


class VolSurfaceRegressionTrainer(BaseTrainer):
    """Train a deterministic residual forecaster on merged vol workbook rows."""

    trainer_id = "vol_regression"
    logger_name = "wgan_option.vol_regression"

    def __init__(self, config: Config, *, config_path: str | None = None):
        super().__init__(config, config_path=config_path)
        self.bundle: Optional[VolSurfaceXlsxBundle] = None
        self.model: Optional[VolSurfaceRegressor] = None
        self.optimizer: Optional[Adam] = None
        self._initial_learning_rate: float | None = None
        self.device = torch.device(
            "cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu"
        )
        self.strike_grid: Optional[torch.Tensor] = None
        self.tau_years: Optional[torch.Tensor] = None

    def _prepare_runtime_config(self, config: Config) -> tuple[Config, Path]:
        return prepare_timestamped_training_config(config, trainer_id=self.trainer_id)

    @property
    def logger(self) -> logging.Logger:
        return self._get_or_create_logger()

    def _ensure_samples_dir(self) -> None:
        if self.run_dir is None:
            return
        samples_path = str(self.config.samples_path).strip()
        if samples_path:
            Path(samples_path).mkdir(parents=True, exist_ok=True)

    def _log_device_info(self) -> None:
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = getattr(device_props, "total_memory", 0) / 1024**3
            self.logger.info(
                "CUDA available: %s (%.1f GB)",
                torch.cuda.get_device_name(0),
                total_memory_gb,
            )
        else:
            self.logger.info("CUDA not available, using CPU")
        self.logger.info("Device: %s", self.device)

    def setup(self) -> None:
        seed_everything(self.config.seed)
        self._log_device_info()

        self.logger.info("*** Loading merged vol-surface dataset ***")
        self.logger.info(
            "Data source: %s (sheet: %s)", self.config.data_path, self.config.sheet_name
        )
        self.bundle = create_configured_vol_surface_dataloaders(self.config)
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, test_samples=%s, "
            "surface_shape=(%s, %s), embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            self.bundle.test_samples,
            len(self.bundle.maturity_grid_days),
            len(self.bundle.strike_grid),
            self.bundle.embedding_dim,
        )
        if self.bundle.split_metadata:
            self.logger.info("Explicit split metadata: %s", self.bundle.split_metadata)

        self.strike_grid = torch.tensor(
            self.bundle.strike_grid, dtype=torch.float32, device=self.device
        ).clamp_min(1e-4)
        maturity_grid_days = torch.tensor(
            self.bundle.maturity_grid_days, dtype=torch.float32, device=self.device
        )
        self.tau_years = (maturity_grid_days / 365.0).clamp_min(1.0 / 365.0)

        self.logger.info("*** Initializing deterministic vol regressor ***")
        self.model = VolSurfaceRegressor(
            channels=int(self.config.channels),
            embedding_dim=int(self.bundle.embedding_dim),
            surface_height=int(len(self.bundle.maturity_grid_days)),
            surface_width=int(len(self.bundle.strike_grid)),
            base_channels=int(self.config.gen_base_channels),
            res_blocks=int(self.config.gen_res_blocks),
            text_hidden_dim=int(self.config.gen_text_hidden_dim),
            text_out_dim=int(self.config.gen_text_out_dim),
            hidden_dim=int(self.config.gen_hidden_dim),
            residual_output_mode=self.config.residual_output_mode,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Model initialized: %s parameters", total_params)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
        )

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    @staticmethod
    def _optimizer_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _create_plateau_scheduler(self, optimizer) -> ReduceLROnPlateau:
        scheduler_min_lr = self._validated_scheduler_min_lr(optimizer)
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.reduce_lr_factor),
            patience=int(self.config.reduce_lr_patience),
            min_lr=scheduler_min_lr,
        )

    def _validated_scheduler_min_lr(self, optimizer) -> float:
        """Return a non-increasing scheduler floor for the optimizer."""

        initial_lr = self._optimizer_lr(optimizer)
        scheduler_min_lr = float(self.config.reduce_lr_min_lr)
        if not np.isfinite(initial_lr) or initial_lr < 0.0:
            raise ValueError(
                f"Optimizer learning rate must be finite and non-negative, got {initial_lr}"
            )
        if not np.isfinite(scheduler_min_lr) or scheduler_min_lr < 0.0:
            raise ValueError(
                "reduce_lr_min_lr must be finite and non-negative, got "
                f"{scheduler_min_lr}"
            )
        if scheduler_min_lr > initial_lr:
            raise ValueError(
                "reduce_lr_min_lr cannot exceed the optimizer's initial learning "
                f"rate: {scheduler_min_lr} > {initial_lr}"
            )
        return scheduler_min_lr

    def _learning_rate_contract(self) -> dict[str, object]:
        """Return checkpoint-safe learning-rate lineage for this run."""

        optimizer_lr = (
            self._optimizer_lr(self.optimizer)
            if self.optimizer is not None
            else float(self.config.learning_rate)
        )
        initial_lr = (
            optimizer_lr
            if self._initial_learning_rate is None
            else float(self._initial_learning_rate)
        )
        trace = [
            {"epoch": int(row["epoch"]), "lr": float(row["lr"])}
            for row in getattr(self, "_metrics_rows", [])
            if "epoch" in row and "lr" in row
        ]
        return {
            "seed": int(self.config.seed),
            "lr_profile": self.config.news_first_lr_profile,
            "lr_profile_sha256": self.config.news_first_lr_profile_sha256,
            "fixed_lr_profile": self.config.news_first_fixed_learning_rate_profile,
            "fixed_lr_profile_sha256": (
                self.config.news_first_fixed_learning_rate_profile_sha256
            ),
            "capacity_seed_profile_sha256": (
                self.config.news_first_capacity_seed_profile_sha256
            ),
            "initial_learning_rate": initial_lr,
            "scheduler_min_lr": float(self.config.reduce_lr_min_lr),
            "lr_trace": trace,
        }

    def _label_reliability_lineage(self) -> dict[str, object]:
        """Return the hash-bound train-label contract for persisted artifacts."""

        return label_reliability_lineage(self.config)

    def _step_plateau_scheduler(
        self,
        *,
        scheduler: Optional[ReduceLROnPlateau],
        optimizer,
        metric_name: str,
        metric_value: float,
    ) -> None:
        if scheduler is None:
            return
        old_lr = self._optimizer_lr(optimizer)
        scheduler.step(metric_value)
        new_lr = self._optimizer_lr(optimizer)
        if not np.isclose(old_lr, new_lr):
            self.logger.info(
                "ReduceLROnPlateau lowered LR from %.6g to %.6g using %s=%.6f",
                old_lr,
                new_lr,
                metric_name,
                metric_value,
            )

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

    def _black_call_price(self, sigma: torch.Tensor) -> torch.Tensor:
        assert self.strike_grid is not None
        assert self.tau_years is not None
        sigma = sigma.clamp_min(1e-4)
        k = self.strike_grid.view(1, 1, -1)
        t = self.tau_years.view(1, -1, 1)
        sqrt_t = torch.sqrt(t)
        d1 = (torch.log(1.0 / k) + 0.5 * sigma.pow(2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return self._normal_cdf(d1) - k * self._normal_cdf(d2)

    @staticmethod
    def _reduce_per_sample(
        values: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
    ) -> torch.Tensor:
        if training_weights:
            return training_weighted_mean(values, sample_weight)
        return weighted_mean(values, sample_weight)

    def _calendar_penalty_per_sample(
        self,
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.tau_years is not None
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(1) < 2:
            return sigma.new_zeros(int(sigma.shape[0]))
        total_variance = sigma.pow(2) * self.tau_years.view(1, -1, 1)
        diff = total_variance[:, 1:, :] - total_variance[:, :-1, :]
        edge_mask = None
        if support_mask is not None:
            cells = support_mask.squeeze(1)
            edge_mask = cells[:, 1:, :] * cells[:, :-1, :]
        return masked_mean_per_sample(F.relu(-diff), edge_mask)

    def calendar_arbitrage_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._calendar_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    def _butterfly_penalty_per_sample(
        self,
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(2) < 3:
            return sigma.new_zeros(int(sigma.shape[0]))
        call_prices = self._black_call_price(sigma)
        second_diff = (
            call_prices[:, :, 2:]
            - 2.0 * call_prices[:, :, 1:-1]
            + call_prices[:, :, :-2]
        )
        triplet_mask = None
        if support_mask is not None:
            cells = support_mask.squeeze(1)
            triplet_mask = cells[:, :, 2:] * cells[:, :, 1:-1] * cells[:, :, :-2]
        return masked_mean_per_sample(F.relu(-second_diff), triplet_mask)

    def butterfly_arbitrage_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._butterfly_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    @staticmethod
    def _smoothness_penalty_per_sample(
        generated_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = generated_surface.squeeze(1)
        batch_size = int(sigma.shape[0])
        penalty = sigma.new_zeros(batch_size)
        cells = support_mask.squeeze(1) if support_mask is not None else None
        if sigma.size(1) > 1:
            maturity_mask = (
                cells[:, 1:, :] * cells[:, :-1, :] if cells is not None else None
            )
            penalty = penalty + masked_mean_per_sample(
                (sigma[:, 1:, :] - sigma[:, :-1, :]).pow(2),
                maturity_mask,
            )
        if sigma.size(2) > 1:
            strike_mask = (
                cells[:, :, 1:] * cells[:, :, :-1] if cells is not None else None
            )
            penalty = penalty + masked_mean_per_sample(
                (sigma[:, :, 1:] - sigma[:, :, :-1]).pow(2),
                strike_mask,
            )
        return penalty

    def smoothness_penalty(
        self,
        generated_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._smoothness_penalty_per_sample(generated_surface, support_mask),
            sample_weight,
            training_weights=training_weights,
        )

    @staticmethod
    def _delta_shrink_per_sample(
        predicted_surface: torch.Tensor,
        current_surface: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_mean_per_sample(
            torch.abs(predicted_surface - current_surface),
            support_mask,
        )

    def delta_shrink_penalty(
        self,
        predicted_surface: torch.Tensor,
        current_surface: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        training_weights: bool = False,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._reduce_per_sample(
            self._delta_shrink_per_sample(
                predicted_surface, current_surface, support_mask
            ),
            sample_weight,
            training_weights=training_weights,
        )

    def _constraint_warmup_epochs(self) -> int:
        return max(0, int(getattr(self.config, "constraint_warmup_epochs", 0)))

    def _constraint_switches_for_epoch(self, epoch: int) -> tuple[bool, bool, bool]:
        if int(epoch) <= self._constraint_warmup_epochs():
            return False, False, False
        return (
            bool(self.config.use_calendar_constraint),
            bool(self.config.use_butterfly_constraint),
            bool(self.config.use_smooth_constraint),
        )

    @staticmethod
    def _allowed_monitor_metrics() -> set[str]:
        return {
            "val_recon",
            "val_current_recon",
            "val_baseline_gap",
            "val_hybrid_score",
            "val_calendar",
            "val_butterfly",
            "val_delta_shrink",
        }

    def _run_epoch(self, loader, *, train: bool, epoch: int) -> Dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None

        mode_ctx = torch.enable_grad() if train else torch.no_grad()
        if train:
            self.model.train()
        else:
            self.model.eval()

        metric_numerators: Dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "calendar": 0.0,
            "butterfly": 0.0,
            "smooth": 0.0,
            "delta_shrink": 0.0,
            "current_recon": 0.0,
        }
        metric_denominator = 0.0
        use_calendar_constraint, use_butterfly_constraint, use_smooth_constraint = (
            self._constraint_switches_for_epoch(epoch)
        )

        with mode_ctx:
            for raw_batch in loader:
                batch = unpack_vol_training_batch(raw_batch)
                current_surface = self._to_device(batch.current_surface)
                text_embedding = self._to_device(batch.text_embedding)
                real_future = self._to_device(batch.target_surface)
                batch_size = int(current_surface.shape[0])
                sample_weight = validated_sample_weights(
                    batch.sample_weight,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=current_surface.dtype,
                )
                label_reliability_weight = validated_label_reliability_weights(
                    batch.label_reliability_weight,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=current_surface.dtype,
                    require_ones=not train,
                )
                recon_weight = reconstruction_training_weights(
                    sample_weight,
                    label_reliability_weight,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=current_surface.dtype,
                    require_label_ones=not train,
                )
                support_mask = validated_surface_mask(
                    batch.support_mask,
                    reference_surface=real_future,
                )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                predicted_future = self.model(current_surface, text_embedding)
                per_sample = {
                    "recon": masked_mean_per_sample(
                        torch.abs(predicted_future - real_future), support_mask
                    ),
                    "calendar": self._calendar_penalty_per_sample(
                        predicted_future, support_mask
                    ),
                    "butterfly": self._butterfly_penalty_per_sample(
                        predicted_future, support_mask
                    ),
                    "smooth": self._smoothness_penalty_per_sample(
                        predicted_future, support_mask
                    ),
                    "delta_shrink": self._delta_shrink_per_sample(
                        predicted_future,
                        current_surface,
                        support_mask,
                    ),
                    "current_recon": masked_mean_per_sample(
                        torch.abs(current_surface - real_future), support_mask
                    ),
                }
                reducer = training_weighted_mean if train else weighted_mean
                reduced = {
                    name: reducer(
                        values,
                        recon_weight if name == "recon" else sample_weight,
                    )
                    for name, values in per_sample.items()
                }

                total_loss = float(self.config.lambda_recon) * reduced["recon"]
                if use_calendar_constraint:
                    total_loss = (
                        total_loss
                        + float(self.config.lambda_calendar) * reduced["calendar"]
                    )
                if use_butterfly_constraint:
                    total_loss = (
                        total_loss
                        + float(self.config.lambda_butterfly) * reduced["butterfly"]
                    )
                if use_smooth_constraint:
                    total_loss = (
                        total_loss
                        + float(self.config.lambda_smooth) * reduced["smooth"]
                    )
                if float(getattr(self.config, "lambda_delta_shrink", 0.0)) > 0.0:
                    total_loss = (
                        total_loss
                        + float(self.config.lambda_delta_shrink)
                        * reduced["delta_shrink"]
                    )

                if train:
                    total_loss.backward()
                    self.optimizer.step()

                batch_numerators = {
                    name: torch.sum(
                        values.detach()
                        * (recon_weight if name == "recon" else sample_weight)
                    )
                    for name, values in per_sample.items()
                }
                total_numerator = (
                    float(self.config.lambda_recon) * batch_numerators["recon"]
                )
                if use_calendar_constraint:
                    total_numerator = total_numerator + float(
                        self.config.lambda_calendar
                    ) * batch_numerators["calendar"]
                if use_butterfly_constraint:
                    total_numerator = total_numerator + float(
                        self.config.lambda_butterfly
                    ) * batch_numerators["butterfly"]
                if use_smooth_constraint:
                    total_numerator = total_numerator + float(
                        self.config.lambda_smooth
                    ) * batch_numerators["smooth"]
                if float(getattr(self.config, "lambda_delta_shrink", 0.0)) > 0.0:
                    total_numerator = total_numerator + float(
                        self.config.lambda_delta_shrink
                    ) * batch_numerators["delta_shrink"]
                metric_numerators["total"] += float(total_numerator.cpu())
                for name, numerator in batch_numerators.items():
                    metric_numerators[name] += float(numerator.cpu())
                metric_denominator += (
                    float(batch_size)
                    if train
                    else float(torch.sum(sample_weight).cpu())
                )

        prefix = "train" if train else "val"
        metrics = {
            f"{prefix}_{name}": numerator / metric_denominator
            if metric_denominator > 0.0
            else 0.0
            for name, numerator in metric_numerators.items()
            if name != "current_recon"
        }
        if not train:
            val_recon = metrics["val_recon"]
            val_current_recon = (
                metric_numerators["current_recon"] / metric_denominator
                if metric_denominator > 0.0
                else 0.0
            )
            val_baseline_gap = val_recon - val_current_recon
            metrics.update(
                {
                    "val_current_recon": val_current_recon,
                    "val_baseline_gap": val_baseline_gap,
                    "val_hybrid_score": val_recon
                    + float(self.config.baseline_penalty_weight)
                    * max(0.0, val_baseline_gap),
                }
            )
        return metrics

    def _init_metrics_file(self) -> None:
        self._metrics_file = os.path.join(
            self.config.metrics_path, "training_metrics.json"
        )
        self._metrics_csv_file = os.path.join(
            self.config.metrics_path, "training_metrics.csv"
        )
        self._best_checkpoint_file = os.path.join(
            self.config.metrics_path, "best_checkpoint.json"
        )
        self._initial_checkpoint_file = os.path.join(
            self.config.metrics_path,
            "initial_checkpoint.json",
        )
        self._best_learned_checkpoint_file = os.path.join(
            self.config.metrics_path,
            "best_learned_checkpoint.json",
        )
        self._metrics_rows = []
        write_metrics_json([], self._metrics_file)
        for stale_path in (
            self._metrics_csv_file,
            self._best_checkpoint_file,
            self._initial_checkpoint_file,
            self._best_learned_checkpoint_file,
            os.path.join(self.config.models_path, "vol_regressor_best.pt"),
            os.path.join(
                self.config.models_path,
                "vol_regressor_initial_epoch0.pt",
            ),
            os.path.join(
                self.config.models_path,
                "vol_regressor_best_learned.pt",
            ),
        ):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    def _append_metrics_row(self, row) -> None:
        self._metrics_rows.append(row)
        self._write_metrics(self._metrics_rows)

    def _write_metrics(self, metrics_rows) -> None:
        write_metrics_json(metrics_rows, self._metrics_file)
        write_metrics_csv(metrics_rows, self._metrics_csv_file)

    def save_model(
        self, epoch: Optional[int] = None, *, label: Optional[str] = None
    ) -> Dict[str, str]:
        assert self.model is not None
        assert self.bundle is not None
        os.makedirs(self.config.models_path, exist_ok=True)
        if epoch is not None and label is not None:
            raise ValueError(
                "Specify either epoch or label when saving a model, not both."
            )
        suffix = ""
        if label:
            suffix = f"_{label}"
        elif epoch is not None:
            suffix = f"_epoch_{epoch:04d}"
        model_path = os.path.join(self.config.models_path, f"vol_regressor{suffix}.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.bundle.embedding_dim,
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                "residual_output_mode": self.model.residual_output_mode,
                "residual_output_fingerprint": self.model.residual_output_fingerprint,
                **self._label_reliability_lineage(),
                **self._learning_rate_contract(),
            },
            model_path,
        )
        return {"model": model_path}

    def _save_best_validation_checkpoint(
        self,
        *,
        epoch: int,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist one validation-selected checkpoint, including epoch zero."""

        artifact_paths = self.save_model(label="best")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "best_epoch": int(epoch),
                "best_metric": current_metric,
                "selection_scope": "baseline_inclusive",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._label_reliability_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._best_checkpoint_file,
        )

    def _save_initial_validation_checkpoint(
        self,
        *,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist the immutable pre-training validation checkpoint."""

        artifact_paths = self.save_model(label="initial_epoch0")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "initial_epoch": 0,
                "best_epoch": 0,
                "best_metric": current_metric,
                "selection_scope": "initial_epoch0",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._label_reliability_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._initial_checkpoint_file,
        )

    def _save_best_learned_validation_checkpoint(
        self,
        *,
        epoch: int,
        monitor_metric: str,
        current_metric: float,
        epoch_stats: dict,
    ) -> None:
        """Persist the best trained checkpoint, excluding epoch zero."""

        if epoch < 1:
            raise ValueError("best learned checkpoint requires epoch >= 1")
        artifact_paths = self.save_model(label="best_learned")
        write_best_checkpoint(
            {
                "monitor_metric": monitor_metric,
                "best_epoch": int(epoch),
                "best_learned_epoch_ge_1": int(epoch),
                "best_metric": current_metric,
                "selection_scope": "trained_epochs_only",
                "capacity_profile": self.config.news_first_capacity_profile,
                "capacity_profile_sha256": (
                    self.config.news_first_capacity_profile_sha256
                ),
                **self._label_reliability_lineage(),
                **self._learning_rate_contract(),
                "metrics": {
                    "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                    "val_current_recon": float(
                        epoch_stats.get("val_current_recon", 0.0)
                    ),
                    "val_baseline_gap": float(epoch_stats.get("val_baseline_gap", 0.0)),
                    "val_hybrid_score": float(epoch_stats.get("val_hybrid_score", 0.0)),
                    "val_calendar": float(epoch_stats.get("val_calendar", 0.0)),
                    "val_butterfly": float(epoch_stats.get("val_butterfly", 0.0)),
                    "val_delta_shrink": float(epoch_stats.get("val_delta_shrink", 0.0)),
                },
                "artifacts": artifact_paths,
            },
            self._best_learned_checkpoint_file,
        )

    def _save_loss_curves(self, metrics_rows) -> None:
        output_path = os.path.join(self.config.metrics_path, "loss_curves.png")
        plot_training_curves(
            metrics_rows,
            title="Deterministic Vol Regression Loss Curves",
            metric_groups=(
                (
                    "Primary losses",
                    (
                        "train_total",
                        "val_total",
                        "train_recon",
                        "val_recon",
                        "val_current_recon",
                        "val_hybrid_score",
                    ),
                ),
                (
                    "Constraint losses",
                    (
                        "train_calendar",
                        "val_calendar",
                        "train_butterfly",
                        "val_butterfly",
                        "train_smooth",
                        "val_smooth",
                        "train_delta_shrink",
                        "val_delta_shrink",
                    ),
                ),
            ),
            output_path=output_path,
        )
        self.logger.info("Loss curve plot saved to: %s", output_path)

    def _train_impl(self) -> Path | None:
        assert self.bundle is not None
        assert self.optimizer is not None
        self._init_metrics_file()
        self._initial_learning_rate = self._optimizer_lr(self.optimizer)
        self.logger.info(
            "*** Start deterministic vol regression: %s epochs, batch_size=%s, lr=%s ***",
            self.config.num_epochs,
            self.config.batch_size,
            self._initial_learning_rate,
        )
        monitor_metric = (
            str(getattr(self.config, "best_checkpoint_metric", "val_recon")).strip()
            or "val_recon"
        )
        if monitor_metric not in self._allowed_monitor_metrics():
            raise ValueError(
                f"Unsupported best_checkpoint_metric='{monitor_metric}'. "
                f"Expected one of {sorted(self._allowed_monitor_metrics())}."
            )
        best_metric = None
        best_epoch = None
        best_learned_metric = None
        best_learned_epoch = None
        patience = max(1, int(self.config.early_stopping_patience))
        min_delta = float(self.config.early_stopping_min_delta)
        min_epochs = max(0, int(self.config.early_stopping_min_epochs))
        best_tracking_enabled = self.bundle.val_loader is not None
        early_stopping_enabled = bool(
            self.config.use_early_stopping and best_tracking_enabled
        )
        evaluate_initial_checkpoint = bool(
            getattr(self.config, "evaluate_initial_checkpoint", False)
        )
        scheduler: Optional[ReduceLROnPlateau] = None
        epochs_without_improvement = 0

        if not self.config.use_reduce_lr_on_plateau:
            self.logger.info("ReduceLROnPlateau is disabled.")
        elif not best_tracking_enabled:
            self.logger.info(
                "ReduceLROnPlateau requested but disabled because no validation split is available."
            )
        else:
            assert self.optimizer is not None
            scheduler = self._create_plateau_scheduler(self.optimizer)
            self.logger.info(
                "ReduceLROnPlateau enabled using %s (factor=%.3f, patience=%d, min_lr=%.6g).",
                monitor_metric,
                float(self.config.reduce_lr_factor),
                int(self.config.reduce_lr_patience),
                float(self.config.reduce_lr_min_lr),
            )

        if not best_tracking_enabled:
            self.logger.info(
                "Validation unavailable; best-checkpoint tracking disabled."
            )
            if evaluate_initial_checkpoint:
                self.logger.info(
                    "Initial-checkpoint evaluation requested but disabled because no validation split is available."
                )
        elif early_stopping_enabled:
            self.logger.info(
                "Best-checkpoint tracking enabled using %s. Early stopping active "
                "(patience=%d, min_delta=%.6f, min_epochs=%d).",
                monitor_metric,
                patience,
                min_delta,
                min_epochs,
            )
        else:
            self.logger.info(
                "Best-checkpoint tracking enabled using %s. Early stopping is disabled.",
                monitor_metric,
            )

        warmup_epochs = self._constraint_warmup_epochs()
        if warmup_epochs > 0:
            self.logger.info(
                "Constraint warmup active: enabled constraint losses will be skipped for the first %d epoch(s) and applied from epoch %d.",
                warmup_epochs,
                warmup_epochs + 1,
            )

        metrics_rows = []
        if best_tracking_enabled and evaluate_initial_checkpoint:
            initial_stats = {"epoch": 0}
            initial_stats.update(
                self._run_epoch(self.bundle.val_loader, train=False, epoch=0)
            )
            assert self.optimizer is not None
            initial_stats["lr"] = self._optimizer_lr(self.optimizer)
            metrics_rows.append(initial_stats)
            self._append_metrics_row(initial_stats)
            if monitor_metric in initial_stats:
                best_metric = resolve_monitor_metric(initial_stats, monitor_metric)
                best_epoch = 0
                epochs_without_improvement = 0
                self._save_initial_validation_checkpoint(
                    monitor_metric=monitor_metric,
                    current_metric=best_metric,
                    epoch_stats=initial_stats,
                )
                self._save_best_validation_checkpoint(
                    epoch=0,
                    monitor_metric=monitor_metric,
                    current_metric=best_metric,
                    epoch_stats=initial_stats,
                )
                self.logger.info(
                    "Initial validation checkpoint saved at epoch 0 with %s=%.6f",
                    monitor_metric,
                    best_metric,
                )
                assert self.optimizer is not None
                self._step_plateau_scheduler(
                    scheduler=scheduler,
                    optimizer=self.optimizer,
                    metric_name=monitor_metric,
                    metric_value=best_metric,
                )
            else:
                self.logger.warning(
                    "Initial-checkpoint evaluation produced no %s metric; epoch-0 best tracking was skipped.",
                    monitor_metric,
                )

        for epoch in range(1, int(self.config.num_epochs) + 1):
            epoch_stats = {"epoch": epoch}
            epoch_stats.update(
                self._run_epoch(self.bundle.train_loader, train=True, epoch=epoch)
            )
            if self.bundle.val_loader is not None:
                epoch_stats.update(
                    self._run_epoch(self.bundle.val_loader, train=False, epoch=epoch)
                )
            assert self.optimizer is not None
            epoch_stats["lr"] = self._optimizer_lr(self.optimizer)
            metrics_rows.append(epoch_stats)
            self._append_metrics_row(epoch_stats)

            val_info = ""
            if "val_recon" in epoch_stats:
                val_info = (
                    f" ValRecon={epoch_stats['val_recon']:.4f}"
                    f" Curr={epoch_stats.get('val_current_recon', 0.0):.4f}"
                    f" Hybrid={epoch_stats.get('val_hybrid_score', 0.0):.4f}"
                )

            self.logger.info(
                "[Epoch %04d/%04d] TrainTotal=%.4f TrainRecon=%.4f%s",
                epoch,
                self.config.num_epochs,
                epoch_stats.get("train_total", 0.0),
                epoch_stats.get("train_recon", 0.0),
                val_info,
            )

            if best_tracking_enabled and monitor_metric in epoch_stats:
                current_metric = resolve_monitor_metric(epoch_stats, monitor_metric)
                if best_learned_metric is None or current_metric < (
                    best_learned_metric - min_delta
                ):
                    best_learned_metric = current_metric
                    best_learned_epoch = epoch
                    self._save_best_learned_validation_checkpoint(
                        epoch=epoch,
                        monitor_metric=monitor_metric,
                        current_metric=current_metric,
                        epoch_stats=epoch_stats,
                    )
                    self.logger.info(
                        "New best learned checkpoint saved at epoch %d with %s=%.6f",
                        epoch,
                        monitor_metric,
                        current_metric,
                    )
                if best_metric is None or current_metric < (best_metric - min_delta):
                    best_metric = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_best_validation_checkpoint(
                        epoch=epoch,
                        monitor_metric=monitor_metric,
                        current_metric=current_metric,
                        epoch_stats=epoch_stats,
                    )
                    self.logger.info(
                        "New best checkpoint saved at epoch %d with %s=%.6f",
                        epoch,
                        monitor_metric,
                        current_metric,
                    )
                elif early_stopping_enabled:
                    epochs_without_improvement += 1
                    self.logger.info(
                        "Early stopping patience %d/%d without %s improvement (current=%.6f, best=%.6f at epoch %d)",
                        epochs_without_improvement,
                        patience,
                        monitor_metric,
                        current_metric,
                        best_metric,
                        best_epoch,
                    )

                assert self.optimizer is not None
                self._step_plateau_scheduler(
                    scheduler=scheduler,
                    optimizer=self.optimizer,
                    metric_name=monitor_metric,
                    metric_value=current_metric,
                )

            if epoch % int(self.config.save_every) == 0:
                self.save_model(epoch)
                self.logger.info("Checkpoint saved at epoch %d", epoch)

            if (
                early_stopping_enabled
                and epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                self.logger.info(
                    "Early stopping triggered at epoch %d. Best %s=%.6f at epoch %d; "
                    "best learned epoch=%s.",
                    epoch,
                    monitor_metric,
                    best_metric,
                    best_epoch,
                    best_learned_epoch,
                )
                break

        self.save_model()
        self._write_metrics(metrics_rows)
        try:
            self._save_loss_curves(metrics_rows)
        except Exception as exc:
            self.logger.warning("Failed to save loss curve plot: %s", exc)
        self.logger.info("*** Training complete ***")
        return self.run_dir

    def _prepare_generate_result(
        self,
        generate_config: GenerateResultConfig | None = None,
        *,
        overrides: Mapping[str, object] | None = None,
        config_path: str | None = None,
    ) -> tuple[GenerateResultConfig, Path]:
        if generate_config is None:
            base_config = self.config if self._runtime_prepared else self.raw_config
            generate_config = build_generate_result_config(
                training_values=config_to_dict(base_config),
                generate_values=_load_generate_result_section(config_path),
                overrides=overrides,
            )
        elif overrides:
            generate_config = replace(generate_config, **dict(overrides))

        base_training_config = (
            self.config if self._runtime_prepared else self.raw_config
        )
        output_root = infer_training_output_root(
            base_training_config, trainer_id=self.trainer_id
        )
        run_dir = self.run_dir or resolve_existing_run_dir(
            output_root=output_root,
            checkpoint_path=generate_config.checkpoint_path or None,
        )
        runtime_config = replace(
            generate_config,
            models_path=str(run_dir),
            metrics_path=str(run_dir),
        )
        resolved_checkpoint_path = resolve_checkpoint_path(
            runtime_config,
            artifact_key="model",
            fallback_filenames=("vol_regressor_best.pt", "vol_regressor.pt"),
        )
        if str(generate_config.output_dir).strip():
            resolved_generate_dir = generate_result_dir(
                run_dir, generate_config.output_dir
            )
        else:
            resolved_generate_dir = checkpoint_named_dir(
                generate_result_dir(run_dir), resolved_checkpoint_path
            )
        resolved_config = replace(
            runtime_config,
            checkpoint_path=str(resolved_checkpoint_path),
            output_dir=str(resolved_generate_dir),
        )
        validate_result_config(resolved_config)
        return resolved_config, resolved_generate_dir

    def _generate_result_impl(
        self, generate_config: GenerateResultConfig, generate_dir: Path
    ) -> Path:
        del generate_dir
        return generate_vol_regression_result(generate_config)


def main(config: Optional[Config] = None) -> None:
    trainer = VolSurfaceRegressionTrainer(config or default_config)
    trainer.train()


if __name__ == "__main__":
    main()
