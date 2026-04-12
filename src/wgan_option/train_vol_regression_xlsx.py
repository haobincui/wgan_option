"""Training wrapper for deterministic merged vol-surface regression."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wgan_option.config import Config, default_config, save_config_yaml
from wgan_option.models.vol_regressor import VolSurfaceRegressor
from wgan_option.utils.merged_xlsx import VolSurfaceXlsxBundle, create_vol_surface_xlsx_dataloaders
from wgan_option.utils.training_artifacts import write_best_checkpoint, write_metrics_csv, write_metrics_json
from wgan_option.utils.training_run_paths import prepare_timestamped_training_config, training_run_config_path
from wgan_option.utils.visualization import plot_training_curves
from wgan_option.utils.vol_forecast_metrics import resolve_monitor_metric, summarize_baseline_aware_metrics


class VolSurfaceRegressionTrainer:
    """Train a deterministic residual forecaster on merged vol workbook rows."""

    def __init__(self, config: Config):
        self.run_dir: Optional[Path] = None
        self.config, self.run_dir = prepare_timestamped_training_config(config)
        self._logger: Optional[logging.Logger] = None
        self.bundle: Optional[VolSurfaceXlsxBundle] = None
        self.model: Optional[VolSurfaceRegressor] = None
        self.optimizer: Optional[Adam] = None
        self.device = torch.device("cuda:0" if (self.config.cuda and torch.cuda.is_available()) else "cpu")
        self.strike_grid: Optional[torch.Tensor] = None
        self.tau_years: Optional[torch.Tensor] = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger("wgan_option.vol_regression")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(stream_handler)
            self._logger = logger
        return self._logger

    def _save_run_config(self) -> None:
        output_path = training_run_config_path(self.config, self.run_dir)
        save_config_yaml(self.config, str(output_path))
        self.logger.info("Resolved config saved to: %s", output_path)

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
            self.logger.info("CUDA available: %s (%.1f GB)", torch.cuda.get_device_name(0), total_memory_gb)
        else:
            self.logger.info("CUDA not available, using CPU")
        self.logger.info("Device: %s", self.device)

    def setup(self) -> None:
        self._log_device_info()

        self.logger.info("*** Loading merged vol-surface dataset ***")
        self.logger.info("Data source: %s (sheet: %s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_vol_surface_xlsx_dataloaders(self.config)
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, surface_shape=(%s, %s), embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            len(self.bundle.maturity_grid_days),
            len(self.bundle.strike_grid),
            self.bundle.embedding_dim,
        )

        self.strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device).clamp_min(1e-4)
        maturity_grid_days = torch.tensor(self.bundle.maturity_grid_days, dtype=torch.float32, device=self.device)
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

    def calendar_arbitrage_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        assert self.tau_years is not None
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(1) < 2:
            return torch.zeros(1, device=self.device).squeeze()
        total_variance = sigma.pow(2) * self.tau_years.view(1, -1, 1)
        diff = total_variance[:, 1:, :] - total_variance[:, :-1, :]
        return F.relu(-diff).mean()

    def butterfly_arbitrage_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        sigma = generated_surface.squeeze(1).clamp_min(1e-4)
        if sigma.size(2) < 3:
            return torch.zeros(1, device=self.device).squeeze()
        call_prices = self._black_call_price(sigma)
        second_diff = call_prices[:, :, 2:] - 2.0 * call_prices[:, :, 1:-1] + call_prices[:, :, :-2]
        return F.relu(-second_diff).mean()

    def smoothness_penalty(self, generated_surface: torch.Tensor) -> torch.Tensor:
        sigma = generated_surface.squeeze(1)
        penalty = torch.zeros(1, device=self.device).squeeze()
        if sigma.size(1) > 1:
            penalty = penalty + (sigma[:, 1:, :] - sigma[:, :-1, :]).pow(2).mean()
        if sigma.size(2) > 1:
            penalty = penalty + (sigma[:, :, 1:] - sigma[:, :, :-1]).pow(2).mean()
        return penalty

    @staticmethod
    def delta_shrink_penalty(predicted_surface: torch.Tensor, current_surface: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(predicted_surface - current_surface))

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

        recon_values: list[float] = []
        current_recon_values: list[float] = []
        running: Dict[str, list[float]] = {
            "total": [],
            "recon": [],
            "calendar": [],
            "butterfly": [],
            "smooth": [],
            "delta_shrink": [],
        }
        use_calendar_constraint, use_butterfly_constraint, use_smooth_constraint = self._constraint_switches_for_epoch(epoch)

        with mode_ctx:
            for current_surface, text_embedding, real_future in loader:
                current_surface = self._to_device(current_surface)
                text_embedding = self._to_device(text_embedding)
                real_future = self._to_device(real_future)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                predicted_future = self.model(current_surface, text_embedding)
                recon_loss = F.l1_loss(predicted_future, real_future)
                calendar_penalty = self.calendar_arbitrage_penalty(predicted_future)
                butterfly_penalty = self.butterfly_arbitrage_penalty(predicted_future)
                smooth_penalty = self.smoothness_penalty(predicted_future)
                delta_shrink = self.delta_shrink_penalty(predicted_future, current_surface)

                total_loss = float(self.config.lambda_recon) * recon_loss
                if use_calendar_constraint:
                    total_loss = total_loss + float(self.config.lambda_calendar) * calendar_penalty
                if use_butterfly_constraint:
                    total_loss = total_loss + float(self.config.lambda_butterfly) * butterfly_penalty
                if use_smooth_constraint:
                    total_loss = total_loss + float(self.config.lambda_smooth) * smooth_penalty
                if float(getattr(self.config, "lambda_delta_shrink", 0.0)) > 0.0:
                    total_loss = total_loss + float(self.config.lambda_delta_shrink) * delta_shrink

                if train:
                    total_loss.backward()
                    self.optimizer.step()

                running["total"].append(float(total_loss.detach().cpu()))
                running["recon"].append(float(recon_loss.detach().cpu()))
                running["calendar"].append(float(calendar_penalty.detach().cpu()))
                running["butterfly"].append(float(butterfly_penalty.detach().cpu()))
                running["smooth"].append(float(smooth_penalty.detach().cpu()))
                running["delta_shrink"].append(float(delta_shrink.detach().cpu()))
                recon_values.append(float(recon_loss.detach().cpu()))
                if not train:
                    current_recon_values.append(float(F.l1_loss(current_surface, real_future).detach().cpu()))

        prefix = "train" if train else "val"
        metrics = {
            f"{prefix}_total": float(np.mean(running["total"])) if running["total"] else 0.0,
            f"{prefix}_recon": float(np.mean(running["recon"])) if running["recon"] else 0.0,
            f"{prefix}_calendar": float(np.mean(running["calendar"])) if running["calendar"] else 0.0,
            f"{prefix}_butterfly": float(np.mean(running["butterfly"])) if running["butterfly"] else 0.0,
            f"{prefix}_smooth": float(np.mean(running["smooth"])) if running["smooth"] else 0.0,
            f"{prefix}_delta_shrink": float(np.mean(running["delta_shrink"])) if running["delta_shrink"] else 0.0,
        }
        if not train:
            metrics.update(
                summarize_baseline_aware_metrics(
                    recon_values,
                    current_recon_values,
                    baseline_penalty_weight=float(self.config.baseline_penalty_weight),
                )
            )
        return metrics

    def _init_metrics_file(self) -> None:
        self._metrics_file = os.path.join(self.config.metrics_path, "training_metrics.json")
        self._metrics_csv_file = os.path.join(self.config.metrics_path, "training_metrics.csv")
        self._best_checkpoint_file = os.path.join(self.config.metrics_path, "best_checkpoint.json")
        self._metrics_rows = []
        write_metrics_json([], self._metrics_file)
        for stale_path in (
            self._metrics_csv_file,
            self._best_checkpoint_file,
            os.path.join(self.config.models_path, "vol_regressor_best.pt"),
        ):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    def _append_metrics_row(self, row) -> None:
        self._metrics_rows.append(row)
        self._write_metrics(self._metrics_rows)

    def _write_metrics(self, metrics_rows) -> None:
        write_metrics_json(metrics_rows, self._metrics_file)
        write_metrics_csv(metrics_rows, self._metrics_csv_file)

    def save_model(self, epoch: Optional[int] = None, *, label: Optional[str] = None) -> Dict[str, str]:
        assert self.model is not None
        assert self.bundle is not None
        os.makedirs(self.config.models_path, exist_ok=True)
        if epoch is not None and label is not None:
            raise ValueError("Specify either epoch or label when saving a model, not both.")
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
            },
            model_path,
        )
        return {"model": model_path}

    def _save_loss_curves(self, metrics_rows) -> None:
        output_path = os.path.join(self.config.metrics_path, "loss_curves.png")
        plot_training_curves(
            metrics_rows,
            title="Deterministic Vol Regression Loss Curves",
            metric_groups=(
                (
                    "Primary losses",
                    ("train_total", "val_total", "train_recon", "val_recon", "val_current_recon", "val_hybrid_score"),
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

    def start_train(self) -> None:
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self._ensure_samples_dir()
        self.setup()
        self._save_run_config()
        assert self.bundle is not None
        self._init_metrics_file()
        self.logger.info(
            "*** Start deterministic vol regression: %s epochs, batch_size=%s, lr=%s ***",
            self.config.num_epochs,
            self.config.batch_size,
            self.config.learning_rate,
        )
        monitor_metric = str(getattr(self.config, "best_checkpoint_metric", "val_recon")).strip() or "val_recon"
        if monitor_metric not in self._allowed_monitor_metrics():
            raise ValueError(
                f"Unsupported best_checkpoint_metric='{monitor_metric}'. "
                f"Expected one of {sorted(self._allowed_monitor_metrics())}."
            )
        best_metric = None
        best_epoch = None
        patience = max(1, int(self.config.early_stopping_patience))
        min_delta = float(self.config.early_stopping_min_delta)
        best_tracking_enabled = self.bundle.val_loader is not None
        early_stopping_enabled = bool(self.config.use_early_stopping and best_tracking_enabled)
        scheduler: Optional[ReduceLROnPlateau] = None
        epochs_without_improvement = 0

        if not self.config.use_reduce_lr_on_plateau:
            self.logger.info("ReduceLROnPlateau is disabled.")
        elif not best_tracking_enabled:
            self.logger.info("ReduceLROnPlateau requested but disabled because no validation split is available.")
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
            self.logger.info("Validation unavailable; best-checkpoint tracking disabled.")
        elif early_stopping_enabled:
            self.logger.info(
                "Best-checkpoint tracking enabled using %s. Early stopping active (patience=%d, min_delta=%.6f).",
                monitor_metric,
                patience,
                min_delta,
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
        for epoch in range(1, int(self.config.num_epochs) + 1):
            epoch_stats = {"epoch": epoch}
            epoch_stats.update(self._run_epoch(self.bundle.train_loader, train=True, epoch=epoch))
            if self.bundle.val_loader is not None:
                epoch_stats.update(self._run_epoch(self.bundle.val_loader, train=False, epoch=epoch))
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
                if best_metric is None or current_metric < (best_metric - min_delta):
                    best_metric = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    artifact_paths = self.save_model(label="best")
                    write_best_checkpoint(
                        {
                            "monitor_metric": monitor_metric,
                            "best_epoch": int(epoch),
                            "best_metric": current_metric,
                            "metrics": {
                                "val_recon": float(epoch_stats.get("val_recon", 0.0)),
                                "val_current_recon": float(epoch_stats.get("val_current_recon", 0.0)),
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

            if early_stopping_enabled and epochs_without_improvement >= patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d. Best %s=%.6f at epoch %d.",
                    epoch,
                    monitor_metric,
                    best_metric,
                    best_epoch,
                )
                break

        self.save_model()
        self._write_metrics(metrics_rows)
        try:
            self._save_loss_curves(metrics_rows)
        except Exception as exc:
            self.logger.warning("Failed to save loss curve plot: %s", exc)
        self.logger.info("*** Training complete ***")

    def dry_run(self) -> None:
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self._ensure_samples_dir()
        self.setup()
        self._save_run_config()
        self.logger.info("Dry run finished. Training was not started.")


def main(config: Optional[Config] = None) -> None:
    trainer = VolSurfaceRegressionTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
