"""Training wrapper for merged SVI workbook rows."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wgan_option.config import Config, default_config, save_config_yaml
from wgan_option.models.svi_regressor import SviRegressor
from wgan_option.utils.merged_xlsx import SVI_FEATURE_ORDER, SviXlsxBundle, create_svi_xlsx_dataloaders
from wgan_option.utils.training_artifacts import write_best_checkpoint, write_metrics_csv, write_metrics_json
from wgan_option.utils.training_run_paths import prepare_timestamped_training_config
from wgan_option.utils.visualization import plot_training_curves


class SviXlsxTrainer:
    """Train a supervised MLP regressor on paired SVI samples."""

    def __init__(self, config: Config):
        self.run_dir: Optional[Path] = None
        self.config, self.run_dir = prepare_timestamped_training_config(
            config,
            include_normalization_stats=True,
        )
        self._logger: Optional[logging.Logger] = None
        self.bundle: Optional[SviXlsxBundle] = None
        self.model: Optional[SviRegressor] = None
        self.optimizer: Optional[Adam] = None
        self.device = torch.device("cuda:0" if (self.config.cuda and torch.cuda.is_available()) else "cpu")

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger("wgan_option.svi_trainer")
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

    def _save_run_config(self):
        os.makedirs(self.config.metrics_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.config.metrics_path, f"run_config_{timestamp}.yaml")
        save_config_yaml(self.config, output_path)
        self.logger.info("Resolved config saved to: %s", output_path)

    def _save_normalization_stats(self):
        assert self.bundle is not None
        output_path = self.config.normalization_stats_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            **self.bundle.normalization_stats,
            "max_slices": int(self.bundle.max_slices),
            "feature_dim": int(len(SVI_FEATURE_ORDER)),
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.logger.info("Normalization stats saved to: %s", output_path)

    def _log_device_info(self):
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = getattr(device_props, "total_memory", 0) / 1024**3
            self.logger.info("CUDA available: %s (%.1f GB)", torch.cuda.get_device_name(0), total_memory_gb)
        else:
            self.logger.info("CUDA not available, using CPU")
        self.logger.info("Device: %s", self.device)

    def setup(self):
        self._log_device_info()

        self.logger.info("*** Loading merged SVI dataset ***")
        self.logger.info("Data source: %s (sheet: %s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_svi_xlsx_dataloaders(self.config)
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, input_dim=%s, regression_dim=%s, embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            self.bundle.current_input_dim,
            self.bundle.regression_dim,
            self.bundle.embedding_dim,
        )

        self.logger.info("*** Initializing SVI regressor ***")
        self.model = SviRegressor(
            current_input_dim=self.bundle.current_input_dim,
            embedding_dim=self.bundle.embedding_dim,
            regression_dim=self.bundle.regression_dim,
            count_classes=self.bundle.max_slices,
            hidden_dim=self.config.svi_hidden_dim,
            dropout=self.config.svi_dropout,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Model initialized: %s parameters", total_params)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
        )
        self._save_normalization_stats()

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

    def _masked_smooth_l1(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        feature_dim = predicted.shape[1] // int(self.config.max_slices)
        expanded_mask = future_mask.unsqueeze(-1).expand(-1, -1, feature_dim).reshape(predicted.shape[0], -1)
        if torch.sum(expanded_mask) <= 0:
            return torch.zeros(1, device=predicted.device).squeeze()
        loss_matrix = F.smooth_l1_loss(predicted, target, reduction="none")
        masked_loss = loss_matrix * expanded_mask
        return masked_loss.sum() / expanded_mask.sum()

    def _run_epoch(self, loader, *, train: bool) -> Dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None

        mode_ctx = torch.enable_grad() if train else torch.no_grad()
        if train:
            self.model.train()
        else:
            self.model.eval()

        running: Dict[str, list] = {"total": [], "regression": [], "count": []}
        with mode_ctx:
            for current_features, text_embedding, future_regression, future_mask, future_count in loader:
                current_features = self._to_device(current_features)
                text_embedding = self._to_device(text_embedding)
                future_regression = self._to_device(future_regression)
                future_mask = self._to_device(future_mask)
                future_count = self._to_device(future_count)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                predicted_regression, predicted_count_logits = self.model(current_features, text_embedding)
                regression_loss = self._masked_smooth_l1(predicted_regression, future_regression, future_mask)
                count_loss = F.cross_entropy(predicted_count_logits, future_count)
                total_loss = regression_loss + float(self.config.count_loss_weight) * count_loss

                if train:
                    total_loss.backward()
                    self.optimizer.step()

                running["total"].append(float(total_loss.detach().cpu()))
                running["regression"].append(float(regression_loss.detach().cpu()))
                running["count"].append(float(count_loss.detach().cpu()))

        prefix = "train" if train else "val"
        return {
            f"{prefix}_total": float(np.mean(running["total"])) if running["total"] else 0.0,
            f"{prefix}_regression": float(np.mean(running["regression"])) if running["regression"] else 0.0,
            f"{prefix}_count": float(np.mean(running["count"])) if running["count"] else 0.0,
        }

    def _init_metrics_file(self):
        self._metrics_file = os.path.join(self.config.metrics_path, "training_metrics.json")
        self._metrics_csv_file = os.path.join(self.config.metrics_path, "training_metrics.csv")
        self._best_checkpoint_file = os.path.join(self.config.metrics_path, "best_checkpoint.json")
        self._metrics_rows = []
        write_metrics_json([], self._metrics_file)
        for stale_path in (
            self._metrics_csv_file,
            self._best_checkpoint_file,
            os.path.join(self.config.models_path, "svi_regressor_best.pt"),
        ):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    def _append_metrics_row(self, row):
        self._metrics_rows.append(row)
        self._write_metrics(self._metrics_rows)

    def _write_metrics(self, metrics_rows):
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
        model_path = os.path.join(self.config.models_path, f"svi_regressor{suffix}.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "embedding_dim": self.bundle.embedding_dim,
                "current_input_dim": self.bundle.current_input_dim,
                "regression_dim": self.bundle.regression_dim,
                "max_slices": self.bundle.max_slices,
                "normalization_stats": self.bundle.normalization_stats,
            },
            model_path,
        )
        return {
            "model": model_path,
        }

    def _save_loss_curves(self, metrics_rows) -> None:
        output_path = os.path.join(self.config.metrics_path, "loss_curves.png")
        plot_training_curves(
            metrics_rows,
            title="SVI Training Loss Curves",
            metric_groups=(
                ("Primary losses", ("train_total", "val_total", "train_regression", "val_regression")),
                ("Count losses", ("train_count", "val_count")),
            ),
            output_path=output_path,
        )
        self.logger.info("Loss curve plot saved to: %s", output_path)

    def start_train(self):
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self.setup()
        self._save_run_config()
        assert self.bundle is not None
        metrics_rows = []
        self._init_metrics_file()
        self.logger.info("*** Start SVI training: %s epochs, batch_size=%s, lr=%s ***",
                         self.config.num_epochs, self.config.batch_size, self.config.learning_rate)
        monitor_metric = "val_regression"
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
            if self.config.use_early_stopping:
                self.logger.info("Early stopping requested but disabled because no validation split is available.")
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

        for epoch in range(1, int(self.config.num_epochs) + 1):
            epoch_stats = {"epoch": epoch}
            epoch_stats.update(self._run_epoch(self.bundle.train_loader, train=True))
            if self.bundle.val_loader is not None:
                epoch_stats.update(self._run_epoch(self.bundle.val_loader, train=False))
            assert self.optimizer is not None
            epoch_stats["lr"] = self._optimizer_lr(self.optimizer)
            metrics_rows.append(epoch_stats)
            self._append_metrics_row(epoch_stats)

            val_info = ""
            if "val_total" in epoch_stats:
                val_info = f" ValTotal={epoch_stats['val_total']:.4f} ValReg={epoch_stats.get('val_regression', 0.0):.4f}"

            self.logger.info(
                "[Epoch %04d/%04d] TrainTotal=%.4f TrainReg=%.4f TrainCount=%.4f%s",
                epoch,
                int(self.config.num_epochs),
                epoch_stats.get("train_total", 0.0),
                epoch_stats.get("train_regression", 0.0),
                epoch_stats.get("train_count", 0.0),
                val_info,
            )

            if best_tracking_enabled and monitor_metric in epoch_stats:
                current_metric = float(epoch_stats[monitor_metric])
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
        self.logger.info("*** SVI training complete ***")

    def dry_run(self):
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self.setup()
        self._save_run_config()
        self.logger.info("Dry run finished. Training was not started.")


def main(config: Optional[Config] = None):
    trainer = SviXlsxTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
