"""Training wrapper for merged SVI workbook rows."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from wgan_option.config import Config, default_config, save_config_yaml
from wgan_option.models.svi_regressor import SviRegressor
from wgan_option.utils.merged_xlsx import SVI_FEATURE_ORDER, SviXlsxBundle, create_svi_xlsx_dataloaders
from wgan_option.utils.visualization import plot_training_curves


class SviXlsxTrainer:
    """Train a supervised MLP regressor on paired SVI samples."""

    def __init__(self, config: Config):
        self.config = config
        self._logger: Optional[logging.Logger] = None
        self.bundle: Optional[SviXlsxBundle] = None
        self.model: Optional[SviRegressor] = None
        self.optimizer: Optional[Adam] = None
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

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
        self._metrics_rows = []
        with open(self._metrics_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    def _append_metrics_row(self, row):
        self._metrics_rows.append(row)
        with open(self._metrics_file, "w", encoding="utf-8") as f:
            json.dump(self._metrics_rows, f, indent=2, ensure_ascii=False)

    def _write_metrics(self, metrics_rows):
        output_file = os.path.join(self.config.metrics_path, "training_metrics.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, indent=2, ensure_ascii=False)

    def save_model(self, epoch: Optional[int] = None):
        assert self.model is not None
        assert self.bundle is not None
        os.makedirs(self.config.models_path, exist_ok=True)
        suffix = f"_epoch_{epoch:04d}" if epoch is not None else ""
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
        self.setup()
        self._save_run_config()
        assert self.bundle is not None
        metrics_rows = []
        self._init_metrics_file()
        self.logger.info("*** Start SVI training: %s epochs, batch_size=%s, lr=%s ***",
                         self.config.num_epochs, self.config.batch_size, self.config.learning_rate)
        for epoch in range(1, int(self.config.num_epochs) + 1):
            epoch_stats = {"epoch": epoch}
            epoch_stats.update(self._run_epoch(self.bundle.train_loader, train=True))
            if self.bundle.val_loader is not None:
                epoch_stats.update(self._run_epoch(self.bundle.val_loader, train=False))
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

            if epoch % int(self.config.save_every) == 0:
                self.save_model(epoch)
                self.logger.info("Checkpoint saved at epoch %d", epoch)

        self.save_model()
        self._write_metrics(metrics_rows)
        try:
            self._save_loss_curves(metrics_rows)
        except Exception as exc:
            self.logger.warning("Failed to save loss curve plot: %s", exc)
        self.logger.info("*** SVI training complete ***")

    def dry_run(self):
        self.setup()
        self._save_run_config()
        self.logger.info("Dry run finished. Training was not started.")


def main(config: Optional[Config] = None):
    trainer = SviXlsxTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
