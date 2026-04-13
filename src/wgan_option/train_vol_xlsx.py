"""Training wrapper for merged vol-surface xlsx datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from wgan_option.config import Config, default_config
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.trainer import WGANTrainer
from wgan_option.utils.merged_xlsx import create_vol_surface_xlsx_dataloaders
from wgan_option.utils.training_run_paths import prepare_timestamped_training_config


class VolSurfaceXlsxTrainer(WGANTrainer):
    """Run the existing WGAN-GP on merged vol-surface workbook rows."""

    trainer_id = "cnn_wgan"

    def _prepare_runtime_config(self, config: Config) -> tuple[Config, Path]:
        return prepare_timestamped_training_config(config, trainer_id=self.trainer_id)

    def setup(self):
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

        self.logger.info("*** Initializing model ***")
        self.model = WGAN_GP(
            config=self.config,
            strike_grid=self.bundle.strike_grid,
            maturity_grid_days=self.bundle.maturity_grid_days,
            embedding_dim=self.bundle.embedding_dim,
        )
        total_params = sum(p.numel() for p in self.model.G.parameters()) + sum(p.numel() for p in self.model.D.parameters())
        self.logger.info("Model initialized: G params=%s, D params=%s, total=%s",
                         sum(p.numel() for p in self.model.G.parameters()),
                         sum(p.numel() for p in self.model.D.parameters()),
                         total_params)


def main(config: Optional[Config] = None):
    trainer = VolSurfaceXlsxTrainer(config or default_config)
    trainer.train()


if __name__ == "__main__":
    main()
