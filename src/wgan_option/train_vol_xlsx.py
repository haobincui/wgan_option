"""Training wrapper for merged vol-surface xlsx datasets."""

from __future__ import annotations

from typing import Optional

from wgan_option.config import Config, default_config
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.train import WGANTrainer
from wgan_option.utils.merged_xlsx import create_vol_surface_xlsx_dataloaders


class VolSurfaceXlsxTrainer(WGANTrainer):
    """Run the existing WGAN-GP on merged vol-surface workbook rows."""

    def setup(self):
        self.logger.info("*** Load merged vol-surface dataset ***")
        self.bundle = create_vol_surface_xlsx_dataloaders(self.config)
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, surface_shape=(%s, %s), embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            len(self.bundle.maturity_grid_days),
            len(self.bundle.strike_grid),
            self.bundle.embedding_dim,
        )

        self.logger.info("*** Initialize model ***")
        self.model = WGAN_GP(
            config=self.config,
            strike_grid=self.bundle.strike_grid,
            maturity_grid_days=self.bundle.maturity_grid_days,
            embedding_dim=self.bundle.embedding_dim,
        )


def main(config: Optional[Config] = None):
    trainer = VolSurfaceXlsxTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
