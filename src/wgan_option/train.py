"""Training orchestrator for conditional WGAN option-surface forecasting."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from wgan_option.config import Config, default_config, save_config_yaml
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.utils.dataloader import ForecastDataBundle, create_bond_option_forecast_dataloaders


class WGANTrainer:
    """High-level trainer wrapper around dataloading + model lifecycle."""

    def __init__(self, config: Config):
        self.config = config
        self._logger: Optional[logging.Logger] = None
        self.bundle: Optional[ForecastDataBundle] = None
        self.model: Optional[WGAN_GP] = None

    @property
    def logger(self) -> logging.Logger:
        """Lazily initialize a stdout logger with consistent format."""
        if self._logger is None:
            logger = logging.getLogger("wgan_option.trainer")
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
        """Save resolved runtime config under metrics directory."""
        os.makedirs(self.config.metrics_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.config.metrics_path, f"run_config_{timestamp}.yaml")
        save_config_yaml(self.config, output_path)
        self.logger.info(f"Resolved config saved to: {output_path}")

    def setup(self):
        """Prepare dataset bundle and model instance before training."""
        self.logger.info("*** Load dataset ***")
        self.bundle = create_bond_option_forecast_dataloaders(self.config)
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

    def start_train(self):
        """Execute full training run."""
        self.setup()
        self._save_run_config()
        self.logger.info("*** Start training ***")
        assert self.model is not None
        assert self.bundle is not None
        self.model.train(self.bundle.train_loader, self.bundle.val_loader)
        self.logger.info("*** Training complete ***")

    def dry_run(self):
        """Run setup and config snapshot without fitting the model."""
        self.setup()
        self._save_run_config()
        self.logger.info("Dry run finished. Training was not started.")


def main(config: Optional[Config] = None):
    """Convenience main for programmatic usage."""
    trainer = WGANTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
