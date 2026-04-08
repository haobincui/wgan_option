"""Training orchestrator for conditional WGAN option-surface forecasting."""

import logging
import sys
from pathlib import Path
from typing import Optional

from wgan_option.config import Config, default_config, save_config_yaml
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.utils.dataloader import ForecastDataBundle, create_bond_option_forecast_dataloaders
from wgan_option.utils.training_run_paths import training_run_config_path


class WGANTrainer:
    """High-level trainer wrapper around dataloading + model lifecycle."""

    def __init__(self, config: Config):
        self.run_dir: Optional[Path] = None
        self.config = self._prepare_runtime_config(config)
        self._logger: Optional[logging.Logger] = None
        self.bundle: Optional[ForecastDataBundle] = None
        self.model: Optional[WGAN_GP] = None

    def _prepare_runtime_config(self, config: Config) -> Config:
        """Allow subclasses to rewrite config paths before training starts."""
        return config

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
        output_path = training_run_config_path(self.config, self.run_dir)
        save_config_yaml(self.config, str(output_path))
        self.logger.info("Resolved config saved to: %s", output_path)

    def _log_device_info(self):
        """Log compute device information."""
        import torch
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = getattr(device_props, "total_memory", 0) / 1024**3
            self.logger.info("CUDA available: %s (%.1f GB)", torch.cuda.get_device_name(0), total_memory_gb)
        else:
            self.logger.info("CUDA not available, using CPU")
        self.logger.info("Device: %s", "cuda:0" if (self.config.cuda and torch.cuda.is_available()) else "cpu")

    def setup(self):
        """Prepare dataset bundle and model instance before training."""
        self._log_device_info()

        self.logger.info("*** Loading dataset ***")
        self.logger.info("Data source: %s (sheet: %s)", self.config.data_path or self.config.option_data_glob, self.config.sheet_name)
        self.bundle = create_bond_option_forecast_dataloaders(self.config)
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

    def start_train(self):
        """Execute full training run."""
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self.setup()
        self._save_run_config()
        self.logger.info("*** Start training: %s epochs, batch_size=%s, lr=%s ***",
                         self.config.num_epochs, self.config.batch_size, self.config.learning_rate)
        assert self.model is not None
        assert self.bundle is not None
        self.model.train(self.bundle.train_loader, self.bundle.val_loader)
        self.logger.info("*** Training complete ***")

    def dry_run(self):
        """Run setup and config snapshot without fitting the model."""
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self.setup()
        self._save_run_config()
        self.logger.info("Dry run finished. Training was not started.")


def main(config: Optional[Config] = None):
    """Convenience main for programmatic usage."""
    trainer = WGANTrainer(config or default_config)
    trainer.start_train()


if __name__ == "__main__":
    main()
