"""Training orchestrator for conditional WGAN option-surface forecasting."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

from trainer import BaseTrainer
from utils.generate_result_runtime import generate_vol_result, validate_result_config
from utils.postprocess_runtime import resolve_checkpoint_path
from utils.result_config import GenerateResultConfig, build_generate_result_config
from utils.training_paths import checkpoint_named_dir, generate_result_dir, infer_training_output_root, resolve_existing_run_dir
from wgan_option.config import Config, config_to_dict, default_config
from wgan_option.config_parsing import load_yaml_mapping
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.utils.dataloader import ForecastDataBundle, create_bond_option_forecast_dataloaders


def _load_generate_result_section(config_path: str | None) -> dict:
    if not config_path:
        return {}
    _, payload = load_yaml_mapping(config_path)
    section = payload.get("generate_result") or {}
    if section and not isinstance(section, dict):
        raise ValueError(f"Config section 'generate_result' in {config_path} must contain a YAML mapping.")
    return dict(section)


class WGANTrainer(BaseTrainer):
    """High-level trainer wrapper around dataloading + model lifecycle."""

    trainer_id = "cnn_wgan"
    logger_name = "wgan_option.trainer"

    def __init__(self, config: Config, *, config_path: str | None = None):
        super().__init__(config, config_path=config_path)
        self.bundle: Optional[ForecastDataBundle] = None
        self.model: Optional[WGAN_GP] = None

    @property
    def logger(self) -> logging.Logger:
        """Lazily initialize the shared trainer logger."""
        return self._get_or_create_logger()

    def _log_device_info(self) -> None:
        """Log compute device information."""
        import torch

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory_gb = getattr(device_props, "total_memory", 0) / 1024**3
            self.logger.info("CUDA available: %s (%.1f GB)", torch.cuda.get_device_name(0), total_memory_gb)
        else:
            self.logger.info("CUDA not available, using CPU")
        self.logger.info("Device: %s", "cuda:0" if (self.config.cuda and torch.cuda.is_available()) else "cpu")

    def setup(self) -> None:
        """Prepare dataset bundle and model instance before training."""
        self._log_device_info()

        self.logger.info("*** Loading dataset ***")
        self.logger.info(
            "Data source: %s (sheet: %s)",
            self.config.data_path or self.config.option_data_glob,
            self.config.sheet_name,
        )
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
        self.logger.info(
            "Model initialized: G params=%s, D params=%s, total=%s",
            sum(p.numel() for p in self.model.G.parameters()),
            sum(p.numel() for p in self.model.D.parameters()),
            total_params,
        )

    def _train_impl(self) -> Path | None:
        self.logger.info(
            "*** Start training: %s epochs, batch_size=%s, lr=%s ***",
            self.config.num_epochs,
            self.config.batch_size,
            self.config.learning_rate,
        )
        assert self.model is not None
        assert self.bundle is not None
        self.model.train(
            self.bundle.train_loader,
            self.bundle.val_loader,
            val_samples=getattr(self.bundle, "val_items", None),
        )
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

        base_training_config = self.config if self._runtime_prepared else self.raw_config
        output_root = infer_training_output_root(base_training_config, trainer_id=self.trainer_id)
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
            artifact_key="generator",
            fallback_filenames=("generator_best.pt", "generator.pt"),
        )
        if str(generate_config.output_dir).strip():
            resolved_generate_dir = generate_result_dir(run_dir, generate_config.output_dir)
        else:
            resolved_generate_dir = checkpoint_named_dir(generate_result_dir(run_dir), resolved_checkpoint_path)
        resolved_config = replace(
            runtime_config,
            checkpoint_path=str(resolved_checkpoint_path),
            output_dir=str(resolved_generate_dir),
        )
        validate_result_config(resolved_config)
        return resolved_config, resolved_generate_dir

    def _generate_result_impl(self, generate_config: GenerateResultConfig, generate_dir: Path) -> Path:
        del generate_dir
        return generate_vol_result(generate_config)


def main(config: Optional[Config] = None, *, config_path: str | None = None):
    """Convenience main for programmatic usage."""

    trainer = WGANTrainer(config or default_config, config_path=config_path)
    trainer.train()


if __name__ == "__main__":
    main()
