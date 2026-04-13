"""Shared abstract trainer lifecycle for training and generate-result workflows."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from utils.training_paths import generate_result_config_path, training_run_config_path


def _config_to_payload(config: Any) -> Any:
    """Convert dataclass-backed config objects into plain YAML-safe payloads."""

    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return config


def save_yaml_payload(config: Any, output_path: str | Path) -> Path:
    """Persist one resolved config payload to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_config_to_payload(config), handle, sort_keys=False, allow_unicode=False)
    return path


class BaseTrainer(ABC):
    """Common lifecycle for trainer-led train / dry-run / generate-result flows."""

    trainer_id = "base"
    logger_name = "trainer"

    def __init__(self, config: Any, *, config_path: str | None = None):
        self.raw_config = config
        self.config = config
        self.config_path = str(config_path) if config_path else None
        self.run_dir: Optional[Path] = None
        self._logger: Optional[logging.Logger] = None
        self._runtime_prepared = False

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(handler)
            self._logger = logger
        return self._logger

    def _prepare_runtime_config(self, config: Any) -> Any:
        """Allow subclasses to rewrite config paths before training starts."""

        return config

    def _normalize_runtime_config(self, prepared: Any) -> tuple[Any, Optional[Path]]:
        if isinstance(prepared, tuple) and len(prepared) == 2:
            config, run_dir = prepared
            return config, None if run_dir is None else Path(run_dir)
        return prepared, None

    def _ensure_runtime_prepared(self) -> None:
        if self._runtime_prepared:
            return
        self.config, self.run_dir = self._normalize_runtime_config(
            self._prepare_runtime_config(self.raw_config)
        )
        self._runtime_prepared = True

    def _ensure_samples_dir(self) -> None:
        samples_path = str(getattr(self.config, "samples_path", "")).strip()
        if samples_path:
            Path(samples_path).mkdir(parents=True, exist_ok=True)

    def _save_training_resolved_config(self) -> Path:
        output_path = training_run_config_path(self.config, self.run_dir)
        save_yaml_payload(self.config, output_path)
        self.logger.info("Training config snapshot saved to: %s", output_path)
        return output_path

    def _save_generate_resolved_config(self, config: Any, generate_dir: str | Path) -> Path:
        output_path = generate_result_config_path(generate_dir)
        save_yaml_payload(config, output_path)
        self.logger.info("Generate-result config snapshot saved to: %s", output_path)
        return output_path

    @abstractmethod
    def setup(self) -> None:
        """Load data/model state required before one training run."""

    @abstractmethod
    def _train_impl(self) -> Path | None:
        """Execute the model-specific training loop."""

    def train(self) -> Path | None:
        self._ensure_runtime_prepared()
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self._ensure_samples_dir()
        self.setup()
        self._save_training_resolved_config()
        return self._train_impl()

    def start_train(self) -> Path | None:
        """Backward-compatible alias for older script entrypoints/tests."""

        return self.train()

    def dry_run(self) -> Path | None:
        self._ensure_runtime_prepared()
        if self.run_dir is not None:
            self.logger.info("Training artifacts will be written under: %s", self.run_dir)
        self._ensure_samples_dir()
        self.setup()
        self._save_training_resolved_config()
        self.logger.info("Dry run finished. Training was not started.")
        return self.run_dir

    @abstractmethod
    def _prepare_generate_result(
        self,
        generate_config: Any | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        config_path: str | None = None,
    ) -> tuple[Any, Path]:
        """Resolve one generate-result config object and its output directory."""

    @abstractmethod
    def _generate_result_impl(self, generate_config: Any, generate_dir: Path) -> Path:
        """Execute the model-specific generate-result workflow."""

    def generate_result(
        self,
        generate_config: Any | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        config_path: str | None = None,
    ) -> Path:
        resolved_config, generate_dir = self._prepare_generate_result(
            generate_config,
            overrides=overrides,
            config_path=config_path or self.config_path,
        )
        self._save_generate_resolved_config(resolved_config, generate_dir)
        return self._generate_result_impl(resolved_config, generate_dir)

    def run_pipeline(
        self,
        generate_config: Any | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        config_path: str | None = None,
    ) -> Path:
        self.train()
        return self.generate_result(
            generate_config,
            overrides=overrides,
            config_path=config_path or self.config_path,
        )
