"""Standalone VolGAN research module for merged-vol scenario generation."""

from .config import (
    VolGANSampleConfig,
    VolGANTrainConfig,
    build_sample_config_from_train_config,
    load_sample_config,
    load_train_config,
)
from .inference import VolGANSampler
from .trainer import VolGANTrainer

__all__ = [
    "VolGANSampleConfig",
    "VolGANTrainConfig",
    "build_sample_config_from_train_config",
    "load_sample_config",
    "load_train_config",
    "VolGANSampler",
    "VolGANTrainer",
]
