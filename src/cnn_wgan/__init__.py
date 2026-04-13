"""Standalone CNN WGAN research module for merged-vol scenario generation."""

from .config import CnnWGANSampleConfig, CnnWGANTrainConfig, load_sample_config, load_train_config
from .inference import CnnWGANSampler
from .trainer import CnnWGANTrainer

__all__ = [
    "CnnWGANSampleConfig",
    "CnnWGANTrainConfig",
    "load_sample_config",
    "load_train_config",
    "CnnWGANSampler",
    "CnnWGANTrainer",
]
