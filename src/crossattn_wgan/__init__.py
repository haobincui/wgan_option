"""Standalone Cross-Attention WGAN research module for merged-vol scenario generation."""

from .config import CrossAttnWGANSampleConfig, CrossAttnWGANTrainConfig, load_sample_config, load_train_config
from .inference import CrossAttnWGANSampler
from .trainer import CrossAttnWGANTrainer

__all__ = [
    "CrossAttnWGANSampleConfig",
    "CrossAttnWGANTrainConfig",
    "load_sample_config",
    "load_train_config",
    "CrossAttnWGANSampler",
    "CrossAttnWGANTrainer",
]
