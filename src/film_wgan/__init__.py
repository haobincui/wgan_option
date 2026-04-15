"""Standalone FiLM WGAN research module for merged-vol scenario generation."""

from .config import FilmWGANSampleConfig, FilmWGANTrainConfig, load_sample_config, load_train_config
from .inference import FilmWGANSampler
from .trainer import FilmWGANTrainer

__all__ = [
    "FilmWGANSampleConfig",
    "FilmWGANTrainConfig",
    "load_sample_config",
    "load_train_config",
    "FilmWGANSampler",
    "FilmWGANTrainer",
]
