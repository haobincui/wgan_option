"""Standalone CNN WGAN research module for merged-vol scenario generation."""

__all__ = [
    "CnnWGANSampleConfig",
    "CnnWGANTrainConfig",
    "load_sample_config",
    "load_train_config",
    "CnnWGANSampler",
    "CnnWGANTrainer",
]


def __getattr__(name: str):
    """Load training dependencies only when package-level symbols are requested."""

    if name in {"CnnWGANSampleConfig", "CnnWGANTrainConfig", "load_sample_config", "load_train_config"}:
        from . import config

        return getattr(config, name)
    if name == "CnnWGANSampler":
        from .inference import CnnWGANSampler

        return CnnWGANSampler
    if name == "CnnWGANTrainer":
        from .trainer import CnnWGANTrainer

        return CnnWGANTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
