"""Typed configuration loaders for the standalone VolGAN module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import torch
import yaml

T = TypeVar("T")


@dataclass
class VolGANTrainConfig:
    """Training configuration for the standalone VolGAN module."""

    data_path: str = "data/processed/svi-all/20260330-01/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "hd"
    train_ratio: float = 0.8
    min_samples_for_training: int = 2

    noise_dim: int = 32
    hidden_dim: int = 128
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    generator_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4

    disc_steps_per_batch: int = 2
    gen_steps_per_batch: int = 1
    real_label_value: float = 0.9
    fake_label_value: float = 0.0
    generator_grad_clip: float = 5.0
    discriminator_grad_clip: float = 5.0

    use_gradient_matching: bool = True
    gradient_match_epochs: int = 5
    alpha_m: float = 1.0
    alpha_tau: float = 1.0
    alpha_clip_min: float = 1e-3
    alpha_clip_max: float = 10.0

    normalize_current_surface: bool = True
    normalize_target_delta: bool = True
    normalize_text_embedding: bool = True

    eval_mc_samples: int = 32
    eval_reweight_beta_mode: str = "fixed"
    eval_reweight_beta: float = 25.0
    eval_aggregation_mode: str = "weighted_mean"
    checkpoint_metric: str = "val_mae_gap_vs_current"

    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0

    output_root: str = ""
    save_every: int = 10


@dataclass
class VolGANSampleConfig:
    """Sampling configuration for the standalone VolGAN module."""

    data_path: str = "data/processed/svi-all/20260330-01/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "hd"
    train_ratio: float = 0.8

    checkpoint_path: str = ""
    seed: int = 42
    cuda: bool = torch.cuda.is_available()

    mc_samples: int = 64
    reweight_beta_mode: str = "fixed"
    reweight_beta: float = 25.0
    quantiles: list[float] | tuple[float, ...] = (0.05, 0.5, 0.95)

    split: str = "val"
    selection_mode: str = "all"
    selection_count: int = 0
    aggregation_mode: str = "weighted_mean"
    output_dir: str = ""
    save_json: bool = True
    save_plots: bool = True


def infer_dataset_family(data_path: str | Path) -> str:
    path = Path(data_path)
    if path.parent.name and path.parent.name not in {"", "."}:
        grandparent = path.parent.parent.name
        if grandparent:
            return grandparent
        return path.parent.name
    return "default"


def default_train_output_root(data_path: str | Path) -> str:
    return str(Path("outputs/training/volgan") / infer_dataset_family(data_path))


def default_generate_result_output_dir(data_path: str | Path) -> str:
    return str(Path("outputs/generate_result/volgan") / infer_dataset_family(data_path))


def _config_to_dict(config: Any) -> Dict[str, Any]:
    return asdict(config)


def _load_yaml_values(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must define a YAML mapping.")
    return payload


def _coerce_config(config_cls: Type[T], values: Dict[str, Any]) -> T:
    allowed = {field.name for field in fields(config_cls)}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(
            f"Unexpected config keys for {config_cls.__name__}: {unexpected}. Allowed keys: {sorted(allowed)}"
        )
    return config_cls(**values)


def load_train_config(config_path: str | Path) -> VolGANTrainConfig:
    values = _load_yaml_values(config_path)
    config = _coerce_config(VolGANTrainConfig, values)
    if "generator_learning_rate" not in values:
        config.generator_learning_rate = float(config.learning_rate)
    if "discriminator_learning_rate" not in values:
        config.discriminator_learning_rate = float(config.learning_rate)
    if not str(config.output_root).strip():
        config.output_root = default_train_output_root(config.data_path)
    return config


def load_sample_config(config_path: str | Path) -> VolGANSampleConfig:
    values = _load_yaml_values(config_path)
    config = _coerce_config(VolGANSampleConfig, values)
    if not str(config.output_dir).strip():
        config.output_dir = default_generate_result_output_dir(config.data_path)
    return config


def save_config_yaml(config: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
