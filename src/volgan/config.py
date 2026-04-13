"""Typed configuration loaders for the standalone VolGAN module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import torch
import yaml
from utils.output_paths import (
    find_best_checkpoint,
    find_latest_run_dir,
    infer_dataset_family as infer_output_dataset_family,
    resolve_output_root,
)
from utils.training_paths import generate_result_dir, infer_run_dir_from_checkpoint
from wgan_option.config_parsing import load_yaml_mapping

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
    models_path: str = ""
    metrics_path: str = ""
    samples_path: str = ""
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
    return infer_output_dataset_family(data_path)


def default_train_output_root(data_path: str | Path) -> str:
    return str(
        resolve_output_root(
            "",
            base_root="outputs/training/volgan",
            data_path=data_path,
        )
    )


def _config_to_dict(config: Any) -> Dict[str, Any]:
    return asdict(config)


def _load_yaml_values(config_path: str | Path) -> Dict[str, Any]:
    _, payload = load_yaml_mapping(config_path)
    return payload


def _coerce_config(config_cls: Type[T], values: Dict[str, Any]) -> T:
    allowed = {field.name for field in fields(config_cls)}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(
            f"Unexpected config keys for {config_cls.__name__}: {unexpected}. Allowed keys: {sorted(allowed)}"
        )
    return config_cls(**values)


def _field_names(config_cls: Type[Any]) -> set[str]:
    return {field.name for field in fields(config_cls)}


def load_train_config(config_path: str | Path) -> VolGANTrainConfig:
    payload = _load_yaml_values(config_path)
    values = payload.get("training", payload)
    if not isinstance(values, dict):
        raise ValueError(f"Config section 'training' in {config_path} must define a YAML mapping.")
    config = _coerce_config(VolGANTrainConfig, values)
    if "generator_learning_rate" not in values:
        config.generator_learning_rate = float(config.learning_rate)
    if "discriminator_learning_rate" not in values:
        config.discriminator_learning_rate = float(config.learning_rate)
    if not str(config.output_root).strip():
        config.output_root = default_train_output_root(config.data_path)
    return config


def _build_sample_defaults(
    *,
    train_config: VolGANTrainConfig,
    run_dir: Path,
    checkpoint_path: str | Path | None = None,
) -> VolGANSampleConfig:
    sample_defaults = VolGANSampleConfig()
    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else find_best_checkpoint(
        run_dir,
        filename="volgan_best.pt",
    )
    return VolGANSampleConfig(
        data_path=train_config.data_path,
        sheet_name=train_config.sheet_name,
        text_embedding_mode=train_config.text_embedding_mode,
        train_ratio=train_config.train_ratio,
        checkpoint_path=str(resolved_checkpoint_path),
        seed=int(train_config.seed),
        cuda=bool(train_config.cuda),
        mc_samples=int(train_config.eval_mc_samples),
        reweight_beta_mode=str(train_config.eval_reweight_beta_mode),
        reweight_beta=float(train_config.eval_reweight_beta),
        quantiles=list(sample_defaults.quantiles),
        split=str(sample_defaults.split),
        selection_mode=str(sample_defaults.selection_mode),
        selection_count=int(sample_defaults.selection_count),
        aggregation_mode=str(train_config.eval_aggregation_mode),
        output_dir=str(generate_result_dir(run_dir)),
        save_json=bool(sample_defaults.save_json),
        save_plots=bool(sample_defaults.save_plots),
    )


def load_sample_config(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> VolGANSampleConfig:
    payload = _load_yaml_values(config_path)
    if "training" in payload or "generate_result" in payload or set(payload).issubset(_field_names(VolGANTrainConfig)):
        train_config = load_train_config(config_path)
        resolved_run_dir = (
            Path(run_dir)
            if run_dir is not None
            else infer_run_dir_from_checkpoint(checkpoint_path) or find_latest_run_dir(train_config.output_root)
        )
        config = _build_sample_defaults(
            train_config=train_config,
            run_dir=resolved_run_dir,
            checkpoint_path=checkpoint_path,
        )
        generate_values = payload.get("generate_result") or {}
        if generate_values and not isinstance(generate_values, dict):
            raise ValueError(f"Config section 'generate_result' in {config_path} must define a YAML mapping.")
        if generate_values:
            config = replace(config, **_coerce_config(VolGANSampleConfig, {**_config_to_dict(config), **generate_values}).__dict__)
        if checkpoint_path is not None:
            config = replace(config, checkpoint_path=str(checkpoint_path))
        config = replace(config, output_dir=str(generate_result_dir(resolved_run_dir, config.output_dir)))
        return config

    config = _coerce_config(VolGANSampleConfig, payload)
    if run_dir is not None:
        config = replace(config, output_dir=str(generate_result_dir(run_dir, config.output_dir)))
    return config


def build_sample_config_from_train_config(
    train_config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> VolGANSampleConfig:
    """Derive generate-result settings directly from one training config."""

    train_config = load_train_config(train_config_path)
    resolved_run_dir = (
        Path(run_dir)
        if run_dir is not None
        else infer_run_dir_from_checkpoint(checkpoint_path) or find_latest_run_dir(train_config.output_root)
    )
    return _build_sample_defaults(
        train_config=train_config,
        run_dir=resolved_run_dir,
        checkpoint_path=checkpoint_path,
    )


def save_config_yaml(config: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
