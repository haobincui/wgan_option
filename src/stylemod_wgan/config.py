"""Typed configuration loaders for the standalone StyleMod WGAN module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Type, TypeVar

import torch
import yaml

from utils.output_paths import default_output_root, find_best_checkpoint
from utils.training_paths import checkpoint_named_dir, generate_result_dir
from wgan_option.config_parsing import load_yaml_config_values, load_yaml_mapping, parse_typed_overrides

T = TypeVar("T")


@dataclass
class StyleModWGANTrainConfig:
    """Training configuration for the standalone StyleMod WGAN module."""

    data_path: str = "data/processed/svi-excel/20260410-174929/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "lp"
    train_ratio: float = 0.8
    min_samples_for_training: int = 2

    normalize_current_surface: bool = True
    normalize_target_delta: bool = True
    normalize_text_embedding: bool = True

    noise_dim: int = 32
    gen_base_channels: int = 32
    disc_base_channels: int = 32
    gen_res_blocks: int = 1
    disc_res_blocks: int = 1
    text_hidden_dim: int = 256
    text_out_dim: int = 128
    fusion_hidden_dim: int = 512
    style_dim: int = 128
    style_noise_scale: float = 0.25
    style_demodulate: bool = True

    learning_rate: float = 1e-4
    generator_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4
    beta_1: float = 0.5
    beta_2: float = 0.9
    critic_iter: int = 5
    lambda_gp: float = 10.0
    num_epochs: int = 100
    batch_size: int = 32

    lambda_calendar: float = 2.0
    lambda_butterfly: float = 2.0
    lambda_smooth: float = 0.1
    lambda_recon: float = 0.0
    use_calendar_constraint: bool = True
    use_butterfly_constraint: bool = True
    use_smooth_constraint: bool = True
    use_recon_constraint: bool = False

    eval_mc_samples: int = 32
    eval_reweight_beta_mode: str = "fixed"
    eval_reweight_beta: float = 25.0
    eval_aggregation_mode: str = "weighted_mean"
    checkpoint_metric: str = "val_mae_gap_vs_current"
    checkpoint_warmup_epochs: int = 10

    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0

    output_root: str = ""
    checkpoints_path: str = ""
    metrics_path: str = ""
    save_every: int = 10


@dataclass
class StyleModWGANSampleConfig:
    """Generate-result configuration for the standalone StyleMod WGAN module."""

    data_path: str = "data/processed/svi-excel/20260410-174929/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "lp"
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


_TRAIN_DEFAULTS = StyleModWGANTrainConfig()
_SAMPLE_DEFAULTS = StyleModWGANSampleConfig()
_TRAIN_FIELD_NAMES = {field.name for field in fields(StyleModWGANTrainConfig)}
_SAMPLE_FIELD_NAMES = {field.name for field in fields(StyleModWGANSampleConfig)}
_SHARED_GENERATE_FIELDS = {
    "data_path",
    "sheet_name",
    "text_embedding_mode",
    "train_ratio",
    "seed",
    "cuda",
}


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a config dataclass into a plain dictionary."""

    return asdict(config)


def _coerce_config(config_cls: Type[T], values: Mapping[str, Any], *, source: str) -> T:
    allowed = {field.name for field in fields(config_cls)}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(
            f"Unexpected config keys for {config_cls.__name__} in {source}: {unexpected}. "
            f"Allowed keys: {sorted(allowed)}"
        )
    return config_cls(**dict(values))


def parse_train_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse typed CLI overrides for training configs."""

    return parse_typed_overrides(
        override_items,
        defaults=config_to_dict(_TRAIN_DEFAULTS),
        example="--set batch_size=64",
    )


def parse_sample_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse typed CLI overrides for generate-result configs."""

    return parse_typed_overrides(
        override_items,
        defaults=config_to_dict(_SAMPLE_DEFAULTS),
        example="--set split=val",
    )


def default_train_output_root(data_path: str | Path) -> str:
    """Return the dataset-aware default training root."""

    return str(default_output_root("outputs/training/stylemod_wgan", data_path))


def _load_generate_sections(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path, payload = load_yaml_mapping(config_path)
    if "training" in payload or "generate_result" in payload:
        training_values = payload.get("training") or {}
        generate_values = payload.get("generate_result") or {}
        if training_values and not isinstance(training_values, dict):
            raise ValueError(f"Config section 'training' in {path} must contain a YAML mapping.")
        if generate_values and not isinstance(generate_values, dict):
            raise ValueError(f"Config section 'generate_result' in {path} must contain a YAML mapping.")
        return dict(training_values), dict(generate_values)
    if set(payload).issubset(_TRAIN_FIELD_NAMES):
        return dict(payload), {}
    return {}, dict(payload)


def build_sample_config(
    *,
    training_values: Mapping[str, Any] | None = None,
    generate_values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> StyleModWGANSampleConfig:
    """Merge shared training fields with one generate-result section."""

    loaded_values = config_to_dict(_SAMPLE_DEFAULTS)

    if training_values:
        unknown_training = sorted(set(training_values) - _TRAIN_FIELD_NAMES)
        if unknown_training:
            raise ValueError(
                f"Unknown training config keys for StyleMod WGAN: {unknown_training}. "
                f"Allowed keys: {sorted(_TRAIN_FIELD_NAMES)}"
            )
        for key in _SHARED_GENERATE_FIELDS:
            if key in training_values:
                loaded_values[key] = training_values[key]

    if generate_values:
        unknown_generate = sorted(set(generate_values) - _SAMPLE_FIELD_NAMES)
        if unknown_generate:
            raise ValueError(
                f"Unknown generate_result config keys for StyleMod WGAN: {unknown_generate}. "
                f"Allowed keys: {sorted(_SAMPLE_FIELD_NAMES)}"
            )
        loaded_values.update(dict(generate_values))

    if overrides:
        unknown_overrides = sorted(set(overrides) - _SAMPLE_FIELD_NAMES)
        if unknown_overrides:
            raise ValueError(
                f"Unknown generate_result overrides for StyleMod WGAN: {unknown_overrides}. "
                f"Allowed keys: {sorted(_SAMPLE_FIELD_NAMES)}"
            )
        loaded_values.update(dict(overrides))

    config = StyleModWGANSampleConfig(**loaded_values)
    if checkpoint_path not in {None, ""}:
        config = replace(config, checkpoint_path=str(checkpoint_path))
    if run_dir is not None:
        resolved_checkpoint_path = (
            Path(config.checkpoint_path)
            if str(config.checkpoint_path).strip()
            else find_best_checkpoint(run_dir, filename="stylemod_wgan_best.pt")
        )
        if str(config.output_dir).strip():
            resolved_output_dir = generate_result_dir(run_dir, config.output_dir)
        else:
            resolved_output_dir = checkpoint_named_dir(generate_result_dir(run_dir), resolved_checkpoint_path)
        config = replace(
            config,
            checkpoint_path=str(resolved_checkpoint_path),
            output_dir=str(resolved_output_dir),
        )
    return config


def build_sample_config_from_train_config(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> StyleModWGANSampleConfig:
    """Build a generate-result config from the merged training YAML."""

    training_values, generate_values = _load_generate_sections(config_path)
    return build_sample_config(
        training_values=training_values,
        generate_values=generate_values,
        overrides=overrides,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )


def load_train_config(
    config_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> StyleModWGANTrainConfig:
    """Load a training config from a flat or merged YAML."""

    _, yaml_values, loaded_values = load_yaml_config_values(
        config_path,
        defaults=config_to_dict(_TRAIN_DEFAULTS),
        overrides=overrides,
        section="training",
    )
    config = _coerce_config(StyleModWGANTrainConfig, loaded_values, source=str(config_path))
    if "generator_learning_rate" not in yaml_values and not (overrides and "generator_learning_rate" in overrides):
        config.generator_learning_rate = float(config.learning_rate)
    if "discriminator_learning_rate" not in yaml_values and not (
        overrides and "discriminator_learning_rate" in overrides
    ):
        config.discriminator_learning_rate = float(config.learning_rate)
    if not str(config.output_root).strip():
        config.output_root = default_train_output_root(config.data_path)
    return config


def load_sample_config(
    config_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> StyleModWGANSampleConfig:
    """Load one generate-result config from either merged or legacy YAML."""

    return build_sample_config_from_train_config(
        config_path,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        overrides=overrides,
    )


def save_config_yaml(config: Any, output_path: str | Path) -> None:
    """Persist one resolved config snapshot."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
