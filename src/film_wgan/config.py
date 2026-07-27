"""Typed configuration loaders for the standalone FiLM WGAN module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Type, TypeVar

import torch
import yaml

from utils.output_paths import default_output_root, find_best_checkpoint
from utils.training_paths import checkpoint_named_dir, generate_result_dir
from wgan_option.config_parsing import load_yaml_config_values, load_yaml_mapping, parse_typed_overrides

T = TypeVar("T")


@dataclass
class FilmWGANTrainConfig:
    """Training configuration for the standalone FiLM WGAN module."""

    data_path: str = "data/processed/svi-excel/20260410-174929/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "lp"
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    split_strategy: str = "legacy_row"
    split_manifest_path: str = ""
    text_alignment_mode: str = "matched"
    text_permutation_seed: int = 20260722
    sample_unit: str = "article_row"
    news_workbook_path: str = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
    text_lineage_mode: str = "legacy"
    text_pooling_mode: str = "mean_l2"
    pair_text_feature_path: str = ""
    text_preprocessing_mode: str = "coordinate_zscore"
    text_transform_path: str = ""
    text_pca_components: int = 128
    text_pca_whiten: bool = False
    text_output_dim: int = 128
    surface_support_mode: str = "full_grid"
    surface_support_path: str = ""
    support_grid_quantile_low: float = 0.05
    support_grid_quantile_high: float = 0.95
    support_strike_bins: int = 16
    support_maturity_bins: int = 16
    support_min_train_pair_cells: int = 1
    report_atm7_metric: bool = True
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
    forecast_mode: str = "stochastic_wgan"
    conditioning_mode: str = "film"
    critic_conditioning_mode: str = "inherit"
    text_dropout: float = 0.0
    text_gate_initial_value: float = 0.0

    learning_rate: float = 1e-4
    generator_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4
    backbone_learning_rate: float = 2e-6
    text_adapter_learning_rate: float = 2e-5
    beta_1: float = 0.5
    beta_2: float = 0.9
    critic_iter: int = 5
    lambda_gp: float = 10.0
    num_epochs: int = 100
    batch_size: int = 32

    lambda_adv: float = 1.0
    lambda_film: float = 0.0
    lambda_mismatch: float = 0.0
    adv_warmup_epochs: int = 0
    lambda_calendar: float = 2.0
    lambda_butterfly: float = 2.0
    lambda_smooth: float = 0.1
    lambda_recon: float = 0.0
    recon_weight_mode: str = "uniform"
    recon_atm_range: float = 0.08
    recon_atm_short_end_max_days: float = 90.0
    recon_atm_multiplier: float = 3.0
    use_calendar_constraint: bool = True
    use_butterfly_constraint: bool = True
    use_smooth_constraint: bool = True
    use_recon_constraint: bool = False

    use_atm_short_loss: bool = False
    lambda_atm_short: float = 0.0
    atm_short_range: float = 0.05
    atm_short_max_days: float = 60.0

    eval_mc_samples: int = 32
    eval_reweight_beta_mode: str = "fixed"
    eval_reweight_beta: float = 25.0
    eval_aggregation_mode: str = "weighted_mean"
    eval_calibration_levels: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.9])
    arbitrage_violation_tolerance: float = 1e-8
    checkpoint_metric: str = "val_mae_gap_vs_current"
    extra_checkpoint_metrics: list[str] = field(default_factory=list)
    checkpoint_warmup_epochs: int = 10

    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0

    output_root: str = ""
    checkpoints_path: str = ""
    metrics_path: str = ""
    initial_generator_checkpoint_path: str = ""
    parent_text_transform_policy: str = "exact"
    freeze_backbone_epochs: int = 0
    save_every: int = 10


@dataclass
class FilmWGANSampleConfig:
    """Generate-result configuration for the standalone FiLM WGAN module."""

    data_path: str = "data/processed/svi-excel/20260410-174929/merged_vol.xlsx"
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "lp"
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    split_strategy: str = "legacy_row"
    split_manifest_path: str = ""
    text_alignment_mode: str = "matched"
    text_permutation_seed: int = 20260722
    sample_unit: str = "article_row"
    news_workbook_path: str = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
    text_lineage_mode: str = "legacy"
    text_pooling_mode: str = "mean_l2"
    pair_text_feature_path: str = ""
    text_preprocessing_mode: str = "coordinate_zscore"
    text_transform_path: str = ""
    text_pca_components: int = 128
    text_pca_whiten: bool = False
    text_output_dim: int = 128
    surface_support_mode: str = "full_grid"
    surface_support_path: str = ""
    support_grid_quantile_low: float = 0.05
    support_grid_quantile_high: float = 0.95
    support_strike_bins: int = 16
    support_maturity_bins: int = 16
    support_min_train_pair_cells: int = 1
    report_atm7_metric: bool = True

    checkpoint_path: str = ""
    seed: int = 42
    cuda: bool = torch.cuda.is_available()

    mc_samples: int = 64
    reweight_beta_mode: str = "fixed"
    reweight_beta: float = 25.0
    quantiles: list[float] | tuple[float, ...] = (0.05, 0.5, 0.95)
    calibration_levels: list[float] | tuple[float, ...] = (0.5, 0.8, 0.9)
    arbitrage_violation_tolerance: float = 1e-8

    split: str = "val"
    selection_mode: str = "all"
    selection_count: int = 0
    aggregation_mode: str = "weighted_mean"
    residual_blend_alpha: float = 1.0
    output_dir: str = ""
    save_json: bool = True
    save_plots: bool = True
    save_full_atm_timeseries: bool = True


_TRAIN_DEFAULTS = FilmWGANTrainConfig()
_SAMPLE_DEFAULTS = FilmWGANSampleConfig()
_TRAIN_FIELD_NAMES = {field.name for field in fields(FilmWGANTrainConfig)}
_SAMPLE_FIELD_NAMES = {field.name for field in fields(FilmWGANSampleConfig)}
_SHARED_GENERATE_FIELDS = {
    "data_path",
    "sheet_name",
    "text_embedding_mode",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "split_strategy",
    "split_manifest_path",
    "text_alignment_mode",
    "text_permutation_seed",
    "sample_unit",
    "news_workbook_path",
    "text_lineage_mode",
    "text_pooling_mode",
    "pair_text_feature_path",
    "text_preprocessing_mode",
    "text_transform_path",
    "text_pca_components",
    "text_pca_whiten",
    "text_output_dim",
    "surface_support_mode",
    "surface_support_path",
    "support_grid_quantile_low",
    "support_grid_quantile_high",
    "support_strike_bins",
    "support_maturity_bins",
    "support_min_train_pair_cells",
    "report_atm7_metric",
    "seed",
    "cuda",
}


def _validate_split_fields(config: FilmWGANTrainConfig | FilmWGANSampleConfig) -> None:
    strategy = str(config.split_strategy).strip().lower()
    if strategy not in {"legacy_row", "grouped_chronological"}:
        raise ValueError(
            "split_strategy must be one of ['legacy_row', 'grouped_chronological'], "
            f"got: {config.split_strategy}"
        )
    if strategy == "legacy_row":
        if not 0.0 < float(config.train_ratio) < 1.0:
            raise ValueError("legacy_row train_ratio must be between 0 and 1.")
        return
    ratios = (float(config.train_ratio), float(config.val_ratio), float(config.test_ratio))
    if any(value <= 0.0 for value in ratios):
        raise ValueError("grouped_chronological train/val/test ratios must all be positive.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"grouped_chronological train/val/test ratios must sum to 1, got {ratios}.")


def _validate_common_fields(config: FilmWGANTrainConfig | FilmWGANSampleConfig) -> None:
    _validate_split_fields(config)
    alignment = str(config.text_alignment_mode).strip().lower()
    if alignment not in {"matched", "permuted"}:
        raise ValueError("text_alignment_mode must be one of ['matched', 'permuted'].")
    sample_unit = str(config.sample_unit).strip().lower()
    if sample_unit not in {"article_row", "surface_pair"}:
        raise ValueError("sample_unit must be one of ['article_row', 'surface_pair'].")
    lineage_mode = str(config.text_lineage_mode).strip().lower()
    if lineage_mode not in {"legacy", "strict"}:
        raise ValueError("text_lineage_mode must be one of ['legacy', 'strict'].")
    if lineage_mode == "strict" and sample_unit != "surface_pair":
        raise ValueError("text_lineage_mode=strict requires sample_unit=surface_pair.")
    pooling_mode = str(config.text_pooling_mode).strip().lower()
    allowed_pooling_modes = {"mean_l2", "bow_log_count_l2", "mean_scores"}
    if pooling_mode not in allowed_pooling_modes:
        raise ValueError(
            f"text_pooling_mode must be one of {sorted(allowed_pooling_modes)}."
        )
    preprocessing_mode = str(config.text_preprocessing_mode).strip().lower()
    if preprocessing_mode not in {"raw_l2", "pca", "coordinate_zscore", "zscore_pad"}:
        raise ValueError(
            "text_preprocessing_mode must be one of "
            "['raw_l2', 'pca', 'coordinate_zscore', 'zscore_pad']."
        )
    if int(config.text_pca_components) <= 0:
        raise ValueError("text_pca_components must be positive.")
    if int(config.text_output_dim) <= 0:
        raise ValueError("text_output_dim must be positive.")
    support_mode = str(config.surface_support_mode).strip().lower()
    if support_mode not in {"full_grid", "raw_observed"}:
        raise ValueError("surface_support_mode must be one of ['full_grid', 'raw_observed'].")
    if support_mode == "raw_observed":
        if sample_unit != "surface_pair":
            raise ValueError("surface_support_mode=raw_observed requires sample_unit=surface_pair.")
        if not str(config.surface_support_path).strip():
            raise ValueError("surface_support_mode=raw_observed requires surface_support_path.")
        if int(config.support_strike_bins) < 2 or int(config.support_maturity_bins) < 2:
            raise ValueError("Raw support grids require at least two strike and maturity bins.")
        if int(config.support_min_train_pair_cells) < 1:
            raise ValueError("support_min_train_pair_cells must be positive.")
        if not (
            0.0
            <= float(config.support_grid_quantile_low)
            < float(config.support_grid_quantile_high)
            <= 1.0
        ):
            raise ValueError(
                "Support quantiles must satisfy 0 <= support_grid_quantile_low "
                "< support_grid_quantile_high <= 1."
            )
    if preprocessing_mode in {"pca", "zscore_pad"} and bool(
        getattr(config, "normalize_text_embedding", False)
    ):
        raise ValueError(
            f"text_preprocessing_mode={preprocessing_mode} requires "
            "normalize_text_embedding=false."
        )
    pair_feature_path = str(config.pair_text_feature_path).strip()
    if pair_feature_path and sample_unit != "surface_pair":
        raise ValueError("pair_text_feature_path requires sample_unit=surface_pair.")
    if pair_feature_path and pooling_mode == "mean_l2":
        raise ValueError(
            "External pair_text_feature_path requires an explicit RQ2 pooling mode."
        )
    if str(config.text_embedding_mode).strip().lower().replace("-", "_") == "zero_lp" and bool(
        getattr(config, "normalize_text_embedding", False)
    ):
        raise ValueError("text_embedding_mode=zero_lp requires normalize_text_embedding=false.")


def _validate_train_fields(config: FilmWGANTrainConfig) -> None:
    _validate_common_fields(config)
    forecast_mode = str(config.forecast_mode).strip().lower()
    conditioning_mode = str(config.conditioning_mode).strip().lower()
    if forecast_mode not in {"stochastic_wgan", "deterministic"}:
        raise ValueError("forecast_mode must be one of ['stochastic_wgan', 'deterministic'].")
    if conditioning_mode not in {"film", "concat", "residual_film"}:
        raise ValueError("conditioning_mode must be one of ['film', 'concat', 'residual_film'].")
    critic_conditioning_mode = str(config.critic_conditioning_mode).strip().lower()
    if critic_conditioning_mode not in {"inherit", "film", "concat", "projection"}:
        raise ValueError(
            "critic_conditioning_mode must be one of ['inherit', 'film', 'concat', 'projection']."
        )
    if conditioning_mode == "residual_film" and critic_conditioning_mode != "projection":
        raise ValueError("conditioning_mode=residual_film requires critic_conditioning_mode=projection.")
    if str(config.initial_generator_checkpoint_path).strip() and conditioning_mode != "residual_film":
        raise ValueError("initial_generator_checkpoint_path is supported only for residual_film.")
    parent_transform_policy = str(config.parent_text_transform_policy).strip().lower()
    if parent_transform_policy not in {"exact", "dimension_only"}:
        raise ValueError(
            "parent_text_transform_policy must be one of ['exact', 'dimension_only']."
        )
    if not 0.0 <= float(config.text_dropout) < 1.0:
        raise ValueError("text_dropout must be in [0, 1).")
    if float(config.lambda_film) < 0.0 or float(config.lambda_mismatch) < 0.0:
        raise ValueError("lambda_film and lambda_mismatch must be non-negative.")
    if int(config.freeze_backbone_epochs) < 0:
        raise ValueError("freeze_backbone_epochs must be non-negative.")
    if float(config.backbone_learning_rate) <= 0.0 or float(config.text_adapter_learning_rate) <= 0.0:
        raise ValueError("backbone_learning_rate and text_adapter_learning_rate must be positive.")
    if forecast_mode == "deterministic" and abs(float(config.lambda_adv)) > 1e-12:
        raise ValueError("forecast_mode=deterministic requires lambda_adv=0.")
    levels = [float(value) for value in config.eval_calibration_levels]
    if any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("eval_calibration_levels must contain values strictly between 0 and 1.")
    if float(config.arbitrage_violation_tolerance) < 0.0:
        raise ValueError("arbitrage_violation_tolerance must be non-negative.")
    if str(config.surface_support_mode).strip().lower() == "raw_observed" and any(
        (
            bool(config.use_calendar_constraint),
            bool(config.use_butterfly_constraint),
            bool(config.use_smooth_constraint),
        )
    ):
        raise ValueError(
            "raw_observed support uses an irregular per-sample mask; calendar, butterfly, "
            "and smoothness penalties must be disabled unless support-aware edge penalties "
            "are explicitly implemented."
        )


def _validate_sample_fields(config: FilmWGANSampleConfig) -> None:
    _validate_common_fields(config)
    if str(config.split).strip().lower() not in {"train", "val", "test", "all"}:
        raise ValueError("split must be one of ['train', 'val', 'test', 'all'].")
    levels = [float(value) for value in config.calibration_levels]
    if any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("calibration_levels must contain values strictly between 0 and 1.")
    if float(config.arbitrage_violation_tolerance) < 0.0:
        raise ValueError("arbitrage_violation_tolerance must be non-negative.")


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

    return str(default_output_root("outputs/training/film_wgan", data_path))


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
) -> FilmWGANSampleConfig:
    """Merge shared training fields with one generate-result section."""

    loaded_values = config_to_dict(_SAMPLE_DEFAULTS)

    if training_values:
        unknown_training = sorted(set(training_values) - _TRAIN_FIELD_NAMES)
        if unknown_training:
            raise ValueError(
                f"Unknown training config keys for FiLM WGAN: {unknown_training}. "
                f"Allowed keys: {sorted(_TRAIN_FIELD_NAMES)}"
            )
        for key in _SHARED_GENERATE_FIELDS:
            if key in training_values:
                loaded_values[key] = training_values[key]

    if generate_values:
        unknown_generate = sorted(set(generate_values) - _SAMPLE_FIELD_NAMES)
        if unknown_generate:
            raise ValueError(
                f"Unknown generate_result config keys for FiLM WGAN: {unknown_generate}. "
                f"Allowed keys: {sorted(_SAMPLE_FIELD_NAMES)}"
            )
        loaded_values.update(dict(generate_values))

    if overrides:
        unknown_overrides = sorted(set(overrides) - _SAMPLE_FIELD_NAMES)
        if unknown_overrides:
            raise ValueError(
                f"Unknown generate_result overrides for FiLM WGAN: {unknown_overrides}. "
                f"Allowed keys: {sorted(_SAMPLE_FIELD_NAMES)}"
            )
        loaded_values.update(dict(overrides))

    config = FilmWGANSampleConfig(**loaded_values)
    _validate_sample_fields(config)
    if checkpoint_path not in {None, ""}:
        config = replace(config, checkpoint_path=str(checkpoint_path))
    if run_dir is not None:
        resolved_checkpoint_path = (
            Path(config.checkpoint_path)
            if str(config.checkpoint_path).strip()
            else find_best_checkpoint(run_dir, filename="film_wgan_best.pt")
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
) -> FilmWGANSampleConfig:
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
) -> FilmWGANTrainConfig:
    """Load a training config from a flat or merged YAML."""

    _, yaml_values, loaded_values = load_yaml_config_values(
        config_path,
        defaults=config_to_dict(_TRAIN_DEFAULTS),
        overrides=overrides,
        section="training",
    )
    config = _coerce_config(FilmWGANTrainConfig, loaded_values, source=str(config_path))
    if "generator_learning_rate" not in yaml_values and not (overrides and "generator_learning_rate" in overrides):
        config.generator_learning_rate = float(config.learning_rate)
    if "discriminator_learning_rate" not in yaml_values and not (
        overrides and "discriminator_learning_rate" in overrides
    ):
        config.discriminator_learning_rate = float(config.learning_rate)
    if not str(config.output_root).strip():
        config.output_root = default_train_output_root(config.data_path)
    _validate_train_fields(config)
    return config


def load_sample_config(
    config_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> FilmWGANSampleConfig:
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
