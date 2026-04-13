"""Configuration model and loading helpers for result-generation scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import yaml
from wgan_option.config_parsing import load_yaml_mapping, load_yaml_config_values, parse_typed_overrides
from wgan_option.surface_grid import (
    DEFAULT_MATURITY_BINS,
    DEFAULT_MATURITY_MAX_DAYS,
    DEFAULT_MATURITY_MIN_DAYS,
    DEFAULT_MONEYNESS_MAX,
    DEFAULT_MONEYNESS_MIN,
    DEFAULT_STRIKE_BINS,
)

DEFAULT_VOL_RESULT_CONFIG_PATH = "configs/generate_result/vol.yaml"
DEFAULT_SVI_RESULT_CONFIG_PATH = "configs/generate_result/svi.yaml"
DEFAULT_VOL_REGRESSION_RESULT_CONFIG_PATH = "configs/generate_result/vol-regression.yaml"

_CONFIG_FIELD_NAMES = {
    "data_path",
    "sheet_name",
    "text_embedding_mode",
    "train_ratio",
    "cuda",
    "seed",
    "num_workers",
    "checkpoint_path",
    "models_path",
    "metrics_path",
    "strike_bins",
    "maturity_bins",
    "moneyness_min",
    "moneyness_max",
    "maturity_min_days",
    "maturity_max_days",
}


@dataclass
class GenerateResultConfig:
    # Data
    data_path: str = ""
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "hd"
    train_ratio: float = 0.8

    # Runtime
    cuda: bool = torch.cuda.is_available()
    seed: int = 42
    num_workers: int = 0

    # Checkpoint resolution
    checkpoint_path: str = ""
    models_path: str = ""
    metrics_path: str = ""
    fallback_mode: str = "none"
    mc_samples: int = 1
    uncertainty_threshold: float = -1.0

    # Sample selection
    split: str = "val"
    selection_mode: str = "first_n"
    sample_id: str = ""
    row_index: int = -1
    limit: int = 5

    # Outputs
    output_dir: str = ""
    save_plots: bool = True
    save_json: bool = True
    plot_style: str = "heatmap_diff"

    # Fixed-grid reconstruction for SVI
    strike_bins: int = DEFAULT_STRIKE_BINS
    maturity_bins: int = DEFAULT_MATURITY_BINS
    moneyness_min: float = DEFAULT_MONEYNESS_MIN
    moneyness_max: float = DEFAULT_MONEYNESS_MAX
    maturity_min_days: int = DEFAULT_MATURITY_MIN_DAYS
    maturity_max_days: int = DEFAULT_MATURITY_MAX_DAYS


def generate_result_config_to_dict(config: GenerateResultConfig) -> Dict[str, Any]:
    """Convert strongly-typed config object into plain dictionary."""
    return asdict(config)


def parse_generate_result_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into typed override dict."""
    return parse_typed_overrides(
        override_items,
        defaults=generate_result_config_to_dict(GenerateResultConfig()),
        example="--set split=all",
    )


def build_generate_result_config(
    *,
    training_values: Mapping[str, Any] | None = None,
    generate_values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> GenerateResultConfig:
    """Merge training-shared fields with one generate-result section."""

    defaults = generate_result_config_to_dict(GenerateResultConfig())
    loaded_values = dict(defaults)

    if training_values:
        for key in _CONFIG_FIELD_NAMES:
            if key in training_values:
                loaded_values[key] = training_values[key]

    if generate_values:
        validate_keys = set(defaults.keys())
        unknown_keys = sorted(set(generate_values.keys()) - validate_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown generate_result config keys: {unknown_keys}. Allowed keys: {sorted(validate_keys)}"
            )
        loaded_values.update(dict(generate_values))

    if overrides:
        unknown_keys = sorted(set(overrides.keys()) - set(defaults.keys()))
        if unknown_keys:
            raise ValueError(
                f"Unknown generate_result override keys: {unknown_keys}. "
                f"Allowed keys: {sorted(defaults.keys())}"
            )
        loaded_values.update(dict(overrides))

    return GenerateResultConfig(**loaded_values)


def load_generate_result_config(
    config_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> GenerateResultConfig:
    """Load YAML config and apply validated CLI overrides."""

    resolved_path = config_path or DEFAULT_VOL_RESULT_CONFIG_PATH
    path, payload = load_yaml_mapping(resolved_path)

    if "training" in payload or "generate_result" in payload:
        training_values = payload.get("training") or {}
        if training_values and not isinstance(training_values, dict):
            raise ValueError(f"Config section 'training' in {path} must contain a YAML mapping.")
        generate_values = payload.get("generate_result") or {}
        if generate_values and not isinstance(generate_values, dict):
            raise ValueError(f"Config section 'generate_result' in {path} must contain a YAML mapping.")
        return build_generate_result_config(
            training_values=training_values,
            generate_values=generate_values,
            overrides=overrides,
        )

    _, _, loaded_values = load_yaml_config_values(
        resolved_path,
        defaults=generate_result_config_to_dict(GenerateResultConfig()),
        overrides=overrides,
    )
    return GenerateResultConfig(**loaded_values)


def save_generate_result_config_yaml(config: GenerateResultConfig, output_path: str | Path) -> Path:
    """Persist resolved runtime config for reproducible result-generation runs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(generate_result_config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
    return path
