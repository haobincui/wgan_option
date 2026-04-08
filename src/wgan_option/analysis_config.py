"""Configuration model and loading helpers for analyze-error scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

DEFAULT_VOL_ANALYSIS_CONFIG_PATH = "configs/analyze_error/vol.yaml"
DEFAULT_SVI_ANALYSIS_CONFIG_PATH = "configs/analyze_error/svi.yaml"


@dataclass
class AnalysisConfig:
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

    # Sample selection
    split: str = "all"
    selection_mode: str = "all"
    sample_id: str = ""
    row_index: int = -1
    limit: int = 5

    # Outputs
    output_dir: str = "outputs/analyze_error"
    histogram_bins: int = 30
    save_mse_histogram: bool = True
    save_bootstrap_histogram: bool = True

    # Bootstrap
    bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    bootstrap_seed: int = 42
    save_bootstrap_distribution: bool = True

    # Fixed-grid reconstruction for SVI
    strike_bins: int = 16
    maturity_bins: int = 16
    moneyness_min: float = 0.7
    moneyness_max: float = 1.3
    maturity_min_days: int = 7
    maturity_max_days: int = 365


def analysis_config_to_dict(config: AnalysisConfig) -> Dict[str, Any]:
    """Convert strongly typed config object into plain dictionary."""

    return asdict(config)


def _validate_config_keys(data: Dict[str, Any], source: str) -> None:
    valid_keys = set(analysis_config_to_dict(AnalysisConfig()).keys())
    unknown_keys = sorted(set(data.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {source}: {unknown_keys}")


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{raw}'")


def _cast_override(raw_value: str, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return _parse_bool(raw_value)
    if isinstance(default_value, int):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, str):
        return raw_value
    raise TypeError(f"Unsupported override type: {type(default_value)}")


def parse_analysis_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into typed override dict."""

    defaults = analysis_config_to_dict(AnalysisConfig())
    overrides: Dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(
                f"Override '{item}' is invalid. Expected KEY=VALUE, "
                f"example: --set split=all"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in defaults:
            raise ValueError(f"Unknown override key: '{key}'")
        overrides[key] = _cast_override(raw_value, defaults[key])
    return overrides


def load_analysis_config(
    config_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> AnalysisConfig:
    """Load YAML config and apply validated CLI overrides."""

    resolved_path = config_path or DEFAULT_VOL_ANALYSIS_CONFIG_PATH
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")

    loaded_values = analysis_config_to_dict(AnalysisConfig())
    with path.open("r", encoding="utf-8") as handle:
        yaml_values = yaml.safe_load(handle) or {}
    if not isinstance(yaml_values, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {resolved_path}")
    _validate_config_keys(yaml_values, source=resolved_path)
    loaded_values.update(yaml_values)

    if overrides:
        _validate_config_keys(overrides, source="cli overrides")
        loaded_values.update(overrides)

    return AnalysisConfig(**loaded_values)


def save_analysis_config_yaml(config: AnalysisConfig, output_path: str | Path) -> Path:
    """Persist resolved runtime config for reproducible analyze-error runs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(analysis_config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
    return path
