"""Configuration model and loading helpers for analyze-error scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml
from wgan_option.config_parsing import load_yaml_config_values, parse_typed_overrides
from wgan_option.surface_grid import (
    DEFAULT_MATURITY_BINS,
    DEFAULT_MATURITY_MAX_DAYS,
    DEFAULT_MATURITY_MIN_DAYS,
    DEFAULT_MONEYNESS_MAX,
    DEFAULT_MONEYNESS_MIN,
    DEFAULT_STRIKE_BINS,
)

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
    strike_bins: int = DEFAULT_STRIKE_BINS
    maturity_bins: int = DEFAULT_MATURITY_BINS
    moneyness_min: float = DEFAULT_MONEYNESS_MIN
    moneyness_max: float = DEFAULT_MONEYNESS_MAX
    maturity_min_days: int = DEFAULT_MATURITY_MIN_DAYS
    maturity_max_days: int = DEFAULT_MATURITY_MAX_DAYS


def analysis_config_to_dict(config: AnalysisConfig) -> Dict[str, Any]:
    """Convert strongly typed config object into plain dictionary."""

    return asdict(config)


def parse_analysis_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into typed override dict."""
    return parse_typed_overrides(
        override_items,
        defaults=analysis_config_to_dict(AnalysisConfig()),
        example="--set split=all",
    )


def load_analysis_config(
    config_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> AnalysisConfig:
    """Load YAML config and apply validated CLI overrides."""

    resolved_path = config_path or DEFAULT_VOL_ANALYSIS_CONFIG_PATH
    _, _, loaded_values = load_yaml_config_values(
        resolved_path,
        defaults=analysis_config_to_dict(AnalysisConfig()),
        overrides=overrides,
    )

    return AnalysisConfig(**loaded_values)


def save_analysis_config_yaml(config: AnalysisConfig, output_path: str | Path) -> Path:
    """Persist resolved runtime config for reproducible analyze-error runs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(analysis_config_to_dict(config), handle, sort_keys=False, allow_unicode=False)
    return path
