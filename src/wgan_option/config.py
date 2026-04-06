"""Configuration model and loading helpers for training/evaluation scripts."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

DEFAULT_CONFIG_PATH = "configs/wgan/train_default.yaml"


@dataclass
class Config:
    # Data
    option_data_glob: str = "data/raw/option_data/*.csv.gz"
    news_embedding_path: str = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
    embedding_column: str = "HD_embedding"
    news_date_column: str = "PD"
    processed_cache_path: str = "data/processed/bond_option_dataset.pt"
    data_path: str = ""
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "hd"

    # Surface grid
    strike_bins: int = 16
    maturity_bins: int = 16
    moneyness_min: float = 0.7
    moneyness_max: float = 1.3
    maturity_min_days: int = 7
    maturity_max_days: int = 365
    prediction_horizon: int = 1
    train_ratio: float = 0.8
    min_samples_for_training: int = 2

    # Proxy implied volatility normalization range
    vol_floor: float = 0.03
    vol_cap: float = 1.50

    # Model dimensions
    channels: int = 1
    embedding_dim: int = 1024
    noise_dim: int = 32
    gen_hidden_dim: int = 512
    disc_hidden_dim: int = 256

    # Optimization
    learning_rate: float = 0.0002
    num_epochs: int = 100
    batch_size: int = 16
    beta_1: float = 0.5
    beta_2: float = 0.9
    discriminator_iter: int = 5
    lambda_gp: float = 10.0

    # Modified loss weights
    lambda_recon: float = 10.0
    lambda_calendar: float = 2.0
    lambda_butterfly: float = 2.0
    lambda_smooth: float = 0.1
    use_calendar_constraint: bool = True
    use_butterfly_constraint: bool = True
    use_smooth_constraint: bool = True
    count_loss_weight: float = 0.2

    # Runtime
    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0
    use_cache: bool = True
    max_slices: int = 4

    # Paths
    models_path: str = "outputs/checkpoints"
    outputs_path: str = "outputs/checkpoints"
    samples_path: str = "outputs/samples"
    metrics_path: str = "outputs/metrics"
    normalization_stats_path: str = "outputs/metrics/normalization_stats.json"
    save_every: int = 10

    # SVI regressor
    svi_hidden_dim: int = 256
    svi_dropout: float = 0.1


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert strongly-typed config object into plain dictionary."""
    return asdict(config)


def _validate_config_keys(data: Dict[str, Any], source: str):
    """Fail fast when unknown keys appear in yaml/CLI overrides."""
    valid_keys = set(config_to_dict(Config()).keys())
    unknown_keys = sorted(set(data.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {source}: {unknown_keys}")


def _parse_bool(raw: str) -> bool:
    """Parse permissive CLI boolean formats."""
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{raw}'")


def _cast_override(raw_value: str, default_value: Any) -> Any:
    """Cast CLI override string into the field type defined by defaults."""
    if isinstance(default_value, bool):
        return _parse_bool(raw_value)
    if isinstance(default_value, int):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, str):
        return raw_value
    raise TypeError(f"Unsupported override type: {type(default_value)}")


def parse_cli_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into typed override dict."""
    defaults = config_to_dict(Config())
    overrides: Dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(
                f"Override '{item}' is invalid. Expected KEY=VALUE, "
                f"example: --set batch_size=32"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in defaults:
            raise ValueError(f"Unknown override key: '{key}'")
        overrides[key] = _cast_override(raw_value, defaults[key])
    return overrides


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load YAML config and apply validated CLI overrides."""
    loaded_values = config_to_dict(Config())
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")

    with open(path, "r", encoding="utf-8") as f:
        yaml_values = yaml.safe_load(f) or {}
    if not isinstance(yaml_values, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {resolved_path}")
    _validate_config_keys(yaml_values, source=resolved_path)
    loaded_values.update(yaml_values)

    if overrides:
        _validate_config_keys(overrides, source="cli overrides")
        loaded_values.update(overrides)

    return Config(**loaded_values)


def save_config_yaml(config: Config, output_path: str):
    """Persist resolved config for reproducible experiment runs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False, allow_unicode=False)


default_config = Config()
