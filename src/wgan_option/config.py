"""Configuration model and loading helpers for training/evaluation scripts."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml
from wgan_option.config_parsing import load_yaml_config_values, parse_typed_overrides

DEFAULT_CONFIG_PATH = "configs/wgan/train_vol_xlsx.yaml"
LEGACY_OUTPUT_PATH_FIELDS = (
    "models_path",
    "outputs_path",
    "samples_path",
    "metrics_path",
    "normalization_stats_path",
)


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
    gen_base_channels: int = 32
    disc_base_channels: int = 32
    gen_res_blocks: int = 0
    disc_res_blocks: int = 0
    gen_text_hidden_dim: int = 256
    gen_text_out_dim: int = 128
    disc_text_hidden_dim: int = 128
    gen_hidden_dim: int = 512
    disc_hidden_dim: int = 256

    # Optimization
    learning_rate: float = 0.0002
    use_reduce_lr_on_plateau: bool = False
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 8
    reduce_lr_min_lr: float = 1e-5
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
    constraint_warmup_epochs: int = 0
    count_loss_weight: float = 0.2

    # Runtime
    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0
    use_cache: bool = True
    max_slices: int = 4

    # Paths
    output_root: str = ""
    models_path: str = "outputs/checkpoints"
    outputs_path: str = "outputs/checkpoints"
    samples_path: str = "outputs/samples"
    metrics_path: str = "outputs/metrics"
    normalization_stats_path: str = "outputs/metrics/normalization_stats.json"
    save_every: int = 10
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # SVI regressor
    svi_hidden_dim: int = 256
    svi_dropout: float = 0.1


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert strongly-typed config object into plain dictionary."""
    return asdict(config)


def derive_training_output_paths(output_root: str) -> Dict[str, Any]:
    """Expand one merged-xlsx output root into concrete artifact paths."""

    normalized_output_root = str(output_root).strip()
    if not normalized_output_root:
        raise ValueError("output_root must be a non-empty path when deriving training output paths.")

    root = Path(normalized_output_root)
    return {
        "models_path": str(root / "checkpoints"),
        "outputs_path": str(root / "checkpoints"),
        "samples_path": str(root / "samples"),
        "metrics_path": str(root / "metrics"),
        "normalization_stats_path": str(root / "metrics" / "normalization_stats.json"),
    }


def _apply_output_root(
    loaded_values: Dict[str, Any],
    *,
    yaml_values: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the single output_root entrypoint into concrete internal paths."""

    output_root = str(loaded_values.get("output_root", "")).strip()
    if not output_root:
        return loaded_values

    explicit_legacy_fields = sorted(
        field_name
        for field_name in LEGACY_OUTPUT_PATH_FIELDS
        if field_name in yaml_values or (overrides is not None and field_name in overrides)
    )
    if explicit_legacy_fields:
        raise ValueError(
            "output_root cannot be combined with explicit legacy path fields. "
            f"Remove these fields and keep only output_root: {explicit_legacy_fields}"
        )

    loaded_values["output_root"] = output_root
    loaded_values.update(derive_training_output_paths(output_root))
    return loaded_values


def parse_cli_overrides(override_items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` items into typed override dict."""
    return parse_typed_overrides(
        override_items,
        defaults=config_to_dict(Config()),
        example="--set batch_size=32",
    )


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load YAML config and apply validated CLI overrides."""
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    _, yaml_values, loaded_values = load_yaml_config_values(
        resolved_path,
        defaults=config_to_dict(Config()),
        overrides=overrides,
    )

    loaded_values = _apply_output_root(
        loaded_values,
        yaml_values=yaml_values,
        overrides=overrides,
    )

    return Config(**loaded_values)


def save_config_yaml(config: Config, output_path: str):
    """Persist resolved config for reproducible experiment runs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False, allow_unicode=False)


default_config = Config()
