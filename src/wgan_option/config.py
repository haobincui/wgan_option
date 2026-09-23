"""Configuration model and loading helpers for training/evaluation scripts."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import yaml
from wgan_option.config_parsing import load_yaml_config_values, parse_typed_overrides
from wgan_option.models.common import (
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GAUSSIAN_GENERATOR_NOISE_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LEGACY_RESIDUAL_OUTPUT_MODE,
    LEGACY_CRITIC_NORMALIZATION_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    normalize_critic_conditioning_mode,
    normalize_critic_normalization_mode,
    normalize_generator_conditioning_mode,
    normalize_generator_current_input_mode,
    normalize_generator_noise_mode,
    normalize_residual_output_mode,
)
from wgan_option.utils.news_first_experiment_core import (
    NO_FULL_TRAINING_STATE,
    NO_PAIR_TEXT_OVERLAY,
    RESUME_DYNAMIC_FULL_TRAINING_STATE,
    RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    SAVE_DYNAMIC_FULL_TRAINING_STATE,
    normalize_full_training_state_mode,
    normalize_pair_text_overlay_mode,
)

DEFAULT_CONFIG_PATH = "configs/wgan/train_vol_xlsx.yaml"
LEGACY_OUTPUT_PATH_FIELDS = (
    "models_path",
    "outputs_path",
    "samples_path",
    "metrics_path",
    "normalization_stats_path",
)
LABEL_RELIABILITY_MODES = frozenset(
    {
        "none",
        "support_filter",
        "soft_weight",
        "support_filter_soft_weight",
    }
)
NO_REFIT_MODE = "none"
FROZEN_EPOCH_LR_REPLAY_REFIT_MODE = "frozen_epoch_lr_replay_v1"
REFIT_MODES = frozenset({NO_REFIT_MODE, FROZEN_EPOCH_LR_REPLAY_REFIT_MODE})


def normalize_refit_mode(value: object) -> str:
    """Return the persisted Stage-B refit contract slug."""

    mode = str(value or NO_REFIT_MODE).strip().lower()
    if mode not in REFIT_MODES:
        raise ValueError(
            f"news_first_refit_mode must be one of {sorted(REFIT_MODES)}, got {mode!r}"
        )
    return mode


def _validated_refit_lr_trace(
    value: object,
    *,
    field_name: str,
    num_epochs: int,
) -> list[dict[str, float | int]]:
    if not isinstance(value, list) or len(value) != int(num_epochs):
        raise ValueError(f"{field_name} must contain exactly {num_epochs} rows")
    result: list[dict[str, float | int]] = []
    for expected_epoch, raw in enumerate(value, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field_name}[{expected_epoch - 1}] must be a mapping")
        if int(raw.get("epoch", -1)) != expected_epoch:
            raise ValueError(f"{field_name} epochs must be exactly 1..{num_epochs}")
        learning_rate = float(raw.get("lr", float("nan")))
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"{field_name} learning rates must be finite and positive")
        result.append({"epoch": expected_epoch, "lr": learning_rate})
    return result


def load_refit_recipe(path: str, expected_sha256: str) -> Dict[str, Any]:
    """Load and hash-validate one immutable fixed-epoch LR replay recipe."""

    recipe_path = Path(str(path)).expanduser()
    payload_bytes = recipe_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != str(expected_sha256).strip().lower():
        raise ValueError(
            "news_first_refit_recipe_sha256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    try:
        raw = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Refit recipe must be valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Refit recipe root must be a JSON object")
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("Refit recipe schema_version must be 1")
    if normalize_refit_mode(raw.get("refit_mode")) != FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
        raise ValueError("Refit recipe refit_mode must be frozen_epoch_lr_replay_v1")
    num_epochs = int(raw.get("num_epochs", 0))
    if num_epochs <= 0:
        raise ValueError("Refit recipe num_epochs must be positive")
    return {
        "schema_version": 1,
        "refit_mode": FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
        "num_epochs": num_epochs,
        "generator_lr_trace": _validated_refit_lr_trace(
            raw.get("generator_lr_trace"),
            field_name="generator_lr_trace",
            num_epochs=num_epochs,
        ),
        "discriminator_lr_trace": _validated_refit_lr_trace(
            raw.get("discriminator_lr_trace"),
            field_name="discriminator_lr_trace",
            num_epochs=num_epochs,
        ),
    }


def normalize_label_reliability_mode(value: object) -> str:
    """Return the stable persisted slug for a label-reliability arm."""

    mode = str(value or "none").strip().lower()
    if mode not in LABEL_RELIABILITY_MODES:
        raise ValueError(
            "news_first_label_reliability_mode must be one of "
            f"{sorted(LABEL_RELIABILITY_MODES)}, got: {mode}"
        )
    return mode


def label_reliability_lineage(config: "Config") -> Dict[str, Any]:
    """Return the persisted train-label contract with legacy-safe defaults."""

    return {
        "news_first_label_reliability_mode": normalize_label_reliability_mode(
            getattr(config, "news_first_label_reliability_mode", "none")
        ),
        "news_first_label_reliability_manifest_path": str(
            getattr(config, "news_first_label_reliability_manifest_path", "") or ""
        ),
        "news_first_label_reliability_manifest_sha256": str(
            getattr(config, "news_first_label_reliability_manifest_sha256", "") or ""
        ),
        "news_first_label_reliability_profile_sha256": str(
            getattr(config, "news_first_label_reliability_profile_sha256", "") or ""
        ),
        "news_first_label_reliability_fold_id": str(
            getattr(config, "news_first_label_reliability_fold_id", "") or ""
        ),
        "news_first_label_reliability_train_pair_universe_sha256": str(
            getattr(
                config,
                "news_first_label_reliability_train_pair_universe_sha256",
                "",
            )
            or ""
        ),
        "news_first_label_reliability_train_data_sha256": str(
            getattr(
                config,
                "news_first_label_reliability_train_data_sha256",
                "",
            )
            or ""
        ),
        "news_first_label_reliability_validation_data_sha256": str(
            getattr(
                config,
                "news_first_label_reliability_validation_data_sha256",
                "",
            )
            or ""
        ),
        "news_first_label_reliability_data_window_contract_sha256": str(
            getattr(
                config,
                "news_first_label_reliability_data_window_contract_sha256",
                "",
            )
            or ""
        ),
        "data_path": str(getattr(config, "data_path", "") or ""),
        "news_first_common_eval_data_path": str(
            getattr(config, "news_first_common_eval_data_path", "") or ""
        ),
        "news_first_train_end_utc": str(
            getattr(config, "news_first_train_end_utc", "") or ""
        ),
        "news_first_validation_end_utc": str(
            getattr(config, "news_first_validation_end_utc", "") or ""
        ),
        "news_first_materialize_test_loader": bool(
            getattr(config, "news_first_materialize_test_loader", True)
        ),
        "news_first_materialize_validation_loader": bool(
            getattr(config, "news_first_materialize_validation_loader", True)
        ),
    }


@dataclass
class Config:
    # Data
    option_data_glob: str = "data/raw/option_data/*.csv.gz"
    news_embedding_path: str = (
        "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
    )
    embedding_column: str = "HD_embedding"
    news_date_column: str = "PD"
    processed_cache_path: str = "data/processed/bond_option_dataset.pt"
    data_path: str = ""
    sheet_name: str = "gan_input_ready"
    text_embedding_mode: str = "hd"
    news_first_common_eval_data_path: str = ""
    news_first_train_end_utc: str = "2023-07-01T00:00:00Z"
    news_first_validation_end_utc: str = "2023-10-01T00:00:00Z"
    support_mask_mode: str = "none"
    news_first_dataset_tolerance_minutes: int = 0
    news_first_text_ablation_mode: str = "real_text"
    news_first_text_information_path: str = "current_surface_plus_real_lp_embedding"
    news_first_text_shuffle_seed: int = 42
    news_first_pair_text_overlay_mode: str = NO_PAIR_TEXT_OVERLAY
    news_first_pair_text_manifest_path: str = ""
    news_first_pair_text_manifest_sha256: str = ""
    news_first_pair_text_profile_sha256: str = ""
    news_first_capacity_profile: str = "default"
    news_first_capacity_profile_sha256: str = ""
    news_first_architecture_profile_sha256: str = ""
    news_first_model_contract_sha256: str = ""
    news_first_surface_grid_profile: str = ""
    news_first_surface_grid_sha256: str = ""
    news_first_capacity_seed_profile_sha256: str = ""
    news_first_lr_profile: str = "default"
    news_first_lr_profile_sha256: str = ""
    news_first_fixed_learning_rate_profile: str = "default"
    news_first_fixed_learning_rate_profile_sha256: str = ""
    news_first_label_reliability_mode: str = "none"
    news_first_label_reliability_manifest_path: str = ""
    news_first_label_reliability_manifest_sha256: str = ""
    news_first_label_reliability_profile_sha256: str = ""
    news_first_label_reliability_fold_id: str = ""
    news_first_label_reliability_train_pair_universe_sha256: str = ""
    news_first_label_reliability_train_data_sha256: str = ""
    news_first_label_reliability_validation_data_sha256: str = ""
    news_first_label_reliability_data_window_contract_sha256: str = ""
    news_first_refit_mode: str = NO_REFIT_MODE
    news_first_refit_recipe_path: str = ""
    news_first_refit_recipe_sha256: str = ""
    news_first_full_training_state_mode: str = NO_FULL_TRAINING_STATE
    news_first_full_training_state_contract_path: str = ""
    news_first_full_training_state_contract_sha256: str = ""
    # Branch-local, weights-only initialization used by the Pure-CNN -> FiLM
    # text-effectiveness experiment.  Unlike ``full_training_state`` resume,
    # this contract deliberately never restores Adam or scheduler state.
    news_first_graft_state_path: str = ""
    news_first_graft_state_sha256: str = ""
    # Optional learned-epoch checkpoints retained for validation-trajectory
    # diagnostics.  Epoch zero is already represented by the immutable
    # ``initial_epoch0`` checkpoint and is therefore not listed here.
    news_first_validation_snapshot_epochs: tuple[int, ...] = ()
    news_first_materialize_validation_loader: bool = True
    news_first_materialize_test_loader: bool = True
    # Internal half-open input window applied before surface/text deserialization.
    # Formal reliability jobs derive these bounds from their persisted fold split.
    news_first_data_window_start_utc_inclusive: str = ""
    news_first_data_window_end_utc_exclusive: str = ""

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
    generator_noise_mode: str = GAUSSIAN_GENERATOR_NOISE_MODE
    generator_current_input_mode: str = FULL_CURRENT_GENERATOR_INPUT_MODE
    generator_conditioning_mode: str = BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
    critic_conditioning_mode: str = LP_CONCAT_CRITIC_CONDITIONING_MODE
    critic_normalization_mode: str = LEGACY_CRITIC_NORMALIZATION_MODE
    gen_base_channels: int = 32
    disc_base_channels: int = 32
    gen_res_blocks: int = 0
    disc_res_blocks: int = 0
    gen_text_hidden_dim: int = 256
    gen_text_out_dim: int = 128
    disc_text_hidden_dim: int = 128
    gen_hidden_dim: int = 512
    disc_hidden_dim: int = 256
    residual_output_mode: str = LEGACY_RESIDUAL_OUTPUT_MODE

    # Optimization
    learning_rate: float = 0.0002
    generator_learning_rate: float = 0.0
    generator_optimizer_profile: str = "uniform_v1"
    generator_text_learning_rate: float = 0.0
    generator_film_learning_rate: float = 0.0
    generator_text_min_learning_rate: float = 0.0
    generator_film_min_learning_rate: float = 0.0
    discriminator_learning_rate: float = 0.0
    lr_scheduler_type: str = "none"
    lr_warmup_epochs: int = 0
    lr_warmup_start_factor: float = 0.1
    gradient_accumulation_steps: int = 1
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
    lambda_delta_shrink: float = 0.0
    use_calendar_constraint: bool = True
    use_butterfly_constraint: bool = True
    use_smooth_constraint: bool = True
    constraint_warmup_epochs: int = 0
    best_checkpoint_metric: str = "val_recon"
    baseline_penalty_weight: float = 2.0
    count_loss_weight: float = 0.2

    # Runtime
    seed: int = 42
    cuda: bool = torch.cuda.is_available()
    num_workers: int = 0
    validation_mc_samples: int = 1
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
    early_stopping_min_epochs: int = 0
    evaluate_initial_checkpoint: bool = False

    # SVI regressor
    svi_hidden_dim: int = 256
    svi_dropout: float = 0.1

    # Branch-local Generator architecture and generic conditioning-LR contract.
    # Appended to preserve the positional order of all pre-existing fields.
    gen_crossattn_heads: int = 4
    gen_crossattn_text_tokens: int = 4
    gen_crossattn_dim: int = 128
    gen_transformer_model_dim: int = 96
    gen_transformer_layers: int = 4
    gen_transformer_heads: int = 8
    gen_transformer_ffn_dim: int = 384
    gen_transformer_dropout: float = 0.1
    gen_style_dim: int = 128
    gen_style_demodulate: bool = True
    generator_conditioning_learning_rate: float = 0.0
    generator_conditioning_min_learning_rate: float = 0.0

    def _validate_generator_architecture_fields(self) -> None:
        mode = self.generator_conditioning_mode
        if mode == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            self.gen_crossattn_heads = int(self.gen_crossattn_heads)
            self.gen_crossattn_text_tokens = int(self.gen_crossattn_text_tokens)
            self.gen_crossattn_dim = int(self.gen_crossattn_dim)
            if self.gen_crossattn_heads <= 0 or self.gen_crossattn_text_tokens <= 0:
                raise ValueError(
                    "Cross-attention heads and text-token count must be positive"
                )
            if (
                self.gen_crossattn_dim <= 0
                or self.gen_crossattn_dim % self.gen_crossattn_heads
            ):
                raise ValueError(
                    "gen_crossattn_dim must be divisible by gen_crossattn_heads"
                )
        elif mode == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            self.gen_transformer_model_dim = int(self.gen_transformer_model_dim)
            self.gen_transformer_layers = int(self.gen_transformer_layers)
            self.gen_transformer_heads = int(self.gen_transformer_heads)
            self.gen_transformer_ffn_dim = int(self.gen_transformer_ffn_dim)
            self.gen_transformer_dropout = float(self.gen_transformer_dropout)
            if self.gen_transformer_model_dim <= 0 or self.gen_transformer_heads <= 0:
                raise ValueError("Transformer dimensions must be positive")
            if self.gen_transformer_model_dim % self.gen_transformer_heads:
                raise ValueError(
                    "gen_transformer_model_dim must be divisible by "
                    "gen_transformer_heads"
                )
            if self.gen_transformer_layers <= 0 or self.gen_transformer_ffn_dim <= 0:
                raise ValueError(
                    "Transformer layers and FFN dimension must be positive"
                )
            if not 0.0 <= self.gen_transformer_dropout < 1.0:
                raise ValueError("gen_transformer_dropout must be in [0, 1)")
        elif mode == STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            self.gen_style_dim = int(self.gen_style_dim)
            if self.gen_style_dim <= 0:
                raise ValueError("gen_style_dim must be positive")
            if not isinstance(self.gen_style_demodulate, bool):
                raise ValueError("gen_style_demodulate must be a boolean")
            if not self.gen_style_demodulate:
                raise ValueError(
                    "stylemod_unet_mask_coords_v1 requires demodulated convolutions"
                )

    def __post_init__(self) -> None:
        self.generator_noise_mode = normalize_generator_noise_mode(
            self.generator_noise_mode
        )
        self.generator_current_input_mode = normalize_generator_current_input_mode(
            self.generator_current_input_mode
        )
        self.generator_conditioning_mode = normalize_generator_conditioning_mode(
            self.generator_conditioning_mode
        )
        self.critic_conditioning_mode = normalize_critic_conditioning_mode(
            self.critic_conditioning_mode
        )
        self.critic_normalization_mode = normalize_critic_normalization_mode(
            self.critic_normalization_mode
        )
        self.residual_output_mode = normalize_residual_output_mode(
            self.residual_output_mode
        )
        architecture_modes = {
            CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }
        self.support_mask_mode = str(self.support_mask_mode or "none").strip().lower()
        if self.support_mask_mode not in {"none", "raw_joint"}:
            raise ValueError(
                "support_mask_mode must be one of ['none', 'raw_joint'], got: "
                f"{self.support_mask_mode}"
            )
        if (
            self.generator_current_input_mode
            == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            and self.support_mask_mode != "raw_joint"
        ):
            raise ValueError(
                "generator_current_input_mode='current_support_masked' requires "
                "support_mask_mode='raw_joint' so an independently derived "
                "current_support_mask is available"
            )
        if (
            self.generator_conditioning_mode
            in {
                FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
                FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                *architecture_modes,
            }
            and int(self.gen_res_blocks) != 0
        ):
            raise ValueError(
                f"{self.generator_conditioning_mode} requires gen_res_blocks=0"
            )
        if (
            self.generator_conditioning_mode
            in {
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                *architecture_modes,
            }
            and self.generator_current_input_mode
            != CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        ):
            raise ValueError(
                f"{self.generator_conditioning_mode} requires "
                "generator_current_input_mode='current_support_masked'"
            )
        if self.generator_conditioning_mode in architecture_modes:
            if int(self.channels) != 1:
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires channels=1"
                )
            if (int(self.maturity_bins), int(self.strike_bins)) != (16, 16):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires an exact 16x16 grid"
                )
            if (
                self.generator_noise_mode != GAUSSIAN_GENERATOR_NOISE_MODE
                or int(self.noise_dim) != 32
            ):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires Gaussian32"
                )
            if self.residual_output_mode != IDENTITY_RESIDUAL_OUTPUT_MODE:
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires identity residual output"
                )
            self._validate_generator_architecture_fields()
        self.news_first_pair_text_overlay_mode = normalize_pair_text_overlay_mode(
            self.news_first_pair_text_overlay_mode
        )
        self.news_first_pair_text_manifest_path = str(
            self.news_first_pair_text_manifest_path or ""
        ).strip()
        self.news_first_pair_text_manifest_sha256 = (
            str(self.news_first_pair_text_manifest_sha256 or "").strip().lower()
        )
        self.news_first_pair_text_profile_sha256 = (
            str(self.news_first_pair_text_profile_sha256 or "").strip().lower()
        )
        pair_text_values = {
            "news_first_pair_text_manifest_path": self.news_first_pair_text_manifest_path,
            "news_first_pair_text_manifest_sha256": (
                self.news_first_pair_text_manifest_sha256
            ),
            "news_first_pair_text_profile_sha256": (
                self.news_first_pair_text_profile_sha256
            ),
        }
        if self.news_first_pair_text_overlay_mode == NO_PAIR_TEXT_OVERLAY:
            if any(pair_text_values.values()):
                raise ValueError(
                    "Pair-text manifest fields require a non-none "
                    "news_first_pair_text_overlay_mode"
                )
        else:
            missing_pair_text = sorted(
                name for name, value in pair_text_values.items() if not value
            )
            if missing_pair_text:
                raise ValueError(
                    "Pair-text overlay requires complete manifest lineage; "
                    f"missing: {missing_pair_text}"
                )
            for field_name in (
                "news_first_pair_text_manifest_sha256",
                "news_first_pair_text_profile_sha256",
            ):
                if (
                    re.fullmatch(r"[0-9a-f]{64}", str(getattr(self, field_name)))
                    is None
                ):
                    raise ValueError(
                        f"{field_name} must be a lowercase 64-character SHA256 digest"
                    )
            if int(self.embedding_dim) != 1024:
                raise ValueError("Pair-text overlays require embedding_dim=1024")
            if str(self.news_first_text_ablation_mode).strip().lower() != "real_text":
                raise ValueError(
                    "Pair-text overlay owns the text treatment and requires "
                    "news_first_text_ablation_mode='real_text'"
                )
        self.news_first_capacity_profile = str(
            self.news_first_capacity_profile or "default"
        ).strip()
        if not self.news_first_capacity_profile:
            self.news_first_capacity_profile = "default"
        self.news_first_capacity_profile_sha256 = str(
            self.news_first_capacity_profile_sha256 or ""
        ).strip()
        self.news_first_capacity_seed_profile_sha256 = str(
            self.news_first_capacity_seed_profile_sha256 or ""
        ).strip()
        self.news_first_lr_profile = str(
            self.news_first_lr_profile or "default"
        ).strip()
        if not self.news_first_lr_profile:
            self.news_first_lr_profile = "default"
        self.news_first_lr_profile_sha256 = str(
            self.news_first_lr_profile_sha256 or ""
        ).strip()
        self.news_first_fixed_learning_rate_profile = str(
            self.news_first_fixed_learning_rate_profile or "default"
        ).strip()
        if not self.news_first_fixed_learning_rate_profile:
            self.news_first_fixed_learning_rate_profile = "default"
        self.news_first_fixed_learning_rate_profile_sha256 = str(
            self.news_first_fixed_learning_rate_profile_sha256 or ""
        ).strip()
        self.news_first_label_reliability_mode = normalize_label_reliability_mode(
            self.news_first_label_reliability_mode
        )
        self.news_first_label_reliability_manifest_path = str(
            self.news_first_label_reliability_manifest_path or ""
        ).strip()
        self.news_first_label_reliability_manifest_sha256 = (
            str(self.news_first_label_reliability_manifest_sha256 or "").strip().lower()
        )
        self.news_first_label_reliability_profile_sha256 = (
            str(self.news_first_label_reliability_profile_sha256 or "").strip().lower()
        )
        self.news_first_label_reliability_fold_id = str(
            self.news_first_label_reliability_fold_id or ""
        ).strip()
        self.news_first_label_reliability_train_pair_universe_sha256 = (
            str(self.news_first_label_reliability_train_pair_universe_sha256 or "")
            .strip()
            .lower()
        )
        self.news_first_label_reliability_train_data_sha256 = (
            str(self.news_first_label_reliability_train_data_sha256 or "")
            .strip()
            .lower()
        )
        self.news_first_label_reliability_validation_data_sha256 = (
            str(self.news_first_label_reliability_validation_data_sha256 or "")
            .strip()
            .lower()
        )
        self.news_first_label_reliability_data_window_contract_sha256 = (
            str(self.news_first_label_reliability_data_window_contract_sha256 or "")
            .strip()
            .lower()
        )
        self.news_first_data_window_start_utc_inclusive = str(
            self.news_first_data_window_start_utc_inclusive or ""
        ).strip()
        self.news_first_data_window_end_utc_exclusive = str(
            self.news_first_data_window_end_utc_exclusive or ""
        ).strip()
        self.news_first_refit_mode = normalize_refit_mode(self.news_first_refit_mode)
        self.news_first_refit_recipe_path = str(
            self.news_first_refit_recipe_path or ""
        ).strip()
        self.news_first_refit_recipe_sha256 = (
            str(self.news_first_refit_recipe_sha256 or "").strip().lower()
        )
        self.news_first_full_training_state_mode = normalize_full_training_state_mode(
            self.news_first_full_training_state_mode
        )
        self.news_first_full_training_state_contract_path = str(
            self.news_first_full_training_state_contract_path or ""
        ).strip()
        self.news_first_full_training_state_contract_sha256 = (
            str(self.news_first_full_training_state_contract_sha256 or "")
            .strip()
            .lower()
        )
        self.news_first_graft_state_path = str(
            self.news_first_graft_state_path or ""
        ).strip()
        self.news_first_graft_state_sha256 = (
            str(self.news_first_graft_state_sha256 or "").strip().lower()
        )
        self.news_first_validation_snapshot_epochs = tuple(
            int(epoch) for epoch in (self.news_first_validation_snapshot_epochs or ())
        )
        if tuple(sorted(set(self.news_first_validation_snapshot_epochs))) != (
            self.news_first_validation_snapshot_epochs
        ):
            raise ValueError(
                "news_first_validation_snapshot_epochs must be sorted and unique"
            )
        if any(
            epoch < 1 or epoch > int(self.num_epochs)
            for epoch in self.news_first_validation_snapshot_epochs
        ):
            raise ValueError(
                "news_first_validation_snapshot_epochs must fall in [1, num_epochs]"
            )
        self.generator_optimizer_profile = str(
            self.generator_optimizer_profile or "uniform_v1"
        ).strip()
        if self.generator_optimizer_profile not in {
            "uniform_v1",
            "film_unet_split_lr_v1",
            "conditioning_split_lr_v2",
        }:
            raise ValueError(
                "generator_optimizer_profile must be one of "
                "['conditioning_split_lr_v2', 'film_unet_split_lr_v1', "
                "'uniform_v1']"
            )
        for field_name in (
            "generator_text_learning_rate",
            "generator_film_learning_rate",
            "generator_text_min_learning_rate",
            "generator_film_min_learning_rate",
            "generator_conditioning_learning_rate",
            "generator_conditioning_min_learning_rate",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            setattr(self, field_name, value)
        split_lr_values = (
            self.generator_text_learning_rate,
            self.generator_film_learning_rate,
            self.generator_text_min_learning_rate,
            self.generator_film_min_learning_rate,
            self.generator_conditioning_learning_rate,
            self.generator_conditioning_min_learning_rate,
        )
        if self.generator_optimizer_profile == "uniform_v1":
            if any(value != 0.0 for value in split_lr_values):
                raise ValueError(
                    "Generator split learning-rate fields require a split-LR "
                    "generator_optimizer_profile"
                )
        elif self.generator_optimizer_profile == "film_unet_split_lr_v1":
            if self.generator_conditioning_mode not in {
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            }:
                raise ValueError(
                    "film_unet_split_lr_v1 requires a FiLM U-Net generator"
                )
            if (
                self.generator_text_learning_rate <= 0.0
                or self.generator_film_learning_rate <= 0.0
            ):
                raise ValueError(
                    "film_unet_split_lr_v1 requires positive generator text and "
                    "FiLM learning rates"
                )
            if (
                self.generator_conditioning_learning_rate != 0.0
                or self.generator_conditioning_min_learning_rate != 0.0
            ):
                raise ValueError(
                    "film_unet_split_lr_v1 does not accept generic conditioning "
                    "learning-rate fields"
                )
        else:
            supported_v2_modes = {
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            }
            if self.generator_conditioning_mode not in supported_v2_modes:
                raise ValueError(
                    "conditioning_split_lr_v2 requires a supported conditional "
                    "Generator"
                )
            if self.generator_text_learning_rate <= 0.0:
                raise ValueError(
                    "conditioning_split_lr_v2 requires a positive generator text "
                    "learning rate"
                )
            backbone_learning_rate = float(self.generator_learning_rate) or float(
                self.learning_rate
            )
            conditioning_learning_rate = (
                self.generator_conditioning_learning_rate
                or self.generator_film_learning_rate
                or backbone_learning_rate
            )
            if not math.isfinite(conditioning_learning_rate) or (
                conditioning_learning_rate <= 0.0
            ):
                raise ValueError(
                    "conditioning_split_lr_v2 requires a positive resolved "
                    "conditioning learning rate"
                )
        self.lr_warmup_epochs = int(self.lr_warmup_epochs)
        self.lr_warmup_start_factor = float(self.lr_warmup_start_factor)
        if self.lr_warmup_epochs < 0:
            raise ValueError("lr_warmup_epochs must be non-negative")
        if not math.isfinite(self.lr_warmup_start_factor) or not (
            0.0 < self.lr_warmup_start_factor <= 1.0
        ):
            raise ValueError(
                "lr_warmup_start_factor must be finite and in the interval (0, 1]"
            )
        if self.lr_warmup_epochs > int(self.num_epochs):
            raise ValueError("lr_warmup_epochs cannot exceed num_epochs")
        configured_scheduler = str(self.lr_scheduler_type).strip().lower()
        if configured_scheduler == "none" and self.use_reduce_lr_on_plateau:
            configured_scheduler = "plateau"
        if self.lr_warmup_epochs > 0 and configured_scheduler == "cosine":
            raise ValueError(
                "Learning-rate warmup currently supports plateau or no scheduler, "
                "not cosine"
            )
        if self.lr_warmup_epochs > 0 and self.news_first_refit_mode != NO_REFIT_MODE:
            raise ValueError(
                "Learning-rate warmup is incompatible with frozen refit LR replay"
            )
        if self.lr_warmup_epochs > 0 and self.news_first_full_training_state_mode in {
            RESUME_DYNAMIC_FULL_TRAINING_STATE,
            RESUME_FROZEN_LR_FULL_TRAINING_STATE,
        }:
            raise ValueError(
                "Learning-rate warmup cannot be repeated during full-state continuation"
            )
        if self.news_first_full_training_state_mode == NO_FULL_TRAINING_STATE:
            if (
                self.news_first_full_training_state_contract_path
                or self.news_first_full_training_state_contract_sha256
            ):
                raise ValueError(
                    "Full-training-state contract fields require a non-none mode"
                )
        else:
            if (
                not self.news_first_full_training_state_contract_path
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    self.news_first_full_training_state_contract_sha256,
                )
                is None
            ):
                raise ValueError(
                    "Full-training-state mode requires a contract path and lowercase "
                    "64-character contract SHA256"
                )
            if (
                self.news_first_full_training_state_mode
                in {
                    RESUME_DYNAMIC_FULL_TRAINING_STATE,
                    RESUME_FROZEN_LR_FULL_TRAINING_STATE,
                }
                and not self.news_first_pair_text_manifest_sha256
            ):
                raise ValueError(
                    "Full-state continuation requires a hash-bound pair-text manifest"
                )
        if bool(self.news_first_graft_state_path) != bool(
            self.news_first_graft_state_sha256
        ):
            raise ValueError(
                "news_first_graft_state_path and SHA256 must be configured together"
            )
        if (
            self.news_first_graft_state_sha256
            and re.fullmatch(r"[0-9a-f]{64}", self.news_first_graft_state_sha256)
            is None
        ):
            raise ValueError(
                "news_first_graft_state_sha256 must be a lowercase 64-character SHA256"
            )
        if (
            self.news_first_graft_state_path
            and self.news_first_full_training_state_mode
            in {
                RESUME_DYNAMIC_FULL_TRAINING_STATE,
                RESUME_FROZEN_LR_FULL_TRAINING_STATE,
            }
        ):
            raise ValueError(
                "A graft initialization cannot be combined with full-state resume"
            )
        if self.news_first_refit_mode == FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
            if not self.news_first_refit_recipe_path:
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 requires news_first_refit_recipe_path"
                )
            if (
                re.fullmatch(r"[0-9a-f]{64}", self.news_first_refit_recipe_sha256)
                is None
            ):
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 requires a lowercase 64-character "
                    "news_first_refit_recipe_sha256"
                )
            if bool(self.news_first_materialize_validation_loader):
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 requires "
                    "news_first_materialize_validation_loader=false"
                )
            if bool(self.news_first_materialize_test_loader):
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 requires "
                    "news_first_materialize_test_loader=false"
                )
            if str(self.lr_scheduler_type).strip().lower() != "none":
                raise ValueError("frozen_epoch_lr_replay_v1 forbids lr_scheduler_type")
            if bool(self.use_reduce_lr_on_plateau):
                raise ValueError("frozen_epoch_lr_replay_v1 forbids ReduceLROnPlateau")
            if bool(self.use_early_stopping):
                raise ValueError("frozen_epoch_lr_replay_v1 forbids early stopping")
            if bool(self.evaluate_initial_checkpoint):
                raise ValueError(
                    "frozen_epoch_lr_replay_v1 forbids initial validation evaluation"
                )
        elif self.news_first_refit_recipe_path or self.news_first_refit_recipe_sha256:
            raise ValueError(
                "news_first_refit_recipe_path/SHA require "
                "news_first_refit_mode=frozen_epoch_lr_replay_v1"
            )
        if (
            self.news_first_full_training_state_mode
            == RESUME_FROZEN_LR_FULL_TRAINING_STATE
            and self.news_first_refit_mode != FROZEN_EPOCH_LR_REPLAY_REFIT_MODE
        ):
            raise ValueError(
                "resume_frozen_lr_replay_v1 requires "
                "news_first_refit_mode=frozen_epoch_lr_replay_v1"
            )
        if (
            self.news_first_full_training_state_mode
            == RESUME_DYNAMIC_FULL_TRAINING_STATE
            and self.news_first_refit_mode != NO_REFIT_MODE
        ):
            raise ValueError("resume_dynamic_v1 requires news_first_refit_mode=none")
        if self.news_first_full_training_state_mode in {
            SAVE_DYNAMIC_FULL_TRAINING_STATE,
            RESUME_DYNAMIC_FULL_TRAINING_STATE,
        }:
            configured_scheduler = str(self.lr_scheduler_type).strip().lower()
            if configured_scheduler == "none" and self.use_reduce_lr_on_plateau:
                configured_scheduler = "plateau"
            if configured_scheduler != "plateau":
                raise ValueError(
                    "Dynamic full-training-state modes require "
                    "ReduceLROnPlateau for both optimizers"
                )
            if not self.news_first_materialize_validation_loader:
                raise ValueError(
                    "Dynamic full-training-state modes require a validation loader"
                )
        reliability_lineage = {
            "news_first_label_reliability_manifest_path": (
                self.news_first_label_reliability_manifest_path
            ),
            "news_first_label_reliability_manifest_sha256": (
                self.news_first_label_reliability_manifest_sha256
            ),
            "news_first_label_reliability_profile_sha256": (
                self.news_first_label_reliability_profile_sha256
            ),
            "news_first_label_reliability_fold_id": (
                self.news_first_label_reliability_fold_id
            ),
            "news_first_label_reliability_train_pair_universe_sha256": (
                self.news_first_label_reliability_train_pair_universe_sha256
            ),
            "news_first_label_reliability_train_data_sha256": (
                self.news_first_label_reliability_train_data_sha256
            ),
            "news_first_label_reliability_validation_data_sha256": (
                self.news_first_label_reliability_validation_data_sha256
            ),
            "news_first_label_reliability_data_window_contract_sha256": (
                self.news_first_label_reliability_data_window_contract_sha256
            ),
        }
        has_reliability_lineage = any(reliability_lineage.values())
        if self.news_first_label_reliability_mode != "none":
            has_reliability_lineage = True
        if has_reliability_lineage:
            missing = sorted(
                name for name, value in reliability_lineage.items() if not value
            )
            if missing:
                raise ValueError(
                    "Label-reliability experiments require complete manifest "
                    f"lineage; missing: {missing}"
                )
            for field_name in (
                "news_first_label_reliability_manifest_sha256",
                "news_first_label_reliability_profile_sha256",
                "news_first_label_reliability_train_pair_universe_sha256",
                "news_first_label_reliability_train_data_sha256",
                "news_first_label_reliability_validation_data_sha256",
                "news_first_label_reliability_data_window_contract_sha256",
            ):
                digest = str(getattr(self, field_name))
                if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError(
                        f"{field_name} must be a lowercase 64-character SHA256 digest"
                    )
            if self.support_mask_mode != "raw_joint":
                raise ValueError(
                    "Label-reliability experiments require "
                    "support_mask_mode='raw_joint'."
                )
            if int(self.news_first_dataset_tolerance_minutes) not in {5, 10, 15, 30}:
                raise ValueError(
                    "Label-reliability experiments require "
                    "news_first_dataset_tolerance_minutes in {5, 10, 15, 30}."
                )
            if bool(self.news_first_materialize_test_loader):
                raise ValueError(
                    "Label-reliability experiments must set "
                    "news_first_materialize_test_loader=false so post-validation "
                    "rows cannot enter a training-stage DataLoader."
                )
            if not str(self.data_path).strip():
                raise ValueError(
                    "Label-reliability experiments require a fold-scoped data_path."
                )
            if not str(self.news_first_common_eval_data_path).strip():
                raise ValueError(
                    "Label-reliability experiments require a fold-scoped "
                    "news_first_common_eval_data_path."
                )
        self.early_stopping_min_epochs = int(self.early_stopping_min_epochs)
        if self.early_stopping_min_epochs < 0:
            raise ValueError("early_stopping_min_epochs must be non-negative")


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert strongly-typed config object into plain dictionary."""
    return asdict(config)


def derive_training_output_paths(output_root: str) -> Dict[str, Any]:
    """Expand one merged-xlsx output root into concrete artifact paths."""

    normalized_output_root = str(output_root).strip()
    if not normalized_output_root:
        raise ValueError(
            "output_root must be a non-empty path when deriving training output paths."
        )

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
        if field_name in yaml_values
        or (overrides is not None and field_name in overrides)
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


def load_config(
    config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """Load YAML config and apply validated CLI overrides."""
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    _, yaml_values, loaded_values = load_yaml_config_values(
        resolved_path,
        defaults=config_to_dict(Config()),
        overrides=overrides,
        section="training",
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
