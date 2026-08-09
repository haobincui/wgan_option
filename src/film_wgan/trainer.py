"""Standalone FiLM WGAN training loop and checkpoint management."""

from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import default_collate

from trainer import BaseTrainer
from utils.output_paths import find_best_checkpoint, prepare_run_dir
from utils.training_paths import checkpoint_named_dir, generate_result_dir, infer_training_output_root, resolve_existing_run_dir
from .arbitrage import butterfly_arbitrage_penalty, calendar_arbitrage_penalty
from .config import (
    FilmWGANSampleConfig,
    FilmWGANTrainConfig,
    build_sample_config,
    build_sample_config_from_train_config,
    config_to_dict,
)
from .data import FilmWGANDataBundle, create_train_val_bundle, denormalize_tensor, normalize_surface_tensor
from .inference import FilmWGANSampler, build_sample_payload, normalization_stats_to_tensors
from .io import config_payload, load_checkpoint, save_checkpoint, write_csv, write_json
from .losses import (
    atm_short_pure_mae,
    build_atm_short_mask,
    build_reconstruction_weight_template,
    critic_transition_matching_loss,
    critic_wgan_loss,
    generator_transition_matching_loss,
    generator_wgan_loss,
    gradient_penalty_terms,
    maturity_smoothness_penalty,
    parameter_count,
    strike_smoothness_penalty,
    weighted_surface_mae,
)
from .matching import (
    TransitionMatchingNegativePlan,
    build_transition_matching_donor_mapping,
)
from .models import (
    VOL_CEIL,
    VOL_FLOOR,
    FilmWGANCritic,
    FilmWGANGenerator,
    reconstruct_future_surface,
    reconstruct_future_surface_terms,
)
from .protocol import (
    CHECKPOINT_SCHEMA_VERSION_V3,
    CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING,
    DIAGNOSTICS_SCHEMA_VERSION,
    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
    TEXT_ALIGNMENT_PLAN_VERSION,
    TRAINING_PROTOCOL_VERSION_V3,
    canonical_payload_sha256,
)
from .text_transform import sha256_file
from .training_plots import plot_training_curves
from wgan_option.config_parsing import load_yaml_mapping


def _load_generate_result_section(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    _, payload = load_yaml_mapping(config_path)
    section = payload.get("generate_result") or {}
    if section and not isinstance(section, dict):
        raise ValueError(f"Config section 'generate_result' in {config_path} must contain a YAML mapping.")
    return dict(section)


def module_state_sha256(module: torch.nn.Module | None) -> str:
    """Hash a module state without relying on torch serialization details."""

    if module is None:
        return ""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_mean(values: list[float], *, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float(default)


def aggregate_gradient_penalty_diagnostics(
    metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Aggregate GP sufficient statistics exactly across unequal batches."""

    count = float(sum(float(row.get("gp_raw_norm_count", 0.0)) for row in metric_rows))
    unsupported_max = max(
        (float(row.get("gp_unsupported_max_abs_gradient", 0.0)) for row in metric_rows),
        default=0.0,
    )
    raw_values = np.asarray(
        [
            float(value)
            for row in metric_rows
            for value in row.get("gp_raw_norm_values", [])  # type: ignore[union-attr]
        ],
        dtype=np.float64,
    )
    if count <= 0.0:
        return {
            "gp_raw_norm_mean": 0.0,
            "gp_raw_norm_std": 0.0,
            "gp_raw_norm_min": 0.0,
            "gp_raw_norm_max": 0.0,
            "gp_raw_norm_p05": 0.0,
            "gp_raw_norm_p50": 0.0,
            "gp_raw_norm_p95": 0.0,
            "gp_raw_norm_outside_0p5_1p5_rate": 0.0,
            "gp_unscaled_penalty": 0.0,
            "gp_unsupported_max_abs_gradient": unsupported_max,
            "gp_raw_norm_count": 0.0,
        }
    value_sum = sum(float(row.get("gp_raw_norm_sum", 0.0)) for row in metric_rows)
    square_sum = sum(
        float(row.get("gp_raw_norm_sum_squares", 0.0)) for row in metric_rows
    )
    mean = value_sum / count
    variance = max(0.0, square_sum / count - mean * mean)
    valid_rows = [
        row for row in metric_rows if float(row.get("gp_raw_norm_count", 0.0)) > 0.0
    ]
    quantiles = (
        np.quantile(raw_values, [0.05, 0.50, 0.95])
        if raw_values.size
        else np.asarray([float("nan")] * 3, dtype=np.float64)
    )
    return {
        "gp_raw_norm_mean": mean,
        "gp_raw_norm_std": math.sqrt(variance),
        "gp_raw_norm_min": min(float(row["gp_raw_norm_min"]) for row in valid_rows),
        "gp_raw_norm_max": max(float(row["gp_raw_norm_max"]) for row in valid_rows),
        "gp_raw_norm_p05": float(quantiles[0]),
        "gp_raw_norm_p50": float(quantiles[1]),
        "gp_raw_norm_p95": float(quantiles[2]),
        "gp_raw_norm_outside_0p5_1p5_rate": sum(
            float(row.get("gp_raw_norm_outside_count", 0.0)) for row in metric_rows
        )
        / count,
        "gp_unscaled_penalty": sum(
            float(row.get("gp_unscaled_penalty_sum", 0.0)) for row in metric_rows
        )
        / count,
        "gp_unsupported_max_abs_gradient": unsupported_max,
        "gp_raw_norm_count": count,
    }


def aggregate_transition_delivery_diagnostics(
    metric_rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate cell-level transition delivery diagnostics without batch bias."""

    supported = float(
        sum(float(row.get("g_transition_supported_count", 0.0)) for row in metric_rows)
    )
    clipped = float(
        sum(float(row.get("g_transition_clipped_count", 0.0)) for row in metric_rows)
    )
    clipped_gap_sum = float(
        sum(float(row.get("g_transition_clipped_abs_gap_sum", 0.0)) for row in metric_rows)
    )
    clipped_max = max(
        (float(row.get("g_transition_clipped_max_abs_gap", 0.0)) for row in metric_rows),
        default=0.0,
    )
    raw_log_max = max(
        (
            float(row.get("g_transition_unclipped_raw_log_max_abs_error", 0.0))
            for row in metric_rows
        ),
        default=0.0,
    )
    normalized_max = max(
        (
            float(row.get("g_transition_unclipped_normalized_max_abs_error", 0.0))
            for row in metric_rows
        ),
        default=0.0,
    )
    roundtrip_max = max(
        (
            float(row.get("g_transition_surface_log_roundtrip_max_abs_error", 0.0))
            for row in metric_rows
        ),
        default=0.0,
    )
    return {
        "g_transition_supported_count": supported,
        "g_transition_clipped_count": clipped,
        "g_transition_clipped_fraction": clipped / supported if supported > 0.0 else 0.0,
        "g_transition_clipped_mean_abs_gap": clipped_gap_sum / clipped if clipped > 0.0 else 0.0,
        "g_transition_clipped_max_abs_gap": clipped_max,
        "g_transition_unclipped_raw_log_max_abs_error": raw_log_max,
        "g_transition_unclipped_normalized_max_abs_error": normalized_max,
        "g_transition_surface_log_roundtrip_max_abs_error": roundtrip_max,
        # V2 aliases retained for plots and downstream readers.
        "g_saturation_rate": clipped / supported if supported > 0.0 else 0.0,
        "g_transition_delivery_max_abs_gap": clipped_max,
        "g_transition_nonsaturated_max_abs_error": normalized_max,
    }


def summarize_matching_logits(
    matched_logits: torch.Tensor,
    mismatched_logits: torch.Tensor,
    *,
    total_targets: int,
    prefix: str,
) -> dict[str, float]:
    """Return target-level held-out matcher metrics and a normal CI audit."""

    positive = matched_logits.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if positive.numel() == 0:
        return {
            f"{prefix}_eligible_targets": 0.0,
            f"{prefix}_eligible_fraction": 0.0,
            f"{prefix}_loss": float("nan"),
            f"{prefix}_positive_logit_mean": float("nan"),
            f"{prefix}_negative_logit_mean": float("nan"),
            f"{prefix}_margin_mean": float("nan"),
            f"{prefix}_pairwise_accuracy": float("nan"),
            f"{prefix}_accuracy_se": float("nan"),
            f"{prefix}_accuracy_ci95_low": float("nan"),
            f"{prefix}_accuracy_ci95_high": float("nan"),
        }
    negative = mismatched_logits.detach().to(dtype=torch.float64, device="cpu").reshape(
        positive.numel(), -1
    )
    pairwise_margin = positive.unsqueeze(1) - negative
    target_accuracy = (
        (pairwise_margin > 0.0).to(torch.float64)
        + 0.5 * (pairwise_margin == 0.0).to(torch.float64)
    ).mean(dim=1)
    accuracy = float(target_accuracy.mean())
    se = (
        float(target_accuracy.std(unbiased=True) / math.sqrt(float(positive.numel())))
        if positive.numel() > 1
        else 0.0
    )
    loss = 0.5 * (
        torch.nn.functional.softplus(-positive).mean()
        + torch.nn.functional.softplus(negative).mean()
    )
    return {
        f"{prefix}_eligible_targets": float(positive.numel()),
        f"{prefix}_eligible_fraction": float(positive.numel()) / float(max(1, total_targets)),
        f"{prefix}_loss": float(loss),
        f"{prefix}_positive_logit_mean": float(positive.mean()),
        f"{prefix}_negative_logit_mean": float(negative.mean()),
        f"{prefix}_margin_mean": float(pairwise_margin.mean()),
        f"{prefix}_pairwise_accuracy": accuracy,
        f"{prefix}_accuracy_se": se,
        f"{prefix}_accuracy_ci95_low": max(0.0, accuracy - 1.96 * se),
        f"{prefix}_accuracy_ci95_high": min(1.0, accuracy + 1.96 * se),
    }


@contextmanager
def _temporarily_freeze_parameters(module: torch.nn.Module):
    """Keep input gradients while preventing parameter gradients for one forward."""

    parameters = list(module.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original_flags):
            parameter.requires_grad_(requires_grad)


@contextmanager
def _temporarily_enable_parameter_gradients(module: torch.nn.Module):
    """Enable complete-module diagnostics while restoring every trainability flag."""

    parameters = list(module.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(True)
        yield parameters
    finally:
        for parameter, requires_grad in zip(parameters, original_flags):
            parameter.requires_grad_(requires_grad)


class FilmWGANTrainer(BaseTrainer):
    """Train the standalone FiLM WGAN model on `merged_vol.xlsx` rows."""

    trainer_id = "film_wgan"
    logger_name = "film_wgan.trainer"

    def __init__(self, config: FilmWGANTrainConfig, *, config_path: str | None = None):
        super().__init__(config, config_path=config_path)
        self.checkpoints_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None
        self.device = torch.device("cuda:0" if (config.cuda and torch.cuda.is_available()) else "cpu")

        self.bundle: Optional[FilmWGANDataBundle] = None
        self.generator: Optional[FilmWGANGenerator] = None
        self.critic: Optional[FilmWGANCritic] = None
        self.generator_optimizer: Optional[Adam] = None
        self.critic_optimizer: Optional[Adam] = None
        self.generator_scheduler: Optional[CosineAnnealingLR] = None
        self.critic_scheduler: Optional[CosineAnnealingLR] = None
        self.normalization = None
        self._strike_grid: Optional[torch.Tensor] = None
        self._maturity_days_grid: Optional[torch.Tensor] = None
        self._recon_weights_surface: Optional[torch.Tensor] = None
        self._recon_weights_flat: Optional[torch.Tensor] = None
        self._atm_short_mask_surface: Optional[torch.Tensor] = None
        self._atm_short_mask_flat: Optional[torch.Tensor] = None
        self._current_epoch: int = 0
        self._disc_step_count: int = 0
        self._gp_warmup_steps: int = 200
        self._grad_clip: float = 5.0
        self._lr_min_ratio: float = 0.1
        self._parent_checkpoint_path: str = ""
        self._parent_checkpoint_sha256: str = ""
        self._initial_generator_state_sha256: str = ""
        self._initial_critic_state_sha256: str = ""
        self._backbone_frozen: bool = False
        self._matching_donor_indices: Optional[torch.Tensor] = None
        self._matching_text_bank: Optional[torch.Tensor] = None
        self._matching_donor_mapping_sha256: str = ""
        self._matching_eligible_samples: int = 0
        self._matching_eligible_samples_by_split: dict[str, int] = {}
        self._matching_plan_violation_counts_by_split: dict[str, dict[str, int]] = {}
        self._matching_donor_indices_by_split: dict[str, torch.Tensor] = {}
        self._matching_text_banks_by_split: dict[str, torch.Tensor] = {}
        self._matching_positive_source_indices_by_split: dict[str, torch.Tensor] = {}
        self._matching_plan_sha256_by_split: dict[str, str] = {}
        self._matching_positive_alignment_sha256: str = ""
        self._matching_positive_alignment_sha256_by_split: dict[str, str] = {}
        self._text_alignment_plan_sha256: str = ""
        self._matching_negative_source_plan_sha256: str = ""
        self._matching_gradient_probe_batch: tuple[torch.Tensor, ...] | None = None
        self._matching_gradient_probe_noise: torch.Tensor | None = None
        self._latest_diagnostics: dict[str, float] = {}
        self._resolved_scheduler_horizon_epochs: int = 0
        self._last_gp_raw_norm_values: list[float] = []

    def _set_seed(self) -> None:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))

    def _prepare_runtime_config(self, config: FilmWGANTrainConfig) -> tuple[FilmWGANTrainConfig, Path]:
        output_root = infer_training_output_root(config, trainer_id=self.trainer_id)
        run_dir = prepare_run_dir(output_root, create=False)
        resolved_config = replace(
            config,
            output_root=str(output_root),
            checkpoints_path=str(run_dir / "checkpoints"),
            metrics_path=str(run_dir / "metrics"),
        )
        self.checkpoints_dir = Path(resolved_config.checkpoints_path)
        self.metrics_dir = Path(resolved_config.metrics_path)
        return resolved_config, run_dir

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _normalize_surface_flat(self, surface_flat: torch.Tensor) -> torch.Tensor:
        assert self.bundle is not None
        assert self.normalization is not None
        height, width = self.bundle.surface_shape
        normalized = normalize_surface_tensor(surface_flat, self.normalization.current_log_mean, self.normalization.current_log_std)
        return normalized.view(surface_flat.size(0), 1, height, width)

    def _load_initial_generator_checkpoint(self) -> None:
        assert self.generator is not None
        assert self.bundle is not None
        checkpoint_value = str(self.config.initial_generator_checkpoint_path).strip()
        if not checkpoint_value:
            return
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"initial_generator_checkpoint_path does not exist: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        if (
            str(self.config.critic_conditioning_mode).strip().lower()
            == "transition_matching"
        ):
            schema_version = int(checkpoint.get("checkpoint_schema_version", 0))
            protocol_version = str(
                checkpoint.get("training_protocol_version", "")
            )
            critic_version = str(
                checkpoint.get("critic_architecture_version", "")
            )
            v3_protocol = (
                str(self.config.training_protocol_version).strip()
                == TRAINING_PROTOCOL_VERSION_V3
            )
            expected_schema = CHECKPOINT_SCHEMA_VERSION_V3 if v3_protocol else 5
            expected_protocol = (
                TRAINING_PROTOCOL_VERSION_V3
                if v3_protocol
                else "film_wgan_transition_matching_v2"
            )
            if (
                schema_version != expected_schema
                or protocol_version != expected_protocol
                or critic_version
                != CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
            ):
                parent_protocol_label = (
                    "schema-6 v3 parent" if v3_protocol else "schema-5 v2 parent"
                )
                raise ValueError(
                    "transition_matching Stage B requires a "
                    f"{parent_protocol_label} "
                    "trained with the same transition-matching critic protocol; "
                    f"found schema={schema_version}, protocol={protocol_version!r}, "
                    f"critic={critic_version!r}."
                )
            if v3_protocol:
                parent_fingerprint = str(
                    checkpoint.get("run_fingerprint_sha256", "")
                ).strip()
                if (
                    len(parent_fingerprint) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in parent_fingerprint
                    )
                ):
                    raise ValueError(
                        "Parent run_fingerprint_sha256 must be a 64-character "
                        "lowercase hex digest for the v3 protocol."
                    )
                try:
                    parent_diagnostics_schema = int(
                        checkpoint.get("diagnostics_schema_version", 0)
                    )
                except (TypeError, ValueError):
                    parent_diagnostics_schema = 0
                if parent_diagnostics_schema != int(DIAGNOSTICS_SCHEMA_VERSION):
                    raise ValueError(
                        "Parent diagnostics_schema_version does not match the v3 protocol."
                    )
                for checkpoint_field, expected_version in (
                    (
                        "text_alignment_plan_version",
                        TEXT_ALIGNMENT_PLAN_VERSION,
                    ),
                    (
                        "matching_negative_source_plan_version",
                        MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
                    ),
                ):
                    if str(checkpoint.get(checkpoint_field, "")) != expected_version:
                        raise ValueError(
                            f"Parent {checkpoint_field} does not match the v3 protocol."
                        )
                for checkpoint_field, expected_sha in (
                    ("text_alignment_plan_sha256", self._text_alignment_plan_sha256),
                    (
                        "matching_negative_source_plan_sha256",
                        self._matching_negative_source_plan_sha256,
                    ),
                ):
                    if str(checkpoint.get(checkpoint_field, "")) != str(expected_sha):
                        raise ValueError(
                            f"Parent {checkpoint_field} does not match the frozen v3 plan artifact."
                        )
        parent_shape = tuple(int(value) for value in checkpoint.get("surface_shape", ()))
        if parent_shape != tuple(self.bundle.surface_shape):
            raise ValueError(
                f"Parent generator surface shape mismatch: expected {self.bundle.surface_shape}, "
                f"found {parent_shape}."
            )
        if int(checkpoint.get("embedding_dim", -1)) != int(self.bundle.embedding_dim):
            raise ValueError(
                f"Parent generator embedding dimension mismatch: expected {self.bundle.embedding_dim}, "
                f"found {checkpoint.get('embedding_dim')}."
            )
        parent_channels = int(checkpoint.get("current_surface_channels", 1))
        if parent_channels != int(self.generator.current_surface_channels):
            raise ValueError(
                "Parent generator current-surface channel mismatch: "
                f"expected {self.generator.current_surface_channels}, found {parent_channels}."
            )
        parent_support_sha = str(checkpoint.get("surface_support_sha256", ""))
        if parent_support_sha != str(self.bundle.surface_support_sha256):
            raise ValueError(
                "Parent generator raw-support artifact SHA256 does not match the current fold."
            )
        parent_mode = str(
            checkpoint.get(
                "conditioning_mode",
                (checkpoint.get("config") or {}).get("conditioning_mode", "film"),
            )
        ).strip().lower()
        if parent_mode != "residual_film":
            raise ValueError(
                "Paired initialization requires a residual_film parent checkpoint; "
                f"found conditioning_mode={parent_mode!r}."
            )
        parent_transform_sha = str(checkpoint.get("text_transform_sha256", ""))
        transform_policy = str(self.config.parent_text_transform_policy).strip().lower()
        if (
            transform_policy == "exact"
            and parent_transform_sha
            and self.bundle.text_transform_sha256
            and parent_transform_sha != self.bundle.text_transform_sha256
        ):
            raise ValueError(
                "Parent generator text-transform SHA256 does not match the current fold artifact."
            )
        if (
            transform_policy == "dimension_only"
            and parent_transform_sha != self.bundle.text_transform_sha256
        ):
            self.logger.info(
                "Parent text-transform SHA differs under dimension_only policy; "
                "embedding dimensions remain strictly matched."
            )
        try:
            self.generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Parent generator state is incompatible with the current architecture: {checkpoint_path}"
            ) from exc
        self._parent_checkpoint_path = str(checkpoint_path.resolve())
        self._parent_checkpoint_sha256 = sha256_file(checkpoint_path)
        freeze_epochs = max(0, int(self.config.freeze_backbone_epochs))
        if freeze_epochs > 0:
            self.generator.set_backbone_trainable(False)
            self._backbone_frozen = True
        self.logger.info(
            "Loaded paired parent generator: %s (sha256=%s, freeze_backbone_epochs=%d)",
            checkpoint_path,
            self._parent_checkpoint_sha256,
            freeze_epochs,
        )

    def _build_generator_optimizer(self) -> Adam:
        assert self.generator is not None
        betas = (float(self.config.beta_1), float(self.config.beta_2))
        if self.generator.conditioning_mode == "residual_film" and self._parent_checkpoint_path:
            backbone = list(self.generator.backbone_parameters())
            adapter = list(self.generator.text_adapter_parameters())
            return Adam(
                [
                    {
                        "params": backbone,
                        "lr": float(self.config.backbone_learning_rate),
                        "name": "surface_backbone",
                    },
                    {
                        "params": adapter,
                        "lr": float(self.config.text_adapter_learning_rate),
                        "name": "text_adapter",
                    },
                ],
                betas=betas,
            )
        return Adam(
            (parameter for parameter in self.generator.parameters() if parameter.requires_grad),
            lr=float(self.config.generator_learning_rate),
            betas=betas,
        )

    def _update_backbone_freeze_state(self, epoch: int) -> None:
        if self.generator is None:
            return
        if not self._parent_checkpoint_path or self.generator.conditioning_mode != "residual_film":
            return
        should_freeze = int(epoch) <= max(0, int(self.config.freeze_backbone_epochs))
        if should_freeze == self._backbone_frozen:
            return
        self.generator.set_backbone_trainable(not should_freeze)
        self._backbone_frozen = should_freeze
        self.logger.info(
            "Surface backbone %s at epoch %d.",
            "frozen" if should_freeze else "unfrozen",
            int(epoch),
        )

    def setup(self) -> None:
        self._set_seed()
        assert self.checkpoints_dir is not None
        assert self.metrics_dir is not None
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Standalone FiLM WGAN outputs: %s", self.run_dir)
        self.logger.info("Loading merged-vol workbook from %s (%s)", self.config.data_path, self.config.sheet_name)
        self.bundle = create_train_val_bundle(self.config)
        self.bundle.split_manifest.to_csv(self.metrics_dir / "split_manifest_resolved.csv", index=False)
        self._prepare_transition_matching_donors()
        self.normalization = normalization_stats_to_tensors(self.bundle.normalization_stats, self.device)
        self._strike_grid = torch.tensor(self.bundle.strike_grid, dtype=torch.float32, device=self.device)
        self._maturity_days_grid = torch.tensor(self.bundle.maturity_days_grid, dtype=torch.float32, device=self.device)
        self._recon_weights_surface = build_reconstruction_weight_template(
            strike_grid=self._strike_grid,
            maturity_days_grid=self._maturity_days_grid,
            mode=self.config.recon_weight_mode,
            atm_range=float(self.config.recon_atm_range),
            short_end_max_days=float(self.config.recon_atm_short_end_max_days),
            atm_multiplier=float(self.config.recon_atm_multiplier),
        )
        self._recon_weights_flat = self._recon_weights_surface.reshape(-1)
        self._atm_short_mask_surface = build_atm_short_mask(
            strike_grid=self._strike_grid,
            maturity_days_grid=self._maturity_days_grid,
            atm_range=float(self.config.atm_short_range),
            max_days=float(self.config.atm_short_max_days),
        )
        self._atm_short_mask_flat = self._atm_short_mask_surface.reshape(-1)
        surface_height, surface_width = self.bundle.surface_shape
        current_surface_channels = (
            2
            if str(self.config.surface_support_mode).strip().lower() == "raw_observed"
            else 1
        )
        deterministic = str(self.config.forecast_mode).strip().lower() == "deterministic"
        generator_noise_dim = 0 if deterministic else int(self.config.noise_dim)
        self.generator = FilmWGANGenerator(
            surface_height=surface_height,
            surface_width=surface_width,
            embedding_dim=self.bundle.embedding_dim,
            noise_dim=generator_noise_dim,
            base_channels=self.config.gen_base_channels,
            res_blocks=self.config.gen_res_blocks,
            text_hidden_dim=self.config.text_hidden_dim,
            text_out_dim=self.config.text_out_dim,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            conditioning_mode=self.config.conditioning_mode,
            text_dropout=float(self.config.text_dropout),
            text_gate_initial_value=float(self.config.text_gate_initial_value),
            current_surface_channels=current_surface_channels,
        ).to(self.device)
        self._load_initial_generator_checkpoint()
        if not deterministic:
            critic_text_dropout = float(self.config.critic_text_dropout)
            if critic_text_dropout < 0.0:
                critic_text_dropout = float(self.config.text_dropout)
            self.critic = FilmWGANCritic(
                surface_height=surface_height,
                surface_width=surface_width,
                embedding_dim=self.bundle.embedding_dim,
                base_channels=self.config.disc_base_channels,
                res_blocks=self.config.disc_res_blocks,
                text_hidden_dim=self.config.text_hidden_dim,
                text_out_dim=self.config.text_out_dim,
                fusion_hidden_dim=self.config.fusion_hidden_dim,
                conditioning_mode=self.config.conditioning_mode,
                critic_conditioning_mode=self.config.critic_conditioning_mode,
                text_dropout=critic_text_dropout,
                current_surface_channels=current_surface_channels,
                matching_logit_scale=float(self.config.matching_logit_scale),
            ).to(self.device)
        self._prepare_matching_gradient_probe()
        self._initial_generator_state_sha256 = module_state_sha256(self.generator)
        self._initial_critic_state_sha256 = module_state_sha256(self.critic)
        write_json(
            self.metrics_dir / "initialization_audit.json",
            {
                "seed": int(self.config.seed),
                "embedding_dim": int(self.bundle.embedding_dim),
                "parent_checkpoint_path": self._parent_checkpoint_path,
                "parent_checkpoint_sha256": self._parent_checkpoint_sha256,
                "parent_text_transform_policy": str(
                    self.config.parent_text_transform_policy
                ),
                "text_transform_path": self.bundle.text_transform_path,
                "text_transform_sha256": self.bundle.text_transform_sha256,
                "surface_support_path": self.bundle.surface_support_path,
                "surface_support_sha256": self.bundle.surface_support_sha256,
                "current_surface_channels": current_surface_channels,
                "generator_text_dropout": float(self.config.text_dropout),
                "critic_text_dropout": (
                    float(self.config.text_dropout)
                    if float(self.config.critic_text_dropout) < 0.0
                    else float(self.config.critic_text_dropout)
                ),
                "gradient_penalty_mode": str(self.config.gradient_penalty_mode),
                "matching_donor_mapping_sha256": self._matching_donor_mapping_sha256,
                "matching_eligible_samples": self._matching_eligible_samples,
                "matching_eligible_samples_by_split": dict(
                    self._matching_eligible_samples_by_split
                ),
                "training_protocol_version": str(
                    self.config.training_protocol_version
                ),
                "diagnostics_schema_version": int(
                    self.config.diagnostics_schema_version
                ),
                "run_fingerprint_sha256": str(
                    self.config.run_fingerprint_sha256
                ),
                "scheduler_horizon_epochs": int(
                    self.config.scheduler_horizon_epochs
                    or self.config.num_epochs
                ),
                "text_alignment_plan_path": str(
                    self.config.text_alignment_plan_path
                ),
                "text_alignment_plan_sha256": self._text_alignment_plan_sha256,
                "matching_negative_source_plan_path": str(
                    self.config.matching_negative_source_plan_path
                ),
                "matching_negative_source_plan_sha256": (
                    self._matching_negative_source_plan_sha256
                ),
                "matching_positive_alignment_sha256": (
                    self._matching_positive_alignment_sha256
                ),
                "matching_positive_alignment_sha256_by_split": dict(
                    self._matching_positive_alignment_sha256_by_split
                ),
                "matching_plan_sha256_by_split": dict(
                    self._matching_plan_sha256_by_split
                ),
                "matching_negative_source_plan_version": (
                    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION
                ),
                "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
                "native_positive_as_negative_count": sum(
                    values["native_positive_as_negative_count"]
                    for values in self._matching_plan_violation_counts_by_split.values()
                ),
                "placebo_positive_as_negative_count": sum(
                    values["placebo_positive_as_negative_count"]
                    for values in self._matching_plan_violation_counts_by_split.values()
                ),
                "matching_plan_violation_counts_by_split": dict(
                    self._matching_plan_violation_counts_by_split
                ),
                "initial_generator_state_sha256": self._initial_generator_state_sha256,
                "initial_critic_state_sha256": self._initial_critic_state_sha256,
            },
        )
        self.generator_optimizer = self._build_generator_optimizer()
        if self.critic is not None:
            self.critic_optimizer = Adam(
                (parameter for parameter in self.critic.parameters() if parameter.requires_grad),
                lr=float(self.config.discriminator_learning_rate),
                betas=(float(self.config.beta_1), float(self.config.beta_2)),
            )
        total_epochs = max(1, int(self.config.num_epochs))
        configured_horizon = int(self.config.scheduler_horizon_epochs)
        scheduler_horizon = configured_horizon if configured_horizon > 0 else total_epochs
        if scheduler_horizon < total_epochs:
            raise ValueError(
                "scheduler_horizon_epochs must be zero or at least num_epochs; "
                f"got horizon={scheduler_horizon} num_epochs={total_epochs}."
            )
        self._resolved_scheduler_horizon_epochs = scheduler_horizon
        self.generator_scheduler = CosineAnnealingLR(
            self.generator_optimizer,
            T_max=scheduler_horizon,
            eta_min=min(float(group["lr"]) for group in self.generator_optimizer.param_groups)
            * self._lr_min_ratio,
        )
        if self.critic_optimizer is not None:
            self.critic_scheduler = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=scheduler_horizon,
                eta_min=float(self.config.discriminator_learning_rate) * self._lr_min_ratio,
            )
        self.logger.info(
            "Dataset ready: train_samples=%s, val_samples=%s, test_samples=%s, surface_shape=%s, embedding_dim=%s",
            self.bundle.train_samples,
            self.bundle.val_samples,
            self.bundle.test_samples,
            self.bundle.surface_shape,
            self.bundle.embedding_dim,
        )
        self.logger.info(
            "Model initialized: G params=%s, C params=%s",
            parameter_count(self.generator.parameters()),
            parameter_count(self.critic.parameters()) if self.critic is not None else 0,
        )
        self.logger.info(
            "Reconstruction weighting: mode=%s atm_range=%.4f short_end_max_days=%.1f atm_multiplier=%.3f",
            str(self.config.recon_weight_mode),
            float(self.config.recon_atm_range),
            float(self.config.recon_atm_short_end_max_days),
            float(self.config.recon_atm_multiplier),
        )
        atm_short_cells = int(self._atm_short_mask_surface.sum().item()) if self._atm_short_mask_surface is not None else 0
        self.logger.info(
            "ATM-short pure loss: enabled=%s lambda=%.4f range=%.4f max_days=%.1f cells=%d/%d",
            bool(self.config.use_atm_short_loss),
            float(self.config.lambda_atm_short),
            float(self.config.atm_short_range),
            float(self.config.atm_short_max_days),
            atm_short_cells,
            surface_height * surface_width,
        )
        self.logger.info(
            "Adversarial weighting: lambda_adv=%.4f adv_warmup_epochs=%d adv_ramp_epochs=%d",
            float(self.config.lambda_adv),
            int(self.config.adv_warmup_epochs),
            int(self.config.adv_ramp_epochs),
        )
        self.logger.info(
            "Scheduler horizon: run_epochs=%d horizon_epochs=%d.",
            total_epochs,
            scheduler_horizon,
        )
        self.logger.info(
            "Critic protocol: conditioning=%s text_dropout=%.4f gp_mode=%s "
            "lambda_matching_d=%.4f lambda_matching_g=%.4f logit_scale=%.4f "
            "matcher_updates_per_generator_batch=%d",
            str(self.config.critic_conditioning_mode),
            (
                float(self.config.text_dropout)
                if float(self.config.critic_text_dropout) < 0.0
                else float(self.config.critic_text_dropout)
            ),
            str(self.config.gradient_penalty_mode),
            float(self.config.lambda_critic_matching),
            float(self.config.lambda_generator_matching),
            float(self.config.matching_logit_scale),
            int(
                str(self.config.critic_conditioning_mode).strip().lower()
                == "transition_matching"
                and float(self.config.lambda_critic_matching) > 0.0
            ),
        )

    def _prepare_transition_matching_donors(self) -> None:
        assert self.bundle is not None
        assert self.metrics_dir is not None
        transition_mode = (
            str(self.config.critic_conditioning_mode).strip().lower()
            == "transition_matching"
        )
        loss_enabled = (
            float(self.config.lambda_critic_matching) > 0.0
            or float(self.config.lambda_generator_matching) > 0.0
        )
        v3_protocol = (
            str(self.config.training_protocol_version).strip()
            == TRAINING_PROTOCOL_VERSION_V3
        )
        if not transition_mode:
            return
        if not (loss_enabled or v3_protocol):
            return

        split_items = {
            "train": self.bundle.train_items,
            "val": self.bundle.val_items,
        }
        native_split_items = {
            "train": self.bundle.native_train_items or self.bundle.train_items,
            "val": self.bundle.native_val_items or self.bundle.val_items,
        }

        negative_plan_frame: pd.DataFrame | None = None
        if v3_protocol:
            alignment_path = Path(self.config.text_alignment_plan_path).expanduser()
            negative_path = Path(
                self.config.matching_negative_source_plan_path
            ).expanduser()
            self._text_alignment_plan_sha256 = sha256_file(alignment_path)
            self._matching_negative_source_plan_sha256 = sha256_file(negative_path)
            negative_plan_frame = pd.read_csv(negative_path)

        positive_alignment_payload: dict[str, object] = {
            "training_protocol_version": str(self.config.training_protocol_version),
            "text_alignment_mode": str(self.config.text_alignment_mode).strip().lower(),
            "splits": {},
        }
        for split, items in split_items.items():
            if not items:
                continue
            native_items = native_split_items[split]
            try:
                if v3_protocol:
                    assert negative_plan_frame is not None
                    plan = TransitionMatchingNegativePlan.from_frame(
                        negative_plan_frame,
                        split=split,
                    )
                    native_sample_ids = tuple(item.sample_id for item in native_items)
                    native_pair_ids = tuple(item.surface_pair_id for item in native_items)
                    if (
                        plan.target_sample_ids != native_sample_ids
                        or plan.target_surface_pair_ids != native_pair_ids
                    ):
                        raise ValueError(
                            f"Frozen negative-source plan identities do not match split={split}."
                        )
                    alignment_plan = self.bundle.text_alignment_plans.get(split)
                    if alignment_plan is None:
                        raise ValueError(
                            f"The data bundle did not retain a frozen text alignment plan for split={split}."
                        )
                    if plan.positive_alignment_sha256 != alignment_plan.sha256:
                        raise ValueError(
                            f"Negative-source and text-alignment plan SHA mismatch for split={split}."
                        )
                    if plan.negative_count != int(self.config.matching_negative_count):
                        raise ValueError(
                            f"Negative-source plan K mismatch for split={split}."
                        )
                    if plan.minimum_supported_cells != int(
                        self.config.matching_min_supported_cells
                    ):
                        raise ValueError(
                            f"Negative-source minimum support mismatch for split={split}."
                        )
                    mapping = plan.negative_source_indices
                    rows = plan.to_frame().to_dict(orient="records")
                    plan_sha = plan.sha256
                    positive_sources = alignment_plan.positive_source_indices(
                        self.config.text_alignment_mode
                    )
                    mapping_values = np.asarray(mapping, dtype=np.int64)
                    canonical_targets = np.arange(len(native_items), dtype=np.int64)
                    eligible_rows = np.all(mapping_values >= 0, axis=1)
                    native_violations = int(
                        (
                            mapping_values[eligible_rows]
                            == canonical_targets[eligible_rows, None]
                        ).sum()
                    )
                    placebo_violations = int(
                        (
                            mapping_values[eligible_rows]
                            == alignment_plan.placebo_source_indices[
                                eligible_rows, None
                            ]
                        ).sum()
                    )
                    self._matching_plan_violation_counts_by_split[split] = {
                        "native_positive_as_negative_count": native_violations,
                        "placebo_positive_as_negative_count": placebo_violations,
                    }
                    if native_violations or placebo_violations:
                        raise ValueError(
                            "Frozen v3 negative-source plan reuses a native/placebo "
                            f"positive for split={split}: native={native_violations}, "
                            f"placebo={placebo_violations}."
                        )
                else:
                    mapping, rows = build_transition_matching_donor_mapping(
                        items,
                        negative_count=int(self.config.matching_negative_count),
                        minimum_supported_cells=int(
                            self.config.matching_min_supported_cells
                        ),
                        seed=int(self.config.matching_negative_seed),
                        duplicate_cosine_threshold=float(
                            self.config.matching_duplicate_cosine_threshold
                        ),
                    )
                    plan_sha = canonical_payload_sha256(
                        {
                            "split": split,
                            "mapping": np.asarray(mapping, dtype=np.int64).tolist(),
                        }
                    )
                    positive_sources = np.arange(len(items), dtype=np.int64)
                    native_items = items
                    self._matching_plan_violation_counts_by_split[split] = {
                        "native_positive_as_negative_count": 0,
                        "placebo_positive_as_negative_count": 0,
                    }
            except ValueError:
                if split == "train" or v3_protocol:
                    raise
                self.logger.warning(
                    "Held-out transition matcher is unavailable for split=%s because "
                    "a valid split-local donor map could not be built.",
                    split,
                    exc_info=True,
                )
                continue

            resolved_path = (
                self.metrics_dir
                / f"transition_matching_negative_source_plan_{split}_resolved.csv"
            )
            write_csv(resolved_path, rows)
            mapping_tensor = torch.as_tensor(
                np.asarray(mapping, dtype=np.int64).copy(),
                dtype=torch.long,
                device=self.device,
            )
            matching_text_bank = np.stack(
                [sample.text_embedding for sample in native_items], axis=0
            ).astype(np.float32)
            if bool(self.config.normalize_text_embedding):
                matching_text_bank = (
                    (matching_text_bank - self.bundle.normalization_stats.text_mean)
                    / self.bundle.normalization_stats.text_std
                ).astype(np.float32)
            self._matching_donor_indices_by_split[split] = mapping_tensor
            self._matching_positive_source_indices_by_split[split] = torch.as_tensor(
                np.asarray(positive_sources, dtype=np.int64).copy(),
                dtype=torch.long,
                device=self.device,
            )
            self._matching_text_banks_by_split[split] = torch.as_tensor(
                matching_text_bank,
                dtype=torch.float32,
                device=self.device,
            )
            self._matching_plan_sha256_by_split[split] = plan_sha
            self._matching_eligible_samples_by_split[split] = int(
                np.all(np.asarray(mapping) >= 0, axis=1).sum()
            )
            positive_rows = [
                {
                    "target_sample_id": native_items[target_index].sample_id,
                    "target_surface_pair_id": native_items[target_index].surface_pair_id,
                    "positive_source_index": int(source_index),
                    "positive_source_sample_id": native_items[int(source_index)].sample_id,
                    "positive_source_surface_pair_id": native_items[
                        int(source_index)
                    ].surface_pair_id,
                }
                for target_index, source_index in enumerate(positive_sources)
            ]
            split_positive_sha = canonical_payload_sha256(
                {"split": split, "rows": positive_rows}
            )
            self._matching_positive_alignment_sha256_by_split[
                split
            ] = split_positive_sha
            positive_alignment_payload["splits"][split] = positive_rows

        self._matching_positive_alignment_sha256 = canonical_payload_sha256(
            positive_alignment_payload
        )
        if "train" in self._matching_donor_indices_by_split:
            self._matching_donor_indices = self._matching_donor_indices_by_split[
                "train"
            ]
            self._matching_text_bank = self._matching_text_banks_by_split["train"]
            self._matching_donor_mapping_sha256 = self._matching_plan_sha256_by_split[
                "train"
            ]
            mapping = self._matching_donor_indices.detach().cpu().numpy()
            self._matching_eligible_samples = int(np.all(mapping >= 0, axis=1).sum())
        self.logger.info(
            "Transition-matching plans: eligible_train=%d/%d K=%d splits=%s file_sha256=%s",
            self._matching_eligible_samples,
            len(self.bundle.train_items),
            int(self.config.matching_negative_count),
            sorted(self._matching_donor_indices_by_split),
            self._matching_negative_source_plan_sha256,
        )

    def _prepare_matching_gradient_probe(self) -> None:
        """Freeze a deterministic, eligible training mini-batch and noise draw."""

        assert self.bundle is not None
        assert self.generator is not None
        self._matching_gradient_probe_batch = None
        self._matching_gradient_probe_noise = None
        requested = int(self.config.matching_gradient_probe_size)
        mapping = self._matching_donor_indices_by_split.get("train")
        dataset = getattr(self.bundle.train_loader, "dataset", None)
        if (
            requested <= 0
            or self.critic is None
            or self.critic.conditioning_mode != "transition_matching"
            or mapping is None
            or dataset is None
        ):
            return
        mapping_eligible = torch.nonzero(
            torch.all(mapping >= 0, dim=1), as_tuple=False
        ).reshape(-1).cpu().numpy()
        if int(mapping.size(0)) != len(dataset):
            raise ValueError(
                "Transition-matching probe mapping and train dataset size differ: "
                f"mapping={int(mapping.size(0))}, dataset={len(dataset)}."
            )
        carrier_eligible: list[int] = []
        for raw_index in mapping_eligible:
            index = int(raw_index)
            item = dataset[index]
            current_flat = item[3]
            extra_index = 5
            if str(self.config.surface_support_mode).strip().lower() == "raw_observed":
                support_mask_flat = item[extra_index]
                extra_index += 1
            else:
                support_mask_flat = torch.ones_like(current_flat)
            if str(self.config.conditioning_mode).strip().lower() == "residual_film":
                has_text = item[extra_index]
            else:
                has_text = torch.ones((), dtype=torch.float32)
            if (
                float(torch.as_tensor(has_text).reshape(-1)[0]) > 0.0
                and float(torch.as_tensor(support_mask_flat).reshape(-1).sum())
                >= float(self.config.matching_min_supported_cells)
            ):
                carrier_eligible.append(index)

        if carrier_eligible:
            eligible = np.asarray(carrier_eligible, dtype=np.int64)
            rng = np.random.default_rng(int(self.config.matching_gradient_probe_seed))
            chosen = np.sort(
                rng.choice(
                    eligible,
                    size=min(requested, int(eligible.size)),
                    replace=False,
                )
            )
            collated = default_collate([dataset[int(index)] for index in chosen])
            self._matching_gradient_probe_batch = tuple(
                self._to_device(value) if isinstance(value, torch.Tensor) else value
                for value in collated
            )
            if self.generator.noise_dim > 0:
                noise_generator = torch.Generator(device=self.device)
                noise_generator.manual_seed(int(self.config.matching_gradient_probe_seed))
                self._matching_gradient_probe_noise = torch.randn(
                    len(chosen),
                    self.generator.noise_dim,
                    generator=noise_generator,
                    device=self.device,
                    dtype=torch.float32,
                )
        else:
            chosen = np.empty((0,), dtype=np.int64)
        if self.metrics_dir is not None:
            write_json(
                self.metrics_dir / "matching_gradient_probe.json",
                {
                    "diagnostics_schema_version": int(
                        self.config.diagnostics_schema_version
                    ),
                    "seed": int(self.config.matching_gradient_probe_seed),
                    "requested_size": requested,
                    "mapping_eligible_size": int(mapping_eligible.size),
                    "carrier_eligible_size": int(len(carrier_eligible)),
                    "selected_size": int(len(chosen)),
                    "train_dataset_indices": [int(value) for value in chosen],
                },
            )

    @staticmethod
    def _probe_gradient_norms(
        loss: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
        *,
        adapter_parameter_ids: set[int],
        retain_graph: bool,
    ) -> tuple[float, float]:
        if not parameters or not loss.requires_grad:
            return 0.0, 0.0
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        all_squares = loss.new_zeros(())
        adapter_squares = loss.new_zeros(())
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            square = gradient.detach().square().sum()
            all_squares = all_squares + square
            if id(parameter) in adapter_parameter_ids:
                adapter_squares = adapter_squares + square
        return (
            float(torch.sqrt(all_squares).cpu()),
            float(torch.sqrt(adapter_squares).cpu()),
        )

    @staticmethod
    def _inactive_matching_gradient_probe_metrics(
        eligible_targets: int = 0,
        *,
        probe_samples: int = 0,
        effective_lambda: float = 0.0,
    ) -> dict[str, float]:
        """Return the complete probe schema when the probe is inactive.

        ``training_metrics.csv`` derives its header from the first epoch.  The
        probe is intentionally inactive during adversarial warmup, so omitting
        the active-only fields here would make the first active epoch impossible
        to append to the same metric table.  Keep every active probe field in
        this payload and use NaN only for measurements that were not made.
        """

        return {
            "diag_g_probe_active": 0.0,
            "diag_g_probe_eligible_targets": float(eligible_targets),
            "diag_g_matching_output_grad_rms_median": float("nan"),
            "diag_g_matching_output_grad_rms_p95": float("nan"),
            "diag_g_nonmatching_output_grad_rms_median": float("nan"),
            "diag_g_matching_output_grad_ratio_median": float("nan"),
            "diag_g_matching_output_grad_ratio_p95": float("nan"),
            "diag_g_matching_nonmatching_output_grad_cosine": float("nan"),
            "diag_g_all_parameter_ratio": float("nan"),
            "diag_g_text_adapter_ratio": float("nan"),
            "g_matching_gradient_probe_samples": float(probe_samples),
            "g_matching_gradient_norm_all": float("nan"),
            "g_nonmatching_gradient_norm_all": float("nan"),
            "g_matching_gradient_ratio_all": float("nan"),
            "g_matching_gradient_norm_text_adapter": float("nan"),
            "g_nonmatching_gradient_norm_text_adapter": float("nan"),
            "g_matching_gradient_ratio_text_adapter": float("nan"),
            "g_matching_gradient_probe_effective_lambda": float(
                effective_lambda
            ),
        }

    def _evaluate_matching_gradient_probe(self) -> dict[str, float]:
        """Measure weighted matching/non-matching generator gradients safely."""

        probe_batch = self._matching_gradient_probe_batch
        probe_samples = (
            int(probe_batch[0].size(0)) if probe_batch is not None else 0
        )
        ramp = self._adversarial_ramp_factor()
        effective_matching_weight = (
            float(self.config.lambda_generator_matching)
            * ramp
        )
        if (
            self.bundle is None
            or self.generator is None
            or self.normalization is None
            or self.critic is None
            or self.critic.conditioning_mode != "transition_matching"
            or probe_batch is None
        ):
            return self._inactive_matching_gradient_probe_metrics(
                probe_samples=probe_samples,
                effective_lambda=effective_matching_weight,
            )
        batch = probe_batch
        (
            current_features,
            text_features,
            _real_delta_norm,
            current_flat,
            target_flat,
        ) = batch[:5]
        extra_index = 5
        if str(self.config.surface_support_mode).strip().lower() == "raw_observed":
            support_mask_flat = batch[extra_index]
            extra_index += 1
        else:
            support_mask_flat = torch.ones_like(current_flat)
        if str(self.config.conditioning_mode).strip().lower() == "residual_film":
            has_text = batch[extra_index]
            extra_index += 1
        else:
            has_text = torch.ones(
                current_features.size(0),
                device=self.device,
                dtype=torch.float32,
            )
        sample_indices = batch[extra_index]
        support_surface = support_mask_flat.view(
            support_mask_flat.size(0),
            1,
            self.bundle.surface_shape[0],
            self.bundle.surface_shape[1],
        )
        eligible, donors = self._matching_indices(
            has_text,
            support_mask_flat,
            sample_indices,
            split="train",
        )
        if eligible.numel() == 0 or effective_matching_weight <= 0.0:
            return self._inactive_matching_gradient_probe_metrics(
                int(eligible.numel()),
                probe_samples=probe_samples,
                effective_lambda=effective_matching_weight,
            )
        parameters = list(self.generator.parameters())
        adapter_ids = (
            {id(parameter) for parameter in self.generator.text_adapter_parameters()}
            if self.generator.conditioning_mode == "residual_film"
            else set()
        )
        generator_was_training = self.generator.training
        critic_was_training = self.critic.training
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        self.generator.eval()
        self.critic.eval()
        try:
            with (
                torch.enable_grad(),
                _temporarily_enable_parameter_gradients(self.generator),
                _temporarily_freeze_parameters(self.critic),
            ):
                generated_delta_norm = self.generator(
                    current_features,
                    text_features,
                    noise=self._matching_gradient_probe_noise,
                    has_text=has_text,
                )
                generated_delta = (
                    denormalize_tensor(
                        generated_delta_norm,
                        self.normalization.delta_mean,
                        self.normalization.delta_std,
                    )
                    if self.config.normalize_target_delta
                    else generated_delta_norm
                )
                reconstruction = reconstruct_future_surface_terms(
                    current_flat,
                    generated_delta,
                )
                generated_future_flat = reconstruction.surface
                generated_future_unmasked = self._normalize_surface_flat(
                    generated_future_flat
                )
                generated_future_surface = generated_future_unmasked * support_surface
                generated_scores = self.critic(
                    generated_future_surface,
                    current_features,
                    text_features,
                    has_text=has_text,
                    support_mask=support_surface,
                )
                delivered_delta = reconstruction.clamped_log_surface - torch.log(
                    torch.clamp(current_flat, min=VOL_FLOOR)
                )
                delivered_delta_norm = (
                    (delivered_delta - self.normalization.delta_mean)
                    / torch.clamp(self.normalization.delta_std, min=1e-6)
                    if self.config.normalize_target_delta
                    else delivered_delta
                )
                positive, negative = self._transition_matching_logits(
                    delivered_delta_norm.view_as(support_surface),
                    text_features,
                    support_surface,
                    eligible,
                    donors,
                    sample_indices=sample_indices,
                    split="train",
                )
                matching_loss = generator_transition_matching_loss(
                    positive, negative
                )
                matching_weighted = effective_matching_weight * matching_loss
                nonmatching_weighted = (
                    float(self.config.lambda_adv)
                    * ramp
                    * generator_wgan_loss(generated_scores)
                )
                future_level = generated_future_flat.view(
                    generated_future_flat.size(0), *self.bundle.surface_shape
                )
                if self.config.use_calendar_constraint:
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_calendar
                    ) * calendar_arbitrage_penalty(
                        future_level, self._strike_grid, self._maturity_days_grid
                    ).mean()
                if self.config.use_butterfly_constraint:
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_butterfly
                    ) * butterfly_arbitrage_penalty(
                        future_level, self._strike_grid, self._maturity_days_grid
                    ).mean()
                if self.config.use_smooth_constraint:
                    future_log = reconstruction.clamped_log_surface.view_as(
                        future_level
                    )
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_smooth
                    ) * (
                        strike_smoothness_penalty(future_log, self._strike_grid)
                        + maturity_smoothness_penalty(
                            future_log, self._maturity_days_grid
                        )
                    )
                if self.config.use_recon_constraint:
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_recon
                    ) * weighted_surface_mae(
                        generated_future_flat,
                        target_flat,
                        support_mask_flat * self._recon_weights_flat,
                    )
                if self.config.use_atm_short_loss:
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_atm_short
                    ) * atm_short_pure_mae(
                        generated_future_flat,
                        target_flat,
                        support_mask_flat * self._atm_short_mask_flat,
                    )
                if float(self.config.lambda_film) > 0.0:
                    nonmatching_weighted = nonmatching_weighted + float(
                        self.config.lambda_film
                    ) * self.generator.film_regularization()
                matching_output_gradient = torch.autograd.grad(
                    matching_weighted,
                    generated_delta_norm,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach()
                nonmatching_output_gradient = torch.autograd.grad(
                    nonmatching_weighted,
                    generated_delta_norm,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach()
                matching_all, matching_adapter = self._probe_gradient_norms(
                    matching_weighted,
                    parameters,
                    adapter_parameter_ids=adapter_ids,
                    retain_graph=True,
                )
                nonmatching_all, nonmatching_adapter = self._probe_gradient_norms(
                    nonmatching_weighted,
                    parameters,
                    adapter_parameter_ids=adapter_ids,
                    retain_graph=False,
                )
        finally:
            self.generator.train(generator_was_training)
            self.critic.train(critic_was_training)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_cpu_rng_state)
            if torch_cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_states)

        def ratio(numerator: float, denominator: float) -> float:
            if numerator == 0.0 and denominator == 0.0:
                return 0.0
            return numerator / max(denominator, 1.0e-12)

        eligible_support = support_mask_flat[eligible].reshape(eligible.numel(), -1)
        matching_selected = matching_output_gradient[eligible].reshape(
            eligible.numel(), -1
        )
        nonmatching_selected = nonmatching_output_gradient[eligible].reshape(
            eligible.numel(), -1
        )
        support_count = eligible_support.sum(dim=1).clamp_min(1.0)
        matching_rms = (
            (matching_selected.square() * eligible_support).sum(dim=1)
            / support_count
        ).sqrt().cpu().numpy()
        nonmatching_rms = (
            (nonmatching_selected.square() * eligible_support).sum(dim=1)
            / support_count
        ).sqrt().cpu().numpy()
        per_target_ratio = np.divide(
            matching_rms,
            np.maximum(nonmatching_rms, 1.0e-12),
        )
        supported_cells = eligible_support > 0.0
        flat_matching = matching_selected[supported_cells]
        flat_nonmatching = nonmatching_selected[supported_cells]
        cosine_denominator = float(
            torch.linalg.vector_norm(flat_matching)
            * torch.linalg.vector_norm(flat_nonmatching)
        )
        output_cosine = (
            float(torch.dot(flat_matching, flat_nonmatching)) / cosine_denominator
            if cosine_denominator > 0.0
            else float("nan")
        )
        return {
            "diag_g_probe_active": 1.0,
            "diag_g_probe_eligible_targets": float(eligible.numel()),
            "diag_g_matching_output_grad_rms_median": float(
                np.median(matching_rms)
            ),
            "diag_g_matching_output_grad_rms_p95": float(
                np.quantile(matching_rms, 0.95)
            ),
            "diag_g_nonmatching_output_grad_rms_median": float(
                np.median(nonmatching_rms)
            ),
            "diag_g_matching_output_grad_ratio_median": float(
                np.median(per_target_ratio)
            ),
            "diag_g_matching_output_grad_ratio_p95": float(
                np.quantile(per_target_ratio, 0.95)
            ),
            "diag_g_matching_nonmatching_output_grad_cosine": output_cosine,
            "diag_g_all_parameter_ratio": ratio(matching_all, nonmatching_all),
            "diag_g_text_adapter_ratio": ratio(
                matching_adapter, nonmatching_adapter
            ),
            "g_matching_gradient_probe_samples": float(current_features.size(0)),
            "g_matching_gradient_norm_all": matching_all,
            "g_nonmatching_gradient_norm_all": nonmatching_all,
            "g_matching_gradient_ratio_all": ratio(
                matching_all, nonmatching_all
            ),
            "g_matching_gradient_norm_text_adapter": matching_adapter,
            "g_nonmatching_gradient_norm_text_adapter": nonmatching_adapter,
            "g_matching_gradient_ratio_text_adapter": ratio(
                matching_adapter, nonmatching_adapter
            ),
            "g_matching_gradient_probe_effective_lambda": float(
                effective_matching_weight
            ),
        }

    def _adversarial_ramp_factor(self) -> float:
        warmup_epochs = max(0, int(self.config.adv_warmup_epochs))
        if self._current_epoch <= warmup_epochs:
            return 0.0
        ramp_epochs = max(0, int(self.config.adv_ramp_epochs))
        if ramp_epochs == 0:
            return 1.0
        return min(1.0, float(self._current_epoch - warmup_epochs) / float(ramp_epochs))

    def _matching_indices(
        self,
        has_text: torch.Tensor,
        support_mask_flat: torch.Tensor,
        sample_indices: torch.Tensor | None,
        *,
        split: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_indices is None:
            raise ValueError(
                "transition matching requires dataset sample indices in each batch."
            )
        donor_mapping = self._matching_donor_indices_by_split.get(split)
        if donor_mapping is None and split == "train":
            donor_mapping = self._matching_donor_indices
        if donor_mapping is None:
            raise RuntimeError(
                f"Transition-matching donor mapping has not been prepared for split={split}."
            )
        sample_indices = sample_indices.reshape(-1).to(
            device=donor_mapping.device,
            dtype=torch.long,
        )
        if sample_indices.numel() != has_text.numel():
            raise ValueError("sample_indices must contain one index per batch row.")
        mapped_donors = donor_mapping[sample_indices]
        support_counts = support_mask_flat.reshape(support_mask_flat.size(0), -1).sum(dim=1)
        eligible_mask = (has_text.reshape(-1) > 0.0) & (
            support_counts >= float(self.config.matching_min_supported_cells)
        ) & torch.all(mapped_donors >= 0, dim=1)
        eligible = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        if eligible.numel() == 0:
            empty = torch.empty(
                (0, int(self.config.matching_negative_count)),
                dtype=torch.long,
                device=eligible.device,
            )
            return eligible, empty
        return eligible, mapped_donors[eligible]

    def _transition_matching_logits(
        self,
        transition_surface: torch.Tensor,
        text_features: torch.Tensor,
        support_surface: torch.Tensor,
        eligible: torch.Tensor,
        donors: torch.Tensor,
        *,
        sample_indices: torch.Tensor | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.critic is not None
        if eligible.numel() == 0 or donors.numel() == 0:
            empty = transition_surface.new_empty((0,))
            return empty, transition_surface.new_empty((0, 0))
        text_bank = self._matching_text_banks_by_split.get(split)
        if text_bank is None and split == "train":
            text_bank = self._matching_text_bank
        if text_bank is None:
            raise RuntimeError(
                f"Transition-matching text bank has not been prepared for split={split}."
            )
        positive_source_mapping = self._matching_positive_source_indices_by_split.get(
            split
        )
        if positive_source_mapping is not None:
            if sample_indices is None:
                raise ValueError(
                    "Canonical transition matching requires sample_indices for positive sources."
                )
            target_indices = sample_indices.reshape(-1).to(
                device=positive_source_mapping.device,
                dtype=torch.long,
            )
            positive_text = text_bank[
                positive_source_mapping[target_indices[eligible]]
            ]
        else:
            positive_text = text_features[eligible]
        positive_has_text = torch.ones(
            eligible.numel(),
            device=transition_surface.device,
            dtype=transition_surface.dtype,
        )
        matched = self.critic.matching_logits(
            transition_surface[eligible],
            positive_text,
            support_mask=support_surface[eligible],
            has_text=positive_has_text,
        ).reshape(-1)

        negative_count = donors.size(1)
        repeated_transition = transition_surface[eligible].repeat_interleave(
            negative_count,
            dim=0,
        )
        repeated_support = support_surface[eligible].repeat_interleave(
            negative_count,
            dim=0,
        )
        donor_text = text_bank[donors.reshape(-1)]
        donor_has_text = torch.ones(
            donor_text.size(0),
            device=transition_surface.device,
            dtype=transition_surface.dtype,
        )
        mismatched = self.critic.matching_logits(
            repeated_transition,
            donor_text,
            support_mask=repeated_support,
            has_text=donor_has_text,
        ).view(eligible.numel(), negative_count)
        return matched, mismatched

    def _discriminator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        real_delta_norm: torch.Tensor,
        current_flat: torch.Tensor,
        target_flat: torch.Tensor,
        has_text: torch.Tensor,
        support_mask_flat: torch.Tensor,
        sample_indices: torch.Tensor | None = None,
        update_matching: bool = True,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.critic is not None
        assert self.critic_optimizer is not None
        assert self.normalization is not None

        self.critic_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(current_features.size(0), self.generator.noise_dim, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            fake_delta_norm = self.generator(
                current_features,
                text_features,
                noise=noise,
                has_text=has_text,
            )
            fake_delta = (
                denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
                if self.config.normalize_target_delta
                else fake_delta_norm
            )
            fake_future_flat = reconstruct_future_surface(current_flat, fake_delta)
        fake_future_unmasked = self._normalize_surface_flat(fake_future_flat)
        real_future_unmasked = self._normalize_surface_flat(target_flat)
        support_surface = support_mask_flat.view(
            support_mask_flat.size(0),
            1,
            fake_future_unmasked.size(2),
            fake_future_unmasked.size(3),
        )
        fake_future_surface = fake_future_unmasked * support_surface
        real_future_surface = real_future_unmasked * support_surface

        fake_scores = self.critic(
            fake_future_surface,
            current_features,
            text_features,
            has_text=has_text,
            support_mask=support_surface,
        )
        real_scores = self.critic(
            real_future_surface,
            current_features,
            text_features,
            has_text=has_text,
            support_mask=support_surface,
        )
        adversarial_disc_loss = critic_wgan_loss(real_scores, fake_scores)
        zero = torch.zeros((), device=self.device, dtype=real_scores.dtype)
        mismatch_loss = torch.zeros((), device=self.device, dtype=real_scores.dtype)
        mismatch_scores = None
        mismatch_enabled = (
            self.critic.conditioning_mode == "projection"
            and float(self.config.lambda_mismatch) > 0.0
            and current_features.size(0) > 1
            and bool(torch.any(has_text > 0.0))
        )
        if mismatch_enabled:
            donor_indices = torch.roll(
                torch.arange(current_features.size(0), device=self.device),
                shifts=1,
            )
            mismatch_scores = self.critic(
                real_future_surface,
                current_features,
                text_features[donor_indices],
                has_text=has_text[donor_indices],
                support_mask=support_surface,
            )
            mismatch_loss = mismatch_scores.mean()
            mismatch_weight = float(self.config.lambda_mismatch)
            adversarial_disc_loss = (
                fake_scores.mean() + mismatch_weight * mismatch_loss
            ) / (1.0 + mismatch_weight) - real_scores.mean()

        matching_loss = zero
        matched_logit_mean = zero
        mismatched_logit_mean = zero
        matching_margin = zero
        matching_pairwise_accuracy = zero
        matching_eligible_fraction = zero
        if (
            self.critic.conditioning_mode == "transition_matching"
            and float(self.config.lambda_critic_matching) > 0.0
            and bool(update_matching)
        ):
            eligible, donors = self._matching_indices(
                has_text,
                support_mask_flat,
                sample_indices,
            )
            matching_eligible_fraction = real_scores.new_tensor(
                float(eligible.numel()) / float(max(1, has_text.numel()))
            )
            if eligible.numel() > 0 and donors.numel() > 0:
                real_transition = real_delta_norm.view(
                    real_delta_norm.size(0),
                    1,
                    real_future_surface.size(2),
                    real_future_surface.size(3),
                )
                matched_logits, mismatched_logits = self._transition_matching_logits(
                    real_transition,
                    text_features,
                    support_surface,
                    eligible,
                    donors,
                    sample_indices=sample_indices,
                )
                matching_loss = critic_transition_matching_loss(
                    matched_logits,
                    mismatched_logits,
                )
                matched_logit_mean = matched_logits.mean()
                mismatched_logit_mean = mismatched_logits.mean()
                matching_margin = (
                    matched_logits.unsqueeze(1) - mismatched_logits
                ).mean()
                matching_pairwise_accuracy = (
                    matched_logits.unsqueeze(1) > mismatched_logits
                ).float().mean()
        self._disc_step_count += 1
        warmup_factor = min(1.0, self._disc_step_count / max(1, self._gp_warmup_steps))
        effective_lambda_gp = float(self.config.lambda_gp) * warmup_factor
        support_aware_gp = (
            str(self.config.gradient_penalty_mode).strip().lower() == "support_masked"
        )
        gp_terms = gradient_penalty_terms(
            critic=self.critic,
            real_future_surface=(real_future_unmasked if support_aware_gp else real_future_surface),
            fake_future_surface=(fake_future_unmasked if support_aware_gp else fake_future_surface),
            current_surface=current_features,
            text_embedding=text_features,
            lambda_gp=effective_lambda_gp,
            has_text=has_text,
            support_mask=(support_surface if support_aware_gp else None),
        )
        gp = gp_terms.penalty
        disc_loss = (
            adversarial_disc_loss
            + gp
            + float(self.config.lambda_critic_matching) * matching_loss
        )
        disc_loss.backward()
        if torch.isfinite(disc_loss):
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self._grad_clip)
            self.critic_optimizer.step()
        else:
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.logger.warning("Non-finite disc_loss detected at step %s; skipping critic update.", self._disc_step_count)
        raw_gp_norms = gp_terms.raw_norms.detach()
        raw_gp_deviation = (raw_gp_norms - 1.0).square()
        self._last_gp_raw_norm_values = [
            float(value) for value in raw_gp_norms.cpu().tolist()
        ]
        return {
            "d_total": float(disc_loss.detach().cpu()),
            "d_real": float(real_scores.mean().detach().cpu()),
            "d_fake": float(fake_scores.mean().detach().cpu()),
            "d_mismatch": float(mismatch_loss.detach().cpu()),
            "d_wgan": float(adversarial_disc_loss.detach().cpu()),
            "d_matching": float(matching_loss.detach().cpu()),
            "d_aux_matching_contribution": float(
                (
                    float(self.config.lambda_critic_matching)
                    * matching_loss
                ).detach().cpu()
            ),
            "d_matching_positive": float(matched_logit_mean.detach().cpu()),
            "d_matching_negative": float(mismatched_logit_mean.detach().cpu()),
            "d_matching_margin": float(matching_margin.detach().cpu()),
            "d_matching_pairwise_accuracy": float(
                matching_pairwise_accuracy.detach().cpu()
            ),
            "d_matching_eligible_fraction": float(
                matching_eligible_fraction.detach().cpu()
            ),
            "gp": float(gp.detach().cpu()),
            "gp_raw_norm_count": float(raw_gp_norms.numel()),
            "gp_raw_norm_sum": float(raw_gp_norms.sum().cpu()),
            "gp_raw_norm_sum_squares": float(raw_gp_norms.square().sum().cpu()),
            "gp_raw_norm_min": float(raw_gp_norms.min().cpu()),
            "gp_raw_norm_max": float(raw_gp_norms.max().cpu()),
            "gp_raw_norm_outside_count": float(
                ((raw_gp_norms < 0.5) | (raw_gp_norms > 1.5)).sum().cpu()
            ),
            "gp_unscaled_penalty_sum": float(raw_gp_deviation.sum().cpu()),
            "gp_unsupported_max_abs_gradient": float(
                gp_terms.unsupported_max_abs_gradient.detach().cpu()
            ),
        }

    def _generator_step(
        self,
        current_features: torch.Tensor,
        text_features: torch.Tensor,
        current_flat: torch.Tensor,
        target_flat: torch.Tensor,
        has_text: torch.Tensor,
        support_mask_flat: torch.Tensor,
        sample_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        assert self.generator is not None
        assert self.generator_optimizer is not None
        assert self.bundle is not None
        assert self.normalization is not None
        assert self._strike_grid is not None
        assert self._maturity_days_grid is not None
        assert self._recon_weights_flat is not None

        self.generator_optimizer.zero_grad(set_to_none=True)
        if (
            self.critic_optimizer is not None
            and self.critic is not None
            and self.critic.conditioning_mode == "transition_matching"
        ):
            # The matching critic is frozen during the generator update.  Also
            # clear the preceding D-step gradients so `.grad` remains an exact
            # audit of this separation rather than stale optimizer state.
            self.critic_optimizer.zero_grad(set_to_none=True)
        noise = None
        if self.generator.noise_dim > 0:
            noise = torch.randn(
                current_features.size(0),
                self.generator.noise_dim,
                device=self.device,
                dtype=torch.float32,
            )
        fake_delta_norm = self.generator(
            current_features,
            text_features,
            noise=noise,
            has_text=has_text,
        )
        fake_delta = (
            denormalize_tensor(fake_delta_norm, self.normalization.delta_mean, self.normalization.delta_std)
            if self.config.normalize_target_delta
            else fake_delta_norm
        )
        reconstruction = reconstruct_future_surface_terms(current_flat, fake_delta)
        fake_future_flat = reconstruction.surface
        current_log = torch.log(torch.clamp(current_flat, min=VOL_FLOOR))
        # This is the exact transition delivered to every downstream loss.
        # Do not recover it through log(exp(.)): the structured helper retains
        # the clamp result before the level-space round trip.
        actual_delta = reconstruction.clamped_log_surface - current_log
        actual_delta_norm = (
            (actual_delta - self.normalization.delta_mean)
            / torch.clamp(self.normalization.delta_std, min=1e-6)
            if self.config.normalize_target_delta
            else actual_delta
        )
        fake_future_unmasked = self._normalize_surface_flat(fake_future_flat)
        support_surface = support_mask_flat.view(
            support_mask_flat.size(0),
            1,
            self.bundle.surface_shape[0],
            self.bundle.surface_shape[1],
        )
        fake_future_surface = fake_future_unmasked * support_surface
        matching_generator_loss = torch.zeros(
            (),
            device=fake_future_flat.device,
            dtype=fake_future_flat.dtype,
        )
        matching_generator_margin = matching_generator_loss
        if self.critic is not None:
            freeze_context = (
                _temporarily_freeze_parameters(self.critic)
                if self.critic.conditioning_mode == "transition_matching"
                else nullcontext()
            )
            with freeze_context:
                fake_scores = self.critic(
                    fake_future_surface,
                    current_features,
                    text_features,
                    has_text=has_text,
                    support_mask=support_surface,
                )
                adv_loss = generator_wgan_loss(fake_scores)
                if (
                    self.critic.conditioning_mode == "transition_matching"
                    and float(self.config.lambda_generator_matching) > 0.0
                ):
                    eligible, donors = self._matching_indices(
                        has_text,
                        support_mask_flat,
                        sample_indices,
                    )
                    if eligible.numel() > 0 and donors.numel() > 0:
                        # Match the transition represented by the delivered,
                        # clamped future surface.  This prevents the generator
                        # from optimizing an out-of-range latent delta that the
                        # adversarial/reconstruction objectives never observe.
                        fake_transition = actual_delta_norm.view(
                            actual_delta_norm.size(0),
                            1,
                            self.bundle.surface_shape[0],
                            self.bundle.surface_shape[1],
                        )
                        matched_logits, mismatched_logits = self._transition_matching_logits(
                            fake_transition,
                            text_features,
                            support_surface,
                            eligible,
                            donors,
                            sample_indices=sample_indices,
                        )
                        matching_generator_loss = generator_transition_matching_loss(
                            matched_logits,
                            mismatched_logits,
                        )
                        matching_generator_margin = (
                            matched_logits.unsqueeze(1) - mismatched_logits
                        ).mean()
        else:
            adv_loss = torch.zeros((), device=fake_future_flat.device, dtype=fake_future_flat.dtype)

        future_surface_level = fake_future_flat.view(current_features.size(0), self.bundle.surface_shape[0], self.bundle.surface_shape[1])
        future_log_surface = reconstruction.clamped_log_surface.view_as(
            future_surface_level
        )
        supported_mask = support_mask_flat > 0.0
        supported_count = supported_mask.sum()
        clipped_supported_mask = supported_mask & reconstruction.clipped_mask
        clipped_count = clipped_supported_mask.sum()
        saturation_rate = clipped_count.to(dtype=support_mask_flat.dtype) / supported_count.clamp_min(1)
        supported_transition_gap = torch.abs(
            actual_delta_norm - fake_delta_norm
        )[supported_mask]
        transition_delivery_max_abs_gap = (
            torch.max(supported_transition_gap)
            if supported_transition_gap.numel() > 0
            else torch.zeros(
                (),
                device=fake_future_flat.device,
                dtype=fake_future_flat.dtype,
            )
        )
        transition_audit_mask = supported_mask & (~reconstruction.clipped_mask)
        if bool(torch.any(transition_audit_mask)):
            transition_nonsaturated_max_abs_error = torch.max(
                torch.abs(actual_delta_norm - fake_delta_norm)[transition_audit_mask]
            )
            transition_unclipped_raw_log_max_abs_error = torch.max(
                torch.abs(actual_delta - fake_delta)[transition_audit_mask]
            )
        else:
            transition_nonsaturated_max_abs_error = torch.zeros(
                (),
                device=fake_future_flat.device,
                dtype=fake_future_flat.dtype,
            )
            transition_unclipped_raw_log_max_abs_error = (
                transition_nonsaturated_max_abs_error
            )
        clipped_gaps = torch.abs(actual_delta_norm - fake_delta_norm)[
            clipped_supported_mask
        ]
        clipped_abs_gap_sum = (
            clipped_gaps.sum()
            if clipped_gaps.numel() > 0
            else torch.zeros((), device=self.device, dtype=fake_future_flat.dtype)
        )
        clipped_max_abs_gap = (
            clipped_gaps.max()
            if clipped_gaps.numel() > 0
            else torch.zeros((), device=self.device, dtype=fake_future_flat.dtype)
        )
        surface_log_roundtrip = torch.abs(
            torch.log(torch.clamp(fake_future_flat, min=VOL_FLOOR))
            - reconstruction.clamped_log_surface
        )[supported_mask]
        surface_log_roundtrip_max_abs_error = (
            surface_log_roundtrip.max()
            if surface_log_roundtrip.numel() > 0
            else torch.zeros((), device=self.device, dtype=fake_future_flat.dtype)
        )
        zero_penalty = torch.zeros(
            (),
            device=fake_future_flat.device,
            dtype=fake_future_flat.dtype,
        )
        calendar_penalty = (
            calendar_arbitrage_penalty(
                future_surface_level,
                self._strike_grid,
                self._maturity_days_grid,
            ).mean()
            if self.config.use_calendar_constraint
            else zero_penalty
        )
        butterfly_penalty = (
            butterfly_arbitrage_penalty(
                future_surface_level,
                self._strike_grid,
                self._maturity_days_grid,
            ).mean()
            if self.config.use_butterfly_constraint
            else zero_penalty
        )
        smooth_penalty = (
            strike_smoothness_penalty(future_log_surface, self._strike_grid)
            + maturity_smoothness_penalty(
                future_log_surface,
                self._maturity_days_grid,
            )
            if self.config.use_smooth_constraint
            else zero_penalty
        )
        recon_penalty = weighted_surface_mae(
            fake_future_flat,
            target_flat,
            support_mask_flat,
        )
        recon_penalty_weighted = weighted_surface_mae(
            fake_future_flat,
            target_flat,
            support_mask_flat * self._recon_weights_flat,
        )
        if self.config.use_atm_short_loss and self._atm_short_mask_flat is not None:
            atm_short_penalty = atm_short_pure_mae(
                fake_future_flat,
                target_flat,
                support_mask_flat * self._atm_short_mask_flat,
            )
        else:
            atm_short_penalty = torch.zeros((), device=fake_future_flat.device, dtype=fake_future_flat.dtype)

        adversarial_ramp_factor = self._adversarial_ramp_factor()
        if self.critic is None:
            adversarial_ramp_factor = 0.0
        effective_lambda_adv = float(self.config.lambda_adv) * adversarial_ramp_factor
        effective_lambda_generator_matching = (
            float(self.config.lambda_generator_matching) * adversarial_ramp_factor
        )

        total_loss = (
            effective_lambda_adv * adv_loss
            + effective_lambda_generator_matching * matching_generator_loss
        )
        if self.config.use_calendar_constraint:
            total_loss = total_loss + float(self.config.lambda_calendar) * calendar_penalty
        if self.config.use_butterfly_constraint:
            total_loss = total_loss + float(self.config.lambda_butterfly) * butterfly_penalty
        if self.config.use_smooth_constraint:
            total_loss = total_loss + float(self.config.lambda_smooth) * smooth_penalty
        if self.config.use_recon_constraint:
            total_loss = total_loss + float(self.config.lambda_recon) * recon_penalty_weighted
        if self.config.use_atm_short_loss:
            total_loss = total_loss + float(self.config.lambda_atm_short) * atm_short_penalty
        film_penalty = self.generator.film_regularization()
        if float(self.config.lambda_film) > 0.0:
            total_loss = total_loss + float(self.config.lambda_film) * film_penalty
        total_loss.backward()
        if torch.isfinite(total_loss):
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self._grad_clip)
            self.generator_optimizer.step()
        else:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.logger.warning("Non-finite generator total_loss detected; skipping generator update.")
        return {
            "g_total": float(total_loss.detach().cpu()),
            "g_adv": float(adv_loss.detach().cpu()),
            "g_adv_effective_lambda": float(effective_lambda_adv),
            "g_matching": float(matching_generator_loss.detach().cpu()),
            "g_matching_margin": float(matching_generator_margin.detach().cpu()),
            "g_matching_effective_lambda": float(effective_lambda_generator_matching),
            "g_saturation_rate": float(saturation_rate.detach().cpu()),
            "g_transition_delivery_max_abs_gap": float(
                transition_delivery_max_abs_gap.detach().cpu()
            ),
            "g_transition_nonsaturated_max_abs_error": float(
                transition_nonsaturated_max_abs_error.detach().cpu()
            ),
            "g_transition_supported_count": float(supported_count.detach().cpu()),
            "g_transition_clipped_count": float(clipped_count.detach().cpu()),
            "g_transition_clipped_abs_gap_sum": float(
                clipped_abs_gap_sum.detach().cpu()
            ),
            "g_transition_clipped_max_abs_gap": float(
                clipped_max_abs_gap.detach().cpu()
            ),
            "g_transition_unclipped_raw_log_max_abs_error": float(
                transition_unclipped_raw_log_max_abs_error.detach().cpu()
            ),
            "g_transition_unclipped_normalized_max_abs_error": float(
                transition_nonsaturated_max_abs_error.detach().cpu()
            ),
            "g_transition_surface_log_roundtrip_max_abs_error": float(
                surface_log_roundtrip_max_abs_error.detach().cpu()
            ),
            "g_calendar": float(calendar_penalty.detach().cpu()),
            "g_butterfly": float(butterfly_penalty.detach().cpu()),
            "g_smooth": float(smooth_penalty.detach().cpu()),
            "g_recon": float(recon_penalty.detach().cpu()),
            "g_recon_weighted": float(recon_penalty_weighted.detach().cpu()),
            "g_atm_short": float(atm_short_penalty.detach().cpu()),
            "g_film": float(film_penalty.detach().cpu()),
        }

    def _evaluate_matching_diagnostics(self) -> dict[str, float]:
        """Evaluate real/generated/mask-only matching on the frozen val plan."""

        assert self.bundle is not None
        assert self.generator is not None
        assert self.normalization is not None
        if (
            self.critic is None
            or self.critic.conditioning_mode != "transition_matching"
            or self.bundle.val_loader is None
            or "val" not in self._matching_donor_indices_by_split
        ):
            return {}

        generator_was_training = self.generator.training
        critic_was_training = self.critic.training
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        self.generator.eval()
        self.critic.eval()
        collected: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {"positive": [], "negative": []}
            for name in ("real", "generated", "mask_only")
        }
        noise_generator = torch.Generator(device=self.device)
        noise_generator.manual_seed(int(self.config.matching_gradient_probe_seed) + 101)
        try:
            with torch.inference_mode():
                for batch in self.bundle.val_loader:
                    (
                        current_features,
                        text_features,
                        real_delta_norm,
                        current_flat,
                        _target_flat,
                    ) = batch[:5]
                    extra_index = 5
                    if (
                        str(self.config.surface_support_mode).strip().lower()
                        == "raw_observed"
                    ):
                        support_mask_flat = batch[extra_index]
                        extra_index += 1
                    else:
                        support_mask_flat = torch.ones_like(current_flat)
                    if (
                        str(self.config.conditioning_mode).strip().lower()
                        == "residual_film"
                    ):
                        has_text = batch[extra_index]
                        extra_index += 1
                    else:
                        has_text = torch.ones(
                            current_features.size(0), dtype=torch.float32
                        )
                    sample_indices = batch[extra_index]
                    current_features = self._to_device(current_features)
                    text_features = self._to_device(text_features)
                    real_delta_norm = self._to_device(real_delta_norm)
                    current_flat = self._to_device(current_flat)
                    support_mask_flat = self._to_device(support_mask_flat)
                    has_text = self._to_device(has_text)
                    sample_indices = self._to_device(sample_indices)
                    support_surface = support_mask_flat.view(
                        support_mask_flat.size(0),
                        1,
                        self.bundle.surface_shape[0],
                        self.bundle.surface_shape[1],
                    )
                    eligible, donors = self._matching_indices(
                        has_text,
                        support_mask_flat,
                        sample_indices,
                        split="val",
                    )
                    if eligible.numel() == 0:
                        continue
                    noise = None
                    if self.generator.noise_dim > 0:
                        noise = torch.randn(
                            current_features.size(0),
                            self.generator.noise_dim,
                            generator=noise_generator,
                            device=self.device,
                            dtype=torch.float32,
                        )
                    generated_delta_norm = self.generator(
                        current_features,
                        text_features,
                        noise=noise,
                        has_text=has_text,
                    )
                    generated_delta = (
                        denormalize_tensor(
                            generated_delta_norm,
                            self.normalization.delta_mean,
                            self.normalization.delta_std,
                        )
                        if self.config.normalize_target_delta
                        else generated_delta_norm
                    )
                    generated_reconstruction = reconstruct_future_surface_terms(
                        current_flat,
                        generated_delta,
                    )
                    generated_delivered = (
                        generated_reconstruction.clamped_log_surface
                        - torch.log(torch.clamp(current_flat, min=VOL_FLOOR))
                    )
                    generated_delivered_norm = (
                        (
                            generated_delivered - self.normalization.delta_mean
                        )
                        / torch.clamp(self.normalization.delta_std, min=1e-6)
                        if self.config.normalize_target_delta
                        else generated_delivered
                    )
                    transition_by_name = {
                        "real": real_delta_norm.view_as(support_surface),
                        "generated": generated_delivered_norm.view_as(
                            support_surface
                        ),
                        # Zero transition with the real support mask isolates
                        # any support-geometry/text shortcut in the matcher.
                        "mask_only": torch.zeros_like(support_surface),
                    }
                    for name, transition in transition_by_name.items():
                        positive, negative = self._transition_matching_logits(
                            transition,
                            text_features,
                            support_surface,
                            eligible,
                            donors,
                            sample_indices=sample_indices,
                            split="val",
                        )
                        collected[name]["positive"].append(positive.cpu())
                        collected[name]["negative"].append(negative.cpu())
        finally:
            self.generator.train(generator_was_training)
            self.critic.train(critic_was_training)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_cpu_rng_state)
            if torch_cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_states)

        metrics: dict[str, float] = {}
        for name, values in collected.items():
            if values["positive"]:
                positive = torch.cat(values["positive"], dim=0)
                negative = torch.cat(values["negative"], dim=0)
            else:
                positive = torch.empty(0)
                negative = torch.empty(
                    (0, int(self.config.matching_negative_count))
                )
            metrics.update(
                summarize_matching_logits(
                    positive,
                    negative,
                    total_targets=len(self.bundle.val_items),
                    prefix=f"val_matching_{name}",
                )
            )
        # Keep the pre-v3 names as real-transition aliases.
        for suffix in (
            "eligible_targets",
            "eligible_fraction",
            "loss",
            "positive_logit_mean",
            "negative_logit_mean",
            "margin_mean",
            "pairwise_accuracy",
            "accuracy_se",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
        ):
            metrics[f"val_matching_{suffix}"] = metrics[
                f"val_matching_real_{suffix}"
            ]
        return metrics

    def _evaluate(self) -> dict[str, float]:
        assert self.bundle is not None
        assert self.generator is not None
        assert self.normalization is not None
        assert self._strike_grid is not None
        assert self._maturity_days_grid is not None
        assert self._recon_weights_surface is not None

        if not self.bundle.val_items:
            return {}

        generator_was_training = self.generator.training
        self.generator.eval()
        mae: list[float] = []
        rmse: list[float] = []
        current_mae: list[float] = []
        current_rmse: list[float] = []
        short_atm_weighted_mae: list[float] = []
        current_short_atm_weighted_mae: list[float] = []
        atm_short_pure_list: list[float] = []
        current_atm_short_pure_list: list[float] = []
        atm_short_win_flags: list[float] = []
        win_flags: list[float] = []
        generated_current_mae: list[float] = []
        real_current_mae: list[float] = []
        calendar: list[float] = []
        butterfly: list[float] = []
        penalty_mean: list[float] = []
        penalty_std: list[float] = []
        weight_entropy: list[float] = []
        mc_surface_mae_se: list[float] = []

        for sample in self.bundle.val_items:
            payload = build_sample_payload(
                generator=self.generator,
                sample=sample,
                normalization=self.normalization,
                noise_dim=int(self.generator.noise_dim),
                mc_samples=int(self.config.eval_mc_samples),
                seed=(
                    int(self.config.evaluation_noise_seed)
                    if int(self.config.evaluation_noise_seed) >= 0
                    else int(self.config.seed)
                ),
                device=self.device,
                reweight_beta_mode=self.config.eval_reweight_beta_mode,
                reweight_beta=float(self.config.eval_reweight_beta),
                aggregation_mode=self.config.eval_aggregation_mode,
                quantiles=(),
                calibration_levels=self.config.eval_calibration_levels,
                arbitrage_violation_tolerance=float(self.config.arbitrage_violation_tolerance),
                checkpoint_path="",
                split="val",
                selection_mode="all",
                recon_weights_surface=self._recon_weights_surface,
                atm_short_mask_surface=self._atm_short_mask_surface,
                normalize_current_surface=bool(self.config.normalize_current_surface),
                normalize_text_embedding=bool(self.config.normalize_text_embedding),
                normalize_target_delta=bool(self.config.normalize_target_delta),
                residual_blend_alpha=1.0,
            )
            mae.append(float(payload["metrics"]["mae"]))
            rmse.append(float(payload["metrics"]["rmse"]))
            current_mae.append(float(payload["current_metrics"]["mae"]))
            current_rmse.append(float(payload["current_metrics"]["rmse"]))
            generated_current_mae.append(float(payload["generated_current_metrics"]["mae"]))
            real_current_mae.append(float(payload["current_metrics"]["mae"]))
            generated_surface = torch.tensor(payload["generated_surface"], dtype=torch.float32, device=self.device)
            current_surface = torch.tensor(payload["current_surface"], dtype=torch.float32, device=self.device)
            target_surface = torch.tensor(payload["target_surface"], dtype=torch.float32, device=self.device)
            evaluation_support = torch.tensor(
                sample.evaluation_support_mask,
                dtype=torch.float32,
                device=self.device,
            )
            short_atm_weighted_mae.append(
                float(
                    weighted_surface_mae(
                        generated_surface,
                        target_surface,
                        self._recon_weights_surface * evaluation_support,
                    )
                    .detach()
                    .cpu()
                )
            )
            current_short_atm_weighted_mae.append(
                float(
                    weighted_surface_mae(
                        current_surface,
                        target_surface,
                        self._recon_weights_surface * evaluation_support,
                    )
                    .detach()
                    .cpu()
                )
            )
            supported_atm_mask = (
                self._atm_short_mask_surface * evaluation_support
                if self._atm_short_mask_surface is not None
                else None
            )
            if supported_atm_mask is not None and float(supported_atm_mask.sum().item()) > 0.0:
                gen_atm_pure = float(
                    atm_short_pure_mae(
                        generated_surface,
                        target_surface,
                        supported_atm_mask,
                    )
                    .detach()
                    .cpu()
                )
                cur_atm_pure = float(
                    atm_short_pure_mae(
                        current_surface,
                        target_surface,
                        supported_atm_mask,
                    )
                    .detach()
                    .cpu()
                )
                atm_short_pure_list.append(gen_atm_pure)
                current_atm_short_pure_list.append(cur_atm_pure)
                atm_short_win_flags.append(1.0 if gen_atm_pure < cur_atm_pure else 0.0)
            win_flags.append(1.0 if float(payload["metrics"]["mae"]) < float(payload["current_metrics"]["mae"]) else 0.0)
            penalty_mean.append(float(payload["penalty_mean"]))
            penalty_std.append(float(payload["penalty_std"]))
            weight_entropy.append(float(payload["weight_entropy"]))
            mc_surface_mae_se.append(
                float(
                    payload["probabilistic_metrics"][
                        "mc_surface_mae_se"
                    ]
                )
            )

            if bool(torch.all(evaluation_support > 0.0)):
                weighted_surface = generated_surface.unsqueeze(0)
                calendar.extend(
                    calendar_arbitrage_penalty(
                        weighted_surface,
                        self._strike_grid,
                        self._maturity_days_grid,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                butterfly.extend(
                    butterfly_arbitrage_penalty(
                        weighted_surface,
                        self._strike_grid,
                        self._maturity_days_grid,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

        self.generator.train(generator_was_training)
        val_mae = _finite_mean(mae, default=float("nan"))
        val_current_mae = _finite_mean(current_mae, default=float("nan"))
        val_short_atm_weighted_mae = _finite_mean(
            short_atm_weighted_mae,
            default=float("nan"),
        )
        val_current_short_atm_weighted_mae = _finite_mean(
            current_short_atm_weighted_mae,
            default=float("nan"),
        )
        val_atm_short_pure_mae = _finite_mean(
            atm_short_pure_list,
            default=float("nan"),
        )
        val_current_atm_short_pure_mae = _finite_mean(
            current_atm_short_pure_list,
            default=float("nan"),
        )
        metrics = {
            "val_mae": val_mae,
            "val_rmse": _finite_mean(rmse, default=float("nan")),
            "val_current_mae": val_current_mae,
            "val_current_rmse": _finite_mean(current_rmse, default=float("nan")),
            "val_short_atm_weighted_mae": val_short_atm_weighted_mae,
            "val_current_short_atm_weighted_mae": val_current_short_atm_weighted_mae,
            "val_short_atm_mae_gap_vs_current": val_short_atm_weighted_mae - val_current_short_atm_weighted_mae,
            "val_atm_short_pure_mae": val_atm_short_pure_mae,
            "val_current_atm_short_pure_mae": val_current_atm_short_pure_mae,
            "val_atm_short_pure_mae_gap_vs_current": val_atm_short_pure_mae - val_current_atm_short_pure_mae,
            "val_atm_short_win_rate_vs_current": _finite_mean(atm_short_win_flags),
            "val_mae_gap_vs_current": val_mae - val_current_mae,
            "val_win_rate_vs_current": _finite_mean(win_flags),
            "val_generated_current_mae": _finite_mean(
                generated_current_mae,
                default=float("nan"),
            ),
            "val_real_current_mae": _finite_mean(
                real_current_mae,
                default=float("nan"),
            ),
            "val_calendar": _finite_mean(calendar, default=float("nan")),
            "val_butterfly": _finite_mean(butterfly, default=float("nan")),
            "val_penalty_mean": _finite_mean(penalty_mean),
            "val_penalty_std": _finite_mean(penalty_std),
            "val_weight_entropy": _finite_mean(weight_entropy),
            "val_mc_surface_mae_se": _finite_mean(
                mc_surface_mae_se,
                default=float("nan"),
            ),
        }
        metrics.update(self._evaluate_matching_diagnostics())
        return metrics

    def _checkpoint_payload(self) -> dict[str, object]:
        assert self.bundle is not None
        assert self.generator is not None
        transition_matching = (
            self.critic is not None
            and self.critic.conditioning_mode == "transition_matching"
        )
        v3_protocol = (
            str(self.config.training_protocol_version).strip()
            == TRAINING_PROTOCOL_VERSION_V3
        )
        return {
            "checkpoint_schema_version": (
                CHECKPOINT_SCHEMA_VERSION_V3 if v3_protocol else 5
            ),
            "architecture_version": "pair_text_residual_film_v1"
            if self.generator.conditioning_mode == "residual_film"
            else "film_wgan_legacy_v2",
            "critic_architecture_version": (
                CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
                if transition_matching
                else "legacy_v1"
            ),
            "training_protocol_version": (
                TRAINING_PROTOCOL_VERSION_V3
                if v3_protocol
                else "film_wgan_transition_matching_v2"
                if transition_matching
                else "film_wgan_v1_compatible"
            ),
            "diagnostics_schema_version": (
                int(self.config.diagnostics_schema_version)
                if v3_protocol
                else 0
            ),
            "diagnostics": dict(self._latest_diagnostics),
            "run_fingerprint_sha256": str(self.config.run_fingerprint_sha256),
            "text_alignment_plan_path": str(self.config.text_alignment_plan_path),
            "text_alignment_plan_sha256": self._text_alignment_plan_sha256,
            "matching_negative_source_plan_path": str(
                self.config.matching_negative_source_plan_path
            ),
            "matching_negative_source_plan_sha256": (
                self._matching_negative_source_plan_sha256
            ),
            "matching_negative_source_plan_version": (
                MATCHING_NEGATIVE_SOURCE_PLAN_VERSION if v3_protocol else ""
            ),
            "text_alignment_plan_version": (
                TEXT_ALIGNMENT_PLAN_VERSION if v3_protocol else ""
            ),
            "matching_negative_source_plan_sha256_by_split": dict(
                self._matching_plan_sha256_by_split
            ),
            "matching_positive_alignment_sha256": (
                self._matching_positive_alignment_sha256
            ),
            "matching_positive_alignment_sha256_by_split": dict(
                self._matching_positive_alignment_sha256_by_split
            ),
            "matching_donor_mapping_sha256": self._matching_donor_mapping_sha256,
            "matching_eligible_samples": int(self._matching_eligible_samples),
            "matching_eligible_samples_by_split": dict(
                self._matching_eligible_samples_by_split
            ),
            "native_positive_as_negative_count": sum(
                values["native_positive_as_negative_count"]
                for values in self._matching_plan_violation_counts_by_split.values()
            ),
            "placebo_positive_as_negative_count": sum(
                values["placebo_positive_as_negative_count"]
                for values in self._matching_plan_violation_counts_by_split.values()
            ),
            "matching_plan_violation_counts_by_split": dict(
                self._matching_plan_violation_counts_by_split
            ),
            "scheduler_horizon_epochs": int(
                self._resolved_scheduler_horizon_epochs
                or self.config.scheduler_horizon_epochs
                or self.config.num_epochs
            ),
            "epoch": int(self._current_epoch),
            "forecast_mode": str(self.config.forecast_mode),
            "conditioning_mode": str(self.config.conditioning_mode),
            "critic_conditioning_mode": str(self.config.critic_conditioning_mode),
            "config": config_payload(self.config),
            "surface_shape": list(self.bundle.surface_shape),
            "strike_grid": self.bundle.strike_grid.astype(float).tolist(),
            "maturity_days_grid": self.bundle.maturity_days_grid.astype(float).tolist(),
            "embedding_dim": int(self.bundle.embedding_dim),
            "normalization_stats": {
                "current_log_mean": self.bundle.normalization_stats.current_log_mean.astype(float).tolist(),
                "current_log_std": self.bundle.normalization_stats.current_log_std.astype(float).tolist(),
                "delta_mean": self.bundle.normalization_stats.delta_mean.astype(float).tolist(),
                "delta_std": self.bundle.normalization_stats.delta_std.astype(float).tolist(),
                "text_mean": self.bundle.normalization_stats.text_mean.astype(float).tolist(),
                "text_std": self.bundle.normalization_stats.text_std.astype(float).tolist(),
            },
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict() if self.critic is not None else None,
            "generator_parameter_count": parameter_count(self.generator.parameters()),
            "critic_parameter_count": parameter_count(self.critic.parameters()) if self.critic is not None else 0,
            "parent_generator_checkpoint_path": self._parent_checkpoint_path,
            "parent_generator_checkpoint_sha256": self._parent_checkpoint_sha256,
            "parent_text_transform_policy": str(self.config.parent_text_transform_policy),
            "initial_generator_state_sha256": self._initial_generator_state_sha256,
            "initial_critic_state_sha256": self._initial_critic_state_sha256,
            "text_transform_path": self.bundle.text_transform_path,
            "text_transform_sha256": self.bundle.text_transform_sha256,
            "surface_support_path": self.bundle.surface_support_path,
            "surface_support_sha256": self.bundle.surface_support_sha256,
            "current_surface_channels": int(self.generator.current_surface_channels),
            "backbone_frozen": bool(self._backbone_frozen),
        }

    def _save_loss_curves(self, metrics_rows: list[dict[str, float]]) -> None:
        plot_training_curves(
            metrics_rows,
            output_path=self.metrics_dir / "loss_curves.png",
            title="Standalone FiLM WGAN Training Curves",
        )

    def _train_impl(self) -> Path:
        assert self.bundle is not None
        assert self.checkpoints_dir is not None
        assert self.metrics_dir is not None

        primary_metric = str(self.config.checkpoint_metric).strip() or "val_mae_gap_vs_current"
        configured_extra_metrics: list[str] = []
        seen_metrics = {primary_metric}
        for raw_metric in getattr(self.config, "extra_checkpoint_metrics", ()) or ():
            metric_name = str(raw_metric).strip()
            if not metric_name or metric_name in seen_metrics:
                continue
            configured_extra_metrics.append(metric_name)
            seen_metrics.add(metric_name)

        summary_metrics: list[str] = []
        for metric_name in [
            primary_metric,
            "val_mae_gap_vs_current",
            "val_short_atm_mae_gap_vs_current",
            "val_atm_short_pure_mae_gap_vs_current",
            *configured_extra_metrics,
        ]:
            if metric_name and metric_name not in summary_metrics:
                summary_metrics.append(metric_name)

        extra_checkpoint_paths = {
            metric_name: self.checkpoints_dir / f"film_wgan_best_{metric_name}.pt"
            for metric_name in configured_extra_metrics
        }
        metric_summary: dict[str, dict[str, object]] = {}
        for metric_name in summary_metrics:
            tracking_mode = "summary_only"
            if metric_name == primary_metric:
                tracking_mode = "primary"
            elif metric_name in configured_extra_metrics:
                tracking_mode = "extra"
            metric_summary[metric_name] = {
                "best_epoch": 0,
                "best_value": None,
                "checkpoint_path": "",
                "tracking_mode": tracking_mode,
                "is_primary": metric_name == primary_metric,
                "is_extra": metric_name in configured_extra_metrics,
            }

        metrics_rows: list[dict[str, float]] = []
        best_metric = float("inf")
        best_epoch = 0
        fallback_metric = float("inf")
        fallback_epoch = 0
        checkpoint_warmup_epochs = max(0, int(getattr(self.config, "checkpoint_warmup_epochs", 0)))
        selection_start_epoch = checkpoint_warmup_epochs + 1
        fallback_checkpoint_path = self.checkpoints_dir / "film_wgan_best_warmup_fallback.pt"
        fallback_used = False

        early_stopping_enabled = bool(getattr(self.config, "use_early_stopping", False))
        early_stopping_patience = max(1, int(getattr(self.config, "early_stopping_patience", 10)))
        early_stopping_min_delta = float(getattr(self.config, "early_stopping_min_delta", 0.0))
        epochs_without_improvement = 0
        early_stopped = False

        if checkpoint_warmup_epochs > 0:
            self.logger.info(
                "Best checkpoint selection warmup enabled: skipping epochs <= %s; selection begins at epoch %s.",
                checkpoint_warmup_epochs,
                selection_start_epoch,
            )
        if early_stopping_enabled:
            self.logger.info(
                "Early stopping enabled (patience=%d, min_delta=%.6f) on %s.",
                early_stopping_patience,
                early_stopping_min_delta,
                primary_metric,
            )

        for epoch in range(1, int(self.config.num_epochs) + 1):
            self._current_epoch = epoch
            self._update_backbone_freeze_state(epoch)
            running: dict[str, list[float]] = {
                "d_total": [],
                "d_real": [],
                "d_fake": [],
                "d_mismatch": [],
                "d_wgan": [],
                "d_matching": [],
                "d_aux_matching_contribution": [],
                "d_matching_positive": [],
                "d_matching_negative": [],
                "d_matching_margin": [],
                "d_matching_pairwise_accuracy": [],
                "d_matching_eligible_fraction": [],
                "gp": [],
                "g_total": [],
                "g_adv": [],
                "g_adv_effective_lambda": [],
                "g_matching": [],
                "g_matching_margin": [],
                "g_matching_effective_lambda": [],
                "g_saturation_rate": [],
                "g_transition_delivery_max_abs_gap": [],
                "g_transition_nonsaturated_max_abs_error": [],
                "g_calendar": [],
                "g_butterfly": [],
                "g_smooth": [],
                "g_recon": [],
                "g_recon_weighted": [],
                "g_atm_short": [],
                "g_film": [],
            }
            gp_metric_rows: list[Mapping[str, object]] = []
            g_metric_rows: list[Mapping[str, float]] = []
            for batch in self.bundle.train_loader:
                (
                    current_features,
                    text_features,
                    real_delta_norm,
                    current_flat,
                    target_flat,
                ) = batch[:5]
                extra_index = 5
                if str(self.config.surface_support_mode).strip().lower() == "raw_observed":
                    support_mask_flat = batch[extra_index]
                    extra_index += 1
                else:
                    support_mask_flat = torch.ones_like(current_flat)
                if str(self.config.conditioning_mode).strip().lower() == "residual_film":
                    has_text = batch[extra_index]
                    extra_index += 1
                else:
                    has_text = torch.ones(current_features.size(0), dtype=torch.float32)
                sample_indices = None
                if (
                    str(self.config.critic_conditioning_mode).strip().lower()
                    == "transition_matching"
                ):
                    sample_indices = batch[extra_index]
                current_features = self._to_device(current_features)
                text_features = self._to_device(text_features)
                real_delta_norm = self._to_device(real_delta_norm)
                current_flat = self._to_device(current_flat)
                target_flat = self._to_device(target_flat)
                has_text = self._to_device(has_text)
                support_mask_flat = self._to_device(support_mask_flat)
                if sample_indices is not None:
                    sample_indices = self._to_device(sample_indices)

                if self.critic is not None:
                    critic_steps = max(1, int(self.config.critic_iter))
                    for critic_step_index in range(critic_steps):
                        update_matching = critic_step_index == 0
                        d_metrics = self._discriminator_step(
                            current_features,
                            text_features,
                            real_delta_norm,
                            current_flat,
                            target_flat,
                            has_text,
                            support_mask_flat,
                            sample_indices,
                            update_matching=update_matching,
                        )
                        gp_metric_rows.append(
                            {
                                **d_metrics,
                                "gp_raw_norm_values": list(
                                    self._last_gp_raw_norm_values
                                ),
                            }
                        )
                        for key, value in d_metrics.items():
                            if key.startswith("d_matching") and not update_matching:
                                continue
                            if key not in running:
                                continue
                            running[key].append(float(value))

                g_metrics = self._generator_step(
                    current_features,
                    text_features,
                    current_flat,
                    target_flat,
                    has_text,
                    support_mask_flat,
                    sample_indices,
                )
                g_metric_rows.append(g_metrics)
                for key, value in g_metrics.items():
                    if key not in running:
                        continue
                    running[key].append(float(value))

            row = {
                "epoch": float(epoch),
                "d_total": float(np.mean(running["d_total"])) if running["d_total"] else 0.0,
                "d_real": float(np.mean(running["d_real"])) if running["d_real"] else 0.0,
                "d_fake": float(np.mean(running["d_fake"])) if running["d_fake"] else 0.0,
                "d_mismatch": float(np.mean(running["d_mismatch"])) if running["d_mismatch"] else 0.0,
                "d_wgan": float(np.mean(running["d_wgan"])) if running["d_wgan"] else 0.0,
                "d_matching": float(np.mean(running["d_matching"])) if running["d_matching"] else 0.0,
                "d_aux_matching_contribution": float(
                    np.mean(running["d_aux_matching_contribution"])
                )
                if running["d_aux_matching_contribution"]
                else 0.0,
                "d_matching_positive": float(np.mean(running["d_matching_positive"]))
                if running["d_matching_positive"]
                else 0.0,
                "d_matching_negative": float(np.mean(running["d_matching_negative"]))
                if running["d_matching_negative"]
                else 0.0,
                "d_matching_margin": float(np.mean(running["d_matching_margin"]))
                if running["d_matching_margin"]
                else 0.0,
                "d_matching_pairwise_accuracy": float(
                    np.mean(running["d_matching_pairwise_accuracy"])
                )
                if running["d_matching_pairwise_accuracy"]
                else 0.0,
                "d_matching_eligible_fraction": float(
                    np.mean(running["d_matching_eligible_fraction"])
                )
                if running["d_matching_eligible_fraction"]
                else 0.0,
                "gp": float(np.mean(running["gp"])) if running["gp"] else 0.0,
                "g_total": float(np.mean(running["g_total"])) if running["g_total"] else 0.0,
                "g_adv": float(np.mean(running["g_adv"])) if running["g_adv"] else 0.0,
                "g_adv_effective_lambda": float(np.mean(running["g_adv_effective_lambda"]))
                if running["g_adv_effective_lambda"]
                else 0.0,
                "g_matching": float(np.mean(running["g_matching"]))
                if running["g_matching"]
                else 0.0,
                "g_matching_margin": float(np.mean(running["g_matching_margin"]))
                if running["g_matching_margin"]
                else 0.0,
                "g_matching_effective_lambda": float(
                    np.mean(running["g_matching_effective_lambda"])
                )
                if running["g_matching_effective_lambda"]
                else 0.0,
                "g_saturation_rate": float(np.mean(running["g_saturation_rate"]))
                if running["g_saturation_rate"]
                else 0.0,
                "g_transition_delivery_max_abs_gap": float(
                    np.max(running["g_transition_delivery_max_abs_gap"])
                )
                if running["g_transition_delivery_max_abs_gap"]
                else 0.0,
                "g_transition_nonsaturated_max_abs_error": float(
                    np.max(running["g_transition_nonsaturated_max_abs_error"])
                )
                if running["g_transition_nonsaturated_max_abs_error"]
                else 0.0,
                "g_calendar": float(np.mean(running["g_calendar"])) if running["g_calendar"] else 0.0,
                "g_butterfly": float(np.mean(running["g_butterfly"])) if running["g_butterfly"] else 0.0,
                "g_smooth": float(np.mean(running["g_smooth"])) if running["g_smooth"] else 0.0,
                "g_recon": float(np.mean(running["g_recon"])) if running["g_recon"] else 0.0,
                "g_recon_weighted": float(np.mean(running["g_recon_weighted"])) if running["g_recon_weighted"] else 0.0,
                "g_atm_short": float(np.mean(running["g_atm_short"])) if running["g_atm_short"] else 0.0,
                "g_film": float(np.mean(running["g_film"])) if running["g_film"] else 0.0,
                "text_gate": float(torch.tanh(self.generator.text_gate).detach().cpu())
                if self.generator is not None and self.generator.conditioning_mode == "residual_film"
                else 0.0,
                "backbone_frozen": 1.0 if self._backbone_frozen else 0.0,
            }
            row.update(aggregate_gradient_penalty_diagnostics(gp_metric_rows))
            row.update(aggregate_transition_delivery_diagnostics(g_metric_rows))
            row.update(self._evaluate())
            row.update(self._evaluate_matching_gradient_probe())
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
            row["lr_generator"] = float(self.generator_optimizer.param_groups[0]["lr"]) if self.generator_optimizer else 0.0
            row["lr_critic"] = float(self.critic_optimizer.param_groups[0]["lr"]) if self.critic_optimizer else 0.0
            self._latest_diagnostics = {
                key: float(value)
                for key, value in row.items()
                if key.startswith(
                    (
                        "gp_",
                        "diag_",
                        "g_transition_",
                        "g_matching_gradient_",
                        "g_nonmatching_gradient_",
                        "val_matching_",
                    )
                )
            }
            metrics_rows.append(row)
            write_json(self.metrics_dir / "training_metrics.json", metrics_rows)
            write_csv(self.metrics_dir / "training_metrics.csv", metrics_rows)
            self._save_loss_curves(metrics_rows)

            monitor_name = primary_metric
            monitor_value = float(row.get(monitor_name, row.get("val_mae", row["g_total"])))
            if monitor_value < fallback_metric:
                fallback_metric = monitor_value
                fallback_epoch = epoch
                save_checkpoint(fallback_checkpoint_path, self._checkpoint_payload())
            improved = False
            if epoch > checkpoint_warmup_epochs and monitor_value < (best_metric - early_stopping_min_delta):
                best_metric = monitor_value
                best_epoch = epoch
                improved = True
                epochs_without_improvement = 0
                save_checkpoint(self.checkpoints_dir / "film_wgan_best.pt", self._checkpoint_payload())
            elif epoch > checkpoint_warmup_epochs and early_stopping_enabled:
                epochs_without_improvement += 1
                self.logger.info(
                    "Early stopping patience %d/%d without %s improvement (current=%.6f, best=%.6f at epoch %d)",
                    epochs_without_improvement,
                    early_stopping_patience,
                    monitor_name,
                    monitor_value,
                    best_metric,
                    best_epoch,
                )
            if epoch > checkpoint_warmup_epochs:
                for metric_name, summary in metric_summary.items():
                    if metric_name not in row:
                        continue
                    metric_value = float(row[metric_name])
                    current_best = summary["best_value"]
                    if current_best is None or metric_value < float(current_best):
                        summary["best_epoch"] = int(epoch)
                        summary["best_value"] = float(metric_value)
                        if summary["tracking_mode"] == "extra":
                            save_checkpoint(extra_checkpoint_paths[metric_name], self._checkpoint_payload())
            save_every = int(self.config.save_every)
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(self.checkpoints_dir / f"film_wgan_epoch_{epoch:04d}.pt", self._checkpoint_payload())

            self.logger.info(
                "epoch=%s g_total=%.6f d_total=%.6f val_mae=%.6f val_current_mae=%.6f gap=%.6f short_atm_gap=%.6f atm_short_pure_gap=%.6f win_rate=%.3f",
                epoch,
                row["g_total"],
                row["d_total"],
                row.get("val_mae", 0.0),
                row.get("val_current_mae", 0.0),
                row.get("val_mae_gap_vs_current", 0.0),
                row.get("val_short_atm_mae_gap_vs_current", 0.0),
                row.get("val_atm_short_pure_mae_gap_vs_current", 0.0),
                row.get("val_win_rate_vs_current", 0.0),
            )

            if (
                early_stopping_enabled
                and epoch > checkpoint_warmup_epochs
                and best_epoch > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                self.logger.info(
                    "Early stopping triggered at epoch %d. Best %s=%.6f at epoch %d.",
                    epoch,
                    monitor_name,
                    best_metric,
                    best_epoch,
                )
                early_stopped = True
                break

        if best_epoch == 0 and fallback_epoch > 0:
            fallback_used = True
            best_metric = fallback_metric
            best_epoch = fallback_epoch
            fallback_checkpoint_path.replace(self.checkpoints_dir / "film_wgan_best.pt")
            self.logger.warning(
                "No epoch exceeded checkpoint_warmup_epochs=%s during training; falling back to overall best epoch=%s.",
                checkpoint_warmup_epochs,
                best_epoch,
            )
        elif fallback_checkpoint_path.exists():
            fallback_checkpoint_path.unlink()

        save_checkpoint(self.checkpoints_dir / "film_wgan_final.pt", self._checkpoint_payload())
        write_json(
            self.metrics_dir / "best_checkpoint.json",
            {
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "checkpoint_metric": str(self.config.checkpoint_metric),
                "checkpoint_warmup_epochs": int(checkpoint_warmup_epochs),
                "selection_start_epoch": int(selection_start_epoch),
                "fallback_used": bool(fallback_used),
                "early_stopped": bool(early_stopped),
                "checkpoint_path": str(self.checkpoints_dir / "film_wgan_best.pt"),
            },
        )
        primary_summary = metric_summary[primary_metric]
        primary_summary["best_epoch"] = int(best_epoch)
        primary_summary["best_value"] = float(best_metric)
        primary_summary["checkpoint_path"] = str(self.checkpoints_dir / "film_wgan_best.pt")
        for metric_name, summary in metric_summary.items():
            if summary["tracking_mode"] == "extra" and int(summary["best_epoch"]) > 0:
                summary["checkpoint_path"] = str(extra_checkpoint_paths[metric_name])
        write_json(
            self.metrics_dir / "best_metrics_summary.json",
            {
                "primary_metric": primary_metric,
                "checkpoint_warmup_epochs": int(checkpoint_warmup_epochs),
                "selection_start_epoch": int(selection_start_epoch),
                "metrics": metric_summary,
            },
        )
        self.logger.info("Standalone FiLM WGAN training complete. Best epoch=%s best_metric=%.6f", best_epoch, best_metric)
        return self.run_dir

    def _prepare_generate_result(
        self,
        generate_config: FilmWGANSampleConfig | None = None,
        *,
        overrides: Mapping[str, object] | None = None,
        config_path: str | None = None,
    ) -> tuple[FilmWGANSampleConfig, Path]:
        base_training_config = self.config if self._runtime_prepared else self.raw_config
        output_root = infer_training_output_root(base_training_config, trainer_id=self.trainer_id)
        checkpoint_override = None if not overrides else overrides.get("checkpoint_path")

        if generate_config is None:
            if config_path:
                generate_config = build_sample_config_from_train_config(
                    config_path,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                    overrides=overrides,
                )
            elif self.config_path:
                generate_config = build_sample_config_from_train_config(
                    self.config_path,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                    overrides=overrides,
                )
            else:
                generate_config = build_sample_config(
                    training_values=config_to_dict(base_training_config),
                    generate_values=_load_generate_result_section(config_path),
                    overrides=overrides,
                    checkpoint_path=None if checkpoint_override in {None, ""} else str(checkpoint_override),
                )
        elif overrides:
            generate_config = replace(generate_config, **dict(overrides))

        run_dir = self.run_dir or resolve_existing_run_dir(
            output_root=output_root,
            checkpoint_path=generate_config.checkpoint_path or None,
        )
        resolved_checkpoint_path = (
            Path(generate_config.checkpoint_path)
            if str(generate_config.checkpoint_path).strip()
            else find_best_checkpoint(run_dir, filename="film_wgan_best.pt")
        )
        if str(generate_config.output_dir).strip():
            resolved_generate_dir = generate_result_dir(run_dir, generate_config.output_dir)
        else:
            resolved_generate_dir = checkpoint_named_dir(generate_result_dir(run_dir), resolved_checkpoint_path)
        resolved_config = replace(
            generate_config,
            checkpoint_path=str(resolved_checkpoint_path),
            output_dir=str(resolved_generate_dir),
        )
        return resolved_config, resolved_generate_dir

    def _generate_result_impl(self, generate_config: FilmWGANSampleConfig, generate_dir: Path) -> Path:
        del generate_dir
        sampler = FilmWGANSampler(generate_config)
        return sampler.sample()
