"""Shared inference helpers for result generation and error analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import embedded_calendar
from quantlib.vol_surface.algo.svi_surface import SviVolSurface
from wgan_option.config import (
    FROZEN_EPOCH_LR_REPLAY_REFIT_MODE,
    NO_REFIT_MODE,
    Config,
    normalize_label_reliability_mode,
    normalize_refit_mode,
)
from wgan_option.models.common import (
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    LP_CONCAT_CRITIC_CONDITIONING_MODE,
    critic_normalization_fingerprint,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GAUSSIAN_GENERATOR_NOISE_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LEGACY_RESIDUAL_OUTPUT_MODE,
    STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    generator_current_input_fingerprint,
    generator_conditioning_fingerprint,
    generator_noise_fingerprint,
    critic_conditioning_fingerprint,
    normalize_critic_conditioning_mode,
    normalize_generator_current_input_mode,
    normalize_generator_conditioning_mode,
    normalize_generator_noise_mode,
    normalize_critic_normalization_mode,
    normalize_residual_output_mode,
    residual_output_fingerprint,
)
from wgan_option.models.generator import Generator
from wgan_option.models.svi_regressor import SviRegressor
from wgan_option.models.vol_regressor import VolSurfaceRegressor
from wgan_option.utils.merged_xlsx import (
    SVI_FEATURE_ORDER,
    _build_svi_matrix_from_params,
    _normalize_svi_matrix,
)

_RECONSTRUCTION_DAYCOUNT = DayCountBusN("BUS250", embedded_calendar(), 250)
_RECONSTRUCTION_VALUATION_DATE = date(2023, 1, 2)
_LABEL_RELIABILITY_LINEAGE_FIELDS = (
    "news_first_label_reliability_mode",
    "news_first_label_reliability_manifest_path",
    "news_first_label_reliability_manifest_sha256",
    "news_first_label_reliability_profile_sha256",
    "news_first_label_reliability_fold_id",
    "news_first_label_reliability_train_pair_universe_sha256",
    "news_first_label_reliability_train_data_sha256",
    "news_first_label_reliability_validation_data_sha256",
    "news_first_label_reliability_data_window_contract_sha256",
    "data_path",
    "news_first_common_eval_data_path",
    "news_first_train_end_utc",
    "news_first_validation_end_utc",
    "news_first_materialize_test_loader",
)


def build_inference_device(use_cuda: bool) -> torch.device:
    """Resolve the device used for inference."""

    return torch.device(
        "cuda:0" if (bool(use_cuda) and torch.cuda.is_available()) else "cpu"
    )


def ensure_matching_embedding_dim(
    *, sample_id: str, embedding: np.ndarray, embedding_dim: int
) -> None:
    """Fail fast if a sample's embedding width mismatches the checkpoint."""

    _coerce_inference_embedding(
        sample_id=sample_id,
        embedding=embedding,
        embedding_dim=embedding_dim,
    )


def _coerce_inference_embedding(
    *, sample_id: str, embedding: np.ndarray, embedding_dim: int
) -> np.ndarray:
    """Align inference-time embeddings to the checkpoint width.

    The `none` text mode is stored at training time as an all-zero vector with a
    positive width (for example width=1), but `load_vol_surface_samples()` and
    `load_svi_paired_samples()` represent `none` rows as zero-length arrays.
    For inference we pad those zero-length arrays back to the checkpoint width
    while preserving the fail-fast behavior for real mismatches.
    """

    embedding_array = np.asarray(embedding, dtype=np.float32)
    if int(embedding_array.size) == int(embedding_dim):
        return embedding_array
    if int(embedding_array.size) == 0 and int(embedding_dim) > 0:
        return np.zeros(int(embedding_dim), dtype=np.float32)
    raise ValueError(
        f"Embedding dimension mismatch for sample {sample_id}: "
        f"expected {embedding_dim}, got {embedding_array.size}"
    )


def deterministic_noise(
    noise_dim: int,
    seed: int,
    global_index: int,
    device: torch.device,
    *,
    sample_offset: int = 0,
) -> torch.Tensor:
    """Build deterministic generator noise for one sample."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(global_index) + int(sample_offset) * 1000003)
    noise = torch.randn((1, int(noise_dim)), generator=generator, dtype=torch.float32)
    return noise.to(device)


def resolve_checkpoint_residual_output_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve and validate a vol checkpoint's surface-output contract.

    Historical checkpoints predate output-contract metadata.  They retain the
    exact historical behavior by resolving to ``legacy_softplus``.  Identity
    residual checkpoints must carry the formula fingerprint so inference can
    never silently apply a different positive-domain transform.
    """

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_has_mode = "residual_output_mode" in raw_config
    config_mode = normalize_residual_output_mode(
        raw_config.get("residual_output_mode", LEGACY_RESIDUAL_OUTPUT_MODE)
    )

    metadata_value = checkpoint.get("residual_output_mode")
    metadata_mode = (
        normalize_residual_output_mode(metadata_value)
        if metadata_value is not None
        else None
    )
    if config_has_mode and metadata_mode is not None and config_mode != metadata_mode:
        raise ValueError(
            "Checkpoint residual_output_mode metadata disagrees with its saved config: "
            f"metadata={metadata_mode!r}, config={config_mode!r}"
        )
    mode = metadata_mode or config_mode
    expected_fingerprint = residual_output_fingerprint(mode)
    saved_fingerprint = checkpoint.get("residual_output_fingerprint")
    if saved_fingerprint is None:
        if mode == IDENTITY_RESIDUAL_OUTPUT_MODE:
            raise ValueError(
                "Identity-residual checkpoint is missing residual_output_fingerprint"
            )
    elif str(saved_fingerprint) != expected_fingerprint:
        raise ValueError(
            "Checkpoint residual-output fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {saved_fingerprint}"
        )
    return mode, expected_fingerprint


def resolve_checkpoint_generator_noise_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve latent semantics while preserving historical Gaussian checkpoints.

    Persisted checkpoints predating this contract contain neither a mode nor a
    fingerprint and therefore retain their historical Gaussian behavior.  A
    zero-noise checkpoint is accepted only when its explicit fingerprint
    matches the saved noise width, so deterministic experiments cannot be
    silently evaluated as stochastic ones (or vice versa).
    """

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_has_mode = "generator_noise_mode" in raw_config
    config_mode = normalize_generator_noise_mode(
        raw_config.get("generator_noise_mode", GAUSSIAN_GENERATOR_NOISE_MODE)
    )
    metadata_value = checkpoint.get("generator_noise_mode")
    metadata_mode = (
        normalize_generator_noise_mode(metadata_value)
        if metadata_value is not None
        else None
    )
    if config_has_mode and metadata_mode is not None and config_mode != metadata_mode:
        raise ValueError(
            "Checkpoint generator_noise_mode metadata disagrees with its saved config: "
            f"metadata={metadata_mode!r}, config={config_mode!r}"
        )
    mode = metadata_mode or config_mode
    noise_dim = int(raw_config.get("noise_dim", Config().noise_dim))
    expected_fingerprint = generator_noise_fingerprint(mode, noise_dim)
    saved_fingerprint = checkpoint.get("generator_noise_fingerprint")
    if saved_fingerprint is None:
        if mode == ZERO_GENERATOR_NOISE_MODE:
            raise ValueError(
                "Zero-noise checkpoint is missing generator_noise_fingerprint"
            )
    elif str(saved_fingerprint) != expected_fingerprint:
        raise ValueError(
            "Checkpoint generator-noise fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {saved_fingerprint}"
        )
    return mode, expected_fingerprint


def resolve_checkpoint_generator_current_input_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve Generator current-input semantics without leaking target support.

    Historical checkpoints predate this contract and therefore keep consuming
    the full current surface.  A checkpoint that requests current-side masking
    must carry the exact contract fingerprint; inference fails closed instead
    of silently falling back to the full or future-aware joint mask.
    """

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_has_mode = "generator_current_input_mode" in raw_config
    config_mode = normalize_generator_current_input_mode(
        raw_config.get(
            "generator_current_input_mode",
            FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
    )
    metadata_value = checkpoint.get("generator_current_input_mode")
    metadata_mode = (
        normalize_generator_current_input_mode(metadata_value)
        if metadata_value is not None
        else None
    )
    if config_has_mode and metadata_mode is not None and config_mode != metadata_mode:
        raise ValueError(
            "Checkpoint generator_current_input_mode metadata disagrees with its "
            f"saved config: metadata={metadata_mode!r}, config={config_mode!r}"
        )
    mode = metadata_mode or config_mode
    if mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE:
        if not config_has_mode:
            raise ValueError(
                "Current-support-masked checkpoint config must explicitly save "
                "generator_current_input_mode"
            )
        support_mode = str(raw_config.get("support_mask_mode", "none")).strip().lower()
        if support_mode != "raw_joint":
            raise ValueError(
                "Current-support-masked checkpoint config requires "
                "support_mask_mode='raw_joint'"
            )
    expected_fingerprint = generator_current_input_fingerprint(mode)
    saved_fingerprint = checkpoint.get("generator_current_input_fingerprint")
    if saved_fingerprint is None:
        if mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE:
            raise ValueError(
                "Current-support-masked checkpoint is missing "
                "generator_current_input_fingerprint"
            )
    elif str(saved_fingerprint) != expected_fingerprint:
        raise ValueError(
            "Checkpoint generator-current-input fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {saved_fingerprint}"
        )
    return mode, expected_fingerprint


def _resolve_checkpoint_conditioning_contract(
    checkpoint: Mapping[str, Any],
    *,
    mode_field: str,
    fingerprint_field: str,
    default_mode: str,
    normalize_mode,
    fingerprint,
) -> tuple[str, str]:
    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_has_mode = mode_field in raw_config
    config_mode = normalize_mode(raw_config.get(mode_field, default_mode))
    metadata_value = checkpoint.get(mode_field)
    metadata_mode = (
        normalize_mode(metadata_value) if metadata_value is not None else None
    )
    if config_has_mode and metadata_mode is not None and config_mode != metadata_mode:
        raise ValueError(
            f"Checkpoint {mode_field} metadata disagrees with its saved config: "
            f"metadata={metadata_mode!r}, config={config_mode!r}"
        )
    mode = metadata_mode or config_mode
    expected_fingerprint = fingerprint(mode)
    saved_fingerprint = checkpoint.get(fingerprint_field)
    if mode != default_mode:
        missing = []
        if not config_has_mode:
            missing.append(f"config.{mode_field}")
        if metadata_mode is None:
            missing.append(mode_field)
        if saved_fingerprint in (None, ""):
            missing.append(fingerprint_field)
        if missing:
            raise ValueError(
                f"Non-default conditioning checkpoint is missing metadata: {missing}"
            )
    if (
        saved_fingerprint not in (None, "")
        and str(saved_fingerprint) != expected_fingerprint
    ):
        raise ValueError(
            f"Checkpoint {fingerprint_field} mismatch: expected "
            f"{expected_fingerprint}, got {saved_fingerprint}"
        )
    return mode, expected_fingerprint


def resolve_checkpoint_generator_conditioning_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve Generator LP conditioning, defaulting historical checkpoints."""

    return _resolve_checkpoint_conditioning_contract(
        checkpoint,
        mode_field="generator_conditioning_mode",
        fingerprint_field="generator_conditioning_fingerprint",
        default_mode=BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        normalize_mode=normalize_generator_conditioning_mode,
        fingerprint=generator_conditioning_fingerprint,
    )


def resolve_checkpoint_critic_conditioning_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve Critic LP conditioning, defaulting historical checkpoints."""

    return _resolve_checkpoint_conditioning_contract(
        checkpoint,
        mode_field="critic_conditioning_mode",
        fingerprint_field="critic_conditioning_fingerprint",
        default_mode=LP_CONCAT_CRITIC_CONDITIONING_MODE,
        normalize_mode=normalize_critic_conditioning_mode,
        fingerprint=critic_conditioning_fingerprint,
    )


def _checkpoint_lr_trace(
    value: object,
    *,
    field_name: str,
    num_epochs: int,
) -> list[dict[str, float | int]]:
    if not isinstance(value, list) or len(value) != num_epochs:
        raise ValueError(f"{field_name} must contain exactly {num_epochs} rows")
    result: list[dict[str, float | int]] = []
    for expected_epoch, row in enumerate(value, 1):
        if not isinstance(row, Mapping) or int(row.get("epoch", -1)) != expected_epoch:
            raise ValueError(f"{field_name} epochs must be exactly 1..{num_epochs}")
        learning_rate = float(row.get("lr", float("nan")))
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"{field_name} learning rates must be finite and positive")
        result.append({"epoch": expected_epoch, "lr": learning_rate})
    return result


def resolve_checkpoint_refit_contract(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate final Stage-B fixed-epoch/LR-replay lineage."""

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_has_mode = "news_first_refit_mode" in raw_config
    config_mode = normalize_refit_mode(
        raw_config.get("news_first_refit_mode", NO_REFIT_MODE)
    )
    metadata_value = checkpoint.get("news_first_refit_mode")
    metadata_mode = (
        normalize_refit_mode(metadata_value) if metadata_value is not None else None
    )
    if config_has_mode and metadata_mode is not None and config_mode != metadata_mode:
        raise ValueError(
            "Checkpoint news_first_refit_mode disagrees with its saved config"
        )
    mode = metadata_mode or config_mode
    result: Dict[str, Any] = {"news_first_refit_mode": mode}
    if mode != FROZEN_EPOCH_LR_REPLAY_REFIT_MODE:
        return result
    required = (
        "news_first_refit_mode",
        "news_first_refit_recipe_path",
        "news_first_refit_recipe_sha256",
        "refit_recipe_schema_version",
        "refit_num_epochs",
        "refit_generator_lr_trace",
        "refit_discriminator_lr_trace",
    )
    missing = [field for field in required if checkpoint.get(field) in (None, "")]
    if not config_has_mode:
        missing.append("config.news_first_refit_mode")
    for field in (
        "news_first_refit_recipe_path",
        "news_first_refit_recipe_sha256",
    ):
        if field not in raw_config:
            missing.append(f"config.{field}")
        elif str(checkpoint.get(field, "")) != str(raw_config.get(field, "")):
            raise ValueError(f"Checkpoint {field} disagrees with its saved config")
    if missing:
        raise ValueError(
            f"Frozen refit checkpoint is missing lineage: {sorted(missing)}"
        )
    if int(checkpoint["refit_recipe_schema_version"]) != 1:
        raise ValueError("Frozen refit checkpoint recipe schema must be 1")
    num_epochs = int(checkpoint["refit_num_epochs"])
    if num_epochs <= 0 or num_epochs != int(raw_config.get("num_epochs", 0)):
        raise ValueError("Frozen refit checkpoint epoch count disagrees with config")
    recipe_sha = str(checkpoint["news_first_refit_recipe_sha256"])
    if len(recipe_sha) != 64 or any(
        char not in "0123456789abcdef" for char in recipe_sha
    ):
        raise ValueError("Frozen refit checkpoint recipe SHA256 is malformed")
    result.update(
        {
            "news_first_refit_recipe_path": str(
                checkpoint["news_first_refit_recipe_path"]
            ),
            "news_first_refit_recipe_sha256": recipe_sha,
            "refit_num_epochs": num_epochs,
            "refit_generator_lr_trace": _checkpoint_lr_trace(
                checkpoint["refit_generator_lr_trace"],
                field_name="refit_generator_lr_trace",
                num_epochs=num_epochs,
            ),
            "refit_discriminator_lr_trace": _checkpoint_lr_trace(
                checkpoint["refit_discriminator_lr_trace"],
                field_name="refit_discriminator_lr_trace",
                num_epochs=num_epochs,
            ),
        }
    )
    return result


def resolve_checkpoint_surface_grid_contract(
    checkpoint: Mapping[str, Any], sample: Any
) -> Dict[str, Any]:
    """Validate explicit grid/model lineage while keeping legacy checkpoints readable."""

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    declared = bool(
        checkpoint.get("surface_grid_profile")
        or raw_config.get("news_first_surface_grid_profile")
    )
    if not declared:
        return {
            "surface_grid_profile": "legacy_unspecified",
            "surface_grid_sha256": "",
            "critic_normalization_mode": normalize_critic_normalization_mode(
                raw_config.get("critic_normalization_mode", None)
            ),
        }

    required = (
        "surface_grid_profile",
        "surface_grid_sha256",
        "surface_shape",
        "strike_grid",
        "maturity_days_grid",
        "critic_normalization_mode",
        "critic_normalization_fingerprint",
        "architecture_profile_sha256",
        "model_contract_sha256",
    )
    missing = [field for field in required if checkpoint.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Explicit-grid checkpoint is missing metadata: {missing}")

    profile = str(checkpoint["surface_grid_profile"])
    digest = str(checkpoint["surface_grid_sha256"])
    if profile != str(raw_config.get("news_first_surface_grid_profile", "")):
        raise ValueError("Checkpoint surface-grid profile disagrees with saved config")
    if digest != str(raw_config.get("news_first_surface_grid_sha256", "")):
        raise ValueError("Checkpoint surface-grid SHA disagrees with saved config")
    for metadata_field, config_field in (
        ("architecture_profile_sha256", "news_first_architecture_profile_sha256"),
        ("model_contract_sha256", "news_first_model_contract_sha256"),
    ):
        if str(checkpoint[metadata_field]) != str(raw_config.get(config_field, "")):
            raise ValueError(f"Checkpoint {metadata_field} disagrees with saved config")

    mode = normalize_critic_normalization_mode(checkpoint["critic_normalization_mode"])
    if mode != normalize_critic_normalization_mode(
        raw_config.get("critic_normalization_mode", None)
    ):
        raise ValueError("Checkpoint critic normalization disagrees with saved config")
    expected_norm_sha = critic_normalization_fingerprint(mode)
    if str(checkpoint["critic_normalization_fingerprint"]) != expected_norm_sha:
        raise ValueError("Checkpoint critic-normalization fingerprint mismatch")

    sample_shape = [
        int(sample.current_surface.shape[-2]),
        int(sample.current_surface.shape[-1]),
    ]
    if [int(value) for value in checkpoint["surface_shape"]] != sample_shape:
        raise ValueError(
            f"Checkpoint/sample surface shape mismatch: "
            f"{checkpoint['surface_shape']} != {sample_shape}"
        )
    sample_strikes = np.asarray(sample.strike_grid, dtype=np.float32)
    sample_maturities = np.asarray(sample.maturity_grid_days, dtype=np.float32)
    saved_strikes = np.asarray(checkpoint["strike_grid"], dtype=np.float32)
    saved_maturities = np.asarray(checkpoint["maturity_days_grid"], dtype=np.float32)
    if not np.array_equal(saved_strikes, sample_strikes):
        raise ValueError("Checkpoint/sample strike grid mismatch")
    if not np.array_equal(saved_maturities, sample_maturities):
        raise ValueError("Checkpoint/sample maturity grid mismatch")
    return {
        "surface_grid_profile": profile,
        "surface_grid_sha256": digest,
        "critic_normalization_mode": mode,
    }


def resolve_checkpoint_label_reliability_contract(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve train-label lineage, defaulting historical checkpoints to none."""

    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    config_lineage: Dict[str, Any] = {
        "news_first_label_reliability_mode": normalize_label_reliability_mode(
            raw_config.get("news_first_label_reliability_mode", "none")
        ),
        "news_first_label_reliability_manifest_path": str(
            raw_config.get("news_first_label_reliability_manifest_path", "") or ""
        ),
        "news_first_label_reliability_manifest_sha256": str(
            raw_config.get("news_first_label_reliability_manifest_sha256", "") or ""
        ),
        "news_first_label_reliability_profile_sha256": str(
            raw_config.get("news_first_label_reliability_profile_sha256", "") or ""
        ),
        "news_first_label_reliability_fold_id": str(
            raw_config.get("news_first_label_reliability_fold_id", "") or ""
        ),
        "news_first_label_reliability_train_pair_universe_sha256": str(
            raw_config.get(
                "news_first_label_reliability_train_pair_universe_sha256",
                "",
            )
            or ""
        ),
        "news_first_label_reliability_train_data_sha256": str(
            raw_config.get("news_first_label_reliability_train_data_sha256", "") or ""
        ),
        "news_first_label_reliability_validation_data_sha256": str(
            raw_config.get(
                "news_first_label_reliability_validation_data_sha256",
                "",
            )
            or ""
        ),
        "news_first_label_reliability_data_window_contract_sha256": str(
            raw_config.get(
                "news_first_label_reliability_data_window_contract_sha256",
                "",
            )
            or ""
        ),
        "data_path": str(raw_config.get("data_path", "") or ""),
        "news_first_common_eval_data_path": str(
            raw_config.get("news_first_common_eval_data_path", "") or ""
        ),
        "news_first_train_end_utc": str(
            raw_config.get("news_first_train_end_utc", "") or ""
        ),
        "news_first_validation_end_utc": str(
            raw_config.get("news_first_validation_end_utc", "") or ""
        ),
        "news_first_materialize_test_loader": bool(
            raw_config.get("news_first_materialize_test_loader", True)
        ),
    }
    validation_field = "news_first_materialize_validation_loader"
    if validation_field in raw_config or validation_field in checkpoint:
        config_lineage[validation_field] = bool(raw_config.get(validation_field, True))
    formal_lineage = bool(
        config_lineage["news_first_label_reliability_mode"] != "none"
        or config_lineage["news_first_label_reliability_manifest_path"]
    )
    metadata_present = {
        field: field in checkpoint for field in _LABEL_RELIABILITY_LINEAGE_FIELDS
    }
    if formal_lineage:
        missing = sorted(
            field for field, present in metadata_present.items() if not present
        )
        if missing:
            raise ValueError(
                "Formal label-reliability checkpoint is missing top-level lineage: "
                f"{missing}"
            )
    for field, present in metadata_present.items():
        if not present:
            continue
        observed = checkpoint[field]
        expected = config_lineage[field]
        if field == "news_first_label_reliability_mode":
            observed = normalize_label_reliability_mode(observed)
        elif field in {
            "news_first_materialize_test_loader",
            "news_first_materialize_validation_loader",
        }:
            observed = bool(observed)
        else:
            observed = str(observed or "")
        if observed != expected:
            raise ValueError(
                "Checkpoint label-reliability metadata disagrees with its saved "
                f"config for {field}: metadata={observed!r}, config={expected!r}"
            )
    if validation_field in checkpoint and bool(checkpoint[validation_field]) != bool(
        config_lineage.get(validation_field, True)
    ):
        raise ValueError(
            "Checkpoint label-reliability metadata disagrees with its saved "
            f"config for {validation_field}"
        )
    return config_lineage


def _sample_current_support_mask_tensor(
    sample: Any,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Return only the sample's current-side raw-support mask.

    The joint ``sample.support_mask`` is intentionally never considered here:
    it depends on the future target and would leak information into Generator
    conditioning.
    """

    current_support_mask = getattr(sample, "current_support_mask", None)
    if current_support_mask is None:
        return None
    mask = torch.as_tensor(
        current_support_mask,
        dtype=torch.float32,
        device=device,
    )
    if mask.ndim in {2, 3}:
        mask = mask.unsqueeze(0)
    return mask


def load_vol_generator(
    checkpoint_path: str | Path, sample: Any, device: torch.device
) -> tuple[Generator, Config, int]:
    """Load the trained generator used by vol-surface inference."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    residual_output_mode, _ = resolve_checkpoint_residual_output_contract(checkpoint)
    generator_noise_mode, _ = resolve_checkpoint_generator_noise_contract(checkpoint)
    generator_current_input_mode, _ = (
        resolve_checkpoint_generator_current_input_contract(checkpoint)
    )
    generator_conditioning_mode, _ = resolve_checkpoint_generator_conditioning_contract(
        checkpoint
    )
    critic_conditioning_mode, _ = resolve_checkpoint_critic_conditioning_contract(
        checkpoint
    )
    resolve_checkpoint_refit_contract(checkpoint)
    resolve_checkpoint_label_reliability_contract(checkpoint)
    resolve_checkpoint_surface_grid_contract(checkpoint, sample)
    train_config = Config(**checkpoint["config"])
    train_config.residual_output_mode = residual_output_mode
    train_config.generator_noise_mode = generator_noise_mode
    train_config.generator_current_input_mode = generator_current_input_mode
    train_config.generator_conditioning_mode = generator_conditioning_mode
    train_config.critic_conditioning_mode = critic_conditioning_mode
    embedding_dim = int(checkpoint.get("embedding_dim", train_config.embedding_dim))
    strike_grid = getattr(sample, "strike_grid", None)
    maturity_grid_days = getattr(sample, "maturity_grid_days", None)
    if generator_conditioning_mode in {
        FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    }:
        if strike_grid is None or maturity_grid_days is None:
            raise ValueError(
                f"{generator_conditioning_mode} inference requires sample "
                "strike_grid and maturity_grid_days"
            )

    model = Generator(
        channels=int(train_config.channels),
        embedding_dim=embedding_dim,
        noise_dim=int(train_config.noise_dim),
        surface_height=int(sample.current_surface.shape[1]),
        surface_width=int(sample.current_surface.shape[2]),
        base_channels=int(getattr(train_config, "gen_base_channels", 32)),
        res_blocks=int(getattr(train_config, "gen_res_blocks", 0)),
        text_hidden_dim=int(getattr(train_config, "gen_text_hidden_dim", 256)),
        text_out_dim=int(getattr(train_config, "gen_text_out_dim", 128)),
        hidden_dim=int(train_config.gen_hidden_dim),
        residual_output_mode=residual_output_mode,
        generator_noise_mode=generator_noise_mode,
        generator_current_input_mode=generator_current_input_mode,
        generator_conditioning_mode=generator_conditioning_mode,
        strike_grid=strike_grid,
        maturity_grid_days=maturity_grid_days,
        crossattn_heads=int(getattr(train_config, "gen_crossattn_heads", 4)),
        crossattn_text_tokens=int(
            getattr(train_config, "gen_crossattn_text_tokens", 4)
        ),
        crossattn_dim=int(getattr(train_config, "gen_crossattn_dim", 128)),
        transformer_model_dim=int(
            getattr(train_config, "gen_transformer_model_dim", 96)
        ),
        transformer_layers=int(getattr(train_config, "gen_transformer_layers", 4)),
        transformer_heads=int(getattr(train_config, "gen_transformer_heads", 8)),
        transformer_ffn_dim=int(getattr(train_config, "gen_transformer_ffn_dim", 384)),
        transformer_dropout=float(
            getattr(train_config, "gen_transformer_dropout", 0.1)
        ),
        style_dim=int(getattr(train_config, "gen_style_dim", 128)),
        style_demodulate=bool(getattr(train_config, "gen_style_demodulate", True)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, train_config, embedding_dim


def infer_vol_surface(
    model: Generator,
    sample: Any,
    *,
    noise_dim: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run one forward pass of the vol generator and return a 2D surface."""

    current_tensor = torch.tensor(
        sample.current_surface, dtype=torch.float32, device=device
    ).unsqueeze(0)
    if (
        hasattr(model, "text_encoder")
        and len(model.text_encoder) > 0
        and hasattr(model.text_encoder[0], "in_features")
    ):
        embedding_dim = int(model.text_encoder[0].in_features)
    else:
        embedding_dim = int(sample.text_embedding.size)
    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    text_tensor = torch.tensor(
        aligned_embedding, dtype=torch.float32, device=device
    ).unsqueeze(0)
    current_support_mask = _sample_current_support_mask_tensor(sample, device=device)
    if model.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE:
        noise = torch.zeros((1, int(noise_dim)), dtype=torch.float32, device=device)
    else:
        noise = deterministic_noise(noise_dim, seed, sample.global_index, device)

    with torch.no_grad():
        generated_surface = (
            model(
                current_tensor,
                text_tensor,
                noise=noise,
                current_support_mask=current_support_mask,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
    return np.asarray(generated_surface, dtype=np.float32)


def infer_vol_surface_mc(
    model: Generator,
    sample: Any,
    *,
    noise_dim: int,
    seed: int,
    device: torch.device,
    mc_samples: int,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Average deterministic MC draws and return the mean surface and uncertainty score."""

    current_tensor = torch.tensor(
        sample.current_surface, dtype=torch.float32, device=device
    ).unsqueeze(0)
    if (
        hasattr(model, "text_encoder")
        and len(model.text_encoder) > 0
        and hasattr(model.text_encoder[0], "in_features")
    ):
        embedding_dim = int(model.text_encoder[0].in_features)
    else:
        embedding_dim = int(sample.text_embedding.size)
    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    text_tensor = torch.tensor(
        aligned_embedding, dtype=torch.float32, device=device
    ).unsqueeze(0)
    current_support_mask = _sample_current_support_mask_tensor(sample, device=device)

    surfaces: list[np.ndarray] = []
    with torch.no_grad():
        effective_mc_samples = (
            1
            if model.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE
            else max(1, int(mc_samples))
        )
        for draw_idx in range(effective_mc_samples):
            if model.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE:
                noise = torch.zeros(
                    (1, int(noise_dim)), dtype=torch.float32, device=device
                )
            else:
                noise = deterministic_noise(
                    noise_dim,
                    seed,
                    sample.global_index,
                    device,
                    sample_offset=draw_idx,
                )
            generated_surface = (
                model(
                    current_tensor,
                    text_tensor,
                    noise=noise,
                    current_support_mask=current_support_mask,
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )
            surfaces.append(np.asarray(generated_surface, dtype=np.float32))

    stacked = np.stack(surfaces, axis=0).astype(np.float32)
    mean_surface = np.mean(stacked, axis=0).astype(np.float32)
    uncertainty_score = float(np.mean(np.std(stacked, axis=0, ddof=0)))
    return mean_surface, uncertainty_score, surfaces


def load_vol_regressor(
    checkpoint_path: str | Path,
    sample: Any,
    device: torch.device,
) -> tuple[VolSurfaceRegressor, Config, int]:
    """Load the trained deterministic vol regressor used by surface inference."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    residual_output_mode, _ = resolve_checkpoint_residual_output_contract(checkpoint)
    resolve_checkpoint_label_reliability_contract(checkpoint)
    train_config = Config(**checkpoint["config"])
    train_config.residual_output_mode = residual_output_mode
    embedding_dim = int(checkpoint.get("embedding_dim", train_config.embedding_dim))

    model = VolSurfaceRegressor(
        channels=int(train_config.channels),
        embedding_dim=embedding_dim,
        surface_height=int(sample.current_surface.shape[1]),
        surface_width=int(sample.current_surface.shape[2]),
        base_channels=int(getattr(train_config, "gen_base_channels", 32)),
        res_blocks=int(getattr(train_config, "gen_res_blocks", 0)),
        text_hidden_dim=int(getattr(train_config, "gen_text_hidden_dim", 256)),
        text_out_dim=int(getattr(train_config, "gen_text_out_dim", 128)),
        hidden_dim=int(train_config.gen_hidden_dim),
        residual_output_mode=residual_output_mode,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, train_config, embedding_dim


def infer_vol_regression_surface(
    model: VolSurfaceRegressor,
    sample: Any,
    *,
    device: torch.device,
) -> np.ndarray:
    """Run one deterministic forward pass of the vol regressor and return a 2D surface."""

    current_tensor = torch.tensor(
        sample.current_surface, dtype=torch.float32, device=device
    ).unsqueeze(0)
    if (
        hasattr(model, "text_encoder")
        and len(model.text_encoder) > 0
        and hasattr(model.text_encoder[0], "in_features")
    ):
        embedding_dim = int(model.text_encoder[0].in_features)
    else:
        embedding_dim = int(sample.text_embedding.size)
    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    text_tensor = torch.tensor(
        aligned_embedding, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        generated_surface = (
            model(current_tensor, text_tensor).detach().cpu().numpy()[0, 0]
        )
    return np.asarray(generated_surface, dtype=np.float32)


def load_svi_regressor(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[SviRegressor, Dict[str, Any]]:
    """Load the trained SVI regressor used by SVI inference."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = SviRegressor(
        current_input_dim=int(checkpoint["current_input_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        regression_dim=int(checkpoint["regression_dim"]),
        count_classes=int(checkpoint["max_slices"]),
        hidden_dim=int(checkpoint.get("config", {}).get("svi_hidden_dim", 256)),
        dropout=float(checkpoint.get("config", {}).get("svi_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_current_svi_feature_vector(
    svi_params: Dict[str, List[float]],
    *,
    max_slices: int,
    normalization_stats: Dict[str, Any],
    sample_label: str,
) -> Tuple[np.ndarray, int]:
    """Convert SVI parameters into the normalized current-state feature vector."""

    matrix, mask, count = _build_svi_matrix_from_params(
        svi_params, max_slices, sample_label=sample_label
    )
    normalized = _normalize_svi_matrix(matrix, mask, normalization_stats)
    feature_vector = np.concatenate(
        [
            normalized.reshape(-1),
            mask.astype(np.float32),
            np.asarray([float(count) / float(max_slices)], dtype=np.float32),
        ],
        axis=0,
    )
    return feature_vector.astype(np.float32), count


def sanitize_svi_params(svi_params: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Sort SVI slices by maturity and enforce positive, strictly increasing business days."""

    rows: List[Dict[str, float]] = []
    slice_count = len(svi_params["business_days"])
    for idx in range(slice_count):
        rows.append(
            {feature: float(svi_params[feature][idx]) for feature in SVI_FEATURE_ORDER}
        )

    rows.sort(key=lambda row: row["business_days"])
    previous_day = 0
    sanitized: List[Dict[str, float]] = []
    for row in rows:
        business_day = max(1, int(round(row["business_days"])))
        if business_day <= previous_day:
            business_day = previous_day + 1
        previous_day = business_day
        sanitized.append(
            {
                "business_days": float(business_day),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "rho": float(row["rho"]),
                "m": float(row["m"]),
                "sigma": max(float(row["sigma"]), 1e-8),
            }
        )

    return {
        feature: [float(row[feature]) for row in sanitized]
        for feature in SVI_FEATURE_ORDER
    }


def denormalize_svi_prediction(
    predicted_regression: np.ndarray,
    *,
    predicted_count: int,
    normalization_stats: Dict[str, Any],
    max_slices: int,
) -> Dict[str, List[float]]:
    """Map normalized SVI regression outputs back to raw SVI parameters."""

    feature_dim = len(SVI_FEATURE_ORDER)
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["std"], dtype=np.float32)
    regression_matrix = predicted_regression.reshape(int(max_slices), feature_dim)
    denormalized = regression_matrix * std.reshape(1, feature_dim) + mean.reshape(
        1, feature_dim
    )

    raw_params = {
        feature: denormalized[:predicted_count, feature_idx].astype(float).tolist()
        for feature_idx, feature in enumerate(SVI_FEATURE_ORDER)
    }
    return sanitize_svi_params(raw_params)


def reconstruct_svi_surface(
    svi_params: Dict[str, List[float]],
    *,
    strike_grid: np.ndarray,
    maturity_days_grid: np.ndarray,
) -> np.ndarray:
    """Reconstruct a vol surface from SVI parameters on a fixed grid."""

    surface = SviVolSurface(
        valuation_date=_RECONSTRUCTION_VALUATION_DATE,
        svi_params=sanitize_svi_params(svi_params),
        vol_daycount=_RECONSTRUCTION_DAYCOUNT,
    )
    grid = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid.tolist()],
        business_days=[int(round(value)) for value in maturity_days_grid.tolist()],
        forward=1.0,
    )
    return np.asarray(grid, dtype=np.float32)


def infer_future_svi(
    model: SviRegressor,
    sample: Any,
    *,
    embedding_dim: int,
    max_slices: int,
    normalization_stats: Dict[str, Any],
    device: torch.device,
) -> tuple[Dict[str, List[float]], int, int]:
    """Run one forward pass of the SVI regressor."""

    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    current_features, current_count = build_current_svi_feature_vector(
        sample.current_svi,
        max_slices=max_slices,
        normalization_stats=normalization_stats,
        sample_label=f"{sample.sample_id}:current",
    )
    current_tensor = torch.tensor(
        current_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    text_tensor = torch.tensor(
        aligned_embedding, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        predicted_regression, predicted_count_logits = model(
            current_tensor, text_tensor
        )

    predicted_regression_np = predicted_regression.detach().cpu().numpy()[0]
    predicted_count = int(torch.argmax(predicted_count_logits, dim=1).item()) + 1
    predicted_svi = denormalize_svi_prediction(
        predicted_regression_np,
        predicted_count=predicted_count,
        normalization_stats=normalization_stats,
        max_slices=max_slices,
    )
    return predicted_svi, current_count, predicted_count
