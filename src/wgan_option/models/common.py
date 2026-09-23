"""Shared building blocks for surface-based neural network models."""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_RESIDUAL_OUTPUT_MODE = "legacy_softplus"
IDENTITY_RESIDUAL_OUTPUT_MODE = "identity_softplus_residual"
RESIDUAL_OUTPUT_EPSILON = 1e-4
GAUSSIAN_GENERATOR_NOISE_MODE = "gaussian"
ZERO_GENERATOR_NOISE_MODE = "zero"
FULL_CURRENT_GENERATOR_INPUT_MODE = "full_current"
CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE = "current_support_masked"
BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE = "bottleneck_concat_v1"
FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE = (
    "film_conv_bottleneck_concat_v1"
)
FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE = (
    "film_conv_no_bottleneck_concat_v1"
)
FILM_UNET_GENERATOR_CONDITIONING_MODE = "film_unet_v1"
FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE = "film_unet_mask_coords_v1"
CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE = "cnn_unet_mask_coords_v1"
CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE = "crossattn_unet_mask_coords_v1"
TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE = (
    "transformer_tokens_mask_coords_v1"
)
STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE = "stylemod_unet_mask_coords_v1"
LP_CONCAT_CRITIC_CONDITIONING_MODE = "lp_concat_v1"
LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE = "lp_disabled_same_shape_v1"
LP_PROJECTION_CRITIC_CONDITIONING_MODE = "lp_projection_v1"
# Kept dependency-free to avoid a Config -> model -> film_wgan -> Config import
# cycle.  Contract tests pin this value to film_wgan.support.SUPPORT_METHOD.
GENERATOR_CURRENT_SUPPORT_METHOD = "raw_bracket_intersection_v1"
SUPPORTED_GENERATOR_NOISE_MODES = frozenset(
    {
        GAUSSIAN_GENERATOR_NOISE_MODE,
        ZERO_GENERATOR_NOISE_MODE,
    }
)
SUPPORTED_GENERATOR_CURRENT_INPUT_MODES = frozenset(
    {
        FULL_CURRENT_GENERATOR_INPUT_MODE,
        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    }
)
SUPPORTED_GENERATOR_CONDITIONING_MODES = frozenset(
    {
        BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        FILM_UNET_GENERATOR_CONDITIONING_MODE,
        FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    }
)
SUPPORTED_CRITIC_CONDITIONING_MODES = frozenset(
    {
        LP_CONCAT_CRITIC_CONDITIONING_MODE,
        LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
        LP_PROJECTION_CRITIC_CONDITIONING_MODE,
    }
)
SUPPORTED_RESIDUAL_OUTPUT_MODES = frozenset(
    {
        LEGACY_RESIDUAL_OUTPUT_MODE,
        IDENTITY_RESIDUAL_OUTPUT_MODE,
    }
)


def normalize_generator_noise_mode(value: str) -> str:
    """Return a validated generator latent-noise treatment."""

    mode = str(value).strip().lower()
    if mode not in SUPPORTED_GENERATOR_NOISE_MODES:
        supported = ", ".join(sorted(SUPPORTED_GENERATOR_NOISE_MODES))
        raise ValueError(
            f"Unsupported generator_noise_mode={value!r}; expected one of: {supported}"
        )
    return mode


def generator_noise_contract(mode: str, noise_dim: int) -> dict[str, object]:
    """Describe the latent-input semantics without changing model shape."""

    normalized = normalize_generator_noise_mode(mode)
    resolved_dim = int(noise_dim)
    if resolved_dim < 0:
        raise ValueError(f"noise_dim must be non-negative, got {resolved_dim}")
    return {
        "schema_version": 1,
        "generator_noise_mode": normalized,
        "noise_dim": resolved_dim,
        "input_distribution": (
            "standard_normal"
            if normalized == GAUSSIAN_GENERATOR_NOISE_MODE
            else "deterministic_zero"
        ),
        "training_rng_policy": (
            "standard_normal"
            if normalized == GAUSSIAN_GENERATOR_NOISE_MODE
            else "burn_standard_normal_then_zero"
        ),
        "evaluation_rng_policy": (
            "stable_standard_normal_mc"
            if normalized == GAUSSIAN_GENERATOR_NOISE_MODE
            else "literal_zero_single_pass"
        ),
        # The zero treatment deliberately retains the same fusion width so the
        # ablation does not confound stochasticity with parameter count.
        "architecture_noise_width_preserved": True,
    }


def generator_noise_fingerprint(mode: str, noise_dim: int) -> str:
    """Return a stable fingerprint for persisted latent-noise semantics."""

    encoded = json.dumps(
        generator_noise_contract(mode, noise_dim),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generator_noise_tensor(
    reference: torch.Tensor,
    *,
    batch_size: int,
    noise_dim: int,
    mode: str,
    preserve_gaussian_rng_progression: bool = False,
) -> torch.Tensor:
    """Create one explicit latent batch using the requested treatment."""

    normalized = normalize_generator_noise_mode(mode)
    shape = (int(batch_size), int(noise_dim))
    if normalized == ZERO_GENERATOR_NOISE_MODE:
        if preserve_gaussian_rng_progression:
            # Fair paired-seed ablations must leave subsequent stochastic
            # layers (notably text Dropout) on the same global RNG stream as
            # the Gaussian treatment.
            torch.randn(shape, device=reference.device, dtype=reference.dtype)
        return reference.new_zeros(shape)
    return torch.randn(shape, device=reference.device, dtype=reference.dtype)


def normalize_generator_current_input_mode(value: str) -> str:
    """Return a validated current-surface encoder-input treatment."""

    mode = str(value).strip().lower()
    if mode not in SUPPORTED_GENERATOR_CURRENT_INPUT_MODES:
        supported = ", ".join(sorted(SUPPORTED_GENERATOR_CURRENT_INPUT_MODES))
        raise ValueError(
            "Unsupported generator_current_input_mode="
            f"{value!r}; expected one of: {supported}"
        )
    return mode


def generator_current_input_contract(mode: str) -> dict[str, object]:
    """Describe which current-surface cells may enter the Generator encoder.

    The masked treatment deliberately changes only the encoder input.  The
    residual output remains anchored to the original, unmasked current surface
    so a zero residual is still the persistence forecast over the entire grid.
    """

    normalized = normalize_generator_current_input_mode(mode)
    masked = normalized == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
    return {
        "schema_version": 1,
        "generator_current_input_mode": normalized,
        "encoder_input": (
            "current_surface * current_support_mask" if masked else "current_surface"
        ),
        "mask_source": "current_raw_support_only" if masked else "none",
        "support_method": GENERATOR_CURRENT_SUPPORT_METHOD if masked else "none",
        "unsupported_cell_fill_value": 0.0 if masked else None,
        "joint_or_target_support_allowed": False if masked else None,
        "residual_anchor": "original_unmasked_current_surface",
        "architecture_shape_preserved": True,
    }


def generator_current_input_fingerprint(mode: str) -> str:
    """Return a stable fingerprint for persisted current-input semantics."""

    encoded = json.dumps(
        generator_current_input_contract(mode),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_generator_conditioning_mode(value: object) -> str:
    """Return the versioned Generator text-conditioning contract slug."""

    mode = str(value or BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE).strip().lower()
    if mode not in SUPPORTED_GENERATOR_CONDITIONING_MODES:
        raise ValueError(
            "generator_conditioning_mode must be one of "
            f"{sorted(SUPPORTED_GENERATOR_CONDITIONING_MODES)}, got {mode!r}"
        )
    return mode


def generator_conditioning_contract(mode: object) -> dict[str, object]:
    """Describe how encoded LP text enters the Generator."""

    normalized = normalize_generator_conditioning_mode(mode)
    if normalized == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": [
                "current_iv_times_support_mask",
                "support_mask",
                "normalized_moneyness",
                "normalized_log_ttm",
            ],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "fully_convolutional_encoder_decoder_with_skips",
            "noise_injection": "broadcast_concat_at_deepest_bottleneck",
            "text_embedding_treatment": "encoded_lp_to_four_virtual_tokens",
            "conditioning_mechanism": "gated_spatial_query_cross_attention",
            "conditioning_injection_points": [
                "bottleneck_4x4",
                "decoder_8x8",
                "decoder_16x16",
            ],
            "conditioning_initialization": "zero_residual_gates",
            "flatten_fusion_mlp": False,
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_1x1_convolution",
        }
    if normalized == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": [
                "current_iv_times_support_mask",
                "support_mask",
                "normalized_moneyness",
                "normalized_log_ttm",
            ],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "transformer_encoder_over_256_surface_tokens",
            "noise_injection": "one_dedicated_noise_token",
            "text_embedding_treatment": "one_encoded_lp_text_token",
            "conditioning_mechanism": "joint_surface_text_noise_self_attention",
            "conditioning_injection_points": ["transformer_token_sequence"],
            "conditioning_initialization": "zero_residual_head",
            "flatten_fusion_mlp": False,
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_tokenwise_linear_projection",
        }
    if normalized == STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": [
                "current_iv_times_support_mask",
                "support_mask",
                "normalized_moneyness",
                "normalized_log_ttm",
            ],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "fully_convolutional_encoder_decoder_with_skips",
            "noise_injection": "broadcast_concat_at_deepest_bottleneck",
            "text_embedding_treatment": "encoded_lp_style_vector",
            "conditioning_mechanism": "demodulated_modulated_convolution",
            "conditioning_injection_points": [
                "encoder_conv1",
                "encoder_conv2",
                "encoder_conv3",
                "bottleneck_conv",
                "decoder_conv1",
                "decoder_conv2",
            ],
            "conditioning_initialization": "zero_style_residual_projections",
            "flatten_fusion_mlp": False,
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_1x1_convolution",
        }
    if normalized == CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": [
                "current_iv_times_support_mask",
                "support_mask",
                "normalized_moneyness",
                "normalized_log_ttm",
            ],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "fully_convolutional_encoder_decoder_with_skips",
            "noise_injection": "broadcast_concat_at_deepest_bottleneck",
            "text_embedding_treatment": "ignored",
            "text_encoder_registered": False,
            "film_formula": "none",
            "film_injection_points": [],
            "film_projection_initialization": "none",
            "film_conditioning_vector": "none",
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_1x1_convolution",
            "shared_global_rng_advanced_by_film_construction": False,
            "construction_rng_alignment": (
                "consume_and_discard_text_encoder_initialization_to_match_film_unet"
            ),
        }
    if normalized == FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "film_formula": "(1 + gamma) * features + beta",
            "film_injection_points": [
                "conv1_pre_activation",
                "conv2_pre_activation",
                "conv3_pre_activation",
            ],
            "film_projection_initialization": "all_zero",
            "film_conditioning_vector": "encoded_lp_text",
            "film_res_blocks_supported": False,
            "shared_global_rng_advanced_by_film_construction": False,
        }
    if normalized == FILM_UNET_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": ["current_iv_times_support_mask"],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "fully_convolutional_encoder_decoder_with_skips",
            "noise_injection": "broadcast_concat_at_deepest_bottleneck",
            "film_formula": "(1 + gamma) * features + beta",
            "film_injection_points": [
                "encoder_conv1_pre_activation",
                "encoder_conv2_pre_activation",
                "encoder_conv3_pre_activation",
                "bottleneck_conv_pre_activation",
                "decoder_conv1_pre_activation",
                "decoder_conv2_pre_activation",
            ],
            "film_projection_initialization": "all_zero",
            "film_conditioning_vector": "encoded_lp_text",
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_1x1_convolution",
            "shared_global_rng_advanced_by_film_construction": False,
        }
    if normalized == FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "generator_conditioning_mode": normalized,
            "bottleneck_text_concat": False,
            "surface_input_channels": [
                "current_iv_times_support_mask",
                "support_mask",
                "normalized_moneyness",
                "normalized_log_ttm",
            ],
            "support_mask_missing_treatment": "fail_closed",
            "surface_path": "fully_convolutional_encoder_decoder_with_skips",
            "noise_injection": "broadcast_concat_at_deepest_bottleneck",
            "film_formula": "(1 + gamma) * features + beta",
            "film_injection_points": [
                "encoder_conv1_pre_activation",
                "encoder_conv2_pre_activation",
                "encoder_conv3_pre_activation",
                "bottleneck_conv_pre_activation",
                "decoder_conv1_pre_activation",
                "decoder_conv2_pre_activation",
            ],
            "film_projection_initialization": "all_zero",
            "film_conditioning_vector": "encoded_lp_text",
            "film_res_blocks_supported": False,
            "residual_head": "zero_initialized_1x1_convolution",
            "shared_global_rng_advanced_by_film_construction": False,
        }
    film = normalized == FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
    return {
        "schema_version": 1,
        "generator_conditioning_mode": normalized,
        "bottleneck_text_concat": True,
        "film_formula": "(1 + gamma) * features + beta" if film else "none",
        "film_injection_points": (
            ["conv1_pre_activation", "conv2_pre_activation", "conv3_pre_activation"]
            if film
            else []
        ),
        "film_projection_initialization": "all_zero" if film else "none",
        "film_conditioning_vector": "encoded_lp_text" if film else "none",
        "film_res_blocks_supported": False if film else True,
        "shared_global_rng_advanced_by_film_construction": False,
    }


def generator_conditioning_fingerprint(mode: object) -> str:
    """Return a stable hash for the Generator conditioning semantics."""

    return hashlib.sha256(
        json.dumps(
            generator_conditioning_contract(mode),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def normalize_critic_conditioning_mode(value: object) -> str:
    """Return the versioned Critic LP-conditioning contract slug."""

    mode = str(value or LP_CONCAT_CRITIC_CONDITIONING_MODE).strip().lower()
    if mode not in SUPPORTED_CRITIC_CONDITIONING_MODES:
        raise ValueError(
            "critic_conditioning_mode must be one of "
            f"{sorted(SUPPORTED_CRITIC_CONDITIONING_MODES)}, got {mode!r}"
        )
    return mode


def critic_conditioning_contract(mode: object) -> dict[str, object]:
    """Describe whether the shape-preserving Critic LP branch carries information."""

    normalized = normalize_critic_conditioning_mode(mode)
    if normalized == LP_PROJECTION_CRITIC_CONDITIONING_MODE:
        return {
            "schema_version": 1,
            "critic_conditioning_mode": normalized,
            "text_encoder_architecture_preserved": True,
            "classifier_input_shape_preserved": False,
            "parameter_count_preserved": True,
            "encoded_lp_treatment": "learned_projection",
            "lp_parameter_gradient": "enabled",
            "score_formula": (
                "unconditional(surface_hidden) + "
                "dot(surface_hidden, projected_encoded_lp) / sqrt(hidden_dim)"
            ),
        }
    disabled = normalized == LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE
    return {
        "schema_version": 1,
        "critic_conditioning_mode": normalized,
        "text_encoder_architecture_preserved": True,
        "classifier_input_shape_preserved": True,
        "parameter_count_preserved": True,
        "encoded_lp_treatment": "zeros_like" if disabled else "identity",
        "lp_parameter_gradient": "zero" if disabled else "enabled",
        "score_formula": "mlp([surface_features, encoded_lp])",
    }


def critic_conditioning_fingerprint(mode: object) -> str:
    """Return a stable hash for the Critic conditioning semantics."""

    return hashlib.sha256(
        json.dumps(
            critic_conditioning_contract(mode),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def generator_current_encoder_input(
    current_surface: torch.Tensor,
    current_support_mask: torch.Tensor | None,
    *,
    mode: str,
) -> torch.Tensor:
    """Build the current-surface tensor consumed by the Generator encoder.

    ``current_support_masked`` fails closed when its independently sourced
    current-side mask is missing or malformed.  In particular, callers must
    not substitute the joint training-loss mask, because that mask contains
    information about future target support.
    """

    normalized = normalize_generator_current_input_mode(mode)
    if normalized == FULL_CURRENT_GENERATOR_INPUT_MODE:
        return current_surface
    if current_support_mask is None:
        raise ValueError(
            "generator_current_input_mode='current_support_masked' requires "
            "an explicit current_support_mask; joint/target support masks must "
            "not be substituted"
        )

    mask = current_support_mask.to(
        device=current_surface.device,
        dtype=current_surface.dtype,
        non_blocking=True,
    )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    expected = (
        int(current_surface.shape[0]),
        1,
        int(current_surface.shape[-2]),
        int(current_surface.shape[-1]),
    )
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"current_support_mask shape must be {expected}, got {tuple(mask.shape)}"
        )
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("current_support_mask values must be finite")
    if bool(((mask != 0.0) & (mask != 1.0)).any()):
        raise ValueError("current_support_mask values must be binary (0 or 1)")
    if bool((mask.reshape(mask.shape[0], -1).sum(dim=1) <= 0).any()):
        raise ValueError(
            "Every current_support_mask row must contain at least one supported cell"
        )
    return current_surface * mask


def normalize_residual_output_mode(value: str) -> str:
    """Return a validated surface-output parameterization name."""

    mode = str(value).strip().lower()
    if mode not in SUPPORTED_RESIDUAL_OUTPUT_MODES:
        supported = ", ".join(sorted(SUPPORTED_RESIDUAL_OUTPUT_MODES))
        raise ValueError(
            f"Unsupported residual_output_mode={value!r}; expected one of: {supported}"
        )
    return mode


def residual_output_contract(mode: str) -> dict[str, object]:
    """Describe the model-output transform used by a persisted checkpoint."""

    normalized = normalize_residual_output_mode(mode)
    formula = {
        LEGACY_RESIDUAL_OUTPUT_MODE: "softplus(current + delta) + epsilon",
        IDENTITY_RESIDUAL_OUTPUT_MODE: (
            "current + softplus(inv_softplus(current - epsilon) + delta) "
            "- softplus(inv_softplus(current - epsilon))"
        ),
    }[normalized]
    return {
        "schema_version": 1,
        "residual_output_mode": normalized,
        "epsilon": RESIDUAL_OUTPUT_EPSILON,
        "formula": formula,
        "identity_when_delta_zero": normalized == IDENTITY_RESIDUAL_OUTPUT_MODE,
    }


def residual_output_fingerprint(mode: str) -> str:
    """Return a stable fingerprint for the output transform contract."""

    encoded = json.dumps(
        residual_output_contract(mode),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_residual_surface_output(
    current_surface: torch.Tensor,
    delta: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Map a raw residual to a positive future surface.

    ``identity_softplus_residual`` works in the unconstrained coordinate whose
    positive-domain map is softplus.  Subtracting the same mapped anchor makes
    a zero residual an exact identity operation for valid volatility inputs
    (``current_surface > RESIDUAL_OUTPUT_EPSILON``), while the limiting output
    remains positive as the residual tends to negative infinity.

    ``legacy_softplus`` intentionally preserves the historical implementation
    for old configs and checkpoints, including its non-identity zero residual.
    """

    normalized = normalize_residual_output_mode(mode)
    if normalized == LEGACY_RESIDUAL_OUTPUT_MODE:
        return F.softplus(current_surface + delta) + RESIDUAL_OUTPUT_EPSILON

    positive_offset = current_surface - RESIDUAL_OUTPUT_EPSILON
    tiny = torch.finfo(current_surface.dtype).tiny
    safe_offset = positive_offset.clamp_min(tiny)
    # Stable inverse softplus: x + log(1 - exp(-x)), written with expm1
    # for accuracy when x is close to zero.
    anchor_coordinate = safe_offset + torch.log(-torch.expm1(-safe_offset))
    anchor_surface = F.softplus(anchor_coordinate)
    shifted_surface = F.softplus(anchor_coordinate + delta)
    return current_surface + (shifted_surface - anchor_surface)


def zero_initialize_residual_head(head: nn.Linear | nn.Conv2d) -> None:
    """Start an identity residual model at the persistence forecast."""

    nn.init.zeros_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def conv2d_out_size(
    size: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    dilation: int = 1,
) -> int:
    """Return the spatial dimension after a Conv2d with the given parameters."""
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def group_count(channels: int) -> int:
    """Pick the largest valid GroupNorm group count for *channels*."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small residual block used to widen the surface encoder without changing resolution."""

    def __init__(self, channels: int):
        super().__init__()
        groups = group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)


LEGACY_CRITIC_NORMALIZATION_MODE = "legacy_instance_norm_v1"
INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE = "instance_norm_group_tail_v1"
CRITIC_NORMALIZATION_MODES = frozenset(
    {
        LEGACY_CRITIC_NORMALIZATION_MODE,
        INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE,
    }
)


def normalize_critic_normalization_mode(value: object) -> str:
    """Return the stable persisted critic-normalization contract slug."""

    mode = str(value or LEGACY_CRITIC_NORMALIZATION_MODE).strip().lower()
    if mode not in CRITIC_NORMALIZATION_MODES:
        raise ValueError(
            "critic_normalization_mode must be one of "
            f"{sorted(CRITIC_NORMALIZATION_MODES)}, got {mode!r}"
        )
    return mode


def critic_normalization_fingerprint(mode: object) -> str:
    normalized = normalize_critic_normalization_mode(mode)
    payload = {
        "schema_version": 1,
        "critic_normalization_mode": normalized,
        "second_downsample_norm": "instance_norm_2d_affine",
        "third_downsample_norm": (
            "group_norm_1_group_affine"
            if normalized == INSTANCE_GROUP_TAIL_CRITIC_NORMALIZATION_MODE
            else "instance_norm_2d_affine"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
