"""Pair-weighted batch reductions and stable validation-noise helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class VolTrainingBatch:
    """Normalized view of legacy and weighted merged-vol batches."""

    current_surface: torch.Tensor
    text_embedding: torch.Tensor
    target_surface: torch.Tensor
    sample_weight: torch.Tensor
    stable_noise_key: torch.Tensor | None
    # Future-aware current∩target mask for losses/evaluation only.
    support_mask: torch.Tensor | None
    # Current-only mask that is safe for model conditioning.
    current_support_mask: torch.Tensor | None
    # Pair-level train-only label reliability multiplier.  It is deliberately
    # last so historical three- through seven-item tuple meanings never move.
    label_reliability_weight: torch.Tensor | None


def unpack_vol_training_batch(batch: Sequence[Any]) -> VolTrainingBatch:
    """Accept legacy batches plus weighted batches with an optional cell mask.

    The six-item shape appends the joint ``support_mask`` after the stable
    sample key.  The seven-item shape then appends ``current_support_mask``.
    The eighth item is the scalar-per-row ``label_reliability_weight``.
    Historical three- through seven-item tuple meanings remain unchanged.
    """

    if len(batch) not in {3, 4, 5, 6, 7, 8}:
        raise ValueError(
            "Expected a 3, 4, 5, 6, 7 or 8 item vol batch, "
            f"got {len(batch)} item(s)."
        )
    current_surface, text_embedding, target_surface = batch[:3]
    if len(batch) >= 4:
        sample_weight = batch[3]
    else:
        sample_weight = torch.ones(
            int(current_surface.shape[0]),
            dtype=current_surface.dtype,
            device=current_surface.device,
        )
    stable_noise_key = batch[4] if len(batch) >= 5 else None
    support_mask = batch[5] if len(batch) >= 6 else None
    current_support_mask = batch[6] if len(batch) >= 7 else None
    label_reliability_weight = batch[7] if len(batch) == 8 else None
    return VolTrainingBatch(
        current_surface=current_surface,
        text_embedding=text_embedding,
        target_surface=target_surface,
        sample_weight=sample_weight.reshape(-1),
        stable_noise_key=stable_noise_key,
        support_mask=support_mask,
        current_support_mask=current_support_mask,
        label_reliability_weight=(
            None
            if label_reliability_weight is None
            else label_reliability_weight.reshape(-1)
        ),
    )


def validated_surface_mask(
    support_mask: torch.Tensor | None,
    *,
    reference_surface: torch.Tensor,
) -> torch.Tensor | None:
    """Validate a per-cell binary mask and normalize it to ``[B, 1, H, W]``."""

    if support_mask is None:
        return None
    mask = support_mask.to(
        device=reference_surface.device,
        dtype=reference_surface.dtype,
        non_blocking=True,
    )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    expected = (
        int(reference_surface.shape[0]),
        1,
        int(reference_surface.shape[-2]),
        int(reference_surface.shape[-1]),
    )
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"support_mask shape must be {expected}, got {tuple(mask.shape)}"
        )
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("support_mask values must be finite.")
    if bool(((mask != 0.0) & (mask != 1.0)).any()):
        raise ValueError("support_mask values must be binary (0 or 1).")
    if bool((mask.reshape(mask.shape[0], -1).sum(dim=1) <= 0).any()):
        raise ValueError(
            "Every support_mask row must contain at least one supported cell."
        )
    return mask


def apply_surface_mask(
    surface: torch.Tensor,
    support_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Zero cells outside raw joint support without changing legacy behavior."""

    if support_mask is None:
        return surface
    return surface * support_mask


def masked_mean_per_sample(
    values: torch.Tensor,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average each row over supported elements, returning zero for empty edge masks."""

    batch_size = int(values.shape[0])
    if support_mask is None:
        return values.reshape(batch_size, -1).mean(dim=1)
    mask = support_mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    try:
        mask = torch.broadcast_to(mask, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"support mask shape {tuple(support_mask.shape)} cannot cover values "
            f"with shape {tuple(values.shape)}"
        ) from exc
    numerator = (values * mask).reshape(batch_size, -1).sum(dim=1)
    denominator = mask.reshape(batch_size, -1).sum(dim=1)
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def validated_sample_weights(
    sample_weight: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return positive finite weights without normalizing away their mass."""

    if sample_weight is None:
        return torch.ones(batch_size, dtype=dtype, device=device)
    weights = sample_weight.to(device=device, dtype=dtype, non_blocking=True).reshape(
        -1
    )
    if int(weights.numel()) != int(batch_size):
        raise ValueError(
            f"sample_weight length must match batch size: {weights.numel()} != {batch_size}"
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("sample_weight values must all be positive and finite.")
    return weights


def validated_label_reliability_weights(
    label_reliability_weight: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    require_ones: bool = False,
) -> torch.Tensor:
    """Validate the bounded train-only multiplier for reconstruction loss."""

    if label_reliability_weight is None:
        weights = torch.ones(batch_size, dtype=dtype, device=device)
    else:
        weights = label_reliability_weight.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ).reshape(-1)
    if int(weights.numel()) != int(batch_size):
        raise ValueError(
            "label_reliability_weight length must match batch size: "
            f"{weights.numel()} != {batch_size}"
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("label_reliability_weight values must all be finite.")
    if bool(((weights < 0.5) | (weights > 2.0)).any()):
        raise ValueError(
            "label_reliability_weight values must be within the closed interval "
            "[0.5, 2.0]."
        )
    if require_ones and bool((weights != 1.0).any()):
        raise ValueError(
            "Validation/evaluation label_reliability_weight values must be exactly 1."
        )
    return weights


def reconstruction_training_weights(
    sample_weight: torch.Tensor | None,
    label_reliability_weight: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    require_label_ones: bool = False,
) -> torch.Tensor:
    """Return ``pair_weight * label_weight`` for reconstruction and nothing else."""

    pair_weights = validated_sample_weights(
        sample_weight,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    label_weights = validated_label_reliability_weights(
        label_reliability_weight,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        require_ones=require_label_ones,
    )
    return pair_weights * label_weights


def weighted_mean(
    per_sample: torch.Tensor, sample_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Return an exact weight-normalized mean for evaluation."""

    if per_sample.ndim == 0:
        return per_sample
    batch_size = int(per_sample.shape[0])
    if batch_size == 0:
        return per_sample.new_zeros(())
    values = per_sample.reshape(batch_size, -1).mean(dim=1)
    weights = validated_sample_weights(
        sample_weight,
        batch_size=batch_size,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.sum(values * weights) / torch.sum(weights)


def training_weighted_mean(
    per_sample: torch.Tensor,
    scaled_sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``mean(scaled_weight * loss)`` for unbiased mini-batch training.

    News-first loaders scale pair-balanced raw weights by ``n_rows / n_pairs``.
    Their population mean is therefore one.  Keeping the denominator equal to
    the batch size avoids making a sample's gradient depend on which other
    pairs happened to land in the same mini-batch.
    """

    if per_sample.ndim == 0:
        return per_sample
    batch_size = int(per_sample.shape[0])
    if batch_size == 0:
        return per_sample.new_zeros(())
    values = per_sample.reshape(batch_size, -1).mean(dim=1)
    weights = validated_sample_weights(
        scaled_sample_weight,
        batch_size=batch_size,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.mean(values * weights)


def stable_key_to_int64(value: str) -> int:
    """Map a persistent sample identifier to a process-independent int64 key."""

    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def stable_tensor_row_keys(*tensors: torch.Tensor) -> torch.Tensor:
    """Hash tensor rows into stable keys for legacy batches without IDs.

    This fallback is intentionally based on sample content, not row position or
    ``global_index``, so validation noise remains unchanged when the same rows
    are rebatched or reordered.
    """

    if not tensors:
        return torch.empty(0, dtype=torch.int64)
    batch_size = int(tensors[0].shape[0])
    if any(int(tensor.shape[0]) != batch_size for tensor in tensors):
        raise ValueError("All tensors must have the same leading batch dimension.")
    cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors]
    keys: list[int] = []
    for row_index in range(batch_size):
        digest = hashlib.sha256()
        for tensor in cpu_tensors:
            row = tensor[row_index]
            digest.update(str(row.dtype).encode("ascii"))
            digest.update(str(tuple(row.shape)).encode("ascii"))
            digest.update(row.numpy().tobytes())
        keys.append(
            int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
            & ((1 << 63) - 1)
        )
    return torch.tensor(keys, dtype=torch.int64)


def _noise_seed(base_seed: int, stable_key: int, draw_index: int) -> int:
    payload = f"{int(base_seed)}|{int(stable_key)}|{int(draw_index)}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def stable_noise_for_keys(
    stable_keys: torch.Tensor | Sequence[int],
    *,
    noise_dim: int,
    base_seed: int,
    draw_index: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create noise determined only by seed, persistent sample key and draw."""

    if isinstance(stable_keys, torch.Tensor):
        keys = [int(value) for value in stable_keys.detach().cpu().reshape(-1).tolist()]
    else:
        keys = [int(value) for value in stable_keys]
    rows = []
    for stable_key in keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_noise_seed(base_seed, stable_key, draw_index))
        rows.append(torch.randn(int(noise_dim), generator=generator, dtype=dtype))
    if not rows:
        return torch.empty((0, int(noise_dim)), dtype=dtype, device=device)
    return torch.stack(rows, dim=0).to(device=device, non_blocking=True)
