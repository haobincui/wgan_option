"""Hash-bound Pure-CNN to FiLM U-Net backbone graft artifacts.

This module is an opt-in, branch-local transfer contract.  It deliberately does
not extend ``news_first_wgan_full_training_state_v1``: a Pure-CNN optimizer
cannot be resumed strictly into a FiLM model because the latter registers text
and modulation parameters.  Instead, the shared convolutional Generator state
and the complete NoLP Critic state are copied, while every optimizer and
scheduler starts fresh and is persisted together with the transfer lineage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from wgan_option.models.common import (
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE,
)
from wgan_option.utils.news_first_experiment_core import (
    capture_rng_state,
    restore_rng_state,
)


GRAFT_STATE_SCHEMA_VERSION = 1
GRAFT_STATE_KIND = "pure_cnn_to_film_graft_state_v1"
GRAFT_MANIFEST_KIND = "pure_cnn_to_film_graft_manifest_v1"
GRAFT_SAVE_PHASE = "after_weight_graft_before_first_optimizer_step"
SUPPORTED_TARGET_GENERATOR_MODES = frozenset(
    {
        CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    }
)

SHARED_GENERATOR_PREFIXES = (
    "surface_encoder.",
    "bottleneck_conv.",
    "decoder_convs.",
    "residual_head.",
)
NEW_FILM_GENERATOR_PREFIXES = (
    "text_encoder.",
    "encoder_film_layers.",
    "bottleneck_film_layer.",
    "decoder_film_layers.",
)
FILM_PROJECTION_PREFIXES = (
    "encoder_film_layers.",
    "bottleneck_film_layer.",
    "decoder_film_layers.",
)
VERIFICATION_TEXT_KEYS = ("zero", "matched", "shuffle")


class PureCnnFilmGraftError(ValueError):
    """Raised when a Pure-CNN/FiLM graft violates its frozen contract."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PureCnnFilmGraftError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _normalized_lineage(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise PureCnnFilmGraftError("graft lineage must be a non-empty mapping")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise PureCnnFilmGraftError("graft lineage keys must be non-empty strings")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PureCnnFilmGraftError("graft lineage must be finite JSON") from error
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise PureCnnFilmGraftError("normalized graft lineage must be an object")
    return normalized


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in module.state_dict().items()
    }


def _keys_with_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(sorted(name for name in state if name.startswith(prefixes)))


def _validate_module_modes(
    pure_cnn_generator: torch.nn.Module,
    target_generator: torch.nn.Module,
    pure_cnn_critic: torch.nn.Module,
    target_critic: torch.nn.Module,
) -> str:
    if (
        getattr(pure_cnn_generator, "generator_conditioning_mode", None)
        != CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
    ):
        raise PureCnnFilmGraftError(
            "source Generator must be a Pure-CNN mask/coords U-Net"
        )
    target_mode = str(getattr(target_generator, "generator_conditioning_mode", ""))
    if target_mode not in SUPPORTED_TARGET_GENERATOR_MODES:
        raise PureCnnFilmGraftError(
            "destination Generator must be a Pure-CNN or FiLM mask/coords U-Net"
        )
    for label, critic in (
        ("source", pure_cnn_critic),
        ("destination", target_critic),
    ):
        if (
            getattr(critic, "critic_conditioning_mode", None)
            != LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE
        ):
            raise PureCnnFilmGraftError(f"{label} Critic must use the NoLP contract")
    return target_mode


def _validate_coordinate_buffers(
    pure_cnn_generator: torch.nn.Module,
    film_generator: torch.nn.Module,
) -> None:
    for name in ("moneyness_coordinate", "log_ttm_coordinate"):
        source = getattr(pure_cnn_generator, name, None)
        destination = getattr(film_generator, name, None)
        if not isinstance(source, torch.Tensor) or not isinstance(
            destination, torch.Tensor
        ):
            raise PureCnnFilmGraftError(
                f"missing non-persistent coordinate buffer: {name}"
            )
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise PureCnnFilmGraftError(
                f"coordinate buffer shape/dtype mismatch: {name}"
            )
        if not torch.equal(source.detach().cpu(), destination.detach().cpu()):
            raise PureCnnFilmGraftError(f"coordinate grid mismatch: {name}")


def _validate_tensor_compatibility(
    source: Mapping[str, torch.Tensor],
    destination: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    *,
    label: str,
) -> None:
    for name in keys:
        if name not in source or name not in destination:
            raise PureCnnFilmGraftError(f"{label} key mismatch: {name}")
        left = source[name]
        right = destination[name]
        if left.shape != right.shape:
            raise PureCnnFilmGraftError(
                f"{label} shape mismatch for {name}: {tuple(left.shape)} != {tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise PureCnnFilmGraftError(
                f"{label} dtype mismatch for {name}: {left.dtype} != {right.dtype}"
            )


def _assert_film_projections_zero(film_generator: torch.nn.Module) -> tuple[str, ...]:
    state = film_generator.state_dict()
    keys = _keys_with_prefixes(state, FILM_PROJECTION_PREFIXES)
    if not keys:
        raise PureCnnFilmGraftError("FiLM Generator has no modulation projection state")
    for name in keys:
        value = state[name]
        if not torch.isfinite(value).all() or int(torch.count_nonzero(value)) != 0:
            raise PureCnnFilmGraftError(f"FiLM projection must be exactly zero: {name}")
    return keys


def graft_pure_cnn_to_film(
    *,
    pure_cnn_generator: torch.nn.Module,
    film_generator: torch.nn.Module,
    pure_cnn_critic: torch.nn.Module,
    film_critic: torch.nn.Module,
) -> dict[str, Any]:
    """Copy one Pure-CNN backbone and complete NoLP Critic into FiLM modules."""

    result = graft_pure_cnn_fresh_restart(
        pure_cnn_generator=pure_cnn_generator,
        target_generator=film_generator,
        pure_cnn_critic=pure_cnn_critic,
        target_critic=film_critic,
    )
    if (
        getattr(film_generator, "generator_conditioning_mode", None)
        != FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
    ):
        raise PureCnnFilmGraftError("FiLM graft requires a FiLM destination Generator")
    return result


def graft_pure_cnn_fresh_restart(
    *,
    pure_cnn_generator: torch.nn.Module,
    target_generator: torch.nn.Module,
    pure_cnn_critic: torch.nn.Module,
    target_critic: torch.nn.Module,
) -> dict[str, Any]:
    """Create a FiLM graft or Pure-CNN identity fresh-restart model state."""

    target_mode = _validate_module_modes(
        pure_cnn_generator,
        target_generator,
        pure_cnn_critic,
        target_critic,
    )
    _validate_coordinate_buffers(pure_cnn_generator, target_generator)
    source_generator = pure_cnn_generator.state_dict()
    destination_generator = target_generator.state_dict()
    copied_keys = _keys_with_prefixes(source_generator, SHARED_GENERATOR_PREFIXES)
    if not copied_keys or set(source_generator) != set(copied_keys):
        unexpected = sorted(set(source_generator) - set(copied_keys))
        raise PureCnnFilmGraftError(
            f"Pure-CNN state contains non-backbone keys: {unexpected}"
        )
    destination_shared = _keys_with_prefixes(
        destination_generator, SHARED_GENERATOR_PREFIXES
    )
    if copied_keys != destination_shared:
        raise PureCnnFilmGraftError("source/destination shared Generator keys differ")
    new_keys = tuple(sorted(set(destination_generator) - set(copied_keys)))
    if target_mode == FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        if not new_keys or any(
            not name.startswith(NEW_FILM_GENERATOR_PREFIXES) for name in new_keys
        ):
            raise PureCnnFilmGraftError(
                "destination FiLM Generator contains unexpected new keys"
            )
    elif new_keys:
        raise PureCnnFilmGraftError(
            "Pure-CNN identity restart must not introduce Generator keys"
        )
    if target_mode == FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        _assert_film_projections_zero(target_generator)
    _validate_tensor_compatibility(
        source_generator,
        destination_generator,
        copied_keys,
        label="Generator",
    )
    assembled_generator = dict(destination_generator)
    for name in copied_keys:
        assembled_generator[name] = source_generator[name].detach().clone()
    target_generator.load_state_dict(assembled_generator, strict=True)

    source_critic = pure_cnn_critic.state_dict()
    destination_critic = target_critic.state_dict()
    critic_keys = tuple(sorted(source_critic))
    if critic_keys != tuple(sorted(destination_critic)):
        raise PureCnnFilmGraftError("source/destination NoLP Critic keys differ")
    _validate_tensor_compatibility(
        source_critic,
        destination_critic,
        critic_keys,
        label="Critic",
    )
    target_critic.load_state_dict(source_critic, strict=True)

    copied_state = target_generator.state_dict()
    for name in copied_keys:
        if not torch.equal(
            copied_state[name].detach().cpu(), source_generator[name].detach().cpu()
        ):
            raise PureCnnFilmGraftError(f"Generator graft copy mismatch: {name}")
    copied_critic_state = target_critic.state_dict()
    for name in critic_keys:
        if not torch.equal(
            copied_critic_state[name].detach().cpu(), source_critic[name].detach().cpu()
        ):
            raise PureCnnFilmGraftError(f"Critic graft copy mismatch: {name}")
    projection_keys = (
        _assert_film_projections_zero(target_generator)
        if target_mode == FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        else ()
    )
    return {
        "target_generator_mode": target_mode,
        "copied_generator_keys": copied_keys,
        "new_generator_keys": new_keys,
        "film_projection_keys": projection_keys,
        "critic_keys": critic_keys,
    }


def verify_graft_text_invariance(
    *,
    pure_cnn_generator: torch.nn.Module,
    film_generator: torch.nn.Module,
    current_surface: torch.Tensor,
    current_support_mask: torch.Tensor,
    noise: torch.Tensor,
    text_embeddings: Mapping[str, torch.Tensor],
    tolerance: float = 1.0e-7,
) -> dict[str, object]:
    """Verify exact epoch-zero equivalence for zero, matched, and shuffle text."""

    threshold = float(tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise PureCnnFilmGraftError(
            "verification tolerance must be finite/non-negative"
        )
    if tuple(text_embeddings) != VERIFICATION_TEXT_KEYS:
        raise PureCnnFilmGraftError(
            f"text embeddings must be ordered as {VERIFICATION_TEXT_KEYS}"
        )
    zero = text_embeddings["zero"]
    tensors = {
        "current_surface": current_surface,
        "current_support_mask": current_support_mask,
        "noise": noise,
        **{f"text:{name}": value for name, value in text_embeddings.items()},
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise PureCnnFilmGraftError("verification inputs must all be tensors")
    if not bool(torch.isfinite(current_surface).all()) or not bool(
        torch.isfinite(noise).all()
    ):
        raise PureCnnFilmGraftError("verification surface/noise must be finite")
    if int(zero.shape[0]) != int(current_surface.shape[0]) or bool(zero.any()):
        raise PureCnnFilmGraftError("zero text must be an all-zero batch")
    for name, text in text_embeddings.items():
        if text.shape != zero.shape or not bool(torch.isfinite(text).all()):
            raise PureCnnFilmGraftError(f"invalid verification text tensor: {name}")

    source_training = pure_cnn_generator.training
    destination_training = film_generator.training
    pure_cnn_generator.eval()
    film_generator.eval()
    try:
        with torch.no_grad():
            reference = pure_cnn_generator(
                current_surface,
                zero,
                noise=noise,
                current_support_mask=current_support_mask,
            )
            candidate_outputs = {
                name: film_generator(
                    current_surface,
                    text,
                    noise=noise,
                    current_support_mask=current_support_mask,
                )
                for name, text in text_embeddings.items()
            }
    finally:
        pure_cnn_generator.train(source_training)
        film_generator.train(destination_training)

    max_abs_by_text: dict[str, float] = {}
    for name, output in candidate_outputs.items():
        if output.shape != reference.shape or not bool(torch.isfinite(output).all()):
            raise PureCnnFilmGraftError(f"invalid graft output for text={name}")
        difference = float((output - reference).abs().max().detach().cpu())
        max_abs_by_text[name] = difference
        if difference > threshold:
            raise PureCnnFilmGraftError(
                f"graft output differs for text={name}: {difference} > {threshold}"
            )
    maximum = max(max_abs_by_text.values())
    return {
        "text_cases": list(VERIFICATION_TEXT_KEYS),
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_by_text": max_abs_by_text,
        "tolerance": threshold,
        "passed": True,
    }


def _capture_graft_rng_state(
    loader_generator: torch.Generator,
    *,
    expected_cuda_device_count: int | None = None,
    cuda_source_device_index: int = 0,
) -> tuple[dict[str, object], dict[str, int]]:
    state = capture_rng_state(loader_generator)
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise PureCnnFilmGraftError("captured CUDA RNG state must be a list")
    source_count = len(cuda_states)
    expected_count = (
        source_count
        if expected_cuda_device_count is None
        else int(expected_cuda_device_count)
    )
    source_index = int(cuda_source_device_index)
    if expected_count < 0:
        raise PureCnnFilmGraftError("expected CUDA device count must be non-negative")
    if expected_count == 0:
        if source_count:
            raise PureCnnFilmGraftError(
                "cannot discard visible CUDA RNG state for a CPU-only graft topology"
            )
        selected_cuda_states: list[object] = []
    elif expected_count == 1:
        if source_index < 0 or source_index >= source_count:
            raise PureCnnFilmGraftError(
                "worker CUDA RNG source index is unavailable: "
                f"index={source_index}, visible={source_count}"
            )
        selected_cuda_states = [cuda_states[source_index]]
    elif expected_count == source_count and source_index == 0:
        selected_cuda_states = list(cuda_states)
    else:
        raise PureCnnFilmGraftError(
            "unsupported CUDA RNG topology projection: "
            f"visible={source_count}, expected={expected_count}, source={source_index}"
        )
    rng_state = {
        "python": state["python"],
        "numpy": state["numpy"],
        "torch_cpu": state["torch_cpu"],
        "torch_cuda": selected_cuda_states,
        "dataloader_generator": state["loader_generator"],
    }
    topology = {
        "source_cuda_device_count": source_count,
        "source_cuda_device_index": source_index if expected_count else -1,
        "worker_cuda_device_count": expected_count,
    }
    return rng_state, topology


def _restore_graft_rng_state(
    state: object, *, loader_generator: torch.Generator
) -> None:
    if not isinstance(state, Mapping):
        raise PureCnnFilmGraftError("graft RNG state must be a mapping")
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "dataloader_generator",
    }
    if set(state) != required:
        raise PureCnnFilmGraftError("graft RNG state keys mismatch")
    restore_rng_state(
        {
            "python": state["python"],
            "numpy": state["numpy"],
            "torch_cpu": state["torch_cpu"],
            "torch_cuda": state["torch_cuda"],
            "loader_generator": state["dataloader_generator"],
        },
        loader_generator=loader_generator,
    )


def _validated_cuda_rng_topology(payload: Mapping[str, object]) -> dict[str, int]:
    raw = payload.get("cuda_rng_topology")
    rng_state = payload.get("rng_state")
    if not isinstance(raw, Mapping) or not isinstance(rng_state, Mapping):
        raise PureCnnFilmGraftError("graft CUDA RNG topology is missing")
    if set(raw) != {
        "source_cuda_device_count",
        "source_cuda_device_index",
        "worker_cuda_device_count",
    }:
        raise PureCnnFilmGraftError("graft CUDA RNG topology keys mismatch")
    topology = {key: int(value) for key, value in raw.items()}
    source_count = topology["source_cuda_device_count"]
    source_index = topology["source_cuda_device_index"]
    worker_count = topology["worker_cuda_device_count"]
    cuda_states = rng_state.get("torch_cuda")
    if (
        source_count < 0
        or worker_count < 0
        or not isinstance(cuda_states, list)
        or len(cuda_states) != worker_count
        or (worker_count == 0 and source_index != -1)
        or (worker_count > 0 and (source_index < 0 or source_index >= source_count))
    ):
        raise PureCnnFilmGraftError("graft CUDA RNG topology is inconsistent")
    return topology


def _validate_parent_checkpoint(
    path: Path,
    expected_sha256: str,
    *,
    pure_cnn_generator: torch.nn.Module,
    pure_cnn_critic: torch.nn.Module,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Pure-CNN parent checkpoint does not exist: {path}")
    expected = _require_sha256(expected_sha256, label="parent checkpoint SHA256")
    observed = sha256_file(path)
    if observed != expected:
        raise PureCnnFilmGraftError(
            f"parent checkpoint SHA256 mismatch: expected={expected}, actual={observed}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PureCnnFilmGraftError("parent checkpoint root must be a mapping")
    required = ("generator_state_dict", "discriminator_state_dict")
    if any(not isinstance(payload.get(name), Mapping) for name in required):
        raise PureCnnFilmGraftError(
            "parent checkpoint must contain Generator and Discriminator state dicts"
        )
    expected_states = (
        ("Generator", _cpu_state_dict(pure_cnn_generator), payload[required[0]]),
        ("Critic", _cpu_state_dict(pure_cnn_critic), payload[required[1]]),
    )
    for label, model_state, checkpoint_state in expected_states:
        if set(model_state) != set(checkpoint_state):
            raise PureCnnFilmGraftError(f"parent checkpoint {label} keys differ")
        for name, tensor in model_state.items():
            candidate = checkpoint_state[name]
            if not isinstance(candidate, torch.Tensor) or not torch.equal(
                tensor, candidate.detach().cpu()
            ):
                raise PureCnnFilmGraftError(
                    f"parent checkpoint {label} state differs: {name}"
                )
    return observed


def _fresh_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    module: torch.nn.Module,
    label: str,
) -> dict[str, object]:
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    module_ids = {id(parameter) for parameter in module.parameters()}
    if (
        len(optimizer_ids) != len(set(optimizer_ids))
        or set(optimizer_ids) != module_ids
    ):
        raise PureCnnFilmGraftError(
            f"{label} optimizer does not cover the destination module exactly"
        )
    state = copy.deepcopy(optimizer.state_dict())
    if state.get("state"):
        raise PureCnnFilmGraftError(f"{label} optimizer must be fresh")
    if not state.get("param_groups"):
        raise PureCnnFilmGraftError(f"{label} optimizer has no parameter groups")
    return state


def _fresh_scheduler_state(
    scheduler: object,
    *,
    optimizer: torch.optim.Optimizer,
    label: str,
) -> dict[str, object]:
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise PureCnnFilmGraftError(
            f"{label} scheduler does not reference the declared optimizer"
        )
    state_method = getattr(scheduler, "state_dict", None)
    if not callable(state_method):
        raise PureCnnFilmGraftError(f"{label} scheduler has no state_dict")
    state = copy.deepcopy(state_method())
    if (
        int(state.get("last_epoch", -1)) != 0
        or int(state.get("num_bad_epochs", -1)) != 0
        or int(state.get("cooldown_counter", -1)) != 0
        or state.get("best") != state.get("mode_worse")
    ):
        raise PureCnnFilmGraftError(f"{label} scheduler must be fresh")
    return state


def _atomic_torch_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"graft artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> Path:
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"graft manifest already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def save_pure_cnn_to_film_graft_state(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    parent_checkpoint_path: str | Path,
    expected_parent_checkpoint_sha256: str,
    pure_cnn_generator: torch.nn.Module,
    target_generator: torch.nn.Module,
    pure_cnn_critic: torch.nn.Module,
    target_discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generator_scheduler: object,
    critic_scheduler: object,
    loader_generator: torch.Generator,
    parent_lineage: Mapping[str, object],
    graft_lineage: Mapping[str, object],
    current_surface: torch.Tensor,
    current_support_mask: torch.Tensor,
    noise: torch.Tensor,
    text_embeddings: Mapping[str, torch.Tensor],
    verification_tolerance: float = 1.0e-7,
    expected_cuda_device_count: int | None = None,
    cuda_source_device_index: int = 0,
) -> dict[str, object]:
    """Graft, validate, and persist one immutable transfer state and manifest."""

    parent_path = Path(parent_checkpoint_path).expanduser().resolve()
    parent_sha = _validate_parent_checkpoint(
        parent_path,
        expected_parent_checkpoint_sha256,
        pure_cnn_generator=pure_cnn_generator,
        pure_cnn_critic=pure_cnn_critic,
    )
    generator_optimizer_state = _fresh_optimizer_state(
        generator_optimizer, module=target_generator, label="Generator"
    )
    discriminator_optimizer_state = _fresh_optimizer_state(
        critic_optimizer, module=target_discriminator, label="Discriminator"
    )
    generator_scheduler_state = _fresh_scheduler_state(
        generator_scheduler,
        optimizer=generator_optimizer,
        label="Generator",
    )
    discriminator_scheduler_state = _fresh_scheduler_state(
        critic_scheduler,
        optimizer=critic_optimizer,
        label="Discriminator",
    )
    key_plan = graft_pure_cnn_fresh_restart(
        pure_cnn_generator=pure_cnn_generator,
        target_generator=target_generator,
        pure_cnn_critic=pure_cnn_critic,
        target_critic=target_discriminator,
    )
    verification = verify_graft_text_invariance(
        pure_cnn_generator=pure_cnn_generator,
        film_generator=target_generator,
        current_surface=current_surface,
        current_support_mask=current_support_mask,
        noise=noise,
        text_embeddings=text_embeddings,
        tolerance=verification_tolerance,
    )
    normalized_parent_lineage = _normalized_lineage(parent_lineage)
    normalized_graft_lineage = _normalized_lineage(graft_lineage)
    optimizer_reset_proof = {
        "generator_optimizer_state_empty": not bool(generator_optimizer_state["state"]),
        "discriminator_optimizer_state_empty": not bool(
            discriminator_optimizer_state["state"]
        ),
        "generator_scheduler_fresh": True,
        "discriminator_scheduler_fresh": True,
        "apply_loads_optimizer_or_scheduler_state": False,
    }
    rng_state, cuda_rng_topology = _capture_graft_rng_state(
        loader_generator,
        expected_cuda_device_count=expected_cuda_device_count,
        cuda_source_device_index=cuda_source_device_index,
    )
    artifact_payload: dict[str, object] = {
        "schema_version": GRAFT_STATE_SCHEMA_VERSION,
        "kind": GRAFT_STATE_KIND,
        "save_phase": GRAFT_SAVE_PHASE,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "target_generator_mode": str(key_plan["target_generator_mode"]),
        "copied_generator_keys": list(key_plan["copied_generator_keys"]),
        "new_generator_keys": list(key_plan["new_generator_keys"]),
        "film_projection_keys": list(key_plan["film_projection_keys"]),
        "critic_keys": list(key_plan["critic_keys"]),
        "generator_state_dict": _cpu_state_dict(target_generator),
        "discriminator_state_dict": _cpu_state_dict(target_discriminator),
        "generator_optimizer_state_dict": generator_optimizer_state,
        "discriminator_optimizer_state_dict": discriminator_optimizer_state,
        "generator_scheduler_state_dict": generator_scheduler_state,
        "discriminator_scheduler_state_dict": discriminator_scheduler_state,
        "optimizer_reset_proof": optimizer_reset_proof,
        "rng_state": rng_state,
        "cuda_rng_topology": cuda_rng_topology,
        "parent_lineage": normalized_parent_lineage,
        "graft_lineage": normalized_graft_lineage,
        "epoch0_equivalence": verification,
    }
    artifact = _atomic_torch_write(
        Path(artifact_path).expanduser().resolve(), artifact_payload
    )
    artifact_sha = sha256_file(artifact)
    manifest_payload: dict[str, object] = {
        "schema_version": GRAFT_STATE_SCHEMA_VERSION,
        "kind": GRAFT_MANIFEST_KIND,
        "artifact_path": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": artifact_sha,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "target_generator_mode": str(key_plan["target_generator_mode"]),
        "copied_generator_keys": list(key_plan["copied_generator_keys"]),
        "new_generator_keys": list(key_plan["new_generator_keys"]),
        "film_projection_keys": list(key_plan["film_projection_keys"]),
        "critic_keys": list(key_plan["critic_keys"]),
        "optimizer_reset_proof": optimizer_reset_proof,
        "parent_lineage": normalized_parent_lineage,
        "graft_lineage": normalized_graft_lineage,
        "epoch0_equivalence": verification,
        "cuda_rng_topology": cuda_rng_topology,
    }
    manifest = _atomic_json_write(
        Path(manifest_path).expanduser().resolve(), manifest_payload
    )
    return {
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "parent_checkpoint_sha256": parent_sha,
        "copied_generator_keys": list(key_plan["copied_generator_keys"]),
        "new_generator_keys": list(key_plan["new_generator_keys"]),
        "target_generator_mode": str(key_plan["target_generator_mode"]),
        "optimizer_reset_proof": optimizer_reset_proof,
        "epoch0_equivalence": verification,
        "cuda_rng_topology": cuda_rng_topology,
    }


def _read_hash_bound_manifest(path: Path, expected_sha256: str) -> dict[str, object]:
    expected = _require_sha256(expected_sha256, label="graft manifest SHA256")
    if not path.is_file():
        raise FileNotFoundError(f"graft manifest does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise PureCnnFilmGraftError(
            f"graft manifest SHA256 mismatch: expected={expected}, actual={observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PureCnnFilmGraftError("graft manifest root must be an object")
    if (
        int(payload.get("schema_version", -1)) != GRAFT_STATE_SCHEMA_VERSION
        or payload.get("kind") != GRAFT_MANIFEST_KIND
    ):
        raise PureCnnFilmGraftError("graft manifest schema/kind mismatch")
    return payload


def read_pure_cnn_graft_metadata(
    *,
    artifact_path: str | Path,
    expected_artifact_sha256: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Read graft metadata only after validating the artifact hash."""

    artifact = Path(artifact_path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"graft artifact does not exist: {artifact}")
    expected = _require_sha256(expected_artifact_sha256, label="graft artifact SHA256")
    observed = sha256_file(artifact)
    if observed != expected:
        raise PureCnnFilmGraftError(
            f"graft artifact SHA256 mismatch: expected={expected}, actual={observed}"
        )
    payload = torch.load(artifact, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise PureCnnFilmGraftError("graft artifact root must be a mapping")
    if (
        int(payload.get("schema_version", -1)) != GRAFT_STATE_SCHEMA_VERSION
        or payload.get("kind") != GRAFT_STATE_KIND
        or payload.get("save_phase") != GRAFT_SAVE_PHASE
    ):
        raise PureCnnFilmGraftError("graft artifact schema/kind/save phase mismatch")
    parent_lineage = _normalized_lineage(payload.get("parent_lineage", {}))
    graft_lineage = _normalized_lineage(payload.get("graft_lineage", {}))
    cuda_rng_topology = _validated_cuda_rng_topology(payload)
    return {
        "artifact_path": str(artifact),
        "artifact_sha256": observed,
        "parent_checkpoint_path": str(payload.get("parent_checkpoint_path", "")),
        "parent_checkpoint_sha256": _require_sha256(
            payload.get("parent_checkpoint_sha256"),
            label="parent checkpoint SHA256",
        ),
        "target_generator_mode": str(payload.get("target_generator_mode", "")),
        "copied_generator_keys": list(payload.get("copied_generator_keys", ())),
        "new_generator_keys": list(payload.get("new_generator_keys", ())),
        "parent_lineage": parent_lineage,
        "graft_lineage": graft_lineage,
        "optimizer_reset_proof": dict(payload.get("optimizer_reset_proof") or {}),
        "epoch0_equivalence": dict(payload.get("epoch0_equivalence") or {}),
        "cuda_rng_topology": cuda_rng_topology,
    }


def load_pure_cnn_to_film_graft_state(
    *,
    artifact_path: str | Path,
    expected_artifact_sha256: str = "",
    manifest_path: str | Path | None = None,
    expected_manifest_sha256: str = "",
    expected_parent_lineage: Mapping[str, object],
    expected_graft_lineage: Mapping[str, object],
    target_generator: torch.nn.Module,
    target_discriminator: torch.nn.Module,
    loader_generator: torch.Generator | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Strictly apply G/D weights and optionally restore the recorded RNG.

    Optimizer and scheduler state is intentionally never loaded.  The caller
    must construct those objects after this function returns, which makes the
    fresh-restart boundary mechanically enforceable in the training path. A
    worker may bind the artifact directly by SHA; an orchestrator may also
    supply the immutable manifest and its SHA.
    """

    artifact = Path(artifact_path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"graft artifact does not exist: {artifact}")
    observed_artifact_sha = sha256_file(artifact)
    manifest: dict[str, object] | None = None
    resolved_manifest_path = ""
    resolved_manifest_sha = ""
    if manifest_path is not None:
        resolved_manifest = Path(manifest_path).expanduser().resolve()
        manifest = _read_hash_bound_manifest(
            resolved_manifest, expected_manifest_sha256
        )
        resolved_manifest_path = str(resolved_manifest)
        resolved_manifest_sha = _require_sha256(
            expected_manifest_sha256, label="graft manifest SHA256"
        )
        if Path(str(manifest.get("artifact_path", ""))).resolve() != artifact:
            raise PureCnnFilmGraftError("graft artifact path differs from manifest")
        if (
            int(manifest.get("artifact_size_bytes", -1)) != artifact.stat().st_size
            or _require_sha256(
                manifest.get("artifact_sha256"), label="graft artifact SHA256"
            )
            != observed_artifact_sha
        ):
            raise PureCnnFilmGraftError("graft artifact size/SHA256 mismatch")
    elif not expected_artifact_sha256:
        raise PureCnnFilmGraftError(
            "direct graft loading requires expected_artifact_sha256"
        )
    if expected_artifact_sha256:
        expected_artifact_sha = _require_sha256(
            expected_artifact_sha256, label="graft artifact SHA256"
        )
        if observed_artifact_sha != expected_artifact_sha:
            raise PureCnnFilmGraftError(
                "graft artifact SHA256 mismatch: "
                f"expected={expected_artifact_sha}, actual={observed_artifact_sha}"
            )
    payload = torch.load(artifact, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise PureCnnFilmGraftError("graft artifact root must be a mapping")
    if (
        int(payload.get("schema_version", -1)) != GRAFT_STATE_SCHEMA_VERSION
        or payload.get("kind") != GRAFT_STATE_KIND
        or payload.get("save_phase") != GRAFT_SAVE_PHASE
    ):
        raise PureCnnFilmGraftError("graft artifact schema/kind/save phase mismatch")
    normalized_parent_lineage = _normalized_lineage(expected_parent_lineage)
    normalized_graft_lineage = _normalized_lineage(expected_graft_lineage)
    mirrored_fields = (
        "parent_checkpoint_path",
        "parent_checkpoint_sha256",
        "target_generator_mode",
        "copied_generator_keys",
        "new_generator_keys",
        "film_projection_keys",
        "critic_keys",
        "optimizer_reset_proof",
        "parent_lineage",
        "graft_lineage",
        "epoch0_equivalence",
        "cuda_rng_topology",
    )
    if (
        payload.get("parent_lineage") != normalized_parent_lineage
        or payload.get("graft_lineage") != normalized_graft_lineage
        or (
            manifest is not None
            and any(payload.get(name) != manifest.get(name) for name in mirrored_fields)
        )
    ):
        raise PureCnnFilmGraftError("graft artifact/manifest lineage metadata drift")
    if not bool(dict(payload.get("epoch0_equivalence") or {}).get("passed")):
        raise PureCnnFilmGraftError("graft artifact lacks successful equivalence proof")
    cuda_rng_topology = _validated_cuda_rng_topology(payload)
    reset_proof = dict(payload.get("optimizer_reset_proof") or {})
    expected_reset_proof = {
        "generator_optimizer_state_empty": True,
        "discriminator_optimizer_state_empty": True,
        "generator_scheduler_fresh": True,
        "discriminator_scheduler_fresh": True,
        "apply_loads_optimizer_or_scheduler_state": False,
    }
    if reset_proof != expected_reset_proof:
        raise PureCnnFilmGraftError("graft optimizer-reset proof is invalid")
    for name in (
        "generator_optimizer_state_dict",
        "discriminator_optimizer_state_dict",
    ):
        state = payload.get(name)
        if not isinstance(state, Mapping) or state.get("state"):
            raise PureCnnFilmGraftError(
                f"graft fresh-state evidence is invalid: {name}"
            )
    for name in (
        "generator_scheduler_state_dict",
        "discriminator_scheduler_state_dict",
    ):
        state = payload.get(name)
        if (
            not isinstance(state, Mapping)
            or int(state.get("last_epoch", -1)) != 0
            or int(state.get("num_bad_epochs", -1)) != 0
        ):
            raise PureCnnFilmGraftError(
                f"graft fresh-state evidence is invalid: {name}"
            )

    target_mode = str(getattr(target_generator, "generator_conditioning_mode", ""))
    if target_mode != str(payload.get("target_generator_mode", "")):
        raise PureCnnFilmGraftError("destination Generator mode differs from graft")
    if target_mode not in SUPPORTED_TARGET_GENERATOR_MODES:
        raise PureCnnFilmGraftError("unsupported graft destination Generator mode")
    if (
        getattr(target_discriminator, "critic_conditioning_mode", None)
        != LP_DISABLED_SAME_SHAPE_CRITIC_CONDITIONING_MODE
    ):
        raise PureCnnFilmGraftError("destination Critic must use the NoLP contract")
    generator_state = payload.get("generator_state_dict")
    discriminator_state = payload.get("discriminator_state_dict")
    if not isinstance(generator_state, Mapping) or not isinstance(
        discriminator_state, Mapping
    ):
        raise PureCnnFilmGraftError("graft artifact model state is incomplete")
    if set(generator_state) != set(target_generator.state_dict()):
        raise PureCnnFilmGraftError("destination Generator state keys differ")
    if set(discriminator_state) != set(target_discriminator.state_dict()):
        raise PureCnnFilmGraftError("destination Critic state keys differ")
    _validate_tensor_compatibility(
        generator_state,
        target_generator.state_dict(),
        tuple(sorted(generator_state)),
        label="loaded Generator",
    )
    _validate_tensor_compatibility(
        discriminator_state,
        target_discriminator.state_dict(),
        tuple(sorted(discriminator_state)),
        label="loaded Critic",
    )
    target_generator.load_state_dict(generator_state, strict=True)
    target_discriminator.load_state_dict(discriminator_state, strict=True)
    if target_mode == FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        _assert_film_projections_zero(target_generator)
    if restore_rng:
        if not isinstance(loader_generator, torch.Generator):
            raise PureCnnFilmGraftError(
                "restore_rng=True requires a DataLoader torch.Generator"
            )
        _restore_graft_rng_state(
            payload.get("rng_state"), loader_generator=loader_generator
        )
    return {
        "artifact_path": str(artifact),
        "artifact_sha256": observed_artifact_sha,
        "manifest_path": resolved_manifest_path,
        "manifest_sha256": resolved_manifest_sha,
        "parent_checkpoint_sha256": str(payload["parent_checkpoint_sha256"]),
        "copied_generator_keys": list(payload["copied_generator_keys"]),
        "new_generator_keys": list(payload["new_generator_keys"]),
        "target_generator_mode": target_mode,
        "parent_lineage": normalized_parent_lineage,
        "graft_lineage": normalized_graft_lineage,
        "optimizer_reset_proof": reset_proof,
        "epoch0_equivalence": dict(payload["epoch0_equivalence"]),
        "cuda_rng_topology": cuda_rng_topology,
        "rng_restored": bool(restore_rng),
    }


# Explicit worker-facing alias: this operation applies only G/D and optional
# RNG state. It cannot accept, and therefore cannot restore, optimizer/scheduler
# objects.
apply_pure_cnn_graft_state = load_pure_cnn_to_film_graft_state
save_pure_cnn_graft_state = save_pure_cnn_to_film_graft_state


__all__ = [
    "FILM_PROJECTION_PREFIXES",
    "GRAFT_MANIFEST_KIND",
    "GRAFT_SAVE_PHASE",
    "GRAFT_STATE_KIND",
    "GRAFT_STATE_SCHEMA_VERSION",
    "NEW_FILM_GENERATOR_PREFIXES",
    "PureCnnFilmGraftError",
    "SHARED_GENERATOR_PREFIXES",
    "SUPPORTED_TARGET_GENERATOR_MODES",
    "VERIFICATION_TEXT_KEYS",
    "apply_pure_cnn_graft_state",
    "graft_pure_cnn_fresh_restart",
    "graft_pure_cnn_to_film",
    "load_pure_cnn_to_film_graft_state",
    "read_pure_cnn_graft_metadata",
    "save_pure_cnn_graft_state",
    "save_pure_cnn_to_film_graft_state",
    "sha256_file",
    "verify_graft_text_invariance",
]
