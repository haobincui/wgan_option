"""Prediction contracts for the Pure-CNN parent -> FiLM text experiment.

The heavy model loading and surface-metric implementation deliberately stays in
the existing news-first inference stack.  This module owns the experiment-level
parts that are easy to get subtly wrong: the 280/80 prediction universes,
checkpoint-freeze gate, sparse validation trajectory, independent wrong-text
mapping, shared Monte-Carlo noise lineage, and hash-bound manifests.

Nothing in this module reads a test panel implicitly.  A caller must first pass
``require_frozen_test_access``; the optional adapter
``evaluate_frozen_test_cell_with_existing_core`` applies that gate before it
delegates to the already-audited production evaluator.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, ContextManager, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_SEEDS = (
    42,
    202,
    404,
    382624741,
    1607127774,
    1662128673,
    2041145538,
    2014889368,
    1343862330,
    779214671,
)
CANONICAL_FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
PARENT_ARM = "pure_cnn_parent"
BRANCH_ARMS = (
    "pure_cnn_continue_no_text",
    "film_zero_text",
    "film_lp_matched",
    "film_lp_shuffle",
    "film_bow",
    "film_sentiment",
)
STANDARD_ARMS = (PARENT_ARM, *BRANCH_ARMS)
PRIMARY_ANALYSIS_ARM = {
    "film_lp_matched": "matched",
    "film_zero_text": "film_zero_text",
    "film_lp_shuffle": "film_lp_shuffle",
}
INTERVENTION_CONDITIONS = ("zero_input", "wrong_input")
INTERVENTION_SOURCE_ARM = "film_lp_matched"
TRAJECTORY_FIXED_EPOCHS = (0, 1, 5, 10, 20, 30)
TRAJECTORY_LABELS = tuple(f"epoch_{epoch}" for epoch in TRAJECTORY_FIXED_EPOCHS) + (
    "best",
)

EXPECTED_STANDARD_CELLS = 280
EXPECTED_INTERVENTION_CELLS = 80
EXPECTED_TRAJECTORY_CELLS = 1_680
VALIDATION_MC_SAMPLES = 16
TEST_MC_SAMPLES = 64
NOISE_DIMENSION = 32

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class TextEffectPredictionError(ValueError):
    """Raised when prediction lineage violates the frozen experiment design."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TextEffectPredictionError(
            "payload is not finite canonical JSON"
        ) from error


def payload_sha256(value: object) -> str:
    """Return a stable SHA-256 for one finite JSON payload."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one artifact without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise TextEffectPredictionError(f"{label} is not a lowercase SHA-256")
    return digest


def _as_frame(
    source: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path, label: str
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise TextEffectPredictionError(f"{label} CSV is missing: {path}")
        frame = pd.read_csv(path, low_memory=False)
    else:
        frame = pd.DataFrame([dict(row) for row in source])
    if frame.empty:
        raise TextEffectPredictionError(f"{label} is empty")
    return frame


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TextEffectPredictionError(f"{label} is missing columns: {missing}")


def canonical_training_jobs(
    jobs: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    folds: Sequence[str] = CANONICAL_FOLDS,
    arms: Sequence[str] = STANDARD_ARMS,
) -> pd.DataFrame:
    """Validate and return the exact seed x fold x arm training universe."""

    frame = _as_frame(jobs, "training jobs")
    _require_columns(frame, {"job_id", "seed", "fold", "arm"}, label="training jobs")
    frame = frame.copy()
    for column in ("job_id", "fold", "arm"):
        frame[column] = frame[column].astype(str).str.strip()
    numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
    if (
        numeric_seed.isna().any()
        or not np.equal(numeric_seed, np.floor(numeric_seed)).all()
    ):
        raise TextEffectPredictionError("training job seeds must be integers")
    frame["seed"] = numeric_seed.astype(int)
    if frame[["job_id", "fold", "arm"]].eq("").any().any():
        raise TextEffectPredictionError("training jobs contain empty identity fields")
    if (
        frame["job_id"].duplicated().any()
        or frame.duplicated(["seed", "fold", "arm"]).any()
    ):
        raise TextEffectPredictionError("training job identities are duplicated")
    expected = {
        (int(seed), str(fold), str(arm))
        for seed in seeds
        for fold in folds
        for arm in arms
    }
    observed = set(frame[["seed", "fold", "arm"]].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise TextEffectPredictionError(
            f"training job universe drift: missing={missing[:3]}, extra={extra[:3]}"
        )
    return frame.sort_values(["seed", "fold", "arm"], kind="stable").reset_index(
        drop=True
    )


def verify_checkpoint_allowlist(
    source: pd.DataFrame | str | Path,
    *,
    expected_job_ids: Sequence[str],
    expected_file_sha256: str | None = None,
    verify_artifacts: bool = True,
) -> pd.DataFrame:
    """Validate the exact frozen G/D pair for every training job."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        if expected_file_sha256 is not None:
            raise TextEffectPredictionError(
                "an in-memory allowlist cannot satisfy a file-SHA contract"
            )
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise TextEffectPredictionError(f"checkpoint allowlist is missing: {path}")
        if expected_file_sha256 is not None and sha256_file(path) != _require_sha256(
            expected_file_sha256, label="checkpoint allowlist SHA"
        ):
            raise TextEffectPredictionError("checkpoint allowlist SHA drift")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "job_id",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
        "size_bytes",
    }
    _require_columns(frame, required, label="checkpoint allowlist")
    frame = frame.copy()
    for column in ("job_id", "checkpoint_role", "checkpoint_path", "checkpoint_sha256"):
        frame[column] = frame[column].astype(str).str.strip()
    sizes = pd.to_numeric(frame["size_bytes"], errors="coerce")
    if (
        sizes.isna().any()
        or not np.equal(sizes, np.floor(sizes)).all()
        or sizes.le(0).any()
    ):
        raise TextEffectPredictionError(
            "checkpoint allowlist sizes must be positive integers"
        )
    frame["size_bytes"] = sizes.astype(int)
    expected_jobs = {str(value) for value in expected_job_ids}
    expected_roles = {"generator_best_learned", "discriminator_best_learned"}
    expected_cells = {
        (job_id, role) for job_id in expected_jobs for role in expected_roles
    }
    observed_cells = set(
        frame[["job_id", "checkpoint_role"]].itertuples(index=False, name=None)
    )
    if (
        observed_cells != expected_cells
        or frame.duplicated(["job_id", "checkpoint_role"]).any()
    ):
        raise TextEffectPredictionError("checkpoint allowlist G/D universe drift")
    for row in frame.itertuples(index=False):
        digest = _require_sha256(row.checkpoint_sha256, label="checkpoint SHA")
        if verify_artifacts:
            path = Path(row.checkpoint_path).expanduser().resolve()
            if not path.is_file():
                raise TextEffectPredictionError(f"checkpoint is missing: {path}")
            if path.stat().st_size != int(row.size_bytes):
                raise TextEffectPredictionError(f"checkpoint size drift: {path}")
            if sha256_file(path) != digest:
                raise TextEffectPredictionError(f"checkpoint SHA drift: {path}")
    return frame.sort_values(["job_id", "checkpoint_role"], kind="stable").reset_index(
        drop=True
    )


def require_frozen_test_access(
    registry: Mapping[str, Any],
    *,
    expected_training_jobs: int = EXPECTED_STANDARD_CELLS,
    verify_allowlist_artifacts: bool = True,
) -> pd.DataFrame:
    """Fail before any test read unless the complete checkpoint freeze is valid."""

    if not isinstance(registry, Mapping):
        raise TextEffectPredictionError("registry must be a mapping")
    if not bool(registry.get("evaluation_frozen")):
        raise TextEffectPredictionError("test access requires evaluation_frozen=true")
    jobs = list(registry.get("jobs") or [])
    if len(jobs) != int(expected_training_jobs):
        raise TextEffectPredictionError(
            f"test access requires {expected_training_jobs} frozen training jobs"
        )
    job_ids = [str(dict(job).get("job_id", "")) for job in jobs]
    if any(not value for value in job_ids) or len(set(job_ids)) != len(job_ids):
        raise TextEffectPredictionError("registry job IDs are empty or duplicated")
    allowlist_path = str(registry.get("checkpoint_allowlist_path") or "").strip()
    allowlist_sha = str(registry.get("checkpoint_allowlist_sha256") or "").strip()
    if not allowlist_path or not allowlist_sha:
        raise TextEffectPredictionError(
            "frozen checkpoint allowlist lineage is missing"
        )
    return verify_checkpoint_allowlist(
        allowlist_path,
        expected_job_ids=job_ids,
        expected_file_sha256=allowlist_sha,
        verify_artifacts=verify_allowlist_artifacts,
    )


def _generator_checkpoint_map(allowlist: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected = allowlist.loc[allowlist["checkpoint_role"].eq("generator_best_learned")]
    return {
        str(row.job_id): {
            "path": str(Path(row.checkpoint_path).expanduser().resolve()),
            "sha256": str(row.checkpoint_sha256),
            "size_bytes": int(row.size_bytes),
            "role": "generator_best_learned",
        }
        for row in selected.itertuples(index=False)
    }


def noise_bank_profile_sha256(
    *,
    split: str,
    seed: int,
    fold: str,
    sample_ids: Sequence[str],
    draws: int,
    tolerance_minutes: int = 5,
    noise_dim: int = NOISE_DIMENSION,
) -> str:
    """Hash an arm-independent stable-key Monte-Carlo noise bank."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"validation", "test"}:
        raise TextEffectPredictionError("noise split must be validation or test")
    identifiers = [str(value).strip() for value in sample_ids]
    if not identifiers or any(not value for value in identifiers):
        raise TextEffectPredictionError("noise-bank sample IDs must be non-empty")
    if len(identifiers) != len(set(identifiers)):
        raise TextEffectPredictionError("noise-bank sample IDs must be unique")
    expected_draws = (
        VALIDATION_MC_SAMPLES if normalized_split == "validation" else TEST_MC_SAMPLES
    )
    if int(draws) != expected_draws or int(noise_dim) != NOISE_DIMENSION:
        raise TextEffectPredictionError(
            f"{normalized_split} noise contract requires MC={expected_draws}, dim=32"
        )
    return payload_sha256(
        {
            "schema_version": 1,
            "kind": "pure_cnn_film_text_effect_shared_mc_noise_bank_v1",
            "method": "stable_noise_for_keys_v1",
            "split": normalized_split,
            "tolerance_minutes": int(tolerance_minutes),
            "fold": str(fold),
            "seed": int(seed),
            "sample_ids": sorted(identifiers),
            "draws": int(draws),
            "noise_dim": int(noise_dim),
        }
    )


def plan_standard_prediction_units(
    jobs: pd.DataFrame | Sequence[Mapping[str, Any]],
    allowlist: pd.DataFrame | str | Path,
    *,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    folds: Sequence[str] = CANONICAL_FOLDS,
    arms: Sequence[str] = STANDARD_ARMS,
    verify_checkpoint_artifacts: bool = True,
) -> pd.DataFrame:
    """Plan exactly one frozen-test prediction for each of the 280 jobs."""

    job_frame = canonical_training_jobs(jobs, seeds=seeds, folds=folds, arms=arms)
    frozen = verify_checkpoint_allowlist(
        allowlist,
        expected_job_ids=job_frame["job_id"].tolist(),
        verify_artifacts=verify_checkpoint_artifacts,
    )
    checkpoint_by_job = _generator_checkpoint_map(frozen)
    rows: list[dict[str, Any]] = []
    for job in job_frame.to_dict(orient="records"):
        checkpoint = checkpoint_by_job[str(job["job_id"])]
        arm = str(job["arm"])
        rows.append(
            {
                "prediction_unit_id": f"standard::{job['job_id']}",
                "prediction_kind": "standard_test",
                "job_id": str(job["job_id"]),
                "seed": int(job["seed"]),
                "fold": str(job["fold"]),
                "arm": arm,
                "analysis_arm": PRIMARY_ANALYSIS_ARM.get(arm, arm),
                "model_profile": (
                    "pure_cnn"
                    if arm in {PARENT_ARM, "pure_cnn_continue_no_text"}
                    else "film_unet"
                ),
                "input_overlay_arm": arm,
                "checkpoint_role": checkpoint["role"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_size_bytes": checkpoint["size_bytes"],
                "split": "test",
                "tolerance_minutes": 5,
                "mc_samples": TEST_MC_SAMPLES,
                "noise_bank_namespace": (
                    f"text_effect/test/05m/{job['fold']}/seed_{int(job['seed'])}"
                ),
            }
        )
    result = (
        pd.DataFrame(rows)
        .sort_values(["seed", "fold", "arm"], kind="stable")
        .reset_index(drop=True)
    )
    expected_count = len(tuple(seeds)) * len(tuple(folds)) * len(tuple(arms))
    if len(result) != expected_count or result["prediction_unit_id"].duplicated().any():
        raise TextEffectPredictionError("standard prediction unit universe drift")
    return result


def plan_intervention_prediction_units(
    jobs: pd.DataFrame | Sequence[Mapping[str, Any]],
    allowlist: pd.DataFrame | str | Path,
    *,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    folds: Sequence[str] = CANONICAL_FOLDS,
    verify_checkpoint_artifacts: bool = True,
) -> pd.DataFrame:
    """Plan the 80 extra zero/wrong inputs for matched FiLM checkpoints.

    ``matched_input`` is deliberately absent: it reuses the corresponding
    standard ``film_lp_matched`` prediction, so it is not a new inference cell.
    """

    job_frame = canonical_training_jobs(
        jobs, seeds=seeds, folds=folds, arms=STANDARD_ARMS
    )
    frozen = verify_checkpoint_allowlist(
        allowlist,
        expected_job_ids=job_frame["job_id"].tolist(),
        verify_artifacts=verify_checkpoint_artifacts,
    )
    checkpoint_by_job = _generator_checkpoint_map(frozen)
    matched = job_frame.loc[job_frame["arm"].eq(INTERVENTION_SOURCE_ARM)]
    rows: list[dict[str, Any]] = []
    for job in matched.to_dict(orient="records"):
        checkpoint = checkpoint_by_job[str(job["job_id"])]
        for condition in INTERVENTION_CONDITIONS:
            rows.append(
                {
                    "prediction_unit_id": f"intervention::{job['job_id']}::{condition}",
                    "prediction_kind": "matched_checkpoint_intervention",
                    "source_job_id": str(job["job_id"]),
                    "job_id": str(job["job_id"]),
                    "seed": int(job["seed"]),
                    "fold": str(job["fold"]),
                    "arm": INTERVENTION_SOURCE_ARM,
                    "input_condition": condition,
                    "input_overlay_arm": (
                        "film_zero_text"
                        if condition == "zero_input"
                        else "film_lp_independent_wrong"
                    ),
                    "checkpoint_role": checkpoint["role"],
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_size_bytes": checkpoint["size_bytes"],
                    "split": "test",
                    "tolerance_minutes": 5,
                    "mc_samples": TEST_MC_SAMPLES,
                    "noise_bank_namespace": (
                        f"text_effect/test/05m/{job['fold']}/seed_{int(job['seed'])}"
                    ),
                }
            )
    result = (
        pd.DataFrame(rows)
        .sort_values(["seed", "fold", "input_condition"], kind="stable")
        .reset_index(drop=True)
    )
    expected_count = (
        len(tuple(seeds)) * len(tuple(folds)) * len(INTERVENTION_CONDITIONS)
    )
    if len(result) != expected_count or result["prediction_unit_id"].duplicated().any():
        raise TextEffectPredictionError("intervention prediction unit universe drift")
    return result


def _normalise_trajectory_label(value: object) -> str:
    label = str(value).strip().lower().replace("-", "_")
    aliases = {
        "generator_initial_epoch0": "epoch_0",
        "initial_epoch0": "epoch_0",
        "generator_best_learned": "best",
        "best_learned": "best",
    }
    label = aliases.get(label, label)
    match = re.fullmatch(r"(?:generator_)?epoch_?(0|1|5|10|20|30)", label)
    if match:
        return f"epoch_{int(match.group(1))}"
    if label == "best":
        return label
    raise TextEffectPredictionError(
        f"unsupported trajectory checkpoint label: {value!r}"
    )


def validate_trajectory_checkpoint_inventory(
    inventory: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
    *,
    branch_jobs: pd.DataFrame,
    verify_artifacts: bool = True,
) -> pd.DataFrame:
    """Require sparse epoch 0/1/5/10/20/30 plus selected-best checkpoints."""

    frame = _as_frame(inventory, "trajectory checkpoint inventory")
    label_column = (
        "checkpoint_label" if "checkpoint_label" in frame.columns else "checkpoint_role"
    )
    _require_columns(
        frame,
        {
            "job_id",
            label_column,
            "epoch",
            "checkpoint_path",
            "checkpoint_sha256",
            "size_bytes",
        },
        label="trajectory checkpoint inventory",
    )
    frame = frame.copy()
    frame["job_id"] = frame["job_id"].astype(str).str.strip()
    frame["checkpoint_label"] = frame[label_column].map(_normalise_trajectory_label)
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    sizes = pd.to_numeric(frame["size_bytes"], errors="coerce")
    if (
        epochs.isna().any()
        or not np.equal(epochs, np.floor(epochs)).all()
        or sizes.isna().any()
        or not np.equal(sizes, np.floor(sizes)).all()
        or sizes.le(0).any()
    ):
        raise TextEffectPredictionError("trajectory epoch/size fields are invalid")
    frame["epoch"] = epochs.astype(int)
    frame["size_bytes"] = sizes.astype(int)
    if frame.duplicated(["job_id", "checkpoint_label"]).any():
        raise TextEffectPredictionError("trajectory checkpoint labels are duplicated")
    expected_jobs = set(branch_jobs["job_id"].astype(str))
    expected_cells = {
        (job_id, label) for job_id in expected_jobs for label in TRAJECTORY_LABELS
    }
    observed_cells = set(
        frame[["job_id", "checkpoint_label"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise TextEffectPredictionError("trajectory checkpoint universe drift")
    for row in frame.itertuples(index=False):
        label = str(row.checkpoint_label)
        if label.startswith("epoch_") and int(row.epoch) != int(label.split("_")[1]):
            raise TextEffectPredictionError(
                f"trajectory epoch disagrees with label: {row.job_id}/{label}"
            )
        if label == "best" and not 1 <= int(row.epoch) <= 240:
            raise TextEffectPredictionError("best trajectory epoch must lie in [1,240]")
        digest = _require_sha256(
            row.checkpoint_sha256, label="trajectory checkpoint SHA"
        )
        if verify_artifacts:
            path = Path(row.checkpoint_path).expanduser().resolve()
            if not path.is_file() or path.stat().st_size != int(row.size_bytes):
                raise TextEffectPredictionError(
                    f"trajectory checkpoint size drift: {path}"
                )
            if sha256_file(path) != digest:
                raise TextEffectPredictionError(
                    f"trajectory checkpoint SHA drift: {path}"
                )
    return frame.sort_values(["job_id", "checkpoint_label"], kind="stable").reset_index(
        drop=True
    )


def plan_validation_trajectory_units(
    jobs: pd.DataFrame | Sequence[Mapping[str, Any]],
    inventory: pd.DataFrame | Sequence[Mapping[str, Any]],
    parent_allowlist: pd.DataFrame | str | Path,
    *,
    seeds: Sequence[int] = CANONICAL_SEEDS,
    folds: Sequence[str] = CANONICAL_FOLDS,
    verify_checkpoint_artifacts: bool = True,
) -> pd.DataFrame:
    """Plan 240 x seven validation snapshots without touching a test panel."""

    all_jobs = canonical_training_jobs(
        jobs, seeds=seeds, folds=folds, arms=STANDARD_ARMS
    )
    branches = all_jobs.loc[all_jobs["arm"].isin(BRANCH_ARMS)].copy()
    frozen = verify_checkpoint_allowlist(
        parent_allowlist,
        expected_job_ids=all_jobs["job_id"].tolist(),
        verify_artifacts=verify_checkpoint_artifacts,
    )
    parent_checkpoint_by_job = _generator_checkpoint_map(frozen)
    parent_job_by_cell = {
        (int(row.seed), str(row.fold)): str(row.job_id)
        for row in all_jobs.loc[all_jobs["arm"].eq(PARENT_ARM)].itertuples(index=False)
    }
    checkpoints = validate_trajectory_checkpoint_inventory(
        inventory,
        branch_jobs=branches,
        verify_artifacts=verify_checkpoint_artifacts,
    )
    checkpoint_rows = {
        (str(row.job_id), str(row.checkpoint_label)): row
        for row in checkpoints.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for job in branches.to_dict(orient="records"):
        parent_job_id = str(
            job.get("parent_job_id")
            or parent_job_by_cell[(int(job["seed"]), str(job["fold"]))]
        )
        expected_parent = parent_job_by_cell[(int(job["seed"]), str(job["fold"]))]
        if parent_job_id != expected_parent:
            raise TextEffectPredictionError(
                f"branch parent lineage drift: {job['job_id']}"
            )
        parent_checkpoint = parent_checkpoint_by_job[parent_job_id]
        for label in TRAJECTORY_LABELS:
            checkpoint = checkpoint_rows[(str(job["job_id"]), label)]
            rows.append(
                {
                    "prediction_unit_id": f"validation::{job['job_id']}::{label}",
                    "prediction_kind": "validation_trajectory",
                    "job_id": str(job["job_id"]),
                    "parent_job_id": parent_job_id,
                    "seed": int(job["seed"]),
                    "fold": str(job["fold"]),
                    "arm": str(job["arm"]),
                    "analysis_arm": PRIMARY_ANALYSIS_ARM.get(
                        str(job["arm"]), str(job["arm"])
                    ),
                    "checkpoint_label": label,
                    "epoch": int(checkpoint.epoch),
                    "checkpoint_path": str(Path(checkpoint.checkpoint_path).resolve()),
                    "checkpoint_sha256": str(checkpoint.checkpoint_sha256),
                    "checkpoint_size_bytes": int(checkpoint.size_bytes),
                    "parent_checkpoint_path": parent_checkpoint["path"],
                    "parent_checkpoint_sha256": parent_checkpoint["sha256"],
                    "split": "validation",
                    "tolerance_minutes": 5,
                    "mc_samples": VALIDATION_MC_SAMPLES,
                    "noise_bank_namespace": (
                        f"text_effect/validation/05m/{job['fold']}/seed_{int(job['seed'])}"
                    ),
                }
            )
    result = (
        pd.DataFrame(rows)
        .sort_values(["seed", "fold", "arm", "checkpoint_label"], kind="stable")
        .reset_index(drop=True)
    )
    expected_count = (
        len(tuple(seeds))
        * len(tuple(folds))
        * len(BRANCH_ARMS)
        * len(TRAJECTORY_LABELS)
    )
    if len(result) != expected_count or result["prediction_unit_id"].duplicated().any():
        raise TextEffectPredictionError("validation trajectory unit universe drift")
    return result


def attach_noise_bank_profiles(
    units: pd.DataFrame,
    *,
    sample_ids_by_split_fold: Mapping[tuple[str, str], Sequence[str]],
) -> pd.DataFrame:
    """Attach the shared MC profile SHA; arm/condition never enters the hash."""

    frame = units.copy()
    _require_columns(
        frame,
        {"prediction_unit_id", "split", "seed", "fold", "mc_samples"},
        label="prediction units",
    )
    profiles: list[str] = []
    for row in frame.itertuples(index=False):
        key = (str(row.split), str(row.fold))
        if key not in sample_ids_by_split_fold:
            raise TextEffectPredictionError(f"missing sample-ID universe for {key}")
        profiles.append(
            noise_bank_profile_sha256(
                split=str(row.split),
                seed=int(row.seed),
                fold=str(row.fold),
                sample_ids=sample_ids_by_split_fold[key],
                draws=int(row.mc_samples),
            )
        )
    frame["noise_bank_profile_sha256"] = profiles
    validate_shared_noise_banks(frame)
    return frame


def validate_shared_noise_banks(manifest: pd.DataFrame) -> None:
    """Ensure every arm/intervention/snapshot shares one bank per seed/fold/split."""

    _require_columns(
        manifest,
        {
            "prediction_unit_id",
            "split",
            "seed",
            "fold",
            "mc_samples",
            "noise_bank_profile_sha256",
        },
        label="noise-bank manifest",
    )
    if manifest["prediction_unit_id"].astype(str).duplicated().any():
        raise TextEffectPredictionError("noise-bank manifest has duplicate units")
    for row in manifest.itertuples(index=False):
        _require_sha256(row.noise_bank_profile_sha256, label="noise-bank profile SHA")
        split = str(row.split)
        expected = VALIDATION_MC_SAMPLES if split == "validation" else TEST_MC_SAMPLES
        if split not in {"validation", "test"} or int(row.mc_samples) != expected:
            raise TextEffectPredictionError("prediction unit MC contract drift")
    counts = manifest.groupby(["split", "seed", "fold"])[
        "noise_bank_profile_sha256"
    ].nunique()
    if not counts.eq(1).all():
        raise TextEffectPredictionError("arms do not share one MC noise bank")


def deterministic_wrong_text_mapping(
    pairs: pd.DataFrame,
    *,
    master_seed: int,
    namespace: str,
    forbidden_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build a deterministic, cross-session, one-to-one wrong-text mapping."""

    _require_columns(pairs, {"pair_id", "session_id"}, label="wrong-text pairs")
    frame = pairs[["pair_id", "session_id"]].copy()
    frame["pair_id"] = frame["pair_id"].astype(str).str.strip()
    frame["session_id"] = frame["session_id"].astype(str).str.strip()
    if (
        frame.empty
        or frame.eq("").any().any()
        or frame["pair_id"].duplicated().any()
        or frame["session_id"].nunique() < 2
    ):
        raise TextEffectPredictionError(
            "wrong-text mapping requires unique pairs across at least two sessions"
        )
    namespace = str(namespace).strip()
    if not namespace:
        raise TextEffectPredictionError("wrong-text namespace must be non-empty")
    sessions = dict(frame.itertuples(index=False, name=None))
    pair_ids = sorted(sessions)
    forbidden = {
        str(key): str(value) for key, value in (forbidden_mapping or {}).items()
    }

    def priority(receiver: str, donor: str) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": "independent_cross_session_derangement_v1",
                    "master_seed": int(master_seed),
                    "namespace": namespace,
                    "receiver": receiver,
                    "donor": donor,
                }
            )
        ).hexdigest()

    candidates = {
        receiver: sorted(
            (
                donor
                for donor in pair_ids
                if donor != receiver
                and sessions[donor] != sessions[receiver]
                and donor != forbidden.get(receiver)
            ),
            key=lambda donor: priority(receiver, donor),
        )
        for receiver in pair_ids
    }
    if any(not values for values in candidates.values()):
        raise TextEffectPredictionError(
            "no independent cross-session wrong-text donor exists for every pair"
        )
    donor_to_receiver: dict[str, str] = {}

    def augment(receiver: str, visited: set[str]) -> bool:
        for donor in candidates[receiver]:
            if donor in visited:
                continue
            visited.add(donor)
            previous = donor_to_receiver.get(donor)
            if previous is None or augment(previous, visited):
                donor_to_receiver[donor] = receiver
                return True
        return False

    receiver_order = sorted(pair_ids, key=lambda value: (len(candidates[value]), value))
    for receiver in receiver_order:
        if not augment(receiver, set()):
            raise TextEffectPredictionError(
                "unable to construct an independent cross-session derangement"
            )
    donor_by_receiver = {
        receiver: donor for donor, receiver in donor_to_receiver.items()
    }
    if set(donor_by_receiver) != set(pair_ids) or set(
        donor_by_receiver.values()
    ) != set(pair_ids):
        raise TextEffectPredictionError("wrong-text mapping is not a pair bijection")
    rows = [
        {
            "pair_id": receiver,
            "session_id": sessions[receiver],
            "donor_pair_id": donor_by_receiver[receiver],
            "donor_session_id": sessions[donor_by_receiver[receiver]],
        }
        for receiver in pair_ids
    ]
    result = pd.DataFrame(rows)
    mapping_payload = sorted((row["pair_id"], row["donor_pair_id"]) for row in rows)
    result["mapping_seed"] = int(master_seed)
    result["mapping_namespace"] = namespace
    result["mapping_sha256"] = payload_sha256(mapping_payload)
    return result


def _embedding_array(value: object, *, dimension: int, label: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise TextEffectPredictionError(f"{label} is not JSON") from error
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TextEffectPredictionError(f"{label} is not numeric") from error
    if result.shape != (int(dimension),) or not np.isfinite(result).all():
        raise TextEffectPredictionError(
            f"{label} must have finite shape ({int(dimension)},)"
        )
    return result


def build_intervention_overlay_frames(
    matched_overlay: pd.DataFrame,
    *,
    master_seed: int,
    namespace: str,
    embedding_dimension: int = 1024,
    training_shuffle_mapping: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create zero and independently-wrong overlays from a matched LP panel."""

    _require_columns(
        matched_overlay,
        {"pair_id", "session_id", "embedding"},
        label="matched intervention overlay",
    )
    matched = matched_overlay[["pair_id", "session_id", "embedding"]].copy()
    matched["pair_id"] = matched["pair_id"].astype(str).str.strip()
    matched["session_id"] = matched["session_id"].astype(str).str.strip()
    if matched["pair_id"].duplicated().any():
        raise TextEffectPredictionError("matched intervention overlay duplicates pairs")
    vectors = {
        str(row.pair_id): _embedding_array(
            row.embedding,
            dimension=embedding_dimension,
            label=f"matched embedding {row.pair_id}",
        )
        for row in matched.itertuples(index=False)
    }
    mapping = deterministic_wrong_text_mapping(
        matched,
        master_seed=int(master_seed),
        namespace=str(namespace),
        forbidden_mapping=training_shuffle_mapping,
    )
    mapping_by_pair = dict(
        mapping[["pair_id", "donor_pair_id"]].itertuples(index=False, name=None)
    )
    sessions = dict(
        matched[["pair_id", "session_id"]].itertuples(index=False, name=None)
    )
    zero_rows = [
        {
            "pair_id": pair_id,
            "session_id": sessions[pair_id],
            "embedding": np.zeros(int(embedding_dimension), dtype=np.float32),
            "input_condition": "zero_input",
        }
        for pair_id in sorted(vectors)
    ]
    wrong_rows = [
        {
            "pair_id": pair_id,
            "session_id": sessions[pair_id],
            "donor_pair_id": mapping_by_pair[pair_id],
            "embedding": vectors[mapping_by_pair[pair_id]].copy(),
            "input_condition": "wrong_input",
            "mapping_sha256": str(mapping["mapping_sha256"].iloc[0]),
        }
        for pair_id in sorted(vectors)
    ]
    return {
        "zero_input": pd.DataFrame(zero_rows),
        "wrong_input": pd.DataFrame(wrong_rows),
        "mapping": mapping,
    }


def standard_pair_metrics_for_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the three primary standard arms with analysis-compatible names."""

    _require_columns(
        frame,
        {
            "arm",
            "seed",
            "fold",
            "pair_id",
            "session_id",
            "target_mae",
            "persistence_mae",
        },
        label="standard pair metrics",
    )
    selected = frame.loc[frame["arm"].isin(PRIMARY_ANALYSIS_ARM)].copy()
    observed = set(selected["arm"].astype(str))
    if observed != set(PRIMARY_ANALYSIS_ARM):
        raise TextEffectPredictionError("primary standard arm universe drift")
    selected["training_arm"] = selected["arm"].astype(str)
    selected["arm"] = selected["training_arm"].map(PRIMARY_ANALYSIS_ARM)
    if selected.duplicated(["arm", "seed", "fold", "pair_id"]).any():
        raise TextEffectPredictionError("standard analysis pair lineage is duplicated")
    return selected.sort_values(
        ["seed", "fold", "arm", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def compose_intervention_analysis_panel(
    standard_pair_metrics: pd.DataFrame,
    counterfactual_pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine reused matched predictions with the 80 new counterfactual cells."""

    _require_columns(
        standard_pair_metrics,
        {"arm", "seed", "fold", "pair_id", "session_id", "target_mae"},
        label="standard pair metrics",
    )
    _require_columns(
        counterfactual_pair_metrics,
        {"input_condition", "seed", "fold", "pair_id", "session_id", "target_mae"},
        label="counterfactual pair metrics",
    )
    matched = standard_pair_metrics.loc[
        standard_pair_metrics["arm"].isin({"film_lp_matched", "matched"})
    ].copy()
    if matched.empty:
        raise TextEffectPredictionError("matched standard predictions are missing")
    matched["input_condition"] = "matched_input"
    counterfactual = counterfactual_pair_metrics.loc[
        counterfactual_pair_metrics["input_condition"].isin(INTERVENTION_CONDITIONS)
    ].copy()
    if set(counterfactual["input_condition"].astype(str)) != set(
        INTERVENTION_CONDITIONS
    ):
        raise TextEffectPredictionError("counterfactual condition universe drift")
    common_columns = [
        column
        for column in matched.columns
        if column in counterfactual.columns or column == "input_condition"
    ]
    required_columns = [
        "input_condition",
        "seed",
        "fold",
        "pair_id",
        "session_id",
        "target_mae",
    ]
    for column in required_columns:
        if column not in common_columns:
            common_columns.append(column)
    result = pd.concat(
        [matched[common_columns], counterfactual[common_columns]], ignore_index=True
    )
    keys = ["input_condition", "seed", "fold", "pair_id"]
    if result.duplicated(keys).any():
        raise TextEffectPredictionError("intervention analysis pairs are duplicated")
    lineage = result.pivot_table(
        index=["seed", "fold", "pair_id", "session_id"],
        columns="input_condition",
        values="target_mae",
        aggfunc="size",
        fill_value=0,
    )
    expected_conditions = {"matched_input", *INTERVENTION_CONDITIONS}
    if set(lineage.columns) != expected_conditions or not lineage.eq(1).all().all():
        raise TextEffectPredictionError(
            "intervention inputs do not share paired lineage"
        )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def build_prediction_manifest_record(
    unit: Mapping[str, Any],
    *,
    panel_path: str | Path,
    overlay_path: str | Path,
    prediction_path: str | Path,
    pair_metrics_path: str | Path,
    row_count: int,
    noise_bank_profile_sha256_value: str,
) -> dict[str, Any]:
    """Build one self-hashed manifest after all prediction files exist."""

    required = {
        "prediction_unit_id",
        "prediction_kind",
        "job_id",
        "seed",
        "fold",
        "checkpoint_path",
        "checkpoint_sha256",
        "mc_samples",
        "split",
    }
    missing = sorted(required - set(unit))
    if missing:
        raise TextEffectPredictionError(f"prediction unit is missing fields: {missing}")
    artifacts = {
        "panel": Path(panel_path).expanduser().resolve(),
        "overlay": Path(overlay_path).expanduser().resolve(),
        "prediction": Path(prediction_path).expanduser().resolve(),
        "pair_metrics": Path(pair_metrics_path).expanduser().resolve(),
    }
    for label, path in artifacts.items():
        if not path.is_file():
            raise TextEffectPredictionError(f"{label} artifact is missing: {path}")
    profile_sha = _require_sha256(
        noise_bank_profile_sha256_value, label="noise-bank profile SHA"
    )
    checkpoint_path = Path(str(unit["checkpoint_path"])).expanduser().resolve()
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != _require_sha256(
        unit["checkpoint_sha256"], label="unit checkpoint SHA"
    ):
        raise TextEffectPredictionError("unit checkpoint drift")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pure_cnn_film_text_effect_prediction_manifest_v1",
        "prediction_unit_id": str(unit["prediction_unit_id"]),
        "prediction_kind": str(unit["prediction_kind"]),
        "job_id": str(unit["job_id"]),
        "seed": int(unit["seed"]),
        "fold": str(unit["fold"]),
        "arm": str(unit.get("arm", "")),
        "input_condition": str(unit.get("input_condition", "")),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(unit["checkpoint_sha256"]),
        "panel_path": str(artifacts["panel"]),
        "panel_sha256": sha256_file(artifacts["panel"]),
        "overlay_path": str(artifacts["overlay"]),
        "overlay_sha256": sha256_file(artifacts["overlay"]),
        "prediction_path": str(artifacts["prediction"]),
        "prediction_sha256": sha256_file(artifacts["prediction"]),
        "pair_metrics_path": str(artifacts["pair_metrics"]),
        "pair_metrics_sha256": sha256_file(artifacts["pair_metrics"]),
        "row_count": int(row_count),
        "split": str(unit["split"]),
        "mc_samples": int(unit["mc_samples"]),
        "noise_bank_profile_sha256": profile_sha,
    }
    if payload["row_count"] <= 0:
        raise TextEffectPredictionError(
            "prediction manifest row_count must be positive"
        )
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def validate_prediction_manifests(
    manifests: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
    units: pd.DataFrame,
    *,
    verify_artifacts: bool = True,
) -> pd.DataFrame:
    """Validate one self-hashed manifest for every planned prediction unit."""

    frame = _as_frame(manifests, "prediction manifests")
    _require_columns(
        frame,
        {
            "prediction_unit_id",
            "job_id",
            "checkpoint_sha256",
            "prediction_path",
            "prediction_sha256",
            "pair_metrics_path",
            "pair_metrics_sha256",
            "noise_bank_profile_sha256",
            "mc_samples",
            "split",
            "payload_sha256",
        },
        label="prediction manifests",
    )
    expected_units = set(units["prediction_unit_id"].astype(str))
    if (
        set(frame["prediction_unit_id"].astype(str)) != expected_units
        or frame["prediction_unit_id"].astype(str).duplicated().any()
    ):
        raise TextEffectPredictionError("prediction manifest unit universe drift")
    unit_by_id = {
        str(row["prediction_unit_id"]): row for row in units.to_dict(orient="records")
    }
    for raw in frame.to_dict(orient="records"):
        unit = unit_by_id[str(raw["prediction_unit_id"])]
        unsigned = {key: value for key, value in raw.items() if key != "payload_sha256"}
        if payload_sha256(unsigned) != str(raw["payload_sha256"]):
            raise TextEffectPredictionError("prediction manifest payload SHA drift")
        if (
            str(raw["job_id"]) != str(unit["job_id"])
            or str(raw["checkpoint_sha256"]) != str(unit["checkpoint_sha256"])
            or int(raw["mc_samples"]) != int(unit["mc_samples"])
            or str(raw["split"]) != str(unit["split"])
        ):
            raise TextEffectPredictionError("prediction manifest/unit lineage drift")
        for path_key, sha_key in (
            ("prediction_path", "prediction_sha256"),
            ("pair_metrics_path", "pair_metrics_sha256"),
            ("panel_path", "panel_sha256"),
            ("overlay_path", "overlay_sha256"),
        ):
            if path_key not in raw or sha_key not in raw:
                raise TextEffectPredictionError(
                    f"prediction manifest lacks {path_key}/{sha_key}"
                )
            _require_sha256(raw[sha_key], label=sha_key)
            if verify_artifacts:
                path = Path(str(raw[path_key])).expanduser().resolve()
                if not path.is_file() or sha256_file(path) != str(raw[sha_key]):
                    raise TextEffectPredictionError(
                        f"prediction artifact drift: {path}"
                    )
    validate_shared_noise_banks(frame)
    return frame.sort_values("prediction_unit_id", kind="stable").reset_index(drop=True)


def evaluate_frozen_test_cell_with_existing_core(
    root: str | Path,
    registry: Mapping[str, Any],
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    *,
    expected_pairs: int,
    profile_context: Callable[[], ContextManager[Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Gate test access, then delegate to the existing production evaluator."""

    require_frozen_test_access(registry)
    from scripts.rq123 import news_first_vol_film_nolp_10seed as existing

    context = profile_context() if profile_context is not None else nullcontext()
    with context:
        return existing._evaluate_prediction_job(  # noqa: SLF001
            Path(root).resolve(),
            job,
            checkpoint,
            expected_pairs=int(expected_pairs),
        )


__all__ = [
    "BRANCH_ARMS",
    "CANONICAL_FOLDS",
    "CANONICAL_SEEDS",
    "EXPECTED_INTERVENTION_CELLS",
    "EXPECTED_STANDARD_CELLS",
    "EXPECTED_TRAJECTORY_CELLS",
    "INTERVENTION_CONDITIONS",
    "PARENT_ARM",
    "STANDARD_ARMS",
    "TEST_MC_SAMPLES",
    "TRAJECTORY_FIXED_EPOCHS",
    "TRAJECTORY_LABELS",
    "TextEffectPredictionError",
    "VALIDATION_MC_SAMPLES",
    "attach_noise_bank_profiles",
    "build_intervention_overlay_frames",
    "build_prediction_manifest_record",
    "canonical_training_jobs",
    "compose_intervention_analysis_panel",
    "deterministic_wrong_text_mapping",
    "evaluate_frozen_test_cell_with_existing_core",
    "noise_bank_profile_sha256",
    "payload_sha256",
    "plan_intervention_prediction_units",
    "plan_standard_prediction_units",
    "plan_validation_trajectory_units",
    "require_frozen_test_access",
    "sha256_file",
    "standard_pair_metrics_for_analysis",
    "validate_prediction_manifests",
    "validate_shared_noise_banks",
    "validate_trajectory_checkpoint_inventory",
    "verify_checkpoint_allowlist",
]
