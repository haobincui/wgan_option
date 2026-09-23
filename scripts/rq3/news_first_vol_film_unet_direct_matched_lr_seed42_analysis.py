"""Analysis for the direct FiLM-LR sweep against frozen Pure-CNN evidence.

The analysis is deliberately downstream-only: it never discovers checkpoints,
launches training, or opens an unfrozen dataset.  Callers must provide the
SHA-addressed 2,500-row FiLM table, the immutable 500-row Pure-CNN table, and
the 20-row FiLM training summary.  The module validates pair/session,
persistence, and MC-noise lineage before computing any comparison.

All results are single-seed retrospective development diagnostics.  The
lowest observed test MAE is reported as a *descriptive point leader* only and
must not be used to select a learning rate.
"""

from __future__ import annotations

import hashlib
import html
from itertools import combinations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42_analysis import (
    DirectFiveArmAnalysisError,
    compute_arm_summary,
    compute_fold_summary,
    fold_session_paired_bootstrap,
    holm_adjust,
    validate_pair_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FILM_LR_ARMS: Mapping[str, float] = {
    "film_lr_2p5e7": 2.5e-7,
    "film_lr_5e7": 5.0e-7,
    "film_lr_1e6": 1.0e-6,
    "film_lr_2p5e6": 2.5e-6,
    "film_lr_5e6": 5.0e-6,
}
PURE_ARM = "pure_cnn_no_text"
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
EXPECTED_FOLD_PAIR_SESSION_COUNTS: Mapping[str, tuple[int, int]] = {
    "f1_2023q1": (110, 34),
    "f2_2023q2": (112, 36),
    "f3_2023q3": (135, 33),
    "f4_2023q4": (143, 45),
}
SEED = 42
TOLERANCE_MINUTES = 5
EXPECTED_FILM_ROWS = 2_500
EXPECTED_PURE_ROWS = 500
EXPECTED_FILM_JOBS = 20
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260831
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
PURE_PAIR_METRICS_SHA256 = (
    "81d9c0626e33bd60e69174b0e470618c43e42724feb655f20840471e0a05436a"
)
NOISE_COLUMN = "noise_bank_profile_sha256"

EXPECTED_GROUPS: Mapping[str, tuple[int, float | None]] = {
    "backbone": (416_353, 5.0e-7),
    "text_encoder": (295_808, 2.5e-6),
    "film": (115_584, None),
    "critic": (729_157, 5.0e-7),
}


class FilmLrAnalysisError(ValueError):
    """Raised when frozen evidence or the analysis contract has drifted."""


@dataclass(frozen=True)
class TrainingEvidence:
    diagnostics: pd.DataFrame
    lr_trace: pd.DataFrame
    external_inputs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FilmLrAnalysis:
    film_pair_metrics: pd.DataFrame
    pure_pair_metrics: pd.DataFrame
    combined_pair_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    ranking: pd.DataFrame
    comparisons: pd.DataFrame
    pairwise_comparisons: pd.DataFrame
    training_diagnostics: pd.DataFrame
    lr_trace: pd.DataFrame
    training_external_inputs: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FilmLrAnalysisError(f"Required file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise FilmLrAnalysisError(f"{label} must be one lowercase SHA-256 digest")
    return digest


def _wrap_direct(label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except DirectFiveArmAnalysisError as exc:
        raise FilmLrAnalysisError(f"{label}: {exc}") from exc


def _validate_noise(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if NOISE_COLUMN not in frame.columns:
        raise FilmLrAnalysisError(f"{label} is missing {NOISE_COLUMN}")
    result = frame.copy()
    result[NOISE_COLUMN] = [
        _require_sha(value, f"{label} {NOISE_COLUMN}") for value in result[NOISE_COLUMN]
    ]
    for (fold, arm), group in result.groupby(["fold", "arm"], sort=True):
        if group[NOISE_COLUMN].nunique(dropna=False) != 1:
            raise FilmLrAnalysisError(
                f"{label} has multiple MC-noise banks for fold={fold}, arm={arm}"
            )
    return result


def validate_film_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = EXPECTED_FILM_ROWS,
) -> pd.DataFrame:
    result = _wrap_direct(
        "invalid FiLM-LR pair metrics",
        validate_pair_metrics,
        source,
        expected_arms=tuple(FILM_LR_ARMS),
        expected_seed=SEED,
        expected_fold_counts=expected_fold_counts,
        expected_tolerance_minutes=TOLERANCE_MINUTES,
        expected_row_count=expected_row_count,
    )
    result = _validate_noise(result, "FiLM-LR pair metrics")
    result["film_learning_rate"] = result["arm"].map(FILM_LR_ARMS).astype(float)
    return result


def validate_pure_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = EXPECTED_PURE_ROWS,
) -> pd.DataFrame:
    result = _wrap_direct(
        "invalid Pure-CNN pair metrics",
        validate_pair_metrics,
        source,
        expected_arms=(PURE_ARM,),
        expected_seed=SEED,
        expected_fold_counts=expected_fold_counts,
        expected_tolerance_minutes=TOLERANCE_MINUTES,
        expected_row_count=expected_row_count,
    )
    result = _validate_noise(result, "Pure-CNN pair metrics")
    result["film_learning_rate"] = np.nan
    return result


def validate_cross_experiment_lineage(
    film: pd.DataFrame,
    pure: pd.DataFrame,
) -> None:
    """Require exact fold/pair/session/persistence/MC-noise comparability."""

    lineage = [
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
        NOISE_COLUMN,
    ]
    candidate = (
        pure[lineage]
        .sort_values(["fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )
    for arm in FILM_LR_ARMS:
        reference = (
            film.loc[film["arm"].eq(arm), lineage]
            .sort_values(["fold", "pair_id"], kind="stable")
            .reset_index(drop=True)
        )
        if len(reference) != len(candidate):
            raise FilmLrAnalysisError(
                f"{arm} and Pure-CNN panels are not pair complete"
            )
        for column in (
            "fold",
            "pair_id",
            "session_id",
            "effective_origin_utc",
            NOISE_COLUMN,
        ):
            if not reference[column].equals(candidate[column]):
                raise FilmLrAnalysisError(
                    f"Cross-experiment {column} lineage differs for {arm} vs Pure CNN"
                )
        if not np.array_equal(
            reference["persistence_mae"].to_numpy(float),
            candidate["persistence_mae"].to_numpy(float),
        ):
            raise FilmLrAnalysisError(
                f"Cross-experiment persistence lineage differs for {arm} vs Pure CNN"
            )


def _read_frame(source: pd.DataFrame | str | Path, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise FilmLrAnalysisError(f"{label} does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def _finite_positive(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FilmLrAnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise FilmLrAnalysisError(f"{label} must be finite and positive")
    return numeric


def _contract_groups(
    payload: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("groups", payload.get("parameter_groups"))
    if isinstance(raw, Mapping):
        groups = {str(key): dict(value) for key, value in raw.items()}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        groups = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise FilmLrAnalysisError(f"{label} parameter_groups is malformed")
            name = str(item.get("group_name", item.get("name", ""))).strip()
            if not name or name in groups:
                raise FilmLrAnalysisError(
                    f"{label} group names are empty or duplicated"
                )
            groups[name] = dict(item)
    else:
        raise FilmLrAnalysisError(f"{label} lacks optimizer parameter groups")
    aliases = {
        "cnn_backbone": "backbone",
        "generator_backbone": "backbone",
        "text": "text_encoder",
        "generator_text": "text_encoder",
        "generator_text_encoder": "text_encoder",
        "global_film": "film",
        "generator_film": "film",
        "film_projection": "film",
        "film_projections": "film",
        "discriminator": "critic",
    }
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, row in groups.items():
        canonical = aliases.get(name, name)
        if canonical in normalized:
            raise FilmLrAnalysisError(
                f"{label} has duplicate canonical group {canonical}"
            )
        normalized[canonical] = row
    if set(normalized) != set(EXPECTED_GROUPS):
        raise FilmLrAnalysisError(
            f"{label} group universe drift: expected={sorted(EXPECTED_GROUPS)}, "
            f"actual={sorted(normalized)}"
        )
    return normalized


def _group_value(group: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    for name in names:
        if name in group:
            return group[name]
    raise FilmLrAnalysisError(f"{label} is missing {list(names)}")


def _normalize_trace(
    raw: Any,
    *,
    job_id: str,
    group_name: str,
    epochs_ran: int,
    initial_lr: float,
    final_lr: float,
) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FilmLrAnalysisError(
                f"{job_id}/{group_name} LR trace is not valid JSON"
            ) from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise FilmLrAnalysisError(f"{job_id}/{group_name} LR trace is empty")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            epoch = int(item.get("epoch", index))
            lr = _finite_positive(
                item.get("lr", item.get("learning_rate")),
                f"{job_id}/{group_name} trace LR",
            )
        else:
            epoch = index
            lr = _finite_positive(item, f"{job_id}/{group_name} trace LR")
        rows.append({"epoch": epoch, "learning_rate": lr})
    epochs = [row["epoch"] for row in rows]
    expected_start = 0 if epochs[0] == 0 else 1
    if epochs != list(range(expected_start, int(epochs_ran) + 1)):
        raise FilmLrAnalysisError(
            f"{job_id}/{group_name} LR trace must be contiguous through epochs_ran"
        )
    if not math.isclose(
        rows[0]["learning_rate"], initial_lr, rel_tol=0.0, abs_tol=1e-18
    ):
        raise FilmLrAnalysisError(f"{job_id}/{group_name} initial LR/trace mismatch")
    if not math.isclose(
        rows[-1]["learning_rate"], final_lr, rel_tol=0.0, abs_tol=1e-18
    ):
        raise FilmLrAnalysisError(f"{job_id}/{group_name} final LR/trace mismatch")
    return rows


def _load_optimizer_contract(
    row: Mapping[str, Any],
    *,
    external_inputs: dict[str, Mapping[str, Any]],
    native_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    job_id = str(row["job_id"])
    inline = row.get("optimizer_contract_json")
    if inline is not None and str(inline).strip():
        try:
            payload = json.loads(str(inline))
        except json.JSONDecodeError as exc:
            raise FilmLrAnalysisError(
                f"{job_id} optimizer_contract_json is invalid"
            ) from exc
        if not isinstance(payload, Mapping):
            raise FilmLrAnalysisError(f"{job_id} optimizer contract must be a mapping")
        return payload
    path_value = row.get("optimizer_contract_path")
    sha_value = row.get("optimizer_contract_sha256")
    if path_value is None or not str(path_value).strip() or sha_value is None:
        if native_contracts is not None and job_id in native_contracts:
            return native_contracts[job_id]
        raise FilmLrAnalysisError(
            f"{job_id} must provide optimizer_contract_json or a SHA-bound contract path"
        )
    path = Path(str(path_value)).resolve()
    expected_sha = _require_sha(sha_value, f"{job_id} optimizer contract SHA")
    if sha256_file(path) != expected_sha:
        raise FilmLrAnalysisError(f"{job_id} optimizer contract SHA drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FilmLrAnalysisError(f"{job_id} optimizer contract must be a mapping")
    external_inputs[job_id] = {
        "role": f"optimizer_contract:{job_id}",
        "path": str(path),
        "sha256": expected_sha,
        "size_bytes": path.stat().st_size,
    }
    return payload


def _record_sha_bound_input(
    inputs: dict[str, Mapping[str, Any]],
    *,
    role: str,
    path: str | Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    target = Path(path).resolve()
    if not target.is_file():
        raise FilmLrAnalysisError(f"Missing SHA-bound training artifact: {target}")
    observed_sha = sha256_file(target)
    if expected_sha256 is not None and observed_sha != _require_sha(
        expected_sha256, f"{role} SHA"
    ):
        raise FilmLrAnalysisError(f"SHA drift for {role}: {target}")
    if expected_size is not None and target.stat().st_size != int(expected_size):
        raise FilmLrAnalysisError(f"Size drift for {role}: {target}")
    key = f"{role}:{target}"
    inputs[key] = {
        "role": role,
        "path": str(target),
        "sha256": observed_sha,
        "size_bytes": target.stat().st_size,
    }
    return target


def _native_artifact(
    status: Mapping[str, Any],
    role: str,
    *,
    job_id: str,
    external_inputs: dict[str, Mapping[str, Any]],
) -> Path:
    matches = [
        row
        for row in list(status.get("artifacts") or [])
        if isinstance(row, Mapping) and str(row.get("artifact_role")) == role
    ]
    if len(matches) != 1:
        raise FilmLrAnalysisError(f"{job_id} must expose exactly one {role} artifact")
    artifact = matches[0]
    return _record_sha_bound_input(
        external_inputs,
        role=f"training_artifact:{job_id}:{role}",
        path=str(artifact.get("path", "")),
        expected_sha256=str(artifact.get("sha256", "")),
        expected_size=int(artifact.get("size_bytes", -1)),
    )


def _native_trace(
    rows: Any,
    *,
    job_id: str,
    group_name: str,
    value_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise FilmLrAnalysisError(
            f"{job_id}/{group_name} native checkpoint LR trace is missing"
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "epoch" not in row or value_key not in row:
            raise FilmLrAnalysisError(
                f"{job_id}/{group_name} native checkpoint LR trace is malformed"
            )
        result.append(
            {
                "epoch": int(row["epoch"]),
                "lr": _finite_positive(
                    row[value_key], f"{job_id}/{group_name} native checkpoint LR"
                ),
            }
        )
    return result


def _load_native_optimizer_contracts(
    training_summary_path: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    external_inputs: dict[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Read the real direct-run split-LR evidence through frozen manifests.

    The direct orchestrator intentionally keeps ``training_summary.csv``
    compact.  The actual group contract is emitted by ``WGAN_GP`` into the
    SHA-bound best-checkpoint JSON and epoch metrics artifacts referenced by
    each immutable job status.  This loader uses that native interface rather
    than inventing analysis-only columns that formal postprocess never writes.
    """

    if (
        training_summary_path.name != "training_summary.csv"
        or training_summary_path.parent.name != "analysis"
    ):
        raise FilmLrAnalysisError(
            "Native optimizer evidence requires <root>/analysis/training_summary.csv"
        )
    root = training_summary_path.parent.parent.resolve()
    registry_path = _record_sha_bound_input(
        external_inputs,
        role="training_task_registry",
        path=root / "registry/task_registry.json",
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping):
        raise FilmLrAnalysisError("Training task registry must be a mapping")
    jobs = {
        str(row.get("job_id")): row
        for row in list(registry.get("jobs") or [])
        if isinstance(row, Mapping)
    }
    expected_job_ids = {str(row["job_id"]) for row in summary_rows}
    if set(jobs) != expected_job_ids:
        raise FilmLrAnalysisError(
            "Native task-registry jobs differ from training_summary.csv"
        )

    model_path = _record_sha_bound_input(
        external_inputs,
        role="training_model_contract",
        path=root / "model_contract.json",
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, Mapping):
        raise FilmLrAnalysisError("Training model contract must be a mapping")
    if (
        int(model.get("generator_parameters", -1))
        != sum(
            count for name, (count, _lr) in EXPECTED_GROUPS.items() if name != "critic"
        )
        or int(model.get("critic_parameters", -1)) != EXPECTED_GROUPS["critic"][0]
    ):
        raise FilmLrAnalysisError("Training model/group parameter-count drift")

    contracts: dict[str, Mapping[str, Any]] = {}
    for summary_row in summary_rows:
        job_id = str(summary_row["job_id"])
        job = jobs[job_id]
        status_path = _record_sha_bound_input(
            external_inputs,
            role=f"training_job_status:{job_id}",
            path=root / "registry/job_status" / f"{job_id}.json",
        )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            not isinstance(status, Mapping)
            or status.get("status") != "completed"
            or str(status.get("job_id")) != job_id
            or str(status.get("job_spec_sha256")) != str(job.get("job_spec_sha256"))
            or str(status.get("training_config_sha256"))
            != str(job.get("training_config_sha256"))
        ):
            raise FilmLrAnalysisError(f"Native completed job status drift: {job_id}")

        best_path = _native_artifact(
            status,
            "best_learned_checkpoint",
            job_id=job_id,
            external_inputs=external_inputs,
        )
        metrics_path = _native_artifact(
            status,
            "training_metrics_csv",
            job_id=job_id,
            external_inputs=external_inputs,
        )
        resolved_path = _native_artifact(
            status,
            "resolved_training_config",
            job_id=job_id,
            external_inputs=external_inputs,
        )
        generator_path = _native_artifact(
            status,
            "generator_best_learned",
            job_id=job_id,
            external_inputs=external_inputs,
        )
        if sha256_file(generator_path) != str(summary_row["checkpoint_sha256"]):
            raise FilmLrAnalysisError(
                f"{job_id} summary checkpoint differs from native Generator artifact"
            )

        best = json.loads(best_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(metrics_path, low_memory=False)
        if not isinstance(best, Mapping) or not isinstance(config, Mapping):
            raise FilmLrAnalysisError(f"Malformed native training evidence: {job_id}")
        if int(best.get("best_epoch", -1)) != int(
            summary_row["best_epoch"]
        ) or not math.isclose(
            float(best.get("best_metric", math.nan)),
            float(summary_row["best_validation_score"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise FilmLrAnalysisError(f"Native best-checkpoint summary drift: {job_id}")
        if str(config.get("generator_optimizer_profile")) != "film_unet_split_lr_v1":
            raise FilmLrAnalysisError(f"{job_id} is not a split-LR Generator run")
        required_metric_columns = {
            "epoch",
            "g_lr_backbone",
            "g_lr_text_encoder",
            "g_lr_film_projection",
            "d_lr",
        }
        if not required_metric_columns.issubset(metrics.columns):
            raise FilmLrAnalysisError(
                f"{job_id} metrics lack split-LR columns: "
                f"{sorted(required_metric_columns - set(metrics.columns))}"
            )
        epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int).tolist()
        epochs_ran = int(summary_row["epochs_ran"])
        if epochs != list(range(0, epochs_ran + 1)):
            raise FilmLrAnalysisError(
                f"{job_id} native metrics epochs must be contiguous 0..epochs_ran"
            )

        initial_groups = best.get("generator_group_initial_learning_rates")
        group_trace = best.get("generator_group_lr_trace")
        if not isinstance(initial_groups, Mapping):
            raise FilmLrAnalysisError(
                f"{job_id} checkpoint lacks generator_group_initial_learning_rates"
            )
        configured = {
            "backbone": _finite_positive(
                config.get("generator_learning_rate"),
                f"{job_id}/backbone configured LR",
            ),
            "text_encoder": _finite_positive(
                config.get("generator_text_learning_rate"),
                f"{job_id}/text_encoder configured LR",
            ),
            "film": _finite_positive(
                config.get("generator_film_learning_rate"),
                f"{job_id}/film configured LR",
            ),
            "critic": _finite_positive(
                config.get("discriminator_learning_rate"),
                f"{job_id}/critic configured LR",
            ),
        }
        checkpoint_trace_by_group = {
            "backbone": _native_trace(
                group_trace,
                job_id=job_id,
                group_name="backbone",
                value_key="backbone",
            ),
            "text_encoder": _native_trace(
                group_trace,
                job_id=job_id,
                group_name="text_encoder",
                value_key="text_encoder",
            ),
            "film": _native_trace(
                group_trace,
                job_id=job_id,
                group_name="film",
                value_key="film_projection",
            ),
            "critic": _native_trace(
                best.get("discriminator_lr_trace"),
                job_id=job_id,
                group_name="critic",
                value_key="lr",
            ),
        }
        initial = {
            "backbone": _finite_positive(
                initial_groups.get("backbone"), f"{job_id}/backbone initial LR"
            ),
            "text_encoder": _finite_positive(
                initial_groups.get("text_encoder"),
                f"{job_id}/text_encoder initial LR",
            ),
            "film": _finite_positive(
                initial_groups.get("film_projection"), f"{job_id}/film initial LR"
            ),
            "critic": _finite_positive(
                best.get("discriminator_initial_learning_rate"),
                f"{job_id}/critic initial LR",
            ),
        }
        metric_columns = {
            "backbone": "g_lr_backbone",
            "text_encoder": "g_lr_text_encoder",
            "film": "g_lr_film_projection",
            "critic": "d_lr",
        }
        groups: dict[str, Mapping[str, Any]] = {}
        best_epoch = int(summary_row["best_epoch"])
        for group_name, (parameter_count, _fixed_lr) in EXPECTED_GROUPS.items():
            checkpoint_trace = checkpoint_trace_by_group[group_name]
            checkpoint_epochs = [int(row["epoch"]) for row in checkpoint_trace]
            if checkpoint_epochs != list(range(0, best_epoch + 1)):
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} checkpoint trace must end at best_epoch"
                )
            metric_values = pd.to_numeric(
                metrics[metric_columns[group_name]], errors="raise"
            ).to_numpy(float)
            checkpoint_values = np.asarray(
                [float(row["lr"]) for row in checkpoint_trace]
            )
            if not np.array_equal(metric_values[: best_epoch + 1], checkpoint_values):
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} checkpoint/metrics LR drift"
                )
            trace = [
                {"epoch": epoch, "lr": float(learning_rate)}
                for epoch, learning_rate in zip(epochs, metric_values, strict=True)
            ]
            groups[group_name] = {
                "parameter_count": parameter_count,
                "configured_learning_rate": configured[group_name],
                "initial_learning_rate": initial[group_name],
                "final_learning_rate": float(trace[-1]["lr"]),
                "lr_trace": trace,
            }
        contracts[job_id] = {
            "schema_version": 1,
            "source": "native_best_checkpoint_metrics_and_resolved_config_v1",
            "groups": groups,
        }
    return contracts


def validate_training_evidence(
    source: pd.DataFrame | str | Path,
    film_pair_metrics: pd.DataFrame,
    *,
    maximum_epochs: int = 240,
) -> TrainingEvidence:
    """Validate epochs, group counts, and complete per-group LR trajectories."""

    source_path = None if isinstance(source, pd.DataFrame) else Path(source).resolve()
    frame = _read_frame(source, "FiLM-LR training summary")
    required = {
        "job_id",
        "fold",
        "arm",
        "best_epoch",
        "epochs_ran",
        "best_validation_score",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise FilmLrAnalysisError(
            f"Training summary is empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise FilmLrAnalysisError(f"Training summary {column} must be non-empty")
    if len(result) != EXPECTED_FILM_JOBS or result["job_id"].duplicated().any():
        raise FilmLrAnalysisError("Training summary must contain 20 unique FiLM jobs")
    expected_cells = {(fold, arm) for fold in FOLDS for arm in FILM_LR_ARMS}
    observed_cells = set(result[["fold", "arm"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise FilmLrAnalysisError("Training summary fold/arm universe drift")
    for column in ("best_epoch", "epochs_ran"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise FilmLrAnalysisError(f"Training summary {column} must be integer")
        result[column] = numeric.astype(int)
    if (
        (result["best_epoch"] < 1).any()
        or (result["epochs_ran"] < result["best_epoch"]).any()
        or (result["epochs_ran"] > int(maximum_epochs)).any()
    ):
        raise FilmLrAnalysisError(
            "Training epochs must satisfy 1 <= best_epoch <= epochs_ran <= maximum"
        )
    scores = pd.to_numeric(result["best_validation_score"], errors="coerce").astype(
        float
    )
    if not np.isfinite(scores.to_numpy()).all():
        raise FilmLrAnalysisError("best_validation_score must be finite")
    result["best_validation_score"] = scores
    result["checkpoint_sha256"] = [
        _require_sha(value, "training checkpoint_sha256")
        for value in result["checkpoint_sha256"]
    ]
    pair_jobs = (
        film_pair_metrics[["job_id", "fold", "arm", "checkpoint_sha256"]]
        .drop_duplicates()
        .sort_values("job_id", kind="stable")
        .reset_index(drop=True)
    )
    declared = (
        result[["job_id", "fold", "arm", "checkpoint_sha256"]]
        .sort_values("job_id", kind="stable")
        .reset_index(drop=True)
    )
    if not declared.equals(pair_jobs):
        raise FilmLrAnalysisError(
            "Training summary jobs/checkpoint hashes differ from pair metrics"
        )

    external_inputs: dict[str, Mapping[str, Any]] = {}
    native_contracts: Mapping[str, Mapping[str, Any]] | None = None
    has_declared_contract = "optimizer_contract_json" in result.columns or {
        "optimizer_contract_path",
        "optimizer_contract_sha256",
    }.issubset(result.columns)
    if not has_declared_contract and source_path is not None:
        native_contracts = _load_native_optimizer_contracts(
            source_path,
            result.to_dict(orient="records"),
            external_inputs=external_inputs,
        )
    diagnostic_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        job_id = str(row["job_id"])
        arm = str(row["arm"])
        contract = _load_optimizer_contract(
            row,
            external_inputs=external_inputs,
            native_contracts=native_contracts,
        )
        groups = _contract_groups(contract, f"{job_id} optimizer contract")
        group_values: dict[str, dict[str, Any]] = {}
        for group_name, (expected_count, fixed_lr) in EXPECTED_GROUPS.items():
            group = groups[group_name]
            count = int(
                _group_value(
                    group,
                    ("parameter_count", "parameters", "num_parameters"),
                    f"{job_id}/{group_name} parameter count",
                )
            )
            if count != expected_count:
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} parameter count drift: {count} != {expected_count}"
                )
            configured = _finite_positive(
                _group_value(
                    group,
                    ("configured_learning_rate", "configured_lr", "target_lr"),
                    f"{job_id}/{group_name} configured LR",
                ),
                f"{job_id}/{group_name} configured LR",
            )
            initial = _finite_positive(
                _group_value(
                    group,
                    ("initial_learning_rate", "initial_lr"),
                    f"{job_id}/{group_name} initial LR",
                ),
                f"{job_id}/{group_name} initial LR",
            )
            final = _finite_positive(
                _group_value(
                    group,
                    ("final_learning_rate", "final_lr"),
                    f"{job_id}/{group_name} final LR",
                ),
                f"{job_id}/{group_name} final LR",
            )
            expected_lr = FILM_LR_ARMS[arm] if group_name == "film" else fixed_lr
            assert expected_lr is not None
            if not math.isclose(configured, expected_lr, rel_tol=0.0, abs_tol=1e-18):
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} configured LR drift: {configured} != {expected_lr}"
                )
            if not math.isclose(initial, configured, rel_tol=0.0, abs_tol=1e-18):
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} configured/initial LR mismatch"
                )
            trace = _normalize_trace(
                _group_value(
                    group,
                    ("lr_trace", "learning_rate_trace"),
                    f"{job_id}/{group_name} LR trace",
                ),
                job_id=job_id,
                group_name=group_name,
                epochs_ran=int(row["epochs_ran"]),
                initial_lr=initial,
                final_lr=final,
            )
            group_values[group_name] = {
                "count": count,
                "configured": configured,
                "initial": initial,
                "final": final,
            }
            trace_rows.extend(
                {
                    "job_id": job_id,
                    "fold": str(row["fold"]),
                    "arm": arm,
                    "film_learning_rate": FILM_LR_ARMS[arm],
                    "parameter_group": group_name,
                    **trace_row,
                }
                for trace_row in trace
            )
        diagnostic_rows.append(
            {
                "job_id": job_id,
                "fold": str(row["fold"]),
                "arm": arm,
                "film_learning_rate": FILM_LR_ARMS[arm],
                "best_epoch": int(row["best_epoch"]),
                "epochs_ran": int(row["epochs_ran"]),
                "best_validation_score": float(row["best_validation_score"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                **{
                    f"{name}_{metric}": values[metric]
                    for name, values in group_values.items()
                    for metric in ("count", "configured", "initial", "final")
                },
            }
        )
    diagnostics = (
        pd.DataFrame(diagnostic_rows)
        .sort_values(["film_learning_rate", "fold"], kind="stable")
        .reset_index(drop=True)
    )
    traces = (
        pd.DataFrame(trace_rows)
        .sort_values(
            ["film_learning_rate", "fold", "parameter_group", "epoch"], kind="stable"
        )
        .reset_index(drop=True)
    )
    return TrainingEvidence(
        diagnostics=diagnostics,
        lr_trace=traces,
        external_inputs=tuple(external_inputs[key] for key in sorted(external_inputs)),
    )


def _build_ranking(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = compute_fold_summary(combined)
    arms = compute_arm_summary(combined, folds)
    pure_mae = float(arms.loc[arms["arm"].eq(PURE_ARM), "equal_fold_mae"].iloc[0])
    ranking = arms[arms["arm"].isin(FILM_LR_ARMS)].copy()
    ranking["film_learning_rate"] = ranking["arm"].map(FILM_LR_ARMS).astype(float)
    ranking["improvement_vs_pure_percent"] = 100.0 * (
        1.0 - ranking["equal_fold_mae"] / pure_mae
    )
    ranking["film_lr_rank"] = (
        ranking["equal_fold_mae"].rank(method="min", ascending=True).astype(int)
    )
    ranking["point_leader_only"] = ranking["film_lr_rank"].eq(1)
    ranking["selection_permitted"] = False
    ranking["interpretation"] = INTERPRETATION
    return (
        folds.sort_values(["fold", "mean_mae", "arm"], kind="stable").reset_index(
            drop=True
        ),
        ranking.sort_values(
            ["film_lr_rank", "film_learning_rate"], kind="stable"
        ).reset_index(drop=True),
    )


def _build_comparisons(
    combined: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted(FILM_LR_ARMS, key=FILM_LR_ARMS.get)
    for index, arm in enumerate(ordered):
        stats = _wrap_direct(
            f"bootstrap failed for {arm} vs {PURE_ARM}",
            fold_session_paired_bootstrap,
            combined,
            focal_arm=arm,
            reference_arm=PURE_ARM,
            expected_folds=expected_folds,
            iterations=int(iterations),
            rng_seed=int(rng_seed) + index,
        )
        stats.update(
            {
                "comparison_id": f"{arm}_vs_{PURE_ARM}",
                "film_learning_rate": FILM_LR_ARMS[arm],
                "multiplicity_family": "film_lr_vs_pure_holm5",
                "confirmatory": False,
                "inference_permitted": False,
                "interpretation": INTERPRETATION,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    adjusted = holm_adjust(
        dict(
            zip(
                result["comparison_id"].astype(str),
                result["p_value_one_sided"].astype(float),
                strict=True,
            )
        )
    )
    result["holm_adjusted_p"] = result["comparison_id"].map(adjusted)
    result["holm_family_size"] = 5
    result["direction_favors_film"] = result["mean_log_mae_ratio"].lt(0.0)
    result["ci_excludes_zero_favoring_film"] = result["ci_95_upper"].lt(0.0)
    result["descriptive_support_gate"] = (
        result["direction_favors_film"]
        & result["ci_excludes_zero_favoring_film"]
        & result["holm_adjusted_p"].lt(0.05)
        & result["focal_nonworse_fold_count"].ge(3)
    )
    return result.sort_values("film_learning_rate", kind="stable").reset_index(
        drop=True
    )


def _build_pairwise_comparisons(
    combined: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted(FILM_LR_ARMS, key=FILM_LR_ARMS.get)
    adjacent_pairs = set(zip(ordered, ordered[1:]))
    for index, (lower, higher) in enumerate(combinations(ordered, 2)):
        stats = _wrap_direct(
            f"bootstrap failed for LR pair {higher} vs {lower}",
            fold_session_paired_bootstrap,
            combined,
            focal_arm=higher,
            reference_arm=lower,
            expected_folds=expected_folds,
            iterations=int(iterations),
            rng_seed=int(rng_seed) + index,
        )
        stats.update(
            {
                "comparison_id": f"{higher}_vs_{lower}",
                "lower_film_learning_rate": FILM_LR_ARMS[lower],
                "higher_film_learning_rate": FILM_LR_ARMS[higher],
                "adjacent_learning_rates": (lower, higher) in adjacent_pairs,
                "multiplicity_family": "all_film_lr_pairs_holm10",
                "confirmatory": False,
                "inference_permitted": False,
                "interpretation": INTERPRETATION,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    if len(result) != math.comb(len(FILM_LR_ARMS), 2):
        raise FilmLrAnalysisError("Internal FiLM-LR pairwise family is incomplete")
    adjusted = holm_adjust(
        dict(
            zip(
                result["comparison_id"].astype(str),
                result["p_value_one_sided"].astype(float),
                strict=True,
            )
        )
    )
    result["holm_adjusted_p"] = result["comparison_id"].map(adjusted)
    result["holm_family_size"] = len(result)
    result["direction_favors_higher_lr"] = result["mean_log_mae_ratio"].lt(0.0)
    result["ci_excludes_zero_favoring_higher_lr"] = result["ci_95_upper"].lt(0.0)
    return result.sort_values(
        ["lower_film_learning_rate", "higher_film_learning_rate"], kind="stable"
    ).reset_index(drop=True)


def analyze_film_lr(
    film_pair_metrics: pd.DataFrame | str | Path,
    pure_pair_metrics: pd.DataFrame | str | Path,
    training_summary: pd.DataFrame | str | Path,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_film_rows: int | None = EXPECTED_FILM_ROWS,
    expected_pure_rows: int | None = EXPECTED_PURE_ROWS,
) -> FilmLrAnalysis:
    film = validate_film_pair_metrics(
        film_pair_metrics,
        expected_fold_counts=expected_fold_counts,
        expected_row_count=expected_film_rows,
    )
    pure = validate_pure_pair_metrics(
        pure_pair_metrics,
        expected_fold_counts=expected_fold_counts,
        expected_row_count=expected_pure_rows,
    )
    validate_cross_experiment_lineage(film, pure)
    training = validate_training_evidence(training_summary, film)
    combined = pd.concat([film, pure], ignore_index=True, sort=False)
    folds, ranking = _build_ranking(combined)
    comparisons = _build_comparisons(
        combined,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed),
    )
    pairwise = _build_pairwise_comparisons(
        combined,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed) + 10_000,
    )
    leader = ranking.sort_values(
        ["equal_fold_mae", "film_learning_rate"], kind="stable"
    ).iloc[0]
    leader_comparison = comparisons.loc[
        comparisons["arm" if "arm" in comparisons.columns else "focal_arm"].eq(
            str(leader["arm"])
        )
    ].iloc[0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "film_unet_direct_matched_lr_seed42_vs_frozen_pure_cnn",
        "interpretation": INTERPRETATION,
        "claim_scope": "single_seed_retrospective_development_descriptive_only",
        "confirmatory": False,
        "inference_permitted": False,
        "test_based_lr_selection_permitted": False,
        "point_leader_is_model_selection": False,
        "point_leader_arm": str(leader["arm"]),
        "point_leader_film_learning_rate": float(leader["film_learning_rate"]),
        "point_leader_equal_fold_mae": float(leader["equal_fold_mae"]),
        "point_leader_vs_pure_mean_log_mae_ratio": float(
            leader_comparison["mean_log_mae_ratio"]
        ),
        "point_leader_vs_pure_ci_95_lower": float(leader_comparison["ci_95_lower"]),
        "point_leader_vs_pure_ci_95_upper": float(leader_comparison["ci_95_upper"]),
        "point_leader_vs_pure_holm_adjusted_p": float(
            leader_comparison["holm_adjusted_p"]
        ),
        "film_lr_values": sorted(FILM_LR_ARMS.values()),
        "seed": SEED,
        "folds": list(expected_fold_counts),
        "tolerance_minutes": TOLERANCE_MINUTES,
        "film_job_count": int(film["job_id"].nunique()),
        "film_pair_metric_rows": int(len(film)),
        "pure_pair_metric_rows": int(len(pure)),
        "paired_panel_rows_per_arm": int(len(pure)),
        "fold_session_count_per_arm": int(
            pure[["fold", "session_id"]].drop_duplicates().shape[0]
        ),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "holm_family_size": 5,
        "pairwise_holm_family_size": 10,
        "capacity_matched": False,
        "training_protocol_equivalence_to_frozen_pure_not_established": True,
        "pure_comparator_role": "frozen_contextual_architecture_and_capacity_reference",
        "lineage_contract": (
            "exact_fold_pair_session_origin_persistence_and_mc_noise_bank_v1"
        ),
        "training_diagnostics_complete": True,
    }
    return FilmLrAnalysis(
        film_pair_metrics=film,
        pure_pair_metrics=pure,
        combined_pair_metrics=combined,
        fold_summary=folds,
        ranking=ranking,
        comparisons=comparisons,
        pairwise_comparisons=pairwise,
        training_diagnostics=training.diagnostics,
        lr_trace=training.lr_trace,
        training_external_inputs=training.external_inputs,
        summary=summary,
    )


def _format(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.10g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    headings = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_format(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_reports(analysis: FilmLrAnalysis) -> tuple[str, str]:
    rank_columns = (
        "film_lr_rank",
        "arm",
        "film_learning_rate",
        "equal_fold_mae",
        "improvement_vs_pure_percent",
        "point_leader_only",
        "selection_permitted",
    )
    comparison_columns = (
        "film_learning_rate",
        "mean_log_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "focal_nonworse_fold_count",
        "descriptive_support_gate",
    )
    pairwise_columns = (
        "lower_film_learning_rate",
        "higher_film_learning_rate",
        "adjacent_learning_rates",
        "mean_log_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
    )
    training_columns = (
        "arm",
        "fold",
        "best_epoch",
        "epochs_ran",
        "backbone_initial",
        "text_encoder_initial",
        "film_initial",
        "film_final",
        "critic_final",
    )
    summary = analysis.summary
    caveat = (
        "这些结果只属于 single-seed retrospective development。点估计第一名不是已冻结的学习率选择；"
        "不得根据这些 test 结果选择 LR。Pure CNN 是冻结的架构/容量参照，且训练协议等价性未建立。"
    )
    markdown = f"""# FiLM projection LR × Pure CNN 结果

## 结论

点估计第一名是 `{summary["point_leader_arm"]}`（FiLM LR `{_format(summary["point_leader_film_learning_rate"])}`），equal-fold MAE 为 `{_format(summary["point_leader_equal_fold_mae"])}`。该名称只描述当前冻结结果，不构成模型或学习率选择。

{caveat}

## LR 排名

{_markdown_table(analysis.ranking, rank_columns)}

## 每个 LR 与 Pure CNN 的配对比较（Holm-5）

负的 log-MAE ratio 表示 FiLM 更好。Bootstrap 为 fold → paired CME-session，共 `{summary["bootstrap_iterations"]}` 次。

{_markdown_table(analysis.comparisons, comparison_columns)}

## LR 内部全配对比较（Holm-10）

五个 LR 的全部 `5 choose 2 = 10` 个比较使用同一个 Holm-10 family；这些结果仍只作描述，不用于选择 LR。

{_markdown_table(analysis.pairwise_comparisons, pairwise_columns)}

## 训练诊断

{_markdown_table(analysis.training_diagnostics, training_columns)}

完整逐轮、逐参数组 LR 轨迹见 `film_lr_group_lr_trace.csv`。

## 数据与解释边界

- FiLM：5 LR × 4 folds，{summary["film_pair_metric_rows"]} pair rows。
- Pure CNN：{summary["pure_pair_metric_rows"]} pair rows。
- fold、pair、session、origin、persistence 和 MC64 noise-bank SHA 已逐项一致。
- `confirmatory=false`，`inference_permitted=false`，`test_based_lr_selection_permitted=false`。
"""
    html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FiLM projection LR × Pure CNN</title>
<style>
body{{margin:0;background:#f4f6f8;color:#17202a;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dfe4ea;border-radius:10px;padding:18px;margin:14px 0;overflow:auto}}h1,h2{{margin-top:0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid #e7ebef;padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.warning{{border-left:5px solid #b45309;background:#fff8eb}}
</style></head><body><main>
<h1>FiLM projection LR × Pure CNN</h1>
<section><h2>结论</h2><p>点估计第一名：<code>{html.escape(str(summary["point_leader_arm"]))}</code>，FiLM LR <code>{_format(summary["point_leader_film_learning_rate"])}</code>，equal-fold MAE <code>{_format(summary["point_leader_equal_fold_mae"])}</code>。这只是描述性 point leader。</p></section>
<section class="warning"><h2>解释边界</h2><p>{html.escape(caveat)}</p></section>
<section><h2>LR 排名</h2>{_html_table(analysis.ranking, rank_columns)}</section>
<section><h2>FiLM vs Pure CNN（Holm-5）</h2>{_html_table(analysis.comparisons, comparison_columns)}</section>
<section><h2>LR 内部全配对比较（Holm-10）</h2>{_html_table(analysis.pairwise_comparisons, pairwise_columns)}<p>全 10 个比较共享同一个 Holm family；不用于 test-based LR selection。</p></section>
<section><h2>训练诊断</h2>{_html_table(analysis.training_diagnostics, training_columns)}<p>完整轨迹位于 <code>film_lr_group_lr_trace.csv</code>。</p></section>
<section><h2>数据合同</h2><p>FiLM {summary["film_pair_metric_rows"]} 行；Pure CNN {summary["pure_pair_metric_rows"]} 行；fold/pair/session/origin/persistence/MC noise 均通过严格配对。</p></section>
</main></body></html>"""
    return markdown, html_report


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
        "utf-8"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FilmLrAnalysisError(f"Existing analysis output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_analysis_bundle(
    analysis: FilmLrAnalysis,
    *,
    film_pair_metrics_path: str | Path,
    film_pair_metrics_sha256: str,
    pure_pair_metrics_path: str | Path,
    pure_pair_metrics_sha256: str,
    training_summary_path: str | Path,
    training_summary_sha256: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    film_source = Path(film_pair_metrics_path).resolve()
    pure_source = Path(pure_pair_metrics_path).resolve()
    training_source = Path(training_summary_path).resolve()
    expected_inputs = {
        "film_pair_metrics_source": (
            film_source,
            _require_sha(film_pair_metrics_sha256, "FiLM pair metrics SHA"),
        ),
        "pure_pair_metrics_source": (
            pure_source,
            _require_sha(pure_pair_metrics_sha256, "Pure pair metrics SHA"),
        ),
        "film_training_summary_source": (
            training_source,
            _require_sha(training_summary_sha256, "training summary SHA"),
        ),
    }
    for role, (path, expected_sha) in expected_inputs.items():
        if sha256_file(path) != expected_sha:
            raise FilmLrAnalysisError(f"{role} SHA-256 drift before write")
    destination = Path(output_dir)
    paths = {
        "ranking": destination / "film_lr_ranking.csv",
        "fold_summary": destination / "film_lr_fold_summary.csv",
        "comparisons": destination / "film_lr_vs_pure_bootstrap_holm.csv",
        "pairwise": destination / "film_lr_pairwise_bootstrap_holm.csv",
        "training_diagnostics": destination / "film_lr_training_diagnostics.csv",
        "lr_trace": destination / "film_lr_group_lr_trace.csv",
        "summary": destination / "film_lr_analysis_summary.json",
        "report_markdown": destination / "film_lr_report.md",
        "report_html": destination / "film_lr_report.html",
        "manifest": destination / "film_lr_analysis_manifest.json",
    }
    markdown, html_report = render_reports(analysis)
    frames = {
        "ranking": analysis.ranking,
        "fold_summary": analysis.fold_summary,
        "comparisons": analysis.comparisons,
        "pairwise": analysis.pairwise_comparisons,
        "training_diagnostics": analysis.training_diagnostics,
        "lr_trace": analysis.lr_trace,
    }
    for role, frame in frames.items():
        _atomic_write(paths[role], _csv_bytes(frame))
    summary = dict(analysis.summary)
    summary.update(
        {
            f"{role}_path": str(path)
            for role, (path, _expected_sha) in expected_inputs.items()
        }
    )
    summary.update(
        {
            f"{role}_sha256": expected_sha
            for role, (_path, expected_sha) in expected_inputs.items()
        }
    )
    _atomic_write(paths["summary"], _canonical_json(summary))
    _atomic_write(paths["report_markdown"], markdown.encode("utf-8"))
    _atomic_write(paths["report_html"], html_report.encode("utf-8"))
    artifacts = []
    for role in (
        "ranking",
        "fold_summary",
        "comparisons",
        "pairwise",
        "training_diagnostics",
        "lr_trace",
        "summary",
        "report_markdown",
        "report_html",
    ):
        path = paths[role].resolve()
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    inputs = [
        {
            "role": role,
            "path": str(path),
            "sha256": expected_sha,
            "size_bytes": path.stat().st_size,
        }
        for role, (path, expected_sha) in expected_inputs.items()
    ]
    inputs.extend(dict(row) for row in analysis.training_external_inputs)
    for row in inputs:
        if sha256_file(row["path"]) != row["sha256"]:
            raise FilmLrAnalysisError(f"Input changed while writing: {row['role']}")
    manifest = {
        "schema_version": 1,
        "kind": "film_unet_direct_matched_lr_seed42_analysis_manifest_v1",
        "audience": "technical",
        "interpretation": INTERPRETATION,
        "confirmatory": False,
        "inference_permitted": False,
        "test_based_lr_selection_permitted": False,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    _atomic_write(paths["manifest"], _canonical_json(manifest))
    return paths


def run_film_lr_analysis(
    *,
    film_pair_metrics_path: str | Path,
    film_pair_metrics_sha256: str,
    pure_pair_metrics_path: str | Path,
    pure_pair_metrics_sha256: str,
    training_summary_path: str | Path,
    training_summary_sha256: str,
    output_dir: str | Path,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Path]:
    sources = (
        (film_pair_metrics_path, film_pair_metrics_sha256, "FiLM pair metrics"),
        (pure_pair_metrics_path, pure_pair_metrics_sha256, "Pure pair metrics"),
        (training_summary_path, training_summary_sha256, "training summary"),
    )
    for path, digest, label in sources:
        if sha256_file(path) != _require_sha(digest, f"{label} SHA"):
            raise FilmLrAnalysisError(f"{label} SHA-256 drift")
    analysis = analyze_film_lr(
        film_pair_metrics_path,
        pure_pair_metrics_path,
        training_summary_path,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    return write_analysis_bundle(
        analysis,
        film_pair_metrics_path=film_pair_metrics_path,
        film_pair_metrics_sha256=film_pair_metrics_sha256,
        pure_pair_metrics_path=pure_pair_metrics_path,
        pure_pair_metrics_sha256=pure_pair_metrics_sha256,
        training_summary_path=training_summary_path,
        training_summary_sha256=training_summary_sha256,
        output_dir=output_dir,
    )


def analyze_experiment(output_root: Path, config: Mapping[str, Any]) -> Path:
    """Orchestrator adapter using the canonical new-root paths."""

    analysis_config = config.get("analysis")
    if not isinstance(analysis_config, Mapping):
        raise FilmLrAnalysisError("config.analysis must be a mapping")
    reference = analysis_config.get("frozen_pure_cnn")
    if not isinstance(reference, Mapping):
        raise FilmLrAnalysisError("analysis.frozen_pure_cnn must be a mapping")
    pure_path = reference.get("pair_metrics_path")
    pure_sha = reference.get("pair_metrics_sha256")
    if not pure_path or not pure_sha:
        raise FilmLrAnalysisError("Frozen Pure-CNN path/SHA are required")
    iterations = int(analysis_config.get("bootstrap_replicates", BOOTSTRAP_ITERATIONS))
    seed = int(analysis_config.get("bootstrap_seed", BOOTSTRAP_SEED))
    root = Path(output_root).resolve()
    film_path = root / "analysis/rq12_pair_metrics.csv.gz"
    training_path = root / "analysis/training_summary.csv"
    paths = run_film_lr_analysis(
        film_pair_metrics_path=film_path,
        film_pair_metrics_sha256=sha256_file(film_path),
        pure_pair_metrics_path=(REPO_ROOT / Path(str(pure_path))).resolve()
        if not Path(str(pure_path)).is_absolute()
        else Path(str(pure_path)).resolve(),
        pure_pair_metrics_sha256=str(pure_sha),
        training_summary_path=training_path,
        training_summary_sha256=sha256_file(training_path),
        output_dir=root / "analysis",
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
    )
    return paths["manifest"]


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "EXPECTED_FOLD_PAIR_SESSION_COUNTS",
    "FILM_LR_ARMS",
    "FilmLrAnalysis",
    "FilmLrAnalysisError",
    "PURE_ARM",
    "PURE_PAIR_METRICS_SHA256",
    "analyze_experiment",
    "analyze_film_lr",
    "render_reports",
    "run_film_lr_analysis",
    "sha256_file",
    "validate_cross_experiment_lineage",
    "validate_film_pair_metrics",
    "validate_pure_pair_metrics",
    "validate_training_evidence",
    "write_analysis_bundle",
]
