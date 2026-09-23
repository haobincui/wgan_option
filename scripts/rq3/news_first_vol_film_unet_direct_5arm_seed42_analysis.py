"""Deterministic analysis for the direct-training FiLM U-Net five-arm study.

The module deliberately contains no experiment discovery and no training code.
The orchestrator must pass the frozen pair-metric path and its SHA-256 digest.
The public entry point validates the complete 20-cell/2,500-row evidence matrix,
runs the predeclared single-seed analyses, and writes a fail-closed bundle of
CSV, JSON, Markdown, and self-contained HTML artifacts.

The statistical estimand for every comparison is the equally weighted mean of
the four fold-level ``log(mean(MAE_focal) / mean(MAE_reference))`` values.
Bootstrap draws first resample folds and then resample paired CME-session
clusters within every sampled fold.  Negative values favour the focal arm.
Because only seed 42 is present, all outputs are explicitly descriptive and
must not be presented as evidence of stability across random initialisations.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
    holm_adjust as _audited_holm_adjust,
)


DIRECT_ARMS = (
    "lp_matched",
    "lp_shuffle",
    "no_text",
    "bow",
    "sentiment",
)
DIRECT_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
DIRECT_SEED = 42
DIRECT_TOLERANCE_MINUTES = 5
EXPECTED_FOLD_PAIR_SESSION_COUNTS: Mapping[str, tuple[int, int]] = {
    "f1_2023q1": (110, 34),
    "f2_2023q2": (112, 36),
    "f3_2023q3": (135, 33),
    "f4_2023q4": (143, 45),
}
EXPECTED_JOB_COUNT = 20
EXPECTED_PAIR_COUNT = 500
EXPECTED_SESSION_COUNT = 148
EXPECTED_PAIR_METRIC_ROWS = 2_500
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260830
INTERPRETATION_LABEL = "retrospective_rolling_development_single_seed_descriptive"

DIRECT_CONTRASTS = (
    {
        "family": "rq1_holm2",
        "research_question": "RQ1",
        "contrast_id": "lp_matched_minus_no_text",
        "focal_arm": "lp_matched",
        "reference_arm": "no_text",
    },
    {
        "family": "rq1_holm2",
        "research_question": "RQ1",
        "contrast_id": "lp_matched_minus_lp_shuffle",
        "focal_arm": "lp_matched",
        "reference_arm": "lp_shuffle",
    },
    {
        "family": "representation_holm2",
        "research_question": "RQ2",
        "contrast_id": "lp_matched_minus_bow",
        "focal_arm": "lp_matched",
        "reference_arm": "bow",
    },
    {
        "family": "representation_holm2",
        "research_question": "RQ2",
        "contrast_id": "lp_matched_minus_sentiment",
        "focal_arm": "lp_matched",
        "reference_arm": "sentiment",
    },
)

REQUIRED_PAIR_METRIC_COLUMNS = (
    "job_id",
    "tolerance_minutes",
    "fold",
    "seed",
    "arm",
    "pair_id",
    "session_id",
    "effective_origin_utc",
    "target_mae",
    "persistence_mae",
    "checkpoint_sha256",
    "prediction_sha256",
)


class DirectFiveArmAnalysisError(ValueError):
    """Raised when frozen evidence or the analysis contract has drifted."""


@dataclass(frozen=True)
class DirectFiveArmAnalysis:
    """All deterministic in-memory tables produced from frozen pair evidence."""

    pair_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    arm_summary: pd.DataFrame
    contrasts: pd.DataFrame
    persistence: pd.DataFrame
    training_summary: pd.DataFrame | None
    summary: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    source = Path(path)
    if not source.is_file():
        raise DirectFiveArmAnalysisError(f"Required file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DirectFiveArmAnalysisError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return digest


def _read_pair_metrics(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise DirectFiveArmAnalysisError(f"Pair metrics do not exist: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - pandas preserves the useful cause
        raise DirectFiveArmAnalysisError(
            f"Could not read pair metrics {path}: {exc}"
        ) from exc


def validate_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_arms: Sequence[str] = DIRECT_ARMS,
    expected_seed: int = DIRECT_SEED,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_tolerance_minutes: int = DIRECT_TOLERANCE_MINUTES,
    expected_row_count: int | None = EXPECTED_PAIR_METRIC_ROWS,
) -> pd.DataFrame:
    """Validate and canonically sort one frozen direct-training evidence table.

    ``expected_fold_counts`` maps each fold to ``(pair_count, session_count)``.
    Tests may supply a smaller explicit universe; the production entry point
    always uses the canonical four-fold counts and exactly 2,500 rows.
    """

    frame = _read_pair_metrics(source)
    missing = sorted(set(REQUIRED_PAIR_METRIC_COLUMNS) - set(frame.columns))
    if frame.empty or missing:
        raise DirectFiveArmAnalysisError(
            f"Pair metrics are empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm", "pair_id", "session_id"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise DirectFiveArmAnalysisError(f"Pair metrics {column} must be non-empty")
    for column in ("seed", "tolerance_minutes"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise DirectFiveArmAnalysisError(
                f"Pair metrics {column} must contain integers"
            )
        result[column] = numeric.astype(int)
    for column in ("target_mae", "persistence_mae"):
        numeric = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0.0).any():
            raise DirectFiveArmAnalysisError(
                f"Pair metrics {column} must be finite and nonnegative"
            )
        result[column] = numeric
    if (result["persistence_mae"] <= 0.0).any():
        raise DirectFiveArmAnalysisError("persistence_mae must be strictly positive")
    origins = pd.to_datetime(result["effective_origin_utc"], utc=True, errors="coerce")
    if origins.isna().any():
        raise DirectFiveArmAnalysisError(
            "effective_origin_utc contains invalid UTC timestamps"
        )
    result["effective_origin_utc"] = origins.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for column in ("checkpoint_sha256", "prediction_sha256"):
        result[column] = [
            _require_sha(value, f"pair metrics {column}") for value in result[column]
        ]

    arms = tuple(map(str, expected_arms))
    folds = tuple(map(str, expected_fold_counts))
    if not arms or len(arms) != len(set(arms)):
        raise DirectFiveArmAnalysisError("expected_arms must be non-empty and unique")
    if not folds or len(folds) != len(set(folds)):
        raise DirectFiveArmAnalysisError(
            "expected_fold_counts must define unique, non-empty folds"
        )
    expected_universes = (
        ("arm", set(arms), set(result["arm"])),
        ("fold", set(folds), set(result["fold"])),
        ("seed", {int(expected_seed)}, set(result["seed"])),
        (
            "tolerance_minutes",
            {int(expected_tolerance_minutes)},
            set(result["tolerance_minutes"]),
        ),
    )
    for label, expected, observed in expected_universes:
        if observed != expected:
            raise DirectFiveArmAnalysisError(
                f"Pair metrics {label} universe drift: "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
    if expected_row_count is not None and len(result) != int(expected_row_count):
        raise DirectFiveArmAnalysisError(
            f"Pair metric row count drift: expected={expected_row_count}, "
            f"actual={len(result)}"
        )
    if result.duplicated(["job_id", "pair_id"]).any():
        raise DirectFiveArmAnalysisError(
            "Pair metrics must contain one row per job_id/pair_id"
        )

    job_cells = result[
        ["job_id", "arm", "fold", "seed", "tolerance_minutes"]
    ].drop_duplicates()
    if job_cells["job_id"].duplicated().any():
        raise DirectFiveArmAnalysisError(
            "One job_id maps to more than one experiment cell"
        )
    expected_cells = {(arm, fold) for arm in arms for fold in folds}
    observed_cells = set(job_cells[["arm", "fold"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells or len(job_cells) != len(expected_cells):
        raise DirectFiveArmAnalysisError(
            "Direct-training job matrix drift: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    for job_id, group in result.groupby("job_id", sort=True):
        for column in ("checkpoint_sha256", "prediction_sha256"):
            if group[column].nunique(dropna=False) != 1:
                raise DirectFiveArmAnalysisError(
                    f"{job_id} contains multiple {column} values"
                )

    lineage_columns = (
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
    )
    for fold, (expected_pairs, expected_sessions) in expected_fold_counts.items():
        fold_frame = result[result["fold"].eq(str(fold))]
        reference: pd.DataFrame | None = None
        for arm in arms:
            panel = (
                fold_frame[fold_frame["arm"].eq(arm)][list(lineage_columns)]
                .sort_values("pair_id", kind="stable")
                .reset_index(drop=True)
            )
            if len(panel) != int(expected_pairs):
                raise DirectFiveArmAnalysisError(
                    f"Pair count drift for fold={fold}, arm={arm}: "
                    f"expected={expected_pairs}, actual={len(panel)}"
                )
            if panel["pair_id"].nunique() != int(expected_pairs):
                raise DirectFiveArmAnalysisError(
                    f"Duplicate pair IDs for fold={fold}, arm={arm}"
                )
            if panel["session_id"].nunique() != int(expected_sessions):
                raise DirectFiveArmAnalysisError(
                    f"Session count drift for fold={fold}, arm={arm}: "
                    f"expected={expected_sessions}, "
                    f"actual={panel['session_id'].nunique()}"
                )
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise DirectFiveArmAnalysisError(
                    "Pair/session/time/persistence universe differs across arms "
                    f"for fold={fold}"
                )
    return result.sort_values(["fold", "arm", "pair_id"], kind="stable").reset_index(
        drop=True
    )


def validate_training_summary(
    source: pd.DataFrame | str | Path,
    pair_metrics: pd.DataFrame,
    *,
    maximum_epochs: int = 240,
) -> pd.DataFrame:
    """Validate the orchestrator-frozen 20-row training diagnostic table."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.is_file():
            raise DirectFiveArmAnalysisError(
                f"Frozen training summary does not exist: {path}"
            )
        frame = pd.read_csv(path, low_memory=False)
    required = {
        "job_id",
        "fold",
        "arm",
        "best_epoch",
        "epochs_ran",
        "final_generator_lr",
        "final_discriminator_lr",
        "best_validation_score",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise DirectFiveArmAnalysisError(
            f"Training summary is empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise DirectFiveArmAnalysisError(
                f"Training summary {column} must be non-empty"
            )
    if len(result) != EXPECTED_JOB_COUNT or result["job_id"].duplicated().any():
        raise DirectFiveArmAnalysisError(
            "Training summary must contain exactly 20 unique jobs"
        )
    expected_cells = {(fold, arm) for fold in DIRECT_FOLDS for arm in DIRECT_ARMS}
    observed_cells = set(result[["fold", "arm"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise DirectFiveArmAnalysisError(
            "Training summary fold/arm universe differs from the direct matrix"
        )
    for column in ("best_epoch", "epochs_ran"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise DirectFiveArmAnalysisError(
                f"Training summary {column} must contain integers"
            )
        result[column] = values.astype(int)
    if (
        (result["best_epoch"] < 1).any()
        or (result["epochs_ran"] < result["best_epoch"]).any()
        or (result["epochs_ran"] > int(maximum_epochs)).any()
    ):
        raise DirectFiveArmAnalysisError(
            "Training epochs must satisfy 1 <= best_epoch <= epochs_ran <= maximum_epochs"
        )
    for column in (
        "final_generator_lr",
        "final_discriminator_lr",
        "best_validation_score",
    ):
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise DirectFiveArmAnalysisError(
                f"Training summary {column} must contain finite values"
            )
        if column.endswith("_lr") and (values <= 0.0).any():
            raise DirectFiveArmAnalysisError(
                f"Training summary {column} must be positive"
            )
        result[column] = values
    result["checkpoint_sha256"] = [
        _require_sha(value, "training summary checkpoint_sha256")
        for value in result["checkpoint_sha256"]
    ]
    pair_jobs = (
        pair_metrics[["job_id", "fold", "arm", "checkpoint_sha256"]]
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
        raise DirectFiveArmAnalysisError(
            "Training summary jobs or checkpoint hashes differ from pair metrics"
        )
    return result.sort_values(["fold", "arm"], kind="stable").reset_index(drop=True)


def compute_fold_summary(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return per-arm/per-fold MAE, persistence comparison, and rank."""

    rows: list[dict[str, Any]] = []
    for (fold, arm), group in pair_metrics.groupby(["fold", "arm"], sort=True):
        model_mae = float(group["target_mae"].mean())
        persistence_mae = float(group["persistence_mae"].mean())
        if model_mae <= 0.0 or persistence_mae <= 0.0:
            raise DirectFiveArmAnalysisError(
                f"Mean MAE must be positive for fold={fold}, arm={arm}"
            )
        log_ratio = float(math.log(model_mae / persistence_mae))
        rows.append(
            {
                "fold": str(fold),
                "arm": str(arm),
                "mean_mae": model_mae,
                "persistence_mae": persistence_mae,
                "log_mae_ratio_vs_persistence": log_ratio,
                "mae_ratio_vs_persistence": float(math.exp(log_ratio)),
                "improvement_vs_persistence_percent": float(
                    100.0 * (1.0 - model_mae / persistence_mae)
                ),
                "pair_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
                "interpretation": INTERPRETATION_LABEL,
            }
        )
    result = pd.DataFrame(rows)
    result["fold_mae_rank"] = (
        result.groupby("fold", sort=True)["mean_mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return result.sort_values(
        ["fold", "fold_mae_rank", "arm"], kind="stable"
    ).reset_index(drop=True)


def compute_arm_summary(
    pair_metrics: pd.DataFrame,
    fold_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return pooled and equally-fold-weighted MAE summaries and ranks."""

    folds = compute_fold_summary(pair_metrics) if fold_summary is None else fold_summary
    rows: list[dict[str, Any]] = []
    for arm, group in pair_metrics.groupby("arm", sort=True):
        arm_folds = folds[folds["arm"].eq(str(arm))]
        if len(arm_folds) != pair_metrics["fold"].nunique():
            raise DirectFiveArmAnalysisError(f"Fold summary is incomplete for {arm}")
        pooled_mae = float(group["target_mae"].mean())
        pooled_persistence = float(group["persistence_mae"].mean())
        equal_fold_mae = float(arm_folds["mean_mae"].mean())
        equal_fold_persistence = float(arm_folds["persistence_mae"].mean())
        rows.append(
            {
                "arm": str(arm),
                "pooled_mae": pooled_mae,
                "equal_fold_mae": equal_fold_mae,
                "pooled_persistence_mae": pooled_persistence,
                "equal_fold_persistence_mae": equal_fold_persistence,
                "pooled_improvement_vs_persistence_percent": float(
                    100.0 * (1.0 - pooled_mae / pooled_persistence)
                ),
                "equal_fold_improvement_vs_persistence_percent": float(
                    100.0 * (1.0 - equal_fold_mae / equal_fold_persistence)
                ),
                "pair_count": int(len(group)),
                "session_count": int(
                    group[["fold", "session_id"]].drop_duplicates().shape[0]
                ),
                "fold_count": int(group["fold"].nunique()),
                "seed": DIRECT_SEED,
                "interpretation": INTERPRETATION_LABEL,
                "confirmatory": False,
            }
        )
    result = pd.DataFrame(rows)
    result["pooled_mae_rank"] = (
        result["pooled_mae"].rank(method="min", ascending=True).astype(int)
    )
    result["equal_fold_mae_rank"] = (
        result["equal_fold_mae"].rank(method="min", ascending=True).astype(int)
    )
    return result.sort_values(
        ["equal_fold_mae_rank", "pooled_mae_rank", "arm"], kind="stable"
    ).reset_index(drop=True)


def _pair_arm_values(
    pair_metrics: pd.DataFrame,
    *,
    focal_arm: str,
    reference_arm: str,
) -> pd.DataFrame:
    keys = ["fold", "pair_id"]
    audit = ["session_id", "effective_origin_utc", "persistence_mae"]
    focal = pair_metrics[pair_metrics["arm"].eq(str(focal_arm))][
        [*keys, *audit, "target_mae"]
    ].rename(
        columns={
            "session_id": "focal_session_id",
            "effective_origin_utc": "focal_origin",
            "persistence_mae": "focal_persistence",
            "target_mae": "focal_value",
        }
    )
    reference = pair_metrics[pair_metrics["arm"].eq(str(reference_arm))][
        [*keys, *audit, "target_mae"]
    ].rename(
        columns={
            "session_id": "reference_session_id",
            "effective_origin_utc": "reference_origin",
            "persistence_mae": "reference_persistence",
            "target_mae": "reference_value",
        }
    )
    if focal.empty or reference.empty:
        raise DirectFiveArmAnalysisError(
            f"Missing comparison arm: {focal_arm} or {reference_arm}"
        )
    paired = focal.merge(
        reference, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise DirectFiveArmAnalysisError(
            f"Comparison is not pair complete: {focal_arm} vs {reference_arm}"
        )
    paired = paired.drop(columns="_merge")
    if not paired["focal_session_id"].equals(paired["reference_session_id"]):
        raise DirectFiveArmAnalysisError("Session lineage differs across arms")
    if not paired["focal_origin"].equals(paired["reference_origin"]):
        raise DirectFiveArmAnalysisError("Origin lineage differs across arms")
    if not np.array_equal(
        paired["focal_persistence"].to_numpy(float),
        paired["reference_persistence"].to_numpy(float),
    ):
        raise DirectFiveArmAnalysisError("Persistence lineage differs across arms")
    paired["session_id"] = paired.pop("focal_session_id")
    paired = paired.drop(
        columns=[
            "reference_session_id",
            "focal_origin",
            "reference_origin",
            "focal_persistence",
            "reference_persistence",
        ]
    )
    return paired.sort_values(["fold", "pair_id"], kind="stable").reset_index(drop=True)


def _persistence_values(pair_metrics: pd.DataFrame, arm: str) -> pd.DataFrame:
    selected = pair_metrics[pair_metrics["arm"].eq(str(arm))][
        ["fold", "pair_id", "session_id", "target_mae", "persistence_mae"]
    ].copy()
    if selected.empty:
        raise DirectFiveArmAnalysisError(f"Missing persistence arm: {arm}")
    return (
        selected.rename(
            columns={"target_mae": "focal_value", "persistence_mae": "reference_value"}
        )
        .sort_values(["fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )


def _fold_then_session_bootstrap(
    paired: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Bootstrap equal-fold log-MAE ratios from paired session clusters."""

    if int(iterations) < 2:
        raise DirectFiveArmAnalysisError("Bootstrap iterations must be at least two")
    required = {"fold", "pair_id", "session_id", "focal_value", "reference_value"}
    missing = sorted(required - set(paired.columns))
    if paired.empty or missing:
        raise DirectFiveArmAnalysisError(
            f"Bootstrap evidence is empty or missing columns: {missing}"
        )
    folds = tuple(map(str, expected_folds))
    if set(paired["fold"]) != set(folds):
        raise DirectFiveArmAnalysisError("Bootstrap fold universe drift")

    fold_points: dict[str, float] = {}
    session_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in folds:
        cell = paired[paired["fold"].eq(fold)].copy()
        if cell.empty:
            raise DirectFiveArmAnalysisError(f"Empty bootstrap fold: {fold}")
        focal = pd.to_numeric(cell["focal_value"], errors="coerce").to_numpy(float)
        reference = pd.to_numeric(cell["reference_value"], errors="coerce").to_numpy(
            float
        )
        if (
            not np.isfinite(focal).all()
            or not np.isfinite(reference).all()
            or focal.mean() <= 0.0
            or reference.mean() <= 0.0
        ):
            raise DirectFiveArmAnalysisError(
                f"Bootstrap MAEs must have finite positive means in {fold}"
            )
        fold_points[fold] = float(math.log(focal.mean() / reference.mean()))
        block = (
            cell.assign(_count=1)
            .groupby("session_id", sort=True, as_index=False)
            .agg(
                focal_sum=("focal_value", "sum"),
                reference_sum=("reference_value", "sum"),
                pair_count=("_count", "sum"),
            )
        )
        session_blocks.append(
            (
                block["focal_sum"].to_numpy(float),
                block["reference_sum"].to_numpy(float),
                block["pair_count"].to_numpy(float),
            )
        )

    point = float(np.mean(list(fold_points.values())))
    rng = np.random.default_rng(int(rng_seed))
    fold_draws = rng.integers(0, len(folds), size=(int(iterations), len(folds)))
    sampled_ratios = np.empty_like(fold_draws, dtype=float)
    for slot in range(len(folds)):
        for fold_index, (focal_sums, reference_sums, counts) in enumerate(
            session_blocks
        ):
            draw_rows = np.flatnonzero(fold_draws[:, slot] == fold_index)
            if not len(draw_rows):
                continue
            session_count = len(focal_sums)
            sampled_sessions = rng.integers(
                0, session_count, size=(len(draw_rows), session_count)
            )
            sampled_focal = focal_sums[sampled_sessions].sum(axis=1)
            sampled_reference = reference_sums[sampled_sessions].sum(axis=1)
            sampled_count = counts[sampled_sessions].sum(axis=1)
            focal_means = sampled_focal / sampled_count
            reference_means = sampled_reference / sampled_count
            if (
                not np.isfinite(focal_means).all()
                or not np.isfinite(reference_means).all()
                or (focal_means <= 0.0).any()
                or (reference_means <= 0.0).any()
            ):
                raise DirectFiveArmAnalysisError(
                    "Bootstrap draw produced a non-positive mean MAE"
                )
            sampled_ratios[draw_rows, slot] = np.log(focal_means / reference_means)
    draws = sampled_ratios.mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    return {
        "focal_equal_fold_mae": float(
            paired.groupby("fold", sort=True)["focal_value"].mean().mean()
        ),
        "reference_equal_fold_mae": float(
            paired.groupby("fold", sort=True)["reference_value"].mean().mean()
        ),
        "mean_log_mae_ratio": point,
        "geometric_mae_ratio": float(math.exp(point)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_value_one_sided": float(p_negative),
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_negative, p_positive))),
        "focal_nonworse_fold_count": int(
            sum(value <= 0.0 for value in fold_points.values())
        ),
        "fold_count": int(len(folds)),
        "pair_count": int(len(paired)),
        "session_count": int(paired[["fold", "session_id"]].drop_duplicates().shape[0]),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(rng_seed),
        "resampling_method": (
            "fold_then_paired_cme_session_cluster_recompute_fold_log_mae_ratio"
        ),
        "fold_log_mae_ratios_json": json.dumps(
            fold_points, sort_keys=True, separators=(",", ":")
        ),
        "difference_direction": ("log_mae_focal_over_reference_negative_is_better"),
    }


def fold_session_paired_bootstrap(
    pair_metrics: pd.DataFrame,
    *,
    focal_arm: str,
    reference_arm: str,
    expected_folds: Sequence[str] = DIRECT_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare two arms with the frozen fold -> paired-session bootstrap."""

    result = _fold_then_session_bootstrap(
        _pair_arm_values(
            pair_metrics, focal_arm=focal_arm, reference_arm=reference_arm
        ),
        expected_folds=expected_folds,
        iterations=iterations,
        rng_seed=rng_seed,
    )
    result.update({"focal_arm": str(focal_arm), "reference_arm": str(reference_arm)})
    return result


def fold_session_persistence_bootstrap(
    pair_metrics: pd.DataFrame,
    *,
    arm: str,
    expected_folds: Sequence[str] = DIRECT_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare one arm with persistence using the identical paired bootstrap."""

    result = _fold_then_session_bootstrap(
        _persistence_values(pair_metrics, arm),
        expected_folds=expected_folds,
        iterations=iterations,
        rng_seed=rng_seed,
    )
    result.update({"focal_arm": str(arm), "reference_arm": "persistence"})
    return result


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Expose the audited deterministic Holm step-down implementation."""

    try:
        return _audited_holm_adjust(p_values)
    except ValueError as exc:
        raise DirectFiveArmAnalysisError(str(exc)) from exc


def _apply_holm(
    frame: pd.DataFrame,
    *,
    family_column: str,
    id_column: str,
    expected_family_sizes: Mapping[str, int],
) -> pd.DataFrame:
    if frame.empty:
        raise DirectFiveArmAnalysisError("Holm input must not be empty")
    parts: list[pd.DataFrame] = []
    observed_families = set(frame[family_column].astype(str))
    if observed_families != set(expected_family_sizes):
        raise DirectFiveArmAnalysisError("Holm family universe drift")
    for family, group in frame.groupby(family_column, sort=True):
        expected_size = int(expected_family_sizes[str(family)])
        if len(group) != expected_size:
            raise DirectFiveArmAnalysisError(
                f"Holm family {family} must contain {expected_size} comparisons"
            )
        ids = group[id_column].astype(str)
        if ids.duplicated().any():
            raise DirectFiveArmAnalysisError(f"Duplicate IDs in Holm family {family}")
        adjusted = holm_adjust(dict(zip(ids, group["p_value_one_sided"], strict=True)))
        group = group.copy()
        group["holm_adjusted_p"] = ids.map(adjusted)
        group["holm_family_size"] = expected_size
        group["direction_favors_focal"] = group["mean_log_mae_ratio"].lt(0.0)
        group["ci_excludes_zero_favoring_focal"] = group["ci_95_upper"].lt(0.0)
        group["holm_rejects_one_sided_0p05"] = group["holm_adjusted_p"].lt(0.05)
        group["confirmatory"] = False
        group["inference_permitted"] = False
        group["claim_scope"] = "single_seed_descriptive_only"
        group["interpretation"] = INTERPRETATION_LABEL
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def analyze_direct_contrasts(
    pair_metrics: pd.DataFrame,
    *,
    contrast_specs: Sequence[Mapping[str, Any]] = DIRECT_CONTRASTS,
    expected_folds: Sequence[str] = DIRECT_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Run the four predeclared contrasts and their two Holm-2 families."""

    if len(contrast_specs) != 4:
        raise DirectFiveArmAnalysisError("Exactly four direct contrasts are required")
    rows: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(contrast_specs):
        spec = dict(raw_spec)
        missing = sorted(
            {"family", "research_question", "contrast_id", "focal_arm", "reference_arm"}
            - set(spec)
        )
        if missing:
            raise DirectFiveArmAnalysisError(
                f"Direct contrast spec is missing keys: {missing}"
            )
        stats = fold_session_paired_bootstrap(
            pair_metrics,
            focal_arm=str(spec["focal_arm"]),
            reference_arm=str(spec["reference_arm"]),
            expected_folds=expected_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + index,
        )
        stats.update(
            {
                "family": str(spec["family"]),
                "research_question": str(spec["research_question"]),
                "contrast_id": str(spec["contrast_id"]),
                "seed": DIRECT_SEED,
            }
        )
        rows.append(stats)
    result = _apply_holm(
        pd.DataFrame(rows),
        family_column="family",
        id_column="contrast_id",
        expected_family_sizes={"rq1_holm2": 2, "representation_holm2": 2},
    )
    return result.sort_values(
        ["research_question", "contrast_id"], kind="stable"
    ).reset_index(drop=True)


def analyze_persistence(
    pair_metrics: pd.DataFrame,
    *,
    arms: Sequence[str] = DIRECT_ARMS,
    expected_folds: Sequence[str] = DIRECT_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED + 50_000,
) -> pd.DataFrame:
    """Run the predeclared Holm-5 arm-vs-persistence descriptive family."""

    if tuple(map(str, arms)) != DIRECT_ARMS:
        raise DirectFiveArmAnalysisError(
            "Persistence family must contain the five canonical arms in order"
        )
    rows: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        stats = fold_session_persistence_bootstrap(
            pair_metrics,
            arm=str(arm),
            expected_folds=expected_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + index,
        )
        stats.update(
            {
                "family": "persistence_holm5",
                "comparison_id": f"{arm}_vs_persistence",
                "seed": DIRECT_SEED,
            }
        )
        rows.append(stats)
    result = _apply_holm(
        pd.DataFrame(rows),
        family_column="family",
        id_column="comparison_id",
        expected_family_sizes={"persistence_holm5": 5},
    )
    return result.sort_values("focal_arm", kind="stable").reset_index(drop=True)


def analyze_direct_5arm(
    pair_metrics: pd.DataFrame | str | Path,
    *,
    training_summary: pd.DataFrame | str | Path | None = None,
    maximum_epochs: int = 240,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> DirectFiveArmAnalysis:
    """Validate canonical evidence and build all in-memory analysis tables."""

    validated = validate_pair_metrics(pair_metrics)
    validated_training = (
        None
        if training_summary is None
        else validate_training_summary(
            training_summary, validated, maximum_epochs=int(maximum_epochs)
        )
    )
    folds = compute_fold_summary(validated)
    arms = compute_arm_summary(validated, folds)
    contrasts = analyze_direct_contrasts(
        validated,
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed),
    )
    persistence = analyze_persistence(
        validated,
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed) + 50_000,
    )
    leader = arms.sort_values(
        ["equal_fold_mae", "pooled_mae", "arm"], kind="stable"
    ).iloc[0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "news_first_vol_film_unet_direct_5arm_seed42",
        "interpretation": INTERPRETATION_LABEL,
        "confirmatory": False,
        "claim_scope": "single_seed_descriptive_only",
        "seed": DIRECT_SEED,
        "tolerance_minutes": DIRECT_TOLERANCE_MINUTES,
        "arms": list(DIRECT_ARMS),
        "folds": list(DIRECT_FOLDS),
        "job_count": int(validated["job_id"].nunique()),
        "pair_metric_row_count": int(len(validated)),
        "pair_count": int(validated[["fold", "pair_id"]].drop_duplicates().shape[0]),
        "session_count": int(
            validated[["fold", "session_id"]].drop_duplicates().shape[0]
        ),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_method": (
            "fold_then_paired_cme_session_cluster_recompute_fold_log_mae_ratio"
        ),
        "rq1_holm_family_size": 2,
        "representation_holm_family_size": 2,
        "persistence_holm_family_size": 5,
        "equal_fold_mae_leader": str(leader["arm"]),
        "equal_fold_mae_leader_value": float(leader["equal_fold_mae"]),
        "cross_seed_inference_permitted": False,
        "training_summary_included": validated_training is not None,
    }
    return DirectFiveArmAnalysis(
        pair_metrics=validated,
        fold_summary=folds,
        arm_summary=arms,
        contrasts=contrasts,
        persistence=persistence,
        training_summary=validated_training,
        summary=summary,
    )


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0"
        if abs(number) < 1e-3 or abs(number) >= 1e4:
            return f"{number:.6e}"
        return f"{number:.8f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[[column for column in columns if column in frame]]
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = [
            _format_value(value).replace("|", "\\|").replace("\n", " ") for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[[column for column in columns if column in frame]]
    headers = "".join(
        f"<th>{html.escape(str(column))}</th>" for column in selected.columns
    )
    rows: list[str] = []
    for row in selected.itertuples(index=False, name=None):
        cells = "".join(
            f"<td>{html.escape(_format_value(value))}</td>" for value in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_direct_reports(analysis: DirectFiveArmAnalysis) -> tuple[str, str]:
    """Render deterministic Markdown and self-contained HTML report strings."""

    rank_columns = (
        "equal_fold_mae_rank",
        "arm",
        "equal_fold_mae",
        "pooled_mae",
        "equal_fold_improvement_vs_persistence_percent",
        "pooled_improvement_vs_persistence_percent",
    )
    comparison_columns = (
        "research_question",
        "contrast_id",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "focal_nonworse_fold_count",
        "fold_count",
    )
    persistence_columns = (
        "focal_arm",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "focal_nonworse_fold_count",
        "fold_count",
    )
    fold_columns = (
        "fold",
        "fold_mae_rank",
        "arm",
        "mean_mae",
        "improvement_vs_persistence_percent",
        "pair_count",
        "session_count",
    )
    training_columns = (
        "fold",
        "arm",
        "best_epoch",
        "epochs_ran",
        "final_generator_lr",
        "final_discriminator_lr",
        "best_validation_score",
    )
    if analysis.training_summary is None:
        training_markdown = "_主编排器未向此纯分析调用提供训练汇总。_"
        training_html = "<p><em>主编排器未向此纯分析调用提供训练汇总。</em></p>"
    else:
        training_markdown = _markdown_table(analysis.training_summary, training_columns)
        training_html = _html_table(analysis.training_summary, training_columns)
    disclaimer = (
        "本报告仅包含 seed 42，属于 retrospective rolling-development 的描述性结果；"
        "Holm 校正和 bootstrap 区间不能证明结果可跨随机初始化复现。"
    )
    markdown = f"""# FiLM U-Net 五分支直接训练：单-Seed结果

> {disclaimer}

## 数据与方法

- 架构：`film_unet_mask_coords_v1 + lp_disabled_same_shape_v1`
- Seed：`42`
- 证据：{analysis.summary["job_count"]} jobs，{analysis.summary["pair_metric_row_count"]} pair-metric rows，{analysis.summary["pair_count"]} unique fold/pairs，{analysis.summary["session_count"]} fold/sessions
- Bootstrap：{analysis.summary["bootstrap_iterations"]:,}次 `fold → paired CME-session cluster`
- 排名主口径：四个fold等权MAE；数值越低越好

## 五个分支排名

{_markdown_table(analysis.arm_summary, rank_columns)}

## 四个预声明对比

负的 `mean_log_mae_ratio` 表示LP分支更好。RQ1与表示比较分别执行Holm-2。

{_markdown_table(analysis.contrasts, comparison_columns)}

## 相对Persistence

五个分支构成一个Holm-5 family。

{_markdown_table(analysis.persistence, persistence_columns)}

## Fold明细

{_markdown_table(analysis.fold_summary, fold_columns)}

## 训练诊断

{training_markdown}

## 解释边界

{disclaimer} 所有 `inference_permitted` 均固定为 `false`，结果不得表述为confirmatory或跨seed稳定性证据。
"""
    style = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#17202a;background:#f5f7fa}
main{max-width:1180px;margin:0 auto;padding:32px}.card{background:white;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 2px 12px #00000012}
h1,h2{color:#16324f}.warning{border-left:5px solid #b9770e;background:#fff8e7;padding:14px}.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e1e8;padding:7px;text-align:right}th{background:#eaf0f5}th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}code{background:#eef2f5;padding:2px 4px;border-radius:4px}
""".strip()
    html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FiLM U-Net direct five-arm seed42</title><style>{style}</style></head>
<body><main><h1>FiLM U-Net 五分支直接训练：单-Seed结果</h1>
<div class="warning">{html.escape(disclaimer)}</div>
<section class="card"><h2>数据与方法</h2><p>seed=42；{analysis.summary["job_count"]} jobs；{analysis.summary["pair_metric_row_count"]} pair-metric rows；bootstrap={analysis.summary["bootstrap_iterations"]:,}次 fold → paired CME-session cluster。</p></section>
<section class="card"><h2>五个分支排名</h2>{_html_table(analysis.arm_summary, rank_columns)}</section>
<section class="card"><h2>四个预声明对比</h2><p>负的mean_log_mae_ratio表示focal arm更好；两组Holm-2。</p>{_html_table(analysis.contrasts, comparison_columns)}</section>
<section class="card"><h2>相对Persistence（Holm-5）</h2>{_html_table(analysis.persistence, persistence_columns)}</section>
<section class="card"><h2>Fold明细</h2>{_html_table(analysis.fold_summary, fold_columns)}</section>
<section class="card"><h2>训练诊断</h2>{training_html}</section>
<section class="card"><h2>解释边界</h2><p>{html.escape(disclaimer)} 所有inference_permitted均为false。</p></section>
</main></body></html>
"""
    return markdown, html_report


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
        "utf-8"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise DirectFiveArmAnalysisError(f"Existing analysis output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_direct_analysis_bundle(
    analysis: DirectFiveArmAnalysis,
    *,
    pair_metrics_path: str | Path,
    pair_metrics_sha256: str,
    output_dir: str | Path,
    training_summary_path: str | Path | None = None,
    training_summary_sha256: str | None = None,
) -> dict[str, Path]:
    """Write deterministic outputs and a SHA-addressed artifact manifest."""

    source = Path(pair_metrics_path).resolve()
    expected_sha = _require_sha(pair_metrics_sha256, "pair metrics SHA-256")
    if sha256_file(source) != expected_sha:
        raise DirectFiveArmAnalysisError("Pair metrics SHA-256 drift before write")
    training_source: Path | None = None
    training_sha: str | None = None
    if training_summary_path is not None or training_summary_sha256 is not None:
        if training_summary_path is None or training_summary_sha256 is None:
            raise DirectFiveArmAnalysisError(
                "Training summary path and SHA-256 must be supplied together"
            )
        if analysis.training_summary is None:
            raise DirectFiveArmAnalysisError(
                "Training summary input was declared but not validated"
            )
        training_source = Path(training_summary_path).resolve()
        training_sha = _require_sha(training_summary_sha256, "training summary SHA-256")
        if sha256_file(training_source) != training_sha:
            raise DirectFiveArmAnalysisError(
                "Training summary SHA-256 drift before write"
            )
    elif analysis.training_summary is not None:
        raise DirectFiveArmAnalysisError(
            "Validated training summary requires a frozen path and SHA-256"
        )
    destination = Path(output_dir)
    paths = {
        "arm_summary": destination / "direct_5arm_arm_ranking.csv",
        "fold_summary": destination / "direct_5arm_fold_ranking.csv",
        "contrasts": destination / "direct_5arm_contrast_bootstrap_holm.csv",
        "persistence": destination / "direct_5arm_persistence_bootstrap_holm.csv",
        "summary": destination / "direct_5arm_analysis_summary.json",
        "report_markdown": destination / "direct_5arm_report.md",
        "report_html": destination / "direct_5arm_report.html",
        "manifest": destination / "analysis_manifest.json",
    }
    markdown, html_report = render_direct_reports(analysis)
    for role, frame in (
        ("arm_summary", analysis.arm_summary),
        ("fold_summary", analysis.fold_summary),
        ("contrasts", analysis.contrasts),
        ("persistence", analysis.persistence),
    ):
        _atomic_write(paths[role], _csv_bytes(frame))
    summary = dict(analysis.summary)
    summary.update(
        {
            "pair_metrics_path": str(source),
            "pair_metrics_sha256": expected_sha,
            "training_summary_path": (
                str(training_source) if training_source is not None else None
            ),
            "training_summary_sha256": training_sha,
        }
    )
    _atomic_write(paths["summary"], _canonical_json(summary).encode("utf-8"))
    _atomic_write(paths["report_markdown"], markdown.encode("utf-8"))
    _atomic_write(paths["report_html"], html_report.encode("utf-8"))
    artifacts = []
    for role in (
        "arm_summary",
        "fold_summary",
        "contrasts",
        "persistence",
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
            "role": "pair_metrics_source",
            "path": str(source),
            "sha256": expected_sha,
            "size_bytes": source.stat().st_size,
        }
    ]
    if training_source is not None and training_sha is not None:
        if sha256_file(training_source) != training_sha:
            raise DirectFiveArmAnalysisError(
                "Training summary changed while analysis outputs were written"
            )
        inputs.append(
            {
                "role": "training_summary_source",
                "path": str(training_source),
                "sha256": training_sha,
                "size_bytes": training_source.stat().st_size,
            }
        )
    if sha256_file(source) != expected_sha:
        raise DirectFiveArmAnalysisError(
            "Pair metrics changed while analysis outputs were written"
        )
    manifest = {
        "schema_version": 1,
        "kind": "news_first_vol_film_unet_direct_5arm_seed42_analysis_manifest_v1",
        "interpretation": INTERPRETATION_LABEL,
        "confirmatory": False,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    _atomic_write(paths["manifest"], _canonical_json(manifest).encode("utf-8"))
    return paths


def run_direct_5arm_analysis(
    *,
    pair_metrics_path: str | Path,
    pair_metrics_sha256: str,
    output_dir: str | Path,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Validate the frozen 2,500 rows, analyze them, and write the bundle.

    This is the expected orchestrator call.  The pair-metric file must already
    be frozen; no path is discovered and no test evidence is opened here.
    """

    source = Path(pair_metrics_path).resolve()
    expected_sha = _require_sha(pair_metrics_sha256, "pair metrics SHA-256")
    actual_sha = sha256_file(source)
    if actual_sha != expected_sha:
        raise DirectFiveArmAnalysisError(
            f"Pair metrics SHA-256 drift: expected={expected_sha}, actual={actual_sha}"
        )
    analysis = analyze_direct_5arm(
        source,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    return write_direct_analysis_bundle(
        analysis,
        pair_metrics_path=source,
        pair_metrics_sha256=expected_sha,
        output_dir=output_dir,
    )


def analyze_experiment(output_root: Path, config: Mapping[str, Any]) -> Path:
    """Analyze the canonical completed experiment and return its manifest path.

    The orchestrator-facing convention is fixed:

    - pair evidence: ``<output_root>/analysis/rq12_pair_metrics.csv.gz``
    - training diagnostics: ``<output_root>/analysis/training_summary.csv``
    - returned artifact: ``<output_root>/analysis/analysis_manifest.json``

    Both inputs are hashed before analysis and rechecked immediately before the
    manifest is committed.  This function never discovers checkpoints or
    predictions and therefore must only be called after the evaluation freeze.
    """

    if not isinstance(config, Mapping):
        raise DirectFiveArmAnalysisError("config must be a mapping")
    matrix = config.get("matrix")
    analysis_config = config.get("analysis")
    training_config = config.get("training")
    folds_config = config.get("folds")
    if not isinstance(matrix, Mapping) or not isinstance(analysis_config, Mapping):
        raise DirectFiveArmAnalysisError(
            "config must contain matrix and analysis mappings"
        )
    if not isinstance(training_config, Mapping) or not isinstance(folds_config, list):
        raise DirectFiveArmAnalysisError(
            "config must contain training mapping and folds list"
        )
    if tuple(map(str, matrix.get("direct_arms", ()))) != DIRECT_ARMS:
        raise DirectFiveArmAnalysisError("config direct_arms contract drift")
    if tuple(map(int, matrix.get("seeds", ()))) != (DIRECT_SEED,):
        raise DirectFiveArmAnalysisError("config seed contract drift")
    if int(matrix.get("expected_training_jobs", -1)) != EXPECTED_JOB_COUNT:
        raise DirectFiveArmAnalysisError("config expected_training_jobs drift")
    if int(matrix.get("expected_prediction_cells", -1)) != EXPECTED_JOB_COUNT:
        raise DirectFiveArmAnalysisError("config expected_prediction_cells drift")
    if int(matrix.get("expected_pair_metric_rows", -1)) != EXPECTED_PAIR_METRIC_ROWS:
        raise DirectFiveArmAnalysisError("config expected_pair_metric_rows drift")
    fold_ids = tuple(str(item.get("id", "")) for item in folds_config)
    if fold_ids != DIRECT_FOLDS:
        raise DirectFiveArmAnalysisError("config fold contract drift")
    if analysis_config.get("interpretation") != INTERPRETATION_LABEL:
        raise DirectFiveArmAnalysisError("config interpretation label drift")
    if bool(analysis_config.get("cross_seed_inference_enabled", True)):
        raise DirectFiveArmAnalysisError(
            "cross-seed inference must remain disabled for seed42"
        )
    iterations = int(
        analysis_config.get("bootstrap_replicates", DEFAULT_BOOTSTRAP_ITERATIONS)
    )
    bootstrap_seed = int(analysis_config.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    if iterations != DEFAULT_BOOTSTRAP_ITERATIONS:
        raise DirectFiveArmAnalysisError(
            "Production analysis requires exactly 10,000 bootstrap replicates"
        )
    maximum_epochs = int(training_config.get("num_epochs", -1))
    if maximum_epochs != 240:
        raise DirectFiveArmAnalysisError("config direct-training epoch cap drift")

    root = Path(output_root).resolve()
    analysis_dir = root / "analysis"
    pair_path = analysis_dir / "rq12_pair_metrics.csv.gz"
    training_path = analysis_dir / "training_summary.csv"
    pair_sha = sha256_file(pair_path)
    training_sha = sha256_file(training_path)
    result = analyze_direct_5arm(
        pair_path,
        training_summary=training_path,
        maximum_epochs=maximum_epochs,
        bootstrap_iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    paths = write_direct_analysis_bundle(
        result,
        pair_metrics_path=pair_path,
        pair_metrics_sha256=pair_sha,
        training_summary_path=training_path,
        training_summary_sha256=training_sha,
        output_dir=analysis_dir,
    )
    return paths["manifest"]


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DIRECT_ARMS",
    "DIRECT_CONTRASTS",
    "DIRECT_FOLDS",
    "DIRECT_SEED",
    "DirectFiveArmAnalysis",
    "DirectFiveArmAnalysisError",
    "INTERPRETATION_LABEL",
    "analyze_experiment",
    "analyze_direct_5arm",
    "analyze_direct_contrasts",
    "analyze_persistence",
    "compute_arm_summary",
    "compute_fold_summary",
    "fold_session_paired_bootstrap",
    "fold_session_persistence_bootstrap",
    "holm_adjust",
    "render_direct_reports",
    "run_direct_5arm_analysis",
    "sha256_file",
    "validate_pair_metrics",
    "validate_training_summary",
    "write_direct_analysis_bundle",
]
