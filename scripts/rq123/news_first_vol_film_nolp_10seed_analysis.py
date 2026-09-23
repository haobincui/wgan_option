"""Frozen, unified RQ1/RQ2/RQ3 analysis for the ten-seed FiLM experiment.

This module intentionally has no training or experiment-discovery code.  The
orchestrator must pass every input path and its already-frozen SHA-256 digest.
The public functions below operate on keyed pair-level evidence and fail
closed when an arm, seed, fold, pair, checkpoint, prediction, or source hash
does not agree with the declared experiment universe.

The inferential label follows the frozen experiment registry:
``retrospective_rolling_development``.  Reports separately and explicitly mark
the 2023 evidence exploratory and non-confirmatory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_ARMS = (
    "parent_current_only",
    "continuation_no_text",
    "lp_matched",
    "lp_shuffle",
    "bow",
    "sentiment",
)
CANONICAL_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
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
CANONICAL_TOLERANCES = (5, 30)
CANONICAL_ARMS_BY_TOLERANCE: Mapping[int, tuple[str, ...]] = {
    5: CANONICAL_ARMS,
    30: (
        "parent_current_only",
        "continuation_no_text",
        "lp_matched",
        "lp_shuffle",
    ),
}
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260822
RETROSPECTIVE_LABEL = "retrospective_rolling_development"
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_MARKET_JUMP_ROOT = REPO_ROOT / "outputs/rq3/atm_skew_jumps_20260810_final"
FROZEN_MARKET_JUMP_RELATIVE_PATH = Path("candidate_pairs.csv")
EXPECTED_MARKET_JUMP_TIERS = ("broad", "primary", "high")
DEFAULT_CURRENT_STATE_COLUMNS = (
    "current_surface_mean",
    "current_surface_std",
    "current_short_atm_mean",
    "current_strike_slope",
    "current_term_slope",
    "current_curvature",
    "current_supported_cell_fraction",
)

PAIR_METRIC_COLUMNS = (
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
    *DEFAULT_CURRENT_STATE_COLUMNS,
)

SCHEDULED_WINDOWS: Mapping[str, tuple[int, int, str]] = {
    "scheduled_0_plus30_primary": (0, 30, "primary"),
    "scheduled_minus10_plus20_robustness": (-10, 20, "robustness"),
    "scheduled_0_plus5_descriptive": (0, 5, "descriptive"),
}

DEFAULT_RQ1_RQ2_CONTRASTS = (
    {
        "research_question": "RQ1",
        "contrast_id": "lp_matched_minus_continuation_no_text",
        "focal_arm": "lp_matched",
        "reference_arm": "continuation_no_text",
    },
    {
        "research_question": "RQ1",
        "contrast_id": "lp_matched_minus_lp_shuffle",
        "focal_arm": "lp_matched",
        "reference_arm": "lp_shuffle",
    },
    {
        "research_question": "RQ2",
        "contrast_id": "lp_matched_minus_bow",
        "focal_arm": "lp_matched",
        "reference_arm": "bow",
    },
    {
        "research_question": "RQ2",
        "contrast_id": "lp_matched_minus_sentiment",
        "focal_arm": "lp_matched",
        "reference_arm": "sentiment",
    },
)

DEFAULT_PERSISTENCE_COMPARISONS = (
    {
        "research_question": "RQ1",
        "comparison_id": "parent_current_only_vs_persistence",
        "arm": "parent_current_only",
        "comparison_role": "diagnostic_only",
    },
    {
        "research_question": "RQ1",
        "comparison_id": "continuation_no_text_vs_persistence",
        "arm": "continuation_no_text",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ1",
        "comparison_id": "lp_matched_vs_persistence",
        "arm": "lp_matched",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ1",
        "comparison_id": "lp_shuffle_vs_persistence",
        "arm": "lp_shuffle",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ2",
        "comparison_id": "continuation_no_text_vs_persistence",
        "arm": "continuation_no_text",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ2",
        "comparison_id": "lp_matched_vs_persistence",
        "arm": "lp_matched",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ2",
        "comparison_id": "bow_vs_persistence",
        "arm": "bow",
        "comparison_role": "secondary",
    },
    {
        "research_question": "RQ2",
        "comparison_id": "sentiment_vs_persistence",
        "arm": "sentiment",
        "comparison_role": "secondary",
    },
)

DEFAULT_RQ3_CONTRASTS = (
    {
        "contrast_id": "lp_matched_vs_continuation_no_text",
        "focal_arm": "lp_matched",
        "reference_arm": "continuation_no_text",
        "contrast_role": "primary",
    },
    {
        "contrast_id": "lp_matched_vs_lp_shuffle",
        "focal_arm": "lp_matched",
        "reference_arm": "lp_shuffle",
        "contrast_role": "primary",
    },
)


class UnifiedAnalysisError(ValueError):
    """Raised when frozen evidence or a statistical contract drifts."""


@dataclass(frozen=True)
class ValidatedEvidence:
    """Validated pair evidence and its two job-level lineage tables."""

    pair_metrics: pd.DataFrame
    prediction_manifest: pd.DataFrame
    checkpoint_manifest: pd.DataFrame


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    source = Path(path)
    if not source.is_file():
        raise UnifiedAnalysisError(f"Required file does not exist: {source}")
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
        raise UnifiedAnalysisError(f"{label} must be one lowercase SHA-256 digest")
    return digest


def _verify_file(path: str | Path, expected_sha256: Any, label: str) -> Path:
    source = Path(path)
    expected = _require_sha(expected_sha256, f"{label} SHA-256")
    actual = sha256_file(source)
    if actual != expected:
        raise UnifiedAnalysisError(
            f"{label} SHA-256 drift: expected={expected}, actual={actual}, path={source}"
        )
    return source


def _read_table(source: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    path = Path(source)
    if not path.is_file():
        raise UnifiedAnalysisError(f"Required table does not exist: {path}")
    try:
        return pd.read_csv(path), path.parent
    except Exception as exc:  # pragma: no cover - pandas preserves the useful cause
        raise UnifiedAnalysisError(f"Could not read CSV table {path}: {exc}") from exc


def _canonical_utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].astype(str).head(3).tolist()
        raise UnifiedAnalysisError(
            f"{label} contains invalid UTC timestamps: {examples}"
        )
    return parsed


def _resolve_declared_path(value: Any, base_dir: Path | None, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise UnifiedAnalysisError(f"{label} path must be non-empty")
    path = Path(text)
    if not path.is_absolute():
        if base_dir is None:
            raise UnifiedAnalysisError(
                f"{label} uses a relative path but no manifest directory is available"
            )
        path = base_dir / path
    return path.resolve()


def _manifest_columns(frame: pd.DataFrame, kind: str) -> tuple[str, str]:
    path_specific = f"{kind}_path"
    sha_specific = f"{kind}_sha256"
    path_options = [column for column in (path_specific, "path") if column in frame]
    sha_options = [column for column in (sha_specific, "sha256") if column in frame]
    if len(path_options) != 1 or len(sha_options) != 1:
        raise UnifiedAnalysisError(
            f"{kind} manifest must contain exactly one path column "
            f"({path_specific!r} or 'path') and exactly one digest column "
            f"({sha_specific!r} or 'sha256')"
        )
    return path_options[0], sha_options[0]


def _validate_job_manifest(
    source: pd.DataFrame | str | Path,
    *,
    kind: str,
    expected_jobs: set[str],
) -> pd.DataFrame:
    frame, base_dir = _read_table(source)
    if "job_id" not in frame:
        raise UnifiedAnalysisError(f"{kind} manifest is missing job_id")
    path_column, sha_column = _manifest_columns(frame, kind)
    result = frame.copy()
    result["job_id"] = result["job_id"].astype(str).str.strip()
    if (
        result.empty
        or result["job_id"].eq("").any()
        or result["job_id"].duplicated().any()
    ):
        raise UnifiedAnalysisError(
            f"{kind} manifest job_id values must be non-empty and unique"
        )
    jobs = set(result["job_id"])
    if jobs != expected_jobs:
        raise UnifiedAnalysisError(
            f"{kind} manifest job universe drift: "
            f"missing={sorted(expected_jobs - jobs)}, extra={sorted(jobs - expected_jobs)}"
        )
    canonical_paths: list[str] = []
    canonical_hashes: list[str] = []
    for row in result.itertuples(index=False):
        values = row._asdict()
        job_id = str(values["job_id"])
        path = _resolve_declared_path(values[path_column], base_dir, f"{kind}/{job_id}")
        expected_sha = _require_sha(values[sha_column], f"{kind}/{job_id} SHA-256")
        _verify_file(path, expected_sha, f"{kind}/{job_id}")
        if "size_bytes" in result:
            size_value = pd.to_numeric(
                pd.Series([values["size_bytes"]]), errors="coerce"
            ).iloc[0]
            if (
                not math.isfinite(float(size_value))
                or int(size_value) != path.stat().st_size
            ):
                raise UnifiedAnalysisError(f"{kind}/{job_id} size_bytes drift")
        canonical_paths.append(str(path))
        canonical_hashes.append(expected_sha)
    result[f"{kind}_path"] = canonical_paths
    result[f"{kind}_sha256"] = canonical_hashes
    metadata_columns = [
        column
        for column in ("arm", "fold", "seed", "tolerance_minutes")
        if column in result
    ]
    keep = ["job_id", *metadata_columns, f"{kind}_path", f"{kind}_sha256"]
    return result[keep].sort_values("job_id", kind="stable").reset_index(drop=True)


def validate_frozen_artifact_manifest(
    manifest_path: str | Path,
    *,
    expected_roles: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate an explicit JSON/CSV artifact manifest without discovery.

    JSON manifests may contain an ``artifacts`` list (and optionally an
    ``inputs`` list); CSV manifests contain one row per artifact.  Every row
    requires ``role``, ``path``, and ``sha256``.  Relative paths are resolved
    against the manifest directory.
    """

    source = Path(manifest_path)
    if not source.is_file():
        raise UnifiedAnalysisError(f"Artifact manifest does not exist: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise UnifiedAnalysisError("Artifact JSON manifest must be an object")
        rows: list[Mapping[str, Any]] = []
        for key in ("inputs", "artifacts"):
            section = payload.get(key, [])
            if not isinstance(section, list):
                raise UnifiedAnalysisError(f"Artifact manifest {key} must be a list")
            rows.extend(section)
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(source)
    required = {"role", "path", "sha256"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise UnifiedAnalysisError(
            f"Artifact manifest is empty or missing columns: {missing}"
        )
    result = frame.copy()
    result["role"] = result["role"].astype(str).str.strip()
    if result["role"].eq("").any() or result["role"].duplicated().any():
        raise UnifiedAnalysisError("Artifact roles must be non-empty and unique")
    if expected_roles is not None and set(result["role"]) != set(
        map(str, expected_roles)
    ):
        expected = set(map(str, expected_roles))
        observed = set(result["role"])
        raise UnifiedAnalysisError(
            "Artifact role universe drift: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    paths: list[str] = []
    digests: list[str] = []
    for row in result.itertuples(index=False):
        values = row._asdict()
        path = _resolve_declared_path(
            values["path"], source.parent, str(values["role"])
        )
        digest = _require_sha(values["sha256"], f"{values['role']} SHA-256")
        _verify_file(path, digest, str(values["role"]))
        if "size_bytes" in result:
            expected_size = int(values["size_bytes"])
            if expected_size != path.stat().st_size:
                raise UnifiedAnalysisError(f"{values['role']} size_bytes drift")
        paths.append(str(path))
        digests.append(digest)
    result["path"] = paths
    result["sha256"] = digests
    return result.sort_values("role", kind="stable").reset_index(drop=True)


def _normalize_arms_by_tolerance(
    expected_arms: Sequence[str],
    expected_tolerances: Sequence[int],
    expected_arms_by_tolerance: Mapping[int, Sequence[str]] | None,
) -> dict[int, tuple[str, ...]]:
    arms = tuple(map(str, expected_arms))
    tolerances = tuple(map(int, expected_tolerances))
    if expected_arms_by_tolerance is None:
        if arms == CANONICAL_ARMS and tolerances == CANONICAL_TOLERANCES:
            return {
                tolerance: tuple(CANONICAL_ARMS_BY_TOLERANCE[tolerance])
                for tolerance in tolerances
            }
        return {tolerance: arms for tolerance in tolerances}
    result = {
        int(tolerance): tuple(map(str, arm_values))
        for tolerance, arm_values in expected_arms_by_tolerance.items()
    }
    if set(result) != set(tolerances):
        raise UnifiedAnalysisError(
            "expected_arms_by_tolerance keys must equal expected_tolerances"
        )
    if not all(values for values in result.values()):
        raise UnifiedAnalysisError("Each tolerance must retain at least one arm")
    if set().union(*(set(values) for values in result.values())) != set(arms):
        raise UnifiedAnalysisError(
            "expected_arms_by_tolerance must cover exactly expected_arms"
        )
    if any(len(values) != len(set(values)) for values in result.values()):
        raise UnifiedAnalysisError("Arms must be unique within each tolerance")
    return result


def _validate_pair_frame(
    source: pd.DataFrame | str | Path,
    *,
    expected_arms: Sequence[str],
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    expected_tolerances: Sequence[int],
    expected_arms_by_tolerance: Mapping[int, Sequence[str]] | None,
    require_complete_matrix: bool,
) -> pd.DataFrame:
    frame, _ = _read_table(source)
    missing = sorted(set(PAIR_METRIC_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise UnifiedAnalysisError(
            f"Pair metrics are empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm", "pair_id", "session_id"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise UnifiedAnalysisError(f"Pair metrics {column} must be non-empty")
    for column in ("seed", "tolerance_minutes"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise UnifiedAnalysisError(f"Pair metrics {column} must contain integers")
        result[column] = numeric.astype(int)
    result["effective_origin_utc"] = _canonical_utc(
        result["effective_origin_utc"], "effective_origin_utc"
    )
    for column in ("target_mae", "persistence_mae"):
        numeric = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any():
            raise UnifiedAnalysisError(
                f"Pair metrics {column} must be finite and nonnegative"
            )
        result[column] = numeric
    for column in DEFAULT_CURRENT_STATE_COLUMNS:
        numeric = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise UnifiedAnalysisError(
                f"Pair metrics {column} must contain finite current-state values"
            )
        result[column] = numeric
    if (result["persistence_mae"] <= 0).any():
        raise UnifiedAnalysisError("Pair metrics persistence_mae must be positive")
    for column in ("checkpoint_sha256", "prediction_sha256"):
        result[column] = [
            _require_sha(value, f"pair metrics {column}") for value in result[column]
        ]
    pair_keys = ["job_id", "pair_id"]
    if result.duplicated(pair_keys).any():
        raise UnifiedAnalysisError(
            "Pair metrics must contain one row per job_id/pair_id"
        )

    arms = tuple(map(str, expected_arms))
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    tolerances = tuple(map(int, expected_tolerances))
    arms_by_tolerance = _normalize_arms_by_tolerance(
        arms, tolerances, expected_arms_by_tolerance
    )
    for label, observed, expected in (
        ("arm", set(result["arm"]), set(arms)),
        ("seed", set(result["seed"]), set(seeds)),
        ("fold", set(result["fold"]), set(folds)),
        ("tolerance", set(result["tolerance_minutes"]), set(tolerances)),
    ):
        if observed != expected:
            raise UnifiedAnalysisError(
                f"Pair metrics {label} universe drift: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )

    job_metadata = ["job_id", "arm", "fold", "seed", "tolerance_minutes"]
    jobs = result[job_metadata].drop_duplicates()
    if jobs["job_id"].duplicated().any():
        raise UnifiedAnalysisError("One job_id maps to more than one experiment cell")
    combos = set(
        jobs[["arm", "fold", "seed", "tolerance_minutes"]].itertuples(
            index=False, name=None
        )
    )
    expected_combos = {
        (arm, fold, seed, tolerance)
        for tolerance in tolerances
        for arm in arms_by_tolerance[tolerance]
        for fold in folds
        for seed in seeds
    }
    if require_complete_matrix and combos != expected_combos:
        raise UnifiedAnalysisError(
            "Pair metric job matrix drift: "
            f"missing={sorted(expected_combos - combos)}, extra={sorted(combos - expected_combos)}"
        )

    for job_id, group in result.groupby("job_id", sort=False):
        for column in ("checkpoint_sha256", "prediction_sha256"):
            if group[column].nunique(dropna=False) != 1:
                raise UnifiedAnalysisError(f"{job_id} has multiple {column} values")

    universe_columns = [
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
        *DEFAULT_CURRENT_STATE_COLUMNS,
    ]
    for (tolerance, fold, seed), group in result.groupby(
        ["tolerance_minutes", "fold", "seed"], sort=True
    ):
        reference: pd.DataFrame | None = None
        for arm in arms_by_tolerance[int(tolerance)]:
            panel = (
                group[group["arm"].eq(arm)][universe_columns]
                .sort_values("pair_id", kind="stable")
                .reset_index(drop=True)
            )
            if panel.empty:
                if require_complete_matrix:
                    raise UnifiedAnalysisError(
                        f"Empty paired panel for tolerance={tolerance}, fold={fold}, seed={seed}, arm={arm}"
                    )
                continue
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise UnifiedAnalysisError(
                    "Pair/session/time/persistence/current-state universe differs across arms for "
                    f"tolerance={tolerance}, fold={fold}, seed={seed}"
                )
    invariant_columns = [
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
        *DEFAULT_CURRENT_STATE_COLUMNS,
    ]
    grouped_pairs = result.groupby(["tolerance_minutes", "fold", "pair_id"], sort=False)
    for column in invariant_columns:
        drift = grouped_pairs[column].nunique(dropna=False).gt(1)
        if drift.any():
            examples = [tuple(value) for value in drift[drift].index[:3]]
            raise UnifiedAnalysisError(
                f"Pair metric {column} differs across arms/seeds: {examples}"
            )
    return result.sort_values(
        ["tolerance_minutes", "fold", "seed", "arm", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def validate_paired_evidence(
    pair_metrics: pd.DataFrame | str | Path,
    prediction_manifest: pd.DataFrame | str | Path,
    checkpoint_manifest: pd.DataFrame | str | Path,
    *,
    expected_arms: Sequence[str] = CANONICAL_ARMS,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    expected_tolerances: Sequence[int] = CANONICAL_TOLERANCES,
    expected_arms_by_tolerance: Mapping[int, Sequence[str]] | None = None,
    require_complete_matrix: bool = True,
) -> ValidatedEvidence:
    """Validate the paired panel and every declared prediction/checkpoint file."""

    pairs = _validate_pair_frame(
        pair_metrics,
        expected_arms=expected_arms,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        expected_tolerances=expected_tolerances,
        expected_arms_by_tolerance=expected_arms_by_tolerance,
        require_complete_matrix=require_complete_matrix,
    )
    expected_jobs = set(pairs["job_id"])
    predictions = _validate_job_manifest(
        prediction_manifest, kind="prediction", expected_jobs=expected_jobs
    )
    checkpoints = _validate_job_manifest(
        checkpoint_manifest, kind="checkpoint", expected_jobs=expected_jobs
    )
    job_hashes = pairs.groupby("job_id", sort=True)[
        ["prediction_sha256", "checkpoint_sha256"]
    ].first()
    prediction_hashes = predictions.set_index("job_id")["prediction_sha256"]
    checkpoint_hashes = checkpoints.set_index("job_id")["checkpoint_sha256"]
    if not job_hashes["prediction_sha256"].equals(
        prediction_hashes.loc[job_hashes.index]
    ):
        raise UnifiedAnalysisError(
            "Prediction manifest hashes differ from pair metrics"
        )
    if not job_hashes["checkpoint_sha256"].equals(
        checkpoint_hashes.loc[job_hashes.index]
    ):
        raise UnifiedAnalysisError(
            "Checkpoint manifest hashes differ from pair metrics"
        )
    job_cells = (
        pairs[["job_id", "arm", "fold", "seed", "tolerance_minutes"]]
        .drop_duplicates()
        .set_index("job_id")
        .sort_index()
    )
    for label, manifest in (
        ("prediction", predictions),
        ("checkpoint", checkpoints),
    ):
        available = [
            column
            for column in ("arm", "fold", "seed", "tolerance_minutes")
            if column in manifest
        ]
        if not available:
            continue
        declared = manifest.set_index("job_id").loc[job_cells.index, available].copy()
        expected = job_cells[available].copy()
        for column in ("seed", "tolerance_minutes"):
            if column in available:
                declared[column] = pd.to_numeric(declared[column], errors="coerce")
        if not declared.equals(expected):
            raise UnifiedAnalysisError(
                f"{label} manifest experiment-cell metadata differs from pair metrics"
            )
    return ValidatedEvidence(pairs, predictions, checkpoints)


def _paired_arm_differences(
    frame: pd.DataFrame,
    *,
    focal_arm: str,
    reference_arm: str,
    value_column: str,
) -> pd.DataFrame:
    if focal_arm == reference_arm:
        raise UnifiedAnalysisError("A contrast must use two different arms")
    if value_column not in frame:
        raise UnifiedAnalysisError(f"Missing contrast value column: {value_column}")
    keys = ["tolerance_minutes", "fold", "seed", "pair_id"]
    audit = ["session_id", "effective_origin_utc", "persistence_mae"]
    focal = frame[frame["arm"].eq(focal_arm)][keys + audit + [value_column]].rename(
        columns={
            "session_id": "focal_session_id",
            "effective_origin_utc": "focal_origin",
            "persistence_mae": "focal_persistence",
            value_column: "focal_value",
        }
    )
    reference = frame[frame["arm"].eq(reference_arm)][
        keys + audit + [value_column]
    ].rename(
        columns={
            "session_id": "reference_session_id",
            "effective_origin_utc": "reference_origin",
            "persistence_mae": "reference_persistence",
            value_column: "reference_value",
        }
    )
    if focal.empty or reference.empty:
        raise UnifiedAnalysisError(
            f"Contrast arms are missing: {focal_arm}, {reference_arm}"
        )
    paired = focal.merge(
        reference, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise UnifiedAnalysisError(
            f"Contrast {focal_arm} vs {reference_arm} is not pair complete"
        )
    paired = paired.drop(columns="_merge")
    if not paired["focal_session_id"].equals(paired["reference_session_id"]):
        raise UnifiedAnalysisError("Session lineage differs across contrasted arms")
    if not paired["focal_origin"].equals(paired["reference_origin"]):
        raise UnifiedAnalysisError("Origin lineage differs across contrasted arms")
    if not np.array_equal(
        paired["focal_persistence"].to_numpy(float),
        paired["reference_persistence"].to_numpy(float),
    ):
        raise UnifiedAnalysisError("Persistence values differ across contrasted arms")
    paired["session_id"] = paired.pop("focal_session_id")
    paired["effective_origin_utc"] = paired.pop("focal_origin")
    paired["persistence_mae"] = paired.pop("focal_persistence")
    paired = paired.drop(
        columns=["reference_session_id", "reference_origin", "reference_persistence"]
    )
    paired["difference"] = paired["focal_value"] - paired["reference_value"]
    paired["focal_arm"] = focal_arm
    paired["reference_arm"] = reference_arm
    return paired


def _cell_mean(frame: pd.DataFrame, value_column: str) -> float:
    value = float(pd.to_numeric(frame[value_column], errors="coerce").mean())
    if not math.isfinite(value):
        raise UnifiedAnalysisError("Bootstrap cell mean is not finite")
    return value


def _hierarchical_bootstrap_values(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
    improvement_direction: str,
) -> dict[str, Any]:
    if int(iterations) < 2:
        raise UnifiedAnalysisError("Bootstrap iterations must be at least two")
    required = {"seed", "fold", "session_id", value_column}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise UnifiedAnalysisError(
            f"Bootstrap input is empty or missing columns: {missing}"
        )
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    observed_cells = set(
        frame[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )
    expected_cells = {(seed, fold) for seed in seeds for fold in folds}
    if observed_cells != expected_cells:
        raise UnifiedAnalysisError(
            "Bootstrap seed/fold cells drift: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )
    cells: dict[tuple[int, str], pd.DataFrame] = {}
    cell_points: dict[tuple[int, str], float] = {}
    for seed in seeds:
        for fold in folds:
            cell = frame[(frame["seed"].eq(seed)) & (frame["fold"].eq(fold))].copy()
            if cell.empty or cell["session_id"].astype(str).str.strip().eq("").any():
                raise UnifiedAnalysisError(
                    f"Empty session cluster in seed={seed}, fold={fold}"
                )
            values = pd.to_numeric(cell[value_column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise UnifiedAnalysisError("Bootstrap values must be finite")
            cells[(seed, fold)] = cell
            cell_points[(seed, fold)] = float(values.mean())
    point = float(np.mean(list(cell_points.values())))
    seed_means = {
        seed: float(np.mean([cell_points[(seed, fold)] for fold in folds]))
        for seed in seeds
    }
    fold_means = {
        fold: float(np.mean([cell_points[(seed, fold)] for seed in seeds]))
        for fold in folds
    }

    rng = np.random.default_rng(int(rng_seed))
    draws = np.empty(int(iterations), dtype=float)
    seed_array = np.asarray(seeds, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=object)
    for draw_index in range(int(iterations)):
        sampled_cell_means: list[float] = []
        for sampled_seed in rng.choice(seed_array, size=len(seed_array), replace=True):
            for sampled_fold in rng.choice(
                fold_array, size=len(fold_array), replace=True
            ):
                cell = cells[(int(sampled_seed), str(sampled_fold))]
                sessions = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
                chosen_sessions = rng.choice(sessions, size=len(sessions), replace=True)
                blocks = [
                    cell[cell["session_id"].eq(session)] for session in chosen_sessions
                ]
                sampled_cell_means.append(_cell_mean(pd.concat(blocks), value_column))
        draws[draw_index] = float(np.mean(sampled_cell_means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    p_two = min(1.0, 2.0 * min(p_negative, p_positive))
    if improvement_direction == "negative":
        p_one = p_negative
        seed_consistent = sum(value <= 0.0 for value in seed_means.values())
        fold_consistent = sum(value <= 0.0 for value in fold_means.values())
    elif improvement_direction == "positive":
        p_one = p_positive
        seed_consistent = sum(value >= 0.0 for value in seed_means.values())
        fold_consistent = sum(value >= 0.0 for value in fold_means.values())
    else:
        raise UnifiedAnalysisError(
            "improvement_direction must be 'negative' or 'positive'"
        )
    return {
        "mean_difference": point,
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_value_one_sided": float(p_one),
        "p_value_two_sided": float(p_two),
        "consistent_seed_count": int(seed_consistent),
        "consistent_fold_count": int(fold_consistent),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "pair_count": int(frame.get("pair_id", pd.Series(dtype=object)).nunique()),
        "session_count": int(frame["session_id"].nunique()),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(rng_seed),
        "resampling_method": "seed_then_fold_then_paired_session_cluster",
        "seed_means_json": json.dumps(
            seed_means, sort_keys=True, separators=(",", ":")
        ),
        "fold_means_json": json.dumps(
            fold_means, sort_keys=True, separators=(",", ":")
        ),
    }


def seed_fold_session_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    focal_arm: str,
    reference_arm: str,
    value_column: str = "target_mae",
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the predeclared seed -> fold -> paired-session bootstrap.

    The estimand is the equally weighted mean of seed/fold cell
    ``log(mean(MAE_focal) / mean(MAE_reference))`` values.  Each sampled
    session contributes all of its paired observations before that cell ratio
    is recomputed.  Negative values therefore favour the focal arm.
    """

    paired = _paired_arm_differences(
        frame,
        focal_arm=str(focal_arm),
        reference_arm=str(reference_arm),
        value_column=value_column,
    )
    if int(iterations) < 2:
        raise UnifiedAnalysisError("Bootstrap iterations must be at least two")
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    observed_cells = set(
        paired[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )
    expected_cells = {(seed, fold) for seed in seeds for fold in folds}
    if observed_cells != expected_cells:
        raise UnifiedAnalysisError(
            "Bootstrap seed/fold cells drift: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    def log_ratio(cell: pd.DataFrame) -> float:
        focal_value = float(cell["focal_value"].mean())
        reference_value = float(cell["reference_value"].mean())
        if (
            not math.isfinite(focal_value)
            or not math.isfinite(reference_value)
            or focal_value <= 0.0
            or reference_value <= 0.0
        ):
            raise UnifiedAnalysisError(
                "RQ1/RQ2 cell mean MAEs must be finite and positive"
            )
        return float(math.log(focal_value / reference_value))

    cells: dict[tuple[int, str], pd.DataFrame] = {}
    cell_points: dict[tuple[int, str], float] = {}
    for seed in seeds:
        for fold in folds:
            cell = paired[paired["seed"].eq(seed) & paired["fold"].eq(fold)].copy()
            if cell.empty:
                raise UnifiedAnalysisError(
                    f"Empty paired cell: seed={seed}, fold={fold}"
                )
            cells[(seed, fold)] = cell
            cell_points[(seed, fold)] = log_ratio(cell)
    point = float(np.mean(list(cell_points.values())))
    seed_means = {
        seed: float(np.mean([cell_points[(seed, fold)] for fold in folds]))
        for seed in seeds
    }
    fold_means = {
        fold: float(np.mean([cell_points[(seed, fold)] for seed in seeds]))
        for fold in folds
    }
    rng = np.random.default_rng(int(rng_seed))
    draws = np.empty(int(iterations), dtype=float)
    seed_array = np.asarray(seeds, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=object)
    for draw_index in range(int(iterations)):
        sampled_ratios: list[float] = []
        for sampled_seed in rng.choice(seed_array, size=len(seed_array), replace=True):
            for sampled_fold in rng.choice(
                fold_array, size=len(fold_array), replace=True
            ):
                cell = cells[(int(sampled_seed), str(sampled_fold))]
                sessions = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
                sampled_sessions = rng.choice(
                    sessions, size=len(sessions), replace=True
                )
                sampled = pd.concat(
                    [cell[cell["session_id"].eq(value)] for value in sampled_sessions],
                    ignore_index=True,
                )
                sampled_ratios.append(log_ratio(sampled))
        draws[draw_index] = float(np.mean(sampled_ratios))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    focal_mean = float(
        paired.groupby(["seed", "fold"], sort=True)["focal_value"].mean().mean()
    )
    reference_mean = float(
        paired.groupby(["seed", "fold"], sort=True)["reference_value"].mean().mean()
    )
    return {
        "focal_arm": str(focal_arm),
        "reference_arm": str(reference_arm),
        "focal_mean_mae": focal_mean,
        "reference_mean_mae": reference_mean,
        "mean_log_mae_ratio": point,
        "geometric_mae_ratio": float(math.exp(point)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_value_one_sided": float(p_negative),
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_negative, p_positive))),
        "consistent_seed_count": int(
            sum(value <= 0.0 for value in seed_means.values())
        ),
        "consistent_fold_count": int(
            sum(value <= 0.0 for value in fold_means.values())
        ),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "pair_count": int(paired["pair_id"].nunique()),
        "session_count": int(paired["session_id"].nunique()),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(rng_seed),
        "resampling_method": "seed_then_fold_then_paired_session_cluster_recompute_cell_log_mae_ratio",
        "seed_log_mae_ratios_json": json.dumps(
            seed_means, sort_keys=True, separators=(",", ":")
        ),
        "fold_log_mae_ratios_json": json.dumps(
            fold_means, sort_keys=True, separators=(",", ":")
        ),
        "difference_direction": "log_mae_focal_over_reference_negative_is_better",
    }


def seed_fold_session_persistence_bootstrap(
    frame: pd.DataFrame,
    *,
    arm: str,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap ``log(mean arm MAE / mean persistence MAE)`` by frozen cell."""

    selected = frame[frame["arm"].astype(str).eq(str(arm))].copy()
    if selected.empty:
        raise UnifiedAnalysisError(f"Persistence comparison arm is missing: {arm}")
    pseudo_arm = "__frozen_persistence_reference__"
    if pseudo_arm in set(frame["arm"].astype(str)):
        raise UnifiedAnalysisError(
            "Reserved persistence pseudo-arm appears in evidence"
        )
    persistence = selected.copy()
    persistence["arm"] = pseudo_arm
    persistence["target_mae"] = persistence["persistence_mae"]
    result = seed_fold_session_paired_bootstrap(
        pd.concat([selected, persistence], ignore_index=True),
        focal_arm=str(arm),
        reference_arm=pseudo_arm,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        iterations=iterations,
        rng_seed=rng_seed,
    )
    result["reference_arm"] = "persistence"
    return result


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic Holm step-down adjusted p-values."""

    if not p_values:
        return {}
    checked: list[tuple[str, float]] = []
    for name, value in p_values.items():
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise UnifiedAnalysisError("Holm p-values must be finite in [0, 1]")
        checked.append((str(name), number))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw))
        adjusted[name] = running
    return adjusted


def apply_holm_and_consistency_gate(
    results: pd.DataFrame,
    *,
    family_columns: Sequence[str] = ("research_question", "tolerance_minutes"),
    alpha: float = 0.05,
    min_nonworse_seed_fraction: float = 0.7,
    min_nonworse_fold_fraction: float = 0.75,
    estimate_column: str | None = None,
) -> pd.DataFrame:
    """Apply Holm within each declared family and the directional gate."""

    if estimate_column is None:
        estimate_column = (
            "mean_log_mae_ratio"
            if "mean_log_mae_ratio" in results
            else "mean_difference"
        )
    required = {
        "contrast_id",
        estimate_column,
        "ci_95_upper",
        "p_value_one_sided",
        "consistent_seed_count",
        "consistent_fold_count",
        "seed_count",
        "fold_count",
        *family_columns,
    }
    missing = sorted(required - set(results.columns))
    if missing or results.empty:
        raise UnifiedAnalysisError(f"Gate input is empty or missing columns: {missing}")
    if not 0 < alpha < 1:
        raise UnifiedAnalysisError("alpha must lie in (0,1)")
    output_parts: list[pd.DataFrame] = []
    group_key: str | list[str]
    group_key = list(family_columns)
    for _, group in results.groupby(group_key, sort=True, dropna=False):
        group = group.copy()
        adjusted = holm_adjust(
            dict(
                zip(
                    group["contrast_id"].astype(str),
                    group["p_value_one_sided"],
                    strict=True,
                )
            )
        )
        group["holm_adjusted_p"] = group["contrast_id"].astype(str).map(adjusted)
        required_seeds = np.ceil(
            group["seed_count"].astype(float) * min_nonworse_seed_fraction
        ).astype(int)
        required_folds = np.ceil(
            group["fold_count"].astype(float) * min_nonworse_fold_fraction
        ).astype(int)
        group["required_consistent_seed_count"] = required_seeds
        group["required_consistent_fold_count"] = required_folds
        group["passes_consistency_gate"] = group["consistent_seed_count"].astype(
            int
        ).ge(required_seeds) & group["consistent_fold_count"].astype(int).ge(
            required_folds
        )
        group["passes_full_gate"] = (
            group[estimate_column].astype(float).lt(0.0)
            & group["ci_95_upper"].astype(float).lt(0.0)
            & group["holm_adjusted_p"].astype(float).lt(alpha)
            & group["passes_consistency_gate"]
        )
        group["alpha"] = float(alpha)
        group["gate_estimate_column"] = str(estimate_column)
        output_parts.append(group)
    return (
        pd.concat(output_parts, ignore_index=True)
        .sort_values([*family_columns, "contrast_id"], kind="stable")
        .reset_index(drop=True)
    )


def analyze_rq1_rq2(
    pair_metrics: pd.DataFrame,
    *,
    contrast_specs: Sequence[Mapping[str, Any]] = DEFAULT_RQ1_RQ2_CONTRASTS,
    tolerance_minutes: int = 5,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Analyze the predeclared RQ1/RQ2 contrasts on the 5m panel only."""

    if int(tolerance_minutes) != 5:
        raise UnifiedAnalysisError("RQ1/RQ2 primary analysis is frozen to 5m")
    selected = pair_metrics[pair_metrics["tolerance_minutes"].astype(int).eq(5)]
    if selected.empty:
        raise UnifiedAnalysisError("No pair metrics for the frozen 5m RQ1/RQ2 panel")
    if not contrast_specs:
        raise UnifiedAnalysisError("RQ1/RQ2 contrast specs must be non-empty")
    rows: list[dict[str, Any]] = []
    seen_contrasts: set[tuple[str, str]] = set()
    for contrast_index, raw_spec in enumerate(contrast_specs):
        spec = dict(raw_spec)
        required = {
            "research_question",
            "contrast_id",
            "focal_arm",
            "reference_arm",
        }
        missing = sorted(required - set(spec))
        if missing:
            raise UnifiedAnalysisError(f"Contrast spec is missing keys: {missing}")
        if "tolerance_minutes" in spec and int(spec["tolerance_minutes"]) != 5:
            raise UnifiedAnalysisError(
                "RQ1/RQ2 contrast specs may only declare tolerance_minutes=5"
            )
        contrast_key = (
            str(spec["research_question"]).upper(),
            str(spec["contrast_id"]),
        )
        if contrast_key in seen_contrasts:
            raise UnifiedAnalysisError(
                f"Duplicate RQ1/RQ2 contrast within family: {contrast_key}"
            )
        seen_contrasts.add(contrast_key)
        result = seed_fold_session_paired_bootstrap(
            selected,
            focal_arm=str(spec["focal_arm"]),
            reference_arm=str(spec["reference_arm"]),
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + contrast_index,
        )
        result.update(
            {
                "research_question": str(spec["research_question"]).upper(),
                "contrast_id": str(spec["contrast_id"]),
                "tolerance_minutes": 5,
                "analysis_role": "primary",
                "interpretation": RETROSPECTIVE_LABEL,
            }
        )
        rows.append(result)
    results = apply_holm_and_consistency_gate(pd.DataFrame(rows))
    results["passes_primary_gate"] = results["passes_full_gate"]
    results["claim_scope"] = "retrospective_primary_exploratory"
    return results


def analyze_persistence_secondary(
    pair_metrics: pd.DataFrame,
    *,
    comparison_specs: Sequence[Mapping[str, Any]] = DEFAULT_PERSISTENCE_COMPARISONS,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED + 50_000,
) -> pd.DataFrame:
    """Analyze the two frozen 5m arm-vs-persistence secondary families."""

    selected = pair_metrics[pair_metrics["tolerance_minutes"].astype(int).eq(5)].copy()
    if selected.empty or not comparison_specs:
        raise UnifiedAnalysisError(
            "Persistence secondary analysis requires 5m evidence and comparisons"
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_spec in enumerate(comparison_specs):
        spec = dict(raw_spec)
        missing = sorted(
            {"research_question", "comparison_id", "arm", "comparison_role"} - set(spec)
        )
        if missing:
            raise UnifiedAnalysisError(
                f"Persistence comparison spec is missing keys: {missing}"
            )
        research_question = str(spec["research_question"]).upper()
        comparison_id = str(spec["comparison_id"])
        key = (research_question, comparison_id)
        if key in seen:
            raise UnifiedAnalysisError(
                f"Duplicate persistence comparison within family: {key}"
            )
        seen.add(key)
        stats = seed_fold_session_persistence_bootstrap(
            selected,
            arm=str(spec["arm"]),
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + index,
        )
        stats.update(
            {
                "research_question": research_question,
                "contrast_id": comparison_id,
                "comparison_id": comparison_id,
                "comparison_role": str(spec["comparison_role"]),
                "analysis_role": "secondary_arm_vs_persistence",
                "tolerance_minutes": 5,
                "interpretation": RETROSPECTIVE_LABEL,
            }
        )
        rows.append(stats)
    results = apply_holm_and_consistency_gate(
        pd.DataFrame(rows), family_columns=("research_question",)
    )
    results["passes_statistical_gate"] = results["passes_full_gate"]
    results["inference_permitted"] = ~results["comparison_role"].eq("diagnostic_only")
    results["passes_secondary_gate"] = results["passes_statistical_gate"] & results[
        "inference_permitted"
    ].astype(bool)
    results["claim_scope"] = np.where(
        results["inference_permitted"].astype(bool),
        "retrospective_secondary_exploratory",
        "retrospective_diagnostic_only",
    )
    return results.sort_values(
        ["research_question", "comparison_id"], kind="stable"
    ).reset_index(drop=True)


def _base_pair_metadata(
    pair_metrics: pd.DataFrame,
    tolerance_minutes: int,
    *,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    selected = pair_metrics[
        pair_metrics["tolerance_minutes"].astype(int).eq(int(tolerance_minutes))
    ].copy()
    if selected.empty:
        raise UnifiedAnalysisError(
            f"No pair metadata for tolerance={tolerance_minutes}"
        )
    selected["effective_origin_utc"] = _canonical_utc(
        selected["effective_origin_utc"], "pair effective_origin_utc"
    )
    missing = sorted(set(map(str, extra_columns)) - set(selected.columns))
    if missing:
        raise UnifiedAnalysisError(
            f"Pair metrics lack frozen current-state columns: {missing}"
        )
    columns = [
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        *map(str, extra_columns),
    ]
    for column in columns[1:]:
        if (
            selected.groupby(["fold", "pair_id"], sort=False)[column]
            .nunique(dropna=False)
            .gt(1)
            .any()
        ):
            raise UnifiedAnalysisError(
                f"Pair metadata {column} differs across arms/seeds"
            )
    for column in map(str, extra_columns):
        numeric = pd.to_numeric(selected[column], errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise UnifiedAnalysisError(
                f"Current-state match column {column} must be finite"
            )
        selected[column] = numeric
    return (
        selected[columns]
        .drop_duplicates()
        .sort_values(["fold", "effective_origin_utc", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_scheduled_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "release_time_utc"}
    missing = sorted(required - set(events.columns))
    if missing or events.empty:
        raise UnifiedAnalysisError(
            f"Scheduled events are empty or missing columns: {missing}"
        )
    result = events.copy()
    result["event_id"] = result["event_id"].astype(str).str.strip()
    if result["event_id"].eq("").any() or result["event_id"].duplicated().any():
        raise UnifiedAnalysisError(
            "Scheduled event_id values must be non-empty and unique"
        )
    result["release_time_utc"] = _canonical_utc(
        result["release_time_utc"], "release_time_utc"
    )
    if (
        "scheduled_or_unscheduled" in result
        and not result["scheduled_or_unscheduled"]
        .astype(str)
        .str.lower()
        .eq("scheduled")
        .all()
    ):
        raise UnifiedAnalysisError("Scheduled-event source contains unscheduled rows")
    return result.sort_values(
        ["release_time_utc", "event_id"], kind="stable"
    ).reset_index(drop=True)


def _event_membership(
    pairs: pd.DataFrame,
    events: pd.DataFrame,
    lower_minutes: int,
    upper_minutes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_times = events["release_time_utc"].to_numpy()
    for pair in pairs.itertuples(index=False):
        origin = pd.Timestamp(pair.effective_origin_utc)
        deltas = np.asarray(
            [
                (origin - pd.Timestamp(event_time)).total_seconds() / 60.0
                for event_time in event_times
            ],
            dtype=float,
        )
        eligible = np.flatnonzero(
            (deltas >= float(lower_minutes)) & (deltas <= float(upper_minutes))
        )
        if len(eligible) == 0:
            continue
        ranked = sorted(
            eligible,
            key=lambda index: (
                abs(float(deltas[index])),
                str(events.iloc[int(index)]["event_id"]),
            ),
        )
        selected_index = int(ranked[0])
        event = events.iloc[selected_index]
        eligible_events = events.iloc[eligible].sort_values(
            ["release_time_utc", "event_id"], kind="stable"
        )
        rows.append(
            {
                "fold": str(pair.fold),
                "pair_id": str(pair.pair_id),
                "session_id": str(pair.session_id),
                "effective_origin_utc": origin,
                "event_id": str(event["event_id"]),
                "event_ids": ";".join(eligible_events["event_id"].astype(str)),
                "event_id_count": int(eligible_events["event_id"].nunique()),
                "release_time_utc": pd.Timestamp(event["release_time_utc"]),
                "event_delta_minutes": float(deltas[selected_index]),
                "overlapping_event_count": int(len(eligible)),
            }
        )
    return pd.DataFrame(rows)


def _ordinary_pool(
    pairs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    minimum_distance_minutes: int,
) -> pd.DataFrame:
    if int(minimum_distance_minutes) < 0:
        raise UnifiedAnalysisError("ordinary minimum distance must be nonnegative")
    event_times = [pd.Timestamp(value) for value in events["release_time_utc"]]
    keep: list[bool] = []
    for row in pairs.itertuples(index=False):
        origin = pd.Timestamp(row.effective_origin_utc)
        minimum = min(
            abs((origin - release).total_seconds()) / 60.0 for release in event_times
        )
        keep.append(minimum >= float(minimum_distance_minutes))
    return pairs.loc[keep].copy()


def _circular_minute_distance(left: float, right: float) -> float:
    direct = abs(left - right)
    return min(direct, 1440.0 - direct)


def build_ordinary_match_plan(
    pair_frame: pd.DataFrame,
    scheduled_events: pd.DataFrame,
    *,
    tolerance_minutes: int = 30,
    windows: Mapping[str, tuple[int, int, str]] = SCHEDULED_WINDOWS,
    match_ratio: int = 1,
    ordinary_minimum_distance_minutes: int = 60,
    clock_caliper_minutes: int | None = 15,
) -> pd.DataFrame:
    """Build deterministic fold/weekday-exact ordinary matches without replacement."""

    if int(match_ratio) <= 0:
        raise UnifiedAnalysisError("match_ratio must be positive")
    pairs = _base_pair_metadata(pair_frame, tolerance_minutes)
    events = _validate_scheduled_events(scheduled_events)
    ordinary = _ordinary_pool(
        pairs,
        events,
        minimum_distance_minutes=int(ordinary_minimum_distance_minutes),
    )
    if ordinary.empty:
        raise UnifiedAnalysisError(
            "No ordinary-news controls satisfy the frozen event-distance buffer"
        )
    output: list[dict[str, Any]] = []
    for window_id, raw_window in windows.items():
        if len(raw_window) != 3:
            raise UnifiedAnalysisError(
                f"Invalid scheduled window contract: {window_id}"
            )
        lower, upper, role = int(raw_window[0]), int(raw_window[1]), str(raw_window[2])
        if lower > upper:
            raise UnifiedAnalysisError(
                f"Scheduled window lower bound exceeds upper: {window_id}"
            )
        scheduled = _event_membership(pairs, events, lower, upper)
        if scheduled.empty:
            raise UnifiedAnalysisError(f"No scheduled pairs in window {window_id}")
        available = set(ordinary.index)
        for event in scheduled.sort_values(
            ["release_time_utc", "effective_origin_utc", "pair_id"], kind="stable"
        ).itertuples(index=False):
            event_origin = pd.Timestamp(event.effective_origin_utc)
            event_weekday = event_origin.weekday()
            candidates = ordinary.loc[
                [
                    index
                    for index in sorted(available)
                    if str(ordinary.loc[index, "fold"]) == str(event.fold)
                    and pd.Timestamp(
                        ordinary.loc[index, "effective_origin_utc"]
                    ).weekday()
                    == event_weekday
                ]
            ].copy()
            if clock_caliper_minutes is not None:
                if int(clock_caliper_minutes) < 0:
                    raise UnifiedAnalysisError(
                        "clock_caliper_minutes must be nonnegative"
                    )
                event_minute = event_origin.hour * 60.0 + event_origin.minute
                clock_distances = candidates["effective_origin_utc"].map(
                    lambda value: _circular_minute_distance(
                        event_minute,
                        pd.Timestamp(value).hour * 60.0 + pd.Timestamp(value).minute,
                    )
                )
                candidates = candidates.loc[
                    clock_distances.le(float(clock_caliper_minutes))
                ].copy()
            matched_set_id = (
                "match_"
                + hashlib.sha256(
                    f"{window_id}|{event.event_id}|{event.fold}|{event.pair_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
            )
            if len(candidates) < int(match_ratio):
                output.append(
                    {
                        "window_id": str(window_id),
                        "window_lower_minutes": lower,
                        "window_upper_minutes": upper,
                        "window_role": role,
                        "matched_set_id": matched_set_id,
                        "match_rank": 0,
                        "match_status": "unmatched",
                        "unmatched_reason": "insufficient_fold_weekday_clock_controls",
                        "fold": str(event.fold),
                        "event_id": str(event.event_id),
                        "event_ids": str(event.event_ids),
                        "event_id_count": int(event.event_id_count),
                        "release_time_utc": pd.Timestamp(event.release_time_utc),
                        "event_pair_id": str(event.pair_id),
                        "event_session_id": str(event.session_id),
                        "event_effective_origin_utc": pd.Timestamp(
                            event.effective_origin_utc
                        ),
                        "event_delta_minutes": float(event.event_delta_minutes),
                        "overlapping_event_count": int(event.overlapping_event_count),
                        "control_pair_id": "",
                        "control_session_id": "",
                        "control_effective_origin_utc": pd.NaT,
                        "clock_distance_minutes": math.nan,
                        "match_distance": math.nan,
                        "matching_method": "fold_weekday_exact_clock_caliper_greedy_no_replacement_calendar_distance_v1",
                    }
                )
                continue
            event_minute = event_origin.hour * 60.0 + event_origin.minute
            distances: list[float] = []
            for control in candidates.itertuples(index=False):
                control_origin = pd.Timestamp(control.effective_origin_utc)
                control_minute = control_origin.hour * 60.0 + control_origin.minute
                minute_component = (
                    _circular_minute_distance(event_minute, control_minute) / 60.0
                )
                date_component = (
                    abs((event_origin.normalize() - control_origin.normalize()).days)
                    / 92.0
                )
                distances.append(
                    float(math.sqrt(minute_component**2 + date_component**2))
                )
            candidates["match_distance"] = distances
            chosen = candidates.sort_values(
                ["match_distance", "effective_origin_utc", "pair_id"], kind="stable"
            ).head(int(match_ratio))
            for rank, (index, control) in enumerate(chosen.iterrows(), start=1):
                available.remove(index)
                output.append(
                    {
                        "window_id": str(window_id),
                        "window_lower_minutes": lower,
                        "window_upper_minutes": upper,
                        "window_role": role,
                        "matched_set_id": matched_set_id,
                        "match_rank": rank,
                        "match_status": "matched",
                        "unmatched_reason": "",
                        "fold": str(event.fold),
                        "event_id": str(event.event_id),
                        "event_ids": str(event.event_ids),
                        "event_id_count": int(event.event_id_count),
                        "release_time_utc": pd.Timestamp(event.release_time_utc),
                        "event_pair_id": str(event.pair_id),
                        "event_session_id": str(event.session_id),
                        "event_effective_origin_utc": pd.Timestamp(
                            event.effective_origin_utc
                        ),
                        "event_delta_minutes": float(event.event_delta_minutes),
                        "overlapping_event_count": int(event.overlapping_event_count),
                        "control_pair_id": str(control["pair_id"]),
                        "control_session_id": str(control["session_id"]),
                        "control_effective_origin_utc": pd.Timestamp(
                            control["effective_origin_utc"]
                        ),
                        "clock_distance_minutes": _circular_minute_distance(
                            event_minute,
                            pd.Timestamp(control["effective_origin_utc"]).hour * 60.0
                            + pd.Timestamp(control["effective_origin_utc"]).minute,
                        ),
                        "match_distance": float(control["match_distance"]),
                        "matching_method": "fold_weekday_exact_clock_caliper_greedy_no_replacement_calendar_distance_v1",
                    }
                )
    result = (
        pd.DataFrame(output)
        .sort_values(
            ["window_id", "release_time_utc", "event_pair_id", "match_rank"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    matched_windows = set(
        result.loc[result["match_status"].eq("matched"), "window_id"].astype(str)
    )
    if matched_windows != set(map(str, windows)):
        raise UnifiedAnalysisError(
            "At least one scheduled window has zero ordinary-news matches: "
            f"{sorted(set(map(str, windows)) - matched_windows)}"
        )
    return result


def _pair_arm_value_map(frame: pd.DataFrame, tolerance_minutes: int) -> pd.DataFrame:
    selected = frame[
        frame["tolerance_minutes"].astype(int).eq(int(tolerance_minutes))
    ].copy()
    keys = ["fold", "seed", "pair_id", "session_id", "arm"]
    if selected.duplicated(keys).any():
        raise UnifiedAnalysisError("Pair arm values are not unique")
    return selected.set_index(keys)["target_mae"].to_frame()


def analyze_rq3_scheduled(
    pair_metrics: pd.DataFrame,
    scheduled_events: pd.DataFrame,
    *,
    match_plan: pd.DataFrame | None = None,
    contrast_specs: Sequence[Mapping[str, Any]] = DEFAULT_RQ3_CONTRASTS,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED + 100_000,
    ordinary_minimum_distance_minutes: int = 60,
    clock_caliper_minutes: int | None = 15,
    minimum_primary_pairs: int = 30,
    minimum_primary_releases: int = 20,
    minimum_primary_sessions: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze frozen 30m scheduled windows against matched ordinary news.

    Returns ``(match_plan, matched_set_differences, window_results)``.  Positive
    scheduled increments mean that the focal representation has more value in
    the scheduled window than in its ordinary-news control.
    """

    expected_plan = build_ordinary_match_plan(
        pair_metrics,
        scheduled_events,
        ordinary_minimum_distance_minutes=ordinary_minimum_distance_minutes,
        clock_caliper_minutes=clock_caliper_minutes,
    )
    plan = expected_plan if match_plan is None else match_plan.copy()
    required_plan = {
        "window_id",
        "window_role",
        "matched_set_id",
        "match_status",
        "fold",
        "event_id",
        "event_ids",
        "event_pair_id",
        "event_session_id",
        "control_pair_id",
    }
    missing = sorted(required_plan - set(plan.columns))
    if missing or plan.empty:
        raise UnifiedAnalysisError(
            f"Scheduled match plan is empty or missing columns: {missing}"
        )
    if match_plan is not None:
        comparable = plan.copy()
        for column in (
            "release_time_utc",
            "event_effective_origin_utc",
            "control_effective_origin_utc",
        ):
            if column in comparable:
                comparable[column] = pd.to_datetime(
                    comparable[column], errors="coerce", utc=True
                )
        comparable = comparable[expected_plan.columns].reset_index(drop=True)
        if not comparable.equals(expected_plan):
            raise UnifiedAnalysisError(
                "Supplied scheduled match plan differs from deterministic replay"
            )
    invalid_statuses = set(plan["match_status"].astype(str)) - {"matched", "unmatched"}
    if invalid_statuses:
        raise UnifiedAnalysisError(
            f"Scheduled match plan contains invalid statuses: {sorted(invalid_statuses)}"
        )
    matched_plan = plan[plan["match_status"].astype(str).eq("matched")].copy()
    if matched_plan.empty:
        raise UnifiedAnalysisError("Scheduled match plan has no matched controls")
    if matched_plan.duplicated(["window_id", "event_pair_id", "control_pair_id"]).any():
        raise UnifiedAnalysisError("Scheduled match plan contains duplicate matches")
    values = _pair_arm_value_map(pair_metrics, 30)
    detail_rows: list[dict[str, Any]] = []
    for spec in contrast_specs:
        focal = str(spec["focal_arm"])
        reference = str(spec["reference_arm"])
        for match in matched_plan.itertuples(index=False):
            for seed in map(int, expected_seeds):
                keys = {
                    "event_focal": (
                        match.fold,
                        seed,
                        match.event_pair_id,
                        match.event_session_id,
                        focal,
                    ),
                    "event_reference": (
                        match.fold,
                        seed,
                        match.event_pair_id,
                        match.event_session_id,
                        reference,
                    ),
                    "control_focal": (
                        match.fold,
                        seed,
                        match.control_pair_id,
                        match.control_session_id,
                        focal,
                    ),
                    "control_reference": (
                        match.fold,
                        seed,
                        match.control_pair_id,
                        match.control_session_id,
                        reference,
                    ),
                }
                try:
                    resolved = {
                        name: float(values.loc[key, "target_mae"])
                        for name, key in keys.items()
                    }
                except KeyError as exc:
                    raise UnifiedAnalysisError(
                        f"Scheduled match lacks paired prediction evidence: {keys}"
                    ) from exc
                event_advantage = resolved["event_reference"] - resolved["event_focal"]
                control_advantage = (
                    resolved["control_reference"] - resolved["control_focal"]
                )
                detail_rows.append(
                    {
                        "window_id": str(match.window_id),
                        "window_role": str(match.window_role),
                        "matched_set_id": str(match.matched_set_id),
                        "fold": str(match.fold),
                        "seed": seed,
                        "session_id": str(match.event_session_id),
                        "event_id": str(match.event_id),
                        "event_ids": str(match.event_ids),
                        "event_pair_id": str(match.event_pair_id),
                        "pair_id": str(match.event_pair_id),
                        "control_pair_id": str(match.control_pair_id),
                        "contrast_id": str(spec["contrast_id"]),
                        "contrast_role": str(spec.get("contrast_role", "descriptive")),
                        "focal_arm": focal,
                        "reference_arm": reference,
                        "event_text_advantage": event_advantage,
                        "control_text_advantage": control_advantage,
                        "scheduled_increment": event_advantage - control_advantage,
                        "difference_direction": "event_minus_control_positive_means_focal_more_valuable",
                    }
                )
    details = pd.DataFrame(detail_rows)
    result_rows: list[dict[str, Any]] = []
    expected_fold_tuple = tuple(map(str, expected_folds))
    expected_fold_set = set(expected_fold_tuple)
    for index, ((window_id, contrast_id), group) in enumerate(
        details.groupby(["window_id", "contrast_id"], sort=True)
    ):
        first = group.iloc[0]
        observed_fold_set = set(group["fold"].astype(str).unique())
        extra_folds = sorted(observed_fold_set - expected_fold_set)
        if extra_folds:
            raise UnifiedAnalysisError(
                f"Scheduled analysis contains unexpected folds: {extra_folds}"
            )
        observed_folds = tuple(
            fold for fold in expected_fold_tuple if fold in observed_fold_set
        )
        missing_folds = tuple(
            fold for fold in expected_fold_tuple if fold not in observed_fold_set
        )
        full_fold_panel = not missing_folds
        if str(first["window_role"]) == "primary" and not full_fold_panel:
            raise UnifiedAnalysisError(
                "Primary scheduled window is missing frozen folds: "
                f"{list(missing_folds)}"
            )
        stats = _hierarchical_bootstrap_values(
            group,
            value_column="scheduled_increment",
            expected_seeds=expected_seeds,
            # Robustness and explicitly descriptive windows may have no matched
            # event/control pair in one or more folds.  They remain transparent
            # descriptive analyses over the observed fold universe; only the
            # primary window is required to retain the frozen four-fold panel.
            expected_folds=observed_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + index,
            improvement_direction="positive",
        )
        window_plan = plan[plan["window_id"].astype(str).eq(str(window_id))]
        candidate_pair_count = int(window_plan["event_pair_id"].nunique())
        unmatched_pair_count = int(
            window_plan.loc[
                window_plan["match_status"].astype(str).eq("unmatched"),
                "event_pair_id",
            ].nunique()
        )
        stats.update(
            {
                "research_question": "RQ3",
                "analysis_type": "scheduled_news_vs_matched_ordinary",
                "window_id": str(window_id),
                "window_role": str(first["window_role"]),
                "contrast_id": str(contrast_id),
                "contrast_role": str(first["contrast_role"]),
                "focal_arm": str(first["focal_arm"]),
                "reference_arm": str(first["reference_arm"]),
                # Coverage admission is based on the deterministic release
                # selected for each matched event pair.  ``event_ids`` retains
                # every overlapping eligible release at that timestamp for
                # audit, but those alternatives must not inflate the release
                # count used by the minimum-coverage gate.
                "event_count": int(group["event_id"].astype(str).nunique()),
                "eligible_event_count": len(
                    {
                        event_id
                        for values in group["event_ids"].astype(str).unique()
                        for event_id in values.split(";")
                        if event_id
                    }
                ),
                "matched_set_count": int(group["matched_set_id"].nunique()),
                "candidate_pair_count": candidate_pair_count,
                "unmatched_pair_count": unmatched_pair_count,
                "expected_fold_count": len(expected_fold_tuple),
                "observed_fold_count": len(observed_folds),
                "full_fold_panel": bool(full_fold_panel),
                "observed_folds_json": json.dumps(list(observed_folds)),
                "missing_folds_json": json.dumps(list(missing_folds)),
                "match_rate": (
                    (candidate_pair_count - unmatched_pair_count) / candidate_pair_count
                ),
                "interpretation": RETROSPECTIVE_LABEL,
            }
        )
        result_rows.append(stats)
    results = pd.DataFrame(result_rows)
    results["holm_adjusted_p"] = np.nan
    for window_id, group in results.groupby("window_id", sort=True):
        adjusted = holm_adjust(
            dict(zip(group["contrast_id"], group["p_value_one_sided"], strict=True))
        )
        results.loc[group.index, "holm_adjusted_p"] = group["contrast_id"].map(adjusted)
    required_seeds = np.ceil(results["seed_count"] * 0.7).astype(int)
    required_folds = np.ceil(results["fold_count"] * 0.75).astype(int)
    results["passes_directional_consistency"] = results["consistent_seed_count"].ge(
        required_seeds
    ) & results["consistent_fold_count"].ge(required_folds)
    results["minimum_primary_pairs"] = int(minimum_primary_pairs)
    results["minimum_primary_releases"] = int(minimum_primary_releases)
    results["minimum_primary_sessions"] = int(minimum_primary_sessions)
    results["coverage_gate_passes"] = (
        results["pair_count"].ge(int(minimum_primary_pairs))
        & results["event_count"].ge(int(minimum_primary_releases))
        & results["session_count"].ge(int(minimum_primary_sessions))
    )
    results["passes_primary_gate"] = (
        results["window_role"].eq("primary")
        & results["contrast_role"].eq("primary")
        & results["mean_difference"].gt(0.0)
        & results["ci_95_lower"].gt(0.0)
        & results["holm_adjusted_p"].lt(0.05)
        & results["passes_directional_consistency"]
        & results["coverage_gate_passes"]
    )
    results["claim_scope"] = np.select(
        [
            results["window_role"].eq("primary") & ~results["coverage_gate_passes"],
            results["window_role"].eq("primary"),
            results["window_role"].eq("robustness"),
        ],
        [
            "retrospective_undercovered_descriptive_only",
            "retrospective_primary_exploratory",
            "retrospective_robustness_only",
        ],
        default="retrospective_descriptive_only",
    )
    return (
        plan,
        details,
        results.sort_values(["window_id", "contrast_id"], kind="stable").reset_index(
            drop=True
        ),
    )


def join_frozen_market_jumps(
    pair_frame: pd.DataFrame,
    market_jumps: pd.DataFrame,
    *,
    expected_tiers: Sequence[str] = EXPECTED_MARKET_JUMP_TIERS,
    tolerance_minutes: int = 30,
    require_all_tiers: bool = True,
) -> pd.DataFrame:
    """Exact-timestamp join to already-frozen market-jump rankings."""

    required = {"pair_id", "origin_time_utc", "session_id", "anomaly_tier"}
    missing = sorted(required - set(market_jumps.columns))
    if missing or market_jumps.empty:
        raise UnifiedAnalysisError(
            f"Market jumps are empty or missing columns: {missing}"
        )
    jumps = market_jumps.copy()
    jumps["origin_time_utc"] = _canonical_utc(
        jumps["origin_time_utc"], "market origin_time_utc"
    )
    jumps["anomaly_tier"] = jumps["anomaly_tier"].astype(str).str.strip().str.lower()
    tiers = tuple(map(str, expected_tiers))
    jumps = jumps[jumps["anomaly_tier"].isin(tiers)].copy()
    if jumps.empty or jumps["origin_time_utc"].duplicated().any():
        raise UnifiedAnalysisError(
            "Frozen market-jump origins must be non-empty and unique"
        )
    base = _base_pair_metadata(pair_frame, tolerance_minutes).rename(
        columns={"pair_id": "prediction_pair_id", "session_id": "prediction_session_id"}
    )
    joined = base.merge(
        jumps,
        left_on="effective_origin_utc",
        right_on="origin_time_utc",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_market"),
    )
    if (
        not joined["prediction_session_id"]
        .astype(str)
        .equals(joined["session_id"].astype(str))
    ):
        raise UnifiedAnalysisError(
            "Market-jump timestamp join session lineage differs from predictions"
        )
    observed = set(joined["anomaly_tier"].astype(str))
    if require_all_tiers and observed != set(tiers):
        raise UnifiedAnalysisError(
            "Market-jump all-tier coverage gate failed: "
            f"missing={sorted(set(tiers) - observed)}, extra={sorted(observed - set(tiers))}"
        )
    source_counts = (
        jumps.groupby("anomaly_tier", sort=True)["pair_id"].nunique().to_dict()
    )
    joined_counts = (
        joined.groupby("anomaly_tier", sort=True)["prediction_pair_id"]
        .nunique()
        .to_dict()
    )
    joined["tier_source_pair_count"] = (
        joined["anomaly_tier"].map(source_counts).astype(int)
    )
    joined["tier_joined_pair_count"] = (
        joined["anomaly_tier"].map(joined_counts).astype(int)
    )
    joined["all_tier_coverage_gate_passes"] = observed == set(tiers)
    joined["join_method"] = "exact_effective_origin_utc_to_frozen_origin_time_utc"
    return joined.sort_values(
        ["anomaly_tier", "effective_origin_utc", "prediction_pair_id"], kind="stable"
    ).reset_index(drop=True)


def build_market_jump_match_plan(
    pair_metrics: pd.DataFrame,
    market_jump_join: pd.DataFrame,
    *,
    state_columns: Sequence[str] = DEFAULT_CURRENT_STATE_COLUMNS,
    tolerance_minutes: int = 30,
    clock_caliper_minutes: int = 15,
    control_buffer_minutes: int = 30,
) -> pd.DataFrame:
    """Match frozen jumps to non-jumps on fold, weekday, clock, and state."""

    if int(tolerance_minutes) != 30:
        raise UnifiedAnalysisError("Market-jump RQ3 matching is frozen to 30m")
    if int(clock_caliper_minutes) < 0 or int(control_buffer_minutes) < 0:
        raise UnifiedAnalysisError("Market matching calipers must be nonnegative")
    if (
        market_jump_join.empty
        or not market_jump_join["all_tier_coverage_gate_passes"].astype(bool).all()
    ):
        raise UnifiedAnalysisError(
            "Market-jump all-tier timestamp coverage is not frozen as passing"
        )
    base = _base_pair_metadata(
        pair_metrics,
        tolerance_minutes,
        extra_columns=state_columns,
    )
    jump_keys = market_jump_join[
        [
            "fold",
            "prediction_pair_id",
            "anomaly_tier",
            "pair_id",
            "origin_time_utc",
        ]
    ].drop_duplicates(["fold", "prediction_pair_id"])
    jumps = base.merge(
        jump_keys,
        left_on=["fold", "pair_id"],
        right_on=["fold", "prediction_pair_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_market"),
    )
    if len(jumps) != len(jump_keys):
        raise UnifiedAnalysisError("Market-jump timestamp join lost prediction pairs")
    flagged = set(zip(jumps["fold"], jumps["pair_id"], strict=True))
    controls = base.loc[
        [
            (str(row.fold), str(row.pair_id)) not in flagged
            for row in base.itertuples(index=False)
        ]
    ].copy()
    jump_times = [pd.Timestamp(value) for value in jumps["effective_origin_utc"]]
    controls = controls.loc[
        controls["effective_origin_utc"].map(
            lambda value: min(
                abs((pd.Timestamp(value) - jump_time).total_seconds()) / 60.0
                for jump_time in jump_times
            )
            > float(control_buffer_minutes)
        )
    ].copy()
    if controls.empty:
        raise UnifiedAnalysisError("No non-jump controls survive the frozen buffer")

    combined = pd.concat(
        [jumps[list(state_columns)], controls[list(state_columns)]],
        ignore_index=True,
    )
    scales: dict[str, float] = {}
    scale_methods: dict[str, str] = {}
    for column in map(str, state_columns):
        values = combined[column].to_numpy(dtype=float)
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        method = "median_absolute_deviation_x_1.4826"
        if not math.isfinite(scale) or scale <= 1e-12:
            q25, q75 = np.quantile(values, (0.25, 0.75))
            scale = float((q75 - q25) / 1.349)
            method = "interquartile_range_div_1.349_fallback"
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(values, ddof=0))
            method = "population_std_fallback"
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
            method = "constant_feature_unit_fallback"
        scales[column] = scale
        scale_methods[column] = method
    candidate_map: dict[int, pd.DataFrame] = {}
    for index, jump in jumps.iterrows():
        origin = pd.Timestamp(jump["effective_origin_utc"])
        minute = origin.hour * 60.0 + origin.minute
        candidate_map[int(index)] = controls.loc[
            controls["fold"].astype(str).eq(str(jump["fold"]))
            & controls["effective_origin_utc"].map(
                lambda value: pd.Timestamp(value).weekday() == origin.weekday()
            )
            & controls["effective_origin_utc"].map(
                lambda value: _circular_minute_distance(
                    minute,
                    pd.Timestamp(value).hour * 60.0 + pd.Timestamp(value).minute,
                )
                <= float(clock_caliper_minutes)
            )
        ].copy()
    tier_order = {"high": 0, "primary": 1, "broad": 2}
    order = sorted(
        jumps.index,
        key=lambda index: (
            len(candidate_map[int(index)]),
            tier_order.get(str(jumps.loc[index, "anomaly_tier"]), 99),
            pd.Timestamp(jumps.loc[index, "effective_origin_utc"]),
            str(jumps.loc[index, "pair_id"]),
        ),
    )
    available = set(controls.index)
    rows: list[dict[str, Any]] = []
    for index in order:
        jump = jumps.loc[index]
        candidates = (
            candidate_map[int(index)]
            .loc[
                [
                    value
                    for value in candidate_map[int(index)].index
                    if value in available
                ]
            ]
            .copy()
        )
        matched_set_id = (
            "jump_match_"
            + hashlib.sha256(
                f"{jump['fold']}|{jump['pair_id']}|{jump['anomaly_tier']}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )
        common = {
            "matched_set_id": matched_set_id,
            "fold": str(jump["fold"]),
            "anomaly_tier": str(jump["anomaly_tier"]),
            "jump_pair_id": str(jump["pair_id"]),
            "jump_session_id": str(jump["session_id"]),
            "jump_effective_origin_utc": pd.Timestamp(jump["effective_origin_utc"]),
            "market_pair_id": str(jump["pair_id_market"]),
            "eligible_control_count": int(len(candidate_map[int(index)])),
            "available_control_count_at_assignment": int(len(candidates)),
        }
        for column in map(str, state_columns):
            common[f"jump_{column}"] = float(jump[column])
        if candidates.empty:
            for column in map(str, state_columns):
                common[f"control_{column}"] = math.nan
                common[f"standardized_difference_{column}"] = math.nan
            rows.append(
                {
                    **common,
                    "match_status": "unmatched",
                    "unmatched_reason": (
                        "no_same_fold_weekday_clock_control"
                        if len(candidate_map[int(index)]) == 0
                        else "eligible_controls_already_assigned"
                    ),
                    "control_pair_id": "",
                    "control_session_id": "",
                    "control_effective_origin_utc": pd.NaT,
                    "clock_distance_minutes": math.nan,
                    "match_distance": math.nan,
                }
            )
            continue
        distances = np.zeros(len(candidates), dtype=float)
        for column in map(str, state_columns):
            distances += (
                (candidates[column].to_numpy(float) - float(jump[column]))
                / scales[column]
            ) ** 2
        candidates["match_distance"] = np.sqrt(distances)
        chosen_index = candidates.sort_values(
            ["match_distance", "effective_origin_utc", "pair_id"], kind="stable"
        ).index[0]
        control = candidates.loc[chosen_index]
        available.remove(chosen_index)
        for column in map(str, state_columns):
            common[f"control_{column}"] = float(control[column])
            common[f"standardized_difference_{column}"] = float(
                (float(control[column]) - float(jump[column])) / scales[column]
            )
        jump_origin = pd.Timestamp(jump["effective_origin_utc"])
        control_origin = pd.Timestamp(control["effective_origin_utc"])
        rows.append(
            {
                **common,
                "match_status": "matched",
                "unmatched_reason": "",
                "control_pair_id": str(control["pair_id"]),
                "control_session_id": str(control["session_id"]),
                "control_effective_origin_utc": control_origin,
                "clock_distance_minutes": _circular_minute_distance(
                    jump_origin.hour * 60.0 + jump_origin.minute,
                    control_origin.hour * 60.0 + control_origin.minute,
                ),
                "match_distance": float(control["match_distance"]),
            }
        )
    result = pd.DataFrame(rows)
    result["matching_method"] = (
        "fold_weekday_clock_greedy_no_replacement_current_state_v1"
    )
    result["clock_caliper_minutes"] = int(clock_caliper_minutes)
    result["control_buffer_minutes"] = int(control_buffer_minutes)
    result["state_columns_json"] = json.dumps(
        list(map(str, state_columns)), separators=(",", ":")
    )
    result["state_scales_json"] = json.dumps(
        scales, sort_keys=True, separators=(",", ":")
    )
    result["state_scale_methods_json"] = json.dumps(
        scale_methods, sort_keys=True, separators=(",", ":")
    )
    result["distance_metric"] = "robust_scale_euclidean"
    result["non_jump_control_definition"] = (
        "not_exact_joined_jump_and_strictly_outside_joined_jump_plus_minus_buffer"
    )
    result["all_source_tiers_present"] = set(result["anomaly_tier"]) == set(
        EXPECTED_MARKET_JUMP_TIERS
    )
    return result.sort_values(
        ["anomaly_tier", "jump_effective_origin_utc", "jump_pair_id"], kind="stable"
    ).reset_index(drop=True)


def analyze_rq3_market_jumps(
    pair_metrics: pd.DataFrame,
    market_jump_join: pd.DataFrame,
    *,
    match_plan: pd.DataFrame | None = None,
    contrast_specs: Sequence[Mapping[str, Any]] = DEFAULT_RQ3_CONTRASTS,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED + 200_000,
    minimum_pairs: int = 30,
    minimum_sessions: int = 10,
    state_columns: Sequence[str] = DEFAULT_CURRENT_STATE_COLUMNS,
    clock_caliper_minutes: int = 15,
    control_buffer_minutes: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate jump-vs-matched-non-jump text-advantage differences-in-differences."""

    expected_plan = build_market_jump_match_plan(
        pair_metrics,
        market_jump_join,
        state_columns=state_columns,
        clock_caliper_minutes=clock_caliper_minutes,
        control_buffer_minutes=control_buffer_minutes,
    )
    plan = expected_plan if match_plan is None else match_plan.copy()
    required = {
        "matched_set_id",
        "fold",
        "anomaly_tier",
        "jump_pair_id",
        "jump_session_id",
        "control_pair_id",
        "control_session_id",
        "match_status",
    }
    missing = sorted(required - set(plan.columns))
    if missing or plan.empty:
        raise UnifiedAnalysisError(f"Market match plan is empty or missing: {missing}")
    if match_plan is not None:
        comparable = plan.copy()
        for column in (
            "jump_effective_origin_utc",
            "control_effective_origin_utc",
        ):
            if column in comparable:
                comparable[column] = pd.to_datetime(
                    comparable[column], errors="coerce", utc=True
                )
        comparable = comparable[expected_plan.columns].reset_index(drop=True)
        if not comparable.equals(expected_plan):
            raise UnifiedAnalysisError(
                "Supplied market-jump match plan differs from deterministic replay"
            )
    matched = plan[plan["match_status"].astype(str).eq("matched")].copy()
    if (
        matched.empty
        or matched.duplicated(["fold", "jump_pair_id", "control_pair_id"]).any()
    ):
        raise UnifiedAnalysisError("Market match plan is empty or duplicated")
    if matched.duplicated(["fold", "control_pair_id"]).any():
        raise UnifiedAnalysisError(
            "Market match plan reuses a non-jump control across jump pairs"
        )
    if not contrast_specs:
        raise UnifiedAnalysisError("Market-jump contrast specs must be non-empty")
    normalized_specs: list[dict[str, Any]] = []
    seen_contrast_ids: set[str] = set()
    for raw_spec in contrast_specs:
        spec = dict(raw_spec)
        missing_spec = sorted({"contrast_id", "focal_arm", "reference_arm"} - set(spec))
        if missing_spec:
            raise UnifiedAnalysisError(
                f"Market-jump contrast spec is missing keys: {missing_spec}"
            )
        contrast_id = str(spec["contrast_id"])
        if contrast_id in seen_contrast_ids:
            raise UnifiedAnalysisError(
                f"Duplicate market-jump contrast_id: {contrast_id}"
            )
        seen_contrast_ids.add(contrast_id)
        normalized_specs.append(spec)
    values = _pair_arm_value_map(pair_metrics, 30)
    detail_rows: list[dict[str, Any]] = []
    for spec in normalized_specs:
        focal = str(spec["focal_arm"])
        reference = str(spec["reference_arm"])
        for match in matched.itertuples(index=False):
            for seed in map(int, expected_seeds):
                keys = {
                    "jump_focal": (
                        match.fold,
                        seed,
                        match.jump_pair_id,
                        match.jump_session_id,
                        focal,
                    ),
                    "jump_reference": (
                        match.fold,
                        seed,
                        match.jump_pair_id,
                        match.jump_session_id,
                        reference,
                    ),
                    "control_focal": (
                        match.fold,
                        seed,
                        match.control_pair_id,
                        match.control_session_id,
                        focal,
                    ),
                    "control_reference": (
                        match.fold,
                        seed,
                        match.control_pair_id,
                        match.control_session_id,
                        reference,
                    ),
                }
                try:
                    resolved = {
                        name: float(values.loc[key, "target_mae"])
                        for name, key in keys.items()
                    }
                except KeyError as exc:
                    raise UnifiedAnalysisError(
                        f"Market match lacks paired prediction evidence: {keys}"
                    ) from exc
                jump_advantage = resolved["jump_reference"] - resolved["jump_focal"]
                control_advantage = (
                    resolved["control_reference"] - resolved["control_focal"]
                )
                detail_rows.append(
                    {
                        "matched_set_id": str(match.matched_set_id),
                        "fold": str(match.fold),
                        "seed": seed,
                        "session_id": str(match.jump_session_id),
                        "anomaly_tier": str(match.anomaly_tier),
                        "jump_pair_id": str(match.jump_pair_id),
                        "pair_id": str(match.jump_pair_id),
                        "jump_effective_origin_utc": pd.Timestamp(
                            match.jump_effective_origin_utc
                        ),
                        "control_pair_id": str(match.control_pair_id),
                        "control_session_id": str(match.control_session_id),
                        "control_effective_origin_utc": pd.Timestamp(
                            match.control_effective_origin_utc
                        ),
                        "match_distance": float(match.match_distance),
                        "contrast_id": str(spec["contrast_id"]),
                        "contrast_role": str(spec.get("contrast_role", "descriptive")),
                        "focal_arm": focal,
                        "reference_arm": reference,
                        "jump_text_advantage": jump_advantage,
                        "control_text_advantage": control_advantage,
                        "jump_increment": jump_advantage - control_advantage,
                        "difference_direction": "jump_minus_control_positive_means_focal_more_valuable",
                    }
                )
    details = pd.DataFrame(detail_rows)
    result_rows: list[dict[str, Any]] = []
    for tier_index, tier in enumerate(("all", "primary", "high")):
        tier_frame = (
            details
            if tier == "all"
            else details[details["anomaly_tier"].astype(str).eq(tier)]
        )
        for contrast_index, spec in enumerate(normalized_specs):
            contrast_id = str(spec["contrast_id"])
            group = tier_frame[
                tier_frame["contrast_id"].astype(str).eq(contrast_id)
            ].copy()
            plan_mask = (
                plan["anomaly_tier"].isin(EXPECTED_MARKET_JUMP_TIERS)
                if tier == "all"
                else plan["anomaly_tier"].eq(tier)
            )
            candidate_count = int(plan.loc[plan_mask, "jump_pair_id"].nunique())
            unmatched_count = int(
                plan.loc[
                    plan_mask & plan["match_status"].eq("unmatched"), "jump_pair_id"
                ].nunique()
            )
            pair_count = int(group["jump_pair_id"].nunique())
            session_count = int(group["session_id"].nunique())
            observed_seeds = set(group["seed"].astype(int))
            observed_folds = set(group["fold"].astype(str))
            full_seed_panel = observed_seeds == set(map(int, expected_seeds))
            full_fold_panel = observed_folds == set(map(str, expected_folds))
            full_seed_fold_panel = full_seed_panel and full_fold_panel
            ordered_observed_folds = tuple(
                fold for fold in map(str, expected_folds) if fold in observed_folds
            )
            if observed_folds != set(ordered_observed_folds):
                raise UnifiedAnalysisError(
                    f"Market-jump details contain unexpected folds: {sorted(observed_folds)}"
                )
            matched_tier_coverage = set(
                plan.loc[plan["match_status"].eq("matched"), "anomaly_tier"].astype(str)
            ) == set(EXPECTED_MARKET_JUMP_TIERS)
            coverage_passes = pair_count >= int(minimum_pairs) and session_count >= int(
                minimum_sessions
            )
            jump_mean = (
                float(group["jump_text_advantage"].mean())
                if not group.empty
                else math.nan
            )
            control_mean = (
                float(group["control_text_advantage"].mean())
                if not group.empty
                else math.nan
            )
            increment_mean = (
                float(group["jump_increment"].mean()) if not group.empty else math.nan
            )
            row: dict[str, Any] = {
                "research_question": "RQ3",
                "analysis_type": "market_jump_vs_matched_non_jump_text_advantage_did",
                "anomaly_tier": tier,
                "analysis_role": (
                    "retrospective_primary_all_tiers"
                    if tier == "all"
                    else "descriptive_primary_high"
                ),
                "contrast_id": contrast_id,
                "contrast_role": str(spec.get("contrast_role", "descriptive")),
                "focal_arm": str(spec["focal_arm"]),
                "reference_arm": str(spec["reference_arm"]),
                "pair_count": pair_count,
                "session_count": session_count,
                "seed_count": int(group["seed"].nunique()),
                "fold_count": int(group["fold"].nunique()),
                "mean_jump_text_advantage": jump_mean,
                "mean_control_text_advantage": control_mean,
                "mean_jump_increment": increment_mean,
                "candidate_pair_count": candidate_count,
                "unmatched_pair_count": unmatched_count,
                "match_rate": (
                    pair_count / candidate_count if candidate_count else math.nan
                ),
                "minimum_pairs": int(minimum_pairs),
                "minimum_sessions": int(minimum_sessions),
                "full_seed_panel": full_seed_panel,
                "full_fold_panel": full_fold_panel,
                "full_seed_fold_panel": full_seed_fold_panel,
                "all_tier_category_coverage_passes": bool(
                    plan["all_source_tiers_present"].astype(bool).all()
                ),
                "matched_tier_category_coverage_passes": matched_tier_coverage,
                "coverage_gate_passes": coverage_passes,
                "estimability_status": (
                    "estimable"
                    if tier == "all" and coverage_passes
                    else (
                        "not_estimable_due_to_coverage"
                        if tier == "all"
                        else "descriptive_severity_only"
                    )
                ),
                "difference_direction": "jump_minus_control_positive_means_focal_more_valuable",
                "interpretation": RETROSPECTIVE_LABEL,
            }
            if tier == "all" and coverage_passes:
                stats = _hierarchical_bootstrap_values(
                    group,
                    value_column="jump_increment",
                    expected_seeds=expected_seeds,
                    expected_folds=ordered_observed_folds,
                    iterations=iterations,
                    rng_seed=int(rng_seed) + tier_index * 1000 + contrast_index,
                    improvement_direction="positive",
                )
                for name, value in stats.items():
                    if name not in {"mean_difference", "pair_count", "session_count"}:
                        row[name] = value
                row["inference_permitted"] = True
            else:
                seed_means = group.groupby("seed", sort=True)["jump_increment"].mean()
                fold_means = group.groupby("fold", sort=True)["jump_increment"].mean()
                row.update(
                    {
                        "bootstrap_se": math.nan,
                        "ci_95_lower": math.nan,
                        "ci_95_upper": math.nan,
                        "p_value_one_sided": math.nan,
                        "p_value_two_sided": math.nan,
                        "consistent_seed_count": int((seed_means >= 0.0).sum()),
                        "consistent_fold_count": int((fold_means >= 0.0).sum()),
                        "bootstrap_iterations": 0,
                        "bootstrap_seed": math.nan,
                        "resampling_method": "descriptive_no_inference",
                        "seed_means_json": json.dumps(
                            seed_means.to_dict(), sort_keys=True, separators=(",", ":")
                        ),
                        "fold_means_json": json.dumps(
                            fold_means.to_dict(), sort_keys=True, separators=(",", ":")
                        ),
                        "inference_permitted": False,
                    }
                )
            result_rows.append(row)
    results = pd.DataFrame(result_rows)
    results["holm_adjusted_p"] = np.nan
    all_rows = results[results["anomaly_tier"].eq("all")]
    inferential_rows = all_rows[all_rows["inference_permitted"].astype(bool)]
    if not inferential_rows.empty:
        adjusted = holm_adjust(
            dict(
                zip(
                    inferential_rows["contrast_id"],
                    inferential_rows["p_value_one_sided"],
                    strict=True,
                )
            )
        )
        results.loc[inferential_rows.index, "holm_adjusted_p"] = inferential_rows[
            "contrast_id"
        ].map(adjusted)
    results["required_consistent_seed_count"] = np.ceil(
        results["seed_count"].astype(float) * 0.7
    ).astype(int)
    results["required_consistent_fold_count"] = np.ceil(
        results["fold_count"].astype(float) * 0.75
    ).astype(int)
    results["passes_directional_consistency"] = results["consistent_seed_count"].ge(
        results["required_consistent_seed_count"]
    ) & results["consistent_fold_count"].ge(results["required_consistent_fold_count"])
    results["passes_primary_gate"] = (
        results["anomaly_tier"].eq("all")
        & results["contrast_role"].eq("primary")
        & results["inference_permitted"].astype(bool)
        & results["mean_jump_increment"].gt(0.0)
        & results["ci_95_lower"].gt(0.0)
        & results["holm_adjusted_p"].lt(0.05)
        & results["passes_directional_consistency"]
    )
    results["claim_scope"] = np.select(
        [
            results["anomaly_tier"].eq("all") & ~results["coverage_gate_passes"],
            results["anomaly_tier"].eq("all"),
        ],
        [
            "retrospective_undercovered_descriptive_only",
            "retrospective_primary_exploratory",
        ],
        default="retrospective_descriptive_primary_high_only",
    )
    return (
        plan,
        details,
        results.sort_values(["anomaly_tier", "contrast_id"], kind="stable").reset_index(
            drop=True
        ),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise UnifiedAnalysisError(f"Existing analysis output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _input_row(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_unified_analysis(
    *,
    pair_metrics_path: str | Path,
    pair_metrics_sha256: str,
    prediction_manifest_path: str | Path,
    prediction_manifest_sha256: str,
    checkpoint_manifest_path: str | Path,
    checkpoint_manifest_sha256: str,
    scheduled_events_path: str | Path,
    scheduled_events_sha256: str,
    market_jump_path: str | Path,
    market_jump_sha256: str,
    output_dir: str | Path,
    expected_arms: Sequence[str] = CANONICAL_ARMS,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    expected_tolerances: Sequence[int] = CANONICAL_TOLERANCES,
    expected_arms_by_tolerance: Mapping[int, Sequence[str]] | None = None,
    rq1_rq2_contrast_specs: Sequence[Mapping[str, Any]] = DEFAULT_RQ1_RQ2_CONTRASTS,
    persistence_comparison_specs: Sequence[
        Mapping[str, Any]
    ] = DEFAULT_PERSISTENCE_COMPARISONS,
    rq3_contrast_specs: Sequence[Mapping[str, Any]] = DEFAULT_RQ3_CONTRASTS,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ordinary_minimum_distance_minutes: int = 60,
    scheduled_clock_caliper_minutes: int | None = 15,
    scheduled_minimum_primary_pairs: int = 30,
    scheduled_minimum_primary_releases: int = 20,
    scheduled_minimum_primary_sessions: int = 10,
    market_jump_minimum_pairs: int = 30,
    market_jump_minimum_sessions: int = 10,
    market_jump_state_columns: Sequence[str] = DEFAULT_CURRENT_STATE_COLUMNS,
    market_jump_clock_caliper_minutes: int = 15,
    market_jump_control_buffer_minutes: int = 30,
    required_market_root: str | Path | None = FROZEN_MARKET_JUMP_ROOT,
) -> dict[str, Path]:
    """Validate frozen inputs, run all analyses, and write a hashed bundle."""

    pair_path = _verify_file(pair_metrics_path, pair_metrics_sha256, "pair metrics")
    prediction_path = _verify_file(
        prediction_manifest_path, prediction_manifest_sha256, "prediction manifest"
    )
    checkpoint_path = _verify_file(
        checkpoint_manifest_path, checkpoint_manifest_sha256, "checkpoint manifest"
    )
    scheduled_path = _verify_file(
        scheduled_events_path, scheduled_events_sha256, "scheduled events"
    )
    jump_path = _verify_file(market_jump_path, market_jump_sha256, "market jumps")
    if required_market_root is not None:
        required = Path(required_market_root).resolve()
        expected_jump = (required / FROZEN_MARKET_JUMP_RELATIVE_PATH).resolve()
        if jump_path.resolve() != expected_jump:
            raise UnifiedAnalysisError(
                f"Market-jump input must be the frozen source {expected_jump}; got {jump_path.resolve()}"
            )

    arms_by_tolerance = _normalize_arms_by_tolerance(
        expected_arms, expected_tolerances, expected_arms_by_tolerance
    )
    evidence = validate_paired_evidence(
        pair_path,
        prediction_path,
        checkpoint_path,
        expected_arms=expected_arms,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        expected_tolerances=expected_tolerances,
        expected_arms_by_tolerance=expected_arms_by_tolerance,
    )
    scheduled_events = pd.read_csv(scheduled_path)
    market_jumps = pd.read_csv(jump_path)
    rq1_rq2 = analyze_rq1_rq2(
        evidence.pair_metrics,
        contrast_specs=rq1_rq2_contrast_specs,
        tolerance_minutes=5,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        iterations=bootstrap_iterations,
        rng_seed=bootstrap_seed,
    )
    persistence_secondary = analyze_persistence_secondary(
        evidence.pair_metrics,
        comparison_specs=persistence_comparison_specs,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        iterations=bootstrap_iterations,
        rng_seed=int(bootstrap_seed) + 50_000,
    )
    match_plan, scheduled_details, scheduled_results = analyze_rq3_scheduled(
        evidence.pair_metrics,
        scheduled_events,
        contrast_specs=rq3_contrast_specs,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        iterations=bootstrap_iterations,
        rng_seed=int(bootstrap_seed) + 100_000,
        ordinary_minimum_distance_minutes=ordinary_minimum_distance_minutes,
        clock_caliper_minutes=scheduled_clock_caliper_minutes,
        minimum_primary_pairs=scheduled_minimum_primary_pairs,
        minimum_primary_releases=scheduled_minimum_primary_releases,
        minimum_primary_sessions=scheduled_minimum_primary_sessions,
    )
    market_join = join_frozen_market_jumps(evidence.pair_metrics, market_jumps)
    market_match_plan, market_details, market_results = analyze_rq3_market_jumps(
        evidence.pair_metrics,
        market_join,
        contrast_specs=rq3_contrast_specs,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        iterations=bootstrap_iterations,
        rng_seed=int(bootstrap_seed) + 200_000,
        minimum_pairs=market_jump_minimum_pairs,
        minimum_sessions=market_jump_minimum_sessions,
        state_columns=market_jump_state_columns,
        clock_caliper_minutes=market_jump_clock_caliper_minutes,
        control_buffer_minutes=market_jump_control_buffer_minutes,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "rq1_rq2_results": destination / "rq1_rq2_results.csv",
        "rq1_rq2_persistence_secondary": destination
        / "rq1_rq2_persistence_secondary.csv",
        "rq3_scheduled_match_plan": destination / "rq3_scheduled_match_plan.csv",
        "rq3_scheduled_details": destination / "rq3_scheduled_matched_differences.csv",
        "rq3_scheduled_results": destination / "rq3_scheduled_results.csv",
        "rq3_market_jump_join": destination / "rq3_market_jump_timestamp_join.csv",
        "rq3_market_jump_match_plan": destination / "rq3_market_jump_match_plan.csv",
        "rq3_market_jump_details": destination
        / "rq3_market_jump_matched_differences.csv",
        "rq3_market_jump_results": destination / "rq3_market_jump_results.csv",
        "analysis_summary": destination / "analysis_summary.json",
        "analysis_manifest": destination / "analysis_artifact_manifest.json",
    }
    for role, frame in (
        ("rq1_rq2_results", rq1_rq2),
        ("rq1_rq2_persistence_secondary", persistence_secondary),
        ("rq3_scheduled_match_plan", match_plan),
        ("rq3_scheduled_details", scheduled_details),
        ("rq3_scheduled_results", scheduled_results),
        ("rq3_market_jump_join", market_join),
        ("rq3_market_jump_match_plan", market_match_plan),
        ("rq3_market_jump_details", market_details),
        ("rq3_market_jump_results", market_results),
    ):
        _write_csv(frame, paths[role])
    market_all = market_results[market_results["anomaly_tier"].eq("all")]
    market_estimability_by_contrast = {
        str(row.contrast_id): str(row.estimability_status)
        for row in market_all.itertuples(index=False)
    }
    market_estimability_values = set(market_estimability_by_contrast.values())
    if len(market_estimability_values) != 1:
        raise UnifiedAnalysisError(
            "All-tier market-jump estimability must agree across contrasts"
        )
    summary = {
        "schema_version": 1,
        "experiment": "news_first_vol_film_nolp_10seed_unified",
        "interpretation": RETROSPECTIVE_LABEL,
        "confirmatory": False,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_method": "seed_then_fold_then_paired_session_cluster",
        "rq1_rq2_primary_tolerance_minutes": 5,
        "rq1_rq2_primary_estimand": "mean_seed_fold_log_mean_mae_focal_over_mean_mae_reference",
        "rq1_rq2_persistence_secondary_estimand": "mean_seed_fold_log_mean_arm_mae_over_mean_persistence_mae",
        "rq1_rq2_persistence_secondary_family_sizes": {
            str(question): int(count)
            for question, count in persistence_secondary.groupby(
                "research_question", sort=True
            )["comparison_id"]
            .nunique()
            .items()
        },
        "arms": list(map(str, expected_arms)),
        "seeds": list(map(int, expected_seeds)),
        "folds": list(map(str, expected_folds)),
        "tolerances_minutes": list(map(int, expected_tolerances)),
        "arms_by_tolerance": {
            str(tolerance): list(arms_by_tolerance[int(tolerance)])
            for tolerance in map(int, expected_tolerances)
        },
        "scheduled_windows": {
            key: {
                "lower_minutes": value[0],
                "upper_minutes": value[1],
                "role": value[2],
            }
            for key, value in SCHEDULED_WINDOWS.items()
        },
        "market_jump_join_method": "exact_effective_origin_utc_to_frozen_origin_time_utc",
        "market_jump_expected_tiers": list(EXPECTED_MARKET_JUMP_TIERS),
        "market_jump_matching": {
            "method": "fold_weekday_clock_greedy_no_replacement_current_state_v1",
            "clock_caliper_minutes": int(market_jump_clock_caliper_minutes),
            "control_buffer_minutes": int(market_jump_control_buffer_minutes),
            "state_columns": list(map(str, market_jump_state_columns)),
            "distance_metric": "robust_scale_euclidean",
        },
        "market_jump_inference_scope": "all_tier_pool_coverage_gated_only",
        "market_jump_estimability_status": next(iter(market_estimability_values)),
        "market_jump_estimability_by_contrast": market_estimability_by_contrast,
        "market_jump_matched_pair_count": int(market_all["pair_count"].max()),
        "market_jump_matched_session_count": int(market_all["session_count"].max()),
        "market_jump_coverage_gate_fields": ["matched_pair_count", "session_count"],
        "market_jump_fold_and_tier_completeness": "audit_only_not_admission_gates",
        "market_jump_primary_high_scope": "descriptive_only",
        "rq1_rq2_gate_pass_count": int(rq1_rq2["passes_full_gate"].sum()),
        "rq1_rq2_persistence_secondary_gate_pass_count": int(
            persistence_secondary["passes_secondary_gate"].sum()
        ),
        "rq3_primary_gate_pass_count": int(
            scheduled_results["passes_primary_gate"].sum()
        ),
        "rq3_market_jump_primary_gate_pass_count": int(
            market_results["passes_primary_gate"].sum()
        ),
        "pair_metric_row_count": int(len(evidence.pair_metrics)),
        "job_count": int(evidence.pair_metrics["job_id"].nunique()),
    }
    _atomic_write_text(
        paths["analysis_summary"],
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    inputs = [
        _input_row("pair_metrics_source", pair_path),
        _input_row("prediction_manifest_source", prediction_path),
        _input_row("checkpoint_manifest_source", checkpoint_path),
        _input_row("scheduled_events_source", scheduled_path),
        _input_row("market_jump_source", jump_path),
    ]
    artifacts = [
        _input_row(role, paths[role]) for role in paths if role != "analysis_manifest"
    ]
    manifest = {
        "schema_version": 1,
        "interpretation": RETROSPECTIVE_LABEL,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    _atomic_write_text(
        paths["analysis_manifest"],
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


__all__ = [
    "CANONICAL_ARMS",
    "CANONICAL_ARMS_BY_TOLERANCE",
    "CANONICAL_FOLDS",
    "CANONICAL_SEEDS",
    "CANONICAL_TOLERANCES",
    "DEFAULT_RQ1_RQ2_CONTRASTS",
    "DEFAULT_RQ3_CONTRASTS",
    "DEFAULT_CURRENT_STATE_COLUMNS",
    "DEFAULT_PERSISTENCE_COMPARISONS",
    "EXPECTED_MARKET_JUMP_TIERS",
    "FROZEN_MARKET_JUMP_ROOT",
    "RETROSPECTIVE_LABEL",
    "SCHEDULED_WINDOWS",
    "UnifiedAnalysisError",
    "ValidatedEvidence",
    "analyze_rq1_rq2",
    "analyze_persistence_secondary",
    "analyze_rq3_market_jumps",
    "analyze_rq3_scheduled",
    "apply_holm_and_consistency_gate",
    "build_market_jump_match_plan",
    "build_ordinary_match_plan",
    "holm_adjust",
    "join_frozen_market_jumps",
    "run_unified_analysis",
    "seed_fold_session_paired_bootstrap",
    "seed_fold_session_persistence_bootstrap",
    "sha256_file",
    "validate_frozen_artifact_manifest",
    "validate_paired_evidence",
]
