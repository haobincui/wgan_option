"""Fold-only inference for the News-first Vol label-reliability experiment.

The capacity, learning rate, text treatment, mask and residual parameterisation
are fixed by the orchestrator.  This module therefore compares only the three
label treatments B/C/D with the unchanged arm A.  Selection consumes inner-fold
validation pair metrics; Q3 is deliberately handled by a separate, allowlisted
code path after selection and checkpoint hashes have been frozen.  There is no
Q4 entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BASELINE_ARM = "A"
CANDIDATE_ARMS = ("B", "C", "D")
ONE_SE_PRIORITY = ("C", "B", "D")
FOLD_IDS = ("F1", "F2", "F3", "F4")
SEEDS = (42, 202, 404)
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260820
Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
Q4_START_UTC = Q3_END_UTC
HIGH_RELIABILITY_THRESHOLDS = {
    "c_i": 16,
    "q_i": 8,
    "m_i": 4,
    "v_i": 0.8,
    "h_i": 0.8,
    "reliability_score": 0.5,
}


class LabelReliabilityAnalysisError(ValueError):
    """Raised when inferential inputs violate the frozen experiment contract."""


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_positive(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result <= 0.0).any():
        raise LabelReliabilityAnalysisError(f"{label} must be finite and positive")
    return result


def reliability_score_with_tau(pair_components: pd.DataFrame, tau_u_q75: float) -> pd.Series:
    """Compute frozen R for evaluation pairs using training-fold tau."""

    tau = float(tau_u_q75)
    if not math.isfinite(tau) or tau <= 0.0:
        raise LabelReliabilityAnalysisError("tau_u_q75 must be finite and positive")
    required = {"c_i", "q_i", "m_i", "u_i", "u_estimable", "v_i", "h_i"}
    missing = sorted(required - set(pair_components.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"Reliability components are missing: {missing}"
        )
    numeric = {
        field: pd.to_numeric(pair_components[field], errors="coerce").astype(float)
        for field in ("c_i", "q_i", "m_i", "u_i", "v_i", "h_i")
    }
    for field in ("c_i", "q_i", "m_i"):
        values = numeric[field].to_numpy()
        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise LabelReliabilityAnalysisError(
                f"{field} must be finite nonnegative integers"
            )
    for field in ("v_i", "h_i"):
        values = numeric[field].to_numpy()
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise LabelReliabilityAnalysisError(f"{field} must be finite in [0,1]")
    estimable = pair_components["u_estimable"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    finite_u = np.isfinite(numeric["u_i"])
    if bool((estimable & (~finite_u | numeric["u_i"].le(0.0))).any()):
        raise LabelReliabilityAnalysisError("Estimable u_i must be finite and positive")
    if bool((~estimable & finite_u).any()):
        raise LabelReliabilityAnalysisError("Unestimable u_i must be NaN")
    s = np.minimum(1.0, np.sqrt(numeric["c_i"] / 16.0))
    d = np.minimum(1.0, np.sqrt(numeric["q_i"] / 8.0))
    e = np.minimum(1.0, np.sqrt(numeric["m_i"] / 4.0))
    b = pd.Series(0.0, index=pair_components.index)
    b.loc[estimable] = 1.0 / (1.0 + (numeric["u_i"].loc[estimable] / tau) ** 2)
    score = (s * d * e * b * numeric["v_i"] * numeric["h_i"]) ** (1.0 / 6.0)
    if not np.isfinite(score.to_numpy()).all() or (score < 0.0).any() or (score > 1.0).any():
        raise LabelReliabilityAnalysisError("Computed reliability score escaped [0,1]")
    return score.astype(float)


def attach_inner_fold_reliability(
    pair_metrics: pd.DataFrame,
    pair_components: pd.DataFrame,
    tau_by_fold: Mapping[str, float],
    fold_intervals: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Join pre-Q3 validation components without estimating tau on validation."""

    required = {"pair_id", "effective_origin_utc", "tolerance_minutes"}
    missing = sorted(required - set(pair_components.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"Pair components are missing join columns: {missing}"
        )
    components = pair_components.copy()
    components = components[
        pd.to_numeric(components["tolerance_minutes"], errors="coerce") == 5
    ].copy()
    times = pd.to_datetime(components["effective_origin_utc"], errors="coerce", utc=True)
    if times.isna().any() or bool((times >= pd.Timestamp(Q3_START_UTC)).any()):
        raise LabelReliabilityAnalysisError("Inner-fold reliability must be pre-Q3")
    parts = []
    for fold in FOLD_IDS:
        if fold not in tau_by_fold or fold not in fold_intervals:
            raise LabelReliabilityAnalysisError(f"Missing reliability contract for {fold}")
        start, end = map(pd.Timestamp, fold_intervals[fold])
        selected = components.loc[(times >= start) & (times < end)].copy()
        if selected["pair_id"].astype(str).duplicated().any():
            raise LabelReliabilityAnalysisError(
                f"Reliability components are not pair unique in {fold}"
            )
        selected["fold_id"] = fold
        selected["reliability_score"] = reliability_score_with_tau(
            selected, float(tau_by_fold[fold])
        )
        parts.append(selected)
    component_panel = pd.concat(parts, ignore_index=True)
    join_columns = [
        "fold_id",
        "pair_id",
        "c_i",
        "q_i",
        "m_i",
        "u_i",
        "u_estimable",
        "v_i",
        "h_i",
        "reliability_score",
    ]
    output = pair_metrics.merge(
        component_panel[join_columns],
        on=["fold_id", "pair_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not output["_merge"].eq("both").all():
        raise LabelReliabilityAnalysisError(
            "Inner-fold pair metrics lack reliability components"
        )
    return output.drop(columns="_merge")


def high_reliability_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the predeclared secondary-panel membership mask."""

    missing = sorted(set(HIGH_RELIABILITY_THRESHOLDS) - set(frame.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"High-reliability panel is missing columns: {missing}"
        )
    mask = pd.Series(True, index=frame.index)
    for field, threshold in HIGH_RELIABILITY_THRESHOLDS.items():
        values = pd.to_numeric(frame[field], errors="coerce")
        mask &= values.ge(float(threshold))
    return mask


def _validate_pair_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "arm_id",
        "fold_id",
        "seed",
        "pair_id",
        "session_id",
        "model_mae",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"Pair metrics are missing required columns: {missing}"
        )
    if frame.empty:
        raise LabelReliabilityAnalysisError("Pair metrics are empty")
    result = frame.copy()
    for column in ("arm_id", "fold_id", "pair_id", "session_id"):
        result[column] = result[column].fillna("").astype(str).str.strip()
        if result[column].eq("").any():
            raise LabelReliabilityAnalysisError(f"{column} must be non-empty")
    result["seed"] = pd.to_numeric(result["seed"], errors="coerce")
    if result["seed"].isna().any() or not np.equal(
        result["seed"], np.floor(result["seed"])
    ).all():
        raise LabelReliabilityAnalysisError("seed must be an integer")
    result["seed"] = result["seed"].astype(int)
    result["model_mae"] = _finite_positive(result["model_mae"], "model_mae")
    keys = ["arm_id", "fold_id", "seed", "pair_id"]
    if result.duplicated(keys).any():
        raise LabelReliabilityAnalysisError(
            "Pair metrics must contain one row per arm/fold/seed/pair"
        )
    if not set(result["arm_id"]).issubset({BASELINE_ARM, *CANDIDATE_ARMS}):
        raise LabelReliabilityAnalysisError("Pair metrics contain an unknown arm")
    return result


def _paired_cells(
    frame: pd.DataFrame, candidate_arm: str
) -> dict[tuple[int, str], pd.DataFrame]:
    subset = frame[frame["arm_id"].isin([BASELINE_ARM, candidate_arm])]
    cells: dict[tuple[int, str], pd.DataFrame] = {}
    observed = set(zip(subset["seed"], subset["fold_id"]))
    expected = {(seed, fold) for seed in SEEDS for fold in FOLD_IDS}
    if observed != expected:
        raise LabelReliabilityAnalysisError(
            f"{candidate_arm} comparison cells drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for seed, fold in sorted(expected):
        cell = subset[(subset["seed"] == seed) & (subset["fold_id"] == fold)]
        baseline = cell[cell["arm_id"] == BASELINE_ARM].set_index("pair_id")
        candidate = cell[cell["arm_id"] == candidate_arm].set_index("pair_id")
        if set(baseline.index) != set(candidate.index):
            raise LabelReliabilityAnalysisError(
                f"{candidate_arm}/{seed}/{fold} is not evaluated on one paired panel"
            )
        paired = baseline[["session_id", "model_mae"]].rename(
            columns={"session_id": "baseline_session", "model_mae": "baseline_mae"}
        ).join(
            candidate[["session_id", "model_mae"]].rename(
                columns={
                    "session_id": "candidate_session",
                    "model_mae": "candidate_mae",
                }
            ),
            how="inner",
        )
        if not paired["baseline_session"].equals(paired["candidate_session"]):
            raise LabelReliabilityAnalysisError(
                f"{candidate_arm}/{seed}/{fold} session lineage differs by arm"
            )
        paired = paired.rename(columns={"baseline_session": "session_id"}).drop(
            columns="candidate_session"
        )
        if paired.empty:
            raise LabelReliabilityAnalysisError(
                f"{candidate_arm}/{seed}/{fold} paired panel is empty"
            )
        cells[(seed, fold)] = paired.reset_index()
    return cells


def _cell_log_ratio(cell: pd.DataFrame) -> float:
    return float(
        math.log(float(cell["candidate_mae"].mean()) / float(cell["baseline_mae"].mean()))
    )


def seed_fold_session_bootstrap(
    frame: pd.DataFrame,
    candidate_arm: str,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare one arm with A using paired seed -> fold -> session resampling.

    Each draw samples three seeds with replacement, then four folds within each
    sampled seed, then paired CME-session clusters within each sampled cell.
    Cell log ratios receive equal weight, preserving the frozen estimand.
    """

    candidate = str(candidate_arm).strip().upper()
    if candidate not in CANDIDATE_ARMS:
        raise LabelReliabilityAnalysisError(
            f"candidate_arm must be one of {CANDIDATE_ARMS}"
        )
    if int(iterations) < 2:
        raise LabelReliabilityAnalysisError("iterations must be at least two")
    clean = _validate_pair_metrics(frame)
    cells = _paired_cells(clean, candidate)
    cell_ratios = {
        key: _cell_log_ratio(value) for key, value in cells.items()
    }
    point = float(np.mean(list(cell_ratios.values())))
    fold_means = {
        fold: float(np.mean([cell_ratios[(s, fold)] for s in SEEDS]))
        for fold in FOLD_IDS
    }
    seed_means = {
        sampled_seed: float(
            np.mean([cell_ratios[(sampled_seed, fold)] for fold in FOLD_IDS])
        )
        for sampled_seed in SEEDS
    }

    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    seed_values = np.asarray(SEEDS, dtype=np.int64)
    fold_values = np.asarray(FOLD_IDS, dtype=object)
    for draw_index in range(int(iterations)):
        draw_cells: list[float] = []
        for sampled_seed in rng.choice(seed_values, size=len(SEEDS), replace=True):
            for sampled_fold in rng.choice(
                fold_values, size=len(FOLD_IDS), replace=True
            ):
                cell = cells[(int(sampled_seed), str(sampled_fold))]
                sessions = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
                sampled_sessions = rng.choice(
                    sessions, size=len(sessions), replace=True
                )
                # Repeated sampled clusters must contribute repeatedly.  Building
                # a list avoids a many-to-many merge that can square duplicates.
                blocks = [
                    cell[cell["session_id"] == session] for session in sampled_sessions
                ]
                sampled = pd.concat(blocks, ignore_index=True)
                draw_cells.append(_cell_log_ratio(sampled))
        draws[draw_index] = float(np.mean(draw_cells))

    lower, upper = np.quantile(draws, [0.025, 0.975])
    one_sided_p = float((1 + np.count_nonzero(draws >= 0.0)) / (len(draws) + 1))
    return {
        "candidate_arm": candidate,
        "baseline_arm": BASELINE_ARM,
        "mean_log_ratio": point,
        "geometric_mae_ratio": float(math.exp(point)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_one_sided": one_sided_p,
        "nonworse_fold_count": int(sum(value <= 0.0 for value in fold_means.values())),
        "nonworse_seed_count": int(sum(value <= 0.0 for value in seed_means.values())),
        "fold_log_ratios": fold_means,
        "seed_log_ratios": {str(key): value for key, value in seed_means.items()},
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "paired_pair_count_min": int(min(len(cell) for cell in cells.values())),
        "paired_session_count_min": int(
            min(cell["session_id"].nunique() for cell in cells.values())
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values for one named family."""

    if not p_values:
        return {}
    items = sorted((str(key), float(value)) for key, value in p_values.items())
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for _, value in items):
        raise LabelReliabilityAnalysisError("Holm p-values must be finite in [0, 1]")
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * raw))
        adjusted[name] = running
    return adjusted


def build_arm_comparisons(
    pair_metrics: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Build the three predeclared comparisons and their Holm decisions."""

    rows = [
        seed_fold_session_bootstrap(
            pair_metrics,
            arm,
            iterations=int(iterations),
            seed=int(seed) + index,
        )
        for index, arm in enumerate(CANDIDATE_ARMS)
    ]
    adjusted = holm_adjust(
        {row["candidate_arm"]: row["p_value_one_sided"] for row in rows}
    )
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        arm = str(row["candidate_arm"])
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"fold_log_ratios", "seed_log_ratios"}
        }
        flat["fold_log_ratios_json"] = json.dumps(
            row["fold_log_ratios"], sort_keys=True, separators=(",", ":")
        )
        flat["seed_log_ratios_json"] = json.dumps(
            row["seed_log_ratios"], sort_keys=True, separators=(",", ":")
        )
        flat["holm_adjusted_p"] = adjusted[arm]
        flat["passes_gate"] = bool(
            float(row["mean_log_ratio"]) < 0.0
            and float(row["ci_upper"]) < 0.0
            and adjusted[arm] < 0.05
            and int(row["nonworse_fold_count"]) >= 3
            and int(row["nonworse_seed_count"]) >= 2
        )
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows).sort_values("candidate_arm").reset_index(drop=True)


def select_reliability_winner(comparisons: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen gate and C>B>D one-standard-error tie-break."""

    required = {
        "candidate_arm",
        "mean_log_ratio",
        "bootstrap_se",
        "ci_upper",
        "holm_adjusted_p",
        "nonworse_fold_count",
        "nonworse_seed_count",
        "passes_gate",
    }
    missing = sorted(required - set(comparisons.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"Comparison table is missing columns: {missing}"
        )
    table = comparisons.copy()
    if set(table["candidate_arm"].astype(str)) != set(CANDIDATE_ARMS) or len(table) != 3:
        raise LabelReliabilityAnalysisError("Selection requires exactly B/C/D versus A")
    passing = table[table["passes_gate"].astype(bool)].copy()
    base = {
        "schema_version": 1,
        "selection_panel": "four_inner_validation_folds",
        "baseline_arm": BASELINE_ARM,
        "candidate_arms": list(CANDIDATE_ARMS),
        "one_se_priority": list(ONE_SE_PRIORITY),
        "q3_used_for_selection": False,
        "q4_accessed": False,
        "comparison_sha256": _payload_sha256(
            table.sort_values("candidate_arm").to_dict(orient="records")
        ),
    }
    if passing.empty:
        result = {
            **base,
            "status": "terminal_no_winner",
            "selected_arm": None,
            "reason": "no_candidate_passed_the_predeclared_gate",
            "eligible_one_se_arms": [],
        }
    else:
        passing["mean_log_ratio"] = pd.to_numeric(
            passing["mean_log_ratio"], errors="raise"
        )
        passing["bootstrap_se"] = pd.to_numeric(
            passing["bootstrap_se"], errors="raise"
        )
        best_row = passing.sort_values(
            ["mean_log_ratio", "candidate_arm"], kind="stable"
        ).iloc[0]
        cutoff = float(best_row["mean_log_ratio"]) + float(best_row["bootstrap_se"])
        eligible = set(
            passing.loc[passing["mean_log_ratio"] <= cutoff, "candidate_arm"].astype(str)
        )
        winner = next(arm for arm in ONE_SE_PRIORITY if arm in eligible)
        result = {
            **base,
            "status": "selected",
            "selected_arm": winner,
            "best_arm": str(best_row["candidate_arm"]),
            "best_mean_log_ratio": float(best_row["mean_log_ratio"]),
            "best_bootstrap_se": float(best_row["bootstrap_se"]),
            "one_se_cutoff": cutoff,
            "eligible_one_se_arms": [
                arm for arm in ONE_SE_PRIORITY if arm in eligible
            ],
        }
    result["selection_payload_sha256"] = _payload_sha256(result)
    return result


def validate_selection_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the immutable selection's content-addressed self hash."""

    result = dict(payload)
    observed = str(result.pop("selection_payload_sha256", ""))
    expected = _payload_sha256(result)
    if observed != expected:
        raise LabelReliabilityAnalysisError("Selection payload SHA256 mismatch")
    result["selection_payload_sha256"] = observed
    return result


def apply_retention_validity_gate(
    comparisons: pd.DataFrame, profile_manifest: pd.DataFrame, *, tolerance_minutes: int = 5
) -> pd.DataFrame:
    """Make B/D ineligible when any fold misses 60% pairs or 80% sessions."""

    required = {
        "arm_id",
        "tolerance_minutes",
        "fold_id",
        "pair_retention_fraction",
        "session_retention_fraction",
    }
    missing = sorted(required - set(profile_manifest.columns))
    if missing:
        raise LabelReliabilityAnalysisError(
            f"Profile manifest is missing retention columns: {missing}"
        )
    result = comparisons.copy()
    invalid: set[str] = set()
    selected = profile_manifest[
        pd.to_numeric(profile_manifest["tolerance_minutes"], errors="coerce")
        == int(tolerance_minutes)
    ]
    for arm in ("B", "D"):
        arm_rows = selected[selected["arm_id"].astype(str) == arm]
        if set(arm_rows["fold_id"].astype(str)) != set(FOLD_IDS):
            invalid.add(arm)
            continue
        pair_retention = pd.to_numeric(
            arm_rows["pair_retention_fraction"], errors="coerce"
        )
        session_retention = pd.to_numeric(
            arm_rows["session_retention_fraction"], errors="coerce"
        )
        if (
            pair_retention.isna().any()
            or session_retention.isna().any()
            or bool((pair_retention < 0.60).any())
            or bool((session_retention < 0.80).any())
        ):
            invalid.add(arm)
    result["retention_gate_valid_all_folds"] = ~result[
        "candidate_arm"
    ].astype(str).isin(invalid)
    result.loc[
        ~result["retention_gate_valid_all_folds"], "passes_gate"
    ] = False
    return result


def validate_q3_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed unless predictions are exclusively inside Q3."""

    if "effective_origin_utc" not in frame.columns:
        raise LabelReliabilityAnalysisError(
            "Q3 predictions require effective_origin_utc"
        )
    timestamps = pd.to_datetime(frame["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise LabelReliabilityAnalysisError("Q3 predictions contain invalid timestamps")
    start = pd.Timestamp(Q3_START_UTC)
    end = pd.Timestamp(Q3_END_UTC)
    if not ((timestamps >= start) & (timestamps < end)).all():
        raise LabelReliabilityAnalysisError(
            "Prediction frame is not Q3-only; Q4 and earlier folds are forbidden"
        )
    return frame.copy()


def q3_checkpoint_allowlist(
    jobs: Sequence[Mapping[str, Any]], selected_arm: str
) -> list[dict[str, Any]]:
    """Allow only A/winner F4 checkpoints from completed conditional stages."""

    winner = str(selected_arm).strip().upper()
    if winner not in CANDIDATE_ARMS:
        raise LabelReliabilityAnalysisError("A valid selected arm is required for Q3")
    allowed_stages = {
        "stage1_regression_05m",
        "stage2_regression_30m",
        "stage3_wgan_05m",
    }
    rows: list[dict[str, Any]] = []
    for raw in jobs:
        job = dict(raw)
        if str(job.get("fold_id")) != "F4":
            continue
        if str(job.get("arm_id")) not in {BASELINE_ARM, winner}:
            continue
        if str(job.get("stage_id")) not in allowed_stages:
            raise LabelReliabilityAnalysisError("Unknown stage in Q3 checkpoint set")
        checkpoint = str(job.get("best_learned_checkpoint_path", "")).strip()
        digest = str(job.get("best_learned_checkpoint_sha256", "")).strip()
        if not checkpoint or len(digest) != 64:
            raise LabelReliabilityAnalysisError(
                f"F4 checkpoint lineage is incomplete for {job.get('job_id')}"
            )
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "stage_id": str(job["stage_id"]),
                "model_family": str(job["model_family"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "arm_id": str(job["arm_id"]),
                "fold_id": "F4",
                "seed": int(job["seed"]),
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": digest,
            }
        )
    expected = 2 * 3 * 3
    if len(rows) != expected:
        raise LabelReliabilityAnalysisError(
            f"Q3 checkpoint allowlist must contain {expected} rows, got {len(rows)}"
        )
    return sorted(rows, key=lambda row: row["job_id"])


def run_label_reliability_analysis(
    experiment_root: str | Path,
    *,
    pair_metrics_path: str | Path | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    profile_manifest_path: str | Path | None = None,
) -> dict[str, Path | dict[str, Any]]:
    """Run fold selection from already-materialized pair metrics.

    Checkpoint inference is intentionally kept outside this pure selection
    boundary.  The orchestrator must first produce `fold_pair_metrics.csv.gz`
    using only matching inner-fold validation windows.
    """

    root = Path(experiment_root).resolve()
    source = (
        Path(pair_metrics_path).resolve()
        if pair_metrics_path is not None
        else root / "analysis" / "fold_pair_metrics.csv.gz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    pair_metrics = pd.read_csv(source, low_memory=False)
    comparisons = build_arm_comparisons(
        pair_metrics, iterations=int(iterations), seed=int(seed)
    )
    retention_path = (
        Path(profile_manifest_path).resolve()
        if profile_manifest_path is not None
        else root / "reliability_profiles" / "reliability_profile_manifest.csv"
    )
    if retention_path.is_file():
        comparisons = apply_retention_validity_gate(
            comparisons, pd.read_csv(retention_path, low_memory=False)
        )
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    comparisons_path = analysis_dir / "label_reliability_comparisons.csv"
    comparisons.to_csv(comparisons_path, index=False)
    selection = select_reliability_winner(comparisons)
    selection_path = root / "label_reliability_selection.json"
    encoded = json.dumps(selection, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if selection_path.exists() and selection_path.read_text(encoding="utf-8") != encoded:
        raise LabelReliabilityAnalysisError(
            "Immutable label-reliability selection already exists with different content"
        )
    selection_path.write_text(encoded, encoding="utf-8")
    return {
        "comparisons_path": comparisons_path,
        "selection_path": selection_path,
        "selection": selection,
    }


__all__ = [
    "LabelReliabilityAnalysisError",
    "apply_retention_validity_gate",
    "build_arm_comparisons",
    "holm_adjust",
    "q3_checkpoint_allowlist",
    "run_label_reliability_analysis",
    "seed_fold_session_bootstrap",
    "select_reliability_winner",
    "validate_selection_payload",
    "validate_q3_prediction_frame",
]
