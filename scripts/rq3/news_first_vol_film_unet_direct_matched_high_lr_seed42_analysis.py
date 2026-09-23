"""Analysis for the direct matched-text high FiLM-LR sweep.

This module is an explicit high-LR profile.  It reuses the audited pairing,
bootstrap, Holm, SHA, and native optimizer-artifact readers from the original
FiLM-LR analysis without mutating that module's globals.  Consequently the
low- and high-LR experiments can be analyzed safely in the same process.

All outputs are single-seed retrospective rolling-development diagnostics.
The point leader is descriptive and must not be treated as a test-selected
learning rate.
"""

from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42_analysis as base,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FILM_LR_ARMS: Mapping[str, float] = {
    "film_lr_5e6": 5.0e-6,
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
    "film_lr_5e5": 5.0e-5,
    "film_lr_1e4": 1.0e-4,
}
PURE_ARM = base.PURE_ARM
FOLDS = base.FOLDS
EXPECTED_FOLD_PAIR_SESSION_COUNTS = base.EXPECTED_FOLD_PAIR_SESSION_COUNTS
SEED = base.SEED
TOLERANCE_MINUTES = base.TOLERANCE_MINUTES
EXPECTED_FILM_ROWS = 2_500
EXPECTED_PURE_ROWS = base.EXPECTED_PURE_ROWS
EXPECTED_FILM_JOBS = 20
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260901
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
PURE_PAIR_METRICS_SHA256 = base.PURE_PAIR_METRICS_SHA256
NOISE_COLUMN = base.NOISE_COLUMN
EXPECTED_GROUPS = base.EXPECTED_GROUPS
# The output schema is intentionally identical to the original five-LR
# analysis.  The formal root, frozen input hashes, and summary experiment name
# distinguish this profile; keeping the kind stable lets the shared verifier
# validate either LR range without a schema-specific branch.
ANALYSIS_MANIFEST_KIND = "film_unet_direct_matched_lr_seed42_analysis_manifest_v1"

FilmLrAnalysisError = base.FilmLrAnalysisError
TrainingEvidence = base.TrainingEvidence
FilmLrAnalysis = base.FilmLrAnalysis
sha256_file = base.sha256_file
validate_pure_pair_metrics = base.validate_pure_pair_metrics


def validate_film_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = EXPECTED_FILM_ROWS,
) -> pd.DataFrame:
    """Validate the exact five-arm high-LR pair universe."""

    result = base._wrap_direct(
        "invalid high-FiLM-LR pair metrics",
        base.validate_pair_metrics,
        source,
        expected_arms=tuple(FILM_LR_ARMS),
        expected_seed=SEED,
        expected_fold_counts=expected_fold_counts,
        expected_tolerance_minutes=TOLERANCE_MINUTES,
        expected_row_count=expected_row_count,
    )
    result = base._validate_noise(result, "high-FiLM-LR pair metrics")
    result["film_learning_rate"] = result["arm"].map(FILM_LR_ARMS).astype(float)
    return result


def validate_cross_experiment_lineage(
    film: pd.DataFrame,
    pure: pd.DataFrame,
) -> None:
    """Require identical pair/session/persistence/noise evidence per arm."""

    lineage = [
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
        NOISE_COLUMN,
    ]
    reference = (
        pure[lineage]
        .sort_values(["fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )
    for arm in FILM_LR_ARMS:
        candidate = (
            film.loc[film["arm"].eq(arm), lineage]
            .sort_values(["fold", "pair_id"], kind="stable")
            .reset_index(drop=True)
        )
        if len(candidate) != len(reference):
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
            if not candidate[column].equals(reference[column]):
                raise FilmLrAnalysisError(
                    f"Cross-experiment {column} lineage differs for {arm} vs Pure CNN"
                )
        if not np.array_equal(
            candidate["persistence_mae"].to_numpy(float),
            reference["persistence_mae"].to_numpy(float),
        ):
            raise FilmLrAnalysisError(
                f"Cross-experiment persistence lineage differs for {arm} vs Pure CNN"
            )


def validate_training_evidence(
    source: pd.DataFrame | str | Path,
    film_pair_metrics: pd.DataFrame,
    *,
    maximum_epochs: int = 240,
) -> TrainingEvidence:
    """Validate all jobs, group sizes, configured LRs, and complete LR traces."""

    source_path = None if isinstance(source, pd.DataFrame) else Path(source).resolve()
    frame = base._read_frame(source, "high-FiLM-LR training summary")
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
        base._require_sha(value, "training checkpoint_sha256")
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
        native_contracts = base._load_native_optimizer_contracts(
            source_path,
            result.to_dict(orient="records"),
            external_inputs=external_inputs,
        )

    diagnostic_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        job_id = str(row["job_id"])
        arm = str(row["arm"])
        contract = base._load_optimizer_contract(
            row,
            external_inputs=external_inputs,
            native_contracts=native_contracts,
        )
        groups = base._contract_groups(contract, f"{job_id} optimizer contract")
        group_values: dict[str, dict[str, Any]] = {}
        for group_name, (expected_count, fixed_lr) in EXPECTED_GROUPS.items():
            group = groups[group_name]
            count = int(
                base._group_value(
                    group,
                    ("parameter_count", "parameters", "num_parameters"),
                    f"{job_id}/{group_name} parameter count",
                )
            )
            if count != expected_count:
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} parameter count drift: "
                    f"{count} != {expected_count}"
                )
            configured = base._finite_positive(
                base._group_value(
                    group,
                    ("configured_learning_rate", "configured_lr", "target_lr"),
                    f"{job_id}/{group_name} configured LR",
                ),
                f"{job_id}/{group_name} configured LR",
            )
            initial = base._finite_positive(
                base._group_value(
                    group,
                    ("initial_learning_rate", "initial_lr"),
                    f"{job_id}/{group_name} initial LR",
                ),
                f"{job_id}/{group_name} initial LR",
            )
            final = base._finite_positive(
                base._group_value(
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
                    f"{job_id}/{group_name} configured LR drift: "
                    f"{configured} != {expected_lr}"
                )
            if not math.isclose(initial, configured, rel_tol=0.0, abs_tol=1e-18):
                raise FilmLrAnalysisError(
                    f"{job_id}/{group_name} configured/initial LR mismatch"
                )
            trace = base._normalize_trace(
                base._group_value(
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
            ["film_learning_rate", "fold", "parameter_group", "epoch"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return TrainingEvidence(
        diagnostics=diagnostics,
        lr_trace=traces,
        external_inputs=tuple(external_inputs[key] for key in sorted(external_inputs)),
    )


def _build_ranking(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = base.compute_fold_summary(combined)
    arms = base.compute_arm_summary(combined, folds)
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
        stats = base._wrap_direct(
            f"bootstrap failed for {arm} vs {PURE_ARM}",
            base.fold_session_paired_bootstrap,
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
                "multiplicity_family": "high_film_lr_vs_pure_holm5",
                "confirmatory": False,
                "inference_permitted": False,
                "interpretation": INTERPRETATION,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    adjusted = base.holm_adjust(
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
        stats = base._wrap_direct(
            f"bootstrap failed for LR pair {higher} vs {lower}",
            base.fold_session_paired_bootstrap,
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
                "multiplicity_family": "all_high_film_lr_pairs_holm10",
                "confirmatory": False,
                "inference_permitted": False,
                "interpretation": INTERPRETATION,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    if len(result) != math.comb(len(FILM_LR_ARMS), 2):
        raise FilmLrAnalysisError("Internal high-FiLM-LR family is incomplete")
    adjusted = base.holm_adjust(
        dict(
            zip(
                result["comparison_id"].astype(str),
                result["p_value_one_sided"].astype(float),
                strict=True,
            )
        )
    )
    result["holm_adjusted_p"] = result["comparison_id"].map(adjusted)
    result["holm_family_size"] = 10
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
    focal_column = "arm" if "arm" in comparisons.columns else "focal_arm"
    leader_comparison = comparisons.loc[
        comparisons[focal_column].eq(str(leader["arm"]))
    ].iloc[0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "film_unet_direct_matched_high_lr_seed42_vs_frozen_pure_cnn",
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
        "pure_comparator_role": (
            "frozen_contextual_architecture_and_capacity_reference"
        ),
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


def render_reports(analysis: FilmLrAnalysis) -> tuple[str, str]:
    """Reuse the generic LR report renderer with high-profile evidence."""

    return base.render_reports(analysis)


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
    """Write an idempotent, SHA-bound high-LR analysis bundle."""

    film_source = Path(film_pair_metrics_path).resolve()
    pure_source = Path(pure_pair_metrics_path).resolve()
    training_source = Path(training_summary_path).resolve()
    expected_inputs = {
        "film_pair_metrics_source": (
            film_source,
            base._require_sha(film_pair_metrics_sha256, "FiLM pair metrics SHA"),
        ),
        "pure_pair_metrics_source": (
            pure_source,
            base._require_sha(pure_pair_metrics_sha256, "Pure pair metrics SHA"),
        ),
        "film_training_summary_source": (
            training_source,
            base._require_sha(training_summary_sha256, "training summary SHA"),
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
        base._atomic_write(paths[role], base._csv_bytes(frame))
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
    base._atomic_write(paths["summary"], base._canonical_json(summary))
    base._atomic_write(paths["report_markdown"], markdown.encode("utf-8"))
    base._atomic_write(paths["report_html"], html_report.encode("utf-8"))

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
        "kind": ANALYSIS_MANIFEST_KIND,
        "audience": "technical",
        "interpretation": INTERPRETATION,
        "confirmatory": False,
        "inference_permitted": False,
        "test_based_lr_selection_permitted": False,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    base._atomic_write(paths["manifest"], base._canonical_json(manifest))
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
        if sha256_file(path) != base._require_sha(digest, f"{label} SHA"):
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
    """Orchestrator adapter for the independent high-LR formal root."""

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
    resolved_pure = Path(str(pure_path))
    if not resolved_pure.is_absolute():
        resolved_pure = (REPO_ROOT / resolved_pure).resolve()
    paths = run_film_lr_analysis(
        film_pair_metrics_path=film_path,
        film_pair_metrics_sha256=sha256_file(film_path),
        pure_pair_metrics_path=resolved_pure,
        pure_pair_metrics_sha256=str(pure_sha),
        training_summary_path=training_path,
        training_summary_sha256=sha256_file(training_path),
        output_dir=root / "analysis",
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
    )
    return paths["manifest"]


__all__ = [
    "ANALYSIS_MANIFEST_KIND",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "EXPECTED_FOLD_PAIR_SESSION_COUNTS",
    "EXPECTED_GROUPS",
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
