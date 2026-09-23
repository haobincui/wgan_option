"""Five-seed robustness analysis for the two frozen FiLM learning rates.

The experiment was scheduled after the seed-42 LR sweep, so this module does
not select a learning rate from its test results.  Its sole primary contrast
is frozen as ``2.5e-5 versus 1e-5``.  Both arms are also compared with the
pair-matched persistence forecast as one Holm-2 secondary family.

All inference uses the shared RQ1--RQ3 seed -> fold -> paired CME-session
bootstrap implementation.  Inputs are explicit, SHA-bound artifacts; this
module has no training, checkpoint-discovery, or prediction side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq123 import news_first_vol_film_nolp_10seed_analysis as unified
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42_analysis as base,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FILM_LR_ARMS: Mapping[str, float] = {
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
}
PRIMARY_FOCAL_ARM = "film_lr_2p5e5"
PRIMARY_REFERENCE_ARM = "film_lr_1e5"
SEEDS = (202, 404, 382624741, 1607127774, 1662128673)
FOLDS = base.FOLDS
EXPECTED_FOLD_PAIR_SESSION_COUNTS = base.EXPECTED_FOLD_PAIR_SESSION_COUNTS
TOLERANCE_MINUTES = 5
EXPECTED_PAIR_ROWS = 5_000
EXPECTED_JOBS = 40
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260902
NOISE_COLUMN = base.NOISE_COLUMN
EXPECTED_GROUPS = base.EXPECTED_GROUPS
INTERPRETATION = "retrospective_rolling_development_post_selection_seed_robustness"
ANALYSIS_MANIFEST_KIND = "film_unet_direct_matched_lr_5seed_analysis_manifest_v1"

FilmLrFiveSeedAnalysisError = base.FilmLrAnalysisError
TrainingEvidence = base.TrainingEvidence
sha256_file = base.sha256_file


@dataclass(frozen=True)
class FilmLrFiveSeedAnalysis:
    """Validated evidence and all deterministic analysis tables."""

    pair_metrics: pd.DataFrame
    seed_fold_summary: pd.DataFrame
    arm_summary: pd.DataFrame
    primary_comparison: pd.DataFrame
    persistence_comparisons: pd.DataFrame
    training_diagnostics: pd.DataFrame
    lr_trace: pd.DataFrame
    training_external_inputs: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def _read_frame(source: pd.DataFrame | str | Path, label: str) -> pd.DataFrame:
    return base._read_frame(source, label)


def validate_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_seeds: Sequence[int] = SEEDS,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = EXPECTED_PAIR_ROWS,
) -> pd.DataFrame:
    """Validate the exact seed/fold/arm panel and its prediction lineage."""

    frame = _read_frame(source, "five-seed FiLM-LR pair metrics")
    seeds = tuple(map(int, expected_seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise FilmLrFiveSeedAnalysisError("expected_seeds must be non-empty and unique")
    observed_seed_values = pd.to_numeric(frame.get("seed"), errors="coerce")
    if (
        observed_seed_values.isna().any()
        or not np.equal(observed_seed_values, np.floor(observed_seed_values)).all()
    ):
        raise FilmLrFiveSeedAnalysisError("Pair metrics seed must contain integers")
    frame = frame.copy()
    frame["seed"] = observed_seed_values.astype(int)
    observed_seeds = set(frame["seed"])
    if observed_seeds != set(seeds):
        raise FilmLrFiveSeedAnalysisError(
            "Pair metrics seed universe drift: "
            f"missing={sorted(set(seeds) - observed_seeds)}, "
            f"extra={sorted(observed_seeds - set(seeds))}"
        )

    validated: list[pd.DataFrame] = []
    per_seed_rows = None
    if expected_row_count is not None:
        if int(expected_row_count) % len(seeds):
            raise FilmLrFiveSeedAnalysisError(
                "Expected row count must be divisible by the seed count"
            )
        per_seed_rows = int(expected_row_count) // len(seeds)
    for seed in seeds:
        validated.append(
            base._wrap_direct(
                f"invalid five-seed pair metrics for seed={seed}",
                base.validate_pair_metrics,
                frame.loc[frame["seed"].eq(seed)].copy(),
                expected_arms=tuple(FILM_LR_ARMS),
                expected_seed=seed,
                expected_fold_counts=expected_fold_counts,
                expected_tolerance_minutes=TOLERANCE_MINUTES,
                expected_row_count=per_seed_rows,
            )
        )
    result = pd.concat(validated, ignore_index=True, sort=False)
    if expected_row_count is not None and len(result) != int(expected_row_count):
        raise FilmLrFiveSeedAnalysisError(
            f"Pair metric row count drift: {len(result)} != {expected_row_count}"
        )
    if NOISE_COLUMN not in result:
        raise FilmLrFiveSeedAnalysisError(f"Pair metrics are missing {NOISE_COLUMN}")
    result[NOISE_COLUMN] = [
        base._require_sha(value, f"pair metrics {NOISE_COLUMN}")
        for value in result[NOISE_COLUMN]
    ]
    for job_id, group in result.groupby("job_id", sort=True):
        if group[NOISE_COLUMN].nunique(dropna=False) != 1:
            raise FilmLrFiveSeedAnalysisError(
                f"{job_id} contains multiple MC-noise-bank profiles"
            )

    lineage = [
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
        NOISE_COLUMN,
    ]
    stable_lineage = lineage[:-1]
    for seed in seeds:
        for fold in expected_fold_counts:
            cell = result[result["seed"].eq(seed) & result["fold"].eq(str(fold))]
            reference: pd.DataFrame | None = None
            for arm in FILM_LR_ARMS:
                panel = (
                    cell[cell["arm"].eq(arm)][lineage]
                    .sort_values("pair_id", kind="stable")
                    .reset_index(drop=True)
                )
                if reference is None:
                    reference = panel
                elif not panel.equals(reference):
                    raise FilmLrFiveSeedAnalysisError(
                        "Pair/session/origin/persistence/noise lineage differs "
                        f"across arms for seed={seed}, fold={fold}"
                    )

    # The market panel itself is frozen across training seeds.  MC banks may
    # be seed-specific, but all other lineage must therefore also agree across
    # seeds, not merely across the two arms within one seed.
    for fold in expected_fold_counts:
        reference = None
        for seed in seeds:
            panel = (
                result[
                    result["seed"].eq(seed)
                    & result["fold"].eq(str(fold))
                    & result["arm"].eq(PRIMARY_REFERENCE_ARM)
                ][stable_lineage]
                .sort_values("pair_id", kind="stable")
                .reset_index(drop=True)
            )
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise FilmLrFiveSeedAnalysisError(
                    f"Frozen market lineage differs across seeds for fold={fold}"
                )

    result["film_learning_rate"] = result["arm"].map(FILM_LR_ARMS).astype(float)
    return result.sort_values(
        ["seed", "fold", "film_learning_rate", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def validate_training_evidence(
    source: pd.DataFrame | str | Path,
    pair_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = SEEDS,
    maximum_epochs: int = 240,
) -> TrainingEvidence:
    """Validate all 40 jobs and every grouped-optimizer LR trajectory."""

    source_path = None if isinstance(source, pd.DataFrame) else Path(source).resolve()
    frame = _read_frame(source, "five-seed FiLM-LR training summary")
    required = {
        "job_id",
        "seed",
        "fold",
        "arm",
        "best_epoch",
        "epochs_ran",
        "best_validation_score",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(frame))
    if frame.empty or missing:
        raise FilmLrFiveSeedAnalysisError(
            f"Training summary is empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise FilmLrFiveSeedAnalysisError(
                f"Training summary {column} must be non-empty"
            )
    seed_values = pd.to_numeric(result["seed"], errors="coerce")
    if (
        seed_values.isna().any()
        or not np.equal(seed_values, np.floor(seed_values)).all()
    ):
        raise FilmLrFiveSeedAnalysisError("Training summary seed must contain integers")
    result["seed"] = seed_values.astype(int)
    seeds = tuple(map(int, expected_seeds))
    if len(result) != EXPECTED_JOBS or result["job_id"].duplicated().any():
        raise FilmLrFiveSeedAnalysisError(
            "Training summary must contain 40 unique FiLM jobs"
        )
    expected_cells = {
        (seed, fold, arm) for seed in seeds for fold in FOLDS for arm in FILM_LR_ARMS
    }
    observed_cells = set(
        result[["seed", "fold", "arm"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise FilmLrFiveSeedAnalysisError(
            "Training summary seed/fold/arm universe drift"
        )

    for column in ("best_epoch", "epochs_ran"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise FilmLrFiveSeedAnalysisError(
                f"Training summary {column} must contain integers"
            )
        result[column] = values.astype(int)
    if (
        (result["best_epoch"] < 1).any()
        or (result["epochs_ran"] < result["best_epoch"]).any()
        or (result["epochs_ran"] > int(maximum_epochs)).any()
    ):
        raise FilmLrFiveSeedAnalysisError(
            "Training epochs must satisfy 1 <= best_epoch <= epochs_ran <= maximum"
        )
    scores = pd.to_numeric(result["best_validation_score"], errors="coerce").astype(
        float
    )
    if not np.isfinite(scores.to_numpy()).all():
        raise FilmLrFiveSeedAnalysisError(
            "best_validation_score must contain finite values"
        )
    result["best_validation_score"] = scores
    result["checkpoint_sha256"] = [
        base._require_sha(value, "training checkpoint_sha256")
        for value in result["checkpoint_sha256"]
    ]
    job_columns = ["job_id", "seed", "fold", "arm", "checkpoint_sha256"]
    declared = (
        result[job_columns].sort_values("job_id", kind="stable").reset_index(drop=True)
    )
    pair_jobs = (
        pair_metrics[job_columns]
        .drop_duplicates()
        .sort_values("job_id", kind="stable")
        .reset_index(drop=True)
    )
    if not declared.equals(pair_jobs):
        raise FilmLrFiveSeedAnalysisError(
            "Training summary jobs/seeds/checkpoint hashes differ from pair metrics"
        )

    external_inputs: dict[str, Mapping[str, Any]] = {}
    native_contracts: Mapping[str, Mapping[str, Any]] | None = None
    has_declared_contract = "optimizer_contract_json" in result or {
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
                raise FilmLrFiveSeedAnalysisError(
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
                raise FilmLrFiveSeedAnalysisError(
                    f"{job_id}/{group_name} configured LR drift: "
                    f"{configured} != {expected_lr}"
                )
            if not math.isclose(initial, configured, rel_tol=0.0, abs_tol=1e-18):
                raise FilmLrFiveSeedAnalysisError(
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
                    "seed": int(row["seed"]),
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
                "seed": int(row["seed"]),
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
        .sort_values(["film_learning_rate", "seed", "fold"], kind="stable")
        .reset_index(drop=True)
    )
    traces = (
        pd.DataFrame(trace_rows)
        .sort_values(
            ["film_learning_rate", "seed", "fold", "parameter_group", "epoch"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return TrainingEvidence(
        diagnostics=diagnostics,
        lr_trace=traces,
        external_inputs=tuple(external_inputs[key] for key in sorted(external_inputs)),
    )


def _summaries(pair_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (seed, fold, arm), group in pair_metrics.groupby(
        ["seed", "fold", "arm"], sort=True
    ):
        model = float(group["target_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        if model <= 0.0 or persistence <= 0.0:
            raise FilmLrFiveSeedAnalysisError("Cell mean MAEs must be positive")
        rows.append(
            {
                "seed": int(seed),
                "fold": str(fold),
                "arm": str(arm),
                "film_learning_rate": FILM_LR_ARMS[str(arm)],
                "mean_mae": model,
                "persistence_mae": persistence,
                "log_mae_ratio_vs_persistence": float(math.log(model / persistence)),
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "pair_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
                "interpretation": INTERPRETATION,
            }
        )
    cells = (
        pd.DataFrame(rows)
        .sort_values(["seed", "fold", "film_learning_rate"], kind="stable")
        .reset_index(drop=True)
    )
    arm_rows = []
    for arm, group in pair_metrics.groupby("arm", sort=True):
        arm_cells = cells[cells["arm"].eq(str(arm))]
        arm_rows.append(
            {
                "arm": str(arm),
                "film_learning_rate": FILM_LR_ARMS[str(arm)],
                "pooled_mae": float(group["target_mae"].mean()),
                "equal_seed_fold_mae": float(arm_cells["mean_mae"].mean()),
                "equal_seed_fold_persistence_mae": float(
                    arm_cells["persistence_mae"].mean()
                ),
                "equal_seed_fold_improvement_vs_persistence_percent": float(
                    100.0
                    * (
                        1.0
                        - arm_cells["mean_mae"].mean()
                        / arm_cells["persistence_mae"].mean()
                    )
                ),
                "seed_count": int(group["seed"].nunique()),
                "fold_count": int(group["fold"].nunique()),
                "pair_rows": int(len(group)),
                "interpretation": INTERPRETATION,
                "test_based_lr_selection_permitted": False,
            }
        )
    arms = pd.DataFrame(arm_rows)
    arms["descriptive_mae_rank"] = (
        arms["equal_seed_fold_mae"].rank(method="min", ascending=True).astype(int)
    )
    return cells, arms.sort_values(
        ["descriptive_mae_rank", "film_learning_rate"], kind="stable"
    ).reset_index(drop=True)


def _primary_comparison(
    pair_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    try:
        stats = unified.seed_fold_session_paired_bootstrap(
            pair_metrics,
            focal_arm=PRIMARY_FOCAL_ARM,
            reference_arm=PRIMARY_REFERENCE_ARM,
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
            iterations=int(iterations),
            rng_seed=int(rng_seed),
        )
    except unified.UnifiedAnalysisError as exc:
        raise FilmLrFiveSeedAnalysisError(f"Primary bootstrap failed: {exc}") from exc
    stats.update(
        {
            "comparison_id": f"{PRIMARY_FOCAL_ARM}_vs_{PRIMARY_REFERENCE_ARM}",
            "comparison_role": "primary_frozen_post_selection_seed_robustness",
            "focal_film_learning_rate": FILM_LR_ARMS[PRIMARY_FOCAL_ARM],
            "reference_film_learning_rate": FILM_LR_ARMS[PRIMARY_REFERENCE_ARM],
            "holm_family_size": 1,
            "holm_adjusted_p": float(stats["p_value_one_sided"]),
            "required_consistent_seed_count": 4,
            "required_consistent_fold_count": 3,
            "passes_retrospective_support_gate": bool(
                float(stats["mean_log_mae_ratio"]) < 0.0
                and float(stats["ci_95_upper"]) < 0.0
                and float(stats["p_value_one_sided"]) < 0.05
                and int(stats["consistent_seed_count"]) >= 4
                and int(stats["consistent_fold_count"]) >= 3
            ),
            "interpretation": INTERPRETATION,
            "confirmatory": False,
            "test_based_lr_selection_permitted": False,
        }
    )
    return pd.DataFrame([stats])


def _persistence_comparisons(
    pair_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, arm in enumerate(FILM_LR_ARMS):
        try:
            stats = unified.seed_fold_session_persistence_bootstrap(
                pair_metrics,
                arm=arm,
                expected_seeds=expected_seeds,
                expected_folds=expected_folds,
                iterations=int(iterations),
                rng_seed=int(rng_seed) + index,
            )
        except unified.UnifiedAnalysisError as exc:
            raise FilmLrFiveSeedAnalysisError(
                f"Persistence bootstrap failed for {arm}: {exc}"
            ) from exc
        stats.update(
            {
                "arm": arm,
                "comparison_id": f"{arm}_vs_persistence",
                "film_learning_rate": FILM_LR_ARMS[arm],
                "comparison_role": "secondary_arm_vs_persistence",
                "multiplicity_family": "two_film_lrs_vs_persistence_holm2",
                "interpretation": INTERPRETATION,
                "confirmatory": False,
                "test_based_lr_selection_permitted": False,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    try:
        adjusted = unified.holm_adjust(
            dict(
                zip(
                    result["comparison_id"].astype(str),
                    result["p_value_one_sided"].astype(float),
                    strict=True,
                )
            )
        )
    except unified.UnifiedAnalysisError as exc:
        raise FilmLrFiveSeedAnalysisError(f"Holm adjustment failed: {exc}") from exc
    result["holm_adjusted_p"] = result["comparison_id"].map(adjusted)
    result["holm_family_size"] = 2
    result["required_consistent_seed_count"] = 4
    result["required_consistent_fold_count"] = 3
    result["passes_retrospective_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["holm_adjusted_p"].lt(0.05)
        & result["consistent_seed_count"].ge(4)
        & result["consistent_fold_count"].ge(3)
    )
    return result.sort_values("film_learning_rate", kind="stable").reset_index(
        drop=True
    )


def analyze_film_lr(
    pair_metrics: pd.DataFrame | str | Path,
    training_summary: pd.DataFrame | str | Path,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_seeds: Sequence[int] = SEEDS,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_pair_rows: int | None = EXPECTED_PAIR_ROWS,
) -> FilmLrFiveSeedAnalysis:
    """Analyze the frozen two-LR, five-seed evidence."""

    pairs = validate_pair_metrics(
        pair_metrics,
        expected_seeds=expected_seeds,
        expected_fold_counts=expected_fold_counts,
        expected_row_count=expected_pair_rows,
    )
    training = validate_training_evidence(
        training_summary, pairs, expected_seeds=expected_seeds
    )
    cells, arms = _summaries(pairs)
    primary = _primary_comparison(
        pairs,
        expected_seeds=expected_seeds,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed),
    )
    persistence = _persistence_comparisons(
        pairs,
        expected_seeds=expected_seeds,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed) + 10_000,
    )
    primary_row = primary.iloc[0]
    descriptive = arms.iloc[0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "film_unet_direct_matched_two_lr_five_seed_robustness",
        "interpretation": INTERPRETATION,
        "claim_scope": "post_selection_seed_robustness_retrospective_development",
        "confirmatory": False,
        "test_based_lr_selection_permitted": False,
        "lr_choice_must_not_use_test_results": True,
        "primary_comparison_id": str(primary_row["comparison_id"]),
        "primary_mean_log_mae_ratio": float(primary_row["mean_log_mae_ratio"]),
        "primary_geometric_mae_ratio": float(primary_row["geometric_mae_ratio"]),
        "primary_ci_95_lower": float(primary_row["ci_95_lower"]),
        "primary_ci_95_upper": float(primary_row["ci_95_upper"]),
        "primary_p_value_one_sided": float(primary_row["p_value_one_sided"]),
        "primary_consistent_seed_count": int(primary_row["consistent_seed_count"]),
        "primary_consistent_fold_count": int(primary_row["consistent_fold_count"]),
        "primary_passes_retrospective_support_gate": bool(
            primary_row["passes_retrospective_support_gate"]
        ),
        "descriptive_lowest_mae_arm": str(descriptive["arm"]),
        "descriptive_lowest_mae_lr": float(descriptive["film_learning_rate"]),
        "descriptive_lowest_equal_seed_fold_mae": float(
            descriptive["equal_seed_fold_mae"]
        ),
        "film_lr_values": sorted(FILM_LR_ARMS.values()),
        "seeds": list(map(int, expected_seeds)),
        "seed_count": len(tuple(expected_seeds)),
        "folds": list(expected_fold_counts),
        "fold_count": len(expected_fold_counts),
        "tolerance_minutes": TOLERANCE_MINUTES,
        "training_job_count": int(pairs["job_id"].nunique()),
        "pair_metric_rows": int(len(pairs)),
        "rows_per_arm": int(len(pairs) // len(FILM_LR_ARMS)),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_method": (
            "seed_then_fold_then_paired_session_cluster_recompute_cell_log_mae_ratio"
        ),
        "primary_holm_family_size": 1,
        "persistence_holm_family_size": 2,
        "lineage_contract": (
            "exact_seed_fold_pair_session_origin_persistence_and_mc_noise_bank_v1"
        ),
        "training_diagnostics_complete": True,
    }
    return FilmLrFiveSeedAnalysis(
        pair_metrics=pairs,
        seed_fold_summary=cells,
        arm_summary=arms,
        primary_comparison=primary,
        persistence_comparisons=persistence,
        training_diagnostics=training.diagnostics,
        lr_trace=training.lr_trace,
        training_external_inputs=training.external_inputs,
        summary=summary,
    )


def _format(value: Any) -> str:
    return base._format(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return base._markdown_table(frame, columns)


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return base._html_table(frame, columns)


def render_reports(analysis: FilmLrFiveSeedAnalysis) -> tuple[str, str]:
    """Render self-contained Chinese Markdown and HTML reports."""

    summary = analysis.summary
    arm_columns = (
        "descriptive_mae_rank",
        "arm",
        "film_learning_rate",
        "equal_seed_fold_mae",
        "equal_seed_fold_improvement_vs_persistence_percent",
        "seed_count",
        "fold_count",
    )
    comparison_columns = (
        "focal_arm",
        "reference_arm",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "p_value_one_sided",
        "consistent_seed_count",
        "consistent_fold_count",
        "passes_retrospective_support_gate",
    )
    persistence_columns = (
        "arm",
        "film_learning_rate",
        "mean_log_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "passes_retrospective_support_gate",
    )
    diagnostics = (
        analysis.training_diagnostics.groupby(["arm", "film_learning_rate"], sort=True)
        .agg(
            job_count=("job_id", "nunique"),
            seed_count=("seed", "nunique"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
            film_initial=("film_initial", "first"),
            text_encoder_initial=("text_encoder_initial", "first"),
            backbone_initial=("backbone_initial", "first"),
            critic_initial=("critic_initial", "first"),
        )
        .reset_index()
    )
    diagnostic_columns = tuple(diagnostics.columns)
    caveat = (
        "本实验是 seed-42 LR 扫描之后预定的五 seed 稳健性验证。主比较固定为 "
        "2.5e-5 对 1e-5；禁止用本次 test 结果重新选择 LR。结果属于 "
        f"{INTERPRETATION}，不是 confirmatory holdout。"
    )
    markdown = f"""# FiLM LR 1e-5 vs 2.5e-5：五 Seed 稳健性验证

## 固定主比较

负的 log-MAE ratio 表示 `2.5e-5` 更好。使用 `{summary["bootstrap_iterations"]}` 次 seed → fold → paired CME-session bootstrap。

{_markdown_table(analysis.primary_comparison, comparison_columns)}

{caveat}

## 描述性汇总

{_markdown_table(analysis.arm_summary, arm_columns)}

排名只描述冻结结果，`test_based_lr_selection_permitted=false`。

## 相对 Persistence（secondary Holm-2）

{_markdown_table(analysis.persistence_comparisons, persistence_columns)}

## 训练与优化器审计

{_markdown_table(diagnostics, diagnostic_columns)}

- 训练任务：{summary["training_job_count"]}；pair metrics：{summary["pair_metric_rows"]} 行。
- 5 seeds × 4 folds × 2 LR 的 checkpoint、pair、session、origin、persistence 与 MC noise lineage 均已核验。
- 完整逐任务诊断与逐 epoch 参数组 LR 分别见 `film_lr_5seed_training_diagnostics.csv` 和 `film_lr_5seed_group_lr_trace.csv`。
"""
    html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FiLM LR 五 Seed 稳健性验证</title><style>
body{{margin:0;background:#f4f6f8;color:#17202a;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dfe4ea;border-radius:10px;padding:18px;margin:14px 0;overflow:auto}}h1,h2{{margin-top:0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid #e7ebef;padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.warning{{border-left:5px solid #b45309;background:#fff8eb}}
</style></head><body><main><h1>FiLM LR 1e-5 vs 2.5e-5：五 Seed 稳健性验证</h1>
<section><h2>固定主比较</h2>{_html_table(analysis.primary_comparison, comparison_columns)}</section>
<section class="warning"><h2>解释边界</h2><p>{html.escape(caveat)}</p></section>
<section><h2>描述性汇总</h2>{_html_table(analysis.arm_summary, arm_columns)}<p>排名不允许用于 test-based LR selection。</p></section>
<section><h2>相对 Persistence（Holm-2）</h2>{_html_table(analysis.persistence_comparisons, persistence_columns)}</section>
<section><h2>训练与优化器审计</h2>{_html_table(diagnostics, diagnostic_columns)}<p>{summary["training_job_count"]} jobs；{summary["pair_metric_rows"]} pair rows。</p></section>
</main></body></html>"""
    return markdown, html_report


def write_analysis_bundle(
    analysis: FilmLrFiveSeedAnalysis,
    *,
    pair_metrics_path: str | Path,
    pair_metrics_sha256: str,
    training_summary_path: str | Path,
    training_summary_sha256: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write an idempotent, SHA-bound analysis bundle."""

    pair_source = Path(pair_metrics_path).resolve()
    training_source = Path(training_summary_path).resolve()
    inputs = {
        "pair_metrics_source": (
            pair_source,
            base._require_sha(pair_metrics_sha256, "pair metrics SHA"),
        ),
        "training_summary_source": (
            training_source,
            base._require_sha(training_summary_sha256, "training summary SHA"),
        ),
    }
    for role, (path, expected_sha) in inputs.items():
        if sha256_file(path) != expected_sha:
            raise FilmLrFiveSeedAnalysisError(f"{role} SHA-256 drift before write")
    destination = Path(output_dir)
    paths = {
        "seed_fold_summary": destination / "film_lr_5seed_seed_fold_summary.csv",
        "arm_summary": destination / "film_lr_5seed_arm_summary.csv",
        "primary_comparison": destination / "film_lr_5seed_primary_bootstrap.csv",
        "persistence_comparisons": destination
        / "film_lr_5seed_vs_persistence_bootstrap_holm.csv",
        "training_diagnostics": destination / "film_lr_5seed_training_diagnostics.csv",
        "lr_trace": destination / "film_lr_5seed_group_lr_trace.csv",
        "summary": destination / "film_lr_5seed_analysis_summary.json",
        "report_markdown": destination / "film_lr_5seed_report.md",
        "report_html": destination / "film_lr_5seed_report.html",
        "manifest": destination / "film_lr_5seed_analysis_manifest.json",
    }
    frames = {
        "seed_fold_summary": analysis.seed_fold_summary,
        "arm_summary": analysis.arm_summary,
        "primary_comparison": analysis.primary_comparison,
        "persistence_comparisons": analysis.persistence_comparisons,
        "training_diagnostics": analysis.training_diagnostics,
        "lr_trace": analysis.lr_trace,
    }
    for role, frame in frames.items():
        base._atomic_write(paths[role], base._csv_bytes(frame))
    summary = dict(analysis.summary)
    for role, (path, digest) in inputs.items():
        summary[f"{role}_path"] = str(path)
        summary[f"{role}_sha256"] = digest
    markdown, html_report = render_reports(analysis)
    base._atomic_write(paths["summary"], base._canonical_json(summary))
    base._atomic_write(paths["report_markdown"], markdown.encode("utf-8"))
    base._atomic_write(paths["report_html"], html_report.encode("utf-8"))

    artifacts = []
    for role in (
        "seed_fold_summary",
        "arm_summary",
        "primary_comparison",
        "persistence_comparisons",
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
    manifest_inputs = [
        {
            "role": role,
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        for role, (path, digest) in inputs.items()
    ]
    manifest_inputs.extend(dict(row) for row in analysis.training_external_inputs)
    for row in manifest_inputs:
        if sha256_file(row["path"]) != row["sha256"]:
            raise FilmLrFiveSeedAnalysisError(
                f"Input changed while writing: {row['role']}"
            )
    manifest = {
        "schema_version": 1,
        "kind": ANALYSIS_MANIFEST_KIND,
        "audience": "technical",
        "interpretation": INTERPRETATION,
        "confirmatory": False,
        "test_based_lr_selection_permitted": False,
        "inputs": manifest_inputs,
        "artifacts": artifacts,
    }
    base._atomic_write(paths["manifest"], base._canonical_json(manifest))
    return paths


def run_film_lr_analysis(
    *,
    pair_metrics_path: str | Path,
    pair_metrics_sha256: str,
    training_summary_path: str | Path,
    training_summary_sha256: str,
    output_dir: str | Path,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Path]:
    for path, digest, label in (
        (pair_metrics_path, pair_metrics_sha256, "pair metrics"),
        (training_summary_path, training_summary_sha256, "training summary"),
    ):
        if sha256_file(path) != base._require_sha(digest, f"{label} SHA"):
            raise FilmLrFiveSeedAnalysisError(f"{label} SHA-256 drift")
    analysis = analyze_film_lr(
        pair_metrics_path,
        training_summary_path,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    return write_analysis_bundle(
        analysis,
        pair_metrics_path=pair_metrics_path,
        pair_metrics_sha256=pair_metrics_sha256,
        training_summary_path=training_summary_path,
        training_summary_sha256=training_summary_sha256,
        output_dir=output_dir,
    )


def analyze_experiment(output_root: Path, config: Mapping[str, Any]) -> Path:
    """Orchestrator adapter for the independent five-seed formal root."""

    analysis_config = config.get("analysis")
    if not isinstance(analysis_config, Mapping):
        raise FilmLrFiveSeedAnalysisError("config.analysis must be a mapping")
    iterations = int(analysis_config.get("bootstrap_replicates", BOOTSTRAP_ITERATIONS))
    seed = int(analysis_config.get("bootstrap_seed", BOOTSTRAP_SEED))
    root = Path(output_root).resolve()
    pair_path = root / "analysis/rq12_pair_metrics.csv.gz"
    training_path = root / "analysis/training_summary.csv"
    paths = run_film_lr_analysis(
        pair_metrics_path=pair_path,
        pair_metrics_sha256=sha256_file(pair_path),
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
    "FOLDS",
    "FilmLrFiveSeedAnalysis",
    "FilmLrFiveSeedAnalysisError",
    "INTERPRETATION",
    "PRIMARY_FOCAL_ARM",
    "PRIMARY_REFERENCE_ARM",
    "SEEDS",
    "analyze_experiment",
    "analyze_film_lr",
    "render_reports",
    "run_film_lr_analysis",
    "sha256_file",
    "validate_pair_metrics",
    "validate_training_evidence",
    "write_analysis_bundle",
]
