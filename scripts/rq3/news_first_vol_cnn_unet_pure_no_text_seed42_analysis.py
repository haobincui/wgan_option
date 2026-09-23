"""Analysis for the seed-42 pure-CNN U-Net no-text ablation.

The new experiment contains one independently trained ``pure_cnn_no_text``
cell for each of the four rolling folds.  Its frozen 500-row pair-metric table
is compared with the immutable seed-42 direct-five-arm evidence for
``no_text``, ``lp_matched``, and ``lp_shuffle``.

The primary estimand is the equally weighted mean of the four fold-level
``log(mean(MAE_pure_cnn) / mean(MAE_film_no_text))`` values.  Bootstrap draws
resample folds first and paired CME-session clusters second.  Negative values
favour the pure CNN.  The comparison is deliberately labelled descriptive:
there is only one seed, the 2023 rolling test periods have prior exposure, and
removing the text encoder plus six FiLM blocks also removes Generator
parameters.  It is therefore an architecture-and-capacity ablation, not a
capacity-matched estimate of a FiLM effect.
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

from scripts.rq3.news_first_vol_film_unet_direct_5arm_seed42_analysis import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DIRECT_ARMS,
    DIRECT_FOLDS,
    DIRECT_SEED,
    DIRECT_TOLERANCE_MINUTES,
    EXPECTED_FOLD_PAIR_SESSION_COUNTS,
    INTERPRETATION_LABEL,
    DirectFiveArmAnalysisError,
    compute_arm_summary,
    compute_fold_summary,
    fold_session_paired_bootstrap,
    fold_session_persistence_bootstrap,
    holm_adjust,
    validate_pair_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PURE_ARM = "pure_cnn_no_text"
HISTORICAL_REFERENCE_ARMS = ("no_text", "lp_matched", "lp_shuffle")
EXPECTED_PURE_JOBS = 4
EXPECTED_PURE_PAIR_METRIC_ROWS = 500
EXPECTED_PAIR_COUNT = 500
EXPECTED_SESSION_COUNT = 148
DEFAULT_BOOTSTRAP_SEED = 20260830
EXPECTED_HISTORICAL_PAIR_METRICS_SHA256 = (
    "1c6acd482ae04546c80ce159dde373f2a1e074acbb4fdd2e4f1972c4e86ac57c"
)
PURE_GENERATOR_PARAMETERS = 416_353
HISTORICAL_FILM_GENERATOR_PARAMETERS = 827_745
CRITIC_PARAMETERS = 729_157
PURE_TOTAL_PARAMETERS = 1_145_510
HISTORICAL_FILM_TOTAL_PARAMETERS = 1_556_902
PURE_GENERATOR_MODE = "cnn_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"

REQUIRED_NOISE_COLUMN = "noise_bank_profile_sha256"


class PureCnnAnalysisError(ValueError):
    """Raised when frozen evidence or the pure-CNN analysis contract drifts."""


@dataclass(frozen=True)
class PureCnnAnalysis:
    """All deterministic tables for the pure-CNN single-seed comparison."""

    pure_pair_metrics: pd.DataFrame
    historical_pair_metrics: pd.DataFrame
    combined_pair_metrics: pd.DataFrame
    fold_summary: pd.DataFrame
    arm_summary: pd.DataFrame
    comparisons: pd.DataFrame
    persistence: pd.DataFrame
    training_summary: pd.DataFrame | None
    summary: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return a regular file's SHA-256 digest."""

    source = Path(path)
    if not source.is_file():
        raise PureCnnAnalysisError(f"Required file does not exist: {source}")
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
        raise PureCnnAnalysisError(f"{label} must be one lowercase SHA-256 digest")
    return digest


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _wrap_direct_error(operation: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except DirectFiveArmAnalysisError as exc:
        raise PureCnnAnalysisError(f"{operation}: {exc}") from exc


def _validate_noise_bank(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if REQUIRED_NOISE_COLUMN not in frame.columns:
        raise PureCnnAnalysisError(
            f"{label} is missing required {REQUIRED_NOISE_COLUMN}"
        )
    result = frame.copy()
    result[REQUIRED_NOISE_COLUMN] = [
        _require_sha(value, f"{label} {REQUIRED_NOISE_COLUMN}")
        for value in result[REQUIRED_NOISE_COLUMN]
    ]
    for (fold, arm), group in result.groupby(["fold", "arm"], sort=True):
        if group[REQUIRED_NOISE_COLUMN].nunique(dropna=False) != 1:
            raise PureCnnAnalysisError(
                f"{label} has multiple MC64 noise banks for fold={fold}, arm={arm}"
            )
    return result


def validate_pure_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = EXPECTED_PURE_PAIR_METRIC_ROWS,
) -> pd.DataFrame:
    """Validate the one-arm/four-fold pure-CNN frozen prediction evidence."""

    result = _wrap_direct_error(
        "invalid pure-CNN pair metrics",
        validate_pair_metrics,
        source,
        expected_arms=(PURE_ARM,),
        expected_seed=DIRECT_SEED,
        expected_fold_counts=expected_fold_counts,
        expected_tolerance_minutes=DIRECT_TOLERANCE_MINUTES,
        expected_row_count=expected_row_count,
    )
    return _validate_noise_bank(result, "pure-CNN pair metrics")


def validate_historical_pair_metrics(
    source: pd.DataFrame | str | Path,
    *,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_row_count: int | None = 2_500,
) -> pd.DataFrame:
    """Validate the complete immutable direct-five-arm source before subsetting."""

    result = _wrap_direct_error(
        "invalid historical direct-five-arm pair metrics",
        validate_pair_metrics,
        source,
        expected_arms=DIRECT_ARMS,
        expected_seed=DIRECT_SEED,
        expected_fold_counts=expected_fold_counts,
        expected_tolerance_minutes=DIRECT_TOLERANCE_MINUTES,
        expected_row_count=expected_row_count,
    )
    return _validate_noise_bank(result, "historical pair metrics")


def validate_cross_experiment_lineage(
    pure_pair_metrics: pd.DataFrame,
    historical_pair_metrics: pd.DataFrame,
) -> None:
    """Require identical pair, session, time, persistence, and MC64 lineages."""

    pure = (
        pure_pair_metrics[
            [
                "fold",
                "pair_id",
                "session_id",
                "effective_origin_utc",
                "persistence_mae",
                REQUIRED_NOISE_COLUMN,
            ]
        ]
        .sort_values(["fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )
    historical = (
        historical_pair_metrics[historical_pair_metrics["arm"].eq("no_text")][
            list(pure.columns)
        ]
        .sort_values(["fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(pure) != len(historical):
        raise PureCnnAnalysisError(
            "Pure-CNN and historical no_text row counts are not pair complete"
        )
    text_columns = (
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        REQUIRED_NOISE_COLUMN,
    )
    for column in text_columns:
        if not pure[column].equals(historical[column]):
            raise PureCnnAnalysisError(
                f"Cross-experiment {column} lineage differs from historical no_text"
            )
    if not np.array_equal(
        pure["persistence_mae"].to_numpy(float),
        historical["persistence_mae"].to_numpy(float),
    ):
        raise PureCnnAnalysisError(
            "Cross-experiment persistence lineage differs from historical no_text"
        )


def validate_training_summary(
    source: pd.DataFrame | str | Path,
    pure_pair_metrics: pd.DataFrame,
    *,
    maximum_epochs: int = 240,
) -> pd.DataFrame:
    """Validate the four-row training diagnostic table for the pure CNN."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.is_file():
            raise PureCnnAnalysisError(f"Training summary does not exist: {path}")
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
        raise PureCnnAnalysisError(
            f"Training summary is empty or missing columns: {missing}"
        )
    result = frame.copy()
    for column in ("job_id", "fold", "arm"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise PureCnnAnalysisError(f"Training summary {column} must be non-empty")
    if len(result) != EXPECTED_PURE_JOBS or result["job_id"].duplicated().any():
        raise PureCnnAnalysisError("Training summary must contain four unique jobs")
    expected_cells = {(fold, PURE_ARM) for fold in DIRECT_FOLDS}
    observed_cells = set(result[["fold", "arm"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise PureCnnAnalysisError("Training summary fold/arm universe drift")
    for column in ("best_epoch", "epochs_ran"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise PureCnnAnalysisError(f"Training summary {column} must be integer")
        result[column] = numeric.astype(int)
    if (
        (result["best_epoch"] < 1).any()
        or (result["epochs_ran"] < result["best_epoch"]).any()
        or (result["epochs_ran"] > int(maximum_epochs)).any()
    ):
        raise PureCnnAnalysisError(
            "Training epochs must satisfy 1 <= best_epoch <= epochs_ran <= cap"
        )
    for column in (
        "final_generator_lr",
        "final_discriminator_lr",
        "best_validation_score",
    ):
        numeric = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise PureCnnAnalysisError(
                f"Training summary {column} must contain finite values"
            )
        if column.endswith("_lr") and (numeric <= 0.0).any():
            raise PureCnnAnalysisError(f"Training summary {column} must be positive")
        result[column] = numeric
    result["checkpoint_sha256"] = [
        _require_sha(value, "training summary checkpoint_sha256")
        for value in result["checkpoint_sha256"]
    ]
    pair_jobs = (
        pure_pair_metrics[["job_id", "fold", "arm", "checkpoint_sha256"]]
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
        raise PureCnnAnalysisError(
            "Training summary jobs/checkpoint hashes differ from pair metrics"
        )
    return result.sort_values("fold", kind="stable").reset_index(drop=True)


def _build_comparisons(
    combined_pair_metrics: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, reference_arm in enumerate(HISTORICAL_REFERENCE_ARMS):
        stats = _wrap_direct_error(
            f"failed bootstrap for {PURE_ARM} vs {reference_arm}",
            fold_session_paired_bootstrap,
            combined_pair_metrics,
            focal_arm=PURE_ARM,
            reference_arm=reference_arm,
            expected_folds=expected_folds,
            iterations=int(iterations),
            rng_seed=int(rng_seed) + index,
        )
        is_primary = reference_arm == "no_text"
        stats.update(
            {
                "comparison_id": f"{PURE_ARM}_minus_{reference_arm}",
                "comparison_role": (
                    "primary_architecture_and_capacity_ablation"
                    if is_primary
                    else "contextual_text_reference"
                ),
                "multiplicity_family": (
                    "primary_single_comparison"
                    if is_primary
                    else "contextual_lp_references_holm2"
                ),
                "capacity_matched": False,
                "seed": DIRECT_SEED,
                "confirmatory": False,
                "inference_permitted": False,
                "claim_scope": "single_seed_retrospective_descriptive_only",
                "interpretation": INTERPRETATION_LABEL,
            }
        )
        rows.append(stats)
    result = pd.DataFrame(rows)
    result["adjusted_p_value_one_sided"] = result["p_value_one_sided"]
    context = result["comparison_role"].eq("contextual_text_reference")
    context_rows = result.loc[context]
    adjusted = holm_adjust(
        dict(
            zip(
                context_rows["comparison_id"].astype(str),
                context_rows["p_value_one_sided"].astype(float),
                strict=True,
            )
        )
    )
    result.loc[context, "adjusted_p_value_one_sided"] = context_rows[
        "comparison_id"
    ].map(adjusted)
    result["direction_favors_pure_cnn"] = result["mean_log_mae_ratio"].lt(0.0)
    result["ci_excludes_zero_favoring_pure_cnn"] = result["ci_95_upper"].lt(0.0)
    return result.sort_values(
        ["comparison_role", "reference_arm"], kind="stable"
    ).reset_index(drop=True)


def _build_persistence(
    pure_pair_metrics: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
) -> pd.DataFrame:
    stats = _wrap_direct_error(
        "failed pure-CNN persistence bootstrap",
        fold_session_persistence_bootstrap,
        pure_pair_metrics,
        arm=PURE_ARM,
        expected_folds=expected_folds,
        iterations=int(iterations),
        rng_seed=int(rng_seed),
    )
    stats.update(
        {
            "comparison_id": f"{PURE_ARM}_vs_persistence",
            "multiplicity_family": "pure_cnn_persistence_single_comparison",
            "adjusted_p_value_one_sided": stats["p_value_one_sided"],
            "direction_favors_pure_cnn": stats["mean_log_mae_ratio"] < 0.0,
            "ci_excludes_zero_favoring_pure_cnn": stats["ci_95_upper"] < 0.0,
            "seed": DIRECT_SEED,
            "confirmatory": False,
            "inference_permitted": False,
            "claim_scope": "single_seed_retrospective_descriptive_only",
            "interpretation": INTERPRETATION_LABEL,
        }
    )
    return pd.DataFrame([stats])


def analyze_pure_cnn(
    pure_pair_metrics: pd.DataFrame | str | Path,
    historical_pair_metrics: pd.DataFrame | str | Path,
    *,
    training_summary: pd.DataFrame | str | Path | None = None,
    maximum_epochs: int = 240,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_fold_counts: Mapping[str, tuple[int, int]] = (
        EXPECTED_FOLD_PAIR_SESSION_COUNTS
    ),
    expected_pure_row_count: int | None = EXPECTED_PURE_PAIR_METRIC_ROWS,
    expected_historical_row_count: int | None = 2_500,
) -> PureCnnAnalysis:
    """Validate both sources and compute all single-seed descriptive results."""

    pure = validate_pure_pair_metrics(
        pure_pair_metrics,
        expected_fold_counts=expected_fold_counts,
        expected_row_count=expected_pure_row_count,
    )
    historical = validate_historical_pair_metrics(
        historical_pair_metrics,
        expected_fold_counts=expected_fold_counts,
        expected_row_count=expected_historical_row_count,
    )
    validate_cross_experiment_lineage(pure, historical)
    validated_training = (
        None
        if training_summary is None
        else validate_training_summary(
            training_summary, pure, maximum_epochs=int(maximum_epochs)
        )
    )
    historical_context = historical[
        historical["arm"].isin(HISTORICAL_REFERENCE_ARMS)
    ].copy()
    combined = pd.concat([pure, historical_context], ignore_index=True, sort=False)
    expected_arms = {PURE_ARM, *HISTORICAL_REFERENCE_ARMS}
    if set(combined["arm"]) != expected_arms:
        raise PureCnnAnalysisError("Combined comparison arm universe drift")
    folds = compute_fold_summary(combined)
    arms = compute_arm_summary(combined, folds)
    comparisons = _build_comparisons(
        combined,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed),
    )
    persistence = _build_persistence(
        pure,
        expected_folds=tuple(expected_fold_counts),
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed) + 50_000,
    )

    pure_row = arms[arms["arm"].eq(PURE_ARM)].iloc[0]
    film_row = arms[arms["arm"].eq("no_text")].iloc[0]
    primary = comparisons[comparisons["reference_arm"].eq("no_text")].iloc[0]
    generator_parameter_delta = (
        PURE_GENERATOR_PARAMETERS - HISTORICAL_FILM_GENERATOR_PARAMETERS
    )
    total_parameter_delta = PURE_TOTAL_PARAMETERS - HISTORICAL_FILM_TOTAL_PARAMETERS
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "news_first_vol_cnn_unet_pure_no_text_seed42",
        "interpretation": INTERPRETATION_LABEL,
        "confirmatory": False,
        "claim_scope": "single_seed_retrospective_descriptive_only",
        "share_readiness": "share_with_caveats",
        "seed": DIRECT_SEED,
        "tolerance_minutes": DIRECT_TOLERANCE_MINUTES,
        "pure_arm": PURE_ARM,
        "historical_reference_arms": list(HISTORICAL_REFERENCE_ARMS),
        "folds": list(expected_fold_counts),
        "pure_job_count": int(pure["job_id"].nunique()),
        "pure_pair_metric_row_count": int(len(pure)),
        "pair_count": int(pure[["fold", "pair_id"]].drop_duplicates().shape[0]),
        "session_count": int(pure[["fold", "session_id"]].drop_duplicates().shape[0]),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_method": (
            "fold_then_paired_cme_session_cluster_recompute_fold_log_mae_ratio"
        ),
        "pure_equal_fold_mae": float(pure_row["equal_fold_mae"]),
        "pure_pooled_mae": float(pure_row["pooled_mae"]),
        "pure_equal_fold_improvement_vs_persistence_percent": float(
            pure_row["equal_fold_improvement_vs_persistence_percent"]
        ),
        "pure_pooled_improvement_vs_persistence_percent": float(
            pure_row["pooled_improvement_vs_persistence_percent"]
        ),
        "film_no_text_equal_fold_mae": float(film_row["equal_fold_mae"]),
        "film_no_text_pooled_mae": float(film_row["pooled_mae"]),
        "pure_minus_film_no_text_equal_fold_mae": float(
            pure_row["equal_fold_mae"] - film_row["equal_fold_mae"]
        ),
        "pure_vs_film_no_text_relative_mae_percent": float(
            100.0 * (pure_row["equal_fold_mae"] / film_row["equal_fold_mae"] - 1.0)
        ),
        "primary_mean_log_mae_ratio": float(primary["mean_log_mae_ratio"]),
        "primary_ci_95_lower": float(primary["ci_95_lower"]),
        "primary_ci_95_upper": float(primary["ci_95_upper"]),
        "primary_p_value_one_sided": float(primary["p_value_one_sided"]),
        "primary_pure_nonworse_fold_count": int(primary["focal_nonworse_fold_count"]),
        "pure_generator_parameters": PURE_GENERATOR_PARAMETERS,
        "film_generator_parameters": HISTORICAL_FILM_GENERATOR_PARAMETERS,
        "critic_parameters_both": CRITIC_PARAMETERS,
        "pure_total_parameters": PURE_TOTAL_PARAMETERS,
        "film_total_parameters": HISTORICAL_FILM_TOTAL_PARAMETERS,
        "generator_parameter_delta": generator_parameter_delta,
        "total_parameter_delta": total_parameter_delta,
        "pure_total_parameter_reduction_percent": float(
            100.0 * (1.0 - PURE_TOTAL_PARAMETERS / HISTORICAL_FILM_TOTAL_PARAMETERS)
        ),
        "capacity_matched_comparison": False,
        "cross_seed_inference_permitted": False,
        "training_summary_included": validated_training is not None,
        "visual_omission_reason": (
            "Exact four-fold and four-arm lookup tables preserve tiny MAE "
            "differences without a potentially misleading truncated axis."
        ),
    }
    return PureCnnAnalysis(
        pure_pair_metrics=pure,
        historical_pair_metrics=historical,
        combined_pair_metrics=combined,
        fold_summary=folds,
        arm_summary=arms,
        comparisons=comparisons,
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


def _result_sentence(analysis: PureCnnAnalysis) -> str:
    summary = analysis.summary
    delta = float(summary["pure_minus_film_no_text_equal_fold_mae"])
    relative = float(summary["pure_vs_film_no_text_relative_mae_percent"])
    direction = "低于" if delta < 0.0 else "高于"
    implication = "纯CNN数值上更好" if delta < 0.0 else "FiLM no_text数值上更好"
    return (
        f"纯CNN四fold等权MAE为 {summary['pure_equal_fold_mae']:.10f}，"
        f"{direction} FiLM no_text 的 {summary['film_no_text_equal_fold_mae']:.10f} "
        f"（相对差异 {relative:+.6f}%）；{implication}。"
    )


def render_reports(analysis: PureCnnAnalysis) -> tuple[str, str]:
    """Render answer-first Markdown and self-contained technical HTML."""

    result_sentence = _result_sentence(analysis)
    primary = analysis.comparisons[
        analysis.comparisons["reference_arm"].eq("no_text")
    ].iloc[0]
    persistence = analysis.persistence.iloc[0]
    rank_columns = (
        "equal_fold_mae_rank",
        "arm",
        "equal_fold_mae",
        "pooled_mae",
        "equal_fold_improvement_vs_persistence_percent",
        "pooled_improvement_vs_persistence_percent",
    )
    comparison_columns = (
        "comparison_role",
        "reference_arm",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "adjusted_p_value_one_sided",
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
        "best_epoch",
        "epochs_ran",
        "final_generator_lr",
        "final_discriminator_lr",
        "best_validation_score",
    )
    if analysis.training_summary is None:
        training_markdown = "_本次分析调用未提供训练汇总。_"
        training_html = "<p><em>本次分析调用未提供训练汇总。</em></p>"
    else:
        training_markdown = _markdown_table(analysis.training_summary, training_columns)
        training_html = _html_table(analysis.training_summary, training_columns)
    caveat = (
        "该结果仅有seed 42，且2023 rolling test区间已有历史暴露；"
        "此外纯CNN比FiLM模型少411,392个Generator参数，因此不是容量匹配消融。"
    )
    interval_statement = (
        f"主对比 mean log-MAE ratio={float(primary['mean_log_mae_ratio']):.8g}，"
        f"95% bootstrap CI=[{float(primary['ci_95_lower']):.8g}, "
        f"{float(primary['ci_95_upper']):.8g}]，"
        f"one-sided p={float(primary['p_value_one_sided']):.6g}。"
    )
    persistence_statement = (
        f"纯CNN相对persistence的等fold改善为 "
        f"{analysis.summary['pure_equal_fold_improvement_vs_persistence_percent']:.6f}%；"
        f"95% log-ratio CI=[{float(persistence['ci_95_lower']):.8g}, "
        f"{float(persistence['ci_95_upper']):.8g}]。"
    )
    markdown = f"""# 纯CNN U-Net无文本消融：单-Seed结果

## 技术摘要

{result_sentence} {interval_statement}

{persistence_statement}

> {caveat} 结果固定为 `retrospective_rolling_development_single_seed_descriptive`，不得解释为跨seed稳定性或FiLM的因果效应。

## 纯CNN与冻结历史对照的排名

主指标是四个rolling fold等权MAE；pooled MAE按500个pair直接汇总。数值越低越好。

{_markdown_table(analysis.arm_summary, rank_columns)}

## 配对不确定性没有消除设计限制

负的 `mean_log_mae_ratio` 表示纯CNN更好。主对比为纯CNN vs FiLM `no_text`；LP两行只作为上下文，并执行Holm-2。

{_markdown_table(analysis.comparisons, comparison_columns)}

## 四个rolling fold的结果

每个fold使用完全相同的pair、CME session、时间戳、persistence和MC64 noise bank进行跨实验比较。

{_markdown_table(analysis.fold_summary, fold_columns)}

## 数据、模型与指标定义

- 新模型：`cnn_unet_mask_coords_v1 + lp_disabled_same_shape_v1`，Generator {PURE_GENERATOR_PARAMETERS:,}参数，Critic {CRITIC_PARAMETERS:,}参数。
- 历史FiLM模型：Generator {HISTORICAL_FILM_GENERATOR_PARAMETERS:,}参数，Critic {CRITIC_PARAMETERS:,}参数。
- 数据：seed 42、5分钟alignment、4个rolling folds、{analysis.summary["pair_count"]} fold/pairs、{analysis.summary["session_count"]} fold/sessions。
- Bootstrap：{analysis.summary["bootstrap_iterations"]:,}次 `fold → paired CME-session cluster`；每次重新计算fold MAE log-ratio。
- `no_text`是与历史文本分支容量匹配的FiLM无文本对照，但它与纯CNN本身不容量匹配。

## 训练诊断

{training_markdown}

## 限制与稳健性

{caveat} 配对bootstrap量化的是当前四个fold/session样本的不确定性，不能补偿单seed、历史数据暴露或参数量差异。精确表格用于呈现极小MAE差异，避免截断坐标轴放大差距。

## 建议的下一步

1. 若纯CNN数值不差，先补3–5个seed确认初始化稳定性。
2. 再增宽纯CNN，使Generator参数接近827,745，区分容量减少与移除FiLM的作用。
3. 保持相同rolling folds和冻结MC64 noise bank；不要根据当前test结果重新选择epoch。

## 仍待回答的问题

- 容量匹配后，纯CNN与FiLM `no_text`的差异是否仍存在？
- 加入matched LP后，跨seed文本增量能否超过当前极小的架构差异？
"""
    style = """
:root{color-scheme:light dark}body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#17202a;background:#f5f7fa}main{max-width:1180px;margin:0 auto;padding:32px}.card{background:white;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 2px 12px #00000012}h1,h2{color:#16324f}.warning{border-left:5px solid #b9770e;background:#fff8e7;padding:14px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d9e1e8;padding:7px;text-align:right}th{background:#eaf0f5}th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}code{background:#eef2f5;padding:2px 4px;border-radius:4px}@media(prefers-color-scheme:dark){body{color:#e8edf2;background:#111820}.card{background:#1b2631}h1,h2{color:#b9d9f4}.warning{background:#3b321d}th{background:#263746}code{background:#263746}}
""".strip()
    html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>纯CNN U-Net无文本消融：单-Seed结果</title><style>{style}</style></head>
<body><main><h1>纯CNN U-Net无文本消融：单-Seed结果</h1>
<section class="card"><h2>技术摘要</h2><p>{html.escape(result_sentence)} {html.escape(interval_statement)}</p><p>{html.escape(persistence_statement)}</p><div class="warning">{html.escape(caveat)} 结果仅为回顾性单seed描述。</div></section>
<section class="card"><h2>纯CNN与冻结历史对照的排名</h2><p>主指标为四个rolling fold等权MAE；数值越低越好。</p>{_html_table(analysis.arm_summary, rank_columns)}</section>
<section class="card"><h2>配对不确定性没有消除设计限制</h2><p>负的mean log-MAE ratio表示纯CNN更好；LP两行仅作上下文。</p>{_html_table(analysis.comparisons, comparison_columns)}</section>
<section class="card"><h2>四个rolling fold的结果</h2><p>跨实验pair、session、时间戳、persistence和MC64 noise bank均完全一致。</p>{_html_table(analysis.fold_summary, fold_columns)}</section>
<section class="card"><h2>数据、模型与指标定义</h2><p>纯CNN G/D={PURE_GENERATOR_PARAMETERS:,}/{CRITIC_PARAMETERS:,}；历史FiLM G/D={HISTORICAL_FILM_GENERATOR_PARAMETERS:,}/{CRITIC_PARAMETERS:,}。seed=42；5分钟alignment；{analysis.summary["pair_count"]} fold/pairs；{analysis.summary["session_count"]} fold/sessions；bootstrap={analysis.summary["bootstrap_iterations"]:,}次。</p></section>
<section class="card"><h2>训练诊断</h2>{training_html}</section>
<section class="card"><h2>限制与稳健性</h2><p>{html.escape(caveat)} 配对bootstrap不能补偿这些设计限制。</p></section>
<section class="card"><h2>建议的下一步</h2><ol><li>补3–5个seed。</li><li>增宽纯CNN做参数量匹配。</li><li>保持fold和MC64 noise bank冻结，不根据test重选epoch。</li></ol></section>
<section class="card"><h2>仍待回答的问题</h2><p>容量匹配后架构差异是否仍存在，以及matched LP能否跨seed提供增量。</p></section>
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
            raise PureCnnAnalysisError(f"Existing analysis output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_analysis_bundle(
    analysis: PureCnnAnalysis,
    *,
    pure_pair_metrics_path: str | Path,
    pure_pair_metrics_sha256: str,
    historical_pair_metrics_path: str | Path,
    historical_pair_metrics_sha256: str,
    output_dir: str | Path,
    training_summary_path: str | Path | None = None,
    training_summary_sha256: str | None = None,
) -> dict[str, Path]:
    """Write deterministic CSV/JSON/Markdown/HTML and a source-addressed manifest."""

    pure_source = Path(pure_pair_metrics_path).resolve()
    historical_source = Path(historical_pair_metrics_path).resolve()
    pure_sha = _require_sha(pure_pair_metrics_sha256, "pure pair metrics SHA-256")
    historical_sha = _require_sha(
        historical_pair_metrics_sha256, "historical pair metrics SHA-256"
    )
    if sha256_file(pure_source) != pure_sha:
        raise PureCnnAnalysisError("Pure pair metrics SHA-256 drift before write")
    if sha256_file(historical_source) != historical_sha:
        raise PureCnnAnalysisError("Historical pair metrics SHA-256 drift before write")
    training_source: Path | None = None
    training_sha: str | None = None
    if training_summary_path is not None or training_summary_sha256 is not None:
        if training_summary_path is None or training_summary_sha256 is None:
            raise PureCnnAnalysisError(
                "Training summary path and SHA-256 must be supplied together"
            )
        if analysis.training_summary is None:
            raise PureCnnAnalysisError(
                "Training summary source was declared but not validated"
            )
        training_source = Path(training_summary_path).resolve()
        training_sha = _require_sha(training_summary_sha256, "training summary SHA")
        if sha256_file(training_source) != training_sha:
            raise PureCnnAnalysisError("Training summary SHA-256 drift before write")
    elif analysis.training_summary is not None:
        raise PureCnnAnalysisError(
            "Validated training summary requires a frozen path and SHA-256"
        )

    destination = Path(output_dir)
    paths = {
        "arm_summary": destination / "pure_cnn_arm_summary.csv",
        "fold_summary": destination / "pure_cnn_fold_summary.csv",
        "comparisons": destination / "pure_cnn_comparison_bootstrap.csv",
        "persistence": destination / "pure_cnn_persistence_bootstrap.csv",
        "summary": destination / "pure_cnn_analysis_summary.json",
        "report_markdown": destination / "pure_cnn_report.md",
        "report_html": destination / "pure_cnn_report.html",
        "manifest": destination / "analysis_manifest.json",
    }
    markdown, html_report = render_reports(analysis)
    for role, frame in (
        ("arm_summary", analysis.arm_summary),
        ("fold_summary", analysis.fold_summary),
        ("comparisons", analysis.comparisons),
        ("persistence", analysis.persistence),
    ):
        _atomic_write(paths[role], _csv_bytes(frame))
    summary = dict(analysis.summary)
    summary.update(
        {
            "pure_pair_metrics_path": str(pure_source),
            "pure_pair_metrics_sha256": pure_sha,
            "historical_pair_metrics_path": str(historical_source),
            "historical_pair_metrics_sha256": historical_sha,
            "training_summary_path": (
                str(training_source) if training_source is not None else None
            ),
            "training_summary_sha256": training_sha,
        }
    )
    _atomic_write(paths["summary"], _canonical_json(summary).encode("utf-8"))
    _atomic_write(paths["report_markdown"], markdown.encode("utf-8"))
    _atomic_write(paths["report_html"], html_report.encode("utf-8"))
    artifacts: list[dict[str, Any]] = []
    for role in (
        "arm_summary",
        "fold_summary",
        "comparisons",
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
            "role": "pure_pair_metrics_source",
            "path": str(pure_source),
            "sha256": pure_sha,
            "size_bytes": pure_source.stat().st_size,
        },
        {
            "role": "historical_direct_5arm_pair_metrics_source",
            "path": str(historical_source),
            "sha256": historical_sha,
            "size_bytes": historical_source.stat().st_size,
        },
    ]
    if training_source is not None and training_sha is not None:
        inputs.append(
            {
                "role": "pure_training_summary_source",
                "path": str(training_source),
                "sha256": training_sha,
                "size_bytes": training_source.stat().st_size,
            }
        )
    if (
        sha256_file(pure_source) != pure_sha
        or sha256_file(historical_source) != historical_sha
    ):
        raise PureCnnAnalysisError("An input changed while outputs were being written")
    manifest = {
        "schema_version": 1,
        "kind": "news_first_vol_cnn_unet_pure_no_text_seed42_analysis_manifest_v1",
        "audience": "technical",
        "interpretation": INTERPRETATION_LABEL,
        "confirmatory": False,
        "capacity_matched_comparison": False,
        "visual_omission_reason": analysis.summary["visual_omission_reason"],
        "inputs": inputs,
        "artifacts": artifacts,
    }
    _atomic_write(paths["manifest"], _canonical_json(manifest).encode("utf-8"))
    return paths


def _validate_config(config: Mapping[str, Any]) -> tuple[Path, str, int, int]:
    matrix = config.get("matrix")
    analysis = config.get("analysis")
    model = config.get("model")
    training = config.get("training")
    folds = config.get("folds")
    if not all(
        isinstance(value, Mapping) for value in (matrix, analysis, model, training)
    ):
        raise PureCnnAnalysisError(
            "config must contain matrix, analysis, model, and training mappings"
        )
    if not isinstance(folds, list):
        raise PureCnnAnalysisError("config folds must be a list")
    direct_arms = tuple(map(str, matrix.get("direct_arms", ())))
    if direct_arms != (PURE_ARM,):
        raise PureCnnAnalysisError("config must contain only pure_cnn_no_text")
    if tuple(map(int, matrix.get("seeds", ()))) != (DIRECT_SEED,):
        raise PureCnnAnalysisError("config seed must be exactly 42")
    for field, expected in (
        ("expected_training_jobs", EXPECTED_PURE_JOBS),
        ("expected_prediction_cells", EXPECTED_PURE_JOBS),
        ("expected_pair_metric_rows", EXPECTED_PURE_PAIR_METRIC_ROWS),
    ):
        if int(matrix.get(field, -1)) != expected:
            raise PureCnnAnalysisError(f"config matrix.{field} drift")
    if tuple(str(row.get("id", "")) for row in folds) != DIRECT_FOLDS:
        raise PureCnnAnalysisError("config rolling fold contract drift")
    if analysis.get("interpretation") != INTERPRETATION_LABEL:
        raise PureCnnAnalysisError("config interpretation label drift")
    if bool(analysis.get("cross_seed_inference_enabled", True)):
        raise PureCnnAnalysisError("cross-seed inference must be disabled")
    historical_path = _resolve_path(
        str(analysis.get("historical_pair_metrics_path", ""))
    )
    historical_sha = _require_sha(
        analysis.get("historical_pair_metrics_sha256"),
        "config historical_pair_metrics_sha256",
    )
    if historical_sha != EXPECTED_HISTORICAL_PAIR_METRICS_SHA256:
        raise PureCnnAnalysisError("Frozen historical pair-metrics SHA drift")
    if sha256_file(historical_path) != historical_sha:
        raise PureCnnAnalysisError("Frozen historical pair-metrics file drift")
    if model.get("generator_conditioning_mode") != PURE_GENERATOR_MODE:
        raise PureCnnAnalysisError("Pure-CNN Generator mode drift")
    if model.get("critic_conditioning_mode") != CRITIC_MODE:
        raise PureCnnAnalysisError("NoLP Critic mode drift")
    observed_counts = {
        "generator": int(model.get("expected_generator_parameters", -1)),
        "critic": int(model.get("expected_critic_parameters", -1)),
        "total": int(model.get("expected_total_parameters", -1)),
    }
    expected_counts = {
        "generator": PURE_GENERATOR_PARAMETERS,
        "critic": CRITIC_PARAMETERS,
        "total": PURE_TOTAL_PARAMETERS,
    }
    if observed_counts != expected_counts:
        raise PureCnnAnalysisError(
            f"Pure-CNN parameter-count contract drift: {observed_counts}"
        )
    iterations = int(analysis.get("bootstrap_replicates", -1))
    if iterations != DEFAULT_BOOTSTRAP_ITERATIONS:
        raise PureCnnAnalysisError("Production analysis requires 10,000 bootstraps")
    bootstrap_seed = int(analysis.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    maximum_epochs = int(training.get("num_epochs", -1))
    if maximum_epochs < 1:
        raise PureCnnAnalysisError("training.num_epochs must be positive")
    return historical_path, historical_sha, bootstrap_seed, maximum_epochs


def analyze_experiment(output_root: Path, config: Mapping[str, Any]) -> Path:
    """Analyze one completed canonical root and return its analysis manifest."""

    historical_path, historical_sha, bootstrap_seed, maximum_epochs = _validate_config(
        config
    )
    root = Path(output_root).resolve()
    analysis_dir = root / "analysis"
    pure_path = analysis_dir / "rq12_pair_metrics.csv.gz"
    training_path = analysis_dir / "training_summary.csv"
    pure_sha = sha256_file(pure_path)
    training_sha = sha256_file(training_path)
    result = analyze_pure_cnn(
        pure_path,
        historical_path,
        training_summary=training_path,
        maximum_epochs=maximum_epochs,
        bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
        bootstrap_seed=bootstrap_seed,
    )
    paths = write_analysis_bundle(
        result,
        pure_pair_metrics_path=pure_path,
        pure_pair_metrics_sha256=pure_sha,
        historical_pair_metrics_path=historical_path,
        historical_pair_metrics_sha256=historical_sha,
        training_summary_path=training_path,
        training_summary_sha256=training_sha,
        output_dir=analysis_dir,
    )
    return paths["manifest"]


__all__ = [
    "CRITIC_PARAMETERS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_HISTORICAL_PAIR_METRICS_SHA256",
    "HISTORICAL_REFERENCE_ARMS",
    "PURE_ARM",
    "PURE_GENERATOR_PARAMETERS",
    "PURE_TOTAL_PARAMETERS",
    "PureCnnAnalysis",
    "PureCnnAnalysisError",
    "analyze_experiment",
    "analyze_pure_cnn",
    "render_reports",
    "sha256_file",
    "validate_cross_experiment_lineage",
    "validate_historical_pair_metrics",
    "validate_pure_pair_metrics",
    "validate_training_summary",
    "write_analysis_bundle",
]
