"""RQ4 fold-pooled descriptive and rolling-OOS MAE analysis.

The input is the single pair-metric table produced after the frozen RQ1
FiLM-CNN and pure-CNN checkpoints have generated all RQ4 special-time
surfaces.  ``train``, ``validation`` and ``test`` rows stay pooled within a
checkpoint fold for the descriptive result.  Only the rows whose original
``source_split`` is ``test`` enter inferential statistics.

The module is deliberately independent of training and prediction code.  It
accepts a DataFrame or CSV/CSV.GZ, validates the full paired lineage, returns
in-memory tables through :func:`analyze_frame`, and can atomically publish a
hash-addressed artifact bundle through :func:`analyze`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_MODELS = ("film_cnn", "pure_cnn")
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
CANONICAL_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
SOURCE_SPLITS = ("train", "validation", "test")
EVENT_REGIMES = ("scheduled_only", "jump_only", "both")
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260904
LINEAGE_ATOL = 1.0e-12
NUMERICAL_TIE_ATOL = 1.0e-15

# pair/session counts in the frozen 30-minute RQ4 special-time panels.
FORMAL_SPLIT_COUNTS: Mapping[str, Mapping[str, tuple[int, int]]] = {
    "f1_2023q1": {
        "train": (53, 25),
        "validation": (37, 19),
        "test": (12, 9),
    },
    "f2_2023q2": {
        "train": (90, 44),
        "validation": (12, 9),
        "test": (26, 11),
    },
    "f3_2023q3": {
        "train": (102, 53),
        "validation": (26, 11),
        "test": (26, 9),
    },
    "f4_2023q4": {
        "train": (128, 64),
        "validation": (26, 9),
        "test": (31, 15),
    },
}
FORMAL_COMBINED_COUNTS: Mapping[str, tuple[int, int]] = {
    "f1_2023q1": (102, 53),
    "f2_2023q2": (128, 64),
    "f3_2023q3": (154, 73),
    "f4_2023q4": (185, 88),
}
FORMAL_TEST_COUNTS = {
    "pairs": 95,
    "sessions": 44,
    "scheduled_any_pairs": 70,
    "jump_any_pairs": 31,
    "both_pairs": 6,
}
REQUIRED_COLUMNS = {
    "model",
    "seed",
    "checkpoint_fold",
    "source_split",
    "pair_id",
    "session_id",
    "target_mae",
    "persistence_mae",
    "event_regime",
    "scheduled_event",
    "market_jump",
    "source_5m_membership",
}

MODEL_ALIASES = {
    "film_cnn": "film_cnn",
    "film-cnn": "film_cnn",
    "film": "film_cnn",
    "lp_matched": "film_cnn",
    "film_lp_matched": "film_cnn",
    "pure_cnn": "pure_cnn",
    "pure-cnn": "pure_cnn",
    "pure_cnn_no_text": "pure_cnn",
}
TRUE_VALUES = {
    "1",
    "1.0",
    "true",
    "t",
    "yes",
    "y",
    "in_5m",
    "in_5m_panel",
    "5m",
    "5m_overlap",
    "overlap_5m",
    "source_5m",
}
FALSE_VALUES = {
    "0",
    "0.0",
    "false",
    "f",
    "no",
    "n",
    "not_in_5m",
    "30m_only",
    "only_30m",
}


class RQ4FoldPooledAnalysisError(ValueError):
    """Raised when pair metrics violate the frozen analysis contract."""


FrameLike = pd.DataFrame | str | Path


def _load_frame(source: FrameLike) -> tuple[pd.DataFrame, str | None]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise RQ4FoldPooledAnalysisError(f"Pair-metric input is missing: {path}")
    # Reading identifiers as strings protects IDs with leading zeroes.  Numeric
    # and Boolean fields are normalized explicitly below.
    return pd.read_csv(path, dtype=str, low_memory=False), str(path)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RQ4FoldPooledAnalysisError(
            f"Pair metrics are missing required columns: {missing}"
        )
    if frame.empty:
        raise RQ4FoldPooledAnalysisError("Pair metrics are empty")


def _normalise_boolean(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise RQ4FoldPooledAnalysisError(f"{label} contains missing values")
    lowered = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(lowered) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise RQ4FoldPooledAnalysisError(
            f"{label} contains invalid Boolean values: {unknown}"
        )
    return lowered.isin(TRUE_VALUES)


def _normalise_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    identifiers = (
        "model",
        "checkpoint_fold",
        "source_split",
        "pair_id",
        "session_id",
        "event_regime",
    )
    for column in identifiers:
        if result[column].isna().any():
            raise RQ4FoldPooledAnalysisError(f"{column} contains missing values")
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise RQ4FoldPooledAnalysisError(f"{column} contains empty values")

    lowered_models = result["model"].str.lower()
    unknown_models = sorted(set(lowered_models) - set(MODEL_ALIASES))
    if unknown_models:
        raise RQ4FoldPooledAnalysisError(f"Unknown model values: {unknown_models}")
    result["model"] = lowered_models.map(MODEL_ALIASES)
    result["source_split"] = (
        result["source_split"].str.lower().replace({"val": "validation"})
    )
    result["event_regime"] = result["event_regime"].str.lower()

    seeds = pd.to_numeric(result["seed"], errors="coerce")
    if seeds.isna().any() or not np.equal(seeds, np.floor(seeds)).all():
        raise RQ4FoldPooledAnalysisError("seed must contain integers")
    result["seed"] = seeds.astype(np.int64)

    for column in ("target_mae", "persistence_mae"):
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all() or values.le(0.0).any():
            raise RQ4FoldPooledAnalysisError(
                f"{column} must be finite and strictly positive"
            )
        result[column] = values

    result["scheduled_event"] = _normalise_boolean(
        result["scheduled_event"], "scheduled_event"
    )
    result["market_jump"] = _normalise_boolean(result["market_jump"], "market_jump")
    result["source_5m_membership"] = _normalise_boolean(
        result["source_5m_membership"], "source_5m_membership"
    )
    return result


def _validate_universes(
    frame: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise RQ4FoldPooledAnalysisError("Expected seeds must be non-empty and unique")
    if not folds or len(set(folds)) != len(folds):
        raise RQ4FoldPooledAnalysisError("Expected folds must be non-empty and unique")
    observed_models = set(frame["model"])
    if observed_models != set(CANONICAL_MODELS):
        raise RQ4FoldPooledAnalysisError(
            f"Model universe drift: observed={sorted(observed_models)}"
        )
    observed_seeds = set(frame["seed"].astype(int))
    if observed_seeds != set(seeds):
        raise RQ4FoldPooledAnalysisError(
            "Seed universe drift: "
            f"missing={sorted(set(seeds) - observed_seeds)}, "
            f"extra={sorted(observed_seeds - set(seeds))}"
        )
    observed_folds = set(frame["checkpoint_fold"])
    if observed_folds != set(folds):
        raise RQ4FoldPooledAnalysisError(
            "Fold universe drift: "
            f"missing={sorted(set(folds) - observed_folds)}, "
            f"extra={sorted(observed_folds - set(folds))}"
        )
    observed_splits = set(frame["source_split"])
    if observed_splits != set(SOURCE_SPLITS):
        raise RQ4FoldPooledAnalysisError(
            f"source_split universe drift: observed={sorted(observed_splits)}"
        )
    observed_regimes = set(frame["event_regime"])
    if not observed_regimes.issubset(EVENT_REGIMES) or not observed_regimes:
        raise RQ4FoldPooledAnalysisError(
            f"event_regime universe drift: observed={sorted(observed_regimes)}"
        )
    return seeds, folds


def _validate_event_labels(frame: pd.DataFrame) -> None:
    expected = np.select(
        [
            frame["scheduled_event"] & frame["market_jump"],
            frame["scheduled_event"] & ~frame["market_jump"],
            ~frame["scheduled_event"] & frame["market_jump"],
        ],
        ["both", "scheduled_only", "jump_only"],
        default="invalid",
    )
    mismatch = frame["event_regime"].to_numpy(dtype=str) != expected
    if np.any(mismatch):
        example = frame.loc[
            mismatch,
            ["pair_id", "event_regime", "scheduled_event", "market_jump"],
        ].head(3)
        raise RQ4FoldPooledAnalysisError(
            "event_regime disagrees with scheduled_event/market_jump flags: "
            f"{example.to_dict(orient='records')}"
        )


def _validate_pairing(
    frame: pd.DataFrame,
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
) -> None:
    keys = ["model", "seed", "checkpoint_fold", "pair_id"]
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(3).to_dict(orient="records")
        raise RQ4FoldPooledAnalysisError(
            f"Duplicate model/seed/fold/pair rows: {examples}"
        )

    expected_replications = len(CANONICAL_MODELS) * len(seeds)
    replication = frame.groupby(["checkpoint_fold", "pair_id"], sort=False).size()
    bad_replication = replication[replication.ne(expected_replications)]
    if not bad_replication.empty:
        raise RQ4FoldPooledAnalysisError(
            "Every fold-routed pair must occur once for both models and every seed; "
            f"bad examples={bad_replication.head(3).to_dict()}"
        )

    expected_cells = {
        (model, int(seed), str(fold))
        for model in CANONICAL_MODELS
        for seed in seeds
        for fold in folds
    }
    observed_cells = set(
        frame[["model", "seed", "checkpoint_fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise RQ4FoldPooledAnalysisError(
            "Model/seed/fold cell universe is incomplete: "
            f"missing={sorted(expected_cells - observed_cells)}"
        )

    lineage_columns = (
        "session_id",
        "source_split",
        "event_regime",
        "scheduled_event",
        "market_jump",
        "source_5m_membership",
    )
    grouped = frame.groupby(["checkpoint_fold", "pair_id"], sort=False)
    for column in lineage_columns:
        drift = grouped[column].nunique(dropna=False).ne(1)
        if drift.any():
            examples = list(drift[drift].index[:3])
            raise RQ4FoldPooledAnalysisError(
                f"Paired {column} lineage drift for fold/pairs: {examples}"
            )
    persistence_span = grouped["persistence_mae"].agg(
        lambda values: float(values.max() - values.min())
    )
    if persistence_span.gt(LINEAGE_ATOL).any():
        examples = list(persistence_span[persistence_span.gt(LINEAGE_ATOL)].index[:3])
        raise RQ4FoldPooledAnalysisError(
            f"Paired persistence_mae lineage drift for fold/pairs: {examples}"
        )


def _panel_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "checkpoint_fold",
        "source_split",
        "pair_id",
        "session_id",
        "event_regime",
        "scheduled_event",
        "market_jump",
        "source_5m_membership",
        "persistence_mae",
    ]
    return (
        frame[columns]
        .drop_duplicates(["checkpoint_fold", "pair_id"])
        .sort_values(["checkpoint_fold", "pair_id"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_formal_counts(frame: pd.DataFrame) -> dict[str, Any]:
    panel = _panel_rows(frame)
    for fold, split_counts in FORMAL_SPLIT_COUNTS.items():
        fold_rows = panel.loc[panel["checkpoint_fold"].eq(fold)]
        for split, (expected_pairs, expected_sessions) in split_counts.items():
            rows = fold_rows.loc[fold_rows["source_split"].eq(split)]
            observed = (len(rows), int(rows["session_id"].nunique()))
            expected = (int(expected_pairs), int(expected_sessions))
            if observed != expected:
                raise RQ4FoldPooledAnalysisError(
                    f"Formal panel count drift for {fold}/{split}: "
                    f"expected={expected}, observed={observed}"
                )
        expected_combined = FORMAL_COMBINED_COUNTS[fold]
        observed_combined = (len(fold_rows), int(fold_rows["session_id"].nunique()))
        if observed_combined != expected_combined:
            raise RQ4FoldPooledAnalysisError(
                f"Formal combined count drift for {fold}: "
                f"expected={expected_combined}, observed={observed_combined}"
            )

    test = panel.loc[panel["source_split"].eq("test")]
    if test["pair_id"].duplicated().any():
        raise RQ4FoldPooledAnalysisError(
            "Formal rolling-test pair IDs must be globally unique across folds"
        )
    observed_test = {
        "pairs": len(test),
        "sessions": int(test["session_id"].nunique()),
        "scheduled_any_pairs": int(test["scheduled_event"].sum()),
        "jump_any_pairs": int(test["market_jump"].sum()),
        "both_pairs": int(test["event_regime"].eq("both").sum()),
    }
    if observed_test != FORMAL_TEST_COUNTS:
        raise RQ4FoldPooledAnalysisError(
            "Formal rolling-test counts drift: "
            f"expected={FORMAL_TEST_COUNTS}, observed={observed_test}"
        )
    expected_rows = (
        sum(value[0] for value in FORMAL_COMBINED_COUNTS.values())
        * len(CANONICAL_MODELS)
        * len(CANONICAL_SEEDS)
    )
    if len(frame) != expected_rows:
        raise RQ4FoldPooledAnalysisError(
            f"Formal metric-row count drift: expected={expected_rows}, "
            f"observed={len(frame)}"
        )
    return {
        "metric_rows": len(frame),
        "fold_routed_pairs": len(panel),
        "test_pairs": len(test),
        "test_sessions": int(test["session_id"].nunique()),
    }


def normalise_pair_metrics(
    source: FrameLike,
    *,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    strict_design_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and fail-closed validate the paired fold-pooled metric table."""

    raw, source_path = _load_frame(source)
    _require_columns(raw)
    frame = _normalise_identifiers(raw)
    seeds, folds = _validate_universes(
        frame, expected_seeds=expected_seeds, expected_folds=expected_folds
    )
    if strict_design_counts and (seeds != CANONICAL_SEEDS or folds != CANONICAL_FOLDS):
        raise RQ4FoldPooledAnalysisError(
            "Formal count validation requires the canonical 10 seeds and four folds"
        )
    _validate_event_labels(frame)
    _validate_pairing(frame, seeds=seeds, folds=folds)
    formal_counts = _validate_formal_counts(frame) if strict_design_counts else None
    order = ["model", "seed", "checkpoint_fold", "source_split", "pair_id"]
    frame = frame.sort_values(order, kind="stable").reset_index(drop=True)
    metadata = {
        "source_path": source_path,
        "expected_seeds": list(seeds),
        "expected_folds": list(folds),
        "strict_design_counts": bool(strict_design_counts),
        "formal_counts": formal_counts,
    }
    return frame, metadata


def _persistence_skill_percent(model_mae: float, persistence_mae: float) -> float:
    if not math.isfinite(model_mae) or not math.isfinite(persistence_mae):
        raise RQ4FoldPooledAnalysisError("Summary MAEs must be finite")
    if model_mae <= 0.0 or persistence_mae <= 0.0:
        raise RQ4FoldPooledAnalysisError("Summary MAEs must be positive")
    return 100.0 * (1.0 - model_mae / persistence_mae)


def _cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "checkpoint_fold"]
    result = (
        frame.groupby(keys, as_index=False, sort=True)
        .agg(
            mean_target_mae=("target_mae", "mean"),
            mean_persistence_mae=("persistence_mae", "mean"),
            pair_count=("pair_id", "size"),
            session_count=("session_id", "nunique"),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    split_counts = (
        frame.groupby(keys + ["source_split"], sort=True)
        .size()
        .unstack("source_split", fill_value=0)
        .reindex(columns=SOURCE_SPLITS, fill_value=0)
        .rename(columns={split: f"{split}_pair_count" for split in SOURCE_SPLITS})
        .reset_index()
    )
    result = result.merge(split_counts, on=keys, how="left", validate="one_to_one")
    result["persistence_skill_percent"] = 100.0 * (
        1.0 - result["mean_target_mae"] / result["mean_persistence_mae"]
    )
    return result


def _model_summary(frame: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in CANONICAL_MODELS:
        model_rows = frame.loc[frame["model"].eq(model)]
        model_cells = cells.loc[cells["model"].eq(model)]
        equal_target = float(model_cells["mean_target_mae"].mean())
        equal_persistence = float(model_cells["mean_persistence_mae"].mean())
        pooled_target = float(model_rows["target_mae"].mean())
        pooled_persistence = float(model_rows["persistence_mae"].mean())
        rows.append(
            {
                "model": model,
                "equal_cell_mean_target_mae": equal_target,
                "equal_cell_mean_persistence_mae": equal_persistence,
                "equal_cell_persistence_skill_percent": (
                    _persistence_skill_percent(equal_target, equal_persistence)
                ),
                "pair_weighted_mean_target_mae": pooled_target,
                "pair_weighted_mean_persistence_mae": pooled_persistence,
                "pair_weighted_persistence_skill_percent": (
                    _persistence_skill_percent(pooled_target, pooled_persistence)
                ),
                "cell_count": len(model_cells),
                "metric_row_count": len(model_rows),
                "fold_routed_pair_count": int(
                    model_rows[["checkpoint_fold", "pair_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "unique_pair_count": int(model_rows["pair_id"].nunique()),
                "unique_session_count": int(model_rows["session_id"].nunique()),
                "seed_count": int(model_rows["seed"].nunique()),
                "fold_count": int(model_rows["checkpoint_fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _cell_model_comparisons(cells: pd.DataFrame) -> pd.DataFrame:
    index = ["seed", "checkpoint_fold"]
    target = cells.pivot(index=index, columns="model", values="mean_target_mae")
    persistence = cells.pivot(
        index=index, columns="model", values="mean_persistence_mae"
    )
    if target.isna().any().any() or set(target.columns) != set(CANONICAL_MODELS):
        raise RQ4FoldPooledAnalysisError("Combined model cells are not exactly paired")
    if not np.allclose(
        persistence["film_cnn"].to_numpy(float),
        persistence["pure_cnn"].to_numpy(float),
        rtol=0.0,
        atol=LINEAGE_ATOL,
    ):
        raise RQ4FoldPooledAnalysisError(
            "Combined model cells have mismatched persistence MAE"
        )
    result = target.rename(
        columns={
            "film_cnn": "film_cnn_mean_target_mae",
            "pure_cnn": "pure_cnn_mean_target_mae",
        }
    ).reset_index()
    result["mean_persistence_mae"] = persistence["film_cnn"].to_numpy(float)
    result["film_to_pure_mae_ratio"] = (
        result["film_cnn_mean_target_mae"] / result["pure_cnn_mean_target_mae"]
    )
    result["log_film_to_pure_mae_ratio"] = np.log(result["film_to_pure_mae_ratio"])
    result["film_improvement_percent"] = 100.0 * (
        1.0 - result["film_to_pure_mae_ratio"]
    )
    difference = result["film_cnn_mean_target_mae"] - result["pure_cnn_mean_target_mae"]
    result["winning_model"] = np.where(
        np.isclose(difference, 0.0, rtol=0.0, atol=NUMERICAL_TIE_ATOL),
        "tie",
        np.where(difference < 0.0, "film_cnn", "pure_cnn"),
    )
    return result.sort_values(index, kind="stable").reset_index(drop=True)


def _combined_comparison_summary(
    frame: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    seed_points = comparisons.groupby("seed", sort=True)[
        "log_film_to_pure_mae_ratio"
    ].mean()
    fold_points = comparisons.groupby("checkpoint_fold", sort=True)[
        "log_film_to_pure_mae_ratio"
    ].mean()
    point = float(comparisons["log_film_to_pure_mae_ratio"].mean())
    film = frame.loc[frame["model"].eq("film_cnn")]
    pure = frame.loc[frame["model"].eq("pure_cnn")]
    film_pooled = float(film["target_mae"].mean())
    pure_pooled = float(pure["target_mae"].mean())
    return pd.DataFrame(
        [
            {
                "comparison_id": "film_cnn_vs_pure_cnn_combined_descriptive",
                "inference_role": "descriptive_only_contains_train_and_validation",
                "equal_cell_mean_log_mae_ratio": point,
                "equal_cell_geometric_mae_ratio": math.exp(point),
                "equal_cell_geometric_improvement_percent": 100.0
                * (1.0 - math.exp(point)),
                "pair_weighted_film_cnn_mae": film_pooled,
                "pair_weighted_pure_cnn_mae": pure_pooled,
                "pair_weighted_mae_ratio": film_pooled / pure_pooled,
                "pair_weighted_improvement_percent": 100.0
                * (1.0 - film_pooled / pure_pooled),
                "film_winning_cell_count": int(
                    comparisons["winning_model"].eq("film_cnn").sum()
                ),
                "pure_winning_cell_count": int(
                    comparisons["winning_model"].eq("pure_cnn").sum()
                ),
                "tied_cell_count": int(comparisons["winning_model"].eq("tie").sum()),
                "film_winning_seed_count": int(seed_points.lt(0.0).sum()),
                "film_winning_fold_count": int(fold_points.lt(0.0).sum()),
                "seed_count": len(seed_points),
                "fold_count": len(fold_points),
            }
        ]
    )


def _strata() -> tuple[tuple[str, str], ...]:
    return (
        ("special_scope", "all_special"),
        ("special_scope", "scheduled_any"),
        ("special_scope", "market_jump_any"),
        ("event_regime", "scheduled_only"),
        ("event_regime", "jump_only"),
        ("event_regime", "both"),
        ("alignment", "source_5m_overlap"),
        ("alignment", "source_30m_only"),
        ("source_split", "train"),
        ("source_split", "validation"),
        ("source_split", "test"),
    )


def _stratum_mask(frame: pd.DataFrame, group: str, value: str) -> pd.Series:
    if value == "all_special":
        return pd.Series(True, index=frame.index)
    if value == "scheduled_any":
        return frame["scheduled_event"]
    if value == "market_jump_any":
        return frame["market_jump"]
    if group == "event_regime":
        return frame["event_regime"].eq(value)
    if value == "source_5m_overlap":
        return frame["source_5m_membership"]
    if value == "source_30m_only":
        return ~frame["source_5m_membership"]
    if group == "source_split":
        return frame["source_split"].eq(value)
    raise RQ4FoldPooledAnalysisError(f"Unknown stratum: {group}/{value}")


def _strata_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum_group, stratum in _strata():
        selected = frame.loc[_stratum_mask(frame, stratum_group, stratum)]
        if selected.empty:
            continue
        cells = selected.groupby(
            ["model", "seed", "checkpoint_fold"], as_index=False, sort=True
        ).agg(
            mean_target_mae=("target_mae", "mean"),
            mean_persistence_mae=("persistence_mae", "mean"),
        )
        for model in CANONICAL_MODELS:
            model_rows = selected.loc[selected["model"].eq(model)]
            model_cells = cells.loc[cells["model"].eq(model)]
            equal_target = float(model_cells["mean_target_mae"].mean())
            equal_persistence = float(model_cells["mean_persistence_mae"].mean())
            pooled_target = float(model_rows["target_mae"].mean())
            pooled_persistence = float(model_rows["persistence_mae"].mean())
            rows.append(
                {
                    "stratum_group": stratum_group,
                    "stratum": stratum,
                    "model": model,
                    "equal_cell_mean_target_mae": equal_target,
                    "equal_cell_mean_persistence_mae": equal_persistence,
                    "equal_cell_persistence_skill_percent": (
                        _persistence_skill_percent(equal_target, equal_persistence)
                    ),
                    "pair_weighted_mean_target_mae": pooled_target,
                    "pair_weighted_mean_persistence_mae": pooled_persistence,
                    "pair_weighted_persistence_skill_percent": (
                        _persistence_skill_percent(pooled_target, pooled_persistence)
                    ),
                    "contributing_cell_count": len(model_cells),
                    "metric_row_count": len(model_rows),
                    "fold_routed_pair_count": int(
                        model_rows[["checkpoint_fold", "pair_id"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "unique_session_count": int(model_rows["session_id"].nunique()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["stratum_group", "stratum", "model"], kind="stable")
        .reset_index(drop=True)
    )


def _paired_test_models(test: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "checkpoint_fold", "pair_id"]
    audit = [
        "session_id",
        "source_split",
        "event_regime",
        "scheduled_event",
        "market_jump",
        "source_5m_membership",
        "persistence_mae",
    ]
    film = test.loc[test["model"].eq("film_cnn"), keys + audit + ["target_mae"]]
    film = film.rename(
        columns={
            **{column: f"{column}_film" for column in audit},
            "target_mae": "focal_mae",
        }
    )
    pure = test.loc[test["model"].eq("pure_cnn"), keys + audit + ["target_mae"]]
    pure = pure.rename(
        columns={
            **{column: f"{column}_pure" for column in audit},
            "target_mae": "reference_mae",
        }
    )
    paired = film.merge(
        pure, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise RQ4FoldPooledAnalysisError(
            "Test-only FiLM/pure pair universe is not one-to-one"
        )
    paired = paired.drop(columns="_merge")
    for column in audit:
        left = paired[f"{column}_film"]
        right = paired[f"{column}_pure"]
        if column == "persistence_mae":
            matches = np.isclose(
                left.to_numpy(float),
                right.to_numpy(float),
                rtol=0.0,
                atol=LINEAGE_ATOL,
            )
        else:
            matches = left.to_numpy() == right.to_numpy()
        if not np.all(matches):
            raise RQ4FoldPooledAnalysisError(
                f"Test-only FiLM/pure {column} lineage drift"
            )
        paired[column] = left
        paired = paired.drop(columns=[f"{column}_film", f"{column}_pure"])
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def _cell_log_ratios(paired: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "checkpoint_fold",
        "pair_id",
        "session_id",
        "focal_mae",
        "reference_mae",
    }
    missing = sorted(required - set(paired.columns))
    if missing:
        raise RQ4FoldPooledAnalysisError(
            f"Paired bootstrap input is missing columns: {missing}"
        )
    cells = paired.groupby(["seed", "checkpoint_fold"], as_index=False, sort=True).agg(
        focal_mae=("focal_mae", "mean"),
        reference_mae=("reference_mae", "mean"),
        pair_count=("pair_id", "size"),
        session_count=("session_id", "nunique"),
    )
    if (
        not np.isfinite(cells[["focal_mae", "reference_mae"]].to_numpy()).all()
        or cells[["focal_mae", "reference_mae"]].le(0.0).any().any()
    ):
        raise RQ4FoldPooledAnalysisError("Paired cell MAEs must be finite and positive")
    cells["log_mae_ratio"] = np.log(cells["focal_mae"] / cells["reference_mae"])
    return cells


def seed_fold_session_bootstrap(
    paired: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Equal-cell log-MAE bootstrap: seed -> fold -> paired CME session."""

    if int(iterations) < 2:
        raise RQ4FoldPooledAnalysisError("Bootstrap iterations must be at least two")
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    cell_points = _cell_log_ratios(paired)
    expected_cells = {(seed, fold) for seed in seeds for fold in folds}
    observed_cells = set(
        cell_points[["seed", "checkpoint_fold"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise RQ4FoldPooledAnalysisError(
            "Bootstrap seed/fold cells drift: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    hierarchy: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seed in seeds:
        for fold in folds:
            cell = paired.loc[
                paired["seed"].eq(seed) & paired["checkpoint_fold"].eq(fold)
            ]
            sessions = cell.groupby("session_id", as_index=False, sort=True).agg(
                focal_sum=("focal_mae", "sum"),
                reference_sum=("reference_mae", "sum"),
                pair_count=("pair_id", "size"),
            )
            if sessions.empty:
                raise RQ4FoldPooledAnalysisError(
                    f"Bootstrap cell has no sessions: seed={seed}, fold={fold}"
                )
            hierarchy[(seed, fold)] = (
                sessions["focal_sum"].to_numpy(float),
                sessions["reference_sum"].to_numpy(float),
                sessions["pair_count"].to_numpy(np.int64),
            )

    point = float(cell_points["log_mae_ratio"].mean())
    seed_points = (
        cell_points.groupby("seed", sort=True)["log_mae_ratio"].mean().to_dict()
    )
    fold_points = (
        cell_points.groupby("checkpoint_fold", sort=True)["log_mae_ratio"]
        .mean()
        .to_dict()
    )
    rng = np.random.default_rng(int(rng_seed))
    seed_array = np.asarray(seeds, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=object)
    draws = np.empty(int(iterations), dtype=float)
    for draw_index in range(int(iterations)):
        draw_cells: list[float] = []
        for sampled_seed in rng.choice(seed_array, size=len(seed_array), replace=True):
            for sampled_fold in rng.choice(
                fold_array, size=len(fold_array), replace=True
            ):
                focal, reference, counts = hierarchy[
                    (int(sampled_seed), str(sampled_fold))
                ]
                indexes = rng.integers(0, len(counts), size=len(counts))
                focal_sum = float(focal[indexes].sum())
                reference_sum = float(reference[indexes].sum())
                sampled_pair_count = int(counts[indexes].sum())
                if sampled_pair_count <= 0 or focal_sum <= 0.0 or reference_sum <= 0.0:
                    raise RQ4FoldPooledAnalysisError(
                        "Bootstrap draw has a non-positive aggregate"
                    )
                # The pair count is shared by focal/reference, so it cancels.
                draw_cells.append(math.log(focal_sum / reference_sum))
        draws[draw_index] = float(np.mean(draw_cells))

    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    return {
        "focal_equal_cell_mean_mae": float(cell_points["focal_mae"].mean()),
        "reference_equal_cell_mean_mae": float(cell_points["reference_mae"].mean()),
        "mean_log_mae_ratio": point,
        "geometric_mae_ratio": math.exp(point),
        "geometric_improvement_percent": 100.0 * (1.0 - math.exp(point)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "geometric_mae_ratio_ci_95_lower": math.exp(float(lower)),
        "geometric_mae_ratio_ci_95_upper": math.exp(float(upper)),
        "p_value_one_sided_focal_better": float(p_negative),
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_negative, p_positive))),
        "consistent_seed_count": int(
            sum(float(value) < 0.0 for value in seed_points.values())
        ),
        "consistent_fold_count": int(
            sum(float(value) < 0.0 for value in fold_points.values())
        ),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "cell_count": len(cell_points),
        "fold_routed_pair_count": int(
            paired[["checkpoint_fold", "pair_id"]].drop_duplicates().shape[0]
        ),
        "unique_pair_count": int(paired["pair_id"].nunique()),
        "unique_session_count": int(paired["session_id"].nunique()),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(rng_seed),
        "seed_log_mae_ratios_json": json.dumps(
            {str(key): float(value) for key, value in seed_points.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "fold_log_mae_ratios_json": json.dumps(
            {str(key): float(value) for key, value in fold_points.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "resampling_method": (
            "seed_then_fold_then_paired_cme_session_cluster_"
            "recompute_equal_cell_log_mae_ratio"
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Deterministic Holm step-down adjustment."""

    if not p_values:
        return {}
    checked: list[tuple[str, float]] = []
    for name, value in p_values.items():
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RQ4FoldPooledAnalysisError(
                "Holm p-values must be finite and lie in [0,1]"
            )
        checked.append((str(name), number))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    total = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw))
        adjusted[name] = running
    return adjusted


def _test_oos_summary(test: pd.DataFrame) -> pd.DataFrame:
    cells = _cell_summary(test)
    summary = _model_summary(test, cells)
    summary.insert(1, "evidence_role", "rolling_oos_retrospective_development")
    return summary


def _direction_rows(
    paired: pd.DataFrame,
    *,
    comparison_id: str,
    focal: str,
    reference: str,
) -> list[dict[str, Any]]:
    cells = _cell_log_ratios(paired)
    rows: list[dict[str, Any]] = []
    for dimension, column in (("seed", "seed"), ("fold", "checkpoint_fold")):
        points = cells.groupby(column, sort=True)["log_mae_ratio"].mean()
        for group_id, value in points.items():
            if math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=NUMERICAL_TIE_ATOL):
                direction = "tie"
            elif float(value) < 0.0:
                direction = f"{focal}_better"
            else:
                direction = f"{reference}_better"
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "focal": focal,
                    "reference": reference,
                    "dimension": dimension,
                    "group_id": str(group_id),
                    "mean_log_mae_ratio": float(value),
                    "direction": direction,
                }
            )
    return rows


def _test_bootstrap_tables(
    test: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    alpha: float,
    minimum_consistent_seeds: int,
    minimum_consistent_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_models = _paired_test_models(test)
    specifications: list[tuple[str, str, str, pd.DataFrame, int, bool]] = [
        (
            "film_cnn_vs_pure_cnn",
            "film_cnn",
            "pure_cnn",
            paired_models,
            int(bootstrap_seed),
            True,
        )
    ]
    for offset, model in enumerate(CANONICAL_MODELS, start=1):
        selected = test.loc[test["model"].eq(model)].copy()
        paired = selected[
            [
                "seed",
                "checkpoint_fold",
                "pair_id",
                "session_id",
                "target_mae",
                "persistence_mae",
            ]
        ].rename(
            columns={
                "target_mae": "focal_mae",
                "persistence_mae": "reference_mae",
            }
        )
        specifications.append(
            (
                f"{model}_vs_persistence",
                model,
                "persistence",
                paired,
                int(bootstrap_seed) + 1009 * offset,
                False,
            )
        )

    result_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for comparison_id, focal, reference, paired, rng_seed, is_primary in specifications:
        row = seed_fold_session_bootstrap(
            paired,
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
            iterations=int(bootstrap_iterations),
            rng_seed=rng_seed,
        )
        row.update(
            {
                "comparison_id": comparison_id,
                "focal": focal,
                "reference": reference,
                "evidence_role": "rolling_oos_retrospective_development",
                "test_type": (
                    "two_sided_primary" if is_primary else "one_sided_focal_better"
                ),
                "holm_family": (
                    "none_single_primary"
                    if is_primary
                    else "models_vs_persistence_holm2"
                ),
                "holm_family_size": 1 if is_primary else 2,
                "alpha": float(alpha),
                "required_consistent_seed_count": int(minimum_consistent_seeds),
                "required_consistent_fold_count": int(minimum_consistent_folds),
            }
        )
        result_rows.append(row)
        direction_rows.extend(
            _direction_rows(
                paired,
                comparison_id=comparison_id,
                focal=focal,
                reference=reference,
            )
        )

    result = pd.DataFrame(result_rows)
    persistence_mask = result["holm_family"].eq("models_vs_persistence_holm2")
    raw_p = dict(
        zip(
            result.loc[persistence_mask, "comparison_id"].astype(str),
            result.loc[persistence_mask, "p_value_one_sided_focal_better"].astype(
                float
            ),
            strict=True,
        )
    )
    if len(raw_p) != 2:
        raise RQ4FoldPooledAnalysisError(
            "Models-versus-persistence family must contain exactly two tests"
        )
    adjusted = holm_adjust(raw_p)
    result["holm_adjusted_p"] = np.nan
    result.loc[persistence_mask, "holm_adjusted_p"] = result.loc[
        persistence_mask, "comparison_id"
    ].map(adjusted)
    primary_mask = result["comparison_id"].eq("film_cnn_vs_pure_cnn")
    result["passes_evidence_gate"] = False
    common_gate = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["consistent_seed_count"].ge(int(minimum_consistent_seeds))
        & result["consistent_fold_count"].ge(int(minimum_consistent_folds))
    )
    result.loc[primary_mask, "passes_evidence_gate"] = common_gate[
        primary_mask
    ] & result.loc[primary_mask, "p_value_two_sided"].lt(float(alpha))
    result.loc[persistence_mask, "passes_evidence_gate"] = common_gate[
        persistence_mask
    ] & result.loc[persistence_mask, "holm_adjusted_p"].lt(float(alpha))
    result["difference_direction"] = "negative_log_ratio_favours_focal"
    direction = (
        pd.DataFrame(direction_rows)
        .sort_values(["comparison_id", "dimension", "group_id"], kind="stable")
        .reset_index(drop=True)
    )
    return (
        result.sort_values("comparison_id", kind="stable").reset_index(drop=True),
        direction,
    )


def analyze_frame(
    source: FrameLike,
    *,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
    minimum_consistent_seeds: int = 7,
    minimum_consistent_folds: int = 3,
    strict_design_counts: bool = True,
) -> dict[str, Any]:
    """Validate evidence and build all descriptive and rolling-OOS tables."""

    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    if not 0.0 < float(alpha) < 1.0:
        raise RQ4FoldPooledAnalysisError("alpha must lie in (0,1)")
    if not 1 <= int(minimum_consistent_seeds) <= len(seeds):
        raise RQ4FoldPooledAnalysisError("Invalid minimum_consistent_seeds")
    if not 1 <= int(minimum_consistent_folds) <= len(folds):
        raise RQ4FoldPooledAnalysisError("Invalid minimum_consistent_folds")
    if int(bootstrap_iterations) < 2:
        raise RQ4FoldPooledAnalysisError("Bootstrap iterations must be at least two")

    frame, metadata = normalise_pair_metrics(
        source,
        expected_seeds=seeds,
        expected_folds=folds,
        strict_design_counts=bool(strict_design_counts),
    )
    if (
        strict_design_counts
        and int(bootstrap_iterations) != DEFAULT_BOOTSTRAP_ITERATIONS
    ):
        raise RQ4FoldPooledAnalysisError(
            "Formal RQ4 inference requires exactly 10,000 bootstrap iterations"
        )
    cells = _cell_summary(frame)
    cell_comparisons = _cell_model_comparisons(cells)
    test = frame.loc[frame["source_split"].eq("test")].copy()
    bootstrap, direction = _test_bootstrap_tables(
        test,
        expected_seeds=seeds,
        expected_folds=folds,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
        alpha=float(alpha),
        minimum_consistent_seeds=int(minimum_consistent_seeds),
        minimum_consistent_folds=int(minimum_consistent_folds),
    )
    metadata.update(
        {
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
            "alpha": float(alpha),
            "minimum_consistent_seeds": int(minimum_consistent_seeds),
            "minimum_consistent_folds": int(minimum_consistent_folds),
            "metric_rows": len(frame),
            "fold_routed_pairs": int(
                frame[["checkpoint_fold", "pair_id"]].drop_duplicates().shape[0]
            ),
            "test_unique_pairs": int(test["pair_id"].nunique()),
            "test_unique_sessions": int(test["session_id"].nunique()),
            "inference_scope": "source_split_test_only",
            "combined_scope": "train_validation_test_descriptive_only",
        }
    )
    return {
        "combined_cell_summary": cells,
        "combined_model_summary": _model_summary(frame, cells),
        "combined_cell_comparisons": cell_comparisons,
        "combined_comparison_summary": _combined_comparison_summary(
            frame, cell_comparisons
        ),
        "combined_strata_summary": _strata_summary(frame),
        "test_only_oos_summary": _test_oos_summary(test),
        "test_only_bootstrap": bootstrap,
        "test_only_direction_consistency": direction,
        "metadata": metadata,
        "_normalised_metrics": frame,
    }


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(REQUIRED_COLUMNS)
    payload = (
        frame[columns]
        .to_csv(index=False, lineterminator="\n", float_format="%.17g")
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    temporary.replace(path)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return f"{number:.9g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_markdown_report(result: Mapping[str, Any]) -> str:
    """Render a concise, deterministic human-readable results report."""

    models = result["combined_model_summary"]
    comparison = result["combined_comparison_summary"]
    strata = result["combined_strata_summary"]
    oos = result["test_only_oos_summary"]
    bootstrap = result["test_only_bootstrap"]
    metadata = result["metadata"]
    requested_strata = strata.loc[
        strata["stratum"].isin(
            ["scheduled_any", "market_jump_any", "both", "source_30m_only"]
        )
    ].copy()

    sections = [
        "# RQ4 Fold-Pooled RQ1 Transfer Evaluation",
        "",
        (
            "This report evaluates the frozen RQ1 FiLM-CNN (`lp_matched`) and "
            "Pure-CNN (`pure_cnn_no_text`) checkpoints on the RQ4 tolerance-30m "
            "special-time panels. Each prediction is the mean of 64 stable Gaussian "
            "draws and every MAE uses the current/target raw-support intersection."
        ),
        "",
        "## Coverage",
        "",
        (
            f"The analysis contains {int(metadata['metric_rows']):,} model records "
            f"over {int(metadata['fold_routed_pairs']):,} fold-routed pairs. The "
            f"test-only appendix contains {int(metadata['test_unique_pairs'])} "
            f"unique pairs from {int(metadata['test_unique_sessions'])} CME sessions."
        ),
        "",
        "## Combined train + validation + test result",
        "",
        (
            "These results are descriptive only because they include training rows "
            "and validation rows used for checkpoint selection."
        ),
        "",
        _markdown_table(
            models,
            (
                "model",
                "equal_cell_mean_target_mae",
                "equal_cell_mean_persistence_mae",
                "equal_cell_persistence_skill_percent",
                "pair_weighted_mean_target_mae",
                "pair_weighted_mean_persistence_mae",
                "pair_weighted_persistence_skill_percent",
            ),
        ),
        "",
        _markdown_table(
            comparison,
            (
                "equal_cell_geometric_mae_ratio",
                "equal_cell_geometric_improvement_percent",
                "pair_weighted_mae_ratio",
                "film_winning_cell_count",
                "pure_winning_cell_count",
                "film_winning_seed_count",
                "film_winning_fold_count",
            ),
        ),
        "",
        (
            "All 80 model-seed-fold cell values and their win directions are in "
            "`combined_cell_summary.csv` and `combined_cell_comparisons.csv`."
        ),
        "",
        "## Special-time strata",
        "",
        _markdown_table(
            requested_strata,
            (
                "stratum",
                "model",
                "equal_cell_mean_target_mae",
                "equal_cell_persistence_skill_percent",
                "pair_weighted_mean_target_mae",
                "fold_routed_pair_count",
            ),
        ),
        "",
        "## Test-only rolling-OOS appendix",
        "",
        (
            "This is retrospective rolling out-of-sample development evidence, not "
            "an unobserved confirmatory holdout."
        ),
        "",
        _markdown_table(
            oos,
            (
                "model",
                "equal_cell_mean_target_mae",
                "equal_cell_mean_persistence_mae",
                "equal_cell_persistence_skill_percent",
                "pair_weighted_mean_target_mae",
                "pair_weighted_persistence_skill_percent",
            ),
        ),
        "",
        _markdown_table(
            bootstrap,
            (
                "comparison_id",
                "geometric_mae_ratio",
                "geometric_mae_ratio_ci_95_lower",
                "geometric_mae_ratio_ci_95_upper",
                "p_value_two_sided",
                "p_value_one_sided_focal_better",
                "holm_adjusted_p",
                "consistent_seed_count",
                "consistent_fold_count",
                "passes_evidence_gate",
            ),
        ),
        "",
        (
            "The ratio confidence limits above are exponentiated from the 95% "
            "bootstrap interval on the equal-cell mean log-MAE ratio. Persistence "
            "comparisons use one-sided tests with Holm correction across two models."
        ),
        "",
    ]
    return "\n".join(sections)


ARTIFACT_FILENAMES = {
    "combined_cell_summary": "combined_cell_summary.csv",
    "combined_model_summary": "combined_model_summary.csv",
    "combined_cell_comparisons": "combined_cell_comparisons.csv",
    "combined_comparison_summary": "combined_comparison_summary.csv",
    "combined_strata_summary": "combined_strata_summary.csv",
    "test_only_oos_summary": "test_only_oos_summary.csv",
    "test_only_bootstrap": "test_only_bootstrap.csv",
    "test_only_direction_consistency": "test_only_direction_consistency.csv",
}


def analyze(
    source: FrameLike,
    output_dir: str | Path,
    *,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
    minimum_consistent_seeds: int = 7,
    minimum_consistent_folds: int = 3,
    strict_design_counts: bool = True,
) -> dict[str, Path]:
    """Run analysis and write a self-auditing artifact bundle."""

    result = analyze_frame(
        source,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
        minimum_consistent_seeds=minimum_consistent_seeds,
        minimum_consistent_folds=minimum_consistent_folds,
        strict_design_counts=strict_design_counts,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for key, filename in ARTIFACT_FILENAMES.items():
        table = result[key]
        if not isinstance(table, pd.DataFrame):
            raise RQ4FoldPooledAnalysisError(f"Analysis result {key} is not a table")
        path = destination / filename
        _write_csv_atomic(table, path)
        paths[key] = path
        artifacts[key] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "rows": len(table),
            "columns": list(table.columns),
        }

    report_path = destination / "rq4_fold_pooled_report.md"
    _write_text_atomic(render_markdown_report(result), report_path)
    paths["report"] = report_path

    metrics = result["_normalised_metrics"]
    metadata = dict(result["metadata"])
    manifest_payload = {
        "schema_version": 1,
        "kind": "rq4_fold_pooled_analysis_manifest_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": metadata.pop("source_path"),
        "input_canonical_sha256": _canonical_frame_sha256(metrics),
        "paired_lineage_validated": True,
        "duplicate_rows_absent": True,
        "combined_results_are_descriptive_only": True,
        "test_only_results_are_rolling_oos_retrospective_development": True,
        "models_vs_persistence_holm_family_size": 2,
        "resampling_hierarchy": "seed_then_fold_then_paired_cme_session",
        "metadata": metadata,
        "artifacts": artifacts,
        "report": {
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
    }
    manifest_path = destination / "analysis_manifest.json"
    _write_json_atomic(manifest_payload, manifest_path)
    paths["manifest"] = manifest_path
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze RQ4 fold-pooled FiLM-CNN and pure-CNN pair MAEs."
    )
    parser.add_argument("--input", required=True, help="Combined pair_metrics CSV(.gz)")
    parser.add_argument(
        "--output-dir", required=True, help="Analysis artifact directory"
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-design-count-drift",
        action="store_true",
        help="Disable only the frozen formal panel-count checks (for fixtures/debugging).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = analyze(
        args.input,
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        strict_design_counts=not args.allow_design_count_drift,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


__all__ = [
    "CANONICAL_FOLDS",
    "CANONICAL_MODELS",
    "CANONICAL_SEEDS",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EVENT_REGIMES",
    "FORMAL_COMBINED_COUNTS",
    "FORMAL_SPLIT_COUNTS",
    "FORMAL_TEST_COUNTS",
    "RQ4FoldPooledAnalysisError",
    "SOURCE_SPLITS",
    "analyze",
    "analyze_frame",
    "holm_adjust",
    "main",
    "normalise_pair_metrics",
    "seed_fold_session_bootstrap",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
