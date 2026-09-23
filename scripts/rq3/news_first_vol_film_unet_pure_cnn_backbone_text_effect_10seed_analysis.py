"""Paired 10-seed diagnostics for FiLM text effects on a pure-CNN backbone.

The module is deliberately side-effect free.  The experiment orchestrator may
pass either in-memory :class:`pandas.DataFrame` objects or CSV paths and is
responsible for publishing the returned tables.  Four predeclared comparisons
are evaluated:

* standard test: ``matched`` versus ``film_zero_text`` and
  ``film_lp_shuffle``;
* frozen-checkpoint intervention: ``matched_input`` versus ``zero_input`` and
  ``wrong_input``.

Each family uses a seed -> fold -> paired CME-session bootstrap followed by a
Holm-2 correction.  Validation trajectories are descriptive: ``Gain`` is
``parent_mae - target_mae`` (positive is better), and the epoch 1--30 AUC is
the trapezoidal integral of that gain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260907
DEFAULT_EPOCH0_EQUIVALENCE_ATOL = 1e-7
CANONICAL_PARENT_ARM = "pure_cnn_continue_no_text"

STANDARD_COMPARISONS = (
    ("matched_vs_film_zero_text", "matched", "film_zero_text"),
    ("matched_vs_film_lp_shuffle", "matched", "film_lp_shuffle"),
)
INTERVENTION_COMPARISONS = (
    ("matched_input_vs_zero_input", "matched_input", "zero_input"),
    ("matched_input_vs_wrong_input", "matched_input", "wrong_input"),
)
TRAJECTORY_FIXED_EPOCHS = (0, 1, 5, 10, 20, 30)
TRAJECTORY_LABELS = tuple(f"epoch_{epoch}" for epoch in TRAJECTORY_FIXED_EPOCHS) + (
    "best",
)
TRAJECTORY_ARM_ALIASES = {
    "film_lp_matched": "matched",
    "film_zero_text": "film_zero_text",
    "film_lp_shuffle": "film_lp_shuffle",
}
TRAJECTORY_COMPARISONS = (
    ("matched_gain_vs_film_zero_text", "matched", "film_zero_text"),
    ("matched_gain_vs_film_lp_shuffle", "matched", "film_lp_shuffle"),
)


class TextEffectAnalysisError(ValueError):
    """Raised when evidence violates the frozen paired-analysis contract."""


FrameLike = pd.DataFrame | str | Path


def _load_frame(source: FrameLike, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise TextEffectAnalysisError(f"{label} CSV is missing: {path}")
        frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise TextEffectAnalysisError(f"{label} evidence is empty")
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TextEffectAnalysisError(f"{label} is missing columns: {missing}")


def _normalise_panel(
    source: FrameLike,
    *,
    condition_column: str,
    conditions: Sequence[str],
    label: str,
    require_persistence: bool,
) -> pd.DataFrame:
    frame = _load_frame(source, label)
    required = {
        condition_column,
        "seed",
        "fold",
        "pair_id",
        "session_id",
        "target_mae",
    }
    if require_persistence:
        required.add("persistence_mae")
    _require_columns(frame, required, label)
    frame = frame.copy()
    frame[condition_column] = frame[condition_column].astype(str).str.strip()
    frame["fold"] = frame["fold"].astype(str).str.strip()
    frame["pair_id"] = frame["pair_id"].astype(str).str.strip()
    frame["session_id"] = frame["session_id"].astype(str).str.strip()
    if frame[[condition_column, "fold", "pair_id", "session_id"]].eq("").any().any():
        raise TextEffectAnalysisError(f"{label} contains empty lineage values")
    seeds = pd.to_numeric(frame["seed"], errors="coerce")
    if seeds.isna().any() or not np.equal(seeds, np.floor(seeds)).all():
        raise TextEffectAnalysisError(f"{label}.seed must contain integers")
    frame["seed"] = seeds.astype(int)
    numeric = ["target_mae"] + (["persistence_mae"] if require_persistence else [])
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all() or values.le(0.0).any():
            raise TextEffectAnalysisError(
                f"{label}.{column} must be finite and strictly positive"
            )
        frame[column] = values
    selected = frame[frame[condition_column].isin(tuple(map(str, conditions)))].copy()
    observed = set(selected[condition_column])
    expected = set(map(str, conditions))
    if observed != expected:
        raise TextEffectAnalysisError(
            f"{label} condition universe drift: missing={sorted(expected - observed)}"
        )
    keys = [condition_column, "seed", "fold", "pair_id"]
    if selected.duplicated(keys).any():
        raise TextEffectAnalysisError(f"{label} contains duplicate condition/pair rows")
    return selected.sort_values(keys, kind="stable").reset_index(drop=True)


def _paired_values(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    focal: str,
    reference: str,
    require_persistence: bool,
) -> pd.DataFrame:
    keys = ["seed", "fold", "pair_id"]
    audit = ["session_id"] + (["persistence_mae"] if require_persistence else [])
    focal_frame = frame[frame[condition_column].eq(focal)][
        keys + audit + ["target_mae"]
    ].rename(
        columns={
            "session_id": "focal_session_id",
            "target_mae": "focal_mae",
            "persistence_mae": "focal_persistence_mae",
        }
    )
    reference_frame = frame[frame[condition_column].eq(reference)][
        keys + audit + ["target_mae"]
    ].rename(
        columns={
            "session_id": "reference_session_id",
            "target_mae": "reference_mae",
            "persistence_mae": "reference_persistence_mae",
        }
    )
    paired = focal_frame.merge(
        reference_frame, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise TextEffectAnalysisError(
            f"{focal} vs {reference} does not have one-to-one pair lineage"
        )
    paired = paired.drop(columns="_merge")
    if not paired["focal_session_id"].equals(paired["reference_session_id"]):
        raise TextEffectAnalysisError(
            f"{focal} vs {reference} has mismatched CME-session lineage"
        )
    if require_persistence and not np.array_equal(
        paired["focal_persistence_mae"].to_numpy(float),
        paired["reference_persistence_mae"].to_numpy(float),
    ):
        raise TextEffectAnalysisError(
            f"{focal} vs {reference} has mismatched persistence lineage"
        )
    paired["session_id"] = paired.pop("focal_session_id")
    paired = paired.drop(columns="reference_session_id")
    if require_persistence:
        paired["persistence_mae"] = paired.pop("focal_persistence_mae")
        paired = paired.drop(columns="reference_persistence_mae")
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def _log_ratio(focal: np.ndarray, reference: np.ndarray) -> float:
    focal_mean = float(np.mean(focal))
    reference_mean = float(np.mean(reference))
    if (
        not math.isfinite(focal_mean)
        or not math.isfinite(reference_mean)
        or focal_mean <= 0.0
        or reference_mean <= 0.0
    ):
        raise TextEffectAnalysisError("Paired cell MAEs must be finite and positive")
    return float(math.log(focal_mean / reference_mean))


def seed_fold_session_paired_bootstrap(
    paired: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the equal-cell mean log MAE ratio using paired sessions."""

    if int(iterations) < 2:
        raise TextEffectAnalysisError("Bootstrap iterations must be at least two")
    _require_columns(
        paired,
        {"seed", "fold", "pair_id", "session_id", "focal_mae", "reference_mae"},
        "paired bootstrap",
    )
    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    observed_cells = set(
        paired[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )
    expected_cells = {(seed, fold) for seed in seeds for fold in folds}
    if observed_cells != expected_cells:
        raise TextEffectAnalysisError(
            "Bootstrap seed/fold cells drift: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    cells: dict[tuple[int, str], dict[str, Any]] = {}
    points: dict[tuple[int, str], float] = {}
    for seed in seeds:
        for fold in folds:
            cell = paired[paired["seed"].eq(seed) & paired["fold"].eq(fold)].copy()
            if cell.empty or cell["session_id"].astype(str).str.strip().eq("").any():
                raise TextEffectAnalysisError(
                    f"Empty paired cell: seed={seed}, fold={fold}"
                )
            sessions = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
            focal_blocks = {
                session: cell.loc[cell["session_id"].eq(session), "focal_mae"].to_numpy(
                    float
                )
                for session in sessions
            }
            reference_blocks = {
                session: cell.loc[
                    cell["session_id"].eq(session), "reference_mae"
                ].to_numpy(float)
                for session in sessions
            }
            focal = cell["focal_mae"].to_numpy(float)
            reference = cell["reference_mae"].to_numpy(float)
            if not np.isfinite(focal).all() or not np.isfinite(reference).all():
                raise TextEffectAnalysisError("Paired MAEs must be finite")
            cells[(seed, fold)] = {
                "sessions": sessions,
                "focal": focal_blocks,
                "reference": reference_blocks,
            }
            points[(seed, fold)] = _log_ratio(focal, reference)

    point = float(np.mean(list(points.values())))
    seed_points = {
        seed: float(np.mean([points[(seed, fold)] for fold in folds])) for seed in seeds
    }
    fold_points = {
        fold: float(np.mean([points[(seed, fold)] for seed in seeds])) for fold in folds
    }
    rng = np.random.default_rng(int(rng_seed))
    seed_array = np.asarray(seeds, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=object)
    draws = np.empty(int(iterations), dtype=float)
    for draw_index in range(int(iterations)):
        sampled_cell_ratios: list[float] = []
        for sampled_seed in rng.choice(seed_array, size=len(seed_array), replace=True):
            for sampled_fold in rng.choice(
                fold_array, size=len(fold_array), replace=True
            ):
                payload = cells[(int(sampled_seed), str(sampled_fold))]
                sessions = payload["sessions"]
                chosen = rng.choice(sessions, size=len(sessions), replace=True)
                focal = np.concatenate(
                    [payload["focal"][session] for session in chosen]
                )
                reference = np.concatenate(
                    [payload["reference"][session] for session in chosen]
                )
                sampled_cell_ratios.append(_log_ratio(focal, reference))
        draws[draw_index] = float(np.mean(sampled_cell_ratios))

    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    cell_means = paired.groupby(["seed", "fold"], sort=True)[
        ["focal_mae", "reference_mae"]
    ].mean()
    focal_mean_mae = float(cell_means["focal_mae"].mean())
    reference_mean_mae = float(cell_means["reference_mae"].mean())
    return {
        "focal_mean_mae": focal_mean_mae,
        "reference_mean_mae": reference_mean_mae,
        "reference_minus_focal_mae": reference_mean_mae - focal_mean_mae,
        "mean_log_mae_ratio": point,
        "geometric_mae_ratio": float(math.exp(point)),
        "geometric_improvement_percent": 100.0 * (1.0 - float(math.exp(point))),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_value_one_sided": float(p_negative),
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_negative, p_positive))),
        "consistent_seed_count": int(
            sum(value <= 0.0 for value in seed_points.values())
        ),
        "consistent_fold_count": int(
            sum(value <= 0.0 for value in fold_points.values())
        ),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "pair_count": int(paired["pair_id"].nunique()),
        "session_count": int(paired["session_id"].nunique()),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(rng_seed),
        "seed_log_mae_ratios_json": json.dumps(
            seed_points, sort_keys=True, separators=(",", ":")
        ),
        "fold_log_mae_ratios_json": json.dumps(
            fold_points, sort_keys=True, separators=(",", ":")
        ),
        "resampling_method": (
            "seed_then_fold_then_paired_cme_session_cluster_"
            "recompute_cell_log_mae_ratio"
        ),
        "difference_direction": "negative_log_ratio_favours_matched",
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic Holm step-down adjusted p-values."""

    if not p_values:
        return {}
    checked: list[tuple[str, float]] = []
    for name, value in p_values.items():
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise TextEffectAnalysisError("Holm p-values must be finite in [0, 1]")
        checked.append((str(name), number))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw))
        adjusted[name] = running
    return adjusted


def _comparison_family(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    specifications: Sequence[tuple[str, str, str]],
    family: str,
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str],
    iterations: int,
    rng_seed: int,
    alpha: float,
    minimum_nonworse_seeds: int,
    minimum_nonworse_folds: int,
    require_persistence: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (comparison_id, focal, reference) in enumerate(specifications):
        paired = _paired_values(
            frame,
            condition_column=condition_column,
            focal=focal,
            reference=reference,
            require_persistence=require_persistence,
        )
        row = seed_fold_session_paired_bootstrap(
            paired,
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
            iterations=iterations,
            rng_seed=int(rng_seed) + index,
        )
        row.update(
            {
                "comparison_id": comparison_id,
                "comparison_family": family,
                "focal_condition": focal,
                "reference_condition": reference,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 2:
        raise TextEffectAnalysisError(f"{family} must contain exactly two comparisons")
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
    result["holm_family_size"] = 2
    result["alpha"] = float(alpha)
    result["required_consistent_seed_count"] = int(minimum_nonworse_seeds)
    result["required_consistent_fold_count"] = int(minimum_nonworse_folds)
    result["passes_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["holm_adjusted_p"].lt(float(alpha))
        & result["consistent_seed_count"].ge(int(minimum_nonworse_seeds))
        & result["consistent_fold_count"].ge(int(minimum_nonworse_folds))
    )
    return result.sort_values("comparison_id", kind="stable").reset_index(drop=True)


def validation_gain_trajectory(
    source: FrameLike,
    *,
    epoch_start: int = 1,
    epoch_end: int = 30,
    epoch0_equivalence_atol: float = DEFAULT_EPOCH0_EQUIVALENCE_ATOL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sparse trajectory gains, cell metrics, and equal-cell summaries.

    Formal experiment evidence contains the prescribed checkpoints
    ``0/1/5/10/20/30/best`` rather than every integer epoch.  The epoch-1--30
    AUC is therefore the trapezoidal integral through the five fixed observed
    checkpoints.  ``best`` remains a separate diagnostic and is never inserted
    into that curve, because its selected epoch can duplicate a fixed epoch.

    For compatibility with synthetic and older callers, a dense epoch-1--30
    panel without ``checkpoint_label`` is still accepted.  Formal sparse input
    may omit ``parent_mae``: it is derived from the epoch-0 prediction and is
    checked within the graft tolerance across all arms in the same
    seed/fold/pair cell.  Every arm's gain is then referenced to the canonical
    ``pure_cnn_continue_no_text`` epoch-0 MAE, so the gain definition has one
    parent rather than six numerically near-identical baselines.
    """

    frame = _load_frame(source, "validation trajectory")
    required = {
        "arm",
        "seed",
        "fold",
        "pair_id",
        "session_id",
        "epoch",
        "target_mae",
    }
    _require_columns(frame, required, "validation trajectory")
    if int(epoch_start) != 1 or int(epoch_end) != 30:
        raise TextEffectAnalysisError("Validation trajectory is frozen to epochs 1--30")
    if (
        not math.isfinite(float(epoch0_equivalence_atol))
        or float(epoch0_equivalence_atol) < 0.0
    ):
        raise TextEffectAnalysisError("epoch0_equivalence_atol must be finite and >= 0")
    frame = frame.copy()
    for column in ("arm", "fold", "pair_id", "session_id"):
        frame[column] = frame[column].astype(str).str.strip()
    if frame[["arm", "fold", "pair_id", "session_id"]].eq("").any().any():
        raise TextEffectAnalysisError("Validation trajectory has empty lineage")
    for column in ("seed", "epoch"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise TextEffectAnalysisError(
                f"Validation trajectory {column} must be integer"
            )
        frame[column] = values.astype(int)
    for column in ("target_mae",):
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all() or values.lt(0.0).any():
            raise TextEffectAnalysisError(
                f"Validation trajectory {column} must be finite and nonnegative"
            )
        frame[column] = values
    frame["arm"] = frame["arm"].replace(TRAJECTORY_ARM_ALIASES)
    sparse = "checkpoint_label" in frame.columns
    if sparse:
        frame["checkpoint_label"] = frame["checkpoint_label"].astype(str).str.strip()
        if frame["checkpoint_label"].eq("").any():
            raise TextEffectAnalysisError(
                "Validation trajectory contains an empty checkpoint label"
            )
        keys = ["arm", "seed", "fold", "pair_id", "checkpoint_label"]
        if frame.empty or frame.duplicated(keys).any():
            raise TextEffectAnalysisError(
                "Validation sparse trajectory is empty or duplicated"
            )
    else:
        if "parent_mae" not in frame.columns:
            raise TextEffectAnalysisError(
                "Dense validation trajectory requires parent_mae"
            )
        frame = frame[frame["epoch"].between(epoch_start, epoch_end)].copy()
        keys = ["arm", "seed", "fold", "pair_id", "epoch"]
        if frame.empty or frame.duplicated(keys).any():
            raise TextEffectAnalysisError(
                "Validation trajectory is empty or duplicated"
            )
        parent_values = pd.to_numeric(frame["parent_mae"], errors="coerce").astype(
            float
        )
        if not np.isfinite(parent_values.to_numpy()).all() or parent_values.lt(0).any():
            raise TextEffectAnalysisError(
                "Validation trajectory parent_mae must be finite and nonnegative"
            )
        frame["parent_mae"] = parent_values

    if sparse:
        baseline = frame.loc[frame["checkpoint_label"].eq("epoch_0")].copy()
        baseline_keys = ["arm", "seed", "fold", "pair_id"]
        if baseline.empty or baseline.duplicated(baseline_keys).any():
            raise TextEffectAnalysisError(
                "Sparse validation trajectory requires one epoch_0 row per arm/pair"
            )
        baseline = baseline[baseline_keys + ["target_mae"]].rename(
            columns={"target_mae": "derived_parent_mae"}
        )
        frame = frame.merge(
            baseline,
            on=baseline_keys,
            how="left",
            validate="many_to_one",
        )
        if frame["derived_parent_mae"].isna().any():
            raise TextEffectAnalysisError(
                "Sparse validation trajectory lacks its epoch-0 baseline"
            )
        cross_arm = baseline.pivot_table(
            index=["seed", "fold", "pair_id"],
            columns="arm",
            values="derived_parent_mae",
            aggfunc="first",
        )
        if CANONICAL_PARENT_ARM not in cross_arm.columns:
            raise TextEffectAnalysisError(
                "Sparse validation trajectory lacks canonical "
                f"{CANONICAL_PARENT_ARM} epoch-0 baseline"
            )
        canonical_values = cross_arm[CANONICAL_PARENT_ARM].to_numpy(float)
        if cross_arm.isna().any().any() or not np.allclose(
            cross_arm.to_numpy(float),
            canonical_values[:, None],
            rtol=0.0,
            atol=float(epoch0_equivalence_atol),
        ):
            raise TextEffectAnalysisError(
                "Epoch-0 parent MAE is not identical across continuation arms"
            )
        canonical = (
            cross_arm[CANONICAL_PARENT_ARM].rename("canonical_parent_mae").reset_index()
        )
        frame = frame.merge(
            canonical,
            on=["seed", "fold", "pair_id"],
            how="left",
            validate="many_to_one",
        )
        if "parent_mae" in frame.columns:
            supplied = pd.to_numeric(frame["parent_mae"], errors="coerce").astype(float)
            if not np.isfinite(supplied.to_numpy()).all() or not np.allclose(
                supplied.to_numpy(float),
                frame["canonical_parent_mae"].to_numpy(float),
                rtol=0.0,
                atol=float(epoch0_equivalence_atol),
            ):
                raise TextEffectAnalysisError(
                    "Supplied parent_mae disagrees with canonical epoch-0 baseline"
                )
        frame = frame.drop(columns="derived_parent_mae")
        frame["parent_mae"] = frame.pop("canonical_parent_mae")

    pair_rows: list[dict[str, Any]] = []
    pair_keys = ["arm", "seed", "fold", "pair_id"]
    for key, group in frame.groupby(pair_keys, sort=True):
        if sparse:
            observed_labels = set(group["checkpoint_label"])
            expected_labels = set(TRAJECTORY_LABELS)
            if observed_labels != expected_labels:
                raise TextEffectAnalysisError(
                    "Validation sparse trajectory lacks prescribed checkpoints "
                    f"for {key}: missing={sorted(expected_labels - observed_labels)}, "
                    f"extra={sorted(observed_labels - expected_labels)}"
                )
            fixed = group.loc[
                group["checkpoint_label"].isin(
                    {f"epoch_{epoch}" for epoch in TRAJECTORY_FIXED_EPOCHS}
                )
            ].sort_values("epoch", kind="stable")
            if tuple(fixed["epoch"]) != TRAJECTORY_FIXED_EPOCHS:
                raise TextEffectAnalysisError(
                    f"Validation sparse checkpoint epoch drift for {key}"
                )
            curve = fixed.loc[fixed["epoch"].between(epoch_start, epoch_end)]
            best = group.loc[group["checkpoint_label"].eq("best")].iloc[0]
        else:
            ordered = group.sort_values("epoch", kind="stable")
            expected_epochs = set(range(epoch_start, epoch_end + 1))
            if set(ordered["epoch"]) != expected_epochs:
                raise TextEffectAnalysisError(
                    f"Validation trajectory lacks epochs 1--30 for {key}"
                )
            curve = ordered
            best = None
        ordered = group.sort_values(
            ["epoch"] + (["checkpoint_label"] if sparse else []), kind="stable"
        )
        if ordered["session_id"].nunique() != 1:
            raise TextEffectAnalysisError(
                f"Session lineage changes across epochs for {key}"
            )
        if not np.allclose(
            ordered["parent_mae"].to_numpy(float),
            float(ordered["parent_mae"].iloc[0]),
            rtol=0.0,
            atol=0.0,
        ):
            raise TextEffectAnalysisError(f"Parent MAE changes across epochs for {key}")
        gain = curve["parent_mae"].to_numpy(float) - curve["target_mae"].to_numpy(float)
        epochs = curve["epoch"].to_numpy(float)
        auc = float(np.trapz(gain, epochs))
        epoch30_gain = float(
            curve.loc[curve["epoch"].eq(epoch_end), "parent_mae"].iloc[0]
            - curve.loc[curve["epoch"].eq(epoch_end), "target_mae"].iloc[0]
        )
        pair_rows.append(
            {
                "arm": str(key[0]),
                "seed": int(key[1]),
                "fold": str(key[2]),
                "pair_id": str(key[3]),
                "session_id": str(ordered["session_id"].iloc[0]),
                "epoch30_gain": epoch30_gain,
                "epoch1_30_gain_trapezoid_auc": auc,
                "epoch1_30_gain_trapezoid_auc_per_interval": auc
                / float(epoch_end - epoch_start),
                "best_epoch": int(best["epoch"]) if best is not None else pd.NA,
                "best_gain": (
                    float(best["parent_mae"] - best["target_mae"])
                    if best is not None
                    else float("nan")
                ),
            }
        )
    frame["gain"] = frame["parent_mae"] - frame["target_mae"]
    pair_metrics = pd.DataFrame(pair_rows)
    cell_summary = (
        pair_metrics.groupby(["arm", "seed", "fold"], sort=True)
        .agg(
            epoch30_gain_mean=("epoch30_gain", "mean"),
            epoch1_30_gain_trapezoid_auc_mean=(
                "epoch1_30_gain_trapezoid_auc",
                "mean",
            ),
            epoch1_30_gain_trapezoid_auc_per_interval_mean=(
                "epoch1_30_gain_trapezoid_auc_per_interval",
                "mean",
            ),
            pair_count=("pair_id", "nunique"),
            session_count=("session_id", "nunique"),
            best_epoch_mean=("best_epoch", "mean"),
            best_gain_mean=("best_gain", "mean"),
        )
        .reset_index()
    )
    arm_summary = (
        cell_summary.groupby("arm", sort=True)
        .agg(
            epoch30_gain_equal_cell_mean=("epoch30_gain_mean", "mean"),
            epoch1_30_gain_trapezoid_auc_equal_cell_mean=(
                "epoch1_30_gain_trapezoid_auc_mean",
                "mean",
            ),
            epoch1_30_gain_trapezoid_auc_per_interval_equal_cell_mean=(
                "epoch1_30_gain_trapezoid_auc_per_interval_mean",
                "mean",
            ),
            seed_count=("seed", "nunique"),
            fold_count=("fold", "nunique"),
            best_epoch_equal_cell_mean=("best_epoch_mean", "mean"),
            best_gain_equal_cell_mean=("best_gain_mean", "mean"),
        )
        .reset_index()
    )
    return (
        frame.sort_values(keys, kind="stable").reset_index(drop=True),
        cell_summary,
        arm_summary,
    )


def validation_trajectory_comparisons(
    cell_summary: pd.DataFrame,
    *,
    minimum_nonworse_seeds: int = 7,
    minimum_nonworse_folds: int = 3,
    expected_seeds: Sequence[int] | None = None,
    expected_folds: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare matched validation gain with equal-budget zero/shuffle arms."""

    required = {
        "arm",
        "seed",
        "fold",
        "epoch30_gain_mean",
        "epoch1_30_gain_trapezoid_auc_per_interval_mean",
    }
    _require_columns(cell_summary, required, "validation trajectory cells")
    cells = cell_summary.copy()
    cells["arm"] = cells["arm"].astype(str).replace(TRAJECTORY_ARM_ALIASES)
    if expected_seeds is not None and expected_folds is not None:
        observed = set(
            cells[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None)
        )
        expected = {
            (int(seed), str(fold)) for seed in expected_seeds for fold in expected_folds
        }
        if observed != expected:
            raise TextEffectAnalysisError(
                "Validation seed/fold universe drift: "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
    rows: list[dict[str, Any]] = []
    for comparison_id, focal, reference in TRAJECTORY_COMPARISONS:
        focal_cells = cells.loc[cells["arm"].eq(focal)].set_index(["seed", "fold"])
        reference_cells = cells.loc[cells["arm"].eq(reference)].set_index(
            ["seed", "fold"]
        )
        if focal_cells.empty or not focal_cells.index.equals(reference_cells.index):
            raise TextEffectAnalysisError(
                f"Validation cell lineage differs for {focal} vs {reference}"
            )
        epoch30_difference = (
            focal_cells["epoch30_gain_mean"] - reference_cells["epoch30_gain_mean"]
        )
        auc_difference = (
            focal_cells["epoch1_30_gain_trapezoid_auc_per_interval_mean"]
            - reference_cells["epoch1_30_gain_trapezoid_auc_per_interval_mean"]
        )
        seed_auc = auc_difference.groupby(level="seed").mean()
        fold_auc = auc_difference.groupby(level="fold").mean()
        seed_epoch30 = epoch30_difference.groupby(level="seed").mean()
        fold_epoch30 = epoch30_difference.groupby(level="fold").mean()
        seed_count = int(((seed_auc >= 0.0) & (seed_epoch30 >= 0.0)).sum())
        fold_count = int(((fold_auc >= 0.0) & (fold_epoch30 >= 0.0)).sum())
        row = {
            "comparison_id": comparison_id,
            "focal_condition": focal,
            "reference_condition": reference,
            "epoch30_gain_difference_equal_cell_mean": float(epoch30_difference.mean()),
            "epoch1_30_auc_per_interval_difference_equal_cell_mean": float(
                auc_difference.mean()
            ),
            "consistent_seed_count": seed_count,
            "consistent_fold_count": fold_count,
            "required_consistent_seed_count": int(minimum_nonworse_seeds),
            "required_consistent_fold_count": int(minimum_nonworse_folds),
        }
        row["passes_optimization_gate"] = bool(
            row["epoch30_gain_difference_equal_cell_mean"] > 0.0
            and row["epoch1_30_auc_per_interval_difference_equal_cell_mean"] > 0.0
            and seed_count >= int(minimum_nonworse_seeds)
            and fold_count >= int(minimum_nonworse_folds)
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("comparison_id", kind="stable")
        .reset_index(drop=True)
    )


def validation_snapshot_summary(trajectory_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize each observed validation checkpoint with equal cell weighting."""

    required = {
        "arm",
        "seed",
        "fold",
        "pair_id",
        "epoch",
        "target_mae",
        "parent_mae",
        "gain",
    }
    _require_columns(trajectory_rows, required, "validation trajectory rows")
    frame = trajectory_rows.copy()
    if "checkpoint_label" not in frame.columns:
        frame["checkpoint_label"] = frame["epoch"].map(
            lambda epoch: f"epoch_{int(epoch)}"
        )
    frame["checkpoint_label"] = frame["checkpoint_label"].astype(str)
    cell = (
        frame.groupby(["arm", "seed", "fold", "checkpoint_label", "epoch"], sort=True)
        .agg(
            target_mae_cell_mean=("target_mae", "mean"),
            parent_mae_cell_mean=("parent_mae", "mean"),
            gain_cell_mean=("gain", "mean"),
            pair_count=("pair_id", "nunique"),
        )
        .reset_index()
    )
    summary = (
        cell.groupby(["arm", "checkpoint_label", "epoch"], sort=True)
        .agg(
            target_mae_equal_cell_mean=("target_mae_cell_mean", "mean"),
            parent_mae_equal_cell_mean=("parent_mae_cell_mean", "mean"),
            gain_equal_cell_mean=("gain_cell_mean", "mean"),
            seed_count=("seed", "nunique"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    label_order = {label: index for index, label in enumerate(TRAJECTORY_LABELS)}

    def order(label: str) -> int:
        if label in label_order:
            return label_order[label]
        suffix = label.rsplit("_", maxsplit=1)[-1]
        return 1_000 + int(suffix)

    summary["checkpoint_order"] = summary["checkpoint_label"].map(order)
    return summary.sort_values(
        ["arm", "checkpoint_order", "epoch"], kind="stable"
    ).reset_index(drop=True)


def conclusion_status(
    *,
    main_all_supported: bool,
    intervention_all_supported: bool,
    trajectory_all_supported: bool = True,
) -> str:
    """Map all three frozen evidence layers to the declared status vocabulary."""

    if trajectory_all_supported and main_all_supported and intervention_all_supported:
        return "stable_text_mae_increment"
    if main_all_supported:
        return "optimization_only"
    if intervention_all_supported:
        return "not_generalized"
    return "no_verified_text_reliance"


def analyze_backbone_text_effect(
    standard_test: FrameLike,
    intervention: FrameLike,
    validation_trajectory: FrameLike,
    *,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    expected_folds: Sequence[str] = CANONICAL_FOLDS,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
    minimum_nonworse_seeds: int = 7,
    minimum_nonworse_folds: int = 3,
) -> dict[str, Any]:
    """Run the frozen standard, intervention, and trajectory analyses."""

    seeds = tuple(map(int, expected_seeds))
    folds = tuple(map(str, expected_folds))
    if (
        not seeds
        or not folds
        or len(set(seeds)) != len(seeds)
        or len(set(folds)) != len(folds)
    ):
        raise TextEffectAnalysisError("Expected seed/fold universes must be unique")
    if not 0.0 < float(alpha) < 1.0:
        raise TextEffectAnalysisError("alpha must lie in (0,1)")
    if not 1 <= int(minimum_nonworse_seeds) <= len(seeds):
        raise TextEffectAnalysisError("Invalid minimum_nonworse_seeds")
    if not 1 <= int(minimum_nonworse_folds) <= len(folds):
        raise TextEffectAnalysisError("Invalid minimum_nonworse_folds")

    standard_conditions = tuple(
        dict.fromkeys(
            condition
            for _, focal, reference in STANDARD_COMPARISONS
            for condition in (focal, reference)
        )
    )
    standard = _normalise_panel(
        standard_test,
        condition_column="arm",
        conditions=standard_conditions,
        label="standard test",
        require_persistence=True,
    )
    intervention_conditions = tuple(
        dict.fromkeys(
            condition
            for _, focal, reference in INTERVENTION_COMPARISONS
            for condition in (focal, reference)
        )
    )
    interventions = _normalise_panel(
        intervention,
        condition_column="input_condition",
        conditions=intervention_conditions,
        label="intervention",
        require_persistence=False,
    )
    main = _comparison_family(
        standard,
        condition_column="arm",
        specifications=STANDARD_COMPARISONS,
        family="standard_test_holm2",
        expected_seeds=seeds,
        expected_folds=folds,
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed),
        alpha=float(alpha),
        minimum_nonworse_seeds=int(minimum_nonworse_seeds),
        minimum_nonworse_folds=int(minimum_nonworse_folds),
        require_persistence=True,
    )
    intervention_result = _comparison_family(
        interventions,
        condition_column="input_condition",
        specifications=INTERVENTION_COMPARISONS,
        family="frozen_checkpoint_intervention_holm2",
        expected_seeds=seeds,
        expected_folds=folds,
        iterations=int(bootstrap_iterations),
        rng_seed=int(bootstrap_seed) + 10_000,
        alpha=float(alpha),
        minimum_nonworse_seeds=int(minimum_nonworse_seeds),
        minimum_nonworse_folds=int(minimum_nonworse_folds),
        require_persistence=False,
    )
    trajectory_rows, trajectory_cells, trajectory_arms = validation_gain_trajectory(
        validation_trajectory
    )
    trajectory_comparisons = validation_trajectory_comparisons(
        trajectory_cells,
        minimum_nonworse_seeds=int(minimum_nonworse_seeds),
        minimum_nonworse_folds=int(minimum_nonworse_folds),
        expected_seeds=seeds,
        expected_folds=folds,
    )
    trajectory_snapshots = validation_snapshot_summary(trajectory_rows)
    trajectory_all = bool(trajectory_comparisons["passes_optimization_gate"].all())
    main_all = bool(main["passes_support_gate"].all())
    intervention_all = bool(intervention_result["passes_support_gate"].all())
    status = conclusion_status(
        main_all_supported=main_all,
        intervention_all_supported=intervention_all,
        trajectory_all_supported=trajectory_all,
    )
    failed_layers = [
        name
        for name, supported in (
            ("validation_optimization", trajectory_all),
            ("frozen_test_generalization", main_all),
            ("same_checkpoint_text_reliance", intervention_all),
        )
        if not supported
    ]
    conclusion = {
        "schema_version": 1,
        "kind": "film_unet_pure_cnn_backbone_text_effect_conclusion_v1",
        "status": status,
        "all_three_evidence_layers_supported": bool(
            trajectory_all and main_all and intervention_all
        ),
        "validation_trajectory_family_all_supported": trajectory_all,
        "main_standard_family_all_supported": main_all,
        "intervention_family_all_supported": intervention_all,
        "main_supported_comparison_count": int(main["passes_support_gate"].sum()),
        "intervention_supported_comparison_count": int(
            intervention_result["passes_support_gate"].sum()
        ),
        "bootstrap_iterations_per_comparison": int(bootstrap_iterations),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "holm_family_size": 2,
        "minimum_nonworse_seeds": int(minimum_nonworse_seeds),
        "minimum_nonworse_folds": int(minimum_nonworse_folds),
        "failed_evidence_layers": failed_layers,
        "trajectory_role": "predeclared_optimization_evidence_layer",
        "gain_definition": "parent_mae_minus_target_mae_positive_is_better",
        "auc_definition": "epoch_1_to_30_trapezoid_integral_of_pair_level_gain",
        "status_rule": {
            "stable_text_mae_increment": "all_three_evidence_layers_pass",
            "optimization_only": (
                "standard_family_passes_but_at_least_one_other_layer_fails"
            ),
            "not_generalized": ("intervention_family_passes_but_standard_family_fails"),
            "no_verified_text_reliance": (
                "neither_standard_nor_intervention_complete_family_passes"
            ),
        },
    }
    return {
        "main_comparisons": main,
        "intervention_comparisons": intervention_result,
        "validation_trajectory": trajectory_rows,
        "validation_trajectory_cells": trajectory_cells,
        "validation_trajectory_arms": trajectory_arms,
        "validation_trajectory_comparisons": trajectory_comparisons,
        "validation_trajectory_snapshots": trajectory_snapshots,
        "conclusion": conclusion,
    }


def conclusion_json(result: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize the conclusion from :func:`analyze_backbone_text_effect`."""

    conclusion = result.get("conclusion")
    if not isinstance(conclusion, Mapping):
        raise TextEffectAnalysisError("Analysis result lacks a conclusion mapping")
    return json.dumps(
        dict(conclusion),
        ensure_ascii=False,
        sort_keys=True,
        indent=int(indent),
        allow_nan=False,
    )


__all__ = [
    "CANONICAL_FOLDS",
    "CANONICAL_PARENT_ARM",
    "CANONICAL_SEEDS",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_EPOCH0_EQUIVALENCE_ATOL",
    "INTERVENTION_COMPARISONS",
    "STANDARD_COMPARISONS",
    "TRAJECTORY_COMPARISONS",
    "TRAJECTORY_FIXED_EPOCHS",
    "TRAJECTORY_LABELS",
    "TextEffectAnalysisError",
    "analyze_backbone_text_effect",
    "conclusion_json",
    "conclusion_status",
    "holm_adjust",
    "seed_fold_session_paired_bootstrap",
    "validation_gain_trajectory",
    "validation_snapshot_summary",
    "validation_trajectory_comparisons",
]
