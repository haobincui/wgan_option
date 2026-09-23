"""Build the Q4-primary and rolling-robustness evidence for Chapter 3.

This is an analysis-only command.  It reads the terminal five-arm direct-training
composite, verifies its frozen lineage, and writes deterministic summaries and
paired bootstrap results without training or predicting any model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import holm_adjust


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / (
    "outputs/experiments/"
    "rq12_news_first_vol_text_10seed_composite_lr2p5e5_"
    "descriptive_no_bootstrap_exact_ttm_rolling_v1"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "outputs/reports/chapter3_q4_primary_rolling_robustness_v2"
)
DEFAULT_ROLLING_BOOTSTRAP_ROOT = REPO_ROOT / (
    "outputs/experiments/"
    "rq12_news_first_vol_text_10seed_modelwise_bootstrap10000_lr2p5e5_"
    "exact_ttm_rolling_v1"
)
SOURCE_PAIR_METRICS = Path("analysis/text_10seed_composite_pair_metrics.csv")
SOURCE_TRAINING_SUMMARY = Path("analysis/text_10seed_composite_training_summary.csv")
SOURCE_ANALYSIS_MANIFEST = Path("analysis/text_10seed_composite_analysis_manifest.json")
SOURCE_QA = Path("text_10seed_composite_qa.json")
SOURCE_OUTPUT_HASHES = Path("text_10seed_composite_output_hashes.csv")
ROLLING_SOURCE_SUMMARY = Path("analysis/model_vs_persistence_bootstrap_10000.csv")
ROLLING_SOURCE_DRAWS = Path("analysis/model_vs_persistence_bootstrap_draws.csv.gz")
ROLLING_SOURCE_ANALYSIS_MANIFEST = Path("bootstrap_analysis_manifest.json")
ROLLING_SOURCE_QA = Path("bootstrap_qa.json")
ROLLING_SOURCE_OUTPUT_HASHES = Path("bootstrap_output_hashes.csv")

ARMS = ("lp_matched", "no_text", "lp_shuffle", "bow", "sentiment")
SEEDS = (
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
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
Q4_FOLD = "f4_2023q4"
EXPECTED_PAIRS = {
    "f1_2023q1": 110,
    "f2_2023q2": 112,
    "f3_2023q3": 135,
    "f4_2023q4": 143,
}
EXPECTED_SESSIONS = {
    "f1_2023q1": 34,
    "f2_2023q2": 36,
    "f3_2023q3": 33,
    "f4_2023q4": 45,
}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260908
MINIMUM_NONWORSE_SEEDS = 7

Q4_ARM_SUMMARY = Path("analysis/q4_arm_summary.csv")
Q4_TRAINING_SUMMARY = Path("analysis/q4_training_summary.csv")
ROLLING_SUMMARY = Path("analysis/rolling_fold_summary.csv")
Q4_CONTRASTS = Path("analysis/q4_primary_contrasts.csv")
Q4_DRAWS = Path("analysis/q4_bootstrap_draws.csv.gz")
FOLD_PERSISTENCE_CONTRASTS = Path("analysis/fold_model_vs_persistence_contrasts.csv")
FOLD_PERSISTENCE_DRAWS = Path(
    "analysis/fold_model_vs_persistence_bootstrap_draws.csv.gz"
)
ROLLING_PERSISTENCE_CONTRASTS = Path(
    "analysis/rolling_model_vs_persistence_contrasts.csv"
)
ROLLING_PERSISTENCE_DRAWS = Path(
    "analysis/rolling_model_vs_persistence_bootstrap_draws.csv.gz"
)
INPUT_MANIFEST = Path("input_manifest.json")
ANALYSIS_MANIFEST = Path("analysis_manifest.json")
QA_PATH = Path("qa.json")
OUTPUT_HASHES = Path("output_hashes.csv")
REPORT_PATH = Path("report.md")


class Chapter3Q4AnalysisError(RuntimeError):
    """Raised when frozen evidence or a derived artifact violates its contract."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise Chapter3Q4AnalysisError(f"Existing artifact drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    content = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _atomic_bytes(path, content)


def _write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> Path:
    content = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    if compressed:
        content = gzip.compress(content, compresslevel=6, mtime=0)
    return _atomic_bytes(path, content)


def _source_file_rows(source_root: Path) -> list[dict[str, Any]]:
    paths = (
        SOURCE_PAIR_METRICS,
        SOURCE_TRAINING_SUMMARY,
        SOURCE_ANALYSIS_MANIFEST,
        SOURCE_QA,
        SOURCE_OUTPUT_HASHES,
    )
    rows = []
    for relative in paths:
        path = source_root / relative
        if not path.is_file():
            raise Chapter3Q4AnalysisError(f"Required source is missing: {path}")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _rolling_source_file_rows(source_root: Path) -> list[dict[str, Any]]:
    paths = (
        ROLLING_SOURCE_SUMMARY,
        ROLLING_SOURCE_DRAWS,
        ROLLING_SOURCE_ANALYSIS_MANIFEST,
        ROLLING_SOURCE_QA,
        ROLLING_SOURCE_OUTPUT_HASHES,
    )
    rows = []
    for relative in paths:
        path = source_root / relative
        if not path.is_file():
            raise Chapter3Q4AnalysisError(f"Required rolling source is missing: {path}")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_source_hash_manifest(source_root: Path) -> None:
    manifest = pd.read_csv(source_root / SOURCE_OUTPUT_HASHES)
    required = {"relative_path", "size_bytes", "sha256"}
    if required - set(manifest):
        raise Chapter3Q4AnalysisError("Source output hash manifest schema drift")
    indexed = manifest.set_index("relative_path")
    for relative in (
        SOURCE_PAIR_METRICS,
        SOURCE_TRAINING_SUMMARY,
        SOURCE_ANALYSIS_MANIFEST,
        SOURCE_QA,
    ):
        key = relative.as_posix()
        if key not in indexed.index:
            raise Chapter3Q4AnalysisError(f"Source hash row is missing: {key}")
        row = indexed.loc[key]
        path = source_root / relative
        if int(path.stat().st_size) != int(row["size_bytes"]) or sha256_file(
            path
        ) != str(row["sha256"]):
            raise Chapter3Q4AnalysisError(f"Frozen source hash drift: {path}")


def load_and_validate_source(
    source_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = resolve_path(source_root)
    _verify_source_hash_manifest(root)
    qa = json.loads((root / SOURCE_QA).read_text(encoding="utf-8"))
    analysis = json.loads((root / SOURCE_ANALYSIS_MANIFEST).read_text(encoding="utf-8"))
    if (
        qa.get("status") != "passed"
        or int(qa.get("direct_training_jobs", -1)) != 200
        or int(qa.get("parent_jobs", -1)) != 0
        or int(qa.get("continuation_jobs", -1)) != 0
        or int(qa.get("pair_metric_rows", -1)) != 25_000
        or analysis.get("inference_status") != "deferred_by_config"
    ):
        raise Chapter3Q4AnalysisError("Source terminal direct-training QA drift")

    pairs = pd.read_csv(root / SOURCE_PAIR_METRICS, low_memory=False)
    training = pd.read_csv(root / SOURCE_TRAINING_SUMMARY, low_memory=False)
    required_pair_columns = {
        "arm",
        "seed",
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "target_mae",
        "persistence_mae",
        "tolerance_minutes",
        "noise_bank_profile_sha256",
    }
    required_training_columns = {
        "arm",
        "seed",
        "fold",
        "best_epoch",
        "epochs_ran",
        "early_stopped",
    }
    if required_pair_columns - set(pairs):
        raise Chapter3Q4AnalysisError("Pair evidence schema drift")
    if required_training_columns - set(training):
        raise Chapter3Q4AnalysisError("Training summary schema drift")

    pairs = pairs.copy()
    training = training.copy()
    pairs["seed"] = pd.to_numeric(pairs["seed"], errors="raise").astype(int)
    training["seed"] = pd.to_numeric(training["seed"], errors="raise").astype(int)
    for frame in (pairs, training):
        frame["arm"] = frame["arm"].astype(str)
        frame["fold"] = frame["fold"].astype(str)
    for column in ("target_mae", "persistence_mae"):
        pairs[column] = pd.to_numeric(pairs[column], errors="raise").astype(float)
    if (
        len(pairs) != 25_000
        or len(training) != 200
        or set(pairs["arm"]) != set(ARMS)
        or set(training["arm"]) != set(ARMS)
        or set(pairs["seed"]) != set(SEEDS)
        or set(training["seed"]) != set(SEEDS)
        or set(pairs["fold"]) != set(FOLDS)
        or set(training["fold"]) != set(FOLDS)
        or pairs.duplicated(["arm", "seed", "fold", "pair_id"]).any()
        or training.duplicated(["arm", "seed", "fold"]).any()
        or pairs[list(required_pair_columns)].isna().any().any()
        or not np.isfinite(pairs[["target_mae", "persistence_mae"]]).all().all()
        or (pairs[["target_mae", "persistence_mae"]] <= 0.0).any().any()
        or set(pd.to_numeric(pairs["tolerance_minutes"], errors="raise")) != {5}
    ):
        raise Chapter3Q4AnalysisError("Source row/value universe drift")

    lineage_columns = [
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
    ]
    reference: pd.DataFrame | None = None
    for arm in ARMS:
        for seed in SEEDS:
            panel = (
                pairs[pairs["arm"].eq(arm) & pairs["seed"].eq(seed)][lineage_columns]
                .sort_values(["fold", "pair_id"], kind="stable")
                .reset_index(drop=True)
            )
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise Chapter3Q4AnalysisError(
                    "Pair/session/persistence lineage differs across arms or seeds"
                )
            for fold in FOLDS:
                cell = panel[panel["fold"].eq(fold)]
                if (
                    len(cell) != EXPECTED_PAIRS[fold]
                    or cell["session_id"].nunique() != EXPECTED_SESSIONS[fold]
                ):
                    raise Chapter3Q4AnalysisError(
                        f"Pair/session count drift: {arm}/{seed}/{fold}"
                    )

    q4 = pairs[pairs["fold"].eq(Q4_FOLD)]
    for seed, cell in q4.groupby("seed", sort=True):
        if cell["noise_bank_profile_sha256"].nunique() != 1:
            raise Chapter3Q4AnalysisError(
                f"Q4 noise bank differs across arms for seed {seed}"
            )
    return pairs, training


def summarize_arms(pairs: pd.DataFrame) -> pd.DataFrame:
    q4 = pairs[pairs["fold"].eq(Q4_FOLD)].copy()
    cells = q4.groupby(["arm", "seed"], sort=True).agg(
        mean_mae=("target_mae", "mean"),
        persistence_mae=("persistence_mae", "mean"),
    )
    rows = []
    for arm, group in cells.reset_index().groupby("arm", sort=True):
        model = float(group["mean_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        rows.append(
            {
                "arm": str(arm),
                "q4_ten_seed_mean_mae": model,
                "q4_seed_mae_sd": float(group["mean_mae"].std(ddof=1)),
                "q4_persistence_mae": persistence,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "seed_count": int(len(group)),
                "pair_count_per_seed": EXPECTED_PAIRS[Q4_FOLD],
                "session_count_per_seed": EXPECTED_SESSIONS[Q4_FOLD],
            }
        )
    result = pd.DataFrame(rows)
    result["q4_mae_rank"] = (
        result["q4_ten_seed_mean_mae"].rank(method="min").astype(int)
    )
    return result.sort_values(["q4_mae_rank", "arm"], kind="stable").reset_index(
        drop=True
    )


def summarize_training(training: pd.DataFrame) -> pd.DataFrame:
    q4 = training[training["fold"].eq(Q4_FOLD)].copy()
    q4["best_epoch"] = pd.to_numeric(q4["best_epoch"], errors="raise").astype(int)
    q4["epochs_ran"] = pd.to_numeric(q4["epochs_ran"], errors="raise").astype(int)
    q4["early_stopped"] = q4["early_stopped"].astype(bool)
    rows = []
    for arm, group in q4.groupby("arm", sort=True):
        rows.append(
            {
                "arm": str(arm),
                "fits": int(len(group)),
                "mean_best_epoch": float(group["best_epoch"].mean()),
                "median_best_epoch": float(group["best_epoch"].median()),
                "minimum_best_epoch": int(group["best_epoch"].min()),
                "maximum_best_epoch": int(group["best_epoch"].max()),
                "early_stopped_fits": int(group["early_stopped"].sum()),
                "ran_to_epoch_240": int(group["epochs_ran"].eq(240).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("arm", kind="stable").reset_index(drop=True)


def summarize_rolling_folds(pairs: pd.DataFrame) -> pd.DataFrame:
    cells = pairs.groupby(["arm", "fold", "seed"], sort=True).agg(
        mean_mae=("target_mae", "mean"),
        persistence_mae=("persistence_mae", "mean"),
        pair_count=("pair_id", "nunique"),
        session_count=("session_id", "nunique"),
    )
    rows = []
    for (arm, fold), group in cells.reset_index().groupby(["arm", "fold"], sort=True):
        model = float(group["mean_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        rows.append(
            {
                "arm": str(arm),
                "fold": str(fold),
                "ten_seed_mean_mae": model,
                "seed_mae_sd": float(group["mean_mae"].std(ddof=1)),
                "persistence_mae": persistence,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "seed_count": int(len(group)),
                "pair_count_per_seed": int(group["pair_count"].iloc[0]),
                "session_count_per_seed": int(group["session_count"].iloc[0]),
            }
        )
    result = pd.DataFrame(rows)
    result["mae_rank_within_fold"] = (
        result.groupby("fold")["ten_seed_mean_mae"].rank(method="min").astype(int)
    )
    return result.sort_values(["fold", "mae_rank_within_fold", "arm"], kind="stable")


def _comparison_panel(frame: pd.DataFrame, focal: str, reference: str) -> pd.DataFrame:
    keys = ["seed", "pair_id"]
    audit = ["session_id"]
    focal_frame = frame[frame["arm"].eq(focal)][keys + audit + ["target_mae"]].rename(
        columns={"target_mae": "focal_mae", "session_id": "focal_session_id"}
    )
    if reference == "persistence":
        reference_frame = frame[frame["arm"].eq(focal)][
            keys + audit + ["persistence_mae"]
        ].rename(
            columns={
                "persistence_mae": "reference_mae",
                "session_id": "reference_session_id",
            }
        )
    else:
        reference_frame = frame[frame["arm"].eq(reference)][
            keys + audit + ["target_mae"]
        ].rename(
            columns={
                "target_mae": "reference_mae",
                "session_id": "reference_session_id",
            }
        )
    paired = focal_frame.merge(
        reference_frame, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all() or not paired["focal_session_id"].equals(
        paired["reference_session_id"]
    ):
        raise Chapter3Q4AnalysisError(
            f"Comparison is not pair/session aligned: {focal} vs {reference}"
        )
    paired["session_id"] = paired.pop("focal_session_id")
    return paired.drop(columns=["reference_session_id", "_merge"]).sort_values(
        keys, kind="stable"
    )


def _log_ratio(focal: np.ndarray, reference: np.ndarray) -> float:
    return float(math.log(float(np.mean(focal)) / float(np.mean(reference))))


def _bootstrap_t_statistic(point: float, standard_error: float) -> float:
    if not math.isfinite(point) or not math.isfinite(standard_error):
        raise Chapter3Q4AnalysisError("Bootstrap statistic must be finite")
    if standard_error <= 0.0:
        raise Chapter3Q4AnalysisError("Bootstrap standard error must be positive")
    return float(point / standard_error)


def significance_stars(adjusted_p: float) -> str:
    value = float(adjusted_p)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise Chapter3Q4AnalysisError("Holm-adjusted p-value must be finite in [0, 1]")
    if value < 0.01:
        return "***"
    if value < 0.05:
        return "**"
    if value < 0.10:
        return "*"
    return ""


def bootstrap_fold_comparison(
    paired: pd.DataFrame,
    *,
    fold: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    rng_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], np.ndarray]:
    fold = str(fold)
    if fold not in FOLDS:
        raise Chapter3Q4AnalysisError(f"Unknown rolling fold: {fold}")
    if int(iterations) < 2:
        raise Chapter3Q4AnalysisError("Bootstrap iterations must be at least two")
    if set(paired["seed"]) != set(SEEDS):
        raise Chapter3Q4AnalysisError("Fold bootstrap seed universe drift")
    cells: dict[int, dict[str, Any]] = {}
    points: dict[int, float] = {}
    for seed in SEEDS:
        cell = paired[paired["seed"].eq(seed)]
        sessions = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
        if (
            len(cell) != EXPECTED_PAIRS[fold]
            or len(sessions) != EXPECTED_SESSIONS[fold]
        ):
            raise Chapter3Q4AnalysisError(
                f"Fold bootstrap cell drift: fold={fold}, seed={seed}"
            )
        cells[seed] = {
            "sessions": sessions,
            "focal": {
                value: cell.loc[cell["session_id"].eq(value), "focal_mae"].to_numpy(
                    float
                )
                for value in sessions
            },
            "reference": {
                value: cell.loc[cell["session_id"].eq(value), "reference_mae"].to_numpy(
                    float
                )
                for value in sessions
            },
        }
        points[seed] = _log_ratio(
            cell["focal_mae"].to_numpy(float),
            cell["reference_mae"].to_numpy(float),
        )

    rng = np.random.default_rng(int(rng_seed))
    seed_array = np.asarray(SEEDS, dtype=np.int64)
    draws = np.empty(int(iterations), dtype=float)
    for draw_index in range(int(iterations)):
        ratios = []
        for sampled_seed in rng.choice(seed_array, size=len(seed_array), replace=True):
            payload = cells[int(sampled_seed)]
            sessions = payload["sessions"]
            chosen = rng.choice(sessions, size=len(sessions), replace=True)
            focal_values = np.concatenate([payload["focal"][value] for value in chosen])
            reference_values = np.concatenate(
                [payload["reference"][value] for value in chosen]
            )
            ratios.append(_log_ratio(focal_values, reference_values))
        draws[draw_index] = float(np.mean(ratios))

    cell_means = paired.groupby("seed", sort=True)[
        ["focal_mae", "reference_mae"]
    ].mean()
    point = float(np.mean(list(points.values())))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_negative = (1.0 + float(np.count_nonzero(draws >= 0.0))) / (len(draws) + 1.0)
    p_positive = (1.0 + float(np.count_nonzero(draws <= 0.0))) / (len(draws) + 1.0)
    focal_mean = float(cell_means["focal_mae"].mean())
    reference_mean = float(cell_means["reference_mae"].mean())
    bootstrap_se = float(draws.std(ddof=1))
    return (
        {
            "fold": fold,
            "focal_mean_mae": focal_mean,
            "reference_mean_mae": reference_mean,
            "reference_minus_focal_mae": reference_mean - focal_mean,
            "mean_log_mae_ratio": point,
            "geometric_improvement_percent": 100.0 * (1.0 - math.exp(point)),
            "bootstrap_se": bootstrap_se,
            "bootstrap_t": _bootstrap_t_statistic(point, bootstrap_se),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "p_value_one_sided": float(p_negative),
            "p_value_two_sided": float(min(1.0, 2.0 * min(p_negative, p_positive))),
            "consistent_seed_count": int(sum(value <= 0 for value in points.values())),
            "seed_count": len(SEEDS),
            "pair_count_per_seed": EXPECTED_PAIRS[fold],
            "session_count_per_seed": EXPECTED_SESSIONS[fold],
            "bootstrap_iterations": int(iterations),
            "bootstrap_seed": int(rng_seed),
            "resampling_method": "seed_then_paired_CME_session_cluster",
            "seed_log_mae_ratios_json": json.dumps(
                points, sort_keys=True, separators=(",", ":")
            ),
        },
        draws,
    )


def bootstrap_q4_comparison(
    paired: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    rng_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], np.ndarray]:
    return bootstrap_fold_comparison(
        paired,
        fold=Q4_FOLD,
        iterations=iterations,
        rng_seed=rng_seed,
    )


def _apply_holm_by_family(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        raise Chapter3Q4AnalysisError("Cannot adjust an empty contrast table")
    result = result.copy()
    result["holm_adjusted_p"] = np.nan
    result["holm_family_size"] = 0
    for family, group in result.groupby("family", sort=False):
        adjusted = holm_adjust(
            dict(
                zip(
                    group["comparison_id"].astype(str),
                    group["p_value_one_sided"].astype(float),
                    strict=True,
                )
            )
        )
        result.loc[group.index, "holm_adjusted_p"] = group["comparison_id"].map(
            adjusted
        )
        result.loc[group.index, "holm_family_size"] = len(group)
    result["holm_family_size"] = result["holm_family_size"].astype(int)
    result["significance_stars"] = result["holm_adjusted_p"].map(significance_stars)
    return result


def q4_contrasts(
    pairs: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    rng_seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = (
        ("rq1_primary", "lp_matched_vs_no_text", "lp_matched", "no_text"),
        (
            "rq1_secondary_persistence",
            "lp_matched_vs_persistence",
            "lp_matched",
            "persistence",
        ),
        (
            "rq1_secondary_persistence",
            "no_text_vs_persistence",
            "no_text",
            "persistence",
        ),
        ("rq2_primary", "lp_matched_vs_bow", "lp_matched", "bow"),
        (
            "rq2_primary",
            "lp_matched_vs_sentiment",
            "lp_matched",
            "sentiment",
        ),
    )
    q4 = pairs[pairs["fold"].eq(Q4_FOLD)].copy()
    rows: list[dict[str, Any]] = []
    draw_rows: list[pd.DataFrame] = []
    for family, comparison, focal, reference in specifications:
        summary, draws = bootstrap_q4_comparison(
            _comparison_panel(q4, focal, reference),
            iterations=iterations,
            rng_seed=rng_seed,
        )
        summary.update(
            {
                "family": family,
                "comparison_id": comparison,
                "focal_arm": focal,
                "reference_arm": reference,
            }
        )
        rows.append(summary)
        draw_rows.append(
            pd.DataFrame(
                {
                    "family": family,
                    "comparison_id": comparison,
                    "draw_index": np.arange(len(draws), dtype=np.int64),
                    "mean_log_mae_ratio": draws,
                }
            )
        )
    result = _apply_holm_by_family(pd.DataFrame(rows))
    result["passes_q4_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["holm_adjusted_p"].lt(0.05)
        & result["consistent_seed_count"].ge(MINIMUM_NONWORSE_SEEDS)
    )
    return (
        result.sort_values(["family", "comparison_id"], kind="stable").reset_index(
            drop=True
        ),
        pd.concat(draw_rows, ignore_index=True),
    )


def fold_persistence_contrasts(
    pairs: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    rng_seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    draw_rows: list[pd.DataFrame] = []
    for fold in FOLDS:
        fold_pairs = pairs[pairs["fold"].eq(fold)].copy()
        family = f"{fold}_model_vs_persistence"
        for arm in ARMS:
            comparison = f"{arm}_vs_persistence"
            summary, draws = bootstrap_fold_comparison(
                _comparison_panel(fold_pairs, arm, "persistence"),
                fold=fold,
                iterations=iterations,
                rng_seed=rng_seed,
            )
            summary.update(
                {
                    "family": family,
                    "comparison_id": comparison,
                    "focal_arm": arm,
                    "reference_arm": "persistence",
                }
            )
            rows.append(summary)
            draw_rows.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "fold": fold,
                        "comparison_id": comparison,
                        "focal_arm": arm,
                        "draw_index": np.arange(len(draws), dtype=np.int64),
                        "mean_log_mae_ratio": draws,
                    }
                )
            )
    result = _apply_holm_by_family(pd.DataFrame(rows))
    result["passes_fold_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["holm_adjusted_p"].lt(0.05)
        & result["consistent_seed_count"].ge(MINIMUM_NONWORSE_SEEDS)
    )
    return (
        result.sort_values(["fold", "focal_arm"], kind="stable").reset_index(drop=True),
        pd.concat(draw_rows, ignore_index=True),
    )


def _verify_rolling_source_hashes(root: Path) -> None:
    manifest_path = root / ROLLING_SOURCE_OUTPUT_HASHES
    if not manifest_path.is_file():
        raise Chapter3Q4AnalysisError(
            f"Rolling bootstrap hash manifest is missing: {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    required_columns = {"relative_path", "size_bytes", "sha256"}
    if required_columns - set(manifest):
        raise Chapter3Q4AnalysisError("Rolling bootstrap hash schema drift")
    indexed = manifest.set_index("relative_path")
    for relative in (
        ROLLING_SOURCE_SUMMARY,
        ROLLING_SOURCE_DRAWS,
        ROLLING_SOURCE_ANALYSIS_MANIFEST,
        ROLLING_SOURCE_QA,
    ):
        key = relative.as_posix()
        if key not in indexed.index:
            raise Chapter3Q4AnalysisError(
                f"Rolling bootstrap hash row is missing: {key}"
            )
        row = indexed.loc[key]
        path = root / relative
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise Chapter3Q4AnalysisError(
                f"Frozen rolling bootstrap source drift: {path}"
            )


def load_and_validate_rolling_bootstrap(
    rolling_bootstrap_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = resolve_path(rolling_bootstrap_root)
    _verify_rolling_source_hashes(root)
    qa = json.loads((root / ROLLING_SOURCE_QA).read_text(encoding="utf-8"))
    analysis = json.loads(
        (root / ROLLING_SOURCE_ANALYSIS_MANIFEST).read_text(encoding="utf-8")
    )
    if (
        qa.get("status") != "passed"
        or int(qa.get("arm_count", -1)) != len(ARMS)
        or int(qa.get("holm_family_size", -1)) != len(ARMS)
        or int(qa.get("iterations_per_arm", -1)) != BOOTSTRAP_ITERATIONS
        or int(qa.get("total_draws", -1)) != len(ARMS) * BOOTSTRAP_ITERATIONS
        or analysis.get("interpretation") != "retrospective_rolling_development"
    ):
        raise Chapter3Q4AnalysisError("Rolling bootstrap terminal QA drift")

    summary = pd.read_csv(root / ROLLING_SOURCE_SUMMARY)
    draws = pd.read_csv(root / ROLLING_SOURCE_DRAWS)
    required_summary = {
        "arm",
        "mean_log_mae_ratio",
        "geometric_cell_ratio_improvement_percent",
        "geometric_improvement_ci_95_lower_percent",
        "geometric_improvement_ci_95_upper_percent",
        "log_ratio_bootstrap_se",
        "p_value_one_sided_sign_tail",
        "holm5_adjusted_p",
        "holm_family_size",
        "consistent_seed_count",
        "consistent_fold_count",
        "bootstrap_iterations",
    }
    required_draws = {"arm", "draw_index", "mean_log_mae_ratio"}
    if required_summary - set(summary) or required_draws - set(draws):
        raise Chapter3Q4AnalysisError("Rolling bootstrap source schema drift")
    if (
        len(summary) != len(ARMS)
        or set(summary["arm"].astype(str)) != set(ARMS)
        or set(draws["arm"].astype(str)) != set(ARMS)
        or len(draws) != len(ARMS) * BOOTSTRAP_ITERATIONS
        or not summary["holm_family_size"].astype(int).eq(len(ARMS)).all()
        or not summary["bootstrap_iterations"]
        .astype(int)
        .eq(BOOTSTRAP_ITERATIONS)
        .all()
    ):
        raise Chapter3Q4AnalysisError("Rolling bootstrap row universe drift")

    expected_holm = holm_adjust(
        dict(
            zip(
                summary["arm"].astype(str),
                summary["p_value_one_sided_sign_tail"].astype(float),
                strict=True,
            )
        )
    )
    summary = summary.copy()
    for row in summary.itertuples(index=False):
        arm = str(row.arm)
        arm_draws = draws[draws["arm"].astype(str).eq(arm)].sort_values(
            "draw_index", kind="stable"
        )
        if (
            len(arm_draws) != BOOTSTRAP_ITERATIONS
            or arm_draws["draw_index"].duplicated().any()
            or not np.array_equal(
                arm_draws["draw_index"].to_numpy(int),
                np.arange(BOOTSTRAP_ITERATIONS, dtype=int),
            )
        ):
            raise Chapter3Q4AnalysisError(
                f"Rolling bootstrap draw universe drift: {arm}"
            )
        observed_se = float(arm_draws["mean_log_mae_ratio"].to_numpy(float).std(ddof=1))
        if not math.isclose(
            observed_se,
            float(row.log_ratio_bootstrap_se),
            rel_tol=1e-10,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(row.holm5_adjusted_p),
            float(expected_holm[arm]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Chapter3Q4AnalysisError(f"Rolling bootstrap calculation drift: {arm}")

    summary["family"] = "rolling_model_vs_persistence"
    summary["comparison_id"] = summary["arm"].astype(str) + "_vs_persistence"
    summary["focal_arm"] = summary["arm"].astype(str)
    summary["reference_arm"] = "persistence"
    summary["bootstrap_se"] = summary["log_ratio_bootstrap_se"].astype(float)
    summary["bootstrap_t"] = [
        _bootstrap_t_statistic(point, standard_error)
        for point, standard_error in zip(
            summary["mean_log_mae_ratio"].astype(float),
            summary["bootstrap_se"].astype(float),
            strict=True,
        )
    ]
    summary["holm_adjusted_p"] = summary["holm5_adjusted_p"].astype(float)
    summary["significance_stars"] = summary["holm_adjusted_p"].map(significance_stars)
    summary["geometric_improvement_percent"] = summary[
        "geometric_cell_ratio_improvement_percent"
    ].astype(float)
    summary["ci_95_lower_percent"] = summary[
        "geometric_improvement_ci_95_lower_percent"
    ].astype(float)
    summary["ci_95_upper_percent"] = summary[
        "geometric_improvement_ci_95_upper_percent"
    ].astype(float)
    return (
        summary.sort_values("arm", kind="stable").reset_index(drop=True),
        draws.sort_values(["arm", "draw_index"], kind="stable").reset_index(drop=True),
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.10g}"
        return str(value)

    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame[list(columns)].itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *rows,
        ]
    )


def _report(
    arms: pd.DataFrame,
    training: pd.DataFrame,
    rolling: pd.DataFrame,
    q4_contrast_frame: pd.DataFrame,
    fold_persistence: pd.DataFrame,
    rolling_persistence: pd.DataFrame,
) -> str:
    return f"""# Chapter 3 Q4-primary evidence

This analysis uses only the frozen `f4_2023q4` predictions for the primary
RQ1/RQ2 results.  The four-fold table is retained solely as a rolling-period
robustness diagnostic.  All evidence is retrospective development evidence.

## Q4 arm summary

{_markdown_table(arms, ("q4_mae_rank", "arm", "q4_ten_seed_mean_mae", "q4_seed_mae_sd", "improvement_vs_persistence_percent"))}

## Q4 training diagnostics

{_markdown_table(training, ("arm", "fits", "mean_best_epoch", "median_best_epoch", "early_stopped_fits", "ran_to_epoch_240"))}

## Q4 paired bootstrap

{_markdown_table(q4_contrast_frame, ("family", "comparison_id", "mean_log_mae_ratio", "bootstrap_t", "ci_95_lower", "ci_95_upper", "holm_adjusted_p", "significance_stars", "consistent_seed_count", "passes_q4_support_gate"))}

## Fold-specific model versus persistence inference

{_markdown_table(fold_persistence, ("fold", "focal_arm", "mean_log_mae_ratio", "bootstrap_t", "ci_95_lower", "ci_95_upper", "holm_adjusted_p", "significance_stars", "consistent_seed_count", "passes_fold_support_gate"))}

## Four-fold model versus persistence inference

{_markdown_table(rolling_persistence, ("focal_arm", "mean_log_mae_ratio", "bootstrap_t", "ci_95_lower_percent", "ci_95_upper_percent", "holm_adjusted_p", "significance_stars", "consistent_seed_count", "consistent_fold_count"))}

## Rolling-fold robustness

{_markdown_table(rolling, ("fold", "mae_rank_within_fold", "arm", "ten_seed_mean_mae", "improvement_vs_persistence_percent"))}
"""


def _artifact_row(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def run_analysis(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    rolling_bootstrap_root: str | Path = DEFAULT_ROLLING_BOOTSTRAP_ROOT,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> Path:
    if int(iterations) != BOOTSTRAP_ITERATIONS:
        raise Chapter3Q4AnalysisError(
            f"Chapter table inference requires exactly {BOOTSTRAP_ITERATIONS} draws"
        )
    source = resolve_path(source_root)
    rolling_source = resolve_path(rolling_bootstrap_root)
    output = resolve_path(output_root)
    pairs, training = load_and_validate_source(source)
    arm_summary = summarize_arms(pairs)
    training_summary = summarize_training(training)
    rolling_summary = summarize_rolling_folds(pairs)
    contrasts, draws = q4_contrasts(pairs, iterations=iterations)
    fold_contrasts, fold_draws = fold_persistence_contrasts(
        pairs, iterations=iterations
    )
    rolling_contrasts, rolling_draws = load_and_validate_rolling_bootstrap(
        rolling_source
    )

    input_payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "chapter3_q4_primary_input_manifest_v2",
        "source_root": str(source),
        "source_artifacts": _source_file_rows(source),
        "rolling_bootstrap_source_root": str(rolling_source),
        "rolling_bootstrap_source_artifacts": _rolling_source_file_rows(rolling_source),
        "fold": Q4_FOLD,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    input_payload["payload_sha256"] = _payload_sha256(input_payload)
    _write_json(output / INPUT_MANIFEST, input_payload)
    _write_csv(output / Q4_ARM_SUMMARY, arm_summary)
    _write_csv(output / Q4_TRAINING_SUMMARY, training_summary)
    _write_csv(output / ROLLING_SUMMARY, rolling_summary)
    _write_csv(output / Q4_CONTRASTS, contrasts)
    _write_csv(output / Q4_DRAWS, draws, compressed=True)
    _write_csv(output / FOLD_PERSISTENCE_CONTRASTS, fold_contrasts)
    _write_csv(output / FOLD_PERSISTENCE_DRAWS, fold_draws, compressed=True)
    _write_csv(output / ROLLING_PERSISTENCE_CONTRASTS, rolling_contrasts)
    _write_csv(output / ROLLING_PERSISTENCE_DRAWS, rolling_draws, compressed=True)
    _atomic_bytes(
        output / REPORT_PATH,
        _report(
            arm_summary,
            training_summary,
            rolling_summary,
            contrasts,
            fold_contrasts,
            rolling_contrasts,
        ).encode("utf-8"),
    )

    analysis_artifacts = (
        INPUT_MANIFEST,
        Q4_ARM_SUMMARY,
        Q4_TRAINING_SUMMARY,
        ROLLING_SUMMARY,
        Q4_CONTRASTS,
        Q4_DRAWS,
        FOLD_PERSISTENCE_CONTRASTS,
        FOLD_PERSISTENCE_DRAWS,
        ROLLING_PERSISTENCE_CONTRASTS,
        ROLLING_PERSISTENCE_DRAWS,
        REPORT_PATH,
    )
    analysis_payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "chapter3_q4_primary_analysis_manifest_v2",
        "interpretation": "retrospective_development_test",
        "confirmatory": False,
        "fold": Q4_FOLD,
        "seed_count": len(SEEDS),
        "pair_count_per_seed": EXPECTED_PAIRS[Q4_FOLD],
        "session_count_per_seed": EXPECTED_SESSIONS[Q4_FOLD],
        "comparison_count": len(contrasts),
        "fold_persistence_comparison_count": len(fold_contrasts),
        "rolling_persistence_comparison_count": len(rolling_contrasts),
        "total_bootstrap_draws": int(len(draws) + len(fold_draws) + len(rolling_draws)),
        "artifacts": [_artifact_row(output, path) for path in analysis_artifacts],
    }
    analysis_payload["payload_sha256"] = _payload_sha256(analysis_payload)
    _write_json(output / ANALYSIS_MANIFEST, analysis_payload)

    qa_payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "chapter3_q4_primary_terminal_qa_v2",
        "status": "passed",
        "source_qa_sha256": sha256_file(source / SOURCE_QA),
        "q4_pair_rows": int(len(pairs[pairs["fold"].eq(Q4_FOLD)])),
        "q4_arm_count": len(arm_summary),
        "q4_training_rows": len(training_summary),
        "rolling_summary_rows": len(rolling_summary),
        "comparison_count": len(contrasts),
        "fold_persistence_comparison_count": len(fold_contrasts),
        "rolling_persistence_comparison_count": len(rolling_contrasts),
        "bootstrap_draw_rows": len(draws),
        "fold_persistence_bootstrap_draw_rows": len(fold_draws),
        "rolling_persistence_bootstrap_draw_rows": len(rolling_draws),
        "bootstrap_iterations_per_comparison": int(iterations),
        "all_outputs_finite": bool(
            np.isfinite(
                pd.concat(
                    [
                        contrasts[
                            [
                                "mean_log_mae_ratio",
                                "bootstrap_t",
                                "holm_adjusted_p",
                            ]
                        ],
                        fold_contrasts[
                            [
                                "mean_log_mae_ratio",
                                "bootstrap_t",
                                "holm_adjusted_p",
                            ]
                        ],
                        rolling_contrasts[
                            [
                                "mean_log_mae_ratio",
                                "bootstrap_t",
                                "holm_adjusted_p",
                            ]
                        ],
                    ],
                    ignore_index=True,
                ).to_numpy(float)
            ).all()
        ),
    }
    qa_payload["payload_sha256"] = _payload_sha256(qa_payload)
    _write_json(output / QA_PATH, qa_payload)

    hashed = (*analysis_artifacts, ANALYSIS_MANIFEST, QA_PATH)
    _write_csv(
        output / OUTPUT_HASHES,
        pd.DataFrame([_artifact_row(output, path) for path in hashed]),
    )
    verify_root(output, expected_iterations=iterations)
    return output


def verify_root(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    expected_iterations: int = BOOTSTRAP_ITERATIONS,
) -> Path:
    root = resolve_path(output_root)
    required = (
        INPUT_MANIFEST,
        Q4_ARM_SUMMARY,
        Q4_TRAINING_SUMMARY,
        ROLLING_SUMMARY,
        Q4_CONTRASTS,
        Q4_DRAWS,
        FOLD_PERSISTENCE_CONTRASTS,
        FOLD_PERSISTENCE_DRAWS,
        ROLLING_PERSISTENCE_CONTRASTS,
        ROLLING_PERSISTENCE_DRAWS,
        REPORT_PATH,
        ANALYSIS_MANIFEST,
        QA_PATH,
        OUTPUT_HASHES,
    )
    missing = [str(root / path) for path in required if not (root / path).is_file()]
    if missing:
        raise Chapter3Q4AnalysisError(f"Derived artifact is missing: {missing}")
    hashes = pd.read_csv(root / OUTPUT_HASHES)
    for row in hashes.to_dict(orient="records"):
        path = root / str(row["relative_path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise Chapter3Q4AnalysisError(f"Derived output hash drift: {path}")
    qa = json.loads((root / QA_PATH).read_text(encoding="utf-8"))
    contrasts = pd.read_csv(root / Q4_CONTRASTS)
    draws = pd.read_csv(root / Q4_DRAWS)
    fold_contrasts = pd.read_csv(root / FOLD_PERSISTENCE_CONTRASTS)
    fold_draws = pd.read_csv(root / FOLD_PERSISTENCE_DRAWS)
    rolling_contrasts = pd.read_csv(root / ROLLING_PERSISTENCE_CONTRASTS)
    rolling_draws = pd.read_csv(root / ROLLING_PERSISTENCE_DRAWS)
    for name, frame in (
        ("q4", contrasts),
        ("fold", fold_contrasts),
        ("rolling", rolling_contrasts),
    ):
        expected_t = frame["mean_log_mae_ratio"].astype(float) / frame[
            "bootstrap_se"
        ].astype(float)
        expected_stars = frame["holm_adjusted_p"].astype(float).map(significance_stars)
        observed_stars = frame["significance_stars"].fillna("").astype(str)
        if not np.allclose(
            expected_t,
            frame["bootstrap_t"].astype(float),
            rtol=1e-12,
            atol=1e-15,
        ) or not observed_stars.equals(expected_stars):
            raise Chapter3Q4AnalysisError(
                f"Derived bootstrap t/star calculation drift: {name}"
            )
    if (
        qa.get("status") != "passed"
        or qa.get("all_outputs_finite") is not True
        or int(qa.get("q4_pair_rows", -1)) != 7_150
        or int(qa.get("comparison_count", -1)) != 5
        or int(qa.get("fold_persistence_comparison_count", -1)) != 20
        or int(qa.get("rolling_persistence_comparison_count", -1)) != 5
        or int(qa.get("bootstrap_iterations_per_comparison", -1))
        != int(expected_iterations)
        or len(contrasts) != 5
        or len(fold_contrasts) != 20
        or len(rolling_contrasts) != 5
        or len(draws) != 5 * int(expected_iterations)
        or len(fold_draws) != 20 * int(expected_iterations)
        or len(rolling_draws) != 5 * int(expected_iterations)
        or draws.groupby("comparison_id").size().to_dict()
        != {
            comparison: int(expected_iterations)
            for comparison in contrasts["comparison_id"]
        }
        or fold_draws.groupby(["fold", "comparison_id"]).size().to_dict()
        != {
            (str(row.fold), str(row.comparison_id)): int(expected_iterations)
            for row in fold_contrasts.itertuples(index=False)
        }
        or rolling_draws.groupby("arm").size().to_dict()
        != {arm: int(expected_iterations) for arm in ARMS}
        or fold_contrasts.groupby("fold").size().to_dict()
        != {fold: len(ARMS) for fold in FOLDS}
        or not fold_contrasts["holm_family_size"].astype(int).eq(len(ARMS)).all()
        or not rolling_contrasts["holm_family_size"].astype(int).eq(len(ARMS)).all()
        or not np.isfinite(
            pd.concat(
                [
                    contrasts[["bootstrap_t", "holm_adjusted_p"]],
                    fold_contrasts[["bootstrap_t", "holm_adjusted_p"]],
                    rolling_contrasts[["bootstrap_t", "holm_adjusted_p"]],
                ],
                ignore_index=True,
            ).to_numpy(float)
        ).all()
    ):
        raise Chapter3Q4AnalysisError("Derived terminal QA drift")
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument(
        "--rolling-bootstrap-root", default=str(DEFAULT_ROLLING_BOOTSTRAP_ROOT)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        result = run_analysis(
            args.source_root,
            args.output_dir,
            rolling_bootstrap_root=args.rolling_bootstrap_root,
            iterations=args.iterations,
        )
    else:
        result = verify_root(args.output_dir, expected_iterations=args.iterations)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
