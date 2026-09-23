"""Composite supervisor and analysis for the 10-seed five-arm text experiment.

The two child experiments deliberately have different Generator contracts:
four text arms use the FiLM U-Net while ``no_text`` uses the pure-CNN U-Net.
This module never trains a model itself.  It runs the two hash-checked child
pipelines sequentially, combines their frozen pair evidence, and reports the
predeclared ten-seed estimands.  Selecting the lowest observed seed is exposed
only as a descriptive oracle and is never used as the primary result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq123 import news_first_vol_film_nolp_10seed_analysis as unified


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_text_10seed_composite.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_text_10seed_composite_lr2p5e5_exact_ttm_rolling_v1"
)
ARMS = ("lp_matched", "no_text", "lp_shuffle", "bow", "sentiment")
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
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
EXPECTED_PAIRS_BY_FOLD = {
    "f1_2023q1": 110,
    "f2_2023q2": 112,
    "f3_2023q3": 135,
    "f4_2023q4": 143,
}
EXPECTED_SESSIONS_BY_FOLD = {
    "f1_2023q1": 34,
    "f2_2023q2": 36,
    "f3_2023q3": 33,
    "f4_2023q4": 45,
}
EXPECTED_PAIR_ROWS = 25_000
EXPECTED_TRAINING_JOBS = 200
INTERPRETATION = "retrospective_rolling_development"
ANALYSIS_KIND = "news_first_vol_text_10seed_composite_analysis_v1"
QA_KIND = "news_first_vol_text_10seed_composite_terminal_qa_v1"


class CompositeError(ValueError):
    """Raised when a child or combined artifact violates the frozen contract."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise CompositeError(f"Existing composite artifact drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_write(path, _canonical_bytes(payload))


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_write(
        path,
        frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
            "utf-8"
        ),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositeError(f"{label} must be a mapping")
    return dict(value)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = resolve_path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = _mapping(raw, "composite config")
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise CompositeError("Composite config schema_version must be 1")
    execution = _mapping(config.get("execution"), "execution")
    if execution.get("mode") != "sequential_children":
        raise CompositeError("Only sequential_children execution is supported")
    children = _mapping(config.get("children"), "children")
    if tuple(children) != ("film_text", "pure_cnn_no_text"):
        raise CompositeError("Composite requires film_text then pure_cnn_no_text")
    expected_child_arms = {
        "film_text": {"lp_matched", "lp_shuffle", "bow", "sentiment"},
        "pure_cnn_no_text": {"no_text"},
    }
    arm_union: set[str] = set()
    job_total = row_total = 0
    for name, expected in expected_child_arms.items():
        child = _mapping(children[name], f"children.{name}")
        for key in ("module", "config", "output_root"):
            if not str(child.get(key, "")).strip():
                raise CompositeError(f"children.{name}.{key} is required")
        arms = set(map(str, child.get("arms") or ()))
        if arms != expected:
            raise CompositeError(f"children.{name}.arms drift")
        source_arms = set(map(str, child.get("source_arms") or ()))
        aliases = {
            str(key): str(value)
            for key, value in _mapping(
                child.get("arm_aliases", {}), f"children.{name}.arm_aliases"
            ).items()
        }
        if not source_arms or set(aliases) - source_arms:
            raise CompositeError(f"children.{name}.source_arms/aliases drift")
        reported_from_source = {aliases.get(arm, arm) for arm in source_arms}
        if reported_from_source != arms:
            raise CompositeError(f"children.{name} source-to-report arm mapping drift")
        if arm_union & arms:
            raise CompositeError("Child arm universes overlap")
        arm_union |= arms
        job_total += int(child.get("expected_training_jobs", -1))
        row_total += int(child.get("expected_pair_metric_rows", -1))
    matrix = _mapping(config.get("matrix"), "matrix")
    if (
        tuple(map(int, matrix.get("seeds") or ())) != SEEDS
        or tuple(map(str, matrix.get("folds") or ())) != FOLDS
        or tuple(map(str, matrix.get("arms") or ())) != ARMS
        or arm_union != set(ARMS)
        or int(matrix.get("tolerance_minutes", -1)) != 5
        or int(matrix.get("expected_training_jobs", -1)) != EXPECTED_TRAINING_JOBS
        or int(matrix.get("expected_pair_metric_rows", -1)) != EXPECTED_PAIR_ROWS
        or job_total != EXPECTED_TRAINING_JOBS
        or row_total != EXPECTED_PAIR_ROWS
    ):
        raise CompositeError("Composite matrix contract drift")
    pair_counts = {
        str(key): int(value)
        for key, value in _mapping(
            matrix.get("expected_pairs_by_fold"), "expected_pairs_by_fold"
        ).items()
    }
    session_counts = {
        str(key): int(value)
        for key, value in _mapping(
            matrix.get("expected_sessions_by_fold"), "expected_sessions_by_fold"
        ).items()
    }
    if pair_counts != EXPECTED_PAIRS_BY_FOLD or session_counts != (
        EXPECTED_SESSIONS_BY_FOLD
    ):
        raise CompositeError("Fold pair/session contract drift")
    analysis = _mapping(config.get("analysis"), "analysis")
    run_bootstrap = analysis.get("run_bootstrap", True)
    if not isinstance(run_bootstrap, bool):
        raise CompositeError("analysis.run_bootstrap must be a boolean")
    if (
        analysis.get("focal_arm") != "lp_matched"
        or tuple(map(str, analysis.get("reference_arms") or ()))
        != ("no_text", "lp_shuffle", "bow", "sentiment")
        or int(analysis.get("bootstrap_replicates", -1)) != 10_000
        or not bool(analysis.get("require_shared_noise_bank"))
        or analysis.get("best_observed_seed_interpretation")
        != "descriptive_oracle_not_model_selection"
    ):
        raise CompositeError("Composite analysis contract drift")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _child_artifacts(name: str, child: Mapping[str, Any]) -> dict[str, Any]:
    root = resolve_path(str(child["output_root"]))
    registry_path = root / "registry/task_registry.json"
    qa_path = root / "qa.json"
    hashes_path = root / "output_hashes.csv"
    pair_path = root / "analysis/rq12_pair_metrics.csv.gz"
    training_path = root / "analysis/training_summary.csv"
    required = (registry_path, qa_path, hashes_path, pair_path, training_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CompositeError(f"Child {name} artifacts are missing: {missing}")
    registry = _read_json(registry_path)
    qa = _read_json(qa_path)
    if not registry.get("terminal_complete") or registry.get("status") != "completed":
        raise CompositeError(f"Child {name} is not terminal-complete")
    if qa.get("status") not in {"pass", "passed"}:
        raise CompositeError(f"Child {name} terminal QA did not pass")
    if int(registry.get("expected_training_jobs", -1)) != int(
        child["expected_training_jobs"]
    ):
        raise CompositeError(f"Child {name} job count drift")
    hashes = pd.read_csv(hashes_path, dtype=str, keep_default_na=False)
    if not {"relative_path", "size_bytes", "sha256"}.issubset(hashes.columns):
        raise CompositeError(f"Child {name} output hash manifest is malformed")
    for path in (pair_path, training_path, qa_path):
        relative = path.relative_to(root).as_posix()
        row = hashes.loc[hashes["relative_path"].eq(relative)]
        if (
            len(row) != 1
            or int(row.iloc[0]["size_bytes"]) != path.stat().st_size
            or row.iloc[0]["sha256"] != sha256_file(path)
        ):
            raise CompositeError(f"Child {name} output hash drift: {relative}")
    return {
        "name": name,
        "root": root,
        "registry": registry_path,
        "qa": qa_path,
        "output_hashes": hashes_path,
        "pair_metrics": pair_path,
        "training_summary": training_path,
    }


def _normalise_pairs(frame: pd.DataFrame, *, source_child: str) -> pd.DataFrame:
    required = {
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
        "noise_bank_profile_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise CompositeError(f"Child pair metrics missing columns: {missing}")
    result = frame.copy()
    result["source_child"] = source_child
    for column in ("job_id", "fold", "arm", "pair_id", "session_id"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise CompositeError(f"Pair metric {column} contains empty values")
    for column in ("seed", "tolerance_minutes"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise CompositeError(f"Pair metric {column} must contain integers")
        result[column] = values.astype(int)
    for column in ("target_mae", "persistence_mae"):
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all() or values.le(0.0).any():
            raise CompositeError(f"Pair metric {column} must be finite and positive")
        result[column] = values
    return result


def validate_combined_pairs(
    source: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
    folds: Sequence[str] = FOLDS,
    arms: Sequence[str] = ARMS,
    expected_pairs_by_fold: Mapping[str, int] = EXPECTED_PAIRS_BY_FOLD,
    expected_sessions_by_fold: Mapping[str, int] = EXPECTED_SESSIONS_BY_FOLD,
    expected_rows: int | None = EXPECTED_PAIR_ROWS,
    require_shared_noise_bank: bool = True,
) -> pd.DataFrame:
    """Validate the complete arm/seed/fold panel and paired lineage."""

    frame = source.copy()
    expected_seeds = tuple(map(int, seeds))
    expected_folds = tuple(map(str, folds))
    expected_arms = tuple(map(str, arms))
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise CompositeError(
            f"Combined pair rows drift: {len(frame)} != {expected_rows}"
        )
    expected_cells = {
        (seed, fold, arm)
        for seed in expected_seeds
        for fold in expected_folds
        for arm in expected_arms
    }
    observed_cells = set(
        frame[["seed", "fold", "arm"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise CompositeError("Combined seed/fold/arm universe drift")
    if frame["job_id"].nunique() != len(expected_cells):
        raise CompositeError("Each combined experiment cell must have one unique job")
    stable = [
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "persistence_mae",
    ]
    if require_shared_noise_bank:
        stable.append("noise_bank_profile_sha256")
    for seed in expected_seeds:
        for fold in expected_folds:
            cell = frame[frame["seed"].eq(seed) & frame["fold"].eq(fold)]
            reference: pd.DataFrame | None = None
            for arm in expected_arms:
                panel = (
                    cell[cell["arm"].eq(arm)][stable]
                    .sort_values("pair_id", kind="stable")
                    .reset_index(drop=True)
                )
                expected_pairs = int(expected_pairs_by_fold[fold])
                expected_sessions = int(expected_sessions_by_fold[fold])
                if len(panel) != expected_pairs or panel["session_id"].nunique() != (
                    expected_sessions
                ):
                    raise CompositeError(
                        f"Pair/session count drift: seed={seed}, fold={fold}, arm={arm}"
                    )
                if reference is None:
                    reference = panel
                elif not panel.equals(reference):
                    raise CompositeError(
                        f"Paired lineage differs across arms: seed={seed}, fold={fold}"
                    )
    market_columns = stable[:-1] if require_shared_noise_bank else stable
    for fold in expected_folds:
        reference = None
        for seed in expected_seeds:
            panel = (
                frame[
                    frame["seed"].eq(seed)
                    & frame["fold"].eq(fold)
                    & frame["arm"].eq(expected_arms[0])
                ][market_columns]
                .sort_values("pair_id", kind="stable")
                .reset_index(drop=True)
            )
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise CompositeError(f"Market panel differs across seeds: fold={fold}")
    return frame.sort_values(
        ["arm", "seed", "fold", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def combine_child_evidence(
    config: Mapping[str, Any], root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    children = _mapping(config["children"], "children")
    pair_parts: list[pd.DataFrame] = []
    training_parts: list[pd.DataFrame] = []
    bindings: list[dict[str, Any]] = []
    for name, child_value in children.items():
        child = _mapping(child_value, f"children.{name}")
        artifacts = _child_artifacts(name, child)
        pairs = _normalise_pairs(
            pd.read_csv(artifacts["pair_metrics"], low_memory=False),
            source_child=name,
        )
        pairs["source_arm"] = pairs["arm"].astype(str)
        aliases = {
            str(key): str(value)
            for key, value in _mapping(
                child.get("arm_aliases", {}), f"children.{name}.arm_aliases"
            ).items()
        }
        pairs["arm"] = pairs["source_arm"].map(
            lambda arm: aliases.get(str(arm), str(arm))
        )
        expected_arms = set(map(str, child["arms"]))
        expected_source_arms = set(map(str, child["source_arms"]))
        if (
            set(pairs["source_arm"]) != expected_source_arms
            or set(pairs["arm"]) != expected_arms
            or len(pairs) != int(child["expected_pair_metric_rows"])
        ):
            raise CompositeError(f"Child {name} pair arm/count drift")
        training = pd.read_csv(artifacts["training_summary"], low_memory=False)
        training = training.copy()
        training["source_arm"] = training["arm"].astype(str)
        training["arm"] = training["source_arm"].map(
            lambda arm: aliases.get(str(arm), str(arm))
        )
        if (
            len(training) != int(child["expected_training_jobs"])
            or set(training["arm"].astype(str)) != expected_arms
        ):
            raise CompositeError(f"Child {name} training summary drift")
        training["source_child"] = name
        pair_parts.append(pairs)
        training_parts.append(training)
        bindings.append(
            {
                "child": name,
                "root": str(artifacts["root"]),
                "config_path": str(resolve_path(str(child["config"]))),
                "config_sha256": sha256_file(resolve_path(str(child["config"]))),
                "output_hashes_path": str(artifacts["output_hashes"]),
                "output_hashes_sha256": sha256_file(artifacts["output_hashes"]),
                "pair_metrics_path": str(artifacts["pair_metrics"]),
                "pair_metrics_sha256": sha256_file(artifacts["pair_metrics"]),
                "training_summary_path": str(artifacts["training_summary"]),
                "training_summary_sha256": sha256_file(artifacts["training_summary"]),
                "qa_path": str(artifacts["qa"]),
                "qa_sha256": sha256_file(artifacts["qa"]),
            }
        )
    matrix = _mapping(config["matrix"], "matrix")
    combined = validate_combined_pairs(
        pd.concat(pair_parts, ignore_index=True, sort=False),
        seeds=matrix["seeds"],
        folds=matrix["folds"],
        arms=matrix["arms"],
        expected_pairs_by_fold=matrix["expected_pairs_by_fold"],
        expected_sessions_by_fold=matrix["expected_sessions_by_fold"],
        expected_rows=int(matrix["expected_pair_metric_rows"]),
        require_shared_noise_bank=bool(config["analysis"]["require_shared_noise_bank"]),
    )
    training = pd.concat(training_parts, ignore_index=True, sort=False)
    if (
        len(training) != int(matrix["expected_training_jobs"])
        or training["job_id"].astype(str).duplicated().any()
    ):
        raise CompositeError("Combined training job universe drift")
    _write_csv(root / "analysis/text_10seed_composite_pair_metrics.csv", combined)
    _write_csv(root / "analysis/text_10seed_composite_training_summary.csv", training)
    return combined, training, bindings


def _summaries(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    for (arm, seed, fold), group in pairs.groupby(["arm", "seed", "fold"], sort=True):
        model = float(group["target_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        fold_rows.append(
            {
                "arm": str(arm),
                "seed": int(seed),
                "fold": str(fold),
                "mean_mae": model,
                "persistence_mae": persistence,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "pair_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
            }
        )
    folds = pd.DataFrame(fold_rows)
    seed_rows: list[dict[str, Any]] = []
    for (arm, seed), group in folds.groupby(["arm", "seed"], sort=True):
        model = float(group["mean_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        seed_rows.append(
            {
                "arm": str(arm),
                "seed": int(seed),
                "equal_fold_mae": model,
                "median_fold_mae": float(group["mean_mae"].median()),
                "equal_fold_persistence_mae": persistence,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "fold_count": int(len(group)),
                "descriptive_oracle_candidate": True,
            }
        )
    seeds = pd.DataFrame(seed_rows)
    arm_rows: list[dict[str, Any]] = []
    for arm, group in seeds.groupby("arm", sort=True):
        ordered = group.sort_values(
            ["equal_fold_mae", "seed"], kind="stable"
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        model = float(group["equal_fold_mae"].mean())
        persistence = float(group["equal_fold_persistence_mae"].mean())
        arm_rows.append(
            {
                "arm": str(arm),
                "ten_seed_equal_weight_mae": model,
                "ten_seed_median_mae": float(group["equal_fold_mae"].median()),
                "ten_seed_mae_sd": float(group["equal_fold_mae"].std(ddof=1)),
                "ten_seed_min_mae": float(group["equal_fold_mae"].min()),
                "ten_seed_max_mae": float(group["equal_fold_mae"].max()),
                "ten_seed_equal_weight_persistence_mae": persistence,
                "improvement_vs_persistence_percent": 100.0
                * (1.0 - model / persistence),
                "best_observed_seed": int(best["seed"]),
                "best_observed_seed_mae": float(best["equal_fold_mae"]),
                "best_observed_seed_improvement_vs_persistence_percent": float(
                    best["improvement_vs_persistence_percent"]
                ),
                "best_observed_seed_role": (
                    "descriptive_oracle_only_not_primary_not_selection"
                ),
                "seed_count": int(len(group)),
            }
        )
    arms = pd.DataFrame(arm_rows)
    arms["ten_seed_mae_rank"] = (
        arms["ten_seed_equal_weight_mae"].rank(method="min", ascending=True).astype(int)
    )
    order = {arm: index for index, arm in enumerate(ARMS)}
    arms["_order"] = arms["arm"].map(order)
    arms = arms.sort_values(["ten_seed_mae_rank", "_order"], kind="stable").drop(
        columns="_order"
    )
    return (
        folds.sort_values(["arm", "seed", "fold"], kind="stable").reset_index(
            drop=True
        ),
        seeds.sort_values(["arm", "seed"], kind="stable").reset_index(drop=True),
        arms.reset_index(drop=True),
    )


def _contrasts(
    pairs: pd.DataFrame,
    *,
    seeds: Sequence[int],
    folds: Sequence[str],
    focal_arm: str,
    reference_arms: Sequence[str],
    iterations: int,
    rng_seed: int,
    alpha: float,
    minimum_nonworse_seeds: int,
    minimum_nonworse_folds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(reference_arms):
        try:
            row = unified.seed_fold_session_paired_bootstrap(
                pairs,
                focal_arm=str(focal_arm),
                reference_arm=str(reference),
                expected_seeds=seeds,
                expected_folds=folds,
                iterations=int(iterations),
                rng_seed=int(rng_seed) + index,
            )
        except unified.UnifiedAnalysisError as exc:
            raise CompositeError(f"Bootstrap failed for {reference}: {exc}") from exc
        row.update(
            {
                "contrast_id": f"{focal_arm}_vs_{reference}",
                "multiplicity_family": "lp_matched_vs_four_alternatives_holm4",
                "interpretation": INTERPRETATION,
                "confirmatory": False,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    adjusted = unified.holm_adjust(
        dict(
            zip(
                result["contrast_id"].astype(str),
                result["p_value_one_sided"].astype(float),
                strict=True,
            )
        )
    )
    result["holm_adjusted_p"] = result["contrast_id"].map(adjusted)
    result["holm_family_size"] = len(reference_arms)
    result["required_consistent_seed_count"] = int(minimum_nonworse_seeds)
    result["required_consistent_fold_count"] = int(minimum_nonworse_folds)
    result["passes_retrospective_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["ci_95_upper"].lt(0.0)
        & result["holm_adjusted_p"].lt(float(alpha))
        & result["consistent_seed_count"].ge(int(minimum_nonworse_seeds))
        & result["consistent_fold_count"].ge(int(minimum_nonworse_folds))
    )
    return result.sort_values("contrast_id", kind="stable").reset_index(drop=True)


def analyze_pairs(
    pairs: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
    folds: Sequence[str] = FOLDS,
    focal_arm: str = "lp_matched",
    reference_arms: Sequence[str] = ("no_text", "lp_shuffle", "bow", "sentiment"),
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260904,
    alpha: float = 0.05,
    minimum_nonworse_seeds: int = 7,
    minimum_nonworse_folds: int = 3,
    run_bootstrap: bool = True,
) -> dict[str, Any]:
    fold_summary, seed_summary, arm_summary = _summaries(pairs)
    result = {
        "fold_summary": fold_summary,
        "seed_summary": seed_summary,
        "arm_summary": arm_summary,
    }
    if run_bootstrap:
        result["contrasts"] = _contrasts(
            pairs,
            seeds=seeds,
            folds=folds,
            focal_arm=focal_arm,
            reference_arms=reference_arms,
            iterations=bootstrap_iterations,
            rng_seed=bootstrap_seed,
            alpha=alpha,
            minimum_nonworse_seeds=minimum_nonworse_seeds,
            minimum_nonworse_folds=minimum_nonworse_folds,
        )
    return result


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[list(columns)].copy()

    def display(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not math.isfinite(float(value)) else f"{float(value):.10g}"
        return str(value)

    rows = [[display(value) for value in row] for row in selected.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([header, rule, *("| " + " | ".join(row) + " |" for row in rows)])


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[list(columns)].copy()
    headings = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in selected.itertuples(index=False, name=None):
        cells = []
        for value in row:
            rendered = (
                f"{float(value):.10g}"
                if isinstance(value, (float, np.floating))
                and math.isfinite(float(value))
                else str(value)
            )
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _reports(
    arm_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    contrasts: pd.DataFrame | None,
    *,
    bootstrap_iterations: int,
    run_bootstrap: bool = True,
) -> tuple[str, str]:
    arm_columns = (
        "ten_seed_mae_rank",
        "arm",
        "ten_seed_equal_weight_mae",
        "ten_seed_median_mae",
        "ten_seed_mae_sd",
        "improvement_vs_persistence_percent",
        "best_observed_seed",
        "best_observed_seed_mae",
    )
    contrast_columns = (
        "contrast_id",
        "mean_log_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "passes_retrospective_support_gate",
    )
    seed_columns = (
        "arm",
        "seed",
        "equal_fold_mae",
        "improvement_vs_persistence_percent",
    )
    if run_bootstrap:
        if contrasts is None:
            raise CompositeError("Bootstrap report requires contrast results")
        markdown_inference = f"""## Paired inference

LP matched is compared with each alternative using {bootstrap_iterations:,} draws of the seed → fold → paired CME-session bootstrap. Negative log-MAE ratios favour LP matched; the four comparisons form one Holm-4 family.

{_markdown_table(contrasts, contrast_columns)}
"""
        html_inference = (
            "<h2>Paired bootstrap and Holm-4</h2>"
            f"<p>{bootstrap_iterations:,} seed → fold → paired CME-session "
            "draws; negative values favour LP matched.</p>"
            f"{_html_table(contrasts, contrast_columns)}"
        )
    else:
        markdown_inference = """## Descriptive analysis only

Bootstrap confidence intervals and Holm-adjusted inference are intentionally deferred (`analysis.run_bootstrap=false`). This report contains the predeclared equal-weight 10-seed averages, dispersion, best-observed-seed descriptors, and all per-seed results only.
"""
        html_inference = (
            "<h2>Descriptive analysis only</h2>"
            "<p>Bootstrap confidence intervals and Holm-adjusted inference are "
            "intentionally deferred (<code>analysis.run_bootstrap=false</code>). "
            "This report contains 10-seed descriptive summaries only.</p>"
        )
    markdown = f"""# FiLM text vs pure-CNN no-text: 10-seed composite

## Primary result

The primary ranking uses the mean of four fold MAEs within each seed and then gives all ten seeds equal weight. The `best_observed_seed` columns are a **descriptive oracle only**: they are reported because requested, but they are not a valid model-selection result and are not used for the conclusion.

{_markdown_table(arm_summary, arm_columns)}

{markdown_inference}

## All seed results

{_markdown_table(seed_summary, seed_columns)}

## Scope

All jobs are random-initialization direct training: there are no parent or continuation stages. Results are labelled `{INTERPRETATION}` and are not confirmatory evidence.
"""
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>10-seed text composite</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#18212b}}table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:1rem 0}}th,td{{border:1px solid #ccd4dc;padding:.42rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f7}}.note{{padding:1rem;background:#fff6d8;border-left:4px solid #d29b00}}</style></head>
<body><h1>FiLM text vs pure-CNN no-text: 10-seed composite</h1>
<p class="note"><strong>Primary estimand:</strong> all ten seeds receive equal weight. Best-observed-seed values are a descriptive oracle only and must not be used as the headline result or for model selection.</p>
<h2>Ten-seed arm summary</h2>{_html_table(arm_summary, arm_columns)}
{html_inference}
<h2>All seed results</h2>{_html_table(seed_summary, seed_columns)}
<h2>Scope</h2><p>Random-initialization direct training only; parent jobs=0 and continuation jobs=0. Interpretation: {INTERPRETATION}; not confirmatory.</p></body></html>
"""
    return markdown, html_report


def analyze_composite(config: Mapping[str, Any], output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    pairs, _training, bindings = combine_child_evidence(config, root)
    matrix = _mapping(config["matrix"], "matrix")
    analysis_config = _mapping(config["analysis"], "analysis")
    run_bootstrap = bool(analysis_config.get("run_bootstrap", True))
    results = analyze_pairs(
        pairs,
        seeds=matrix["seeds"],
        folds=matrix["folds"],
        focal_arm=str(analysis_config["focal_arm"]),
        reference_arms=analysis_config["reference_arms"],
        bootstrap_iterations=int(analysis_config["bootstrap_replicates"]),
        bootstrap_seed=int(analysis_config["bootstrap_seed"]),
        alpha=float(analysis_config["holm_alpha"]),
        minimum_nonworse_seeds=int(analysis_config["minimum_nonworse_seeds"]),
        minimum_nonworse_folds=int(analysis_config["minimum_nonworse_folds"]),
        run_bootstrap=run_bootstrap,
    )
    paths = {
        "fold_summary": root / "analysis/text_10seed_composite_fold_summary.csv",
        "seed_summary": root / "analysis/text_10seed_composite_seed_summary.csv",
        "arm_summary": root / "analysis/text_10seed_composite_arm_summary.csv",
    }
    if run_bootstrap:
        paths["contrasts"] = root / "analysis/text_10seed_composite_bootstrap_holm.csv"
    for key, path in paths.items():
        _write_csv(path, results[key])
    markdown, html_report = _reports(
        results["arm_summary"],
        results["seed_summary"],
        results.get("contrasts"),
        bootstrap_iterations=int(analysis_config["bootstrap_replicates"]),
        run_bootstrap=run_bootstrap,
    )
    report_md = _atomic_write(
        root / "report/text_10seed_composite_report.md", markdown.encode("utf-8")
    )
    report_html = _atomic_write(
        root / "report/text_10seed_composite_report.html",
        html_report.encode("utf-8"),
    )
    input_manifest = {
        "schema_version": 1,
        "kind": "news_first_vol_text_10seed_composite_input_binding_v1",
        "children": bindings,
    }
    input_manifest["payload_sha256"] = hashlib.sha256(
        _canonical_bytes(input_manifest)
    ).hexdigest()
    input_path = _write_json(
        root / "analysis/text_10seed_composite_input_manifest.json", input_manifest
    )
    output_files = [
        root / "analysis/text_10seed_composite_pair_metrics.csv",
        root / "analysis/text_10seed_composite_training_summary.csv",
        *paths.values(),
        report_md,
        report_html,
    ]
    arm_summary = results["arm_summary"]
    summary = {
        "schema_version": 1,
        "kind": ANALYSIS_KIND,
        "interpretation": INTERPRETATION,
        "confirmatory": False,
        "primary_result_uses_all_seeds_equal_weight": True,
        "best_observed_seed_role": "descriptive_oracle_only_not_primary_not_selection",
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "direct_training_jobs": EXPECTED_TRAINING_JOBS,
        "pair_metric_rows": EXPECTED_PAIR_ROWS,
        "seed_count": len(SEEDS),
        "fold_count": len(FOLDS),
        "arm_count": len(ARMS),
        "bootstrap_enabled": run_bootstrap,
        "bootstrap_replicates": (
            int(analysis_config["bootstrap_replicates"]) if run_bootstrap else 0
        ),
        "holm_family_size": (
            len(analysis_config["reference_arms"]) if run_bootstrap else 0
        ),
        "inference_status": ("completed" if run_bootstrap else "deferred_by_config"),
        "descriptive_leader_arm": str(arm_summary.iloc[0]["arm"]),
        "descriptive_leader_ten_seed_equal_weight_mae": float(
            arm_summary.iloc[0]["ten_seed_equal_weight_mae"]
        ),
        "input_manifest": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "artifacts": [
            {
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    summary["payload_sha256"] = hashlib.sha256(_canonical_bytes(summary)).hexdigest()
    return _write_json(
        root / "analysis/text_10seed_composite_analysis_manifest.json", summary
    )


def _validate_direct_job_contract(
    name: str,
    jobs: Sequence[Mapping[str, Any]],
    child_qa: Mapping[str, Any],
    *,
    expected_jobs: int,
) -> None:
    if len(jobs) != int(expected_jobs):
        raise CompositeError(f"Child {name} registry job count drift")
    invalid = [
        str(job.get("job_id", ""))
        for job in jobs
        if str(job.get("stage", "")) != "direct_arms"
        or bool(str(job.get("parent_state_path", "")).strip())
        or bool(str(job.get("recipe_path", "")).strip())
        or bool(str(job.get("parent_job_id", "")).strip())
        or bool(str(job.get("continuation_job_id", "")).strip())
    ]
    if invalid:
        raise CompositeError(
            f"Child {name} is not random-initialization direct training: {invalid[:3]}"
        )
    if (
        int(child_qa.get("parent_jobs", -1)) != 0
        or int(child_qa.get("continuation_jobs", -1)) != 0
    ):
        raise CompositeError(
            f"Child {name} QA does not attest zero parent/continuation"
        )


def _verify_direct_children(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for name, value in _mapping(config["children"], "children").items():
        child = _mapping(value, f"children.{name}")
        artifacts = _child_artifacts(name, child)
        registry = _read_json(artifacts["registry"])
        child_qa = _read_json(artifacts["qa"])
        jobs = registry.get("jobs") or []
        _validate_direct_job_contract(
            name,
            jobs,
            child_qa,
            expected_jobs=int(child["expected_training_jobs"]),
        )
        evidence.append(
            {
                "child": name,
                "direct_training_jobs": len(jobs),
                "parent_jobs": 0,
                "continuation_jobs": 0,
                "registry_sha256": sha256_file(artifacts["registry"]),
                "qa_sha256": sha256_file(artifacts["qa"]),
            }
        )
    return evidence


def qa(config: Mapping[str, Any], output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    direct_evidence = _verify_direct_children(config)
    pair_path = root / "analysis/text_10seed_composite_pair_metrics.csv"
    training_path = root / "analysis/text_10seed_composite_training_summary.csv"
    manifest_path = root / "analysis/text_10seed_composite_analysis_manifest.json"
    required = (pair_path, training_path, manifest_path)
    if any(not path.is_file() for path in required):
        raise CompositeError("Composite analysis artifacts are incomplete")
    pairs = pd.read_csv(pair_path, low_memory=False)
    matrix = _mapping(config["matrix"], "matrix")
    validate_combined_pairs(
        pairs,
        seeds=matrix["seeds"],
        folds=matrix["folds"],
        arms=matrix["arms"],
        expected_pairs_by_fold=matrix["expected_pairs_by_fold"],
        expected_sessions_by_fold=matrix["expected_sessions_by_fold"],
        expected_rows=int(matrix["expected_pair_metric_rows"]),
        require_shared_noise_bank=bool(config["analysis"]["require_shared_noise_bank"]),
    )
    training = pd.read_csv(training_path, low_memory=False)
    if len(training) != EXPECTED_TRAINING_JOBS:
        raise CompositeError("Composite training summary count drift")
    manifest = _read_json(manifest_path)
    unsigned = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if manifest.get("kind") != ANALYSIS_KIND or hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest() != manifest.get("payload_sha256"):
        raise CompositeError("Composite analysis manifest drift")
    run_bootstrap = bool(config["analysis"].get("run_bootstrap", True))
    if (
        bool(manifest.get("bootstrap_enabled")) != run_bootstrap
        or int(manifest.get("bootstrap_replicates", -1))
        != (int(config["analysis"]["bootstrap_replicates"]) if run_bootstrap else 0)
        or int(manifest.get("holm_family_size", -1))
        != (len(config["analysis"]["reference_arms"]) if run_bootstrap else 0)
    ):
        raise CompositeError("Composite inference-mode manifest drift")
    bootstrap_path = root / "analysis/text_10seed_composite_bootstrap_holm.csv"
    if not run_bootstrap and bootstrap_path.exists():
        raise CompositeError("Deferred analysis must not contain bootstrap/Holm output")
    for row in manifest.get("artifacts") or []:
        path = Path(str(row["path"])).resolve()
        if (
            root not in path.parents
            or not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise CompositeError(f"Composite analysis artifact drift: {path}")
    payload = {
        "schema_version": 1,
        "kind": QA_KIND,
        "status": "passed",
        "interpretation": INTERPRETATION,
        "confirmatory": False,
        "direct_training_jobs": EXPECTED_TRAINING_JOBS,
        "parent_jobs": 0,
        "continuation_jobs": 0,
        "prediction_cells": EXPECTED_TRAINING_JOBS,
        "pair_metric_rows": EXPECTED_PAIR_ROWS,
        "arm_count": len(ARMS),
        "seed_count": len(SEEDS),
        "fold_count": len(FOLDS),
        "best_observed_seed_role": "descriptive_oracle_only_not_primary_not_selection",
        "bootstrap_enabled": run_bootstrap,
        "inference_status": ("completed" if run_bootstrap else "deferred_by_config"),
        "child_direct_training_evidence": direct_evidence,
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "source_config_sha256": str(config["source_config_sha256"]),
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return _write_json(root / "text_10seed_composite_qa.json", payload)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, path)
    return path


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")


def _journal(root: Path, stage: str, status_value: str, **details: Any) -> Path:
    control = root.with_name(root.name + "_control")
    return _replace_json(
        control / "text_10seed_composite_pipeline_journal.json",
        {
            "schema_version": 1,
            "kind": "news_first_vol_text_10seed_composite_pipeline_journal_v1",
            "stage": stage,
            "status": status_value,
            "pid": os.getpid(),
            "updated_at_utc": _utc_now(),
            **details,
        },
    )


def _acquire_lock(root: Path) -> tuple[Path, int]:
    control = root.with_name(root.name + "_control")
    control.mkdir(parents=True, exist_ok=True)
    path = (control / "text_10seed_composite_pipeline.lock").resolve()
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Live composite supervisor already exists: {path}") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    (control / "text_10seed_composite_pipeline.pid").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )
    return path, descriptor


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _child_status(child: Mapping[str, Any]) -> dict[str, Any]:
    root = resolve_path(str(child["output_root"]))
    registry = root / "registry/task_registry.json"
    if not root.exists():
        return {"status": "absent", "root": str(root)}
    if not registry.is_file():
        return {"status": "invalid_partial_root", "root": str(root)}
    payload = _read_json(registry)
    return {
        "status": payload.get("status"),
        "root": str(root),
        "registered_jobs": len(payload.get("jobs") or []),
        "expected_jobs": int(child["expected_training_jobs"]),
        "predictions_frozen": bool(payload.get("predictions_frozen")),
        "terminal_complete": bool(payload.get("terminal_complete")),
    }


def status(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    children = {
        name: _child_status(_mapping(value, f"children.{name}"))
        for name, value in _mapping(config["children"], "children").items()
    }
    pair_path = root / "analysis/text_10seed_composite_pair_metrics.csv"
    qa_path = root / "text_10seed_composite_qa.json"
    journal_path = root.with_name(root.name + "_control") / (
        "text_10seed_composite_pipeline_journal.json"
    )
    journal = _read_json(journal_path) if journal_path.is_file() else {}
    return {
        "status": (
            "completed"
            if qa_path.is_file()
            else journal.get("status", "absent" if not root.exists() else "partial")
        ),
        "stage": journal.get("stage"),
        "root": str(root),
        "children": children,
        "combined_pair_rows": (
            sum(1 for _ in pair_path.open("rb")) - 1 if pair_path.is_file() else 0
        ),
        "expected_pair_rows": EXPECTED_PAIR_ROWS,
        "qa_complete": qa_path.is_file(),
    }


def _run_child(name: str, child: Mapping[str, Any]) -> None:
    current = _child_status(child)
    if current.get("terminal_complete"):
        _child_artifacts(name, child)
        return
    module = importlib.import_module(str(child["module"]))
    runner = getattr(module, "run_pipeline", None)
    if not callable(runner):
        raise CompositeError(f"Child module lacks run_pipeline: {child['module']}")
    arguments = (
        str(resolve_path(str(child["config"]))),
        str(resolve_path(str(child["output_root"]))),
    )
    if name == "film_text":
        film_profile = getattr(module, "film_text_profile", None)
        multiseed_module = getattr(module, "multiseed", None)
        multiseed_profile = getattr(multiseed_module, "multiseed_profile", None)
        if not callable(film_profile) or not callable(multiseed_profile):
            raise CompositeError(
                "FiLM child must expose film_text_profile and multiseed_profile"
            )
        # Keep both reusable-wrapper profiles installed for the complete child
        # lifecycle.  The child runner enters narrower profiles itself, but without
        # these outer contexts its postprocess step restores the generic profile
        # between calls and can falsely report frozen source-manifest drift.
        with film_profile():
            with multiseed_profile():
                runner(*arguments, resume=True)
    else:
        runner(*arguments, resume=True)
    _child_artifacts(name, child)


def _write_output_hashes(root: Path) -> Path:
    paths = [
        root / "text_10seed_composite_resolved_config.yaml",
        root / "text_10seed_composite_qa.json",
        *sorted((root / "analysis").glob("text_10seed_composite_*")),
        *sorted((root / "report").glob("text_10seed_composite_*")),
    ]
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    ]
    return _write_csv(
        root / "text_10seed_composite_output_hashes.csv", pd.DataFrame(rows)
    )


def _verify_output_hashes(root: Path) -> None:
    path = root / "text_10seed_composite_output_hashes.csv"
    if not path.is_file():
        raise CompositeError("Composite output hash manifest is missing")
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    for row in rows.itertuples(index=False):
        target = Path(row.path).resolve()
        if (
            root not in target.parents
            or target.stat().st_size != int(row.size_bytes)
            or sha256_file(target) != row.sha256
        ):
            raise CompositeError(f"Composite terminal output drift: {target}")


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    del resume
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    qa_path = root / "text_10seed_composite_qa.json"
    if qa_path.is_file():
        qa(config, root)
        _verify_output_hashes(root)
        return root
    _lock_path, descriptor = _acquire_lock(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = yaml.safe_dump(
            dict(config), sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        _atomic_write(root / "text_10seed_composite_resolved_config.yaml", resolved)
        children = _mapping(config["children"], "children")
        for name, value in children.items():
            _journal(root, f"child:{name}", "running")
            _run_child(name, _mapping(value, f"children.{name}"))
        _journal(root, "analysis", "running")
        analyze_composite(config, root)
        _journal(root, "qa", "running")
        qa(config, root)
        _write_output_hashes(root)
        _verify_output_hashes(root)
        _journal(root, "terminal", "completed")
        return root
    except BaseException as exc:
        _journal(root, "pipeline", "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _release_lock(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-pipeline", "status"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run-pipeline":
        result: Any = run_pipeline(
            args.config, args.output_dir, resume=bool(args.resume)
        )
        payload: Any = str(result)
    else:
        payload = status(args.output_dir, args.config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CompositeError",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "FOLDS",
    "SEEDS",
    "analyze_composite",
    "analyze_pairs",
    "combine_child_evidence",
    "load_config",
    "main",
    "qa",
    "run_pipeline",
    "status",
    "validate_combined_pairs",
    "validate_config",
]
