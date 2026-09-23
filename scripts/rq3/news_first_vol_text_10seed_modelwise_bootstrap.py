"""Parallel model-wise bootstrap add-on for the frozen 10-seed text experiment.

The source experiment is immutable.  This module binds its terminal artifacts by
SHA, runs five arm-vs-persistence bootstraps in a separate analysis-only root,
and publishes results only after every chunk and terminal QA pass.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import html
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
from typing import Any, Iterator, Mapping, Sequence

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.rq123 import (  # noqa: E402
    news_first_vol_film_nolp_10seed_analysis as unified,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/rq3/news_first_vol_text_10seed_modelwise_bootstrap.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "outputs/experiments/"
    "rq12_news_first_vol_text_10seed_modelwise_bootstrap10000_"
    "lr2p5e5_exact_ttm_rolling_v1"
)
SCRIPT_PATH = Path(__file__).resolve()
CANONICAL_ARMS = ("lp_matched", "lp_shuffle", "no_text", "bow", "sentiment")
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
SUMMARY_RELATIVE = Path("analysis/model_vs_persistence_bootstrap_10000.csv")
DRAWS_RELATIVE = Path("analysis/model_vs_persistence_bootstrap_draws.csv.gz")
RESOURCE_RELATIVE = Path("analysis/bootstrap_chunk_resources.csv")
CHUNK_PLAN_RELATIVE = Path("analysis/bootstrap_chunk_plan.csv")
INPUT_MANIFEST_RELATIVE = Path("bootstrap_input_manifest.json")
ANALYSIS_MANIFEST_RELATIVE = Path("bootstrap_analysis_manifest.json")
QA_RELATIVE = Path("bootstrap_qa.json")
OUTPUT_HASHES_RELATIVE = Path("bootstrap_output_hashes.csv")
REPORT_MD_RELATIVE = Path("report/modelwise_bootstrap_report.md")
REPORT_HTML_RELATIVE = Path("report/modelwise_bootstrap_report.html")
RESOLVED_CONFIG_RELATIVE = Path("resolved_config.yaml")


class BootstrapError(RuntimeError):
    """Fail-closed error for the derived bootstrap analysis."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_bytes(path, _canonical_bytes(dict(value)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, frame: pd.DataFrame, *, gzip: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    compression: Any = (
        {"method": "gzip", "compresslevel": 6, "mtime": 0} if gzip else None
    )
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        compression=compression,
    )
    os.replace(temporary, path)
    return path


def _file_row(
    role: str, path: Path, *, relative_to: Path | None = None
) -> dict[str, Any]:
    target = path.resolve()
    if not target.is_file():
        raise BootstrapError(f"Required file is missing: {target}")
    row: dict[str, Any] = {
        "artifact_role": role,
        "path": str(target),
        "size_bytes": int(target.stat().st_size),
        "sha256": sha256_file(target),
    }
    if relative_to is not None:
        row["relative_path"] = target.relative_to(relative_to.resolve()).as_posix()
        row.pop("path")
    return row


def _verify_file_row(row: Mapping[str, Any], *, root: Path | None = None) -> None:
    if root is None:
        path = Path(str(row["path"])).resolve()
    else:
        path = (root / str(row["relative_path"])).resolve()
        if root.resolve() not in path.parents:
            raise BootstrapError(f"Output manifest escapes root: {path}")
    if (
        not path.is_file()
        or path.stat().st_size != int(row["size_bytes"])
        or sha256_file(path) != str(row["sha256"])
    ):
        raise BootstrapError(f"Manifest row drift: {path}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BootstrapError(f"{label} must be a mapping")
    return dict(value)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(raw, "config")
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = sha256_file(config_path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise BootstrapError("schema_version must be 1")
    matrix = _mapping(config.get("matrix"), "matrix")
    if (
        tuple(map(str, matrix.get("arms") or ())) != CANONICAL_ARMS
        or tuple(map(int, matrix.get("seeds") or ())) != CANONICAL_SEEDS
        or tuple(map(str, matrix.get("folds") or ())) != CANONICAL_FOLDS
        or int(matrix.get("tolerance_minutes", -1)) != 5
        or int(matrix.get("expected_pair_rows", -1)) != 25_000
        or int(matrix.get("expected_pairs_per_arm", -1)) != 5_000
    ):
        raise BootstrapError("Frozen matrix contract drift")
    bootstrap = _mapping(config.get("bootstrap"), "bootstrap")
    iterations = int(bootstrap.get("iterations_per_arm", -1))
    chunks = int(bootstrap.get("chunks_per_arm", -1))
    per_chunk = int(bootstrap.get("draws_per_chunk", -1))
    workers = int(bootstrap.get("max_workers", -1))
    if (
        iterations != 10_000
        or chunks != 10
        or per_chunk != 1_000
        or chunks * per_chunk != iterations
        or workers != 50
        or not bool(bootstrap.get("shared_resampling_schedule"))
        or bootstrap.get("multiprocessing_start_method")
        not in {
            "spawn",
            "forkserver",
        }
    ):
        raise BootstrapError("Formal parallel bootstrap contract drift")
    source = _mapping(config.get("source"), "source")
    required_source = {
        "root",
        "pair_metrics",
        "pair_metrics_sha256",
        "qa",
        "qa_sha256",
        "analysis_manifest",
        "analysis_manifest_sha256",
        "input_manifest",
        "input_manifest_sha256",
        "output_hashes",
        "output_hashes_sha256",
        "resolved_config",
        "resolved_config_sha256",
    }
    if not required_source.issubset(source):
        raise BootstrapError("Source binding is incomplete")


def _source_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _mapping(config["source"], "source")
    root = resolve_path(str(source["root"]))
    bindings = (
        ("source_pair_metrics", "pair_metrics", "pair_metrics_sha256"),
        ("source_terminal_qa", "qa", "qa_sha256"),
        ("source_analysis_manifest", "analysis_manifest", "analysis_manifest_sha256"),
        ("source_input_manifest", "input_manifest", "input_manifest_sha256"),
        ("source_output_hashes", "output_hashes", "output_hashes_sha256"),
        ("source_resolved_config", "resolved_config", "resolved_config_sha256"),
    )
    rows: list[dict[str, Any]] = []
    for role, path_key, sha_key in bindings:
        path = (root / str(source[path_key])).resolve()
        row = _file_row(role, path)
        if row["sha256"] != str(source[sha_key]):
            raise BootstrapError(f"Frozen source SHA drift: {path}")
        rows.append(row)
    rows.extend(
        (
            _file_row("bootstrap_code", SCRIPT_PATH),
            _file_row("bootstrap_config", Path(str(config["_config_path"]))),
        )
    )
    return rows


def _validate_source(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = _source_rows(config)
    for row in rows:
        _verify_file_row(row)
    source = _mapping(config["source"], "source")
    root = resolve_path(str(source["root"]))
    qa = _read_json(root / str(source["qa"]))
    analysis = _read_json(root / str(source["analysis_manifest"]))
    if (
        qa.get("status") != "passed"
        or int(qa.get("direct_training_jobs", -1)) != 200
        or int(qa.get("parent_jobs", -1)) != 0
        or int(qa.get("continuation_jobs", -1)) != 0
        or bool(qa.get("bootstrap_enabled"))
        or analysis.get("inference_status") != "deferred_by_config"
        or bool(analysis.get("bootstrap_enabled"))
    ):
        raise BootstrapError("Source terminal/zero-parent/deferred-bootstrap QA drift")
    pair_path = root / str(source["pair_metrics"])
    frame = pd.read_csv(pair_path, low_memory=False)
    required = {
        "arm",
        "seed",
        "fold",
        "pair_id",
        "session_id",
        "effective_origin_utc",
        "tolerance_minutes",
        "target_mae",
        "persistence_mae",
    }
    if required - set(frame):
        raise BootstrapError(
            f"Pair evidence columns missing: {sorted(required - set(frame))}"
        )
    matrix = _mapping(config["matrix"], "matrix")
    frame = frame.copy()
    frame["arm"] = frame["arm"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["fold"] = frame["fold"].astype(str)
    frame["target_mae"] = pd.to_numeric(frame["target_mae"], errors="raise")
    frame["persistence_mae"] = pd.to_numeric(frame["persistence_mae"], errors="raise")
    key = ["arm", "seed", "fold", "pair_id"]
    if (
        len(frame) != int(matrix["expected_pair_rows"])
        or frame.duplicated(key).any()
        or set(frame["arm"]) != set(CANONICAL_ARMS)
        or set(frame["seed"]) != set(CANONICAL_SEEDS)
        or set(frame["fold"]) != set(CANONICAL_FOLDS)
        or set(pd.to_numeric(frame["tolerance_minutes"], errors="raise")) != {5}
        or not np.isfinite(frame[["target_mae", "persistence_mae"]].to_numpy()).all()
        or (frame[["target_mae", "persistence_mae"]] <= 0).any().any()
        or frame[["pair_id", "session_id", "effective_origin_utc"]]
        .astype(str)
        .apply(lambda column: column.str.strip().eq("").any())
        .any()
    ):
        raise BootstrapError("Pair evidence universe/value contract drift")
    expected_pairs = {
        str(k): int(v) for k, v in matrix["expected_pairs_by_fold"].items()
    }
    expected_sessions = {
        str(k): int(v) for k, v in matrix["expected_sessions_by_fold"].items()
    }
    for arm in CANONICAL_ARMS:
        selected = frame[frame["arm"].eq(arm)]
        if len(selected) != int(matrix["expected_pairs_per_arm"]):
            raise BootstrapError(f"Arm pair count drift: {arm}")
        for seed in CANONICAL_SEEDS:
            for fold in CANONICAL_FOLDS:
                cell = selected[selected["seed"].eq(seed) & selected["fold"].eq(fold)]
                if (
                    len(cell) != expected_pairs[fold]
                    or cell["session_id"].nunique() != expected_sessions[fold]
                ):
                    raise BootstrapError(
                        f"Cell pair/session drift: {arm}/{seed}/{fold}"
                    )
    lineage = ["fold", "pair_id", "session_id", "effective_origin_utc"]
    reference = None
    for arm in CANONICAL_ARMS:
        for seed in CANONICAL_SEEDS:
            panel = (
                frame[frame["arm"].eq(arm) & frame["seed"].eq(seed)][
                    lineage + ["persistence_mae"]
                ]
                .sort_values(["fold", "pair_id"], kind="stable")
                .reset_index(drop=True)
            )
            if reference is None:
                reference = panel
            elif not panel.equals(reference):
                raise BootstrapError(
                    "Market lineage/persistence differs across arms or seeds"
                )
    return frame.sort_values(key, kind="stable").reset_index(drop=True)


def _cell_payloads(
    frame: pd.DataFrame, arm: str
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    selected = frame[frame["arm"].eq(arm)].copy()
    cells: dict[tuple[int, str], dict[str, Any]] = {}
    cell_points: dict[tuple[int, str], tuple[float, float, float]] = {}
    for seed in CANONICAL_SEEDS:
        for fold in CANONICAL_FOLDS:
            cell = selected[selected["seed"].eq(seed) & selected["fold"].eq(fold)]
            labels = cell["session_id"].drop_duplicates().to_numpy(dtype=object)
            model_sums: list[float] = []
            persistence_sums: list[float] = []
            counts: list[int] = []
            for label in labels:
                block = cell[cell["session_id"].eq(label)]
                model_sums.append(float(block["target_mae"].sum()))
                persistence_sums.append(float(block["persistence_mae"].sum()))
                counts.append(len(block))
            entry = {
                "labels": labels,
                "label_to_index": {
                    str(label): index for index, label in enumerate(labels)
                },
                "model_sums": np.asarray(model_sums, dtype=float),
                "persistence_sums": np.asarray(persistence_sums, dtype=float),
                "counts": np.asarray(counts, dtype=np.int64),
            }
            cells[(seed, fold)] = entry
            model = float(cell["target_mae"].mean())
            persistence = float(cell["persistence_mae"].mean())
            cell_points[(seed, fold)] = (
                model,
                persistence,
                math.log(model / persistence),
            )
    model_point = float(np.mean([value[0] for value in cell_points.values()]))
    persistence_point = float(np.mean([value[1] for value in cell_points.values()]))
    log_point = float(np.mean([value[2] for value in cell_points.values()]))
    seed_means = {
        seed: float(np.mean([cell_points[(seed, fold)][2] for fold in CANONICAL_FOLDS]))
        for seed in CANONICAL_SEEDS
    }
    fold_means = {
        fold: float(np.mean([cell_points[(seed, fold)][2] for seed in CANONICAL_SEEDS]))
        for fold in CANONICAL_FOLDS
    }
    points = {
        "model_mean_mae": model_point,
        "persistence_mean_mae": persistence_point,
        "mean_log_mae_ratio": log_point,
        "consistent_seed_count": sum(value <= 0 for value in seed_means.values()),
        "consistent_fold_count": sum(value <= 0 for value in fold_means.values()),
        "seed_log_ratios": seed_means,
        "fold_log_ratios": fold_means,
        "pair_count": int(selected["pair_id"].nunique()),
        "session_count": int(selected["session_id"].nunique()),
    }
    return cells, points


def _bootstrap_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(int(task["rng_seed"]))
    cells = task["cells"]
    iterations = int(task["iterations"])
    seeds = np.asarray(CANONICAL_SEEDS, dtype=np.int64)
    folds = np.asarray(CANONICAL_FOLDS, dtype=object)
    model_draws = np.empty(iterations, dtype=float)
    persistence_draws = np.empty(iterations, dtype=float)
    log_draws = np.empty(iterations, dtype=float)
    for draw_index in range(iterations):
        sampled_models: list[float] = []
        sampled_persistence: list[float] = []
        sampled_logs: list[float] = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            for sampled_fold in rng.choice(folds, size=len(folds), replace=True):
                entry = cells[(int(sampled_seed), str(sampled_fold))]
                labels = entry["labels"]
                chosen = rng.choice(labels, size=len(labels), replace=True)
                indexes = np.fromiter(
                    (entry["label_to_index"][str(label)] for label in chosen),
                    dtype=np.int64,
                    count=len(chosen),
                )
                denominator = int(entry["counts"][indexes].sum())
                model = float(entry["model_sums"][indexes].sum() / denominator)
                persistence = float(
                    entry["persistence_sums"][indexes].sum() / denominator
                )
                sampled_models.append(model)
                sampled_persistence.append(persistence)
                sampled_logs.append(math.log(model / persistence))
        model_draws[draw_index] = float(np.mean(sampled_models))
        persistence_draws[draw_index] = float(np.mean(sampled_persistence))
        log_draws[draw_index] = float(np.mean(sampled_logs))
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "arm": str(task["arm"]),
        "chunk_index": int(task["chunk_index"]),
        "rng_seed": int(task["rng_seed"]),
        "pid": os.getpid(),
        "duration_seconds": float(time.perf_counter() - started),
        "peak_rss_kib": peak_rss,
        "model_draws": model_draws,
        "persistence_draws": persistence_draws,
        "log_draws": log_draws,
    }


def _chunk_schedule(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[int]]:
    bootstrap = _mapping(config["bootstrap"], "bootstrap")
    entropy = [int(value) for value in bootstrap["seed_entropy"]]
    chunks = int(bootstrap["chunks_per_arm"])
    children = np.random.SeedSequence(entropy).spawn(chunks)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    rows: list[dict[str, Any]] = []
    per_chunk = int(bootstrap["draws_per_chunk"])
    for arm in CANONICAL_ARMS:
        for chunk_index, (child, rng_seed) in enumerate(
            zip(children, seeds, strict=True)
        ):
            rows.append(
                {
                    "task_id": f"{arm}_chunk_{chunk_index:02d}",
                    "arm": arm,
                    "chunk_index": chunk_index,
                    "draw_start": chunk_index * per_chunk,
                    "draw_end_exclusive": (chunk_index + 1) * per_chunk,
                    "draw_count": per_chunk,
                    "rng_seed": rng_seed,
                    "seed_sequence_spawn_key_json": json.dumps(list(child.spawn_key)),
                }
            )
    return pd.DataFrame(rows), seeds


def _summaries(
    draws: pd.DataFrame,
    points: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    bootstrap = _mapping(config["bootstrap"], "bootstrap")
    iterations = int(bootstrap["iterations_per_arm"])
    alpha_tail = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    rows: list[dict[str, Any]] = []
    for arm in CANONICAL_ARMS:
        selected = draws[draws["arm"].eq(arm)].sort_values("draw_index")
        if len(selected) != iterations:
            raise BootstrapError(f"Draw count drift: {arm}")
        model_draws = selected["model_mean_mae"].to_numpy(float)
        persistence_draws = selected["persistence_mean_mae"].to_numpy(float)
        log_draws = selected["mean_log_mae_ratio"].to_numpy(float)
        if not np.isfinite(
            np.column_stack((model_draws, persistence_draws, log_draws))
        ).all():
            raise BootstrapError(f"Non-finite draws: {arm}")
        point = points[arm]
        mae_lower, mae_upper = np.quantile(model_draws, (alpha_tail, 1 - alpha_tail))
        log_lower, log_upper = np.quantile(log_draws, (alpha_tail, 1 - alpha_tail))
        p_negative = (1.0 + float(np.count_nonzero(log_draws >= 0.0))) / (
            iterations + 1.0
        )
        p_positive = (1.0 + float(np.count_nonzero(log_draws <= 0.0))) / (
            iterations + 1.0
        )
        log_point = float(point["mean_log_mae_ratio"])
        rows.append(
            {
                "arm": arm,
                "model_equal_cell_mean_mae": float(point["model_mean_mae"]),
                "model_mae_bootstrap_se": float(model_draws.std(ddof=1)),
                "model_mae_ci_95_lower": float(mae_lower),
                "model_mae_ci_95_upper": float(mae_upper),
                "persistence_equal_cell_mean_mae": float(point["persistence_mean_mae"]),
                "arithmetic_equal_cell_mean_improvement_percent": 100.0
                * (
                    1.0
                    - float(point["model_mean_mae"])
                    / float(point["persistence_mean_mae"])
                ),
                "mean_log_mae_ratio": log_point,
                "geometric_mae_ratio": float(math.exp(log_point)),
                "geometric_cell_ratio_improvement_percent": 100.0
                * (1.0 - math.exp(log_point)),
                "geometric_improvement_ci_95_lower_percent": 100.0
                * (1.0 - math.exp(float(log_upper))),
                "geometric_improvement_ci_95_upper_percent": 100.0
                * (1.0 - math.exp(float(log_lower))),
                "log_ratio_bootstrap_se": float(log_draws.std(ddof=1)),
                "log_ratio_ci_95_lower": float(log_lower),
                "log_ratio_ci_95_upper": float(log_upper),
                "p_value_one_sided_sign_tail": float(p_negative),
                "p_value_two_sided_sign_tail": float(
                    min(1.0, 2.0 * min(p_negative, p_positive))
                ),
                "consistent_seed_count": int(point["consistent_seed_count"]),
                "consistent_fold_count": int(point["consistent_fold_count"]),
                "seed_count": len(CANONICAL_SEEDS),
                "fold_count": len(CANONICAL_FOLDS),
                "pair_count": int(point["pair_count"]),
                "session_count": int(point["session_count"]),
                "bootstrap_iterations": iterations,
                "p_value_interpretation": str(bootstrap["p_value_interpretation"]),
                "resampling_method": str(bootstrap["resampling_method"]),
                "seed_log_mae_ratios_json": json.dumps(
                    point["seed_log_ratios"], sort_keys=True, separators=(",", ":")
                ),
                "fold_log_mae_ratios_json": json.dumps(
                    point["fold_log_ratios"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    result = pd.DataFrame(rows)
    adjusted = unified.holm_adjust(
        dict(
            zip(
                result["arm"].astype(str),
                result["p_value_one_sided_sign_tail"].astype(float),
                strict=True,
            )
        )
    )
    result["holm5_adjusted_p"] = result["arm"].map(adjusted)
    result["holm_family_size"] = len(CANONICAL_ARMS)
    result["passes_retrospective_support_gate"] = (
        result["mean_log_mae_ratio"].lt(0.0)
        & result["log_ratio_ci_95_upper"].lt(0.0)
        & result["holm5_adjusted_p"].lt(float(bootstrap["holm_alpha"]))
        & result["consistent_seed_count"].ge(int(bootstrap["minimum_nonworse_seeds"]))
        & result["consistent_fold_count"].ge(int(bootstrap["minimum_nonworse_folds"]))
    )
    result["mae_rank"] = (
        result["model_equal_cell_mean_mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    order = {arm: index for index, arm in enumerate(CANONICAL_ARMS)}
    result["_order"] = result["arm"].map(order)
    return (
        result.sort_values(["mae_rank", "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    rows = []
    for values in frame[list(columns)].itertuples(index=False, name=None):
        rendered = []
        for value in values:
            rendered.append(
                f"{float(value):.10g}" if isinstance(value, float) else str(value)
            )
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *rows,
        ]
    )


def _reports(summary: pd.DataFrame) -> tuple[str, str]:
    columns = (
        "mae_rank",
        "arm",
        "model_equal_cell_mean_mae",
        "model_mae_ci_95_lower",
        "model_mae_ci_95_upper",
        "arithmetic_equal_cell_mean_improvement_percent",
        "geometric_cell_ratio_improvement_percent",
        "geometric_improvement_ci_95_lower_percent",
        "geometric_improvement_ci_95_upper_percent",
        "p_value_one_sided_sign_tail",
        "holm5_adjusted_p",
        "passes_retrospective_support_gate",
    )
    table = _markdown_table(summary, columns)
    markdown = f"""# Model-wise 10,000-draw bootstrap vs Persistence

Each of the five models uses all ten seeds and four rolling folds.  The five
models were evaluated in parallel with a shared resampling schedule.  Each
bootstrap draw resamples seed, then fold, then CME-session clusters; every pair
inside a selected session is retained before cell means are recomputed.

{table}

`arithmetic_equal_cell_mean_improvement_percent` is the ratio of the two raw
equal-cell mean MAEs.  The bootstrap estimand and its confidence interval are
instead based on the mean cell log-ratio; its transformed point and interval are
reported in the `geometric_*` columns.  These are different estimands and must
not be interchanged.

The p-values are approximate bootstrap sign-tail probabilities, not exact
null-centered tests.  Holm-5 treats the five model-vs-persistence rows as one
family.  With 10,000 draws the minimum tail probability is 1/10,001 and the
Monte Carlo SE near p=0.05 is about 0.0022.

The historical nested protocol independently resamples the same market panel
inside each model seed.  It is retained for comparability but may yield narrower
uncertainty than a cross-classified market-session analysis.  Results are
`retrospective_rolling_development`, not confirmatory evidence.
"""
    headings = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for values in summary[list(columns)].itertuples(index=False, name=None):
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(f'{value:.10g}' if isinstance(value, float) else str(value))}</td>"
                for value in values
            )
            + "</tr>"
        )
    html_report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Model-wise bootstrap</title><style>body{{font-family:system-ui,sans-serif;max-width:1800px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:.78rem}}th,td{{border:1px solid #ccd4dc;padding:.38rem;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#eef3f7}}</style></head><body>
<h1>Model-wise 10,000-draw bootstrap vs Persistence</h1>
<p>Five models, all ten seeds and four rolling folds; shared seed → fold → paired CME-session resampling schedule.</p>
<table><thead><tr>{headings}</tr></thead><tbody>{"".join(body)}</tbody></table>
<p>Arithmetic mean-MAE improvement and geometric cell-log-ratio improvement are different estimands. Confidence intervals shown for the geometric bootstrap estimand must not be applied to the arithmetic column.</p>
<p>Approximate bootstrap sign-tail probabilities; Holm-5. Nested historical resampling can understate cross-seed market-panel dependence. Retrospective development evidence only.</p>
</body></html>"""
    return markdown, html_report


def _input_manifest(
    config: Mapping[str, Any], chunk_plan: pd.DataFrame
) -> dict[str, Any]:
    rows = _source_rows(config)
    profile = {
        "seed_entropy": list(map(int, config["bootstrap"]["seed_entropy"])),
        "shared_resampling_schedule": True,
        "chunk_seeds": chunk_plan.drop_duplicates("chunk_index")
        .sort_values("chunk_index")[
            ["chunk_index", "rng_seed", "seed_sequence_spawn_key_json"]
        ]
        .to_dict(orient="records"),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "modelwise_bootstrap_input_manifest_v1",
        "source_artifacts": rows,
        "shared_resampling_profile": profile,
        "shared_resampling_profile_sha256": payload_sha256(profile),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _analysis_manifest(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [
        SUMMARY_RELATIVE,
        DRAWS_RELATIVE,
        RESOURCE_RELATIVE,
        CHUNK_PLAN_RELATIVE,
        REPORT_MD_RELATIVE,
        REPORT_HTML_RELATIVE,
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "modelwise_bootstrap_analysis_manifest_v1",
        "interpretation": str(config["interpretation"]),
        "confirmatory": False,
        "arm_count": len(CANONICAL_ARMS),
        "iterations_per_arm": int(config["bootstrap"]["iterations_per_arm"]),
        "total_draws": len(CANONICAL_ARMS)
        * int(config["bootstrap"]["iterations_per_arm"]),
        "worker_tasks": len(CANONICAL_ARMS)
        * int(config["bootstrap"]["chunks_per_arm"]),
        "max_workers": int(config["bootstrap"]["max_workers"]),
        "holm_family_size": len(CANONICAL_ARMS),
        "artifacts": [
            _file_row(f"analysis:{path.name}", root / path, relative_to=root)
            for path in artifacts
        ],
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _qa_payload(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = pd.read_csv(root / SUMMARY_RELATIVE)
    draws = pd.read_csv(root / DRAWS_RELATIVE, low_memory=False)
    chunk_plan = pd.read_csv(root / CHUNK_PLAN_RELATIVE)
    expected = int(config["bootstrap"]["iterations_per_arm"])
    if (
        len(summary) != len(CANONICAL_ARMS)
        or set(summary["arm"]) != set(CANONICAL_ARMS)
        or len(draws) != len(CANONICAL_ARMS) * expected
        or draws.groupby("arm").size().to_dict()
        != {arm: expected for arm in CANONICAL_ARMS}
        or len(chunk_plan) != 50
        or chunk_plan["task_id"].duplicated().any()
        or not np.isfinite(
            draws[
                ["model_mean_mae", "persistence_mean_mae", "mean_log_mae_ratio"]
            ].to_numpy()
        ).all()
    ):
        raise BootstrapError("Terminal bootstrap row/count/finiteness QA failed")
    reference = None
    for arm in CANONICAL_ARMS:
        values = (
            draws[draws["arm"].eq(arm)]
            .sort_values("draw_index")["persistence_mean_mae"]
            .to_numpy(float)
        )
        if reference is None:
            reference = values
        elif not np.allclose(values, reference, rtol=0.0, atol=1e-15):
            raise BootstrapError(
                "Shared resampling persistence draws differ across arms"
            )
    if (
        not (summary["holm_family_size"] == 5).all()
        or not summary["holm5_adjusted_p"].between(0, 1).all()
        or not (summary["bootstrap_iterations"] == expected).all()
    ):
        raise BootstrapError("Holm/iteration QA failed")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "modelwise_bootstrap_terminal_qa_v1",
        "status": "passed",
        "source_root_immutable": True,
        "arm_count": 5,
        "iterations_per_arm": expected,
        "total_draws": len(draws),
        "chunk_tasks": len(chunk_plan),
        "max_workers": int(config["bootstrap"]["max_workers"]),
        "shared_resampling_schedule": True,
        "holm_family_size": 5,
        "p_value_interpretation": str(config["bootstrap"]["p_value_interpretation"]),
        "interpretation": str(config["interpretation"]),
        "confirmatory": False,
        "input_manifest_sha256": sha256_file(root / INPUT_MANIFEST_RELATIVE),
        "analysis_manifest_sha256": sha256_file(root / ANALYSIS_MANIFEST_RELATIVE),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _output_paths() -> list[Path]:
    return [
        RESOLVED_CONFIG_RELATIVE,
        INPUT_MANIFEST_RELATIVE,
        ANALYSIS_MANIFEST_RELATIVE,
        QA_RELATIVE,
        SUMMARY_RELATIVE,
        DRAWS_RELATIVE,
        RESOURCE_RELATIVE,
        CHUNK_PLAN_RELATIVE,
        REPORT_MD_RELATIVE,
        REPORT_HTML_RELATIVE,
    ]


def _write_output_hashes(root: Path) -> Path:
    rows = [
        _file_row(f"output:{path.as_posix()}", root / path, relative_to=root)
        for path in _output_paths()
    ]
    return _write_csv(root / OUTPUT_HASHES_RELATIVE, pd.DataFrame(rows))


def verify_root(root_or_path: str | Path) -> Path:
    root = resolve_path(root_or_path)
    required = [*_output_paths(), OUTPUT_HASHES_RELATIVE]
    missing = [str(root / path) for path in required if not (root / path).is_file()]
    if missing:
        raise BootstrapError(f"Terminal root is incomplete: {missing}")
    config_raw = yaml.safe_load((root / RESOLVED_CONFIG_RELATIVE).read_text("utf-8"))
    config = _mapping(config_raw, "resolved config")
    config["_config_path"] = str(DEFAULT_CONFIG.resolve())
    config["_config_sha256"] = sha256_file(DEFAULT_CONFIG)
    validate_config(config)
    input_manifest = _read_json(root / INPUT_MANIFEST_RELATIVE)
    unsigned_input = {
        key: value for key, value in input_manifest.items() if key != "payload_sha256"
    }
    if input_manifest.get(
        "kind"
    ) != "modelwise_bootstrap_input_manifest_v1" or payload_sha256(
        unsigned_input
    ) != input_manifest.get("payload_sha256"):
        raise BootstrapError("Input manifest signature drift")
    for row in input_manifest.get("source_artifacts") or []:
        _verify_file_row(row)
    analysis = _read_json(root / ANALYSIS_MANIFEST_RELATIVE)
    unsigned_analysis = {
        key: value for key, value in analysis.items() if key != "payload_sha256"
    }
    if analysis.get(
        "kind"
    ) != "modelwise_bootstrap_analysis_manifest_v1" or payload_sha256(
        unsigned_analysis
    ) != analysis.get("payload_sha256"):
        raise BootstrapError("Analysis manifest signature drift")
    for row in analysis.get("artifacts") or []:
        _verify_file_row(row, root=root)
    qa = _read_json(root / QA_RELATIVE)
    unsigned_qa = {key: value for key, value in qa.items() if key != "payload_sha256"}
    if qa.get("status") != "passed" or payload_sha256(unsigned_qa) != qa.get(
        "payload_sha256"
    ):
        raise BootstrapError("QA signature/status drift")
    hashes = pd.read_csv(
        root / OUTPUT_HASHES_RELATIVE, dtype=str, keep_default_na=False
    )
    if len(hashes) != len(_output_paths()):
        raise BootstrapError("Output hash manifest count drift")
    for row in hashes.to_dict(orient="records"):
        _verify_file_row(row, root=root)
    _qa_payload(root, config)
    return root


def _journal(control: Path, stage: str, status: str, **details: Any) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "modelwise_bootstrap_pipeline_journal_v1",
        "pid": os.getpid(),
        "stage": stage,
        "status": status,
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        **details,
    }
    return _write_json(control / "pipeline_journal.json", payload)


@contextmanager
def _pipeline_lock(root: Path) -> Iterator[Path]:
    control = Path(f"{root}_control")
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "pipeline.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise BootstrapError(
            f"Bootstrap supervisor already running: {lock_path}"
        ) from exc
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        _atomic_bytes(control / "pipeline.pid", f"{os.getpid()}\n".encode())
        yield control
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_analysis(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    config = load_config(config_path)
    root = resolve_path(output_dir)
    if root.exists():
        return verify_root(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with _pipeline_lock(root) as control:
        attempt = root.parent / f".{root.name}.attempt-{os.getpid()}"
        if attempt.exists():
            raise BootstrapError(f"Attempt directory already exists: {attempt}")
        attempt.mkdir(parents=True)
        try:
            _journal(control, "validate_source", "running")
            pairs = _validate_source(config)
            resolved = {
                key: value for key, value in config.items() if not key.startswith("_")
            }
            _atomic_bytes(
                attempt / RESOLVED_CONFIG_RELATIVE,
                yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True).encode(
                    "utf-8"
                ),
            )
            chunk_plan, chunk_seeds = _chunk_schedule(config)
            _write_csv(attempt / CHUNK_PLAN_RELATIVE, chunk_plan)
            input_manifest = _input_manifest(config, chunk_plan)
            _write_json(attempt / INPUT_MANIFEST_RELATIVE, input_manifest)
            payloads: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
            points: dict[str, dict[str, Any]] = {}
            for arm in CANONICAL_ARMS:
                payloads[arm], points[arm] = _cell_payloads(pairs, arm)
            tasks: list[dict[str, Any]] = []
            bootstrap = _mapping(config["bootstrap"], "bootstrap")
            for arm in CANONICAL_ARMS:
                for chunk_index, rng_seed in enumerate(chunk_seeds):
                    tasks.append(
                        {
                            "arm": arm,
                            "chunk_index": chunk_index,
                            "rng_seed": rng_seed,
                            "iterations": int(bootstrap["draws_per_chunk"]),
                            "cells": payloads[arm],
                        }
                    )
            if len(tasks) != 50:
                raise BootstrapError("Expected exactly 50 worker tasks")
            _journal(
                control, "bootstrap", "running", completed_chunks=0, total_chunks=50
            )
            context = mp.get_context(str(bootstrap["multiprocessing_start_method"]))
            results: list[dict[str, Any]] = []
            with ProcessPoolExecutor(
                max_workers=int(bootstrap["max_workers"]), mp_context=context
            ) as executor:
                futures = [executor.submit(_bootstrap_chunk, task) for task in tasks]
                for completed, future in enumerate(as_completed(futures), start=1):
                    results.append(future.result())
                    if completed % 5 == 0 or completed == len(futures):
                        _journal(
                            control,
                            "bootstrap",
                            "running",
                            completed_chunks=completed,
                            total_chunks=50,
                        )
            results.sort(
                key=lambda row: (
                    CANONICAL_ARMS.index(str(row["arm"])),
                    int(row["chunk_index"]),
                )
            )
            draw_frames: list[pd.DataFrame] = []
            resource_rows: list[dict[str, Any]] = []
            per_chunk = int(bootstrap["draws_per_chunk"])
            for result in results:
                chunk = int(result["chunk_index"])
                local = np.arange(per_chunk, dtype=int)
                draw_frames.append(
                    pd.DataFrame(
                        {
                            "arm": str(result["arm"]),
                            "chunk_index": chunk,
                            "local_draw_index": local,
                            "draw_index": chunk * per_chunk + local,
                            "rng_seed": int(result["rng_seed"]),
                            "model_mean_mae": result["model_draws"],
                            "persistence_mean_mae": result["persistence_draws"],
                            "mean_log_mae_ratio": result["log_draws"],
                        }
                    )
                )
                resource_rows.append(
                    {
                        "arm": str(result["arm"]),
                        "chunk_index": chunk,
                        "rng_seed": int(result["rng_seed"]),
                        "worker_pid": int(result["pid"]),
                        "duration_seconds": float(result["duration_seconds"]),
                        "peak_rss_kib": int(result["peak_rss_kib"]),
                        "draw_count": per_chunk,
                    }
                )
            draws = pd.concat(draw_frames, ignore_index=True).sort_values(
                ["arm", "draw_index"], kind="stable"
            )
            summary = _summaries(draws, points, config)
            _write_csv(attempt / DRAWS_RELATIVE, draws, gzip=True)
            _write_csv(attempt / SUMMARY_RELATIVE, summary)
            _write_csv(attempt / RESOURCE_RELATIVE, pd.DataFrame(resource_rows))
            markdown, html_report = _reports(summary)
            _atomic_bytes(attempt / REPORT_MD_RELATIVE, markdown.encode("utf-8"))
            _atomic_bytes(attempt / REPORT_HTML_RELATIVE, html_report.encode("utf-8"))
            analysis_manifest = _analysis_manifest(attempt, config)
            _write_json(attempt / ANALYSIS_MANIFEST_RELATIVE, analysis_manifest)
            qa = _qa_payload(attempt, config)
            _write_json(attempt / QA_RELATIVE, qa)
            _write_output_hashes(attempt)
            verify_root(attempt)
            os.replace(attempt, root)
            verify_root(root)
            _journal(
                control,
                "terminal",
                "completed",
                output_root=str(root),
                total_draws=50_000,
            )
            return root
        except BaseException as exc:
            _journal(
                control,
                "pipeline",
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                attempt_root=str(attempt),
            )
            raise


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = resolve_path(output_dir)
    control = Path(f"{root}_control")
    journal = control / "pipeline_journal.json"
    if root.exists():
        try:
            verify_root(root)
            return {"status": "completed", "stage": "terminal", "root": str(root)}
        except (BootstrapError, FileNotFoundError, ValueError) as exc:
            return {"status": "invalid", "root": str(root), "error": str(exc)}
    if journal.is_file():
        return _read_json(journal)
    return {"status": "absent", "root": str(root)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "qa"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        print(run_analysis(args.config, args.output_dir))
    elif args.action == "qa":
        print(verify_root(args.output_dir))
    else:
        print(json.dumps(status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
