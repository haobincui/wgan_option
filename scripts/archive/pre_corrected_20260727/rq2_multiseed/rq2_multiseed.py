#!/usr/bin/env python3
"""RQ2 multi-seed controlled experiment utilities.

The shell scripts in this directory provide stable entrypoints.  This helper
does the auditable work: input snapshotting, run registry collection,
checkpoint selection, sample-level metric reconstruction, and final tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:  # scipy is present in the py312 experiment environment.
    from scipy import stats
except Exception:  # pragma: no cover - fallback for lightweight inspection.
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_ROOT = REPO_ROOT / "outputs/experiments/rq2_multiseed_textbase_20260627"
ORIGINAL_TEXT_CONFIG = (
    REPO_ROOT
    / "outputs/training/film_wgan/svi-excel/20260417_131244/metrics/training_resolved_config.yaml"
)
FROZEN_CONFIG = REPO_ROOT / "configs/film_wgan/train_rq2_multiseed_textbase.yaml"
ENRICHED_WORKBOOK = REPO_ROOT / "data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx"
FEATURE_DIR = REPO_ROOT / "data/processed/text_features/rq2/20260625-075653"

DEFAULT_SEEDS = (42, 101, 202, 303, 404)
EXPECTED_SAMPLE_COUNT = 3711
TRAIN_COUNT = 2968
EVAL_COUNT = 743
AFTER10_SELECTION_METRIC = "val_mae_gap_vs_current"
SELECTED_CHECKPOINT_NAME = "film_wgan_best_val_mae_gap_vs_current.pt"

MODELS: dict[str, dict[str, Any]] = {
    "text": {"text_embedding_mode": "lp", "normalize_text_embedding": True},
    "no_text": {"text_embedding_mode": "none", "normalize_text_embedding": False},
    "bow": {"text_embedding_mode": "bow", "normalize_text_embedding": True},
    "llm_sentiment": {"text_embedding_mode": "llm_sentiment", "normalize_text_embedding": True},
}

MAIN_METRICS = ("surface_mae", "short_atm_mae", "atm7_abs_err")
ALL_METRICS = (
    "surface_mae",
    "surface_gap",
    "short_atm_mae",
    "short_atm_gap",
    "atm7_abs_err",
    "atm7_gap",
)
FULL_PAIRS = (
    ("text", "no_text"),
    ("text", "bow"),
    ("text", "llm_sentiment"),
    ("no_text", "bow"),
    ("no_text", "llm_sentiment"),
    ("bow", "llm_sentiment"),
)
TEXT_BASELINE_PAIRS = (
    ("text", "no_text"),
    ("text", "bow"),
    ("text", "llm_sentiment"),
)

SAME_MODEL_ALLOWED_DIFFS = {"seed", "output_root", "checkpoints_path", "metrics_path"}
SAME_SEED_ALLOWED_DIFFS = {
    "text_embedding_mode",
    "normalize_text_embedding",
    "output_root",
    "checkpoints_path",
    "metrics_path",
}


@dataclass(frozen=True)
class RunRecord:
    model: str
    seed: int
    run_dir: Path
    resolved_config: Path
    metrics_csv: Path
    checkpoint_path: Path


def _repo_rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return dict(payload)


def _write_yaml(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path, *, category: str, notes: str = "") -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return {
            "relative_path": _repo_rel(dst),
            "source_path": _repo_rel(src),
            "category": category,
            "size_bytes": dst.stat().st_size,
            "sha256": _sha256(dst),
            "notes": notes,
        }
    return {
        "relative_path": _repo_rel(dst),
        "source_path": _repo_rel(src),
        "category": category,
        "size_bytes": 0,
        "sha256": "",
        "notes": f"missing optional source; {notes}".strip("; "),
    }


def _run_git_command(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive only.
        return f"failed to run git {' '.join(args)}: {exc}"


def _atomic_write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _parse_seeds(seed_text: str | None) -> list[int]:
    if not seed_text:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def _path_arg(value: str | Path, default: Path) -> Path:
    text = str(value or "").strip()
    path = Path(text) if text else default
    return path if path.is_absolute() else REPO_ROOT / path


def _split_for_index(global_index: int, train_count: int) -> str:
    return "train" if int(global_index) < int(train_count) else "eval"


def _normalise_for_compare(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [_normalise_for_compare(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_for_compare(val) for key, val in sorted(value.items())}
    return value


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _config_diff(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    keys = set(left) | set(right)
    return {
        key
        for key in keys
        if _normalise_for_compare(left.get(key)) != _normalise_for_compare(right.get(key))
    }


def prepare_experiment(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root).resolve()
    original_config = _path_arg(getattr(args, "original_config", ""), ORIGINAL_TEXT_CONFIG)
    frozen_config = _path_arg(getattr(args, "frozen_config", ""), FROZEN_CONFIG)
    data_path = _path_arg(getattr(args, "data_path", ""), ENRICHED_WORKBOOK)
    feature_dir = _path_arg(getattr(args, "feature_dir", ""), FEATURE_DIR)
    manifest_rows: list[dict[str, Any]] = []

    for rel_dir in [
        "inputs/configs",
        "inputs/scripts_snapshot",
        "inputs/data",
        "inputs/text_features",
        "training_runs",
        "logs",
        "registry",
        "checkpoint_selection",
        "comparisons",
        "final_tables",
    ]:
        (exp_root / rel_dir).mkdir(parents=True, exist_ok=True)

    launch_registry = exp_root / "registry/launch_registry.csv"
    if not launch_registry.exists():
        launch_registry.write_text(
            "launch_ts_utc,model,seed,text_embedding_mode,normalize_text_embedding,"
            "output_root,work_dir,pid_file,log_file,pid,status\n",
            encoding="utf-8",
        )

    manifest_rows.append(
        _copy_file(
            original_config,
            exp_root / "inputs/configs/original_text_training_resolved_config.yaml",
            category="config",
            notes="Original LP text resolved config used as controlled-experiment base.",
        )
    )
    manifest_rows.append(
        _copy_file(
            frozen_config,
            exp_root / "inputs/configs" / frozen_config.name,
            category="config",
            notes="Frozen multi-seed textbase config.",
        )
    )
    manifest_rows.append(
        _copy_file(
            data_path,
            exp_root / "inputs/data" / data_path.name,
            category="data",
            notes="Shared enriched workbook for all four input representations.",
        )
    )

    feature_targets = [
        "bow_features.xlsx",
        "bow_manifest.json",
        "bow_vocabulary.json",
        "llm_sentiment_features.xlsx",
        "llm_sentiment_manifest.json",
        "openai_sentiment_cache.jsonl",
    ]
    for name in feature_targets:
        manifest_rows.append(
            _copy_file(
                feature_dir / name,
                exp_root / f"inputs/text_features/{name}",
                category="text_features",
                notes="RQ2 feature artifact snapshot.",
            )
        )

    snapshot_roots = [
        REPO_ROOT / "run_film_wgan_svi_excel.sh",
        REPO_ROOT / "scripts/film_wgan",
        REPO_ROOT / "scripts/rq2_multiseed",
        REPO_ROOT / "src/film_wgan",
        REPO_ROOT / "src/utils/standalone_cli.py",
        REPO_ROOT / "src/utils/output_paths.py",
        REPO_ROOT / "src/utils/training_paths.py",
        REPO_ROOT / "src/wgan_option/config_parsing.py",
    ]
    for src in snapshot_roots:
        dst = exp_root / "inputs/scripts_snapshot" / _repo_rel(src)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            manifest_rows.append(
                {
                    "relative_path": _repo_rel(dst),
                    "source_path": _repo_rel(src),
                    "category": "script_snapshot",
                    "size_bytes": sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()),
                    "sha256": "",
                    "notes": "Directory snapshot; per-file hashes are intentionally not expanded here.",
                }
            )
        else:
            manifest_rows.append(
                _copy_file(src, dst, category="script_snapshot", notes="Runnable code snapshot.")
            )

    git_state = exp_root / "inputs/git_state.txt"
    git_state.write_text(
        "\n".join(
            [
                f"captured_at_utc: {_now_utc()}",
                "",
                "[git rev-parse --abbrev-ref HEAD]",
                _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
                "",
                "[git rev-parse HEAD]",
                _run_git_command(["rev-parse", "HEAD"]),
                "",
                "[git status --short]",
                _run_git_command(["status", "--short"]),
                "",
                "[git diff --stat]",
                _run_git_command(["diff", "--stat"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_rows.append(
        {
            "relative_path": _repo_rel(git_state),
            "source_path": "git working tree",
            "category": "git_state",
            "size_bytes": git_state.stat().st_size,
            "sha256": _sha256(git_state),
            "notes": "Branch, HEAD, status, and diff summary at prepare time.",
        }
    )

    protocol = exp_root / "README.md"
    protocol.write_text(
        "\n".join(
            [
                f"# {getattr(args, 'experiment_title', 'RQ2 Multi-Seed Controlled Experiment')}",
                "",
                "This archive is prepared for the thesis-facing FiLM WGAN RQ2 multi-seed experiment.",
                "",
                "Controlled rule:",
                "",
                "```text",
                f"base config = {_repo_rel(original_config)}",
                f"training config = {_repo_rel(frozen_config)}",
                f"data workbook = {_repo_rel(data_path)}",
                "same model across seeds: seed-only change",
                "same seed across models: text representation-only change",
                "paper-facing checkpoint: epoch > 10, lowest val_mae_gap_vs_current",
                "```",
                "",
                "Models:",
                "",
                "```text",
                "text          = lp",
                "no_text       = none",
                "bow           = n-gram log-count BoW",
                "llm_sentiment = ChatGPT-style multidimensional sentiment scores",
                "```",
                "",
                "Run order:",
                "",
                "```bash",
                "bash scripts/rq2_multiseed/prepare_experiment.sh",
                "bash scripts/rq2_multiseed/start_training_matrix_background.sh",
                "bash scripts/rq2_multiseed/monitor_training.sh",
                "bash scripts/rq2_multiseed/collect_run_registry.sh",
                "bash scripts/rq2_multiseed/start_generate_matrix_background.sh",
                "bash scripts/rq2_multiseed/monitor_training.sh",
                "bash scripts/rq2_multiseed/build_comparison_archive.sh",
                "bash scripts/rq2_multiseed/package_experiment.sh",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_rows.append(
        {
            "relative_path": _repo_rel(protocol),
            "source_path": "generated",
            "category": "documentation",
            "size_bytes": protocol.stat().st_size,
            "sha256": _sha256(protocol),
            "notes": "Experiment protocol README.",
        }
    )

    manifest = pd.DataFrame(manifest_rows)
    _atomic_write_dataframe(manifest, exp_root / "inputs/data_manifest.csv")
    print(f"Prepared experiment root: {_repo_rel(exp_root)}")
    print(f"Input manifest: {_repo_rel(exp_root / 'inputs/data_manifest.csv')}")


def _find_run_dir(exp_root: Path, model: str, seed: int) -> Path | None:
    seed_root = exp_root / "training_runs" / model / f"seed_{seed}"
    if not seed_root.exists():
        return None
    candidates = [
        path
        for path in seed_root.iterdir()
        if path.is_dir() and (path / "metrics/training_resolved_config.yaml").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def _validate_run_config(path: Path, model: str, seed: int, data_path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing resolved config"
    config = _read_yaml(path)
    expected = MODELS[model]
    problems = []
    if int(config.get("seed", -1)) != int(seed):
        problems.append(f"seed={config.get('seed')} expected {seed}")
    actual_data_path = Path(str(config.get("data_path", "")))
    if not actual_data_path.is_absolute():
        actual_data_path = REPO_ROOT / actual_data_path
    if actual_data_path.resolve() != data_path.resolve():
        problems.append(f"data_path={config.get('data_path')} expected {_repo_rel(data_path)}")
    if str(config.get("text_embedding_mode")) != expected["text_embedding_mode"]:
        problems.append(
            f"text_embedding_mode={config.get('text_embedding_mode')} expected {expected['text_embedding_mode']}"
        )
    if bool(config.get("normalize_text_embedding")) != bool(expected["normalize_text_embedding"]):
        problems.append(
            "normalize_text_embedding="
            f"{config.get('normalize_text_embedding')} expected {expected['normalize_text_embedding']}"
        )
    return not problems, "; ".join(problems)


def _ranking_for_metrics(metrics_csv: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_csv)
    if "epoch" not in metrics or AFTER10_SELECTION_METRIC not in metrics:
        return pd.DataFrame()
    after10 = metrics[pd.to_numeric(metrics["epoch"], errors="coerce") > 10].copy()
    after10[AFTER10_SELECTION_METRIC] = pd.to_numeric(
        after10[AFTER10_SELECTION_METRIC], errors="coerce"
    )
    after10 = after10.dropna(subset=[AFTER10_SELECTION_METRIC])
    if after10.empty:
        return pd.DataFrame()
    after10 = after10.sort_values([AFTER10_SELECTION_METRIC, "epoch"], ascending=[True, True])
    after10.insert(0, "rank_after10", range(1, len(after10) + 1))
    return after10


def collect_run_registry(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root).resolve()
    seeds = _parse_seeds(args.seeds)
    data_path = _path_arg(getattr(args, "data_path", ""), ENRICHED_WORKBOOK)
    run_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for model in MODELS:
        for seed in seeds:
            run_dir = _find_run_dir(exp_root, model, seed)
            if run_dir is None:
                run_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "run_dir": "",
                        "resolved_config": "",
                        "metrics_csv": "",
                        "checkpoint_path": "",
                        "status": "missing_run_dir",
                        "notes": f"No completed run found under training_runs/{model}/seed_{seed}.",
                    }
                )
                selected_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "run_dir": "",
                        "selected_epoch": "",
                        "selection_metric": AFTER10_SELECTION_METRIC,
                        "selection_metric_value": "",
                        "checkpoint_path": "",
                        "checkpoint_exists": False,
                        "selected_epoch_gt_10": False,
                        "status": "missing_run_dir",
                        "notes": "Run has not been collected yet.",
                    }
                )
                continue

            resolved_config = run_dir / "metrics/training_resolved_config.yaml"
            metrics_csv = run_dir / "metrics/training_metrics.csv"
            checkpoint_path = run_dir / "checkpoints" / SELECTED_CHECKPOINT_NAME
            config_ok, config_notes = _validate_run_config(resolved_config, model, seed, data_path)
            status = "ok" if config_ok and metrics_csv.exists() else "invalid"
            run_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_dir": _repo_rel(run_dir),
                    "resolved_config": _repo_rel(resolved_config),
                    "metrics_csv": _repo_rel(metrics_csv),
                    "checkpoint_path": _repo_rel(checkpoint_path),
                    "status": status,
                    "notes": config_notes,
                }
            )

            ranking = _ranking_for_metrics(metrics_csv)
            if ranking.empty:
                selected_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "run_dir": _repo_rel(run_dir),
                        "selected_epoch": "",
                        "selection_metric": AFTER10_SELECTION_METRIC,
                        "selection_metric_value": "",
                        "checkpoint_path": _repo_rel(checkpoint_path),
                        "checkpoint_exists": checkpoint_path.exists(),
                        "selected_epoch_gt_10": False,
                        "status": "missing_after10_ranking",
                        "notes": "No epoch > 10 row with val_mae_gap_vs_current was available.",
                    }
                )
                continue

            for _, row in ranking.iterrows():
                keep_cols = [
                    "epoch",
                    AFTER10_SELECTION_METRIC,
                    "val_short_atm_mae_gap_vs_current",
                    "val_atm_short_pure_mae_gap_vs_current",
                    "val_mae",
                    "val_current_mae",
                ]
                out = {
                    "model": model,
                    "seed": seed,
                    "run_dir": _repo_rel(run_dir),
                    "rank_after10": int(row["rank_after10"]),
                }
                for col in keep_cols:
                    out[col] = row[col] if col in ranking.columns else ""
                ranking_rows.append(out)

            best = ranking.iloc[0]
            selected_epoch = int(best["epoch"])
            selected_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_dir": _repo_rel(run_dir),
                    "selected_epoch": selected_epoch,
                    "selection_metric": AFTER10_SELECTION_METRIC,
                    "selection_metric_value": float(best[AFTER10_SELECTION_METRIC]),
                    "checkpoint_path": _repo_rel(checkpoint_path),
                    "checkpoint_exists": checkpoint_path.exists(),
                    "selected_epoch_gt_10": selected_epoch > 10,
                    "status": "ok" if checkpoint_path.exists() and selected_epoch > 10 else "invalid",
                    "notes": config_notes,
                }
            )

    _atomic_write_dataframe(pd.DataFrame(run_rows), exp_root / "registry/run_registry.csv")
    _atomic_write_dataframe(
        pd.DataFrame(ranking_rows), exp_root / "checkpoint_selection/checkpoint_after10_rankings.csv"
    )
    _atomic_write_dataframe(
        pd.DataFrame(selected_rows), exp_root / "checkpoint_selection/selected_checkpoints.csv"
    )
    print(f"Run registry: {_repo_rel(exp_root / 'registry/run_registry.csv')}")
    print(f"Selected checkpoints: {_repo_rel(exp_root / 'checkpoint_selection/selected_checkpoints.csv')}")


def _load_run_records(exp_root: Path) -> list[RunRecord]:
    selected_path = exp_root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected checkpoint registry: {selected_path}")
    selected = pd.read_csv(selected_path)
    records: list[RunRecord] = []
    for _, row in selected.iterrows():
        if str(row.get("status")) != "ok":
            continue
        run_dir = REPO_ROOT / str(row["run_dir"])
        records.append(
            RunRecord(
                model=str(row["model"]),
                seed=int(row["seed"]),
                run_dir=run_dir,
                resolved_config=run_dir / "metrics/training_resolved_config.yaml",
                metrics_csv=run_dir / "metrics/training_metrics.csv",
                checkpoint_path=run_dir / "checkpoints" / SELECTED_CHECKPOINT_NAME,
            )
        )
    return records


def _infer_expected_sample_count(records: list[RunRecord]) -> int:
    if not records:
        return EXPECTED_SAMPLE_COUNT
    sample_dir = records[0].run_dir / "after10_val_mae_all_json/samples"
    if not sample_dir.exists():
        return EXPECTED_SAMPLE_COUNT
    return len(sorted(sample_dir.glob("*.json")))


def _sample_metrics_from_json(path: Path, *, model: str, seed: int, train_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated = np.asarray(payload["generated_surface"], dtype=float)
    target = np.asarray(payload.get("target_surface", payload.get("real_surface")), dtype=float)
    current = np.asarray(payload["current_surface"], dtype=float)
    strikes = np.asarray(payload["strike_grid"], dtype=float)
    maturities = np.asarray(payload["maturity_days_grid"], dtype=float)

    if generated.shape != target.shape or generated.shape != current.shape:
        raise ValueError(f"Surface shapes do not match in {path}")
    if generated.ndim != 2:
        raise ValueError(f"Expected 2D surface in {path}")

    abs_err = np.abs(generated - target)
    current_abs_err = np.abs(current - target)
    short_atm_mask = (np.abs(strikes[None, :] - 1.0) <= 0.06) & (maturities[:, None] <= 90.0)
    if short_atm_mask.shape != generated.shape:
        raise ValueError(f"Short-ATM mask shape mismatch for {path}")
    if not np.any(short_atm_mask):
        raise ValueError(f"Short-ATM mask is empty for {path}")

    maturity_idx = int(np.argmin(maturities))
    atm_idx = int(np.argmin(np.abs(strikes - 1.0)))
    model_atm7 = float(generated[maturity_idx, atm_idx])
    target_atm7 = float(target[maturity_idx, atm_idx])
    current_atm7 = float(current[maturity_idx, atm_idx])

    surface_mae = float(abs_err.mean())
    current_surface_mae = float(current_abs_err.mean())
    short_atm_mae = float(abs_err[short_atm_mask].mean())
    current_short_atm_mae = float(current_abs_err[short_atm_mask].mean())
    atm7_abs_err = abs(model_atm7 - target_atm7)
    current_atm7_abs_err = abs(current_atm7 - target_atm7)

    global_index = int(payload["global_index"])
    split = _split_for_index(global_index, train_count)
    return {
        "model": model,
        "seed": seed,
        "sample_id": payload.get("sample_id", ""),
        "global_index": global_index,
        "split": split,
        "source_json": _repo_rel(path),
        "surface_mae": surface_mae,
        "current_surface_mae": current_surface_mae,
        "surface_gap": surface_mae - current_surface_mae,
        "short_atm_mae": short_atm_mae,
        "current_short_atm_mae": current_short_atm_mae,
        "short_atm_gap": short_atm_mae - current_short_atm_mae,
        "atm7_abs_err": atm7_abs_err,
        "current_atm7_abs_err": current_atm7_abs_err,
        "atm7_gap": atm7_abs_err - current_atm7_abs_err,
        "generated_7d_atm_vol": model_atm7,
        "target_7d_atm_vol": target_atm7,
        "current_7d_atm_vol": current_atm7,
        "nearest_atm_strike": float(strikes[atm_idx]),
        "shortest_maturity_days": float(maturities[maturity_idx]),
    }


def _collect_sample_metrics(
    records: list[RunRecord],
    *,
    expected_sample_count: int,
    train_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        sample_dir = record.run_dir / "after10_val_mae_all_json/samples"
        if not sample_dir.exists():
            raise FileNotFoundError(f"Missing generate-result samples: {sample_dir}")
        json_paths = sorted(sample_dir.glob("*.json"))
        if len(json_paths) != expected_sample_count:
            raise ValueError(
                f"{record.model} seed {record.seed} has {len(json_paths)} JSON samples; "
                f"expected {expected_sample_count}: {sample_dir}"
            )
        rows.extend(
            _sample_metrics_from_json(path, model=record.model, seed=record.seed, train_count=train_count)
            for path in json_paths
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["model", "seed", "global_index", "sample_id"]).reset_index(drop=True)
    return frame


def _validate_config_consistency(records: list[RunRecord], exp_root: Path) -> pd.DataFrame:
    configs: dict[tuple[str, int], dict[str, Any]] = {
        (record.model, record.seed): _read_yaml(record.resolved_config) for record in records
    }
    failures: list[dict[str, Any]] = []
    seeds = sorted({seed for _, seed in configs})

    for model in MODELS:
        model_keys = [(m, seed) for (m, seed) in configs if m == model]
        if len(model_keys) < 2:
            continue
        base_key = sorted(model_keys, key=lambda item: item[1])[0]
        base = configs[base_key]
        for key in sorted(model_keys, key=lambda item: item[1])[1:]:
            diff = _config_diff(base, configs[key])
            unexpected = sorted(diff - SAME_MODEL_ALLOWED_DIFFS)
            if unexpected:
                failures.append(
                    {
                        "comparison_type": "same_model_across_seeds",
                        "left": f"{base_key[0]}/seed_{base_key[1]}",
                        "right": f"{key[0]}/seed_{key[1]}",
                        "unexpected_diff_keys": ";".join(unexpected),
                        "allowed_diff_keys": ";".join(sorted(SAME_MODEL_ALLOWED_DIFFS)),
                    }
                )

    for seed in seeds:
        seed_keys = [(model, s) for (model, s) in configs if s == seed]
        if len(seed_keys) < 2:
            continue
        base_key = ("text", seed) if ("text", seed) in configs else sorted(seed_keys)[0]
        base = configs[base_key]
        for key in sorted(seed_keys):
            if key == base_key:
                continue
            diff = _config_diff(base, configs[key])
            unexpected = sorted(diff - SAME_SEED_ALLOWED_DIFFS)
            if unexpected:
                failures.append(
                    {
                        "comparison_type": "same_seed_across_models",
                        "left": f"{base_key[0]}/seed_{base_key[1]}",
                        "right": f"{key[0]}/seed_{key[1]}",
                        "unexpected_diff_keys": ";".join(unexpected),
                        "allowed_diff_keys": ";".join(sorted(SAME_SEED_ALLOWED_DIFFS)),
                    }
                )

    failure_frame = pd.DataFrame(failures)
    _atomic_write_dataframe(failure_frame, exp_root / "comparisons/config_validation_failures.csv")
    return failure_frame


def _metric_summary_by_seed(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "eval", "all"]:
        data = sample_metrics if split_name == "all" else sample_metrics[sample_metrics["split"] == split_name]
        for (model, seed), group in data.groupby(["model", "seed"], sort=True):
            row: dict[str, Any] = {
                "model": model,
                "seed": int(seed),
                "split": split_name,
                "n_samples": int(len(group)),
            }
            for metric in MAIN_METRICS:
                current_col = {
                    "surface_mae": "current_surface_mae",
                    "short_atm_mae": "current_short_atm_mae",
                    "atm7_abs_err": "current_atm7_abs_err",
                }[metric]
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{current_col}_mean"] = float(group[current_col].mean())
                row[f"{metric}_gap_mean"] = float((group[metric] - group[current_col]).mean())
                row[f"{metric}_win_rate_vs_current"] = float((group[metric] < group[current_col]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_vectors(
    sample_metrics: pd.DataFrame,
    *,
    left: str,
    right: str,
    seed: int,
    split_name: str,
    metric: str,
) -> pd.DataFrame:
    data = sample_metrics[sample_metrics["seed"] == seed]
    if split_name != "all":
        data = data[data["split"] == split_name]
    subset = data[data["model"].isin([left, right])][
        ["model", "sample_id", "global_index", metric]
    ].copy()
    pivot = subset.pivot_table(
        index=["global_index", "sample_id"], columns="model", values=metric, aggfunc="first"
    ).reset_index()
    if left not in pivot.columns or right not in pivot.columns:
        raise ValueError(f"Missing model in paired data: {left} vs {right}, seed {seed}, {split_name}, {metric}")
    pivot = pivot.dropna(subset=[left, right])
    return pivot


def _ttest_1samp(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return {
            "n": n,
            "mean": float(values.mean()) if n else math.nan,
            "std": math.nan,
            "t_stat": math.nan,
            "df": max(n - 1, 0),
            "p_two_sided": math.nan,
            "p_mean_gt_0": math.nan,
            "p_mean_lt_0": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    se = std / math.sqrt(n)
    if se == 0:
        t_stat = math.inf if mean > 0 else -math.inf if mean < 0 else 0.0
    else:
        t_stat = mean / se
    df = n - 1
    if stats is not None:
        p_two = float(stats.t.sf(abs(t_stat), df) * 2.0)
        p_gt = float(stats.t.sf(t_stat, df))
        p_lt = float(stats.t.cdf(t_stat, df))
        tcrit = float(stats.t.ppf(0.975, df))
    else:
        p_two = p_gt = p_lt = math.nan
        tcrit = math.nan
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t_stat": float(t_stat),
        "df": df,
        "p_two_sided": p_two,
        "p_mean_gt_0": p_gt,
        "p_mean_lt_0": p_lt,
        "ci95_low": mean - tcrit * se if math.isfinite(tcrit) else math.nan,
        "ci95_high": mean + tcrit * se if math.isfinite(tcrit) else math.nan,
    }


def _wilcoxon(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if stats is None or values.size < 1 or np.allclose(values, 0.0):
        return {
            "wilcoxon_p_two_sided": math.nan,
            "wilcoxon_p_mean_gt_0": math.nan,
            "wilcoxon_p_mean_lt_0": math.nan,
        }
    try:
        return {
            "wilcoxon_p_two_sided": float(stats.wilcoxon(values, alternative="two-sided").pvalue),
            "wilcoxon_p_mean_gt_0": float(stats.wilcoxon(values, alternative="greater").pvalue),
            "wilcoxon_p_mean_lt_0": float(stats.wilcoxon(values, alternative="less").pvalue),
        }
    except Exception:
        return {
            "wilcoxon_p_two_sided": math.nan,
            "wilcoxon_p_mean_gt_0": math.nan,
            "wilcoxon_p_mean_lt_0": math.nan,
        }


def _build_pairwise_outputs(sample_metrics: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed_rows: list[dict[str, Any]] = []
    sample_test_rows: list[dict[str, Any]] = []

    for left, right in FULL_PAIRS:
        for seed in seeds:
            for split_name in ["train", "eval", "all"]:
                for metric in MAIN_METRICS:
                    paired = _paired_vectors(
                        sample_metrics,
                        left=left,
                        right=right,
                        seed=seed,
                        split_name=split_name,
                        metric=metric,
                    )
                    if len(paired) == 0:
                        continue
                    right_minus_left = paired[right].to_numpy(dtype=float) - paired[left].to_numpy(dtype=float)
                    left_minus_right = -right_minus_left
                    test = _ttest_1samp(right_minus_left)
                    by_seed_rows.append(
                        {
                            "left_model": left,
                            "right_model": right,
                            "seed": seed,
                            "split": split_name,
                            "metric": metric,
                            "n_samples": int(len(paired)),
                            "left_mean": float(paired[left].mean()),
                            "right_mean": float(paired[right].mean()),
                            "right_minus_left_mean": float(right_minus_left.mean()),
                            "left_minus_right_mean": float(left_minus_right.mean()),
                            "left_better_sample_win_rate": float((right_minus_left > 0).mean()),
                            "right_better_sample_win_rate": float((right_minus_left < 0).mean()),
                        }
                    )
                    sample_test_rows.append(
                        {
                            "left_model": left,
                            "right_model": right,
                            "seed": seed,
                            "split": split_name,
                            "metric": metric,
                            "difference": "right_minus_left",
                            "interpretation_positive": "left model lower error / better",
                            "interpretation_negative": "right model lower error / better",
                            **test,
                        }
                    )

    by_seed = pd.DataFrame(by_seed_rows)
    sample_tests = pd.DataFrame(sample_test_rows)
    return by_seed, sample_tests


def _build_seed_level_tests(pairwise_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (left, right, split_name, metric), group in pairwise_by_seed.groupby(
        ["left_model", "right_model", "split", "metric"], sort=True
    ):
        values = group.sort_values("seed")["right_minus_left_mean"].to_numpy(dtype=float)
        test = _ttest_1samp(values)
        wilcoxon = _wilcoxon(values)
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "split": split_name,
                "metric": metric,
                "difference": "right_minus_left_by_seed",
                "interpretation_positive": "left model lower error / better",
                "interpretation_negative": "right model lower error / better",
                "seed_count": int(len(values)),
                "left_win_seed_count": int((values > 0).sum()),
                "right_win_seed_count": int((values < 0).sum()),
                "tie_seed_count": int((values == 0).sum()),
                "sample_win_rate_mean_by_seed_left": float(group["left_better_sample_win_rate"].mean()),
                "sample_win_rate_mean_by_seed_right": float(group["right_better_sample_win_rate"].mean()),
                **test,
                **wilcoxon,
            }
        )
    return pd.DataFrame(rows)


def _build_final_tables(summary_by_seed: pd.DataFrame, seed_tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_summary = summary_by_seed[summary_by_seed["split"] == "eval"].copy()
    rows: list[dict[str, Any]] = []
    for model, group in eval_summary.groupby("model", sort=True):
        for metric in MAIN_METRICS:
            rows.append(
                {
                    "model": model,
                    "split": "eval",
                    "metric": metric,
                    "mean_error_across_seeds": float(group[f"{metric}_mean"].mean()),
                    "std_error_across_seeds": float(group[f"{metric}_mean"].std(ddof=1)),
                    "min_seed_mean_error": float(group[f"{metric}_mean"].min()),
                    "max_seed_mean_error": float(group[f"{metric}_mean"].max()),
                    "mean_gap_vs_current_across_seeds": float(group[f"{metric}_gap_mean"].mean()),
                    "mean_win_rate_vs_current_across_seeds": float(
                        group[f"{metric}_win_rate_vs_current"].mean()
                    ),
                    "seed_count": int(group["seed"].nunique()),
                }
            )
    thesis_eval = pd.DataFrame(rows)

    text_vs = seed_tests[
        (seed_tests["split"] == "eval")
        & (seed_tests["metric"].isin(MAIN_METRICS))
        & seed_tests["left_model"].eq("text")
        & seed_tests["right_model"].isin(["no_text", "bow", "llm_sentiment"])
    ].copy()
    if not text_vs.empty:
        text_vs["paper_claim_direction"] = "right_minus_text > 0 means LP text has lower eval MAE"
        text_vs["text_better_supported_by_seed_mean"] = text_vs["mean"] > 0
        text_vs["text_better_one_sided_t_p"] = text_vs["p_mean_gt_0"]
        text_vs["text_better_one_sided_wilcoxon_p"] = text_vs["wilcoxon_p_mean_gt_0"]
    return thesis_eval, text_vs


def _build_best_seed_eval_tables(
    summary_by_seed: pd.DataFrame,
    sample_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_summary = summary_by_seed[summary_by_seed["split"] == "eval"].copy()
    best_rows: list[dict[str, Any]] = []
    best_lookup: dict[tuple[str, str], int] = {}
    for model, group in eval_summary.groupby("model", sort=True):
        for metric in MAIN_METRICS:
            ordered = group.sort_values([f"{metric}_mean", "seed"], ascending=[True, True])
            best = ordered.iloc[0]
            seed = int(best["seed"])
            best_lookup[(model, metric)] = seed
            best_rows.append(
                {
                    "model": model,
                    "split": "eval",
                    "metric": metric,
                    "best_seed": seed,
                    "best_seed_mean_error": float(best[f"{metric}_mean"]),
                    "best_seed_gap_vs_current": float(best[f"{metric}_gap_mean"]),
                    "best_seed_win_rate_vs_current": float(best[f"{metric}_win_rate_vs_current"]),
                    "n_samples": int(best["n_samples"]),
                    "selection_rule": "lowest eval mean MAE for this model and metric",
                }
            )

    pair_rows: list[dict[str, Any]] = []
    eval_samples = sample_metrics[sample_metrics["split"] == "eval"].copy()
    for baseline in ("no_text", "bow", "llm_sentiment"):
        for metric in MAIN_METRICS:
            text_seed = best_lookup.get(("text", metric))
            baseline_seed = best_lookup.get((baseline, metric))
            if text_seed is None or baseline_seed is None:
                continue
            subset = eval_samples[
                ((eval_samples["model"] == "text") & (eval_samples["seed"] == text_seed))
                | ((eval_samples["model"] == baseline) & (eval_samples["seed"] == baseline_seed))
            ][["model", "sample_id", "global_index", metric]].copy()
            pivot = subset.pivot_table(
                index=["global_index", "sample_id"], columns="model", values=metric, aggfunc="first"
            ).reset_index()
            if "text" not in pivot or baseline not in pivot:
                continue
            pivot = pivot.dropna(subset=["text", baseline])
            diff = pivot[baseline].to_numpy(dtype=float) - pivot["text"].to_numpy(dtype=float)
            test = _ttest_1samp(diff)
            pair_rows.append(
                {
                    "left_model": "text",
                    "right_model": baseline,
                    "split": "eval",
                    "metric": metric,
                    "text_best_seed": int(text_seed),
                    "baseline_best_seed": int(baseline_seed),
                    "difference": "baseline_minus_text",
                    "interpretation_positive": "LP text lower eval MAE / better",
                    "interpretation_negative": "baseline lower eval MAE / better",
                    "text_mean": float(pivot["text"].mean()),
                    "baseline_mean": float(pivot[baseline].mean()),
                    "baseline_minus_text_mean": float(diff.mean()) if len(diff) else math.nan,
                    **test,
                }
            )
    return pd.DataFrame(best_rows), pd.DataFrame(pair_rows)


def _write_manifest(exp_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(exp_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(exp_root)
            category = rel.parts[0] if rel.parts else "root"
            rows.append(
                {
                    "relative_path": str(rel),
                    "source_path": "generated in experiment archive",
                    "category": category,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path) if path.stat().st_size < 1024 * 1024 * 512 else "",
                    "notes": "sha256 omitted only for files >= 512 MiB",
                }
            )
    _atomic_write_dataframe(pd.DataFrame(rows), exp_root / "manifest.csv")


def build_comparison_archive(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root).resolve()
    seeds = _parse_seeds(args.seeds)
    records = _load_run_records(exp_root)
    if len(records) != len(MODELS) * len(seeds):
        raise SystemExit(
            f"Expected {len(MODELS) * len(seeds)} collected runs, found {len(records)}. "
            "Run collect_run_registry.sh after all training runs finish."
        )
    expected_sample_count = int(getattr(args, "expected_sample_count", 0) or 0)
    if expected_sample_count <= 0:
        expected_sample_count = _infer_expected_sample_count(records)
    train_count = int(getattr(args, "train_count", 0) or 0)
    if train_count <= 0:
        train_count = int(math.floor(expected_sample_count * 0.8))
    eval_count = max(expected_sample_count - train_count, 0)

    config_failures = _validate_config_consistency(records, exp_root)
    validation: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "expected_models": list(MODELS),
        "expected_seeds": seeds,
        "expected_run_count": len(MODELS) * len(seeds),
        "collected_run_count": len(records),
        "expected_sample_count": expected_sample_count,
        "train_count": train_count,
        "eval_count": eval_count,
        "config_consistency_ok": bool(config_failures.empty),
    }
    if not config_failures.empty:
        validation["status"] = "failed_config_consistency"
        _write_json(validation, exp_root / "validation_summary.json")
        raise SystemExit(
            "Config consistency validation failed. See "
            f"{_repo_rel(exp_root / 'comparisons/config_validation_failures.csv')}"
        )

    selected = pd.read_csv(exp_root / "checkpoint_selection/selected_checkpoints.csv")
    ok_selected = selected[selected["status"].eq("ok")].copy()
    validation["selected_checkpoints_ok"] = bool(
        len(ok_selected) == len(records)
        and _as_bool_series(ok_selected["checkpoint_exists"]).all()
        and _as_bool_series(ok_selected["selected_epoch_gt_10"]).all()
    )

    sample_metrics = _collect_sample_metrics(
        records,
        expected_sample_count=expected_sample_count,
        train_count=train_count,
    )
    _atomic_write_dataframe(
        sample_metrics, exp_root / "comparisons/sample_metrics_all_models_all_seeds.csv"
    )

    counts = (
        sample_metrics.groupby(["model", "seed", "split"], sort=True)
        .size()
        .reset_index(name="n_samples")
    )
    all_counts = (
        sample_metrics.groupby(["model", "seed"], sort=True)
        .size()
        .reset_index(name="n_samples")
    )
    validation["sample_count_by_model_seed"] = all_counts.to_dict(orient="records")
    validation["split_count_by_model_seed"] = counts.to_dict(orient="records")
    validation["sample_metrics_row_count"] = int(len(sample_metrics))
    validation["sample_metrics_expected_row_count"] = int(len(MODELS) * len(seeds) * expected_sample_count)

    summary_by_seed = _metric_summary_by_seed(sample_metrics)
    _atomic_write_dataframe(summary_by_seed, exp_root / "comparisons/model_metric_summary_by_seed.csv")

    pairwise_by_seed, sample_level_tests = _build_pairwise_outputs(sample_metrics, seeds)
    _atomic_write_dataframe(
        pairwise_by_seed, exp_root / "comparisons/pairwise_diff_summary_by_seed.csv"
    )
    _atomic_write_dataframe(
        sample_level_tests, exp_root / "comparisons/sample_level_tests_by_seed.csv"
    )

    seed_tests = _build_seed_level_tests(pairwise_by_seed)
    _atomic_write_dataframe(seed_tests, exp_root / "comparisons/seed_level_tests.csv")

    thesis_eval, thesis_text_vs = _build_final_tables(summary_by_seed, seed_tests)
    _atomic_write_dataframe(thesis_eval, exp_root / "final_tables/thesis_eval_main_metrics.csv")
    _atomic_write_dataframe(thesis_text_vs, exp_root / "final_tables/thesis_text_vs_baselines.csv")
    best_seed_by_metric, best_seed_pairs = _build_best_seed_eval_tables(summary_by_seed, sample_metrics)
    _atomic_write_dataframe(best_seed_by_metric, exp_root / "final_tables/best_seed_by_model_metric_eval.csv")
    _atomic_write_dataframe(
        best_seed_pairs,
        exp_root / "final_tables/best_seed_pairwise_text_vs_baselines_eval_p_values.csv",
    )

    validation["pairwise_seed_test_rows"] = int(len(seed_tests))
    validation["expected_pairwise_seed_test_rows"] = len(FULL_PAIRS) * 3 * len(MAIN_METRICS)
    validation["sample_level_test_rows"] = int(len(sample_level_tests))
    validation["expected_sample_level_test_rows"] = len(FULL_PAIRS) * len(seeds) * 3 * len(MAIN_METRICS)
    validation["final_eval_rows"] = int(len(thesis_eval))
    validation["final_text_vs_rows"] = int(len(thesis_text_vs))
    validation["best_seed_rows"] = int(len(best_seed_by_metric))
    validation["best_seed_pair_rows"] = int(len(best_seed_pairs))
    validation["status"] = "ok"
    _write_json(validation, exp_root / "validation_summary.json")

    _write_manifest(exp_root)
    print(f"Comparison archive built under: {_repo_rel(exp_root)}")
    print(f"Final eval table: {_repo_rel(exp_root / 'final_tables/thesis_eval_main_metrics.csv')}")
    print(f"Text-vs-baselines table: {_repo_rel(exp_root / 'final_tables/thesis_text_vs_baselines.csv')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_help = f"default: {_repo_rel(DEFAULT_EXP_ROOT)}"
    prepare = subparsers.add_parser("prepare", help="Create experiment root and snapshot inputs.")
    prepare.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT), help=common_help)
    prepare.add_argument("--original-config", default=str(ORIGINAL_TEXT_CONFIG))
    prepare.add_argument("--frozen-config", default=str(FROZEN_CONFIG))
    prepare.add_argument("--data-path", default=str(ENRICHED_WORKBOOK))
    prepare.add_argument("--feature-dir", default=str(FEATURE_DIR))
    prepare.add_argument("--experiment-title", default="RQ2 Multi-Seed Controlled Experiment")
    prepare.set_defaults(func=prepare_experiment)

    collect = subparsers.add_parser("collect", help="Collect run registry and after-10 checkpoint rankings.")
    collect.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT), help=common_help)
    collect.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    collect.add_argument("--data-path", default=str(ENRICHED_WORKBOOK))
    collect.set_defaults(func=collect_run_registry)

    build = subparsers.add_parser("build-comparison", help="Build sample metrics, tests, and final tables.")
    build.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT), help=common_help)
    build.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    build.add_argument("--expected-sample-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    build.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    build.set_defaults(func=build_comparison_archive)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
