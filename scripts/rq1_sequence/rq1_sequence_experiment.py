#!/usr/bin/env python3
"""Orchestrate the raw-vol RQ1 intraday rolling-sequence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from film_wgan.text_transform import sha256_file  # noqa: E402
from scripts.rq1_sequence.dataset_builder import (  # noqa: E402
    FOLDS,
    build_sequence_datasets,
)


EXPERIMENT_PREFIX = "rq1_intraday_sequence_raw_vol_"
DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq1_sequence_textbase.yaml"
DEFAULT_SOURCE_ROOT = Path(
    "/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone"
)
DEFAULT_DATABASE = (
    DEFAULT_SOURCE_ROOT
    / "data/processed/raw-market-index/raw_2022_2023_london_itm5_relaxed_v2"
    / "market_surface_index.sqlite"
)
DEFAULT_NEWS_WORKBOOK = (
    DEFAULT_SOURCE_ROOT
    / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
)
DEFAULT_SESSION_CALENDAR = (
    ROOT / "data/reference/cme_treasury_globex_closures_2022_2023.csv"
)
DEFAULT_SEEDS = (42, 202, 404)

PARENT_VARIANT = "history_no_text_parent"
CONTINUED_VARIANT = "history_no_text_continued"
CURRENT_ONLY_VARIANT = "current_only_no_text"
COUNT_VARIANT = "history_news_count"
MATCHED_VARIANT = "history_matched_lp"
SHUFFLED_VARIANT = "history_shuffled_lp"
VARIANTS = (
    PARENT_VARIANT,
    CONTINUED_VARIANT,
    CURRENT_ONLY_VARIANT,
    COUNT_VARIANT,
    MATCHED_VARIANT,
    SHUFFLED_VARIANT,
)
STAGE_B_VARIANTS = (
    CONTINUED_VARIANT,
    COUNT_VARIANT,
    MATCHED_VARIANT,
    SHUFFLED_VARIANT,
)
VARIANT_STAGE = {
    PARENT_VARIANT: "stage_a_parent",
    CONTINUED_VARIANT: "stage_b_continued_control",
    CURRENT_ONLY_VARIANT: "single_stage_current_only_control",
    COUNT_VARIANT: "stage_b_news_arrival_control",
    MATCHED_VARIANT: "stage_b_lp_treatment",
    SHUFFLED_VARIANT: "stage_b_shuffled_placebo",
}

POINT_METRICS = ("surface_mae", "short_atm_mae")
PROBABILISTIC_METRICS = (
    "energy_score",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "interval_width_50",
    "interval_width_80",
    "interval_width_90",
    "calibration_error",
    "scenario_spread",
)
FINANCIAL_METRICS = (
    "calendar_violation_rate",
    "butterfly_violation_rate",
)
INFERENTIAL_METRICS = (
    *POINT_METRICS,
    "energy_score",
    "calibration_error",
    *FINANCIAL_METRICS,
)
CONTRASTS = (
    (MATCHED_VARIANT, CONTINUED_VARIANT, "incremental_lp_text"),
    (PARENT_VARIANT, CURRENT_ONLY_VARIANT, "history_vs_current_only"),
    (MATCHED_VARIANT, COUNT_VARIANT, "lp_semantics_vs_news_arrival"),
    (MATCHED_VARIANT, SHUFFLED_VARIANT, "matched_vs_shuffled_lp"),
)
CURRENT_ERROR_COLUMNS = {
    "surface_mae": "current_mae",
    "short_atm_mae": "current_atm_short_pure_mae",
}


def _contrast_family(contrast: str) -> str:
    if contrast == "incremental_lp_text":
        return "primary_incremental_text"
    if contrast.endswith("_vs_persistence"):
        return "persistence_context"
    return "control_ablations"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _assert_py312() -> None:
    if Path(sys.prefix).name != "py312":
        raise RuntimeError(
            "RQ1 sequence workflow must run in conda environment 'py312'; "
            f"current prefix={sys.prefix}."
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _normalize_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be a non-empty unique list: {seeds}")
    if any(value < 0 or value >= 2**31 for value in seeds):
        raise ValueError("Seeds must be integers in [0, 2**31).")
    return seeds


def _latest_experiment() -> Path:
    candidates = sorted((ROOT / "outputs/experiments").glob(f"{EXPERIMENT_PREFIX}*"))
    if not candidates:
        raise FileNotFoundError("No prepared RQ1 sequence experiment exists.")
    return candidates[-1]


def _resolve_root(value: str, *, create: bool = False) -> Path:
    if str(value).strip():
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else ROOT / candidate
    if create:
        return ROOT / "outputs/experiments" / f"{EXPERIMENT_PREFIX}{_utc_timestamp()}"
    return _latest_experiment()


def _experiment_design(root: Path) -> dict[str, Any]:
    path = root / "inputs/experiment_design.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment design: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_seeds(root: Path) -> tuple[int, ...]:
    return _normalize_seeds(_experiment_design(root)["seeds"])


def _freeze_input(source: Path, destination: Path) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix == ".sqlite":
        wal_path = Path(f"{source}-wal")
        if wal_path.is_file() and wal_path.stat().st_size > 0:
            raise RuntimeError(
                "Cannot freeze a SQLite database with a non-empty WAL; "
                f"checkpoint the database first: {wal_path}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"Existing frozen input differs from source: {destination}")
        return "existing"
    reflink = subprocess.run(
        [
            "cp",
            "--reflink=always",
            "--preserve=mode,timestamps",
            str(source),
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if reflink.returncode == 0:
        return "reflink_copy"
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)
    return "copy"


def _input_manifest(root: Path, source_rows: Sequence[Mapping[str, Any]]) -> Path:
    source_by_relative = {
        str(row["relative_path"]): dict(row)
        for row in source_rows
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "inputs").rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        source = source_by_relative.get(relative, {})
        rows.append(
            {
                "relative_path": relative,
                "source_path": source.get("source_path", "generated"),
                "import_method": source.get("import_method", "generated"),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "category": source.get("category", "generated_input"),
            }
        )
    output = root / "inputs/input_manifest.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _fold_config(root: Path, fold: str) -> Path:
    return root / "folds" / fold / "train_rq1_sequence_textbase.yaml"


def _fold_test_count(root: Path, fold: str) -> int:
    manifest = pd.read_csv(root / "folds" / fold / "split_manifest.csv")
    return int(manifest["split"].astype(str).eq("test").sum())


def prepare_experiment(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root, create=True)
    if root.exists() and any(root.iterdir()) and not args.reuse:
        raise FileExistsError(f"Experiment root is not empty: {root}")
    for relative in (
        "inputs/configs",
        "inputs/data",
        "folds",
        "training_runs",
        "logs",
        "registry",
        "checkpoint_selection",
        "generated_results",
        "comparisons",
        "final_tables",
        "docs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    sources = {
        "config": Path(args.config).expanduser().resolve(),
        "database": Path(args.database).expanduser().resolve(),
        "news": Path(args.news_workbook).expanduser().resolve(),
        "calendar": Path(args.session_calendar).expanduser().resolve(),
    }
    for value in sources.values():
        if not value.is_file():
            raise FileNotFoundError(value)

    destinations = {
        "config": root / "inputs/configs/train_rq1_sequence_textbase_source.yaml",
        "database": root / "inputs/data/market_surface_index.sqlite",
        "news": root / "inputs/data/news_with_openai_embeddings_large.xlsx",
        "calendar": root / "inputs/data/cme_treasury_globex_closures_2022_2023.csv",
    }
    categories = {
        "config": "frozen_config",
        "database": "raw_surface_index",
        "news": "factiva_news",
        "calendar": "market_calendar",
    }
    source_rows: list[dict[str, Any]] = []
    for key, source in sources.items():
        method = _freeze_input(source, destinations[key])
        source_rows.append(
            {
                "relative_path": str(destinations[key].relative_to(root)),
                "source_path": str(source),
                "import_method": method,
                "category": categories[key],
            }
        )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (root / "inputs/git_state.txt").write_text(
        f"commit={git_commit}\n{git_status}", encoding="utf-8"
    )

    dataset_summary = build_sequence_datasets(
        database_path=destinations["database"],
        news_workbook=destinations["news"],
        session_calendar_path=destinations["calendar"],
        output_root=root,
        permutation_seed=int(args.permutation_seed),
    )
    source_config = _read_yaml(destinations["config"])
    base_training = dict(source_config.get("training") or source_config)
    generate = dict(source_config.get("generate_result") or {})
    for fold in FOLDS:
        fold_dir = root / "folds" / fold
        fold_training = dict(base_training)
        fold_training.update(
            {
                "data_path": str(fold_dir / "sequence_dataset.npz"),
                "split_manifest_path": str(fold_dir / "split_manifest.csv"),
                "surface_support_path": str(fold_dir / "raw_support_artifact.json"),
                "text_transform_path": str(fold_dir / "text_transform.npz"),
                "output_root": "",
                "checkpoints_path": "",
                "metrics_path": "",
            }
        )
        _write_yaml(
            _fold_config(root, fold),
            {"training": fold_training, "generate_result": generate},
        )

    seeds = _normalize_seeds(args.seeds)
    split_counts = {
        fold: json.loads(
            (root / "folds" / fold / "dataset_validation.json").read_text(
                encoding="utf-8"
            )
        )["split_counts"]
        for fold in FOLDS
    }
    design = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "surface_source": "raw",
        "source_timezone": "Europe/London",
        "publication_availability_lag_minutes": 0,
        "history_steps": 12,
        "history_interval_minutes": 5,
        "history_min_valid_steps": 6,
        "forecast_horizon_minutes": 5,
        "news_buckets_minutes": [[0, 15], [15, 30], [30, 60]],
        "text_condition_dim": 389,
        "seeds": list(seeds),
        "folds": list(FOLDS),
        "variants": list(VARIANTS),
        "variant_stages": VARIANT_STAGE,
        "expected_training_runs": len(FOLDS) * len(seeds) * len(VARIANTS),
        "split_counts": split_counts,
        "dataset_summary": dataset_summary,
    }
    _write_json(root / "inputs/experiment_design.json", design)
    manifest_path = _input_manifest(root, source_rows)
    _write_json(
        root / "validation_summary.json",
        {
            "status": "prepared",
            "development_only": True,
            "surface_source": "raw",
            "candidate_count": dataset_summary["candidate_count"],
            "with_news": dataset_summary["with_news"],
            "without_news": dataset_summary["without_news"],
            "fold_split_counts": split_counts,
            "expected_training_runs": design["expected_training_runs"],
            "input_manifest": str(manifest_path),
        },
    )
    print(root)
    return root


def _run_dirs(output_root: Path) -> list[Path]:
    if not output_root.is_dir():
        return []
    return sorted(path for path in output_root.iterdir() if path.is_dir())


def _completed_run(output_root: Path) -> Path | None:
    candidates = [
        path
        for path in _run_dirs(output_root)
        if (path / "checkpoints/film_wgan_best.pt").is_file()
        and (path / "checkpoints/film_wgan_final.pt").is_file()
        and (path / "metrics/best_checkpoint.json").is_file()
    ]
    return candidates[-1] if candidates else None


def _output_root(root: Path, variant: str, fold: str, seed: int) -> Path:
    return root / "training_runs" / variant / fold / f"seed_{seed}"


def _parent_checkpoint(root: Path, fold: str, seed: int) -> Path:
    run = _completed_run(_output_root(root, PARENT_VARIANT, fold, seed))
    if run is None:
        raise FileNotFoundError(
            f"Stage-A parent is not complete for {fold}/seed_{seed}."
        )
    return run / "checkpoints/film_wgan_best.pt"


def _variant_overrides(
    root: Path,
    *,
    variant: str,
    fold: str,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "seed": int(seed),
        "output_root": str(_output_root(root, variant, fold, seed)),
        "checkpoints_path": "",
        "metrics_path": "",
        "surface_input_mode": "rolling_history",
        "history_input_policy": "full_history",
        "conditioning_mode": "residual_film",
        "sequence_text_variant": "no_text",
        "initial_generator_checkpoint_path": "",
        "freeze_backbone_epochs": 0,
        "lambda_film": 0.0,
        "lambda_mismatch": 0.0,
        "num_epochs": 30,
    }
    if variant == PARENT_VARIANT:
        return values
    if variant == CURRENT_ONLY_VARIANT:
        values["history_input_policy"] = "current_only"
        return values
    parent = _parent_checkpoint(root, fold, seed)
    values["initial_generator_checkpoint_path"] = str(parent)
    if variant == CONTINUED_VARIANT:
        return values
    values.update(
        {
            "freeze_backbone_epochs": 5,
            "lambda_film": 1.0e-4,
            "lambda_mismatch": 0.5,
        }
    )
    if variant == COUNT_VARIANT:
        values["sequence_text_variant"] = "news_count"
    elif variant == MATCHED_VARIANT:
        values["sequence_text_variant"] = "matched_lp"
    elif variant == SHUFFLED_VARIANT:
        values["sequence_text_variant"] = "shuffled_lp"
    else:
        raise ValueError(f"Unsupported sequence variant: {variant}")
    return values


def _cli_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def train_one(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    seed = int(args.seed)
    if seed not in _experiment_seeds(root):
        raise ValueError(f"Seed {seed} is not frozen in this experiment.")
    output_root = _output_root(root, args.variant, args.fold, seed)
    completed = _completed_run(output_root)
    if completed is not None:
        print(f"reused={completed}")
        return completed
    if _run_dirs(output_root) and not args.resume:
        raise RuntimeError(
            f"Incomplete run exists under {output_root}; rerun with --resume."
        )
    command = [
        sys.executable,
        "scripts/film_wgan/main.py",
        "train",
        "--config",
        str(_fold_config(root, args.fold)),
        "--train-only",
    ]
    for key, value in _variant_overrides(
        root,
        variant=args.variant,
        fold=args.fold,
        seed=seed,
    ).items():
        command.extend(["--set", _cli_override(key, value)])
    _run_logged(
        command,
        root / "logs/training" / args.variant / args.fold / f"seed_{seed}.log",
    )
    completed = _completed_run(output_root)
    if completed is None:
        raise RuntimeError(f"Training ended without a complete run: {output_root}")
    _write_json(
        output_root / "run_pointer.json",
        {
            "variant": args.variant,
            "fold": args.fold,
            "seed": seed,
            "run_dir": str(completed),
            "checkpoint": str(completed / "checkpoints/film_wgan_best.pt"),
        },
    )
    print(f"completed={completed}")
    return completed


def train_matrix(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    seeds = _experiment_seeds(root)
    selected_folds = [args.fold] if args.fold else list(FOLDS)
    selected_seeds = [int(args.seed)] if args.seed is not None else list(seeds)
    selected_variants = [args.variant] if args.variant else list(VARIANTS)
    for fold in selected_folds:
        for seed in selected_seeds:
            ordered = [
                variant
                for variant in VARIANTS
                if variant in selected_variants
            ]
            for variant in ordered:
                train_one(
                    argparse.Namespace(
                        experiment_root=str(root),
                        fold=fold,
                        seed=seed,
                        variant=variant,
                        resume=args.resume,
                    )
                )
    _write_training_registry(root)
    return root


def _write_training_registry(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in _experiment_seeds(root):
            for variant in VARIANTS:
                run = _completed_run(_output_root(root, variant, fold, seed))
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "training_stage": VARIANT_STAGE[variant],
                        "status": "complete" if run else "pending_or_incomplete",
                        "run_dir": "" if run is None else str(run),
                        "checkpoint": (
                            ""
                            if run is None
                            else str(run / "checkpoints/film_wgan_best.pt")
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    path = root / "registry/training_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def monitor(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    frame = _write_training_registry(root)
    epochs: list[int] = []
    for row in frame.itertuples(index=False):
        latest_epoch = 0
        run_root = _output_root(root, str(row.variant), str(row.fold), int(row.seed))
        runs = _run_dirs(run_root)
        if runs:
            metrics = runs[-1] / "metrics/training_metrics.csv"
            if metrics.is_file():
                values = pd.read_csv(metrics, usecols=["epoch"])
                if not values.empty:
                    latest_epoch = int(values["epoch"].max())
        epochs.append(latest_epoch)
    frame["latest_epoch"] = epochs
    frame.to_csv(root / "registry/monitor_status.csv", index=False)
    print(frame.groupby("status").size().to_string())
    active = frame[frame["status"] != "complete"]
    if not active.empty:
        print(active.tail(20).to_string(index=False))
    return root


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def collect_checkpoints(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for fold in FOLDS:
        for seed in _experiment_seeds(root):
            parent_path = _parent_checkpoint(root, fold, seed)
            parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
            parent_state_sha = _state_dict_sha256(parent_payload["generator_state_dict"])
            stage_b_critic_shas: list[str] = []
            for variant in VARIANTS:
                run = _completed_run(_output_root(root, variant, fold, seed))
                if run is None:
                    failures.append(f"missing_run:{variant}/{fold}/seed_{seed}")
                    continue
                best_path = run / "metrics/best_checkpoint.json"
                best = json.loads(best_path.read_text(encoding="utf-8"))
                checkpoint = run / "checkpoints/film_wgan_best.pt"
                selected_epoch = int(best["best_epoch"])
                if selected_epoch <= 10:
                    failures.append(
                        f"warmup_checkpoint:{variant}/{fold}/seed_{seed}/epoch_{selected_epoch}"
                    )
                config = _read_yaml(run / "metrics/training_resolved_config.yaml")
                training = dict(config.get("training") or config)
                audit = json.loads(
                    (run / "metrics/initialization_audit.json").read_text(
                        encoding="utf-8"
                    )
                )
                configured_parent = str(
                    training.get("initial_generator_checkpoint_path", "")
                )
                expected_parent = str(parent_path) if variant in STAGE_B_VARIANTS else ""
                parent_ok = configured_parent == expected_parent
                generator_ok = (
                    audit.get("initial_generator_state_sha256") == parent_state_sha
                    if variant in STAGE_B_VARIANTS
                    else True
                )
                if not parent_ok:
                    failures.append(f"parent_path:{variant}/{fold}/seed_{seed}")
                if not generator_ok:
                    failures.append(f"parent_state:{variant}/{fold}/seed_{seed}")
                if variant in STAGE_B_VARIANTS:
                    stage_b_critic_shas.append(
                        str(audit.get("initial_critic_state_sha256", ""))
                    )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "training_stage": VARIANT_STAGE[variant],
                        "selected_epoch": selected_epoch,
                        "validation_surface_mae": float(best["best_metric"]),
                        "checkpoint_metric": str(best["checkpoint_metric"]),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "parent_checkpoint_path": configured_parent,
                        "parent_checkpoint_sha256": (
                            sha256_file(parent_path)
                            if variant in STAGE_B_VARIANTS
                            else ""
                        ),
                        "run_dir": str(run),
                    }
                )
                audit_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "expected_parent": expected_parent,
                        "configured_parent": configured_parent,
                        "parent_path_ok": int(parent_ok),
                        "initial_generator_matches_parent": int(generator_ok),
                        "initial_generator_state_sha256": audit.get(
                            "initial_generator_state_sha256", ""
                        ),
                        "initial_critic_state_sha256": audit.get(
                            "initial_critic_state_sha256", ""
                        ),
                        "sequence_text_variant": training.get(
                            "sequence_text_variant", ""
                        ),
                        "history_input_policy": training.get(
                            "history_input_policy", ""
                        ),
                    }
                )
            if stage_b_critic_shas and len(set(stage_b_critic_shas)) != 1:
                failures.append(f"stage_b_critic_initialization:{fold}/seed_{seed}")
    selection = pd.DataFrame(rows)
    selection.to_csv(
        root / "checkpoint_selection/selected_checkpoints.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        root / "checkpoint_selection/initialization_audit.csv", index=False
    )
    if failures:
        _write_json(
            root / "checkpoint_selection/validation_failures.json", failures
        )
        raise ValueError(
            "Checkpoint validation failed; inspect checkpoint_selection/validation_failures.json."
        )
    print(f"selected_checkpoints={len(selection)}")
    return root


def generate_one(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    run = _completed_run(
        _output_root(root, args.variant, args.fold, int(args.seed))
    )
    if run is None:
        raise FileNotFoundError(
            f"Missing completed run: {args.variant}/{args.fold}/seed_{args.seed}"
        )
    output = run / "development_test_json"
    summary = output / "summary.csv"
    expected = _fold_test_count(root, args.fold)
    sample_json_count = (
        len(list((output / "samples").glob("*.json")))
        if (output / "samples").is_dir()
        else 0
    )
    if (
        summary.is_file()
        and len(pd.read_csv(summary)) == expected
        and sample_json_count == expected
    ):
        _write_json(
            root
            / "generated_results"
            / args.variant
            / args.fold
            / f"seed_{args.seed}"
            / "result_pointer.json",
            {
                "run_dir": str(run),
                "checkpoint": str(run / "checkpoints/film_wgan_best.pt"),
                "output_dir": str(output),
                "summary": str(summary),
                "sample_json_count": expected,
            },
        )
        print(f"reused={summary}")
        return output
    checkpoint = run / "checkpoints/film_wgan_best.pt"
    command = [
        sys.executable,
        "scripts/film_wgan/main.py",
        "generate-result",
        "--config",
        str(run / "metrics/training_resolved_config.yaml"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        "development_test_json",
        "--split",
        "test",
        "--selection-mode",
        "all",
        "--selection-count",
        "0",
        "--no-plot",
        "--set",
        "save_full_atm_timeseries=false",
    ]
    _run_logged(
        command,
        root / "logs/generate" / args.variant / args.fold / f"seed_{args.seed}.log",
    )
    sample_json_count = (
        len(list((output / "samples").glob("*.json")))
        if (output / "samples").is_dir()
        else 0
    )
    if (
        not summary.is_file()
        or len(pd.read_csv(summary)) != expected
        or sample_json_count != expected
    ):
        raise ValueError(
            f"Generated test sample count mismatch for {args.variant}/{args.fold}/"
            f"seed_{args.seed}; expected {expected}."
        )
    _write_json(
        root
        / "generated_results"
        / args.variant
        / args.fold
        / f"seed_{args.seed}"
        / "result_pointer.json",
        {
            "run_dir": str(run),
            "checkpoint": str(checkpoint),
            "output_dir": str(output),
            "summary": str(summary),
            "sample_json_count": sample_json_count,
        },
    )
    print(f"completed={summary}")
    return output


def generate_matrix(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.is_file():
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in _experiment_seeds(root):
            for variant in VARIANTS:
                output = generate_one(
                    argparse.Namespace(
                        experiment_root=str(root),
                        fold=fold,
                        seed=seed,
                        variant=variant,
                    )
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "summary_path": str(output / "summary.csv"),
                        "sample_count": _fold_test_count(root, fold),
                    }
                )
                pd.DataFrame(rows).to_csv(
                    root / "registry/generate_registry.csv", index=False
                )
    return root


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    frame = frame[np.isfinite(pd.to_numeric(frame["difference"], errors="coerce"))]
    daily_groups = [
        group["difference"].to_numpy(dtype=np.float64)
        for _day, group in frame.groupby("trading_day", sort=True)
    ]
    if not daily_groups:
        raise ValueError("Cluster bootstrap received no trading-day clusters.")
    observed = float(frame["difference"].mean())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        chosen = rng.integers(0, len(daily_groups), size=len(daily_groups))
        values = np.concatenate([daily_groups[int(value)] for value in chosen])
        draws[index] = float(values.mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    centered = draws - observed
    p_two = (1.0 + float(np.sum(np.abs(centered) >= abs(observed)))) / (
        float(iterations) + 1.0
    )
    p_one = (1.0 + float(np.sum(centered >= observed))) / (
        float(iterations) + 1.0
    )
    return {
        "mean_difference": observed,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "p_two_sided": min(1.0, p_two),
        "p_one_sided_focal_better": min(1.0, p_one),
        "paired_origin_count": int(len(frame)),
        "pair_win_rate": float((frame["difference"] > 0.0).mean()),
        "pair_tie_rate": float((frame["difference"] == 0.0).mean()),
        "cluster_count": len(daily_groups),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
    }


def _dm_hac(frame: pd.DataFrame, *, max_lag: int = 5) -> dict[str, float | int]:
    frame = frame[np.isfinite(pd.to_numeric(frame["difference"], errors="coerce"))]
    daily = (
        frame.groupby("trading_day", sort=True)["difference"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    count = int(daily.size)
    if count < 2:
        return {
            "daily_observations": count,
            "hac_max_lag": 0,
            "dm_hac_statistic": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_focal_better": float("nan"),
        }
    lag = min(int(max_lag), count - 1)
    centered = daily - daily.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / count))
    mean = float(daily.mean())
    statistic = mean / standard_error if standard_error > 0.0 else float("nan")
    distribution = stats.t(df=count - 1)
    return {
        "daily_observations": count,
        "hac_max_lag": lag,
        "mean_daily_difference": mean,
        "hac_standard_error": standard_error,
        "dm_hac_statistic": statistic,
        "p_two_sided": float(2.0 * distribution.sf(abs(statistic))),
        "p_one_sided_focal_better": float(distribution.sf(statistic)),
    }


def _holm(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (len(array) - rank) * float(array[index])),
        )
        adjusted[index] = running
    return adjusted.astype(float).tolist()


def _paired_differences(samples: pd.DataFrame) -> pd.DataFrame:
    # Coverage and interval width are reported descriptively. They do not have
    # a monotone "lower is better" direction, so they are excluded here.
    metrics = list(INFERENTIAL_METRICS)
    index_columns = [
        "fold",
        "seed",
        "surface_pair_id",
        "trading_day",
        "current_snapshot_time_utc",
    ]
    rows: list[dict[str, Any]] = []
    for focal, comparator, contrast in CONTRASTS:
        left = (
            samples[samples["variant"] == focal]
            .set_index(index_columns)
            .sort_index()
        )
        right = (
            samples[samples["variant"] == comparator]
            .set_index(index_columns)
            .sort_index()
        )
        if not left.index.equals(right.index):
            missing_left = right.index.difference(left.index)
            missing_right = left.index.difference(right.index)
            raise ValueError(
                f"Pair mismatch for {contrast}: missing focal={len(missing_left)}, "
                f"missing comparator={len(missing_right)}."
            )
        for metric in metrics:
            if metric not in left.columns or metric not in right.columns:
                continue
            differences = (
                pd.to_numeric(right[metric], errors="coerce")
                - pd.to_numeric(left[metric], errors="coerce")
            )
            for key, value in differences.items():
                rows.append(
                    {
                        "fold": key[0],
                        "seed": int(key[1]),
                        "surface_pair_id": key[2],
                        "trading_day": key[3],
                        "current_snapshot_time_utc": key[4],
                        "contrast": contrast,
                        "focal_model": focal,
                        "comparator_model": comparator,
                        "metric": metric,
                        "difference": float(value),
                        "difference_direction": "comparator_minus_focal",
                    }
                )
    for variant in VARIANTS:
        group = samples[samples["variant"] == variant]
        for metric, current_column in CURRENT_ERROR_COLUMNS.items():
            for row in group.itertuples(index=False):
                rows.append(
                    {
                        "fold": row.fold,
                        "seed": int(row.seed),
                        "surface_pair_id": row.surface_pair_id,
                        "trading_day": row.trading_day,
                        "current_snapshot_time_utc": row.current_snapshot_time_utc,
                        "contrast": f"{variant}_vs_persistence",
                        "focal_model": variant,
                        "comparator_model": "persistence",
                        "metric": metric,
                        "difference": float(getattr(row, current_column))
                        - float(getattr(row, metric)),
                        "difference_direction": "persistence_minus_model",
                    }
                )
    return pd.DataFrame(rows)


def build_comparison(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.is_file():
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    selected = pd.read_csv(selected_path)
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        summary_path = Path(row.run_dir) / "development_test_json/summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        frame = pd.read_csv(summary_path)
        if len(frame) != _fold_test_count(root, str(row.fold)):
            raise ValueError(f"Unexpected test count: {summary_path}")
        frame.insert(0, "variant", str(row.variant))
        frame.insert(0, "seed", int(row.seed))
        frame.insert(0, "fold", str(row.fold))
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    expected = len(VARIANTS) * len(_experiment_seeds(root)) * sum(
        _fold_test_count(root, fold) for fold in FOLDS
    )
    if len(samples) != expected:
        raise ValueError(f"Expected {expected} generated rows, found {len(samples)}.")
    sample_path = root / "comparisons/rq1_test_sample_metrics.csv"
    samples.to_csv(sample_path, index=False)

    metric_columns = [
        metric
        for metric in [*POINT_METRICS, *PROBABILISTIC_METRICS, *FINANCIAL_METRICS]
        if metric in samples.columns
    ]
    pair_metrics = (
        samples.groupby(
            [
                "fold",
                "seed",
                "variant",
                "surface_pair_id",
                "trading_day",
                "current_snapshot_time_utc",
            ],
            as_index=False,
        )[metric_columns]
        .mean()
    )
    pair_metrics.to_csv(root / "comparisons/rq1_test_pair_metrics.csv", index=False)
    seed_averaged_pair_metrics = (
        pair_metrics.groupby(
            [
                "fold",
                "variant",
                "surface_pair_id",
                "trading_day",
                "current_snapshot_time_utc",
            ],
            as_index=False,
        )[metric_columns]
        .mean()
    )
    seed_averaged_pair_metrics.to_csv(
        root / "comparisons/rq1_test_pair_metrics_seed_averaged.csv",
        index=False,
    )

    differences = _paired_differences(samples)
    differences.to_csv(
        root / "comparisons/rq1_pairwise_differences.csv", index=False
    )
    averaged = (
        differences.groupby(
            [
                "fold",
                "surface_pair_id",
                "trading_day",
                "current_snapshot_time_utc",
                "contrast",
                "focal_model",
                "comparator_model",
                "metric",
                "difference_direction",
            ],
            as_index=False,
        )["difference"]
        .mean()
    )

    bootstrap_rows: list[dict[str, Any]] = []
    hac_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for (contrast, metric), group in averaged.groupby(
        ["contrast", "metric"], sort=True
    ):
        group = group[
            np.isfinite(pd.to_numeric(group["difference"], errors="coerce"))
        ]
        if group.empty:
            continue
        identity = group.iloc[0]
        bootstrap_rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "focal_model": identity["focal_model"],
                "comparator_model": identity["comparator_model"],
                "difference_direction": identity["difference_direction"],
                **_cluster_bootstrap(
                    group,
                    iterations=int(args.bootstrap_iterations),
                    seed=int(args.bootstrap_seed),
                ),
            }
        )
        hac_rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "focal_model": identity["focal_model"],
                "comparator_model": identity["comparator_model"],
                **_dm_hac(group, max_lag=5),
            }
        )
        seed_means = (
            differences[
                (differences["contrast"] == contrast)
                & (differences["metric"] == metric)
            ]
            .groupby("seed", sort=True)["difference"]
            .mean()
            .dropna()
        )
        values = seed_means.to_numpy(dtype=np.float64)
        t_result = stats.ttest_1samp(values, popmean=0.0)
        try:
            wilcoxon = stats.wilcoxon(values, zero_method="wilcox", method="exact")
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        seed_rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "seed_count": len(values),
                "mean_seed_difference": float(values.mean()),
                "std_seed_difference": float(values.std(ddof=1)),
                "positive_seed_count": int((values > 0.0).sum()),
                "paired_t_statistic": float(t_result.statistic),
                "paired_t_p_two_sided": float(t_result.pvalue),
                "paired_t_p_one_sided_focal_better": float(
                    stats.t.sf(float(t_result.statistic), df=len(values) - 1)
                ),
                "wilcoxon_exact_p_two_sided": wilcoxon_p,
                "seed_differences_json": json.dumps(
                    {str(key): float(value) for key, value in seed_means.items()},
                    sort_keys=True,
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["holm_family"] = bootstrap["contrast"].map(_contrast_family)
    bootstrap["holm_p_two_sided_within_metric_family"] = np.nan
    for (_metric, _family), indices in bootstrap.groupby(
        ["metric", "holm_family"]
    ).groups.items():
        bootstrap.loc[
            indices, "holm_p_two_sided_within_metric_family"
        ] = _holm(
            bootstrap.loc[indices, "p_two_sided"].tolist()
        )
    hac = pd.DataFrame(hac_rows)
    seed_tests = pd.DataFrame(seed_rows)
    bootstrap.to_csv(
        root / "comparisons/rq1_cluster_bootstrap_ci.csv", index=False
    )
    hac.to_csv(root / "comparisons/rq1_dm_hac_tests.csv", index=False)
    seed_tests.to_csv(
        root / "comparisons/rq1_seed_level_tests.csv", index=False
    )

    fold_summary = (
        differences.groupby(
            ["fold", "contrast", "focal_model", "comparator_model", "metric"]
        )["difference"]
        .agg(
            mean_difference="mean",
            std_difference="std",
            sample_rows="count",
            positive_pair_rows=lambda values: int((values > 0.0).sum()),
            pair_row_win_rate=lambda values: float((values > 0.0).mean()),
            pair_row_tie_rate=lambda values: float((values == 0.0).mean()),
        )
        .reset_index()
    )
    fold_summary.to_csv(
        root / "comparisons/rq1_fold_level_differences.csv", index=False
    )

    primary = bootstrap[
        (bootstrap["contrast"] == "incremental_lp_text")
        & bootstrap["metric"].isin(POINT_METRICS)
    ].copy()
    primary = primary.merge(
        seed_tests[
            seed_tests["contrast"] == "incremental_lp_text"
        ][
            [
                "metric",
                "seed_count",
                "positive_seed_count",
                "paired_t_p_two_sided",
                "paired_t_p_one_sided_focal_better",
                "wilcoxon_exact_p_two_sided",
            ]
        ],
        on="metric",
        how="left",
        validate="one_to_one",
    )
    primary.to_csv(
        root / "final_tables/thesis_rq1_primary_test_table.csv", index=False
    )
    bootstrap[
        bootstrap["contrast"].isin(
            {
                "history_vs_current_only",
                "lp_semantics_vs_news_arrival",
                "matched_vs_shuffled_lp",
            }
        )
        & bootstrap["metric"].isin(POINT_METRICS)
    ].to_csv(
        root / "final_tables/thesis_rq1_architecture_ablations.csv", index=False
    )
    model_summary = (
        samples.groupby("variant", as_index=False)[metric_columns]
        .mean()
    )
    model_summary.to_csv(
        root / "final_tables/development_rq1_model_overall_metrics.csv", index=False
    )
    model_summary[["variant", *[m for m in PROBABILISTIC_METRICS if m in model_summary]]].to_csv(
        root / "final_tables/thesis_rq1_probabilistic_table.csv", index=False
    )
    model_summary[["variant", *[m for m in FINANCIAL_METRICS if m in model_summary]]].to_csv(
        root / "final_tables/thesis_rq1_financial_consistency.csv", index=False
    )

    _write_json(
        root / "validation_summary.json",
        {
            "status": "ok",
            "development_only": True,
            "surface_source": "raw",
            "training_run_count": int(len(selected)),
            "selected_checkpoint_count": int(len(selected)),
            "all_selected_epochs_after_10": bool(
                (selected["selected_epoch"] > 10).all()
            ),
            "generated_sample_rows": int(len(samples)),
            "test_origin_count": int(
                samples[["fold", "surface_pair_id"]].drop_duplicates().shape[0]
            ),
            "seeds": list(_experiment_seeds(root)),
            "fold_test_counts": {
                fold: _fold_test_count(root, fold) for fold in FOLDS
            },
            "difference_direction": "comparator_error_minus_focal_error",
            "positive_difference_means_focal_better": True,
            "primary_results": json.loads(primary.to_json(orient="records")),
        },
    )
    _write_archive_manifest(root)
    return root


def _write_archive_manifest(root: Path) -> Path:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    output = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def results_pipeline(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    generate_matrix(argparse.Namespace(experiment_root=str(root)))
    return build_comparison(
        argparse.Namespace(
            experiment_root=str(root),
            bootstrap_iterations=int(args.bootstrap_iterations),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    )


def package_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    _write_archive_manifest(root)
    output = (
        Path(args.output).expanduser()
        if str(args.output).strip()
        else root.parent / f"{root.name}.zip"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--experiment-root", default="")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.add_argument("--database", default=str(DEFAULT_DATABASE))
    prepare.add_argument("--news-workbook", default=str(DEFAULT_NEWS_WORKBOOK))
    prepare.add_argument("--session-calendar", default=str(DEFAULT_SESSION_CALENDAR))
    prepare.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    prepare.add_argument("--permutation-seed", type=int, default=20260722)
    prepare.add_argument("--reuse", action="store_true")

    train = commands.add_parser("train-one")
    train.add_argument("--experiment-root", default="")
    train.add_argument("--fold", choices=list(FOLDS), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--variant", choices=list(VARIANTS), required=True)
    train.add_argument("--resume", action="store_true")

    matrix = commands.add_parser("train-matrix")
    matrix.add_argument("--experiment-root", default="")
    matrix.add_argument("--fold", choices=list(FOLDS), default="")
    matrix.add_argument("--seed", type=int, default=None)
    matrix.add_argument("--variant", choices=list(VARIANTS), default="")
    matrix.add_argument("--resume", action="store_true")

    monitor_parser = commands.add_parser("monitor")
    monitor_parser.add_argument("--experiment-root", default="")

    collect = commands.add_parser("collect-checkpoints")
    collect.add_argument("--experiment-root", default="")

    generate = commands.add_parser("generate-one")
    generate.add_argument("--experiment-root", default="")
    generate.add_argument("--fold", choices=list(FOLDS), required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--variant", choices=list(VARIANTS), required=True)

    generate_all = commands.add_parser("generate")
    generate_all.add_argument("--experiment-root", default="")

    compare = commands.add_parser("compare")
    compare.add_argument("--experiment-root", default="")
    compare.add_argument("--bootstrap-iterations", type=int, default=10000)
    compare.add_argument("--bootstrap-seed", type=int, default=20260722)

    results = commands.add_parser("results-pipeline")
    results.add_argument("--experiment-root", default="")
    results.add_argument("--bootstrap-iterations", type=int, default=10000)
    results.add_argument("--bootstrap-seed", type=int, default=20260722)

    package = commands.add_parser("package")
    package.add_argument("--experiment-root", default="")
    package.add_argument("--output", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    handlers = {
        "prepare": prepare_experiment,
        "train-one": train_one,
        "train-matrix": train_matrix,
        "monitor": monitor,
        "collect-checkpoints": collect_checkpoints,
        "generate-one": generate_one,
        "generate": generate_matrix,
        "compare": build_comparison,
        "results-pipeline": results_pipeline,
        "package": package_experiment,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    main()
