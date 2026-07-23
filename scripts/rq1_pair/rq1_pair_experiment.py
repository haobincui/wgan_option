#!/usr/bin/env python3
"""RQ1 pair-level text rolling-development experiment orchestration."""

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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import load_train_config  # noqa: E402
from film_wgan.data import (  # noqa: E402
    apply_text_alignment,
    build_split_manifest_frame,
    create_train_val_bundle,
    load_film_wgan_samples,
)
from film_wgan.text_transform import sha256_file  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq1_pair_textbase.yaml"
DEFAULT_WORKBOOK = (
    ROOT
    / "data/processed/raw-excel/raw_vol_w5_s3_20260629-144428/merged_vol_rq2_text.xlsx"
)
DEFAULT_NEWS_WORKBOOK = ROOT / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
SUMMARY_DOCUMENT = ROOT / "docs/summary/rq_research_logic_and_methodology_review_20260722.md"
EXPERIMENT_PREFIX = "rq1_pair_text_raw_vol_rolling_"
EXPECTED_SURFACE_MODEL = "raw"

SEEDS = (42, 202, 404)
VARIANTS = (
    "pair_pca_no_text_residual",
    "pair_pca_text_residual_pretrained",
    "pair_pca_shuffled_residual_pretrained",
    "pair_pca_text_full_film",
    "pair_pca_text_concat",
    "pair_l2_text_full_film",
)
FOLDS = {
    "2023Q1": {
        "train_end": "2022Q3",
        "validation": "2022Q4",
        "test": "2023Q1",
        "counts": (1621, 425, 521),
    },
    "2023Q2": {
        "train_end": "2022Q4",
        "validation": "2023Q1",
        "test": "2023Q2",
        "counts": (2046, 521, 333),
    },
    "2023Q3": {
        "train_end": "2023Q1",
        "validation": "2023Q2",
        "test": "2023Q3",
        "counts": (2567, 333, 365),
    },
    "2023Q4": {
        "train_end": "2023Q2",
        "validation": "2023Q3",
        "test": "2023Q4",
        "counts": (2900, 365, 378),
    },
}
POINT_METRICS = ("surface_mae", "short_atm_mae", "atm7_abs_err")
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
FINANCIAL_METRICS = ("calendar_violation_rate", "butterfly_violation_rate")
CONTRASTS = (
    ("pair_pca_text_residual_pretrained", "pair_pca_no_text_residual", "incremental_text"),
    ("pair_pca_text_residual_pretrained", "pair_pca_shuffled_residual_pretrained", "matched_vs_shuffled"),
    ("pair_pca_text_full_film", "pair_pca_text_concat", "film_vs_concat"),
    ("pair_pca_text_full_film", "pair_l2_text_full_film", "pca_vs_raw_l2"),
    ("pair_pca_text_residual_pretrained", "pair_pca_text_full_film", "residual_package_vs_full_film"),
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _latest_experiment() -> Path:
    candidates = sorted((ROOT / "outputs/experiments").glob(f"{EXPERIMENT_PREFIX}*"))
    if not candidates:
        raise FileNotFoundError("No raw-vol RQ1 pair-text experiment exists; run prepare first.")
    return candidates[-1]


def _resolve_root(value: str | None, *, create: bool = False) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    if create:
        return ROOT / "outputs/experiments" / f"{EXPERIMENT_PREFIX}{_utc_timestamp()}"
    return _latest_experiment()


def _assert_py312() -> None:
    if Path(sys.prefix).name != "py312":
        raise RuntimeError(f"Run RQ1 pair workflow in conda env 'py312'; current prefix={sys.prefix}.")


def _quarter(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return str(timestamp.to_period("Q"))


def _fold_dir(root: Path, fold: str) -> Path:
    return root / "inputs/folds" / fold


def _fold_config(root: Path, fold: str) -> Path:
    return _fold_dir(root, fold) / "train_rq1_pair_textbase.yaml"


def _variant_overrides(
    variant: str,
    *,
    fold_config: dict[str, Any],
    output_root: Path,
    seed: int,
    parent_checkpoint: Path | None,
) -> dict[str, Any]:
    training = dict(fold_config["training"])
    overrides: dict[str, Any] = {
        "seed": int(seed),
        "output_root": str(output_root),
        "checkpoints_path": "",
        "metrics_path": "",
        "text_embedding_mode": "lp",
        "text_alignment_mode": "matched",
        "text_preprocessing_mode": "pca",
        "normalize_text_embedding": False,
        "conditioning_mode": "residual_film",
        "critic_conditioning_mode": "projection",
        "lambda_film": 1.0e-4,
        "lambda_mismatch": 0.5,
        "initial_generator_checkpoint_path": "",
        "freeze_backbone_epochs": 0,
    }
    if variant == "pair_pca_no_text_residual":
        overrides.update(
            text_embedding_mode="zero_lp",
            lambda_film=0.0,
            lambda_mismatch=0.0,
        )
    elif variant == "pair_pca_text_residual_pretrained":
        if parent_checkpoint is None:
            raise FileNotFoundError("Residual text training requires its paired no-text checkpoint.")
        overrides.update(
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
        )
    elif variant == "pair_pca_shuffled_residual_pretrained":
        if parent_checkpoint is None:
            raise FileNotFoundError("Shuffled residual training requires its paired no-text checkpoint.")
        overrides.update(
            text_alignment_mode="permuted",
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
        )
    elif variant == "pair_pca_text_full_film":
        overrides.update(
            conditioning_mode="film",
            critic_conditioning_mode="inherit",
            lambda_film=0.0,
            lambda_mismatch=0.0,
        )
    elif variant == "pair_pca_text_concat":
        overrides.update(
            conditioning_mode="concat",
            critic_conditioning_mode="inherit",
            lambda_film=0.0,
            lambda_mismatch=0.0,
        )
    elif variant == "pair_l2_text_full_film":
        overrides.update(
            conditioning_mode="film",
            critic_conditioning_mode="inherit",
            text_preprocessing_mode="raw_l2",
            text_transform_path="",
            lambda_film=0.0,
            lambda_mismatch=0.0,
        )
    else:
        raise ValueError(f"Unsupported RQ1 pair variant: {variant}")
    if variant != "pair_l2_text_full_film":
        overrides["text_transform_path"] = str(training["text_transform_path"])
    return overrides


def _cli_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float)):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
    else:
        raise TypeError(f"Unsupported CLI override value for {key}: {type(value)}")
    return f"{key}={rendered}"


def _validate_raw_surface_workbook(path: Path, *, sheet_name: str) -> str:
    try:
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols=["surface_model"],
        )
    except ValueError as exc:
        raise ValueError(
            f"RQ1 raw-vol workbook must contain a surface_model column in sheet "
            f"{sheet_name!r}: {path}"
        ) from exc
    models = {
        str(value).strip().lower()
        for value in frame["surface_model"].dropna().tolist()
        if str(value).strip()
    }
    if models != {EXPECTED_SURFACE_MODEL}:
        raise ValueError(
            f"RQ1 raw-vol workflow requires surface_model={EXPECTED_SURFACE_MODEL!r}; "
            f"found {sorted(models)} in {path}."
        )
    return EXPECTED_SURFACE_MODEL


def _run(command: Sequence[str], *, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(list(command), cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


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


def prepare_experiment(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root, create=True)
    if root.exists() and any(root.iterdir()) and not args.reuse:
        raise FileExistsError(f"Experiment directory is not empty: {root}")
    for relative in (
        "inputs/configs",
        "inputs/data",
        "inputs/folds",
        "inputs/docs",
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

    source_config = Path(args.config).resolve()
    source_workbook = Path(args.workbook).resolve()
    source_news = Path(args.news_workbook).resolve()
    for source in (source_config, source_workbook, source_news):
        if not source.is_file():
            raise FileNotFoundError(source)
    base_payload = _read_yaml(source_config)
    training = dict(base_payload.get("training") or base_payload)
    generate = dict(base_payload.get("generate_result") or {})
    surface_model = _validate_raw_surface_workbook(
        source_workbook,
        sheet_name=str(training.get("sheet_name", "gan_input_ready")),
    )
    copied_config = root / "inputs/configs/train_rq1_pair_textbase_source.yaml"
    copied_workbook = root / "inputs/data/merged_vol_rq2_text.xlsx"
    copied_news = root / "inputs/data/news_with_openai_embeddings_large.xlsx"
    shutil.copy2(source_config, copied_config)
    shutil.copy2(source_workbook, copied_workbook)
    shutil.copy2(source_news, copied_news)
    if SUMMARY_DOCUMENT.is_file():
        shutil.copy2(SUMMARY_DOCUMENT, root / "inputs/docs/research_logic_review.md")
    git_state = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / "inputs/git_state.txt").write_text(
        f"commit={commit}\n{git_state}",
        encoding="utf-8",
    )

    training.update(
        data_path=str(copied_workbook),
        news_workbook_path=str(copied_news),
        output_root="",
        checkpoints_path="",
        metrics_path="",
    )
    base_frozen = root / "inputs/configs/train_rq1_pair_textbase.yaml"
    _write_yaml(base_frozen, {"training": training, "generate_result": generate})
    row_config = load_train_config(base_frozen)
    row_samples = load_film_wgan_samples(row_config)
    base_manifest = build_split_manifest_frame(row_config, row_samples)
    pair_quarters = {
        sample.surface_pair_id: _quarter(sample.current_snapshot_time_utc)
        for sample in row_samples
    }

    fold_validation: dict[str, Any] = {}
    for fold, specification in FOLDS.items():
        fold_root = _fold_dir(root, fold)
        fold_root.mkdir(parents=True, exist_ok=True)
        split_by_pair: dict[str, str] = {}
        for pair_id, quarter in pair_quarters.items():
            if quarter <= str(specification["train_end"]):
                split = "train"
            elif quarter == str(specification["validation"]):
                split = "val"
            elif quarter == str(specification["test"]):
                split = "test"
            else:
                split = "excluded"
            split_by_pair[pair_id] = split
        manifest = base_manifest.copy()
        manifest["split"] = manifest["surface_pair_id"].astype(str).map(split_by_pair)
        manifest["outer_test_fold"] = fold
        manifest_path = fold_root / "split_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        transform_path = fold_root / "text_transform.npz"
        fold_training = dict(training)
        fold_training.update(
            split_manifest_path=str(manifest_path),
            text_transform_path=str(transform_path),
            text_embedding_mode="lp",
            sample_unit="surface_pair",
            text_preprocessing_mode="pca",
            normalize_text_embedding=False,
        )
        fold_config_path = _fold_config(root, fold)
        _write_yaml(
            fold_config_path,
            {"training": fold_training, "generate_result": generate},
        )
        bundle = create_train_val_bundle(load_train_config(fold_config_path))
        actual_counts = (bundle.train_samples, bundle.val_samples, bundle.test_samples)
        expected_counts = tuple(int(value) for value in specification["counts"])
        if actual_counts != expected_counts:
            raise ValueError(
                f"Rolling fold {fold} count mismatch: expected {expected_counts}, found {actual_counts}."
            )
        pair_rows = []
        permutation_rows = []
        for split, items in (
            ("train", bundle.train_items),
            ("val", bundle.val_items),
            ("test", bundle.test_items),
        ):
            for sample in items:
                pair_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "surface_pair_id": sample.surface_pair_id,
                        "sample_id": sample.sample_id,
                        "current_snapshot_time_utc": sample.current_snapshot_time_utc,
                        "target_snapshot_time_utc": sample.target_snapshot_time_utc,
                        "news_count": sample.metadata["news_count"],
                        "unique_embedding_count": sample.metadata["unique_embedding_count"],
                        "source_sample_ids": json.dumps(sample.metadata["source_sample_ids"]),
                        "article_ids": json.dumps(sample.metadata["article_ids"]),
                    }
                )
            shuffled_config = load_train_config(
                fold_config_path,
                overrides={"text_alignment_mode": "permuted"},
            )
            permuted_items = apply_text_alignment(shuffled_config, items, split=split)
            for target, donor_view in zip(items, permuted_items):
                permutation_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "target_surface_pair_id": target.surface_pair_id,
                        "target_sample_id": target.sample_id,
                        "donor_surface_pair_id": donor_view.metadata["text_source_surface_pair_id"],
                        "donor_sample_id": donor_view.metadata["text_source_sample_id"],
                        "permutation_seed": int(shuffled_config.text_permutation_seed),
                    }
                )
        pd.DataFrame(pair_rows).to_csv(fold_root / "pair_lineage_audit.csv", index=False)
        pd.DataFrame(permutation_rows).to_csv(
            fold_root / "text_permutation_mapping.csv",
            index=False,
        )
        fold_validation[fold] = {
            "train_pairs": bundle.train_samples,
            "validation_pairs": bundle.val_samples,
            "test_pairs": bundle.test_samples,
            "excluded_rows": int((manifest["split"] == "excluded").sum()),
            "text_transform_sha256": sha256_file(transform_path),
        }

    input_rows = []
    for path in sorted((root / "inputs").rglob("*")):
        if path.is_file():
            input_rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(input_rows).to_csv(root / "inputs/input_manifest.csv", index=False)
    _write_json(
        root / "validation_summary.json",
        {
            "status": "prepared",
            "development_only": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "surface_model": surface_model,
            "source_workbook": str(source_workbook),
            "source_workbook_sha256": sha256_file(source_workbook),
            "models": list(VARIANTS),
            "seeds": list(SEEDS),
            "folds": fold_validation,
            "expected_training_runs": len(FOLDS) * len(SEEDS) * len(VARIANTS),
        },
    )
    print(root)
    return root


def train_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    registry_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_payload = _read_yaml(_fold_config(root, fold))
        for seed in SEEDS:
            parent_checkpoint: Path | None = None
            for variant in VARIANTS:
                output_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                completed = _completed_run(output_root)
                if completed is not None:
                    if variant == "pair_pca_no_text_residual":
                        parent_checkpoint = completed / "checkpoints/film_wgan_best.pt"
                    registry_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "variant": variant,
                            "status": "reused",
                            "run_dir": str(completed),
                            "checkpoint": str(completed / "checkpoints/film_wgan_best.pt"),
                        }
                    )
                    continue
                if not args.resume and _run_dirs(output_root):
                    raise RuntimeError(
                        f"Incomplete run exists under {output_root}; rerun with --resume to preserve it and restart."
                    )
                overrides = _variant_overrides(
                    variant,
                    fold_config=fold_payload,
                    output_root=output_root,
                    seed=seed,
                    parent_checkpoint=parent_checkpoint,
                )
                command = [
                    sys.executable,
                    "scripts/film_wgan/main.py",
                    "train",
                    "--config",
                    str(_fold_config(root, fold)),
                    "--train-only",
                ]
                for key, value in overrides.items():
                    command.extend(["--set", _cli_override(key, value)])
                log_path = root / "logs" / variant / fold / f"seed_{seed}.log"
                _run(command, log_path=log_path)
                completed = _completed_run(output_root)
                if completed is None:
                    raise RuntimeError(f"Training finished without a best checkpoint: {output_root}")
                if variant == "pair_pca_no_text_residual":
                    parent_checkpoint = completed / "checkpoints/film_wgan_best.pt"
                registry_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "status": "completed",
                        "run_dir": str(completed),
                        "checkpoint": str(completed / "checkpoints/film_wgan_best.pt"),
                    }
                )
                pd.DataFrame(registry_rows).to_csv(
                    root / "registry/launch_registry.csv",
                    index=False,
                )
    pd.DataFrame(registry_rows).to_csv(root / "registry/launch_registry.csv", index=False)
    return root


def monitor(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for variant in VARIANTS:
                output_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                runs = _run_dirs(output_root)
                latest = runs[-1] if runs else None
                status = "pending"
                epoch = 0
                if latest is not None:
                    status = "running_or_incomplete"
                    metrics_path = latest / "metrics/training_metrics.csv"
                    if metrics_path.is_file():
                        metrics = pd.read_csv(metrics_path)
                        epoch = int(metrics["epoch"].max()) if not metrics.empty else 0
                    if (latest / "checkpoints/film_wgan_final.pt").is_file():
                        status = "complete"
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "status": status,
                        "latest_epoch": epoch,
                        "run_dir": "" if latest is None else str(latest),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "registry/monitor_status.csv", index=False)
    print(frame.groupby("status").size().to_string())
    active = frame[frame["status"] == "running_or_incomplete"]
    if not active.empty:
        print(active[["fold", "seed", "variant", "latest_epoch", "run_dir"]].to_string(index=False))
    return root


def collect_checkpoints(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    rows = []
    resolved_configs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            for variant in VARIANTS:
                run_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                run_dir = _completed_run(run_root)
                if run_dir is None:
                    raise FileNotFoundError(f"Missing completed run: {variant}/{fold}/seed_{seed}")
                best = json.loads((run_dir / "metrics/best_checkpoint.json").read_text(encoding="utf-8"))
                epoch = int(best["best_epoch"])
                if epoch <= 10:
                    raise ValueError(f"Selected checkpoint must be after epoch 10: {run_dir} epoch={epoch}")
                checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
                resolved_config_path = run_dir / "metrics/training_resolved_config.yaml"
                resolved_configs[(fold, seed, variant)] = _read_yaml(resolved_config_path)
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "selected_epoch": epoch,
                        "validation_surface_mae": float(best["best_metric"]),
                        "checkpoint_metric": str(best["checkpoint_metric"]),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "run_dir": str(run_dir),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "checkpoint_selection/selected_checkpoints.csv", index=False)
    allowed_variant_fields = {
        "text_embedding_mode",
        "text_alignment_mode",
        "text_preprocessing_mode",
        "text_transform_path",
        "normalize_text_embedding",
        "conditioning_mode",
        "critic_conditioning_mode",
        "lambda_film",
        "lambda_mismatch",
        "initial_generator_checkpoint_path",
        "freeze_backbone_epochs",
        "seed",
        "output_root",
        "checkpoints_path",
        "metrics_path",
    }
    audit_rows = []
    failures = []
    for fold in FOLDS:
        reference = resolved_configs[(fold, SEEDS[0], "pair_pca_no_text_residual")]
        reference = dict(reference.get("training") or reference)
        for seed in SEEDS:
            for variant in VARIANTS:
                candidate_raw = resolved_configs[(fold, seed, variant)]
                candidate = dict(candidate_raw.get("training") or candidate_raw)
                differing = sorted(
                    key
                    for key in set(reference) | set(candidate)
                    if reference.get(key) != candidate.get(key)
                )
                unexpected = sorted(set(differing) - allowed_variant_fields)
                audit_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "differing_fields": json.dumps(differing),
                        "unexpected_fields": json.dumps(unexpected),
                    }
                )
                if unexpected:
                    failures.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "variant": variant,
                            "unexpected_fields": unexpected,
                        }
                    )
    pd.DataFrame(audit_rows).to_csv(
        root / "checkpoint_selection/resolved_config_audit.csv",
        index=False,
    )
    if failures:
        _write_json(
            root / "checkpoint_selection/resolved_config_validation_failures.json",
            failures,
        )
        raise ValueError("Resolved configs differ outside the pre-registered variant/seed/path fields.")
    print(f"Selected {len(frame)} development checkpoints.")
    return root


def generate_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.is_file():
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    selected = pd.read_csv(selected_path)
    registry_rows = []
    for row in selected.itertuples(index=False):
        run_dir = Path(row.run_dir)
        output_dir = run_dir / "development_test_json"
        summary_path = output_dir / "summary.csv"
        expected = int(FOLDS[str(row.fold)]["counts"][2])
        if summary_path.is_file() and len(pd.read_csv(summary_path)) == expected:
            status = "reused"
        else:
            command = [
                sys.executable,
                "scripts/film_wgan/main.py",
                "generate-result",
                "--config",
                str(run_dir / "metrics/training_resolved_config.yaml"),
                "--checkpoint",
                str(row.checkpoint_path),
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
            _run(
                command,
                log_path=root / "logs/generate" / str(row.variant) / str(row.fold) / f"seed_{row.seed}.log",
            )
            if not summary_path.is_file() or len(pd.read_csv(summary_path)) != expected:
                raise ValueError(
                    f"Generate-result row count mismatch for {row.variant}/{row.fold}/seed_{row.seed}."
                )
            status = "completed"
        registry_rows.append(
            {
                "fold": row.fold,
                "seed": int(row.seed),
                "variant": row.variant,
                "status": status,
                "summary_path": str(summary_path),
                "sample_count": expected,
            }
        )
        pd.DataFrame(registry_rows).to_csv(
            root / "registry/generate_registry.csv",
            index=False,
        )
    return root


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    cluster_values = [
        group["difference"].to_numpy(dtype=np.float64)
        for _day, group in frame.groupby("trading_day", sort=True)
    ]
    if not cluster_values:
        return float("nan"), float("nan"), float("nan"), float("nan")
    observed = float(frame["difference"].mean())
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(iterations), dtype=np.float64)
    cluster_count = len(cluster_values)
    for index in range(int(iterations)):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        draw = np.concatenate([cluster_values[int(item)] for item in selected])
        boot[index] = float(draw.mean())
    lower, upper = np.quantile(boot, [0.025, 0.975])
    p_two = min(
        1.0,
        2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))),
    )
    return observed, float(lower), float(upper), float(p_two)


def _holm_adjust(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    count = len(array)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(array[index])))
        adjusted[index] = running
    return adjusted.astype(float).tolist()


def build_comparison(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    (root / "comparisons").mkdir(parents=True, exist_ok=True)
    (root / "final_tables").mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry/generate_registry.csv"
    if not registry_path.is_file():
        raise FileNotFoundError("Run generate before building the comparison archive.")
    registry = pd.read_csv(registry_path)
    rows = []
    for record in registry.itertuples(index=False):
        summary = pd.read_csv(record.summary_path)
        summary["fold"] = str(record.fold)
        summary["seed"] = int(record.seed)
        summary["variant"] = str(record.variant)
        rows.append(summary)
    samples = pd.concat(rows, ignore_index=True)
    expected_rows = sum(int(spec["counts"][2]) for spec in FOLDS.values()) * len(SEEDS) * len(VARIANTS)
    if len(samples) != expected_rows:
        raise ValueError(f"Combined development sample count mismatch: {len(samples)} != {expected_rows}.")
    samples_path = root / "comparisons/development_test_sample_metrics.csv"
    samples.to_csv(samples_path, index=False)

    metric_columns = [*POINT_METRICS, *PROBABILISTIC_METRICS, *FINANCIAL_METRICS]
    summary_rows = []
    for keys, group in samples.groupby(["variant", "fold", "seed"], sort=True):
        row = {"variant": keys[0], "fold": keys[1], "seed": int(keys[2]), "n_pairs": len(group)}
        row.update({metric: float(group[metric].mean()) for metric in metric_columns})
        summary_rows.append(row)
    model_summary = pd.DataFrame(summary_rows)
    model_summary.to_csv(root / "comparisons/development_model_metrics_by_fold_seed.csv", index=False)

    difference_rows = []
    for focal, baseline, contrast in CONTRASTS:
        left = samples[samples["variant"] == focal]
        right = samples[samples["variant"] == baseline]
        keys = ["fold", "seed", "surface_pair_id"]
        merged = left.merge(right, on=keys, suffixes=("_focal", "_baseline"), validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Pair matching failed for {contrast}.")
        for metric in POINT_METRICS:
            for row in merged.itertuples(index=False):
                difference_rows.append(
                    {
                        "contrast": contrast,
                        "focal_variant": focal,
                        "baseline_variant": baseline,
                        "fold": getattr(row, "fold"),
                        "seed": int(getattr(row, "seed")),
                        "surface_pair_id": getattr(row, "surface_pair_id"),
                        "current_snapshot_time_utc": getattr(row, "current_snapshot_time_utc_focal"),
                        "metric": metric,
                        "focal_error": float(getattr(row, f"{metric}_focal")),
                        "baseline_error": float(getattr(row, f"{metric}_baseline")),
                        "difference": float(
                            getattr(row, f"{metric}_baseline") - getattr(row, f"{metric}_focal")
                        ),
                    }
                )
    differences = pd.DataFrame(difference_rows)
    differences["trading_day"] = pd.to_datetime(
        differences["current_snapshot_time_utc"],
        utc=True,
    ).dt.date.astype(str)
    differences.to_csv(root / "comparisons/development_pairwise_differences.csv", index=False)

    fold_summary = (
        differences.groupby(
            ["contrast", "focal_variant", "baseline_variant", "fold", "seed", "metric"],
            as_index=False,
        )
        .agg(mean_difference=("difference", "mean"), pair_count=("difference", "size"))
    )
    fold_summary.to_csv(root / "comparisons/development_fold_seed_differences.csv", index=False)

    seed_directions = differences.groupby(
        ["contrast", "focal_variant", "baseline_variant", "seed", "metric"],
        as_index=False,
    ).agg(
        mean_difference=("difference", "mean"),
        pair_count=("difference", "size"),
    )
    fold_directions = fold_summary.groupby(
        ["contrast", "focal_variant", "baseline_variant", "seed", "metric"],
        as_index=False,
    ).agg(
        positive_fold_count=("mean_difference", lambda values: int(np.sum(np.asarray(values) > 0.0))),
        fold_count=("mean_difference", "size"),
    )
    seed_directions = seed_directions.merge(
        fold_directions,
        on=["contrast", "focal_variant", "baseline_variant", "seed", "metric"],
        validate="one_to_one",
    )
    seed_directions.to_csv(root / "comparisons/development_seed_direction_summary.csv", index=False)
    seed_test_rows = []
    for (contrast, focal, baseline, metric), group in seed_directions.groupby(
        ["contrast", "focal_variant", "baseline_variant", "metric"],
        sort=True,
    ):
        values = group.sort_values("seed")["mean_difference"].to_numpy(dtype=np.float64)
        t_result = stats.ttest_1samp(values, popmean=0.0)
        if np.allclose(values, 0.0):
            wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon = stats.wilcoxon(
                values,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="exact",
            )
            wilcoxon_statistic, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        seed_test_rows.append(
            {
                "contrast": contrast,
                "focal_variant": focal,
                "baseline_variant": baseline,
                "metric": metric,
                "seed_count": len(values),
                "mean_difference": float(values.mean()),
                "std_difference": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "positive_seed_count": int(np.sum(values > 0.0)),
                "paired_t_statistic": float(t_result.statistic),
                "paired_t_p_two_sided": float(t_result.pvalue),
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_exact_p_two_sided": wilcoxon_p,
                "difference_direction": "baseline_minus_focal",
            }
        )
    seed_tests = pd.DataFrame(seed_test_rows)
    seed_tests.to_csv(root / "comparisons/development_seed_level_tests.csv", index=False)

    bootstrap_rows = []
    for (contrast, focal, baseline, metric), group in differences.groupby(
        ["contrast", "focal_variant", "baseline_variant", "metric"],
        sort=True,
    ):
        seed_average = (
            group.groupby(
                ["fold", "surface_pair_id", "current_snapshot_time_utc", "trading_day"],
                as_index=False,
            )["difference"]
            .mean()
        )
        stable_offset = int(hashlib.sha256(f"{contrast}|{metric}".encode()).hexdigest()[:8], 16)
        mean_diff, ci_low, ci_high, p_two = _cluster_bootstrap(
            seed_average,
            iterations=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed) + stable_offset,
        )
        bootstrap_rows.append(
            {
                "contrast": contrast,
                "focal_variant": focal,
                "baseline_variant": baseline,
                "metric": metric,
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                "pair_count": len(seed_average),
                "trading_day_clusters": int(seed_average["trading_day"].nunique()),
                "mean_difference": mean_diff,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "p_two_sided": p_two,
                "p_one_sided_focal_better": p_two / 2.0 if mean_diff > 0.0 else 1.0 - p_two / 2.0,
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "bootstrap_seed": int(args.bootstrap_seed),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["p_holm_within_contrast"] = np.nan
    for _contrast, indexes in bootstrap.groupby("contrast").groups.items():
        index_list = list(indexes)
        bootstrap.loc[index_list, "p_holm_within_contrast"] = _holm_adjust(
            bootstrap.loc[index_list, "p_two_sided"].tolist()
        )
    bootstrap.to_csv(root / "comparisons/development_cluster_bootstrap_ci.csv", index=False)

    primary = bootstrap[
        (bootstrap["contrast"] == "incremental_text")
        & (bootstrap["metric"] == "surface_mae")
    ].copy()
    primary.to_csv(root / "final_tables/development_rq1_primary_test.csv", index=False)
    bootstrap.to_csv(root / "final_tables/development_rq1_point_metric_contrasts.csv", index=False)
    model_summary.to_csv(root / "final_tables/development_rq1_metrics_by_fold_seed.csv", index=False)
    model_summary[
        ["variant", "fold", "seed", "n_pairs", *PROBABILISTIC_METRICS]
    ].to_csv(root / "final_tables/development_rq1_probabilistic_metrics.csv", index=False)
    model_summary[
        ["variant", "fold", "seed", "n_pairs", *FINANCIAL_METRICS]
    ].to_csv(root / "final_tables/development_rq1_financial_consistency.csv", index=False)

    status = {
        "status": "ok",
        "development_only": True,
        "claim_limit": "No untouched 2024+ confirmation data are available.",
        "training_runs": int(len(registry)),
        "sample_metric_rows": int(len(samples)),
        "pairwise_difference_rows": int(len(differences)),
        "fold_test_pair_counts": {
            fold: int(spec["counts"][2]) for fold, spec in FOLDS.items()
        },
        "primary_difference_direction": "no_text_error_minus_text_error",
        "positive_means_text_better": True,
    }
    _write_json(root / "validation_summary.json", status)
    readme = f"""# RQ1 Raw-Vol Pair-Level Text Rolling Development

This archive is development evidence only. It does not claim an untouched final
confirmation because the available news workbook ends in 2023.

Surface input is reconstructed directly from raw implied-volatility points
(`surface_model=raw`) using strike interpolation and maturity total-variance
interpolation. It does not use an SVI-calibrated surface.

Primary contrast:

```text
pair_pca_no_text_residual - pair_pca_text_residual_pretrained
positive => matched LP text has lower error
```

The four outer tests are non-overlapping 2023 quarters. Text is pooled once per
surface pair, transformed by fold-train-only PCA-128, and the residual text
models are initialized from the paired fold/seed no-text generator.

Primary table: `final_tables/development_rq1_primary_test.csv`
All point-metric contrasts: `final_tables/development_rq1_point_metric_contrasts.csv`
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    _build_manifest(root)
    return root


def _build_manifest(root: Path) -> Path:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    target = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def package_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    _build_manifest(root)
    output = Path(args.output) if args.output else root.with_suffix(".zip")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--experiment-root", default="")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    prepare.add_argument("--news-workbook", default=str(DEFAULT_NEWS_WORKBOOK))
    prepare.add_argument("--reuse", action="store_true")

    train = subparsers.add_parser("train-matrix")
    train.add_argument("--experiment-root", default="")
    train.add_argument("--resume", action="store_true")

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--experiment-root", default="")

    collect = subparsers.add_parser("collect-checkpoints")
    collect.add_argument("--experiment-root", default="")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--experiment-root", default="")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--experiment-root", default="")
    compare.add_argument("--bootstrap-iterations", type=int, default=10000)
    compare.add_argument("--bootstrap-seed", type=int, default=20260722)

    package = subparsers.add_parser("package")
    package.add_argument("--experiment-root", default="")
    package.add_argument("--output", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        return prepare_experiment(args)
    if args.command == "train-matrix":
        return train_matrix(args)
    if args.command == "monitor":
        return monitor(args)
    if args.command == "collect-checkpoints":
        return collect_checkpoints(args)
    if args.command == "generate":
        return generate_matrix(args)
    if args.command == "compare":
        return build_comparison(args)
    if args.command == "package":
        return package_experiment(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
