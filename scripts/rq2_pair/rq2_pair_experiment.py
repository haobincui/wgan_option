#!/usr/bin/env python3
"""RQ2 continuation-based LP, BoW, and ChatGPT sentiment validation."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
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
from film_wgan.data import create_train_val_bundle  # noqa: E402
from film_wgan.text_transform import FilmWGANTextTransform, sha256_file  # noqa: E402
from scripts.rq2_pair.pair_features import (  # noqa: E402
    PairFeatureArtifacts,
    build_fold_pair_features,
)


DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq2_pair_textbase.yaml"
DEFAULT_SOURCE_RQ1 = ""
DEFAULT_FEATURE_ROOT = (
    ROOT / "data/processed/text_features/rq2/20260625-075653"
)
EXPERIMENT_PREFIX = "rq2_pair_representation_raw_vol_continuation_"

SEEDS = (42, 202, 404)
FOLDS = {
    "2023Q1": {"counts": (1621, 425, 521)},
    "2023Q2": {"counts": (2046, 521, 333)},
    "2023Q3": {"counts": (2567, 333, 365)},
    "2023Q4": {"counts": (2900, 365, 378)},
}

SOURCE_PARENT = "pair_pca_no_text_residual"
SOURCE_NO_TEXT = "pair_pca_no_text_continued"
SOURCE_LP = "pair_pca_text_residual_pretrained"
NEW_BOW = "pair_pca_bow_residual_pretrained"
NEW_SENTIMENT = "pair_sentiment_residual_pretrained"
NEW_VARIANTS = (NEW_BOW, NEW_SENTIMENT)
MODEL_VARIANTS = {
    "continued_no_text": SOURCE_NO_TEXT,
    "lp": SOURCE_LP,
    "bow": NEW_BOW,
    "llm_sentiment": NEW_SENTIMENT,
}
POINT_METRICS = (
    "surface_mae",
    "short_atm_mae",
    "supported_shortest_atm_abs_err",
)
CURRENT_METRICS = {
    "surface_mae": "current_mae",
    "short_atm_mae": "current_atm_short_pure_mae",
    "supported_shortest_atm_abs_err": "current_supported_shortest_atm_abs_err",
}
PRIMARY_CONTRASTS = (
    ("lp", "bow", "lp_vs_bow"),
    ("lp", "llm_sentiment", "lp_vs_llm_sentiment"),
)
SECONDARY_CONTRASTS = (
    ("lp", "continued_no_text", "lp_vs_continued_no_text"),
    ("bow", "continued_no_text", "bow_vs_continued_no_text"),
    ("llm_sentiment", "continued_no_text", "llm_sentiment_vs_continued_no_text"),
    ("bow", "llm_sentiment", "bow_vs_llm_sentiment"),
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _assert_py312() -> None:
    if Path(sys.prefix).name != "py312":
        raise RuntimeError(
            f"Run RQ2 in conda env 'py312'; current prefix is {sys.prefix}."
        )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_results_pipeline_status(
    root: Path,
    *,
    status: str,
    phase: str,
    started_at_utc: str | None = None,
    error: BaseException | None = None,
) -> Path:
    target = root / "registry/results_pipeline_status.json"
    existing: dict[str, Any] = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "experiment_root": str(root),
        "status": status,
        "phase": phase,
        "started_at_utc": (
            started_at_utc
            or str(existing.get("started_at_utc") or now)
        ),
        "updated_at_utc": now,
        "finished_at_utc": (
            now if status in {"completed", "failed"} else None
        ),
        "error_type": type(error).__name__ if error is not None else "",
        "error": str(error) if error is not None else "",
    }
    _write_json(target, payload)
    return target


def _latest_experiment() -> Path:
    candidates = sorted(
        (ROOT / "outputs/experiments").glob(f"{EXPERIMENT_PREFIX}*")
    )
    if not candidates:
        raise FileNotFoundError("No prepared RQ2 pair experiment exists.")
    return candidates[-1]


def _latest_rq1_experiment() -> Path:
    candidates = sorted(
        (ROOT / "outputs/experiments").glob(
            "rq1_pair_text_raw_vol_continuation_*"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed London-time RQ1 continuation experiment exists."
        )
    return candidates[-1]


def _resolve_root(value: str | None, *, create: bool = False) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    if create:
        return (
            ROOT
            / "outputs/experiments"
            / f"{EXPERIMENT_PREFIX}{_utc_timestamp()}"
        )
    return _latest_experiment()


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


def _cli_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float)):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
    else:
        raise TypeError(f"Unsupported override type for {key}: {type(value)}")
    return f"{key}={rendered}"


def _run_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_dir())


def _completed_run(path: Path) -> Path | None:
    candidates = [
        item
        for item in _run_dirs(path)
        if (item / "checkpoints/film_wgan_best.pt").is_file()
        and (item / "checkpoints/film_wgan_final.pt").is_file()
        and (item / "metrics/best_checkpoint.json").is_file()
    ]
    return candidates[-1] if candidates else None


def _link_or_copy(
    source: Path,
    target: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    experiment_root: Path,
    category: str,
) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_sha = sha256_file(source)
        target_sha = sha256_file(target)
        if source_sha != target_sha:
            raise ValueError(f"Existing imported artifact differs from source: {target}")
        method = (
            "hardlink"
            if source.stat().st_dev == target.stat().st_dev
            and source.stat().st_ino == target.stat().st_ino
            else "copy_existing"
        )
    else:
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                raise
            shutil.copy2(source, target)
            method = "copy"
        source_sha = sha256_file(source)
        target_sha = sha256_file(target)
        if source_sha != target_sha:
            raise ValueError(f"Imported artifact SHA mismatch: {source} -> {target}")
    manifest_rows.append(
        {
            "relative_path": str(target.relative_to(experiment_root)),
            "source_path": str(source),
            "category": category,
            "import_method": method,
            "size_bytes": int(target.stat().st_size),
            "sha256": target_sha,
        }
    )
    return method


def _link_tree(
    source: Path,
    target: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    experiment_root: Path,
    category: str,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _link_or_copy(
                path,
                target / path.relative_to(source),
                manifest_rows=manifest_rows,
                experiment_root=experiment_root,
                category=category,
            )


def _source_selected(source_root: Path) -> pd.DataFrame:
    path = source_root / "checkpoint_selection/selected_checkpoints.csv"
    frame = pd.read_csv(path)
    required = {
        "fold",
        "seed",
        "variant",
        "selected_epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "run_dir",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source selected-checkpoint registry is missing: {missing}")
    expected = {
        (fold, seed, variant)
        for fold in FOLDS
        for seed in SEEDS
        for variant in (SOURCE_PARENT, SOURCE_NO_TEXT, SOURCE_LP)
    }
    available = {
        (str(row.fold), int(row.seed), str(row.variant))
        for row in frame.itertuples(index=False)
    }
    missing_runs = sorted(expected - available)
    if missing_runs:
        raise ValueError(f"Source RQ1 is missing required runs: {missing_runs}")
    return frame


def _source_generated(source_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_root / "registry/generate_registry.csv")
    expected = {
        (fold, seed, variant)
        for fold in FOLDS
        for seed in SEEDS
        for variant in (SOURCE_NO_TEXT, SOURCE_LP)
    }
    available = {
        (str(row.fold), int(row.seed), str(row.variant))
        for row in frame.itertuples(index=False)
    }
    missing = sorted(expected - available)
    if missing:
        raise ValueError(f"Source RQ1 generate registry is missing: {missing}")
    return frame


def _imported_checkpoint(root: Path, fold: str, seed: int, variant: str) -> Path:
    return (
        root
        / "inputs/imported_rq1/checkpoints"
        / variant
        / fold
        / f"seed_{seed}/film_wgan_best.pt"
    )


def _imported_result(root: Path, fold: str, seed: int, variant: str) -> Path:
    return (
        root
        / "inputs/imported_rq1/results"
        / variant
        / fold
        / f"seed_{seed}/development_test_json"
    )


def _fold_feature_dir(root: Path, fold: str) -> Path:
    return root / "inputs/folds" / fold / "representations"


def _fold_config(root: Path, fold: str, variant: str) -> Path:
    return root / "inputs/folds" / fold / f"train_{variant}.yaml"


def _fold_counts(root: Path, fold: str) -> tuple[int, int, int]:
    lineage_path = root / "inputs/folds" / fold / "pair_lineage_audit.csv"
    if not lineage_path.is_file():
        return tuple(int(value) for value in FOLDS[fold]["counts"])
    lineage = pd.read_csv(lineage_path, usecols=["split"])
    return tuple(
        int((lineage["split"].astype(str) == split).sum())
        for split in ("train", "val", "test")
    )


def _existing_pair_feature_artifacts(
    output_dir: Path,
) -> PairFeatureArtifacts | None:
    paths = {
        "bow_feature_path": output_dir / "bow_pair_features.csv",
        "bow_vocabulary_path": output_dir / "bow_vocabulary.json",
        "bow_transform_path": output_dir / "bow_text_transform.npz",
        "sentiment_feature_path": output_dir
        / "llm_sentiment_pair_features.csv",
        "sentiment_transform_path": output_dir
        / "llm_sentiment_text_transform.npz",
        "audit_path": output_dir / "pair_feature_audit.csv",
    }
    manifest_path = output_dir / "feature_manifest.json"
    if not manifest_path.is_file() or not all(
        path.is_file() for path in paths.values()
    ):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("sentiment_model_id") != "gpt-5.4-mini"
        or manifest.get("sentiment_prompt_version")
        != "sun2026_zero_shot_chatgpt_v1"
    ):
        return None
    expected_hashes = {
        "bow_feature_path": str(manifest.get("bow_feature_sha256", "")),
        "bow_transform_path": str(manifest.get("bow_transform_sha256", "")),
        "sentiment_feature_path": str(
            manifest.get("sentiment_feature_sha256", "")
        ),
        "sentiment_transform_path": str(
            manifest.get("sentiment_transform_sha256", "")
        ),
    }
    for key, expected in expected_hashes.items():
        if not expected or sha256_file(paths[key]) != expected:
            raise ValueError(
                f"Existing pair feature artifact failed SHA validation: {paths[key]}"
            )
    return PairFeatureArtifacts(**paths)


def _representation_settings(
    root: Path,
    fold: str,
    variant: str,
) -> dict[str, Any]:
    feature_dir = _fold_feature_dir(root, fold)
    if variant == NEW_BOW:
        return {
            "text_embedding_mode": "bow",
            "text_pooling_mode": "bow_log_count_l2",
            "pair_text_feature_path": str(feature_dir / "bow_pair_features.csv"),
            "text_preprocessing_mode": "pca",
            "text_transform_path": str(feature_dir / "bow_text_transform.npz"),
        }
    if variant == NEW_SENTIMENT:
        return {
            "text_embedding_mode": "llm_sentiment",
            "text_pooling_mode": "mean_scores",
            "pair_text_feature_path": str(
                feature_dir / "llm_sentiment_pair_features.csv"
            ),
            "text_preprocessing_mode": "zscore_pad",
            "text_transform_path": str(
                feature_dir / "llm_sentiment_text_transform.npz"
            ),
        }
    raise ValueError(f"Unsupported RQ2 variant: {variant}")


def prepare_experiment(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root, create=True)
    if root.exists() and any(root.iterdir()) and not args.reuse:
        raise FileExistsError(f"Experiment directory is not empty: {root}")
    for relative in (
        "inputs/configs",
        "inputs/data",
        "inputs/folds",
        "inputs/imported_rq1",
        "inputs/text_features",
        "inputs/audit",
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

    source_root = (
        Path(args.source_rq1).expanduser().resolve()
        if str(args.source_rq1).strip()
        else _latest_rq1_experiment().resolve()
    )
    feature_root = Path(args.feature_root).expanduser().resolve()
    source_config = Path(args.config).expanduser().resolve()
    source_validation = json.loads(
        (source_root / "validation_summary.json").read_text(encoding="utf-8")
    )
    if source_validation.get("status") != "ok":
        raise ValueError(f"Source RQ1 validation status is not ok: {source_root}")
    if source_validation.get("news_source_timezone") != "Europe/London":
        raise ValueError(
            "RQ2 requires an RQ1 source rebuilt with "
            "news_source_timezone=Europe/London."
        )
    selected = _source_selected(source_root)
    generated = _source_generated(source_root)
    import_rows: list[dict[str, Any]] = []

    source_workbook = source_root / "inputs/data/merged_vol_rq2_text.xlsx"
    source_news = (
        source_root / "inputs/data/news_with_openai_embeddings_large.xlsx"
    )
    copied_workbook = root / "inputs/data/merged_vol_rq2_text.xlsx"
    copied_news = root / "inputs/data/news_with_openai_embeddings_large.xlsx"
    for source, target, category in (
        (source_config, root / "inputs/configs/train_rq2_pair_textbase_source.yaml", "config"),
        (source_workbook, copied_workbook, "raw_vol_workbook"),
        (source_news, copied_news, "news_workbook"),
        (
            source_root / "checkpoint_selection/selected_checkpoints.csv",
            root / "inputs/imported_rq1/selected_checkpoints_source.csv",
            "rq1_registry",
        ),
        (
            source_root / "checkpoint_selection/paired_stage_validation.csv",
            root / "inputs/imported_rq1/paired_stage_validation_source.csv",
            "rq1_registry",
        ),
        (
            source_root / "registry/generate_registry.csv",
            root / "inputs/imported_rq1/generate_registry_source.csv",
            "rq1_registry",
        ),
        (
            source_root / "validation_summary.json",
            root / "inputs/imported_rq1/validation_summary_source.json",
            "rq1_registry",
        ),
    ):
        _link_or_copy(
            source,
            target,
            manifest_rows=import_rows,
            experiment_root=root,
            category=category,
        )

    sentiment_sources = {
        "llm_sentiment_features.xlsx": "sentiment_scores",
        "llm_sentiment_manifest.json": "sentiment_scores",
        "openai_sentiment_cache.jsonl": "sentiment_raw_cache",
        "bow_features.xlsx": "full_sample_bow_audit_only",
        "bow_manifest.json": "full_sample_bow_audit_only",
        "bow_vocabulary.json": "full_sample_bow_audit_only",
    }
    for filename, category in sentiment_sources.items():
        target_dir = (
            root / "inputs/audit/full_sample_bow"
            if category == "full_sample_bow_audit_only"
            else root / "inputs/text_features"
        )
        _link_or_copy(
            feature_root / filename,
            target_dir / filename,
            manifest_rows=import_rows,
            experiment_root=root,
            category=category,
        )
    sentiment_manifest = json.loads(
        (
            root / "inputs/text_features/llm_sentiment_manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected_sentiment_manifest = {
        "model_id": "gpt-5.4-mini",
        "prompt_version": "sun2026_zero_shot_chatgpt_v1",
        "base_feature_dim": 3,
        "row_count": 14900,
        "api_errors": 0,
    }
    sentiment_manifest_errors = {
        key: {
            "expected": expected,
            "found": sentiment_manifest.get(key),
        }
        for key, expected in expected_sentiment_manifest.items()
        if sentiment_manifest.get(key) != expected
    }
    if sentiment_manifest_errors:
        raise ValueError(
            "Frozen ChatGPT sentiment manifest mismatch: "
            f"{sentiment_manifest_errors}"
        )

    imported_checkpoint_rows: list[dict[str, Any]] = []
    for record in selected.itertuples(index=False):
        variant = str(record.variant)
        if variant not in {SOURCE_PARENT, SOURCE_NO_TEXT, SOURCE_LP}:
            continue
        fold = str(record.fold)
        seed = int(record.seed)
        source_run = Path(record.run_dir)
        actual_source_checkpoint_sha = sha256_file(record.checkpoint_path)
        if actual_source_checkpoint_sha != str(record.checkpoint_sha256):
            raise ValueError(
                f"Source checkpoint registry SHA mismatch: {record.checkpoint_path}"
            )
        if int(record.selected_epoch) <= 10:
            raise ValueError(
                f"Imported source checkpoint must be after epoch 10: "
                f"{variant}/{fold}/seed_{seed}."
            )
        target_checkpoint = _imported_checkpoint(root, fold, seed, variant)
        _link_or_copy(
            Path(record.checkpoint_path),
            target_checkpoint,
            manifest_rows=import_rows,
            experiment_root=root,
            category="rq1_checkpoint",
        )
        target_config = (
            root
            / "inputs/imported_rq1/configs"
            / variant
            / fold
            / f"seed_{seed}/training_resolved_config.yaml"
        )
        _link_or_copy(
            source_run / "metrics/training_resolved_config.yaml",
            target_config,
            manifest_rows=import_rows,
            experiment_root=root,
            category="rq1_resolved_config",
        )
        if variant in {SOURCE_NO_TEXT, SOURCE_LP}:
            source_result = Path(
                generated[
                    (generated["fold"].astype(str) == fold)
                    & (generated["seed"].astype(int) == seed)
                    & (generated["variant"].astype(str) == variant)
                ]["summary_path"].iloc[0]
            ).parent
            _link_tree(
                source_result,
                _imported_result(root, fold, seed, variant),
                manifest_rows=import_rows,
                experiment_root=root,
                category="rq1_test_result",
            )
        imported_checkpoint_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "variant": variant,
                "selected_epoch": int(record.selected_epoch),
                "source_checkpoint_path": str(record.checkpoint_path),
                "source_checkpoint_sha256": str(record.checkpoint_sha256),
                "imported_checkpoint_path": str(target_checkpoint),
                "imported_checkpoint_sha256": sha256_file(target_checkpoint),
            }
        )
    pd.DataFrame(imported_checkpoint_rows).to_csv(
        root / "inputs/imported_rq1/imported_checkpoints.csv",
        index=False,
    )

    base_payload = _read_yaml(source_config)
    training_base = dict(base_payload.get("training") or base_payload)
    generate_base = dict(base_payload.get("generate_result") or {})
    training_base.update(
        data_path=str(copied_workbook),
        news_workbook_path=str(copied_news),
        output_root="",
        checkpoints_path="",
        metrics_path="",
    )
    _write_yaml(
        root / "inputs/configs/train_rq2_pair_textbase.yaml",
        {"training": training_base, "generate_result": generate_base},
    )

    fold_summary: dict[str, Any] = {}
    sentiment_path = root / "inputs/text_features/llm_sentiment_features.xlsx"
    for fold, specification in FOLDS.items():
        source_fold = source_root / "inputs/folds" / fold
        target_fold = root / "inputs/folds" / fold
        for filename in (
            "split_manifest.csv",
            "pair_lineage_audit.csv",
            "text_transform.npz",
            "text_transform_metadata.json",
            "raw_surface_support.json",
        ):
            _link_or_copy(
                source_fold / filename,
                target_fold / ("lp_" + filename if filename.startswith("text_transform") else filename),
                manifest_rows=import_rows,
                experiment_root=root,
                category="rq1_fold_artifact",
            )
        artifacts = (
            _existing_pair_feature_artifacts(_fold_feature_dir(root, fold))
            if args.reuse
            else None
        )
        if artifacts is None:
            artifacts = build_fold_pair_features(
                fold=fold,
                lineage_path=target_fold / "pair_lineage_audit.csv",
                news_workbook_path=copied_news,
                sentiment_feature_path=sentiment_path,
                output_dir=_fold_feature_dir(root, fold),
                input_workbook_path=copied_workbook,
                vocabulary_size=1024,
                output_dim=128,
            )
        lineage = pd.read_csv(target_fold / "pair_lineage_audit.csv")
        actual_counts = tuple(
            int((lineage["split"].astype(str) == split).sum())
            for split in ("train", "val", "test")
        )
        if min(actual_counts) <= 0:
            raise ValueError(
                f"Fold {fold} has an empty train/validation/test split: {actual_counts}."
            )
        for variant in NEW_VARIANTS:
            fold_training = dict(training_base)
            fold_training.update(
                split_manifest_path=str(target_fold / "split_manifest.csv"),
                surface_support_path=str(target_fold / "raw_surface_support.json"),
                **_representation_settings(root, fold, variant),
            )
            config_path = _fold_config(root, fold, variant)
            _write_yaml(
                config_path,
                {"training": fold_training, "generate_result": generate_base},
            )
            if not args.skip_loader_validation:
                bundle = create_train_val_bundle(load_train_config(config_path))
                bundle_counts = (
                    bundle.train_samples,
                    bundle.val_samples,
                    bundle.test_samples,
                )
                if bundle_counts != actual_counts or bundle.embedding_dim != 128:
                    raise ValueError(
                        f"Loader validation failed for {fold}/{variant}: "
                        f"counts={bundle_counts}, embedding_dim={bundle.embedding_dim}."
                    )
        fold_summary[fold] = {
            "train_pairs": actual_counts[0],
            "validation_pairs": actual_counts[1],
            "test_pairs": actual_counts[2],
            "bow_feature_path": str(artifacts.bow_feature_path),
            "bow_feature_sha256": sha256_file(artifacts.bow_feature_path),
            "bow_transform_sha256": sha256_file(artifacts.bow_transform_path),
            "sentiment_feature_path": str(artifacts.sentiment_feature_path),
            "sentiment_feature_sha256": sha256_file(
                artifacts.sentiment_feature_path
            ),
            "sentiment_transform_sha256": sha256_file(
                artifacts.sentiment_transform_path
            ),
        }

    pd.DataFrame(import_rows).to_csv(
        root / "inputs/rq1_import_manifest.csv",
        index=False,
    )
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
    _write_json(
        root / "validation_summary.json",
        {
            "status": "prepared",
            "development_only": True,
            "surface_model": "raw",
            "news_source_timezone": "Europe/London",
            "source_rq1_experiment": str(source_root),
            "source_rq1_validation_status": source_validation.get("status"),
            "folds": fold_summary,
            "seeds": list(SEEDS),
            "new_variants": list(NEW_VARIANTS),
            "expected_new_training_runs": len(FOLDS) * len(SEEDS) * len(NEW_VARIANTS),
            "input_dimension": 128,
            "full_sample_bow_usage": "audit_only_not_training",
        },
    )
    print(root)
    return root


def _training_overrides(
    root: Path,
    *,
    fold: str,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    parent = _imported_checkpoint(root, fold, seed, SOURCE_PARENT)
    if not parent.is_file():
        raise FileNotFoundError(parent)
    return {
        **_representation_settings(root, fold, variant),
        "seed": int(seed),
        "output_root": str(
            root / "training_runs" / variant / fold / f"seed_{seed}"
        ),
        "checkpoints_path": "",
        "metrics_path": "",
        "initial_generator_checkpoint_path": str(parent),
        "parent_text_transform_policy": "dimension_only",
        "freeze_backbone_epochs": 5,
    }


def train_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    selected_folds = [args.fold] if getattr(args, "fold", "") else list(FOLDS)
    selected_seeds = [int(args.seed)] if getattr(args, "seed", None) is not None else list(SEEDS)
    selected_variants = (
        [args.variant] if getattr(args, "variant", "") else list(NEW_VARIANTS)
    )
    registry_rows: list[dict[str, Any]] = []
    for fold in selected_folds:
        for seed in selected_seeds:
            for variant in selected_variants:
                output_root = (
                    root / "training_runs" / variant / fold / f"seed_{seed}"
                )
                completed = _completed_run(output_root)
                status = "reused" if completed is not None else "pending"
                if completed is None:
                    if not args.resume and _run_dirs(output_root):
                        raise RuntimeError(
                            f"Incomplete run exists under {output_root}; use --resume."
                        )
                    overrides = _training_overrides(
                        root,
                        fold=fold,
                        seed=seed,
                        variant=variant,
                    )
                    command = [
                        sys.executable,
                        "scripts/film_wgan/main.py",
                        "train",
                        "--config",
                        str(_fold_config(root, fold, variant)),
                        "--train-only",
                    ]
                    for key, value in overrides.items():
                        command.extend(["--set", _cli_override(key, value)])
                    _run(
                        command,
                        log_path=root
                        / "logs/training"
                        / variant
                        / fold
                        / f"seed_{seed}.log",
                    )
                    completed = _completed_run(output_root)
                    if completed is None:
                        raise RuntimeError(
                            f"Training ended without complete checkpoint: {output_root}"
                        )
                    status = "completed"
                parent = _imported_checkpoint(root, fold, seed, SOURCE_PARENT)
                initialization = json.loads(
                    (completed / "metrics/initialization_audit.json").read_text(
                        encoding="utf-8"
                    )
                )
                registry_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "status": status,
                        "run_dir": str(completed),
                        "parent_checkpoint_path": str(parent),
                        "parent_checkpoint_sha256": sha256_file(parent),
                        "initial_generator_state_sha256": initialization[
                            "initial_generator_state_sha256"
                        ],
                        "initial_critic_state_sha256": initialization[
                            "initial_critic_state_sha256"
                        ],
                        "text_transform_sha256": initialization[
                            "text_transform_sha256"
                        ],
                    }
                )
                if not getattr(args, "no_registry_write", False):
                    pd.DataFrame(registry_rows).to_csv(
                        root / "registry/launch_registry.csv",
                        index=False,
                    )
    if not getattr(args, "no_registry_write", False):
        pd.DataFrame(registry_rows).to_csv(
            root / "registry/launch_registry.csv",
            index=False,
        )
    return root


def monitor(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            for variant in NEW_VARIANTS:
                output_root = (
                    root / "training_runs" / variant / fold / f"seed_{seed}"
                )
                runs = _run_dirs(output_root)
                latest = runs[-1] if runs else None
                status = "pending"
                epoch = 0
                if latest is not None:
                    status = "running_or_incomplete"
                    metrics = latest / "metrics/training_metrics.csv"
                    if metrics.is_file():
                        frame = pd.read_csv(metrics)
                        epoch = int(frame["epoch"].max()) if not frame.empty else 0
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
        print(
            active[
                ["fold", "seed", "variant", "latest_epoch", "run_dir"]
            ].to_string(index=False)
        )
    return root


def collect_checkpoints(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    source = pd.read_csv(
        root / "inputs/imported_rq1/imported_checkpoints.csv"
    )
    source_lookup = source.set_index(["fold", "seed", "variant"])
    source_stage_audit = pd.read_csv(
        root / "inputs/imported_rq1/paired_stage_validation_source.csv"
    )
    rows: list[dict[str, Any]] = []
    initialization_rows: list[dict[str, Any]] = []
    four_branch_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            parent = source_lookup.loc[(fold, seed, SOURCE_PARENT)]
            expected_parent_sha = str(parent["imported_checkpoint_sha256"])
            pair_initialization: list[dict[str, str]] = []
            for variant in NEW_VARIANTS:
                run_root = (
                    root / "training_runs" / variant / fold / f"seed_{seed}"
                )
                run_dir = _completed_run(run_root)
                if run_dir is None:
                    raise FileNotFoundError(
                        f"Missing completed run: {variant}/{fold}/seed_{seed}"
                    )
                best = json.loads(
                    (run_dir / "metrics/best_checkpoint.json").read_text(
                        encoding="utf-8"
                    )
                )
                epoch = int(best["best_epoch"])
                if epoch <= 10:
                    raise ValueError(
                        f"Selected epoch must be >10: {variant}/{fold}/seed_{seed}"
                    )
                if str(best["checkpoint_metric"]) != "val_mae":
                    raise ValueError(
                        f"Unexpected checkpoint metric for {run_dir}: "
                        f"{best['checkpoint_metric']}"
                    )
                resolved_path = run_dir / "metrics/training_resolved_config.yaml"
                payload = _read_yaml(resolved_path)
                training = dict(payload.get("training") or payload)
                expected_settings = _representation_settings(
                    root, fold, variant
                )
                errors = [
                    f"config:{key}"
                    for key, expected in expected_settings.items()
                    if str(training.get(key)) != str(expected)
                ]
                parent_path = Path(
                    str(training["initial_generator_checkpoint_path"])
                )
                parent_sha = sha256_file(parent_path)
                if parent_sha != expected_parent_sha:
                    errors.append("parent_checkpoint_sha256")
                if int(training.get("freeze_backbone_epochs", -1)) != 5:
                    errors.append("freeze_backbone_epochs")
                if (
                    str(training.get("parent_text_transform_policy"))
                    != "dimension_only"
                ):
                    errors.append("parent_text_transform_policy")
                initialization = json.loads(
                    (run_dir / "metrics/initialization_audit.json").read_text(
                        encoding="utf-8"
                    )
                )
                if int(initialization["embedding_dim"]) != 128:
                    errors.append("embedding_dim")
                if errors:
                    raise ValueError(
                        f"Controlled config validation failed for {run_dir}: {errors}"
                    )
                checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
                row = {
                    "fold": fold,
                    "seed": seed,
                    "variant": variant,
                    "selected_epoch": epoch,
                    "validation_surface_mae": float(best["best_metric"]),
                    "checkpoint_metric": str(best["checkpoint_metric"]),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "run_dir": str(run_dir),
                    "parent_checkpoint_path": str(parent_path),
                    "parent_checkpoint_sha256": parent_sha,
                    "pair_text_feature_path": str(
                        training["pair_text_feature_path"]
                    ),
                    "pair_text_feature_sha256": sha256_file(
                        training["pair_text_feature_path"]
                    ),
                    "text_transform_path": str(training["text_transform_path"]),
                    "text_transform_sha256": sha256_file(
                        training["text_transform_path"]
                    ),
                    "initial_generator_state_sha256": initialization[
                        "initial_generator_state_sha256"
                    ],
                    "initial_critic_state_sha256": initialization[
                        "initial_critic_state_sha256"
                    ],
                }
                rows.append(row)
                pair_initialization.append(
                    {
                        "variant": variant,
                        "generator": row[
                            "initial_generator_state_sha256"
                        ],
                        "critic": row["initial_critic_state_sha256"],
                    }
                )
            generator_equal = len(
                {row["generator"] for row in pair_initialization}
            ) == 1
            critic_equal = len(
                {row["critic"] for row in pair_initialization}
            ) == 1
            initialization_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "parent_checkpoint_sha256": expected_parent_sha,
                    "generator_initialization_equal": generator_equal,
                    "critic_initialization_equal": critic_equal,
                    "status": (
                        "ok" if generator_equal and critic_equal else "failed"
                    ),
                    "details": json.dumps(pair_initialization),
                }
            )
            if not generator_equal or not critic_equal:
                raise ValueError(
                    f"BoW/sentiment initialization mismatch for {fold}/seed_{seed}."
                )
            expected_generator_hash = pair_initialization[0]["generator"]
            expected_critic_hash = pair_initialization[0]["critic"]
            for source_variant in (SOURCE_NO_TEXT, SOURCE_LP):
                audit = source_stage_audit[
                    (source_stage_audit["fold"].astype(str) == fold)
                    & (source_stage_audit["seed"].astype(int) == seed)
                    & (
                        source_stage_audit["variant"].astype(str)
                        == source_variant
                    )
                ]
                errors: list[str] = []
                if len(audit) != 1 or str(audit.iloc[0]["status"]) != "ok":
                    errors.append("source_paired_stage_audit")
                elif (
                    str(audit.iloc[0]["expected_parent_checkpoint_sha256"])
                    != expected_parent_sha
                ):
                    errors.append("source_parent_checkpoint_sha256")
                source_config_path = (
                    root
                    / "inputs/imported_rq1/configs"
                    / source_variant
                    / fold
                    / f"seed_{seed}/training_resolved_config.yaml"
                )
                source_payload = _read_yaml(source_config_path)
                source_training = dict(
                    source_payload.get("training") or source_payload
                )
                expected_source_fields = {
                    "seed": seed,
                    "conditioning_mode": "residual_film",
                    "critic_conditioning_mode": "projection",
                    "freeze_backbone_epochs": 5,
                    "text_preprocessing_mode": "pca",
                    "text_pca_components": 128,
                }
                errors.extend(
                    f"source_config:{key}"
                    for key, expected in expected_source_fields.items()
                    if source_training.get(key) != expected
                )
                four_branch_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": source_variant,
                        "parent_checkpoint_sha256": expected_parent_sha,
                        "expected_initial_generator_state_sha256": expected_generator_hash,
                        "expected_initial_critic_state_sha256": expected_critic_hash,
                        "runtime_initialization_hash_observed": False,
                        "evidence": (
                            "imported_rq1_parent_sha_resolved_config_and_paired_stage_audit"
                        ),
                        "status": "ok" if not errors else "failed",
                        "errors": json.dumps(errors),
                    }
                )
                if errors:
                    raise ValueError(
                        f"Imported RQ1 Stage-B audit failed for "
                        f"{source_variant}/{fold}/seed_{seed}: {errors}"
                    )
            for actual in pair_initialization:
                four_branch_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": actual["variant"],
                        "parent_checkpoint_sha256": expected_parent_sha,
                        "expected_initial_generator_state_sha256": actual[
                            "generator"
                        ],
                        "expected_initial_critic_state_sha256": actual[
                            "critic"
                        ],
                        "runtime_initialization_hash_observed": True,
                        "evidence": "runtime_initialization_audit_json",
                        "status": "ok",
                        "errors": "[]",
                    }
                )
    selected = pd.DataFrame(rows)
    if len(selected) != 24:
        raise ValueError(f"Expected 24 new checkpoints, found {len(selected)}.")
    selected.to_csv(
        root / "checkpoint_selection/selected_checkpoints.csv",
        index=False,
    )
    pd.DataFrame(initialization_rows).to_csv(
        root / "checkpoint_selection/initialization_audit.csv",
        index=False,
    )
    four_branch = pd.DataFrame(four_branch_rows)
    if (
        len(four_branch) != 48
        or set(four_branch["status"].astype(str)) != {"ok"}
    ):
        raise ValueError("Four-branch initialization audit is incomplete.")
    four_branch.to_csv(
        root / "checkpoint_selection/four_branch_initialization_audit.csv",
        index=False,
    )
    return root


def generate_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.is_file():
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    selected = pd.read_csv(selected_path)
    registry_rows: list[dict[str, Any]] = []

    for fold in FOLDS:
        expected = _fold_counts(root, fold)[2]
        for seed in SEEDS:
            for model, variant in (
                ("continued_no_text", SOURCE_NO_TEXT),
                ("lp", SOURCE_LP),
            ):
                summary = (
                    _imported_result(root, fold, seed, variant)
                    / "summary.csv"
                )
                if not summary.is_file() or len(pd.read_csv(summary)) != expected:
                    raise ValueError(
                        f"Imported test result mismatch: {variant}/{fold}/seed_{seed}"
                    )
                registry_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "variant": variant,
                        "status": "imported",
                        "summary_path": str(summary),
                        "sample_count": expected,
                    }
                )

    for record in selected.itertuples(index=False):
        run_dir = Path(record.run_dir)
        output_dir = run_dir / "development_test_json"
        summary = output_dir / "summary.csv"
        expected = _fold_counts(root, str(record.fold))[2]
        if summary.is_file() and len(pd.read_csv(summary)) == expected:
            status = "reused"
        else:
            command = [
                sys.executable,
                "scripts/film_wgan/main.py",
                "generate-result",
                "--config",
                str(run_dir / "metrics/training_resolved_config.yaml"),
                "--checkpoint",
                str(record.checkpoint_path),
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
                log_path=root
                / "logs/generate"
                / str(record.variant)
                / str(record.fold)
                / f"seed_{record.seed}.log",
            )
            if not summary.is_file() or len(pd.read_csv(summary)) != expected:
                raise ValueError(
                    f"Generate-result mismatch: "
                    f"{record.variant}/{record.fold}/seed_{record.seed}"
                )
            status = "completed"
        model = "bow" if str(record.variant) == NEW_BOW else "llm_sentiment"
        registry_rows.append(
            {
                "fold": record.fold,
                "seed": int(record.seed),
                "model": model,
                "variant": record.variant,
                "status": status,
                "summary_path": str(summary),
                "sample_count": expected,
            }
        )
        pd.DataFrame(registry_rows).to_csv(
            root / "registry/generate_registry.csv",
            index=False,
        )
    if len(registry_rows) != 48:
        raise ValueError(f"Expected 48 model/fold/seed results, found {len(registry_rows)}.")
    return root


def _holm_adjust(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    count = len(array)
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (count - rank) * float(array[index])),
        )
        adjusted[index] = running
    return adjusted.astype(float).tolist()


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    clusters = [
        group["difference"].to_numpy(dtype=np.float64)
        for _day, group in frame.groupby("trading_day", sort=True)
    ]
    if not clusters:
        raise ValueError("Cluster bootstrap received no trading-day clusters.")
    observed = float(frame["difference"].mean())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    cluster_count = len(clusters)
    for index in range(int(iterations)):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        draw = np.concatenate([clusters[int(item)] for item in selected])
        draws[index] = float(draw.mean())
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (
        float(iterations) + 1.0
    )
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (
        float(iterations) + 1.0
    )
    return {
        "mean_difference": observed,
        "ci_95_lower": float(ci_low),
        "ci_95_upper": float(ci_high),
        "p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
        "p_one_sided_focal_better": p_lower,
    }


def _dm_hac_daily(
    frame: pd.DataFrame,
    *,
    max_lag: int,
) -> dict[str, float | int]:
    daily = (
        frame.groupby("trading_day", sort=True)["difference"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    count = len(daily)
    if count < 2:
        return {
            "daily_observations": count,
            "hac_max_lag": 0,
            "mean_daily_difference": float(daily.mean()) if count else float("nan"),
            "hac_standard_error": float("nan"),
            "dm_hac_statistic": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_focal_better": float("nan"),
        }
    lag_limit = min(int(max_lag), count - 1)
    centered = daily - float(daily.mean())
    gamma_zero = float(np.dot(centered, centered) / count)
    long_run_variance = gamma_zero
    for lag in range(1, lag_limit + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        bartlett_weight = 1.0 - lag / float(lag_limit + 1)
        long_run_variance += 2.0 * bartlett_weight * covariance
    variance_mean = max(long_run_variance, 0.0) / count
    standard_error = float(np.sqrt(variance_mean))
    statistic = (
        float(daily.mean()) / standard_error
        if standard_error > 0.0
        else float("inf") if float(daily.mean()) > 0.0
        else float("-inf") if float(daily.mean()) < 0.0
        else 0.0
    )
    distribution = stats.t(df=count - 1)
    return {
        "daily_observations": count,
        "hac_max_lag": lag_limit,
        "mean_daily_difference": float(daily.mean()),
        "hac_standard_error": standard_error,
        "dm_hac_statistic": statistic,
        "p_two_sided": float(2.0 * distribution.sf(abs(statistic))),
        "p_one_sided_focal_better": float(distribution.sf(statistic)),
    }


def _validate_model_matching(samples: pd.DataFrame) -> None:
    key_columns = ["fold", "seed", "surface_pair_id"]
    expected_keys: pd.DataFrame | None = None
    reference: pd.DataFrame | None = None
    audit_columns = [
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        "current_mae",
        "current_atm_short_pure_mae",
        "current_supported_shortest_atm_abs_err",
    ]
    for model in MODEL_VARIANTS:
        frame = samples[samples["model"] == model].sort_values(
            key_columns
        )
        if frame.duplicated(key_columns).any():
            raise ValueError(f"Duplicate test pair rows for model={model}.")
        keys = frame[key_columns].reset_index(drop=True)
        if expected_keys is None:
            expected_keys = keys
            reference = frame.reset_index(drop=True)
            continue
        if not keys.equals(expected_keys):
            raise ValueError(f"Pair matching failed for model={model}.")
        assert reference is not None
        candidate = frame.reset_index(drop=True)
        for column in audit_columns:
            if column not in reference or column not in candidate:
                raise ValueError(f"Missing matching audit column: {column}")
            if column.endswith("_utc"):
                if not (
                    reference[column].astype(str).to_numpy()
                    == candidate[column].astype(str).to_numpy()
                ).all():
                    raise ValueError(
                        f"Timestamp mismatch for model={model}, column={column}."
                    )
            else:
                maximum = float(
                    np.max(
                        np.abs(
                            reference[column].to_numpy(dtype=np.float64)
                            - candidate[column].to_numpy(dtype=np.float64)
                        )
                    )
                )
                if maximum > 1e-8:
                    raise ValueError(
                        f"Current baseline mismatch for model={model}, "
                        f"column={column}, max_abs={maximum}."
                    )


def _build_pairwise_differences(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for focal, baseline, contrast in (
        *PRIMARY_CONTRASTS,
        *SECONDARY_CONTRASTS,
    ):
        focal_frame = samples[samples["model"] == focal]
        baseline_frame = samples[samples["model"] == baseline]
        keys = ["fold", "seed", "surface_pair_id"]
        merged = focal_frame.merge(
            baseline_frame,
            on=keys,
            suffixes=("_focal", "_baseline"),
            validate="one_to_one",
        )
        if len(merged) != len(focal_frame) or len(merged) != len(
            baseline_frame
        ):
            raise ValueError(f"Incomplete pair matching for {contrast}.")
        for metric in POINT_METRICS:
            frame = merged[
                [
                    *keys,
                    "current_snapshot_time_utc_focal",
                    f"{metric}_focal",
                    f"{metric}_baseline",
                ]
            ].copy()
            frame.columns = [
                *keys,
                "current_snapshot_time_utc",
                "focal_error",
                "baseline_error",
            ]
            frame["contrast"] = contrast
            frame["focal_model"] = focal
            frame["baseline_model"] = baseline
            frame["metric"] = metric
            frame["difference"] = (
                frame["baseline_error"] - frame["focal_error"]
            )
            frame["difference_direction"] = "baseline_minus_focal"
            frame["positive_means_focal_better"] = True
            rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    output["trading_day"] = pd.to_datetime(
        output["current_snapshot_time_utc"],
        utc=True,
    ).dt.date.astype(str)
    return output


def _build_seed_tests(differences: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        differences.groupby(
            ["contrast", "focal_model", "baseline_model", "seed", "metric"],
            as_index=False,
        )
        .agg(
            mean_difference=("difference", "mean"),
            pair_count=("difference", "size"),
        )
    )
    fold_seed = (
        differences.groupby(
            [
                "contrast",
                "focal_model",
                "baseline_model",
                "fold",
                "seed",
                "metric",
            ],
            as_index=False,
        )
        .agg(mean_difference=("difference", "mean"))
    )
    rows: list[dict[str, Any]] = []
    for keys, group in seed_means.groupby(
        ["contrast", "focal_model", "baseline_model", "metric"],
        sort=True,
    ):
        contrast, focal, baseline, metric = keys
        values = group.sort_values("seed")["mean_difference"].to_numpy(
            dtype=np.float64
        )
        t_test = stats.ttest_1samp(values, popmean=0.0)
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
            wilcoxon_statistic = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        matching_folds = fold_seed[
            (fold_seed["contrast"] == contrast)
            & (fold_seed["metric"] == metric)
        ]
        rows.append(
            {
                "contrast": contrast,
                "focal_model": focal,
                "baseline_model": baseline,
                "metric": metric,
                "seed_count": len(values),
                "mean_difference": float(values.mean()),
                "std_difference": float(values.std(ddof=1)),
                "positive_seed_count": int(np.sum(values > 0.0)),
                "fold_seed_count": int(len(matching_folds)),
                "positive_fold_seed_count": int(
                    np.sum(
                        matching_folds["mean_difference"].to_numpy(
                            dtype=np.float64
                        )
                        > 0.0
                    )
                ),
                "paired_t_statistic": float(t_test.statistic),
                "paired_t_p_two_sided": float(t_test.pvalue),
                "paired_t_p_one_sided_focal_better": (
                    float(t_test.pvalue) / 2.0
                    if float(t_test.statistic) > 0.0
                    else 1.0 - float(t_test.pvalue) / 2.0
                ),
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_exact_p_two_sided": wilcoxon_p,
                "difference_direction": "baseline_minus_focal",
            }
        )
    return pd.DataFrame(rows)


def _build_inference_tables(
    differences: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    for keys, group in differences.groupby(
        ["contrast", "focal_model", "baseline_model", "metric"],
        sort=True,
    ):
        contrast, focal, baseline, metric = keys
        seed_average = (
            group.groupby(
                [
                    "fold",
                    "surface_pair_id",
                    "current_snapshot_time_utc",
                    "trading_day",
                ],
                as_index=False,
            )["difference"]
            .mean()
        )
        stable_offset = int(
            hashlib.sha256(f"{contrast}|{metric}".encode()).hexdigest()[:8],
            16,
        )
        inference = _cluster_bootstrap(
            seed_average,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed) + stable_offset,
        )
        bootstrap_rows.append(
            {
                "contrast": contrast,
                "focal_model": focal,
                "baseline_model": baseline,
                "metric": metric,
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                "pair_count": int(len(seed_average)),
                "trading_day_clusters": int(
                    seed_average["trading_day"].nunique()
                ),
                **inference,
                "bootstrap_iterations": int(bootstrap_iterations),
                "bootstrap_seed": int(bootstrap_seed),
            }
        )
        dm_rows.append(
            {
                "contrast": contrast,
                "focal_model": focal,
                "baseline_model": baseline,
                "metric": metric,
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                **_dm_hac_daily(seed_average, max_lag=5),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["holm_family"] = ""
    bootstrap["p_holm_two_sided"] = np.nan
    primary_mask = (
        bootstrap["contrast"].isin(
            [contrast for _focal, _baseline, contrast in PRIMARY_CONTRASTS]
        )
        & (bootstrap["metric"] == "surface_mae")
    )
    bootstrap.loc[primary_mask, "holm_family"] = "primary_surface_lp_vs_two_representations"
    bootstrap.loc[primary_mask, "p_holm_two_sided"] = _holm_adjust(
        bootstrap.loc[primary_mask, "p_two_sided"].tolist()
    )
    for metric in ("short_atm_mae", "supported_shortest_atm_abs_err"):
        mask = bootstrap["metric"] == metric
        bootstrap.loc[mask, "holm_family"] = f"secondary_{metric}_all_contrasts"
        bootstrap.loc[mask, "p_holm_two_sided"] = _holm_adjust(
            bootstrap.loc[mask, "p_two_sided"].tolist()
        )
    context_mask = (
        (bootstrap["metric"] == "surface_mae")
        & ~primary_mask
    )
    bootstrap.loc[context_mask, "holm_family"] = "context_surface_secondary_contrasts"
    bootstrap.loc[context_mask, "p_holm_two_sided"] = _holm_adjust(
        bootstrap.loc[context_mask, "p_two_sided"].tolist()
    )
    return bootstrap, pd.DataFrame(dm_rows)


def _build_persistence_context(
    samples: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_VARIANTS:
        frame = samples[samples["model"] == model]
        for metric in POINT_METRICS:
            current_metric = CURRENT_METRICS[metric]
            working = frame[
                [
                    "fold",
                    "seed",
                    "surface_pair_id",
                    "current_snapshot_time_utc",
                    metric,
                    current_metric,
                ]
            ].copy()
            working["difference"] = (
                working[current_metric] - working[metric]
            )
            working["trading_day"] = pd.to_datetime(
                working["current_snapshot_time_utc"],
                utc=True,
            ).dt.date.astype(str)
            seed_average = (
                working.groupby(
                    [
                        "fold",
                        "surface_pair_id",
                        "current_snapshot_time_utc",
                        "trading_day",
                    ],
                    as_index=False,
                )["difference"]
                .mean()
            )
            stable_offset = int(
                hashlib.sha256(
                    f"persistence|{model}|{metric}".encode()
                ).hexdigest()[:8],
                16,
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "difference_direction": "current_minus_model",
                    "positive_means_model_better_than_persistence": True,
                    "mean_model_error": float(frame[metric].mean()),
                    "mean_current_error": float(frame[current_metric].mean()),
                    **_cluster_bootstrap(
                        seed_average,
                        iterations=bootstrap_iterations,
                        seed=bootstrap_seed + stable_offset,
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output["p_holm_two_sided_within_metric"] = np.nan
    for _metric, indexes in output.groupby("metric").groups.items():
        index_list = list(indexes)
        output.loc[
            index_list, "p_holm_two_sided_within_metric"
        ] = _holm_adjust(output.loc[index_list, "p_two_sided"].tolist())
    return output


def _build_manifest(root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    target = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def build_comparison(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    (root / "comparisons").mkdir(parents=True, exist_ok=True)
    (root / "final_tables").mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry/generate_registry.csv"
    if not registry_path.is_file():
        raise FileNotFoundError("Run generate before compare.")
    registry = pd.read_csv(registry_path)
    rows: list[pd.DataFrame] = []
    for record in registry.itertuples(index=False):
        frame = pd.read_csv(record.summary_path)
        frame["fold"] = str(record.fold)
        frame["seed"] = int(record.seed)
        frame["model"] = str(record.model)
        frame["variant"] = str(record.variant)
        rows.append(frame)
    samples = pd.concat(rows, ignore_index=True)
    expected = (
        len(MODEL_VARIANTS)
        * len(SEEDS)
        * sum(_fold_counts(root, fold)[2] for fold in FOLDS)
    )
    if len(samples) != expected:
        raise ValueError(
            f"Combined sample metrics row mismatch: {len(samples)} != {expected}."
        )
    _validate_model_matching(samples)
    samples.to_csv(
        root / "comparisons/development_rq2_test_sample_metrics.csv",
        index=False,
    )

    model_overall = (
        samples.groupby(["model", "variant", "fold", "seed"], as_index=False)
        .agg(
            n_pairs=("surface_pair_id", "size"),
            surface_mae=("surface_mae", "mean"),
            short_atm_mae=("short_atm_mae", "mean"),
            supported_shortest_atm_abs_err=(
                "supported_shortest_atm_abs_err",
                "mean",
            ),
            current_surface_mae=("current_mae", "mean"),
            current_short_atm_mae=("current_atm_short_pure_mae", "mean"),
            current_supported_shortest_atm_abs_err=(
                "current_supported_shortest_atm_abs_err",
                "mean",
            ),
        )
    )
    model_overall.to_csv(
        root / "final_tables/development_rq2_model_overall_metrics.csv",
        index=False,
    )

    differences = _build_pairwise_differences(samples)
    differences.to_csv(
        root / "comparisons/development_rq2_pairwise_differences.csv",
        index=False,
    )
    seed_tests = _build_seed_tests(differences)
    seed_tests.to_csv(
        root / "comparisons/development_rq2_seed_level_tests.csv",
        index=False,
    )
    bootstrap, dm_tests = _build_inference_tables(
        differences,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    bootstrap.to_csv(
        root / "comparisons/development_rq2_cluster_bootstrap_ci.csv",
        index=False,
    )
    dm_tests.to_csv(
        root / "comparisons/development_rq2_dm_hac_tests.csv",
        index=False,
    )
    persistence = _build_persistence_context(
        samples,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    primary_names = {
        contrast for _focal, _baseline, contrast in PRIMARY_CONTRASTS
    }
    primary = bootstrap[
        bootstrap["contrast"].isin(primary_names)
        & (bootstrap["metric"] == "surface_mae")
    ].copy()
    if len(primary) != 2:
        raise ValueError(f"Expected two RQ2 primary tests, found {len(primary)}.")
    primary.to_csv(
        root / "final_tables/development_rq2_primary_lp_vs_baselines.csv",
        index=False,
    )
    incremental_names = {
        "lp_vs_continued_no_text",
        "bow_vs_continued_no_text",
        "llm_sentiment_vs_continued_no_text",
    }
    bootstrap[
        bootstrap["contrast"].isin(incremental_names)
    ].to_csv(
        root / "final_tables/development_rq2_incremental_value_vs_no_text.csv",
        index=False,
    )
    persistence.to_csv(
        root / "final_tables/development_rq2_vs_persistence.csv",
        index=False,
    )

    full_support = bool(
        (primary["mean_difference"] > 0.0).all()
        and (primary["ci_95_lower"] > 0.0).all()
        and (primary["p_holm_two_sided"] < 0.05).all()
    )
    supported_count = int(
        (
            (primary["mean_difference"] > 0.0)
            & (primary["ci_95_lower"] > 0.0)
            & (primary["p_holm_two_sided"] < 0.05)
        ).sum()
    )
    support = (
        "full_support"
        if full_support
        else "partial_support"
        if supported_count == 1
        else "not_supported"
    )
    result_summary = {
        "status": "ok",
        "evidence_status": "2023_rolling_development",
        "claim_limit": "Not an untouched 2024+ confirmation experiment.",
        "surface_model": "raw",
        "news_source_timezone": "Europe/London",
        "models": list(MODEL_VARIANTS),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "new_training_runs": 24,
        "comparison_sample_rows": int(len(samples)),
        "test_surface_pairs_per_seed": int(
            sum(_fold_counts(root, fold)[2] for fold in FOLDS)
        ),
        "primary_difference_direction": "representation_error_minus_lp_error",
        "positive_means_lp_better": True,
        "primary_holm_family_size": 2,
        "h2_support": support,
        "primary_results": primary.to_dict(orient="records"),
    }
    _write_json(
        root / "final_tables/development_rq2_result_summary.json",
        result_summary,
    )
    _write_json(
        root / "validation_summary.json",
        {
            **result_summary,
            "selected_checkpoint_count": int(
                len(
                    pd.read_csv(
                        root
                        / "checkpoint_selection/selected_checkpoints.csv"
                    )
                )
            ),
            "generate_registry_rows": int(len(registry)),
            "input_dimension": 128,
            "full_sample_bow_usage": "audit_only_not_training",
            "four_branch_initialization_audit": (
                "checkpoint_selection/four_branch_initialization_audit.csv"
            ),
        },
    )
    readme = """# RQ2 Raw-Vol Continuation Representation Validation

This archive is 2023 rolling-development evidence, not untouched 2024+
confirmation. It compares LP semantic embeddings with fold-train-only BoW
log-count features and frozen ChatGPT sentiment scores.

All Stage-B branches use the same raw-vol surfaces, rolling folds, seeds,
Stage-A no-text parent generator, fresh critic initialization, five frozen
backbone epochs, losses, and training budget.

Primary differences:

```text
BoW surface MAE - LP surface MAE
sentiment surface MAE - LP surface MAE
positive => LP has lower error
```

BoW uses a 1024-term unigram/bigram vocabulary fitted only on unique articles
from each fold's training pairs. Pair vectors are
`L2(log1p(sum article term counts))`, followed by fold-train-only PCA-128.
The archived historical full-sample BoW vocabulary is audit-only.

ChatGPT sentiment uses the frozen `gpt-5.4-mini` scores from prompt
`sun2026_zero_shot_chatgpt_v1`: macroeconomic uncertainty, institutional
action, and risk-off intensity. Scores are averaged within each pair, z-scored
using fold training pairs only, then zero-padded from 3 to 128 dimensions.

Primary table:
`final_tables/development_rq2_primary_lp_vs_baselines.csv`

Incremental value versus controlled continued no-text:
`final_tables/development_rq2_incremental_value_vs_no_text.csv`

Persistence context:
`final_tables/development_rq2_vs_persistence.csv`
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    _build_manifest(root)
    return root


def results_pipeline(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    started_at = datetime.now(timezone.utc).isoformat()
    phase = "collect_checkpoints"
    _write_results_pipeline_status(
        root,
        status="running",
        phase=phase,
        started_at_utc=started_at,
    )
    try:
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
        phase = "generate_test_results"
        _write_results_pipeline_status(
            root,
            status="running",
            phase=phase,
            started_at_utc=started_at,
        )
        generate_matrix(argparse.Namespace(experiment_root=str(root)))
        phase = "build_comparison"
        _write_results_pipeline_status(
            root,
            status="running",
            phase=phase,
            started_at_utc=started_at,
        )
        result = build_comparison(
            argparse.Namespace(
                experiment_root=str(root),
                bootstrap_iterations=int(args.bootstrap_iterations),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
    except Exception as exc:
        _write_results_pipeline_status(
            root,
            status="failed",
            phase=phase,
            started_at_utc=started_at,
            error=exc,
        )
        raise
    _write_results_pipeline_status(
        root,
        status="completed",
        phase="completed",
        started_at_utc=started_at,
    )
    return result


def monitor_results(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    registry_path = root / "registry/generate_registry.csv"
    result_path = (
        root / "final_tables/development_rq2_result_summary.json"
    )
    pipeline_status_path = root / "registry/results_pipeline_status.json"
    selected_count = (
        len(pd.read_csv(selected_path)) if selected_path.is_file() else 0
    )
    generated_count = 0
    generated_samples = 0
    if registry_path.is_file():
        registry = pd.read_csv(registry_path)
        generated_count = len(registry)
        generated_samples = int(registry["sample_count"].sum())
    status = (
        json.loads(result_path.read_text(encoding="utf-8")).get("status")
        if result_path.is_file()
        else "pending"
    )
    pipeline_status = (
        json.loads(pipeline_status_path.read_text(encoding="utf-8"))
        if pipeline_status_path.is_file()
        else {}
    )
    required_outputs = (
        "comparisons/development_rq2_test_sample_metrics.csv",
        "comparisons/development_rq2_pairwise_differences.csv",
        "comparisons/development_rq2_cluster_bootstrap_ci.csv",
        "comparisons/development_rq2_dm_hac_tests.csv",
        "comparisons/development_rq2_seed_level_tests.csv",
        "final_tables/development_rq2_primary_lp_vs_baselines.csv",
        "final_tables/development_rq2_incremental_value_vs_no_text.csv",
        "final_tables/development_rq2_vs_persistence.csv",
        "final_tables/development_rq2_result_summary.json",
        "validation_summary.json",
        "manifest.csv",
    )
    missing_outputs = [
        relative
        for relative in required_outputs
        if not (root / relative).is_file()
    ]
    payload = {
        "experiment_root": str(root),
        "pipeline_status": pipeline_status.get("status", "not_started"),
        "pipeline_phase": pipeline_status.get("phase", "not_started"),
        "pipeline_updated_at_utc": pipeline_status.get(
            "updated_at_utc", ""
        ),
        "pipeline_error": pipeline_status.get("error", ""),
        "selected_new_checkpoints": selected_count,
        "expected_selected_new_checkpoints": 24,
        "generated_model_fold_seed_results": generated_count,
        "expected_generated_model_fold_seed_results": 48,
        "generated_sample_rows": generated_samples,
        "expected_generated_sample_rows": (
            len(MODEL_VARIANTS)
            * len(SEEDS)
            * sum(_fold_counts(root, fold)[2] for fold in FOLDS)
        ),
        "comparison_status": status,
        "required_output_count": len(required_outputs),
        "missing_outputs": missing_outputs,
        "outputs_complete": not missing_outputs,
    }
    _write_json(root / "registry/results_monitor_status.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return root


def package_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    _build_manifest(root)
    output = Path(args.output).expanduser() if args.output else root.with_suffix(".zip")
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
                archive.write(
                    path,
                    arcname=str(Path(root.name) / path.relative_to(root)),
                )
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--experiment-root", default="")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.add_argument("--source-rq1", default=str(DEFAULT_SOURCE_RQ1))
    prepare.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    prepare.add_argument("--reuse", action="store_true")
    prepare.add_argument("--skip-loader-validation", action="store_true")

    train = subparsers.add_parser("train-matrix")
    train.add_argument("--experiment-root", default="")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--fold", choices=list(FOLDS), default="")
    train.add_argument("--seed", type=int, choices=list(SEEDS), default=None)
    train.add_argument("--variant", choices=list(NEW_VARIANTS), default="")
    train.add_argument(
        "--no-registry-write",
        action="store_true",
        help="Skip shared registry writes for externally parallelized single-run jobs.",
    )

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

    pipeline = subparsers.add_parser("results-pipeline")
    pipeline.add_argument("--experiment-root", default="")
    pipeline.add_argument("--bootstrap-iterations", type=int, default=10000)
    pipeline.add_argument("--bootstrap-seed", type=int, default=20260722)

    results_monitor = subparsers.add_parser("monitor-results")
    results_monitor.add_argument("--experiment-root", default="")

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
    if args.command == "results-pipeline":
        return results_pipeline(args)
    if args.command == "monitor-results":
        return monitor_results(args)
    if args.command == "package":
        return package_experiment(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
