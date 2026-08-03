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
from film_wgan.support import parse_raw_surface_params  # noqa: E402
from film_wgan.text_transform import sha256_file  # noqa: E402
from film_wgan.text_lineage import write_text_lineage_artifacts  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq1_pair_textbase.yaml"
DEFAULT_WORKBOOK = (
    ROOT
    / "data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx"
)
DEFAULT_NEWS_WORKBOOK = ROOT / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
SUMMARY_DOCUMENT = ROOT / "docs/summary/rq_research_logic_and_methodology_review_20260722.md"
EXPERIMENT_PREFIX = "rq1_pair_text_raw_vol_continuation_"
EXPECTED_SURFACE_MODEL = "raw"
EXPECTED_NEWS_SOURCE_TIMEZONE = "Europe/London"
SHORT_MATURITY_CANDIDATES = (7.0, 14.0, 21.0, 30.0, 45.0)
SHORT_ATM_MIN_TRAIN_PAIR_COVERAGE = 0.80
SHORT_ATM_MIN_TRAIN_FOUR_CELL_COVERAGE = 0.65
SHORT_ATM_MAX_TRAIN_CELL_SHARE = 0.80
RAW_MATURITY_QUANTILES = (
    0.0,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    1.0,
)

SEED_SET_VERSION = "rq1_15_seed_v1"
SEED_GENERATION_MASTER_SEED = 20260722
# Keep the original three seeds, then append 12 values drawn once from the
# preregistered master seed. The explicit tuple is the reproducibility record.
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
    880839647,
    2131106188,
    2137949052,
    1340168857,
    1682828188,
)
PARENT_VARIANT = "pair_pca_no_text_residual"
CONTINUATION_VARIANT = "pair_pca_no_text_continued"
TEXT_RESIDUAL_VARIANT = "pair_pca_text_residual_pretrained"
SHUFFLED_RESIDUAL_VARIANT = "pair_pca_shuffled_residual_pretrained"
PAIRED_STAGE_B_VARIANTS = (
    CONTINUATION_VARIANT,
    TEXT_RESIDUAL_VARIANT,
    SHUFFLED_RESIDUAL_VARIANT,
)
VARIANTS = (
    PARENT_VARIANT,
    CONTINUATION_VARIANT,
    TEXT_RESIDUAL_VARIANT,
    SHUFFLED_RESIDUAL_VARIANT,
    "pair_pca_text_full_film",
    "pair_pca_text_concat",
    "pair_l2_text_full_film",
)
VARIANT_STAGES = {
    PARENT_VARIANT: "stage_a_parent",
    CONTINUATION_VARIANT: "stage_b_continuation_control",
    TEXT_RESIDUAL_VARIANT: "stage_b_text_treatment",
    SHUFFLED_RESIDUAL_VARIANT: "stage_b_shuffled_placebo",
    "pair_pca_text_full_film": "single_stage_ablation",
    "pair_pca_text_concat": "single_stage_ablation",
    "pair_l2_text_full_film": "single_stage_ablation",
}
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
POINT_METRICS = (
    "surface_mae",
    "short_atm_mae",
    "supported_shortest_atm_abs_err",
)
PROBABILISTIC_METRICS = (
    "energy_score",
    "variogram_score",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "interval_width_50",
    "interval_width_80",
    "interval_width_90",
    "calibration_error",
    "scenario_spread",
    "mc_surface_mae_se",
)
FINANCIAL_METRICS = ("calendar_violation_rate", "butterfly_violation_rate")
CONTRASTS = (
    (TEXT_RESIDUAL_VARIANT, CONTINUATION_VARIANT, "incremental_text"),
    (TEXT_RESIDUAL_VARIANT, PARENT_VARIANT, "text_vs_parent"),
    (CONTINUATION_VARIANT, PARENT_VARIANT, "continuation_effect"),
    (TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT, "matched_vs_shuffled"),
    ("pair_pca_text_full_film", "pair_pca_text_concat", "film_vs_concat"),
    ("pair_pca_text_full_film", "pair_l2_text_full_film", "pca_vs_raw_l2"),
    (TEXT_RESIDUAL_VARIANT, "pair_pca_text_full_film", "residual_package_vs_full_film"),
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


def _raw_maturity_distribution_rows(
    *,
    fold: str,
    split_items: Sequence[tuple[str, Sequence[Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantile_names = (
        "min_days",
        "p05_days",
        "p10_days",
        "p25_days",
        "p50_days",
        "p75_days",
        "p90_days",
        "p95_days",
        "max_days",
    )
    for split, items in split_items:
        side_values: dict[str, list[float]] = {"current": [], "target": []}
        for sample in items:
            current = parse_raw_surface_params(
                sample.metadata.get("current_surface_param_json", "")
            )
            target = parse_raw_surface_params(
                sample.metadata.get("target_surface_param_json", "")
            )
            side_values["current"].extend(
                float(value) for value in current["business_days"]
            )
            side_values["target"].extend(
                float(value) for value in target["business_days"]
            )
        combined = side_values["current"] + side_values["target"]
        for side, values in (*side_values.items(), ("combined", combined)):
            maturity_days = np.asarray(values, dtype=np.float64)
            if maturity_days.size == 0:
                raise ValueError(f"{fold}/{split}/{side} has no raw maturity observations.")
            quantiles = np.quantile(maturity_days, RAW_MATURITY_QUANTILES)
            row: dict[str, Any] = {
                "fold": fold,
                "split": split,
                "surface_side": side,
                "pair_count": len(items),
                "raw_slice_count": int(maturity_days.size),
                "unique_maturity_count": int(np.unique(maturity_days).size),
            }
            row.update(
                {
                    name: float(value)
                    for name, value in zip(quantile_names, quantiles)
                }
            )
            rows.append(row)
    return rows


def _short_atm_support_rows(
    *,
    fold: str,
    split_items: Sequence[tuple[str, Sequence[Any]]],
    atm_range: float,
    selected_max_days: float,
    maturity_candidates: Sequence[float] = SHORT_MATURITY_CANDIDATES,
) -> list[dict[str, Any]]:
    if not any(
        np.isclose(float(candidate), float(selected_max_days))
        for candidate in maturity_candidates
    ):
        raise ValueError(
            f"Selected short maturity {selected_max_days} is not present in "
            f"candidates {tuple(maturity_candidates)}."
        )
    rows: list[dict[str, Any]] = []
    for split, items in split_items:
        if not items:
            raise ValueError(
                f"{fold}/{split} has no samples for short-ATM support audit."
            )
        strike_grid = np.asarray(items[0].strike_grid, dtype=np.float64)
        maturity_grid = np.asarray(items[0].maturity_days_grid, dtype=np.float64)
        support_masks = np.stack(
            [np.asarray(sample.evaluation_support_mask, dtype=bool) for sample in items]
        )
        total_supported_cells = int(support_masks.sum())
        if total_supported_cells <= 0:
            raise ValueError(f"{fold}/{split} has no evaluation-supported raw-vol cells.")
        strike_mask = np.abs(strike_grid - 1.0) <= float(atm_range) + 1e-9
        for max_days in maturity_candidates:
            local_template = (
                maturity_grid[:, None] <= float(max_days) + 1e-9
            ) & strike_mask[None, :]
            local_counts = (support_masks & local_template).sum(axis=(1, 2))
            eligible = local_counts > 0
            four_or_more = local_counts >= 4
            local_supported_cells = int(local_counts.sum())
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "atm_range": float(atm_range),
                    "maturity_max_business_days": float(max_days),
                    "selected": int(
                        np.isclose(float(max_days), float(selected_max_days))
                    ),
                    "pair_count": len(items),
                    "grid_atm_strike_count": int(strike_mask.sum()),
                    "grid_short_maturity_count": int(
                        (maturity_grid <= float(max_days) + 1e-9).sum()
                    ),
                    "grid_local_cell_count": int(local_template.sum()),
                    "total_supported_cell_count": total_supported_cells,
                    "local_supported_cell_count": local_supported_cells,
                    "local_supported_cell_share": (
                        float(local_supported_cells / total_supported_cells)
                    ),
                    "eligible_pair_count": int(eligible.sum()),
                    "eligible_pair_ratio": float(eligible.mean()),
                    "four_cell_pair_count": int(four_or_more.sum()),
                    "four_cell_pair_ratio": float(four_or_more.mean()),
                    "median_local_cells_all_pairs": float(np.median(local_counts)),
                    "median_local_cells_eligible_pairs": (
                        float(np.median(local_counts[eligible]))
                        if bool(eligible.any())
                        else 0.0
                    ),
                }
            )
    return rows


def _validate_selected_short_atm_rows(rows: Sequence[dict[str, Any]]) -> None:
    selected_train_rows = [
        row
        for row in rows
        if row["split"] == "train" and int(row["selected"]) == 1
    ]
    if not selected_train_rows:
        raise ValueError("Short-ATM audit has no selected train-fold rows.")
    failures: list[str] = []
    for row in selected_train_rows:
        fold = str(row["fold"])
        if int(row["grid_local_cell_count"]) <= 0:
            failures.append(f"{fold}: selected short-ATM grid is empty")
        if int(row["local_supported_cell_count"]) >= int(
            row["total_supported_cell_count"]
        ):
            failures.append(f"{fold}: selected short-ATM support equals full support")
        if float(row["eligible_pair_ratio"]) < SHORT_ATM_MIN_TRAIN_PAIR_COVERAGE:
            failures.append(
                f"{fold}: eligible_pair_ratio={row['eligible_pair_ratio']:.3f} "
                f"< {SHORT_ATM_MIN_TRAIN_PAIR_COVERAGE:.3f}"
            )
        if (
            float(row["four_cell_pair_ratio"])
            < SHORT_ATM_MIN_TRAIN_FOUR_CELL_COVERAGE
        ):
            failures.append(
                f"{fold}: four_cell_pair_ratio={row['four_cell_pair_ratio']:.3f} "
                f"< {SHORT_ATM_MIN_TRAIN_FOUR_CELL_COVERAGE:.3f}"
            )
        if (
            float(row["local_supported_cell_share"])
            >= SHORT_ATM_MAX_TRAIN_CELL_SHARE
        ):
            failures.append(
                f"{fold}: local_supported_cell_share="
                f"{row['local_supported_cell_share']:.3f} "
                f">= {SHORT_ATM_MAX_TRAIN_CELL_SHARE:.3f}"
            )
    if failures:
        raise ValueError("Invalid selected short-ATM region: " + "; ".join(failures))


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


def _normalize_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise ValueError("At least one experiment seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Experiment seeds must be unique: {seeds}")
    if any(seed < 0 or seed >= 2**31 for seed in seeds):
        raise ValueError("Experiment seeds must be integers in [0, 2**31).")
    return seeds


def _experiment_seeds(root: Path) -> tuple[int, ...]:
    """Read the frozen seed set while preserving legacy experiment archives."""
    candidate_paths = (
        root / "inputs/experiment_design.json",
        root / "final_tables/development_rq1_result_summary.json",
        root / "validation_summary.json",
    )
    for path in candidate_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("seeds")
        if values:
            return _normalize_seeds(values)
    registry_path = root / "registry/launch_registry.csv"
    if registry_path.is_file():
        registry = pd.read_csv(registry_path, usecols=["seed"])
        values = sorted(int(value) for value in registry["seed"].dropna().unique())
        if values:
            return _normalize_seeds(values)
    return _normalize_seeds(SEEDS)


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


def _fold_counts(root: Path, fold: str) -> tuple[int, int, int]:
    lineage_path = _fold_dir(root, fold) / "pair_lineage_audit.csv"
    if not lineage_path.is_file():
        return tuple(int(value) for value in FOLDS[fold]["counts"])
    lineage = pd.read_csv(lineage_path, usecols=["split"])
    return tuple(
        int((lineage["split"].astype(str) == split).sum())
        for split in ("train", "val", "test")
    )


def _annotate_cross_split_text_duplicates(
    pair_rows: list[dict[str, Any]],
    text_lineage_rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(pair_rows)
    lineage = text_lineage_rows.set_index("sample_id", drop=False)
    definitions = {
        "exact_embedding_duplicate_with_train": "embedding_sha256",
        "exact_text_duplicate_with_train": "lp_text_sha256",
        "near_text_candidate_duplicate_with_train": (
            "near_duplicate_cluster_id"
        ),
    }

    def _pair_values(source_sample_ids: str, column: str) -> set[str]:
        sample_ids = json.loads(str(source_sample_ids))
        values = set()
        for sample_id in sample_ids:
            if sample_id not in lineage.index:
                raise ValueError(f"Missing text-lineage audit row for {sample_id}.")
            value = str(lineage.loc[sample_id, column])
            if value and value != "nan":
                values.add(value)
        return values

    for output_column, lineage_column in definitions.items():
        pair_values = [
            _pair_values(value, lineage_column)
            for value in frame["source_sample_ids"]
        ]
        train_values: set[str] = set()
        for split, values in zip(frame["split"].astype(str), pair_values):
            if split == "train":
                train_values.update(values)
        frame[output_column] = [
            int(split != "train" and bool(values & train_values))
            for split, values in zip(frame["split"].astype(str), pair_values)
        ]
    return frame


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
    if variant == PARENT_VARIANT:
        overrides.update(
            text_embedding_mode="zero_lp",
            lambda_film=0.0,
            lambda_mismatch=0.0,
        )
    elif variant == CONTINUATION_VARIANT:
        if parent_checkpoint is None:
            raise FileNotFoundError("No-text continuation requires its paired no-text checkpoint.")
        overrides.update(
            text_embedding_mode="zero_lp",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
        )
    elif variant == TEXT_RESIDUAL_VARIANT:
        if parent_checkpoint is None:
            raise FileNotFoundError("Residual text training requires its paired no-text checkpoint.")
        overrides.update(
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
        )
    elif variant == SHUFFLED_RESIDUAL_VARIANT:
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
        )
    except ValueError as exc:
        raise ValueError(
            f"Failed to read RQ1 raw-vol workbook sheet {sheet_name!r}: {path}"
        ) from exc
    if "surface_model" not in frame.columns:
        raise ValueError(
            f"RQ1 raw-vol workbook must contain a surface_model column in "
            f"sheet {sheet_name!r}: {path}"
        )
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
    missing_lineage = sorted(
        {"source_timezone", "timestamp_parse_status"} - set(frame.columns)
    )
    if missing_lineage:
        raise ValueError(
            f"RQ1 workbook is missing London-time lineage columns "
            f"{missing_lineage}: {path}"
        )
    timezones = {
        str(value).strip()
        for value in frame["source_timezone"].dropna().tolist()
        if str(value).strip()
    }
    if timezones != {EXPECTED_NEWS_SOURCE_TIMEZONE}:
        raise ValueError(
            "RQ1 workbook must be rebuilt using Factiva "
            f"source_timezone={EXPECTED_NEWS_SOURCE_TIMEZONE!r}; "
            f"found {sorted(timezones)} in {path}."
        )
    parse_statuses = {
        str(value).strip()
        for value in frame["timestamp_parse_status"].dropna().tolist()
        if str(value).strip()
    }
    if parse_statuses != {"ok"}:
        raise ValueError(
            f"RQ1 workbook contains non-ok timestamp parse states: {sorted(parse_statuses)}."
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


def _stage_registry_fields(
    variant: str,
    *,
    parent_checkpoint: Path | None,
    parent_checkpoint_sha256: str,
) -> dict[str, str]:
    if variant not in VARIANT_STAGES:
        raise ValueError(f"Missing training-stage metadata for variant: {variant}")
    is_stage_b = variant in PAIRED_STAGE_B_VARIANTS
    if is_stage_b and parent_checkpoint is None:
        raise FileNotFoundError(f"{variant} requires its paired Stage-A no-text checkpoint.")
    return {
        "training_stage": VARIANT_STAGES[variant],
        "parent_variant": PARENT_VARIANT if is_stage_b else "",
        "parent_checkpoint_path": str(parent_checkpoint) if is_stage_b else "",
        "parent_checkpoint_sha256": parent_checkpoint_sha256 if is_stage_b else "",
    }


def prepare_experiment(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root, create=True)
    experiment_seeds = _normalize_seeds(
        getattr(args, "seeds", None) or SEEDS
    )
    if root.exists() and any(root.iterdir()) and not args.reuse:
        raise FileExistsError(f"Experiment directory is not empty: {root}")
    for relative in (
        "inputs/configs",
        "inputs/data",
        "inputs/folds",
        "inputs/docs",
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
    text_audit_paths = write_text_lineage_artifacts(
        news_workbook_path=copied_news,
        output_dir=root / "inputs/audit/text_lineage",
    )
    text_lineage_rows = pd.read_csv(text_audit_paths["row_audit"])
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
    _write_json(
        root / "inputs/experiment_design.json",
        {
            "seed_set_version": (
                SEED_SET_VERSION
                if experiment_seeds == tuple(SEEDS)
                else "custom"
            ),
            "seed_generation_master_seed": SEED_GENERATION_MASTER_SEED,
            "seeds": list(experiment_seeds),
            "seed_count": len(experiment_seeds),
            "fold_count": len(FOLDS),
            "variant_count": len(VARIANTS),
            "expected_training_runs": (
                len(FOLDS) * len(experiment_seeds) * len(VARIANTS)
            ),
        },
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
    maturity_audit_rows: list[dict[str, Any]] = []
    short_atm_audit_rows: list[dict[str, Any]] = []
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
        surface_support_path = fold_root / "raw_surface_support.json"
        fold_training = dict(training)
        fold_training.update(
            split_manifest_path=str(manifest_path),
            text_transform_path=str(transform_path),
            surface_support_path=str(surface_support_path),
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
        fold_config = load_train_config(fold_config_path)
        bundle = create_train_val_bundle(fold_config)
        actual_counts = (bundle.train_samples, bundle.val_samples, bundle.test_samples)
        if min(actual_counts) <= 0:
            raise ValueError(
                f"Rolling fold {fold} has an empty train/validation/test split: {actual_counts}."
            )
        split_items = (
            ("train", bundle.train_items),
            ("val", bundle.val_items),
            ("test", bundle.test_items),
        )
        maturity_audit_rows.extend(
            _raw_maturity_distribution_rows(
                fold=fold,
                split_items=split_items,
            )
        )
        fold_short_atm_rows = _short_atm_support_rows(
            fold=fold,
            split_items=split_items,
            atm_range=float(fold_config.atm_short_range),
            selected_max_days=float(fold_config.atm_short_max_days),
        )
        short_atm_audit_rows.extend(fold_short_atm_rows)
        selected_train_row = next(
            row
            for row in fold_short_atm_rows
            if row["split"] == "train" and int(row["selected"]) == 1
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
                        "excluded_source_sample_ids": json.dumps(
                            sample.metadata.get("excluded_source_sample_ids", [])
                        ),
                        "excluded_text_lineage_reasons": json.dumps(
                            sample.metadata.get("excluded_text_lineage_reasons", [])
                        ),
                        "supported_cell_count": int(
                            sample.metadata.get("evaluation_supported_cell_count", 0)
                        ),
                        "supported_cell_fraction": float(
                            sample.metadata.get("evaluation_supported_fraction", 0.0)
                        ),
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
        pair_lineage_frame = _annotate_cross_split_text_duplicates(
            pair_rows,
            text_lineage_rows,
        )
        pair_lineage_frame.to_csv(
            fold_root / "pair_lineage_audit.csv",
            index=False,
        )
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
            "surface_support_sha256": sha256_file(surface_support_path),
            "surface_support_path": str(surface_support_path),
            "short_atm_range": float(fold_config.atm_short_range),
            "short_maturity_max_business_days": float(
                fold_config.atm_short_max_days
            ),
            "short_atm_train_eligible_pair_ratio": float(
                selected_train_row["eligible_pair_ratio"]
            ),
            "short_atm_train_four_cell_pair_ratio": float(
                selected_train_row["four_cell_pair_ratio"]
            ),
            "short_atm_train_local_supported_cell_share": float(
                selected_train_row["local_supported_cell_share"]
            ),
        }

    _validate_selected_short_atm_rows(short_atm_audit_rows)
    maturity_audit_path = root / "inputs/audit/raw_maturity_distribution_by_fold.csv"
    short_atm_audit_path = root / "inputs/audit/short_atm_support_by_fold.csv"
    pd.DataFrame(maturity_audit_rows).to_csv(maturity_audit_path, index=False)
    pd.DataFrame(short_atm_audit_rows).to_csv(short_atm_audit_path, index=False)
    selected_train_rows = [
        row
        for row in short_atm_audit_rows
        if row["split"] == "train" and int(row["selected"]) == 1
    ]
    short_atm_selection_path = root / "inputs/audit/short_atm_selection.json"
    _write_json(
        short_atm_selection_path,
        {
            "selection_basis": "rolling_fold_train_pairs_only",
            "maturity_unit": "business_days",
            "atm_definition": "abs(moneyness - 1.0) <= atm_range",
            "selected_atm_range": float(training["atm_short_range"]),
            "selected_maturity_max_business_days": float(
                training["atm_short_max_days"]
            ),
            "candidate_maturity_max_business_days": list(
                SHORT_MATURITY_CANDIDATES
            ),
            "acceptance_thresholds": {
                "minimum_train_eligible_pair_ratio": (
                    SHORT_ATM_MIN_TRAIN_PAIR_COVERAGE
                ),
                "minimum_train_four_cell_pair_ratio": (
                    SHORT_ATM_MIN_TRAIN_FOUR_CELL_COVERAGE
                ),
                "maximum_train_local_supported_cell_share": (
                    SHORT_ATM_MAX_TRAIN_CELL_SHARE
                ),
            },
            "selected_train_fold_results": selected_train_rows,
            "maturity_distribution_csv": str(maturity_audit_path),
            "support_audit_csv": str(short_atm_audit_path),
        },
    )

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
            "news_source_timezone": EXPECTED_NEWS_SOURCE_TIMEZONE,
            "text_lineage_manifest": str(text_audit_paths["manifest"]),
            "text_lineage_manifest_sha256": sha256_file(
                text_audit_paths["manifest"]
            ),
            "short_atm_selection": str(short_atm_selection_path),
            "short_atm_selection_sha256": sha256_file(
                short_atm_selection_path
            ),
            "models": list(VARIANTS),
            "seeds": list(experiment_seeds),
            "folds": fold_validation,
            "expected_training_runs": (
                len(FOLDS) * len(experiment_seeds) * len(VARIANTS)
            ),
        },
    )
    print(root)
    return root


def train_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    experiment_seeds = _experiment_seeds(root)
    selected_folds = [args.fold] if getattr(args, "fold", "") else list(FOLDS)
    requested_seed = getattr(args, "seed", None)
    if requested_seed is not None and int(requested_seed) not in experiment_seeds:
        raise ValueError(
            f"Seed {requested_seed} is not registered for this experiment; "
            f"expected one of {experiment_seeds}."
        )
    selected_seeds = (
        [int(requested_seed)]
        if requested_seed is not None
        else list(experiment_seeds)
    )
    selected_variants = (
        [args.variant] if getattr(args, "variant", "") else list(VARIANTS)
    )
    registry_rows: list[dict[str, Any]] = []
    for fold in selected_folds:
        fold_payload = _read_yaml(_fold_config(root, fold))
        for seed in selected_seeds:
            parent_run = _completed_run(
                root
                / "training_runs"
                / PARENT_VARIANT
                / fold
                / f"seed_{seed}"
            )
            parent_checkpoint = (
                parent_run / "checkpoints/film_wgan_best.pt"
                if parent_run is not None
                else None
            )
            parent_checkpoint_sha256 = (
                sha256_file(parent_checkpoint)
                if parent_checkpoint is not None
                else ""
            )
            for variant in selected_variants:
                output_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                completed = _completed_run(output_root)
                if completed is not None:
                    if variant == PARENT_VARIANT:
                        parent_checkpoint = completed / "checkpoints/film_wgan_best.pt"
                        parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
                    registry_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "variant": variant,
                            "status": "reused",
                            "run_dir": str(completed),
                            "checkpoint": str(completed / "checkpoints/film_wgan_best.pt"),
                            **_stage_registry_fields(
                                variant,
                                parent_checkpoint=parent_checkpoint,
                                parent_checkpoint_sha256=parent_checkpoint_sha256,
                            ),
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
                if variant == PARENT_VARIANT:
                    parent_checkpoint = completed / "checkpoints/film_wgan_best.pt"
                    parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
                registry_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "status": "completed",
                        "run_dir": str(completed),
                        "checkpoint": str(completed / "checkpoints/film_wgan_best.pt"),
                        **_stage_registry_fields(
                            variant,
                            parent_checkpoint=parent_checkpoint,
                            parent_checkpoint_sha256=parent_checkpoint_sha256,
                        ),
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
    experiment_seeds = _experiment_seeds(root)
    rows = []
    for fold in FOLDS:
        for seed in experiment_seeds:
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


def _config_artifact_path(value: Any) -> Path | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _cached_sha256(path: Path | None, cache: dict[Path, str]) -> str:
    if path is None or not path.is_file():
        return ""
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def _paired_stage_audit(
    selected: pd.DataFrame,
    resolved_configs: dict[tuple[str, int, str], dict[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected_seeds = _normalize_seeds(seeds or SEEDS)
    selected_by_run = selected.set_index(["fold", "seed", "variant"], verify_integrity=True)
    sha_cache: dict[Path, str] = {}
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_variant_values = {
        CONTINUATION_VARIANT: {
            "text_embedding_mode": "zero_lp",
            "text_alignment_mode": "matched",
            "lambda_film": 0.0,
            "lambda_mismatch": 0.0,
        },
        TEXT_RESIDUAL_VARIANT: {
            "text_embedding_mode": "lp",
            "text_alignment_mode": "matched",
            "lambda_film": 1.0e-4,
            "lambda_mismatch": 0.5,
        },
        SHUFFLED_RESIDUAL_VARIANT: {
            "text_embedding_mode": "lp",
            "text_alignment_mode": "permuted",
            "lambda_film": 1.0e-4,
            "lambda_mismatch": 0.5,
        },
    }
    common_expected = {
        "text_preprocessing_mode": "pca",
        "normalize_text_embedding": False,
        "conditioning_mode": "residual_film",
        "critic_conditioning_mode": "projection",
        "freeze_backbone_epochs": 5,
    }

    for fold in FOLDS:
        for seed in selected_seeds:
            parent_key = (fold, seed, PARENT_VARIANT)
            parent_row = selected_by_run.loc[parent_key]
            expected_parent_path = Path(str(parent_row["checkpoint_path"])).resolve()
            expected_parent_sha = str(parent_row["checkpoint_sha256"])
            parent_payload = resolved_configs[parent_key]
            parent_training = dict(parent_payload.get("training") or parent_payload)
            expected_transform_path = _config_artifact_path(parent_training.get("text_transform_path"))
            expected_transform_sha = _cached_sha256(expected_transform_path, sha_cache)

            for variant in PAIRED_STAGE_B_VARIANTS:
                key = (fold, seed, variant)
                payload = resolved_configs[key]
                training = dict(payload.get("training") or payload)
                configured_parent_path = _config_artifact_path(
                    training.get("initial_generator_checkpoint_path")
                )
                configured_parent_sha = _cached_sha256(configured_parent_path, sha_cache)
                transform_path = _config_artifact_path(training.get("text_transform_path"))
                transform_sha = _cached_sha256(transform_path, sha_cache)
                errors: list[str] = []

                if configured_parent_path != expected_parent_path:
                    errors.append("parent_checkpoint_path_mismatch")
                if configured_parent_sha != expected_parent_sha:
                    errors.append("parent_checkpoint_sha256_mismatch")
                if transform_path != expected_transform_path:
                    errors.append("text_transform_path_mismatch")
                if not expected_transform_sha or transform_sha != expected_transform_sha:
                    errors.append("text_transform_sha256_mismatch")

                expected_values = {**common_expected, **expected_variant_values[variant]}
                mismatched_fields = sorted(
                    field
                    for field, expected in expected_values.items()
                    if training.get(field) != expected
                )
                errors.extend(f"config_mismatch:{field}" for field in mismatched_fields)
                audit_rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "variant": variant,
                        "training_stage": VARIANT_STAGES[variant],
                        "parent_variant": PARENT_VARIANT,
                        "expected_parent_checkpoint_path": str(expected_parent_path),
                        "configured_parent_checkpoint_path": (
                            "" if configured_parent_path is None else str(configured_parent_path)
                        ),
                        "expected_parent_checkpoint_sha256": expected_parent_sha,
                        "configured_parent_checkpoint_sha256": configured_parent_sha,
                        "expected_text_transform_path": (
                            "" if expected_transform_path is None else str(expected_transform_path)
                        ),
                        "configured_text_transform_path": (
                            "" if transform_path is None else str(transform_path)
                        ),
                        "expected_text_transform_sha256": expected_transform_sha,
                        "configured_text_transform_sha256": transform_sha,
                        "status": "ok" if not errors else "failed",
                        "errors": json.dumps(errors),
                    }
                )
                if errors:
                    failures.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "variant": variant,
                            "errors": errors,
                        }
                    )
    return pd.DataFrame(audit_rows), failures


def collect_checkpoints(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    experiment_seeds = _experiment_seeds(root)
    rows = []
    resolved_configs: dict[tuple[str, int, str], dict[str, Any]] = {}
    artifact_sha_cache: dict[Path, str] = {}
    for fold in FOLDS:
        for seed in experiment_seeds:
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
                resolved_payload = _read_yaml(resolved_config_path)
                resolved_configs[(fold, seed, variant)] = resolved_payload
                resolved_training = dict(resolved_payload.get("training") or resolved_payload)
                configured_parent_path = _config_artifact_path(
                    resolved_training.get("initial_generator_checkpoint_path")
                )
                transform_path = _config_artifact_path(resolved_training.get("text_transform_path"))
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "training_stage": VARIANT_STAGES[variant],
                        "parent_variant": PARENT_VARIANT if variant in PAIRED_STAGE_B_VARIANTS else "",
                        "parent_checkpoint_path": (
                            "" if configured_parent_path is None else str(configured_parent_path)
                        ),
                        "parent_checkpoint_sha256": _cached_sha256(
                            configured_parent_path,
                            artifact_sha_cache,
                        ),
                        "text_transform_path": "" if transform_path is None else str(transform_path),
                        "text_transform_sha256": _cached_sha256(
                            transform_path,
                            artifact_sha_cache,
                        ),
                        "selected_epoch": epoch,
                        "validation_surface_mae": float(best["best_metric"]),
                        "checkpoint_metric": str(best["checkpoint_metric"]),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "run_dir": str(run_dir),
                    }
                )
    frame = pd.DataFrame(rows)
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
        reference = resolved_configs[
            (fold, experiment_seeds[0], "pair_pca_no_text_residual")
        ]
        reference = dict(reference.get("training") or reference)
        for seed in experiment_seeds:
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
    paired_stage_audit, paired_stage_failures = _paired_stage_audit(
        frame,
        resolved_configs,
        seeds=experiment_seeds,
    )
    paired_stage_audit.to_csv(
        root / "checkpoint_selection/paired_stage_validation.csv",
        index=False,
    )
    if paired_stage_failures:
        _write_json(
            root / "checkpoint_selection/paired_stage_validation_failures.json",
            paired_stage_failures,
        )
        failures.extend(paired_stage_failures)
    if failures:
        _write_json(
            root / "checkpoint_selection/resolved_config_validation_failures.json",
            failures,
        )
        raise ValueError(
            "Checkpoint/config validation failed; inspect checkpoint_selection validation reports."
        )
    frame.to_csv(root / "checkpoint_selection/selected_checkpoints.csv", index=False)
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
        expected = _fold_counts(root, str(row.fold))[2]
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
    frame = frame.loc[
        np.isfinite(pd.to_numeric(frame["difference"], errors="coerce"))
    ].copy()
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


def _dm_hac_daily(
    frame: pd.DataFrame,
    *,
    max_lag: int = 5,
) -> dict[str, float | int]:
    frame = frame.loc[
        np.isfinite(pd.to_numeric(frame["difference"], errors="coerce"))
    ].copy()
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
            "mean_daily_difference": (
                float(daily.mean()) if count else float("nan")
            ),
            "hac_standard_error": float("nan"),
            "dm_hac_statistic": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_focal_better": float("nan"),
        }
    lag_limit = min(int(max_lag), count - 1)
    centered = daily - float(daily.mean())
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_limit + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        long_run_variance += (
            2.0
            * (1.0 - lag / float(lag_limit + 1))
            * covariance
        )
    standard_error = float(
        np.sqrt(max(long_run_variance, 0.0) / count)
    )
    mean = float(daily.mean())
    statistic = (
        mean / standard_error
        if standard_error > 0.0
        else float("inf") if mean > 0.0
        else float("-inf") if mean < 0.0
        else 0.0
    )
    distribution = stats.t(df=count - 1)
    return {
        "daily_observations": count,
        "hac_max_lag": lag_limit,
        "mean_daily_difference": mean,
        "hac_standard_error": standard_error,
        "dm_hac_statistic": statistic,
        "p_two_sided": float(
            2.0 * distribution.sf(abs(statistic))
        ),
        "p_one_sided_focal_better": float(
            distribution.sf(statistic)
        ),
    }


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


def _write_result_summary(
    root: Path,
    *,
    samples: pd.DataFrame,
    bootstrap: pd.DataFrame,
    seed_tests: pd.DataFrame,
) -> None:
    experiment_seeds = _experiment_seeds(root)
    metric_columns = [*POINT_METRICS, *PROBABILISTIC_METRICS, *FINANCIAL_METRICS]
    overall_rows = []
    for variant, group in samples.groupby("variant", sort=True):
        row = {
            "variant": str(variant),
            "sample_rows": int(len(group)),
            "pair_count": int(
                group[["fold", "surface_pair_id"]].drop_duplicates().shape[0]
            ),
            "fold_count": int(group["fold"].nunique()),
            "seed_count": int(group["seed"].nunique()),
        }
        row.update({metric: float(group[metric].mean()) for metric in metric_columns})
        row.update(
            {
                f"{metric}_pair_count": int(group[metric].count())
                for metric in metric_columns
            }
        )
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall_path = root / "final_tables/development_rq1_model_overall_metrics.csv"
    overall.to_csv(overall_path, index=False)

    primary = bootstrap[bootstrap["contrast"] == "incremental_text"].copy()
    primary_seed = seed_tests[seed_tests["contrast"] == "incremental_text"][
        [
            "metric",
            "seed_count",
            "positive_seed_count",
            "paired_t_p_two_sided",
            "wilcoxon_exact_p_two_sided",
        ]
    ]
    primary = primary.merge(primary_seed, on="metric", how="left", validate="one_to_one")
    primary_path = root / "final_tables/development_rq1_primary_all_metrics.csv"
    primary.to_csv(primary_path, index=False)

    summary = {
        "status": "ok",
        "development_only": True,
        "experiment_root": str(root),
        "difference_direction": "no_text_error_minus_text_error",
        "positive_means_text_better": True,
        "folds": list(FOLDS),
        "seeds": list(experiment_seeds),
        "variants": list(VARIANTS),
        "primary_metrics": json.loads(primary.to_json(orient="records")),
        "model_overall_point_metrics": json.loads(
            overall[
                [
                    "variant",
                    "surface_mae",
                    "short_atm_mae",
                    "supported_shortest_atm_abs_err",
                ]
            ].to_json(
                orient="records"
            )
        ),
        "result_files": {
            "primary_all_metrics": str(primary_path),
            "model_overall_metrics": str(overall_path),
            "all_point_metric_contrasts": str(
                root / "final_tables/development_rq1_point_metric_contrasts.csv"
            ),
            "dm_hac_tests": str(
                root / "comparisons/development_dm_hac_tests.csv"
            ),
            "seed_level_tests": str(
                root / "comparisons/development_seed_level_tests.csv"
            ),
            "duplicate_sensitivity": str(
                root
                / "final_tables/development_rq1_duplicate_sensitivity.csv"
            ),
        },
    }
    _write_json(root / "final_tables/development_rq1_result_summary.json", summary)

    primary_columns = [
        "metric",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_two_sided",
        "p_holm_within_contrast",
        "positive_seed_count",
        "seed_count",
    ]
    markdown_lines = [
        "# RQ1 Development Result Summary",
        "",
        "Difference is `no_text_error - text_error`; positive values favor matched LP text.",
        "",
        "| " + " | ".join(primary_columns) + " |",
        "|" + "|".join(["---"] * len(primary_columns)) + "|",
    ]
    for row in primary[primary_columns].itertuples(index=False):
        markdown_lines.append(
            "| "
            + " | ".join(
                str(value) if isinstance(value, str) else f"{float(value):.10g}"
                for value in row
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            "This is rolling 2023 development evidence, not an untouched 2024+ confirmation.",
            "",
        ]
    )
    (root / "final_tables/development_rq1_result_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def build_comparison(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    experiment_seeds = _experiment_seeds(root)
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
    if "alignment_type" not in samples.columns:
        samples["alignment_type"] = "exact"
    samples["alignment_type"] = (
        samples["alignment_type"].fillna("exact").astype(str)
    )
    duplicate_audits = []
    duplicate_columns = [
        "fold",
        "surface_pair_id",
        "exact_embedding_duplicate_with_train",
        "exact_text_duplicate_with_train",
        "near_text_candidate_duplicate_with_train",
    ]
    for fold in FOLDS:
        lineage = pd.read_csv(_fold_dir(root, fold) / "pair_lineage_audit.csv")
        duplicate_audits.append(lineage[duplicate_columns])
    samples = samples.merge(
        pd.concat(duplicate_audits, ignore_index=True),
        on=["fold", "surface_pair_id"],
        how="left",
        validate="many_to_one",
    )
    expected_rows = (
        sum(_fold_counts(root, fold)[2] for fold in FOLDS)
        * len(experiment_seeds)
        * len(VARIANTS)
    )
    if len(samples) != expected_rows:
        raise ValueError(f"Combined development sample count mismatch: {len(samples)} != {expected_rows}.")
    samples_path = root / "comparisons/development_test_sample_metrics.csv"
    samples.to_csv(samples_path, index=False)

    metric_columns = [*POINT_METRICS, *PROBABILISTIC_METRICS, *FINANCIAL_METRICS]
    summary_rows = []
    for keys, group in samples.groupby(["variant", "fold", "seed"], sort=True):
        row = {"variant": keys[0], "fold": keys[1], "seed": int(keys[2]), "n_pairs": len(group)}
        row.update({metric: float(group[metric].mean()) for metric in metric_columns})
        row.update(
            {
                f"{metric}_pair_count": int(group[metric].count())
                for metric in metric_columns
            }
        )
        summary_rows.append(row)
    model_summary = pd.DataFrame(summary_rows)
    model_summary["financial_metric_status"] = (
        "not_reported_for_irregular_raw_support"
    )
    model_summary.to_csv(root / "comparisons/development_model_metrics_by_fold_seed.csv", index=False)
    alignment_model_summary = (
        samples.groupby(
            ["variant", "fold", "seed", "alignment_type"],
            as_index=False,
        )
        .agg(
            n_pairs=("surface_pair_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in metric_columns
            },
            **{
                f"{metric}_pair_count": (metric, "count")
                for metric in metric_columns
            },
        )
    )
    alignment_model_summary.to_csv(
        root / "comparisons/development_alignment_stratum_metrics.csv",
        index=False,
    )

    difference_rows = []
    for focal, baseline, contrast in CONTRASTS:
        left = samples[samples["variant"] == focal]
        right = samples[samples["variant"] == baseline]
        keys = ["fold", "seed", "surface_pair_id"]
        merged = left.merge(right, on=keys, suffixes=("_focal", "_baseline"), validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Pair matching failed for {contrast}.")
        if not merged["alignment_type_focal"].equals(
            merged["alignment_type_baseline"]
        ):
            raise ValueError(
                f"Alignment-stratum matching failed for {contrast}."
            )
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
                        "alignment_type": getattr(
                            row,
                            "alignment_type_focal",
                        ),
                        "metric": metric,
                        "focal_error": float(getattr(row, f"{metric}_focal")),
                        "baseline_error": float(getattr(row, f"{metric}_baseline")),
                        "difference": float(
                            getattr(row, f"{metric}_baseline") - getattr(row, f"{metric}_focal")
                        ),
                        "exact_embedding_duplicate_with_train": int(
                            getattr(
                                row,
                                "exact_embedding_duplicate_with_train_focal",
                            )
                        ),
                        "exact_text_duplicate_with_train": int(
                            getattr(row, "exact_text_duplicate_with_train_focal")
                        ),
                        "near_text_candidate_duplicate_with_train": int(
                            getattr(
                                row,
                                "near_text_candidate_duplicate_with_train_focal",
                            )
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
        .agg(
            mean_difference=("difference", "mean"),
            pair_count=("difference", "count"),
            total_pair_count=("difference", "size"),
        )
    )
    fold_summary.to_csv(root / "comparisons/development_fold_seed_differences.csv", index=False)

    seed_directions = differences.groupby(
        ["contrast", "focal_variant", "baseline_variant", "seed", "metric"],
        as_index=False,
    ).agg(
        mean_difference=("difference", "mean"),
        pair_count=("difference", "count"),
        total_pair_count=("difference", "size"),
    )
    fold_directions = fold_summary.groupby(
        ["contrast", "focal_variant", "baseline_variant", "seed", "metric"],
        as_index=False,
    ).agg(
        positive_fold_count=("mean_difference", lambda values: int(np.sum(np.asarray(values) > 0.0))),
        fold_count=("mean_difference", "count"),
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
        values = (
            group.sort_values("seed")["mean_difference"]
            .dropna()
            .to_numpy(dtype=np.float64)
        )
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
    dm_rows = []
    for (contrast, focal, baseline, metric), group in differences.groupby(
        ["contrast", "focal_variant", "baseline_variant", "metric"],
        sort=True,
    ):
        group = group.loc[
            np.isfinite(pd.to_numeric(group["difference"], errors="coerce"))
        ].copy()
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
        dm_rows.append(
            {
                "contrast": contrast,
                "focal_variant": focal,
                "baseline_variant": baseline,
                "metric": metric,
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                **_dm_hac_daily(seed_average, max_lag=5),
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
    alignment_bootstrap_rows = []
    for (
        contrast,
        focal,
        baseline,
        metric,
        alignment_type,
    ), group in differences.groupby(
        [
            "contrast",
            "focal_variant",
            "baseline_variant",
            "metric",
            "alignment_type",
        ],
        sort=True,
    ):
        group = group.loc[
            np.isfinite(pd.to_numeric(group["difference"], errors="coerce"))
        ].copy()
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
            hashlib.sha256(
                f"{contrast}|{metric}|{alignment_type}".encode()
            ).hexdigest()[:8],
            16,
        )
        mean_diff, ci_low, ci_high, p_two = _cluster_bootstrap(
            seed_average,
            iterations=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed) + stable_offset,
        )
        alignment_bootstrap_rows.append(
            {
                "contrast": contrast,
                "focal_variant": focal,
                "baseline_variant": baseline,
                "metric": metric,
                "alignment_type": alignment_type,
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                "pair_count": int(len(seed_average)),
                "trading_day_clusters": int(
                    seed_average["trading_day"].nunique()
                ),
                "mean_difference": mean_diff,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "p_two_sided": p_two,
                "p_one_sided_focal_better": (
                    p_two / 2.0
                    if mean_diff > 0.0
                    else 1.0 - p_two / 2.0
                ),
            }
        )
    alignment_bootstrap = pd.DataFrame(alignment_bootstrap_rows)
    alignment_bootstrap.to_csv(
        root / "comparisons/development_alignment_stratum_contrasts.csv",
        index=False,
    )
    alignment_bootstrap.to_csv(
        root / "final_tables/development_rq1_alignment_strata.csv",
        index=False,
    )
    pd.DataFrame(dm_rows).to_csv(
        root / "comparisons/development_dm_hac_tests.csv",
        index=False,
    )

    duplicate_sensitivity_rows = []
    incremental = differences[differences["contrast"] == "incremental_text"]
    duplicate_policies = {
        "all_pairs": np.ones(len(incremental), dtype=bool),
        "exclude_exact_embedding_seen_in_train": (
            incremental["exact_embedding_duplicate_with_train"].to_numpy() == 0
        ),
        "exclude_exact_text_seen_in_train": (
            incremental["exact_text_duplicate_with_train"].to_numpy() == 0
        ),
        "exclude_any_exact_duplicate_seen_in_train": (
            (
                incremental["exact_embedding_duplicate_with_train"].to_numpy()
                + incremental["exact_text_duplicate_with_train"].to_numpy()
            )
            == 0
        ),
        "exclude_near_text_seen_in_train": (
            incremental[
                "near_text_candidate_duplicate_with_train"
            ].to_numpy()
            == 0
        ),
        "exclude_any_exact_or_near_duplicate_seen_in_train": (
            (
                incremental["exact_embedding_duplicate_with_train"].to_numpy()
                + incremental["exact_text_duplicate_with_train"].to_numpy()
                + incremental[
                    "near_text_candidate_duplicate_with_train"
                ].to_numpy()
            )
            == 0
        ),
    }
    for policy, keep_mask in duplicate_policies.items():
        policy_frame = incremental.loc[keep_mask].copy()
        for metric, metric_frame in policy_frame.groupby("metric", sort=True):
            metric_frame = metric_frame.loc[
                np.isfinite(
                    pd.to_numeric(metric_frame["difference"], errors="coerce")
                )
            ].copy()
            seed_average = (
                metric_frame.groupby(
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
                hashlib.sha256(f"{policy}|{metric}".encode()).hexdigest()[:8],
                16,
            )
            mean_diff, ci_low, ci_high, p_two = _cluster_bootstrap(
                seed_average,
                iterations=int(args.bootstrap_iterations),
                seed=int(args.bootstrap_seed) + stable_offset,
            )
            duplicate_sensitivity_rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "pair_count": int(len(seed_average)),
                    "trading_day_clusters": int(
                        seed_average["trading_day"].nunique()
                    ),
                    "mean_no_text_minus_text": mean_diff,
                    "ci_95_lower": ci_low,
                    "ci_95_upper": ci_high,
                    "p_two_sided": p_two,
                    "p_one_sided_text_better": (
                        p_two / 2.0 if mean_diff > 0.0 else 1.0 - p_two / 2.0
                    ),
                }
            )
    pd.DataFrame(duplicate_sensitivity_rows).to_csv(
        root / "final_tables/development_rq1_duplicate_sensitivity.csv",
        index=False,
    )

    primary = bootstrap[
        (bootstrap["contrast"] == "incremental_text")
        & (bootstrap["metric"] == "surface_mae")
    ].copy()
    primary.to_csv(root / "final_tables/development_rq1_primary_test.csv", index=False)
    primary.to_csv(
        root / "final_tables/development_rq1_primary_controlled_incremental_text.csv",
        index=False,
    )
    bootstrap[
        bootstrap["contrast"].isin(("text_vs_parent", "continuation_effect"))
    ].to_csv(
        root / "final_tables/development_rq1_parent_continuation_diagnostics.csv",
        index=False,
    )
    bootstrap.to_csv(root / "final_tables/development_rq1_point_metric_contrasts.csv", index=False)
    model_summary.to_csv(root / "final_tables/development_rq1_metrics_by_fold_seed.csv", index=False)
    model_summary[
        ["variant", "fold", "seed", "n_pairs", *PROBABILISTIC_METRICS]
    ].to_csv(root / "final_tables/development_rq1_probabilistic_metrics.csv", index=False)
    model_summary[
        [
            "variant",
            "fold",
            "seed",
            "n_pairs",
            "financial_metric_status",
            *FINANCIAL_METRICS,
        ]
    ].to_csv(root / "final_tables/development_rq1_financial_consistency.csv", index=False)
    _write_result_summary(
        root,
        samples=samples,
        bootstrap=bootstrap,
        seed_tests=seed_tests,
    )

    status = {
        "status": "ok",
        "development_only": True,
        "claim_limit": "No untouched 2024+ confirmation data are available.",
        "training_runs": int(len(registry)),
        "seeds": list(experiment_seeds),
        "seed_count": len(experiment_seeds),
        "expected_training_runs": (
            len(FOLDS) * len(experiment_seeds) * len(VARIANTS)
        ),
        "sample_metric_rows": int(len(samples)),
        "pairwise_difference_rows": int(len(differences)),
        "fold_test_pair_counts": {
            fold: _fold_counts(root, fold)[2] for fold in FOLDS
        },
        "news_source_timezone": EXPECTED_NEWS_SOURCE_TIMEZONE,
        "stage_b_parent_validation": "checkpoint_selection/paired_stage_validation.csv",
        "primary_focal_variant": TEXT_RESIDUAL_VARIANT,
        "primary_baseline_variant": CONTINUATION_VARIANT,
        "primary_difference_direction": "continued_no_text_error_minus_text_error",
        "positive_means_text_better": True,
        "financial_consistency_status": (
            "not_reported_for_irregular_raw_support"
        ),
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
pair_pca_no_text_continued - pair_pca_text_residual_pretrained
positive => matched LP text has lower error
```

The continued no-text and matched/shuffled text branches all start from the
same fold/seed Stage-A no-text generator. They use the same Stage-B epoch
budget, five-epoch backbone freeze, low-learning-rate backbone schedule, fresh
critic behavior, and train-only PCA transform. This controls for improvement
caused only by an additional optimization stage.

The four outer tests are non-overlapping 2023 quarters. Text is pooled once per
surface pair, transformed by fold-train-only PCA-128, and the residual text
models are initialized from the paired fold/seed no-text generator.

Primary table:
`final_tables/development_rq1_primary_controlled_incremental_text.csv`

Parent/continuation diagnostics:
`final_tables/development_rq1_parent_continuation_diagnostics.csv`

Primary three-metric summary: `final_tables/development_rq1_primary_all_metrics.csv`
Machine-readable summary: `final_tables/development_rq1_result_summary.json`
All point-metric contrasts: `final_tables/development_rq1_point_metric_contrasts.csv`
DM/HAC tests: `comparisons/development_dm_hac_tests.csv`
Exact/near-duplicate sensitivity:
`final_tables/development_rq1_duplicate_sensitivity.csv`

Raw-vol metrics are evaluated only on the fold-train observed-support mask.
Unsupported 7-day ATM and broad-grid static-arbitrage metrics are not reported
for this irregular local-support experiment.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    _build_manifest(root)
    return root


def run_results_pipeline(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    experiment_seeds = _experiment_seeds(root)
    status_path = root / "registry/results_pipeline_status.json"
    state: dict[str, Any] = {
        "status": "running",
        "current_stage": "preflight",
        "experiment_root": str(root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "bootstrap_seed": int(args.bootstrap_seed),
        "stages": {},
    }

    def update_stage(stage: str, status: str, **extra: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        state["current_stage"] = stage
        state["updated_at_utc"] = now
        stage_state = dict(state["stages"].get(stage) or {})
        stage_state["status"] = status
        if status == "running":
            stage_state["started_at_utc"] = now
        if status in {"completed", "failed"}:
            stage_state["finished_at_utc"] = now
        stage_state.update(extra)
        state["stages"][stage] = stage_state
        _write_json(status_path, state)

    try:
        update_stage("collect_checkpoints", "running")
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
        selected = pd.read_csv(root / "checkpoint_selection/selected_checkpoints.csv")
        expected_runs = len(FOLDS) * len(experiment_seeds) * len(VARIANTS)
        if len(selected) != expected_runs:
            raise ValueError(
                f"Selected checkpoint count mismatch: {len(selected)} != {expected_runs}."
            )
        update_stage(
            "collect_checkpoints",
            "completed",
            selected_checkpoint_count=int(len(selected)),
        )

        update_stage("generate_test_results", "running")
        generate_matrix(argparse.Namespace(experiment_root=str(root)))
        generated = pd.read_csv(root / "registry/generate_registry.csv")
        if len(generated) != expected_runs:
            raise ValueError(
                f"Generate registry count mismatch: {len(generated)} != {expected_runs}."
            )
        update_stage(
            "generate_test_results",
            "completed",
            generated_run_count=int(len(generated)),
            generated_sample_count=int(generated["sample_count"].sum()),
        )

        update_stage("build_comparison", "running")
        build_comparison(
            argparse.Namespace(
                experiment_root=str(root),
                bootstrap_iterations=int(args.bootstrap_iterations),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        )
        summary_path = root / "final_tables/development_rq1_result_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing final result summary: {summary_path}")
        update_stage(
            "build_comparison",
            "completed",
            result_summary_path=str(summary_path),
        )

        state["status"] = "completed"
        state["current_stage"] = "completed"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["updated_at_utc"] = state["finished_at_utc"]
        _write_json(status_path, state)
        print(f"RQ1 result pipeline completed: {root}")
        print(f"Result summary: {summary_path}")
        return root
    except Exception as exc:
        failed_stage = str(state.get("current_stage", "unknown"))
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error_message"] = str(exc)
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["updated_at_utc"] = state["finished_at_utc"]
        stage_state = dict(state["stages"].get(failed_stage) or {})
        stage_state["status"] = "failed"
        stage_state["finished_at_utc"] = state["finished_at_utc"]
        stage_state["error_type"] = type(exc).__name__
        stage_state["error_message"] = str(exc)
        state["stages"][failed_stage] = stage_state
        _write_json(status_path, state)
        raise


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
    prepare.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    prepare.add_argument("--reuse", action="store_true")

    train = subparsers.add_parser("train-matrix")
    train.add_argument("--experiment-root", default="")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--fold", choices=list(FOLDS), default="")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--variant", choices=list(VARIANTS), default="")
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

    results = subparsers.add_parser("results-pipeline")
    results.add_argument("--experiment-root", default="")
    results.add_argument("--bootstrap-iterations", type=int, default=10000)
    results.add_argument("--bootstrap-seed", type=int, default=20260722)

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
        return run_results_pipeline(args)
    if args.command == "package":
        return package_experiment(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
