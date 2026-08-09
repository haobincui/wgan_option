#!/usr/bin/env python3
"""RQ1 pair-level text rolling-development experiment orchestration."""

from __future__ import annotations

import argparse
import fcntl
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
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import (  # noqa: E402
    build_sample_config,
    config_to_dict,
    load_train_config,
)
from film_wgan.data import (  # noqa: E402
    apply_text_alignment,
    build_split_manifest_frame,
    create_train_val_bundle,
    load_film_wgan_samples,
)
from film_wgan.matching import (  # noqa: E402
    TextAlignmentPlan,
    TransitionMatchingNegativePlan,
    build_text_alignment_plan,
    build_transition_matching_negative_source_plan,
)
from film_wgan.support import parse_raw_surface_params  # noqa: E402
from film_wgan.protocol import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION_V3,
    CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING,
    DIAGNOSTICS_SCHEMA_VERSION,
    EPOCH_POLICY_CONTINUATION_ANCHOR,
    EXPERIMENT_DESIGN_SCHEMA_VERSION,
    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
    TEXT_ALIGNMENT_PLAN_VERSION,
    TRAINING_PROTOCOL_VERSION_V3,
    canonical_payload_sha256,
)
from film_wgan.text_transform import sha256_file  # noqa: E402
from film_wgan.text_lineage import write_text_lineage_artifacts  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq1_pair_textbase_v3_pilot.yaml"
DEFAULT_WORKBOOK = (
    ROOT
    / "data/processed/raw-excel-session/rq123_cme_session_20260729-131219/merged_vol_rq2_text.xlsx"
)
DEFAULT_NEWS_WORKBOOK = ROOT / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
V3_CANONICAL_WORKBOOK_SHA256 = (
    "8d8dbd2d3187bd52c5c1d4e651116348fb34143322e2fbe3735eec1d0cb03c71"
)
PREPARATION_RUN_FINGERPRINT_SHA256 = "0" * 64
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
V3_PILOT_FOLDS = ("2023Q1",)
V3_PILOT_SEEDS = (42, 202, 404)
V3_PILOT_Q1_SPLIT_COUNTS = (339, 133, 103)
V3_PILOT_Q1_MATCHING_ELIGIBLE = {"train": 187, "val": 74}
V2_DIAGNOSTIC_ARCHIVE = "rq1_film_wgan_v2_pilot"
V2_DIAGNOSTIC_CLAIM_SCOPE = "diagnostic_only"
V3_PILOT_VARIANTS = (
    PARENT_VARIANT,
    CONTINUATION_VARIANT,
    TEXT_RESIDUAL_VARIANT,
    SHUFFLED_RESIDUAL_VARIANT,
)
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


def _experiment_design(root: Path) -> dict[str, Any]:
    path = root / "inputs/experiment_design.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected experiment design mapping in {path}.")
    return payload


def _experiment_scope(
    root: Path,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """Return the frozen run matrix, with legacy archive fallbacks."""

    design = _experiment_design(root)
    folds = tuple(str(value) for value in design.get("run_folds") or FOLDS)
    seeds = _normalize_seeds(design.get("run_seeds") or design.get("seeds") or _experiment_seeds(root))
    variants = tuple(str(value) for value in design.get("run_variants") or VARIANTS)
    unknown_folds = sorted(set(folds) - set(FOLDS))
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_folds:
        raise ValueError(f"Unknown folds in experiment scope: {unknown_folds}")
    if unknown_variants:
        raise ValueError(f"Unknown variants in experiment scope: {unknown_variants}")
    if not folds or not variants:
        raise ValueError("Experiment run_folds and run_variants must be non-empty.")
    expected = len(folds) * len(seeds) * len(variants)
    configured_expected = design.get("expected_training_runs")
    if configured_expected is not None and int(configured_expected) != expected:
        raise ValueError(
            "Experiment-design run count mismatch: "
            f"configured={configured_expected} derived={expected}."
        )
    return folds, seeds, variants


def _is_v3_design(root: Path) -> bool:
    design = _experiment_design(root)
    return (
        int(design.get("experiment_design_schema_version", 0))
        == EXPERIMENT_DESIGN_SCHEMA_VERSION
        and str(design.get("training_protocol_version", ""))
        == TRAINING_PROTOCOL_VERSION_V3
    )


def _v3_preparation_overrides(*, validate_frozen_plans: bool) -> dict[str, Any]:
    """Return non-training overrides used while freezing a v3 fold.

    A run fingerprint is variant-specific, so it cannot be written into the
    shared fold YAML.  The all-zero value is used only to exercise the strict
    config/data loader after the frozen plans exist; it is never passed to a
    training command.
    """

    if validate_frozen_plans:
        return {"run_fingerprint_sha256": PREPARATION_RUN_FINGERPRINT_SHA256}
    return {
        "training_protocol_version": "",
        "run_fingerprint_sha256": "",
        "text_alignment_plan_path": "",
        "matching_negative_source_plan_path": "",
        "lambda_mismatch": 0.0,
        "lambda_critic_matching": 0.0,
        "lambda_generator_matching": 0.0,
    }


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
    continuation_anchor_epoch: int | None = None,
) -> dict[str, Any]:
    training = dict(fold_config["training"])
    residual_critic_mode = str(
        training.get("critic_conditioning_mode", "projection")
    ).strip().lower()
    transition_matching = residual_critic_mode == "transition_matching"
    v3_protocol = (
        str(training.get("training_protocol_version", ""))
        == TRAINING_PROTOCOL_VERSION_V3
    )
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
        "critic_conditioning_mode": residual_critic_mode,
        "lambda_film": 1.0e-4,
        "lambda_mismatch": (
            0.0 if transition_matching else float(training.get("lambda_mismatch", 0.5))
        ),
        "lambda_critic_matching": (
            float(training.get("lambda_critic_matching", 0.0))
            if transition_matching
            else 0.0
        ),
        "lambda_generator_matching": (
            float(training.get("lambda_generator_matching", 0.0))
            if transition_matching
            else 0.0
        ),
        "initial_generator_checkpoint_path": "",
        "freeze_backbone_epochs": 0,
    }
    if v3_protocol:
        overrides.update(
            training_protocol_version=TRAINING_PROTOCOL_VERSION_V3,
            scheduler_horizon_epochs=60,
            num_epochs=60,
            use_early_stopping=True,
            early_stopping_patience=10,
        )
    continuation_values: dict[str, Any] = {}
    if v3_protocol:
        continuation_values.update(
            scheduler_horizon_epochs=100,
            num_epochs=100,
            use_early_stopping=True,
            early_stopping_patience=15,
        )
    if variant == PARENT_VARIANT:
        overrides.update(
            text_embedding_mode="zero_lp",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            lambda_critic_matching=0.0,
            lambda_generator_matching=0.0,
        )
    elif variant == CONTINUATION_VARIANT:
        if parent_checkpoint is None:
            raise FileNotFoundError("No-text continuation requires its paired no-text checkpoint.")
        overrides.update(
            text_embedding_mode="zero_lp",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            lambda_critic_matching=0.0,
            lambda_generator_matching=0.0,
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
            **continuation_values,
        )
    elif variant == TEXT_RESIDUAL_VARIANT:
        if parent_checkpoint is None:
            raise FileNotFoundError("Residual text training requires its paired no-text checkpoint.")
        if v3_protocol and (
            continuation_anchor_epoch is None
            or not 11 <= int(continuation_anchor_epoch) <= 100
        ):
            raise ValueError(
                "Residual text training requires a continuation anchor epoch in [11, 100]."
            )
        anchor_values: dict[str, Any] = {}
        if v3_protocol:
            anchor_values.update(
                scheduler_horizon_epochs=100,
                num_epochs=int(continuation_anchor_epoch),
                use_early_stopping=False,
                early_stopping_patience=15,
            )
        overrides.update(
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
            **anchor_values,
        )
    elif variant == SHUFFLED_RESIDUAL_VARIANT:
        if parent_checkpoint is None:
            raise FileNotFoundError("Shuffled residual training requires its paired no-text checkpoint.")
        if v3_protocol and (
            continuation_anchor_epoch is None
            or not 11 <= int(continuation_anchor_epoch) <= 100
        ):
            raise ValueError(
                "Shuffled residual training requires a continuation anchor epoch in [11, 100]."
            )
        anchor_values = {}
        if v3_protocol:
            anchor_values.update(
                scheduler_horizon_epochs=100,
                num_epochs=int(continuation_anchor_epoch),
                use_early_stopping=False,
                early_stopping_patience=15,
            )
        overrides.update(
            text_alignment_mode="permuted",
            initial_generator_checkpoint_path=str(parent_checkpoint),
            freeze_backbone_epochs=5,
            **anchor_values,
        )
    elif variant == "pair_pca_text_full_film":
        overrides.update(
            conditioning_mode="film",
            critic_conditioning_mode="inherit",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            lambda_critic_matching=0.0,
            lambda_generator_matching=0.0,
        )
    elif variant == "pair_pca_text_concat":
        overrides.update(
            conditioning_mode="concat",
            critic_conditioning_mode="inherit",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            lambda_critic_matching=0.0,
            lambda_generator_matching=0.0,
        )
    elif variant == "pair_l2_text_full_film":
        overrides.update(
            conditioning_mode="film",
            critic_conditioning_mode="inherit",
            text_preprocessing_mode="raw_l2",
            text_transform_path="",
            lambda_film=0.0,
            lambda_mismatch=0.0,
            lambda_critic_matching=0.0,
            lambda_generator_matching=0.0,
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


def _best_epoch(run_dir: Path) -> int:
    path = run_dir / "metrics/best_checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("best_epoch", 0))


def _continuation_anchor(root: Path, fold: str, seed: int) -> tuple[Path, int]:
    run_dir = _completed_run(
        root
        / "training_runs"
        / CONTINUATION_VARIANT
        / fold
        / f"seed_{seed}"
    )
    if run_dir is None:
        raise FileNotFoundError(
            "Matched and shuffled Stage-B runs require the completed paired "
            f"continuation anchor: fold={fold}, seed={seed}."
        )
    epoch = _best_epoch(run_dir)
    if not 11 <= epoch <= 100:
        raise ValueError(
            f"Continuation anchor epoch must be in [11, 100], found {epoch}: {run_dir}"
        )
    return run_dir, epoch


def _comparison_checkpoint(
    run_dir: Path,
    *,
    variant: str,
    continuation_anchor_epoch: int | None,
    v3_protocol: bool = True,
) -> tuple[Path, int]:
    if v3_protocol and variant in {
        TEXT_RESIDUAL_VARIANT,
        SHUFFLED_RESIDUAL_VARIANT,
    }:
        if continuation_anchor_epoch is None:
            raise ValueError(f"{variant} requires a continuation anchor epoch.")
        checkpoint = run_dir / "checkpoints/film_wgan_final.pt"
        fields = _checkpoint_protocol_fields(checkpoint)
        epoch = int(fields["epoch"])
        if epoch != int(continuation_anchor_epoch):
            raise ValueError(
                f"Anchored comparison checkpoint epoch mismatch for {variant}: "
                f"found={epoch} expected={continuation_anchor_epoch}."
            )
        return checkpoint, epoch
    checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
    epoch = _best_epoch(run_dir)
    if (
        v3_protocol
        and variant == CONTINUATION_VARIANT
        and not 11 <= epoch <= 100
    ):
        raise ValueError(
            f"Continuation comparison epoch must be in [11, 100], found {epoch}."
        )
    return checkpoint, epoch


def _validate_run_protocol(
    run_dir: Path,
    *,
    critic_mode: str,
    require_v3: bool = False,
    require_schema5: bool = False,
    expected_fingerprint: str = "",
    checkpoint_path: Path | None = None,
) -> None:
    """Validate v3 exactly while retaining schema-5 archive readability."""

    transition_mode = (
        str(critic_mode).strip().lower() == "transition_matching"
    )
    if not (transition_mode or require_v3 or require_schema5):
        return
    selected_checkpoint = checkpoint_path or (
        run_dir / "checkpoints/film_wgan_best.pt"
    )
    fields = _checkpoint_protocol_fields(selected_checkpoint)
    schema = int(fields["checkpoint_schema_version"])
    protocol = str(fields["training_protocol_version"])
    critic = str(fields["critic_architecture_version"])
    fingerprint = str(fields["run_fingerprint_sha256"])
    if require_v3:
        errors = []
        if schema != CHECKPOINT_SCHEMA_VERSION_V3:
            errors.append(
                f"schema={schema} expected={CHECKPOINT_SCHEMA_VERSION_V3}"
            )
        if protocol != TRAINING_PROTOCOL_VERSION_V3:
            errors.append(
                f"protocol={protocol!r} expected={TRAINING_PROTOCOL_VERSION_V3!r}"
            )
        if critic != CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING:
            errors.append(
                "critic="
                f"{critic!r} expected={CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING!r}"
            )
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            errors.append(f"invalid run_fingerprint_sha256={fingerprint!r}")
        if expected_fingerprint and fingerprint != expected_fingerprint:
            errors.append(
                "run_fingerprint_sha256="
                f"{fingerprint!r} expected={expected_fingerprint!r}"
            )
        if not errors:
            return
        raise ValueError(
            "Refusing to reuse an incompatible completed run in the v3 "
            f"transition-matching experiment: run={run_dir}; "
            + "; ".join(errors)
            + ". Use a new experiment root and retrain Stage A."
        )

    expected_protocol = (
        "film_wgan_transition_matching_v2"
        if transition_mode
        else "film_wgan_v1_compatible"
    )
    expected_critic = (
        CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
        if transition_mode
        else "legacy_v1"
    )
    if schema != 5 or protocol != expected_protocol or critic != expected_critic:
        raise ValueError(
            "Refusing to reuse a non-v2 completed run in a transition-matching "
            f"experiment: run={run_dir}, schema={schema}, protocol={protocol!r}, "
            f"critic={critic!r}. Use a new experiment root and retrain Stage A."
        )


def _variant_critic_mode(variant: str, *, residual_critic_mode: str) -> str:
    if variant == PARENT_VARIANT or variant in PAIRED_STAGE_B_VARIANTS:
        return str(residual_critic_mode).strip().lower()
    return "inherit"


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


def _upsert_launch_registry(
    root: Path,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Merge run records without dropping rows from earlier CLI invocations."""

    registry_path = root / "registry/launch_registry.csv"
    if not rows:
        return registry_path

    incoming = pd.DataFrame(rows)
    key_columns = ["fold", "seed", "variant"]
    missing = [column for column in key_columns if column not in incoming.columns]
    if missing:
        raise ValueError(f"Launch registry rows are missing key columns: {missing}")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(".csv.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if registry_path.is_file() and registry_path.stat().st_size > 0:
            existing = pd.read_csv(registry_path)
            existing_missing = [
                column for column in key_columns if column not in existing.columns
            ]
            if existing_missing:
                raise ValueError(
                    "Existing launch registry is missing key columns: "
                    f"{existing_missing}"
                )
            combined = pd.concat(
                [existing, incoming],
                ignore_index=True,
                sort=False,
            )
        else:
            combined = incoming

        combined = combined.drop_duplicates(subset=key_columns, keep="last")
        temporary_path = registry_path.with_suffix(".csv.tmp")
        combined.to_csv(temporary_path, index=False)
        temporary_path.replace(registry_path)
    return registry_path


def _assert_frozen_v2_worktree(root: Path) -> None:
    """Keep every transition-matching run on the commit frozen at prepare."""

    config_path = root / "inputs/configs/train_rq1_pair_textbase.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = _read_yaml(config_path)
    training = dict(payload.get("training") or payload)
    if (
        str(training.get("critic_conditioning_mode", "")).strip().lower()
        != "transition_matching"
    ):
        return

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "transition_matching training requires a clean committed worktree."
        )

    git_state_path = root / "inputs/git_state.txt"
    if not git_state_path.is_file():
        raise FileNotFoundError(git_state_path)
    git_state_lines = git_state_path.read_text(encoding="utf-8").splitlines()
    first_line = git_state_lines[0] if git_state_lines else ""
    if not first_line.startswith("commit=") or not first_line.removeprefix("commit="):
        raise ValueError(f"Invalid frozen git state: {git_state_path}")
    frozen_commit = first_line.removeprefix("commit=").strip()
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != frozen_commit:
        raise RuntimeError(
            "Refusing to mix transition_matching runs across commits: "
            f"experiment={frozen_commit}, current={current_commit}. "
            "Prepare a new experiment root."
        )


def _verify_input_manifest(root: Path) -> None:
    manifest_path = root / "inputs/input_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"Frozen input manifest is missing columns: {sorted(required - set(manifest.columns))}"
        )
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file():
            failures.append(f"missing:{row.relative_path}")
            continue
        if path.stat().st_size != int(row.size_bytes):
            failures.append(f"size:{row.relative_path}")
            continue
        if sha256_file(path) != str(row.sha256):
            failures.append(f"sha256:{row.relative_path}")
    if failures:
        raise ValueError(
            "Frozen input manifest verification failed: " + ", ".join(failures)
        )


def _verify_v3_plan_contents(root: Path) -> None:
    validation_path = root / "validation_summary.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    q1 = dict((validation.get("folds") or {}).get("2023Q1") or {})
    observed_counts = (
        int(q1.get("train_pairs", -1)),
        int(q1.get("validation_pairs", -1)),
        int(q1.get("test_pairs", -1)),
    )
    lineage_path = _fold_dir(root, "2023Q1") / "pair_lineage_audit.csv"
    manifest = pd.read_csv(root / "inputs/input_manifest.csv")
    lineage_relative_path = str(lineage_path.relative_to(root))
    if lineage_relative_path not in set(manifest["relative_path"].astype(str)):
        raise ValueError(
            "Q1 pair-lineage audit is not protected by the frozen input manifest."
        )
    lineage = pd.read_csv(lineage_path)
    required_lineage_columns = {"surface_pair_id", "split"}
    if not required_lineage_columns.issubset(lineage.columns):
        raise ValueError(
            "Q1 pair-lineage audit omits split identity columns: "
            f"{sorted(required_lineage_columns - set(lineage.columns))}."
        )
    lineage_identity = lineage.loc[:, ["surface_pair_id", "split"]].copy()
    lineage_identity["surface_pair_id"] = lineage_identity[
        "surface_pair_id"
    ].astype(str)
    lineage_identity["split"] = lineage_identity["split"].astype(str).str.lower()
    if lineage_identity["surface_pair_id"].duplicated().any():
        raise ValueError("Q1 pair-lineage audit contains duplicate surface_pair_id rows.")
    derived_counts = tuple(
        int((lineage_identity["split"] == split).sum())
        for split in ("train", "val", "test")
    )
    if set(lineage_identity["split"]) != {"train", "val", "test"}:
        raise ValueError("Q1 pair-lineage audit contains an unexpected split label.")
    if (
        derived_counts != V3_PILOT_Q1_SPLIT_COUNTS
        or observed_counts != derived_counts
    ):
        raise ValueError(
            "Canonical Q1 split counts changed: "
            f"lineage={derived_counts} summary={observed_counts} "
            f"expected={V3_PILOT_Q1_SPLIT_COUNTS}."
        )

    for fold in FOLDS:
        training = dict(
            (_read_yaml(_fold_config(root, fold)).get("training") or {})
        )
        alignment_path = _config_artifact_path(
            training.get("text_alignment_plan_path")
        )
        negative_path = _config_artifact_path(
            training.get("matching_negative_source_plan_path")
        )
        if alignment_path is None or negative_path is None:
            raise ValueError(f"Frozen v3 plan paths are empty for {fold}.")
        if negative_path.name != "transition_matching_negative_source_plan.csv":
            raise ValueError(f"Unexpected v3 negative-plan filename: {negative_path}")
        summary_path = (
            negative_path.parent
            / "transition_matching_negative_source_summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        alignment_frame = pd.read_csv(alignment_path)
        negative_frame = pd.read_csv(negative_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("text_alignment_plan_sha256") != sha256_file(
            alignment_path
        ) or summary.get(
            "matching_negative_source_plan_sha256"
        ) != sha256_file(negative_path):
            raise ValueError(f"Frozen plan file SHA summary mismatch for {fold}.")
        for split in ("train", "val"):
            alignment = TextAlignmentPlan.from_frame(
                alignment_frame,
                split=split,
            )
            negative = TransitionMatchingNegativePlan.from_frame(
                negative_frame,
                split=split,
            )
            if negative.positive_alignment_sha256 != alignment.sha256:
                raise ValueError(
                    f"Positive/negative canonical plan SHA mismatch: {fold}/{split}"
                )
            split_rows = negative_frame[
                negative_frame["split"].astype(str).str.lower() == split
            ].copy()
            eligible_rows = split_rows[
                pd.to_numeric(split_rows["target_eligible"], errors="coerce")
                == 1
            ]
            if fold == "2023Q1":
                eligible_count = int(
                    eligible_rows["target_dataset_index"].nunique()
                )
                expected_eligible = V3_PILOT_Q1_MATCHING_ELIGIBLE[split]
                if eligible_count != expected_eligible:
                    raise ValueError(
                        f"Canonical Q1 {split} matching eligibility changed: "
                        f"{eligible_count} != {expected_eligible}."
                    )
            if not eligible_rows.empty:
                native_hits = int(
                    (
                        eligible_rows["negative_source_index"].astype(int)
                        == eligible_rows["native_positive_source_index"].astype(int)
                    ).sum()
                )
                placebo_hits = int(
                    (
                        eligible_rows["negative_source_index"].astype(int)
                        == eligible_rows["placebo_positive_source_index"].astype(int)
                    ).sum()
                )
                fallback_count = int(
                    pd.to_numeric(
                        eligible_rows.get(
                            "round_reuse_fallback",
                            pd.Series(np.zeros(len(eligible_rows))),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )
                per_target_reuse = bool(
                    eligible_rows.duplicated(
                        ["target_dataset_index", "negative_source_index"]
                    ).any()
                )
                rank_reuse = bool(
                    eligible_rows.duplicated(
                        ["negative_rank", "negative_source_index"]
                    ).any()
                )
                if (
                    native_hits
                    or placebo_hits
                    or fallback_count
                    or per_target_reuse
                    or rank_reuse
                ):
                    raise ValueError(
                        "Unsafe symmetric-negative plan audit for "
                        f"{fold}/{split}: native={native_hits}, "
                        f"placebo={placebo_hits}, fallback={fallback_count}, "
                        f"target_reuse={per_target_reuse}, rank_reuse={rank_reuse}."
                    )


def verify_existing(args: argparse.Namespace) -> Path:
    """Read-only verification for an immutable prepared v3 experiment root."""

    root = _resolve_root(args.experiment_root)
    if not _is_v3_design(root):
        raise ValueError(
            "verify-existing requires an experiment-design schema-2 v3 root."
        )
    design = _experiment_design(root)
    expected_fields = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
        "critic_architecture_version": (
            CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
        ),
        "matching_negative_source_plan_version": (
            MATCHING_NEGATIVE_SOURCE_PLAN_VERSION
        ),
        "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
        "epoch_policy": EPOCH_POLICY_CONTINUATION_ANCHOR,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
    }
    mismatches = {
        key: {"found": design.get(key), "expected": expected}
        for key, expected in expected_fields.items()
        if design.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"V3 experiment-design protocol mismatch: {mismatches}")
    if tuple(str(value) for value in design.get("prepared_folds", ())) != tuple(
        FOLDS
    ) or int(design.get("prepared_fold_count", 0)) != len(FOLDS):
        raise ValueError("The v3 root must freeze all four rolling folds.")
    folds, seeds, variants = _experiment_scope(root)
    if (
        folds != V3_PILOT_FOLDS
        or seeds != V3_PILOT_SEEDS
        or variants != V3_PILOT_VARIANTS
    ):
        raise ValueError(
            "The v3 validation pilot scope must be exactly "
            f"folds={V3_PILOT_FOLDS}, seeds={V3_PILOT_SEEDS}, "
            f"variants={V3_PILOT_VARIANTS}; found {(folds, seeds, variants)}."
        )
    if int(design.get("run_fold_count", 0)) != len(V3_PILOT_FOLDS):
        raise ValueError("The v3 validation pilot run_fold_count must be one.")
    superseded = dict(design.get("superseded_experiment") or {})
    if superseded != {
        "experiment_name": V2_DIAGNOSTIC_ARCHIVE,
        "claim_scope": V2_DIAGNOSTIC_CLAIM_SCOPE,
        "checkpoint_reuse_allowed": False,
    }:
        raise ValueError(
            "The v3 design must retain the v2 pilot as a read-only, "
            "diagnostic-only archive."
        )
    _assert_frozen_v2_worktree(root)
    _verify_input_manifest(root)
    frozen_workbook = root / "inputs/data/merged_vol_rq2_text.xlsx"
    if sha256_file(frozen_workbook) != V3_CANONICAL_WORKBOOK_SHA256:
        raise ValueError("Frozen v3 workbook is not the canonical CME-session input.")
    _verify_v3_plan_contents(root)
    for fold in FOLDS:
        payload = _read_yaml(_fold_config(root, fold))
        training = dict(payload.get("training") or payload)
        if str(training.get("training_protocol_version", "")) != TRAINING_PROTOCOL_VERSION_V3:
            raise ValueError(f"Fold {fold} does not freeze the v3 training protocol.")
        for field in (
            "text_alignment_plan_path",
            "matching_negative_source_plan_path",
        ):
            path = _config_artifact_path(training.get(field))
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"Fold {fold} is missing frozen {field}: {training.get(field)!r}"
                )
    print(root)
    return root


def _freeze_fold_matching_plans(
    *,
    fold: str,
    fold_root: Path,
    preparation_bundle: Any,
    preparation_config: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Freeze train/validation positive alignments and common negatives."""

    native_partitions = {
        "train": preparation_bundle.native_train_items,
        "val": preparation_bundle.native_val_items,
    }
    alignment_plans = {}
    negative_plans = {}
    for split, native_items in native_partitions.items():
        if not native_items:
            raise ValueError(f"Cannot freeze an empty v3 {split} plan for {fold}.")
        alignment_plan = build_text_alignment_plan(
            native_items,
            split=split,
            seed=int(preparation_config.text_permutation_seed),
        )
        negative_plan = build_transition_matching_negative_source_plan(
            native_items,
            text_alignment_plan=alignment_plan,
            split=split,
            negative_count=int(preparation_config.matching_negative_count),
            minimum_supported_cells=int(
                preparation_config.matching_min_supported_cells
            ),
            seed=int(preparation_config.matching_negative_seed),
            duplicate_cosine_threshold=float(
                preparation_config.matching_duplicate_cosine_threshold
            ),
        )
        alignment_plans[split] = alignment_plan
        negative_plans[split] = negative_plan

    alignment_path = fold_root / "text_alignment_plan.csv"
    negative_path = fold_root / "transition_matching_negative_source_plan.csv"
    pd.concat(
        [alignment_plans[split].to_frame() for split in ("train", "val")],
        ignore_index=True,
    ).to_csv(alignment_path, index=False)
    pd.concat(
        [negative_plans[split].to_frame() for split in ("train", "val")],
        ignore_index=True,
    ).to_csv(negative_path, index=False)

    audit_payload = {
        "fold": fold,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION_V3,
        "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
        "matching_negative_source_plan_version": (
            MATCHING_NEGATIVE_SOURCE_PLAN_VERSION
        ),
        "text_alignment_plan_path": str(alignment_path),
        "text_alignment_plan_sha256": sha256_file(alignment_path),
        "matching_negative_source_plan_path": str(negative_path),
        "matching_negative_source_plan_sha256": sha256_file(negative_path),
        "splits": {
            split: {
                "pair_count": len(alignment_plans[split].target_sample_ids),
                "text_alignment_canonical_sha256": alignment_plans[split].sha256,
                "matching_negative_canonical_sha256": negative_plans[split].sha256,
                "matching_eligible_targets": int(
                    negative_plans[split].eligible_mask.sum()
                ),
                "matching_negative_count": int(
                    negative_plans[split].negative_count
                ),
                "matching_summary": dict(negative_plans[split].summary),
            }
            for split in ("train", "val")
        },
        "outer_test_plan_frozen": False,
        "outer_test_use_forbidden": True,
    }
    _write_json(
        fold_root / "transition_matching_negative_source_summary.json",
        audit_payload,
    )
    _write_json(fold_root / "matching_plan_audit.json", audit_payload)
    return alignment_path, negative_path, audit_payload


def _verify_fold_matching_plans(
    *,
    fold: str,
    bundle: Any,
    alignment_path: Path,
    negative_path: Path,
) -> None:
    alignment_frame = pd.read_csv(alignment_path)
    negative_frame = pd.read_csv(negative_path)
    native_partitions = {
        "train": bundle.native_train_items,
        "val": bundle.native_val_items,
    }
    for split, native_items in native_partitions.items():
        alignment_plan = bundle.text_alignment_plans.get(split)
        if alignment_plan is None:
            raise ValueError(f"Strict v3 bundle omitted {fold}/{split} alignment plan.")
        disk_alignment = type(alignment_plan).from_frame(
            alignment_frame,
            split=split,
        )
        negative_plan = TransitionMatchingNegativePlan.from_frame(
            negative_frame,
            split=split,
        )
        expected_ids = tuple(item.sample_id for item in native_items)
        expected_pairs = tuple(item.surface_pair_id for item in native_items)
        if (
            alignment_plan.sha256 != disk_alignment.sha256
            or alignment_plan.target_sample_ids != expected_ids
            or alignment_plan.target_surface_pair_ids != expected_pairs
        ):
            raise ValueError(f"Frozen alignment identities changed for {fold}/{split}.")
        if (
            negative_plan.target_sample_ids != expected_ids
            or negative_plan.target_surface_pair_ids != expected_pairs
            or negative_plan.positive_alignment_sha256 != alignment_plan.sha256
        ):
            raise ValueError(f"Frozen negative identities changed for {fold}/{split}.")


def prepare_experiment(args: argparse.Namespace) -> Path:
    _assert_py312()
    source_config = Path(args.config).resolve()
    source_workbook = Path(args.workbook).resolve()
    source_news = Path(args.news_workbook).resolve()
    for source in (source_config, source_workbook, source_news):
        if not source.is_file():
            raise FileNotFoundError(source)
    base_payload = _read_yaml(source_config)
    training = dict(base_payload.get("training") or base_payload)
    generate = dict(base_payload.get("generate_result") or {})
    if str(training.get("training_protocol_version", "")) != TRAINING_PROTOCOL_VERSION_V3:
        raise ValueError(
            "RQ1 v3 prepare requires the explicit frozen training protocol "
            f"{TRAINING_PROTOCOL_VERSION_V3!r}."
        )
    source_workbook_sha256 = sha256_file(source_workbook)
    if source_workbook_sha256 != V3_CANONICAL_WORKBOOK_SHA256:
        raise ValueError(
            "RQ1 v3 requires the canonical CME-session workbook: "
            f"found sha256={source_workbook_sha256}, "
            f"expected={V3_CANONICAL_WORKBOOK_SHA256}."
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
    if (
        str(training.get("critic_conditioning_mode", "")).strip().lower()
        == "transition_matching"
        and any(
            line.strip() and not line.startswith("##")
            for line in git_state.splitlines()
        )
    ):
        raise RuntimeError(
            "transition_matching experiments require a clean committed worktree; "
            "commit the v3 implementation before prepare."
        )

    experiment_seeds = _normalize_seeds(
        getattr(args, "seeds", None) or V3_PILOT_SEEDS
    )
    run_folds = tuple(
        str(value) for value in (getattr(args, "run_folds", None) or FOLDS)
    )
    run_variants = tuple(
        str(value) for value in (getattr(args, "run_variants", None) or VARIANTS)
    )
    unknown_folds = sorted(set(run_folds) - set(FOLDS))
    unknown_variants = sorted(set(run_variants) - set(VARIANTS))
    if unknown_folds:
        raise ValueError(f"Unknown requested run folds: {unknown_folds}")
    if unknown_variants:
        raise ValueError(f"Unknown requested run variants: {unknown_variants}")
    matrix_profile = str(
        getattr(args, "matrix_profile", "validation_pilot")
    )
    if (
        matrix_profile != "validation_pilot"
        or experiment_seeds != V3_PILOT_SEEDS
        or run_folds != V3_PILOT_FOLDS
        or run_variants != V3_PILOT_VARIANTS
    ):
        raise ValueError(
            "RQ1 v3 prepare is fixed to the validation pilot scope: "
            f"matrix_profile=validation_pilot, folds={V3_PILOT_FOLDS}, "
            f"seeds={V3_PILOT_SEEDS}, variants={V3_PILOT_VARIANTS}."
        )
    root = _resolve_root(args.experiment_root, create=True)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Experiment directory is not empty; frozen v3 roots are immutable: "
            f"{root}. Use verify-existing for read-only verification or choose a new root."
        )
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
    (root / "inputs/git_state.txt").write_text(
        f"commit={commit}\n{git_state}",
        encoding="utf-8",
    )
    _write_json(
        root / "inputs/experiment_design.json",
        {
            "experiment_design_schema_version": EXPERIMENT_DESIGN_SCHEMA_VERSION,
            "matrix_profile": matrix_profile,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION_V3,
            "critic_architecture_version": (
                CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
            ),
            "matching_negative_source_plan_version": (
                MATCHING_NEGATIVE_SOURCE_PLAN_VERSION
            ),
            "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
            "epoch_policy": EPOCH_POLICY_CONTINUATION_ANCHOR,
            "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "seed_set_version": (
                SEED_SET_VERSION
                if experiment_seeds == tuple(SEEDS)
                else "custom"
            ),
            "seed_generation_master_seed": SEED_GENERATION_MASTER_SEED,
            "seeds": list(experiment_seeds),
            "seed_count": len(experiment_seeds),
            "prepared_folds": list(FOLDS),
            "prepared_fold_count": len(FOLDS),
            "run_folds": list(run_folds),
            "run_fold_count": len(run_folds),
            "run_seeds": list(experiment_seeds),
            "run_variants": list(run_variants),
            "fold_count": len(run_folds),
            "variant_count": len(run_variants),
            "expected_training_runs": (
                len(run_folds) * len(experiment_seeds) * len(run_variants)
            ),
            "superseded_experiment": {
                "experiment_name": V2_DIAGNOSTIC_ARCHIVE,
                "claim_scope": V2_DIAGNOSTIC_CLAIM_SCOPE,
                "checkpoint_reuse_allowed": False,
            },
        },
    )

    training.update(
        data_path=str(copied_workbook),
        news_workbook_path=str(copied_news),
        training_protocol_version=TRAINING_PROTOCOL_VERSION_V3,
        output_root="",
        checkpoints_path="",
        metrics_path="",
    )
    base_frozen = root / "inputs/configs/train_rq1_pair_textbase.yaml"
    _write_yaml(base_frozen, {"training": training, "generate_result": generate})
    row_config = load_train_config(
        base_frozen,
        overrides=_v3_preparation_overrides(validate_frozen_plans=False),
    )
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
        alignment_plan_path = fold_root / "text_alignment_plan.csv"
        negative_plan_path = (
            fold_root / "transition_matching_negative_source_plan.csv"
        )
        fold_training = dict(training)
        fold_training.update(
            split_manifest_path=str(manifest_path),
            text_transform_path=str(transform_path),
            surface_support_path=str(surface_support_path),
            text_alignment_plan_path=str(alignment_plan_path),
            matching_negative_source_plan_path=str(negative_plan_path),
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
        preparation_config = load_train_config(
            fold_config_path,
            overrides=_v3_preparation_overrides(validate_frozen_plans=False),
        )
        preparation_bundle = create_train_val_bundle(preparation_config)
        (
            alignment_plan_path,
            negative_plan_path,
            matching_plan_audit,
        ) = _freeze_fold_matching_plans(
            fold=fold,
            fold_root=fold_root,
            preparation_bundle=preparation_bundle,
            preparation_config=preparation_config,
        )
        fold_config = load_train_config(
            fold_config_path,
            overrides=_v3_preparation_overrides(validate_frozen_plans=True),
        )
        bundle = create_train_val_bundle(fold_config)
        _verify_fold_matching_plans(
            fold=fold,
            bundle=bundle,
            alignment_path=alignment_plan_path,
            negative_path=negative_plan_path,
        )
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
        for split, items, native_items in (
            ("train", bundle.train_items, bundle.native_train_items),
            ("val", bundle.val_items, bundle.native_val_items),
            ("test", bundle.test_items, bundle.native_test_items),
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
            if split == "test":
                continue
            shuffled_config = load_train_config(
                fold_config_path,
                overrides={
                    **_v3_preparation_overrides(validate_frozen_plans=True),
                    "text_alignment_mode": "permuted",
                },
            )
            permuted_items = apply_text_alignment(
                shuffled_config,
                native_items,
                split=split,
                alignment_plan=bundle.text_alignment_plans[split],
            )
            for target, donor_view in zip(native_items, permuted_items):
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
            "text_alignment_plan_path": str(alignment_plan_path),
            "text_alignment_plan_sha256": sha256_file(alignment_plan_path),
            "matching_negative_source_plan_path": str(negative_plan_path),
            "matching_negative_source_plan_sha256": sha256_file(
                negative_plan_path
            ),
            "matching_plan_audit": matching_plan_audit,
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
            "source_workbook_sha256": source_workbook_sha256,
            "news_source_timezone": EXPECTED_NEWS_SOURCE_TIMEZONE,
            "text_lineage_manifest": str(text_audit_paths["manifest"]),
            "text_lineage_manifest_sha256": sha256_file(
                text_audit_paths["manifest"]
            ),
            "short_atm_selection": str(short_atm_selection_path),
            "short_atm_selection_sha256": sha256_file(
                short_atm_selection_path
            ),
            "models": list(run_variants),
            "seeds": list(experiment_seeds),
            "run_folds": list(run_folds),
            "run_variants": list(run_variants),
            "folds": fold_validation,
            "expected_training_runs": (
                len(run_folds) * len(experiment_seeds) * len(run_variants)
            ),
            "superseded_experiment": {
                "experiment_name": V2_DIAGNOSTIC_ARCHIVE,
                "claim_scope": V2_DIAGNOSTIC_CLAIM_SCOPE,
                "checkpoint_reuse_allowed": False,
            },
        },
    )
    print(root)
    return root


def train_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    v3_design = _is_v3_design(root)
    if v3_design:
        verify_existing(argparse.Namespace(experiment_root=str(root)))
    else:
        _assert_frozen_v2_worktree(root)
    run_folds, experiment_seeds, run_variants = _experiment_scope(root)
    selected_folds = [args.fold] if getattr(args, "fold", "") else list(run_folds)
    if any(fold not in run_folds for fold in selected_folds):
        raise ValueError(
            f"Requested folds {selected_folds} are outside frozen run_folds={run_folds}."
        )
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
        [args.variant] if getattr(args, "variant", "") else list(run_variants)
    )
    if any(variant not in run_variants for variant in selected_variants):
        raise ValueError(
            "Requested variants are outside the frozen run scope: "
            f"requested={selected_variants}, frozen={run_variants}."
        )
    registry_rows: list[dict[str, Any]] = []
    for fold in selected_folds:
        fold_payload = _read_yaml(_fold_config(root, fold))
        fold_training = dict(fold_payload.get("training") or fold_payload)
        expected_critic_mode = str(
            fold_training.get("critic_conditioning_mode", "projection")
        )
        v3_matrix = (
            v3_design
            and str(fold_training.get("training_protocol_version", ""))
            == TRAINING_PROTOCOL_VERSION_V3
        )
        negative_plan_path = _config_artifact_path(
            fold_training.get("matching_negative_source_plan_path")
        )
        alignment_plan_path = _config_artifact_path(
            fold_training.get("text_alignment_plan_path")
        )
        negative_plan_sha256 = (
            sha256_file(negative_plan_path)
            if negative_plan_path is not None and negative_plan_path.is_file()
            else ""
        )
        alignment_plan_sha256 = (
            sha256_file(alignment_plan_path)
            if alignment_plan_path is not None and alignment_plan_path.is_file()
            else ""
        )
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
            if parent_run is not None:
                parent_fingerprint = ""
                if v3_matrix:
                    parent_overrides = _variant_overrides(
                        PARENT_VARIANT,
                        fold_config=fold_payload,
                        output_root=(
                            root
                            / "training_runs"
                            / PARENT_VARIANT
                            / fold
                            / f"seed_{seed}"
                        ),
                        seed=seed,
                        parent_checkpoint=None,
                    )
                    parent_fingerprint = _run_fingerprint(
                        root,
                        fold=fold,
                        seed=seed,
                        variant=PARENT_VARIANT,
                        training=fold_training,
                        overrides=parent_overrides,
                        parent_checkpoint_sha256="",
                    )
                _validate_run_protocol(
                    parent_run,
                    critic_mode=expected_critic_mode,
                    require_v3=v3_matrix,
                    require_schema5=(
                        not v3_matrix
                        and expected_critic_mode.strip().lower()
                        == "transition_matching"
                    ),
                    expected_fingerprint=parent_fingerprint,
                )
            parent_checkpoint_sha256 = (
                sha256_file(parent_checkpoint)
                if parent_checkpoint is not None
                else ""
            )
            for variant in selected_variants:
                output_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                continuation_anchor_epoch: int | None = None
                if v3_matrix and variant in {
                    TEXT_RESIDUAL_VARIANT,
                    SHUFFLED_RESIDUAL_VARIANT,
                }:
                    continuation_run, continuation_anchor_epoch = _continuation_anchor(
                        root,
                        fold,
                        seed,
                    )
                    continuation_overrides = _variant_overrides(
                        CONTINUATION_VARIANT,
                        fold_config=fold_payload,
                        output_root=(
                            root
                            / "training_runs"
                            / CONTINUATION_VARIANT
                            / fold
                            / f"seed_{seed}"
                        ),
                        seed=seed,
                        parent_checkpoint=parent_checkpoint,
                    )
                    continuation_fingerprint = _run_fingerprint(
                        root,
                        fold=fold,
                        seed=seed,
                        variant=CONTINUATION_VARIANT,
                        training=fold_training,
                        overrides=continuation_overrides,
                        parent_checkpoint_sha256=parent_checkpoint_sha256,
                    )
                    _validate_run_protocol(
                        continuation_run,
                        critic_mode=expected_critic_mode,
                        require_v3=v3_matrix,
                        expected_fingerprint=continuation_fingerprint,
                    )
                overrides = _variant_overrides(
                    variant,
                    fold_config=fold_payload,
                    output_root=output_root,
                    seed=seed,
                    parent_checkpoint=parent_checkpoint,
                    continuation_anchor_epoch=continuation_anchor_epoch,
                )
                run_fingerprint_sha256 = ""
                if v3_matrix:
                    run_fingerprint_sha256 = _run_fingerprint(
                        root,
                        fold=fold,
                        seed=seed,
                        variant=variant,
                        training=fold_training,
                        overrides=overrides,
                        parent_checkpoint_sha256=(
                            parent_checkpoint_sha256
                            if variant in PAIRED_STAGE_B_VARIANTS
                            else ""
                        ),
                    )
                    overrides["run_fingerprint_sha256"] = run_fingerprint_sha256
                completed = _completed_run(output_root)
                if completed is not None:
                    comparison_checkpoint, comparison_epoch = _comparison_checkpoint(
                        completed,
                        variant=variant,
                        continuation_anchor_epoch=continuation_anchor_epoch,
                        v3_protocol=v3_matrix,
                    )
                    _validate_run_protocol(
                        completed,
                        critic_mode=_variant_critic_mode(
                            variant,
                            residual_critic_mode=expected_critic_mode,
                        ),
                        require_v3=v3_matrix,
                        expected_fingerprint=run_fingerprint_sha256,
                        checkpoint_path=comparison_checkpoint,
                    )
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
                            "checkpoint": str(comparison_checkpoint),
                            "epoch_policy": (
                                EPOCH_POLICY_CONTINUATION_ANCHOR
                                if v3_matrix
                                else "legacy_best_checkpoint"
                            ),
                            "comparison_epoch": int(comparison_epoch),
                            "comparison_checkpoint": str(comparison_checkpoint),
                            "comparison_checkpoint_sha256": sha256_file(
                                comparison_checkpoint
                            ),
                            "anchor_variant": (
                                CONTINUATION_VARIANT
                                if variant
                                in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                                and v3_matrix
                                else ""
                            ),
                            "run_fingerprint_sha256": run_fingerprint_sha256,
                            "matching_negative_source_plan_sha256": (
                                negative_plan_sha256
                            ),
                            "text_alignment_plan_sha256": alignment_plan_sha256,
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
                comparison_checkpoint, comparison_epoch = _comparison_checkpoint(
                    completed,
                    variant=variant,
                    continuation_anchor_epoch=continuation_anchor_epoch,
                    v3_protocol=v3_matrix,
                )
                _validate_run_protocol(
                    completed,
                    critic_mode=_variant_critic_mode(
                        variant,
                        residual_critic_mode=expected_critic_mode,
                    ),
                    require_v3=v3_matrix,
                    expected_fingerprint=run_fingerprint_sha256,
                    checkpoint_path=comparison_checkpoint,
                )
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
                        "checkpoint": str(comparison_checkpoint),
                        "epoch_policy": (
                            EPOCH_POLICY_CONTINUATION_ANCHOR
                            if v3_matrix
                            else "legacy_best_checkpoint"
                        ),
                        "comparison_epoch": int(comparison_epoch),
                        "comparison_checkpoint": str(comparison_checkpoint),
                        "comparison_checkpoint_sha256": sha256_file(
                            comparison_checkpoint
                        ),
                        "anchor_variant": (
                            CONTINUATION_VARIANT
                            if variant
                            in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                            and v3_matrix
                            else ""
                        ),
                        "run_fingerprint_sha256": run_fingerprint_sha256,
                        "matching_negative_source_plan_sha256": (
                            negative_plan_sha256
                        ),
                        "text_alignment_plan_sha256": alignment_plan_sha256,
                        **_stage_registry_fields(
                            variant,
                            parent_checkpoint=parent_checkpoint,
                            parent_checkpoint_sha256=parent_checkpoint_sha256,
                        ),
                    }
                )
                if not getattr(args, "no_registry_write", False):
                    _upsert_launch_registry(root, registry_rows[-1:])
    if not getattr(args, "no_registry_write", False):
        _upsert_launch_registry(root, registry_rows)
    return root


def monitor(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    run_folds, experiment_seeds, run_variants = _experiment_scope(root)
    rows = []
    for fold in run_folds:
        for seed in experiment_seeds:
            for variant in run_variants:
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


_RUN_FINGERPRINT_EXCLUDED_CONFIG_FIELDS = {
    "checkpoints_path",
    "metrics_path",
    "output_root",
    "run_fingerprint_sha256",
}


def _frozen_commit(root: Path) -> str:
    path = root / "inputs/git_state.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("commit=") or not first_line.removeprefix("commit=").strip():
        raise ValueError(f"Invalid frozen git state: {path}")
    return first_line.removeprefix("commit=").strip()


def _run_fingerprint(
    root: Path,
    *,
    fold: str,
    seed: int,
    variant: str,
    training: dict[str, Any],
    overrides: dict[str, Any],
    parent_checkpoint_sha256: str,
) -> str:
    """Hash the complete scientific identity of an RQ1 training run."""

    resolved = {**training, **overrides}
    semantic_config = {
        key: value
        for key, value in resolved.items()
        if key not in _RUN_FINGERPRINT_EXCLUDED_CONFIG_FIELDS
    }
    artifact_fields = (
        "data_path",
        "news_workbook_path",
        "split_manifest_path",
        "text_transform_path",
        "surface_support_path",
        "text_alignment_plan_path",
        "matching_negative_source_plan_path",
    )
    artifact_hashes: dict[str, str] = {}
    for field in artifact_fields:
        path = _config_artifact_path(resolved.get(field))
        artifact_hashes[field] = "" if path is None else sha256_file(path)
    design = _experiment_design(root)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION_V3,
        "critic_architecture_version": CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING,
        "matching_negative_source_plan_version": MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
        "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
        "epoch_policy": EPOCH_POLICY_CONTINUATION_ANCHOR,
        "experiment_design_schema_version": int(
            design.get("experiment_design_schema_version", 0)
        ),
        "frozen_commit": _frozen_commit(root),
        "fold": str(fold),
        "seed": int(seed),
        "variant": str(variant),
        "training_stage": VARIANT_STAGES[variant],
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "epoch_anchor": {
            "policy": EPOCH_POLICY_CONTINUATION_ANCHOR,
            "anchor_variant": (
                CONTINUATION_VARIANT
                if variant in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                else ""
            ),
            "comparison_epoch": (
                int(overrides["num_epochs"])
                if variant in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                else None
            ),
        },
        "semantic_config": semantic_config,
        "artifact_sha256": artifact_hashes,
    }
    return canonical_payload_sha256(payload)


def _checkpoint_protocol_fields(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_schema_version": int(
            checkpoint.get("checkpoint_schema_version", 0)
        ),
        "training_protocol_version": str(
            checkpoint.get("training_protocol_version", "")
        ),
        "critic_architecture_version": str(
            checkpoint.get("critic_architecture_version", "")
        ),
        "run_fingerprint_sha256": str(
            checkpoint.get("run_fingerprint_sha256", "")
        ),
        "epoch": int(checkpoint.get("epoch", 0)),
        "initial_generator_state_sha256": str(
            checkpoint.get("initial_generator_state_sha256", "")
        ),
        "initial_critic_state_sha256": str(
            checkpoint.get("initial_critic_state_sha256", "")
        ),
        "text_alignment_plan_sha256": str(
            checkpoint.get("text_alignment_plan_sha256", "")
        ),
        "matching_negative_source_plan_sha256": str(
            checkpoint.get("matching_negative_source_plan_sha256", "")
        ),
        "scheduler_horizon_epochs": int(
            checkpoint.get("scheduler_horizon_epochs", 0)
        ),
    }


def _scheduler_trace_through_epoch(
    run_dir: Path,
    *,
    comparison_epoch: int,
    require_exact_end: bool,
) -> tuple[pd.DataFrame, str]:
    """Load the frozen learning-rate trace used for a paired Stage-B comparison."""

    metrics_path = run_dir / "metrics/training_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    required = {"epoch", "lr_generator", "lr_critic"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(
            f"Training metrics omit scheduler trace columns {missing}: {metrics_path}"
        )
    numeric = metrics.loc[:, ["epoch", "lr_generator", "lr_critic"]].copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric[["lr_generator", "lr_critic"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(f"Scheduler trace contains non-finite values: {metrics_path}")
    epochs = numeric["epoch"].to_numpy(dtype=np.float64)
    if not np.equal(epochs, np.floor(epochs)).all():
        raise ValueError(f"Scheduler trace contains non-integer epochs: {metrics_path}")
    numeric["epoch"] = epochs.astype(np.int64)
    if numeric["epoch"].duplicated().any():
        raise ValueError(f"Scheduler trace contains duplicate epochs: {metrics_path}")
    comparison_epoch = int(comparison_epoch)
    expected_epochs = list(range(1, comparison_epoch + 1))
    trace = numeric.loc[numeric["epoch"] <= comparison_epoch].sort_values(
        "epoch"
    )
    if trace["epoch"].tolist() != expected_epochs:
        raise ValueError(
            "Scheduler trace must contain every epoch 1..E exactly once: "
            f"E={comparison_epoch} path={metrics_path}"
        )
    if require_exact_end and numeric["epoch"].tolist() != expected_epochs:
        raise ValueError(
            "Exact-E Stage-B arm contains epochs outside 1..E: "
            f"E={comparison_epoch} path={metrics_path}"
        )
    trace_payload = [
        {
            "epoch": int(row.epoch),
            "lr_generator": float(row.lr_generator),
            "lr_critic": float(row.lr_critic),
        }
        for row in trace.itertuples(index=False)
    ]
    return trace, canonical_payload_sha256(trace_payload)


def _paired_stage_audit(
    selected: pd.DataFrame,
    resolved_configs: dict[tuple[str, int, str], dict[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    folds: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected_seeds = _normalize_seeds(seeds or SEEDS)
    selected_folds = tuple(str(value) for value in (folds or FOLDS))
    selected_by_run = selected.set_index(["fold", "seed", "variant"], verify_integrity=True)
    sha_cache: dict[Path, str] = {}
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for fold in selected_folds:
        for seed in selected_seeds:
            parent_key = (fold, seed, PARENT_VARIANT)
            parent_row = selected_by_run.loc[parent_key]
            expected_parent_path = Path(str(parent_row["checkpoint_path"])).resolve()
            expected_parent_sha = str(parent_row["checkpoint_sha256"])
            parent_payload = resolved_configs[parent_key]
            parent_training = dict(parent_payload.get("training") or parent_payload)
            expected_transform_path = _config_artifact_path(parent_training.get("text_transform_path"))
            expected_transform_sha = _cached_sha256(expected_transform_path, sha_cache)
            matched_key = (fold, seed, TEXT_RESIDUAL_VARIANT)
            matched_payload = resolved_configs[matched_key]
            matched_training = dict(matched_payload.get("training") or matched_payload)
            residual_critic_mode = str(
                matched_training.get("critic_conditioning_mode", "projection")
            )
            common_expected = {
                "text_preprocessing_mode": "pca",
                "normalize_text_embedding": False,
                "conditioning_mode": "residual_film",
                "critic_conditioning_mode": residual_critic_mode,
                "freeze_backbone_epochs": 5,
            }
            expected_variant_values = {
                CONTINUATION_VARIANT: {
                    "text_embedding_mode": "zero_lp",
                    "text_alignment_mode": "matched",
                    "lambda_film": 0.0,
                    "lambda_mismatch": 0.0,
                    "lambda_critic_matching": 0.0,
                    "lambda_generator_matching": 0.0,
                },
                TEXT_RESIDUAL_VARIANT: {
                    "text_embedding_mode": "lp",
                    "text_alignment_mode": "matched",
                    "lambda_film": 1.0e-4,
                    "lambda_mismatch": float(matched_training.get("lambda_mismatch", 0.0)),
                    "lambda_critic_matching": float(
                        matched_training.get("lambda_critic_matching", 0.0)
                    ),
                    "lambda_generator_matching": float(
                        matched_training.get("lambda_generator_matching", 0.0)
                    ),
                },
                SHUFFLED_RESIDUAL_VARIANT: {
                    "text_embedding_mode": "lp",
                    "text_alignment_mode": "permuted",
                    "lambda_film": 1.0e-4,
                    "lambda_mismatch": float(matched_training.get("lambda_mismatch", 0.0)),
                    "lambda_critic_matching": float(
                        matched_training.get("lambda_critic_matching", 0.0)
                    ),
                    "lambda_generator_matching": float(
                        matched_training.get("lambda_generator_matching", 0.0)
                    ),
                },
            }
            v3_protocol = (
                str(matched_training.get("training_protocol_version", ""))
                == TRAINING_PROTOCOL_VERSION_V3
            )
            stage_checkpoint_fields: dict[str, dict[str, Any]] = {}
            expected_initial_generator_sha = ""
            expected_initial_critic_sha = ""
            expected_alignment_plan_sha = ""
            expected_negative_plan_sha = ""
            continuation_epoch = 0
            scheduler_trace_hashes: dict[str, str] = {}
            scheduler_trace_errors: dict[str, list[str]] = {
                variant: [] for variant in PAIRED_STAGE_B_VARIANTS
            }
            if v3_protocol:
                missing_stage_rows = [
                    variant
                    for variant in PAIRED_STAGE_B_VARIANTS
                    if (fold, seed, variant) not in selected_by_run.index
                ]
                if missing_stage_rows:
                    failures.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "variant": "stage_b_set",
                            "errors": [
                                f"missing_selected_variant:{variant}"
                                for variant in missing_stage_rows
                            ],
                        }
                    )
                    continue
                stage_checkpoint_fields = {
                    variant: _checkpoint_protocol_fields(
                        Path(
                            str(
                                selected_by_run.loc[
                                    (fold, seed, variant), "checkpoint_path"
                                ]
                            )
                        )
                    )
                    for variant in PAIRED_STAGE_B_VARIANTS
                }
                expected_initial_generator_sha = stage_checkpoint_fields[
                    CONTINUATION_VARIANT
                ]["initial_generator_state_sha256"]
                expected_initial_critic_sha = stage_checkpoint_fields[
                    CONTINUATION_VARIANT
                ]["initial_critic_state_sha256"]
                if "text_alignment_plan_sha256" in selected_by_run.columns:
                    expected_alignment_plan_sha = str(
                        selected_by_run.loc[
                            (fold, seed, CONTINUATION_VARIANT),
                            "text_alignment_plan_sha256",
                        ]
                    )
                if (
                    "matching_negative_source_plan_sha256"
                    in selected_by_run.columns
                ):
                    expected_negative_plan_sha = str(
                        selected_by_run.loc[
                            (fold, seed, CONTINUATION_VARIANT),
                            "matching_negative_source_plan_sha256",
                        ]
                    )
                continuation_epoch = int(
                    selected_by_run.loc[
                        (fold, seed, CONTINUATION_VARIANT), "selected_epoch"
                    ]
                )
                scheduler_traces: dict[str, pd.DataFrame] = {}
                for variant in PAIRED_STAGE_B_VARIANTS:
                    selected_row = selected_by_run.loc[(fold, seed, variant)]
                    run_dir = Path(str(selected_row["run_dir"])).resolve()
                    expected_checkpoint_name = (
                        "film_wgan_best.pt"
                        if variant == CONTINUATION_VARIANT
                        else "film_wgan_final.pt"
                    )
                    expected_checkpoint_path = (
                        run_dir / "checkpoints" / expected_checkpoint_name
                    ).resolve()
                    selected_checkpoint_path = Path(
                        str(selected_row["checkpoint_path"])
                    ).resolve()
                    if selected_checkpoint_path != expected_checkpoint_path:
                        scheduler_trace_errors[variant].append(
                            "comparison_checkpoint_role_mismatch"
                        )
                    if int(selected_row["selected_epoch"]) != continuation_epoch:
                        scheduler_trace_errors[variant].append(
                            "comparison_epoch_mismatch"
                        )
                    if variant == CONTINUATION_VARIANT:
                        try:
                            if _best_epoch(run_dir) != continuation_epoch:
                                scheduler_trace_errors[variant].append(
                                    "continuation_best_epoch_mismatch"
                                )
                        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                            scheduler_trace_errors[variant].append(
                                "continuation_best_epoch_unreadable:"
                                f"{type(exc).__name__}"
                            )
                    try:
                        trace, trace_sha = _scheduler_trace_through_epoch(
                            run_dir,
                            comparison_epoch=continuation_epoch,
                            require_exact_end=(variant != CONTINUATION_VARIANT),
                        )
                    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                        scheduler_trace_errors[variant].append(
                            f"scheduler_trace_unreadable:{type(exc).__name__}"
                        )
                    else:
                        scheduler_traces[variant] = trace
                        scheduler_trace_hashes[variant] = trace_sha
                reference_trace = scheduler_traces.get(CONTINUATION_VARIANT)
                if reference_trace is not None:
                    reference_values = reference_trace[
                        ["epoch", "lr_generator", "lr_critic"]
                    ].to_numpy(dtype=np.float64)
                    for variant in (
                        TEXT_RESIDUAL_VARIANT,
                        SHUFFLED_RESIDUAL_VARIANT,
                    ):
                        candidate_trace = scheduler_traces.get(variant)
                        if candidate_trace is None:
                            continue
                        candidate_values = candidate_trace[
                            ["epoch", "lr_generator", "lr_critic"]
                        ].to_numpy(dtype=np.float64)
                        if not np.array_equal(candidate_values, reference_values):
                            scheduler_trace_errors[variant].append(
                                "scheduler_trace_mismatch"
                            )

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
                checkpoint_fields = stage_checkpoint_fields.get(variant, {})
                errors.extend(scheduler_trace_errors.get(variant, []))

                if configured_parent_path != expected_parent_path:
                    errors.append("parent_checkpoint_path_mismatch")
                if configured_parent_sha != expected_parent_sha:
                    errors.append("parent_checkpoint_sha256_mismatch")
                if transform_path != expected_transform_path:
                    errors.append("text_transform_path_mismatch")
                if not expected_transform_sha or transform_sha != expected_transform_sha:
                    errors.append("text_transform_sha256_mismatch")

                expected_values = {**common_expected, **expected_variant_values[variant]}
                legacy_optional_defaults = {
                    "lambda_critic_matching": 0.0,
                    "lambda_generator_matching": 0.0,
                }
                mismatched_fields = sorted(
                    field
                    for field, expected in expected_values.items()
                    if training.get(
                        field,
                        legacy_optional_defaults.get(field),
                    )
                    != expected
                )
                errors.extend(f"config_mismatch:{field}" for field in mismatched_fields)
                if v3_protocol:
                    if (
                        not expected_initial_generator_sha
                        or checkpoint_fields["initial_generator_state_sha256"]
                        != expected_initial_generator_sha
                    ):
                        errors.append("initial_generator_state_sha256_mismatch")
                    if (
                        not expected_initial_critic_sha
                        or checkpoint_fields["initial_critic_state_sha256"]
                        != expected_initial_critic_sha
                    ):
                        errors.append("initial_critic_state_sha256_mismatch")
                    selected_row = selected_by_run.loc[(fold, seed, variant)]
                    if (
                        not expected_alignment_plan_sha
                        or str(selected_row["text_alignment_plan_sha256"])
                        != expected_alignment_plan_sha
                        or checkpoint_fields["text_alignment_plan_sha256"]
                        != expected_alignment_plan_sha
                    ):
                        errors.append("text_alignment_plan_sha256_mismatch")
                    if (
                        not expected_negative_plan_sha
                        or str(
                            selected_row[
                                "matching_negative_source_plan_sha256"
                            ]
                        )
                        != expected_negative_plan_sha
                        or checkpoint_fields[
                            "matching_negative_source_plan_sha256"
                        ]
                        != expected_negative_plan_sha
                    ):
                        errors.append(
                            "matching_negative_source_plan_sha256_mismatch"
                        )
                    expected_schedule = {
                        "scheduler_horizon_epochs": 100,
                        "early_stopping_patience": 15,
                        "num_epochs": (
                            100
                            if variant == CONTINUATION_VARIANT
                            else continuation_epoch
                        ),
                        "use_early_stopping": (
                            variant == CONTINUATION_VARIANT
                        ),
                    }
                    for field, expected in expected_schedule.items():
                        if training.get(field) != expected:
                            errors.append(f"schedule_mismatch:{field}")
                    if checkpoint_fields["scheduler_horizon_epochs"] != 100:
                        errors.append("checkpoint_scheduler_horizon_mismatch")
                    if (
                        variant
                        in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                        and int(selected_row["selected_epoch"])
                        != continuation_epoch
                    ):
                        errors.append("comparison_epoch_mismatch")
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
                        "initial_generator_state_sha256": checkpoint_fields.get(
                            "initial_generator_state_sha256", ""
                        ),
                        "initial_critic_state_sha256": checkpoint_fields.get(
                            "initial_critic_state_sha256", ""
                        ),
                        "text_alignment_plan_sha256": checkpoint_fields.get(
                            "text_alignment_plan_sha256", ""
                        ),
                        "matching_negative_source_plan_sha256": (
                            checkpoint_fields.get(
                                "matching_negative_source_plan_sha256", ""
                            )
                        ),
                        "comparison_epoch": int(
                            selected_by_run.loc[
                                (fold, seed, variant), "selected_epoch"
                            ]
                            if (fold, seed, variant) in selected_by_run.index
                            and "selected_epoch" in selected_by_run.columns
                            else 0
                        ),
                        "scheduler_trace_sha256": scheduler_trace_hashes.get(
                            variant, ""
                        ),
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


def _validation_metric_at_epoch(
    run_dir: Path,
    *,
    epoch: int,
    metric: str = "val_mae",
) -> float:
    metrics_path = run_dir / "metrics/training_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    if "epoch" not in metrics.columns or metric not in metrics.columns:
        raise ValueError(f"Training metrics omit epoch/{metric}: {metrics_path}")
    selected = metrics[
        pd.to_numeric(metrics["epoch"], errors="coerce") == int(epoch)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {metric} row at epoch={epoch}: {metrics_path}"
        )
    return float(selected.iloc[0][metric])


def collect_checkpoints(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    v3_design = _is_v3_design(root)
    if v3_design:
        verify_existing(argparse.Namespace(experiment_root=str(root)))
    else:
        _assert_frozen_v2_worktree(root)
    run_folds, experiment_seeds, run_variants = _experiment_scope(root)
    rows = []
    resolved_configs: dict[tuple[str, int, str], dict[str, Any]] = {}
    artifact_sha_cache: dict[Path, str] = {}
    for fold in run_folds:
        fold_payload = _read_yaml(_fold_config(root, fold))
        fold_training = dict(fold_payload.get("training") or fold_payload)
        residual_critic_mode = str(
            fold_training.get("critic_conditioning_mode", "projection")
        )
        for seed in experiment_seeds:
            parent_run = _completed_run(
                root
                / "training_runs"
                / PARENT_VARIANT
                / fold
                / f"seed_{seed}"
            )
            if parent_run is None:
                raise FileNotFoundError(
                    f"Missing completed parent run: {fold}/seed_{seed}"
                )
            parent_checkpoint = parent_run / "checkpoints/film_wgan_best.pt"
            parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
            continuation_anchor_epoch: int | None = None
            if v3_design:
                _continuation_run, continuation_anchor_epoch = _continuation_anchor(
                    root,
                    fold,
                    seed,
                )
            for variant in run_variants:
                run_root = root / "training_runs" / variant / fold / f"seed_{seed}"
                run_dir = _completed_run(run_root)
                if run_dir is None:
                    raise FileNotFoundError(f"Missing completed run: {variant}/{fold}/seed_{seed}")
                variant_anchor_epoch = (
                    continuation_anchor_epoch
                    if variant
                    in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                    else None
                )
                overrides = _variant_overrides(
                    variant,
                    fold_config=fold_payload,
                    output_root=run_root,
                    seed=seed,
                    parent_checkpoint=(
                        parent_checkpoint
                        if variant in PAIRED_STAGE_B_VARIANTS
                        else None
                    ),
                    continuation_anchor_epoch=variant_anchor_epoch,
                )
                run_fingerprint_sha256 = ""
                if v3_design:
                    run_fingerprint_sha256 = _run_fingerprint(
                        root,
                        fold=fold,
                        seed=seed,
                        variant=variant,
                        training=fold_training,
                        overrides=overrides,
                        parent_checkpoint_sha256=(
                            parent_checkpoint_sha256
                            if variant in PAIRED_STAGE_B_VARIANTS
                            else ""
                        ),
                    )
                checkpoint, epoch = _comparison_checkpoint(
                    run_dir,
                    variant=variant,
                    continuation_anchor_epoch=variant_anchor_epoch,
                    v3_protocol=v3_design,
                )
                if epoch <= 10:
                    raise ValueError(
                        f"Selected checkpoint must be after epoch 10: {run_dir} epoch={epoch}"
                    )
                _validate_run_protocol(
                    run_dir,
                    critic_mode=_variant_critic_mode(
                        variant,
                        residual_critic_mode=residual_critic_mode,
                    ),
                    require_v3=v3_design,
                    require_schema5=(
                        not v3_design
                        and residual_critic_mode.strip().lower()
                        == "transition_matching"
                    ),
                    expected_fingerprint=run_fingerprint_sha256,
                    checkpoint_path=checkpoint,
                )
                best = json.loads((run_dir / "metrics/best_checkpoint.json").read_text(encoding="utf-8"))
                resolved_config_path = run_dir / "metrics/training_resolved_config.yaml"
                resolved_payload = _read_yaml(resolved_config_path)
                resolved_configs[(fold, seed, variant)] = resolved_payload
                resolved_training = dict(resolved_payload.get("training") or resolved_payload)
                if v3_design:
                    if (
                        str(resolved_training.get("run_fingerprint_sha256", ""))
                        != run_fingerprint_sha256
                    ):
                        raise ValueError(
                            f"Resolved run fingerprint mismatch: {variant}/{fold}/seed_{seed}"
                        )
                    mismatched_overrides = sorted(
                        key
                        for key, expected in overrides.items()
                        if resolved_training.get(key) != expected
                    )
                    if mismatched_overrides:
                        raise ValueError(
                            "Resolved config differs from frozen v3 orchestration for "
                            f"{variant}/{fold}/seed_{seed}: {mismatched_overrides}"
                        )
                configured_parent_path = _config_artifact_path(
                    resolved_training.get("initial_generator_checkpoint_path")
                )
                transform_path = _config_artifact_path(resolved_training.get("text_transform_path"))
                alignment_path = _config_artifact_path(
                    resolved_training.get("text_alignment_plan_path")
                )
                negative_path = _config_artifact_path(
                    resolved_training.get("matching_negative_source_plan_path")
                )
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
                        "epoch_policy": (
                            EPOCH_POLICY_CONTINUATION_ANCHOR
                            if v3_design
                            else "legacy_best_checkpoint"
                        ),
                        "anchor_variant": (
                            CONTINUATION_VARIANT
                            if v3_design
                            and variant
                            in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}
                            else ""
                        ),
                        "comparison_epoch": epoch,
                        "comparison_checkpoint": str(checkpoint),
                        "comparison_checkpoint_sha256": sha256_file(checkpoint),
                        "run_fingerprint_sha256": run_fingerprint_sha256,
                        "text_alignment_plan_sha256": _cached_sha256(
                            alignment_path,
                            artifact_sha_cache,
                        ),
                        "matching_negative_source_plan_sha256": _cached_sha256(
                            negative_path,
                            artifact_sha_cache,
                        ),
                        "selected_epoch": epoch,
                        "validation_surface_mae": _validation_metric_at_epoch(
                            run_dir,
                            epoch=epoch,
                        ),
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
        "lambda_critic_matching",
        "lambda_generator_matching",
        "initial_generator_checkpoint_path",
        "freeze_backbone_epochs",
        "seed",
        "output_root",
        "checkpoints_path",
        "metrics_path",
        "num_epochs",
        "scheduler_horizon_epochs",
        "use_early_stopping",
        "early_stopping_patience",
        "run_fingerprint_sha256",
    }
    audit_rows = []
    failures = []
    for fold in run_folds:
        reference = resolved_configs[
            (fold, experiment_seeds[0], "pair_pca_no_text_residual")
        ]
        reference = dict(reference.get("training") or reference)
        for seed in experiment_seeds:
            for variant in run_variants:
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
        folds=run_folds,
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


_V3_FROZEN_GENERATE_FIELDS = (
    "evaluation_noise_seed",
    "mc_samples",
    "reweight_beta_mode",
    "reweight_beta",
    "quantiles",
    "calibration_levels",
    "arbitrage_violation_tolerance",
    "aggregation_mode",
    "residual_blend_alpha",
    "save_json",
    "save_plots",
    "save_full_atm_timeseries",
)


def _materialize_generation_config(
    run_dir: Path,
    *,
    frozen_generate: dict[str, Any],
    require_explicit_v3_fields: bool,
) -> tuple[Path, dict[str, Any]]:
    """Merge one run's flat training snapshot with the frozen sample protocol.

    Training writes only the resolved ``training`` section.  Passing that file
    directly to ``generate-result`` silently falls back to sample defaults for
    fields such as ``reweight_beta``.  Persist a merged config so list-valued
    fields and every other frozen generation value reach the actual CLI.
    """

    if require_explicit_v3_fields:
        missing = sorted(
            field for field in _V3_FROZEN_GENERATE_FIELDS if field not in frozen_generate
        )
        if missing:
            raise ValueError(
                "The frozen v3 generate_result protocol omits required fields: "
                f"{missing}."
            )
    resolved_path = run_dir / "metrics/training_resolved_config.yaml"
    resolved_payload = _read_yaml(resolved_path)
    resolved_training = (
        dict(resolved_payload["training"])
        if "training" in resolved_payload
        else dict(resolved_payload)
    )
    resolved_sample = config_to_dict(
        build_sample_config(
            training_values=resolved_training,
            generate_values=frozen_generate,
        )
    )
    target = run_dir / "metrics/validation_pilot_generate_config.yaml"
    _write_yaml(
        target,
        {
            "training": resolved_training,
            "generate_result": resolved_sample,
        },
    )
    return target, resolved_sample


def generate_matrix(args: argparse.Namespace) -> Path:
    _assert_py312()
    root = _resolve_root(args.experiment_root)
    v3_design = _is_v3_design(root)
    if v3_design:
        verify_existing(argparse.Namespace(experiment_root=str(root)))
    run_folds, experiment_seeds, run_variants = _experiment_scope(root)
    frozen_config = _read_yaml(
        root / "inputs/configs/train_rq1_pair_textbase.yaml"
    )
    generate_config = dict(frozen_config.get("generate_result") or {})
    generation_split = str(generate_config.get("split", "test")).strip().lower()
    if generation_split not in {"val", "test"}:
        raise ValueError(
            "RQ1 matrix generation supports only an explicit val or test split."
        )
    if v3_design and generation_split != "val":
        raise RuntimeError(
            "The v3 validation pilot is locked to split=val; outer-test "
            "generation is forbidden in this experiment root."
        )
    generation_output_dir = str(
        generate_config.get(
            "output_dir",
            "validation_pilot_json"
            if generation_split == "val"
            else "development_test_json",
        )
    ).strip()
    output_path = Path(generation_output_dir)
    if (
        not generation_output_dir
        or output_path.is_absolute()
        or ".." in output_path.parts
    ):
        raise ValueError("generate_result.output_dir must be a safe relative path.")
    if (
        str(generate_config.get("selection_mode", "all")).strip().lower()
        != "all"
        or int(generate_config.get("selection_count", 0)) != 0
    ):
        raise ValueError(
            "RQ1 matrix generation requires selection_mode=all and selection_count=0."
        )
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not selected_path.is_file():
        collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    selected = pd.read_csv(selected_path)
    if v3_design:
        expected_keys = {
            (fold, int(seed), variant)
            for fold in run_folds
            for seed in experiment_seeds
            for variant in run_variants
        }
        observed_keys = {
            (str(row.fold), int(row.seed), str(row.variant))
            for row in selected.itertuples(index=False)
        }
        if observed_keys != expected_keys or len(selected) != len(expected_keys):
            raise ValueError(
                "Selected v3 checkpoints do not exactly match the frozen run scope."
            )
    registry_rows = []
    for row in selected.itertuples(index=False):
        run_dir = Path(row.run_dir)
        checkpoint_path = Path(row.checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        generation_config_path, resolved_generate_config = (
            _materialize_generation_config(
                run_dir,
                frozen_generate=generate_config,
                require_explicit_v3_fields=v3_design,
            )
        )
        resolved_payload = _read_yaml(
            run_dir / "metrics/training_resolved_config.yaml"
        )
        resolved_training = dict(
            resolved_payload.get("training") or resolved_payload
        )
        effective_generate_config = {
            **resolved_generate_config,
            "checkpoint_path": str(checkpoint_path),
        }
        generation_config_sha256 = sha256_file(generation_config_path)
        if v3_design:
            if checkpoint_sha256 != str(row.checkpoint_sha256):
                raise ValueError(
                    f"Selected checkpoint SHA changed before generation: {checkpoint_path}"
                )
            expected_fingerprint = str(row.run_fingerprint_sha256)
            if (
                str(resolved_training.get("run_fingerprint_sha256", ""))
                != expected_fingerprint
            ):
                raise ValueError(
                    f"Resolved fingerprint changed before generation: {run_dir}"
                )
            _validate_run_protocol(
                run_dir,
                critic_mode=str(
                    resolved_training.get(
                        "critic_conditioning_mode",
                        "transition_matching",
                    )
                ),
                require_v3=True,
                expected_fingerprint=expected_fingerprint,
                checkpoint_path=checkpoint_path,
            )
        output_dir = run_dir / generation_output_dir
        summary_path = output_dir / "summary.csv"
        generation_fingerprint_sha256 = ""
        generation_fingerprint_path = output_dir / "generation_fingerprint.json"
        generation_fingerprint_valid = not v3_design
        if v3_design:
            generation_fingerprint_sha256 = canonical_payload_sha256(
                {
                    "run_fingerprint_sha256": str(row.run_fingerprint_sha256),
                    "checkpoint_sha256": checkpoint_sha256,
                    "generation_config_sha256": generation_config_sha256,
                    "effective_generate_config": effective_generate_config,
                }
            )
            if generation_fingerprint_path.is_file():
                recorded_generation = json.loads(
                    generation_fingerprint_path.read_text(encoding="utf-8")
                )
                generation_fingerprint_valid = (
                    str(recorded_generation.get("generation_fingerprint_sha256", ""))
                    == generation_fingerprint_sha256
                )
        split_index = 1 if generation_split == "val" else 2
        expected = _fold_counts(root, str(row.fold))[split_index]
        if (
            summary_path.is_file()
            and len(pd.read_csv(summary_path)) == expected
            and generation_fingerprint_valid
        ):
            status = "reused"
        else:
            command = [
                sys.executable,
                "scripts/film_wgan/main.py",
                "generate-result",
                "--config",
                str(generation_config_path),
                "--checkpoint",
                str(checkpoint_path),
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
        if v3_design:
            _verify_generated_sample_protocol(
                output_dir,
                expected_count=expected,
                effective_generate_config=effective_generate_config,
            )
            if status == "completed":
                _write_json(
                    generation_fingerprint_path,
                    {
                        "generation_fingerprint_sha256": generation_fingerprint_sha256,
                        "run_fingerprint_sha256": str(row.run_fingerprint_sha256),
                        "checkpoint_sha256": checkpoint_sha256,
                        "split": generation_split,
                        "generation_config_sha256": generation_config_sha256,
                        "effective_generate_config_sha256": canonical_payload_sha256(
                            effective_generate_config
                        ),
                    },
                )
        registry_rows.append(
            {
                "fold": row.fold,
                "seed": int(row.seed),
                "variant": row.variant,
                "status": status,
                "split": generation_split,
                "output_dir": generation_output_dir,
                "summary_path": str(summary_path),
                "sample_count": expected,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "run_fingerprint_sha256": (
                    str(row.run_fingerprint_sha256) if v3_design else ""
                ),
                "generation_fingerprint_sha256": (
                    generation_fingerprint_sha256 if v3_design else ""
                ),
                "generation_config_path": str(generation_config_path),
                "generation_config_sha256": generation_config_sha256,
            }
        )
        pd.DataFrame(registry_rows).to_csv(
            root / "registry/generate_registry.csv",
            index=False,
        )
    return root


VALIDATION_PILOT_CONTRASTS = (
    (
        TEXT_RESIDUAL_VARIANT,
        CONTINUATION_VARIANT,
        "matched_vs_continuation",
    ),
    (
        TEXT_RESIDUAL_VARIANT,
        SHUFFLED_RESIDUAL_VARIANT,
        "matched_vs_shuffled",
    ),
    (
        CONTINUATION_VARIANT,
        PARENT_VARIANT,
        "continuation_vs_parent",
    ),
)


def _validation_duplicate_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    exact_embedding = (
        frame["exact_embedding_duplicate_with_train"].to_numpy(dtype=int) > 0
    )
    exact_text = frame["exact_text_duplicate_with_train"].to_numpy(dtype=int) > 0
    near_text = (
        frame["near_text_candidate_duplicate_with_train"].to_numpy(dtype=int)
        > 0
    )
    return {
        "all_pairs": np.ones(len(frame), dtype=bool),
        "exclude_exact_embedding_seen_in_train": ~exact_embedding,
        "exclude_exact_text_seen_in_train": ~exact_text,
        "exclude_any_exact_duplicate_seen_in_train": ~(
            exact_embedding | exact_text
        ),
        "exclude_near_text_seen_in_train": ~near_text,
        "exclude_any_exact_or_near_duplicate_seen_in_train": ~(
            exact_embedding | exact_text | near_text
        ),
    }


def _json_payloads_by_pair(output_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "samples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair_id = str(payload.get("surface_pair_id", ""))
        if not pair_id or pair_id in payloads:
            raise ValueError(f"Invalid/duplicate generated surface_pair_id: {path}")
        payloads[pair_id] = payload
    return payloads


def _verify_generated_sample_protocol(
    output_dir: Path,
    *,
    expected_count: int,
    effective_generate_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Close the loop between the frozen request and emitted sample metadata."""

    payloads = _json_payloads_by_pair(output_dir)
    if len(payloads) != int(expected_count):
        raise ValueError(
            "Generated JSON count does not match the frozen split: "
            f"{len(payloads)} != {expected_count} in {output_dir}."
        )
    expected_mc_samples = int(effective_generate_config["mc_samples"])
    expected_split = str(effective_generate_config["split"])
    expected_aggregation = str(effective_generate_config["aggregation_mode"])
    expected_alignment = str(effective_generate_config["text_alignment_mode"])
    expected_blend = float(effective_generate_config["residual_blend_alpha"])
    beta_mode = str(effective_generate_config["reweight_beta_mode"]).strip().lower()
    expected_fixed_beta = float(effective_generate_config["reweight_beta"])
    errors: list[str] = []
    for pair_id, payload in payloads.items():
        metadata = dict(payload.get("metadata") or {})
        observed = {
            "mc_samples": metadata.get("mc_samples"),
            "split": metadata.get("split"),
            "aggregation_mode": metadata.get("aggregation_mode"),
            "text_alignment_mode": metadata.get("text_alignment_mode"),
            "residual_blend_alpha": metadata.get("residual_blend_alpha"),
        }
        if (
            int(observed["mc_samples"] or -1) != expected_mc_samples
            or str(observed["split"]) != expected_split
            or str(observed["aggregation_mode"]) != expected_aggregation
            or str(observed["text_alignment_mode"]) != expected_alignment
            or not np.isclose(
                float(observed["residual_blend_alpha"]),
                expected_blend,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            errors.append(f"{pair_id}:metadata={observed}")
            continue
        if beta_mode == "fixed" and not np.isclose(
            float(payload.get("effective_beta", float("nan"))),
            expected_fixed_beta,
            rtol=0.0,
            atol=1.0e-12,
        ):
            errors.append(
                f"{pair_id}:effective_beta={payload.get('effective_beta')!r} "
                f"expected={expected_fixed_beta}"
            )
    if errors:
        raise ValueError(
            "Generated samples do not match the effective frozen protocol: "
            + "; ".join(errors[:5])
        )
    return payloads


def _collect_text_swap_sensitivity(
    *,
    root: Path,
    selected: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Regenerate matched/shuffled arms with their frozen positive texts swapped."""

    frozen_payload = _read_yaml(
        root / "inputs/configs/train_rq1_pair_textbase.yaml"
    )
    frozen_generate = dict(frozen_payload.get("generate_result") or {})
    evaluation_noise_seed = int(
        frozen_generate.get("evaluation_noise_seed", -1)
    )
    if evaluation_noise_seed < 0:
        raise ValueError("Text-swap sensitivity requires a fixed evaluation noise seed.")
    registry_by_key = registry.set_index(
        ["fold", "seed", "variant"], verify_integrity=True
    )
    rows: list[dict[str, Any]] = []
    for record in selected.itertuples(index=False):
        variant = str(record.variant)
        if variant not in {TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT}:
            continue
        key = (str(record.fold), int(record.seed), variant)
        registry_record = registry_by_key.loc[key]
        checkpoint_path = Path(str(record.checkpoint_path)).resolve()
        if (
            Path(str(registry_record["checkpoint_path"])).resolve()
            != checkpoint_path
            or str(registry_record["checkpoint_sha256"])
            != str(record.checkpoint_sha256)
            or sha256_file(checkpoint_path) != str(record.checkpoint_sha256)
        ):
            raise ValueError(
                f"Text-swap checkpoint identity changed after selection: {key}"
            )
        resolved_payload = _read_yaml(
            Path(record.run_dir) / "metrics/training_resolved_config.yaml"
        )
        resolved_training = dict(
            resolved_payload.get("training") or resolved_payload
        )
        generation_config_path, resolved_generate_config = (
            _materialize_generation_config(
                Path(record.run_dir),
                frozen_generate=frozen_generate,
                require_explicit_v3_fields=True,
            )
        )
        _validate_run_protocol(
            Path(record.run_dir),
            critic_mode=str(
                resolved_training.get(
                    "critic_conditioning_mode", "transition_matching"
                )
            ),
            require_v3=True,
            expected_fingerprint=str(record.run_fingerprint_sha256),
            checkpoint_path=checkpoint_path,
        )
        base_output_dir = Path(str(registry_record["summary_path"])).parent
        run_dir = Path(record.run_dir)
        swap_output_name = "validation_pilot_text_swap_json"
        swap_output_dir = run_dir / swap_output_name
        swapped_alignment = (
            "permuted" if variant == TEXT_RESIDUAL_VARIANT else "matched"
        )
        effective_swap_generate_config = {
            **resolved_generate_config,
            "checkpoint_path": str(checkpoint_path),
            "output_dir": swap_output_name,
            "text_alignment_mode": swapped_alignment,
            "evaluation_noise_seed": evaluation_noise_seed,
        }
        generation_config_sha256 = sha256_file(generation_config_path)
        swap_fingerprint_sha256 = canonical_payload_sha256(
            {
                "run_fingerprint_sha256": str(record.run_fingerprint_sha256),
                "checkpoint_sha256": str(record.checkpoint_sha256),
                "generation_config_sha256": generation_config_sha256,
                "effective_generate_config": effective_swap_generate_config,
            }
        )
        swap_fingerprint_path = swap_output_dir / "generation_fingerprint.json"
        swap_fingerprint_valid = False
        if swap_fingerprint_path.is_file():
            recorded_swap = json.loads(
                swap_fingerprint_path.read_text(encoding="utf-8")
            )
            swap_fingerprint_valid = (
                str(recorded_swap.get("generation_fingerprint_sha256", ""))
                == swap_fingerprint_sha256
            )
        expected = _fold_counts(root, str(record.fold))[1]
        effective_base_generate_config = {
            **resolved_generate_config,
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(registry_record["output_dir"]),
        }
        base_payloads = _verify_generated_sample_protocol(
            base_output_dir,
            expected_count=expected,
            effective_generate_config=effective_base_generate_config,
        )
        swapped_payloads = _json_payloads_by_pair(swap_output_dir)
        generated_swap = len(swapped_payloads) != expected or not swap_fingerprint_valid
        if generated_swap:
            command = [
                sys.executable,
                "scripts/film_wgan/main.py",
                "generate-result",
                "--config",
                str(generation_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                swap_output_name,
                "--set",
                f"text_alignment_mode={swapped_alignment}",
                "--set",
                f"evaluation_noise_seed={evaluation_noise_seed}",
            ]
            _run(
                command,
                log_path=(
                    root
                    / "logs/generate_text_swap"
                    / variant
                    / str(record.fold)
                    / f"seed_{record.seed}.log"
                ),
            )
        swapped_payloads = _verify_generated_sample_protocol(
            swap_output_dir,
            expected_count=expected,
            effective_generate_config=effective_swap_generate_config,
        )
        if generated_swap:
            _write_json(
                swap_fingerprint_path,
                {
                    "generation_fingerprint_sha256": swap_fingerprint_sha256,
                    "run_fingerprint_sha256": str(record.run_fingerprint_sha256),
                    "checkpoint_sha256": str(record.checkpoint_sha256),
                    "split": "val",
                    "text_alignment_mode": swapped_alignment,
                    "evaluation_noise_seed": evaluation_noise_seed,
                    "generation_config_sha256": generation_config_sha256,
                    "effective_generate_config_sha256": canonical_payload_sha256(
                        effective_swap_generate_config
                    ),
                },
            )
        if set(swapped_payloads) != set(base_payloads) or len(swapped_payloads) != expected:
            raise ValueError(f"Text-swap validation JSON identities mismatch: {key}")

        for pair_id, native in base_payloads.items():
            swapped = swapped_payloads[pair_id]
            expected_native_alignment = (
                "matched" if variant == TEXT_RESIDUAL_VARIANT else "permuted"
            )
            native_alignment = str(
                native.get("metadata", {}).get("text_alignment_mode", "")
            )
            swapped_alignment_value = str(
                swapped.get("metadata", {}).get("text_alignment_mode", "")
            )
            if (
                native_alignment != expected_native_alignment
                or swapped_alignment_value != swapped_alignment
            ):
                raise ValueError(
                    f"Text-swap alignment binding mismatch: {key}/{pair_id}"
                )
            native_surface = np.asarray(native["generated_surface"], dtype=np.float64)
            swapped_surface = np.asarray(
                swapped["generated_surface"], dtype=np.float64
            )
            current_surface = np.asarray(native["current_surface"], dtype=np.float64)
            support = np.asarray(
                native["evaluation_support_mask"], dtype=bool
            )
            if (
                native_surface.shape != swapped_surface.shape
                or native_surface.shape != current_surface.shape
                or native_surface.shape != support.shape
                or not bool(support.any())
            ):
                raise ValueError(f"Invalid text-swap surface/support shapes: {key}/{pair_id}")
            surface_gap = swapped_surface - native_surface
            native_delta = np.log(np.clip(native_surface, 1.0e-8, None)) - np.log(
                np.clip(current_surface, 1.0e-8, None)
            )
            swapped_delta = np.log(
                np.clip(swapped_surface, 1.0e-8, None)
            ) - np.log(np.clip(current_surface, 1.0e-8, None))
            delta_gap = swapped_delta - native_delta
            rows.append(
                {
                    "fold": str(record.fold),
                    "seed": int(record.seed),
                    "variant": variant,
                    "surface_pair_id": pair_id,
                    "swap_direction": (
                        "matched_native_to_placebo"
                        if variant == TEXT_RESIDUAL_VARIANT
                        else "shuffled_placebo_to_native"
                    ),
                    "native_alignment_mode": native_alignment,
                    "swapped_alignment_mode": swapped_alignment_value,
                    "supported_cell_count": int(support.sum()),
                    "supported_surface_mae_native_vs_swapped": float(
                        np.mean(np.abs(surface_gap[support]))
                    ),
                    "supported_log_delta_rms_native_vs_swapped": float(
                        np.sqrt(np.mean(np.square(delta_gap[support])))
                    ),
                    "evaluation_noise_seed": int(
                        evaluation_noise_seed
                    ),
                    "run_fingerprint_sha256": str(
                        record.run_fingerprint_sha256
                    ),
                }
            )
    return pd.DataFrame(rows)


def _validation_admission_gates(
    *,
    diagnostics: pd.DataFrame,
    seed_summary: pd.DataFrame,
    duplicate_sensitivity: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate every preregistered pilot admission threshold fail-closed."""

    gates: dict[str, Any] = {}

    def record(
        name: str,
        *,
        passed: bool,
        threshold: Any,
        observed: Any,
        reason: str,
        missing_fields: Sequence[str] = (),
    ) -> None:
        gates[name] = {
            "passed": bool(passed) and not missing_fields,
            "threshold": threshold,
            "observed": observed,
            "reason": (
                f"missing required fields: {list(missing_fields)}"
                if missing_fields
                else reason
            ),
        }

    comparison = diagnostics[diagnostics["is_comparison_epoch"] == 1].copy()
    matched = comparison[comparison["variant"] == TEXT_RESIDUAL_VARIANT]
    shuffled = comparison[comparison["variant"] == SHUFFLED_RESIDUAL_VARIANT]
    matcher_fields = (
        "val_matching_real_pairwise_accuracy",
        "val_matching_real_accuracy_ci95_low",
        "val_matching_real_margin_mean",
    )
    missing = [field for field in matcher_fields if field not in matched]
    matched_values = []
    matched_pass = False
    if not missing and len(matched) == len(V3_PILOT_SEEDS):
        for row in matched.itertuples(index=False):
            values = {
                "seed": int(row.seed),
                "accuracy": float(getattr(row, matcher_fields[0])),
                "ci95_low": float(getattr(row, matcher_fields[1])),
                "margin": float(getattr(row, matcher_fields[2])),
            }
            matched_values.append(values)
        matched_pass = all(
            value["accuracy"] >= 0.60
            and value["ci95_low"] > 0.50
            and value["margin"] > 0.0
            for value in matched_values
        )
    record(
        "matched_heldout_matcher",
        passed=matched_pass,
        threshold={"accuracy_min": 0.60, "ci95_low_strict_min": 0.50, "margin_min": 0.0},
        observed=matched_values,
        reason="all matched seeds satisfy accuracy, confidence, and margin",
        missing_fields=missing,
    )

    accuracy_field = "val_matching_real_pairwise_accuracy"
    accuracy_difference_values = []
    accuracy_difference_pass = False
    missing = [
        field
        for field in (accuracy_field,)
        if field not in matched or field not in shuffled
    ]
    if not missing and len(matched) == len(shuffled) == len(V3_PILOT_SEEDS):
        merged_accuracy = matched[["fold", "seed", accuracy_field]].merge(
            shuffled[["fold", "seed", accuracy_field]],
            on=["fold", "seed"],
            suffixes=("_matched", "_shuffled"),
            validate="one_to_one",
        )
        differences = (
            pd.to_numeric(
                merged_accuracy[f"{accuracy_field}_matched"], errors="coerce"
            )
            - pd.to_numeric(
                merged_accuracy[f"{accuracy_field}_shuffled"], errors="coerce"
            )
        )
        accuracy_difference_values = [
            {"seed": int(seed), "matched_minus_shuffled": float(value)}
            for seed, value in zip(merged_accuracy["seed"], differences)
        ]
        accuracy_difference_pass = bool(
            float(differences.mean()) >= 0.05
            and int(np.sum(differences.to_numpy() > 0.0)) >= 2
        )
    record(
        "matched_vs_shuffled_matcher_accuracy",
        passed=accuracy_difference_pass,
        threshold={"mean_difference_min": 0.05, "positive_seed_count_min": 2},
        observed=accuracy_difference_values,
        reason="matched matcher accuracy exceeds shuffled in magnitude and seed direction",
        missing_fields=missing,
    )

    shuffled_fields = (
        "val_matching_real_pairwise_accuracy",
        "val_matching_real_accuracy_ci95_low",
        "val_matching_real_accuracy_ci95_high",
    )
    missing = [field for field in shuffled_fields if field not in shuffled]
    shuffled_values = []
    shuffled_pass = False
    if not missing and len(shuffled) == len(V3_PILOT_SEEDS):
        for row in shuffled.itertuples(index=False):
            value = {
                "seed": int(row.seed),
                "accuracy": float(getattr(row, shuffled_fields[0])),
                "ci95_low": float(getattr(row, shuffled_fields[1])),
                "ci95_high": float(getattr(row, shuffled_fields[2])),
            }
            shuffled_values.append(value)
        shuffled_pass = all(
            0.45 <= value["accuracy"] <= 0.55
            and value["ci95_low"] <= 0.50 <= value["ci95_high"]
            for value in shuffled_values
        )
    record(
        "shuffled_matcher_at_chance",
        passed=shuffled_pass,
        threshold={"accuracy_range": [0.45, 0.55], "ci95_must_include": 0.50},
        observed=shuffled_values,
        reason="all shuffled seeds remain statistically compatible with chance",
        missing_fields=missing,
    )

    diagnostic_arms = comparison[
        comparison["variant"].isin(
            [TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT]
        )
    ]
    gp_fields = (
        "gp_raw_norm_mean",
        "gp_unscaled_penalty",
        "gp_raw_norm_outside_0p5_1p5_rate",
    )
    missing = [field for field in gp_fields if field not in diagnostic_arms]
    gp_values = []
    gp_pass = False
    if not missing and len(diagnostic_arms) == 2 * len(V3_PILOT_SEEDS):
        for row in diagnostic_arms.itertuples(index=False):
            gp_values.append(
                {
                    "seed": int(row.seed),
                    "variant": str(row.variant),
                    "raw_norm_mean": float(getattr(row, gp_fields[0])),
                    "unscaled_penalty": float(getattr(row, gp_fields[1])),
                    "outside_rate": float(getattr(row, gp_fields[2])),
                }
            )
        gp_pass = all(
            0.75 <= value["raw_norm_mean"] <= 1.25
            and value["unscaled_penalty"] <= 0.10
            and value["outside_rate"] <= 0.10
            for value in gp_values
        )
    record(
        "support_aware_gradient_penalty",
        passed=gp_pass,
        threshold={
            "raw_norm_mean_range": [0.75, 1.25],
            "unscaled_penalty_max": 0.10,
            "outside_0p5_1p5_rate_max": 0.10,
        },
        observed=gp_values,
        reason="all matched/shuffled comparison checkpoints satisfy GP thresholds",
        missing_fields=missing,
    )

    unsupported_field = "gp_unsupported_max_abs_gradient"
    missing = [unsupported_field] if unsupported_field not in diagnostic_arms else []
    unsupported_values = []
    unsupported_pass = False
    if not missing and len(diagnostic_arms) == 2 * len(V3_PILOT_SEEDS):
        unsupported_values = [
            {
                "seed": int(row.seed),
                "variant": str(row.variant),
                "max_abs_gradient": float(getattr(row, unsupported_field)),
            }
            for row in diagnostic_arms.itertuples(index=False)
        ]
        unsupported_pass = all(
            value["max_abs_gradient"] <= 1.0e-7
            for value in unsupported_values
        )
    record(
        "unsupported_gradient_zero",
        passed=unsupported_pass,
        threshold={"max_abs_gradient_max": 1.0e-7},
        observed=unsupported_values,
        reason="unsupported GP gradients remain numerically zero",
        missing_fields=missing,
    )

    probe_fields = (
        "diag_g_probe_active",
        "diag_g_matching_output_grad_ratio_median",
        "diag_g_matching_output_grad_ratio_p95",
    )
    missing = [field for field in probe_fields if field not in matched]
    probe_values = []
    probe_pass = False
    if not missing and len(matched) == len(V3_PILOT_SEEDS):
        probe_values = [
            {
                "seed": int(row.seed),
                "active": float(getattr(row, probe_fields[0])),
                "ratio_median": float(getattr(row, probe_fields[1])),
                "ratio_p95": float(getattr(row, probe_fields[2])),
            }
            for row in matched.itertuples(index=False)
        ]
        probe_pass = all(
            value["active"] == 1.0
            and 0.05 <= value["ratio_median"] <= 0.20
            and value["ratio_p95"] < 0.50
            for value in probe_values
        )
    record(
        "generator_matching_output_gradient_ratio",
        passed=probe_pass,
        threshold={"median_range": [0.05, 0.20], "p95_strict_max": 0.50},
        observed=probe_values,
        reason="matched generator matching gradients are material but bounded",
        missing_fields=missing,
    )

    last_five_arms = diagnostics[
        diagnostics["variant"].isin(
            [TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT]
        )
    ]
    roundtrip_fields = (
        "g_transition_unclipped_raw_log_max_abs_error",
        "g_transition_unclipped_normalized_max_abs_error",
    )
    missing = [field for field in roundtrip_fields if field not in last_five_arms]
    roundtrip_observed = {}
    roundtrip_pass = False
    if not missing and not last_five_arms.empty:
        raw_values = pd.to_numeric(
            last_five_arms[roundtrip_fields[0]], errors="coerce"
        ).to_numpy(dtype=np.float64)
        normalized_values = pd.to_numeric(
            last_five_arms[roundtrip_fields[1]], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite = np.isfinite(raw_values) & np.isfinite(normalized_values)
        nonfinite_count = int((~finite).sum())
        raw_max = float(np.max(raw_values)) if nonfinite_count == 0 else None
        normalized_max = (
            float(np.max(normalized_values)) if nonfinite_count == 0 else None
        )
        roundtrip_observed = {
            "last_five_raw_log_max": raw_max,
            "last_five_normalized_max": normalized_max,
            "nonfinite_count": nonfinite_count,
        }
        roundtrip_pass = bool(
            nonfinite_count == 0
            and raw_max is not None
            and normalized_max is not None
            and raw_max <= 1.0e-6
            and normalized_max <= 1.0e-3
        )
    record(
        "transition_roundtrip",
        passed=roundtrip_pass,
        threshold={"raw_log_error_max": 1.0e-6, "normalized_error_max": 1.0e-3},
        observed=roundtrip_observed,
        reason="unclipped transition delivery remains numerically consistent",
        missing_fields=missing,
    )

    clipped_field = "g_transition_clipped_fraction"
    missing = [clipped_field] if clipped_field not in last_five_arms else []
    clipped_values = []
    clipped_pass = False
    if not missing and not last_five_arms.empty:
        for keys, group in last_five_arms.groupby(
            ["fold", "seed", "variant"], sort=True
        ):
            values = pd.to_numeric(
                group[clipped_field], errors="coerce"
            ).to_numpy(dtype=np.float64)
            comparison_values = pd.to_numeric(
                group.loc[group["is_comparison_epoch"] == 1, clipped_field],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            nonfinite_count = int((~np.isfinite(values)).sum())
            valid_comparison = bool(
                comparison_values.size == 1
                and np.isfinite(comparison_values[0])
            )
            comparison_value = (
                float(comparison_values[0]) if valid_comparison else None
            )
            median_value = (
                float(np.median(values)) if nonfinite_count == 0 else None
            )
            clipped_values.append(
                {
                    "fold": str(keys[0]),
                    "seed": int(keys[1]),
                    "variant": str(keys[2]),
                    "comparison_epoch_fraction": comparison_value,
                    "last_five_median_fraction": median_value,
                    "nonfinite_count": nonfinite_count,
                }
            )
        clipped_pass = all(
            value["nonfinite_count"] == 0
            and value["comparison_epoch_fraction"] is not None
            and value["last_five_median_fraction"] is not None
            and value["comparison_epoch_fraction"] <= 0.001
            and value["last_five_median_fraction"] <= 0.001
            for value in clipped_values
        )
    record(
        "transition_clipping",
        passed=clipped_pass,
        threshold={"comparison_and_last_five_median_max": 0.001},
        observed=clipped_values,
        reason="comparison checkpoint and final-five clipping stay negligible",
        missing_fields=missing,
    )

    mae_values = []
    mae_pass = True
    for contrast in ("matched_vs_continuation", "matched_vs_shuffled"):
        rows = seed_summary[
            (seed_summary["contrast"] == contrast)
            & (seed_summary["metric"] == "surface_mae")
        ]
        values = rows["mean_baseline_minus_focal"].to_numpy(dtype=float)
        item_pass = bool(
            len(values) == len(V3_PILOT_SEEDS)
            and float(np.mean(values)) > 0.0
            and int(np.sum(values > 0.0)) >= 2
        )
        mae_values.append(
            {
                "comparison": contrast,
                "mean_baseline_minus_matched": (
                    float(np.mean(values)) if len(values) else None
                ),
                "positive_seed_count": int(np.sum(values > 0.0)),
                "passed": item_pass,
            }
        )
        mae_pass = mae_pass and item_pass
    matched_samples = samples[samples["variant"] == TEXT_RESIDUAL_VARIANT]
    persistence_by_seed = (
        matched_samples.assign(
            persistence_minus_matched=(
                pd.to_numeric(matched_samples["current_mae"], errors="coerce")
                - pd.to_numeric(matched_samples["surface_mae"], errors="coerce")
            )
        )
        .groupby(["fold", "seed"], as_index=False)["persistence_minus_matched"]
        .mean()
    ) if {"current_mae", "surface_mae"}.issubset(matched_samples.columns) else pd.DataFrame()
    persistence_values = (
        persistence_by_seed["persistence_minus_matched"].to_numpy(dtype=float)
        if not persistence_by_seed.empty
        else np.asarray([], dtype=float)
    )
    persistence_pass = bool(
        len(persistence_values) == len(V3_PILOT_SEEDS)
        and float(np.mean(persistence_values)) > 0.0
        and int(np.sum(persistence_values > 0.0)) >= 2
    )
    mae_values.append(
        {
            "comparison": "matched_vs_persistence",
            "mean_baseline_minus_matched": (
                float(np.mean(persistence_values))
                if len(persistence_values)
                else None
            ),
            "positive_seed_count": int(np.sum(persistence_values > 0.0)),
            "passed": persistence_pass,
        }
    )
    mae_pass = mae_pass and persistence_pass
    record(
        "matched_surface_mae",
        passed=mae_pass,
        threshold={"mean_improvement_strict_min": 0.0, "positive_seed_count_min": 2},
        observed=mae_values,
        reason="matched MAE improves on continuation, shuffled, and persistence",
        missing_fields=(
            []
            if {"current_mae", "surface_mae"}.issubset(samples.columns)
            else ["current_mae", "surface_mae"]
        ),
    )

    duplicate_policy = "exclude_any_exact_or_near_duplicate_seen_in_train"
    duplicate_rows = duplicate_sensitivity[
        (duplicate_sensitivity["policy"] == duplicate_policy)
        & (duplicate_sensitivity["metric"] == "surface_mae")
        & duplicate_sensitivity["contrast"].isin(
            ["matched_vs_continuation", "matched_vs_shuffled"]
        )
    ]
    duplicate_values = duplicate_rows[
        [
            "contrast",
            "unique_pairs",
            "seed_mean_baseline_minus_focal",
            "positive_seed_count",
        ]
    ].to_dict(orient="records") if len(duplicate_rows) else []
    duplicate_pass = bool(
        len(duplicate_rows) == 2
        and np.all(
            pd.to_numeric(
                duplicate_rows["seed_mean_baseline_minus_focal"], errors="coerce"
            ).to_numpy()
            > 0.0
        )
    )
    record(
        "duplicate_free_direction",
        passed=duplicate_pass,
        threshold={"duplicate_free_mean_baseline_minus_matched_strict_min": 0.0},
        observed=duplicate_values,
        reason="removing exact/near train duplicates does not reverse matched comparisons",
        missing_fields=(
            []
            if {
                "policy",
                "metric",
                "contrast",
                "seed_mean_baseline_minus_focal",
            }.issubset(duplicate_sensitivity.columns)
            else ["duplicate_sensitivity_fields"]
        ),
    )

    return {
        "passed": bool(gates) and all(item["passed"] for item in gates.values()),
        "fail_closed": True,
        "checkpoint_selection_use": False,
        "gates": gates,
    }


def summarize_validation_pilot(args: argparse.Namespace) -> Path:
    """Create descriptive validation-only pilot diagnostics, never test claims."""

    _assert_py312()
    root = _resolve_root(args.experiment_root)
    verify_existing(argparse.Namespace(experiment_root=str(root)))
    run_folds, experiment_seeds, run_variants = _experiment_scope(root)
    registry_path = root / "registry/generate_registry.csv"
    selected_path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not registry_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            "Run collect-checkpoints and generate before summarize-validation."
        )
    registry = pd.read_csv(registry_path)
    selected = pd.read_csv(selected_path)
    expected_keys = {
        (fold, int(seed), variant)
        for fold in run_folds
        for seed in experiment_seeds
        for variant in run_variants
    }
    registry_keys = {
        (str(row.fold), int(row.seed), str(row.variant))
        for row in registry.itertuples(index=False)
    }
    selected_keys = {
        (str(row.fold), int(row.seed), str(row.variant))
        for row in selected.itertuples(index=False)
    }
    if (
        registry_keys != expected_keys
        or selected_keys != expected_keys
        or len(registry) != len(expected_keys)
        or len(selected) != len(expected_keys)
    ):
        raise ValueError("Validation pilot registries do not exactly match scope.")
    if set(registry["split"].astype(str).str.lower()) != {"val"}:
        raise RuntimeError("summarize-validation accepts validation outputs only.")

    required_plan_columns = {
        "matching_negative_source_plan_sha256",
        "text_alignment_plan_sha256",
        "run_fingerprint_sha256",
        "comparison_epoch",
    }
    missing_plan_columns = sorted(required_plan_columns - set(selected.columns))
    if missing_plan_columns:
        raise ValueError(
            f"Selected checkpoint registry lacks v3 fields: {missing_plan_columns}"
        )
    for fold, group in selected.groupby("fold", sort=True):
        if (
            group["matching_negative_source_plan_sha256"].nunique() != 1
            or group["text_alignment_plan_sha256"].nunique() != 1
        ):
            raise ValueError(f"Plan file hashes diverged across arms for {fold}.")
    if selected["run_fingerprint_sha256"].nunique() != len(selected):
        raise ValueError("Every v3 run must have a unique run fingerprint.")
    selected_index = selected.set_index(["fold", "seed", "variant"])
    for fold in run_folds:
        for seed in experiment_seeds:
            anchor_epoch = int(
                selected_index.loc[
                    (fold, seed, CONTINUATION_VARIANT),
                    "comparison_epoch",
                ]
            )
            for variant in (TEXT_RESIDUAL_VARIANT, SHUFFLED_RESIDUAL_VARIANT):
                if int(
                    selected_index.loc[(fold, seed, variant), "comparison_epoch"]
                ) != anchor_epoch:
                    raise ValueError(
                        f"Stage-B comparison epoch drift: {fold}/seed_{seed}/{variant}"
                    )

    sample_rows = []
    for record in registry.itertuples(index=False):
        summary = pd.read_csv(record.summary_path)
        summary["fold"] = str(record.fold)
        summary["seed"] = int(record.seed)
        summary["variant"] = str(record.variant)
        sample_rows.append(summary)
    samples = pd.concat(sample_rows, ignore_index=True)
    expected_sample_rows = (
        sum(_fold_counts(root, fold)[1] for fold in run_folds)
        * len(experiment_seeds)
        * len(run_variants)
    )
    if len(samples) != expected_sample_rows:
        raise ValueError(
            "Combined validation row count mismatch: "
            f"{len(samples)} != {expected_sample_rows}."
        )

    duplicate_columns = [
        "fold",
        "surface_pair_id",
        "exact_embedding_duplicate_with_train",
        "exact_text_duplicate_with_train",
        "near_text_candidate_duplicate_with_train",
    ]
    lineage_rows = []
    for fold in run_folds:
        lineage = pd.read_csv(_fold_dir(root, fold) / "pair_lineage_audit.csv")
        lineage_rows.append(
            lineage[lineage["split"].astype(str) == "val"][duplicate_columns]
        )
    samples = samples.merge(
        pd.concat(lineage_rows, ignore_index=True),
        on=["fold", "surface_pair_id"],
        how="left",
        validate="many_to_one",
    )
    if samples[duplicate_columns[2:]].isna().any().any():
        raise ValueError("Validation duplicate-lineage merge left unmatched pairs.")

    available_metrics = [metric for metric in POINT_METRICS if metric in samples]
    if "surface_mae" not in available_metrics:
        raise ValueError("Validation summaries must contain surface_mae.")
    model_summary = (
        samples.groupby(["fold", "seed", "variant"], as_index=False)
        .agg(
            n_pairs=("surface_pair_id", "size"),
            **{metric: (metric, "mean") for metric in available_metrics},
        )
    )

    difference_frames = []
    pair_keys = ["fold", "seed", "surface_pair_id"]
    for focal, baseline, contrast in VALIDATION_PILOT_CONTRASTS:
        left = samples[samples["variant"] == focal]
        right = samples[samples["variant"] == baseline]
        merged = left.merge(
            right,
            on=pair_keys,
            suffixes=("_focal", "_baseline"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Validation pair matching failed for {contrast}.")
        for metric in available_metrics:
            difference_frames.append(
                pd.DataFrame(
                    {
                        "contrast": contrast,
                        "focal_variant": focal,
                        "baseline_variant": baseline,
                        "fold": merged["fold"],
                        "seed": merged["seed"].astype(int),
                        "surface_pair_id": merged["surface_pair_id"],
                        "metric": metric,
                        "focal_error": pd.to_numeric(
                            merged[f"{metric}_focal"], errors="coerce"
                        ),
                        "baseline_error": pd.to_numeric(
                            merged[f"{metric}_baseline"], errors="coerce"
                        ),
                        "difference": (
                            pd.to_numeric(
                                merged[f"{metric}_baseline"], errors="coerce"
                            )
                            - pd.to_numeric(
                                merged[f"{metric}_focal"], errors="coerce"
                            )
                        ),
                        **{
                            column: merged[f"{column}_focal"].astype(int)
                            for column in duplicate_columns[2:]
                        },
                    }
                )
            )
    differences = pd.concat(difference_frames, ignore_index=True)
    seed_summary = (
        differences.groupby(
            [
                "contrast",
                "focal_variant",
                "baseline_variant",
                "fold",
                "seed",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mean_baseline_minus_focal=("difference", "mean"),
            pair_count=("difference", "count"),
        )
    )
    contrast_summary = (
        seed_summary.groupby(
            ["contrast", "focal_variant", "baseline_variant", "metric"],
            as_index=False,
        )
        .agg(
            seed_mean_baseline_minus_focal=(
                "mean_baseline_minus_focal",
                "mean",
            ),
            seed_std_baseline_minus_focal=(
                "mean_baseline_minus_focal",
                "std",
            ),
            positive_seed_count=(
                "mean_baseline_minus_focal",
                lambda values: int(np.sum(np.asarray(values) > 0.0)),
            ),
            seed_count=("seed", "nunique"),
        )
    )
    contrast_summary["difference_direction"] = "baseline_minus_focal"
    contrast_summary["positive_means_focal_better"] = True
    contrast_summary["inference_status"] = "descriptive_pilot_only"

    sensitivity_rows = []
    for (contrast, metric), group in differences.groupby(
        ["contrast", "metric"], sort=True
    ):
        finite = group[np.isfinite(group["difference"].to_numpy(dtype=float))]
        for policy, mask in _validation_duplicate_masks(finite).items():
            retained = finite.loc[mask]
            by_seed = retained.groupby(["fold", "seed"], as_index=False).agg(
                mean_difference=("difference", "mean"),
                pair_count=("difference", "count"),
            )
            sensitivity_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "policy": policy,
                    "pair_seed_rows": int(len(retained)),
                    "unique_pairs": int(
                        retained[["fold", "surface_pair_id"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "seed_count": int(by_seed["seed"].nunique()),
                    "seed_mean_baseline_minus_focal": (
                        float(by_seed["mean_difference"].mean())
                        if not by_seed.empty
                        else float("nan")
                    ),
                    "positive_seed_count": int(
                        np.sum(by_seed["mean_difference"].to_numpy() > 0.0)
                    ),
                    "inference_status": "descriptive_pilot_only",
                }
            )

    diagnostic_rows = []
    for record in selected.itertuples(index=False):
        metrics = pd.read_csv(Path(record.run_dir) / "metrics/training_metrics.csv")
        numeric_epoch = pd.to_numeric(metrics["epoch"], errors="coerce")
        comparison_rows = metrics[numeric_epoch == int(record.comparison_epoch)]
        if len(comparison_rows) != 1:
            raise ValueError(
                f"Missing comparison-epoch diagnostics: {record.run_dir}"
            )
        epoch_rows = metrics[numeric_epoch <= int(record.comparison_epoch)].tail(5)
        if len(epoch_rows) != min(5, int(record.comparison_epoch)):
            raise ValueError(f"Incomplete last-five diagnostics: {record.run_dir}")
        diagnostic_columns = sorted(
            column
            for column in metrics.columns
            if column.startswith(
                (
                    "gp_",
                    "g_transition_",
                    "diag_g_",
                    "g_matching_gradient_",
                    "g_nonmatching_gradient_",
                    "val_matching_",
                )
            )
        )
        for window_index, (_index, row) in enumerate(epoch_rows.iterrows(), start=1):
            diagnostic_rows.append(
                {
                    "fold": str(record.fold),
                    "seed": int(record.seed),
                    "variant": str(record.variant),
                    "epoch": int(row["epoch"]),
                    "comparison_epoch": int(record.comparison_epoch),
                    "last_five_window_index": window_index,
                    "is_comparison_epoch": int(
                        int(row["epoch"]) == int(record.comparison_epoch)
                    ),
                    **{column: row[column] for column in diagnostic_columns},
                }
            )

    comparisons_dir = root / "comparisons"
    final_dir = root / "final_tables"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(
        comparisons_dir / "validation_pilot_sample_metrics.csv", index=False
    )
    differences.to_csv(
        comparisons_dir / "validation_pilot_pairwise_differences.csv", index=False
    )
    model_summary.to_csv(
        final_dir / "validation_pilot_metrics_by_seed.csv", index=False
    )
    seed_summary.to_csv(
        final_dir / "validation_pilot_contrasts_by_seed.csv", index=False
    )
    contrast_summary.to_csv(
        final_dir / "validation_pilot_contrasts.csv", index=False
    )
    duplicate_sensitivity = pd.DataFrame(sensitivity_rows)
    duplicate_sensitivity.to_csv(
        final_dir / "validation_pilot_duplicate_sensitivity.csv", index=False
    )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(
        final_dir / "validation_pilot_checkpoint_diagnostics.csv", index=False
    )
    text_swap = _collect_text_swap_sensitivity(
        root=root,
        selected=selected,
        registry=registry,
    )
    text_swap.to_csv(
        final_dir / "validation_pilot_text_swap_sensitivity.csv", index=False
    )
    text_swap_summary = (
        text_swap.groupby(["variant", "swap_direction", "seed"], as_index=False)
        .agg(
            pair_count=("surface_pair_id", "size"),
            supported_surface_mae_native_vs_swapped=(
                "supported_surface_mae_native_vs_swapped",
                "mean",
            ),
            supported_log_delta_rms_native_vs_swapped=(
                "supported_log_delta_rms_native_vs_swapped",
                "mean",
            ),
        )
    )
    text_swap_summary.to_csv(
        final_dir / "validation_pilot_text_swap_sensitivity_by_seed.csv",
        index=False,
    )

    primary = contrast_summary[
        (contrast_summary["metric"] == "surface_mae")
        & contrast_summary["contrast"].isin(
            ["matched_vs_continuation", "matched_vs_shuffled"]
        )
    ]
    admission = _validation_admission_gates(
        diagnostics=diagnostics,
        seed_summary=seed_summary,
        duplicate_sensitivity=duplicate_sensitivity,
        samples=samples,
    )
    summary_payload = {
        "status": "ok",
        "claim_scope": "validation_pilot_description_only",
        "formal_test_status": "forbidden",
        "outer_test_consumed": False,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION_V3,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
        "epoch_policy": EPOCH_POLICY_CONTINUATION_ANCHOR,
        "folds": list(run_folds),
        "seeds": list(experiment_seeds),
        "variants": list(run_variants),
        "training_runs": len(expected_keys),
        "validation_sample_metric_rows": int(len(samples)),
        "surface_mae_directional_summary": primary.to_dict(orient="records"),
        "admission": admission,
        "duplicate_sensitivity": (
            "final_tables/validation_pilot_duplicate_sensitivity.csv"
        ),
        "checkpoint_diagnostics": (
            "final_tables/validation_pilot_checkpoint_diagnostics.csv"
        ),
        "text_swap_sensitivity": (
            "final_tables/validation_pilot_text_swap_sensitivity.csv"
        ),
        "text_swap_sensitivity_by_seed": text_swap_summary.to_dict(
            orient="records"
        ),
    }
    _write_json(final_dir / "validation_pilot_summary.json", summary_payload)
    print(final_dir / "validation_pilot_summary.json")
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
    registry_splits = set(
        registry.get("split", pd.Series(["test"] * len(registry)))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if registry_splits != {"test"}:
        raise RuntimeError(
            "Validation-only pilot outputs cannot enter the test comparison "
            "or results pipeline. Inspect validation metrics in the run "
            "directories and freeze a separate test protocol first."
        )
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
    frozen_config_path = root / "inputs/configs/train_rq1_pair_textbase.yaml"
    if frozen_config_path.is_file():
        frozen_config = _read_yaml(frozen_config_path)
        frozen_generate = dict(frozen_config.get("generate_result") or {})
        if str(frozen_generate.get("split", "test")).strip().lower() != "test":
            raise RuntimeError(
                "results-pipeline is disabled for validation-only pilot protocols; "
                "it would otherwise create misleading test-labelled tables."
            )
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
    prepare.add_argument(
        "--seeds", nargs="+", type=int, default=list(V3_PILOT_SEEDS)
    )
    prepare.add_argument(
        "--run-folds",
        nargs="+",
        choices=list(FOLDS),
        default=list(V3_PILOT_FOLDS),
    )
    prepare.add_argument(
        "--run-variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(V3_PILOT_VARIANTS),
    )
    prepare.add_argument("--matrix-profile", default="validation_pilot")

    verify = subparsers.add_parser("verify-existing")
    verify.add_argument("--experiment-root", required=True)

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

    summarize_validation = subparsers.add_parser("summarize-validation")
    summarize_validation.add_argument("--experiment-root", required=True)
    summarize_validation_pilot = subparsers.add_parser(
        "summarize-validation-pilot"
    )
    summarize_validation_pilot.add_argument("--experiment-root", required=True)

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
    if args.command == "verify-existing":
        return verify_existing(args)
    if args.command == "train-matrix":
        return train_matrix(args)
    if args.command == "monitor":
        return monitor(args)
    if args.command == "collect-checkpoints":
        return collect_checkpoints(args)
    if args.command == "generate":
        return generate_matrix(args)
    if args.command in {"summarize-validation", "summarize-validation-pilot"}:
        return summarize_validation_pilot(args)
    if args.command == "compare":
        return build_comparison(args)
    if args.command == "results-pipeline":
        return run_results_pipeline(args)
    if args.command == "package":
        return package_experiment(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
