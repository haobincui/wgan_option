"""Analysis for the seed-42, FiLM-LR=2.5e-5 text ablation.

Only four arms are trained by the companion orchestrator.  The matched-LP arm
is an immutable, hash-bound reference from the completed high-FiLM-LR sweep.
This module first verifies that external lineage, combines the four new arms
with exactly the four frozen matched-LP fold cells, and then delegates the
five-arm statistics to the already-audited direct-analysis implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42_analysis as base


NEW_ARMS = ("lp_shuffle", "no_text", "bow", "sentiment")
REFERENCE_SOURCE_ARM = "film_lr_2p5e5"
REFERENCE_REPORT_ARM = "lp_matched"
EXPECTED_NEW_ROWS = 2_000
EXPECTED_REFERENCE_ROWS = 500
EXPECTED_COMBINED_ROWS = 2_500
ANALYSIS_KIND = "direct_text_ablation_lr2p5e5_seed42_analysis_manifest_v1"


class TextAblationAnalysisError(ValueError):
    """Raised when frozen text-ablation evidence has drifted."""


def sha256_file(path: str | Path) -> str:
    return base.sha256_file(path)


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
            raise TextAblationAnalysisError(f"Existing analysis output drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return path


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _reference(config: Mapping[str, Any]) -> dict[str, Any]:
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise TextAblationAnalysisError("config.analysis must be a mapping")
    reference = analysis.get("frozen_matched_lp")
    if not isinstance(reference, Mapping):
        raise TextAblationAnalysisError(
            "config.analysis.frozen_matched_lp must be a mapping"
        )
    return dict(reference)


def verify_frozen_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the external matched-LP evidence and checkpoint allowlist."""

    reference = _reference(config)
    if (
        reference.get("arm") != REFERENCE_SOURCE_ARM
        or reference.get("report_arm") != REFERENCE_REPORT_ARM
        or int(reference.get("seed", -1)) != 42
        or bool(reference.get("retrain", True))
        or float(reference.get("film_learning_rate", -1.0)) != 2.5e-5
    ):
        raise TextAblationAnalysisError("Frozen matched-LP identity drift")
    root = _resolve(reference["experiment_root"])
    declarations = {
        "output_hash_manifest": (
            _resolve(reference["output_hash_manifest_path"]),
            str(reference["output_hash_manifest_sha256"]),
        ),
        "pair_metrics": (
            _resolve(reference["pair_metrics_path"]),
            str(reference["pair_metrics_sha256"]),
        ),
        "training_summary": (
            _resolve(reference["training_summary_path"]),
            str(reference["training_summary_sha256"]),
        ),
        "checkpoint_allowlist": (
            _resolve(reference["checkpoint_allowlist_path"]),
            str(reference["checkpoint_allowlist_sha256"]),
        ),
    }
    for role, (path, expected_sha) in declarations.items():
        if root not in path.parents or sha256_file(path) != expected_sha:
            raise TextAblationAnalysisError(f"Frozen matched-LP {role} drift")
    outputs = pd.read_csv(
        declarations["output_hash_manifest"][0], dtype=str, keep_default_na=False
    )
    for role in ("pair_metrics", "training_summary", "checkpoint_allowlist"):
        path, expected_sha = declarations[role]
        selected = outputs.loc[
            outputs["relative_path"].eq(path.relative_to(root).as_posix())
        ]
        if (
            len(selected) != 1
            or selected.iloc[0]["sha256"] != expected_sha
            or int(selected.iloc[0]["size_bytes"]) != path.stat().st_size
        ):
            raise TextAblationAnalysisError(
                f"Frozen output manifest does not bind {role}"
            )
    allowlist = pd.read_csv(
        declarations["checkpoint_allowlist"][0], dtype=str, keep_default_na=False
    )
    generators = allowlist.loc[
        allowlist["arm"].eq(REFERENCE_SOURCE_ARM)
        & allowlist["checkpoint_role"].eq("generator_best_learned")
    ].copy()
    expected_by_fold = {
        str(key): str(value)
        for key, value in dict(
            reference.get("generator_checkpoint_sha256_by_fold") or {}
        ).items()
    }
    observed_by_fold = dict(
        zip(generators["fold"], generators["checkpoint_sha256"], strict=True)
    )
    if expected_by_fold != observed_by_fold or set(expected_by_fold) != set(
        base.DIRECT_FOLDS
    ):
        raise TextAblationAnalysisError("Frozen matched-LP checkpoint hashes drift")
    for row in generators.itertuples(index=False):
        checkpoint = Path(row.checkpoint_path).resolve()
        if (
            root not in checkpoint.parents
            or checkpoint.stat().st_size != int(row.size_bytes)
            or sha256_file(checkpoint) != row.checkpoint_sha256
        ):
            raise TextAblationAnalysisError(
                f"Frozen matched-LP checkpoint file drift: {checkpoint}"
            )
    return {
        "root": root,
        "reference": reference,
        "declarations": declarations,
        "generator_checkpoint_sha256_by_fold": expected_by_fold,
    }


def _combined_pair_metrics(root: Path, frozen: Mapping[str, Any]) -> Path:
    new_path = root / "analysis/rq12_pair_metrics.csv.gz"
    new = base.validate_pair_metrics(
        new_path,
        expected_arms=NEW_ARMS,
        expected_row_count=EXPECTED_NEW_ROWS,
    )
    source_path = frozen["declarations"]["pair_metrics"][0]
    source = pd.read_csv(source_path, low_memory=False)
    reference = source.loc[
        source["arm"].astype(str).eq(REFERENCE_SOURCE_ARM)
        & pd.to_numeric(source["seed"], errors="raise").astype(int).eq(42)
        & pd.to_numeric(source["tolerance_minutes"], errors="raise").astype(int).eq(5)
    ].copy()
    if len(reference) != EXPECTED_REFERENCE_ROWS:
        raise TextAblationAnalysisError(
            "Frozen matched-LP reference must contain exactly 500 pair rows"
        )
    reference["arm"] = REFERENCE_REPORT_ARM
    combined = pd.concat([reference, new], ignore_index=True)
    noise_column = "noise_bank_profile_sha256"
    if noise_column not in combined:
        raise TextAblationAnalysisError(
            "Combined evidence lacks the MC64 noise-bank lineage"
        )
    for fold in base.DIRECT_FOLDS:
        fold_rows = combined.loc[combined["fold"].astype(str).eq(fold)]
        by_arm = fold_rows.groupby("arm", sort=False)[noise_column].nunique(
            dropna=False
        )
        if (
            set(by_arm.index.astype(str)) != {REFERENCE_REPORT_ARM, *NEW_ARMS}
            or not by_arm.eq(1).all()
            or fold_rows[noise_column].astype(str).nunique(dropna=False) != 1
        ):
            raise TextAblationAnalysisError(
                f"Matched-LP and new arms do not share one MC64 noise bank: {fold}"
            )
    validated = base.validate_pair_metrics(combined)
    if len(validated) != EXPECTED_COMBINED_ROWS:
        raise TextAblationAnalysisError("Combined five-arm pair universe drift")
    destination = root / "analysis/combined_five_arm_pair_metrics.csv"
    return _atomic_write(
        destination,
        validated.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
            "utf-8"
        ),
    )


def _combined_training_summary(
    root: Path, frozen: Mapping[str, Any], combined_pairs: pd.DataFrame
) -> Path:
    new_path = root / "analysis/training_summary.csv"
    if not new_path.is_file():
        raise TextAblationAnalysisError("New 16-job training summary is missing")
    new = pd.read_csv(new_path, low_memory=False)
    expected_cells = {(fold, arm) for fold in base.DIRECT_FOLDS for arm in NEW_ARMS}
    if (
        len(new) != 16
        or new["job_id"].astype(str).duplicated().any()
        or set(new[["fold", "arm"]].itertuples(index=False, name=None))
        != expected_cells
    ):
        raise TextAblationAnalysisError("New training-summary 16-cell universe drift")
    source = pd.read_csv(
        frozen["declarations"]["training_summary"][0], low_memory=False
    )
    reference = source.loc[source["arm"].astype(str).eq(REFERENCE_SOURCE_ARM)].copy()
    if len(reference) != 4:
        raise TextAblationAnalysisError(
            "Frozen matched-LP training summary must contain four folds"
        )
    reference["arm"] = REFERENCE_REPORT_ARM
    combined = pd.concat([reference, new], ignore_index=True, sort=False)
    base.validate_training_summary(combined, combined_pairs, maximum_epochs=240)
    destination = root / "analysis/combined_five_arm_training_summary.csv"
    return _atomic_write(
        destination,
        combined.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
            "utf-8"
        ),
    )


def analyze_experiment(output_root: Path, config: Mapping[str, Any]) -> Path:
    """Build the hash-bound combined five-arm descriptive analysis bundle."""

    root = Path(output_root).resolve()
    matrix = config.get("matrix")
    analysis_config = config.get("analysis")
    if not isinstance(matrix, Mapping) or not isinstance(analysis_config, Mapping):
        raise TextAblationAnalysisError("config matrix/analysis mappings are required")
    if tuple(map(str, matrix.get("direct_arms", ()))) != NEW_ARMS:
        raise TextAblationAnalysisError("Text-ablation arm order drift")
    if int(matrix.get("expected_training_jobs", -1)) != 16:
        raise TextAblationAnalysisError("Text-ablation training-job count drift")
    if int(matrix.get("expected_pair_metric_rows", -1)) != EXPECTED_NEW_ROWS:
        raise TextAblationAnalysisError("Text-ablation pair-row count drift")
    if int(analysis_config.get("bootstrap_replicates", -1)) != 10_000:
        raise TextAblationAnalysisError("Analysis requires 10,000 bootstrap draws")
    frozen = verify_frozen_reference(config)
    combined_path = _combined_pair_metrics(root, frozen)
    combined_pairs = base.validate_pair_metrics(combined_path)
    combined_training_path = _combined_training_summary(root, frozen, combined_pairs)
    analysis = base.analyze_direct_5arm(
        combined_pairs,
        training_summary=combined_training_path,
        maximum_epochs=240,
        bootstrap_iterations=10_000,
        bootstrap_seed=int(analysis_config["bootstrap_seed"]),
    )
    analysis.summary.update(
        experiment="news_first_vol_film_unet_text_ablation_lr2p5e5_seed42",
        comparison_design=(
            "four_new_direct_arms_plus_hash_frozen_compute_matched_lp_reference"
        ),
        film_learning_rate=2.5e-5,
        newly_trained_job_count=16,
        frozen_reference_job_count=4,
        test_based_representation_selection_permitted=False,
    )
    bundle = base.write_direct_analysis_bundle(
        analysis,
        pair_metrics_path=combined_path,
        pair_metrics_sha256=sha256_file(combined_path),
        output_dir=root / "analysis",
        training_summary_path=combined_training_path,
        training_summary_sha256=sha256_file(combined_training_path),
    )
    reference_binding = {
        "schema_version": 1,
        "kind": "frozen_matched_lp_lr2p5e5_reference_binding_v1",
        "source_arm": REFERENCE_SOURCE_ARM,
        "report_arm": REFERENCE_REPORT_ARM,
        "seed": 42,
        "film_learning_rate": 2.5e-5,
        "retrained": False,
        "source_files": [
            {
                "role": role,
                "path": str(path),
                "sha256": expected_sha,
                "size_bytes": path.stat().st_size,
            }
            for role, (path, expected_sha) in frozen["declarations"].items()
        ],
        "generator_checkpoint_sha256_by_fold": frozen[
            "generator_checkpoint_sha256_by_fold"
        ],
    }
    reference_binding["payload_sha256"] = hashlib.sha256(
        _canonical_bytes(reference_binding)
    ).hexdigest()
    binding_path = _atomic_write(
        root / "analysis/frozen_matched_lp_reference_binding.json",
        _canonical_bytes(reference_binding),
    )
    outer = {
        "schema_version": 1,
        "kind": ANALYSIS_KIND,
        "interpretation": "retrospective_rolling_development_single_seed_descriptive",
        "confirmatory": False,
        "new_training_jobs": 16,
        "frozen_reference_jobs": 4,
        "combined_pair_metric_rows": 2_500,
        "inputs": [
            {
                "role": "new_pair_metrics",
                "path": str((root / "analysis/rq12_pair_metrics.csv.gz").resolve()),
                "sha256": sha256_file(root / "analysis/rq12_pair_metrics.csv.gz"),
            },
            {
                "role": "combined_pair_metrics",
                "path": str(combined_path.resolve()),
                "sha256": sha256_file(combined_path),
            },
            {
                "role": "combined_training_summary",
                "path": str(combined_training_path.resolve()),
                "sha256": sha256_file(combined_training_path),
            },
            {
                "role": "frozen_matched_lp_binding",
                "path": str(binding_path.resolve()),
                "sha256": sha256_file(binding_path),
            },
        ],
        "inner_analysis_manifest": {
            "path": str(bundle["manifest"].resolve()),
            "sha256": sha256_file(bundle["manifest"]),
        },
    }
    outer["payload_sha256"] = hashlib.sha256(_canonical_bytes(outer)).hexdigest()
    return _atomic_write(
        root / "analysis/text_ablation_analysis_manifest.json",
        _canonical_bytes(outer),
    )


__all__ = [
    "ANALYSIS_KIND",
    "NEW_ARMS",
    "TextAblationAnalysisError",
    "analyze_experiment",
    "verify_frozen_reference",
]
