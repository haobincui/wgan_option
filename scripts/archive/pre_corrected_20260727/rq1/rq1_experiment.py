#!/usr/bin/env python3
"""Reproducible RQ1 text-increment experiment orchestration and statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
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

from film_wgan.baselines import PCARidgeForecaster  # noqa: E402
from film_wgan.config import FilmWGANTrainConfig, load_train_config  # noqa: E402
from film_wgan.data import build_text_permutation_mapping, create_train_val_bundle, write_split_manifest  # noqa: E402
from film_wgan.inference import summarize_surface_scenarios  # noqa: E402
from film_wgan.losses import build_atm_short_mask, build_reconstruction_weight_template  # noqa: E402
from film_wgan.plotting import extract_atm_short_value  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/film_wgan/train_rq1_textbase.yaml"
DEFAULT_WORKBOOK = ROOT / "data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx"
DEFAULT_ORIGINAL_TEXT_CONFIG = (
    ROOT
    / "outputs/archive/outputs_pre_20260722-135827/training/film_wgan/svi-excel/20260417_131244/metrics"
    / "training_resolved_config.yaml"
)
PLAN_DOCUMENT = ROOT / "docs/summary/20260722-141716/rq1_code_modification_plan.md"
SEEDS = (42, 101, 202, 303, 404)
NEURAL_VARIANTS = (
    "film_wgan_text",
    "film_wgan_no_text",
    "film_wgan_shuffled_text",
    "concat_wgan_text",
    "film_cnn_text",
)
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
    ("film_wgan_text", "film_wgan_no_text"),
    ("film_wgan_text", "film_wgan_shuffled_text"),
    ("film_wgan_text", "concat_wgan_text"),
    ("film_wgan_text", "film_cnn_text"),
    ("film_wgan_text", "persistence"),
    ("film_wgan_text", "pca_ridge_no_text"),
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _latest_experiment() -> Path:
    candidates = sorted((ROOT / "outputs/experiments").glob("rq1_incremental_text_*"))
    if not candidates:
        raise FileNotFoundError("No prepared RQ1 experiment found. Run prepare first.")
    return candidates[-1]


def _resolve_experiment_root(value: str | None, *, create_default: bool = False) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    if create_default:
        return ROOT / "outputs/experiments" / f"rq1_incremental_text_{_utc_timestamp()}"
    return _latest_experiment()


def _experiment_config_path(root: Path) -> Path:
    return root / "inputs/configs/train_rq1_textbase.yaml"


def _run(command: Sequence[str], *, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(list(command), cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(list(command), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _assert_environment(expected_name: str) -> None:
    if Path(sys.prefix).name != str(expected_name):
        raise RuntimeError(
            f"RQ1 orchestration must run inside conda environment {expected_name!r}; current prefix is {sys.prefix}."
        )


def _variant_overrides(variant: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "text_embedding_mode": "lp",
        "normalize_text_embedding": True,
        "text_alignment_mode": "matched",
        "forecast_mode": "stochastic_wgan",
        "conditioning_mode": "film",
        "lambda_adv": 0.1,
    }
    if variant == "film_wgan_no_text":
        overrides.update(text_embedding_mode="zero_lp", normalize_text_embedding=False)
    elif variant == "film_wgan_shuffled_text":
        overrides["text_alignment_mode"] = "permuted"
    elif variant == "concat_wgan_text":
        overrides["conditioning_mode"] = "concat"
    elif variant == "film_cnn_text":
        overrides.update(forecast_mode="deterministic", lambda_adv=0.0)
    elif variant != "film_wgan_text":
        raise ValueError(f"Unsupported RQ1 variant: {variant}")
    return overrides


def _validate_textbase_provenance(original: dict[str, Any], frozen: dict[str, Any]) -> None:
    allowed_changes = {
        "data_path",
        "output_root",
        "checkpoints_path",
        "metrics_path",
        "split_strategy",
        "split_manifest_path",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "text_alignment_mode",
        "text_permutation_seed",
        "forecast_mode",
        "conditioning_mode",
        "checkpoint_metric",
        "checkpoint_warmup_epochs",
        "extra_checkpoint_metrics",
        "eval_calibration_levels",
        "arbitrage_violation_tolerance",
        "min_samples_for_training",
    }
    def equivalent(left: Any, right: Any) -> bool:
        if left == right:
            return True
        if isinstance(left, bool) or isinstance(right, bool):
            return False
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
        except (TypeError, ValueError):
            return False

    unexpected = {
        key: {"original": original.get(key), "frozen": frozen.get(key)}
        for key in sorted(set(original) & set(frozen))
        if key not in allowed_changes and not equivalent(original.get(key), frozen.get(key))
    }
    if unexpected:
        raise ValueError(f"RQ1 frozen config diverges from the original text run: {unexpected}")


def prepare_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root, create_default=True)
    if root.exists() and any(root.iterdir()) and not args.reuse:
        raise FileExistsError(f"Experiment directory is not empty: {root}")
    for relative in (
        "inputs/configs",
        "inputs/data",
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
    original_text_config = Path(args.original_text_resolved_config).resolve()
    if not source_config.exists() or not source_workbook.exists() or not original_text_config.exists():
        raise FileNotFoundError(
            "Missing RQ1 input: "
            f"config={source_config} workbook={source_workbook} original_text_config={original_text_config}"
        )
    copied_workbook = root / "inputs/data/merged_vol_rq2_text.xlsx"
    shutil.copy2(source_workbook, copied_workbook)
    shutil.copy2(source_config, root / "inputs/configs/source_train_rq1_textbase.yaml")
    copied_original_config = root / "inputs/configs/original_text_training_resolved_config.yaml"
    shutil.copy2(original_text_config, copied_original_config)
    if PLAN_DOCUMENT.exists():
        shutil.copy2(PLAN_DOCUMENT, root / "inputs/docs/rq1_code_modification_plan.md")

    payload = _read_yaml(source_config)
    training = dict(payload.get("training") or payload)
    generate = dict(payload.get("generate_result") or {})
    original_payload = _read_yaml(original_text_config)
    original_training = dict(original_payload.get("training") or original_payload)
    _validate_textbase_provenance(original_training, training)
    split_manifest = root / "inputs/split_manifest.csv"
    training.update(
        {
            "data_path": str(copied_workbook),
            "split_manifest_path": "",
            "output_root": "",
            "checkpoints_path": "",
            "metrics_path": "",
        }
    )
    frozen_path = _experiment_config_path(root)
    _write_yaml(frozen_path, {"training": training, "generate_result": generate})
    frozen_config = load_train_config(frozen_path)
    write_split_manifest(frozen_config, split_manifest)
    training["split_manifest_path"] = str(split_manifest)
    _write_yaml(frozen_path, {"training": training, "generate_result": generate})

    manifest = pd.read_csv(split_manifest)
    permutation_mapping = build_text_permutation_mapping(manifest, seed=20260722)
    if bool(
        (
            permutation_mapping["target_surface_pair_id"].astype(str)
            == permutation_mapping["donor_surface_pair_id"].astype(str)
        ).any()
    ):
        raise ValueError("Shuffled-text permutation contains a same-surface-pair donor.")
    permutation_mapping.to_csv(root / "inputs/text_permutation_mapping.csv", index=False)
    counts = manifest.groupby("split").agg(rows=("sample_id", "size"), pairs=("surface_pair_id", "nunique"))
    expected = {"train": (2639, 2039), "val": (533, 437), "test": (539, 437)}
    actual = {split: (int(row.rows), int(row.pairs)) for split, row in counts.iterrows()}
    if actual != expected:
        raise ValueError(f"Unexpected grouped split counts: expected {expected}, found {actual}")

    git_state = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    (root / "inputs/git_state.txt").write_text(f"commit={git_head}\n{git_state}", encoding="utf-8")
    preregistration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Incremental predictive value of LP text in FiLM-WGAN IVS forecasts",
        "primary_difference": "baseline_error_minus_text_error",
        "primary_metric": "surface_mae",
        "secondary_metrics": list(POINT_METRICS[1:]),
        "seeds": list(SEEDS),
        "checkpoint_rule": "lowest validation surface_mae at epoch > 10",
        "split_counts": actual,
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 20260722,
        "best_seed_selection": False,
        "test_accessed": False,
    }
    _write_yaml(root / "inputs/preregistration.yaml", preregistration)
    pd.DataFrame(
        [
            {
                "relative_path": str(copied_workbook.relative_to(root)),
                "source_path": str(source_workbook),
                "sha256": _sha256(copied_workbook),
                "size_bytes": copied_workbook.stat().st_size,
            },
            {
                "relative_path": str(frozen_path.relative_to(root)),
                "source_path": str(source_config),
                "sha256": _sha256(frozen_path),
                "size_bytes": frozen_path.stat().st_size,
            },
            {
                "relative_path": str(copied_original_config.relative_to(root)),
                "source_path": str(original_text_config),
                "sha256": _sha256(copied_original_config),
                "size_bytes": copied_original_config.stat().st_size,
            },
            {
                "relative_path": str(split_manifest.relative_to(root)),
                "source_path": "generated from frozen workbook",
                "sha256": _sha256(split_manifest),
                "size_bytes": split_manifest.stat().st_size,
            },
            {
                "relative_path": "inputs/text_permutation_mapping.csv",
                "source_path": "generated from split_manifest.csv",
                "sha256": _sha256(root / "inputs/text_permutation_mapping.csv"),
                "size_bytes": (root / "inputs/text_permutation_mapping.csv").stat().st_size,
            },
        ]
    ).to_csv(root / "inputs/data_manifest.csv", index=False)
    _write_json(
        root / "experiment.json",
        {"experiment_root": str(root), "config_path": str(frozen_path), "seeds": list(SEEDS)},
    )
    print(root)
    return root


def _read_registry(root: Path) -> pd.DataFrame:
    path = root / "registry/run_registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=["variant", "seed", "run_dir", "status"])
    return pd.read_csv(path)


def run_training_matrix(args: argparse.Namespace) -> Path:
    _assert_environment(args.env_name)
    root = _resolve_experiment_root(args.experiment_root)
    config_path = _experiment_config_path(root)
    if not config_path.exists():
        raise FileNotFoundError(f"Prepared config missing: {config_path}")
    registry_rows = _read_registry(root).to_dict("records")
    completed = {
        (str(row["variant"]), int(row["seed"]))
        for row in registry_rows
        if str(row.get("status")) == "complete" and Path(str(row.get("run_dir"))).exists()
    }
    for variant in NEURAL_VARIANTS:
        for seed in SEEDS:
            if args.variant and variant != args.variant:
                continue
            if args.seed is not None and seed != int(args.seed):
                continue
            if (variant, seed) in completed and not args.rerun:
                continue
            output_root = root / "training_runs" / variant / f"seed_{seed}"
            output_root.mkdir(parents=True, exist_ok=True)
            before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
            overrides = _variant_overrides(variant)
            overrides.update(seed=seed, output_root=str(output_root))
            command = [
                sys.executable,
                "scripts/film_wgan/main.py",
                "train",
                "--config",
                str(config_path),
                "--train-only",
            ]
            for key, value in overrides.items():
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                command.extend(["--set", f"{key}={rendered}"])
            log_path = root / "logs" / variant / f"seed_{seed}.log"
            started = datetime.now(timezone.utc).isoformat()
            try:
                _run(command, log_path=log_path)
                after = [path.resolve() for path in output_root.iterdir() if path.is_dir() and path.resolve() not in before]
                if len(after) != 1:
                    raise RuntimeError(f"Expected one new run directory under {output_root}, found {after}")
                run_dir = after[0]
                status = "complete"
                error = ""
            except Exception as exc:
                run_dir = Path("")
                status = "failed"
                error = str(exc)
            registry_rows = [
                row
                for row in registry_rows
                if not (str(row.get("variant")) == variant and int(row.get("seed", -1)) == seed)
            ]
            registry_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": status,
                    "started_at_utc": started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "log_path": str(log_path),
                    "error": error,
                }
            )
            pd.DataFrame(registry_rows).sort_values(["variant", "seed"]).to_csv(
                root / "registry/run_registry.csv", index=False
            )
            if status != "complete":
                raise RuntimeError(f"RQ1 training failed for {variant} seed={seed}: {error}")
    return root


def collect_checkpoints(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root)
    registry = _read_registry(root)
    expected = {(variant, seed) for variant in NEURAL_VARIANTS for seed in SEEDS}
    found = {(str(row.variant), int(row.seed)) for row in registry.itertuples() if str(row.status) == "complete"}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Cannot freeze checkpoints; missing completed runs: {missing}")
    selected_rows: list[dict[str, Any]] = []
    config_rows: list[tuple[str, int, dict[str, Any]]] = []
    for row in registry.itertuples(index=False):
        if (str(row.variant), int(row.seed)) not in expected:
            continue
        run_dir = Path(str(row.run_dir))
        best_path = run_dir / "metrics/best_checkpoint.json"
        checkpoint = run_dir / "checkpoints/film_wgan_best.pt"
        resolved = run_dir / "metrics/training_resolved_config.yaml"
        if not best_path.exists() or not checkpoint.exists() or not resolved.exists():
            raise FileNotFoundError(f"Incomplete training artifacts under {run_dir}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if int(best["best_epoch"]) <= 10 or str(best["checkpoint_metric"]) != "val_mae":
            raise ValueError(f"Invalid paper checkpoint selection in {best_path}: {best}")
        config_payload = _read_yaml(resolved)
        config_rows.append((str(row.variant), int(row.seed), config_payload))
        selected_rows.append(
            {
                "variant": str(row.variant),
                "seed": int(row.seed),
                "run_dir": str(run_dir),
                "checkpoint_path": str(checkpoint),
                "best_epoch": int(best["best_epoch"]),
                "best_val_surface_mae": float(best["best_metric"]),
                "resolved_config_path": str(resolved),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )

    allowed_variant_fields = {
        "text_embedding_mode",
        "normalize_text_embedding",
        "text_alignment_mode",
        "forecast_mode",
        "conditioning_mode",
        "lambda_adv",
        "seed",
        "output_root",
        "checkpoints_path",
        "metrics_path",
    }
    failures: list[dict[str, Any]] = []
    reference = config_rows[0][2]
    for variant, seed, payload in config_rows[1:]:
        keys = set(reference) | set(payload)
        for key in sorted(keys):
            if reference.get(key) != payload.get(key) and key not in allowed_variant_fields:
                failures.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "field": key,
                        "reference": json.dumps(reference.get(key), default=str),
                        "actual": json.dumps(payload.get(key), default=str),
                    }
                )
    failure_path = root / "checkpoint_selection/config_validation_failures.csv"
    pd.DataFrame(failures, columns=["variant", "seed", "field", "reference", "actual"]).to_csv(
        failure_path, index=False
    )
    if failures:
        raise ValueError(f"Resolved configs differ outside the allowed RQ1 fields; see {failure_path}")
    pd.DataFrame(selected_rows).sort_values(["variant", "seed"]).to_csv(
        root / "checkpoint_selection/selected_checkpoints.csv", index=False
    )
    return root


def run_generate_test_matrix(args: argparse.Namespace) -> Path:
    _assert_environment(args.env_name)
    root = _resolve_experiment_root(args.experiment_root)
    collect_checkpoints(argparse.Namespace(experiment_root=str(root)))
    selected = pd.read_csv(root / "checkpoint_selection/selected_checkpoints.csv")
    if len(selected) != len(NEURAL_VARIANTS) * len(SEEDS):
        raise ValueError("All 25 checkpoints must be frozen before test access.")
    access_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        run_dir = Path(str(row.run_dir))
        checkpoint_path = Path(str(row.checkpoint_path))
        if _sha256(checkpoint_path) != str(row.checkpoint_sha256):
            raise ValueError(f"Frozen checkpoint SHA256 mismatch before test access: {checkpoint_path}")
        output_dir = run_dir / "rq1_test_json"
        summary = output_dir / "summary.csv"
        run_metadata = output_dir / "run_metadata.json"
        if summary.exists() and run_metadata.exists() and not args.rerun:
            count = len(pd.read_csv(summary))
            metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
            if count == 539 and int(metadata.get("selected_samples", -1)) == 539:
                access_rows.append(
                    {
                        "accessed_at_utc": datetime.fromtimestamp(
                            summary.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                        "variant": str(row.variant),
                        "seed": int(row.seed),
                        "checkpoint_sha256": str(row.checkpoint_sha256),
                        "summary_path": str(summary),
                    }
                )
                continue
        command = [
            sys.executable,
            "scripts/film_wgan/main.py",
            "generate-result",
            "--config",
            str(row.resolved_config_path),
            "--checkpoint",
            str(row.checkpoint_path),
            "--output-dir",
            "rq1_test_json",
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
        _run(command, log_path=root / "logs" / str(row.variant) / f"seed_{int(row.seed)}_generate.log")
        if not summary.exists() or not run_metadata.exists() or len(pd.read_csv(summary)) != 539:
            raise ValueError(f"Unexpected RQ1 test result count in {summary}")
        access_rows.append(
            {
                "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
                "variant": str(row.variant),
                "seed": int(row.seed),
                "checkpoint_sha256": str(row.checkpoint_sha256),
                "summary_path": str(summary),
            }
        )
    if access_rows:
        pd.DataFrame(access_rows).to_csv(root / "registry/generated_result_registry.csv", index=False)
        access_path = root / "registry/test_access_log.csv"
        old = pd.read_csv(access_path).to_dict("records") if access_path.exists() else []
        combined = pd.DataFrame(old + access_rows)
        combined = combined.drop_duplicates(
            subset=["variant", "seed", "checkpoint_sha256"],
            keep="first",
        ).sort_values(["variant", "seed"])
        combined.to_csv(access_path, index=False)
    prereg_path = root / "inputs/preregistration.yaml"
    prereg = _read_yaml(prereg_path)
    prereg["test_accessed"] = True
    prereg["first_test_access_utc"] = prereg.get("first_test_access_utc") or datetime.now(timezone.utc).isoformat()
    _write_yaml(prereg_path, prereg)
    return root


def _baseline_summary_row(sample: Any, predicted: np.ndarray, *, model: str, config: FilmWGANTrainConfig) -> dict[str, Any]:
    strike = sample.strike_grid
    maturity = sample.maturity_days_grid
    recon_weights = build_reconstruction_weight_template(
        strike_grid=torch.tensor(strike),
        maturity_days_grid=torch.tensor(maturity),
        mode=config.recon_weight_mode,
        atm_range=config.recon_atm_range,
        short_end_max_days=config.recon_atm_short_end_max_days,
        atm_multiplier=config.recon_atm_multiplier,
    )
    atm_mask = build_atm_short_mask(
        strike_grid=torch.tensor(strike),
        maturity_days_grid=torch.tensor(maturity),
        atm_range=config.atm_short_range,
        max_days=config.atm_short_max_days,
    )
    summary = summarize_surface_scenarios(
        surface_stack=np.asarray(predicted, dtype=np.float64)[None, ...],
        current_surface=sample.current_surface,
        target_surface=sample.target_surface,
        strike_grid=strike,
        maturity_days_grid=maturity,
        reweight_beta_mode="fixed",
        reweight_beta=0.0,
        aggregation_mode="weighted_mean",
        calibration_levels=config.eval_calibration_levels,
        arbitrage_violation_tolerance=config.arbitrage_violation_tolerance,
        recon_weights_surface=recon_weights,
        atm_short_mask_surface=atm_mask,
    )
    predicted_atm = extract_atm_short_value(predicted, strike_grid=strike, maturity_days_grid=maturity)
    target_atm = extract_atm_short_value(sample.target_surface, strike_grid=strike, maturity_days_grid=maturity)
    current_atm = extract_atm_short_value(sample.current_surface, strike_grid=strike, maturity_days_grid=maturity)
    return {
        "model": model,
        "seed": np.nan,
        "sample_id": sample.sample_id,
        "global_index": sample.global_index,
        "surface_pair_id": sample.surface_pair_id,
        "split": "test",
        "news_timestamp_utc": sample.timestamp,
        "current_snapshot_time_utc": sample.current_snapshot_time_utc,
        "target_snapshot_time_utc": sample.target_snapshot_time_utc,
        "surface_mae": summary["metrics"]["mae"],
        "short_atm_mae": summary["short_atm_metrics"]["atm_short_pure_mae"],
        "atm7_abs_err": abs(float(predicted_atm["value"]) - float(target_atm["value"])),
        "current_surface_mae": summary["current_metrics"]["mae"],
        "current_short_atm_mae": summary["short_atm_metrics"]["current_atm_short_pure_mae"],
        "current_atm7_abs_err": abs(float(current_atm["value"]) - float(target_atm["value"])),
        **summary["probabilistic_metrics"],
        **summary["arbitrage_metrics"],
    }


def run_pca_ridge(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root)
    config = load_train_config(_experiment_config_path(root))
    bundle = create_train_val_bundle(config)
    model, selection, audit = PCARidgeForecaster.select(
        train_samples=bundle.train_items,
        val_samples=bundle.val_items,
    )
    output = root / "generated_results/pca_ridge_no_text"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "pca_ridge_model.joblib")
    pd.DataFrame(audit).to_csv(output / "validation_grid_search.csv", index=False)
    _write_json(output / "selection.json", asdict(selection))
    predictions = model.predict(bundle.test_items)
    pca_rows = [
        _baseline_summary_row(sample, predicted, model="pca_ridge_no_text", config=config)
        for sample, predicted in zip(bundle.test_items, predictions)
    ]
    persistence_rows = [
        _baseline_summary_row(sample, sample.current_surface, model="persistence", config=config)
        for sample in bundle.test_items
    ]
    pd.DataFrame(pca_rows).to_csv(output / "summary.csv", index=False)
    persistence_output = root / "generated_results/persistence"
    persistence_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(persistence_rows).to_csv(persistence_output / "summary.csv", index=False)
    return root


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _cluster_bootstrap(values: pd.DataFrame, *, iterations: int = 10000, seed: int = 20260722) -> dict[str, float]:
    grouped = [group["difference"].to_numpy(dtype=float) for _date, group in values.groupby("trading_date")]
    if len(grouped) < 2:
        raise ValueError("Cluster bootstrap requires at least two trading dates.")
    observed = float(values["difference"].mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[int(group_index)] for group_index in chosen])
        draws[index] = float(sample.mean())
    centered = draws - observed
    p_two = float((np.count_nonzero(np.abs(centered) >= abs(observed)) + 1) / (iterations + 1))
    p_better = float((np.count_nonzero(centered >= observed) + 1) / (iterations + 1))
    return {
        "mean_difference": observed,
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "p_two_sided": p_two,
        "p_text_better": p_better,
        "cluster_count": float(len(grouped)),
    }


def _dm_hac(differences: np.ndarray) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    count = len(values)
    mean = float(values.mean())
    centered = values - mean
    max_lag = min(count - 1, max(1, int(math.ceil(count ** (1.0 / 3.0)))))
    long_run = float(np.dot(centered, centered) / count)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    statistic = mean / standard_error if standard_error > 0.0 else math.copysign(math.inf, mean)
    return {
        "mean_difference": mean,
        "dm_statistic": statistic,
        "hac_standard_error": standard_error,
        "hac_max_lag": float(max_lag),
        "p_two_sided": float(2.0 * stats.norm.sf(abs(statistic))),
        "p_text_better": float(stats.norm.sf(statistic)),
    }


def _exact_wilcoxon_p(differences: Sequence[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value by sign enumeration."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[~np.isclose(values, 0.0, rtol=0.0, atol=1e-15)]
    if values.size == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(values), method="average")
    observed = abs(float(np.dot(np.sign(values), ranks)))
    sign_count = 1 << int(values.size)
    exceedances = 0
    for pattern in range(sign_count):
        signs = np.asarray(
            [1.0 if pattern & (1 << index) else -1.0 for index in range(values.size)],
            dtype=np.float64,
        )
        if abs(float(np.dot(signs, ranks))) >= observed - 1e-12:
            exceedances += 1
    return float(exceedances / sign_count)


def _load_neural_sample_metrics(root: Path) -> pd.DataFrame:
    selected = pd.read_csv(root / "checkpoint_selection/selected_checkpoints.csv")
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        summary_path = Path(str(row.run_dir)) / "rq1_test_json/summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing test summary: {summary_path}")
        frame = pd.read_csv(summary_path)
        if len(frame) != 539:
            raise ValueError(f"Expected 539 rows in {summary_path}, found {len(frame)}")
        frame.insert(0, "model", str(row.variant))
        frame.insert(1, "seed", int(row.seed))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_comparison(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root)
    run_pca_ridge(argparse.Namespace(experiment_root=str(root)))
    neural = _load_neural_sample_metrics(root)
    baseline_frames = [
        pd.read_csv(root / "generated_results/persistence/summary.csv"),
        pd.read_csv(root / "generated_results/pca_ridge_no_text/summary.csv"),
    ]
    sample_metrics = pd.concat([neural, *baseline_frames], ignore_index=True, sort=False)
    sample_path = root / "comparisons/rq1_test_sample_metrics.csv"
    sample_metrics.to_csv(sample_path, index=False)

    numeric_metrics = list(dict.fromkeys([*POINT_METRICS, *PROBABILISTIC_METRICS, *FINANCIAL_METRICS]))
    pair_aggregations = {metric: "mean" for metric in numeric_metrics if metric in sample_metrics.columns}
    pair_aggregations.update(
        {
            "news_timestamp_utc": "first",
            "current_snapshot_time_utc": "first",
            "target_snapshot_time_utc": "first",
        }
    )
    pair_metrics = (
        sample_metrics.groupby(["model", "seed", "surface_pair_id"], dropna=False)
        .agg(pair_aggregations)
        .reset_index()
    )
    pair_metrics.to_csv(root / "comparisons/rq1_test_pair_metrics.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for (model, seed), frame in pair_metrics.groupby(["model", "seed"], dropna=False):
        for metric in numeric_metrics:
            if metric in frame:
                summaries.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "mean": float(frame[metric].mean()),
                        "std_across_pairs": float(frame[metric].std(ddof=1)),
                        "pair_count": len(frame),
                    }
                )
    summary_by_seed = pd.DataFrame(summaries)
    summary_by_seed.to_csv(root / "comparisons/rq1_model_summary_by_seed.csv", index=False)

    pairwise_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for text_model, baseline_model in CONTRASTS:
        for metric in POINT_METRICS:
            text = pair_metrics[pair_metrics.model == text_model][
                ["seed", "surface_pair_id", "current_snapshot_time_utc", metric]
            ].rename(columns={metric: "text_error"})
            baseline = pair_metrics[pair_metrics.model == baseline_model][["seed", "surface_pair_id", metric]].rename(
                columns={metric: "baseline_error"}
            )
            if baseline["seed"].isna().all():
                baseline = baseline.drop(columns="seed")
                merged = text.merge(baseline, on="surface_pair_id", validate="many_to_one")
            else:
                merged = text.merge(baseline, on=["seed", "surface_pair_id"], validate="one_to_one")
            merged["difference"] = merged["baseline_error"] - merged["text_error"]
            merged["text_model"] = text_model
            merged["baseline_model"] = baseline_model
            merged["metric"] = metric
            pairwise_rows.extend(merged.to_dict("records"))

            seed_average = (
                merged.groupby("surface_pair_id")
                .agg(
                    difference=("difference", "mean"),
                    current_snapshot_time_utc=("current_snapshot_time_utc", "first"),
                )
                .reset_index()
            )
            seed_average["trading_date"] = pd.to_datetime(
                seed_average["current_snapshot_time_utc"], utc=True
            ).dt.date.astype(str)
            cluster = _cluster_bootstrap(seed_average)
            cluster_rows.append(
                {
                    "text_model": text_model,
                    "baseline_model": baseline_model,
                    "metric": metric,
                    "pair_count": len(seed_average),
                    "text_win_rate": float(np.mean(seed_average["difference"] > 0.0)),
                    **cluster,
                }
            )
            ordered = seed_average.sort_values("current_snapshot_time_utc")
            dm_rows.append(
                {
                    "text_model": text_model,
                    "baseline_model": baseline_model,
                    "metric": metric,
                    **_dm_hac(ordered["difference"].to_numpy()),
                }
            )
            differences_by_seed = merged.groupby("seed", dropna=False)["difference"].mean().dropna().to_numpy()
            if len(differences_by_seed) >= 2:
                t_result = stats.ttest_1samp(differences_by_seed, popmean=0.0)
                wilcoxon_p = _exact_wilcoxon_p(differences_by_seed)
                seed_rows.append(
                    {
                        "text_model": text_model,
                        "baseline_model": baseline_model,
                        "metric": metric,
                        "seed_count": len(differences_by_seed),
                        "mean_difference": float(np.mean(differences_by_seed)),
                        "std_difference": float(np.std(differences_by_seed, ddof=1)),
                        "paired_t_statistic": float(t_result.statistic),
                        "paired_t_p_two_sided": float(t_result.pvalue),
                        "paired_t_p_text_better": float(
                            stats.t.sf(float(t_result.statistic), df=len(differences_by_seed) - 1)
                        ),
                        "wilcoxon_p_two_sided": wilcoxon_p,
                        "text_win_seed_count": int(np.count_nonzero(differences_by_seed > 0.0)),
                    }
                )

    pd.DataFrame(pairwise_rows).to_csv(root / "comparisons/rq1_pairwise_differences.csv", index=False)
    cluster_frame = pd.DataFrame(cluster_rows)
    cluster_frame["p_two_sided_holm"] = _holm_adjust(cluster_frame["p_two_sided"].to_numpy())
    cluster_frame.to_csv(root / "comparisons/rq1_cluster_bootstrap_ci.csv", index=False)
    dm_frame = pd.DataFrame(dm_rows)
    dm_frame["p_two_sided_holm"] = _holm_adjust(dm_frame["p_two_sided"].to_numpy())
    dm_frame.to_csv(root / "comparisons/rq1_dm_hac_tests.csv", index=False)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(root / "comparisons/rq1_seed_level_tests.csv", index=False)

    point_seed = summary_by_seed[summary_by_seed.metric.isin(POINT_METRICS)]
    final_primary = (
        point_seed.groupby(["model", "metric"], dropna=False)
        .agg(mean_test_error=("mean", "mean"), std_across_seeds=("mean", "std"), seed_count=("mean", "count"))
        .reset_index()
    )
    final_primary.to_csv(root / "final_tables/thesis_rq1_point_metric_summary.csv", index=False)
    primary_key = (
        (cluster_frame["text_model"] == "film_wgan_text")
        & (cluster_frame["baseline_model"] == "film_wgan_no_text")
        & (cluster_frame["metric"] == "surface_mae")
    )
    primary_cluster = cluster_frame.loc[primary_key].copy()
    primary_dm = dm_frame.loc[
        (dm_frame["text_model"] == "film_wgan_text")
        & (dm_frame["baseline_model"] == "film_wgan_no_text")
        & (dm_frame["metric"] == "surface_mae")
    ].copy()
    primary_seed = seed_frame.loc[
        (seed_frame["text_model"] == "film_wgan_text")
        & (seed_frame["baseline_model"] == "film_wgan_no_text")
        & (seed_frame["metric"] == "surface_mae")
    ].copy()
    if len(primary_cluster) != 1 or len(primary_dm) != 1 or len(primary_seed) != 1:
        raise ValueError("The preregistered RQ1 primary test must resolve to exactly one result row.")
    primary_table = primary_cluster.rename(
        columns={
            "mean_difference": "no_text_minus_text_mean",
            "ci_lower": "cluster_bootstrap_ci_lower",
            "ci_upper": "cluster_bootstrap_ci_upper",
            "p_two_sided": "cluster_bootstrap_p_two_sided",
            "p_text_better": "cluster_bootstrap_p_text_better",
        }
    )
    primary_table = primary_table.merge(
        primary_dm[
            ["text_model", "baseline_model", "metric", "dm_statistic", "hac_standard_error", "p_two_sided", "p_text_better"]
        ].rename(columns={"p_two_sided": "dm_p_two_sided", "p_text_better": "dm_p_text_better"}),
        on=["text_model", "baseline_model", "metric"],
        validate="one_to_one",
    )
    primary_table = primary_table.merge(
        primary_seed[
            [
                "text_model",
                "baseline_model",
                "metric",
                "seed_count",
                "paired_t_statistic",
                "paired_t_p_two_sided",
                "paired_t_p_text_better",
                "wilcoxon_p_two_sided",
                "text_win_seed_count",
            ]
        ],
        on=["text_model", "baseline_model", "metric"],
        validate="one_to_one",
    )
    primary_table.to_csv(root / "final_tables/thesis_rq1_primary_test_table.csv", index=False)
    cluster_frame.to_csv(root / "final_tables/thesis_rq1_architecture_ablations.csv", index=False)
    for metrics, filename in (
        (PROBABILISTIC_METRICS, "thesis_rq1_probabilistic_table.csv"),
        (FINANCIAL_METRICS, "thesis_rq1_financial_consistency.csv"),
    ):
        table = (
            summary_by_seed[summary_by_seed.metric.isin(metrics)]
            .groupby(["model", "metric"], dropna=False)
            .agg(mean=("mean", "mean"), std_across_seeds=("mean", "std"), seed_count=("mean", "count"))
            .reset_index()
        )
        table.to_csv(root / "final_tables" / filename, index=False)

    validation = {
        "status": "ok",
        "sample_rows": int(len(sample_metrics)),
        "neural_test_rows": int(len(neural)),
        "test_rows_per_neural_run": 539,
        "test_unique_surface_pairs": int(pair_metrics.surface_pair_id.nunique()),
        "neural_run_count": int(len(neural.groupby(["model", "seed"]))),
        "best_seed_selection": False,
    }
    _write_json(root / "validation_summary.json", validation)
    _write_archive_docs(root)
    _write_manifest(root)
    return root


def _write_archive_docs(root: Path) -> None:
    (root / "docs/README.md").write_text(
        "# RQ1 Strict Out-of-Sample Archive\n\n"
        "This archive tests LP text-conditioned FiLM-WGAN against controlled no-text, "
        "placebo, architecture, persistence, and PCA-Ridge baselines. Positive "
        "`baseline_error_minus_text_error` means the LP text model is better. Final "
        "claims use the untouched grouped chronological test split and all five seeds.\n",
        encoding="utf-8",
    )
    (root / "docs/methodology.md").write_text(
        "# Methodology\n\n"
        "Surface pairs are split 70/15/15 chronologically, normalization is train-only, "
        "validation selects checkpoints after epoch 10, and test rows are first aggregated "
        "to unique surface pairs. Inference uses trading-day cluster bootstrap and HAC/DM "
        "diagnostics; no best seed is selected.\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def validate_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root)
    manifest = pd.read_csv(root / "inputs/split_manifest.csv")
    pair_split_counts = manifest.groupby("surface_pair_id")["split"].nunique()
    if int(pair_split_counts.max()) != 1:
        raise ValueError("At least one surface pair crosses an RQ1 split.")
    counts = manifest.groupby("split").agg(rows=("sample_id", "size"), pairs=("surface_pair_id", "nunique"))
    expected = {"train": (2639, 2039), "val": (533, 437), "test": (539, 437)}
    actual = {split: (int(row.rows), int(row.pairs)) for split, row in counts.iterrows()}
    if actual != expected:
        raise ValueError(f"Unexpected split counts: {actual}")
    validation_path = root / "validation_summary.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "ok":
            raise ValueError(f"Comparison validation is not ok: {validation}")
    print(json.dumps({"status": "ok", "split_counts": actual}, indent=2))
    return root


def package_experiment(args: argparse.Namespace) -> Path:
    root = _resolve_experiment_root(args.experiment_root)
    archive_base = root.parent / root.name
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=root.parent, base_dir=root.name))
    print(archive_path)
    return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--experiment-root")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    prepare.add_argument("--original-text-resolved-config", default=str(DEFAULT_ORIGINAL_TEXT_CONFIG))
    prepare.add_argument("--reuse", action="store_true")
    prepare.set_defaults(func=prepare_experiment)

    train = subparsers.add_parser("train-matrix")
    train.add_argument("--experiment-root")
    train.add_argument("--env-name", default="py312")
    train.add_argument("--variant", choices=NEURAL_VARIANTS)
    train.add_argument("--seed", type=int, choices=SEEDS)
    train.add_argument("--rerun", action="store_true")
    train.set_defaults(func=run_training_matrix)

    collect = subparsers.add_parser("collect-checkpoints")
    collect.add_argument("--experiment-root")
    collect.set_defaults(func=collect_checkpoints)

    generate = subparsers.add_parser("generate-test")
    generate.add_argument("--experiment-root")
    generate.add_argument("--env-name", default="py312")
    generate.add_argument("--rerun", action="store_true")
    generate.set_defaults(func=run_generate_test_matrix)

    pca = subparsers.add_parser("run-pca-ridge")
    pca.add_argument("--experiment-root")
    pca.set_defaults(func=run_pca_ridge)

    compare = subparsers.add_parser("build-comparison")
    compare.add_argument("--experiment-root")
    compare.set_defaults(func=build_comparison)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--experiment-root")
    validate.set_defaults(func=validate_experiment)

    package = subparsers.add_parser("package")
    package.add_argument("--experiment-root")
    package.set_defaults(func=package_experiment)
    return parser


def main(argv: Iterable[str] | None = None) -> Any:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    main()
