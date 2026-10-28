#!/usr/bin/env python3
"""Validation-only conformal calibration for RQ1 sequence intervals."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
DEFAULT_VARIANT = "history_matched_lp"
LEVELS = (0.5, 0.8, 0.9)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {path}")
    return path


def _selection_row(root: Path, *, fold: str, seed: int, variant: str) -> pd.Series:
    path = root / "checkpoint_selection/selected_checkpoints.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint selection: {path}")
    frame = pd.read_csv(path)
    selected = frame[
        frame["fold"].eq(fold)
        & frame["seed"].eq(int(seed))
        & frame["variant"].eq(variant)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one selected checkpoint for {variant}/{fold}/seed_{seed}; "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def _expected_split_count(root: Path, *, fold: str, split: str) -> int:
    manifest = pd.read_csv(root / "folds" / fold / "split_manifest.csv")
    return int(manifest["split"].eq(split).sum())


def _valid_generated_output(output: Path, *, expected: int) -> bool:
    summary = output / "summary.csv"
    samples = output / "samples"
    if not summary.is_file() or not samples.is_dir():
        return False
    if len(pd.read_csv(summary)) != expected:
        return False
    paths = list(samples.glob("*.json"))
    if len(paths) != expected:
        return False
    if not paths:
        return False
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    return "q_0.50" in payload.get("quantile_surfaces", {})


def generate_validation_one(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    selected = _selection_row(
        root,
        fold=args.fold,
        seed=int(args.seed),
        variant=args.variant,
    )
    run_dir = Path(str(selected["run_dir"]))
    output = run_dir / "development_val_conformal_json"
    expected = _expected_split_count(root, fold=args.fold, split="val")
    pointer = (
        root
        / "conformal_calibration/generated_validation"
        / args.variant
        / args.fold
        / f"seed_{args.seed}.json"
    )
    if _valid_generated_output(output, expected=expected):
        _write_json(
            pointer,
            {"output_dir": str(output), "sample_count": expected, "reused": True},
        )
        print(f"reused={output / 'summary.csv'}")
        return output

    command = [
        sys.executable,
        "scripts/film_wgan/main.py",
        "generate-result",
        "--config",
        str(run_dir / "metrics/training_resolved_config.yaml"),
        "--checkpoint",
        str(selected["checkpoint_path"]),
        "--output-dir",
        output.name,
        "--split",
        "val",
        "--selection-mode",
        "all",
        "--selection-count",
        "0",
        "--no-plot",
        "--set",
        "save_full_atm_timeseries=false",
    ]
    log_path = (
        root
        / "logs/conformal_generate"
        / args.variant
        / args.fold
        / f"seed_{args.seed}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not _valid_generated_output(output, expected=expected):
        raise ValueError(
            f"Validation generation mismatch for {args.variant}/{args.fold}/seed_{args.seed}."
        )
    _write_json(
        pointer,
        {"output_dir": str(output), "sample_count": expected, "reused": False},
    )
    print(f"completed={output / 'summary.csv'}")
    return output


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        raise ValueError("Cannot calculate a weighted quantile without valid values.")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / float(np.sum(weights))
    index = int(np.searchsorted(cumulative, float(quantile), side="left"))
    return float(values[min(index, len(values) - 1)])


def _day_balanced_weights(days: np.ndarray) -> np.ndarray:
    day_values = np.asarray(days, dtype=str).reshape(-1)
    unique, counts = np.unique(day_values, return_counts=True)
    count_by_day = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    day_count = len(unique)
    if day_count == 0:
        raise ValueError("No trading days are available for conformal calibration.")
    return np.asarray(
        [1.0 / (day_count * count_by_day[day]) for day in day_values],
        dtype=np.float64,
    )


def _conformal_quantile_level(level: float, trading_day_count: int) -> float:
    if trading_day_count <= 0:
        raise ValueError("trading_day_count must be positive.")
    return min(1.0, math.ceil((trading_day_count + 1) * float(level)) / trading_day_count)


def _load_sample_payloads(output: Path) -> list[dict[str, Any]]:
    summary = pd.read_csv(output / "summary.csv")
    trading_days = {
        str(row.sample_id): str(row.trading_day)
        for row in summary[["sample_id", "trading_day"]].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "samples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(payload["sample_id"])
        median = np.asarray(payload["quantile_surfaces"]["q_0.50"], dtype=np.float64)
        target = np.asarray(payload["target_surface"], dtype=np.float64)
        mask = np.asarray(payload["evaluation_support_mask"], dtype=bool)
        if median.shape != target.shape or mask.shape != target.shape:
            raise ValueError(f"Surface shape mismatch in {path}.")
        valid = mask & np.isfinite(median) & np.isfinite(target)
        if not np.any(valid):
            raise ValueError(f"No supported calibration cells in {path}.")
        rows.append(
            {
                "sample_id": sample_id,
                "surface_pair_id": str(payload["surface_pair_id"]),
                "trading_day": trading_days[sample_id],
                "median": median,
                "target": target,
                "valid": valid,
                "probabilistic_metrics": payload["probabilistic_metrics"],
            }
        )
    return rows


def _fit_radii(validation: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[float, float]]:
    scores: list[np.ndarray] = []
    days: list[np.ndarray] = []
    for sample in validation:
        score = np.abs(sample["target"] - sample["median"])[sample["valid"]]
        scores.append(score)
        days.append(np.repeat(sample["trading_day"], len(score)))
    score_values = np.concatenate(scores)
    day_values = np.concatenate(days)
    weights = _day_balanced_weights(day_values)
    day_count = len(np.unique(day_values))
    radii: dict[float, float] = {}
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        quantile_level = _conformal_quantile_level(level, day_count)
        radius = _weighted_quantile(score_values, weights, quantile_level)
        radii[level] = radius
        achieved = float(np.sum(weights * (score_values <= radius)))
        rows.append(
            {
                "nominal_level": level,
                "conformal_quantile_level": quantile_level,
                "radius": radius,
                "interval_width": 2.0 * radius,
                "validation_coverage": achieved,
                "validation_trading_days": day_count,
                "validation_supported_cells": len(score_values),
                "calibration_weighting": "equal_total_weight_per_trading_day",
            }
        )
    return rows, radii


def calibrate(args: argparse.Namespace) -> Path:
    root = _resolve_root(args.experiment_root)
    output_root = root / "conformal_calibration" / args.variant
    output_root.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in args.seeds:
            selected = _selection_row(root, fold=fold, seed=int(seed), variant=args.variant)
            run_dir = Path(str(selected["run_dir"]))
            validation = _load_sample_payloads(run_dir / "development_val_conformal_json")
            test = _load_sample_payloads(run_dir / "development_test_json")
            fitted_rows, radii = _fit_radii(validation)
            for row in fitted_rows:
                parameter_rows.append(
                    {"fold": fold, "seed": int(seed), "variant": args.variant, **row}
                )
            for sample in test:
                absolute_error = np.abs(sample["target"] - sample["median"])[sample["valid"]]
                raw = sample["probabilistic_metrics"]
                for level, radius in radii.items():
                    suffix = str(int(round(level * 100.0)))
                    sample_rows.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "variant": args.variant,
                            "sample_id": sample["sample_id"],
                            "surface_pair_id": sample["surface_pair_id"],
                            "trading_day": sample["trading_day"],
                            "nominal_level": level,
                            "raw_coverage": float(raw[f"coverage_{suffix}"]),
                            "raw_interval_width": float(raw[f"interval_width_{suffix}"]),
                            "conformal_coverage": float(np.mean(absolute_error <= radius)),
                            "conformal_interval_width": 2.0 * radius,
                            "conformal_radius": radius,
                            "supported_cell_count": len(absolute_error),
                        }
                    )

    parameters = pd.DataFrame(parameter_rows)
    samples = pd.DataFrame(sample_rows)
    parameters.to_csv(output_root / "validation_calibration_parameters.csv", index=False)
    samples.to_csv(output_root / "test_conformal_sample_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for level, group in samples.groupby("nominal_level", sort=True):
        daily = group.groupby("trading_day", as_index=False).agg(
            raw_coverage=("raw_coverage", "mean"),
            conformal_coverage=("conformal_coverage", "mean"),
            raw_interval_width=("raw_interval_width", "mean"),
            conformal_interval_width=("conformal_interval_width", "mean"),
        )
        summary_rows.append(
            {
                "scope": "overall_test",
                "fold": "all",
                "variant": args.variant,
                "nominal_level": level,
                "raw_coverage_day_weighted": daily["raw_coverage"].mean(),
                "conformal_coverage_day_weighted": daily["conformal_coverage"].mean(),
                "raw_interval_width_day_weighted": daily["raw_interval_width"].mean(),
                "conformal_interval_width_day_weighted": daily[
                    "conformal_interval_width"
                ].mean(),
                "raw_calibration_error": abs(daily["raw_coverage"].mean() - level),
                "conformal_calibration_error": abs(
                    daily["conformal_coverage"].mean() - level
                ),
                "test_trading_days": daily["trading_day"].nunique(),
                "test_sample_seed_rows": len(group),
            }
        )
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_daily = fold_group.groupby("trading_day", as_index=False).agg(
                raw_coverage=("raw_coverage", "mean"),
                conformal_coverage=("conformal_coverage", "mean"),
                raw_interval_width=("raw_interval_width", "mean"),
                conformal_interval_width=("conformal_interval_width", "mean"),
            )
            summary_rows.append(
                {
                    "scope": "fold_test",
                    "fold": fold,
                    "variant": args.variant,
                    "nominal_level": level,
                    "raw_coverage_day_weighted": fold_daily["raw_coverage"].mean(),
                    "conformal_coverage_day_weighted": fold_daily[
                        "conformal_coverage"
                    ].mean(),
                    "raw_interval_width_day_weighted": fold_daily[
                        "raw_interval_width"
                    ].mean(),
                    "conformal_interval_width_day_weighted": fold_daily[
                        "conformal_interval_width"
                    ].mean(),
                    "raw_calibration_error": abs(
                        fold_daily["raw_coverage"].mean() - level
                    ),
                    "conformal_calibration_error": abs(
                        fold_daily["conformal_coverage"].mean() - level
                    ),
                    "test_trading_days": fold_daily["trading_day"].nunique(),
                    "test_sample_seed_rows": len(fold_group),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "test_conformal_summary.csv", index=False)

    readme = f"""# RQ1 Validation Conformal Calibration Pilot

Variant: `{args.variant}`

This pilot uses validation data only to fit symmetric intervals around the generated
median surface. Absolute residual scores receive equal total weight per trading day.
The finite-sample quantile correction uses the number of validation trading days.
Parameters are fitted independently for every rolling fold and seed, frozen, and then
applied to the corresponding test quarter. Test observations never enter calibration.

This procedure calibrates marginal supported-cell coverage. It does not repair the
under-dispersed WGAN scenario distribution and does not claim simultaneous full-surface
coverage.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    _write_json(
        output_root / "validation_summary.json",
        {
            "status": "ok",
            "variant": args.variant,
            "folds": list(FOLDS),
            "seeds": [int(seed) for seed in args.seeds],
            "levels": list(LEVELS),
            "calibration_split": "val",
            "evaluation_split": "test",
            "method": "median_centered_absolute_residual_split_conformal",
            "weighting": "equal_total_weight_per_trading_day",
            "test_leakage": False,
        },
    )
    print(summary[summary["scope"].eq("overall_test")].to_string(index=False))
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-validation-one")
    generate.add_argument("--experiment-root", required=True)
    generate.add_argument("--fold", choices=FOLDS, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--variant", default=DEFAULT_VARIANT)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--experiment-root", required=True)
    calibration.add_argument("--variant", default=DEFAULT_VARIANT)
    calibration.add_argument("--seeds", nargs="+", type=int, default=[42, 202, 404])
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "generate-validation-one":
        return generate_validation_one(args)
    return calibrate(args)


if __name__ == "__main__":
    main()
