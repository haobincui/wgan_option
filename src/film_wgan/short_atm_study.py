"""Short-end ATM experiment orchestration for standalone FiLM WGAN runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from utils.postprocess_runtime import write_summary_csv

from .config import load_train_config
from .io import utc_timestamp, write_json
from .trainer import FilmWGANTrainer

PRIMARY_METRIC = "val_atm_short_pure_mae_gap_vs_current"
EXTRA_CHECKPOINT_METRICS = (
    "val_mae_gap_vs_current",
    "val_short_atm_mae_gap_vs_current",
)
DEFAULT_BLEND_ALPHAS = (0.70, 0.85, 1.00)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_STUDY_ROOT = Path("outputs/training/film_wgan/short_atm_study")
DEFAULT_BASE_CONFIG_PATH = "configs/film_wgan/train_lp_exp_F3.yaml"

TRAINING_METRICS_FOR_SELECTION = (
    "val_atm_short_pure_mae_gap_vs_current",
    "val_short_atm_mae_gap_vs_current",
    "val_mae_gap_vs_current",
    "val_atm_short_win_rate_vs_current",
    "val_win_rate_vs_current",
    "val_calendar",
    "val_butterfly",
)
EXPORT_METRICS = (
    "mae",
    "rmse",
    "max_abs",
    "current_mae",
    "current_rmse",
    "current_max_abs",
    "mae_gap_vs_current",
    "rmse_gap_vs_current",
    "max_abs_gap_vs_current",
    "win_flag_vs_current",
    "short_atm_weighted_mae",
    "current_short_atm_weighted_mae",
    "short_atm_mae_gap_vs_current",
    "short_atm_weighted_win_flag_vs_current",
    "atm_short_pure_mae",
    "current_atm_short_pure_mae",
    "atm_short_pure_mae_gap_vs_current",
    "atm_short_pure_win_flag_vs_current",
    "generated_current_mae",
    "penalty_mean",
    "penalty_std",
    "weight_entropy",
)
QUARTILE_EXPORT_METRICS = (
    "mae_gap_vs_current",
    "win_flag_vs_current",
    "short_atm_mae_gap_vs_current",
    "short_atm_weighted_win_flag_vs_current",
    "atm_short_pure_mae_gap_vs_current",
    "atm_short_pure_win_flag_vs_current",
)
CHECKPOINT_EXPORT_SPECS = (
    ("primary", "film_wgan_best.pt"),
    ("val_mae_gap_vs_current", "film_wgan_best_val_mae_gap_vs_current.pt"),
    ("val_short_atm_mae_gap_vs_current", "film_wgan_best_val_short_atm_mae_gap_vs_current.pt"),
)


@dataclass(frozen=True)
class ShortATMGridSpec:
    """One fixed grid cell in the short-end ATM experiment matrix."""

    config_id: str
    purpose: str
    recon_atm_range: float
    recon_atm_short_end_max_days: float
    recon_atm_multiplier: float
    lambda_atm_short: float


GRID_SPECS = (
    ShortATMGridSpec(
        config_id="A",
        purpose="Current short-end control equivalent to the strongest short-end baseline.",
        recon_atm_range=0.04,
        recon_atm_short_end_max_days=60.0,
        recon_atm_multiplier=16.0,
        lambda_atm_short=50.0,
    ),
    ShortATMGridSpec(
        config_id="B",
        purpose="Short-end control plus stronger ATM-short supervision.",
        recon_atm_range=0.04,
        recon_atm_short_end_max_days=60.0,
        recon_atm_multiplier=16.0,
        lambda_atm_short=75.0,
    ),
    ShortATMGridSpec(
        config_id="C",
        purpose="Wider reconstruction band control with unchanged ATM-short supervision.",
        recon_atm_range=0.06,
        recon_atm_short_end_max_days=90.0,
        recon_atm_multiplier=8.0,
        lambda_atm_short=50.0,
    ),
    ShortATMGridSpec(
        config_id="D",
        purpose="Wider reconstruction band plus stronger ATM-short supervision.",
        recon_atm_range=0.06,
        recon_atm_short_end_max_days=90.0,
        recon_atm_multiplier=8.0,
        lambda_atm_short=75.0,
    ),
)


def build_short_atm_grid_specs() -> tuple[ShortATMGridSpec, ...]:
    """Return the fixed 2x2 experiment grid used by the short-end ATM study."""

    return GRID_SPECS


def select_blend_scan_config_ids(
    primary_summary_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 2,
) -> list[str]:
    """Return the top configuration ids ranked by primary short-end ATM metric."""

    ranked = sorted(
        primary_summary_rows,
        key=lambda row: float(row["val_atm_short_pure_mae_gap_vs_current_mean"]),
    )
    return [str(row["config_id"]) for row in ranked[: max(0, int(top_k))]]


def choose_recommended_config(
    primary_summary_rows: Sequence[Mapping[str, Any]],
    *,
    control_config_id: str = "A",
) -> dict[str, Any]:
    """Apply the study selection rule and return the chosen configuration."""

    if not primary_summary_rows:
        return {
            "recommended_config_id": "",
            "control_config_id": str(control_config_id),
            "control_present": False,
            "qualified_config_ids": [],
            "reason": "no_primary_rows",
        }

    rows_by_id = {str(row["config_id"]): dict(row) for row in primary_summary_rows}
    control = rows_by_id.get(control_config_id)
    if control is None:
        fallback = min(primary_summary_rows, key=lambda row: float(row["val_atm_short_pure_mae_gap_vs_current_mean"]))
        return {
            "recommended_config_id": str(fallback["config_id"]),
            "control_config_id": str(control_config_id),
            "control_present": False,
            "qualified_config_ids": [str(fallback["config_id"])],
            "reason": "control_missing",
        }

    qualified: list[dict[str, Any]] = []
    control_primary = float(control["val_atm_short_pure_mae_gap_vs_current_mean"])
    for row in primary_summary_rows:
        config_id = str(row["config_id"])
        if config_id == control_config_id:
            continue
        primary = float(row["val_atm_short_pure_mae_gap_vs_current_mean"])
        mae_gap = float(row["val_mae_gap_vs_current_mean"])
        calendar = float(row["val_calendar_mean"])
        butterfly = float(row["val_butterfly_mean"])
        improvement_vs_control = control_primary - primary
        passes = (
            improvement_vs_control >= 2e-4
            and mae_gap <= -0.0018
            and calendar <= 2e-5
            and butterfly <= 2e-6
        )
        if passes:
            enriched = dict(row)
            enriched["improvement_vs_control"] = improvement_vs_control
            qualified.append(enriched)

    if qualified:
        best = min(qualified, key=lambda row: float(row["val_atm_short_pure_mae_gap_vs_current_mean"]))
        return {
            "recommended_config_id": str(best["config_id"]),
            "control_config_id": str(control_config_id),
            "control_present": True,
            "qualified_config_ids": [str(row["config_id"]) for row in qualified],
            "reason": "qualified_better_than_control",
        }
    return {
        "recommended_config_id": str(control_config_id),
        "control_config_id": str(control_config_id),
        "control_present": True,
        "qualified_config_ids": [],
        "reason": "no_config_passed_threshold",
    }


def _study_manifest_path(study_dir: str | Path) -> Path:
    return Path(study_dir) / "study_manifest.json"


def _load_manifest(study_dir: str | Path) -> dict[str, Any]:
    path = _study_manifest_path(study_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing short ATM study manifest: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_manifest(study_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = _study_manifest_path(study_dir)
    write_json(path, dict(manifest))
    return path


def _default_study_dir(base_output_root: str | Path = DEFAULT_STUDY_ROOT) -> Path:
    return Path(base_output_root) / utc_timestamp()


def _config_common_overrides() -> dict[str, Any]:
    return {
        "lambda_adv": 0.05,
        "adv_warmup_epochs": 10,
        "use_atm_short_loss": True,
        "atm_short_range": 0.04,
        "atm_short_max_days": 60.0,
        "checkpoint_metric": PRIMARY_METRIC,
        "extra_checkpoint_metrics": list(EXTRA_CHECKPOINT_METRICS),
        "num_epochs": 30,
        "use_early_stopping": True,
        "early_stopping_patience": 7,
    }


def _config_overrides_for_spec(
    spec: ShortATMGridSpec,
    *,
    seed: int,
    run_output_root: str | Path,
) -> dict[str, Any]:
    overrides = _config_common_overrides()
    overrides.update(
        {
            "seed": int(seed),
            "output_root": str(run_output_root),
            "recon_atm_range": float(spec.recon_atm_range),
            "recon_atm_short_end_max_days": float(spec.recon_atm_short_end_max_days),
            "recon_atm_multiplier": float(spec.recon_atm_multiplier),
            "lambda_atm_short": float(spec.lambda_atm_short),
        }
    )
    return overrides


def _run_output_root(study_dir: str | Path, *, config_id: str, seed: int) -> Path:
    return Path(study_dir) / "runs" / f"{config_id}_seed_{int(seed)}"


def _checkpoint_exports_for_run(run_dir: str | Path) -> dict[str, str]:
    checkpoints_dir = Path(run_dir) / "checkpoints"
    return {
        label: str(checkpoints_dir / filename)
        for label, filename in CHECKPOINT_EXPORT_SPECS
    }


def _load_resolved_training_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "metrics" / "training_resolved_config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _load_training_metric_rows(run_dir: str | Path) -> dict[int, dict[str, float]]:
    metrics_path = Path(run_dir) / "metrics" / "training_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[int, dict[str, float]] = {}
    for row in rows:
        epoch = int(float(row["epoch"]))
        output[epoch] = {
            key: float(value)
            for key, value in row.items()
            if key != "epoch" and value not in {"", None}
        }
    return output


def _load_best_metric_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics" / "best_metrics_summary.json"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _training_checkpoint_rows(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    metric_summary = _load_best_metric_summary(run_dir)
    training_rows = _load_training_metric_rows(run_dir)
    metrics_payload = metric_summary.get("metrics") or {}
    primary_metric = str(metric_summary.get("primary_metric") or PRIMARY_METRIC)
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    for label, _filename in CHECKPOINT_EXPORT_SPECS:
        metric_name = primary_metric if label == "primary" else label
        info = metrics_payload.get(metric_name) or {}
        best_epoch = int(info.get("best_epoch") or 0)
        checkpoint_rows[label] = {
            "selection_metric": metric_name,
            "best_epoch": best_epoch,
            "training_row": training_rows.get(best_epoch, {}),
        }
    return checkpoint_rows


def _summary_rows(summary_csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(summary_csv_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0)) if values else 0.0


def _numeric_mean_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in {None, ""}:
            continue
        values.append(float(raw))
    return _mean(values)


def _export_summary_row(
    *,
    run_record: Mapping[str, Any],
    checkpoint_label: str,
    generate_dir: str | Path,
) -> dict[str, Any]:
    summary_path = Path(generate_dir) / "summary.csv"
    rows = _summary_rows(summary_path)
    training_lookup = _training_checkpoint_rows(run_record["run_dir"])
    training_entry = training_lookup.get(checkpoint_label, {})
    training_row = dict(training_entry.get("training_row") or {})
    summary = {
        "config_id": str(run_record["config_id"]),
        "seed": int(run_record["seed"]),
        "run_id": str(run_record["run_id"]),
        "checkpoint_label": str(checkpoint_label),
        "selection_metric": str(training_entry.get("selection_metric") or ""),
        "best_epoch": int(training_entry.get("best_epoch") or 0),
        "run_dir": str(run_record["run_dir"]),
        "generate_dir": str(generate_dir),
        "summary_csv": str(summary_path),
        "sample_count": len(rows),
    }
    for metric_name in TRAINING_METRICS_FOR_SELECTION:
        summary[metric_name] = float(training_row.get(metric_name, 0.0))
    for metric_name in EXPORT_METRICS:
        summary[f"export_{metric_name}"] = _numeric_mean_from_rows(rows, metric_name)
    summary["export_residual_blend_alpha"] = _numeric_mean_from_rows(rows, "residual_blend_alpha")
    return summary


def _aggregate_rows_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for group, group_rows in grouped.items():
        output = {key: value for key, value in zip(group_keys, group)}
        output["seed_count"] = len(group_rows)
        for metric_name in metric_keys:
            values = [float(row[metric_name]) for row in group_rows]
            output[f"{metric_name}_mean"] = _mean(values)
            output[f"{metric_name}_std"] = _std(values)
        aggregated.append(output)
    return sorted(aggregated, key=lambda row: tuple(row[key] for key in group_keys))


def _quartile_rows(
    *,
    run_record: Mapping[str, Any],
    summary_dir: str | Path,
) -> list[dict[str, Any]]:
    rows = _summary_rows(Path(summary_dir) / "summary.csv")
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["current_atm_short_pure_mae"]))
    buckets: list[list[Mapping[str, Any]]] = [[], [], [], []]
    for index, row in enumerate(ordered):
        bucket_index = min(3, int(index * 4 / max(1, len(ordered))))
        buckets[bucket_index].append(row)

    output: list[dict[str, Any]] = []
    for quartile_index, bucket_rows in enumerate(buckets, start=1):
        quartile_row = {
            "config_id": str(run_record["config_id"]),
            "seed": int(run_record["seed"]),
            "run_id": str(run_record["run_id"]),
            "quartile": quartile_index,
            "sample_count": len(bucket_rows),
        }
        for metric_name in QUARTILE_EXPORT_METRICS:
            quartile_row[metric_name] = _numeric_mean_from_rows(bucket_rows, metric_name)
        quartile_row["current_atm_short_pure_mae"] = _numeric_mean_from_rows(
            bucket_rows,
            "current_atm_short_pure_mae",
        )
        output.append(quartile_row)
    return output


def train_short_atm_grid(
    *,
    base_config_path: str = DEFAULT_BASE_CONFIG_PATH,
    study_dir: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save_json: bool = False,
    save_plots: bool = False,
) -> Path:
    """Run the 2x2x3 short-end ATM training grid and export all checkpoint summaries."""

    resolved_study_dir = Path(study_dir) if study_dir is not None else _default_study_dir()
    resolved_study_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_dir": str(resolved_study_dir),
        "base_config_path": str(base_config_path),
        "seeds": [int(seed) for seed in seeds],
        "blend_alphas": [float(alpha) for alpha in DEFAULT_BLEND_ALPHAS],
        "grid_specs": [asdict(spec) for spec in build_short_atm_grid_specs()],
        "runs": [],
    }
    _save_manifest(resolved_study_dir, manifest)

    for spec in build_short_atm_grid_specs():
        for seed in seeds:
            run_output_root = _run_output_root(resolved_study_dir, config_id=spec.config_id, seed=int(seed))
            overrides = _config_overrides_for_spec(spec, seed=int(seed), run_output_root=run_output_root)
            trainer = FilmWGANTrainer(
                load_train_config(base_config_path, overrides=overrides),
                config_path=str(base_config_path),
            )
            run_dir = trainer.train()
            if run_dir is None:
                raise RuntimeError(f"Training returned no run directory for {spec.config_id} seed={seed}")
            run_record = {
                "run_id": f"{spec.config_id}_seed_{int(seed)}",
                "config_id": spec.config_id,
                "purpose": spec.purpose,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "train_overrides": overrides,
                "checkpoints": _checkpoint_exports_for_run(run_dir),
                "checkpoint_exports": {},
                "blend_exports": {},
            }
            manifest["runs"].append(run_record)
            _save_manifest(resolved_study_dir, manifest)
            checkpoint_exports = export_checkpoint_evaluations(
                study_dir=resolved_study_dir,
                run_record=run_record,
                base_config_path=base_config_path,
                save_json=save_json,
                save_plots=save_plots,
            )
            run_record["checkpoint_exports"] = checkpoint_exports
            _save_manifest(resolved_study_dir, manifest)

    return resolved_study_dir


def export_checkpoint_evaluations(
    *,
    study_dir: str | Path,
    run_record: Mapping[str, Any],
    base_config_path: str,
    save_json: bool,
    save_plots: bool,
) -> dict[str, str]:
    """Export generate-result summaries for the primary and extra checkpoint metrics."""

    resolved_training_config = _load_resolved_training_config(run_record["run_dir"])
    eval_mc_samples = int(resolved_training_config.get("eval_mc_samples", 32))
    trainer = FilmWGANTrainer(load_train_config(base_config_path), config_path=str(base_config_path))
    exported: dict[str, str] = {}
    for checkpoint_label, checkpoint_filename in CHECKPOINT_EXPORT_SPECS:
        checkpoint_path = Path(run_record["run_dir"]) / "checkpoints" / checkpoint_filename
        if not checkpoint_path.exists():
            continue
        generate_dir = trainer.generate_result(
            overrides={
                "checkpoint_path": str(checkpoint_path),
                "output_dir": f"generate_result/checkpoint_eval/{checkpoint_label}",
                "split": "val",
                "selection_mode": "all",
                "selection_count": 0,
                "mc_samples": eval_mc_samples,
                "save_json": bool(save_json),
                "save_plots": bool(save_plots),
                "seed": int(run_record["seed"]),
                "residual_blend_alpha": 1.0,
            },
            config_path=str(base_config_path),
        )
        exported[checkpoint_label] = str(generate_dir)

    manifest = _load_manifest(study_dir)
    for manifest_run in manifest.get("runs", []):
        if str(manifest_run.get("run_id")) == str(run_record["run_id"]):
            manifest_run["checkpoint_exports"] = exported
            break
    _save_manifest(study_dir, manifest)
    return exported


def summarize_short_atm_study(study_dir: str | Path) -> dict[str, Any]:
    """Aggregate checkpoint exports, quartile analysis, and configuration selection."""

    manifest = _load_manifest(study_dir)
    checkpoint_summary_by_run: list[dict[str, Any]] = []
    primary_quartile_by_run: list[dict[str, Any]] = []
    for run_record in manifest.get("runs", []):
        for checkpoint_label, generate_dir in (run_record.get("checkpoint_exports") or {}).items():
            checkpoint_summary_by_run.append(
                _export_summary_row(
                    run_record=run_record,
                    checkpoint_label=str(checkpoint_label),
                    generate_dir=generate_dir,
                )
            )
        primary_dir = (run_record.get("checkpoint_exports") or {}).get("primary")
        if primary_dir:
            primary_quartile_by_run.extend(
                _quartile_rows(
                    run_record=run_record,
                    summary_dir=primary_dir,
                )
            )

    checkpoint_summary_path = Path(study_dir) / "checkpoint_summary_by_run.csv"
    write_summary_csv(checkpoint_summary_by_run, checkpoint_summary_path)
    checkpoint_summary_by_config = _aggregate_rows_by_group(
        checkpoint_summary_by_run,
        group_keys=("config_id", "checkpoint_label"),
        metric_keys=TRAINING_METRICS_FOR_SELECTION + tuple(f"export_{name}" for name in EXPORT_METRICS),
    )
    checkpoint_summary_by_config_path = Path(study_dir) / "checkpoint_summary_by_config.csv"
    write_summary_csv(checkpoint_summary_by_config, checkpoint_summary_by_config_path)

    primary_summary_rows = [row for row in checkpoint_summary_by_run if str(row["checkpoint_label"]) == "primary"]
    primary_summary_by_config = _aggregate_rows_by_group(
        primary_summary_rows,
        group_keys=("config_id",),
        metric_keys=TRAINING_METRICS_FOR_SELECTION + tuple(f"export_{name}" for name in EXPORT_METRICS),
    )
    primary_summary_by_config_path = Path(study_dir) / "primary_summary_by_config.csv"
    write_summary_csv(primary_summary_by_config, primary_summary_by_config_path)

    primary_quartile_summary_by_config = _aggregate_rows_by_group(
        primary_quartile_by_run,
        group_keys=("config_id", "quartile"),
        metric_keys=("current_atm_short_pure_mae",) + QUARTILE_EXPORT_METRICS,
    )
    primary_quartile_summary_path = Path(study_dir) / "primary_quartile_summary.csv"
    write_summary_csv(primary_quartile_summary_by_config, primary_quartile_summary_path)

    selected_config_ids = select_blend_scan_config_ids(primary_summary_by_config, top_k=2)
    recommendation = choose_recommended_config(primary_summary_by_config)
    summary_payload = {
        "study_dir": str(study_dir),
        "primary_metric": PRIMARY_METRIC,
        "selected_for_blend_scan": selected_config_ids,
        **recommendation,
    }
    summary_json_path = Path(study_dir) / "selection_summary.json"
    write_json(summary_json_path, summary_payload)

    manifest["reports"] = {
        "checkpoint_summary_by_run": str(checkpoint_summary_path),
        "checkpoint_summary_by_config": str(checkpoint_summary_by_config_path),
        "primary_summary_by_config": str(primary_summary_by_config_path),
        "primary_quartile_summary": str(primary_quartile_summary_path),
        "selection_summary": str(summary_json_path),
    }
    manifest["selected_for_blend_scan"] = selected_config_ids
    manifest["recommended_config_id"] = summary_payload["recommended_config_id"]
    _save_manifest(study_dir, manifest)
    return summary_payload


def _alpha_tag(alpha: float) -> str:
    return f"alpha_{float(alpha):.2f}".replace(".", "p")


def run_blend_scan(
    study_dir: str | Path,
    *,
    base_config_path: str | None = None,
    blend_alphas: Sequence[float] = DEFAULT_BLEND_ALPHAS,
    save_json: bool = False,
    save_plots: bool = False,
) -> dict[str, Any]:
    """Run residual blend scans on the top two primary configurations."""

    manifest = _load_manifest(study_dir)
    resolved_base_config_path = str(base_config_path or manifest.get("base_config_path") or DEFAULT_BASE_CONFIG_PATH)
    top_config_ids = list(manifest.get("selected_for_blend_scan") or [])
    if not top_config_ids:
        top_config_ids = select_blend_scan_config_ids(
            _aggregate_rows_by_group(
                [
                    _export_summary_row(
                        run_record=run_record,
                        checkpoint_label="primary",
                        generate_dir=(run_record.get("checkpoint_exports") or {}).get("primary"),
                    )
                    for run_record in manifest.get("runs", [])
                    if (run_record.get("checkpoint_exports") or {}).get("primary")
                ],
                group_keys=("config_id",),
                metric_keys=TRAINING_METRICS_FOR_SELECTION,
            ),
            top_k=2,
        )

    blend_rows_by_run: list[dict[str, Any]] = []
    for run_record in manifest.get("runs", []):
        if str(run_record["config_id"]) not in set(top_config_ids):
            continue
        resolved_training_config = _load_resolved_training_config(run_record["run_dir"])
        eval_mc_samples = int(resolved_training_config.get("eval_mc_samples", 32))
        trainer = FilmWGANTrainer(load_train_config(resolved_base_config_path), config_path=resolved_base_config_path)
        primary_checkpoint_path = Path(run_record["run_dir"]) / "checkpoints" / "film_wgan_best.pt"
        if not primary_checkpoint_path.exists():
            continue
        run_record.setdefault("blend_exports", {})
        for alpha in blend_alphas:
            alpha = float(alpha)
            alpha_key = _alpha_tag(alpha)
            if np.isclose(alpha, 1.0):
                existing_primary = (run_record.get("checkpoint_exports") or {}).get("primary")
                if existing_primary:
                    run_record["blend_exports"][alpha_key] = existing_primary
                    continue
            generate_dir = trainer.generate_result(
                overrides={
                    "checkpoint_path": str(primary_checkpoint_path),
                    "output_dir": f"generate_result/blend_scan/{alpha_key}",
                    "split": "val",
                    "selection_mode": "all",
                    "selection_count": 0,
                    "mc_samples": eval_mc_samples,
                    "save_json": bool(save_json),
                    "save_plots": bool(save_plots),
                    "seed": int(run_record["seed"]),
                    "residual_blend_alpha": alpha,
                },
                config_path=resolved_base_config_path,
            )
            run_record["blend_exports"][alpha_key] = str(generate_dir)

        for alpha_key, generate_dir in sorted((run_record.get("blend_exports") or {}).items()):
            blend_row = {
                "config_id": str(run_record["config_id"]),
                "seed": int(run_record["seed"]),
                "run_id": str(run_record["run_id"]),
                "alpha_key": str(alpha_key),
                "generate_dir": str(generate_dir),
            }
            summary_rows = _summary_rows(Path(generate_dir) / "summary.csv")
            blend_row["sample_count"] = len(summary_rows)
            blend_row["residual_blend_alpha"] = _numeric_mean_from_rows(summary_rows, "residual_blend_alpha")
            for metric_name in (
                "mae_gap_vs_current",
                "win_flag_vs_current",
                "short_atm_mae_gap_vs_current",
                "short_atm_weighted_win_flag_vs_current",
                "atm_short_pure_mae_gap_vs_current",
                "atm_short_pure_win_flag_vs_current",
            ):
                blend_row[metric_name] = _numeric_mean_from_rows(summary_rows, metric_name)
            blend_rows_by_run.append(blend_row)

    blend_rows_by_run_path = Path(study_dir) / "blend_scan_by_run.csv"
    write_summary_csv(blend_rows_by_run, blend_rows_by_run_path)
    blend_summary_by_config = _aggregate_rows_by_group(
        blend_rows_by_run,
        group_keys=("config_id", "alpha_key"),
        metric_keys=(
            "residual_blend_alpha",
            "mae_gap_vs_current",
            "win_flag_vs_current",
            "short_atm_mae_gap_vs_current",
            "short_atm_weighted_win_flag_vs_current",
            "atm_short_pure_mae_gap_vs_current",
            "atm_short_pure_win_flag_vs_current",
        ),
    )
    blend_summary_by_config_path = Path(study_dir) / "blend_scan_by_config.csv"
    write_summary_csv(blend_summary_by_config, blend_summary_by_config_path)

    best_alpha_by_config = {}
    for config_id in sorted({str(row["config_id"]) for row in blend_summary_by_config}):
        config_rows = [row for row in blend_summary_by_config if str(row["config_id"]) == config_id]
        if not config_rows:
            continue
        best_row = min(
            config_rows,
            key=lambda row: float(row["atm_short_pure_mae_gap_vs_current_mean"]),
        )
        best_alpha_by_config[config_id] = {
            "alpha_key": str(best_row["alpha_key"]),
            "residual_blend_alpha": float(best_row["residual_blend_alpha_mean"]),
            "atm_short_pure_mae_gap_vs_current_mean": float(best_row["atm_short_pure_mae_gap_vs_current_mean"]),
        }

    blend_summary_json_path = Path(study_dir) / "blend_scan_summary.json"
    blend_summary_payload = {
        "study_dir": str(study_dir),
        "selected_config_ids": top_config_ids,
        "best_alpha_by_config": best_alpha_by_config,
    }
    write_json(blend_summary_json_path, blend_summary_payload)

    manifest["blend_alphas"] = [float(alpha) for alpha in blend_alphas]
    manifest["blend_summary"] = str(blend_summary_json_path)
    _save_manifest(study_dir, manifest)
    return blend_summary_payload


def run_short_atm_study(
    *,
    base_config_path: str = DEFAULT_BASE_CONFIG_PATH,
    study_dir: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    blend_alphas: Sequence[float] = DEFAULT_BLEND_ALPHAS,
    save_json: bool = False,
    save_plots: bool = False,
) -> Path:
    """Run the full train -> summarize -> blend-scan workflow."""

    resolved_study_dir = train_short_atm_grid(
        base_config_path=base_config_path,
        study_dir=study_dir,
        seeds=seeds,
        save_json=save_json,
        save_plots=save_plots,
    )
    summarize_short_atm_study(resolved_study_dir)
    run_blend_scan(
        resolved_study_dir,
        base_config_path=base_config_path,
        blend_alphas=blend_alphas,
        save_json=save_json,
        save_plots=save_plots,
    )
    return resolved_study_dir
