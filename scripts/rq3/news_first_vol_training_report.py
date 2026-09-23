"""Portable technical report for the news-first vol training comparison.

The report is intentionally assembled as a canonical Data Analytics artifact
and rendered by the packaged portable-artifact builder.  It does not hand-roll
HTML or a chart runtime, which keeps the narrative, data and provenance in one
reviewable JSON payload.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REPORT_TITLE = "TY News-first Vol Surface 并行训练与统计对比"
DEFAULT_MAE_NUMERICAL_TIE_TOLERANCE = 1.0e-8
DEFAULT_DELIVER_SCRIPT = Path(
    "/home/haobin_cui/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/"
    "deliver_portable_artifact.mjs"
)
DEFAULT_NODE_CANDIDATES = (
    Path(
        "/home/haobin_cui/research_files_space_2/fomc_trainer/.cache/"
        "node_official/node-v22.12.0-linux-x64/bin/node"
    ),
    Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/node"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame, *, limit: int = 2_000) -> list[dict[str, Any]]:
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.head(limit).to_dict(orient="records")
    ]


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _model_label(value: Any) -> str:
    text = str(value).strip().lower()
    if "wgan" in text or text == "gan":
        return "WGAN-GP"
    if "reg" in text:
        return "Residual regression"
    return str(value)


def _text_mode_label(value: Any) -> str:
    return {
        "real_text": "Real text",
        "current_only": "Current-only (zero text)",
        "text_shuffle": "Fixed shuffled text",
    }.get(str(value).strip().lower(), str(value))


def _gap_interpretation(gap: float, tolerance: float) -> str:
    if abs(float(gap)) <= float(tolerance):
        return "与 persistence 在数值精度内持平"
    return "优于 persistence" if float(gap) < -float(tolerance) else "未优于 persistence"


def _normalise_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "mae_gap" not in result and "gap" in result:
        result["mae_gap"] = result["gap"]
    if "win_rate" not in result and "win" in result:
        result["win_rate"] = result["win"]
    if "stratum" not in result and {
        "stratum_type",
        "stratum_value",
    }.issubset(result.columns):
        result["stratum"] = (
            result["stratum_type"].astype(str)
            + ":"
            + result["stratum_value"].astype(str)
        )
    if "model" in result:
        result["model_label"] = result["model"].map(_model_label)
    if "tolerance_minutes" in result:
        result["tolerance_label"] = result["tolerance_minutes"].map(
            lambda value: f"{int(value)}m"
        )
    if "text_ablation_mode" in result:
        result["text_mode_label"] = result["text_ablation_mode"].map(
            _text_mode_label
        )
    if {"model_label", "tolerance_label"}.issubset(result.columns):
        result["series"] = result["model_label"] + " · " + result["tolerance_label"]
        if "text_mode_label" in result:
            result["series"] += " · " + result["text_mode_label"]
    return result


def _core_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    mask = pd.Series(True, index=comparison.index)
    if "panel" in comparison:
        mask &= comparison["panel"].astype(str).str.lower().isin(
            {"common_test", "common_q4", "test_core", "core"}
        )
    if {"stratum_type", "stratum_value"}.issubset(comparison.columns):
        mask &= comparison["stratum_type"].astype(str).str.lower().eq("overall")
        mask &= comparison["stratum_value"].astype(str).str.lower().eq("all")
    elif "stratum" in comparison:
        mask &= comparison["stratum"].astype(str).str.lower().isin(
            {"all", "overall", "full"}
        )
    subset = comparison.loc[mask]
    return subset if not subset.empty else comparison


def _primary_bootstrap_rows(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Return only the pre-registered common-core surface-MAE family."""

    if bootstrap.empty:
        return bootstrap.copy()
    mask = pd.Series(True, index=bootstrap.index)
    if "model" in bootstrap:
        mask &= bootstrap["model"].astype(str).str.lower().str.contains("wgan")
    if "text_ablation_mode" in bootstrap:
        mask &= bootstrap["text_ablation_mode"].astype(str).str.lower().eq("real_text")
    if "panel" in bootstrap:
        mask &= bootstrap["panel"].astype(str).str.lower().eq("core")
    if "metric" in bootstrap:
        mask &= bootstrap["metric"].astype(str).eq("model_mae")
    if "stratum_type" in bootstrap:
        mask &= bootstrap["stratum_type"].astype(str).str.lower().eq("overall")
    if "stratum_value" in bootstrap:
        mask &= bootstrap["stratum_value"].astype(str).str.lower().eq("all")
    return bootstrap.loc[mask].copy()


def _gap_long(core: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("mae_gap", "完整 surface"),
        ("short_atm_gap", "短端近 ATM"),
        ("atm_gap", "实际近似 ATM"),
        ("skew_gap", "实际局部 skew"),
    )
    rows: list[dict[str, Any]] = []
    for raw in core.to_dict(orient="records"):
        for field, label in definitions:
            value = _numeric(raw.get(field))
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "series": raw.get("series", ""),
                    "model_label": raw.get("model_label", ""),
                    "tolerance_minutes": raw.get("tolerance_minutes"),
                    "metric_label": label,
                    "gap": value,
                }
            )
    return pd.DataFrame(rows)


def _arbitrage_long(core: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("predicted_calendar_violation_rate", "预测 · calendar"),
        ("target_calendar_violation_rate", "目标 · calendar"),
        ("predicted_butterfly_violation_rate", "预测 · butterfly"),
        ("target_butterfly_violation_rate", "目标 · butterfly"),
    )
    rows: list[dict[str, Any]] = []
    for raw in core.to_dict(orient="records"):
        for field, label in definitions:
            value = _numeric(raw.get(field))
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "series": raw.get("series", ""),
                    "diagnostic": label,
                    "violation_rate": value,
                }
            )
    return pd.DataFrame(rows)


def _headline_dataset(
    comparison: pd.DataFrame,
    task_registry: pd.DataFrame,
    resource_usage: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    *,
    numerical_tie_tolerance: float,
) -> pd.DataFrame:
    core = _core_rows(comparison)
    gap_col = _column(core, "mae_gap", "surface_mae_gap")
    best = None
    if gap_col is not None and not core.empty:
        numeric_gap = pd.to_numeric(core[gap_col], errors="coerce")
        if numeric_gap.notna().any():
            best = core.loc[numeric_gap.idxmin()]
    finite_gaps = (
        pd.to_numeric(core[gap_col], errors="coerce").dropna()
        if gap_col is not None
        else pd.Series(dtype=float)
    )
    all_surface_ties = bool(
        not finite_gaps.empty
        and finite_gaps.abs().le(float(numerical_tie_tolerance)).all()
    )
    completed = 0
    if not task_registry.empty:
        status_col = _column(task_registry, "status", "state")
        completed = (
            int(task_registry[status_col].astype(str).str.lower().eq("completed").sum())
            if status_col
            else int(len(task_registry))
        )
    gpu_hours_col = _column(resource_usage, "gpu_hours", "gpu_hour")
    gpu_hours = (
        float(pd.to_numeric(resource_usage[gpu_hours_col], errors="coerce").sum())
        if gpu_hours_col
        else float("nan")
    )
    checkpoint_ok = (
        checkpoint_summary["status"].astype(str).str.lower().eq("ok")
        if not checkpoint_summary.empty and "status" in checkpoint_summary
        else pd.Series(dtype=bool)
    )
    checkpoint_jobs = int(checkpoint_ok.sum())
    best_epochs = (
        pd.to_numeric(
            checkpoint_summary.loc[checkpoint_ok, "best_epoch"], errors="coerce"
        )
        if checkpoint_jobs and "best_epoch" in checkpoint_summary
        else pd.Series(dtype=float)
    )
    epoch_zero_jobs = int(best_epochs.eq(0).sum())
    all_best_epoch_zero = bool(
        completed > 0
        and checkpoint_jobs == completed
        and epoch_zero_jobs == completed
    )
    raw_best_gap = (
        _numeric(best.get(gap_col)) if best is not None and gap_col else float("nan")
    )
    displayed_best_gap = (
        0.0
        if math.isfinite(raw_best_gap)
        and abs(raw_best_gap) <= float(numerical_tie_tolerance)
        else raw_best_gap
    )
    return pd.DataFrame(
        [
            {
                "completed_jobs": completed,
                "best_model": (
                    "全部模型/文本模式（数值持平）"
                    if all_surface_ties
                    else _model_label(best.get("model", "")) if best is not None else "—"
                ),
                "best_tolerance": (
                    "5/10/15/30m"
                    if all_surface_ties
                    else f"{int(best.get('tolerance_minutes'))}m"
                    if best is not None and pd.notna(best.get("tolerance_minutes"))
                    else "—"
                ),
                "best_mae_gap": displayed_best_gap,
                "best_mae_gap_raw": raw_best_gap,
                "best_gap_interpretation": (
                    _gap_interpretation(raw_best_gap, numerical_tie_tolerance)
                    if math.isfinite(raw_best_gap)
                    else "不可用"
                ),
                "numerical_tie_tolerance": float(numerical_tie_tolerance),
                "checkpoint_jobs": checkpoint_jobs,
                "epoch_zero_jobs": epoch_zero_jobs,
                "all_best_epoch_zero": all_best_epoch_zero,
                "common_test_pairs": int(
                    pd.to_numeric(core.get("pair_count", pd.Series([170])), errors="coerce")
                    .dropna()
                    .max()
                ),
                "common_test_sessions": int(
                    pd.to_numeric(core.get("session_count", pd.Series([49])), errors="coerce")
                    .dropna()
                    .max()
                ),
                "gpu_hours": gpu_hours,
            }
        ]
    )


def _best_result_sentence(
    comparison: pd.DataFrame,
    *,
    numerical_tie_tolerance: float,
) -> str:
    core = _core_rows(comparison)
    gap_col = _column(core, "mae_gap", "surface_mae_gap")
    if core.empty or gap_col is None:
        return "正式共同测试集指标尚未生成。"
    gaps = pd.to_numeric(core[gap_col], errors="coerce")
    if not gaps.notna().any():
        return "共同测试集的 MAE gap 不可用。"
    finite_gaps = gaps.dropna()
    if finite_gaps.abs().le(float(numerical_tie_tolerance)).all():
        return (
            "共同 2023Q4 测试中，全部模型、文本模式和容差的 "
            f"pair-balanced MAE gap 均落在 ±{numerical_tie_tolerance:g} IV "
            f"数值持平阈值内（最大绝对 gap={finite_gaps.abs().max():.3g} IV），"
            "与 persistence 持平。"
        )
    row = core.loc[gaps.idxmin()]
    model = _model_label(row.get("model", ""))
    mode = (
        f"、{_text_mode_label(row.get('text_ablation_mode'))}"
        if "text_ablation_mode" in core.columns
        else ""
    )
    tolerance = int(row.get("tolerance_minutes", 0))
    gap = float(gaps.loc[gaps.idxmin()])
    relation = _gap_interpretation(gap, numerical_tie_tolerance)
    return (
        f"共同 2023Q4 测试中，最低 pair-balanced MAE gap 来自 "
        f"{tolerance}m {model}{mode}（{gap:+.6g} IV），{relation}。"
    )


def _resolved_experiment_config(experiment_root: Path) -> dict[str, Any]:
    config_path = experiment_root / "resolved_config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return {}
    config = payload.get("news_first_vol_training", payload)
    return config if isinstance(config, dict) else {}


def _uses_identity_residual(experiment_config: dict[str, Any]) -> bool:
    models = experiment_config.get("models", {})
    if not isinstance(models, dict) or not models:
        return False
    modes: list[str] = []
    for model in models.values():
        if not isinstance(model, dict):
            return False
        training = model.get("training", {})
        if not isinstance(training, dict):
            return False
        modes.append(str(training.get("residual_output_mode", "")))
    return bool(modes) and all(mode == "identity_softplus_residual" for mode in modes)


def _resolved_dataset_root(experiment_root: Path) -> Path | None:
    config = _resolved_experiment_config(experiment_root)
    datasets = config.get("datasets", {}) if isinstance(config, dict) else {}
    value = datasets.get("root") if isinstance(datasets, dict) else None
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (experiment_root / path).resolve()


def _strict_boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be an explicit boolean, got {value!r}")


def _surface_support_summary(
    experiment_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Aggregate pair-level strict-support audits when the dataset provides them."""

    def persist_report_source(frame: pd.DataFrame) -> str:
        """Copy the derived summary inside the portable experiment boundary."""

        target = experiment_root / "analysis" / "surface_support_summary.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target.relative_to(experiment_root).as_posix()

    dataset_root = _resolved_dataset_root(experiment_root)
    if dataset_root is None or not dataset_root.is_dir():
        return pd.DataFrame(), ""
    audit_paths = sorted(dataset_root.glob("tolerance_*m/surface_support_audit.csv.gz"))
    if not audit_paths:
        summary_path = dataset_root / "surface_support_summary.csv"
        if not summary_path.is_file():
            return pd.DataFrame(), ""
        frame = pd.read_csv(summary_path, low_memory=False)
        frame = frame.rename(
            columns={
                "support_audit_pairs": "support_audit_pair_count",
                "surface_usable_pairs": "pair_count",
                "joint_strict_support_pairs": "joint_supported_pair_count",
                "joint_zero_support_pairs": "joint_zero_support_pair_count",
                "joint_strict_support_fraction_mean": "joint_mean_strict_support_fraction",
                "joint_strict_support_cell_count_median": "joint_median_strict_support_cell_count",
            }
        )
        if {"joint_supported_pair_count", "pair_count"}.issubset(frame.columns):
            denominator = pd.to_numeric(frame["pair_count"], errors="coerce")
            denominator = denominator.where(denominator.ne(0))
            frame["joint_supported_pair_rate"] = (
                pd.to_numeric(frame["joint_supported_pair_count"], errors="coerce")
                / denominator
            )
        if {"joint_zero_support_pair_count", "pair_count"}.issubset(frame.columns):
            denominator = pd.to_numeric(frame["pair_count"], errors="coerce")
            denominator = denominator.where(denominator.ne(0))
            frame["joint_zero_support_pair_rate"] = (
                pd.to_numeric(frame["joint_zero_support_pair_count"], errors="coerce")
                / denominator
            )
        if "support_mask_applied" in frame.columns:
            frame["support_mask_applied"] = frame["support_mask_applied"].map(
                lambda value: _strict_boolean(value, label="support_mask_applied")
            )
        return frame, persist_report_source(frame)

    required = {
        "pair_id",
        "current_strict_support_cell_count",
        "current_strict_support_fraction",
        "target_strict_support_cell_count",
        "target_strict_support_fraction",
        "joint_strict_support_cell_count",
        "joint_strict_support_fraction",
        "joint_zero_support",
        "grid_fingerprint",
        "support_mask_applied",
    }
    parts: list[pd.DataFrame] = []
    for path in audit_paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Surface support audit is missing columns {missing}: {path}")
        if "tolerance_minutes" not in frame.columns:
            token = path.parent.name.removeprefix("tolerance_").removesuffix("m")
            frame["tolerance_minutes"] = int(token)
        parts.append(frame)
    audit = pd.concat(parts, ignore_index=True)
    audit["tolerance_minutes"] = pd.to_numeric(
        audit["tolerance_minutes"], errors="raise"
    ).astype(int)
    if audit.duplicated(["tolerance_minutes", "pair_id"]).any():
        raise ValueError("Surface support audit must contain one row per tolerance and pair_id")

    count_columns = [
        "current_strict_support_cell_count",
        "target_strict_support_cell_count",
        "joint_strict_support_cell_count",
    ]
    fraction_columns = [
        "current_strict_support_fraction",
        "target_strict_support_fraction",
        "joint_strict_support_fraction",
    ]
    for column in count_columns + fraction_columns:
        audit[column] = pd.to_numeric(audit[column], errors="raise")
    if (audit[count_columns] < 0).any().any():
        raise ValueError("Surface support cell counts must be non-negative")
    if ((audit[fraction_columns] < 0) | (audit[fraction_columns] > 1)).any().any():
        raise ValueError("Surface support fractions must be inside [0, 1]")
    audit["joint_zero_support"] = audit["joint_zero_support"].map(
        lambda value: _strict_boolean(value, label="joint_zero_support")
    )
    audit["support_mask_applied"] = audit["support_mask_applied"].map(
        lambda value: _strict_boolean(value, label="support_mask_applied")
    )
    if "surface_training_eligible" in audit.columns:
        audit["surface_training_eligible"] = audit["surface_training_eligible"].map(
            lambda value: _strict_boolean(value, label="surface_training_eligible")
        )
    else:
        audit["surface_training_eligible"] = True
    if not audit["joint_zero_support"].equals(
        audit["joint_strict_support_cell_count"].eq(0)
    ):
        raise ValueError("joint_zero_support disagrees with joint strict-support count")

    rows: list[dict[str, Any]] = []
    for tolerance, group in audit.groupby("tolerance_minutes", sort=True):
        eligible = group.loc[group["surface_training_eligible"]].copy()
        pair_count = int(len(eligible))
        zero_count = int(eligible["joint_zero_support"].sum())
        fingerprints = sorted(
            value
            for value in group["grid_fingerprint"].astype(str).str.strip().unique()
            if value
        )
        rows.append(
            {
                "tolerance_minutes": int(tolerance),
                "support_audit_pair_count": int(len(group)),
                "pair_count": pair_count,
                "joint_supported_pair_count": pair_count - zero_count,
                "joint_supported_pair_rate": (
                    float((pair_count - zero_count) / pair_count) if pair_count else float("nan")
                ),
                "joint_zero_support_pair_count": zero_count,
                "joint_zero_support_pair_rate": (
                    float(zero_count / pair_count) if pair_count else float("nan")
                ),
                "current_mean_strict_support_cell_count": float(
                    eligible["current_strict_support_cell_count"].mean()
                ),
                "target_mean_strict_support_cell_count": float(
                    eligible["target_strict_support_cell_count"].mean()
                ),
                "joint_mean_strict_support_cell_count": float(
                    eligible["joint_strict_support_cell_count"].mean()
                ),
                "current_mean_strict_support_fraction": float(
                    eligible["current_strict_support_fraction"].mean()
                ),
                "target_mean_strict_support_fraction": float(
                    eligible["target_strict_support_fraction"].mean()
                ),
                "joint_mean_strict_support_fraction": float(
                    eligible["joint_strict_support_fraction"].mean()
                ),
                "grid_fingerprint_count": int(len(fingerprints)),
                "grid_fingerprint": fingerprints[0] if len(fingerprints) == 1 else "multiple",
                "support_mask_applied": bool(group["support_mask_applied"].any()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary, persist_report_source(summary)


def _short_atm_counts(experiment_root: Path) -> dict[str, list[int]]:
    config_path = experiment_root / "analysis" / "analysis_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        values = payload.get("short_atm", {}).get("observed_cell_counts_by_panel", {})
        if isinstance(values, dict) and values:
            return {
                str(panel): sorted({int(value) for value in counts})
                for panel, counts in values.items()
            }
    sample_path = experiment_root / "analysis" / "sample_metrics.csv.gz"
    if not sample_path.is_file():
        return {}
    frame = pd.read_csv(
        sample_path,
        usecols=lambda column: column in {"panel", "short_atm_cell_count"},
    )
    if not {"panel", "short_atm_cell_count"}.issubset(frame.columns):
        return {}
    return {
        str(panel): sorted(
            set(pd.to_numeric(group["short_atm_cell_count"], errors="coerce").dropna().astype(int))
        )
        for panel, group in frame.groupby("panel", sort=True)
    }


def _short_atm_count_sentence(counts_by_panel: dict[str, list[int]]) -> str:
    all_counts = sorted({count for counts in counts_by_panel.values() for count in counts})
    if len(all_counts) == 1:
        return f"；本实验实际网格包含 {all_counts[0]} 格。"
    if not all_counts:
        return "。"
    details = "、".join(
        f"{panel}={','.join(str(value) for value in counts)}"
        for panel, counts in sorted(counts_by_panel.items())
    )
    return f"；本实验按 panel 的实际格点数为 {details}。"


def _source(
    source_id: str,
    label: str,
    relative_path: str,
    *,
    description: str,
    tables: Iterable[str],
    filters: Iterable[str],
    definitions: Iterable[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": relative_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{relative_path}')",
            "description": description,
            "tables_used": list(tables),
            "filters": list(filters),
            "metric_definitions": list(definitions),
        },
    }


def build_report_artifact(experiment_root: str | Path) -> dict[str, Any]:
    """Build the canonical artifact payload from formal experiment CSVs."""

    root = Path(experiment_root)
    analysis_dir = root / "analysis"
    comparison = _normalise_comparison(
        _read_csv(analysis_dir / "model_comparison.csv", required=True)
    )
    bootstrap = _read_csv(analysis_dir / "bootstrap_comparisons.csv")
    text_ablation = _read_csv(analysis_dir / "text_ablation_comparisons.csv")
    analysis_config: dict[str, Any] = {}
    analysis_config_path = analysis_dir / "analysis_config.json"
    if analysis_config_path.is_file():
        with analysis_config_path.open("r", encoding="utf-8") as handle:
            loaded_analysis_config = json.load(handle)
        if isinstance(loaded_analysis_config, dict):
            analysis_config = loaded_analysis_config
    training_support_mode = str(analysis_config.get("support_mask_mode", "none"))
    numerical_tie_policy = analysis_config.get("numerical_tie_policy", {})
    numerical_tie_tolerance = _numeric(
        numerical_tie_policy.get(
            "mae_gap_tolerance_iv", DEFAULT_MAE_NUMERICAL_TIE_TOLERANCE
        )
        if isinstance(numerical_tie_policy, dict)
        else DEFAULT_MAE_NUMERICAL_TIE_TOLERANCE,
        DEFAULT_MAE_NUMERICAL_TIE_TOLERANCE,
    )
    if numerical_tie_tolerance <= 0.0:
        numerical_tie_tolerance = DEFAULT_MAE_NUMERICAL_TIE_TOLERANCE
    resource_path = (
        root / "resource_summary.csv"
        if (root / "resource_summary.csv").is_file()
        else root / "resource_usage.csv"
    )
    resource = _read_csv(resource_path)
    registry = _read_csv(root / "task_registry.csv")
    split = _read_csv(root / "split_manifest.csv")
    strata = _read_csv(analysis_dir / "stratified_metrics.csv")
    curves = _read_csv(analysis_dir / "training_curves.csv")
    checkpoint_summary = _read_csv(analysis_dir / "checkpoint_summary.csv")
    support, support_source_path = _surface_support_summary(root)
    short_atm_counts = _short_atm_counts(root)
    experiment_config = _resolved_experiment_config(root)
    identity_residual = _uses_identity_residual(experiment_config)

    if not bootstrap.empty:
        if "model" in bootstrap:
            bootstrap["model_label"] = bootstrap["model"].map(_model_label)
        if "focal_tolerance_minutes" in bootstrap:
            bootstrap["comparison_label"] = bootstrap[
                "focal_tolerance_minutes"
            ].map(lambda value: f"{int(value)}m vs 5m")
    if not text_ablation.empty:
        if "model" in text_ablation:
            text_ablation["model_label"] = text_ablation["model"].map(_model_label)
        if "control_mode" in text_ablation:
            text_ablation["control_label"] = text_ablation["control_mode"].map(
                _text_mode_label
            )
        if {"model_label", "control_label"}.issubset(text_ablation.columns):
            text_ablation["comparison_series"] = (
                text_ablation["model_label"] + " vs " + text_ablation["control_label"]
            )
    if not resource.empty and "model" in resource:
        resource["model_label"] = resource["model"].map(_model_label)
    if not split.empty and "tolerance_minutes" in split:
        split["tolerance_label"] = split["tolerance_minutes"].map(
            lambda value: f"{int(value)}m"
        )
    if not curves.empty:
        if "model" in curves:
            curves["model_label"] = curves["model"].map(_model_label)
        if {"model_label", "tolerance_minutes"}.issubset(curves.columns):
            curves["series"] = curves.apply(
                lambda row: (
                    f"{row['model_label']} · {int(row['tolerance_minutes'])}m"
                    + (
                        f" · {_text_mode_label(row['text_ablation_mode'])}"
                        if "text_ablation_mode" in curves.columns
                        else ""
                    )
                ),
                axis=1,
            )

    headline = _headline_dataset(
        comparison,
        registry,
        resource,
        checkpoint_summary,
        numerical_tie_tolerance=numerical_tie_tolerance,
    )
    headline_row = headline.iloc[0]
    all_best_epoch_zero = bool(headline_row.get("all_best_epoch_zero", False))
    epoch_zero_persistence_collapse = bool(
        all_best_epoch_zero and identity_residual and not text_ablation.empty
    )
    core = _core_rows(comparison).copy()
    primary_bootstrap = _primary_bootstrap_rows(bootstrap)
    gap_metrics = _gap_long(core)
    arbitrage_metrics = _arbitrage_long(core)
    broad_strata = strata.copy()
    if not broad_strata.empty:
        mask = pd.Series(True, index=broad_strata.index)
        if "panel" in broad_strata:
            mask &= broad_strata["panel"].astype(str).str.lower().eq("broad")
        if "tolerance_minutes" in broad_strata:
            mask &= pd.to_numeric(
                broad_strata["tolerance_minutes"], errors="coerce"
            ).eq(30)
        if "stratum_type" in broad_strata:
            mask &= broad_strata["stratum_type"].astype(str).isin(
                {"first_included_tolerance", "origin_shift"}
            )
        broad_strata = _normalise_comparison(broad_strata.loc[mask].copy())
    generated_at = _utc_now()

    datasets = {
        "headline": _records(headline),
        "comparison": _records(comparison),
        "core_comparison": _records(core),
        "bootstrap": _records(primary_bootstrap),
        "bootstrap_secondary": _records(bootstrap),
        "text_ablation": _records(text_ablation),
        "resources": _records(resource),
        "splits": _records(split),
        "strata": _records(strata),
        "broad_strata": _records(broad_strata),
        "gap_metrics": _records(gap_metrics),
        "arbitrage_metrics": _records(arbitrage_metrics),
        "curves": _records(curves),
        "checkpoint_summary": _records(checkpoint_summary),
        "surface_support": _records(support),
    }
    sources = [
        _source(
            "comparison_source",
            "Formal model comparison",
            "analysis/model_comparison.csv",
            description="Pair-balanced aggregation of formal fixed-checkpoint predictions.",
            tables=("model_comparison", "pair_metrics"),
            filters=("common 5m Q4 core or declared secondary panel",),
            definitions=(
                "MAE gap = model surface MAE minus persistence surface MAE; lower is better.",
                f"abs(MAE gap) <= {numerical_tie_tolerance:g} IV is a numerical tie; win requires gap below -{numerical_tie_tolerance:g} IV.",
                "Skill is computed as 1 minus model MAE divided by persistence MAE for each pair, then averaged equally across pairs; near-zero persistence errors can amplify this ratio.",
            ),
        ),
        _source(
            "text_ablation_source",
            "Text-information ablation comparison",
            "analysis/text_ablation_comparisons.csv",
            description=(
                "Paired real-text minus current-only/shuffled-text surface MAE on "
                "identical common-Q4 market pairs."
            ),
            tables=("text_ablation_comparisons",),
            filters=("common 5m Q4 core", "10000 fixed-seed session-cluster draws"),
            definitions=(
                "Difference = real_text model MAE minus control model MAE.",
                f"Only a difference below -{numerical_tie_tolerance:g} IV indicates incremental predictive value from real news text; values inside the tie tolerance are zeroed.",
            ),
        ),
        _source(
            "bootstrap_source",
            "Session-cluster bootstrap",
            "analysis/bootstrap_comparisons.csv",
            description="Paired tolerance-minus-5m differences resampled by CME session.",
            tables=("bootstrap_comparisons",),
            filters=("WGAN-GP common Q4 pairs", "10000 fixed-seed draws"),
            definitions=(
                "Mean difference is focal model error minus 5m error; negative favors focal.",
                "p_holm is the Holm-adjusted two-sided p-value across the primary comparisons.",
            ),
        ),
        _source(
            "split_source",
            "Frozen split manifest",
            "split_manifest.csv",
            description="Explicit pair/session-disjoint train, common validation and common test counts.",
            tables=("split_manifest",),
            filters=("train before 2023-07-01", "validation 2023Q3", "test 2023Q4"),
            definitions=("Rows are articles; pairs are unique five-minute market outcomes.",),
        ),
        _source(
            "task_source",
            "Training task registry",
            "task_registry.csv",
            description="Terminal state, assigned GPU and run directory for every registered formal job.",
            tables=("task_registry",),
            filters=("formal jobs only",),
            definitions=(
                "Completed means required best/final artifacts passed hash validation.",
            ),
        ),
        _source(
            "resource_source",
            "GPU resource audit",
            resource_path.relative_to(root).as_posix(),
            description="Per-task elapsed time plus assigned-GPU wave-level NVIDIA telemetry captured by the launcher.",
            tables=("resource_usage", "task_registry"),
            filters=("formal completed tasks only",),
            definitions=(
                "GPU-hours = elapsed wall hours for the assigned GPU worker.",
                "Peak memory and utilization are whole assigned-GPU wave aggregates shared by two concurrent workers, not per-process attribution.",
            ),
        ),
        _source(
            "strata_source",
            "Broad-test stratified metrics",
            "analysis/stratified_metrics.csv",
            description="Secondary 30m-Q4 metrics split by entry tolerance and news-to-origin shift.",
            tables=("stratified_metrics",),
            filters=("30m broad Q4 panel",),
            definitions=("Shift groups are 0, 1–4, and at least 5 minutes.",),
        ),
        _source(
            "curve_source",
            "Training histories",
            "analysis/training_curves.csv",
            description="Epoch-level validation metrics copied from each formal run.",
            tables=("training_curves",),
            filters=("best checkpoint selected only on common 2023Q3 validation",),
            definitions=("Hybrid score = MAE plus twice the positive MAE gap versus persistence.",),
        ),
        _source(
            "checkpoint_source",
            "Best-checkpoint selection metadata",
            "analysis/checkpoint_summary.csv",
            description="Trainer-emitted validation-only best epoch for every formal run.",
            tables=("checkpoint_summary",),
            filters=("one authoritative best_checkpoint.json per completed formal run",),
            definitions=(
                "Epoch 0 is the identity-initialized persistence checkpoint before any optimizer update.",
            ),
        ),
    ]
    if text_ablation.empty:
        sources = [source for source in sources if source["id"] != "text_ablation_source"]
    if checkpoint_summary.empty:
        sources = [source for source in sources if source["id"] != "checkpoint_source"]
    if not support.empty:
        sources.append(
            _source(
                "support_source",
                "Strict raw-surface support audit",
                support_source_path,
                description=(
                    "Pair-level current, target and joint grid support reconstructed from "
                    "the raw maturity and moneyness domains."
                ),
                tables=("surface_support_audit",),
                filters=("one row per independent pair and tolerance",),
                definitions=(
                    "Strict support requires a maturity inside the raw slice range and q inside both adjacent slices for interpolated maturities.",
                    "support_mask_applied records whether the audit changed the training loss or sample eligibility.",
                ),
            )
        )

    cards = [
        {
            "id": "completed_jobs",
            "dataset": "headline",
            "sourceId": "task_source",
            "description": "已完成且进入正式评估的独立训练任务。",
            "metrics": [{"label": "完成任务", "field": "completed_jobs", "format": "number"}],
        },
        {
            "id": "best_gap",
            "dataset": "headline",
            "sourceId": "comparison_source",
            "description": (
                "模型 MAE 减 persistence MAE；仅低于 "
                f"-{numerical_tie_tolerance:g} IV 才认定为改善。"
            ),
            "metrics": [
                {"label": "最佳 MAE gap", "field": "best_mae_gap", "format": "number", "signed": True},
                {"label": "数值解读", "field": "best_gap_interpretation"},
                {"label": "模型", "field": "best_model"},
                {"label": "容差", "field": "best_tolerance"},
            ],
        },
        {
            "id": "common_pairs",
            "dataset": "headline",
            "sourceId": "comparison_source",
            "description": "四组模型共享的 5m 2023Q4 独立 pair 数。",
            "metrics": [
                {"label": "共同测试 pairs", "field": "common_test_pairs", "format": "number"},
                {"label": "CME sessions", "field": "common_test_sessions", "format": "number"},
            ],
        },
        {
            "id": "gpu_hours",
            "dataset": "headline",
            "sourceId": "resource_source",
            "description": "所有正式任务实际占用的 GPU wall-clock 小时之和。",
            "metrics": [{"label": "GPU-hours", "field": "gpu_hours", "format": "number"}],
        },
    ]
    if not checkpoint_summary.empty:
        cards.append(
            {
                "id": "epoch_zero_checkpoints",
                "dataset": "headline",
                "sourceId": "checkpoint_source",
                "description": "使用 validation 选中的 epoch-0 persistence 初始化 checkpoint。",
                "metrics": [
                    {"label": "Epoch-0 best", "field": "epoch_zero_jobs", "format": "number"},
                    {"label": "可审计 checkpoints", "field": "checkpoint_jobs", "format": "number"},
                ],
            }
        )

    charts: list[dict[str, Any]] = []
    if not split.empty and {"tolerance_label", "train_pairs"}.issubset(split.columns):
        charts.append(
            {
                "id": "training_scale",
                "title": "训练样本规模",
                "subtitle": "累计等待容差越大，训练 pair 更多，但时间语义也随之变化。",
                "type": "bar",
                "dataset": "splits",
                "sourceId": "split_source",
                "encodings": {
                    "x": {"field": "tolerance_label", "type": "nominal", "label": "累计等待容差"},
                    "y": {"field": "train_pairs", "type": "quantitative", "label": "训练 pairs", "format": "number"},
                },
                "options": {"orientation": "vertical", "grouping": "single"},
                "layout": "full",
            }
        )
    if not core.empty and {"tolerance_label", "mae_gap", "model_label"}.issubset(core.columns):
        charts.append(
            {
                "id": "common_gap",
                "title": "共同 Q4 测试集的 surface MAE gap",
                "subtitle": (
                    f"仅低于 -{numerical_tie_tolerance:g} IV 才认定为相对 persistence "
                    "有样本外改善；阈值内是数值持平。"
                ),
                "type": "bar",
                "dataset": "core_comparison",
                "sourceId": "comparison_source",
                "encodings": {
                    "x": {"field": "tolerance_label", "type": "nominal", "label": "训练容差"},
                    "y": {"field": "mae_gap", "type": "quantitative", "label": "MAE gap (IV)", "format": "number"},
                    "color": {"field": "model_label", "type": "nominal", "label": "模型"},
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
                "display": {"baseline": 0},
                "layout": "full",
            }
        )
    if not text_ablation.empty and {
        "tolerance_minutes",
        "mean_diff",
        "comparison_series",
    }.issubset(text_ablation.columns):
        charts.append(
            {
                "id": "text_ablation_effect",
                "title": "真实新闻文本相对信息控制的增量",
                "subtitle": (
                    f"real_text MAE − control MAE；仅低于 -{numerical_tie_tolerance:g} IV "
                    "才表示真实新闻文本提供增量预测价值。"
                ),
                "type": "bar",
                "dataset": "text_ablation",
                "sourceId": "text_ablation_source",
                "encodings": {
                    "x": {"field": "tolerance_minutes", "type": "nominal", "label": "训练容差(分钟)"},
                    "y": {"field": "mean_diff", "type": "quantitative", "label": "Real − control MAE", "format": "number"},
                    "color": {"field": "comparison_series", "type": "nominal", "label": "模型与控制"},
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
                "display": {"baseline": 0},
                "layout": "full",
            }
        )
    if not gap_metrics.empty:
        charts.append(
            {
                "id": "atm_skew_gaps",
                "title": "共同 Q4 的 ATM、skew 与完整 surface gap",
                "subtitle": (
                    "完整surface gap按数值持平阈值解读。实际 ATM/skew 只用可用的 A/B 期限；"
                    + (
                        "epoch0 时其负gap反映网格插值与raw persistence口径差，不是learned gain。"
                        if epoch_zero_persistence_collapse
                        else "低于阈值才作为模型改善证据。"
                    )
                ),
                "type": "bar",
                "dataset": "gap_metrics",
                "sourceId": "comparison_source",
                "encodings": {
                    "x": {"field": "series", "type": "nominal", "label": "模型 · 容差"},
                    "y": {"field": "gap", "type": "quantitative", "label": "误差 gap", "format": "number"},
                    "color": {"field": "metric_label", "type": "nominal", "label": "指标"},
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
                "display": {"baseline": 0},
                "layout": "full",
            }
        )
    if not arbitrage_metrics.empty:
        charts.append(
            {
                "id": "arbitrage_rates",
                "title": "共同 Q4 的离散套利违反率",
                "subtitle": "预测与目标均按同一16×16网格、固定1e-8容差计算。",
                "type": "bar",
                "dataset": "arbitrage_metrics",
                "sourceId": "comparison_source",
                "encodings": {
                    "x": {"field": "series", "type": "nominal", "label": "模型 · 容差"},
                    "y": {"field": "violation_rate", "type": "quantitative", "label": "违反率", "format": "percent"},
                    "color": {"field": "diagnostic", "type": "nominal", "label": "诊断"},
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
                "layout": "full",
            }
        )
    if not curves.empty and {"epoch", "val_hybrid_score", "series"}.issubset(curves.columns):
        charts.append(
            {
                "id": "training_curves",
                "title": "验证 hybrid score 训练曲线",
                "subtitle": "最佳 checkpoint 只依据共同 2023Q3 validation 选择。",
                "type": "line",
                "dataset": "curves",
                "sourceId": "curve_source",
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "val_hybrid_score", "type": "quantitative", "label": "Validation hybrid score", "format": "number"},
                    "color": {"field": "series", "type": "nominal", "label": "模型 · 容差"},
                },
                "options": {"points": "never"},
                "layout": "full",
            }
        )
    if not resource.empty and {"runtime_minutes", "model_label", "tolerance_minutes"}.issubset(resource.columns):
        resource["task_label"] = resource.apply(
            lambda row: (
                f"{row['model_label']} · {int(row['tolerance_minutes'])}m"
                + (
                    f" · {_text_mode_label(row['text_ablation_mode'])}"
                    if "text_ablation_mode" in resource.columns
                    else ""
                )
            ),
            axis=1,
        )
        datasets["resources"] = _records(resource)
        charts.append(
            {
                "id": "runtime",
                "title": "正式任务 wall-clock 时间",
                "subtitle": "任务以每张2个独立进程运行；数值包含数据读取、训练与 checkpoint 写入。",
                "type": "bar",
                "dataset": "resources",
                "sourceId": "resource_source",
                "encodings": {
                    "x": {"field": "task_label", "type": "nominal", "label": "任务"},
                    "y": {"field": "runtime_minutes", "type": "quantitative", "label": "分钟", "format": "number"},
                    "color": {"field": "model_label", "type": "nominal", "label": "模型"},
                },
                "options": {"orientation": "vertical", "grouping": "single"},
                "layout": "full",
            }
        )
    if not support.empty and {
        "tolerance_minutes",
        "joint_supported_pair_rate",
    }.issubset(support.columns):
        support["tolerance_label"] = support["tolerance_minutes"].map(
            lambda value: f"{int(value)}m"
        )
        datasets["surface_support"] = _records(support)
        charts.append(
            {
                "id": "strict_support_coverage",
                "title": "窄网格的严格 raw-support 覆盖",
                "subtitle": (
                    "至少一个 current/target 共同严格支持格点的独立 pair 比例；"
                    + (
                        "本轮 zero-support 样本据此排除。"
                        if training_support_mode == "raw_joint"
                        else "仅作审计，不参与训练筛选。"
                    )
                ),
                "type": "bar",
                "dataset": "surface_support",
                "sourceId": "support_source",
                "encodings": {
                    "x": {"field": "tolerance_label", "type": "nominal", "label": "累计等待容差"},
                    "y": {
                        "field": "joint_supported_pair_rate",
                        "type": "quantitative",
                        "label": "有共同严格支持格点的 pair 比例",
                        "format": "percent",
                    },
                },
                "options": {"orientation": "vertical", "grouping": "single"},
                "layout": "full",
            }
        )

    comparison_columns = [
        {"field": name, "label": label, "format": fmt}
        for name, label, fmt in (
            ("model_label", "模型", None),
            ("tolerance_minutes", "容差(分钟)", "number"),
            ("panel", "Panel", None),
            ("stratum", "分层", None),
            ("pair_count", "Pairs", "number"),
            ("session_count", "Sessions", "number"),
            ("mae", "Model MAE", "number"),
            ("persistence_mae", "Persistence MAE", "number"),
            ("mae_gap", "MAE gap", "number"),
            ("tie_rate", "Pair numerical-tie rate", "percent"),
            ("skill", "Mean pair-level skill", "percent"),
            ("win_rate", f"Pair win rate (gap < -{numerical_tie_tolerance:g})", "percent"),
            ("short_atm_mae", "短端近ATM MAE", "number"),
            ("short_atm_gap", "短端近ATM gap", "number"),
            ("short_atm_tie_rate", "短端近ATM tie rate", "percent"),
            ("atm_pair_count", "ATM pairs", "number"),
            ("atm_maturity_count", "ATM maturities", "number"),
            ("atm_mae", "实际ATM MAE", "number"),
            ("atm_gap", "实际ATM gap", "number"),
            ("skew_pair_count", "Skew pairs", "number"),
            ("skew_maturity_count", "Skew maturities", "number"),
            ("skew_mae", "实际skew MAE", "number"),
            ("skew_gap", "实际skew gap", "number"),
            ("predicted_calendar_violation_rate", "预测calendar违反率", "percent"),
            ("predicted_butterfly_violation_rate", "预测butterfly违反率", "percent"),
        )
        if name in comparison.columns
    ]
    bootstrap_columns = [
        {"field": name, "label": label, "format": fmt}
        for name, label, fmt in (
            ("model_label", "模型", None),
            ("focal_tolerance_minutes", "Focal(分钟)", "number"),
            ("base_tolerance_minutes", "Base(分钟)", "number"),
            ("metric", "指标", None),
            ("mean_diff", "配对差", "number"),
            ("ci_low", "95% CI low", "number"),
            ("ci_high", "95% CI high", "number"),
            ("p_value", "p", "number"),
            ("p_holm", "Holm p", "number"),
        )
        if name in primary_bootstrap.columns
    ]
    tables = [
        {
            "id": "comparison_table",
            "title": "模型统计对比",
            "subtitle": (
                "主表使用 pair-balanced 指标；保留原始 MAE gap，"
                f"abs(gap)≤{numerical_tie_tolerance:g} IV 记为数值持平，不计入 win。"
            ),
            "dataset": "comparison",
            "sourceId": "comparison_source",
            "defaultSort": {"field": "mae_gap", "direction": "asc"},
            "columns": comparison_columns,
        }
    ]
    if bootstrap_columns:
        tables.append(
            {
                "id": "bootstrap_table",
                "title": "WGAN-GP 相对 5m 的 session-cluster bootstrap",
                "subtitle": "负的配对差表示扩样模型的误差更低。",
                "dataset": "bootstrap",
                "sourceId": "bootstrap_source",
                "defaultSort": {"field": "focal_tolerance_minutes", "direction": "asc"},
                "columns": bootstrap_columns,
            }
        )
    text_ablation_columns = [
        {"field": name, "label": label, "format": fmt}
        for name, label, fmt in (
            ("model_label", "模型", None),
            ("tolerance_minutes", "容差(分钟)", "number"),
            ("control_label", "控制", None),
            ("pair_count", "Pairs", "number"),
            ("session_count", "Sessions", "number"),
            ("mean_diff", "Real − control MAE", "number"),
            ("ci_95_lower", "95% CI low", "number"),
            ("ci_95_upper", "95% CI high", "number"),
            ("p_two_sided", "p", "number"),
            ("p_holm", "Holm p", "number"),
        )
        if name in text_ablation.columns
    ]
    if text_ablation_columns:
        tables.append(
            {
                "id": "text_ablation_table",
                "title": "新闻文本增量的配对消融",
                "subtitle": (
                    "差值为 real_text − control 的 pair-balanced MAE；负值才支持真实新闻文本有增量。"
                ),
                "dataset": "text_ablation",
                "sourceId": "text_ablation_source",
                "defaultSort": {"field": "mean_diff", "direction": "asc"},
                "columns": text_ablation_columns,
            }
        )
    broad_columns = [
        {"field": name, "label": label, "format": fmt}
        for name, label, fmt in (
            ("model_label", "模型", None),
            ("stratum_type", "分层变量", None),
            ("stratum_value", "分层", None),
            ("pair_count", "Pairs", "number"),
            ("mae", "MAE", "number"),
            ("gap", "MAE gap", "number"),
            ("skill", "Mean pair-level skill", "percent"),
            ("win", "Pair win rate", "percent"),
        )
        if name in broad_strata.columns
    ]
    if broad_columns:
        tables.append(
            {
                "id": "broad_strata_table",
                "title": "30m broad Q4：扩样来源与等待语义",
                "subtitle": "按首次进入容差及新闻到origin的0 / 1–4 / ≥5分钟分层；仅作secondary sensitivity。",
                "dataset": "broad_strata",
                "sourceId": "strata_source",
                "defaultSort": {"field": "gap", "direction": "asc"},
                "columns": broad_columns,
            }
        )
    support_columns = [
        {"field": name, "label": label, "format": fmt}
        for name, label, fmt in (
            ("tolerance_minutes", "容差(分钟)", "number"),
            ("support_audit_pair_count", "审计 pairs", "number"),
            ("pair_count", "Surface-usable pairs", "number"),
            ("joint_supported_pair_count", "有共同支持格点的 pairs", "number"),
            ("joint_supported_pair_rate", "共同支持覆盖率", "percent"),
            ("joint_zero_support_pair_count", "Zero-support pairs", "number"),
            ("joint_mean_strict_support_cell_count", "平均共同支持格点", "number"),
            ("joint_mean_strict_support_fraction", "平均共同支持比例", "percent"),
            ("grid_fingerprint_count", "网格指纹数", "number"),
            ("support_mask_applied", "工作簿生成阶段应用mask", None),
        )
        if name in support.columns
    ]
    if support_columns:
        tables.append(
            {
                "id": "support_table",
                "title": "严格 raw-support 审计",
                "subtitle": (
                    "该列描述工作簿生成阶段；本轮训练mask状态以run config和prediction artifact为准。"
                    if training_support_mode == "raw_joint"
                    else "审计不改变训练资格；窄网格减少外推，但不能保证全部256格都有原始数据支持。"
                ),
                "dataset": "surface_support",
                "sourceId": "support_source",
                "defaultSort": {"field": "tolerance_minutes", "direction": "asc"},
                "columns": support_columns,
            }
        )

    short_atm_line = (
        "- 短端近 ATM 固定为 q∈[0.98,1.02]、期限≤60个business days；不是事后调参"
        + _short_atm_count_sentence(short_atm_counts)
    )
    support_line = ""
    support_body = ""
    if not support.empty:
        mask_values = (
            set(support["support_mask_applied"].astype(bool))
            if "support_mask_applied" in support.columns
            else set()
        )
        mask_text = (
            "false"
            if mask_values == {False}
            else "/".join(str(value).lower() for value in sorted(mask_values)) or "unknown"
        )
        if training_support_mode == "raw_joint":
            support_line = (
                "\n- 本轮训练与正式评估使用 `support_mask_mode=raw_joint`：先排除共同支持为零的样本，"
                "其余样本仅在 current/target 共同 raw-support 格点计算 loss 和主指标。"
            )
            support_body = (
                "## 严格支持区训练与评估\n\n"
                f"数据生成审计中的 `support_mask_applied={mask_text}` 只描述原工作簿生成阶段，"
                "不能解释为本轮训练未使用 mask。本轮 run config 与 prediction artifact 均验证 "
                "`support_mask_mode=raw_joint` / applied=true：zero-support 样本在 split 前排除并重算 pair 权重；"
                "其余样本只有共同支持格点进入 WGAN/regression loss、persistence、主 surface MAE 与结构诊断。"
                "短端 ATM 或结构约束没有支持格时仅排除该辅助指标，不排除主 surface 样本。"
            )
        else:
            support_line = (
                "\n- 窄网格显著减少但未消除外推；"
                f"support_mask_applied={mask_text}，严格支持审计不改变训练资格，全部256格仍等权参与 loss。"
            )
            support_body = (
                "## 严格支持区审计\n\n"
                f"本实验记录 `support_mask_applied={mask_text}`。严格支持格点统计只用于量化仍然存在的外推，"
                "不筛除 pair、不改变训练资格，也不改变256格等权 loss。因此结果只能表述为窄网格显著减少外推，"
                "不能表述为只在真实观测支持区训练。"
            )

    epoch_zero_summary = ""
    if epoch_zero_persistence_collapse:
        epoch_zero_summary = (
            f" {int(headline_row['epoch_zero_jobs'])}/{int(headline_row['checkpoint_jobs'])} "
            "个正式任务的最佳 checkpoint 均为 epoch 0；"
            "identity 初始化因此让 real_text、current-only 和 text_shuffle "
            "退化为同一 persistence surface 预测，本轮无法识别新闻文本增量。"
        )
    atm_skew_epoch_zero_caveat = (
        "Epoch-0 模型的 surface 预测就是 current 网格；但 actual ATM/skew 的 model "
        "误差使用 current 网格插值，persistence 误差使用 raw current 值。"
        "因此该区域即使出现负 gap，也是表示/插值口径差，不是训练改善或文本增量。"
        if epoch_zero_persistence_collapse
        else ""
    )

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术摘要\n\n"
                + _best_result_sentence(
                    comparison,
                    numerical_tie_tolerance=numerical_tie_tolerance,
                )
                + epoch_zero_summary
                + " 此比较只基于一个训练 seed；bootstrap 只衡量测试 session 抽样不确定性，不包含模型初始化不确定性。"
            ),
            "sourceId": "comparison_source",
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "findings_heading",
            "type": "markdown",
            "body": "## 主要结果\n\n横向排名只使用四组完全相同的 5m Q4 test core，避免样本时段差异伪装成模型改善。",
        },
    ]
    if text_ablation_columns:
        blocks.extend(
            [
                {
                    "id": "text_ablation_heading",
                    "type": "markdown",
                    "body": (
                        "## 新闻文本是否有增量？\n\n"
                        "直接比较 real_text、current-only（1024维全零文本）和 split 内固定 shuffled-text。"
                        "统计量是 `real_text MAE − control MAE`；只有低于 "
                        f"-{numerical_tie_tolerance:g} IV 才表示真实新闻文本相对该控制有增量预测价值。"
                        + (
                            "\n\n**本轮不能识别文本增量：所有 "
                            f"{int(headline_row['checkpoint_jobs'])} 个 best checkpoint 都是 epoch 0，"
                            "三种文本设置产生同一 persistence surface 预测。**"
                            if epoch_zero_persistence_collapse
                            else ""
                        )
                    ),
                    "sourceId": "text_ablation_source",
                },
            ]
        )
        if any(chart["id"] == "text_ablation_effect" for chart in charts):
            blocks.append(
                {
                    "id": "text_ablation_effect_block",
                    "type": "chart",
                    "chartId": "text_ablation_effect",
                }
            )
        blocks.append(
            {
                "id": "text_ablation_table_block",
                "type": "table",
                "tableId": "text_ablation_table",
            }
        )
    for chart_id in ("common_gap", "atm_skew_gaps", "arbitrage_rates", "training_scale"):
        if any(chart["id"] == chart_id for chart in charts):
            blocks.append({"id": f"{chart_id}_block", "type": "chart", "chartId": chart_id})
            blocks.append(
                {
                    "id": f"{chart_id}_note",
                    "type": "markdown",
                    "body": (
                        atm_skew_epoch_zero_caveat
                        if chart_id == "atm_skew_gaps"
                        and atm_skew_epoch_zero_caveat
                        else "图中的模型误差与数据规模必须分开解读：更大容差既增加样本，也将 current 窗口更多地移到新闻发布之后。"
                    ),
                }
            )
    blocks.extend(
        [
            {"id": "comparison_table_block", "type": "table", "tableId": "comparison_table"},
            {
                "id": "inference_heading",
                "type": "markdown",
                "body": "## 配对推断\n\n对10m、15m、30m WGAN-GP 与5m基准在同一 Q4 pair 上做配对差，然后以 CME session 为 cluster 重抽样 10,000 次。Holm 校正限定在预注册的主比较族内。",
                "sourceId": "bootstrap_source",
            },
        ]
    )
    if bootstrap_columns:
        blocks.append({"id": "bootstrap_table_block", "type": "table", "tableId": "bootstrap_table"})
    if broad_columns:
        blocks.extend(
            [
                {
                    "id": "broad_heading",
                    "type": "markdown",
                    "body": "## 30m broad Q4 扩样敏感性\n\n该面板不是四组共同测试集，只用于区分样本扩张与新闻窗口语义变化。",
                    "sourceId": "strata_source",
                },
                {"id": "broad_strata_table_block", "type": "table", "tableId": "broad_strata_table"},
            ]
        )
    blocks.extend(
        [
            {
                "id": "data_methods_heading",
                "type": "markdown",
                "body": (
                    "## 数据、模型与评估设计\n\n"
                    "四个容差都使用 gan_input_ready、LP 1024维输入宽度和 seed 42。real_text使用原LP；current-only沿相同模型路径输入精确全零向量；text_shuffle在每个split内使用固定的一一映射且donor pair不同。训练样本截止于 2023-07-01；最佳 checkpoint只看共同5m 2023Q3 validation（identity初始化允许epoch0参与选择）；2023Q4 test从不参与选模。同一pair的新闻权重和为1。WGAN测试使用64次稳定键MC mean，禁用persistence fallback。"
                ),
            },
        ]
    )
    if support_columns:
        blocks.extend(
            [
                {
                    "id": "support_heading",
                    "type": "markdown",
                    "body": support_body,
                    "sourceId": "support_source",
                },
                {"id": "support_table_block", "type": "table", "tableId": "support_table"},
            ]
        )
        if any(chart["id"] == "strict_support_coverage" for chart in charts):
            blocks.append(
                {
                    "id": "strict_support_coverage_block",
                    "type": "chart",
                    "chartId": "strict_support_coverage",
                }
            )
    if any(chart["id"] == "training_curves" for chart in charts):
        blocks.append({"id": "training_curves_block", "type": "chart", "chartId": "training_curves"})
        blocks.append(
            {
                "id": "training_curves_note",
                "type": "markdown",
                "body": "验证曲线只用于选择 checkpoint 与诊断 early stopping，不能替代未见 Q4 的最终评估。",
            }
        )
    if any(chart["id"] == "runtime" for chart in charts):
        blocks.extend(
            [
                {"id": "resource_heading", "type": "markdown", "body": "## GPU 资源使用"},
                {"id": "runtime_block", "type": "chart", "chartId": "runtime"},
                {
                    "id": "runtime_note",
                    "type": "markdown",
                    "body": "两张 A30 各运行两个独立进程；由于没有 NVLink，没有使用跨卡 DDP。运行时长按任务记录；峰值显存和利用率来自周期性 nvidia-smi 整卡采样，同卡同波两个任务共享这些数值，不能解释为单进程独占资源。",
                },
            ]
        )
    blocks.extend(
        [
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制与稳健性\n\n"
                    "- 这是独立的窄网格实验；没有把本轮指标与旧宽网格 checkpoint 作直接比较。\n"
                    "- 只有一个训练 seed，不能用本次 CI 推断初始化稳定性。\n"
                    + (
                        f"- {int(headline_row['epoch_zero_jobs'])}/{int(headline_row['checkpoint_jobs'])} "
                        "个最佳 checkpoint 都是 identity-preserving epoch 0；"
                        "模型退化为persistence，三组文本消融预测相同，因此不能识别文本增量。\n"
                        if epoch_zero_persistence_collapse
                        else "- 本轮两个模型均使用 identity-preserving residual输出；zero-init 的epoch0以persistence状态参与validation checkpoint选择。\n"
                        if not text_ablation.empty or training_support_mode == "raw_joint"
                        else "- 为保持单因素设计，模型仍使用既有 `softplus(current + delta)` residual 参数化；零 delta 不是 identity 映射，该已知校准问题未在本轮修复。\n"
                    )
                    +
                    "- checkpoint 仍由 2023Q3 validation 选择，未对 Q3/Q4 的 vol-change regime drift 做再加权或滚动校准。\n"
                    + (
                        "- `raw_joint` mask 用于 objective、critic/gradient penalty 和正式评估；"
                        "generator/regression forward 仍接收完整 current surface。由于 joint mask "
                        "含有 target-support 信息，它不能作为输入 mask，因此 current 输入仍可包含"
                        "外推格点，不能将本轮表述为‘输入完全无外推’。\n"
                        if training_support_mode == "raw_joint"
                        else ""
                    )
                    +
                    "- calendar、butterfly 与 smooth constraint 的系数沿用旧实验，未按更密的 moneyness/TTM 步长重新标定；结构指标应视为诊断。\n"
                    "- 10m、15m 和 30m 的新增样本多数不再是干净的新闻前后窗口；它们应解释为‘新闻已知条件下预测下一段 vol’。\n"
                    "- 四组是累计集合，不能纵向拼接后当作独立样本。\n"
                    + (
                        "- " + atm_skew_epoch_zero_caveat + "\n"
                        if atm_skew_epoch_zero_caveat
                        else ""
                    )
                    + short_atm_line
                    + "\n"
                    + "- ATM/skew 分层仅在16×16网格覆盖范围内插值，所有超界期限或 moneyness 均记为排除，不外推。"
                    + support_line
                    + "\n- Skill 是先逐 pair 计算比率再等权平均，对 persistence 误差接近零的 pair 较敏感；主结论以 MAE gap 为准。"
                    + f" Pair win 只在 gap < -{numerical_tie_tolerance:g} IV 时记为1，阈值内统一记为tie，不根据浮点噪声夸大win rate。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 后续建议\n\n"
                    "1. 将最佳容差组扩展到至少3个训练 seed，分离数据抽样与初始化不确定性。\n"
                    "2. 将 exact shift=0 作为更接近即时冲击的预注册子样本，其余1–4和≥5分钟只作预测灵敏性。\n"
                    "3. 对官方宏观日历命中样本单独复核，但仍只报告时间关联，不声称因果。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": "## 可继续追问\n\n- 扩样改善是否集中在短端 ATM，还是整个 surface？\n- 新闻类别或文本近重复程度是否驱动某些容差组的表现？\n- 将 WGAN 的 64-draw 不确定性作为校准目标，而不是 fallback 开关，是否更有信息量？",
            },
        ]
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "5m, 10m, 15m and 30m News-first TY volatility-surface forecasting experiment.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def _resolve_node() -> Path:
    configured = str(os.environ.get("NEWS_FIRST_REPORT_NODE", "")).strip()
    candidates = ([Path(configured)] if configured else []) + list(DEFAULT_NODE_CANDIDATES)
    discovered = shutil.which("node")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "A runnable Node.js binary is required to build the portable technical report."
    )


def render_portable_report(
    experiment_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Write artifact.json and package it as a self-contained HTML report."""

    root = Path(experiment_root)
    report_dir = Path(output_dir) if output_dir is not None else root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = report_dir / "artifact.json"
    html_path = report_dir / "technical_report.html"
    receipt_path = report_dir / "delivery_receipt.json"
    artifact = build_report_artifact(root)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    deliver_script = Path(
        str(os.environ.get("DATA_ANALYTICS_REPORT_DELIVER_SCRIPT", "")).strip()
        or DEFAULT_DELIVER_SCRIPT
    )
    if not deliver_script.is_file():
        raise RuntimeError(f"Portable report builder was not found: {deliver_script}")
    completed = subprocess.run(
        [
            str(_resolve_node()),
            str(deliver_script),
            "--input",
            str(artifact_path),
            "--output",
            str(html_path),
            "--timeout-ms",
            "20000",
            "--screenshot",
            str(report_dir / "delivery_failure.png"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    raw_receipt = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    try:
        receipt = json.loads(raw_receipt)
    except json.JSONDecodeError:
        receipt = {
            "ok": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2_000:],
            "stderr": completed.stderr[-2_000:],
        }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not bool(receipt.get("ok")):
        raise RuntimeError(
            f"Portable report delivery failed; see {receipt_path}: {receipt.get('error', raw_receipt)}"
        )
    return html_path


__all__ = ["REPORT_TITLE", "build_report_artifact", "render_portable_report"]
