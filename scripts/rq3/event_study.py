"""Event/no-event volatility analysis for RQ3.

This module is post-processing only: it labels existing samples by event
windows and computes current-to-target volatility movement diagnostics.
"""

from __future__ import annotations

import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "event_time_utc", "event_name", "event_type"]
GAN_SHEET = "gan_input_ready"
METRIC_COLUMNS = [
    "mae",
    "current_mae",
    "mae_gap_vs_current",
    "win_flag_vs_current",
    "short_atm_mae_gap_vs_current",
    "atm_short_pure_mae_gap_vs_current",
]
WORKBOOK_EVENT_TEST_METRICS = [
    "surface_jump_mae",
    "surface_jump_rmse",
    "surface_jump_max_abs",
    "atm_short_abs_jump",
    "atm_short_signed_jump",
]
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260625


def resolve_event_window(
    *,
    window_minutes: float = 30.0,
    pre_window_minutes: float | None = None,
    post_window_minutes: float | None = None,
) -> dict[str, float | str]:
    """Resolve symmetric or explicitly asymmetric event-window bounds."""
    if pre_window_minutes is None and post_window_minutes is None:
        half_width = abs(float(window_minutes))
        return {
            "window_mode": "symmetric",
            "window_minutes": float(window_minutes),
            "pre_window_minutes": half_width,
            "post_window_minutes": half_width,
        }
    if pre_window_minutes is None or post_window_minutes is None:
        raise ValueError(
            "For an asymmetric event window, provide both pre_window_minutes and post_window_minutes."
        )
    pre = float(pre_window_minutes)
    post = float(post_window_minutes)
    if pre < 0.0 or post < 0.0:
        raise ValueError("pre_window_minutes and post_window_minutes must be non-negative.")
    return {
        "window_mode": "asymmetric",
        "window_minutes": float(window_minutes),
        "pre_window_minutes": pre,
        "post_window_minutes": post,
    }


def timestamp_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_timestamp_utc(value: Any, *, field_name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} contains an invalid UTC timestamp: {value!r}")
    return pd.Timestamp(parsed).tz_convert("UTC")


def parse_serialized_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [float(item) for item in parsed]
    raise ValueError(f"Unsupported serialized list value: {value!r}")


def parse_surface(row: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strike_grid = np.asarray(parse_serialized_list(row["strike_grid"]), dtype=np.float32)
    maturity_days_grid = np.asarray(parse_serialized_list(row["maturity_days_grid"]), dtype=np.float32)
    if strike_grid.size <= 0 or maturity_days_grid.size <= 0:
        raise ValueError("strike_grid and maturity_days_grid must be non-empty.")
    surface_shape = (int(maturity_days_grid.size), int(strike_grid.size))
    current = np.asarray(parse_serialized_list(row["current_surface_flat"]), dtype=np.float32).reshape(surface_shape)
    target = np.asarray(parse_serialized_list(row["target_surface_flat"]), dtype=np.float32).reshape(surface_shape)
    return current, target, strike_grid, maturity_days_grid


def atm_short_indices(strike_grid: Sequence[float], maturity_days_grid: Sequence[float]) -> tuple[int, int]:
    strikes = np.asarray(strike_grid, dtype=np.float32)
    maturities = np.asarray(maturity_days_grid, dtype=np.float32)
    atm_idx = int(np.argmin(np.abs(strikes - 1.0)))
    short_idx = int(np.argmin(maturities))
    return atm_idx, short_idx


def surface_jump_metrics(
    current: np.ndarray,
    target: np.ndarray,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> dict[str, float]:
    diff = target.astype(np.float32) - current.astype(np.float32)
    abs_diff = np.abs(diff)
    atm_idx, short_idx = atm_short_indices(strike_grid, maturity_days_grid)
    current_atm = float(current[short_idx, atm_idx])
    target_atm = float(target[short_idx, atm_idx])
    return {
        "surface_jump_mae": float(np.mean(abs_diff)),
        "surface_jump_rmse": float(math.sqrt(float(np.mean(np.square(diff))))),
        "surface_jump_max_abs": float(np.max(abs_diff)),
        "atm_short_current_vol": current_atm,
        "atm_short_target_vol": target_atm,
        "atm_short_signed_jump": float(target_atm - current_atm),
        "atm_short_abs_jump": float(abs(target_atm - current_atm)),
        "atm_strike": float(np.asarray(strike_grid, dtype=np.float32)[atm_idx]),
        "short_maturity_days": float(np.asarray(maturity_days_grid, dtype=np.float32)[short_idx]),
    }


def load_events(events_csv: str | Path, *, allow_empty: bool = False) -> pd.DataFrame:
    path = Path(events_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Event calendar CSV does not exist: {path}. "
            "Create one with columns: event_id,event_time_utc,event_name,event_type."
        )
    frame = pd.read_csv(path, dtype=object)
    missing = [column for column in ["event_id", "event_time_utc"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Event calendar {path} is missing required columns: {missing}")
    if frame.empty and not bool(allow_empty):
        raise ValueError(
            f"Event calendar {path} has no event rows. "
            "Paper-facing RQ3 requires a non-empty event calendar; use --allow-zero-announcement only for diagnostics."
        )
    for column in EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[EVENT_COLUMNS].copy()
    frame["event_time"] = frame["event_time_utc"].map(lambda value: parse_timestamp_utc(value, field_name="event_time_utc"))
    return frame


def validate_announcement_count(
    labeled: pd.DataFrame,
    *,
    allow_zero_announcement: bool = False,
    context: str = "RQ3 analysis",
) -> None:
    announcement_count = int(labeled["is_announcement_window"].sum()) if "is_announcement_window" in labeled.columns else 0
    if announcement_count == 0 and not bool(allow_zero_announcement):
        raise ValueError(
            f"{context} produced zero announcement-window samples. "
            "Check the event calendar, split, timestamps, or window_minutes. "
            "Use --allow-zero-announcement only for quiet-only diagnostics."
        )


def label_rows_by_event(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    timestamp_column: str = "news_timestamp_utc",
    window_minutes: float = 30.0,
    pre_window_minutes: float | None = None,
    post_window_minutes: float | None = None,
) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise ValueError(f"Input frame is missing timestamp column: {timestamp_column}")
    labeled = frame.copy()
    timestamps = labeled[timestamp_column].map(lambda value: parse_timestamp_utc(value, field_name=timestamp_column))
    window = resolve_event_window(
        window_minutes=window_minutes,
        pre_window_minutes=pre_window_minutes,
        post_window_minutes=post_window_minutes,
    )
    pre_window_seconds = float(window["pre_window_minutes"]) * 60.0
    post_window_seconds = float(window["post_window_minutes"]) * 60.0

    event_ids: list[str] = []
    event_names: list[str] = []
    event_types: list[str] = []
    event_times: list[str] = []
    event_deltas: list[float | None] = []
    flags: list[bool] = []

    event_records = list(events.to_dict("records"))
    for timestamp in timestamps:
        best_record: Mapping[str, Any] | None = None
        best_delta_seconds: float | None = None
        for record in event_records:
            delta_seconds = float((timestamp - record["event_time"]).total_seconds())
            abs_delta_seconds = abs(delta_seconds)
            if -pre_window_seconds <= delta_seconds <= post_window_seconds and (
                best_delta_seconds is None or abs_delta_seconds < abs(best_delta_seconds)
            ):
                best_record = record
                best_delta_seconds = delta_seconds
        if best_record is None:
            flags.append(False)
            event_ids.append("")
            event_names.append("")
            event_types.append("")
            event_times.append("")
            event_deltas.append(None)
            continue
        flags.append(True)
        event_ids.append(str(best_record.get("event_id", "")))
        event_names.append(str(best_record.get("event_name", "")))
        event_types.append(str(best_record.get("event_type", "")))
        event_times.append(str(best_record.get("event_time_utc", "")))
        event_deltas.append(float(best_delta_seconds or 0.0) / 60.0)

    labeled["is_announcement_window"] = flags
    labeled["event_group"] = np.where(labeled["is_announcement_window"], "announcement", "quiet")
    labeled["event_id"] = event_ids
    labeled["event_name"] = event_names
    labeled["event_type"] = event_types
    labeled["event_time_utc"] = event_times
    labeled["event_time_delta_minutes"] = event_deltas
    labeled["event_window_mode"] = str(window["window_mode"])
    labeled["event_pre_window_minutes"] = float(window["pre_window_minutes"])
    labeled["event_post_window_minutes"] = float(window["post_window_minutes"])
    return labeled


def chronological_split(frame: pd.DataFrame, *, split: str, train_ratio: float) -> pd.DataFrame:
    normalized_split = str(split).strip().lower()
    ordered = frame.sort_values(["news_timestamp_utc", "_rq3_original_index"], kind="mergesort").reset_index(drop=True)
    ordered["_rq3_ordered_index"] = range(len(ordered))
    if normalized_split == "all":
        ordered["split"] = "all"
        return ordered
    total = len(ordered)
    if total < 2:
        split_idx = total
    else:
        split_idx = int(total * float(train_ratio))
        split_idx = max(1, min(total - 1, split_idx))
    ordered["split"] = np.where(ordered["_rq3_ordered_index"] < split_idx, "train", "val")
    if normalized_split not in {"train", "val"}:
        raise ValueError(f"split must be one of ['train', 'val', 'all'], got: {split}")
    return ordered[ordered["split"] == normalized_split].reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, source: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def build_workbook_sample_metrics(
    workbook_frame: pd.DataFrame,
    *,
    split: str = "val",
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    required = [
        "news_timestamp_utc",
        "current_surface_flat",
        "target_surface_flat",
        "strike_grid",
        "maturity_days_grid",
    ]
    _require_columns(workbook_frame, required, source="Workbook sheet")
    frame = workbook_frame.copy()
    if "training_candidate_flag" in frame.columns:
        candidate_flags = pd.to_numeric(frame["training_candidate_flag"], errors="coerce").fillna(0).astype(int)
        filtered = frame[candidate_flags == 1].copy()
        if not filtered.empty:
            frame = filtered
    frame["_rq3_original_index"] = range(len(frame))
    selected = chronological_split(frame, split=split, train_ratio=train_ratio)

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        current, target, strike_grid, maturity_days_grid = parse_surface(row)
        sample_id = str(row.get("sample_id", f"row_{int(row['_rq3_original_index'])}"))
        output_row = {
            "sample_id": sample_id,
            "global_index": int(row.get("global_index", row.get("_rq3_ordered_index", len(rows)))),
            "news_timestamp_utc": str(row["news_timestamp_utc"]),
            "current_snapshot_time_utc": str(row.get("current_snapshot_time_utc", "")),
            "target_snapshot_time_utc": str(row.get("target_snapshot_time_utc", "")),
            "pair_quality_label": str(row.get("pair_quality_label", "")),
            "current_weighted_iv_rmse": row.get("current_weighted_iv_rmse", None),
            "target_weighted_iv_rmse": row.get("target_weighted_iv_rmse", None),
            "split": str(row.get("split", split)),
            "_rq3_ordered_index": int(row.get("_rq3_ordered_index", len(rows))),
        }
        for metadata_column in (
            "event_group",
            "has_news",
            "news_cluster_id",
            "quiet_buffer_minutes",
            "quiet_grid_minutes",
        ):
            if metadata_column in row.index:
                output_row[metadata_column] = row.get(metadata_column)
        output_row.update(surface_jump_metrics(current, target, strike_grid, maturity_days_grid))
        rows.append(output_row)
    return pd.DataFrame(rows)


def summarize_groups(labeled: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "surface_jump_mae",
        "surface_jump_rmse",
        "surface_jump_max_abs",
        "atm_short_abs_jump",
        "atm_short_signed_jump",
    ]
    rows: list[dict[str, Any]] = []
    for group, group_frame in labeled.groupby("event_group", dropna=False):
        row: dict[str, Any] = {
            "event_group": str(group),
            "sample_count": int(len(group_frame)),
            "announcement_count": int(group_frame["is_announcement_window"].sum()),
        }
        for column in metric_columns:
            if column in group_frame.columns:
                values = pd.to_numeric(group_frame[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean()) if not values.dropna().empty else float("nan")
                row[f"{column}_median"] = float(values.median()) if not values.dropna().empty else float("nan")
        rows.append(row)
    order = {"announcement": 0, "quiet": 1}
    return pd.DataFrame(rows).sort_values("event_group", key=lambda series: series.map(order).fillna(99))


def summarize_events(labeled: pd.DataFrame) -> pd.DataFrame:
    event_rows = labeled[labeled["is_announcement_window"].astype(bool)].copy()
    if event_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "event_type",
                "event_time_utc",
                "sample_count",
                "first_news_timestamp_utc",
                "last_news_timestamp_utc",
                "surface_jump_mae_mean",
                "atm_short_abs_jump_mean",
            ]
        )
    rows: list[dict[str, Any]] = []
    for event_id, group in event_rows.groupby("event_id", dropna=False):
        ordered = group.sort_values("news_timestamp_utc")
        rows.append(
            {
                "event_id": str(event_id),
                "event_name": str(ordered["event_name"].iloc[0]),
                "event_type": str(ordered["event_type"].iloc[0]),
                "event_time_utc": str(ordered["event_time_utc"].iloc[0]),
                "sample_count": int(len(ordered)),
                "first_news_timestamp_utc": str(ordered["news_timestamp_utc"].iloc[0]),
                "last_news_timestamp_utc": str(ordered["news_timestamp_utc"].iloc[-1]),
                "surface_jump_mae_mean": float(pd.to_numeric(ordered["surface_jump_mae"], errors="coerce").mean()),
                "atm_short_abs_jump_mean": float(pd.to_numeric(ordered["atm_short_abs_jump"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)


def quality_audit(labeled: pd.DataFrame) -> pd.DataFrame:
    if "pair_quality_label" not in labeled.columns:
        return pd.DataFrame(columns=["event_group", "pair_quality_label", "sample_count", "group_total", "sample_rate"])
    rows: list[dict[str, Any]] = []
    for group, group_frame in labeled.groupby("event_group", dropna=False):
        total = int(len(group_frame))
        labels = group_frame["pair_quality_label"].fillna("").astype(str)
        for label, count in labels.value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "event_group": str(group),
                    "pair_quality_label": str(label),
                    "sample_count": int(count),
                    "group_total": total,
                    "sample_rate": float(count) / float(total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _welch_t_test(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    left_n = float(left.size)
    right_n = float(right.size)
    se_sq = left_var / left_n + right_var / right_n
    if se_sq <= 0:
        return {
            "t_stat": float("nan"),
            "welch_df": float("nan"),
            "p_two_sided": float("nan"),
            "p_event_greater": float("nan"),
        }
    t_stat = float((np.mean(left) - np.mean(right)) / math.sqrt(se_sq))
    numerator = se_sq * se_sq
    denominator = 0.0
    if left_n > 1:
        denominator += (left_var / left_n) ** 2 / (left_n - 1.0)
    if right_n > 1:
        denominator += (right_var / right_n) ** 2 / (right_n - 1.0)
    if denominator <= 0:
        return {
            "t_stat": t_stat,
            "welch_df": float("nan"),
            "p_two_sided": float("nan"),
            "p_event_greater": float("nan"),
        }
    df = float(numerator / denominator)
    from scipy import stats

    p_event_greater = float(stats.t.sf(t_stat, df))
    return {
        "t_stat": t_stat,
        "welch_df": df,
        "p_two_sided": float(min(1.0, 2.0 * min(p_event_greater, 1.0 - p_event_greater))),
        "p_event_greater": p_event_greater,
    }


def _bootstrap_mean_difference_ci(
    announcement: np.ndarray,
    quiet: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    announcement_size = int(announcement.size)
    quiet_size = int(quiet.size)
    diffs = np.empty(int(iterations), dtype=np.float64)
    for idx in range(int(iterations)):
        announcement_sample = announcement[rng.integers(0, announcement_size, size=announcement_size)]
        quiet_sample = quiet[rng.integers(0, quiet_size, size=quiet_size)]
        diffs[idx] = float(np.mean(announcement_sample) - np.mean(quiet_sample))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def event_vs_quiet_tests(
    labeled: pd.DataFrame,
    *,
    metrics: Sequence[str] = WORKBOOK_EVENT_TEST_METRICS,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    announcement_frame = labeled[labeled["event_group"] == "announcement"]
    quiet_frame = labeled[labeled["event_group"] == "quiet"]
    for metric in metrics:
        row: dict[str, Any] = {
            "metric": str(metric),
            "difference": "announcement_minus_quiet",
            "interpretation_positive": "announcement window has larger IVS jump",
            "interpretation_negative": "quiet window has larger IVS jump",
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
        }
        if metric not in labeled.columns:
            row.update(
                {
                    "announcement_n": 0,
                    "quiet_n": 0,
                    "announcement_mean": float("nan"),
                    "quiet_mean": float("nan"),
                    "announcement_median": float("nan"),
                    "quiet_median": float("nan"),
                    "announcement_minus_quiet_mean": float("nan"),
                    "t_stat": float("nan"),
                    "welch_df": float("nan"),
                    "p_two_sided": float("nan"),
                    "p_event_greater": float("nan"),
                    "bootstrap_ci95_low": float("nan"),
                    "bootstrap_ci95_high": float("nan"),
                }
            )
            rows.append(row)
            continue
        announcement = pd.to_numeric(announcement_frame[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        quiet = pd.to_numeric(quiet_frame[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        row.update(
            {
                "announcement_n": int(announcement.size),
                "quiet_n": int(quiet.size),
                "announcement_mean": float(np.mean(announcement)) if announcement.size else float("nan"),
                "quiet_mean": float(np.mean(quiet)) if quiet.size else float("nan"),
                "announcement_median": float(np.median(announcement)) if announcement.size else float("nan"),
                "quiet_median": float(np.median(quiet)) if quiet.size else float("nan"),
            }
        )
        if announcement.size < 2 or quiet.size < 2:
            row.update(
                {
                    "announcement_minus_quiet_mean": float("nan"),
                    "t_stat": float("nan"),
                    "welch_df": float("nan"),
                    "p_two_sided": float("nan"),
                    "p_event_greater": float("nan"),
                    "bootstrap_ci95_low": float("nan"),
                    "bootstrap_ci95_high": float("nan"),
                }
            )
            rows.append(row)
            continue
        diff_mean = float(np.mean(announcement) - np.mean(quiet))
        test_values = _welch_t_test(announcement, quiet)
        ci_low, ci_high = _bootstrap_mean_difference_ci(
            announcement,
            quiet,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        row.update(
            {
                "announcement_minus_quiet_mean": diff_mean,
                **test_values,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_event_template(output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=EVENT_COLUMNS).to_csv(output, index=False)
    return output


def _write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def _write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output


def plot_event_cases(labeled: pd.DataFrame, output_dir: str | Path, *, max_events: int = 3) -> list[str]:
    event_rows = labeled[labeled["is_announcement_window"].astype(bool)].copy()
    if event_rows.empty or int(max_events) <= 0:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / "plots" / "rq3_atm_short_event_cases"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for event_id, group in list(event_rows.groupby("event_id", dropna=False))[: int(max_events)]:
        ordered = group.sort_values(["news_timestamp_utc", "sample_id"])
        timestamps = [parse_timestamp_utc(value, field_name="news_timestamp_utc").to_pydatetime() for value in ordered["news_timestamp_utc"]]
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
        ax.plot(timestamps, ordered["atm_short_current_vol"].astype(float), label="Current", linewidth=2.0)
        ax.plot(timestamps, ordered["atm_short_target_vol"].astype(float), label="Target", linewidth=2.0, linestyle="--")
        ax.set_title(f"RQ3 ATM-short IV around event: {event_id}")
        ax.set_xlabel("News timestamp (UTC)")
        ax.set_ylabel("Implied volatility")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.tight_layout()
        safe_event_id = str(event_id).replace("/", "_").replace(" ", "_") or "event"
        output_path = plot_dir / f"{safe_event_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        written.append(str(output_path))
    return written


def analyze_workbook(
    *,
    merged_vol_path: str | Path,
    events_csv: str | Path,
    output_dir: str | Path,
    sheet_name: str = GAN_SHEET,
    window_minutes: float = 30.0,
    pre_window_minutes: float | None = None,
    post_window_minutes: float | None = None,
    split: str = "val",
    train_ratio: float = 0.8,
    save_plots: bool = True,
    max_case_events: int = 3,
    allow_zero_announcement: bool = False,
) -> Path:
    workbook_path = Path(merged_vol_path).expanduser()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Merged-vol workbook does not exist: {workbook_path}")
    events = load_events(events_csv, allow_empty=allow_zero_announcement)
    workbook_frame = pd.read_excel(workbook_path, sheet_name=sheet_name, dtype=object)
    sample_metrics = build_workbook_sample_metrics(workbook_frame, split=split, train_ratio=train_ratio)
    window = resolve_event_window(
        window_minutes=window_minutes,
        pre_window_minutes=pre_window_minutes,
        post_window_minutes=post_window_minutes,
    )
    labeled = label_rows_by_event(
        sample_metrics,
        events,
        window_minutes=window_minutes,
        pre_window_minutes=pre_window_minutes,
        post_window_minutes=post_window_minutes,
    )
    validate_announcement_count(
        labeled,
        allow_zero_announcement=allow_zero_announcement,
        context="RQ3 workbook analysis",
    )

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(labeled, output / "rq3_labeled_samples.csv")
    _write_csv(summarize_groups(labeled), output / "rq3_group_summary.csv")
    _write_csv(summarize_events(labeled), output / "rq3_event_summary.csv")
    _write_csv(event_vs_quiet_tests(labeled), output / "rq3_event_vs_quiet_tests.csv")
    _write_csv(quality_audit(labeled), output / "rq3_quality_audit.csv")
    plot_paths = plot_event_cases(labeled, output, max_events=max_case_events) if save_plots else []
    _write_json(
        {
            "mode": "workbook",
            "merged_vol_path": str(workbook_path),
            "events_csv": str(Path(events_csv).expanduser()),
            "output_dir": str(output),
            "sheet_name": str(sheet_name),
            "window_minutes": float(window_minutes),
            "window_mode": str(window["window_mode"]),
            "pre_window_minutes": float(window["pre_window_minutes"]),
            "post_window_minutes": float(window["post_window_minutes"]),
            "event_window_definition": (
                f"[-{float(window['pre_window_minutes'])}, +{float(window['post_window_minutes'])}] "
                "minutes around event_time_utc"
            ),
            "split": str(split),
            "train_ratio": float(train_ratio),
            "sample_count": int(len(labeled)),
            "event_count": int(len(events)),
            "announcement_count": int(labeled["is_announcement_window"].sum()) if not labeled.empty else 0,
            "allow_zero_announcement": bool(allow_zero_announcement),
            "plot_paths": plot_paths,
        },
        output / "rq3_run_manifest.json",
    )
    return output


def parse_result_specs(result_specs: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in result_specs:
        if "=" not in str(spec):
            raise ValueError(f"--result must use label=path format, got: {spec!r}")
        label, raw_path = str(spec).split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError(f"--result must use non-empty label=path format, got: {spec!r}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Result summary CSV does not exist for label {label!r}: {path}")
        parsed.append((label.strip(), path))
    if not parsed:
        raise ValueError("At least one --result label=path item is required.")
    return parsed


def load_result_summaries(result_specs: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in parse_result_specs(result_specs):
        frame = pd.read_csv(path, dtype=object)
        if "news_timestamp_utc" not in frame.columns:
            raise ValueError(f"Result summary {path} is missing required column: news_timestamp_utc")
        frame.insert(0, "model_label", label)
        frame.insert(1, "result_summary_path", str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_result_groups(labeled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, event_group), group in labeled.groupby(["model_label", "event_group"], dropna=False):
        row: dict[str, Any] = {
            "model_label": str(model_label),
            "event_group": str(event_group),
            "sample_count": int(len(group)),
            "announcement_count": int(group["is_announcement_window"].sum()),
        }
        for column in METRIC_COLUMNS:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce")
                if not values.dropna().empty:
                    row[f"{column}_mean"] = float(values.mean())
                    row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_label", "event_group"]).reset_index(drop=True)


def analyze_results(
    *,
    result_specs: Sequence[str],
    events_csv: str | Path,
    output_dir: str | Path,
    window_minutes: float = 30.0,
    pre_window_minutes: float | None = None,
    post_window_minutes: float | None = None,
    allow_zero_announcement: bool = False,
) -> Path:
    events = load_events(events_csv, allow_empty=allow_zero_announcement)
    results = load_result_summaries(result_specs)
    window = resolve_event_window(
        window_minutes=window_minutes,
        pre_window_minutes=pre_window_minutes,
        post_window_minutes=post_window_minutes,
    )
    labeled = label_rows_by_event(
        results,
        events,
        window_minutes=window_minutes,
        pre_window_minutes=pre_window_minutes,
        post_window_minutes=post_window_minutes,
    )
    validate_announcement_count(
        labeled,
        allow_zero_announcement=allow_zero_announcement,
        context="RQ3 result analysis",
    )
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(labeled, output / "rq3_result_labeled_samples.csv")
    _write_csv(summarize_result_groups(labeled), output / "rq3_result_group_metrics.csv")
    _write_json(
        {
            "mode": "result",
            "result_specs": list(result_specs),
            "events_csv": str(Path(events_csv).expanduser()),
            "output_dir": str(output),
            "window_minutes": float(window_minutes),
            "window_mode": str(window["window_mode"]),
            "pre_window_minutes": float(window["pre_window_minutes"]),
            "post_window_minutes": float(window["post_window_minutes"]),
            "event_window_definition": (
                f"[-{float(window['pre_window_minutes'])}, +{float(window['post_window_minutes'])}] "
                "minutes around event_time_utc"
            ),
            "sample_count": int(len(labeled)),
            "event_count": int(len(events)),
            "announcement_count": int(labeled["is_announcement_window"].sum()) if not labeled.empty else 0,
            "allow_zero_announcement": bool(allow_zero_announcement),
            "metric_columns": [column for column in METRIC_COLUMNS if column in labeled.columns],
        },
        output / "rq3_result_manifest.json",
    )
    return output
