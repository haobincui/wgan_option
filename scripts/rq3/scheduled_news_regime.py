"""Frozen-model RQ3 scheduled-news versus matched ordinary-news analysis."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.support import (
    RawSurfaceSupportArtifact,
    raw_support_mask,
    reconstruct_raw_surface,
)


REQUIRED_MODELS = ("continued_no_text", "lp", "bow", "llm_sentiment")
POINT_METRICS = (
    "surface_mae",
    "short_atm_mae",
    "supported_shortest_atm_abs_err",
)
CURRENT_METRICS = {
    "surface_mae": "current_mae",
    "short_atm_mae": "current_atm_short_pure_mae",
    "supported_shortest_atm_abs_err": (
        "current_supported_shortest_atm_abs_err"
    ),
}
CONTRASTS = (
    ("lp", "continued_no_text", "lp_vs_continued_no_text", "primary"),
    ("lp", "bow", "lp_vs_bow", "secondary_representation"),
    ("lp", "llm_sentiment", "lp_vs_llm_sentiment", "secondary_representation"),
)
RQ1_VARIANTS = {
    "continued_no_text": "pair_pca_no_text_continued",
    "lp": "pair_pca_text_residual_pretrained",
    "shuffled_lp": "pair_pca_shuffled_residual_pretrained",
}
CALENDAR_REQUIRED_COLUMNS = {
    "event_id",
    "event_family",
    "release_name",
    "release_time_local",
    "release_timezone",
    "release_time_utc",
    "scheduled_or_unscheduled",
    "official_source",
    "source_retrieval_date",
    "calendar_version",
}
MATCH_COVARIATES = (
    "minute_of_day_sin",
    "minute_of_day_cos",
    "weekday_sin",
    "weekday_cos",
    "current_surface_mean",
    "current_surface_std",
    "current_supported_shortest_atm",
    "current_short_atm_mean",
    "current_strike_slope",
    "current_term_slope",
    "current_curvature",
    "current_weighted_iv_rmse",
    "current_surface_slice_count",
    "current_supported_cell_fraction",
    "recent_surface_level_std_24h",
    "news_count",
)
CONDITIONAL_COVARIATES = (
    "current_surface_mean",
    "current_surface_std",
    "current_supported_shortest_atm",
    "current_weighted_iv_rmse",
    "recent_surface_level_std_24h",
    "current_supported_cell_fraction",
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_py312() -> None:
    if Path(sys.prefix).name != "py312":
        raise RuntimeError(
            "RQ3 must run in conda environment 'py312'; "
            f"current prefix is {sys.prefix}."
        )


def _canonical_timestamp(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _surface_pair_id(current: Any, target: Any) -> str:
    payload = f"{_canonical_timestamp(current)}|{_canonical_timestamp(target)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_offset(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _parse_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel().tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected serialized list, got {type(parsed).__name__}.")
    return [float(item) for item in parsed]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return target


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _require_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Missing {label}: {target}")
    return target


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256_file(source) != _sha256_file(target):
            raise ValueError(f"Existing archived input differs from source: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def load_event_calendar(path: str | Path) -> pd.DataFrame:
    """Load and rigorously validate a frozen scheduled-event calendar."""

    source = _require_file(path, "scheduled event calendar")
    frame = pd.read_csv(source)
    missing = sorted(CALENDAR_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Event calendar {source} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"Event calendar is empty: {source}")
    if frame["event_id"].astype(str).duplicated().any():
        duplicates = frame.loc[
            frame["event_id"].astype(str).duplicated(keep=False), "event_id"
        ].astype(str)
        raise ValueError(f"Duplicate event_id values: {sorted(duplicates.unique())}")
    frame = frame.copy()
    frame["release_time"] = pd.to_datetime(
        frame["release_time_utc"],
        utc=True,
        errors="raise",
    )
    if frame["release_time"].isna().any():
        raise ValueError("Event calendar contains invalid release_time_utc values.")
    local_as_utc = pd.to_datetime(
        frame["release_time_local"],
        utc=True,
        errors="raise",
    )
    if not local_as_utc.equals(frame["release_time"]):
        raise ValueError(
            "release_time_local offsets do not agree with release_time_utc."
        )
    for column in (
        "event_family",
        "release_timezone",
        "official_source",
        "calendar_version",
    ):
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Event calendar contains blank {column} values.")
    scheduled = frame["scheduled_or_unscheduled"].astype(str).str.lower()
    if not scheduled.eq("scheduled").all():
        raise ValueError(
            "Primary RQ3 calendar may contain only scheduled releases; "
            "put unscheduled events in a separate robustness calendar."
        )
    if "priority" not in frame.columns:
        frame["priority"] = np.arange(len(frame), dtype=int)
    frame["priority"] = pd.to_numeric(frame["priority"], errors="raise").astype(int)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["event_family"] = frame["event_family"].astype(str)
    frame["release_local_date"] = frame["release_time_local"].astype(str).str[:10]
    frame["release_trading_day"] = frame["release_local_date"]
    return frame.sort_values(
        ["release_time", "priority", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_model_matching(samples: pd.DataFrame) -> None:
    if "alignment_type" not in samples.columns:
        samples["alignment_type"] = "exact"
    required = {
        "fold",
        "seed",
        "surface_pair_id",
        "model",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        *POINT_METRICS,
        *CURRENT_METRICS.values(),
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Frozen RQ2 metrics are missing columns: {missing}")
    actual_models = set(samples["model"].astype(str).unique())
    missing_models = sorted(set(REQUIRED_MODELS) - actual_models)
    if missing_models:
        raise ValueError(f"Frozen RQ2 metrics are missing models: {missing_models}")
    keys = ["fold", "seed", "surface_pair_id"]
    reference: pd.DataFrame | None = None
    for model in REQUIRED_MODELS:
        frame = samples[samples["model"].astype(str) == model].sort_values(keys)
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate frozen prediction keys for model={model}.")
        if reference is None:
            reference = frame.reset_index(drop=True)
            continue
        candidate = frame.reset_index(drop=True)
        if not candidate[keys].equals(reference[keys]):
            raise ValueError(f"Frozen pair keys do not match for model={model}.")
        for column in (
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
            "alignment_type",
        ):
            if not np.array_equal(
                candidate[column].astype(str).to_numpy(),
                reference[column].astype(str).to_numpy(),
            ):
                raise ValueError(f"Frozen timestamp mismatch for model={model}.")
        for column in CURRENT_METRICS.values():
            difference = np.abs(
                candidate[column].to_numpy(dtype=np.float64)
                - reference[column].to_numpy(dtype=np.float64)
            )
            if float(np.nanmax(difference)) > 1e-8:
                raise ValueError(
                    f"Frozen persistence metric mismatch for model={model}, "
                    f"column={column}."
                )


def load_frozen_predictions(
    rq1_metrics_path: str | Path,
    rq2_metrics_path: str | Path,
    *,
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load four-model predictions and audit duplicated RQ1 LP/no-text rows."""

    rq1_path = _require_file(rq1_metrics_path, "RQ1 frozen test metrics")
    rq2_path = _require_file(rq2_metrics_path, "RQ2 frozen test metrics")
    rq1 = pd.read_csv(rq1_path)
    rq2 = pd.read_csv(rq2_path)
    _validate_model_matching(rq2)

    keys = ["fold", "seed", "surface_pair_id"]
    audit_rows: list[dict[str, Any]] = []
    for model in ("continued_no_text", "lp"):
        variant = RQ1_VARIANTS[model]
        left = rq1[rq1["variant"].astype(str) == variant]
        right = rq2[rq2["model"].astype(str) == model]
        merged = left.merge(
            right,
            on=keys,
            suffixes=("_rq1", "_rq2"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"RQ1/RQ2 frozen-key mismatch for model={model}.")
        for metric in POINT_METRICS:
            maximum = float(
                np.max(
                    np.abs(
                        merged[f"{metric}_rq1"].to_numpy(dtype=np.float64)
                        - merged[f"{metric}_rq2"].to_numpy(dtype=np.float64)
                    )
                )
            )
            audit_rows.append(
                {
                    "model": model,
                    "rq1_variant": variant,
                    "metric": metric,
                    "matched_rows": len(merged),
                    "max_abs_difference": maximum,
                    "tolerance": float(tolerance),
                    "status": "ok" if maximum <= tolerance else "mismatch",
                }
            )
            if maximum > tolerance:
                raise ValueError(
                    f"RQ1/RQ2 frozen prediction mismatch for {model}/{metric}: "
                    f"{maximum} > {tolerance}."
                )

    shuffled = rq1[
        rq1["variant"].astype(str) == RQ1_VARIANTS["shuffled_lp"]
    ].copy()
    if shuffled.empty:
        raise ValueError("RQ1 frozen metrics do not contain shuffled-LP placebo.")
    shuffled["model"] = "shuffled_lp"
    common_columns = sorted(set(rq2.columns) & set(shuffled.columns))
    for column in rq2.columns:
        if column not in shuffled.columns:
            shuffled[column] = np.nan
    combined = pd.concat(
        [rq2, shuffled[rq2.columns]],
        ignore_index=True,
    )
    return combined, pd.DataFrame(audit_rows)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(np.mean(selected)) if selected.size else float("nan")


def _surface_features(
    row: Any,
    *,
    support_artifact: RawSurfaceSupportArtifact | None = None,
) -> dict[str, float]:
    if support_artifact is None:
        strikes = np.asarray(_parse_list(row.strike_grid), dtype=np.float64)
        maturities = np.asarray(
            _parse_list(row.maturity_days_grid),
            dtype=np.float64,
        )
        surface = np.asarray(
            _parse_list(row.current_surface_flat),
            dtype=np.float64,
        )
        if strikes.size == 0 or maturities.size == 0:
            raise ValueError("Surface grid is empty while building RQ3 covariates.")
        expected = int(strikes.size * maturities.size)
        if surface.size != expected:
            raise ValueError(
                f"Current surface size {surface.size} does not match grid {expected}."
            )
        matrix = surface.reshape((maturities.size, strikes.size))
        support = np.ones_like(matrix, dtype=bool)
    else:
        strikes = np.asarray(support_artifact.strike_grid, dtype=np.float64)
        maturities = np.asarray(
            support_artifact.maturity_days_grid,
            dtype=np.float64,
        )
        params = row.current_surface_param_json
        matrix = reconstruct_raw_surface(
            params,
            strike_grid=strikes,
            maturity_days_grid=maturities,
        ).astype(np.float64)
        support = raw_support_mask(
            params,
            strike_grid=strikes,
            maturity_days_grid=maturities,
        )
        if not np.any(support):
            raise ValueError(
                "Current raw surface has no cells inside its frozen fold support."
            )

    supported_locations = np.argwhere(support)
    shortest_index = int(np.min(supported_locations[:, 0]))
    strike_candidates = np.flatnonzero(support[shortest_index])
    atm_index = int(
        strike_candidates[
            np.argmin(np.abs(strikes[strike_candidates] - 1.0))
        ]
    )
    short_atm_mask = (
        support
        & (maturities[:, None] <= 90.0)
        & (np.abs(strikes[None, :] - 1.0) <= 0.06)
    )
    supported_strikes = strikes[np.any(support, axis=0)]
    supported_maturities = maturities[np.any(support, axis=1)]
    strike_low, strike_high = np.quantile(supported_strikes, [0.25, 0.75])
    maturity_low, maturity_high = np.quantile(
        supported_maturities,
        [0.25, 0.75],
    )
    low_strike_mask = support & (strikes[None, :] <= strike_low)
    high_strike_mask = support & (strikes[None, :] >= strike_high)
    low_maturity_mask = support & (maturities[:, None] <= maturity_low)
    high_maturity_mask = support & (maturities[:, None] >= maturity_high)
    center_mask = support & (np.abs(strikes[None, :] - 1.0) <= 0.06)
    wing_cutoff = float(
        np.quantile(np.abs(supported_strikes - 1.0), 0.75)
    )
    wing_mask = support & (
        np.abs(strikes[None, :] - 1.0) >= wing_cutoff
    )
    return {
        "current_surface_mean": _masked_mean(matrix, support),
        "current_surface_std": float(np.std(matrix[support])),
        "current_supported_shortest_atm": float(
            matrix[shortest_index, atm_index]
        ),
        "current_supported_shortest_maturity_days": float(
            maturities[shortest_index]
        ),
        "current_short_atm_mean": _masked_mean(matrix, short_atm_mask),
        "current_strike_slope": float(
            _masked_mean(matrix, high_strike_mask)
            - _masked_mean(matrix, low_strike_mask)
        ),
        "current_term_slope": float(
            _masked_mean(matrix, high_maturity_mask)
            - _masked_mean(matrix, low_maturity_mask)
        ),
        "current_curvature": float(
            _masked_mean(matrix, wing_mask)
            - _masked_mean(matrix, center_mask)
        ),
        "current_supported_cell_count": int(np.sum(support)),
        "current_supported_cell_fraction": float(np.mean(support)),
    }


def build_forecast_origin_covariates(
    workbook_path: str | Path,
    pair_frame: pd.DataFrame,
    *,
    sheet_name: str = "gan_input_ready",
    support_artifact_paths: Mapping[str, str | Path] | None = None,
) -> pd.DataFrame:
    """Build forecast-origin covariates without target surfaces or errors."""

    workbook = _require_file(workbook_path, "raw-vol source workbook")
    required_columns = [
        "sample_id",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        "current_weighted_iv_rmse",
        "current_surface_slice_count",
        "pair_quality_label",
        "alignment_type",
        "origin_shift_minutes",
    ]
    if support_artifact_paths:
        required_columns.append("current_surface_param_json")
    else:
        required_columns.extend(
            ["strike_grid", "maturity_days_grid", "current_surface_flat"]
        )
    source = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        usecols=lambda column: column in required_columns,
    )
    missing = sorted(set(required_columns) - set(source.columns))
    optional_alignment_columns = {
        "alignment_type",
        "origin_shift_minutes",
    }
    hard_missing = sorted(set(missing) - optional_alignment_columns)
    if hard_missing:
        raise ValueError(
            f"Raw-vol workbook is missing columns: {hard_missing}"
        )
    if "alignment_type" not in source.columns:
        source["alignment_type"] = "exact"
    if "origin_shift_minutes" not in source.columns:
        source["origin_shift_minutes"] = 0.0
    source = source.copy()
    source["surface_pair_id"] = [
        _surface_pair_id(current, target)
        for current, target in zip(
            source["current_snapshot_time_utc"],
            source["target_snapshot_time_utc"],
        )
    ]
    needed = set(pair_frame["surface_pair_id"].astype(str))
    source = source[source["surface_pair_id"].astype(str).isin(needed)]
    if source.empty:
        raise ValueError("No frozen test pairs matched the raw-vol workbook.")

    artifacts = {
        str(fold): RawSurfaceSupportArtifact.load(
            _require_file(path, f"{fold} raw support artifact")
        )
        for fold, path in (support_artifact_paths or {}).items()
    }
    pair_folds = (
        pair_frame[["fold", "surface_pair_id"]]
        .drop_duplicates()
        .set_index("surface_pair_id")["fold"]
        .astype(str)
    )
    if pair_folds.index.duplicated().any():
        raise ValueError("A surface pair appears in more than one RQ3 fold.")

    rows: list[dict[str, Any]] = []
    for pair_id, group in source.groupby("surface_pair_id", sort=False):
        first = group.iloc[0]
        invariant_columns = [
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
            (
                "current_surface_param_json"
                if artifacts
                else "current_surface_flat"
            ),
        ]
        for column in invariant_columns:
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(
                    f"Surface pair {pair_id} has inconsistent workbook {column}."
                )
        fold = str(pair_folds.loc[str(pair_id)])
        if artifacts and fold not in artifacts:
            raise ValueError(f"Missing raw support artifact for fold={fold}.")
        features = _surface_features(
            first,
            support_artifact=artifacts.get(fold),
        )
        stamp = pd.Timestamp(first["current_snapshot_time_utc"])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        minute = stamp.hour * 60 + stamp.minute + stamp.second / 60.0
        weekday = stamp.weekday()
        shifts = pd.to_numeric(
            group["origin_shift_minutes"],
            errors="coerce",
        ).dropna()
        minimum_shift = float(shifts.min()) if not shifts.empty else 0.0
        if minimum_shift <= 0.0:
            pair_alignment_type = "exact"
        elif minimum_shift <= 15.0:
            pair_alignment_type = "intraday_shift"
        else:
            pair_alignment_type = "session_shift"
        row = {
            "fold": fold,
            "surface_pair_id": str(pair_id),
            "current_snapshot_time_utc": _canonical_timestamp(stamp),
            "target_snapshot_time_utc": _canonical_timestamp(
                first["target_snapshot_time_utc"]
            ),
            "minute_of_day": float(minute),
            "minute_of_day_sin": float(np.sin(2.0 * np.pi * minute / 1440.0)),
            "minute_of_day_cos": float(np.cos(2.0 * np.pi * minute / 1440.0)),
            "weekday": int(weekday),
            "weekday_sin": float(np.sin(2.0 * np.pi * weekday / 7.0)),
            "weekday_cos": float(np.cos(2.0 * np.pi * weekday / 7.0)),
            "current_weighted_iv_rmse": pd.to_numeric(
                pd.Series([first["current_weighted_iv_rmse"]]),
                errors="coerce",
            ).iloc[0],
            "current_surface_slice_count": pd.to_numeric(
                pd.Series([first["current_surface_slice_count"]]),
                errors="coerce",
            ).iloc[0],
            "pair_quality_label": str(first["pair_quality_label"]),
            "alignment_type": pair_alignment_type,
            "origin_shift_minutes_min": minimum_shift,
            "origin_shift_minutes_max": (
                float(shifts.max()) if not shifts.empty else 0.0
            ),
            "origin_shift_minutes_mean": (
                float(shifts.mean()) if not shifts.empty else 0.0
            ),
            "source_row_count": int(len(group)),
            "raw_support_artifact_sha256": (
                _sha256_file(support_artifact_paths[fold])
                if support_artifact_paths
                else ""
            ),
            **features,
        }
        rows.append(row)
    covariates = pd.DataFrame(rows)

    timeline = covariates.sort_values("current_snapshot_time_utc").copy()
    times = pd.to_datetime(
        timeline["current_snapshot_time_utc"],
        utc=True,
    ).astype("int64").to_numpy()
    values = timeline["current_surface_mean"].to_numpy(dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(values * values)])
    horizon_ns = int(pd.Timedelta(hours=24).value)
    recent: list[float] = []
    for index, stamp_ns in enumerate(times):
        left = int(np.searchsorted(times, stamp_ns - horizon_ns, side="left"))
        count = index - left
        if count < 2:
            recent.append(float("nan"))
            continue
        total = prefix[index] - prefix[left]
        total_sq = prefix_sq[index] - prefix_sq[left]
        variance = max(total_sq / count - (total / count) ** 2, 0.0)
        recent.append(float(np.sqrt(variance)))
    timeline["recent_surface_level_std_24h"] = recent
    covariates = covariates.merge(
        timeline[["surface_pair_id", "recent_surface_level_std_24h"]],
        on="surface_pair_id",
        validate="one_to_one",
    )

    lineage = pair_frame[
        ["fold", "surface_pair_id", "news_count"]
    ].drop_duplicates(["fold", "surface_pair_id"])
    covariates = lineage.merge(
        covariates.drop(columns=["fold"]),
        on="surface_pair_id",
        validate="one_to_one",
    )
    if len(covariates) != len(lineage):
        raise ValueError(
            "Not every frozen test pair has forecast-origin workbook covariates."
        )
    return covariates


def label_information_regimes(
    pairs: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    pre_window_minutes: int,
    post_window_minutes: int,
    ordinary_buffer_minutes: int,
) -> pd.DataFrame:
    """Label scheduled-news and ordinary-news candidates."""

    if pre_window_minutes < 0 or post_window_minutes < 0:
        raise ValueError("Event-window minute values must be non-negative.")
    if ordinary_buffer_minutes <= max(pre_window_minutes, post_window_minutes):
        raise ValueError("ordinary_buffer_minutes must exceed the event window.")
    output = pairs.copy()
    timestamps = pd.to_datetime(
        output["current_snapshot_time_utc"],
        utc=True,
    )
    event_times = calendar["release_time"].tolist()
    event_ns = calendar["release_time"].astype("int64").to_numpy()
    stamp_ns = timestamps.astype("int64").to_numpy()
    nearest_minutes = np.full(len(output), np.inf, dtype=np.float64)
    if event_ns.size:
        for index, value in enumerate(stamp_ns):
            position = int(np.searchsorted(event_ns, value))
            candidates = []
            if position < len(event_ns):
                candidates.append(abs(int(event_ns[position]) - int(value)))
            if position > 0:
                candidates.append(abs(int(event_ns[position - 1]) - int(value)))
            nearest_minutes[index] = (
                min(candidates) / float(pd.Timedelta(minutes=1).value)
                if candidates
                else np.inf
            )

    labels: list[dict[str, Any]] = []
    for stamp in timestamps:
        delta_minutes = (
            (stamp - calendar["release_time"]).dt.total_seconds() / 60.0
        )
        matches = calendar[
            (delta_minutes >= -float(pre_window_minutes))
            & (delta_minutes <= float(post_window_minutes))
        ].copy()
        if matches.empty:
            labels.append(
                {
                    "event_id": "",
                    "event_ids": "[]",
                    "event_family": "",
                    "release_name": "",
                    "release_time_utc": "",
                    "release_local_date": "",
                    "release_trading_day": "",
                    "minutes_from_release": np.nan,
                    "overlap_event_count": 0,
                }
            )
            continue
        matches["absolute_delta"] = np.abs(
            (
                stamp - matches["release_time"]
            ).dt.total_seconds()
            / 60.0
        )
        matches["signed_delta"] = (
            (stamp - matches["release_time"]).dt.total_seconds() / 60.0
        )
        matches = matches.sort_values(
            ["priority", "absolute_delta", "event_id"],
            kind="stable",
        )
        selected = matches.iloc[0]
        labels.append(
            {
                "event_id": str(selected["event_id"]),
                "event_ids": json.dumps(
                    matches["event_id"].astype(str).tolist()
                ),
                "event_family": str(selected["event_family"]),
                "release_name": str(selected["release_name"]),
                "release_time_utc": _canonical_timestamp(
                    selected["release_time"]
                ),
                "release_local_date": str(selected["release_local_date"]),
                "release_trading_day": str(
                    selected["release_trading_day"]
                ),
                "minutes_from_release": float(selected["signed_delta"]),
                "overlap_event_count": int(len(matches)),
            }
        )
    label_frame = pd.DataFrame(labels, index=output.index)
    output = pd.concat([output, label_frame], axis=1)
    output["nearest_event_minutes"] = nearest_minutes
    output["regime"] = np.where(
        output["event_id"].astype(str).ne(""),
        "scheduled_news",
        np.where(
            output["nearest_event_minutes"] >= ordinary_buffer_minutes,
            "ordinary_news_candidate",
            "event_buffer_excluded",
        ),
    )
    return output


def _circular_minute_distance(left: float, right: np.ndarray) -> np.ndarray:
    difference = np.abs(right - float(left))
    return np.minimum(difference, 1440.0 - difference)


@dataclass(frozen=True)
class MatchResult:
    manifest: pd.DataFrame
    balance: pd.DataFrame
    unmatched: pd.DataFrame


def _fill_and_scale(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    output = frame.copy()
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(output[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median).astype(float)
        scale = float(filled.std(ddof=0))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        output[column] = filled
        medians[column] = median
        scales[column] = scale
    return output, medians, scales


def _smd(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce").dropna().to_numpy(float)
    right_values = pd.to_numeric(right, errors="coerce").dropna().to_numpy(float)
    if left_values.size == 0 or right_values.size == 0:
        return float("nan")
    pooled = math.sqrt(
        (float(np.var(left_values)) + float(np.var(right_values))) / 2.0
    )
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(left_values) - np.mean(right_values)) / pooled)


def match_scheduled_to_ordinary(
    labeled: pd.DataFrame,
    *,
    time_caliper_minutes: int = 15,
    match_ratio: int = 1,
    exact_weekday: bool = True,
) -> MatchResult:
    """Greedy deterministic no-replacement matching using origin covariates."""

    if match_ratio <= 0:
        raise ValueError("match_ratio must be positive.")
    scheduled = labeled[labeled["regime"] == "scheduled_news"].copy()
    ordinary = labeled[
        labeled["regime"] == "ordinary_news_candidate"
    ].copy()
    if scheduled.empty:
        raise ValueError("No scheduled-news samples fall inside the event window.")
    if ordinary.empty:
        raise ValueError("No ordinary-news controls survive the event buffer.")

    combined = pd.concat([scheduled, ordinary], ignore_index=True)
    combined, medians, scales = _fill_and_scale(combined, MATCH_COVARIATES)
    scheduled = combined.iloc[: len(scheduled)].copy()
    ordinary = combined.iloc[len(scheduled) :].copy()
    ordinary.index = np.arange(len(ordinary))
    available = set(ordinary.index.tolist())

    candidate_counts: dict[int, int] = {}
    for index, event in scheduled.iterrows():
        controls = ordinary[
            (ordinary["fold"].astype(str) == str(event["fold"]))
            & (
                _circular_minute_distance(
                    float(event["minute_of_day"]),
                    ordinary["minute_of_day"].to_numpy(float),
                )
                <= float(time_caliper_minutes)
            )
        ]
        if exact_weekday:
            controls = controls[
                controls["weekday"].astype(int) == int(event["weekday"])
            ]
        candidate_counts[int(index)] = int(len(controls))
    order = sorted(
        scheduled.index,
        key=lambda index: (
            candidate_counts[int(index)],
            str(scheduled.loc[index, "release_time_utc"]),
            str(scheduled.loc[index, "surface_pair_id"]),
        ),
    )

    manifest_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for event_index in order:
        event = scheduled.loc[event_index]
        candidate_indexes = [
            index
            for index in available
            if str(ordinary.loc[index, "fold"]) == str(event["fold"])
        ]
        if candidate_indexes:
            candidate = ordinary.loc[candidate_indexes]
            minute_distance = _circular_minute_distance(
                float(event["minute_of_day"]),
                candidate["minute_of_day"].to_numpy(float),
            )
            candidate = candidate[
                minute_distance <= float(time_caliper_minutes)
            ].copy()
            if exact_weekday:
                candidate = candidate[
                    candidate["weekday"].astype(int) == int(event["weekday"])
                ]
        else:
            candidate = ordinary.iloc[0:0].copy()
        if len(candidate) < match_ratio:
            unmatched_rows.append(
                {
                    "surface_pair_id": str(event["surface_pair_id"]),
                    "fold": str(event["fold"]),
                    "event_id": str(event["event_id"]),
                    "event_family": str(event["event_family"]),
                    "candidate_count_before_no_replacement": candidate_counts[
                        int(event_index)
                    ],
                    "available_candidate_count": int(len(candidate)),
                    "reason": "insufficient_time_caliper_controls",
                }
            )
            continue
        distances = np.zeros(len(candidate), dtype=np.float64)
        for column in MATCH_COVARIATES:
            distances += (
                (
                    candidate[column].to_numpy(dtype=np.float64)
                    - float(event[column])
                )
                / float(scales[column])
            ) ** 2
        distances += (
            _circular_minute_distance(
                float(event["minute_of_day"]),
                candidate["minute_of_day"].to_numpy(float),
            )
            / max(float(time_caliper_minutes), 1.0)
        ) ** 2
        candidate["match_distance"] = np.sqrt(distances)
        candidate = candidate.sort_values(
            ["match_distance", "current_snapshot_time_utc", "surface_pair_id"],
            kind="stable",
        ).head(match_ratio)
        matched_set_id = (
            f"set_{str(event['event_id'])}_"
            f"{str(event['surface_pair_id'])[:12]}"
        )
        for rank, (control_index, control) in enumerate(
            candidate.iterrows(),
            start=1,
        ):
            row: dict[str, Any] = {
                "matched_set_id": matched_set_id,
                "event_surface_pair_id": str(event["surface_pair_id"]),
                "control_surface_pair_id": str(control["surface_pair_id"]),
                "fold": str(event["fold"]),
                "event_id": str(event["event_id"]),
                "event_ids": str(event["event_ids"]),
                "event_family": str(event["event_family"]),
                "release_name": str(event["release_name"]),
                "release_time_utc": str(event["release_time_utc"]),
                "release_local_date": str(event["release_local_date"]),
                "release_trading_day": str(event["release_trading_day"]),
                "event_current_snapshot_time_utc": str(
                    event["current_snapshot_time_utc"]
                ),
                "control_current_snapshot_time_utc": str(
                    control["current_snapshot_time_utc"]
                ),
                "minutes_from_release": float(event["minutes_from_release"]),
                "match_rank": int(rank),
                "match_distance": float(control["match_distance"]),
                "time_distance_minutes": float(
                    _circular_minute_distance(
                        float(event["minute_of_day"]),
                        np.asarray([float(control["minute_of_day"])]),
                    )[0]
                ),
            }
            for column in MATCH_COVARIATES:
                row[f"event_{column}"] = float(event[column])
                row[f"control_{column}"] = float(control[column])
                row[f"difference_{column}"] = float(
                    event[column] - control[column]
                )
            manifest_rows.append(row)
            available.remove(int(control_index))

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        raise ValueError("No scheduled-news sample could be matched.")
    matched_event_ids = set(manifest["event_surface_pair_id"])
    matched_control_ids = set(manifest["control_surface_pair_id"])
    matched_events = scheduled[
        scheduled["surface_pair_id"].astype(str).isin(matched_event_ids)
    ]
    matched_controls = ordinary[
        ordinary["surface_pair_id"].astype(str).isin(matched_control_ids)
    ]
    balance_rows = []
    for column in MATCH_COVARIATES:
        before = _smd(scheduled[column], ordinary[column])
        after = _smd(matched_events[column], matched_controls[column])
        balance_rows.append(
            {
                "covariate": column,
                "event_count_before": int(len(scheduled)),
                "control_count_before": int(len(ordinary)),
                "event_count_after": int(len(matched_events)),
                "control_count_after": int(len(matched_controls)),
                "smd_before": before,
                "smd_after": after,
                "abs_smd_after": abs(after) if np.isfinite(after) else np.nan,
                "flag_abs_smd_over_0_10": bool(
                    np.isfinite(after) and abs(after) > 0.10
                ),
                "imputation_median": medians[column],
                "scaling_std": scales[column],
            }
        )
    unmatched = pd.DataFrame(unmatched_rows)
    return MatchResult(
        manifest=manifest.sort_values(
            ["release_time_utc", "matched_set_id", "match_rank"]
        ).reset_index(drop=True),
        balance=pd.DataFrame(balance_rows),
        unmatched=unmatched,
    )


def build_loss_differentials(samples: pd.DataFrame) -> pd.DataFrame:
    """Create paired baseline-minus-LP loss differentials."""

    if "alignment_type" not in samples.columns:
        samples = samples.copy()
        samples["alignment_type"] = "exact"
    rows: list[pd.DataFrame] = []
    comparisons = [
        *CONTRASTS,
        ("lp", "shuffled_lp", "lp_vs_shuffled_lp", "placebo_text_alignment"),
    ]
    for focal, baseline, contrast, family in comparisons:
        left = samples[samples["model"].astype(str) == focal]
        right = samples[samples["model"].astype(str) == baseline]
        keys = ["fold", "seed", "surface_pair_id"]
        merged = left.merge(
            right,
            on=keys,
            suffixes=("_focal", "_baseline"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Incomplete frozen matching for {contrast}.")
        if not merged["alignment_type_focal"].equals(
            merged["alignment_type_baseline"]
        ):
            raise ValueError(
                f"Alignment-stratum matching failed for {contrast}."
            )
        for metric in POINT_METRICS:
            frame = merged[
                [
                    *keys,
                    "current_snapshot_time_utc_focal",
                    "alignment_type_focal",
                    f"{metric}_focal",
                    f"{metric}_baseline",
                ]
            ].copy()
            frame.columns = [
                *keys,
                "current_snapshot_time_utc",
                "alignment_type",
                "focal_error",
                "baseline_error",
            ]
            frame["contrast"] = contrast
            frame["contrast_family"] = family
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


def build_all_oos_conditional_rows(
    differences: pd.DataFrame,
    covariates: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    windows: Mapping[str, Mapping[str, int]],
    ordinary_buffer_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label every OOS pair and retain both seed and seed-averaged losses."""

    seed_frames: list[pd.DataFrame] = []
    covariate_columns = [
        column
        for column in covariates.columns
        if column
        not in {
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
            "news_count",
        }
    ]
    for window_name, window in windows.items():
        labeled = label_information_regimes(
            covariates,
            calendar,
            pre_window_minutes=int(window["pre_minutes"]),
            post_window_minutes=int(window["post_minutes"]),
            ordinary_buffer_minutes=int(ordinary_buffer_minutes),
        )
        label_columns = [
            *covariate_columns,
            "regime",
            "event_id",
            "event_family",
            "release_time_utc",
            "release_trading_day",
            "minutes_from_release",
            "nearest_event_minutes",
        ]
        frame = differences.merge(
            labeled[label_columns],
            on=["fold", "surface_pair_id"],
            how="left",
            validate="many_to_one",
        )
        if frame["regime"].isna().any():
            raise ValueError(
                f"Missing all-OOS regime labels for window={window_name}."
            )
        frame["window"] = str(window_name)
        frame["scheduled_news_indicator"] = (
            frame["regime"].astype(str).eq("scheduled_news").astype(int)
        )
        current_day = pd.to_datetime(
            frame["current_snapshot_time_utc"],
            utc=True,
        ).dt.date.astype(str)
        release_day = frame["release_trading_day"].fillna("").astype(str)
        frame["inference_cluster_day"] = np.where(
            frame["scheduled_news_indicator"].eq(1)
            & release_day.str.strip().ne(""),
            release_day,
            current_day,
        )
        frame["text_advantage"] = frame["difference"].astype(float)
        seed_frames.append(frame)

    seed_rows = pd.concat(seed_frames, ignore_index=True)
    identity = [
        "window",
        "fold",
        "surface_pair_id",
        "current_snapshot_time_utc",
        "contrast",
        "contrast_family",
        "focal_model",
        "baseline_model",
        "metric",
        "difference_direction",
        "positive_means_focal_better",
        "regime",
        "scheduled_news_indicator",
        "event_id",
        "event_family",
        "release_time_utc",
        "release_trading_day",
        "minutes_from_release",
        "nearest_event_minutes",
        "inference_cluster_day",
    ]
    carried = [
        column
        for column in covariate_columns
        if column not in {"fold", "surface_pair_id"}
    ]
    average_rows = (
        seed_rows.groupby([*identity, *carried], as_index=False, dropna=False)
        .agg(
            text_advantage=("text_advantage", "mean"),
            focal_error=("focal_error", "mean"),
            baseline_error=("baseline_error", "mean"),
            seed_count=("seed", "nunique"),
            positive_seed_count=(
                "text_advantage",
                lambda values: int(
                    np.sum(np.asarray(values, dtype=np.float64) > 0.0)
                ),
            ),
        )
    )
    return seed_rows, average_rows


def _scaled_numeric_controls(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if np.any(finite) else 0.0
        values = np.where(finite, values, fill)
        scale = float(np.std(values))
        if scale <= 1e-10:
            continue
        arrays.append((values - float(np.mean(values))) / scale)
        names.append(str(column))
    if not arrays:
        return np.empty((len(frame), 0), dtype=np.float64), []
    return np.column_stack(arrays), names


def _cluster_robust_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    clusters: Sequence[str],
) -> tuple[np.ndarray, int]:
    cluster_values = np.asarray([str(value) for value in clusters])
    unique_clusters = np.unique(cluster_values)
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for cluster in unique_clusters:
        mask = cluster_values == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n = int(len(residual))
    k = int(x.shape[1])
    g = int(len(unique_clusters))
    if g > 1 and n > k:
        covariance *= (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    return covariance, g


def _all_oos_conditional_regression(
    frame: pd.DataFrame,
    *,
    covariates: Sequence[str] = CONDITIONAL_COVARIATES,
) -> dict[str, Any]:
    """Estimate the support-weighted scheduled-news conditional effect."""

    data = frame[np.isfinite(frame["text_advantage"].to_numpy(float))].copy()
    scheduled = data["scheduled_news_indicator"].to_numpy(dtype=np.float64)
    if len(data) < 4 or np.unique(scheduled).size != 2:
        return {
            "conditional_scheduled_effect": float("nan"),
            "conditional_cluster_se": float("nan"),
            "conditional_t_statistic": float("nan"),
            "conditional_p_two_sided": float("nan"),
            "conditional_p_one_sided_focal_more_valuable": float("nan"),
            "conditional_ci_95_lower": float("nan"),
            "conditional_ci_95_upper": float("nan"),
            "conditional_observations": int(len(data)),
            "conditional_clusters": 0,
            "conditional_rank": 0,
            "event_family_reference": "",
            "event_family_fixed_effect_count": 0,
            "fold_fixed_effect_count": 0,
            "forecast_origin_control_count": 0,
        }

    control_x, control_names = _scaled_numeric_controls(data, covariates)
    fold_frame = pd.get_dummies(
        data["fold"].astype(str),
        prefix="fold",
        drop_first=True,
        dtype=float,
    )
    scheduled_families = sorted(
        {
            str(value)
            for value in data.loc[
                data["scheduled_news_indicator"].eq(1),
                "event_family",
            ]
            if str(value).strip()
        }
    )
    family_reference = scheduled_families[0] if scheduled_families else ""
    family_names = scheduled_families[1:]
    family_x = np.column_stack(
        [
            (
                data["scheduled_news_indicator"].eq(1)
                & data["event_family"].astype(str).eq(family)
            ).to_numpy(dtype=np.float64)
            for family in family_names
        ]
    ) if family_names else np.empty((len(data), 0), dtype=np.float64)
    fold_x = fold_frame.to_numpy(dtype=np.float64)
    x = np.column_stack(
        [
            np.ones(len(data), dtype=np.float64),
            scheduled,
            family_x,
            fold_x,
            control_x,
        ]
    )
    names = [
        "intercept",
        "scheduled_news_indicator",
        *[f"event_family[{value}]" for value in family_names],
        *fold_frame.columns.astype(str).tolist(),
        *[f"control[{value}]" for value in control_names],
    ]
    y = data["text_advantage"].to_numpy(dtype=np.float64)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta
    covariance, cluster_count = _cluster_robust_covariance(
        x,
        residual,
        data["inference_cluster_day"].astype(str).tolist(),
    )

    contrast = np.zeros(x.shape[1], dtype=np.float64)
    contrast[1] = 1.0
    scheduled_count = max(int(np.sum(scheduled)), 1)
    for offset, family in enumerate(family_names, start=2):
        contrast[offset] = float(
            np.sum(
                (scheduled == 1.0)
                & data["event_family"].astype(str).eq(family).to_numpy()
            )
            / scheduled_count
        )
    estimate = float(contrast @ beta)
    variance = float(contrast @ covariance @ contrast)
    standard_error = (
        float(np.sqrt(max(variance, 0.0)))
        if cluster_count > 1
        else float("nan")
    )
    statistic = (
        estimate / standard_error
        if np.isfinite(standard_error) and standard_error > 0.0
        else float("nan")
    )
    distribution = stats.t(df=max(cluster_count - 1, 1))
    critical = float(distribution.ppf(0.975))
    return {
        "conditional_scheduled_effect": estimate,
        "conditional_cluster_se": standard_error,
        "conditional_t_statistic": statistic,
        "conditional_p_two_sided": (
            float(2.0 * distribution.sf(abs(statistic)))
            if np.isfinite(statistic)
            else float("nan")
        ),
        "conditional_p_one_sided_focal_more_valuable": (
            float(distribution.sf(statistic))
            if np.isfinite(statistic)
            else float("nan")
        ),
        "conditional_ci_95_lower": (
            estimate - critical * standard_error
            if np.isfinite(standard_error)
            else float("nan")
        ),
        "conditional_ci_95_upper": (
            estimate + critical * standard_error
            if np.isfinite(standard_error)
            else float("nan")
        ),
        "conditional_observations": int(len(data)),
        "conditional_clusters": int(cluster_count),
        "conditional_rank": int(np.linalg.matrix_rank(x)),
        "conditional_design_columns": json.dumps(names),
        "event_family_reference": family_reference,
        "event_family_fixed_effect_count": int(len(family_names)),
        "fold_fixed_effect_count": int(fold_x.shape[1]),
        "forecast_origin_control_count": int(control_x.shape[1]),
    }


def _newey_west_long_run_covariance(
    values: np.ndarray,
    *,
    max_lag: int,
) -> tuple[np.ndarray, int]:
    matrix = np.asarray(values, dtype=np.float64)
    count = int(matrix.shape[0])
    if count < 2:
        return np.full((matrix.shape[1], matrix.shape[1]), np.nan), 0
    lag_limit = min(int(max_lag), count - 1)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    covariance = centered.T @ centered / count
    for lag in range(1, lag_limit + 1):
        gamma = centered[lag:].T @ centered[:-lag] / count
        weight = 1.0 - lag / float(lag_limit + 1)
        covariance += weight * (gamma + gamma.T)
    return covariance, lag_limit


def _giacomini_white_test(
    frame: pd.DataFrame,
    *,
    max_lag: int = 5,
    covariates: Sequence[str] = CONDITIONAL_COVARIATES,
) -> dict[str, Any]:
    """Test E[h_t * loss_difference_t] = 0 using daily HAC moments."""

    data = frame[np.isfinite(frame["text_advantage"].to_numpy(float))].copy()
    control_x, control_names = _scaled_numeric_controls(data, covariates)
    fold_x = pd.get_dummies(
        data["fold"].astype(str),
        prefix="fold",
        drop_first=True,
        dtype=float,
    ).to_numpy(dtype=np.float64)
    instruments = np.column_stack(
        [
            np.ones(len(data), dtype=np.float64),
            data["scheduled_news_indicator"].to_numpy(dtype=np.float64),
            control_x,
            fold_x,
        ]
    )
    instrument_names = [
        "intercept",
        "scheduled_news_indicator",
        *[f"control[{value}]" for value in control_names],
        *[
            f"fold_instrument_{index + 1}"
            for index in range(fold_x.shape[1])
        ],
    ]
    moments = instruments * data["text_advantage"].to_numpy(float)[:, None]
    moment_frame = pd.DataFrame(moments, columns=instrument_names)
    moment_frame["cluster_day"] = data["inference_cluster_day"].astype(str).to_numpy()
    daily = (
        moment_frame.groupby("cluster_day", sort=True)[instrument_names]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    long_run, lag_limit = _newey_west_long_run_covariance(
        daily,
        max_lag=max_lag,
    )
    if len(daily) < 2 or not np.all(np.isfinite(long_run)):
        return {
            "gw_statistic": float("nan"),
            "gw_df": 0,
            "gw_p_value": float("nan"),
            "gw_daily_observations": int(len(daily)),
            "gw_hac_max_lag": int(lag_limit),
            "gw_instruments": json.dumps(instrument_names),
        }
    mean_moment = np.mean(daily, axis=0)
    rank = int(np.linalg.matrix_rank(long_run, tol=1e-10))
    statistic = float(
        len(daily) * mean_moment @ np.linalg.pinv(long_run) @ mean_moment
    )
    return {
        "gw_statistic": statistic,
        "gw_df": rank,
        "gw_p_value": (
            float(stats.chi2.sf(statistic, df=rank))
            if rank > 0
            else float("nan")
        ),
        "gw_daily_observations": int(len(daily)),
        "gw_hac_max_lag": int(lag_limit),
        "gw_instruments": json.dumps(instrument_names),
    }


def build_all_oos_inference(
    average_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build thesis-facing all-OOS conditional predictive-ability results."""

    rows: list[dict[str, Any]] = []
    grouping = [
        "window",
        "contrast",
        "contrast_family",
        "focal_model",
        "baseline_model",
        "metric",
    ]
    for keys, group in average_rows.groupby(grouping, sort=True):
        (
            window,
            contrast,
            contrast_family,
            focal_model,
            baseline_model,
            metric,
        ) = keys
        selected_seed_rows = seed_rows[
            seed_rows["window"].astype(str).eq(str(window))
            & seed_rows["contrast"].astype(str).eq(str(contrast))
            & seed_rows["metric"].astype(str).eq(str(metric))
        ]
        seed_effects = []
        for _seed, seed_group in selected_seed_rows.groupby("seed", sort=True):
            scheduled_values = seed_group.loc[
                seed_group["scheduled_news_indicator"].eq(1),
                "text_advantage",
            ]
            other_values = seed_group.loc[
                seed_group["scheduled_news_indicator"].eq(0),
                "text_advantage",
            ]
            seed_effects.append(
                float(scheduled_values.mean() - other_values.mean())
                if len(scheduled_values) and len(other_values)
                else float("nan")
            )
        scheduled_values = group.loc[
            group["scheduled_news_indicator"].eq(1),
            "text_advantage",
        ]
        other_values = group.loc[
            group["scheduled_news_indicator"].eq(0),
            "text_advantage",
        ]
        rows.append(
            {
                "analysis_type": "all_oos_conditional_predictive_robustness",
                "window": str(window),
                "contrast": str(contrast),
                "contrast_family": str(contrast_family),
                "focal_model": str(focal_model),
                "baseline_model": str(baseline_model),
                "metric": str(metric),
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                "all_oos_text_advantage_mean": float(
                    group["text_advantage"].mean()
                ),
                "scheduled_text_advantage_mean": float(
                    scheduled_values.mean()
                ),
                "non_scheduled_text_advantage_mean": float(
                    other_values.mean()
                ),
                "unadjusted_scheduled_increment": float(
                    scheduled_values.mean() - other_values.mean()
                ),
                "oos_pair_count": int(group["surface_pair_id"].nunique()),
                "scheduled_pair_count": int(
                    group.loc[
                        group["scheduled_news_indicator"].eq(1),
                        "surface_pair_id",
                    ].nunique()
                ),
                "scheduled_release_count": int(
                    group.loc[
                        group["scheduled_news_indicator"].eq(1),
                        "event_id",
                    ].nunique()
                ),
                "scheduled_release_day_count": int(
                    group.loc[
                        group["scheduled_news_indicator"].eq(1),
                        "inference_cluster_day",
                    ].nunique()
                ),
                "seed_count": int(
                    selected_seed_rows["seed"].nunique()
                ),
                "positive_seed_scheduled_increment_count": int(
                    np.sum(np.asarray(seed_effects, dtype=float) > 0.0)
                ),
                "seed_scheduled_increments": json.dumps(seed_effects),
                **_all_oos_conditional_regression(group),
                **_giacomini_white_test(group, max_lag=5),
            }
        )
    output = pd.DataFrame(rows)
    output["holm_family"] = ""
    output["conditional_p_holm_two_sided"] = np.nan
    representation_mask = (
        output["window"].eq("primary_0_5")
        & output["metric"].eq("surface_mae")
        & output["contrast"].isin(["lp_vs_bow", "lp_vs_llm_sentiment"])
    )
    output.loc[
        representation_mask,
        "holm_family",
    ] = "rq3_secondary_representation_surface"
    output.loc[
        representation_mask,
        "conditional_p_holm_two_sided",
    ] = _holm_adjust(
        output.loc[
            representation_mask,
            "conditional_p_two_sided",
        ].tolist()
    )
    return output


def build_overall_dm_hac(differences: pd.DataFrame) -> pd.DataFrame:
    """Run unconditional equal-accuracy diagnostics on all OOS pairs."""

    pair_average = (
        differences.groupby(
            [
                "fold",
                "surface_pair_id",
                "current_snapshot_time_utc",
                "contrast",
                "contrast_family",
                "focal_model",
                "baseline_model",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            text_advantage=("difference", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    pair_average["trading_day"] = pd.to_datetime(
        pair_average["current_snapshot_time_utc"],
        utc=True,
    ).dt.date.astype(str)
    rows: list[dict[str, Any]] = []
    grouping = [
        "contrast",
        "contrast_family",
        "focal_model",
        "baseline_model",
        "metric",
    ]
    for keys, group in pair_average.groupby(grouping, sort=True):
        daily = (
            group.groupby("trading_day", sort=True)["text_advantage"]
            .mean()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "contrast": keys[0],
                "contrast_family": keys[1],
                "focal_model": keys[2],
                "baseline_model": keys[3],
                "metric": keys[4],
                "difference_direction": "baseline_minus_focal",
                "positive_means_focal_better": True,
                "mean_text_advantage": float(
                    group["text_advantage"].mean()
                ),
                "pair_count": int(group["surface_pair_id"].nunique()),
                **_dm_hac(daily, max_lag=5),
            }
        )
    return pd.DataFrame(rows)


def build_matched_set_differences(
    differences: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    window_name: str,
    analysis_type: str = "scheduled_news",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one matching manifest to all seeds, contrasts, and metrics."""

    seed_rows: list[dict[str, Any]] = []
    keys = ["fold", "seed", "surface_pair_id", "contrast", "metric"]
    indexed = differences.set_index(keys)
    contrast_meta = (
        differences[
            [
                "contrast",
                "contrast_family",
                "focal_model",
                "baseline_model",
            ]
        ]
        .drop_duplicates("contrast")
        .set_index("contrast")
    )
    seeds = sorted(int(value) for value in differences["seed"].unique())
    contrast_metrics = differences[
        ["contrast", "metric"]
    ].drop_duplicates()
    for matched in manifest.itertuples(index=False):
        for item in contrast_metrics.itertuples(index=False):
            contrast = str(item.contrast)
            metric = str(item.metric)
            metadata = contrast_meta.loc[contrast]
            for seed in seeds:
                event_key = (
                    str(matched.fold),
                    seed,
                    str(matched.event_surface_pair_id),
                    contrast,
                    metric,
                )
                control_key = (
                    str(matched.fold),
                    seed,
                    str(matched.control_surface_pair_id),
                    contrast,
                    metric,
                )
                try:
                    event_row = indexed.loc[event_key]
                    control_row = indexed.loc[control_key]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing frozen differential for matched set "
                        f"{matched.matched_set_id}: {exc}"
                    ) from exc
                seed_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "window": window_name,
                        "matched_set_id": str(matched.matched_set_id),
                        "fold": str(matched.fold),
                        "seed": seed,
                        "event_id": str(matched.event_id),
                        "event_family": str(matched.event_family),
                        "release_time_utc": str(matched.release_time_utc),
                        "release_local_date": str(matched.release_local_date),
                        "release_trading_day": str(
                            matched.release_trading_day
                        ),
                        "event_surface_pair_id": str(
                            matched.event_surface_pair_id
                        ),
                        "control_surface_pair_id": str(
                            matched.control_surface_pair_id
                        ),
                        "contrast": contrast,
                        "contrast_family": str(
                            metadata["contrast_family"]
                        ),
                        "focal_model": str(metadata["focal_model"]),
                        "baseline_model": str(metadata["baseline_model"]),
                        "metric": metric,
                        "event_text_advantage": float(
                            event_row["difference"]
                        ),
                        "control_text_advantage": float(
                            control_row["difference"]
                        ),
                        "scheduled_news_increment": float(
                            event_row["difference"]
                            - control_row["difference"]
                        ),
                        "difference_direction": (
                            "event_text_advantage_minus_control_text_advantage"
                        ),
                        "positive_means_focal_more_valuable_in_event": True,
                    }
                )
    seed_frame = pd.DataFrame(seed_rows)
    average = (
        seed_frame.groupby(
            [
                "analysis_type",
                "window",
                "matched_set_id",
                "fold",
                "event_id",
                "event_family",
                "release_time_utc",
                "release_local_date",
                "release_trading_day",
                "event_surface_pair_id",
                "control_surface_pair_id",
                "contrast",
                "contrast_family",
                "focal_model",
                "baseline_model",
                "metric",
                "difference_direction",
                "positive_means_focal_more_valuable_in_event",
            ],
            as_index=False,
        )
        .agg(
            event_text_advantage=("event_text_advantage", "mean"),
            control_text_advantage=("control_text_advantage", "mean"),
            scheduled_news_increment=("scheduled_news_increment", "mean"),
            seed_count=("seed", "nunique"),
            positive_seed_count=(
                "scheduled_news_increment",
                lambda values: int(np.sum(np.asarray(values) > 0.0)),
            ),
        )
    )
    return seed_frame, average


def _cluster_bootstrap(
    values: np.ndarray,
    clusters: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "cluster": clusters}
    ).dropna()
    grouped = frame.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    if grouped.empty:
        raise ValueError("Cluster bootstrap received no observations.")
    observed = float(frame["value"].mean())
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    cluster_count = len(grouped)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(
        0,
        cluster_count,
        size=(int(iterations), cluster_count),
    )
    draws = np.sum(sums[selected], axis=1) / np.sum(counts[selected], axis=1)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    null_draws = draws - observed
    p_two = (float(np.sum(np.abs(null_draws) >= abs(observed))) + 1.0) / (
        float(iterations) + 1.0
    )
    p_one = (float(np.sum(null_draws >= observed)) + 1.0) / (
        float(iterations) + 1.0
    )
    return {
        "mean_scheduled_news_increment": observed,
        "ci_95_lower": float(ci_low),
        "ci_95_upper": float(ci_high),
        "p_two_sided_centered_cluster_bootstrap": p_two,
        "p_one_sided_focal_more_valuable": p_one,
        "observation_count": int(len(frame)),
        "cluster_count": int(cluster_count),
    }


def _cluster_sign_randomization(
    values: np.ndarray,
    clusters: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "cluster": clusters}
    ).dropna()
    grouped = frame.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    total_count = float(grouped["count"].sum())
    observed = float(frame["value"].mean())
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(int(iterations), len(grouped)),
    )
    draws = signs @ sums / total_count
    return {
        "randomization_p_two_sided": (
            float(np.sum(np.abs(draws) >= abs(observed))) + 1.0
        )
        / (float(iterations) + 1.0),
        "randomization_p_one_sided_focal_more_valuable": (
            float(np.sum(draws >= observed)) + 1.0
        )
        / (float(iterations) + 1.0),
    }


def _dm_hac(values: np.ndarray, *, max_lag: int = 5) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = len(array)
    if count < 2:
        return {
            "daily_observations": count,
            "hac_max_lag": 0,
            "dm_hac_statistic": float("nan"),
            "dm_hac_p_two_sided": float("nan"),
            "dm_hac_p_one_sided": float("nan"),
        }
    lag_limit = min(int(max_lag), count - 1)
    centered = array - float(np.mean(array))
    long_run = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_limit + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run += (
            2.0
            * (1.0 - lag / float(lag_limit + 1))
            * covariance
        )
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    statistic = (
        float(np.mean(array)) / standard_error
        if standard_error > 0.0
        else float("nan")
    )
    distribution = stats.t(df=count - 1)
    return {
        "daily_observations": count,
        "hac_max_lag": lag_limit,
        "dm_hac_statistic": statistic,
        "dm_hac_p_two_sided": (
            float(2.0 * distribution.sf(abs(statistic)))
            if np.isfinite(statistic)
            else float("nan")
        ),
        "dm_hac_p_one_sided": (
            float(distribution.sf(statistic))
            if np.isfinite(statistic)
            else float("nan")
        ),
    }


def _conditional_ols(
    set_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    covariates: Sequence[str] = CONDITIONAL_COVARIATES,
) -> dict[str, float | int]:
    """Matched-set conditional predictive regression with cluster-robust SE."""

    differences = manifest[
        ["matched_set_id", *[f"difference_{item}" for item in covariates]]
    ].drop_duplicates("matched_set_id")
    data = set_frame.merge(
        differences,
        on="matched_set_id",
        validate="one_to_one",
    )
    y = data["scheduled_news_increment"].to_numpy(dtype=np.float64)
    raw_x = data[
        [f"difference_{item}" for item in covariates]
    ].to_numpy(dtype=np.float64)
    raw_x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_x.size:
        scale = np.std(raw_x, axis=0)
        keep = scale > 1e-10
        raw_x = raw_x[:, keep]
        if raw_x.size:
            raw_x = raw_x / scale[keep]
    fold_dummies = pd.get_dummies(
        data["fold"].astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy(dtype=np.float64)
    x = np.column_stack([np.ones(len(y)), raw_x, fold_dummies])
    if len(y) <= x.shape[1]:
        x = np.ones((len(y), 1), dtype=np.float64)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta
    clusters = data["release_trading_day"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n = len(y)
    k = x.shape[1]
    g = len(unique_clusters)
    if g > 1 and n > k:
        covariance *= (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    standard_error = (
        float(np.sqrt(max(covariance[0, 0], 0.0)))
        if g > 1
        else float("nan")
    )
    statistic = (
        float(beta[0]) / standard_error
        if standard_error > 0.0
        else float("nan")
    )
    distribution = stats.t(df=max(g - 1, 1))
    return {
        "conditional_intercept": float(beta[0]),
        "conditional_cluster_se": standard_error,
        "conditional_t_statistic": statistic,
        "conditional_p_two_sided": (
            float(2.0 * distribution.sf(abs(statistic)))
            if np.isfinite(statistic)
            else float("nan")
        ),
        "conditional_p_one_sided_focal_more_valuable": (
            float(distribution.sf(statistic))
            if np.isfinite(statistic)
            else float("nan")
        ),
        "conditional_control_count": int(x.shape[1] - 1),
        "conditional_origin_covariate_count": int(raw_x.shape[1]),
        "conditional_fold_fixed_effect_count": int(fold_dummies.shape[1]),
        "conditional_observations": int(n),
        "conditional_clusters": int(g),
    }


def _holm_adjust(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    output = np.full(len(array), np.nan, dtype=np.float64)
    if not np.any(finite):
        return output.tolist()
    indexes = np.flatnonzero(finite)
    order = indexes[np.argsort(array[indexes])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (count - rank) * float(array[index])),
        )
        output[index] = running
    return output.tolist()


def build_inference(
    set_average: pd.DataFrame,
    seed_sets: pd.DataFrame,
    manifest_by_window: Mapping[str, pd.DataFrame],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inference_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    grouping = [
        "analysis_type",
        "window",
        "contrast",
        "contrast_family",
        "focal_model",
        "baseline_model",
        "metric",
    ]
    for keys, group in set_average.groupby(grouping, sort=True):
        (
            analysis_type,
            window,
            contrast,
            contrast_family,
            focal,
            baseline,
            metric,
        ) = keys
        stable_seed = (
            int(bootstrap_seed)
            + _stable_offset(
                f"{analysis_type}|{window}|{contrast}|{metric}"
            )
        )
        bootstrap = _cluster_bootstrap(
            group["scheduled_news_increment"].to_numpy(float),
            group["release_trading_day"].astype(str).tolist(),
            iterations=int(bootstrap_iterations),
            seed=stable_seed,
        )
        randomization = _cluster_sign_randomization(
            group["scheduled_news_increment"].to_numpy(float),
            group["release_trading_day"].astype(str).tolist(),
            iterations=int(bootstrap_iterations),
            seed=stable_seed + 1,
        )
        daily = (
            group.groupby("release_trading_day", sort=True)[
                "scheduled_news_increment"
            ]
            .mean()
            .to_numpy(float)
        )
        base = {
            "analysis_type": analysis_type,
            "window": window,
            "contrast": contrast,
            "contrast_family": contrast_family,
            "focal_model": focal,
            "baseline_model": baseline,
            "metric": metric,
            "difference_direction": (
                "event_text_advantage_minus_control_text_advantage"
            ),
            "positive_means_focal_more_valuable_in_event": True,
            "matched_set_count": int(group["matched_set_id"].nunique()),
            "release_count": int(group["event_id"].nunique()),
            "event_family_count": int(group["event_family"].nunique()),
            "positive_seed_direction_count": int(
                np.sum(
                    seed_sets[
                        seed_sets["analysis_type"].astype(str).eq(
                            str(analysis_type)
                        )
                        & seed_sets["window"].astype(str).eq(str(window))
                        & seed_sets["contrast"].astype(str).eq(str(contrast))
                        & seed_sets["metric"].astype(str).eq(str(metric))
                    ]
                    .groupby("seed")["scheduled_news_increment"]
                    .mean()
                    .to_numpy(dtype=np.float64)
                    > 0.0
                )
            ),
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
            **bootstrap,
            **randomization,
            **_dm_hac(daily, max_lag=5),
        }
        inference_rows.append(base)
        manifest = manifest_by_window[str(window)]
        conditional_rows.append(
            {
                **{
                    key: base[key]
                    for key in (
                        "analysis_type",
                        "window",
                        "contrast",
                        "contrast_family",
                        "focal_model",
                        "baseline_model",
                        "metric",
                        "difference_direction",
                    )
                },
                **_conditional_ols(group, manifest),
            }
        )
    inference = pd.DataFrame(inference_rows)
    inference["holm_family"] = ""
    inference["p_holm_two_sided"] = np.nan
    secondary_metric_mask = (
        inference["analysis_type"].eq("scheduled_news")
        & inference["window"].eq("primary_0_5")
        & inference["metric"].isin(
            ["short_atm_mae", "supported_shortest_atm_abs_err"]
        )
        & inference["contrast"].isin(
            ["lp_vs_continued_no_text", "lp_vs_bow", "lp_vs_llm_sentiment"]
        )
    )
    for metric in ("short_atm_mae", "supported_shortest_atm_abs_err"):
        mask = secondary_metric_mask & inference["metric"].eq(metric)
        inference.loc[mask, "holm_family"] = f"secondary_metric_{metric}"
        inference.loc[mask, "p_holm_two_sided"] = _holm_adjust(
            inference.loc[
                mask, "p_two_sided_centered_cluster_bootstrap"
            ].tolist()
        )
    representation_mask = (
        inference["analysis_type"].eq("scheduled_news")
        & inference["window"].eq("primary_0_5")
        & inference["metric"].eq("surface_mae")
        & inference["contrast"].isin(["lp_vs_bow", "lp_vs_llm_sentiment"])
    )
    inference.loc[
        representation_mask, "holm_family"
    ] = "secondary_representation_surface"
    inference.loc[
        representation_mask, "p_holm_two_sided"
    ] = _holm_adjust(
        inference.loc[
            representation_mask,
            "p_two_sided_centered_cluster_bootstrap",
        ].tolist()
    )
    return inference, pd.DataFrame(conditional_rows)


def build_fomc_case_study(
    seed_sets: pd.DataFrame,
    average_sets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_seed = seed_sets[
        seed_sets["event_family"].astype(str).eq("FOMC")
    ].copy()
    raw_average = average_sets[
        average_sets["event_family"].astype(str).eq("FOMC")
    ].copy()
    if raw_average.empty:
        return raw_seed, pd.DataFrame()
    seed_frame = (
        raw_seed.groupby(
            [
                "window",
                "event_id",
                "event_family",
                "release_time_utc",
                "release_local_date",
                "fold",
                "seed",
                "contrast",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mean_scheduled_news_increment=(
                "scheduled_news_increment",
                "mean",
            ),
            matched_set_count=("matched_set_id", "nunique"),
        )
    )
    average = (
        raw_average.groupby(
            [
                "window",
                "event_id",
                "event_family",
                "release_time_utc",
                "release_local_date",
                "fold",
                "contrast",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            scheduled_news_increment=(
                "scheduled_news_increment",
                "mean",
            ),
            matched_set_count=("matched_set_id", "nunique"),
        )
    )
    rows: list[dict[str, Any]] = []
    grouping = ["window", "contrast", "metric"]
    for keys, group in average.groupby(grouping, sort=True):
        window, contrast, metric = keys
        values = group["scheduled_news_increment"].to_numpy(float)
        nonzero = values[~np.isclose(values, 0.0)]
        sign_p = (
            float(
                stats.binomtest(
                    int(np.sum(nonzero > 0.0)),
                    n=len(nonzero),
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if len(nonzero)
            else 1.0
        )
        leave_one_out = [
            float(np.mean(np.delete(values, index)))
            for index in range(len(values))
            if len(values) > 1
        ]
        rows.append(
            {
                "window": window,
                "contrast": contrast,
                "metric": metric,
                "meeting_count": int(group["event_id"].nunique()),
                "matched_set_count": int(group["matched_set_count"].sum()),
                "mean_scheduled_news_increment": float(np.mean(values)),
                "positive_meeting_count": int(np.sum(values > 0.0)),
                "exact_sign_p_two_sided": sign_p,
                "leave_one_out_min": (
                    float(np.min(leave_one_out))
                    if leave_one_out
                    else float("nan")
                ),
                "leave_one_out_max": (
                    float(np.max(leave_one_out))
                    if leave_one_out
                    else float("nan")
                ),
                "evidence_status": "development_underpowered_case_study",
            }
        )
    return seed_frame, pd.DataFrame(rows)


def build_event_family_summary(average_sets: pd.DataFrame) -> pd.DataFrame:
    return (
        average_sets.groupby(
            [
                "analysis_type",
                "window",
                "event_family",
                "contrast",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mean_scheduled_news_increment=(
                "scheduled_news_increment",
                "mean",
            ),
            matched_set_count=("matched_set_id", "nunique"),
            release_count=("event_id", "nunique"),
            positive_set_count=(
                "scheduled_news_increment",
                lambda values: int(np.sum(np.asarray(values) > 0.0)),
            ),
        )
    )


def build_persistence_context(
    samples: pd.DataFrame,
    labeled_primary: pd.DataFrame,
    primary_manifest: pd.DataFrame,
) -> pd.DataFrame:
    membership = {}
    for row in primary_manifest.itertuples(index=False):
        membership[str(row.event_surface_pair_id)] = "scheduled_news"
        membership[str(row.control_surface_pair_id)] = "matched_ordinary_news"
    frame = samples[
        samples["model"].astype(str).isin(REQUIRED_MODELS)
    ].copy()
    frame["regime"] = frame["surface_pair_id"].astype(str).map(membership)
    frame = frame[frame["regime"].notna()]
    rows = []
    for keys, group in frame.groupby(["model", "regime"], sort=True):
        model, regime = keys
        for metric in POINT_METRICS:
            current_metric = CURRENT_METRICS[metric]
            difference = (
                group[current_metric].to_numpy(float)
                - group[metric].to_numpy(float)
            )
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "metric": metric,
                    "mean_model_error": float(group[metric].mean()),
                    "mean_current_error": float(group[current_metric].mean()),
                    "mean_current_minus_model": float(np.mean(difference)),
                    "positive_means_model_better_than_persistence": True,
                    "sample_seed_rows": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def _shift_calendar(calendar: pd.DataFrame, business_days: int) -> pd.DataFrame:
    shifted = calendar.copy()
    shifted["release_time"] = shifted["release_time"] + pd.offsets.BDay(
        int(business_days)
    )
    shifted["release_time_utc"] = shifted["release_time"].map(
        _canonical_timestamp
    )
    shifted["event_id"] = shifted["event_id"].astype(str).map(
        lambda value: f"placebo_shift_{business_days:+d}_{value}"
    )
    shifted["event_family"] = shifted["event_family"].astype(str).map(
        lambda value: f"PLACEBO_SHIFT_{business_days:+d}_{value}"
    )
    shifted["release_local_date"] = (
        shifted["release_time"].dt.date.astype(str)
    )
    shifted["release_trading_day"] = shifted["release_local_date"]
    return shifted


def _same_clock_placebo_labeled(
    labeled_primary: pd.DataFrame,
    primary_manifest: pd.DataFrame,
) -> pd.DataFrame:
    output = labeled_primary.copy()
    output["event_id"] = ""
    output["event_ids"] = "[]"
    output["event_family"] = ""
    output["release_name"] = ""
    output["release_time_utc"] = ""
    output["release_local_date"] = ""
    output["release_trading_day"] = ""
    output["minutes_from_release"] = np.nan
    output["overlap_event_count"] = 0
    output["regime"] = np.where(
        output["regime"].eq("ordinary_news_candidate"),
        "ordinary_news_candidate",
        "event_buffer_excluded",
    )
    control_map = (
        primary_manifest[
            [
                "control_surface_pair_id",
                "event_id",
                "event_family",
            ]
        ]
        .drop_duplicates("control_surface_pair_id")
        .set_index("control_surface_pair_id")
    )
    mask = output["surface_pair_id"].astype(str).isin(control_map.index)
    for index in output.index[mask]:
        pair_id = str(output.at[index, "surface_pair_id"])
        source = control_map.loc[pair_id]
        stamp = _canonical_timestamp(output.at[index, "current_snapshot_time_utc"])
        output.at[index, "event_id"] = f"same_clock_{source['event_id']}_{pair_id[:8]}"
        output.at[index, "event_ids"] = json.dumps(
            [output.at[index, "event_id"]]
        )
        output.at[index, "event_family"] = "PLACEBO_SAME_CLOCK"
        output.at[index, "release_name"] = "Same-clock ordinary-news placebo"
        output.at[index, "release_time_utc"] = stamp
        output.at[index, "release_local_date"] = stamp[:10]
        output.at[index, "release_trading_day"] = stamp[:10]
        output.at[index, "minutes_from_release"] = 0.0
        output.at[index, "overlap_event_count"] = 1
        output.at[index, "regime"] = "scheduled_news"
    used_controls = set(primary_manifest["control_surface_pair_id"].astype(str))
    event_ids = set(primary_manifest["event_surface_pair_id"].astype(str))
    exclusion = used_controls | event_ids
    candidate_mask = (
        output["regime"].eq("ordinary_news_candidate")
        & ~output["surface_pair_id"].astype(str).isin(exclusion)
    )
    output.loc[
        output["regime"].eq("ordinary_news_candidate") & ~candidate_mask,
        "regime",
    ] = "event_buffer_excluded"
    return output


def _archive_inputs(
    output_root: Path,
    *,
    config_path: Path,
    rq1_path: Path,
    rq2_path: Path,
    workbook_path: Path,
    calendar_path: Path,
    support_artifact_paths: Mapping[str, Path],
) -> pd.DataFrame:
    specs = [
        ("config", config_path, output_root / "inputs/configs/source_config.yaml"),
        (
            "rq1_frozen_predictions",
            rq1_path,
            output_root / "inputs/frozen_predictions/rq1_test_metrics.csv",
        ),
        (
            "rq2_frozen_predictions",
            rq2_path,
            output_root / "inputs/frozen_predictions/rq2_test_metrics.csv",
        ),
        (
            "raw_vol_workbook",
            workbook_path,
            output_root / "inputs/data/merged_vol_rq2_text.xlsx",
        ),
        (
            "event_calendar",
            calendar_path,
            output_root / "inputs/event_calendars/scheduled_macro_events.csv",
        ),
    ]
    for fold, source in sorted(support_artifact_paths.items()):
        specs.append(
            (
                f"raw_support_artifact_{fold}",
                source,
                output_root / f"inputs/support_artifacts/{fold}.json",
            )
        )
    rows = []
    for category, source, target in specs:
        method = _link_or_copy(source, target)
        rows.append(
            {
                "category": category,
                "source_path": str(source),
                "archived_path": str(target),
                "import_method": method,
                "size_bytes": int(source.stat().st_size),
                "sha256": _sha256_file(source),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "inputs/source_prediction_manifest.csv", index=False)
    return frame


def _git_state() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"commit={commit}\n{status}"


def _write_documentation(output_root: Path, config: Mapping[str, Any]) -> None:
    readme = f"""# RQ3 Conditional Predictive Robustness

This archive is a frozen-model post-processing analysis of the completed
raw-vol RQ1/RQ2 rolling-development predictions.

Primary contrast:

```text
text_advantage = no_text_error - LP_error

text_advantage ~ scheduled_news_indicator
                 + event-family fixed effects
                 + fold fixed effects
                 + forecast-origin controls

positive conditional scheduled effect
=> LP is relatively more valuable during scheduled-news windows
```

Primary metric: `surface_mae`

Primary window: `[0,+5]` minutes relative to a frozen scheduled release.

All out-of-sample pairs enter the primary conditional regression. Inference is
clustered by release/trading day and accompanied by a Giacomini-White
conditional predictive-ability test and an overall Diebold-Mariano/HAC test.
Greedy scheduled-vs-ordinary matching is retained only as secondary robustness.

All forecast-origin surface controls are reconstructed on each fold's frozen,
train-only raw-vol support grid. The unsupported raw-vol 7-day ATM metric is
not reported; the supported shortest-maturity ATM error is used instead.

No model is retrained and no seed, checkpoint, event window, or event family is
selected using RQ3 errors. This is predictive-regime evidence, not a causal
event-study estimate. FOMC results remain meeting-level development evidence.

Bootstrap iterations: {int(config['bootstrap_iterations'])}
Bootstrap seed: {int(config['bootstrap_seed'])}
"""
    methodology = """# Methodology

1. Validate RQ1 and RQ2 duplicated LP/no-text predictions at tolerance 1e-8.
2. Validate matching fold-train raw-support artifacts from RQ1 and RQ2.
3. Reconstruct forecast-origin covariates only inside observed raw support.
4. Freeze scheduled-event labels before joining forecast errors.
5. Calculate baseline-minus-LP loss differences for all OOS pairs and seeds.
6. Average differences across the three registered seeds at pair level.
7. Estimate the scheduled-news coefficient with event-family and fold fixed
   effects plus forecast-origin controls and release/trading-day clustered SE.
8. Report the Giacomini-White HAC moment test and overall DM/HAC diagnostic.
9. As secondary robustness, define ordinary news as at least 60 minutes from
   scheduled releases and match within fold/time/weekday without replacement.
10. Report FOMC by meeting and date-shift, same-clock, and shuffled-text
    placebos.

Primary metrics are `surface_mae`, `short_atm_mae`, and
`supported_shortest_atm_abs_err`. Raw-vol `atm7_abs_err` is deliberately
excluded because seven days falls below the empirical maturity support.

The design is a conditional predictive-ability analysis. It does not identify
the causal effect of a release or its semantic content. A causal official
release study would require pre/post official-release surfaces and an external
surprise measure.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    docs = output_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "methodology.md").write_text(methodology, encoding="utf-8")


def _build_manifest(output_root: Path) -> Path:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    target = output_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def _status_path(output_root: Path) -> Path:
    return output_root / "registry/pipeline_status.json"


def _update_status(
    output_root: Path,
    *,
    status: str,
    phase: str,
    started_at_utc: str,
    error: BaseException | None = None,
) -> None:
    now = _iso_now()
    _write_json(
        _status_path(output_root),
        {
            "status": status,
            "phase": phase,
            "started_at_utc": started_at_utc,
            "updated_at_utc": now,
            "finished_at_utc": now if status in {"completed", "failed"} else None,
            "output_root": str(output_root),
            "error_type": type(error).__name__ if error else "",
            "error": str(error) if error else "",
        },
    )


def run_scheduled_news_regime(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    rq1_experiment_override: str | Path | None = None,
    rq2_experiment_override: str | Path | None = None,
    event_calendar_override: str | Path | None = None,
) -> Path:
    """Run the complete frozen-prediction RQ3 archive pipeline."""

    _assert_py312()
    source_config = _require_file(config_path, "RQ3 config")
    payload = _read_yaml(source_config)
    config = payload.get("rq3", payload)
    if not isinstance(config, dict):
        raise ValueError("RQ3 config must contain a mapping.")

    rq1_value = rq1_experiment_override or config.get("rq1_experiment")
    rq2_value = rq2_experiment_override or config.get("rq2_experiment")
    if not rq1_value or not rq2_value:
        raise ValueError(
            "RQ3 requires rq1_experiment and rq2_experiment paths, either in "
            "the config or as CLI overrides."
        )
    rq1_experiment = _resolve_path(rq1_value)
    rq2_experiment = _resolve_path(rq2_value)
    rq1_metrics = rq1_experiment / str(
        config.get(
            "rq1_metrics_relative_path",
            "comparisons/development_test_sample_metrics.csv",
        )
    )
    rq2_metrics = rq2_experiment / str(
        config.get(
            "rq2_metrics_relative_path",
            "comparisons/development_rq2_test_sample_metrics.csv",
        )
    )
    workbook = _resolve_path(
        config.get("workbook_path")
        or str(rq2_experiment / "inputs/data/merged_vol_rq2_text.xlsx")
    )
    calendar_path = _resolve_path(
        event_calendar_override or config["event_calendar_path"]
    )
    output_root = (
        _resolve_path(output_dir)
        if output_dir
        else ROOT
        / "outputs/experiments"
        / f"rq3_scheduled_news_regime_raw_vol_{_utc_timestamp()}"
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"RQ3 output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = _iso_now()
    _update_status(
        output_root,
        status="running",
        phase="initializing",
        started_at_utc=started,
    )

    resolved = {
        **config,
        "rq1_experiment": str(rq1_experiment),
        "rq2_experiment": str(rq2_experiment),
        "rq1_metrics_path": str(rq1_metrics),
        "rq2_metrics_path": str(rq2_metrics),
        "workbook_path": str(workbook),
        "event_calendar_path": str(calendar_path),
        "output_root": str(output_root),
        "news_source_timezone": str(
            config.get("news_source_timezone", "Europe/London")
        ),
    }
    try:
        expected_timezone = str(resolved["news_source_timezone"])
        for label, experiment in (
            ("RQ1", rq1_experiment),
            ("RQ2", rq2_experiment),
        ):
            validation_path = experiment / "validation_summary.json"
            validation = json.loads(
                _require_file(
                    validation_path,
                    f"{label} validation summary",
                ).read_text(encoding="utf-8")
            )
            if validation.get("news_source_timezone") != expected_timezone:
                raise ValueError(
                    f"{label} was not built with news_source_timezone="
                    f"{expected_timezone}: {validation_path}"
                )
        workbook_timezone = pd.read_excel(
            _require_file(workbook, "raw-vol workbook"),
            sheet_name=str(config.get("workbook_sheet_name", "gan_input_ready")),
            usecols=["source_timezone"],
        )["source_timezone"]
        workbook_timezones = {
            str(value).strip()
            for value in workbook_timezone.dropna().tolist()
            if str(value).strip()
        }
        if workbook_timezones != {expected_timezone}:
            raise ValueError(
                f"RQ3 workbook timezone mismatch: expected {expected_timezone}, "
                f"found {sorted(workbook_timezones)}."
            )
        for directory in (
            "inputs",
            "matched_samples",
            "comparisons",
            "final_tables",
            "registry",
            "docs",
        ):
            (output_root / directory).mkdir(parents=True, exist_ok=True)
        _write_yaml(
            output_root / "inputs/configs/resolved_config.yaml",
            {"rq3": resolved},
        )
        (output_root / "inputs/git_state.txt").write_text(
            _git_state(),
            encoding="utf-8",
        )
        rq1_support_paths = {
            path.parent.name: path
            for path in sorted(
                (rq1_experiment / "inputs/folds").glob(
                    "*/raw_surface_support.json"
                )
            )
        }
        rq2_support_paths = {
            path.parent.name: path
            for path in sorted(
                (rq2_experiment / "inputs/folds").glob(
                    "*/raw_surface_support.json"
                )
            )
        }
        if not rq1_support_paths:
            raise FileNotFoundError(
                f"RQ1 has no frozen raw support artifacts: {rq1_experiment}"
            )
        if set(rq1_support_paths) != set(rq2_support_paths):
            raise ValueError(
                "RQ1/RQ2 raw support fold sets do not match: "
                f"{sorted(rq1_support_paths)} vs {sorted(rq2_support_paths)}."
            )
        for fold, rq1_support in rq1_support_paths.items():
            rq2_support = rq2_support_paths[fold]
            if _sha256_file(rq1_support) != _sha256_file(rq2_support):
                raise ValueError(
                    f"RQ1/RQ2 raw support artifact mismatch for fold={fold}."
                )
        _archive_inputs(
            output_root,
            config_path=source_config,
            rq1_path=_require_file(rq1_metrics, "RQ1 frozen metrics"),
            rq2_path=_require_file(rq2_metrics, "RQ2 frozen metrics"),
            workbook_path=_require_file(workbook, "raw-vol workbook"),
            calendar_path=_require_file(calendar_path, "event calendar"),
            support_artifact_paths=rq1_support_paths,
        )

        _update_status(
            output_root,
            status="running",
            phase="loading_frozen_predictions",
            started_at_utc=started,
        )
        samples, source_audit = load_frozen_predictions(
            rq1_metrics,
            rq2_metrics,
            tolerance=float(config.get("source_tolerance", 1e-8)),
        )
        source_audit.to_csv(
            output_root / "inputs/frozen_prediction_consistency_audit.csv",
            index=False,
        )
        samples.to_csv(
            output_root
            / "comparisons/development_rq3_frozen_sample_metrics.csv",
            index=False,
        )
        pair_frame = (
            samples[samples["model"].astype(str) == "lp"][
                [
                    "fold",
                    "seed",
                    "surface_pair_id",
                    "current_snapshot_time_utc",
                    "target_snapshot_time_utc",
                    "news_count",
                ]
            ]
            .drop_duplicates(["fold", "surface_pair_id"])
            .copy()
        )
        covariates = build_forecast_origin_covariates(
            workbook,
            pair_frame,
            sheet_name=str(config.get("workbook_sheet_name", "gan_input_ready")),
            support_artifact_paths=rq1_support_paths,
        )
        covariates.to_csv(
            output_root / "inputs/forecast_origin_covariates.csv",
            index=False,
        )
        calendar = load_event_calendar(calendar_path)
        calendar.drop(columns=["release_time"]).to_csv(
            output_root / "inputs/event_calendars/validated_event_calendar.csv",
            index=False,
        )
        differences = build_loss_differentials(samples)

        windows = config.get(
            "windows",
            {
                "primary_0_5": {"pre_minutes": 0, "post_minutes": 5},
                "robustness_m10_p20": {
                    "pre_minutes": 10,
                    "post_minutes": 20,
                },
                "robustness_0_30": {
                    "pre_minutes": 0,
                    "post_minutes": 30,
                },
            },
        )
        ordinary_buffer = int(config.get("ordinary_buffer_minutes", 60))
        caliper = int(config.get("time_caliper_minutes", 15))
        match_ratio = int(config.get("match_ratio", 1))
        exact_weekday = bool(config.get("exact_weekday", True))
        if match_ratio != 1:
            raise ValueError(
                "The registered primary design requires 1:1 no-replacement "
                "matching; set match_ratio=1."
            )

        _update_status(
            output_root,
            status="running",
            phase="all_oos_conditional_predictive_analysis",
            started_at_utc=started,
        )
        all_oos_seed_rows, all_oos_average_rows = (
            build_all_oos_conditional_rows(
                differences,
                covariates,
                calendar,
                windows=windows,
                ordinary_buffer_minutes=ordinary_buffer,
            )
        )
        all_oos_seed_rows.to_csv(
            output_root
            / "comparisons/development_rq3_all_oos_seed_rows.csv",
            index=False,
        )
        all_oos_average_rows.to_csv(
            output_root
            / "comparisons/development_rq3_all_oos_conditional_rows.csv",
            index=False,
        )
        all_oos_inference = build_all_oos_inference(
            all_oos_average_rows,
            all_oos_seed_rows,
        )
        all_oos_inference.to_csv(
            output_root
            / "comparisons/development_rq3_conditional_predictive_ability.csv",
            index=False,
        )
        all_oos_inference[
            [
                "analysis_type",
                "window",
                "contrast",
                "metric",
                "gw_statistic",
                "gw_df",
                "gw_p_value",
                "gw_daily_observations",
                "gw_hac_max_lag",
                "gw_instruments",
            ]
        ].to_csv(
            output_root
            / "comparisons/development_rq3_giacomini_white_tests.csv",
            index=False,
        )
        overall_dm = build_overall_dm_hac(differences)
        overall_dm.to_csv(
            output_root
            / "comparisons/development_rq3_overall_dm_hac.csv",
            index=False,
        )

        all_labeled = []
        all_manifests = []
        all_balance = []
        all_unmatched = []
        all_seed_sets = []
        all_average_sets = []
        manifest_by_window: dict[str, pd.DataFrame] = {}

        _update_status(
            output_root,
            status="running",
            phase="matching_scheduled_to_ordinary_news",
            started_at_utc=started,
        )
        primary_labeled: pd.DataFrame | None = None
        primary_manifest: pd.DataFrame | None = None
        for window_name, window in windows.items():
            labeled = label_information_regimes(
                covariates,
                calendar,
                pre_window_minutes=int(window["pre_minutes"]),
                post_window_minutes=int(window["post_minutes"]),
                ordinary_buffer_minutes=ordinary_buffer,
            )
            matching = match_scheduled_to_ordinary(
                labeled,
                time_caliper_minutes=caliper,
                match_ratio=match_ratio,
                exact_weekday=exact_weekday,
            )
            labeled["window"] = str(window_name)
            matching.manifest["window"] = str(window_name)
            matching.balance["window"] = str(window_name)
            if not matching.unmatched.empty:
                matching.unmatched["window"] = str(window_name)
            seed_sets, average_sets = build_matched_set_differences(
                differences,
                matching.manifest,
                window_name=str(window_name),
            )
            all_labeled.append(labeled)
            all_manifests.append(matching.manifest)
            all_balance.append(matching.balance)
            if not matching.unmatched.empty:
                all_unmatched.append(matching.unmatched)
            all_seed_sets.append(seed_sets)
            all_average_sets.append(average_sets)
            manifest_by_window[str(window_name)] = matching.manifest
            if str(window_name) == "primary_0_5":
                primary_labeled = labeled
                primary_manifest = matching.manifest
        if primary_labeled is None or primary_manifest is None:
            raise ValueError("Config must define the primary_0_5 event window.")

        labeled_frame = pd.concat(all_labeled, ignore_index=True)
        manifest_frame = pd.concat(all_manifests, ignore_index=True)
        balance_frame = pd.concat(all_balance, ignore_index=True)
        unmatched_frame = (
            pd.concat(all_unmatched, ignore_index=True)
            if all_unmatched
            else pd.DataFrame()
        )
        seed_sets = pd.concat(all_seed_sets, ignore_index=True)
        average_sets = pd.concat(all_average_sets, ignore_index=True)

        labeled_frame.to_csv(
            output_root / "matched_samples/scheduled_news_samples.csv",
            index=False,
        )
        primary_labeled[
            primary_labeled["regime"].eq("ordinary_news_candidate")
        ].to_csv(
            output_root / "matched_samples/ordinary_news_candidates.csv",
            index=False,
        )
        manifest_frame.to_csv(
            output_root / "matched_samples/matched_set_manifest.csv",
            index=False,
        )
        balance_frame.to_csv(
            output_root / "matched_samples/matching_balance.csv",
            index=False,
        )
        unmatched_frame.to_csv(
            output_root / "matched_samples/unmatched_scheduled_news.csv",
            index=False,
        )
        differences = differences.merge(
            primary_labeled[
                ["fold", "surface_pair_id", "regime"]
            ],
            on=["fold", "surface_pair_id"],
            how="left",
            validate="many_to_one",
        )
        differences.to_csv(
            output_root
            / "comparisons/development_rq3_sample_loss_differentials.csv",
            index=False,
        )
        seed_sets.to_csv(
            output_root
            / "comparisons/development_rq3_matched_set_seed_differences.csv",
            index=False,
        )
        seed_direction_summary = (
            seed_sets.groupby(
                [
                    "analysis_type",
                    "window",
                    "contrast",
                    "contrast_family",
                    "focal_model",
                    "baseline_model",
                    "metric",
                    "seed",
                ],
                as_index=False,
            )
            .agg(
                mean_scheduled_news_increment=(
                    "scheduled_news_increment",
                    "mean",
                ),
                matched_set_count=("matched_set_id", "nunique"),
                positive_set_count=(
                    "scheduled_news_increment",
                    lambda values: int(np.sum(np.asarray(values) > 0.0)),
                ),
            )
        )
        seed_direction_summary.to_csv(
            output_root
            / "comparisons/development_rq3_seed_direction_summary.csv",
            index=False,
        )
        average_sets.to_csv(
            output_root
            / "comparisons/development_rq3_matched_set_differences.csv",
            index=False,
        )

        _update_status(
            output_root,
            status="running",
            phase="statistical_inference",
            started_at_utc=started,
        )
        iterations = int(config.get("bootstrap_iterations", 10000))
        bootstrap_seed = int(config.get("bootstrap_seed", 20260722))
        inference, conditional = build_inference(
            average_sets,
            seed_sets,
            manifest_by_window,
            bootstrap_iterations=iterations,
            bootstrap_seed=bootstrap_seed,
        )
        inference.to_csv(
            output_root
            / "comparisons/development_rq3_secondary_matched_cluster_bootstrap.csv",
            index=False,
        )
        conditional.to_csv(
            output_root
            / "comparisons/development_rq3_secondary_matched_conditional_regression.csv",
            index=False,
        )
        inference_keys = [
            "analysis_type",
            "window",
            "contrast",
            "contrast_family",
            "focal_model",
            "baseline_model",
            "metric",
            "difference_direction",
        ]
        combined_inference = inference.merge(
            conditional,
            on=inference_keys,
            validate="one_to_one",
        )
        family_summary = build_event_family_summary(average_sets)
        family_summary.to_csv(
            output_root
            / "comparisons/development_rq3_event_family_results.csv",
            index=False,
        )
        fomc_seed, fomc_summary = build_fomc_case_study(
            seed_sets,
            average_sets,
        )
        fomc_seed.to_csv(
            output_root
            / "comparisons/development_rq3_fomc_meeting_seed_results.csv",
            index=False,
        )
        fomc_summary.to_csv(
            output_root
            / "comparisons/development_rq3_fomc_meeting_results.csv",
            index=False,
        )
        persistence = build_persistence_context(
            samples,
            primary_labeled,
            primary_manifest,
        )
        persistence.to_csv(
            output_root
            / "comparisons/development_rq3_persistence_context.csv",
            index=False,
        )

        _update_status(
            output_root,
            status="running",
            phase="placebo_analysis",
            started_at_utc=started,
        )
        placebo_seed_frames = []
        placebo_average_frames = []
        placebo_manifests: dict[str, pd.DataFrame] = {}
        for shift in (-1, 1):
            name = f"placebo_shift_{shift:+d}_0_5"
            shifted = _shift_calendar(calendar, shift)
            labeled = label_information_regimes(
                covariates,
                shifted,
                pre_window_minutes=0,
                post_window_minutes=5,
                ordinary_buffer_minutes=ordinary_buffer,
            )
            try:
                matching = match_scheduled_to_ordinary(
                    labeled,
                    time_caliper_minutes=caliper,
                    match_ratio=match_ratio,
                    exact_weekday=exact_weekday,
                )
            except ValueError:
                continue
            seed_frame, average_frame = build_matched_set_differences(
                differences.drop(columns=["regime"]),
                matching.manifest,
                window_name=name,
                analysis_type="placebo_date_shift",
            )
            placebo_seed_frames.append(seed_frame)
            placebo_average_frames.append(average_frame)
            placebo_manifests[name] = matching.manifest
        same_clock_labeled = _same_clock_placebo_labeled(
            primary_labeled,
            primary_manifest,
        )
        try:
            same_clock_match = match_scheduled_to_ordinary(
                same_clock_labeled,
                time_caliper_minutes=caliper,
                match_ratio=match_ratio,
                exact_weekday=exact_weekday,
            )
        except ValueError:
            same_clock_match = None
        if same_clock_match is not None:
            name = "placebo_same_clock_ordinary_news"
            seed_frame, average_frame = build_matched_set_differences(
                differences.drop(columns=["regime"]),
                same_clock_match.manifest,
                window_name=name,
                analysis_type="placebo_same_clock",
            )
            placebo_seed_frames.append(seed_frame)
            placebo_average_frames.append(average_frame)
            placebo_manifests[name] = same_clock_match.manifest
        if placebo_average_frames:
            placebo_seed = pd.concat(placebo_seed_frames, ignore_index=True)
            placebo_average = pd.concat(
                placebo_average_frames,
                ignore_index=True,
            )
            placebo_inference, _placebo_conditional = build_inference(
                placebo_average,
                placebo_seed,
                placebo_manifests,
                bootstrap_iterations=iterations,
                bootstrap_seed=bootstrap_seed,
            )
        else:
            placebo_seed = pd.DataFrame()
            placebo_average = pd.DataFrame()
            placebo_inference = pd.DataFrame()
        placebo_seed.to_csv(
            output_root
            / "comparisons/development_rq3_placebo_seed_differences.csv",
            index=False,
        )
        placebo_average.to_csv(
            output_root
            / "comparisons/development_rq3_placebo_differences.csv",
            index=False,
        )
        placebo_inference.to_csv(
            output_root
            / "comparisons/development_rq3_placebo_results.csv",
            index=False,
        )

        primary_table = all_oos_inference[
            all_oos_inference["window"].eq("primary_0_5")
            & all_oos_inference["contrast"].eq(
                "lp_vs_continued_no_text"
            )
            & all_oos_inference["metric"].eq("surface_mae")
        ].copy()
        secondary_table = all_oos_inference[
            all_oos_inference["window"].eq("primary_0_5")
            & ~(
                all_oos_inference["contrast"].eq(
                    "lp_vs_continued_no_text"
                )
                & all_oos_inference["metric"].eq("surface_mae")
            )
        ].copy()
        matched_secondary_table = combined_inference[
            combined_inference["analysis_type"].eq("scheduled_news")
            & combined_inference["window"].eq("primary_0_5")
        ].copy()
        primary_table.to_csv(
            output_root
            / "final_tables/development_rq3_primary_conditional_robustness.csv",
            index=False,
        )
        secondary_table.to_csv(
            output_root
            / "final_tables/development_rq3_secondary_all_oos.csv",
            index=False,
        )
        matched_secondary_table.to_csv(
            output_root
            / "final_tables/development_rq3_secondary_matched_news.csv",
            index=False,
        )
        fomc_summary.to_csv(
            output_root
            / "final_tables/development_thesis_rq3_fomc_case_study.csv",
            index=False,
        )
        placebo_inference.to_csv(
            output_root
            / "final_tables/development_thesis_rq3_placebo_summary.csv",
            index=False,
        )

        primary_all_oos_rows = all_oos_average_rows[
            all_oos_average_rows["window"].eq("primary_0_5")
            & all_oos_average_rows["scheduled_news_indicator"].eq(1)
        ]
        primary_release_count = int(
            primary_all_oos_rows["event_id"].nunique()
        )
        primary_release_days = int(
            primary_all_oos_rows["inference_cluster_day"].nunique()
        )
        balance_flags = int(
            balance_frame[
                balance_frame["window"].eq("primary_0_5")
            ]["flag_abs_smd_over_0_10"].sum()
        )
        primary_row = (
            primary_table.iloc[0].to_dict()
            if len(primary_table) == 1
            else {}
        )
        primary_sets_for_rule = average_sets[
            average_sets["analysis_type"].eq("scheduled_news")
            & average_sets["window"].eq("primary_0_5")
            & average_sets["contrast"].eq("lp_vs_continued_no_text")
            & average_sets["metric"].eq("surface_mae")
        ]
        family_leave_one_out = {
            str(family): float(
                primary_sets_for_rule[
                    ~primary_sets_for_rule["event_family"].astype(str).eq(
                        str(family)
                    )
                ]["scheduled_news_increment"].mean()
            )
            for family in sorted(
                primary_sets_for_rule["event_family"].astype(str).unique()
            )
        }
        event_family_stable = bool(
            family_leave_one_out
            and all(
                np.isfinite(value) and value > 0.0
                for value in family_leave_one_out.values()
            )
        )
        placebo_primary = placebo_inference[
            placebo_inference.get(
                "contrast",
                pd.Series(dtype=str),
            ).astype(str).eq("lp_vs_continued_no_text")
            & placebo_inference.get(
                "metric",
                pd.Series(dtype=str),
            ).astype(str).eq("surface_mae")
        ] if not placebo_inference.empty else pd.DataFrame()
        placebo_null = bool(
            placebo_primary.empty
            or not np.any(
                (
                    placebo_primary[
                        "mean_scheduled_news_increment"
                    ].to_numpy(dtype=np.float64)
                    > 0.0
                )
                & (
                    placebo_primary[
                        "p_one_sided_focal_more_valuable"
                    ].to_numpy(dtype=np.float64)
                    < 0.05
                )
            )
        )
        minimum_release_days = int(config.get("minimum_release_days", 20))
        support = bool(
            primary_row
            and float(primary_row["conditional_scheduled_effect"]) > 0.0
            and float(primary_row["conditional_ci_95_lower"]) > 0.0
            and float(
                primary_row[
                    "conditional_p_one_sided_focal_more_valuable"
                ]
            )
            < 0.05
            and float(primary_row["gw_p_value"]) < 0.05
            and int(
                primary_row[
                    "positive_seed_scheduled_increment_count"
                ]
            )
            >= 2
            and primary_release_days >= minimum_release_days
        )
        validation = {
            "status": "ok",
            "evidence_status": (
                "development"
                if primary_release_days >= minimum_release_days
                else "development_underpowered"
            ),
            "frozen_model_policy": "no_retraining_no_seed_selection",
            "source_prediction_rows": int(len(samples)),
            "unique_test_pairs": int(
                pair_frame["surface_pair_id"].nunique()
            ),
            "models": sorted(samples["model"].astype(str).unique()),
            "seeds": sorted(int(value) for value in samples["seed"].unique()),
            "folds": sorted(samples["fold"].astype(str).unique()),
            "primary_scheduled_pair_count": int(
                primary_all_oos_rows["surface_pair_id"].nunique()
            ),
            "primary_matched_set_count": int(
                primary_manifest["matched_set_id"].nunique()
            ),
            "primary_release_count": primary_release_count,
            "primary_release_day_count": primary_release_days,
            "primary_event_families": sorted(
                primary_all_oos_rows["event_family"].astype(str).unique()
            ),
            "ordinary_buffer_minutes": ordinary_buffer,
            "time_caliper_minutes": caliper,
            "exact_weekday": exact_weekday,
            "matching_balance_flags_over_0_10": balance_flags,
            "minimum_release_days": minimum_release_days,
            "event_family_leave_one_out_mean": family_leave_one_out,
            "event_family_direction_stable": event_family_stable,
            "placebo_null_requirement_passed": placebo_null,
            "full_support_primary_rule": support,
            "primary_analysis": "all_oos_conditional_predictive_robustness",
            "secondary_analysis": (
                "greedy_matched_scheduled_vs_ordinary_news"
            ),
            "known_limitations": [
                "2023 rolling-development evidence, not 2024+ confirmation",
                "FOMC is an underpowered meeting-level case study",
                (
                    "raw-vol 7d ATM is not reported because 7d lies below "
                    "observed maturity support"
                ),
                (
                    "scheduled-news labels identify a predictive regime, "
                    "not a causal semantic-news effect"
                ),
                "lead-text placebo unavailable in frozen predictions",
                "no-news quiet requires separate mixed-data retraining",
            ],
        }
        _write_json(output_root / "validation_summary.json", validation)
        _write_documentation(
            output_root,
            {
                **resolved,
                "bootstrap_iterations": iterations,
                "bootstrap_seed": bootstrap_seed,
            },
        )
        _build_manifest(output_root)
        _update_status(
            output_root,
            status="completed",
            phase="completed",
            started_at_utc=started,
        )
        _build_manifest(output_root)
        return output_root
    except BaseException as exc:
        _update_status(
            output_root,
            status="failed",
            phase="failed",
            started_at_utc=started,
            error=exc,
        )
        raise
