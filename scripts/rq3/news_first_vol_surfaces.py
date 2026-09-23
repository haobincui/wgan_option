"""Build cumulative news-first raw-vol surface sensitivity datasets.

The market response horizon remains five minutes in every dataset.  Only the
maximum wait from a Factiva availability timestamp to the first eligible
market pair changes (5/10/15/20/30 minutes).  This module intentionally leaves
the frozen five-minute :class:`SessionAlignmentPolicy` unchanged.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import scripts._path_setup  # noqa: F401

from scripts.raw_vol.relaxed_time_pipeline import _materialize_dataset_inputs
from scripts.rq3.market_jump_news import parse_factiva_news
from scripts.rq3.news_first_vol_alignment import (
    DEFAULT_TOLERANCES_MINUTES,
    build_news_first_alignments,
    prepare_pair_universe,
)
from film_wgan.support import SUPPORT_METHOD, raw_support_mask
from wgan_option.market.treasury_sessions import TreasuryGlobexSessionCalendar
from wgan_option.merge.merge_vol_core import build_vol_workbook_frames
from wgan_option.merge_support import write_workbook
from wgan_option.surface_grid import build_surface_grids
from wgan_option.surface_generation.market_index import connect_market_index


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK_SHEETS = (
    "news_surface_pair_audit",
    "surface_side_detail",
    "gan_input_ready",
    "gan_input_atm_ab",
)
LEGACY_TOLERANCES_MINUTES = (5, 10, 15, 30)
SUPPORTED_TOLERANCE_SETS = (
    LEGACY_TOLERANCES_MINUTES,
    DEFAULT_TOLERANCES_MINUTES,
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    compressed: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if compressed and target.suffix != ".gz":
        target = target.with_suffix(target.suffix + ".gz")
    frame.to_csv(
        target,
        index=False,
        compression="gzip" if target.suffix == ".gz" else None,
    )
    return target


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = payload.get("news_first_vol_surfaces", payload)
    if not isinstance(config, dict):
        raise ValueError("news_first_vol_surfaces config must be a mapping")
    return dict(config)


def _validate_tolerance_minutes(values: Sequence[Any]) -> tuple[int, ...]:
    tolerances = tuple(int(value) for value in values)
    if tolerances not in SUPPORTED_TOLERANCE_SETS:
        raise ValueError(
            "This sensitivity export requires ordered tolerance_minutes "
            "[5, 10, 15, 30] or [5, 10, 15, 20, 30]"
        )
    return tolerances


def _pair_key_columns(frame: pd.DataFrame) -> tuple[str, str]:
    origin = (
        "effective_origin_utc"
        if "effective_origin_utc" in frame.columns
        else "current_snapshot_time_utc"
    )
    target = (
        "target_anchor_utc"
        if "target_anchor_utc" in frame.columns
        else "target_snapshot_time_utc"
    )
    missing = [column for column in (origin, target) if column not in frame.columns]
    if missing:
        raise ValueError(f"Pair audit is missing timestamp columns: {missing}")
    return origin, target


def _add_pair_weights(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    if result.empty:
        result["pair_article_count"] = pd.Series(dtype="int64")
        result["sample_weight"] = pd.Series(dtype="float64")
        return result
    if "pair_id" not in result.columns or result["pair_id"].fillna("").eq("").any():
        raise ValueError("Training rows require a non-empty pair_id")
    counts = result.groupby("pair_id", sort=False)["news_row_id"].transform("size")
    result["pair_article_count"] = counts.astype(int)
    result["sample_weight"] = 1.0 / result["pair_article_count"].astype(float)
    return result


def _quality_summary(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "pair_id",
        "slice_pair_id",
        "metric_status",
        "pair_atm_quality",
        "maturity_date",
    }
    missing = sorted(required - set(pair_metrics.columns))
    if missing:
        raise ValueError(f"Pair metrics are missing required columns: {missing}")
    work = pair_metrics.copy()
    work["_atm_ab"] = work["metric_status"].astype(str).eq("ok") & work[
        "pair_atm_quality"
    ].astype(str).isin(["A", "B"])
    work["_atm_abc"] = work["metric_status"].astype(str).eq("ok") & work[
        "pair_atm_quality"
    ].astype(str).isin(["A", "B", "C"])
    if "skew_status" in work.columns and "skew_quality" in work.columns:
        work["_skew_ab"] = work["skew_status"].astype(str).eq("ok") & work[
            "skew_quality"
        ].astype(str).isin(["A", "B"])
    else:
        work["_skew_ab"] = False

    def joined_dates(group: pd.DataFrame, mask: str) -> str:
        values = sorted(
            set(group.loc[group[mask], "maturity_date"].dropna().astype(str))
        )
        return ";".join(values)

    rows: list[dict[str, Any]] = []
    for pair_id, group in work.groupby("pair_id", sort=False):
        rows.append(
            {
                "pair_id": str(pair_id),
                "pair_slice_count": int(len(group)),
                "atm_ab_slice_count": int(group["_atm_ab"].sum()),
                "atm_abc_slice_count": int(group["_atm_abc"].sum()),
                "skew_ab_slice_count": int(group["_skew_ab"].sum()),
                "atm_ab_maturity_dates": joined_dates(group, "_atm_ab"),
                "has_atm_ab_maturity": bool(group["_atm_ab"].any()),
            }
        )
    return pd.DataFrame(rows)


def build_training_views(
    pair_audit: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    tolerance_minutes: int,
    all_tolerances: Sequence[int] = DEFAULT_TOLERANCES_MINUTES,
) -> dict[str, pd.DataFrame]:
    """Create article-level surface views and explicit maturity-grain bridges."""

    audit = pair_audit.copy().reset_index(drop=True)
    required_audit = {
        "news_row_id",
        "publication_market_state",
        "pair_quality_label",
    }
    missing = sorted(required_audit - set(audit.columns))
    if missing:
        raise ValueError(f"Pair audit is missing required columns: {missing}")
    if audit["news_row_id"].duplicated().any():
        raise ValueError("Pair audit contains duplicate news_row_id values")
    origin_column, target_column = _pair_key_columns(audit)

    metrics = pair_metrics.copy()
    if metrics["slice_pair_id"].duplicated().any():
        raise ValueError("Pair metrics contain duplicate slice_pair_id values")
    metric_pair_map = metrics[
        ["pair_id", "origin_time_utc", "target_time_utc"]
    ].drop_duplicates()
    if metric_pair_map["pair_id"].duplicated().any():
        raise ValueError("Pair metrics map pair_id to multiple time pairs")

    if "pair_id" not in audit.columns:
        audit = audit.merge(
            metric_pair_map,
            left_on=[origin_column, target_column],
            right_on=["origin_time_utc", "target_time_utc"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["origin_time_utc", "target_time_utc"])
    else:
        audit["pair_id"] = audit["pair_id"].fillna("").astype(str)
        missing_pair = audit["pair_id"].eq("") & audit[origin_column].fillna("").ne("")
        if missing_pair.any():
            lookup = metric_pair_map.rename(columns={"pair_id": "_metric_pair_id"})
            audit = audit.merge(
                lookup,
                left_on=[origin_column, target_column],
                right_on=["origin_time_utc", "target_time_utc"],
                how="left",
                validate="many_to_one",
            )
            audit.loc[missing_pair, "pair_id"] = audit.loc[
                missing_pair, "_metric_pair_id"
            ].fillna("")
            audit = audit.drop(
                columns=["_metric_pair_id", "origin_time_utc", "target_time_utc"]
            )

    summary = _quality_summary(metrics)
    audit = audit.merge(summary, on="pair_id", how="left", validate="many_to_one")
    for column in (
        "pair_slice_count",
        "atm_ab_slice_count",
        "atm_abc_slice_count",
        "skew_ab_slice_count",
    ):
        audit[column] = (
            pd.to_numeric(audit[column], errors="coerce").fillna(0).astype(int)
        )
    audit["has_atm_ab_maturity"] = audit["atm_ab_slice_count"].gt(0)
    audit["atm_ab_maturity_dates"] = audit["atm_ab_maturity_dates"].fillna("")
    audit["dataset_tolerance_minutes"] = int(tolerance_minutes)
    if "first_included_tolerance_minutes" not in audit.columns:
        shifts = pd.to_numeric(
            audit.get("origin_tolerance_minutes_used"), errors="coerce"
        )
        ordered = sorted(int(value) for value in all_tolerances)
        audit["first_included_tolerance_minutes"] = shifts.map(
            lambda value: (
                next((item for item in ordered if value <= item), pd.NA)
                if pd.notna(value)
                else pd.NA
            )
        )

    eligible = (
        pd.to_numeric(
            audit.get("training_eligible", audit.get("training_candidate_flag", 0)),
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
    )
    surface_usable = audit["pair_quality_label"].astype(str).eq("usable")
    open_news = audit["publication_market_state"].astype(str).eq("open")
    audit["surface_training_eligible"] = eligible & surface_usable & open_news
    audit["atm_ab_training_eligible"] = (
        audit["surface_training_eligible"] & audit["has_atm_ab_maturity"]
    )
    audit["surface_pair_quality_label"] = audit["pair_quality_label"].astype(str)
    audit["training_candidate_flag"] = audit["surface_training_eligible"].astype(int)
    closed = audit["publication_market_state"].astype(str).eq("closed")
    if "exclude_reason" not in audit.columns:
        audit["exclude_reason"] = ""
    audit.loc[closed, "exclude_reason"] = "publication_market_closed"

    ready = _add_pair_weights(audit.loc[audit["surface_training_eligible"]].copy())
    atm_ready = _add_pair_weights(audit.loc[audit["atm_ab_training_eligible"]].copy())

    ab_metrics = metrics.loc[
        metrics["metric_status"].astype(str).eq("ok")
        & metrics["pair_atm_quality"].astype(str).isin(["A", "B"])
    ].copy()
    strict_news = atm_ready[
        [
            column
            for column in (
                "news_row_id",
                "sample_id",
                "news_available_time_utc",
                "news_timestamp_utc",
                "pair_id",
                "pair_article_count",
            )
            if column in atm_ready.columns
        ]
    ].copy()
    article_bridge = strict_news.merge(
        ab_metrics,
        on="pair_id",
        how="inner",
        validate="many_to_many",
        suffixes=("_news", ""),
    )
    if not article_bridge.empty:
        article_bridge["pair_maturity_count"] = (
            article_bridge.groupby("pair_id", sort=False)["slice_pair_id"]
            .transform("nunique")
            .astype(int)
        )
        article_bridge["sample_weight"] = 1.0 / (
            article_bridge["pair_article_count"].astype(float)
            * article_bridge["pair_maturity_count"].astype(float)
        )
        article_bridge["dataset_tolerance_minutes"] = int(tolerance_minutes)
        article_bridge = article_bridge.sort_values(
            ["origin_time_utc", "news_row_id", "maturity_date"], kind="stable"
        ).reset_index(drop=True)

    strict_pair_ids = set(atm_ready["pair_id"].astype(str))
    pair_outcomes = ab_metrics.loc[
        ab_metrics["pair_id"].astype(str).isin(strict_pair_ids)
    ].copy()
    if not pair_outcomes.empty:
        news_counts = atm_ready.groupby("pair_id", sort=False).agg(
            matched_news_count=("news_row_id", "size"),
            matched_news_timestamp_count=(
                "news_available_time_utc"
                if "news_available_time_utc" in atm_ready.columns
                else "news_timestamp_utc",
                "nunique",
            ),
        )
        pair_outcomes = pair_outcomes.merge(
            news_counts,
            left_on="pair_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        pair_outcomes["pair_maturity_count"] = (
            pair_outcomes.groupby("pair_id", sort=False)["slice_pair_id"]
            .transform("nunique")
            .astype(int)
        )
        pair_outcomes["pair_maturity_weight"] = 1.0 / pair_outcomes[
            "pair_maturity_count"
        ].astype(float)
        pair_outcomes["dataset_tolerance_minutes"] = int(tolerance_minutes)
        pair_outcomes = pair_outcomes.sort_values(
            ["origin_time_utc", "maturity_date"], kind="stable"
        ).reset_index(drop=True)

    matched_pair_ids = set(
        audit.loc[audit["pair_id"].fillna("").ne(""), "pair_id"].astype(str)
    )
    matched_metrics = metrics.loc[
        metrics["pair_id"].astype(str).isin(matched_pair_ids)
    ].copy()
    matched_metrics["dataset_tolerance_minutes"] = int(tolerance_minutes)

    return {
        "news_surface_pair_audit": audit,
        "surface_side_detail": pd.DataFrame(),
        "gan_input_ready": ready,
        "gan_input_atm_ab": atm_ready,
        "news_atm_ab_maturity_bridge": article_bridge,
        "pair_atm_ab_outcomes": pair_outcomes,
        "matched_pair_slice_metrics": matched_metrics,
    }


def _serialized_array_properties(value: Any) -> tuple[int, bool]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 0, False
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        text = str(value).strip()
        if not text:
            return 0, False
        parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected a serialized list")
    values = np.asarray(parsed, dtype=float)
    return len(parsed), bool(np.isfinite(values).all())


def _surface_grid_fingerprint(
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> str:
    payload = {
        "schema_version": 1,
        "support_method": SUPPORT_METHOD,
        "strike_grid": [float(value) for value in strike_grid],
        "maturity_days_grid": [
            int(round(float(value))) for value in maturity_days_grid
        ],
        "surface_shape": [len(maturity_days_grid), len(strike_grid)],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_pair_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return ""
    values = group[column].dropna().tolist()
    normalized = {
        json.dumps(value, sort_keys=True) if isinstance(value, Mapping) else str(value)
        for value in values
        if str(value).strip()
    }
    if len(normalized) > 1:
        pair_id = str(group["pair_id"].iloc[0])
        raise ValueError(
            f"Pair {pair_id} maps to multiple {column} values; support audit "
            "cannot choose an article-dependent surface."
        )
    return values[0] if values else ""


def _strict_support_mask(
    surface_params: Any,
    *,
    surface_model: str,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> tuple[np.ndarray, str]:
    shape = (len(maturity_days_grid), len(strike_grid))
    if str(surface_model).strip().lower() != "raw":
        return np.zeros(shape, dtype=bool), "unsupported_surface_model"
    if surface_params is None or not str(surface_params).strip():
        return np.zeros(shape, dtype=bool), "missing_surface_params"
    try:
        mask = raw_support_mask(
            surface_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.zeros(shape, dtype=bool), "invalid_surface_params"
    if mask.shape != shape:
        raise ValueError(
            f"Strict support mask shape {mask.shape} does not match grid {shape}."
        )
    return mask.astype(bool, copy=False), "ok"


def build_surface_support_audit(
    pair_audit: pd.DataFrame,
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    tolerance_minutes: int,
) -> pd.DataFrame:
    """Build one audit row per matched pair without changing eligibility.

    A cell is strict-support eligible only when it avoids both maturity
    extrapolation and strike clamping on the current and target raw surfaces.
    The masks are summarized but are deliberately not serialized or consumed by
    training in this sensitivity experiment.
    """

    required = {
        "pair_id",
        "surface_model",
        "current_surface_param_json",
        "target_surface_param_json",
    }
    missing = sorted(required - set(pair_audit.columns))
    if missing:
        raise ValueError(f"Pair audit is missing support columns: {missing}")
    strikes = [float(value) for value in strike_grid]
    maturities = [int(round(float(value))) for value in maturity_days_grid]
    if len(strikes) < 1 or len(maturities) < 1:
        raise ValueError("Surface support audit requires a non-empty grid.")
    total_cells = len(strikes) * len(maturities)
    fingerprint = _surface_grid_fingerprint(strikes, maturities)
    matched = pair_audit.loc[
        pair_audit["pair_id"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    rows: list[dict[str, Any]] = []
    for pair_id, group in matched.groupby("pair_id", sort=False):
        surface_model = str(_unique_pair_value(group, "surface_model") or "")
        current_params = _unique_pair_value(group, "current_surface_param_json")
        target_params = _unique_pair_value(group, "target_surface_param_json")
        current_mask, current_status = _strict_support_mask(
            current_params,
            surface_model=surface_model,
            strike_grid=strikes,
            maturity_days_grid=maturities,
        )
        target_mask, target_status = _strict_support_mask(
            target_params,
            surface_model=surface_model,
            strike_grid=strikes,
            maturity_days_grid=maturities,
        )
        joint_mask = current_mask & target_mask
        current_count = int(current_mask.sum())
        target_count = int(target_mask.sum())
        joint_count = int(joint_mask.sum())
        eligible = (
            group.get("surface_training_eligible", pd.Series(False, index=group.index))
            .astype(bool)
            .any()
        )
        rows.append(
            {
                "tolerance_minutes": int(tolerance_minutes),
                "pair_id": str(pair_id),
                "effective_origin_utc": _unique_pair_value(
                    group, "effective_origin_utc"
                ),
                "target_snapshot_time_utc": _unique_pair_value(
                    group, "target_snapshot_time_utc"
                ),
                "surface_model": surface_model,
                "surface_training_eligible": bool(eligible),
                "grid_cell_count": int(total_cells),
                "current_strict_support_cell_count": current_count,
                "target_strict_support_cell_count": target_count,
                "joint_strict_support_cell_count": joint_count,
                "current_strict_support_fraction": current_count / total_cells,
                "target_strict_support_fraction": target_count / total_cells,
                "joint_strict_support_fraction": joint_count / total_cells,
                "current_zero_support": current_count == 0,
                "target_zero_support": target_count == 0,
                "joint_zero_support": joint_count == 0,
                "current_support_status": current_status,
                "target_support_status": target_status,
                "support_method": SUPPORT_METHOD,
                "grid_fingerprint": fingerprint,
                "support_mask_applied": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_surface_support(
    support_audit: pd.DataFrame,
    *,
    tolerance_minutes: int,
) -> dict[str, Any]:
    """Summarize strict support on the unchanged surface-training population."""

    if support_audit.empty:
        return {
            "tolerance_minutes": int(tolerance_minutes),
            "support_audit_pairs": 0,
            "surface_usable_pairs": 0,
            "joint_strict_support_pairs": 0,
            "joint_zero_support_pairs": 0,
            "joint_strict_support_fraction_mean": 0.0,
            "joint_strict_support_cell_count_median": 0.0,
            "support_mask_applied": False,
            "grid_fingerprint": "",
        }
    eligible = support_audit["surface_training_eligible"].astype(bool)
    eligible_rows = support_audit.loc[eligible]
    positive = eligible_rows["joint_strict_support_cell_count"].astype(int).gt(0)
    fingerprints = support_audit["grid_fingerprint"].dropna().astype(str).unique()
    return {
        "tolerance_minutes": int(tolerance_minutes),
        "support_audit_pairs": int(support_audit["pair_id"].nunique()),
        "surface_usable_pairs": int(eligible_rows["pair_id"].nunique()),
        "joint_strict_support_pairs": int(positive.sum()),
        "joint_zero_support_pairs": int((~positive).sum()),
        "joint_strict_support_fraction_mean": (
            float(eligible_rows["joint_strict_support_fraction"].mean())
            if not eligible_rows.empty
            else 0.0
        ),
        "joint_strict_support_cell_count_median": (
            float(eligible_rows["joint_strict_support_cell_count"].median())
            if not eligible_rows.empty
            else 0.0
        ),
        "support_mask_applied": False,
        "grid_fingerprint": str(fingerprints[0]) if len(fingerprints) == 1 else "",
    }


def validate_dataset_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    tolerance_minutes: int,
    expected_news_rows: int | None = None,
    expected_surface_cells: int = 256,
) -> dict[str, Any]:
    """Validate dataset grain and return a compact machine-readable summary."""

    missing = sorted(set(WORKBOOK_SHEETS) - set(frames))
    if missing:
        raise ValueError(f"Dataset frames are missing workbook sheets: {missing}")
    audit = frames["news_surface_pair_audit"]
    ready = frames["gan_input_ready"]
    atm_ready = frames["gan_input_atm_ab"]
    article_bridge = frames["news_atm_ab_maturity_bridge"]
    pair_outcomes = frames["pair_atm_ab_outcomes"]
    failures: list[str] = []
    if expected_news_rows is not None and len(audit) != int(expected_news_rows):
        failures.append("news_row_count")
    if audit["news_row_id"].duplicated().any():
        failures.append("duplicate_news_row_id")
    if (
        not ready.empty
        and not ready["publication_market_state"].astype(str).eq("open").all()
    ):
        failures.append("closed_news_in_surface_training")
    if not atm_ready.empty and not atm_ready["has_atm_ab_maturity"].astype(bool).all():
        failures.append("non_atm_ab_row_in_strict_training")

    weight_ok = True
    for frame in (ready, atm_ready):
        if not frame.empty:
            sums = frame.groupby("pair_id", sort=False)["sample_weight"].sum()
            weight_ok = weight_ok and bool(np.allclose(sums.to_numpy(float), 1.0))
    if not weight_ok:
        failures.append("pair_weights_do_not_sum_to_one")

    article_key = ["news_row_id", "pair_id", "maturity_date", "underlying_contract_id"]
    pair_key = ["pair_id", "maturity_date", "underlying_contract_id"]
    article_unique = not article_bridge.duplicated(article_key).any()
    pair_unique = not pair_outcomes.duplicated(pair_key).any()
    if not article_unique:
        failures.append("duplicate_article_maturity_key")
    if not pair_unique:
        failures.append("duplicate_pair_maturity_key")

    for label, frame in (("surface", ready), ("atm_ab", atm_ready)):
        if frame.empty:
            continue
        for column, expected in (
            ("current_surface_flat", int(expected_surface_cells)),
            ("target_surface_flat", int(expected_surface_cells)),
            ("hd_embedding", 1024),
            ("lp_embedding", 1024),
        ):
            if column not in frame.columns:
                failures.append(f"{label}_missing_{column}")
                continue
            properties = frame[column].map(_serialized_array_properties)
            if not properties.map(lambda item: item[0] == expected).all():
                failures.append(f"{label}_{column}_length")
            if not properties.map(lambda item: item[1]).all():
                failures.append(f"{label}_{column}_nonfinite")

    bridge_weight_ok = True
    if not article_bridge.empty:
        sums = article_bridge.groupby("pair_id", sort=False)["sample_weight"].sum()
        bridge_weight_ok = bool(np.allclose(sums.to_numpy(float), 1.0))
    outcome_weight_ok = True
    if not pair_outcomes.empty:
        sums = pair_outcomes.groupby("pair_id", sort=False)[
            "pair_maturity_weight"
        ].sum()
        outcome_weight_ok = bool(np.allclose(sums.to_numpy(float), 1.0))
    if not bridge_weight_ok:
        failures.append("article_maturity_weights_do_not_sum_to_one")
    if not outcome_weight_ok:
        failures.append("pair_maturity_weights_do_not_sum_to_one")

    support_summary: dict[str, Any] = {}
    support_audit = frames.get("surface_support_audit")
    if support_audit is not None:
        required_support_columns = {
            "pair_id",
            "grid_cell_count",
            "current_strict_support_cell_count",
            "target_strict_support_cell_count",
            "joint_strict_support_cell_count",
            "current_strict_support_fraction",
            "target_strict_support_fraction",
            "joint_strict_support_fraction",
            "current_support_status",
            "target_support_status",
            "grid_fingerprint",
            "support_mask_applied",
        }
        missing_support = sorted(required_support_columns - set(support_audit.columns))
        if missing_support:
            failures.append("support_audit_missing_columns")
        else:
            if support_audit["pair_id"].duplicated().any():
                failures.append("duplicate_support_pair_id")
            expected_pair_ids = set(
                audit.loc[
                    audit["pair_id"].fillna("").astype(str).str.strip().ne(""),
                    "pair_id",
                ].astype(str)
            )
            actual_pair_ids = set(support_audit["pair_id"].astype(str))
            if actual_pair_ids != expected_pair_ids:
                failures.append("support_pair_coverage")
            if support_audit["support_mask_applied"].astype(bool).any():
                failures.append("support_mask_was_applied")
            if support_audit["grid_fingerprint"].astype(str).nunique() != 1:
                failures.append("support_grid_fingerprint")
            total = pd.to_numeric(support_audit["grid_cell_count"], errors="coerce")
            for prefix in ("current", "target", "joint"):
                counts = pd.to_numeric(
                    support_audit[f"{prefix}_strict_support_cell_count"],
                    errors="coerce",
                )
                fractions = pd.to_numeric(
                    support_audit[f"{prefix}_strict_support_fraction"],
                    errors="coerce",
                )
                valid = (
                    counts.notna()
                    & total.notna()
                    & counts.ge(0)
                    & counts.le(total)
                    & fractions.between(0.0, 1.0, inclusive="both")
                    & np.isclose(
                        fractions.to_numpy(dtype=float),
                        (counts / total).to_numpy(dtype=float),
                    )
                )
                if not bool(valid.all()):
                    failures.append(f"invalid_{prefix}_support_summary")
            eligible_support = support_audit.loc[
                support_audit.get(
                    "surface_training_eligible",
                    pd.Series(False, index=support_audit.index),
                ).astype(bool)
            ]
            if (
                not eligible_support.empty
                and not (
                    eligible_support["current_support_status"].astype(str).eq("ok")
                    & eligible_support["target_support_status"].astype(str).eq("ok")
                ).all()
            ):
                failures.append("usable_pair_support_parse_status")
            support_summary = summarize_surface_support(
                support_audit,
                tolerance_minutes=tolerance_minutes,
            )

    timestamp_column = (
        "news_available_time_utc"
        if "news_available_time_utc" in ready.columns
        else "news_timestamp_utc"
    )
    atm_timestamp_column = (
        "news_available_time_utc"
        if "news_available_time_utc" in atm_ready.columns
        else "news_timestamp_utc"
    )
    summary = {
        "tolerance_minutes": int(tolerance_minutes),
        "news_rows": int(len(audit)),
        "matched_open_news": int(
            (
                pd.to_numeric(audit.get("training_eligible", 0), errors="coerce")
                .fillna(0)
                .eq(1)
                & audit["publication_market_state"].astype(str).eq("open")
            ).sum()
        ),
        "surface_usable_news": int(len(ready)),
        "surface_usable_timestamps": int(ready[timestamp_column].nunique())
        if not ready.empty
        else 0,
        "surface_usable_pairs": int(ready["pair_id"].nunique())
        if not ready.empty
        else 0,
        "atm_ab_surface_news": int(len(atm_ready)),
        "atm_ab_surface_timestamps": int(atm_ready[atm_timestamp_column].nunique())
        if not atm_ready.empty
        else 0,
        "atm_ab_surface_pairs": int(atm_ready["pair_id"].nunique())
        if not atm_ready.empty
        else 0,
        "atm_ab_news_maturity_rows": int(len(article_bridge)),
        "atm_ab_pair_maturities": int(len(pair_outcomes)),
        "pair_weights_sum_to_one": bool(weight_ok),
        "article_maturity_weights_sum_to_one": bool(bridge_weight_ok),
        "pair_maturity_weights_sum_to_one": bool(outcome_weight_ok),
        "article_maturity_key_unique": bool(article_unique),
        "pair_maturity_key_unique": bool(pair_unique),
        "closed_training_rows": int(
            (~ready["publication_market_state"].astype(str).eq("open")).sum()
        )
        if not ready.empty
        else 0,
        **support_summary,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    return summary


def _git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "status": status}
    except Exception as exc:  # pragma: no cover
        return {"commit": "", "status": f"unavailable:{exc}"}


def _source_manifest(paths: Mapping[str, Path]) -> pd.DataFrame:
    rows = []
    for label, path in paths.items():
        rows.append(
            {
                "source": label,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def _code_manifest(config_path: Path) -> pd.DataFrame:
    code_paths = {
        "source_config": config_path,
        "rq3_cli": ROOT / "scripts/rq3/main.py",
        "news_first_alignment": ROOT / "scripts/rq3/news_first_vol_alignment.py",
        "news_first_surface_builder": ROOT / "scripts/rq3/news_first_vol_surfaces.py",
        "surface_materializer": ROOT / "scripts/raw_vol/relaxed_time_pipeline.py",
        "vol_workbook_builder": ROOT / "src/wgan_option/merge/merge_vol_core.py",
        "surface_grid_builder": ROOT / "src/wgan_option/surface_grid.py",
        "strict_surface_support": ROOT / "src/film_wgan/support.py",
        "canonical_news_parser": ROOT / "scripts/rq3/market_jump_news.py",
    }
    return _source_manifest(code_paths).rename(columns={"source": "component"})


def _field_dictionary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for table, frame in tables.items():
        for column, dtype in frame.dtypes.items():
            role = "field"
            if column in {"news_row_id", "pair_id", "slice_pair_id"}:
                role = "join_key"
            elif column.endswith("_utc"):
                role = "timestamp"
            elif "quality" in column or "eligible" in column or "reason" in column:
                role = "quality_audit"
            elif "weight" in column:
                role = "weight"
            rows.append(
                {
                    "table": table,
                    "column": str(column),
                    "dtype": str(dtype),
                    "role": role,
                    "description": f"Exported {table} field `{column}`.",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["table", "column"])


def _check_expected_counts(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        rows.append(
            {
                "check": field,
                "actual": actual_value,
                "expected": expected_value,
                "status": "pass" if actual_value == expected_value else "fail",
            }
        )
    return rows


def run_news_first_vol_surfaces(
    config_path: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Build the configured cumulative, training-compatible workbooks."""

    config = _read_config(config_path)
    inputs = dict(config.get("inputs", {}))
    analysis = dict(config.get("analysis", {}))
    grid = dict(config.get("grid", {}))
    quality = dict(config.get("quality", {}))
    validation_config = dict(config.get("validation", {}))
    tolerances = _validate_tolerance_minutes(analysis.get("tolerance_minutes", []))
    horizon_minutes = int(analysis.get("horizon_minutes", 5))
    current_window_minutes = int(analysis.get("current_window_minutes", 5))
    if horizon_minutes != 5 or current_window_minutes != 5:
        raise ValueError(
            "News-first surface datasets require fixed five-minute market windows"
        )
    if bool(analysis.get("include_closed_in_training", False)):
        raise ValueError("Closed-to-next-open news must remain audit-only")
    integer_maturity_days = grid.get("integer_maturity_days", False)
    if not isinstance(integer_maturity_days, bool):
        raise ValueError("grid.integer_maturity_days must be true or false")
    strike_bins = int(grid.get("strike_bins", 16))
    maturity_bins = int(grid.get("maturity_bins", 16))
    moneyness_min = float(grid.get("moneyness_min", 0.70))
    moneyness_max = float(grid.get("moneyness_max", 1.30))
    maturity_min_days = int(grid.get("maturity_min_days", 7))
    maturity_max_days = int(grid.get("maturity_max_days", 365))
    maturity_days_nodes_value = grid.get("maturity_days_nodes")
    maturity_days_nodes: list[float] | None = None
    if maturity_days_nodes_value is not None:
        if not isinstance(maturity_days_nodes_value, list):
            raise ValueError("grid.maturity_days_nodes must be a YAML list")
        if any(isinstance(value, bool) for value in maturity_days_nodes_value):
            raise ValueError("grid.maturity_days_nodes must contain numbers")
        maturity_days_nodes = [float(value) for value in maturity_days_nodes_value]
        if len(maturity_days_nodes) != maturity_bins:
            raise ValueError(
                "grid.maturity_days_nodes must contain exactly maturity_bins values"
            )
        if not math.isclose(
            maturity_days_nodes[0], maturity_min_days, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            maturity_days_nodes[-1], maturity_max_days, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "grid.maturity_days_nodes endpoints must match "
                "maturity_min_days and maturity_max_days"
            )
    support_strike_grid, support_maturity_grid = build_surface_grids(
        strike_bins=strike_bins,
        maturity_bins=maturity_bins,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max,
        maturity_min_days=maturity_min_days,
        maturity_max_days=maturity_max_days,
        integer_maturity_days=integer_maturity_days,
        maturity_days_nodes=maturity_days_nodes,
        dtype=np.float64,
    )
    support_query_maturities = [
        int(round(float(value))) for value in support_maturity_grid
    ]

    source_paths = {
        "market_index_sqlite": _resolve_path(inputs["market_index_sqlite"]),
        "pair_slice_metrics_csv": _resolve_path(inputs["pair_slice_metrics_csv"]),
        "session_calendar_csv": _resolve_path(inputs["session_calendar_csv"]),
        "rate_curve_csv": _resolve_path(inputs["rate_curve_csv"]),
        "news_xlsx": _resolve_path(inputs["news_xlsx"]),
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Configured {label} does not exist: {path}")
    if "with_ty_plus_trade_counts" in source_paths["news_xlsx"].name.lower():
        raise ValueError(
            "Refusing legacy news workbook with incorrect New York timestamps"
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_timezone = str(inputs.get("news_source_timezone", "Europe/London"))
    if source_timezone != "Europe/London":
        raise ValueError("The authoritative Factiva timestamps must use Europe/London")
    if tuple(analysis.get("atm_qualities", ["A", "B"])) != ("A", "B"):
        raise ValueError("This frozen training view requires ATM qualities [A, B]")

    resolved_config = {
        "news_first_vol_surfaces": {
            **config,
            "inputs": {
                **inputs,
                **{key: str(value) for key, value in source_paths.items()},
            },
        }
    }
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    pair_metrics = pd.read_csv(source_paths["pair_slice_metrics_csv"], low_memory=False)
    news_audit = parse_factiva_news(
        source_paths["news_xlsx"], source_timezone=source_timezone
    )
    expected_news_rows = int(validation_config.get("expected_news_rows", 14900))
    if len(news_audit) != expected_news_rows:
        raise ValueError(
            f"Canonical Factiva row count changed: {len(news_audit)} != {expected_news_rows}"
        )
    invalid_news = ~news_audit["timestamp_parse_status"].astype(str).eq("ok")
    if invalid_news.any():
        raise ValueError(
            f"Factiva contains {int(invalid_news.sum())} invalid timestamps"
        )

    session_calendar = TreasuryGlobexSessionCalendar.from_csv(
        source_paths["session_calendar_csv"]
    )
    pair_universe = prepare_pair_universe(
        pair_metrics,
        session_calendar=session_calendar,
        horizon_minutes=horizon_minutes,
        current_window_minutes=current_window_minutes,
    )
    expected_pairs = int(validation_config.get("expected_market_pairs", 34450))
    if len(pair_universe) != expected_pairs:
        raise ValueError(
            f"Pair universe count changed: {len(pair_universe)} != {expected_pairs}"
        )
    alignments = build_news_first_alignments(
        news_audit,
        pair_metrics,
        session_calendar=session_calendar,
        tolerances=tolerances,
        horizon_minutes=horizon_minutes,
        current_window_minutes=current_window_minutes,
    )
    print(
        f"Aligned {len(news_audit):,} Factiva rows against "
        f"{len(pair_universe):,} evaluable five-minute market pairs.",
        flush=True,
    )

    _write_csv(news_audit, output_root / "news_master.csv", compressed=True)
    shutil.copy2(
        source_paths["session_calendar_csv"],
        output_root / "cme_session_calendar_snapshot.csv",
    )

    shared_dir = output_root / "shared_surface_inputs"
    shared_dir.mkdir(parents=True, exist_ok=True)
    connection = connect_market_index(source_paths["market_index_sqlite"])
    try:
        print("Materializing the shared 30-minute surface superset...", flush=True)
        _materialize_dataset_inputs(
            connection,
            alignment=alignments[max(tolerances)],
            output_dir=shared_dir,
            source_timezone=source_timezone,
            publication_availability_lag_minutes=0,
            window_minutes=horizon_minutes,
            rate_curve_path=source_paths["rate_curve_csv"],
            min_strikes_per_expiry=int(quality.get("min_strikes_per_expiry", 2)),
            min_expiries_per_minute=int(quality.get("min_expiries_per_surface", 2)),
            option_filter_mode=str(
                quality.get("option_filter_mode", "otm_preferred_itm_fallback")
            ),
            max_itm_moneyness_distance=float(
                quality.get("max_itm_moneyness_distance", 0.05)
            ),
            alignment_policy={
                "news_alignment_mode": "news_first_forward_sensitivity",
                "tolerance_minutes": list(tolerances),
                "horizon_minutes": horizon_minutes,
                "current_window_minutes": current_window_minutes,
                "include_closed_in_training": False,
                "alignment_pair_universe": "pair_slice_metrics",
                "require_common_maturity": True,
            },
        )
    finally:
        connection.close()

    expected_counts = dict(validation_config.get("expected_counts", {}))
    enforce_counts = bool(validation_config.get("enforce_expected_counts", True))
    dataset_summaries: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, str]] = []
    support_summaries: list[dict[str, Any]] = []

    for tolerance in tolerances:
        print(f"Building cumulative tolerance {tolerance}m...", flush=True)
        group_dir = output_root / f"tolerance_{tolerance:02d}m"
        group_dir.mkdir(parents=True, exist_ok=True)
        alignment = alignments[tolerance].copy()
        alignment_path = _write_csv(alignment, group_dir / "news_market_alignment.csv")
        _write_csv(alignment, group_dir / "news_alignment_audit.csv", compressed=True)

        raw_frames = build_vol_workbook_frames(
            shared_dir,
            news_xlsx_path=source_paths["news_xlsx"],
            source_timezone=source_timezone,
            publication_availability_lag_minutes=0,
            offset_minutes=horizon_minutes,
            strike_bins=strike_bins,
            maturity_bins=maturity_bins,
            moneyness_min=moneyness_min,
            moneyness_max=moneyness_max,
            maturity_min_days=maturity_min_days,
            maturity_max_days=maturity_max_days,
            integer_maturity_days=integer_maturity_days,
            maturity_days_nodes=maturity_days_nodes,
            alignment_csv_path=alignment_path,
        )

        pair_audit = raw_frames["news_surface_pair_audit"].copy()
        extra_columns = [
            column
            for column in alignment.columns
            if column not in pair_audit.columns and column not in {"sample_id"}
        ]
        if extra_columns:
            pair_audit = pair_audit.merge(
                alignment[["news_row_id", *extra_columns]],
                on="news_row_id",
                how="left",
                validate="one_to_one",
            )
        lineage_columns = [
            "news_row_id",
            *[
                column
                for column in (
                    "workbook_sha256",
                    "lp_text_sha256",
                    "publication_group_size",
                    "publication_collision_count",
                    "article_id_group_size",
                    "lp_text_group_size",
                )
                if column in news_audit.columns and column not in pair_audit.columns
            ],
        ]
        if len(lineage_columns) > 1:
            pair_audit = pair_audit.merge(
                news_audit[lineage_columns],
                on="news_row_id",
                how="left",
                validate="one_to_one",
            )
        views = build_training_views(
            pair_audit,
            pair_metrics,
            tolerance,
            tolerances,
        )
        side_detail = raw_frames["surface_side_detail"].copy()
        side_extra = views["news_surface_pair_audit"][
            [
                "news_row_id",
                "pair_id",
                "dataset_tolerance_minutes",
                "first_included_tolerance_minutes",
                "surface_training_eligible",
                "atm_ab_training_eligible",
            ]
        ]
        side_detail = side_detail.merge(
            side_extra, on="news_row_id", how="left", validate="many_to_one"
        )
        views["surface_side_detail"] = side_detail
        support_audit = build_surface_support_audit(
            views["news_surface_pair_audit"],
            strike_grid=support_strike_grid,
            maturity_days_grid=support_query_maturities,
            tolerance_minutes=tolerance,
        )
        views["surface_support_audit"] = support_audit
        support_summary = summarize_surface_support(
            support_audit,
            tolerance_minutes=tolerance,
        )
        support_summaries.append(support_summary)

        workbook_path = write_workbook(
            group_dir / "merged_vol.xlsx",
            {sheet: views[sheet] for sheet in WORKBOOK_SHEETS},
        )
        _write_csv(
            views["matched_pair_slice_metrics"],
            group_dir / "matched_pair_slice_metrics.csv",
            compressed=True,
        )
        _write_csv(
            views["news_atm_ab_maturity_bridge"],
            group_dir / "news_atm_ab_maturity_bridge.csv",
            compressed=True,
        )
        _write_csv(
            views["pair_atm_ab_outcomes"],
            group_dir / "pair_atm_ab_outcomes.csv",
            compressed=True,
        )
        _write_csv(
            support_audit,
            group_dir / "surface_support_audit.csv",
            compressed=True,
        )
        exclusions = (
            views["news_surface_pair_audit"]
            .assign(
                exclusion_reason=lambda frame: np.where(
                    frame["surface_training_eligible"],
                    "included_surface_training",
                    np.where(
                        frame["publication_market_state"].astype(str).eq("closed"),
                        "publication_market_closed",
                        np.where(
                            frame["pair_id"].fillna("").eq(""),
                            frame.get("unmatched_reason", "unmatched"),
                            frame["exclude_reason"].replace("", "surface_not_usable"),
                        ),
                    ),
                )
            )
            .groupby("exclusion_reason", dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        _write_csv(exclusions, group_dir / "exclusion_reason_summary.csv")

        validation = validate_dataset_frames(
            views,
            tolerance_minutes=tolerance,
            expected_news_rows=expected_news_rows,
            expected_surface_cells=(
                int(grid.get("strike_bins", 16)) * int(grid.get("maturity_bins", 16))
            ),
        )
        tolerance_expected = dict(expected_counts.get(str(tolerance), {}))
        checks = _check_expected_counts(validation, tolerance_expected)
        failed_expected = [row for row in checks if row["status"] == "fail"]
        validation["expected_count_checks"] = checks
        validation["workbook_path"] = str(workbook_path)
        validation["workbook_size_bytes"] = int(workbook_path.stat().st_size)
        if failed_expected:
            validation["status"] = "fail"
            validation["failures"] = [
                *validation["failures"],
                *[f"expected:{row['check']}" for row in failed_expected],
            ]
        _write_json(group_dir / "validation_summary.json", validation)
        if validation["status"] != "pass" and enforce_counts:
            raise ValueError(
                f"{tolerance}-minute dataset validation failed: "
                f"{validation['failures']}"
            )

        dataset_summaries.append(validation)
        validation_rows.extend(
            {"tolerance_minutes": tolerance, **row} for row in checks
        )
        for table in (
            "news_surface_pair_audit",
            "surface_side_detail",
            "gan_input_ready",
            "gan_input_atm_ab",
            "news_atm_ab_maturity_bridge",
            "pair_atm_ab_outcomes",
            "matched_pair_slice_metrics",
            "surface_support_audit",
        ):
            schema = _field_dictionary(
                {f"tolerance_{tolerance:02d}m/{table}": views[table]}
            )
            dictionary_rows.extend(schema.to_dict(orient="records"))

        del raw_frames, pair_audit, views, side_detail, side_extra, exclusions
        gc.collect()

    summary_frame = pd.DataFrame(dataset_summaries)
    _write_csv(summary_frame, output_root / "dataset_summary.csv")
    _write_csv(
        pd.DataFrame(support_summaries),
        output_root / "surface_support_summary.csv",
    )
    _write_csv(pd.DataFrame(validation_rows), output_root / "data_quality_summary.csv")
    field_dictionary = pd.DataFrame(dictionary_rows).drop_duplicates(
        ["table", "column"]
    )
    _write_csv(field_dictionary, output_root / "field_dictionary.csv")
    _write_json(
        output_root / "field_dictionary.json",
        field_dictionary.to_dict(orient="records"),
    )

    source_manifest = _source_manifest(source_paths)
    _write_csv(source_manifest, output_root / "source_manifest.csv")
    config_source_path = Path(config_path).expanduser().resolve()
    _write_csv(_code_manifest(config_source_path), output_root / "code_manifest.csv")
    git_state = _git_state()
    (output_root / "run_manifest.env").write_text(
        "\n".join(
            [
                f"created_at_utc={_now_utc()}",
                f"git_commit={git_state['commit']}",
                f"git_status={json.dumps(git_state['status'])}",
                "analysis_role=news_first_surface_sensitivity",
                "market_horizon_minutes=5",
                "tolerances_minutes=" + ",".join(str(value) for value in tolerances),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overall_status = (
        "pass"
        if all(item.get("status") == "pass" for item in dataset_summaries)
        else "fail"
    )
    _write_json(
        output_root / "validation_summary.json",
        {
            "status": overall_status,
            "created_at_utc": _now_utc(),
            "news_row_count": int(len(news_audit)),
            "pair_universe_count": int(len(pair_universe)),
            "tolerances_minutes": list(tolerances),
            "expected_counts_basis": validation_config.get("expected_counts_basis", ""),
            "reconciliation_note": validation_config.get("reconciliation_note", ""),
            "datasets": dataset_summaries,
            "interpretation": (
                "Only shift=0 is a clean pre/post boundary. Shifted samples are "
                "news-known-before-forecast sensitivity data, not causal shocks."
            ),
        },
    )
    output_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "dataset_output_sha256.txt"
    )
    (output_root / "dataset_output_sha256.txt").write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(output_root)}\n"
            for path in output_files
        ),
        encoding="utf-8",
    )
    print(f"Completed news-first vol-surface export: {output_root}", flush=True)
    return output_root


__all__ = [
    "WORKBOOK_SHEETS",
    "build_surface_support_audit",
    "build_training_views",
    "run_news_first_vol_surfaces",
    "summarize_surface_support",
    "validate_dataset_frames",
]
