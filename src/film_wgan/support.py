"""Train-fold-only support grids for non-parametric raw-vol surfaces."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wgan_option.merge_support import build_surface_from_params


SUPPORT_SCHEMA_VERSION = 1
SUPPORT_METHOD = "raw_bracket_intersection_v1"


def _sha256_ids(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_raw_surface_params(value: Any) -> dict[str, list[Any]]:
    """Parse and validate the raw surface schema stored in merged workbooks."""

    if isinstance(value, Mapping):
        payload = dict(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Raw surface parameters are missing.")
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Raw surface parameters are empty.")
        payload = {}
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
                break
        if not payload:
            raise ValueError("Raw surface parameters are not valid JSON.")

    required = ("business_days", "percent_strikes", "implied_vols")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Raw surface parameters are missing keys: {missing}")
    lengths = [len(payload[key]) if isinstance(payload[key], list) else -1 for key in required]
    if len(set(lengths)) != 1 or lengths[0] < 2:
        raise ValueError(
            "Raw surface parameters require matching outer lists and at least two maturity slices."
        )

    cleaned: dict[int, list[tuple[float, float]]] = {}
    for business_day, strikes, vols in zip(
        payload["business_days"],
        payload["percent_strikes"],
        payload["implied_vols"],
    ):
        if not isinstance(strikes, list) or not isinstance(vols, list) or len(strikes) != len(vols):
            raise ValueError("Each raw surface slice requires matching strike and IV lists.")
        day = int(round(float(business_day)))
        if day <= 0:
            continue
        points = cleaned.setdefault(day, [])
        for strike, vol in zip(strikes, vols):
            strike_value = float(strike)
            vol_value = float(vol)
            if (
                np.isfinite(strike_value)
                and np.isfinite(vol_value)
                and strike_value > 0.0
                and vol_value > 0.0
            ):
                points.append((strike_value, vol_value))

    business_days: list[int] = []
    percent_strikes: list[list[float]] = []
    implied_vols: list[list[float]] = []
    for day, points in sorted(cleaned.items()):
        by_strike: dict[float, list[float]] = {}
        for strike, vol in points:
            by_strike.setdefault(float(strike), []).append(float(vol))
        if not by_strike:
            continue
        business_days.append(day)
        percent_strikes.append(sorted(by_strike))
        implied_vols.append(
            [float(np.mean(by_strike[strike])) for strike in percent_strikes[-1]]
        )
    if len(business_days) < 2:
        raise ValueError("Raw surface parameters contain fewer than two valid maturity slices.")
    return {
        "business_days": business_days,
        "percent_strikes": percent_strikes,
        "implied_vols": implied_vols,
    }


def raw_support_mask(
    surface_params: Mapping[str, Any],
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> np.ndarray:
    """Return cells supported without strike clamping or maturity extrapolation.

    At an exact maturity, the observed strike range of that slice is used. Between
    maturities, a strike must lie in the intersection of the two bracketing slice
    ranges because total variance uses both slices.
    """

    params = parse_raw_surface_params(surface_params)
    days = np.asarray(params["business_days"], dtype=np.float64)
    bounds = np.asarray(
        [
            (float(min(strikes)), float(max(strikes)))
            for strikes in params["percent_strikes"]
        ],
        dtype=np.float64,
    )
    strikes = np.asarray(strike_grid, dtype=np.float64)
    maturities = np.asarray(maturity_days_grid, dtype=np.float64)
    mask = np.zeros((maturities.size, strikes.size), dtype=bool)

    for maturity_index, maturity in enumerate(maturities):
        if maturity < days[0] or maturity > days[-1]:
            continue
        exact = np.flatnonzero(np.isclose(days, maturity, atol=1e-8, rtol=0.0))
        if exact.size:
            lower, upper = bounds[int(exact[0])]
        else:
            upper_index = int(np.searchsorted(days, maturity, side="right"))
            lower_index = upper_index - 1
            lower = max(bounds[lower_index, 0], bounds[upper_index, 0])
            upper = min(bounds[lower_index, 1], bounds[upper_index, 1])
        if lower > upper:
            continue
        mask[maturity_index] = (strikes >= lower - 1e-12) & (strikes <= upper + 1e-12)
    return mask


def reconstruct_raw_surface(
    surface_params: Mapping[str, Any],
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> np.ndarray:
    """Reconstruct raw IV values on one frozen support grid."""

    params = parse_raw_surface_params(surface_params)
    surface = build_surface_from_params(
        surface_model="raw",
        surface_params=params,
        valuation_date=date(2000, 1, 3),
    )
    values = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid],
        business_days=[int(round(float(value))) for value in maturity_days_grid],
        forward=1.0,
    )
    result = np.asarray(values, dtype=np.float32)
    expected_shape = (len(maturity_days_grid), len(strike_grid))
    if result.shape != expected_shape or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"Invalid reconstructed raw surface with shape {result.shape}.")
    return result


@dataclass(frozen=True)
class RawSurfaceSupportArtifact:
    """Frozen support-grid definition fitted only on one fold's train pairs."""

    schema_version: int
    method: str
    created_at_utc: str
    input_workbook_sha256: str
    train_pair_ids_sha256: str
    train_pair_count: int
    strike_bins: int
    maturity_bins: int
    quantile_low: float
    quantile_high: float
    strike_grid: list[float]
    maturity_days_grid: list[float]
    train_current_support_rate: list[list[float]]
    train_target_support_rate: list[list[float]]
    train_pair_support_rate: list[list[float]]
    train_pair_supported_cell_count_min: int
    train_pair_supported_cell_count_median: float
    train_pair_supported_cell_count_max: int

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "RawSurfaceSupportArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = cls(**payload)
        if artifact.schema_version != SUPPORT_SCHEMA_VERSION or artifact.method != SUPPORT_METHOD:
            raise ValueError(
                f"Unsupported raw support artifact schema/method: "
                f"{artifact.schema_version}/{artifact.method}"
            )
        return artifact


def fit_raw_support_artifact(
    records: Sequence[Mapping[str, Any]],
    *,
    input_workbook_sha256: str,
    strike_bins: int = 16,
    maturity_bins: int = 16,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> RawSurfaceSupportArtifact:
    """Fit a rectangular grid from raw points in training records only."""

    if not records:
        raise ValueError("Cannot fit raw support artifact without training records.")
    if int(strike_bins) < 2 or int(maturity_bins) < 2:
        raise ValueError("Raw support grid requires at least two strike and maturity bins.")
    if not 0.0 <= float(quantile_low) < float(quantile_high) <= 1.0:
        raise ValueError("Support grid quantiles must satisfy 0 <= low < high <= 1.")

    parsed_records: list[tuple[str, dict[str, list[Any]], dict[str, list[Any]]]] = []
    all_strikes: list[float] = []
    all_days: list[float] = []
    for record in records:
        pair_id = str(record["surface_pair_id"])
        current_params = parse_raw_surface_params(record["current_surface_params"])
        target_params = parse_raw_surface_params(record["target_surface_params"])
        parsed_records.append((pair_id, current_params, target_params))
        for params in (current_params, target_params):
            all_days.extend(float(value) for value in params["business_days"])
            for strike_slice in params["percent_strikes"]:
                all_strikes.extend(float(value) for value in strike_slice)

    strike_low, strike_high = np.quantile(
        np.asarray(all_strikes, dtype=np.float64),
        [float(quantile_low), float(quantile_high)],
    )
    maturity_low, maturity_high = np.quantile(
        np.asarray(all_days, dtype=np.float64),
        [float(quantile_low), float(quantile_high)],
    )
    maturity_low = float(np.ceil(maturity_low))
    maturity_high = float(np.floor(maturity_high))
    if strike_low >= strike_high or maturity_low >= maturity_high:
        raise ValueError(
            "Training raw points do not define a non-empty support grid after quantile trimming."
        )

    strike_grid = np.linspace(strike_low, strike_high, int(strike_bins), dtype=np.float64)
    maturity_grid = np.rint(
        np.linspace(maturity_low, maturity_high, int(maturity_bins), dtype=np.float64)
    )
    if np.unique(maturity_grid).size != int(maturity_bins):
        raise ValueError(
            "Training maturity support is too narrow for the requested number of unique bins."
        )

    current_masks: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    pair_masks: list[np.ndarray] = []
    pair_ids: list[str] = []
    for pair_id, current_params, target_params in parsed_records:
        current_mask = raw_support_mask(
            current_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        target_mask = raw_support_mask(
            target_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        current_masks.append(current_mask)
        target_masks.append(target_mask)
        pair_masks.append(current_mask & target_mask)
        pair_ids.append(pair_id)

    current_stack = np.stack(current_masks).astype(np.float64)
    target_stack = np.stack(target_masks).astype(np.float64)
    pair_stack = np.stack(pair_masks).astype(np.float64)
    supported_counts = pair_stack.reshape(pair_stack.shape[0], -1).sum(axis=1).astype(int)
    return RawSurfaceSupportArtifact(
        schema_version=SUPPORT_SCHEMA_VERSION,
        method=SUPPORT_METHOD,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        input_workbook_sha256=str(input_workbook_sha256),
        train_pair_ids_sha256=_sha256_ids(pair_ids),
        train_pair_count=len(pair_ids),
        strike_bins=int(strike_bins),
        maturity_bins=int(maturity_bins),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        strike_grid=[float(value) for value in strike_grid],
        maturity_days_grid=[float(value) for value in maturity_grid],
        train_current_support_rate=current_stack.mean(axis=0).tolist(),
        train_target_support_rate=target_stack.mean(axis=0).tolist(),
        train_pair_support_rate=pair_stack.mean(axis=0).tolist(),
        train_pair_supported_cell_count_min=int(supported_counts.min()),
        train_pair_supported_cell_count_median=float(np.median(supported_counts)),
        train_pair_supported_cell_count_max=int(supported_counts.max()),
    )


def validate_support_artifact_lineage(
    artifact: RawSurfaceSupportArtifact,
    *,
    train_pair_ids: Sequence[str],
    input_workbook_sha256: str,
) -> None:
    expected_ids_sha = _sha256_ids([str(value) for value in train_pair_ids])
    if artifact.train_pair_ids_sha256 != expected_ids_sha:
        raise ValueError(
            "Raw support artifact was not fitted on the resolved fold's ordered training pairs."
        )
    if artifact.input_workbook_sha256 != str(input_workbook_sha256):
        raise ValueError("Raw support artifact input workbook SHA256 mismatch.")
