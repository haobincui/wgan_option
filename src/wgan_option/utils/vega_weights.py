"""Current-only Black-76 quote Vega grids with explicit source multiplicity.

The ACT/365 clock belongs to each observed quote. Business days are used only
to interpolate the resulting Vega surface, never as a pricing-year proxy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

CELL_WEIGHT_MODE = "black76_current_vega_v1"
SCHEMA_VERSION = 1
SECONDS_PER_YEAR = 365.0 * 86400.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"Vega provenance requires an explicit UTC-offset timestamp: {value!r}")
    return result.tz_convert("UTC")


def utc_anchor(value: Any) -> str:
    result = utc_timestamp(value)
    if result != result.floor("min"):
        raise ValueError(f"Current snapshot must be an exact minute: {value!r}")
    return result.isoformat().replace("+00:00", "Z")


def black76_quote_vega(row: Mapping[str, Any]) -> float:
    """Price derivative per unit annual IV, at the quote's recorded inputs."""
    required = (
        "spot", "strike", "implied_vol", "discount_factor", "continuous_rate",
        "trade_datetime_utc", "expiration_datetime_utc", "underlying_trade_datetime_utc",
        "calibration_datetime_utc", "rate_curve_date", "rate_curve_sha256",
    )
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing Black-76 quote metadata: {missing}")
    if str(row.get("pricing_model", "")) != "black76":
        raise ValueError("Current Vega quotes must use pricing_model=black76.")
    quote = utc_timestamp(row["trade_datetime_utc"])
    expiry = utc_timestamp(row["expiration_datetime_utc"])
    underlying = utc_timestamp(row["underlying_trade_datetime_utc"])
    anchor = utc_timestamp(row["calibration_datetime_utc"])
    if not underlying <= quote < anchor:
        raise ValueError("Vega quote provenance must satisfy underlying <= quote < current anchor.")
    if row.get("window_start_utc") and row.get("window_end_utc"):
        if not utc_timestamp(row["window_start_utc"]) <= quote < utc_timestamp(row["window_end_utc"]):
            raise ValueError("Vega quote is outside its recorded half-open current window.")
        if utc_timestamp(row["window_end_utc"]) != anchor:
            raise ValueError("Current Vega quote window ends at a different snapshot.")
    tau = float((expiry.value - quote.value) / 1e9 / SECONDS_PER_YEAR)
    forward, strike, vol, discount, rate = (
        float(row[key]) for key in ("spot", "strike", "implied_vol", "discount_factor", "continuous_rate")
    )
    if not all(math.isfinite(x) for x in (forward, strike, vol, discount, rate, tau)):
        raise ValueError("Black-76 quote inputs must be finite.")
    if min(forward, strike, vol, discount, tau) <= 0:
        raise ValueError("Black-76 quote inputs must be positive (except the rate).")
    if not math.isclose(discount, math.exp(-rate * tau), abs_tol=2e-12, rel_tol=2e-12):
        raise ValueError("Recorded discount factor does not match the quote ACT/365 clock.")
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * tau) / (vol * math.sqrt(tau))
    vega = discount * forward * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(tau)
    if not math.isfinite(vega) or vega <= 0:
        raise ValueError("Black-76 quote Vega must be finite and positive; no proxy fallback is allowed.")
    return vega


def quote_identity(row: Mapping[str, Any]) -> tuple[int, str, float, float]:
    """Raw parser identity; keep UTC nanoseconds and original trade volume."""
    return (
        int(utc_timestamp(row["trade_datetime_utc"]).value),
        str(row["contract_id"]), float(row["price"]), float(row["weight"]),
    )


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights))
    if not pairs or any(not math.isfinite(float(v)) or not math.isfinite(float(w)) or w <= 0 for v, w in pairs):
        raise ValueError("A Vega bucket requires finite observations and positive volumes.")
    threshold = sum(float(w) for _, w in pairs) / 2.0
    running = 0.0
    for value, weight in pairs:
        running += float(weight)
        if running >= threshold:
            return float(value)
    return float(pairs[-1][0])


def restore_quote_multiplicities(
    rows: Sequence[Mapping[str, Any]],
    source_manifest_path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[tuple[int, str, float, float], int], list[dict[str, Any]]]:
    """Recover duplicate counts from SHA-verified raw files, without repricing.

    Read every source file whose declared filename date range can contain a
    desired UTC quote. Matching reproduces the original parser's finite,
    positive Price/Volume filter; Type was not a filter in that parser.
    """
    wanted = {quote_identity(row) for row in rows}
    if not wanted:
        raise ValueError("No accepted current quotes were supplied.")
    dates = {pd.Timestamp(key[0], tz="UTC").date() for key in wanted}
    contracts = {key[1] for key in wanted}
    with Path(source_manifest_path).open(newline="") as handle:
        sources = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for source in sources:
        if source.get("scan_status") != "complete":
            raise ValueError("Raw source manifest includes an incomplete source scan.")
        path = Path(source["source_path"])
        match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", path.name)
        if match:
            lower, upper = (pd.Timestamp(value).date() for value in match.groups())
            if not any(lower <= day <= upper for day in dates):
                continue
        selected.append(dict(source))
    counts: Counter[tuple[int, str, float, float]] = Counter()
    audits: list[dict[str, Any]] = []
    for index, source in enumerate(selected, 1):
        path = Path(source["source_path"])
        if not path.is_file():
            raise ValueError(f"Raw source for Vega multiplicity is missing: {path}")
        digest = sha256_file(path)
        if digest != source["sha256"] or path.stat().st_size != int(source["size_bytes"]):
            raise ValueError(f"Raw source SHA/size does not match the frozen manifest: {path}")
        matched = 0
        for chunk in pd.read_csv(path, usecols=["#RIC", "Date-Time", "Price", "Volume"], chunksize=100000):
            chunk = chunk[chunk["#RIC"].isin(contracts)].copy()
            if chunk.empty:
                continue
            chunk["Price"] = pd.to_numeric(chunk["Price"], errors="coerce")
            chunk["Volume"] = pd.to_numeric(chunk["Volume"], errors="coerce")
            chunk = chunk[chunk["Price"].notna() & (chunk["Price"] > 0) & chunk["Volume"].notna() & (chunk["Volume"] > 0)].copy()
            chunk["_time"] = pd.to_datetime(chunk["Date-Time"], errors="coerce", utc=True)
            chunk = chunk[chunk["_time"].notna()]
            for contract, price, volume, timestamp in chunk[["#RIC", "Price", "Volume", "_time"]].itertuples(index=False, name=None):
                key = (int(timestamp.value), str(contract), float(price), float(volume))
                if key in wanted:
                    counts[key] += 1
                    matched += 1
        audits.append({"source_path": str(path.resolve()), "sha256": digest, "size_bytes": path.stat().st_size, "matched_raw_rows": matched})
        if progress and (index == 1 or index % 20 == 0 or index == len(selected)):
            progress(f"SHA-verified raw files {index}/{len(selected)}; matching quote occurrences {sum(counts.values())}")
    absent = wanted - counts.keys()
    if absent:
        raise ValueError(f"Cannot restore {len(absent)} accepted quote identities from frozen raw sources; examples: {list(absent)[:3]}")
    return dict(counts), audits


def build_current_vega_grid(
    rows: Sequence[Mapping[str, Any]],
    canonical_params: Mapping[str, Any],
    multiplicities: Mapping[tuple[int, str, float, float], int],
    strike_grid: Sequence[float],
    maturity_grid_days: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Verify canonical q/IV buckets, then interpolate current quote Vegas."""
    from film_wgan.support import parse_raw_surface_params, raw_support_mask

    params = parse_raw_surface_params(canonical_params)
    buckets: dict[tuple[int, float], list[tuple[float, float, float, float]]] = defaultdict(list)
    identities: set[tuple[int, str, float, float]] = set()
    expiries: dict[int, set[str]] = defaultdict(set)
    quote_audit: list[dict[str, Any]] = []
    anchor_set = {utc_anchor(row["calibration_datetime_utc"]) for row in rows}
    if len(anchor_set) != 1:
        raise ValueError("A current Vega grid must be built from exactly one snapshot.")
    for row in rows:
        if str(row.get("passes_precalib_filter", "")).lower() != "true":
            raise ValueError("Vega buckets can only consume accepted precalibration quotes.")
        identity = quote_identity(row)
        if identity in identities:
            raise ValueError("Deduplicated quote provenance is ambiguous within a current snapshot.")
        identities.add(identity)
        count = int(multiplicities.get(identity, 0))
        if count < 1:
            raise ValueError("An accepted quote has no restored raw multiplicity.")
        vega = black76_quote_vega(row)
        day, strike = int(row["business_days"]), float(row["strike"])
        volume = float(row["weight"]) * count
        q, iv = float(row["percent_strike"]), float(row["implied_vol"])
        if not math.isclose(q, strike / float(row["spot"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Quote percent strike disagrees with its recorded F/K.")
        expiries[day].add(str(row["expiration_datetime_utc"]))
        buckets[(day, strike)].append((q, iv, vega, volume))
        quote_audit.append({
            "current_snapshot_time_utc": next(iter(anchor_set)),
            "trade_datetime_utc": row["trade_datetime_utc"], "contract_id": identity[1],
            "business_days": day, "strike": strike, "price": identity[2],
            "original_volume": identity[3], "multiplicity": count,
            "restored_volume": volume, "black76_quote_vega": vega,
        })
    if any(len(values) != 1 for values in expiries.values()):
        raise ValueError("A business-day bucket contains conflicting actual expiries.")
    rebuilt: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for (day, _), observations in buckets.items():
        volumes = [point[3] for point in observations]
        q = sum(point[0] * point[3] for point in observations) / sum(volumes)
        rebuilt[day].append((q, weighted_median([x[1] for x in observations], volumes), weighted_median([x[2] for x in observations], volumes)))
    verified: list[tuple[np.ndarray, np.ndarray]] = []
    q_error = iv_error = 0.0
    for day, canonical_q, canonical_iv in zip(params["business_days"], params["percent_strikes"], params["implied_vols"]):
        points = sorted(rebuilt.get(day, []))
        by_q: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for q, iv, vega in points:
            by_q[q].append((iv, vega))
        qq = np.asarray(sorted(by_q), dtype=np.float64)
        ivs = np.asarray([np.mean([x[0] for x in by_q[q]]) for q in qq], dtype=np.float64)
        vegas = np.asarray([np.mean([x[1] for x in by_q[q]]) for q in qq], dtype=np.float64)
        if len(qq) != len(canonical_q):
            raise ValueError(f"Current raw canonical bucket count cannot be reproduced for day={day}.")
        this_q_error = float(np.max(np.abs(qq - canonical_q)))
        this_iv_error = float(np.max(np.abs(ivs - canonical_iv)))
        q_error, iv_error = max(q_error, this_q_error), max(iv_error, this_iv_error)
        if this_q_error > 1e-10 or this_iv_error > 1e-10:
            raise ValueError(f"Current canonical q/IV buckets differ after restoration: day={day}, q_error={this_q_error}, iv_error={this_iv_error}.")
        # Canonical coordinates are the reference, after equality verification.
        verified.append((np.asarray(canonical_q, dtype=np.float64), vegas))
    support = raw_support_mask(params, strike_grid=strike_grid, maturity_days_grid=maturity_grid_days)
    grid = np.zeros(support.shape, dtype=np.float64)
    days = np.asarray(params["business_days"], dtype=np.float64)
    for i, day in enumerate(maturity_grid_days):
        if not np.any(support[i]):
            continue
        exact = np.flatnonzero(np.isclose(days, day, atol=1e-8, rtol=0))
        q_values = np.asarray(strike_grid, dtype=np.float64)[support[i]]
        if exact.size:
            q, values = verified[int(exact[0])]
            grid[i, support[i]] = np.interp(q_values, q, values)
        else:
            upper = int(np.searchsorted(days, day, side="right"))
            lower = upper - 1
            alpha = float((day - days[lower]) / (days[upper] - days[lower]))
            lo = np.interp(q_values, *verified[lower])
            hi = np.interp(q_values, *verified[upper])
            grid[i, support[i]] = (1 - alpha) * lo + alpha * hi
    if np.any(~np.isfinite(grid)) or np.any(grid[support] <= 0) or np.any(grid[~support] != 0):
        raise ValueError("Current Vega grid failed finite/positive/support validation.")
    audit = {
        "current_snapshot_time_utc": next(iter(anchor_set)), "accepted_quote_rows": len(rows),
        "restored_quote_occurrences": sum(int(multiplicities[quote_identity(row)]) for row in rows),
        "restored_duplicate_occurrences": sum(int(multiplicities[quote_identity(row)]) - 1 for row in rows),
        "canonical_slice_count": len(days), "canonical_point_count": sum(len(q) for q in params["percent_strikes"]),
        "maximum_q_reproduction_error": q_error, "maximum_iv_reproduction_error": iv_error,
        "current_strict_support_cells": int(support.sum()),
        "minimum_positive_vega": float(grid[support].min()) if support.any() else None,
        "maximum_vega": float(grid.max()),
        "maximum_quote_datetime_utc": max(utc_timestamp(row["trade_datetime_utc"]) for row in rows).isoformat().replace("+00:00", "Z"),
        "canonical_params_sha256": _json_hash(params),
    }
    return grid.astype(np.float32), audit, quote_audit


def grid_fingerprint(strike_grid: Sequence[float], maturity_grid_days: Sequence[float]) -> str:
    return _json_hash({"strike_grid": [float(x) for x in strike_grid], "maturity_grid_days": [float(x) for x in maturity_grid_days]})


def cell_weight_fingerprint(anchor: str, grid: np.ndarray, grid_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(utc_anchor(anchor).encode())
    digest.update(grid_sha256.encode())
    digest.update(np.asarray(grid, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _read_manifest(path: str, mtime_ns: int, size: int) -> tuple[dict[str, Any], str]:
    del mtime_ns, size
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("cell_weight_mode") != CELL_WEIGHT_MODE or payload.get("mode") != CELL_WEIGHT_MODE:
        raise ValueError("Unsupported current Vega manifest schema/mode.")
    return payload, sha256_file(path)


@lru_cache(maxsize=4)
def _read_archive(path: str, mtime_ns: int, size: int, expected_sha256: str) -> dict[str, Any]:
    del mtime_ns, size
    if sha256_file(path) != expected_sha256:
        raise ValueError("Current Vega NPZ SHA does not match its manifest.")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    anchors = [str(x) for x in payload["current_snapshot_time_utc"]]
    if len(anchors) != len(set(anchors)) or anchors != sorted(anchors):
        raise ValueError("Current Vega archive snapshot keys are duplicated or unordered.")
    grid = payload["vega_weights"]
    expected_shape = (len(anchors), len(payload["maturity_grid_days"]), len(payload["strike_grid"]))
    if grid.dtype != np.float32 or grid.shape != expected_shape or np.any(~np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("Current Vega archive contains invalid weight arrays.")
    support = payload["current_strict_support"]
    if support.shape != grid.shape or np.any(grid[~support] != 0) or np.any(grid[support] <= 0):
        raise ValueError("Current Vega archive violates current strict support.")
    grid_sha = grid_fingerprint(payload["strike_grid"], payload["maturity_grid_days"])
    fingerprints = payload["cell_weight_fingerprints"]
    if len(fingerprints) != len(anchors):
        raise ValueError("Current Vega archive fingerprints have invalid length.")
    for index, anchor in enumerate(anchors):
        if cell_weight_fingerprint(anchor, grid[index], grid_sha) != str(fingerprints[index]):
            raise ValueError("Current Vega cell fingerprint mismatch.")
    payload["anchor_index"] = {anchor: index for index, anchor in enumerate(anchors)}
    payload["grid_fingerprint"] = grid_sha
    return payload


def load_vega_weight_grid(
    manifest_path: str | Path,
    current_snapshot_time_utc: str,
    strike_grid: Sequence[float],
    maturity_grid_days: Sequence[float],
    expected_manifest_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a SHA-checked current-only float32 [H,W] grid with bounded caches."""
    path = Path(manifest_path).resolve()
    stat = path.stat()
    manifest, manifest_sha = _read_manifest(str(path), stat.st_mtime_ns, stat.st_size)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise ValueError("Current Vega manifest SHA does not match the experiment pin.")
    archive_path = (path.parent / manifest["weights_path"]).resolve()
    stat = archive_path.stat()
    archive = _read_archive(str(archive_path), stat.st_mtime_ns, stat.st_size, manifest["weights_sha256"])
    if archive["grid_fingerprint"] != manifest["grid_fingerprint"]:
        raise ValueError("Current Vega manifest/archive grid fingerprints differ.")
    for provided, stored in ((strike_grid, archive["strike_grid"]), (maturity_grid_days, archive["maturity_grid_days"])):
        array = np.asarray(provided, dtype=np.float64)
        if array.shape != stored.shape or not np.allclose(array, stored, rtol=0, atol=1e-7):
            raise ValueError("Requested surface grid differs from the pinned current Vega grid.")
    anchor = utc_anchor(current_snapshot_time_utc)
    if anchor not in archive["anchor_index"]:
        raise ValueError(f"Current snapshot is absent from the Vega cache: {anchor}")
    index = archive["anchor_index"][anchor]
    return archive["vega_weights"][index].copy(), {
        "cell_weight_mode": CELL_WEIGHT_MODE, "current_snapshot_time_utc": anchor,
        "cell_weight_fingerprint": str(archive["cell_weight_fingerprints"][index]),
        "manifest_sha256": manifest_sha, "grid_fingerprint": archive["grid_fingerprint"],
        "source_method": manifest["source_method"],
    }
