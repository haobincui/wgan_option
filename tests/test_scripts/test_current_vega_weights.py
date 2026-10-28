"""Black-76 Vega provenance, duplicate restoration and cache integrity tests."""

from __future__ import annotations

import csv
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from wgan_option.market.black76 import black76_price
from wgan_option.utils.vega_weights import (
    CELL_WEIGHT_MODE,
    _read_archive,
    _read_manifest,
    black76_quote_vega,
    build_current_vega_grid,
    cell_weight_fingerprint,
    grid_fingerprint,
    load_vega_weight_grid,
    quote_identity,
    restore_quote_multiplicities,
    sha256_file,
    weighted_median,
)


def quote(*, day=2, q=1.0, forward=120.0, iv=0.2, volume=10, contract="TYTEST", time="2026-10-07T13:01:00Z"):
    expiry = "2026-10-09T21:00:00Z" if day == 2 else "2026-10-16T21:00:00Z"
    tau = (pd.Timestamp(expiry).value - pd.Timestamp(time).value) / 1e9 / (365 * 86400)
    return {
        "contract_id": contract, "trade_datetime_utc": time,
        "calibration_datetime_utc": "2026-10-07T13:05:00Z",
        "underlying_trade_datetime_utc": time,
        "expiration_datetime_utc": expiry,
        "window_start_utc": "2026-10-07T13:00:00Z",
        "window_end_utc": "2026-10-07T13:05:00Z",
        "spot": forward, "strike": forward * q,
        "implied_vol": iv, "discount_factor": math.exp(-0.03 * tau),
        "continuous_rate": 0.03, "rate_curve_date": "2026-10-07",
        "rate_curve_sha256": "frozen_curve", "price": 1.0,
        "weight": volume, "percent_strike": q, "business_days": day,
        "pricing_model": "black76", "passes_precalib_filter": "true",
    }


class CurrentVegaTests(unittest.TestCase):
    def tearDown(self):
        _read_archive.cache_clear()
        _read_manifest.cache_clear()

    def test_quote_vega_matches_black76_price_finite_difference(self):
        row = quote(q=0.985)
        tau = (pd.Timestamp(row["expiration_datetime_utc"]).value - pd.Timestamp(row["trade_datetime_utc"]).value) / 1e9 / (365 * 86400)
        epsilon = 1e-6
        def price(vol):
            return black76_price(futures_price=row["spot"], strike=row["strike"], tau=tau, volatility=vol, discount_factor=row["discount_factor"], option_type="CALL")
        finite_difference = (price(row["implied_vol"] + epsilon) - price(row["implied_vol"] - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(black76_quote_vega(row), finite_difference, places=7)
        different_business_clock = dict(row, business_days=250)
        self.assertEqual(black76_quote_vega(row), black76_quote_vega(different_business_clock))

    def test_causal_and_missing_metadata_fail_closed(self):
        for changes in (
            {"underlying_trade_datetime_utc": "2026-10-07T13:02:00Z"},
            {"calibration_datetime_utc": "2026-10-07T13:01:00Z"},
            {"expiration_datetime_utc": ""},
            {"discount_factor": 0.5},
            {"pricing_model": "legacy_black_scholes"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                black76_quote_vega(dict(quote(), **changes))

    def test_source_duplicate_restoration_and_sha_pin(self):
        row = quote(volume=179, time="2026-10-07T13:01:00.666137588Z")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw_2026-10-07_2026-10-08.csv.gz"
            with gzip.open(raw, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["#RIC", "Date-Time", "Price", "Volume"])
                for _ in range(2):
                    writer.writerow([row["contract_id"], "2026-10-07T08:01:00.666137588-05", row["price"], row["weight"]])
                writer.writerow([row["contract_id"], "2026-10-07T08:01:00.666137589-05", row["price"], row["weight"]])
            manifest = root / "source_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_path", "sha256", "size_bytes", "scan_status"])
                writer.writeheader()
                writer.writerow({"source_path": str(raw), "sha256": sha256_file(raw), "size_bytes": raw.stat().st_size, "scan_status": "complete"})
            counts, audit = restore_quote_multiplicities([row], manifest)
            self.assertEqual(counts[quote_identity(row)], 2)
            self.assertEqual(audit[0]["matched_raw_rows"], 2)
            with raw.open("ab") as handle:
                handle.write(b"source drift")
            with self.assertRaisesRegex(ValueError, "SHA/size"):
                restore_quote_multiplicities([row], manifest)

    def test_bucket_multiplicity_and_linear_vega_interpolation(self):
        rows = [
            quote(day=d, q=q, contract=f"TY{d}_{q}", iv=iv)
            for d in (2, 7) for q, iv in ((0.98, 0.2), (1.02, 0.3))
        ]
        # Same strike, different F: original raw duplicate volume changes q and
        # the selected median IV, exactly the failure seen in the real cache.
        extra = dict(rows[0], spot=rows[0]["spot"] * 1.01, percent_strike=0.98 / 1.01, implied_vol=0.25, price=1.1)
        rows.append(extra)
        counts = {quote_identity(row): 1 for row in rows}
        counts[quote_identity(extra)] = 2
        canonical_q = (0.98 + 2 * extra["percent_strike"]) / 3
        params = {"business_days": [2, 7], "percent_strikes": [[canonical_q, 1.02], [0.98, 1.02]], "implied_vols": [[0.25, 0.3], [0.2, 0.3]]}
        grid, audit, _ = build_current_vega_grid(rows, params, counts, [0.9, 1.0, 1.1], [1, 2, 4.5, 7, 8])
        self.assertEqual(grid.dtype, np.float32)
        self.assertTrue(np.all(grid[:, [0, 2]] == 0))
        self.assertTrue(np.all(grid[[0, 4]] == 0))
        self.assertAlmostEqual(float(grid[2, 1]), (float(grid[1, 1]) + float(grid[3, 1])) / 2, places=6)
        self.assertEqual(audit["restored_duplicate_occurrences"], 1)
        with self.assertRaisesRegex(ValueError, "canonical q/IV"):
            build_current_vega_grid(rows, params, {key: 1 for key in counts}, [1.0], [2, 7])
        with self.assertRaises(ValueError):
            build_current_vega_grid(rows + [dict(rows[0], calibration_datetime_utc="2026-10-07T13:06:00Z")], params, counts, [1.0], [2, 7])

    def test_weighted_median_matches_original_half_volume_rule(self):
        self.assertEqual(weighted_median([5.0, 1.0], [1.0, 1.0]), 1.0)
        with self.assertRaises(ValueError):
            weighted_median([1.0], [0.0])

    def test_loader_grid_anchor_integrity_and_cache(self):
        anchor = "2026-10-07T13:05:00Z"
        q, days = [0.97, 1.03], [2.0, 7.0]
        grid = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        fingerprint = cell_weight_fingerprint(anchor, grid, grid_fingerprint(q, days))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "weights.npz"
            np.savez_compressed(archive, current_snapshot_time_utc=np.asarray([anchor]), vega_weights=grid[None], current_strict_support=(grid > 0)[None], strike_grid=np.asarray(q), maturity_grid_days=np.asarray(days), cell_weight_fingerprints=np.asarray([fingerprint]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "mode": CELL_WEIGHT_MODE, "cell_weight_mode": CELL_WEIGHT_MODE, "weights_path": archive.name, "weights_sha256": sha256_file(archive), "grid_fingerprint": grid_fingerprint(q, days), "source_method": "test_current_only"}))
            manifest_sha = sha256_file(manifest)
            with patch("wgan_option.utils.vega_weights.np.load", wraps=np.load) as np_load:
                result, metadata = load_vega_weight_grid(manifest, anchor, np.asarray(q, dtype=np.float32), days, manifest_sha)
                result[1, 1] = 999
                again, _ = load_vega_weight_grid(manifest, anchor, q, days, manifest_sha)
                self.assertEqual(np_load.call_count, 1)
                self.assertEqual(again[1, 1], 3)
                self.assertEqual(metadata["cell_weight_fingerprint"], fingerprint)
                self.assertEqual(metadata["manifest_sha256"], manifest_sha)
            for kwargs in (
                {"expected_manifest_sha256": "wrong"},
                {"strike_grid": [0.9, 1.1]},
                {"current_snapshot_time_utc": "2026-10-07T13:06:00Z"},
            ):
                arguments = {"manifest_path": manifest, "current_snapshot_time_utc": anchor, "strike_grid": q, "maturity_grid_days": days}
                arguments.update(kwargs)
                with self.assertRaises(ValueError):
                    load_vega_weight_grid(**arguments)
            with archive.open("ab") as handle:
                handle.write(b"cache drift")
            with self.assertRaisesRegex(ValueError, "NPZ SHA"):
                load_vega_weight_grid(manifest, anchor, q, days, manifest_sha)


if __name__ == "__main__":
    unittest.main()
