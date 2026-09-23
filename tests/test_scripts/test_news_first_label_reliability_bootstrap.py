"""Focused tests for the offline raw-trade label reliability bootstrap."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from wgan_option.surface_generation.label_reliability import (  # noqa: E402
    BOOTSTRAP_METHODS,
    LabelReliabilityError,
    _apply_otm_preferred_itm_fallback,
    compute_reliability_scores,
    run_label_reliability_bootstrap,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params(vols: list[list[float]]) -> str:
    return json.dumps(
        {
            "business_days": [10, 20],
            "percent_strikes": [[0.9, 1.0], [0.9, 1.0]],
            "implied_vols": vols,
        },
        sort_keys=True,
    )


def _build_inputs(root: Path, *, replicated: bool) -> dict[str, object]:
    origin = "2023-06-01T00:00:00Z"
    target = "2023-06-01T00:05:00Z"
    official_vols: list[list[float]] = []
    raw_rows: list[dict[str, object]] = [
        {
            "#RIC": "TYM3",
            "Date-Time": "2023-05-31T23:59:30Z",
            "Price": 100.0,
            "Volume": 50.0,
        }
    ]
    priced_rows: list[dict[str, object]] = []
    for maturity_index, (business_days, maturity_date) in enumerate(
        ((10, "2023-06-15"), (20, "2023-06-29"))
    ):
        official_slice: list[float] = []
        for strike_index, strike in enumerate((90.0, 100.0)):
            low_iv = 0.20 + 0.02 * maturity_index + 0.01 * strike_index
            observations = [("A", 1.0, low_iv)]
            if replicated:
                observations.append(("B", 3.0, low_iv + 0.02))
                official_slice.append(low_iv + 0.02)
            else:
                official_slice.append(low_iv)
            for observation_index, (suffix, volume, implied_vol) in enumerate(observations):
                contract = f"O{business_days}_{int(strike)}_{suffix}"
                timestamp = f"2023-06-01T00:0{1 + observation_index}:00Z"
                price = 1.0 + 0.1 * maturity_index + 0.01 * strike_index + 0.001 * observation_index
                raw_row = {
                    "#RIC": contract,
                    "Date-Time": timestamp,
                    "Price": price,
                    "Volume": volume,
                }
                raw_rows.append(raw_row)
                priced_rows.append(
                    {
                        "target_datetime_utc": origin,
                        "window_side": "backward",
                        "window_start_utc": origin,
                        "window_end_utc": target,
                        "trade_datetime_utc": timestamp,
                        "calibration_datetime_utc": target,
                        "business_days": business_days,
                        "maturity_date": maturity_date,
                        "contract_id": contract,
                        "strike": strike,
                        "price": price,
                        "spot": 100.0,
                        "percent_strike": strike / 100.0,
                        "pricing_model": "black76",
                        "rate_curve_date": "2023-05-31",
                        "continuous_rate": 0.04,
                        "discount_factor": 0.99,
                        "rate_curve_sha256": "a" * 64,
                        "implied_vol": implied_vol,
                        "is_otm": True,
                        "passes_precalib_filter": True,
                        "filter_reason": "",
                        "weight": volume,
                        "underlying_contract_id": "TYM3",
                        "underlying_trade_datetime_utc": "2023-05-31T23:59:30Z",
                        "underlying_staleness_seconds": 30.0,
                        "underlying_match_mode": "last_prior_trade",
                    }
                )
        official_vols.append(official_slice)
    if replicated:
        # Exact raw occurrences are diagnostic multiplicity.  They must not be
        # copied into the canonical calibration input.  Four copies also make
        # the secondary occurrence-weighted sensitivity visibly non-zero.
        raw_rows.extend([dict(raw_rows[1]) for _ in range(3)])

    raw_path = root / "daily_raw.csv.gz"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False, compression="gzip")
    priced_path = root / "precalib_target_cache.csv.gz"
    pd.DataFrame(priced_rows).to_csv(priced_path, index=False, compression="gzip")
    current_params = _params([[0.19, 0.20], [0.21, 0.22]])
    target_params = _params(official_vols)
    pair_paths: dict[int, Path] = {}
    for tolerance in (5, 30):
        pair_path = root / f"pair_manifest_{tolerance:02d}m.csv.gz"
        pd.DataFrame(
            [
                {
                    "tolerance_minutes": tolerance,
                    "pair_id": f"pair_{tolerance:02d}",
                    "session_id": "2023-06-01",
                    "effective_origin_utc": origin,
                    "current_snapshot_time_utc": origin,
                    "target_snapshot_time_utc": target,
                    "current_surface_param_json": current_params,
                    "target_surface_param_json": target_params,
                    "joint_strict_support_cell_count": 4,
                    "strike_grid": "[0.9,1.0]",
                    "maturity_days_grid": "[10,20]",
                }
            ]
        ).to_csv(pair_path, index=False, compression="gzip")
        pair_paths[tolerance] = pair_path
    return {
        "label_reliability": {
            "pair_manifests": pair_paths,
            "pre_q3_end_utc": "2023-07-01T00:00:00Z",
            "forecast_horizon_minutes": 5,
            "priced_target_rows": priced_path,
            "raw_files": [raw_path],
            "grid": {
                "strike_grid": [0.9, 1.0],
                "maturity_days_grid": [10, 20],
            },
            "expected_grid_shape": [2, 2],
            "draws": 24,
            "seed": 42,
            "methods": list(BOOTSTRAP_METHODS),
            "minimum_valid_draw_fraction": 0.8,
            "canonical_replay_tolerance": 1.0e-12,
        }
    }


class LabelReliabilityBootstrapTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_preserves_lineage_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _build_inputs(root, replicated=True)
            first = root / "first"
            second = root / "second"
            first_manifest = run_label_reliability_bootstrap(config, first)
            second_manifest = run_label_reliability_bootstrap(config, second)

            self.assertEqual(first_manifest.name, "label_bootstrap_manifest.json")
            self.assertEqual(
                sorted(path.name for path in first.iterdir()),
                sorted(path.name for path in second.iterdir()),
            )
            for first_path in sorted(first.iterdir()):
                self.assertEqual(_sha256(first_path), _sha256(second / first_path.name))
            self.assertEqual(
                run_label_reliability_bootstrap(config, first, resume=True),
                first_manifest,
            )

            manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pricing_inversion_mode"], "cached_black76_iv_not_independently_rerun")
            self.assertIn("window_side=backward", manifest["target_cache_semantics"])
            self.assertFalse(manifest["formal_1000_draw_contract_satisfied"])
            lineage = pd.read_csv(first / "raw_trade_lineage.csv.gz")
            option_lineage = lineage.loc[lineage["is_option_occurrence"]]
            self.assertEqual(len(option_lineage), 11)
            self.assertEqual(int(option_lineage["duplicate_count"].max()), 4)
            self.assertTrue(lineage["source_file_sha256"].str.len().eq(64).all())
            self.assertTrue(lineage["source_row_ordinal"].ge(0).all())

            replay = pd.read_csv(first / "canonical_dedup_replay.csv.gz")
            self.assertEqual(int(replay["base_eligible"].sum()), 16)
            self.assertEqual(replay.loc[replay["base_eligible"], "row_fingerprint"].nunique(), 8)
            self.assertFalse(replay["underlying_raw_locator_missing"].any())
            pair_metrics = pd.read_csv(
                first / "label_reliability_pair_metrics.csv.gz"
            )
            self.assertEqual(set(pair_metrics["c_i"]), {4})
            self.assertEqual(set(pair_metrics["q_i"]), {4})
            self.assertEqual(set(pair_metrics["m_i"]), {4})
            self.assertTrue(pair_metrics["u_estimable"].all())
            self.assertTrue(pair_metrics["u_i"].gt(0).all())
            self.assertTrue(
                pair_metrics["official_target_replay_max_abs_error"].le(1.0e-12).all()
            )
            self.assertTrue(pair_metrics["official_target_support_matches_replay"].all())
            self.assertTrue(pair_metrics["formal_joint_support_matches_replay"].all())
            self.assertTrue(pair_metrics["occurrence_weighted_max_abs_error"].gt(0).all())

            draws = pd.read_csv(first / "bootstrap_draw_status.csv.gz")
            self.assertEqual(len(draws), 2 * 2 * 24)
            self.assertEqual(set(draws["method"]), set(BOOTSTRAP_METHODS))
            cells = pd.read_csv(first / "label_reliability_cell_metrics.csv.gz")
            for column in (
                "support_probability",
                "bootstrap_mad_iv",
                "canonical_current_iv",
                "canonical_delta_iv",
                "bootstrap_delta_mean",
                "pair_raw_occurrence_count",
                "pair_volume_ess",
                "pair_singleton_bucket_fraction",
            ):
                self.assertIn(column, cells)
            buckets = pd.read_csv(first / "label_reliability_bucket_metrics.csv.gz")
            self.assertEqual(len(buckets), 8)
            self.assertTrue(buckets["raw_occurrence_count"].ge(2).all())
            with np.load(first / "cell_arrays.npz", allow_pickle=False) as arrays:
                self.assertIn("bootstrap_mad", arrays.files)
                self.assertEqual(arrays["canonical_target"].shape, (4, 2, 2))

    def test_singleton_buckets_are_not_estimable_zero_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _build_inputs(root, replicated=False)
            manifest = run_label_reliability_bootstrap(config, root / "singletons")
            metrics = pd.read_csv(
                manifest.parent / "label_reliability_pair_metrics.csv.gz"
            )
            self.assertEqual(set(metrics["m_i"]), {0})
            self.assertEqual(set(metrics["u_i"]), {0.0})
            self.assertFalse(metrics["u_estimable"].any())
            self.assertEqual(set(metrics["singleton_bucket_fraction"]), {1.0})

    def test_otm_ownership_and_nearest_itm_cap_match_production_rule(self) -> None:
        rows = pd.DataFrame(
            [
                {"business_days": 10, "strike": 90.0, "is_otm": True, "percent_strike": 0.90},
                {"business_days": 10, "strike": 90.0, "is_otm": False, "percent_strike": 0.90},
                {"business_days": 10, "strike": 99.0, "is_otm": False, "percent_strike": 0.99},
                {"business_days": 10, "strike": 98.0, "is_otm": False, "percent_strike": 0.98},
            ]
        )
        rows["bootstrap_weight"] = 1.0
        selected = _apply_otm_preferred_itm_fallback(rows)
        self.assertEqual(set(selected["strike"]), {90.0, 99.0})
        self.assertEqual(
            dict(zip(selected["strike"], selected["surface_input_role_replay"])),
            {90.0: "otm", 99.0: "itm_fallback"},
        )

    def test_fold_local_score_formula_and_unestimable_floor(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "pair_id": "a",
                    "tolerance_minutes": 5,
                    "session_id": "s",
                    "effective_origin_utc": "2023-01-01T00:00:00Z",
                    "c_i": 16,
                    "q_i": 8,
                    "m_i": 4,
                    "u_i": 1.0,
                    "u_estimable": True,
                    "v_i": 1.0,
                    "h_i": 1.0,
                },
                {
                    "pair_id": "b",
                    "tolerance_minutes": 5,
                    "session_id": "s",
                    "effective_origin_utc": "2023-01-01T00:05:00Z",
                    "c_i": 16,
                    "q_i": 8,
                    "m_i": 0,
                    "u_i": 0.0,
                    "u_estimable": False,
                    "v_i": 1.0,
                    "h_i": 1.0,
                },
            ]
        )
        scored = compute_reliability_scores(metrics, ["a", "b"], 5).set_index("pair_id")
        self.assertEqual(float(scored.loc["a", "tau_f"]), 1.0)
        self.assertAlmostEqual(float(scored.loc["a", "uncertainty_factor_b_i"]), 0.5)
        self.assertAlmostEqual(float(scored.loc["a", "R_i"]), 0.5 ** (1.0 / 6.0))
        self.assertEqual(float(scored.loc["b", "R_i"]), 0.0)
        self.assertEqual(float(scored.loc["b", "r_i"]), 0.5)

    def test_forward_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _build_inputs(root, replicated=True)
            payload = config["label_reliability"]
            priced_path = Path(payload["priced_target_rows"])
            priced = pd.read_csv(priced_path)
            priced["window_side"] = "forward"
            forward_path = root / "forward_cache.csv.gz"
            priced.to_csv(forward_path, index=False, compression="gzip")
            payload["priced_target_rows"] = forward_path
            with self.assertRaisesRegex(LabelReliabilityError, "No priced target rows"):
                run_label_reliability_bootstrap(config, root / "invalid")


if __name__ == "__main__":
    unittest.main()
