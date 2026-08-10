from __future__ import annotations

import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.rq3.market_jump_detection import (  # noqa: E402
    _aggregate_anchor_records,
    _robust_scale,
    add_rolling_jump_skewness,
    aggregate_market_pairs,
    build_pair_slice_metrics,
    cluster_candidate_episodes,
    load_valid_pairs,
    score_metric_changes,
    select_approximate_atm,
    weighted_median,
)


ORIGIN = "2022-01-03T14:00:00Z"
TARGET = "2022-01-03T14:05:00Z"


def _surface_point(
    *,
    anchor: str,
    slice_id: str,
    strike: float,
    q: float,
    implied_vol: float,
    weight: float = 1.0,
    maturity_date: str = "2022-03-31",
    underlying: str = "TYH2",
    business_days: int = 60,
) -> dict[str, object]:
    return {
        "slice_id": slice_id,
        "anchor_time_utc": anchor,
        "business_days": business_days,
        "maturity_date": maturity_date,
        "underlying_contract_id": underlying,
        "strike": float(strike),
        "strike_over_forward": float(q),
        "log_moneyness": float(math.log(q)),
        "implied_vol": float(implied_vol),
        "weight_sum": float(weight),
    }


def _pair_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "pair_1",
                "origin_time_utc": ORIGIN,
                "target_time_utc": TARGET,
                "origin_time_london": "2022-01-03T14:00:00+00:00",
                "target_time_london": "2022-01-03T14:05:00+00:00",
                "session_id": "2022-01-03",
                "same_cme_continuous_session": True,
            }
        ]
    )


def _two_snapshot_points(
    current: list[tuple[float, float, float]],
    target: list[tuple[float, float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strike, q, implied_vol in current:
        rows.append(
            _surface_point(
                anchor=ORIGIN,
                slice_id="current_slice",
                strike=strike,
                q=q,
                implied_vol=implied_vol,
            )
        )
    for strike, q, implied_vol in target:
        rows.append(
            _surface_point(
                anchor=TARGET,
                slice_id="target_slice",
                strike=strike,
                q=q,
                implied_vol=implied_vol,
            )
        )
    return pd.DataFrame(rows)


class TestRQ3MarketJumpDetection(unittest.TestCase):
    def test_frozen_surface_params_override_audited_reconstruction_drift(self):
        records = [
            {
                "passes_precalib_filter": "true",
                "business_days": "10",
                "strike": "99",
                "weight": "2",
                "implied_vol": "0.11",
                "percent_strike": "0.991",
                "spot": "100",
                "price": "1",
                "maturity_date": "2022-01-17",
                "underlying_contract_id": "TYH2",
                "option_type": "PUT",
                "surface_input_role": "otm",
                "contract_id": "P99",
            },
            {
                "passes_precalib_filter": "true",
                "business_days": "10",
                "strike": "101",
                "weight": "1",
                "implied_vol": "0.09",
                "percent_strike": "1.011",
                "spot": "100",
                "price": "1",
                "maturity_date": "2022-01-17",
                "underlying_contract_id": "TYH2",
                "option_type": "CALL",
                "surface_input_role": "otm",
                "contract_id": "C101",
            },
        ]
        params = {
            "business_days": [10],
            "percent_strikes": [[0.990, 1.010]],
            "implied_vols": [[0.12, 0.08]],
        }

        points, audit = _aggregate_anchor_records(ORIGIN, records, params)

        self.assertEqual([point["strike_over_forward"] for point in points], [0.99, 1.01])
        self.assertEqual([point["implied_vol"] for point in points], [0.12, 0.08])
        self.assertEqual(
            [point["precalib_reconstructed_strike_over_forward"] for point in points],
            [0.991, 1.011],
        )
        self.assertEqual(audit["mismatch_slices"], 1)
        self.assertEqual(audit["q_mismatch_slices"], 1)
        self.assertEqual(audit["iv_mismatch_slices"], 1)

        precision_only_params = {
            **params,
            "percent_strikes": [[0.991, 1.011]],
            "implied_vols": [[0.11 + 5e-12, 0.09 - 5e-12]],
        }
        _, precision_audit = _aggregate_anchor_records(
            ORIGIN,
            records,
            precision_only_params,
        )
        self.assertEqual(precision_audit["mismatch_slices"], 0)
        self.assertEqual(precision_audit["iv_mismatch_slices"], 0)

    def test_pair_session_check_includes_the_current_backward_window(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "pairs.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    "CREATE TABLE valid_pair ("
                    "origin_time_utc TEXT, target_time_utc TEXT, "
                    "current_params_sha256 TEXT, target_params_sha256 TEXT)"
                )
                connection.executemany(
                    "INSERT INTO valid_pair VALUES (?, ?, ?, ?)",
                    [
                        (
                            "2022-02-27T23:01:00Z",
                            "2022-02-27T23:06:00Z",
                            "current_1",
                            "target_1",
                        ),
                        (
                            "2022-02-27T23:05:00Z",
                            "2022-02-27T23:10:00Z",
                            "current_2",
                            "target_2",
                        ),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            pairs = load_valid_pairs(
                database_path,
                ROOT_DIR / "data/reference/cme_treasury_globex_closures_2022_2023.csv",
            )

        self.assertEqual(pairs["same_cme_continuous_session"].tolist(), [False, True])
        self.assertEqual(
            pairs["analysis_window_start_utc"].tolist(),
            ["2022-02-27T22:56:00Z", "2022-02-27T23:00:00Z"],
        )

    def test_weighted_median_is_left_continuous_and_has_zero_weight_fallback(self):
        self.assertEqual(weighted_median([3.0, 1.0, 2.0], [1.0, 1.0, 2.0]), 2.0)
        self.assertEqual(weighted_median([2.0, 1.0], [1.0, 1.0]), 1.0)
        self.assertEqual(weighted_median([1.0, 3.0], [0.0, 0.0]), 2.0)
        self.assertTrue(math.isnan(weighted_median([], [])))

    def test_nearest_observed_atm_is_not_exact_and_ties_use_weight_then_strike(self):
        lower_q = math.exp(-0.01)
        upper_q = math.exp(0.01)
        points = pd.DataFrame(
            [
                _surface_point(
                    anchor=ORIGIN,
                    slice_id="weight_tie",
                    strike=99.0,
                    q=lower_q,
                    implied_vol=0.10,
                    weight=5.0,
                ),
                _surface_point(
                    anchor=ORIGIN,
                    slice_id="weight_tie",
                    strike=101.0,
                    q=upper_q,
                    implied_vol=0.11,
                    weight=10.0,
                ),
                _surface_point(
                    anchor=TARGET,
                    slice_id="strike_tie",
                    strike=98.0,
                    q=lower_q,
                    implied_vol=0.12,
                    weight=10.0,
                ),
                _surface_point(
                    anchor=TARGET,
                    slice_id="strike_tie",
                    strike=102.0,
                    q=upper_q,
                    implied_vol=0.13,
                    weight=10.0,
                ),
            ]
        )

        annotated, selected = select_approximate_atm(points)
        chosen = selected.set_index("slice_id")

        self.assertEqual(chosen.loc["weight_tie", "atm_strike"], 101.0)
        self.assertEqual(chosen.loc["strike_tie", "atm_strike"], 98.0)
        self.assertEqual(chosen["atm_tie_count"].tolist(), [2, 2])
        self.assertFalse(chosen["is_exact_atm"].any())
        self.assertEqual(int(annotated["is_nearest_atm_candidate"].sum()), 4)
        self.assertEqual(int(annotated["is_selected_atm"].sum()), 2)

    def test_pair_atm_uses_one_common_nominal_strike(self):
        points = _two_snapshot_points(
            current=[
                (99.0, 0.999, 0.10),
                (100.0, 1.020, 0.11),
            ],
            target=[
                (99.0, 0.980, 0.20),
                (100.0, 1.001, 0.25),
            ],
        )

        current_independent = points.loc[
            points["anchor_time_utc"].eq(ORIGIN), "log_moneyness"
        ].abs().idxmin()
        target_independent = points.loc[
            points["anchor_time_utc"].eq(TARGET), "log_moneyness"
        ].abs().idxmin()
        self.assertEqual(points.loc[current_independent, "strike"], 99.0)
        self.assertEqual(points.loc[target_independent, "strike"], 100.0)

        metrics, _ = build_pair_slice_metrics(points, _pair_audit())
        metric = metrics.iloc[0]
        self.assertEqual(metric["metric_status"], "ok")
        self.assertEqual(metric["pair_atm_strike"], 100.0)
        self.assertAlmostEqual(metric["current_atm_iv"], 0.11)
        self.assertAlmostEqual(metric["target_atm_iv"], 0.25)
        self.assertAlmostEqual(metric["delta_atm_iv"], 0.14)
        self.assertEqual(metric["pair_atm_quality"], "B")

        pair_audit = pd.concat(
            [
                _pair_audit(),
                _pair_audit().assign(
                    pair_id="cross_session_pair",
                    same_cme_continuous_session=False,
                ),
            ],
            ignore_index=True,
        )
        filtered_metrics, pair_rows = build_pair_slice_metrics(points, pair_audit)
        self.assertEqual(len(filtered_metrics), 1)
        self.assertEqual(len(pair_rows), 2)

    def test_two_sided_secant_and_ols_skew_have_expected_sign_and_units(self):
        q_values = [0.98, 1.0, 1.02]
        strikes = [98.0, 100.0, 102.0]
        current = [
            (strike, q, 0.10 - 0.50 * math.log(q))
            for strike, q in zip(strikes, q_values)
        ]
        target = [
            (strike, q, 0.11 - 0.70 * math.log(q))
            for strike, q in zip(strikes, q_values)
        ]

        metrics, _ = build_pair_slice_metrics(
            _two_snapshot_points(current, target),
            _pair_audit(),
        )
        metric = metrics.iloc[0]

        self.assertEqual(metric["skew_status"], "ok")
        self.assertEqual(metric["skew_quality"], "A")
        self.assertAlmostEqual(metric["current_atm_iv_skew_secant"], -0.50)
        self.assertAlmostEqual(metric["target_atm_iv_skew_secant"], -0.70)
        self.assertAlmostEqual(metric["delta_atm_iv_skew_secant"], -0.20)
        self.assertAlmostEqual(metric["current_atm_iv_skew_ols"], -0.50)
        self.assertAlmostEqual(metric["target_atm_iv_skew_ols"], -0.70)
        self.assertAlmostEqual(metric["current_skew_ols_r2"], 1.0)
        self.assertAlmostEqual(metric["target_skew_ols_r2"], 1.0)

    def test_skew_rejects_one_wing_and_too_small_log_moneyness_span(self):
        one_wing = [(98.0, 0.98, 0.12), (99.0, 0.99, 0.11), (100.0, 1.0, 0.10)]
        one_wing_metrics, _ = build_pair_slice_metrics(
            _two_snapshot_points(one_wing, one_wing),
            _pair_audit(),
        )
        self.assertEqual(
            one_wing_metrics.iloc[0]["skew_status"],
            "missing_two_sided_support",
        )
        self.assertEqual(one_wing_metrics.iloc[0]["skew_quality"], "unusable")

        narrow_q = [math.exp(-0.002), 1.0, math.exp(0.002)]
        narrow = [
            (strike, q, implied_vol)
            for strike, q, implied_vol in zip(
                [99.8, 100.0, 100.2], narrow_q, [0.11, 0.10, 0.09]
            )
        ]
        narrow_metrics, _ = build_pair_slice_metrics(
            _two_snapshot_points(narrow, narrow),
            _pair_audit(),
        )
        self.assertEqual(
            narrow_metrics.iloc[0]["skew_status"],
            "log_moneyness_span_below_minimum",
        )
        self.assertEqual(narrow_metrics.iloc[0]["skew_quality"], "unusable")
        self.assertNotIn("current_atm_iv_skew_secant", narrow_metrics.columns)

    def test_rolling_skew_minimum_counts_and_gap_reset(self):
        first_run = pd.date_range("2022-01-03T14:00:00Z", periods=35, freq="1min")
        second_run = pd.date_range("2022-01-03T14:41:00Z", periods=31, freq="1min")
        timestamps = first_run.append(second_run)
        changes = [((index % 7) - 3) ** 3 / 10000.0 for index in range(len(timestamps))]
        metrics = pd.DataFrame(
            {
                "origin_time_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "maturity_date": "2022-03-31",
                "underlying_contract_id": "TYH2",
                "session_id": "2022-01-03",
                "metric_status": "ok",
                "pair_atm_quality": "A",
                "delta_atm_iv": changes,
            }
        )

        result = add_rolling_jump_skewness(metrics)

        self.assertEqual(result.loc[13, "rolling_jump_skew_30m_n"], 14)
        self.assertTrue(math.isnan(result.loc[13, "rolling_jump_skew_30m"]))
        self.assertEqual(result.loc[14, "rolling_jump_skew_30m_n"], 15)
        self.assertTrue(math.isfinite(result.loc[14, "rolling_jump_skew_30m"]))
        self.assertEqual(result.loc[28, "rolling_jump_skew_60m_n"], 29)
        self.assertTrue(math.isnan(result.loc[28, "rolling_jump_skew_60m"]))
        self.assertEqual(result.loc[29, "rolling_jump_skew_60m_n"], 30)
        self.assertTrue(math.isfinite(result.loc[29, "rolling_jump_skew_60m"]))

        second_start = len(first_run)
        self.assertEqual(result.loc[second_start, "rolling_jump_skew_30m_n"], 1)
        self.assertEqual(result.loc[second_start, "rolling_jump_skew_60m_n"], 1)
        self.assertTrue(math.isnan(result.loc[second_start, "rolling_jump_skew_30m"]))
        self.assertEqual(result.loc[second_start + 14, "rolling_jump_skew_30m_n"], 15)
        self.assertTrue(
            math.isfinite(result.loc[second_start + 14, "rolling_jump_skew_30m"])
        )
        self.assertTrue(
            math.isnan(result.loc[second_start + 14, "delta_rolling_jump_skew_30m"])
        )
        self.assertTrue(
            math.isfinite(result.loc[second_start + 15, "delta_rolling_jump_skew_30m"])
        )

    def test_rolling_change_uses_previous_valid_value_after_temporary_nan(self):
        timestamps = list(pd.date_range("2022-01-03T14:00:00Z", periods=15, freq="1min"))
        timestamps.extend(
            pd.to_datetime(
                [
                    "2022-01-03T14:19:00Z",
                    "2022-01-03T14:24:00Z",
                    "2022-01-03T14:29:00Z",
                    "2022-01-03T14:34:00Z",
                    "2022-01-03T14:39:00Z",
                    "2022-01-03T14:44:00Z",
                ],
                utc=True,
            )
        )
        timestamps.extend(pd.date_range("2022-01-03T14:45:00Z", periods=16, freq="1min"))
        metrics = pd.DataFrame(
            {
                "origin_time_utc": [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in timestamps],
                "maturity_date": "2022-03-31",
                "underlying_contract_id": "TYH2",
                "session_id": "session",
                "metric_status": "ok",
                "pair_atm_quality": "A",
                "delta_atm_iv": [((index % 7) - 3) ** 3 / 10000.0 for index in range(len(timestamps))],
            }
        )

        result = add_rolling_jump_skewness(metrics)
        recovered = result[
            result["rolling_jump_skew_30m"].notna()
            & result["rolling_jump_skew_30m"].shift(1).isna()
        ]
        self.assertGreaterEqual(len(recovered), 2)
        recovered_after_first = recovered.iloc[1]
        self.assertTrue(
            math.isfinite(recovered_after_first["delta_rolling_jump_skew_30m"])
        )

    def test_robust_scale_falls_back_from_mad_to_iqr_then_std(self):
        scale, method = _robust_scale(np.asarray([0.0, 1.0, 2.0]))
        self.assertEqual(method, "mad")
        self.assertAlmostEqual(scale, 1.4826)

        scale, method = _robust_scale(np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
        self.assertEqual(method, "iqr")
        self.assertAlmostEqual(scale, 1.0 / 1.349)

        scale, method = _robust_scale(np.asarray([0.0] * 7 + [100.0]))
        self.assertEqual(method, "std")
        self.assertAlmostEqual(scale, np.std([0.0] * 7 + [100.0], ddof=1))

        scale, method = _robust_scale(np.asarray([4.0, 4.0, 4.0]))
        self.assertEqual(method, "none")
        self.assertTrue(math.isnan(scale))

    def test_tier_boundaries_and_rolling_metric_cap(self):
        changes = [0.0, 1.0, 2.0, 2.0]
        qualities = ["A", "C", "A", "B"]
        metrics = pd.DataFrame(
            [
                {
                    "slice_pair_id": f"slice_{index}",
                    "pair_id": f"pair_{index}",
                    "origin_time_utc": f"2022-01-03T14:0{index}:00Z",
                    "target_time_utc": f"2022-01-03T14:0{index + 5}:00Z",
                    "session_id": "2022-01-03",
                    "maturity_date": "2022-03-31",
                    "origin_business_days": 60,
                    "underlying_contract_id": "TYH2",
                    "delta_atm_iv": change,
                    "pair_atm_quality": quality,
                    "delta_rolling_jump_skew_60m": change,
                    "rolling_skew_quality": "B",
                }
                for index, (change, quality) in enumerate(zip(changes, qualities))
            ]
        )
        scale, _ = _robust_scale(np.asarray(changes))
        exact_top_z = (2.0 - np.median(changes)) / scale
        thresholds = {
            "high": {"percentile": 1.0, "robust_z": exact_top_z},
            "primary": {"percentile": 1.0, "robust_z": exact_top_z},
            "broad": {"percentile": 0.99, "robust_z": 99.0},
        }

        ranking = score_metric_changes(
            metrics,
            min_group_size=1,
            tier_thresholds=thresholds,
        )
        by_key = ranking.set_index(["metric_name", "slice_pair_id"])

        self.assertEqual(by_key.loc[("atm_iv_jump", "slice_2"), "anomaly_tier"], "high")
        self.assertEqual(
            by_key.loc[("atm_iv_jump", "slice_3"), "anomaly_tier"],
            "primary",
        )
        self.assertEqual(
            by_key.loc[("rolling_jump_skew_change", "slice_2"), "anomaly_tier"],
            "broad",
        )
        self.assertEqual(
            by_key.loc[("rolling_jump_skew_change", "slice_3"), "anomaly_tier"],
            "broad",
        )
        self.assertNotIn(
            "high",
            ranking.loc[
                ranking["metric_name"].eq("rolling_jump_skew_change"),
                "anomaly_tier",
            ].tolist(),
        )
        self.assertNotIn(
            "primary",
            ranking.loc[
                ranking["metric_name"].eq("rolling_jump_skew_change"),
                "anomaly_tier",
            ].tolist(),
        )

    def test_pair_aggregation_and_episode_clustering_do_not_count_duplicate_pairs(self):
        pair_audit = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "origin_time_utc": "2022-01-03T10:00:00Z",
                    "target_time_utc": "2022-01-03T10:05:00Z",
                    "session_id": "session",
                }
            ]
        )
        duplicate_metrics = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "slice_pair_id": "same_slice",
                    "anomaly_tier": "high",
                    "anomaly_tier_order": 3,
                    "abs_robust_z": 8.0,
                    "abs_empirical_percentile": 1.0,
                    "metric_name": "atm_iv_jump",
                    "metric_change": 0.02,
                    "metric_quality": "A",
                    "maturity_date": "2022-03-31",
                    "signed_direction": "up",
                },
                {
                    "pair_id": "p1",
                    "slice_pair_id": "same_slice",
                    "anomaly_tier": "high",
                    "anomaly_tier_order": 3,
                    "abs_robust_z": 8.0,
                    "abs_empirical_percentile": 1.0,
                    "metric_name": "atm_iv_jump",
                    "metric_change": 0.02,
                    "metric_quality": "A",
                    "maturity_date": "2022-03-31",
                    "signed_direction": "up",
                },
            ]
        )
        pair_rankings, candidates = aggregate_market_pairs(pair_audit, duplicate_metrics)
        self.assertEqual(len(pair_rankings), 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["flagged_slice_count"], 1)

        tier_consistent_metrics = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "slice_pair_id": "high_atm",
                    "anomaly_tier": "high",
                    "anomaly_tier_order": 3,
                    "abs_robust_z": 8.0,
                    "abs_empirical_percentile": 0.9995,
                    "metric_name": "atm_iv_jump",
                    "metric_change": 0.02,
                    "metric_quality": "A",
                    "maturity_date": "2022-03-31",
                    "signed_direction": "up",
                },
                {
                    "pair_id": "p1",
                    "slice_pair_id": "broad_rolling",
                    "anomaly_tier": "broad",
                    "anomaly_tier_order": 1,
                    "abs_robust_z": 50.0,
                    "abs_empirical_percentile": 1.0,
                    "metric_name": "rolling_jump_skew_change",
                    "metric_change": -2.0,
                    "metric_quality": "B",
                    "maturity_date": "2022-04-29",
                    "signed_direction": "down",
                },
            ]
        )
        tier_consistent_pairs, _ = aggregate_market_pairs(
            pair_audit,
            tier_consistent_metrics,
        )
        summary = tier_consistent_pairs.iloc[0]
        self.assertEqual(summary["peak_metric_name"], "atm_iv_jump")
        self.assertEqual(summary["peak_metric_quality"], "A")
        self.assertEqual(summary["max_abs_robust_z"], 8.0)
        self.assertEqual(summary["all_metric_max_abs_robust_z"], 50.0)
        self.assertEqual(summary["flagged_maturity_dates"], "2022-03-31;2022-04-29")
        self.assertEqual(summary["flagged_directions"], "down;up")

        candidate_rows = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "origin_time_utc": "2022-01-03T10:00:00Z",
                    "target_time_utc": "2022-01-03T10:05:00Z",
                    "session_id": "session",
                    "anomaly_tier": "high",
                    "anomaly_tier_order": 3,
                    "max_abs_robust_z": 8.0,
                    "max_abs_empirical_percentile": 1.0,
                    "peak_metric_name": "atm_iv_jump",
                    "flagged_metric_names": "atm_iv_jump",
                },
                {
                    "pair_id": "p1",  # Deliberate upstream/news-style duplicate.
                    "origin_time_utc": "2022-01-03T10:00:00Z",
                    "target_time_utc": "2022-01-03T10:05:00Z",
                    "session_id": "session",
                    "anomaly_tier": "high",
                    "anomaly_tier_order": 3,
                    "max_abs_robust_z": 8.0,
                    "max_abs_empirical_percentile": 1.0,
                    "peak_metric_name": "atm_iv_jump",
                    "flagged_metric_names": "atm_iv_jump",
                },
                {
                    "pair_id": "p2",
                    "origin_time_utc": "2022-01-03T10:10:00Z",
                    "target_time_utc": "2022-01-03T10:15:00Z",
                    "session_id": "session",
                    "anomaly_tier": "primary",
                    "anomaly_tier_order": 2,
                    "max_abs_robust_z": 6.0,
                    "max_abs_empirical_percentile": 0.999,
                    "peak_metric_name": "smile_skew_jump",
                    "flagged_metric_names": "smile_skew_jump",
                },
                {
                    "pair_id": "p3",
                    "origin_time_utc": "2022-01-03T10:21:00Z",
                    "target_time_utc": "2022-01-03T10:26:00Z",
                    "session_id": "session",
                    "anomaly_tier": "broad",
                    "anomaly_tier_order": 1,
                    "max_abs_robust_z": 4.5,
                    "max_abs_empirical_percentile": 0.995,
                    "peak_metric_name": "atm_iv_jump",
                    "flagged_metric_names": "atm_iv_jump",
                },
            ]
        )

        expected_episode_counts = {0: 3, 5: 2, 10: 1}
        for tolerance, expected_count in expected_episode_counts.items():
            with self.subTest(tolerance=tolerance):
                episodes, members = cluster_candidate_episodes(
                    candidate_rows,
                    tolerance_minutes=tolerance,
                )
                self.assertEqual(len(episodes), expected_count)
                self.assertEqual(len(members), 3)
                self.assertEqual(set(members["pair_id"]), {"p1", "p2", "p3"})
                self.assertEqual(int(episodes["market_pair_count"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
