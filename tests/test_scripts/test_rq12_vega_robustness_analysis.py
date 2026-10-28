"""Protect the paired Vega robustness estimand and replayable inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq123 import shared_panel_bootstrap_core as core
from scripts.rq3 import rq12_vega_robustness_analysis as analysis


ARMS = ("lp_matched", "lp_shuffle", "pure_cnn_no_text", "bow", "sentiment")
VERSIONS = ("unweighted_control", "vega_retrained")


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair_frame(
    *,
    identical: bool = False,
    seeds: tuple[int, ...] = (42, 202),
    fold_counts: tuple[int, ...] = (2, 7),
) -> pd.DataFrame:
    """Small crossed panel with unequal fold and session sizes.

    A market pair's lineage and persistence are identical across versions,
    architectures, text arms and seeds. Model error varies on all axes.
    """

    rows: list[dict[str, object]] = []
    for fold_index, pair_count in enumerate(fold_counts):
        fold = f"f{fold_index + 1}_2023q{fold_index + 1}"
        quarter_origin = pd.Timestamp("2023-01-10T13:30:00Z") + pd.DateOffset(
            months=3 * fold_index
        )
        for pair_index in range(pair_count):
            pair_id = f"{fold}_pair_{pair_index}"
            # Session zero has one pair; session one has all remaining pairs.
            session_index = int(pair_index > 0)
            origin = quarter_origin + pd.Timedelta(
                days=session_index, minutes=15 * pair_index
            )
            lineage = {
                "fold": fold,
                "tolerance_minutes": 5,
                "pair_id": pair_id,
                "sample_id": f"pair::{pair_id}",
                "session_id": f"{fold}_session_{session_index}",
                "current_snapshot_time_utc": origin.isoformat(),
                "target_snapshot_time_utc": (
                    origin + pd.Timedelta(minutes=5)
                ).isoformat(),
                "supported_cell_count": 16 + pair_index,
                "support_mask_fingerprint": _fingerprint(f"support::{pair_id}"),
                "cell_weight_fingerprint": _fingerprint(f"vega::{pair_id}"),
                "pair_truth_fingerprint": _fingerprint(f"truth::{pair_id}"),
                "cell_weight_manifest_sha256": "a" * 64,
                "persistence_plain_mae": 0.004 * (1 + fold_index + pair_index / 10),
                "persistence_vega_mae": 0.003 * (1 + fold_index + pair_index / 8),
            }
            for seed_index, seed in enumerate(seeds):
                for arm_index, arm in enumerate(ARMS):
                    old_error = (
                        0.002
                        * (1 + fold_index) ** 2
                        * (1 + seed_index * 0.05)
                        * (1 + arm_index * 0.04)
                        * (1 + pair_index * 0.03)
                    )
                    ratio = (
                        1.0
                        if identical
                        else 0.86
                        + fold_index * 0.13
                        + seed_index * 0.015
                        + pair_index * 0.007
                        + arm_index * 0.004
                    )
                    for version in VERSIONS:
                        error = old_error * (
                            ratio if version == "vega_retrained" else 1.0
                        )
                        rows.append(
                            {
                                **lineage,
                                "experiment_version": version,
                                "architecture": (
                                    "pure" if arm == "pure_cnn_no_text" else "film"
                                ),
                                "arm": arm,
                                "seed": seed,
                                "plain_mae": error,
                                "vega_mae": error * (0.8 + pair_index * 0.006),
                            }
                        )
    return pd.DataFrame(rows)


def _schedule(frame: pd.DataFrame, *, iterations: int = 128) -> core.Schedule:
    panel = analysis.prepare_input_panel(frame, require_frozen_counts=False)
    return core.make_schedule(panel, iterations=iterations, rng_seed=20260904)


def _holm(values: np.ndarray) -> np.ndarray:
    """Independent step-down calculation, retaining the original arm order."""

    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = maximum
    return adjusted


class VegaRobustnessAnalysisTests(unittest.TestCase):
    def test_identical_forecasts_have_zero_effect_and_unit_p_values(self) -> None:
        frame = _pair_frame(identical=True)
        summary, draws, _ = analysis.analyze_frame(
            frame, _schedule(frame), require_frozen_counts=False
        )

        self.assertEqual(set(summary.arm), set(ARMS))
        for column in (
            "mean_log_ratio", "bootstrap_se", "ci_lower", "ci_upper",
            "geometric_gain_percent", "gain_ci_lower_percent", "gain_ci_upper_percent",
        ):
            np.testing.assert_array_equal(summary[column].to_numpy(float), 0.0)
        np.testing.assert_array_equal(summary.p_two.to_numpy(float), 1.0)
        np.testing.assert_array_equal(summary.p_holm.to_numpy(float), 1.0)
        np.testing.assert_array_equal(draws["log_ratio_draws"], 0.0)

    def test_reversing_reference_reverses_effect_and_preserves_two_sided_p(self) -> None:
        frame = _pair_frame()
        schedule = _schedule(frame)
        forward, forward_draws, _ = analysis.analyze_frame(
            frame, schedule, require_frozen_counts=False
        )
        reversed_frame = frame.copy()
        reversed_frame["experiment_version"] = reversed_frame["experiment_version"].map(
            dict(zip(VERSIONS, reversed(VERSIONS), strict=True))
        )
        reverse, reverse_draws, _ = analysis.analyze_frame(
            reversed_frame, schedule, require_frozen_counts=False
        )
        forward = forward.set_index("arm").sort_index()
        reverse = reverse.set_index("arm").sort_index()

        np.testing.assert_allclose(
            forward.mean_log_ratio, -reverse.mean_log_ratio, rtol=0, atol=2e-14
        )
        np.testing.assert_array_equal(forward.p_two, reverse.p_two)
        np.testing.assert_array_equal(forward.p_holm, reverse.p_holm)
        np.testing.assert_allclose(
            forward.ci_lower, -reverse.ci_upper, rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(
            forward_draws["log_ratio_draws"], -reverse_draws["log_ratio_draws"],
            rtol=0, atol=2e-14,
        )

    def test_forty_equal_cells_use_mean_log_ratio_and_not_a_grand_ratio(self) -> None:
        frame = _pair_frame(seeds=tuple(range(10)), fold_counts=(2, 3, 5, 9))
        summary, _, _ = analysis.analyze_frame(
            frame, _schedule(frame, iterations=16), require_frozen_counts=False
        )
        cells = frame.groupby(["arm", "seed", "fold", "experiment_version"])[
            "vega_mae"
        ].mean().unstack("experiment_version")
        expected = np.log(cells.vega_retrained / cells.unweighted_control).groupby(
            "arm"
        ).mean()
        grand = cells.groupby("arm").mean()
        grand_log_ratio = np.log(grand.vega_retrained / grand.unweighted_control)
        observed = summary.set_index("arm")

        np.testing.assert_array_equal(cells.groupby("arm").size().to_numpy(), 40)
        np.testing.assert_allclose(
            observed.mean_log_ratio.reindex(expected.index), expected,
            rtol=0, atol=2e-14,
        )
        self.assertTrue((np.abs(expected - grand_log_ratio) > 0.01).all())
        for version, summary_column in (
            ("unweighted_control", "old_vega_mae"),
            ("vega_retrained", "new_vega_mae"),
        ):
            np.testing.assert_allclose(
                observed[summary_column].reindex(grand.index), grand[version],
                rtol=0, atol=2e-17,
            )
        # Fold pair counts must not silently replace equal-cell weights.
        pooled = frame.groupby(["arm", "experiment_version"])["vega_mae"].mean()
        self.assertGreater(
            abs(pooled.loc[(ARMS[0], "vega_retrained")] - grand.loc[ARMS[0], "vega_retrained"]),
            0.0001,
        )

    def test_geometric_gain_interval_reverses_log_interval_endpoints(self) -> None:
        frame = _pair_frame()
        summary, _, _ = analysis.analyze_frame(
            frame, _schedule(frame, iterations=256), require_frozen_counts=False
        )
        self.assertTrue((summary.ci_upper > summary.ci_lower).all())
        np.testing.assert_allclose(
            summary.geometric_gain_percent, 100 * (1 - np.exp(summary.mean_log_ratio)),
            rtol=0, atol=2e-13,
        )
        np.testing.assert_allclose(
            summary.gain_ci_lower_percent, 100 * (1 - np.exp(summary.ci_upper)),
            rtol=0, atol=2e-13,
        )
        np.testing.assert_allclose(
            summary.gain_ci_upper_percent, 100 * (1 - np.exp(summary.ci_lower)),
            rtol=0, atol=2e-13,
        )
        self.assertTrue(
            (summary.gain_ci_upper_percent > summary.gain_ci_lower_percent).all()
        )

    def test_adapter_rejects_missing_pairs_and_shared_evidence_drift(self) -> None:
        base = _pair_frame()
        invalid: dict[str, pd.DataFrame] = {
            "missing_pair": base.drop(index=0),
            "duplicate_pair": pd.concat([base, base.iloc[[0]]], ignore_index=True),
        }
        for column in (
            "pair_truth_fingerprint", "support_mask_fingerprint", "cell_weight_fingerprint",
            "cell_weight_manifest_sha256", "sample_id", "session_id",
            "current_snapshot_time_utc", "target_snapshot_time_utc",
        ):
            changed = base.copy()
            changed.loc[0, column] = (
                "2023-01-10T13:31:00+00:00"
                if "time_utc" in column else _fingerprint(f"different::{column}")
            )
            invalid[column] = changed
        for column in (
            "supported_cell_count", "persistence_plain_mae", "persistence_vega_mae",
        ):
            changed = base.copy()
            changed.loc[0, column] *= 2
            invalid[column] = changed
        missing_arm = base[base.arm.ne("sentiment")].copy()
        invalid["missing_arm"] = missing_arm
        invalid["missing_version"] = base[
            base.experiment_version.eq("unweighted_control")
        ].copy()

        for name, frame in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                analysis.prepare_input_panel(frame, require_frozen_counts=False)

    def test_small_adapter_is_separate_from_production_count_gate(self) -> None:
        frame = _pair_frame()
        panel = analysis.prepare_input_panel(frame, require_frozen_counts=False)
        self.assertEqual(len(panel.conditions), 10)
        self.assertEqual(len(panel.seeds), 2)
        self.assertEqual(len(panel.folds), 2)
        with self.assertRaises(ValueError):
            analysis.prepare_input_panel(frame)

    def test_shared_schedule_is_replayable_and_reusable_for_changed_errors(self) -> None:
        frame = _pair_frame()
        schedule = _schedule(frame)
        independent_schedule = _schedule(frame.sample(frac=1, random_state=23))
        for name in ("draw_id", "seed_weights", "fold_indices", "session_weights"):
            np.testing.assert_array_equal(
                getattr(schedule, name), getattr(independent_schedule, name)
            )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "shared_schedule.npz"
            core.save_schedule(schedule, archive)
            replay = core.load_schedule(archive)
            first, first_draws, _ = analysis.analyze_frame(
                frame, schedule, require_frozen_counts=False
            )
            second, second_draws, _ = analysis.analyze_frame(
                frame.sample(frac=1, random_state=17), replay, require_frozen_counts=False
            )
            pd.testing.assert_frame_equal(first, second)
            for name in first_draws:
                np.testing.assert_array_equal(first_draws[name], second_draws[name])

            changed = frame.copy()
            new_mask = changed.experiment_version.eq("vega_retrained")
            changed.loc[new_mask, ["plain_mae", "vega_mae"]] *= 0.7
            third, third_draws, _ = analysis.analyze_frame(
                changed, replay, require_frozen_counts=False
            )
            np.testing.assert_allclose(
                third_draws["log_ratio_draws"],
                first_draws["log_ratio_draws"] + np.log(0.7), rtol=0, atol=2e-14,
            )
            np.testing.assert_allclose(
                third.mean_log_ratio, first.mean_log_ratio + np.log(0.7),
                rtol=0, atol=2e-14,
            )

            # A different market panel cannot silently reuse these weights.
            changed_market = frame.copy()
            selected = changed_market.pair_id.eq(changed_market.iloc[0].pair_id)
            changed_market.loc[selected, "session_id"] += "_different"
            with self.assertRaises(core.ScheduleCompatibilityError):
                analysis.analyze_frame(
                    changed_market, replay, require_frozen_counts=False
                )

    def test_saved_draws_reconstruct_percentile_interval_and_holm_five(self) -> None:
        frame = _pair_frame()
        summary, draws, _ = analysis.analyze_frame(
            frame, _schedule(frame, iterations=256), require_frozen_counts=False
        )
        self.assertEqual(draws["mean_draws"].shape, (256, 10))
        self.assertEqual(draws["log_ratio_draws"].shape, (256, 5))
        arm_names = [str(value) for value in draws["arm_names"]]
        ordered = summary.set_index("arm").reindex(arm_names)
        values = np.asarray(draws["log_ratio_draws"], dtype=float)
        lower, upper = np.quantile(values, (0.025, 0.975), axis=0)
        p_two = np.minimum(
            1.0,
            2 * np.minimum(
                (1 + (values >= 0).sum(axis=0)) / 257,
                (1 + (values <= 0).sum(axis=0)) / 257,
            ),
        )
        np.testing.assert_allclose(ordered.ci_lower, lower, rtol=0, atol=2e-14)
        np.testing.assert_allclose(ordered.ci_upper, upper, rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            ordered.bootstrap_se, values.std(axis=0, ddof=1), rtol=0, atol=2e-14
        )
        np.testing.assert_array_equal(ordered.p_two.to_numpy(float), p_two)
        np.testing.assert_array_equal(ordered.p_holm.to_numpy(float), _holm(p_two))


if __name__ == "__main__":
    unittest.main()
