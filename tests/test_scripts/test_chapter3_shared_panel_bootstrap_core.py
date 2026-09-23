"""Contract tests for the shared seed-by-market bootstrap core."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq123.shared_panel_bootstrap_core import (
    METHOD_VERSION,
    PanelValidationError,
    ScheduleCompatibilityError,
    load_schedule,
    make_schedule,
    prepare_panel,
    run_bootstrap,
    save_schedule,
)


MARKET = (
    ("f1", "p1", "f1_session_1", 1.0),
    ("f1", "p2", "f1_session_1", 4.0),
    ("f1", "p3", "f1_session_2", 2.0),
    ("f2", "p4", "f2_session_1", 8.0),
    ("f2", "p5", "f2_session_2", 3.0),
    ("f2", "p6", "f2_session_2", 6.0),
    ("f2", "p7", "f2_session_3", 9.0),
)


def _panel_frame(
    *,
    seeds: tuple[int, ...] = (11, 22),
    identical_seeds: bool = False,
    identical_conditions: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in ("focal", "control"):
        for seed_index, seed in enumerate(seeds):
            seed_factor = 1.0 if identical_seeds else 1.0 + 0.04 * seed_index
            for pair_index, (fold, pair_id, session_id, base) in enumerate(MARKET):
                if identical_conditions or condition == "control":
                    condition_factor = 1.0
                else:
                    condition_factor = (
                        0.72 + 0.025 * pair_index + (0.08 if fold == "f2" else 0.0)
                    )
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "fold": fold,
                        "pair_id": pair_id,
                        "session_id": session_id,
                        "value": base * seed_factor * condition_factor,
                        # Persistence is a property of the market pair, not a
                        # condition or model seed.
                        "persistence_mae": 1.25 * base,
                        "ignored_source_column": "kept-out-of-core",
                    }
                )
    return pd.DataFrame(rows)


def _manual_observed(frame: pd.DataFrame, estimand: str) -> np.ndarray:
    ordered = sorted(frame["condition"].unique())
    if estimand == "equal_cell":
        values = (
            frame.groupby(["condition", "seed", "fold"], sort=True)["value"]
            .mean()
            .groupby("condition", sort=True)
            .mean()
        )
    else:
        values = frame.groupby("condition", sort=True)["value"].mean()
    return values.reindex(ordered).to_numpy(float)


class PreparePanelTests(unittest.TestCase):
    def test_attributes_fingerprints_and_observed_means_are_canonical(self) -> None:
        source = _panel_frame()
        panel = prepare_panel(source.sample(frac=1.0, random_state=9))
        reordered = prepare_panel(
            source.sort_values(["condition", "seed"], ascending=[False, False])
        )

        self.assertEqual(panel.conditions, ("control", "focal"))
        self.assertEqual(panel.seeds, (11, 22))
        self.assertEqual(panel.folds, ("f1", "f2"))
        self.assertEqual(
            panel.sessions_by_fold,
            (
                ("f1_session_1", "f1_session_2"),
                ("f2_session_1", "f2_session_2", "f2_session_3"),
            ),
        )
        self.assertTrue(panel.has_persistence)
        self.assertEqual(panel.market_fingerprint, reordered.market_fingerprint)
        self.assertEqual(panel.panel_fingerprint, reordered.panel_fingerprint)
        np.testing.assert_allclose(
            panel.observed_means("equal_cell"),
            _manual_observed(source, "equal_cell"),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            panel.observed_means("pooled_pair"),
            _manual_observed(source, "pooled_pair"),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertNotIn("ignored_source_column", panel.frame.columns)

    def test_market_fingerprint_excludes_conditions_seeds_and_values(self) -> None:
        base = prepare_panel(_panel_frame(seeds=(11,), identical_seeds=True))
        clones = _panel_frame(seeds=(11, 22, 33), identical_seeds=True)
        clones.loc[clones["condition"].eq("focal"), "value"] *= 1.001
        changed = prepare_panel(clones)
        self.assertEqual(base.market_fingerprint, changed.market_fingerprint)
        self.assertNotEqual(base.panel_fingerprint, changed.panel_fingerprint)

    def test_rejects_incomplete_duplicate_empty_and_nonpositive_panels(self) -> None:
        base = _panel_frame()
        cases: dict[str, pd.DataFrame] = {}
        cases["incomplete"] = base.drop(index=base.index[0]).copy()
        cases["duplicate"] = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        cases["empty_lineage"] = base.copy()
        cases["empty_lineage"].loc[0, "session_id"] = "  "
        cases["nonpositive"] = base.copy()
        cases["nonpositive"].loc[0, "value"] = 0.0
        cases["nonfinite"] = base.copy()
        cases["nonfinite"].loc[0, "value"] = np.inf
        cases["persistence_drift"] = base.copy()
        cases["persistence_drift"].loc[0, "persistence_mae"] *= 2.0
        for name, candidate in cases.items():
            with self.subTest(name=name), self.assertRaises(PanelValidationError):
                prepare_panel(candidate)

    def test_rejects_missing_columns_and_noninteger_seed(self) -> None:
        with self.assertRaisesRegex(PanelValidationError, "missing canonical"):
            prepare_panel(_panel_frame().drop(columns="value"))
        candidate = _panel_frame()
        candidate["seed"] = candidate["seed"].astype(float)
        candidate.loc[0, "seed"] = 1.5
        with self.assertRaisesRegex(PanelValidationError, "seed"):
            prepare_panel(candidate)


class ScheduleTests(unittest.TestCase):
    def test_ten_thousand_draw_shape_integer_weights_and_metadata(self) -> None:
        panel = prepare_panel(_panel_frame())
        schedule = make_schedule(panel, iterations=10_000, rng_seed=20260904)

        self.assertEqual(schedule.draw_id.shape, (10_000,))
        self.assertEqual(schedule.seed_weights.shape, (10_000, 2))
        self.assertEqual(schedule.fold_indices.shape, (10_000, 2))
        self.assertEqual(schedule.session_weights.shape, (10_000, 2, 3))
        self.assertTrue(np.issubdtype(schedule.seed_weights.dtype, np.integer))
        self.assertTrue(np.issubdtype(schedule.session_weights.dtype, np.integer))
        np.testing.assert_array_equal(schedule.seed_weights.sum(axis=1), 2)
        self.assertEqual(schedule.metadata["method_version"], METHOD_VERSION)
        self.assertEqual(schedule.metadata["rngseed"], 20260904)
        self.assertEqual(schedule.metadata["market_fingerprint"], panel.market_fingerprint)
        self.assertEqual(schedule.metadata["seeds"], [11, 22])
        self.assertEqual(schedule.metadata["iterations"], 10_000)

        for draw in range(10_000):
            for occurrence in range(2):
                fold_index = int(schedule.fold_indices[draw, occurrence])
                session_count = len(panel.sessions_by_fold[fold_index])
                weights = schedule.session_weights[draw, occurrence]
                self.assertEqual(int(weights[:session_count].sum()), session_count)
                self.assertTrue(np.all(weights[session_count:] == 0))

    def test_market_rng_is_invariant_to_cloned_seed_count(self) -> None:
        one_seed = prepare_panel(
            _panel_frame(seeds=(11,), identical_seeds=True)
        )
        cloned_seeds = prepare_panel(
            _panel_frame(seeds=(11, 22, 33), identical_seeds=True)
        )
        one_schedule = make_schedule(one_seed, iterations=256, rng_seed=88)
        clone_schedule = make_schedule(cloned_seeds, iterations=256, rng_seed=88)

        self.assertEqual(one_seed.market_fingerprint, cloned_seeds.market_fingerprint)
        np.testing.assert_array_equal(
            one_schedule.fold_indices, clone_schedule.fold_indices
        )
        np.testing.assert_array_equal(
            one_schedule.session_weights, clone_schedule.session_weights
        )
        self.assertEqual(one_schedule.seed_weights.shape, (256, 1))
        self.assertEqual(clone_schedule.seed_weights.shape, (256, 3))

        for estimand in ("equal_cell", "pooled_pair"):
            one = run_bootstrap(one_seed, one_schedule, estimand)
            clone = run_bootstrap(cloned_seeds, clone_schedule, estimand)
            np.testing.assert_allclose(
                one.mean_draws, clone.mean_draws, rtol=0.0, atol=2e-15
            )
            np.testing.assert_allclose(
                one.contrast("focal", "control")["draws"],
                clone.contrast("focal", "control")["draws"],
                rtol=0.0,
                atol=2e-15,
            )

    def test_duplicate_fold_occurrences_have_independent_child_draws(self) -> None:
        panel = prepare_panel(_panel_frame())
        schedule = make_schedule(panel, iterations=512, rng_seed=19)
        duplicate_draws = np.flatnonzero(
            schedule.fold_indices[:, 0] == schedule.fold_indices[:, 1]
        )
        self.assertGreater(len(duplicate_draws), 0)
        different_child_weights = False
        for draw in duplicate_draws:
            fold_index = int(schedule.fold_indices[draw, 0])
            session_count = len(panel.sessions_by_fold[fold_index])
            if not np.array_equal(
                schedule.session_weights[draw, 0, :session_count],
                schedule.session_weights[draw, 1, :session_count],
            ):
                different_child_weights = True
                break
        self.assertTrue(different_child_weights)

    def test_save_load_is_one_npz_and_exactly_replayable(self) -> None:
        panel = prepare_panel(_panel_frame())
        schedule = make_schedule(panel, iterations=128, rng_seed=71)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            save_schedule(schedule, path)
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(
                    set(archive.files),
                    {
                        "draw_id",
                        "seed_weights",
                        "fold_indices",
                        "session_weights",
                        "metadata",
                    },
                )
            loaded = load_schedule(path)

        self.assertEqual(loaded.metadata, schedule.metadata)
        for name in ("draw_id", "seed_weights", "fold_indices", "session_weights"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(schedule, name))
        before = run_bootstrap(panel, schedule).mean_draws
        after = run_bootstrap(panel, loaded).mean_draws
        np.testing.assert_array_equal(before, after)

    def test_worker_chunks_partition_one_prebuilt_schedule(self) -> None:
        """Workers consume archived draw slices; they do not remake short schedules."""

        panel = prepare_panel(_panel_frame())
        schedule = make_schedule(panel, iterations=101, rng_seed=333)
        worker_draw_ids = tuple(np.array_split(schedule.draw_id, 4))
        np.testing.assert_array_equal(
            np.concatenate(worker_draw_ids), np.arange(101, dtype=np.int64)
        )
        for worker_ids in worker_draw_ids:
            np.testing.assert_array_equal(
                schedule.seed_weights[worker_ids],
                np.take(schedule.seed_weights, worker_ids, axis=0),
            )
            np.testing.assert_array_equal(
                schedule.session_weights[worker_ids],
                np.take(schedule.session_weights, worker_ids, axis=0),
            )


class RunBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _panel_frame()
        self.panel = prepare_panel(self.source)
        self.schedule = make_schedule(self.panel, iterations=400, rng_seed=123)

    def test_result_shapes_point_estimands_and_swap_identity(self) -> None:
        equal = run_bootstrap(self.panel, self.schedule, "equal_cell")
        pooled = run_bootstrap(self.panel, self.schedule, "pooled_pair")
        for result in (equal, pooled):
            self.assertEqual(result.conditions, ("control", "focal"))
            self.assertEqual(result.observed_means.shape, (2,))
            self.assertEqual(result.mean_draws.shape, (400, 2))

            forward = result.contrast("focal", "control")
            reverse = result.contrast("control", "focal")
            self.assertEqual(
                set(forward),
                {
                    "point",
                    "bootstrap_se",
                    "ci_lower",
                    "ci_upper",
                    "p_one",
                    "p_two",
                    "statistic",
                    "draws",
                    "seed_consistent",
                    "fold_consistent",
                },
            )
            self.assertAlmostEqual(forward["point"], -reverse["point"], places=15)
            np.testing.assert_allclose(
                forward["draws"], -reverse["draws"], rtol=0.0, atol=0.0
            )
            self.assertAlmostEqual(forward["ci_lower"], -reverse["ci_upper"], places=15)
            self.assertAlmostEqual(forward["ci_upper"], -reverse["ci_lower"], places=15)

        cells = self.source.groupby(
            ["condition", "seed", "fold"], sort=True
        )["value"].mean().unstack("condition")
        expected_equal = float(np.log(cells["focal"] / cells["control"]).mean())
        pooled_means = self.source.groupby("condition", sort=True)["value"].mean()
        expected_pooled = float(np.log(pooled_means["focal"] / pooled_means["control"]))
        self.assertAlmostEqual(
            equal.contrast("focal", "control")["point"], expected_equal, places=15
        )
        self.assertAlmostEqual(
            pooled.contrast("focal", "control")["point"], expected_pooled, places=15
        )
        self.assertNotAlmostEqual(expected_equal, expected_pooled, places=8)

    def test_first_draw_respects_pair_weighting_with_unequal_sessions(self) -> None:
        result = run_bootstrap(self.panel, self.schedule, "equal_cell")
        condition_index = result.conditions.index("focal")
        expected = 0.0
        for occurrence in range(len(self.panel.folds)):
            fold_index = int(self.schedule.fold_indices[0, occurrence])
            fold = self.panel.folds[fold_index]
            sessions = self.panel.sessions_by_fold[fold_index]
            session_weights = self.schedule.session_weights[
                0, occurrence, : len(sessions)
            ]
            for seed_index, seed in enumerate(self.panel.seeds):
                seed_weight = int(self.schedule.seed_weights[0, seed_index])
                if seed_weight == 0:
                    continue
                selected_values: list[float] = []
                focal = self.source[
                    self.source["condition"].eq("focal")
                    & self.source["seed"].eq(seed)
                    & self.source["fold"].eq(fold)
                ]
                for session, weight in zip(sessions, session_weights, strict=True):
                    block = focal.loc[focal["session_id"].eq(session), "value"].tolist()
                    for _ in range(int(weight)):
                        selected_values.extend(block)
                expected += seed_weight * float(np.mean(selected_values))
        expected /= float(len(self.panel.seeds) * len(self.panel.folds))
        self.assertAlmostEqual(result.mean_draws[0, condition_index], expected, places=15)

    def test_row_condition_reorder_and_value_changes_reuse_schedule(self) -> None:
        reordered = prepare_panel(
            self.source.sample(frac=1.0, random_state=91).reset_index(drop=True)
        )
        original = run_bootstrap(self.panel, self.schedule)
        repeated = run_bootstrap(reordered, self.schedule)
        np.testing.assert_array_equal(original.observed_means, repeated.observed_means)
        np.testing.assert_array_equal(original.mean_draws, repeated.mean_draws)

        changed_source = self.source.copy()
        changed_source.loc[changed_source["condition"].eq("focal"), "value"] *= 1.01
        changed = prepare_panel(changed_source)
        self.assertEqual(changed.market_fingerprint, self.panel.market_fingerprint)
        self.assertNotEqual(changed.panel_fingerprint, self.panel.panel_fingerprint)
        changed_result = run_bootstrap(changed, self.schedule)
        self.assertFalse(np.array_equal(original.mean_draws, changed_result.mean_draws))

    def test_wrong_market_panel_is_rejected(self) -> None:
        wrong_source = self.source.copy()
        wrong_source.loc[wrong_source["pair_id"].eq("p1"), "pair_id"] = "changed_p1"
        wrong = prepare_panel(wrong_source)
        with self.assertRaisesRegex(ScheduleCompatibilityError, "market_fingerprint"):
            run_bootstrap(wrong, self.schedule)

    def test_degenerate_contrast_has_zero_se_and_no_statistic(self) -> None:
        panel = prepare_panel(_panel_frame(identical_conditions=True))
        schedule = make_schedule(panel, iterations=64, rng_seed=7)
        result = run_bootstrap(panel, schedule)
        contrast = result.contrast("focal", "control")
        np.testing.assert_array_equal(contrast["draws"], np.zeros(64))
        self.assertEqual(contrast["point"], 0.0)
        self.assertEqual(contrast["bootstrap_se"], 0.0)
        self.assertIsNone(contrast["statistic"])
        self.assertEqual(contrast["p_one"], 1.0)
        self.assertEqual(contrast["p_two"], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
