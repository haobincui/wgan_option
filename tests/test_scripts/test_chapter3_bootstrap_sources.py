from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from scripts.rq123 import chapter3_bootstrap_sources as sources_module
from scripts.rq123.shared_panel_bootstrap_core import (
    make_schedule,
    prepare_panel,
    run_bootstrap,
)


class Chapter3SourceConfigContractTests(unittest.TestCase):
    def test_formal_recipe_rejects_any_draw_count_other_than_10000(self) -> None:
        payload = yaml.safe_load(
            sources_module.DEFAULT_CONFIG.read_text(encoding="utf-8")
        )
        payload["bootstrap"]["iterations"] = 9_999
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "invalid.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                sources_module.Chapter3BootstrapSourceError,
                "requires 10,000 draws",
            ):
                sources_module.load_sources(config_path)

    def test_formal_recipe_rejects_unsupported_bootstrap_variants(self) -> None:
        variants = {
            "float_iterations": ("iterations", 10_000.0),
            "boolean_rng_seed": ("rng_seed", True),
            "float_rng_seed": ("rng_seed", 20_260_904.0),
            "negative_rng_seed": ("rng_seed", -1),
            "confidence_level": ("confidence_level", 0.90),
            "point_estimate": ("point_estimate", "bootstrap_mean"),
            "seed_resampling": ("seed_resampling", "without_replacement"),
            "fold_resampling": ("fold_resampling", "without_replacement"),
            "session_resampling": (
                "session_resampling",
                "without_replacement",
            ),
            "market_weights": ("market_weights", "independent_by_seed"),
            "resampling_contract": ("resampling_contract", "legacy_v1"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, (key, value) in variants.items():
                with self.subTest(variant=name):
                    payload = yaml.safe_load(
                        sources_module.DEFAULT_CONFIG.read_text(encoding="utf-8")
                    )
                    payload["bootstrap"][key] = value
                    config_path = Path(temporary) / f"{name}.yaml"
                    config_path.write_text(
                        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                    )
                    with self.assertRaises(
                        sources_module.Chapter3BootstrapSourceError
                    ):
                        sources_module.load_sources(config_path)


class Chapter3FrozenSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = yaml.safe_load(
            sources_module.DEFAULT_CONFIG.read_text(encoding="utf-8")
        )
        frozen_paths = [
            sources_module.REPO_ROOT / spec["path"]
            for section in (
                "sources",
                "legacy_sources",
                "qa_artifacts",
                "passthrough_artifacts",
            )
            for spec in payload.get(section, {}).values()
        ]
        missing = [str(path) for path in frozen_paths if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Chapter 3 frozen source fixtures are unavailable: "
                + ", ".join(missing[:3])
            )
        cls.sources, cls.manifest, cls.config = sources_module.load_sources()
        cls.jobs = sources_module.build_jobs(cls.sources, cls.config)

    def test_real_sources_cover_every_canonical_job_panel(self) -> None:
        self.assertEqual(
            set(self.sources),
            {
                "direct",
                "rq3_full",
                "rq3_branch",
                "rq3_intervention",
                "rq3_validation",
                "architecture",
                "alignment",
            },
        )
        self.assertEqual(len(self.manifest), 30)
        self.assertEqual(len(self.jobs), 15)
        self.assertEqual(
            sum(len(job["contrasts"]) for job in self.jobs), 93
        )
        required = set(sources_module.CANONICAL_COLUMNS)
        for source_id, frame in self.sources.items():
            with self.subTest(source=source_id):
                self.assertFalse(frame.empty)
                self.assertTrue(required.issubset(frame.columns))
                self.assertTrue((frame["value"] > 0).all())
        for job in self.jobs:
            with self.subTest(job=job["job_id"]):
                self.assertEqual(
                    set(job["panel"].columns),
                    required | {"persistence_mae"},
                )
                panel = prepare_panel(job["panel"])
                self.assertIn("persistence", panel.conditions)

    def test_cross_source_value_perturbation_is_rejected(self) -> None:
        mutated = dict(self.sources)
        mutated["rq3_branch"] = self.sources["rq3_branch"].copy()
        index = mutated["rq3_branch"].index[
            mutated["rq3_branch"]["condition"].eq("film_lp_matched")
        ][0]
        mutated["rq3_branch"].loc[index, "value"] += 1e-8
        with self.assertRaisesRegex(
            sources_module.Chapter3BootstrapSourceError,
            "Cross-source value drift",
        ):
            sources_module._audit_cross_source_lineage(mutated)

    def test_cross_source_missing_market_row_is_rejected(self) -> None:
        mutated = dict(self.sources)
        branch = self.sources["rq3_branch"]
        index = branch.index[branch["condition"].eq("film_lp_shuffle")][0]
        mutated["rq3_branch"] = branch.drop(index=index)
        with self.assertRaisesRegex(
            sources_module.Chapter3BootstrapSourceError,
            "Cross-source market lineage drift",
        ):
            sources_module._audit_cross_source_lineage(mutated)

    def test_all_reproducible_legacy_points_keep_the_original_estimand(self) -> None:
        checked = 0
        maximum_absolute_difference = 0.0
        for job in self.jobs:
            panel = prepare_panel(job["panel"])
            result = run_bootstrap(
                panel,
                make_schedule(panel, iterations=2, rng_seed=20_260_904),
                estimand=job["estimand"],
            )
            for contrast in job["contrasts"]:
                legacy = contrast.get("legacy")
                if not legacy or legacy.get("point") is None:
                    continue
                point = result.contrast(
                    contrast["focal"], contrast["reference"]
                )["point"]
                difference = abs(point - float(legacy["point"]))
                maximum_absolute_difference = max(
                    maximum_absolute_difference, difference
                )
                checked += 1
        self.assertEqual(checked, 73)
        self.assertLessEqual(maximum_absolute_difference, 2e-12)

    def test_hypothesis_directions_and_complete_holm_families_are_frozen(self) -> None:
        contrasts = [
            contrast
            for job in self.jobs
            for contrast in job["contrasts"]
        ]
        self.assertEqual(
            {contrast["alternative"] for contrast in contrasts}, {"one_sided"}
        )
        counts = Counter(
            contrast["family_id"]
            for contrast in contrasts
            if contrast["apply_holm"]
        )
        self.assertEqual(len(counts), 25)
        for family_id, count in counts.items():
            declared = re.search(r"(?:^|_)holm(\d+)(?:_|$)", family_id)
            self.assertIsNotNone(declared, family_id)
            self.assertEqual(count, int(declared.group(1)), family_id)
        self.assertEqual(
            counts["rq1_lp_matched_vs_no_text_rolling_holm4"], 4
        )
        self.assertEqual(counts["rq2_representations_overall_holm2"], 2)
        self.assertEqual(counts["rq3_branch_holm2"], 2)
        self.assertEqual(counts["rq3_intervention_holm2"], 2)
        self.assertEqual(
            counts["rq3_validation_epoch0_to_epoch30_holm4"], 4
        )
        self.assertEqual(
            counts["rq3_validation_epoch30_to_best_holm4"], 4
        )


if __name__ == "__main__":
    unittest.main()
