"""Negative controls for the Chapter 3 corrected-inference archive.

Every semantic tamper test updates the affected file's recorded SHA-256.  A
failure therefore demonstrates independent archive verification rather than
merely detecting a stale outer checksum.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.rq123 import chapter3_shared_panel_bootstrap_v2 as analysis


def _synthetic_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (11, 22):
        for fold in ("f1", "f2"):
            for session in range(2 if fold == "f1" else 3):
                for pair in range(session + 1):
                    baseline = 1.0 + 0.04 * pair + 0.1 * session + 0.01 * seed
                    for condition, ratio in (("a", 0.98), ("b", 1.0), ("c", 1.02)):
                        rows.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "fold": fold,
                                "pair_id": f"{fold}-s{session}-p{pair}",
                                "session_id": f"{fold}-s{session}",
                                "value": baseline * ratio,
                            }
                        )
    return pd.DataFrame(rows)


class Chapter3BootstrapVerificationNegativeControls(unittest.TestCase):
    """Exercise fail-closed verification against independently copied archives."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)
        panel = _synthetic_panel()
        cls.config_path = cls.base / "config.yaml"
        cls.config_path.write_text("schema_version: 2\n", encoding="utf-8")
        cls.source_path = cls.base / "source.csv"
        panel.to_csv(cls.source_path, index=False)
        source_manifest = [
            {
                "path": str(cls.source_path),
                "sha256": analysis.sha256_file(cls.source_path),
                "source_id": "synthetic",
                "role": "data",
            }
        ]
        jobs = [
            {
                "job_id": "with_contrasts",
                "panel": panel,
                "estimand": "equal_cell",
                "metadata": {"scope": "negative_control", "fold": "overall"},
                "contrasts": [
                    {
                        "contrast_id": "a_vs_b",
                        "focal": "a",
                        "reference": "b",
                        "family_id": "synthetic_holm2",
                        "alternative": "one_sided",
                        "apply_holm": True,
                        "support_gate": {
                            "alpha": 0.1,
                            "minimum_nonworse_seeds": 1,
                            "minimum_nonworse_folds": 1,
                            "require_ci_below_zero": True,
                        },
                    },
                    {
                        "contrast_id": "c_vs_b",
                        "focal": "c",
                        "reference": "b",
                        "family_id": "synthetic_holm2",
                        "alternative": "two_sided",
                        "apply_holm": True,
                    },
                ],
            },
            {
                "job_id": "absolute_mae_only",
                "panel": panel,
                "estimand": "pooled_pair",
                "metadata": {"scope": "negative_control", "fold": "overall"},
                "contrasts": [],
            },
        ]
        config = {"bootstrap": {"iterations": 10_000, "rng_seed": 73}}
        cls.archive = cls.base / "baseline_archive"
        with (
            patch.object(
                analysis,
                "load_sources",
                return_value=({}, source_manifest, config),
            ),
            patch.object(analysis, "build_jobs", return_value=jobs),
        ):
            analysis.run_analysis(cls.config_path, cls.archive)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _copy_archive(self, name: str) -> Path:
        root = self.base / name
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(self.archive, root)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    @staticmethod
    def _rehash(root: Path, relative_path: str) -> None:
        manifest_path = root / analysis.HASH_MANIFEST
        hashes = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        selected = hashes["relative_path"].eq(relative_path)
        if int(selected.sum()) != 1:
            raise AssertionError(f"Expected one hash row for {relative_path}")
        hashes.loc[selected, "sha256"] = analysis.sha256_file(root / relative_path)
        hashes.to_csv(manifest_path, index=False)

    def test_baseline_replays_and_zero_contrast_job_is_archived(self) -> None:
        result = analysis.verify_analysis(self.archive, replay=True)
        self.assertTrue(result["passed"])
        manifest = json.loads(
            (self.archive / "analysis_manifest.json").read_text(encoding="utf-8")
        )
        job = next(row for row in manifest["jobs"] if row["job_id"] == "absolute_mae_only")
        self.assertEqual(job["contrasts"], [])
        with np.load(self.archive / job["draws_path"], allow_pickle=False) as draws:
            self.assertEqual(draws["log_ratio_draws"].shape, (10_000, 0))

    def test_hash_manifest_rejects_missing_row_and_path_traversal(self) -> None:
        missing = self._copy_archive("missing_hash_row")
        manifest_path = missing / analysis.HASH_MANIFEST
        hashes = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        hashes.iloc[:-1].to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "file-set drift"):
            analysis.verify_analysis(missing, replay=False)

        traversal = self._copy_archive("traversing_hash_path")
        manifest_path = traversal / analysis.HASH_MANIFEST
        hashes = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        hashes.loc[0, "relative_path"] = "../outside"
        hashes.to_csv(manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "Unsafe output archive path"):
            analysis.verify_analysis(traversal, replay=False)

    def test_full_recipe_tampering_fails_after_manifest_rehash(self) -> None:
        mutations = {
            "focal": lambda recipe: recipe.update(focal="c"),
            "alternative": lambda recipe: recipe.update(alternative="two_sided"),
            "holm": lambda recipe: recipe.update(family_id="replacement_family"),
            "support_gate": lambda recipe: recipe["support_gate"].update(alpha=0.2),
            "legacy": lambda recipe: recipe.update(legacy={"point": 0.0}),
        }
        for label, mutate in mutations.items():
            with self.subTest(field=label):
                root = self._copy_archive(f"recipe_{label}")
                path = root / "analysis_manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                mutate(manifest["jobs"][0]["contrasts"][0])
                analysis._write_json(path, manifest)
                self._rehash(root, "analysis_manifest.json")
                with self.assertRaises(ValueError):
                    analysis.verify_analysis(root, replay=True)

    def test_derived_artifacts_fail_after_each_file_is_rehashed(self) -> None:
        def mutate_arms(path: Path) -> None:
            frame = pd.read_csv(path)
            frame.loc[0, "observed_mean_mae"] *= 1.1
            analysis._write_csv(path, frame)

        def mutate_contrasts(path: Path) -> None:
            frame = pd.read_csv(path)
            frame.loc[0, "significance_stars"] = "tampered"
            analysis._write_csv(path, frame)

        def mutate_comparison(path: Path) -> None:
            frame = pd.read_csv(path)
            frame.loc[0, "new_point"] += 0.25
            analysis._write_csv(path, frame)

        def mutate_values(path: Path) -> None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["jobs"]["with_contrasts"]["arms"]["a"][
                "observed_mean_mae"
            ] += 0.25
            analysis._write_json(path, payload)

        def mutate_qa(path: Path) -> None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["contrast_count"] += 1
            analysis._write_json(path, payload)

        def mutate_report(path: Path) -> None:
            path.write_text(
                path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
            )

        mutations = {
            "analysis/all_arm_summary.csv": mutate_arms,
            "analysis/all_contrasts.csv": mutate_contrasts,
            "analysis/new_old_comparison.csv": mutate_comparison,
            "chapter3_values.json": mutate_values,
            "qa.json": mutate_qa,
            "report.md": mutate_report,
        }
        for index, (relative_path, mutate) in enumerate(mutations.items()):
            with self.subTest(path=relative_path):
                root = self._copy_archive(f"derived_{index}")
                mutate(root / relative_path)
                self._rehash(root, relative_path)
                with self.assertRaises(ValueError):
                    analysis.verify_analysis(root, replay=True)

    def test_wrong_schedule_method_version_fails_after_rehash(self) -> None:
        root = self._copy_archive("wrong_schedule_version")
        manifest = json.loads(
            (root / "analysis_manifest.json").read_text(encoding="utf-8")
        )
        relative_path = manifest["jobs"][0]["schedule_path"]
        schedule_path = root / relative_path
        with np.load(schedule_path, allow_pickle=False) as archive:
            payload = {name: np.array(archive[name], copy=True) for name in archive.files}
        metadata = json.loads(str(payload["metadata"].reshape(()).item()))
        metadata["method_version"] = "wrong_method_version"
        payload["metadata"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        with schedule_path.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        self._rehash(root, relative_path)
        with self.assertRaisesRegex(ValueError, "unsupported bootstrap method_version"):
            analysis.verify_analysis(root, replay=True)


if __name__ == "__main__":
    unittest.main()
