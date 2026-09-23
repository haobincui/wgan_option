from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_generator_film_critic_factorial as factorial
from scripts.rq3 import (
    news_first_vol_generator_film_critic_factorial_analysis as analysis,
)
from scripts.rq3 import news_first_vol_generator_film_critic_factorial_report as report
from wgan_option.utils.text_ablation import REAL_TEXT, TEXT_SHUFFLE


def _contracts() -> list[dict[str, object]]:
    return [
        {
            "development_job_id": f"dev_{index:02d}",
            "training_contract_sha256": f"contract_{index:02d}",
        }
        for index in range(16)
    ]


def _text_rows(
    *, supported_architecture: tuple[str, str] | None = None
) -> pd.DataFrame:
    rows = []
    for generator, critic in analysis.ARCHITECTURES:
        rows.append(
            {
                "generator_conditioning_mode": generator,
                "critic_conditioning_mode": critic,
                "alignment_supported": (generator, critic) == supported_architecture,
            }
        )
    return pd.DataFrame(rows)


def _factorial_pair_metrics() -> pd.DataFrame:
    rows = []
    pairs = (
        ("session_a", "pair_a"),
        ("session_a", "pair_b"),
        ("session_b", "pair_c"),
        ("session_b", "pair_d"),
    )
    for tolerance in (5, 30):
        for generator, critic in analysis.ARCHITECTURES:
            for text in (TEXT_SHUFFLE, REAL_TEXT):
                signs = analysis._factor_signs(generator, critic, text)
                cell_value = (
                    0.10
                    + 0.001 * signs["G"]
                    + 0.002 * signs["D"]
                    - 0.003 * signs["T"]
                    + 0.004 * signs["G"] * signs["D"] * signs["T"]
                )
                job_id = f"{generator}|{critic}|{text}|{tolerance}"
                for session_id, pair_id in pairs:
                    rows.append(
                        {
                            "job_id": job_id,
                            "generator_conditioning_mode": generator,
                            "critic_conditioning_mode": critic,
                            "text_ablation_mode": text,
                            "tolerance_minutes": tolerance,
                            "seed": 42,
                            "session_id": session_id,
                            "pair_id": pair_id,
                            "model_mae": cell_value,
                            "persistence_mae": 0.11,
                        }
                    )
    return pd.DataFrame(rows)


class SessionBootstrapTests(unittest.TestCase):
    def test_is_deterministic_and_explicitly_single_seed(self) -> None:
        differences = [-0.03, -0.01, 0.02, -0.02, 0.01, -0.04]
        sessions = ["a", "a", "b", "b", "c", "c"]
        first = analysis.session_cluster_bootstrap(
            differences, sessions, iterations=500, seed=91
        )
        second = analysis.session_cluster_bootstrap(
            differences, sessions, iterations=500, seed=91
        )
        self.assertEqual(first, second)
        self.assertEqual(first["inference_unit"], "cme_session_cluster_single_seed")
        self.assertEqual(first["pair_count"], 6)
        self.assertEqual(first["session_count"], 3)

    def test_zero_contrast_is_an_exact_numerical_tie(self) -> None:
        result = analysis.session_cluster_bootstrap(
            [0.0, 0.0, 0.0, 0.0], ["a", "a", "b", "b"], iterations=100
        )
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["bootstrap_se"], 0.0)
        self.assertEqual(result["p_two_sided"], 1.0)


class SelectionTests(unittest.TestCase):
    def test_no_eligible_candidate_falls_back_to_anchor(self) -> None:
        rows = []
        for generator, critic in analysis.ARCHITECTURES[1:]:
            rows.append(
                {
                    "candidate_generator_conditioning_mode": generator,
                    "candidate_critic_conditioning_mode": critic,
                    "mean_difference": -0.1,
                    "bootstrap_se": 0.01,
                    "eligible": False,
                }
            )
        selection = analysis._selection_from_tables(
            pd.DataFrame(rows),
            _text_rows(),
            _contracts(),
            artifact_hashes={"x": "sha"},
            panel_lineage={"counts": analysis.EXPECTED_Q3_COUNTS},
        )
        self.assertEqual(
            selection["selection_label"], "no_supported_architecture_change"
        )
        self.assertEqual(
            selection["winner"]["generator_conditioning_mode"],
            analysis.ANCHOR_GENERATOR,
        )
        analysis._verify_self_hash(selection, "selection_sha256", "selection")

    def test_one_se_rule_uses_frozen_architecture_priority(self) -> None:
        candidates = [
            {
                "candidate_generator_conditioning_mode": analysis.FILM_GENERATOR,
                "candidate_critic_conditioning_mode": analysis.DISABLED_CRITIC,
                "mean_difference": -2.0,
                "bootstrap_se": 0.6,
                "eligible": True,
            },
            {
                "candidate_generator_conditioning_mode": analysis.ANCHOR_GENERATOR,
                "candidate_critic_conditioning_mode": analysis.DISABLED_CRITIC,
                "mean_difference": -1.5,
                "bootstrap_se": 0.1,
                "eligible": True,
            },
            {
                "candidate_generator_conditioning_mode": analysis.FILM_GENERATOR,
                "candidate_critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                "mean_difference": -1.45,
                "bootstrap_se": 0.1,
                "eligible": True,
            },
        ]
        selection = analysis._selection_from_tables(
            pd.DataFrame(candidates),
            _text_rows(
                supported_architecture=(
                    analysis.ANCHOR_GENERATOR,
                    analysis.DISABLED_CRITIC,
                )
            ),
            _contracts(),
            artifact_hashes={},
            panel_lineage={},
        )
        self.assertEqual(
            selection["winner"]["generator_conditioning_mode"],
            analysis.ANCHOR_GENERATOR,
        )
        self.assertEqual(
            selection["winner"]["critic_conditioning_mode"],
            analysis.DISABLED_CRITIC,
        )
        self.assertTrue(selection["text_alignment_supported"])

    def test_three_way_factorial_contrast_uses_all_eight_cells(self) -> None:
        paired = analysis._factorial_pair_contrast(
            _factorial_pair_metrics(), tolerance=5, factors=("G", "D", "T")
        )
        # A coefficient 0.004 on coded G*D*T implies the conventional
        # difference-of-differences-of-differences of 8 * 0.004.
        np.testing.assert_allclose(paired["difference"], 0.032, rtol=0, atol=1e-12)

    def test_q4_primary_family_contains_exactly_four_contrasts(self) -> None:
        selection = {
            "winner": {
                "generator_conditioning_mode": analysis.FILM_GENERATOR,
                "critic_conditioning_mode": analysis.DISABLED_CRITIC,
            }
        }
        contrasts = analysis._primary_q4_contrasts(
            _factorial_pair_metrics(), selection, tolerance=5
        )
        self.assertEqual(len(contrasts), 4)
        self.assertEqual(set(contrasts["holm_family"]), {"q4_primary_holm4"})
        self.assertEqual(
            set(contrasts["contrast"]),
            {
                "winner_minus_persistence",
                "winner_minus_anchor",
                "winner_real_minus_winner_shuffle",
                "generator_x_critic_x_text_interaction",
            },
        )


class RecipeAndGateTests(unittest.TestCase):
    def test_freeze_writes_exact_replay_schema_and_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            selection_path = root / "analysis" / analysis.Q3_SELECTION
            selection_path.write_text("{}\n", encoding="utf-8")
            jobs = []
            contracts = []
            for index in range(16):
                job = {
                    "job_id": f"dev_{index:02d}",
                    "generator_conditioning_mode": "g",
                    "critic_conditioning_mode": "d",
                    "text_ablation_mode": "real_text",
                    "tolerance_minutes": 5 if index < 8 else 30,
                    "seed": 42,
                }
                jobs.append(job)
                contract = {
                    "development_job_id": job["job_id"],
                    "generator_conditioning_mode": "g",
                    "critic_conditioning_mode": "d",
                    "text_ablation_mode": "real_text",
                    "tolerance_minutes": job["tolerance_minutes"],
                    "seed": 42,
                    "best_learned_epoch": 2,
                    "generator_lr_trace": [
                        {"epoch": 1, "lr": 5e-7},
                        {"epoch": 2, "lr": 2.5e-7},
                    ],
                    "discriminator_lr_trace": [
                        {"epoch": 1, "lr": 5e-7},
                        {"epoch": 2, "lr": 5e-7},
                    ],
                    "best_learned_generator_sha256": f"checkpoint_{index}",
                    "best_learned_discriminator_sha256": f"critic_{index}",
                    "training_config_sha256": f"config_{index}",
                    "code_manifest_sha256": "code",
                }
                contract["training_contract_sha256"] = factorial._payload_sha256(
                    contract
                )
                contracts.append(contract)
            selection = {
                "development_training_contracts": contracts,
                "development_matrix_sha256": "matrix",
                "selection_sha256": "selection-payload",
            }
            with (
                mock.patch.object(analysis, "_resolved", return_value={}),
                mock.patch.object(analysis, "_assert_q3_gate"),
                mock.patch.object(
                    analysis,
                    "_load_selection",
                    return_value=(selection, selection_path),
                ),
                mock.patch.object(analysis, "_stage_jobs", return_value=jobs),
                mock.patch.object(
                    analysis,
                    "_development_contract",
                    side_effect=lambda _root, job: contracts[
                        int(str(job["job_id"])[-2:])
                    ],
                ),
            ):
                manifest_path = analysis.freeze_refit_recipes(root, selection)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            analysis._verify_self_hash(manifest, "manifest_sha256", "manifest")
            self.assertEqual(manifest["recipe_count"], 16)
            self.assertEqual(len(manifest["recipes"]), 16)
            recipe = json.loads(
                Path(manifest["recipes"][0]["recipe_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(recipe),
                {
                    "schema_version",
                    "refit_mode",
                    "num_epochs",
                    "generator_lr_trace",
                    "discriminator_lr_trace",
                },
            )
            factorial._validate_refit_recipe(recipe)

    def test_q4_gate_failure_occurs_before_any_panel_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            panel = mock.Mock(side_effect=AssertionError("Q4 panel must not be read"))
            with (
                mock.patch.object(
                    analysis,
                    "_q4_gate",
                    side_effect=analysis.FilmCriticAnalysisError("gate closed"),
                ),
                mock.patch.object(analysis, "_q4_panel", panel),
            ):
                with self.assertRaisesRegex(
                    analysis.FilmCriticAnalysisError, "gate closed"
                ):
                    analysis.run_film_critic_q4_analysis(temporary)
            panel.assert_not_called()

    def test_q4_uses_one_common_5m_panel_for_all_sixteen_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            selection_path = root / "selection.json"
            selection_path.write_text("{}\n", encoding="utf-8")
            jobs = [
                {"job_id": f"job_{tolerance}_{index}", "tolerance_minutes": tolerance}
                for tolerance in (5, 30)
                for index in range(8)
            ]
            registry = {
                "selection_path": str(selection_path),
                "refit_recipe_manifest_sha256": "recipes",
                "q4_allowlist_sha256": "allowlist",
            }
            selection = {
                "selection_sha256": "selection",
                "winner": {
                    "generator_conditioning_mode": analysis.ANCHOR_GENERATOR,
                    "critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                },
            }
            common_panel = pd.DataFrame(
                {"pair_id": ["p"], "session_id": ["s"], "sample_id": ["n"]}
            )
            lineage = {
                "counts": analysis.EXPECTED_Q4_COMMON_COUNTS,
                "pair_universe_sha256": "pairs",
            }
            evaluated_panels: list[pd.DataFrame] = []

            def fake_evaluate(_root, selected_jobs, *, panel, **_kwargs):
                self.assertEqual(len(selected_jobs), 8)
                evaluated_panels.append(panel)
                return pd.DataFrame(
                    {
                        "tolerance_minutes": [selected_jobs[0]["tolerance_minutes"]],
                        "pair_id": ["p"],
                        "session_id": ["s"],
                    }
                )

            contrast = pd.DataFrame(
                {
                    "contrast": ["x"],
                    "p_two_sided": [1.0],
                    "holm_p": [1.0],
                }
            )
            with (
                mock.patch.object(
                    analysis, "_q4_gate", return_value=(registry, selection, jobs)
                ),
                mock.patch.object(
                    analysis,
                    "_resolved",
                    return_value={"split": {"q4_mc_samples": 64}},
                ),
                mock.patch.object(
                    analysis, "_q4_panel", return_value=(common_panel, lineage)
                ) as panel_loader,
                mock.patch.object(
                    analysis, "_evaluate_jobs", side_effect=fake_evaluate
                ),
                mock.patch.object(
                    analysis, "_primary_q4_contrasts", return_value=contrast
                ),
                mock.patch.object(
                    analysis,
                    "_historical_q4_overlap",
                    return_value={
                        "historically_exposed": True,
                        "pair_overlap_label": "143/143",
                    },
                ),
            ):
                result = analysis.run_film_critic_q4_analysis(root)
            panel_loader.assert_called_once_with(
                root.resolve(), {"split": {"q4_mc_samples": 64}}
            )
            self.assertEqual(len(evaluated_panels), 2)
            self.assertIs(evaluated_panels[0], common_panel)
            self.assertIs(evaluated_panels[1], common_panel)
            self.assertEqual(
                result["panel_lineage"]["30m_secondary"][
                    "broad_30m_q4_rows_passed_to_evaluator"
                ],
                0,
            )


class ReportTests(unittest.TestCase):
    def test_report_states_single_seed_and_historical_q4_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_dir = root / "analysis"
            analysis_dir.mkdir()
            architecture = pd.DataFrame(
                [
                    {
                        "candidate_generator_conditioning_mode": analysis.FILM_GENERATOR,
                        "candidate_critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                        "mean_difference": -1e-6,
                        "ci_95_lower": -2e-6,
                        "ci_95_upper": 1e-7,
                        "holm_p": 0.2,
                        "eligible": False,
                    }
                ]
            )
            text = pd.DataFrame(
                [
                    {
                        "generator_conditioning_mode": analysis.ANCHOR_GENERATOR,
                        "critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                        "mean_difference": -1e-7,
                        "ci_95_lower": -2e-7,
                        "ci_95_upper": 1e-7,
                        "holm_p": 0.4,
                        "alignment_supported": False,
                    }
                ]
            )
            architecture.to_csv(analysis_dir / analysis.Q3_ARCH_CONTRASTS, index=False)
            text.to_csv(analysis_dir / analysis.Q3_TEXT_CONTRASTS, index=False)
            for name in (
                analysis.Q3_PAIR_METRICS,
                analysis.Q3_CELL_SCORES,
                analysis.Q3_FACTORIAL_EFFECTS,
            ):
                (analysis_dir / name).write_bytes(b"fixture")
            q3_hashes = {
                name: factorial._sha256_file(analysis_dir / name)
                for name in (
                    analysis.Q3_PAIR_METRICS,
                    analysis.Q3_CELL_SCORES,
                    analysis.Q3_ARCH_CONTRASTS,
                    analysis.Q3_TEXT_CONTRASTS,
                    analysis.Q3_FACTORIAL_EFFECTS,
                )
            }
            selection = analysis._self_hashed_payload(
                {
                    "selection_label": "no_supported_architecture_change",
                    "winner": {
                        "generator_conditioning_mode": analysis.ANCHOR_GENERATOR,
                        "critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                        "text_ablation_mode": REAL_TEXT,
                    },
                    "anchor": {
                        "generator_conditioning_mode": analysis.ANCHOR_GENERATOR,
                        "critic_conditioning_mode": analysis.ANCHOR_CRITIC,
                    },
                    "text_alignment_supported": False,
                    "analysis_artifact_sha256": q3_hashes,
                },
                "selection_sha256",
            )
            factorial._write_json(analysis_dir / analysis.Q3_SELECTION, selection)
            primary = pd.DataFrame(
                [
                    {
                        "contrast": "winner_minus_persistence",
                        "mean_difference": 0.0,
                        "ci_95_lower": -1e-6,
                        "ci_95_upper": 1e-6,
                        "holm_p": 1.0,
                        "holm_significant": False,
                        "improvement_supported": False,
                    }
                ]
            )
            primary.to_csv(analysis_dir / analysis.Q4_PRIMARY_CONTRASTS, index=False)
            (analysis_dir / analysis.Q4_PAIR_METRICS).write_bytes(b"pairs")
            (analysis_dir / analysis.Q4_SECONDARY).write_bytes(b"secondary")
            q4_hashes = {
                name: factorial._sha256_file(analysis_dir / name)
                for name in (
                    analysis.Q4_PAIR_METRICS,
                    analysis.Q4_PRIMARY_CONTRASTS,
                    analysis.Q4_SECONDARY,
                )
            }
            q4 = analysis._self_hashed_payload(
                {
                    "selection_sha256": selection["selection_sha256"],
                    "confirmatory_claim_permitted": False,
                    "historical_q4_exposure": {
                        "historically_exposed": True,
                        "pair_overlap_label": "143/143",
                    },
                    "panel_lineage": {
                        "5m_primary": {"counts": {"pairs": 143, "sessions": 45}}
                    },
                    "artifact_sha256": q4_hashes,
                },
                "summary_sha256",
            )
            factorial._write_json(analysis_dir / analysis.Q4_SUMMARY, q4)
            path = report.render_film_critic_report(root)
            markdown = path.read_text(encoding="utf-8")
            html = path.with_suffix(".html").read_text(encoding="utf-8")
            for value in (
                "single-seed experiment (seed 42)",
                "143/143",
                "retrospective/frozen exploratory, not confirmatory",
                "does **not** estimate seed-to-seed",
            ):
                self.assertIn(value, markdown)
            self.assertIn("not confirmatory", html)


if __name__ == "__main__":
    unittest.main()
