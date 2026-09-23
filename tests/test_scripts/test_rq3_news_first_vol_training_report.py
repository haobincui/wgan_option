from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.rq3 import news_first_vol_training_report as report


class TestNewsFirstVolTrainingReport(unittest.TestCase):
    def _experiment(self, root: Path) -> Path:
        analysis = root / "analysis"
        analysis.mkdir(parents=True)
        comparison_rows = []
        for model in ("wgan", "regression"):
            for tolerance in (5, 10, 15, 30):
                comparison_rows.append(
                    {
                        "model": model,
                        "tolerance_minutes": tolerance,
                        "panel": "common_test",
                        "stratum": "all",
                        "pair_count": 170,
                        "session_count": 49,
                        "mae": 0.006 + tolerance / 100_000,
                        "persistence_mae": 0.0065,
                        "mae_gap": -0.0005 + tolerance / 100_000,
                        "skill": 0.05,
                        "win_rate": 0.55,
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(
            analysis / "model_comparison.csv", index=False
        )
        pd.DataFrame(
            [
                {"panel": "core", "short_atm_cell_count": 6},
                {"panel": "broad", "short_atm_cell_count": 6},
            ]
        ).to_csv(analysis / "sample_metrics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(
            [
                {
                    "model": "wgan",
                    "focal_tolerance_minutes": 10,
                    "base_tolerance_minutes": 5,
                    "metric": "mae",
                    "mean_diff": -0.0001,
                    "ci_low": -0.0002,
                    "ci_high": -0.00001,
                    "p_value": 0.02,
                    "p_holm": 0.04,
                }
            ]
        ).to_csv(analysis / "bootstrap_comparisons.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": model,
                    "tolerance_minutes": tolerance,
                    "runtime_minutes": 2.0,
                    "gpu_hours": 2.0 / 60.0,
                }
                for model in ("wgan", "regression")
                for tolerance in (5, 10, 15, 30)
            ]
        ).to_csv(root / "resource_usage.csv", index=False)
        pd.DataFrame(
            [{"job_id": index, "status": "completed"} for index in range(8)]
        ).to_csv(root / "task_registry.csv", index=False)
        pd.DataFrame(
            [
                {
                    "tolerance_minutes": tolerance,
                    "train_rows": rows,
                    "train_pairs": pairs,
                }
                for tolerance, rows, pairs in (
                    (5, 1234, 952),
                    (10, 1506, 1115),
                    (15, 1748, 1230),
                    (30, 2263, 1452),
                )
            ]
        ).to_csv(root / "split_manifest.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "wgan",
                    "tolerance_minutes": 5,
                    "epoch": epoch,
                    "val_hybrid_score": 0.02 / epoch,
                }
                for epoch in (1, 2)
            ]
        ).to_csv(analysis / "training_curves.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "wgan",
                    "tolerance_minutes": 30,
                    "panel": "broad",
                    "stratum_type": "origin_shift",
                    "stratum_value": ">=5m",
                    "pair_count": 105,
                    "mae": 0.0078,
                    "gap": 0.0016,
                    "skill": -0.2,
                    "win": 0.4,
                }
            ]
        ).to_csv(analysis / "stratified_metrics.csv", index=False)
        return root

    def test_build_artifact_is_source_backed_and_report_shaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._experiment(Path(tmp))
            artifact = report.build_report_artifact(root)

        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(artifact["manifest"]["title"], report.REPORT_TITLE)
        self.assertEqual(len(artifact["snapshot"]["datasets"]["comparison"]), 8)
        self.assertTrue(artifact["manifest"]["cards"])
        self.assertTrue(artifact["manifest"]["charts"])
        self.assertTrue(artifact["manifest"]["tables"])
        source_ids = {item["id"] for item in artifact["sources"]}
        for item in (
            artifact["manifest"]["cards"]
            + artifact["manifest"]["charts"]
            + artifact["manifest"]["tables"]
        ):
            self.assertIn(item["sourceId"], source_ids)
        for table in artifact["manifest"]["tables"]:
            declared = {column["field"] for column in table["columns"]}
            self.assertIn(table["defaultSort"]["field"], declared)
        self.assertNotIn("/home/", json.dumps(artifact, ensure_ascii=False))
        rendered = json.dumps(artifact, ensure_ascii=False)
        self.assertIn("实际网格包含 6 格", rendered)
        self.assertNotIn("support_source", source_ids)
        self.assertIn("runtime", {chart["id"] for chart in artifact["manifest"]["charts"]})

    def test_support_audit_is_aggregated_and_narrow_grid_limit_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._experiment(Path(tmp) / "experiment")
            dataset_root = Path(tmp) / "dataset"
            root.joinpath("resolved_config.yaml").write_text(
                "news_first_vol_training:\n"
                "  datasets:\n"
                f"    root: {dataset_root}\n",
                encoding="utf-8",
            )
            (root / "analysis" / "analysis_config.json").write_text(
                json.dumps(
                    {
                        "support_mask_mode": "raw_joint",
                        "short_atm": {
                            "observed_cell_counts_by_panel": {
                                "core": [160],
                                "broad": [160],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            for tolerance in (5, 10, 15, 30):
                tolerance_dir = dataset_root / f"tolerance_{tolerance:02d}m"
                tolerance_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "tolerance_minutes": tolerance,
                            "pair_id": f"pair_{tolerance}_supported",
                            "surface_training_eligible": True,
                            "current_strict_support_cell_count": 160,
                            "current_strict_support_fraction": 160 / 256,
                            "target_strict_support_cell_count": 160,
                            "target_strict_support_fraction": 160 / 256,
                            "joint_strict_support_cell_count": 160,
                            "joint_strict_support_fraction": 160 / 256,
                            "joint_zero_support": False,
                            "grid_fingerprint": "narrow-grid-v1",
                            "support_mask_applied": False,
                        },
                        {
                            "tolerance_minutes": tolerance,
                            "pair_id": f"pair_{tolerance}_zero",
                            "surface_training_eligible": True,
                            "current_strict_support_cell_count": 0,
                            "current_strict_support_fraction": 0.0,
                            "target_strict_support_cell_count": 0,
                            "target_strict_support_fraction": 0.0,
                            "joint_strict_support_cell_count": 0,
                            "joint_strict_support_fraction": 0.0,
                            "joint_zero_support": True,
                            "grid_fingerprint": "narrow-grid-v1",
                            "support_mask_applied": False,
                        },
                    ]
                ).to_csv(
                    tolerance_dir / "surface_support_audit.csv.gz",
                    index=False,
                    compression="gzip",
                )

            artifact = report.build_report_artifact(root)
            persisted_support_source_path = next(
                source["path"]
                for source in artifact["sources"]
                if source["id"] == "support_source"
            )
            persisted_support_source_exists = (
                root / persisted_support_source_path
            ).is_file()

        support = artifact["snapshot"]["datasets"]["surface_support"]
        self.assertEqual(len(support), 4)
        self.assertTrue(all(row["joint_supported_pair_rate"] == 0.5 for row in support))
        self.assertTrue(all(row["support_mask_applied"] is False for row in support))
        self.assertIn("support_source", {source["id"] for source in artifact["sources"]})
        self.assertEqual(
            persisted_support_source_path, "analysis/surface_support_summary.csv"
        )
        self.assertTrue(persisted_support_source_exists)
        self.assertIn("support_table", {table["id"] for table in artifact["manifest"]["tables"]})
        self.assertIn(
            "strict_support_coverage",
            {chart["id"] for chart in artifact["manifest"]["charts"]},
        )
        rendered = json.dumps(artifact, ensure_ascii=False)
        self.assertIn("实际网格包含 160 格", rendered)
        self.assertIn("support_mask_applied=false", rendered)
        self.assertIn("工作簿生成阶段", rendered)
        self.assertIn("support_mask_mode=raw_joint", rendered)
        self.assertIn("输入完全无外推", rendered)
        self.assertIn("identity-preserving", rendered)
        self.assertNotIn("softplus(current + delta)", rendered)
        self.assertIn("Q3/Q4", rendered)
        self.assertIn("未按更密的 moneyness/TTM 步长重新标定", rendered)
        self.assertIn("没有把本轮指标与旧宽网格 checkpoint 作直接比较", rendered)

    def test_render_writes_artifact_html_and_delivery_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._experiment(Path(tmp))
            dummy_node = root / "node"
            dummy_node.touch()
            dummy_builder = root / "builder.mjs"
            dummy_builder.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "html": "technical_report.html",
                        "stages": {
                            "validation": "passed",
                            "package": "passed",
                            "verification": "structural_only",
                        },
                    }
                ),
                stderr="",
            )
            with (
                mock.patch.object(report, "DEFAULT_DELIVER_SCRIPT", dummy_builder),
                mock.patch.object(report, "_resolve_node", return_value=dummy_node),
                mock.patch.object(subprocess, "run", return_value=completed),
            ):
                output = report.render_portable_report(root)

            self.assertEqual(output, root / "report" / "technical_report.html")
            self.assertTrue((root / "report" / "artifact.json").is_file())
            receipt = json.loads(
                (root / "report" / "delivery_receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(receipt["ok"])

    def test_masked_ablation_report_explains_signed_text_estimand_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._experiment(Path(tmp))
            rows = []
            for model in ("wgan", "regression"):
                for tolerance in (5, 10, 15, 30):
                    for control in ("current_only", "text_shuffle"):
                        rows.append(
                            {
                                "model": model,
                                "tolerance_minutes": tolerance,
                                "panel": "core",
                                "metric": "model_mae",
                                "control_mode": control,
                                "pair_count": 130,
                                "session_count": 45,
                                "mean_diff": 0.0,
                                "ci_95_lower": 0.0,
                                "ci_95_upper": 0.0,
                                "p_two_sided": 1.0,
                                "p_holm": 1.0,
                            }
                        )
            pd.DataFrame(rows).to_csv(
                root / "analysis" / "text_ablation_comparisons.csv", index=False
            )
            (root / "analysis" / "analysis_config.json").write_text(
                json.dumps(
                    {
                        "support_mask_mode": "raw_joint",
                        "numerical_tie_policy": {"mae_gap_tolerance_iv": 1e-8},
                        "short_atm": {
                            "configured_grid_candidate_cell_count": 160,
                            "observed_cell_counts_by_panel": {
                                "core": [0, 40],
                                "broad": [0, 40],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            modes = ("real_text", "current_only", "text_shuffle")
            formal_rows = []
            for model in ("wgan", "regression"):
                for tolerance in (5, 10, 15, 30):
                    for mode in modes:
                        formal_rows.append(
                            {
                                "model": model,
                                "text_ablation_mode": mode,
                                "tolerance_minutes": tolerance,
                                "panel": "core",
                                "stratum_type": "overall",
                                "stratum_value": "all",
                                "pair_count": 130,
                                "session_count": 45,
                                "mae": 0.0017 - 3.99e-11,
                                "persistence_mae": 0.0017,
                                "gap": -3.99e-11,
                                "tie_rate": 1.0,
                                "win": 0.0,
                                "atm_gap": -0.000131,
                                "skew_gap": -0.01558,
                            }
                        )
            pd.DataFrame(formal_rows).to_csv(
                root / "analysis" / "model_comparison.csv", index=False
            )
            checkpoint_rows = []
            for model in ("wgan", "regression"):
                for tolerance in (5, 10, 15, 30):
                    for mode in modes:
                        checkpoint_rows.append(
                            {
                                "run_id": f"{model}_{mode}_{tolerance:02d}m",
                                "model": model,
                                "text_ablation_mode": mode,
                                "support_mask_mode": "raw_joint",
                                "seed": 42,
                                "tolerance_minutes": tolerance,
                                "monitor_metric": "val_hybrid_score",
                                "best_epoch": 0,
                                "best_metric": 0.001,
                                "metadata_path": "runs/example/metrics/best_checkpoint.json",
                                "status": "ok",
                            }
                        )
            pd.DataFrame(checkpoint_rows).to_csv(
                root / "analysis" / "checkpoint_summary.csv", index=False
            )
            pd.DataFrame(
                [
                    {"job_id": row["run_id"], "status": "completed"}
                    for row in checkpoint_rows
                ]
            ).to_csv(root / "task_registry.csv", index=False)
            root.joinpath("resolved_config.yaml").write_text(
                "news_first_vol_training:\n"
                "  models:\n"
                "    wgan:\n"
                "      training:\n"
                "        residual_output_mode: identity_softplus_residual\n"
                "    regression:\n"
                "      training:\n"
                "        residual_output_mode: identity_softplus_residual\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "model": "wgan",
                        "tolerance_minutes": 5,
                        "text_ablation_mode": mode,
                        "epoch": 0,
                        "val_hybrid_score": 0.01,
                    }
                    for mode in modes
                ]
            ).to_csv(root / "analysis" / "training_curves.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": "wgan",
                        "tolerance_minutes": 5,
                        "text_ablation_mode": mode,
                        "runtime_minutes": 2.0,
                        "gpu_hours": 2.0 / 60.0,
                    }
                    for mode in modes
                ]
            ).to_csv(root / "resource_usage.csv", index=False)
            artifact = report.build_report_artifact(root)

        rendered = json.dumps(artifact, ensure_ascii=False)
        self.assertIn("real_text MAE − control MAE", rendered)
        self.assertIn("-1e-08 IV", rendered)
        self.assertIn("identity-preserving", rendered)
        self.assertNotIn("零 delta 不是 identity", rendered)
        self.assertIn("24/24", rendered)
        self.assertIn("与 persistence 持平", rendered)
        self.assertIn("本轮无法识别新闻文本增量", rendered)
        self.assertIn("表示/插值口径差", rendered)
        self.assertIn("不是训练改善或文本增量", rendered)
        self.assertIn("不根据浮点噪声夸大win rate", rendered)
        self.assertNotIn("优于 persistence 基线", rendered)
        self.assertEqual(
            artifact["snapshot"]["datasets"]["headline"][0]["best_mae_gap"],
            0.0,
        )
        self.assertIn(
            "checkpoint_source", {source["id"] for source in artifact["sources"]}
        )
        self.assertEqual(
            len({row["series"] for row in artifact["snapshot"]["datasets"]["curves"]}),
            3,
        )
        self.assertEqual(
            len(
                {
                    row["task_label"]
                    for row in artifact["snapshot"]["datasets"]["resources"]
                }
            ),
            3,
        )
        self.assertIn(
            "text_ablation_effect",
            {chart["id"] for chart in artifact["manifest"]["charts"]},
        )
        self.assertIn(
            "text_ablation_table",
            {table["id"] for table in artifact["manifest"]["tables"]},
        )


if __name__ == "__main__":
    unittest.main()
