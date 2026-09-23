from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.rq3.news_first_vol_surfaces import (  # noqa: E402
    LEGACY_TOLERANCES_MINUTES,
    SUPPORTED_TOLERANCE_SETS,
    build_surface_support_audit,
    build_training_views,
    run_news_first_vol_surfaces,
    summarize_surface_support,
    validate_dataset_frames,
    _surface_grid_fingerprint,
    _validate_tolerance_minutes,
)
from scripts.rq3.news_first_vol_alignment import (  # noqa: E402
    DEFAULT_TOLERANCES_MINUTES,
)
from wgan_option.merge.merge_vol_core import (  # noqa: E402
    _grid_definition,
    _reconstruct_surface_flat,
)
from wgan_option.merge_support import write_workbook  # noqa: E402
from wgan_option.surface_grid import build_surface_grids  # noqa: E402
from wgan_option.utils.merged_xlsx_parsing import (  # noqa: E402
    _parse_serialized_vector,
    _parse_surface_shape,
    _read_sheet,
    _resolve_text_embedding,
)


TOLERANCES = DEFAULT_TOLERANCES_MINUTES
LEGACY_CONFIG_TOLERANCES = LEGACY_TOLERANCES_MINUTES


def _serial(value: float, count: int) -> str:
    return json.dumps([value] * count)


def _raw_params(*, lower: float = 0.97, upper: float = 1.03) -> str:
    return json.dumps(
        {
            "business_days": [7, 38],
            "percent_strikes": [
                [lower, 1.0, upper],
                [lower, 1.0, upper],
            ],
            "implied_vols": [
                [0.21, 0.20, 0.22],
                [0.22, 0.21, 0.23],
            ],
        }
    )


def _pair_audit_fixture() -> pd.DataFrame:
    """Small article-grain fixture covering every training exclusion path."""

    rows: list[dict[str, object]] = []
    definitions = [
        # Two articles deliberately share one market pair.
        (
            1,
            "pair_1",
            "2022-01-03T14:00:00Z",
            "2022-01-03T14:05:00Z",
            "open",
            "usable",
            1,
        ),
        (
            2,
            "pair_1",
            "2022-01-03T14:00:00Z",
            "2022-01-03T14:05:00Z",
            "open",
            "usable",
            1,
        ),
        # A market pair may have good ATM support but fail surface quality.
        (
            3,
            "pair_2",
            "2022-01-03T14:10:00Z",
            "2022-01-03T14:15:00Z",
            "open",
            "poor_fit",
            1,
        ),
        # Closed-news rows remain auditable but must never enter training.
        (
            4,
            "pair_3",
            "2022-01-03T23:00:00Z",
            "2022-01-03T23:05:00Z",
            "closed",
            "usable",
            0,
        ),
        # A usable surface without A/B ATM support remains in the broad GAN view.
        (
            5,
            "pair_4",
            "2022-01-03T14:20:00Z",
            "2022-01-03T14:25:00Z",
            "open",
            "usable",
            1,
        ),
        # Unmatched news must still be represented in the all-news audit.
        (6, "", "", "", "open", "no_pair", 0),
    ]
    for (
        news_row_id,
        pair_id,
        origin,
        target,
        market_state,
        quality,
        eligible,
    ) in definitions:
        publication = {
            1: "2022-01-03T14:00:00Z",
            2: "2022-01-03T13:58:00Z",
            3: "2022-01-03T14:08:00Z",
            4: "2022-01-03T22:00:00Z",
            5: "2022-01-03T14:17:00Z",
            6: "2022-01-03T12:00:00Z",
        }[news_row_id]
        shift = {
            1: 0,
            2: 2,
            3: 2,
            4: 60,
            5: 3,
            6: None,
        }[news_row_id]
        rows.append(
            {
                "sample_id": f"news_{news_row_id}",
                "news_row_id": news_row_id,
                "news_timestamp_utc": publication,
                "news_available_time_utc": publication,
                "publication_timestamp_utc": publication,
                "timestamp_parse_status": "ok",
                "publication_market_state": market_state,
                "has_match": int(bool(pair_id)),
                "training_eligible": eligible,
                "pair_id": pair_id,
                "effective_origin_utc": origin,
                "current_snapshot_time_utc": origin,
                "target_anchor_utc": target,
                "target_snapshot_time_utc": target,
                "origin_shift_minutes": shift,
                "origin_tolerance_minutes_used": 5 if pair_id else None,
                "first_included_tolerance_minutes": 5 if pair_id else None,
                "alignment_type": "exact"
                if shift == 0
                else ("intraday_shift" if pair_id else "unmatched"),
                "pair_quality_label": quality,
                "surface_model": "raw",
                "current_surface_param_json": (_raw_params() if pair_id else ""),
                "target_surface_param_json": (
                    _raw_params(lower=0.98, upper=1.02) if pair_id else ""
                ),
                "training_candidate_flag": int(quality == "usable"),
                "exclude_reason": "" if quality == "usable" else quality,
                "hd_embedding": _serial(0.1, 1024),
                "lp_embedding": _serial(0.2, 1024),
                "hd_dim": 1024,
                "lp_dim": 1024,
                "surface_shape": "[16, 16]",
                "strike_grid": _serial(1.0, 16),
                "maturity_days_grid": _serial(30.0, 16),
                "current_surface_flat": _serial(0.20, 256) if pair_id else "",
                "target_surface_flat": _serial(0.21, 256) if pair_id else "",
            }
        )
    return pd.DataFrame(rows)


def _pair_metrics_fixture() -> pd.DataFrame:
    rows = [
        (
            "pair_1",
            "2022-01-03T14:00:00Z",
            "2022-01-03T14:05:00Z",
            "2022-02-18",
            "TYH2",
            "A",
        ),
        (
            "pair_1",
            "2022-01-03T14:00:00Z",
            "2022-01-03T14:05:00Z",
            "2022-03-18",
            "TYH2",
            "B",
        ),
        (
            "pair_2",
            "2022-01-03T14:10:00Z",
            "2022-01-03T14:15:00Z",
            "2022-02-18",
            "TYH2",
            "A",
        ),
        (
            "pair_3",
            "2022-01-03T23:00:00Z",
            "2022-01-03T23:05:00Z",
            "2022-02-18",
            "TYH2",
            "B",
        ),
        (
            "pair_4",
            "2022-01-03T14:20:00Z",
            "2022-01-03T14:25:00Z",
            "2022-02-18",
            "TYH2",
            "C",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "slice_pair_id": f"{pair_id}_{maturity}",
                "pair_id": pair_id,
                "origin_time_utc": origin,
                "target_time_utc": target,
                "maturity_date": maturity,
                "underlying_contract_id": underlying,
                "metric_status": "ok",
                "pair_atm_quality": quality,
                "current_atm_iv": 0.20,
                "target_atm_iv": 0.21,
                "delta_atm_iv": 0.01,
            }
            for pair_id, origin, target, maturity, underlying, quality in rows
        ]
    )


def _side_detail_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"news_{news_row_id}",
                "news_row_id": news_row_id,
                "side": side,
                "source_direction": direction,
                "surface_flat": _serial(value, 256),
                "side_quality_label": "usable",
            }
            for news_row_id in range(1, 7)
            for side, direction, value in (
                ("current_back", "backward", 0.20),
                ("target_forward", "forward", 0.21),
            )
        ]
    )


class TestRQ3NewsFirstVolSurfaces(unittest.TestCase):
    def _views(self) -> dict[str, pd.DataFrame]:
        return build_training_views(
            _pair_audit_fixture(),
            _pair_metrics_fixture(),
            5,
            TOLERANCES,
        )

    def test_tolerance_contract_adds_20m_without_invalidating_legacy_configs(self):
        self.assertEqual(DEFAULT_TOLERANCES_MINUTES, (5, 10, 15, 20, 30))
        self.assertEqual(
            SUPPORTED_TOLERANCE_SETS,
            ((5, 10, 15, 30), (5, 10, 15, 20, 30)),
        )
        self.assertEqual(
            _validate_tolerance_minutes([5, 10, 15, 30]),
            LEGACY_CONFIG_TOLERANCES,
        )
        self.assertEqual(
            _validate_tolerance_minutes([5, 10, 15, 20, 30]),
            TOLERANCES,
        )
        with self.assertRaisesRegex(ValueError, "ordered tolerance_minutes"):
            _validate_tolerance_minutes([5, 10, 20, 15, 30])

    def test_builds_four_workbook_sheets_and_open_only_training_views(self):
        views = self._views()
        required = {
            "news_surface_pair_audit",
            "surface_side_detail",
            "gan_input_ready",
            "gan_input_atm_ab",
        }
        self.assertTrue(required.issubset(views))

        audit = views["news_surface_pair_audit"]
        self.assertEqual(audit["news_row_id"].tolist(), [1, 2, 3, 4, 5, 6])

        ready = views["gan_input_ready"]
        self.assertEqual(set(ready["news_row_id"]), {1, 2, 5})
        self.assertEqual(set(ready["publication_market_state"]), {"open"})
        self.assertEqual(set(ready["pair_quality_label"]), {"usable"})

        atm_ab = views["gan_input_atm_ab"]
        self.assertEqual(set(atm_ab["news_row_id"]), {1, 2})
        self.assertEqual(set(atm_ab["pair_id"]), {"pair_1"})
        self.assertTrue(atm_ab["has_atm_ab_maturity"].astype(bool).all())

        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "merged_vol.xlsx"
            write_workbook(
                workbook_path,
                {
                    name: views[name]
                    for name in (
                        "news_surface_pair_audit",
                        "surface_side_detail",
                        "gan_input_ready",
                        "gan_input_atm_ab",
                    )
                },
            )
            with pd.ExcelFile(workbook_path, engine="openpyxl") as excel_file:
                self.assertEqual(
                    excel_file.sheet_names,
                    [
                        "news_surface_pair_audit",
                        "surface_side_detail",
                        "gan_input_ready",
                        "gan_input_atm_ab",
                    ],
                )
            # Exercise the same dependency-free parsing layer used by the
            # merged-vol training loader.  The CI image intentionally has no
            # full torch install, so importing the DataLoader wrapper here
            # would turn a workbook compatibility test into an environment test.
            loaded = _read_sheet(str(workbook_path), "gan_input_ready")
            first = loaded.iloc[0]
            self.assertEqual(_parse_surface_shape(first["surface_shape"]), (16, 16))
            self.assertEqual(
                _parse_serialized_vector(first["current_surface_flat"]).size, 256
            )
            self.assertEqual(
                _parse_serialized_vector(first["target_surface_flat"]).size, 256
            )
            self.assertEqual(
                _resolve_text_embedding(
                    first["hd_embedding"], first["lp_embedding"], "hd"
                ).size,
                1024,
            )
            self.assertEqual(
                _resolve_text_embedding(
                    first["hd_embedding"], first["lp_embedding"], "lp"
                ).size,
                1024,
            )

    def test_pair_weights_sum_to_one_without_dropping_colliding_articles(self):
        views = self._views()
        ready = views["gan_input_ready"]
        shared = ready[ready["pair_id"] == "pair_1"].sort_values("news_row_id")

        self.assertEqual(shared["news_row_id"].tolist(), [1, 2])
        self.assertEqual(shared["pair_article_count"].astype(int).tolist(), [2, 2])
        self.assertEqual(shared["sample_weight"].astype(float).tolist(), [0.5, 0.5])
        weight_sums = ready.groupby("pair_id")["sample_weight"].sum()
        self.assertTrue((weight_sums.sub(1.0).abs() < 1e-12).all())
        atm_weight_sums = (
            views["gan_input_atm_ab"].groupby("pair_id")["sample_weight"].sum()
        )
        self.assertTrue((atm_weight_sums.sub(1.0).abs() < 1e-12).all())

    def test_atm_bridges_have_declared_article_and_pair_maturity_grains(self):
        views = self._views()
        article_bridge = views["news_atm_ab_maturity_bridge"]
        pair_outcomes = views["pair_atm_ab_outcomes"]

        article_key = [
            "news_row_id",
            "pair_id",
            "maturity_date",
            "underlying_contract_id",
        ]
        pair_key = ["pair_id", "maturity_date", "underlying_contract_id"]
        self.assertFalse(article_bridge.duplicated(article_key).any())
        self.assertFalse(pair_outcomes.duplicated(pair_key).any())
        self.assertEqual(len(article_bridge), 4)
        self.assertEqual(len(pair_outcomes), 2)
        self.assertEqual(set(article_bridge["pair_atm_quality"]), {"A", "B"})

    def test_validation_summary_and_manifest_fields_are_traceable(self):
        views = self._views()
        validation = validate_dataset_frames(
            views,
            tolerance_minutes=5,
            expected_news_rows=6,
        )

        self.assertEqual(validation["tolerance_minutes"], 5)
        self.assertEqual(validation["news_rows"], 6)
        self.assertEqual(validation["surface_usable_news"], 3)
        self.assertEqual(validation["surface_usable_pairs"], 2)
        self.assertEqual(validation["atm_ab_surface_news"], 2)
        self.assertEqual(validation["atm_ab_surface_pairs"], 1)
        self.assertEqual(validation["atm_ab_news_maturity_rows"], 4)
        self.assertEqual(validation["atm_ab_pair_maturities"], 2)
        self.assertTrue(validation["pair_weights_sum_to_one"])
        self.assertTrue(validation["article_maturity_key_unique"])
        self.assertTrue(validation["pair_maturity_key_unique"])
        self.assertEqual(validation["status"], "pass")

    def test_integer_maturity_grid_is_exact_and_legacy_default_is_unchanged(self):
        legacy_strikes, legacy_maturities = build_surface_grids(dtype=np.float64)
        np.testing.assert_allclose(legacy_strikes, np.linspace(0.70, 1.30, 16))
        np.testing.assert_allclose(legacy_maturities, np.linspace(7, 365, 16))

        strikes, maturities = build_surface_grids(
            strike_bins=16,
            maturity_bins=16,
            moneyness_min=0.97,
            moneyness_max=1.03,
            maturity_min_days=7,
            maturity_max_days=38,
            integer_maturity_days=True,
            dtype=np.float64,
        )
        self.assertEqual(
            maturities.astype(int).tolist(),
            [7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 32, 34, 36, 38],
        )
        self.assertEqual(len(np.unique(maturities)), 16)
        self.assertAlmostEqual(float(strikes[0]), 0.97)
        self.assertAlmostEqual(float(strikes[-1]), 1.03)
        self.assertAlmostEqual(float(strikes[7]), 0.998)
        self.assertAlmostEqual(float(strikes[8]), 1.002)
        with self.assertRaisesRegex(ValueError, "duplicate business-day nodes"):
            build_surface_grids(
                maturity_bins=16,
                maturity_min_days=7,
                maturity_max_days=10,
                integer_maturity_days=True,
            )

    def test_integer_grid_serialization_matches_surface_query_axis(self):
        strikes, maturities, shape = _grid_definition(
            strike_bins=16,
            maturity_bins=16,
            moneyness_min=0.97,
            moneyness_max=1.03,
            maturity_min_days=7,
            maturity_max_days=38,
            integer_maturity_days=True,
        )
        self.assertEqual(shape, "[16, 16]")
        self.assertEqual(
            maturities,
            [7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 32, 34, 36, 38],
        )

        surface = Mock()
        surface.implied_vol_surface.return_value = [[0.20] * 16 for _ in range(16)]
        flat = _reconstruct_surface_flat(surface, strikes, maturities)
        self.assertEqual(len(flat), 256)
        self.assertEqual(
            surface.implied_vol_surface.call_args.kwargs["business_days"],
            maturities,
        )

    def test_explicit_short_tenor_maturity_nodes_preserve_legacy_grid(self):
        expected = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 18, 21, 24, 28, 33, 38]
        strikes, maturities = build_surface_grids(
            strike_bins=16,
            maturity_bins=16,
            moneyness_min=0.97,
            moneyness_max=1.03,
            maturity_min_days=1,
            maturity_max_days=38,
            integer_maturity_days=True,
            maturity_days_nodes=expected,
            dtype=np.float64,
        )
        self.assertEqual(maturities.astype(int).tolist(), expected)

        config = yaml.safe_load(
            (
                ROOT_DIR
                / "configs/rq3/news_first_vol_surfaces_narrow_grid_short_ttm.yaml"
            ).read_text(encoding="utf-8")
        )["news_first_vol_surfaces"]
        self.assertEqual(config["grid"]["maturity_days_nodes"], expected)
        fingerprint = _surface_grid_fingerprint(strikes, maturities)
        self.assertEqual(
            fingerprint,
            "9ab1c2578cdd8c82d3243a8518e14471d7204d841139f90671575d74d706ecf6",
        )
        expected_counts = config["validation"]["expected_counts"]
        self.assertEqual(
            [
                expected_counts[str(value)]["joint_strict_support_pairs"]
                for value in LEGACY_CONFIG_TOLERANCES
            ],
            [999, 1161, 1260, 1445],
        )
        self.assertTrue(
            all(
                expected_counts[str(value)]["grid_fingerprint"] == fingerprint
                for value in LEGACY_CONFIG_TOLERANCES
            )
        )

        _, serialized_maturities, shape = _grid_definition(
            strike_bins=16,
            maturity_bins=16,
            moneyness_min=0.97,
            moneyness_max=1.03,
            maturity_min_days=1,
            maturity_max_days=38,
            integer_maturity_days=True,
            maturity_days_nodes=expected,
        )
        self.assertEqual(serialized_maturities, expected)
        self.assertEqual(shape, "[16, 16]")

        with self.assertRaisesRegex(ValueError, "exactly maturity_bins"):
            build_surface_grids(
                maturity_bins=16,
                integer_maturity_days=True,
                maturity_days_nodes=expected[:-1],
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_surface_grids(
                maturity_bins=16,
                integer_maturity_days=True,
                maturity_days_nodes=[*expected[:6], 7, *expected[7:]],
            )
        with self.assertRaisesRegex(ValueError, "whole business days"):
            build_surface_grids(
                maturity_bins=16,
                integer_maturity_days=True,
                maturity_days_nodes=[1.5, *expected[1:]],
            )

    def test_exact_ttm_config_freezes_nodes_fingerprint_and_support_counts(self):
        expected = [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38]
        config = yaml.safe_load(
            (
                ROOT_DIR
                / "configs/rq3/news_first_vol_surfaces_narrow_grid_exact_ttm.yaml"
            ).read_text(encoding="utf-8")
        )["news_first_vol_surfaces"]
        self.assertEqual(config["grid"]["maturity_days_nodes"], expected)

        strikes, maturities = build_surface_grids(
            strike_bins=16,
            maturity_bins=16,
            moneyness_min=0.97,
            moneyness_max=1.03,
            maturity_min_days=1,
            maturity_max_days=38,
            integer_maturity_days=True,
            maturity_days_nodes=expected,
            dtype=np.float64,
        )
        fingerprint = _surface_grid_fingerprint(strikes, maturities)
        self.assertEqual(
            fingerprint,
            "7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8",
        )
        expected_counts = config["validation"]["expected_counts"]
        self.assertEqual(
            [
                expected_counts[str(value)]["joint_strict_support_pairs"]
                for value in LEGACY_CONFIG_TOLERANCES
            ],
            [1026, 1196, 1294, 1475],
        )
        self.assertTrue(
            all(
                expected_counts[str(value)]["grid_fingerprint"] == fingerprint
                for value in LEGACY_CONFIG_TOLERANCES
            )
        )

    def test_support_audit_is_pair_grain_and_does_not_filter_training(self):
        views = self._views()
        support = build_surface_support_audit(
            views["news_surface_pair_audit"],
            strike_grid=[0.97, 0.99, 1.01, 1.03],
            maturity_days_grid=[7, 17, 28, 38],
            tolerance_minutes=5,
        )
        self.assertEqual(len(support), 4)
        self.assertFalse(support["pair_id"].duplicated().any())
        pair_1 = support.loc[support["pair_id"].eq("pair_1")].iloc[0]
        self.assertEqual(pair_1["current_strict_support_cell_count"], 16)
        self.assertEqual(pair_1["target_strict_support_cell_count"], 8)
        self.assertEqual(pair_1["joint_strict_support_cell_count"], 8)
        self.assertEqual(pair_1["joint_strict_support_fraction"], 0.5)
        self.assertFalse(bool(pair_1["joint_zero_support"]))
        self.assertFalse(bool(pair_1["support_mask_applied"]))
        self.assertRegex(str(pair_1["grid_fingerprint"]), r"^[0-9a-f]{64}$")

        views["surface_support_audit"] = support
        validation = validate_dataset_frames(
            views,
            tolerance_minutes=5,
            expected_news_rows=6,
        )
        self.assertEqual(validation["joint_strict_support_pairs"], 2)
        self.assertEqual(validation["surface_usable_pairs"], 2)
        self.assertEqual(validation["status"], "pass")
        summary = summarize_surface_support(support, tolerance_minutes=5)
        self.assertFalse(summary["support_mask_applied"])

    def test_run_writes_all_tolerance_workbooks_and_root_lineage_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_paths = {
                "market_index_sqlite": root / "market.sqlite",
                "pair_slice_metrics_csv": root / "pair_slice_metrics.csv",
                "session_calendar_csv": root / "sessions.csv",
                "rate_curve_csv": root / "rates.csv",
                "news_xlsx": root / "news.xlsx",
            }
            for key, path in source_paths.items():
                if key != "pair_slice_metrics_csv":
                    path.write_bytes(b"fixture")
            _pair_metrics_fixture().to_csv(
                source_paths["pair_slice_metrics_csv"], index=False
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "news_first_vol_surfaces": {
                            "inputs": {
                                **{
                                    key: str(path) for key, path in source_paths.items()
                                },
                                "news_source_timezone": "Europe/London",
                            },
                            "analysis": {
                                "tolerance_minutes": list(TOLERANCES),
                                "horizon_minutes": 5,
                                "current_window_minutes": 5,
                                "atm_qualities": ["A", "B"],
                                "include_closed_in_training": False,
                            },
                            "grid": {},
                            "quality": {},
                            "validation": {
                                "enforce_expected_counts": False,
                                "expected_news_rows": 6,
                                "expected_market_pairs": 4,
                                "expected_counts": {},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            pair_audit = _pair_audit_fixture()
            news_audit = pair_audit.assign(
                workbook_sha256=[f"workbook_{index}" for index in range(1, 7)],
                lp_text_sha256=[f"lp_text_{index}" for index in range(1, 7)],
            )
            alignments = {tolerance: pair_audit.copy() for tolerance in TOLERANCES}
            raw_frames = {
                "news_surface_pair_audit": pair_audit,
                "surface_side_detail": _side_detail_fixture(),
            }
            connection = Mock()
            with (
                patch(
                    "scripts.rq3.news_first_vol_surfaces.parse_factiva_news",
                    return_value=news_audit,
                ),
                patch(
                    "scripts.rq3.news_first_vol_surfaces.prepare_pair_universe",
                    return_value=pd.DataFrame(
                        {"pair_id": ["pair_1", "pair_2", "pair_3", "pair_4"]}
                    ),
                ),
                patch(
                    "scripts.rq3.news_first_vol_surfaces.build_news_first_alignments",
                    return_value=alignments,
                ) as build_alignments,
                patch(
                    "scripts.rq3.news_first_vol_surfaces.TreasuryGlobexSessionCalendar.from_csv",
                    return_value=Mock(),
                ),
                patch(
                    "scripts.rq3.news_first_vol_surfaces.connect_market_index",
                    return_value=connection,
                ),
                patch(
                    "scripts.rq3.news_first_vol_surfaces._materialize_dataset_inputs"
                ) as materialize,
                patch(
                    "scripts.rq3.news_first_vol_surfaces.build_vol_workbook_frames",
                    side_effect=lambda *args, **kwargs: {
                        name: frame.copy() for name, frame in raw_frames.items()
                    },
                ),
            ):
                output = run_news_first_vol_surfaces(
                    config_path, output_dir=root / "output"
                )

            self.assertEqual(output, (root / "output").resolve())
            alignment_kwargs = build_alignments.call_args.kwargs
            self.assertEqual(alignment_kwargs["tolerances"], TOLERANCES)
            self.assertEqual(alignment_kwargs["horizon_minutes"], 5)
            self.assertEqual(alignment_kwargs["current_window_minutes"], 5)
            materialize.assert_called_once()
            self.assertEqual(materialize.call_args.kwargs["window_minutes"], 5)
            self.assertIs(
                materialize.call_args.kwargs["alignment"],
                alignments[max(TOLERANCES)],
            )
            connection.close.assert_called_once()
            for tolerance in TOLERANCES:
                dataset_dir = output / f"tolerance_{tolerance:02d}m"
                workbook_path = dataset_dir / "merged_vol.xlsx"
                self.assertTrue(workbook_path.is_file())
                with pd.ExcelFile(workbook_path, engine="openpyxl") as excel_file:
                    self.assertEqual(
                        excel_file.sheet_names,
                        [
                            "news_surface_pair_audit",
                            "surface_side_detail",
                            "gan_input_ready",
                            "gan_input_atm_ab",
                        ],
                    )
                workbook_audit = _read_sheet(
                    str(workbook_path), "news_surface_pair_audit"
                )
                self.assertEqual(
                    workbook_audit["workbook_sha256"].tolist(),
                    news_audit["workbook_sha256"].tolist(),
                )
                self.assertEqual(
                    workbook_audit["lp_text_sha256"].tolist(),
                    news_audit["lp_text_sha256"].tolist(),
                )
                validation = json.loads(
                    (dataset_dir / "validation_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(validation["status"], "pass")
                self.assertEqual(validation["tolerance_minutes"], tolerance)
                self.assertTrue(
                    (dataset_dir / "surface_support_audit.csv.gz").is_file()
                )

            root_validation = json.loads(
                (output / "validation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(root_validation["status"], "pass")
            self.assertEqual(root_validation["tolerances_minutes"], list(TOLERANCES))
            manifest = (output / "run_manifest.env").read_text(encoding="utf-8")
            self.assertIn("git_commit=", manifest)
            self.assertIn("market_horizon_minutes=5", manifest)
            self.assertIn("tolerances_minutes=5,10,15,20,30", manifest)
            source_manifest = pd.read_csv(output / "source_manifest.csv")
            self.assertEqual(set(source_manifest["source"]), set(source_paths))
            self.assertTrue(
                source_manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
            )
            self.assertTrue((output / "dataset_output_sha256.txt").is_file())
            support_summary = pd.read_csv(output / "surface_support_summary.csv")
            self.assertEqual(
                support_summary["tolerance_minutes"].astype(int).tolist(),
                list(TOLERANCES),
            )
            self.assertFalse(support_summary["support_mask_applied"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
