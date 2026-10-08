"""Semantic contracts for the two frozen Chapter 3 forecast figures."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq3 import chapter3_forecast_figure_data as data


def surface_pair(pair_id: str = "pair", mask: np.ndarray | None = None) -> dict:
    joint = np.ones((16, 16), dtype=bool) if mask is None else mask.copy()
    realised, predicted = np.full((16, 16), 0.10), np.full((16, 16), 0.15)
    realised[11, 7:9] = [0.10, 0.20]
    predicted[11, 7:9] = [0.15, 0.25]
    return {"fold": data.FOLDS[0], "pair_id": pair_id, "realised": realised,
            "predicted": predicted, "joint_mask": joint, **data.support_statistics(joint)}


def sign_manifest(path: Path, payload: dict) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def frozen_fixture(root: Path) -> list[tuple[Path, Path, Path]]:
    paths = []
    params = {"business_days": [1, 38], "percent_strikes": [[0.97, 1.03], [0.97, 1.03]],
              "implied_vols": [[0.1, 0.1], [0.1, 0.1]]}
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    for fold in data.FOLDS:
        pred_path = root / "evaluation/predictions/tolerance_05m" / fold / "seed_42/lp_matched.csv.gz"
        panel_path = root / "evaluation/test_panels/tolerance_05m" / f"{fold}.csv.gz"
        pred_path.parent.mkdir(parents=True)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        prediction = {
            "pair_id": f"{fold}-pair", "session_id": f"{fold}-session",
            "predicted_surface_flat": json.dumps([0.15] * 256),
            "support_mask_flat": json.dumps([1] * 256), "supported_cell_count": 256,
            "model": "wgan", "arm": "lp_matched", "fold": fold, "seed": 42,
            "tolerance_minutes": 5, "prediction_status": "ok", "prediction_mc_samples": 64,
            "prediction_fallback": False, "support_mask_mode": "raw_joint",
            "text_ablation_mode": "real_text",
            "text_information_path": "current_surface_plus_real_lp_embedding",
            "checkpoint_sha256": data.sha256_file(checkpoint),
            "noise_bank_profile_sha256": "1" * 64,
        }
        panel = {
            "pair_id": prediction["pair_id"], "session_id": prediction["session_id"],
            "effective_origin_utc": "2023-07-12T12:00:00Z",
            "current_snapshot_time_utc": "2023-07-12T12:00:00Z",
            "target_snapshot_time_utc": "2023-07-12T12:05:00Z",
            "strike_grid": json.dumps(data.MONEYNESS_GRID.tolist()),
            "maturity_days_grid": json.dumps(data.MATURITY_GRID.tolist()),
            "current_surface_flat": json.dumps([0.1] * 256),
            "target_surface_flat": json.dumps([0.2] * 256),
            "current_surface_param_json": json.dumps(params),
            "target_surface_param_json": json.dumps(params),
        }
        pd.DataFrame([prediction]).to_csv(pred_path, index=False)
        pd.DataFrame([panel]).to_csv(panel_path, index=False)
        manifest_path = pred_path.with_name("lp_matched.csv.manifest.json")
        manifest = {
            "kind": "rq123_prediction_manifest_v1", "arm": "lp_matched", "fold": fold,
            "seed": 42, "tolerance_minutes": 5, "prediction_mc_samples": 64, "row_count": 1,
            "prediction_path": str(pred_path), "prediction_sha256": data.sha256_file(pred_path),
            "panel_path": str(panel_path), "panel_sha256": data.sha256_file(panel_path),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": data.sha256_file(checkpoint),
            "noise_bank_profile_sha256": prediction["noise_bank_profile_sha256"],
        }
        sign_manifest(manifest_path, manifest)
        paths.append((pred_path, panel_path, manifest_path))
    return paths


def refresh_source_hash(source: Path, manifest_path: Path, field: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = data.sha256_file(source)
    sign_manifest(manifest_path, manifest)


class ForecastFigureDataTests(unittest.TestCase):
    def test_atm_interpolates_variance_in_physical_iv_units(self) -> None:
        frame = pd.DataFrame([surface_pair()])
        result = data.with_atm_values(frame, data.MONEYNESS_GRID, data.MATURITY_GRID)
        self.assertEqual(len(result), len(frame))
        self.assertTrue(result.atm_valid.iloc[0])
        self.assertAlmostEqual(result.atm_realised_iv.iloc[0], np.sqrt(0.025))
        self.assertAlmostEqual(result.atm_predicted_iv.iloc[0], np.sqrt(0.0425))
        self.assertNotAlmostEqual(result.atm_realised_iv.iloc[0], 0.15)
        self.assertAlmostEqual(100 * result.atm_realised_iv.iloc[0], 15.811388300841896)
        np.testing.assert_array_equal(frame.realised.iloc[0], result.realised.iloc[0])

    def test_invalid_atm_endpoints_remain_missing_without_filling(self) -> None:
        left, right = np.ones((16, 16), bool), np.ones((16, 16), bool)
        left[11, 7], right[11, 8] = False, False
        frame = pd.DataFrame([surface_pair("left", left), surface_pair("right", right), surface_pair("valid")])
        result = data.with_atm_values(frame, data.MONEYNESS_GRID, data.MATURITY_GRID)
        self.assertEqual(result.pair_id.tolist(), frame.pair_id.tolist())
        self.assertEqual(result.atm_valid.tolist(), [False, False, True])
        self.assertTrue(result.loc[:1, ["atm_realised_iv", "atm_predicted_iv"]].isna().all().all())

    def test_support_hole_excludes_every_touching_quad(self) -> None:
        mask = np.ones((16, 16), bool)
        self.assertEqual(data.support_statistics(mask)["supported_quad_count"], 225)
        mask[7, 7] = False
        stats = data.support_statistics(mask)
        self.assertEqual(stats["supported_quad_count"], 221)
        self.assertEqual(stats["supported_cell_count"], 255)
        self.assertTrue(stats["surface_eligible"])
        scattered = np.zeros((16, 16), bool)
        scattered[::2, ::2] = True
        self.assertFalse(data.support_statistics(scattered)["surface_eligible"])

    def test_selection_is_invariant_to_input_order_and_ignores_outcomes(self) -> None:
        frame = pd.DataFrame([surface_pair(f"pair-{index:02}") for index in range(10)])
        frame.loc[0, "surface_eligible"] = False
        chosen = data.select_surface_pair(frame).pair_id
        permuted = frame.sample(frac=1.0, random_state=4)
        permuted["predicted"] = [np.full((16, 16), 999.0)] * len(permuted)
        self.assertEqual(data.select_surface_pair(permuted).pair_id, chosen)
        eligible = frame.loc[frame.surface_eligible].sort_values(["fold", "pair_id"])
        index = np.random.default_rng(20261008).choice(len(eligible))
        self.assertEqual(chosen, eligible.iloc[index].pair_id)

    def test_detail_links_break_at_missing_session_large_and_zero_gaps(self) -> None:
        frame = pd.DataFrame({
            "pair_id": [f"p{index}" for index in range(9)],
            "target_time_utc": pd.to_datetime([
                "2023-07-12T12:00Z", "2023-07-12T12:03Z", "2023-07-12T12:04Z",
                "2023-07-12T12:05Z", "2023-07-12T12:06Z", "2023-07-12T12:12Z",
                "2023-07-12T12:12Z", "2023-07-12T12:15Z", "2023-07-12T12:20Z"], utc=True),
            "session_id": ["a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "atm_valid": [True, True, False, True, True, True, True, True, True],
        })
        self.assertEqual(data.detail_link_indices(frame.sample(frac=1.0, random_state=1)),
                         [(0, 1), (6, 7), (7, 8)])
        frame["atm_valid"] = frame.atm_valid.astype("boolean")
        frame.loc[1, "atm_valid"] = pd.NA
        self.assertEqual(data.detail_link_indices(frame), [(6, 7), (7, 8)])

    def test_valid_frozen_inputs_preserve_rows_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen_fixture(root)
            pairs, moneyness, maturities, provenance = data.load_frozen_inputs(root)
            self.assertEqual(len(pairs), 4)
            self.assertEqual(provenance["surface_eligible_count"], 4)
            self.assertEqual(len(provenance["sources"]), 4)
            self.assertEqual(str(pairs.target_time_utc.dt.tz), "UTC")
            self.assertEqual(pairs.joint_mask.iloc[0].shape, (16, 16))
            result = data.with_atm_values(pairs, moneyness, maturities)
            self.assertTrue(result.atm_valid.all())
            self.assertTrue(np.allclose(result.atm_realised_iv, 0.2))

    def test_hash_drift_is_rejected_before_csv_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, panel, _ = frozen_fixture(root)[0]
            panel.write_bytes(b"changed frozen input")
            with self.assertRaisesRegex(ValueError, "Panel SHA256 mismatch"):
                data.load_frozen_inputs(root)

    def test_malformed_surface_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred, _, manifest = frozen_fixture(root)[0]
            frame = pd.read_csv(pred)
            frame.loc[0, "predicted_surface_flat"] = json.dumps([0.15] * 255)
            frame.to_csv(pred, index=False)
            refresh_source_hash(pred, manifest, "prediction_sha256")
            with self.assertRaisesRegex(ValueError, "must have shape"):
                data.load_frozen_inputs(root)

    def test_unmatched_and_duplicate_pairs_are_rejected(self) -> None:
        for mode in ("unmatched", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, panel, manifest = frozen_fixture(root)[0]
                frame = pd.read_csv(panel)
                if mode == "unmatched":
                    frame.loc[0, "pair_id"] = "other-pair"
                else:
                    frame = pd.concat([frame, frame], ignore_index=True)
                frame.to_csv(panel, index=False)
                refresh_source_hash(panel, manifest, "panel_sha256")
                with self.assertRaisesRegex(ValueError, "Unmatched pair_id|Duplicate pair_id"):
                    data.load_frozen_inputs(root)

    def test_raw_joint_mask_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pred, _, manifest = frozen_fixture(root)[0]
            frame = pd.read_csv(pred)
            mask = [1] * 256
            mask[0] = 0
            frame.loc[0, "support_mask_flat"] = json.dumps(mask)
            frame.loc[0, "supported_cell_count"] = 255
            frame.to_csv(pred, index=False)
            refresh_source_hash(pred, manifest, "prediction_sha256")
            with self.assertRaisesRegex(ValueError, "Raw joint support mask mismatch"):
                data.load_frozen_inputs(root)


if __name__ == "__main__":
    unittest.main()
