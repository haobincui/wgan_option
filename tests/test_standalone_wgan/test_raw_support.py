from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from film_wgan.support import (
    RawSurfaceSupportArtifact,
    fit_raw_support_artifact,
    parse_raw_surface_params,
    raw_support_mask,
    reconstruct_raw_surface,
    validate_support_artifact_lineage,
)


def _params(
    *,
    days: tuple[int, int] = (20, 60),
    first_bounds: tuple[float, float] = (0.90, 1.10),
    second_bounds: tuple[float, float] = (0.95, 1.05),
) -> dict[str, list]:
    return {
        "business_days": list(days),
        "percent_strikes": [
            [first_bounds[0], 1.0, first_bounds[1]],
            [second_bounds[0], 1.0, second_bounds[1]],
        ],
        "implied_vols": [
            [0.20, 0.19, 0.21],
            [0.22, 0.20, 0.23],
        ],
    }


class RawSupportTests(unittest.TestCase):
    def test_support_uses_bracketing_strike_intersection(self):
        mask = raw_support_mask(
            _params(),
            strike_grid=[0.90, 0.95, 1.00, 1.05, 1.10],
            maturity_days_grid=[10, 20, 40, 60, 80],
        )
        np.testing.assert_array_equal(mask[0], np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(mask[1], [True, True, True, True, True])
        np.testing.assert_array_equal(mask[2], [False, True, True, True, False])
        np.testing.assert_array_equal(mask[3], [False, True, True, True, False])
        np.testing.assert_array_equal(mask[4], np.zeros(5, dtype=bool))

    def test_parser_averages_duplicate_strikes_and_requires_two_maturities(self):
        params = {
            "business_days": [20, 20, 60],
            "percent_strikes": [[1.0], [1.0], [0.95, 1.05]],
            "implied_vols": [[0.20], [0.22], [0.23, 0.24]],
        }
        parsed = parse_raw_surface_params(params)
        self.assertEqual(parsed["business_days"], [20, 60])
        self.assertEqual(parsed["percent_strikes"][0], [1.0])
        self.assertAlmostEqual(parsed["implied_vols"][0][0], 0.21)

    def test_reconstruction_and_artifact_round_trip(self):
        records = [
            {
                "surface_pair_id": f"pair_{index}",
                "current_surface_params": _params(),
                "target_surface_params": _params(
                    first_bounds=(0.91, 1.09),
                    second_bounds=(0.96, 1.04),
                ),
            }
            for index in range(4)
        ]
        artifact = fit_raw_support_artifact(
            records,
            input_workbook_sha256="abc",
            strike_bins=4,
            maturity_bins=4,
            quantile_low=0.0,
            quantile_high=1.0,
        )
        self.assertEqual(artifact.train_pair_count, 4)
        self.assertEqual(len(artifact.strike_grid), 4)
        self.assertEqual(len(artifact.maturity_days_grid), 4)
        reconstructed = reconstruct_raw_surface(
            _params(),
            strike_grid=artifact.strike_grid,
            maturity_days_grid=artifact.maturity_days_grid,
        )
        self.assertEqual(reconstructed.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(reconstructed)))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "support.json"
            artifact.save(path)
            loaded = RawSurfaceSupportArtifact.load(path)
            self.assertEqual(loaded.strike_grid, artifact.strike_grid)
            validate_support_artifact_lineage(
                loaded,
                train_pair_ids=[f"pair_{index}" for index in range(4)],
                input_workbook_sha256="abc",
            )
            with self.assertRaisesRegex(ValueError, "ordered training pairs"):
                validate_support_artifact_lineage(
                    loaded,
                    train_pair_ids=["wrong"],
                    input_workbook_sha256="abc",
                )


if __name__ == "__main__":
    unittest.main()
