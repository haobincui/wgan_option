from __future__ import annotations

import unittest

import numpy as np

from scripts.rq1_sequence.conformal_calibration import (
    _conformal_quantile_level,
    _day_balanced_weights,
    _weighted_quantile,
)


class Rq1SequenceConformalTest(unittest.TestCase):
    def test_day_balanced_weights_give_each_day_equal_mass(self) -> None:
        days = np.asarray(["a", "a", "a", "b"])
        weights = _day_balanced_weights(days)
        self.assertAlmostEqual(float(weights[days == "a"].sum()), 0.5)
        self.assertAlmostEqual(float(weights[days == "b"].sum()), 0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_weighted_quantile_uses_day_balanced_distribution(self) -> None:
        values = np.asarray([0.0, 0.0, 0.0, 10.0])
        days = np.asarray(["a", "a", "a", "b"])
        weights = _day_balanced_weights(days)
        self.assertEqual(_weighted_quantile(values, weights, 0.5), 0.0)
        self.assertEqual(_weighted_quantile(values, weights, 0.8), 10.0)

    def test_conformal_level_has_finite_day_correction(self) -> None:
        self.assertEqual(_conformal_quantile_level(0.9, 20), 0.95)
        self.assertEqual(_conformal_quantile_level(0.9, 100), 0.91)


if __name__ == "__main__":
    unittest.main()
