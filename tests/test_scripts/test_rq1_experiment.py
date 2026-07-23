import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.baselines import PCARidgeForecaster  # noqa: E402
from film_wgan.data import FilmWGANSample  # noqa: E402
from scripts.rq1.rq1_experiment import (  # noqa: E402
    _cluster_bootstrap,
    _dm_hac,
    _exact_wilcoxon_p,
    _holm_adjust,
    _validate_textbase_provenance,
)


def _sample(index: int) -> FilmWGANSample:
    current = np.asarray([[0.20, 0.21], [0.22, 0.23]], dtype=np.float32) + index * 0.001
    delta = np.asarray([[0.001, -0.001], [0.002, -0.002]], dtype=np.float32) * (1.0 + index / 20.0)
    timestamp = pd.Timestamp("2022-01-01T12:00:00Z") + pd.Timedelta(minutes=index * 5)
    return FilmWGANSample(
        sample_id=f"sample_{index}",
        global_index=index,
        timestamp=timestamp.isoformat(),
        current_snapshot_time_utc=timestamp.isoformat(),
        target_snapshot_time_utc=(timestamp + pd.Timedelta(minutes=5)).isoformat(),
        current_surface=current,
        target_surface=current * np.exp(delta),
        strike_grid=np.asarray([0.9, 1.1], dtype=np.float32),
        maturity_days_grid=np.asarray([7.0, 30.0], dtype=np.float32),
        text_embedding=np.zeros(3, dtype=np.float32),
        metadata={},
    )


class TestRQ1PcaRidge(unittest.TestCase):
    def test_selection_uses_validation_and_predicts_fixed_shape(self):
        train = [_sample(index) for index in range(12)]
        validation = [_sample(index) for index in range(12, 16)]
        model, selection, audit = PCARidgeForecaster.select(
            train_samples=train,
            val_samples=validation,
            component_grid=(2, 4),
            alpha_grid=(0.1, 1.0),
        )

        predicted = model.predict(validation)
        self.assertEqual(tuple(predicted.shape), (4, 2, 2))
        self.assertIn(selection.n_components, {2, 4})
        self.assertIn(selection.alpha, {0.1, 1.0})
        self.assertEqual(len(audit), 4)
        self.assertTrue(np.isfinite(predicted).all())


class TestRQ1Statistics(unittest.TestCase):
    def test_cluster_bootstrap_preserves_positive_baseline_minus_text_direction(self):
        frame = pd.DataFrame(
            {
                "trading_date": ["2022-01-01"] * 3 + ["2022-01-02"] * 3 + ["2022-01-03"] * 3,
                "difference": [0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.02, 0.01],
            }
        )
        result = _cluster_bootstrap(frame, iterations=500, seed=7)
        self.assertGreater(result["mean_difference"], 0.0)
        self.assertGreater(result["ci_lower"], 0.0)

    def test_dm_hac_uses_positive_difference_as_text_better(self):
        result = _dm_hac(np.linspace(0.01, 0.03, 30))
        self.assertGreater(result["dm_statistic"], 0.0)
        self.assertLess(result["p_text_better"], 0.05)

    def test_holm_adjustment_is_monotone_and_not_smaller_than_raw(self):
        raw = [0.01, 0.04, 0.02]
        adjusted = _holm_adjust(raw)
        self.assertTrue(all(value >= original for value, original in zip(adjusted, raw)))
        ordered = sorted(zip(raw, adjusted))
        self.assertTrue(all(ordered[index][1] <= ordered[index + 1][1] for index in range(len(ordered) - 1)))

    def test_exact_wilcoxon_enumerates_all_seed_sign_assignments(self):
        self.assertAlmostEqual(_exact_wilcoxon_p([1.0, 2.0, 3.0, 4.0, 5.0]), 0.0625)
        self.assertEqual(_exact_wilcoxon_p([0.0] * 5), 1.0)


class TestRQ1ConfigProvenance(unittest.TestCase):
    def test_only_preregistered_textbase_changes_are_allowed(self):
        original = {"lambda_recon": 20.0, "train_ratio": 0.8, "checkpoint_metric": "old"}
        frozen = {"lambda_recon": 20.0, "train_ratio": 0.7, "checkpoint_metric": "val_mae"}
        _validate_textbase_provenance(original, frozen)
        frozen["lambda_recon"] = 10.0
        with self.assertRaisesRegex(ValueError, "lambda_recon"):
            _validate_textbase_provenance(original, frozen)


if __name__ == "__main__":
    unittest.main()
