from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42_analysis as subject,
)


FOLD_COUNTS = {
    "f1_2023q1": (110, 34),
    "f2_2023q2": (112, 36),
    "f3_2023q3": (135, 33),
    "f4_2023q4": (143, 45),
}


def _rows(arms: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_index, (fold, (pair_count, session_count)) in enumerate(
        FOLD_COUNTS.items()
    ):
        for arm_index, arm in enumerate(arms):
            for pair_index in range(pair_count):
                pair_id = f"{fold}_pair_{pair_index:03d}"
                rows.append(
                    {
                        "job_id": f"job_{fold}_{arm}",
                        "tolerance_minutes": 5,
                        "fold": fold,
                        "seed": 42,
                        "arm": arm,
                        "pair_id": pair_id,
                        "session_id": f"{fold}_session_{pair_index % session_count:03d}",
                        "effective_origin_utc": (
                            f"2023-0{fold_index + 1}-01T00:{pair_index % 60:02d}:00Z"
                        ),
                        "target_mae": 0.002 + arm_index * 1e-5 + pair_index * 1e-8,
                        "persistence_mae": 0.0021 + pair_index * 1e-8,
                        "checkpoint_sha256": hashlib.sha256(
                            f"checkpoint-{arm}".encode()
                        ).hexdigest(),
                        "prediction_sha256": hashlib.sha256(
                            f"prediction-{arm}".encode()
                        ).hexdigest(),
                        "noise_bank_profile_sha256": hashlib.sha256(
                            f"noise-{fold}".encode()
                        ).hexdigest(),
                    }
                )
    return rows


class DirectTextAblationAnalysisTests(unittest.TestCase):
    def test_combines_four_new_arms_with_one_frozen_matched_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            new = pd.DataFrame(_rows(subject.NEW_ARMS))
            new.to_csv(root / "analysis/rq12_pair_metrics.csv.gz", index=False)
            reference_path = root / "frozen_reference.csv"
            pd.DataFrame(_rows((subject.REFERENCE_SOURCE_ARM,))).to_csv(
                reference_path, index=False
            )
            frozen = {"declarations": {"pair_metrics": (reference_path, "unused")}}
            combined_path = subject._combined_pair_metrics(root, frozen)
            combined = pd.read_csv(combined_path)
            self.assertEqual(len(combined), 2_500)
            self.assertEqual(
                set(combined["arm"]),
                {"lp_matched", "lp_shuffle", "no_text", "bow", "sentiment"},
            )
            self.assertNotIn(subject.REFERENCE_SOURCE_ARM, set(combined["arm"]))

    def test_combination_rejects_reference_pair_lineage_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            pd.DataFrame(_rows(subject.NEW_ARMS)).to_csv(
                root / "analysis/rq12_pair_metrics.csv.gz", index=False
            )
            reference = pd.DataFrame(_rows((subject.REFERENCE_SOURCE_ARM,)))
            reference.loc[0, "session_id"] = "f1_2023q1_session_001"
            reference_path = root / "frozen_reference.csv"
            reference.to_csv(reference_path, index=False)
            frozen = {"declarations": {"pair_metrics": (reference_path, "unused")}}
            with self.assertRaisesRegex(ValueError, "differs across arms"):
                subject._combined_pair_metrics(root, frozen)

    def test_combination_rejects_cross_root_noise_bank_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            pd.DataFrame(_rows(subject.NEW_ARMS)).to_csv(
                root / "analysis/rq12_pair_metrics.csv.gz", index=False
            )
            reference = pd.DataFrame(_rows((subject.REFERENCE_SOURCE_ARM,)))
            reference.loc[
                reference["fold"].eq("f2_2023q2"), "noise_bank_profile_sha256"
            ] = hashlib.sha256(b"different-noise-bank").hexdigest()
            reference_path = root / "frozen_reference.csv"
            reference.to_csv(reference_path, index=False)
            frozen = {"declarations": {"pair_metrics": (reference_path, "unused")}}
            with self.assertRaisesRegex(ValueError, "share one MC64 noise bank"):
                subject._combined_pair_metrics(root, frozen)


if __name__ == "__main__":
    unittest.main()
