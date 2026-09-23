from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_intervention_recovery as subject,
)


class InterventionRecoveryTests(unittest.TestCase):
    def test_corrected_writer_strips_only_redundant_wrong_text_donor(self) -> None:
        from wgan_option.utils import news_first_experiment_core as core

        captured = {}

        def fake(path, *, mode, namespace, records, transform=None):
            captured.update(mode=mode, records=list(records), transform=transform)
            return {"path": str(path)}

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "write_pair_text_overlay_manifest", fake):
                with subject._corrected_overlay_writer():
                    core.write_pair_text_overlay_manifest(
                        Path(directory) / "overlay.json",
                        mode="lp_mean_l2",
                        namespace="wrong",
                        records=[
                            {
                                "pair_id": "a",
                                "session_id": "s1",
                                "embedding": [0.0] * 1024,
                                "donor_pair_id": "b",
                            }
                        ],
                        transform={
                            "method": "independent_cross_session_derangement_v1",
                            "mapping_path": "/tmp/mapping.csv",
                            "mapping_sha256": "a" * 64,
                        },
                    )
        self.assertNotIn("donor_pair_id", captured["records"][0])
        self.assertEqual(captured["mode"], "lp_mean_l2")


if __name__ == "__main__":
    unittest.main()
