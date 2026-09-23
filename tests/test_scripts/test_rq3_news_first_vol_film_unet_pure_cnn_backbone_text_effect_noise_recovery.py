from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_noise_recovery as subject,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapt_units_separates_declared_and_concrete_noise(
    tmp_path: Path, monkeypatch
) -> None:
    pair_path = tmp_path / "pairs.csv"
    pd.DataFrame({"noise_bank_profile_sha256": ["b" * 64, "b" * 64]}).to_csv(
        pair_path, index=False
    )
    cell = {
        "noise_bank_profile_sha256": "a" * 64,
        "metadata": {
            "declared_shared_noise_bank_profile_sha256": "a" * 64,
            "core_noise_bank_profile_sha256": "b" * 64,
        },
    }
    artifacts = {
        "pair_metrics": {
            "path": str(pair_path),
            "size_bytes": pair_path.stat().st_size,
            "sha256": _sha(pair_path),
        }
    }
    monkeypatch.setattr(
        subject.experiment,
        "_parallel_cell_artifacts",
        lambda _root, _stage, _unit_id: (cell, artifacts),
    )
    units = pd.DataFrame(
        [
            {
                "prediction_unit_id": "u1",
                "seed": 42,
                "fold": "f1",
                "noise_bank_profile_sha256": "a" * 64,
            },
            {
                "prediction_unit_id": "u2",
                "seed": 42,
                "fold": "f1",
                "noise_bank_profile_sha256": "a" * 64,
            },
        ]
    )
    result = subject._adapt_units_to_concrete_noise(tmp_path, "standard", units)
    assert set(result["noise_bank_profile_sha256"]) == {"b" * 64}
    assert set(units["noise_bank_profile_sha256"]) == {"a" * 64}
