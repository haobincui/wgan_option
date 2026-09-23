"""Audited recovery for independent wrong-text overlay materialization.

The frozen intervention builder already writes the receiver-to-donor mapping
as a separate, hashed CSV and embeds the donor LP vector in each receiver row.
Its inference mode is intentionally ``lp_mean_l2``.  Such manifests must not
also carry ``donor_pair_id`` (that field is reserved for the training-time
``lp_shuffle`` mode), but the builder accidentally forwarded it.

This recovery strips only that redundant field at the manifest boundary while
requiring the separately frozen mapping lineage.  It then delegates to the
noise-lineage recovery, so checkpoints, embeddings, mappings, and predictions
remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_noise_recovery as noise_recovery,
)


RECOVERY_KIND = "pure_cnn_film_text_effect_intervention_overlay_recovery_v1"
RECOVERY_PATH = "registry/intervention_overlay_recovery.json"


@contextmanager
def _corrected_overlay_writer() -> Iterator[None]:
    from wgan_option.utils import news_first_experiment_core as overlay_core

    original = overlay_core.write_pair_text_overlay_manifest

    def corrected(
        path: str | Path,
        *,
        mode: object,
        namespace: str,
        records: Sequence[Mapping[str, object]],
        transform: Mapping[str, object] | None = None,
    ) -> dict[str, str]:
        normalized = overlay_core.normalize_pair_text_overlay_mode(mode)
        donor_records = [row for row in records if str(row.get("donor_pair_id", ""))]
        if normalized == "lp_mean_l2" and donor_records:
            metadata = dict(transform or {})
            if (
                metadata.get("method") != "independent_cross_session_derangement_v1"
                or not str(metadata.get("mapping_path", ""))
                or not str(metadata.get("mapping_sha256", ""))
            ):
                raise ValueError(
                    "Wrong-text donor stripping requires frozen mapping lineage"
                )
            records = [
                {key: value for key, value in row.items() if key != "donor_pair_id"}
                for row in records
            ]
        return original(
            path,
            mode=mode,
            namespace=namespace,
            records=records,
            transform=transform,
        )

    overlay_core.write_pair_text_overlay_manifest = corrected
    try:
        yield
    finally:
        overlay_core.write_pair_text_overlay_manifest = original


def _write_audit(root: Path) -> Path:
    source = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECOVERY_KIND,
        "status": "installed",
        "scientific_contract_changed": False,
        "checkpoint_embedding_mapping_or_prediction_changed": False,
        "repair_scope": "wrong_text_overlay_redundant_donor_field_only",
        "required_mapping_method": "independent_cross_session_derangement_v1",
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": experiment.sha256_file(source),
    }
    path = root / RECOVERY_PATH
    if path.is_file():
        observed = lifecycle._read_signed(path, kind=RECOVERY_KIND)
        if {
            key: value for key, value in observed.items() if key != "payload_sha256"
        } != payload:
            raise ValueError("Intervention-overlay recovery audit drift")
        return path
    return lifecycle._write_signed(path, payload)


def run(output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir).expanduser().resolve()
    _write_audit(root)
    with _corrected_overlay_writer():
        return noise_recovery.run(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--output-dir", default=experiment.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.action == "run":
        print(run(args.output_dir), flush=True)
    else:
        import json

        from scripts.rq3 import (
            news_first_vol_film_unet_pure_cnn_backbone_text_effect_recovery as recovery,
        )

        print(json.dumps(recovery.status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
