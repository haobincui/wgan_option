"""Audited recovery for the parallel MC64 noise-lineage aggregation bug.

The frozen prediction plan records an arm-independent *declared* noise-bank
contract.  The underlying predictor records a second SHA for the concrete
noise implementation.  Parallel workers correctly preserve both values, but
the frozen public aggregator compared the concrete SHA in pair evidence with
the declared SHA.  Those hashes describe different layers and are not
expected to be equal.

This recovery leaves every checkpoint and prediction artifact unchanged.  It
verifies the declared SHA against the cell manifest, verifies the concrete SHA
against the pair evidence, substitutes the concrete SHA only for aggregation,
and then delegates the rest of the terminal pipeline to the existing audited
recovery runner.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_recovery as recovery,
)


RECOVERY_KIND = "pure_cnn_film_text_effect_noise_lineage_recovery_v1"
RECOVERY_PATH = "registry/prediction_noise_lineage_recovery.json"


def _adapt_units_to_concrete_noise(
    root: Path, stage: str, units: pd.DataFrame
) -> pd.DataFrame:
    """Validate both noise-lineage layers and return concrete-SHA units."""

    adapted = units.copy()
    concrete: list[str] = []
    for unit in units.to_dict(orient="records"):
        unit_id = str(unit["prediction_unit_id"])
        cell, artifacts = experiment._parallel_cell_artifacts(root, stage, unit_id)
        metadata = dict(cell.get("metadata") or {})
        declared = str(unit["noise_bank_profile_sha256"])
        if (
            str(cell.get("noise_bank_profile_sha256")) != declared
            or str(metadata.get("declared_shared_noise_bank_profile_sha256"))
            != declared
        ):
            raise ValueError(f"Declared noise-bank lineage drift: {unit_id}")
        concrete_sha = str(metadata.get("core_noise_bank_profile_sha256", ""))
        if len(concrete_sha) != 64:
            raise ValueError(f"Concrete noise-bank SHA is missing: {unit_id}")
        evidence = pd.read_csv(artifacts["pair_metrics"]["path"], dtype=str)
        observed = set(evidence["noise_bank_profile_sha256"].astype(str))
        if observed != {concrete_sha}:
            raise ValueError(f"Concrete noise-bank lineage drift: {unit_id}")
        concrete.append(concrete_sha)
    adapted["noise_bank_profile_sha256"] = concrete
    grouped = adapted.groupby(["seed", "fold"])["noise_bank_profile_sha256"].nunique()
    if not grouped.eq(1).all():
        raise ValueError(f"Concrete MC noise is not shared within seed/fold: {stage}")
    return adapted


@contextmanager
def _corrected_aggregation() -> Iterator[None]:
    original_standard = experiment._parallel_standard_aggregate
    original_intervention = experiment._parallel_intervention_aggregate

    def standard(root: Path, units: pd.DataFrame) -> tuple[Path, Path]:
        return original_standard(
            root, _adapt_units_to_concrete_noise(root, "standard", units)
        )

    def intervention(root: Path, units: pd.DataFrame) -> tuple[Path, Path]:
        return original_intervention(
            root, _adapt_units_to_concrete_noise(root, "interventions", units)
        )

    experiment._parallel_standard_aggregate = standard
    experiment._parallel_intervention_aggregate = intervention
    try:
        yield
    finally:
        experiment._parallel_standard_aggregate = original_standard
        experiment._parallel_intervention_aggregate = original_intervention


def _write_audit(root: Path) -> Path:
    source = Path(__file__).resolve()
    path = root / RECOVERY_PATH
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECOVERY_KIND,
        "status": "installed",
        "scientific_contract_changed": False,
        "checkpoint_or_prediction_recomputed": False,
        "repair_scope": "parallel_prediction_noise_sha_aggregation_only",
        "declared_sha_role": "arm_independent_shared_noise_contract",
        "concrete_sha_role": "core_prediction_noise_implementation",
        "validation": (
            "declared SHA equals unit and cell metadata; concrete SHA equals pair "
            "evidence and is unique within every seed/fold"
        ),
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": experiment.sha256_file(source),
    }
    if path.is_file():
        observed = lifecycle._read_signed(path, kind=RECOVERY_KIND)
        if observed != {**payload, "payload_sha256": observed["payload_sha256"]}:
            raise ValueError("Noise-lineage recovery audit drift")
        return path
    return lifecycle._write_signed(path, payload)


def run(
    output_dir: str | Path = experiment.DEFAULT_OUTPUT_DIR,
    *,
    shared_workers_per_gpu: int = recovery.DEFAULT_SHARED_WORKERS_PER_GPU,
    idle_workers_per_gpu: int = recovery.DEFAULT_IDLE_WORKERS_PER_GPU,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    _write_audit(root)
    with _corrected_aggregation():
        return recovery.run_shared_pipeline(
            root,
            shared_workers_per_gpu=shared_workers_per_gpu,
            idle_workers_per_gpu=idle_workers_per_gpu,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--output-dir", default=experiment.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shared-workers-per-gpu", type=int, default=4)
    parser.add_argument("--idle-workers-per-gpu", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        print(
            run(
                args.output_dir,
                shared_workers_per_gpu=args.shared_workers_per_gpu,
                idle_workers_per_gpu=args.idle_workers_per_gpu,
            ),
            flush=True,
        )
    else:
        import json

        print(json.dumps(recovery.status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
