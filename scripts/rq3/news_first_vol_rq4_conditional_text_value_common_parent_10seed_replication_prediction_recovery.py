"""Audited prediction-boundary recovery for the RQ4 replication.

All training, checkpoint freezing, validation trajectories, and standard
prediction cells are already complete.  The frozen public aggregator confuses
the declared shared-noise contract SHA with the concrete predictor SHA, and
the intervention overlay writer forwards a redundant donor field.  Both
execution-layer repairs already have focused tests and were used by the
preceding experiment.  This wrapper installs them without changing frozen
checkpoints, prediction values, or the scientific contract, then delegates to
the RQ4 lifecycle recovery runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from scripts.rq3 import (
    news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication as wrapper,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_lifecycle as lifecycle,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_intervention_recovery as intervention_recovery,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_noise_recovery as noise_recovery,
)
from scripts.rq3 import (  # noqa: E402
    news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_recovery as rq4_recovery,
)


RECOVERY_KIND = "rq4_prediction_boundary_recovery_v1"
RECOVERY_PATH = "registry/rq4_prediction_boundary_recovery.json"
RECOVERY_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_prediction_recovery.py"
)
SUPERVISOR_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_prediction_recovery_supervisor.py"
)


def _source_row(role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "artifact_role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": experiment.sha256_file(resolved),
    }


def _write_audit(root: Path) -> Path:
    source_rows = [
        _source_row("rq4_prediction_recovery", experiment.REPO_ROOT / RECOVERY_SOURCE),
        _source_row(
            "rq4_prediction_recovery_supervisor",
            experiment.REPO_ROOT / SUPERVISOR_SOURCE,
        ),
        _source_row("noise_lineage_repair", Path(noise_recovery.__file__)),
        _source_row(
            "intervention_overlay_repair", Path(intervention_recovery.__file__)
        ),
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECOVERY_KIND,
        "status": "installed",
        "scientific_contract_changed": False,
        "training_or_checkpoint_recomputed": False,
        "completed_standard_prediction_cells_recomputed": False,
        "repair_scope": [
            "declared_vs_concrete_mc64_sha_aggregation",
            "redundant_intervention_donor_field_at_manifest_boundary",
        ],
        "source_rows": source_rows,
    }
    path = root / RECOVERY_PATH
    if path.is_file():
        observed = lifecycle._read_signed(path, kind=RECOVERY_KIND)
        expected = dict(payload)
        expected["payload_sha256"] = observed["payload_sha256"]
        if observed != expected:
            raise ValueError("RQ4 prediction recovery audit drift")
        return path
    return lifecycle._write_signed(path, payload)


def run(output_dir: str | Path = wrapper.DEFAULT_OUTPUT_DIR) -> Path:
    wrapper._configure()
    root = Path(output_dir).expanduser().resolve()
    experiment.validate_partial_root(root, verify_stage_roots=True)
    _write_audit(root)
    # The overlay repair applies while intervention inputs are materialized;
    # the noise repair applies only while standard/intervention cells are
    # aggregated.  Worker generation and every numeric prediction stay on the
    # original frozen code path.
    with (
        intervention_recovery._corrected_overlay_writer(),
        noise_recovery._corrected_aggregation(),
    ):
        return rq4_recovery.run_recovery(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--output-dir", default=wrapper.DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        print(run(args.output_dir), flush=True)
    else:
        import json

        print(json.dumps(rq4_recovery.status(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run", "main"]
