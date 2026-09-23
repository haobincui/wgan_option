"""Thin spawn-safe CLI for parallel text-effect prediction cells.

The module intentionally has no Torch, pandas, model, or evaluator imports at
module scope.  This matters when ``multiprocessing`` re-imports the launch
module under ``spawn``: each child first runs the scheduler initializer, which
sets ``CUDA_VISIBLE_DEVICES`` to one physical GPU, and only then imports the
production prediction callback.

Example (safe for ``nohup``/``setsid``)::

    python -m scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_supervisor \
      --units outputs/.../evaluation/parallel_prediction_units.json \
      --control-dir outputs/..._control/prediction \
      --artifact-root outputs/... \
      --gpu-ids 0,1 --workers-per-gpu 4 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKER_ENTRYPOINT = (
    "scripts.rq3."
    "news_first_vol_film_unet_pure_cnn_backbone_text_effect_prediction_worker:"
    "run_prediction_unit"
)


def _coerce_csv_value(key: str, value: str) -> Any:
    stripped = value.strip()
    if key in {
        "seed",
        "tolerance_minutes",
        "mc_samples",
        "checkpoint_size_bytes",
        "epoch",
        "physical_gpu_id",
        "logical_device",
    }:
        return int(stripped)
    if key in {"expected_artifact_roles"}:
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError(f"CSV {key} must encode a JSON list")
        return parsed
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped


def load_units(path_value: str | Path) -> list[dict[str, Any]]:
    """Read a JSON list/``{"units": [...]}`` or a typed CSV unit manifest."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction unit manifest is missing: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("units") if isinstance(raw, Mapping) else raw
        if not isinstance(rows, list) or not all(
            isinstance(row, Mapping) for row in rows
        ):
            raise ValueError("JSON prediction manifest must contain a unit list")
        result = [dict(row) for row in rows]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            result = [
                {
                    key: _coerce_csv_value(key, value)
                    for key, value in row.items()
                    if value is not None and value.strip() != ""
                }
                for row in csv.DictReader(stream)
            ]
    else:
        raise ValueError("Prediction unit manifest must be .json or .csv")
    if not result:
        raise ValueError("Prediction unit manifest is empty")
    identifiers = [str(row.get("prediction_unit_id", "")) for row in result]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Prediction unit IDs are empty or duplicated")
    return result


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--gpu-ids must be a unique comma list")
    return result


def run(
    *,
    units_path: str | Path,
    control_dir: str | Path,
    artifact_root: str | Path,
    gpu_ids: Sequence[int] = (0, 1),
    workers_per_gpu: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Launch the generic scheduler without importing the callback early."""

    # Keep this import inside the original supervisor process.  The callback
    # module is named as a string and is imported in children only after the
    # scheduler initializer has restricted GPU visibility.
    from scripts.rq3 import (
        news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction as parallel,
    )

    units = load_units(units_path)
    return parallel.run_parallel_prediction_units(
        units,
        worker_entrypoint=WORKER_ENTRYPOINT,
        control_dir=control_dir,
        artifact_root=artifact_root,
        gpu_ids=tuple(map(int, gpu_ids)),
        workers_per_gpu=int(workers_per_gpu),
        resume=bool(resume),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", required=True, help="Frozen JSON/CSV unit manifest")
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, default=(0, 1))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.workers_per_gpu) <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    result = run(
        units_path=args.units,
        control_dir=args.control_dir,
        artifact_root=args.artifact_root,
        gpu_ids=args.gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module CLI
    raise SystemExit(main())


__all__ = ["WORKER_ENTRYPOINT", "load_units", "main", "run"]
