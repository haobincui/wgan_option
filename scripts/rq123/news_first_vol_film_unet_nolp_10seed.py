"""Run the frozen ten-seed RQ1--RQ3 matrix with the selected FiLM U-Net.

This module is intentionally a thin, branch-local wrapper around the audited
legacy-width orchestrator.  The data, arm, continuation, evaluation, and
analysis contracts remain unchanged; only the immutable model/profile and
experiment identity are replaced while an action is executing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import html
import io
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator, Mapping, Sequence

from scripts.rq123 import news_first_vol_film_nolp_10seed as core


DEFAULT_CONFIG = (
    "configs/rq123/"
    "news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_"
    "parent30_cont240_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240_rolling_v1"
)
GENERATOR_MODE = "film_unet_mask_coords_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
CAPACITY_PROFILE = "c32"
WORKER_MODULE = "scripts.rq123.news_first_vol_film_unet_nolp_10seed"
SOURCE_CODE_RELATIVE_PATHS = (
    *core.SOURCE_CODE_RELATIVE_PATHS,
    "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py",
)
EXPECTED_PARAMETER_COUNTS = {
    "generator": 827_745,
    "critic": 729_157,
    "total": 1_556_902,
}
EXPECTED_ARCHITECTURE_PROFILE_SHA256 = (
    "2b62f513e37aab49e447c75bcae25ef7d6c8e02fc9e0fe1fea6e7628ac7f3543"
)
_CORE_POSTPROCESS = core.postprocess

_CORE_PROFILE_OVERRIDES: Mapping[str, Any] = {
    "DEFAULT_CONFIG": DEFAULT_CONFIG,
    "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
    "EXPERIMENT_KIND": EXPERIMENT_KIND,
    "GENERATOR_MODE": GENERATOR_MODE,
    "CRITIC_MODE": CRITIC_MODE,
    "CAPACITY_PROFILE": CAPACITY_PROFILE,
    "WORKER_MODULE": WORKER_MODULE,
    "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
    "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
    "EXPECTED_ARCHITECTURE_PROFILE_SHA256": (EXPECTED_ARCHITECTURE_PROFILE_SHA256),
}


@contextmanager
def unet_nolp_profile() -> Iterator[None]:
    """Temporarily install the U-Net contract and restore every core global."""

    replacements = {
        **_CORE_PROFILE_OVERRIDES,
        "postprocess": _profile_postprocess,
    }
    originals = {name: getattr(core, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(core, name, value)


def _immutable_bytes(path: Path, payload: bytes) -> Path:
    """Create one deterministic report artifact without rewriting frozen output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Frozen branch-epoch report drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Branch-epoch cell report cannot be empty")
    buffer = io.StringIO(newline="")
    columns = tuple(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {column: row.get(column, "") for column in columns} for row in rows
    )
    return buffer.getvalue().encode("utf-8")


def _epoch_statistics(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty branch-epoch distribution")
    return {
        "cell_count": len(ordered),
        "minimum": ordered[0],
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "maximum": ordered[-1],
        "e_equals_one_cells": int(sum(value == 1 for value in ordered)),
        "frequency": {
            str(epoch): int(count) for epoch, count in sorted(Counter(ordered).items())
        },
    }


def _branch_epoch_report_paths(root: Path) -> tuple[Path, ...]:
    report = root / "report"
    return (
        report / "branch_epoch_cells.csv",
        report / "branch_epoch_summary.json",
        report / "branch_epoch_report.md",
        report / "branch_epoch_report.html",
    )


def _write_branch_epoch_report(root: Path) -> tuple[Path, ...]:
    """Report every frozen continuation E and replayed G/D LR boundary."""

    registry = core.read_registry(root)
    if not registry.get("branch_recipes_frozen"):
        raise RuntimeError("Branch epoch reporting requires frozen recipes")
    paths = _branch_epoch_report_paths(root)
    if registry.get("terminal_complete") and not all(path.is_file() for path in paths):
        raise RuntimeError(
            "Completed experiment is missing frozen branch-epoch reports"
        )

    recipes = core._validate_recipe_manifest(root)
    continuation_jobs = {
        str(job["job_id"]): job
        for job in registry.get("jobs") or []
        if str(job.get("stage")) == core.CONTINUATION_STAGE
    }
    if set(recipes) != set(continuation_jobs):
        raise ValueError("Continuation jobs and frozen branch recipes disagree")

    fold_order = {fold: index for index, fold in enumerate(core.FOLDS)}
    seed_order = {seed: index for index, seed in enumerate(core.SEEDS)}
    rows: list[dict[str, Any]] = []
    for continuation_id, recipe in recipes.items():
        job = continuation_jobs[continuation_id]
        payload = core.read_json(recipe["path"])
        epochs = int(payload["num_epochs"])
        generator_trace = list(payload["generator_lr_trace"])
        discriminator_trace = list(payload["discriminator_lr_trace"])
        if len(generator_trace) != epochs or len(discriminator_trace) != epochs:
            raise ValueError(f"Incomplete LR replay trace: {continuation_id}")
        rows.append(
            {
                "tolerance_minutes": int(job["tolerance_minutes"]),
                "fold": str(job["fold"]),
                "seed": int(job["seed"]),
                "continuation_job_id": continuation_id,
                "shared_branch_epochs_E": epochs,
                "generator_lr_first": float(generator_trace[0]["lr"]),
                "generator_lr_last": float(generator_trace[-1]["lr"]),
                "discriminator_lr_first": float(discriminator_trace[0]["lr"]),
                "discriminator_lr_last": float(discriminator_trace[-1]["lr"]),
                "parent_state_sha256": str(payload["parent_state_sha256"]),
                "recipe_sha256": str(recipe["sha256"]),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["tolerance_minutes"]),
            fold_order[str(row["fold"])],
            seed_order[int(row["seed"])],
        )
    )
    if len(rows) != 80:
        raise ValueError("Branch epoch report must contain exactly 80 cells")

    all_epochs = [int(row["shared_branch_epochs_E"]) for row in rows]
    summary = {
        "schema_version": 1,
        "kind": "rq123_frozen_branch_epoch_distribution_v1",
        "selection_rule": (
            "continuation_no_text best additional epoch E; every text branch "
            "starts from the same parent full-state and replays exactly E epochs"
        ),
        "post_hoc_epoch_adjustment": False,
        "global": _epoch_statistics(all_epochs),
        "by_tolerance": {
            str(tolerance): _epoch_statistics(
                [
                    int(row["shared_branch_epochs_E"])
                    for row in rows
                    if int(row["tolerance_minutes"]) == tolerance
                ]
            )
            for tolerance in core.TOLERANCES
        },
    }

    summary_rows = [
        ("all", summary["global"]),
        *(
            (f"{value}m", summary["by_tolerance"][str(value)])
            for value in core.TOLERANCES
        ),
    ]
    markdown = [
        "# Frozen branch epoch distribution",
        "",
        "Each text arm starts from its cell's common parent full-state and uses "
        "the no-text continuation's frozen additional epoch count `E` and G/D "
        "learning-rate traces. `E=1` is retained without post-hoc adjustment.",
        "",
        "| panel | cells | min E | median E | mean E | max E | E=1 cells |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in summary_rows:
        markdown.append(
            f"| {label} | {stats['cell_count']} | {stats['minimum']} | "
            f"{stats['median']:.1f} | {stats['mean']:.3f} | "
            f"{stats['maximum']} | {stats['e_equals_one_cells']} |"
        )
    markdown.extend(
        [
            "",
            "## Cell-level frozen recipes",
            "",
            "| tolerance | fold | seed | E | G LR first | G LR last | D LR first | D LR last |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        markdown.append(
            f"| {row['tolerance_minutes']}m | {row['fold']} | {row['seed']} | "
            f"{row['shared_branch_epochs_E']} | {row['generator_lr_first']:.12g} | "
            f"{row['generator_lr_last']:.12g} | {row['discriminator_lr_first']:.12g} | "
            f"{row['discriminator_lr_last']:.12g} |"
        )
    markdown_text = "\n".join(markdown) + "\n"

    html_summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td><td>{stats['cell_count']}</td>"
        f"<td>{stats['minimum']}</td><td>{stats['median']:.1f}</td>"
        f"<td>{stats['mean']:.3f}</td><td>{stats['maximum']}</td>"
        f"<td>{stats['e_equals_one_cells']}</td></tr>"
        for label, stats in summary_rows
    )
    html_cell_rows = "".join(
        "<tr>"
        f"<td>{row['tolerance_minutes']}m</td>"
        f"<td>{html.escape(str(row['fold']))}</td>"
        f"<td>{row['seed']}</td><td>{row['shared_branch_epochs_E']}</td>"
        f"<td>{row['generator_lr_first']:.12g}</td>"
        f"<td>{row['generator_lr_last']:.12g}</td>"
        f"<td>{row['discriminator_lr_first']:.12g}</td>"
        f"<td>{row['discriminator_lr_last']:.12g}</td></tr>"
        for row in rows
    )
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Frozen branch epoch distribution</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.35rem;text-align:right}"
        "th:nth-child(2),td:nth-child(2){text-align:left}</style></head><body>"
        "<h1>Frozen branch epoch distribution</h1>"
        "<p>Every text arm uses the no-text continuation's frozen E and G/D LR "
        "trace; E=1 is retained without post-hoc adjustment.</p>"
        "<table><thead><tr><th>panel</th><th>cells</th><th>min E</th>"
        "<th>median E</th><th>mean E</th><th>max E</th><th>E=1 cells</th>"
        f"</tr></thead><tbody>{html_summary_rows}</tbody></table>"
        "<h2>Cell-level frozen recipes</h2><table><thead><tr>"
        "<th>tolerance</th><th>fold</th><th>seed</th><th>E</th>"
        "<th>G LR first</th><th>G LR last</th><th>D LR first</th><th>D LR last</th>"
        f"</tr></thead><tbody>{html_cell_rows}</tbody></table></body></html>\n"
    )

    csv_path, json_path, markdown_path, html_path = paths
    _immutable_bytes(csv_path, _csv_bytes(rows))
    _immutable_bytes(
        json_path,
        (
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    _immutable_bytes(markdown_path, markdown_text.encode("utf-8"))
    _immutable_bytes(html_path, html_text.encode("utf-8"))
    return paths


def _profile_postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    root = Path(output_dir).resolve()
    registry = core.read_registry(root)
    if registry.get("branch_recipes_frozen"):
        _write_branch_epoch_report(root)
    return _CORE_POSTPROCESS(root, resume=resume)


def run_action(
    action: str,
    *,
    config_path: str = DEFAULT_CONFIG,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
    job_id_value: str = "",
    worker_dry_run: bool = False,
) -> Any:
    """Dispatch one core action under the temporary U-Net profile."""

    with unet_nolp_profile():
        return core.run_action(
            action,
            config_path=config_path,
            output_dir=output_dir,
            resume=resume,
            job_id_value=job_id_value,
            worker_dry_run=worker_dry_run,
        )


def _parser() -> argparse.ArgumentParser:
    parser = core._parser()
    parser.set_defaults(config=DEFAULT_CONFIG, output_dir=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_action(
        args.action,
        config_path=args.config,
        output_dir=args.output_dir,
        resume=bool(args.resume),
        job_id_value=args.job_id,
        worker_dry_run=bool(args.worker_dry_run),
    )
    print(
        core.json.dumps(result, indent=2, default=str)
        if isinstance(result, dict)
        else result
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
