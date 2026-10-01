"""Analyze and render the three-seed F4 FiLM/Pure-CNN capacity sweep.

The module is deliberately independent of the lifecycle runner.  The runner
freezes one canonical pair-metric file and calls :func:`postprocess_experiment`;
the pure :func:`summarize_pair_metrics` entry point is also available for unit
tests and downstream audits.
"""

from __future__ import annotations

from collections.abc import Mapping
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd


SCHEMA_VERSION = 1
ANALYSIS_KIND = "f4_film_pure_capacity_3seed_analysis_v1"
FOLD_ID = "f4_2023q4"
ARCHITECTURES = ("film_cnn", "pure_cnn")
CAPACITY_IDS = ("c08", "c12", "c16", "c24", "c32", "c48")
CURRENT_CAPACITY_ID = "c32"
SEEDS = (42, 202, 404)
PAIR_COUNT = 143
SESSION_COUNT = 45
EXPECTED_ROWS = len(ARCHITECTURES) * len(CAPACITY_IDS) * len(SEEDS) * PAIR_COUNT
TABLE_LABEL = "tab:ch3:f4_film_pure_capacity_robustness"

PAIR_METRICS_NAME = "f4_pair_metrics.csv.gz"
SUMMARY_CSV_NAME = "f4_capacity_summary.csv"
SUMMARY_JSON_NAME = "f4_capacity_summary.json"
TABLE_TEX_NAME = "f4_capacity_table.tex"

REQUIRED_COLUMNS = (
    "architecture",
    "capacity_id",
    "seed",
    "pair_id",
    "session_id",
    "target_mae",
    "noise_bank_profile_sha256",
)
PARAMETER_COLUMNS = (
    "generator_parameters",
    "critic_parameters",
    "total_wgan_parameters",
)


class F4CapacityAnalysisError(ValueError):
    """Raised when frozen capacity evidence violates the analysis contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_write_text(path: Path, text: str) -> Path:
    return _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    csv_payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return _atomic_write_bytes(
        path, gzip.compress(csv_payload, compresslevel=9, mtime=0)
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _parameter_value(
    payload: Mapping[str, Any], canonical_name: str, short_name: str
) -> int:
    if canonical_name in payload:
        raw = payload[canonical_name]
    elif short_name in payload:
        raw = payload[short_name]
    else:
        raise F4CapacityAnalysisError(
            f"Parameter contract is missing {canonical_name!r}"
        )
    if isinstance(raw, bool):
        raise F4CapacityAnalysisError(f"{canonical_name} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise F4CapacityAnalysisError(
            f"{canonical_name} must be a positive integer"
        ) from exc
    if value <= 0 or float(raw) != value:
        raise F4CapacityAnalysisError(f"{canonical_name} must be a positive integer")
    return value


def _parameter_contract_from_mapping(
    parameter_counts: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[int, int, int]]:
    if set(parameter_counts) != set(ARCHITECTURES):
        raise F4CapacityAnalysisError(
            f"Parameter architectures must be exactly {ARCHITECTURES}"
        )
    contract: dict[tuple[str, str], tuple[int, int, int]] = {}
    for architecture in ARCHITECTURES:
        architecture_payload = parameter_counts[architecture]
        if not isinstance(architecture_payload, Mapping):
            raise F4CapacityAnalysisError(
                f"Parameter contract for {architecture} must be a mapping"
            )
        if set(architecture_payload) != set(CAPACITY_IDS):
            raise F4CapacityAnalysisError(
                f"Parameter capacities for {architecture} must be exactly {CAPACITY_IDS}"
            )
        for capacity_id in CAPACITY_IDS:
            cell = architecture_payload[capacity_id]
            if not isinstance(cell, Mapping):
                raise F4CapacityAnalysisError(
                    f"Parameter contract for {architecture}/{capacity_id} must be a mapping"
                )
            generator = _parameter_value(cell, "generator_parameters", "generator")
            critic = _parameter_value(cell, "critic_parameters", "critic")
            total = _parameter_value(cell, "total_wgan_parameters", "total")
            contract[(architecture, capacity_id)] = (generator, critic, total)
    return contract


def _parameter_contract_from_frame(
    frame: pd.DataFrame,
) -> dict[tuple[str, str], tuple[int, int, int]]:
    present = [column in frame.columns for column in PARAMETER_COLUMNS]
    if any(present) and not all(present):
        raise F4CapacityAnalysisError(
            f"Pair metrics must contain all or none of {PARAMETER_COLUMNS}"
        )
    if not all(present):
        return {}

    contract: dict[tuple[str, str], tuple[int, int, int]] = {}
    for (architecture, capacity_id), cell in frame.groupby(
        ["architecture", "capacity_id"], sort=False
    ):
        values: list[int] = []
        for column in PARAMETER_COLUMNS:
            numeric = pd.to_numeric(cell[column], errors="coerce")
            unique = numeric.drop_duplicates()
            if (
                numeric.isna().any()
                or len(unique) != 1
                or not math.isfinite(float(unique.iloc[0]))
                or float(unique.iloc[0]) <= 0
                or float(unique.iloc[0]) != int(unique.iloc[0])
            ):
                raise F4CapacityAnalysisError(
                    f"Non-constant positive integer {column} for "
                    f"{architecture}/{capacity_id}"
                )
            values.append(int(unique.iloc[0]))
        contract[(str(architecture), str(capacity_id))] = tuple(values)  # type: ignore[assignment]
    return contract


def _validate_parameter_contract(
    contract: Mapping[tuple[str, str], tuple[int, int, int]],
) -> None:
    expected = {
        (architecture, capacity_id)
        for architecture in ARCHITECTURES
        for capacity_id in CAPACITY_IDS
    }
    if set(contract) != expected:
        raise F4CapacityAnalysisError(
            "Incomplete architecture/capacity parameter contract"
        )

    for key, (generator, critic, total) in contract.items():
        if generator + critic != total:
            raise F4CapacityAnalysisError(
                f"Total WGAN parameters do not equal G+D for {key[0]}/{key[1]}"
            )

    for capacity_id in CAPACITY_IDS:
        film = contract[("film_cnn", capacity_id)]
        pure = contract[("pure_cnn", capacity_id)]
        if film[1] != pure[1]:
            raise F4CapacityAnalysisError(
                f"FiLM/Pure Critics must be parameter-matched at {capacity_id}"
            )
        if film[0] <= pure[0] or film[2] <= pure[2]:
            raise F4CapacityAnalysisError(
                f"FiLM must contain more parameters than Pure-CNN at {capacity_id}"
            )

    for architecture in ARCHITECTURES:
        totals = [
            contract[(architecture, capacity_id)][2] for capacity_id in CAPACITY_IDS
        ]
        if any(right <= left for left, right in zip(totals, totals[1:])):
            raise F4CapacityAnalysisError(
                f"Total WGAN parameters must increase with capacity for {architecture}"
            )


def _validated_frame(
    pair_metrics: pd.DataFrame,
    *,
    parameter_counts: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[int, int, int]]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in pair_metrics]
    if missing:
        raise F4CapacityAnalysisError(f"Missing pair-metric columns: {missing}")

    frame = pair_metrics.copy()
    if len(frame) != EXPECTED_ROWS:
        raise F4CapacityAnalysisError(
            f"Expected {EXPECTED_ROWS:,} frozen pair rows, found {len(frame):,}"
        )
    frame["architecture"] = frame["architecture"].astype(str)
    frame["capacity_id"] = frame["capacity_id"].astype(str)
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame["session_id"] = frame["session_id"].astype(str)
    frame["noise_bank_profile_sha256"] = (
        frame["noise_bank_profile_sha256"].astype(str).str.lower()
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame["target_mae"] = pd.to_numeric(frame["target_mae"], errors="coerce")

    if frame["seed"].isna().any() or any(
        float(value) != int(value) for value in frame["seed"]
    ):
        raise F4CapacityAnalysisError("Seeds must be finite integers")
    frame["seed"] = frame["seed"].astype(int)
    if (
        not frame["target_mae"].map(math.isfinite).all()
        or (frame["target_mae"] < 0).any()
    ):
        raise F4CapacityAnalysisError("target_mae must be finite and non-negative")
    if (frame["pair_id"].str.len() == 0).any() or (
        frame["session_id"].str.len() == 0
    ).any():
        raise F4CapacityAnalysisError("Pair and session identifiers must be non-empty")
    if not frame["noise_bank_profile_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise F4CapacityAnalysisError(
            "noise_bank_profile_sha256 must contain lowercase-compatible SHA-256 values"
        )

    if set(frame["architecture"]) != set(ARCHITECTURES):
        raise F4CapacityAnalysisError(f"Architectures must be exactly {ARCHITECTURES}")
    if set(frame["capacity_id"]) != set(CAPACITY_IDS):
        raise F4CapacityAnalysisError(f"Capacities must be exactly {CAPACITY_IDS}")
    if set(frame["seed"]) != set(SEEDS):
        raise F4CapacityAnalysisError(f"Seeds must be exactly {SEEDS}")

    if "fold" in frame and set(frame["fold"].astype(str)) != {FOLD_ID}:
        raise F4CapacityAnalysisError(f"Only {FOLD_ID} rows are permitted")
    if "tolerance_minutes" in frame:
        tolerance = pd.to_numeric(frame["tolerance_minutes"], errors="coerce")
        if tolerance.isna().any() or set(tolerance.astype(int)) != {5}:
            raise F4CapacityAnalysisError("Only the five-minute F4 panel is permitted")
    if "prediction_mc_samples" in frame:
        samples = pd.to_numeric(frame["prediction_mc_samples"], errors="coerce")
        if samples.isna().any() or set(samples.astype(int)) != {64}:
            raise F4CapacityAnalysisError("F4 predictions must use MC64")

    cell_keys = ["architecture", "capacity_id", "seed"]
    expected_cells = len(ARCHITECTURES) * len(CAPACITY_IDS) * len(SEEDS)
    grouped = frame.groupby(cell_keys, sort=False, dropna=False)
    if grouped.ngroups != expected_cells:
        raise F4CapacityAnalysisError(f"Expected {expected_cells} complete cells")
    for key, cell in grouped:
        if len(cell) != PAIR_COUNT or cell["pair_id"].nunique() != PAIR_COUNT:
            raise F4CapacityAnalysisError(
                f"Cell {key} must contain exactly {PAIR_COUNT} unique pairs"
            )
        if cell["session_id"].nunique() != SESSION_COUNT:
            raise F4CapacityAnalysisError(
                f"Cell {key} must contain exactly {SESSION_COUNT} sessions"
            )
        if cell["noise_bank_profile_sha256"].nunique() != 1:
            raise F4CapacityAnalysisError(f"Cell {key} has multiple MC64 noise banks")

    pair_lineage = frame[["pair_id", "session_id"]].drop_duplicates()
    if (
        len(pair_lineage) != PAIR_COUNT
        or pair_lineage["pair_id"].nunique() != PAIR_COUNT
    ):
        raise F4CapacityAnalysisError(
            "Architecture/capacity/seed cells do not share one pair/session panel"
        )
    reference_pairs = set(pair_lineage["pair_id"])
    for key, cell in grouped:
        if set(cell["pair_id"]) != reference_pairs:
            raise F4CapacityAnalysisError(f"Pair-panel drift in cell {key}")

    noise_by_seed = frame.groupby("seed")["noise_bank_profile_sha256"].nunique()
    if not noise_by_seed.eq(1).all():
        raise F4CapacityAnalysisError(
            "Architectures and capacities do not share one MC64 bank within seed"
        )

    frame_contract = _parameter_contract_from_frame(frame)
    mapping_contract = (
        _parameter_contract_from_mapping(parameter_counts)
        if parameter_counts is not None
        else {}
    )
    if frame_contract and mapping_contract and frame_contract != mapping_contract:
        raise F4CapacityAnalysisError(
            "Pair-metric parameter counts disagree with the supplied contract"
        )
    contract = frame_contract or mapping_contract
    if not contract:
        raise F4CapacityAnalysisError(
            "Parameter counts must be present in pair metrics or supplied explicitly"
        )
    _validate_parameter_contract(contract)
    return frame, contract


def summarize_pair_metrics(
    pair_metrics: pd.DataFrame,
    parameter_counts: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate frozen pair evidence and return one row per capacity.

    MAE is first averaged over the 143 pairs within each seed and then averaged
    equally over the three seeds.  Improvement is computed from the unrounded
    architecture-specific mean relative to that architecture's c32 mean.
    """

    frame, contract = _validated_frame(
        pair_metrics,
        parameter_counts=parameter_counts,
    )
    seed_means = (
        frame.groupby(["architecture", "capacity_id", "seed"], as_index=False)[
            "target_mae"
        ]
        .mean()
        .rename(columns={"target_mae": "seed_mae"})
    )
    cell_means = seed_means.groupby(["architecture", "capacity_id"], as_index=False)[
        "seed_mae"
    ].mean()
    means = {
        (str(row.architecture), str(row.capacity_id)): float(row.seed_mae)
        for row in cell_means.itertuples(index=False)
    }
    references = {
        architecture: means[(architecture, CURRENT_CAPACITY_ID)]
        for architecture in ARCHITECTURES
    }
    if any(not math.isfinite(value) or value <= 0 for value in references.values()):
        raise F4CapacityAnalysisError("c32 reference MAE must be finite and positive")

    rows: list[dict[str, Any]] = []
    for order, capacity_id in enumerate(CAPACITY_IDS):
        row: dict[str, Any] = {
            "capacity_order": order,
            "capacity_id": capacity_id,
            "is_current_capacity": capacity_id == CURRENT_CAPACITY_ID,
        }
        for architecture, prefix in (
            ("film_cnn", "film_cnn"),
            ("pure_cnn", "pure_cnn"),
        ):
            generator, critic, total = contract[(architecture, capacity_id)]
            observed_mae = means[(architecture, capacity_id)]
            improvement = 100.0 * (1.0 - observed_mae / references[architecture])
            if capacity_id == CURRENT_CAPACITY_ID:
                improvement = 0.0
            row.update(
                {
                    f"{prefix}_generator_parameters": generator,
                    f"{prefix}_critic_parameters": critic,
                    f"{prefix}_total_wgan_parameters": total,
                    f"{prefix}_observed_mae": observed_mae,
                    f"{prefix}_improvement_vs_c32_pct": improvement,
                    f"{prefix}_seed_count": len(SEEDS),
                    f"{prefix}_pair_count": PAIR_COUNT,
                    f"{prefix}_seed_pair_rows": len(SEEDS) * PAIR_COUNT,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _format_improvement(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded == 0:
        return "0.0000"
    return f"{rounded:+.4f}"


def render_latex_table(summary: pd.DataFrame) -> str:
    """Render the complete one-panel capacity table without custom macros."""

    if list(summary["capacity_id"].astype(str)) != list(CAPACITY_IDS):
        raise F4CapacityAnalysisError("Summary capacity order drifted")
    if summary["is_current_capacity"].astype(bool).tolist() != [
        capacity_id == CURRENT_CAPACITY_ID for capacity_id in CAPACITY_IDS
    ]:
        raise F4CapacityAnalysisError("Summary current-capacity marker drifted")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrr@{\hspace{8pt}}rrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{FiLM-CNN} & \multicolumn{3}{c}{Pure-CNN} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        (
            r"Capacity & \shortstack{Total WGAN\\parameters}"
            r" & \shortstack{Observed\\MAE}"
            r" & \shortstack{Improvement vs.\\FiLM c32 (\%)}"
            r" & \shortstack{Total WGAN\\parameters}"
            r" & \shortstack{Observed\\MAE}"
            r" & \shortstack{Improvement vs.\\Pure c32 (\%)} \\"
        ),
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        capacity_id = str(row.capacity_id)
        label = (
            r"\shortstack[l]{\textbf{c32 (current}\\"
            r"\textbf{capacity; reference)}}"
            if capacity_id == CURRENT_CAPACITY_ID
            else capacity_id
        )
        lines.append(
            " & ".join(
                (
                    label,
                    f"{int(row.film_cnn_total_wgan_parameters):,}",
                    f"{float(row.film_cnn_observed_mae):.10f}",
                    _format_improvement(row.film_cnn_improvement_vs_c32_pct),
                    f"{int(row.pure_cnn_total_wgan_parameters):,}",
                    f"{float(row.pure_cnn_observed_mae):.10f}",
                    _format_improvement(row.pure_cnn_improvement_vs_c32_pct),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{F4 (2023Q4) model-capacity robustness test for the "
                r"directly trained FiLM-CNN and Pure-CNN systems. Each MAE is "
                r"the equal-weight mean of three seed-level means on the common "
                r"143-pair, 45-session five-minute test panel (429 seed--pair "
                r"rows per architecture--capacity cell). Training uses pairs "
                r"before 2023-07-01, checkpoint selection uses 2023Q3, and "
                r"2023Q4 is the held-out test period rather than training data. Total "
                r"WGAN parameters equal Generator plus Critic parameters; a common "
                r"cXX label matches spatial-network and Critic widths, not total "
                r"parameters across architectures. Within each architecture, "
                r"improvement is computed from unrounded "
                r"means as $100(1-\mathrm{MAE}_{c}/\mathrm{MAE}_{c32})$; positive "
                r"values indicate lower error than the current c32 capacity. "
                r"FiLM-CNN versus Pure-CNN comparisons should use the MAE columns, "
                r"not compare the two architecture-specific improvement columns.}"
            ),
            rf"\label{{{TABLE_LABEL}}}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines)


def _records_for_json(summary: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in summary.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                record[key] = value
            elif isinstance(value, int):
                record[key] = value
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise F4CapacityAnalysisError(f"Non-finite summary value: {key}")
                record[key] = value
            else:
                record[key] = value.item() if hasattr(value, "item") else value
        records.append(record)
    return records


def postprocess_experiment(
    output_root: Path,
    pair_metrics_path: Path,
    parameter_counts: Mapping[str, Any] | None = None,
) -> Mapping[str, Path]:
    """Validate frozen F4 evidence and write deterministic analysis artifacts."""

    root = Path(output_root).resolve()
    source = Path(pair_metrics_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    input_sha256 = _sha256_file(source)
    pair_metrics = pd.read_csv(source, low_memory=False)
    validated, _ = _validated_frame(
        pair_metrics,
        parameter_counts=parameter_counts,
    )
    summary = summarize_pair_metrics(validated, parameter_counts=parameter_counts)

    capacity_order = {
        capacity_id: index for index, capacity_id in enumerate(CAPACITY_IDS)
    }
    architecture_order = {
        architecture: index for index, architecture in enumerate(ARCHITECTURES)
    }
    frozen_pairs = validated.assign(
        _architecture_order=validated["architecture"].map(architecture_order),
        _capacity_order=validated["capacity_id"].map(capacity_order),
    ).sort_values(
        ["_architecture_order", "_capacity_order", "seed", "pair_id"],
        kind="stable",
    )
    frozen_pairs = frozen_pairs.drop(
        columns=["_architecture_order", "_capacity_order"]
    ).reset_index(drop=True)

    analysis_dir = root / "analysis"
    pair_path = _atomic_write_gzip_csv(
        analysis_dir / PAIR_METRICS_NAME,
        frozen_pairs,
    )
    summary_csv_path = _atomic_write_text(
        analysis_dir / SUMMARY_CSV_NAME,
        summary.to_csv(index=False, float_format="%.17g"),
    )
    latex_path = _atomic_write_text(
        analysis_dir / TABLE_TEX_NAME,
        render_latex_table(summary),
    )
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "fold": FOLD_ID,
        "architectures": list(ARCHITECTURES),
        "capacity_ids": list(CAPACITY_IDS),
        "current_capacity_id": CURRENT_CAPACITY_ID,
        "seeds": list(SEEDS),
        "pair_count": PAIR_COUNT,
        "session_count": SESSION_COUNT,
        "seed_pair_rows_per_architecture_capacity": len(SEEDS) * PAIR_COUNT,
        "pair_metric_rows": EXPECTED_ROWS,
        "aggregation": "equal_seed_mean_of_within_seed_pair_mae_v1",
        "improvement_formula": "100*(1-mae_capacity/mae_c32)_within_architecture",
        "table_label": TABLE_LABEL,
        "input_pair_metrics_path": str(source),
        "input_pair_metrics_sha256": input_sha256,
        "artifacts": {
            PAIR_METRICS_NAME: _sha256_file(pair_path),
            SUMMARY_CSV_NAME: _sha256_file(summary_csv_path),
            TABLE_TEX_NAME: _sha256_file(latex_path),
        },
        "rows": _records_for_json(summary),
    }
    summary_json_path = _atomic_write_text(
        analysis_dir / SUMMARY_JSON_NAME,
        _canonical_json(summary_payload),
    )
    return {
        "pair_metrics": pair_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "latex_table": latex_path,
    }


__all__ = [
    "ANALYSIS_KIND",
    "ARCHITECTURES",
    "CAPACITY_IDS",
    "CURRENT_CAPACITY_ID",
    "EXPECTED_ROWS",
    "F4CapacityAnalysisError",
    "PAIR_COUNT",
    "SEEDS",
    "TABLE_LABEL",
    "postprocess_experiment",
    "render_latex_table",
    "summarize_pair_metrics",
]
