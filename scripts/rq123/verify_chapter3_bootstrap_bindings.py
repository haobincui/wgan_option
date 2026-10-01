"""Read-only checks binding Chapter 3's displayed values to the v2 archive.

This is separate from the frozen numerical implementation: editing prose does
not change the implementation hashes recorded when the draws were generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINDINGS = ROOT / "docs/chapter3_bootstrap_bindings.json"
LEGACY_CAPACITY_TABLE_LABEL = (
    "tab:ch3:legacy_full_wgan_capacity_vs_fixed_pure_cnn"
)
F4_CAPACITY_TABLE_LABEL = "tab:ch3:f4_film_pure_capacity_robustness"
F4_CAPACITY_ANALYSIS_KIND = "f4_film_pure_capacity_3seed_analysis_v1"
F4_CAPACITY_SUMMARY_JSON = "f4_capacity_summary.json"
F4_CAPACITY_SUMMARY_CSV = "f4_capacity_summary.csv"
F4_CAPACITY_PAIR_METRICS = "f4_pair_metrics.csv.gz"
F4_CAPACITY_TABLE_TEX = "f4_capacity_table.tex"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _table_blocks(tex: str) -> list[str]:
    return re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL)


def _unique_table(tables: list[str], label: str) -> str:
    matching = [table for table in tables if "\\label{" + label + "}" in table]
    if len(matching) != 1:
        raise ValueError(f"Missing or duplicate externally tracked table: {label}")
    return matching[0]


def _verify_f4_summary_contract(summary: dict) -> None:
    expected = {
        "schema_version": 1,
        "kind": F4_CAPACITY_ANALYSIS_KIND,
        "fold": "f4_2023q4",
        "architectures": ["film_cnn", "pure_cnn"],
        "capacity_ids": ["c08", "c12", "c16", "c24", "c32", "c48"],
        "current_capacity_id": "c32",
        "seeds": [42, 202, 404],
        "pair_count": 143,
        "session_count": 45,
        "seed_pair_rows_per_architecture_capacity": 429,
        "pair_metric_rows": 5148,
        "aggregation": "equal_seed_mean_of_within_seed_pair_mae_v1",
        "improvement_formula": (
            "100*(1-mae_capacity/mae_c32)_within_architecture"
        ),
        "table_label": F4_CAPACITY_TABLE_LABEL,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"F4 capacity summary contract drift at {field}")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("F4 capacity summary must contain six rows")
    if [row.get("capacity_id") for row in rows if isinstance(row, dict)] != expected[
        "capacity_ids"
    ]:
        raise ValueError("F4 capacity summary capacity rows drifted")


def _verified_source_file(
    source: dict, role: str, expected_name: str | None = None
) -> Path:
    path_key = f"{role}_path"
    hash_key = f"{role}_sha256"
    if path_key not in source or hash_key not in source:
        raise ValueError(f"F4 capacity source is missing {role}")
    path = _resolve(source[path_key]).resolve()
    if (expected_name is not None and path.name != expected_name) or not path.is_file():
        raise ValueError(f"Missing or misnamed F4 capacity source: {role}")
    digest = source[hash_key]
    if not _valid_sha256(digest) or _sha256_path(path) != digest:
        raise ValueError(f"F4 capacity source hash drift: {role}")
    return path


def _verify_f4_capacity_source(record: dict, chapter_table: str) -> None:
    source = record.get("source")
    if not isinstance(source, dict):
        raise ValueError("Active F4 capacity table requires source provenance")
    summary_path = _verified_source_file(
        source, "summary_json", F4_CAPACITY_SUMMARY_JSON
    )
    pair_path = _verified_source_file(
        source, "pair_metrics", F4_CAPACITY_PAIR_METRICS
    )
    summary_csv_path = _verified_source_file(
        source, "summary_csv", F4_CAPACITY_SUMMARY_CSV
    )
    latex_path = _verified_source_file(source, "latex_table", F4_CAPACITY_TABLE_TEX)
    input_path = _verified_source_file(source, "input_pair_metrics")
    if {pair_path.parent, summary_csv_path.parent, latex_path.parent} != {
        summary_path.parent
    }:
        raise ValueError("F4 capacity analysis artifacts do not share one directory")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid F4 capacity summary JSON") from exc
    if not isinstance(summary, dict):
        raise ValueError("F4 capacity summary JSON must be an object")
    _verify_f4_summary_contract(summary)

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("F4 capacity summary artifact registry is missing")
    artifact_contract = {
        F4_CAPACITY_PAIR_METRICS: source["pair_metrics_sha256"],
        F4_CAPACITY_SUMMARY_CSV: source["summary_csv_sha256"],
        F4_CAPACITY_TABLE_TEX: source["latex_table_sha256"],
    }
    for name, digest in artifact_contract.items():
        if artifacts.get(name) != digest:
            raise ValueError(f"F4 capacity nested artifact hash drift: {name}")

    recorded_input_path = _resolve(summary.get("input_pair_metrics_path", "")).resolve()
    if recorded_input_path != input_path:
        raise ValueError("F4 capacity input pair-metric path drift")
    if summary.get("input_pair_metrics_sha256") != source[
        "input_pair_metrics_sha256"
    ]:
        raise ValueError("F4 capacity input pair-metric hash linkage drift")

    generated_tables = _table_blocks(latex_path.read_text(encoding="utf-8"))
    if len(generated_tables) != 1 or (
        "\\label{" + F4_CAPACITY_TABLE_LABEL + "}" not in generated_tables[0]
    ):
        raise ValueError("Generated F4 capacity LaTeX table label drift")
    if generated_tables[0] != chapter_table:
        raise ValueError(
            "Chapter F4 capacity table is not byte-identical to its generated source"
        )


def _verify_external_tables(payload: dict, tex: str, tables: list[str]) -> tuple[int, int]:
    records = payload.get("external_table_changes") or []
    if not records:
        return 0, 0
    if not isinstance(records, list):
        raise ValueError("external_table_changes must be a list")
    labels = [record.get("label") for record in records if isinstance(record, dict)]
    if len(labels) != len(records) or len(labels) != len(set(labels)):
        raise ValueError("Invalid or duplicate external table record")

    verified = 0
    superseded = 0
    for record in records:
        label = record["label"]
        table = _unique_table(tables, label)
        digest = record.get("current_sha256")
        if not _valid_sha256(digest) or hashlib.sha256(
            table.encode("utf-8")
        ).hexdigest() != digest:
            raise ValueError(f"Externally tracked table changed: {label}")
        verified += 1

        if label == F4_CAPACITY_TABLE_LABEL:
            if "\\label{" + LEGACY_CAPACITY_TABLE_LABEL + "}" in tex:
                raise ValueError("Superseded legacy capacity label was reintroduced")
            old = record.get("supersedes")
            if not isinstance(old, dict) or old.get("label") != (
                LEGACY_CAPACITY_TABLE_LABEL
            ) or old.get("status") != "superseded":
                raise ValueError("Active F4 capacity record lacks superseded provenance")
            if not _valid_sha256(old.get("baseline_sha256")) or not _valid_sha256(
                old.get("last_active_sha256")
            ):
                raise ValueError("Superseded capacity-table hashes are invalid")
            _verify_f4_capacity_source(record, table)
            superseded += 1

    if F4_CAPACITY_TABLE_LABEL in labels and LEGACY_CAPACITY_TABLE_LABEL in labels:
        raise ValueError("Legacy capacity table cannot remain an active external record")
    return verified, superseded


def _source(values: dict, source: dict):
    if source["collection"] not in {"arms", "contrasts", "ratios"}:
        raise ValueError("Unknown source collection")
    return values["jobs"][source["job_id"]][source["collection"]][
        source["item_id"]
    ][source["field"]]


def _value(values: dict, binding: dict):
    if ("source" in binding) == ("source_expression" in binding):
        raise ValueError("Exactly one source or source_expression is required")
    if "source" in binding:
        return _source(values, binding["source"])
    expression = binding["source_expression"]
    operands = [_source(values, item) for item in expression["operands"]]
    if expression["operator"] == "negate":
        if len(operands) != 1:
            raise ValueError("Negation requires one source operand")
        return -operands[0]
    if len(operands) != 2:
        raise ValueError("Derived expressions require two source operands")
    left, right = operands
    if expression["operator"] == "difference":
        return left - right
    if expression["operator"] == "ratio":
        return left / right
    if expression["operator"] == "one_minus_ratio_percent":
        return 100.0 * (1.0 - left / right)
    raise ValueError("Unknown source expression")


def render(value, spec: dict) -> str:
    """Render explicit Python decimal or LaTeX scientific display contracts."""
    style = spec["style"]
    if style == "stars":
        if value not in {"", "*", "**", "***"}:
            raise ValueError("Invalid significance stars")
        return value
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Cannot bind a non-finite display value")
    digits = int(spec.get("digits", 4))
    plus = "+" if spec.get("explicit_plus", False) else ""
    if style == "fixed":
        result = format(value, f"{plus}.{digits}f")
    elif style == "significant":
        result = format(value, f"{plus}.{digits}g")
    elif style == "latex_scientific":
        if digits < 1:
            raise ValueError("Scientific notation needs >=1 significant digit")
        mantissa, exponent = format(value, f"{plus}.{digits - 1}e").split("e")
        result = f"{mantissa}\\times10^{{{int(exponent)}}}"
    else:
        raise ValueError(f"Unknown format style: {style}")
    return f"${result}$" if spec.get("math_mode", False) else result


def _unique_position(text: str, literal: str) -> int:
    if not literal or text.count(literal) != 1:
        raise ValueError(f"Anchor is not unique: {literal!r}")
    return text.index(literal)


def verify_bindings(path: Path = DEFAULT_BINDINGS) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        "chapter3_bootstrap_tex_bindings"
    ):
        raise ValueError("Unsupported Chapter 3 bindings schema")
    bindings = payload.get("bindings", [])
    if not bindings:
        raise ValueError("Bindings are incomplete: no numerical bindings")
    if payload.get("status") != "complete":
        raise ValueError("Bindings are incomplete: final synchronization is not marked complete")
    tex_path = _resolve(payload["tex_path"])
    values_path = _resolve(payload["values_path"])
    tex = tex_path.read_text(encoding="utf-8")
    passthrough = payload.get("passthrough_table_hashes", [])
    tables = _table_blocks(tex)
    for item in passthrough:
        matching = [table for table in tables if "\\label{" + item["label"] + "}" in table]
        if len(matching) != 1:
            raise ValueError(f"Missing or duplicate descriptive table: {item['label']}")
        digest = hashlib.sha256(matching[0].encode("utf-8")).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Excluded/descriptive table changed: {item['label']}")
    verified_external, superseded_external = _verify_external_tables(
        payload, tex, tables
    )
    values = json.loads(values_path.read_text(encoding="utf-8"))
    if values.get("kind") != "chapter3_shared_market_panel_bootstrap_v2":
        raise ValueError("Bindings must reference corrected v2 results")
    excluded_spec = payload["scope"]["excluded_anchor_range"]
    excluded_start = _unique_position(tex, excluded_spec["start"])
    excluded_end = _unique_position(tex, excluded_spec["end"])
    if excluded_start >= excluded_end:
        raise ValueError("Invalid excluded RQ4 anchor range")
    ids = [binding["binding_id"] for binding in bindings]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate binding identifier")
    atom_count = 0
    jobs = set()
    for binding in bindings:
        anchor = binding["anchor"]
        if anchor["kind"] == "latex_label":
            marker = "\\label{" + anchor["value"] + "}"
        elif anchor["kind"] == "unique_text":
            marker = anchor["value"]
        else:
            raise ValueError("Unknown anchor kind")
        position = _unique_position(tex, marker)
        if excluded_start <= position < excluded_end:
            raise ValueError("An RQ4 result cannot be a corrected v2 binding")
        line = tex[:position].count("\n")
        lines = tex.splitlines(keepends=True)
        radius = int(anchor["window_lines"])
        if radius < 0:
            raise ValueError("Invalid anchor window")
        first_line = max(0, line - radius)
        window = "".join(lines[first_line : line + radius + 1])
        snippet = binding["expected_substring"]
        if not snippet or window.count(snippet) != 1:
            raise ValueError(f"Chapter text drift at {binding['binding_id']}")
        snippet_position = sum(map(len, lines[:first_line])) + window.index(snippet)
        if snippet_position < excluded_end and snippet_position + len(snippet) > excluded_start:
            raise ValueError("A binding cannot overlap the excluded RQ4 results")
        if not binding["values"]:
            raise ValueError("A numerical binding must have at least one value")
        for item in binding["values"]:
            literal = item["literal"]
            if literal and literal not in snippet:
                raise ValueError(f"Literal absent from {binding['binding_id']}: {literal}")
            expected = render(_value(values, item), item["format"])
            if expected != literal:
                raise ValueError(
                    f"Numerical drift at {binding['binding_id']}: "
                    f"{literal!r} != expected {expected!r}"
                )
            sources = ([item["source"]] if "source" in item else
                       item["source_expression"]["operands"])
            jobs.update(source["job_id"] for source in sources)
            atom_count += 1
    return {
        "passed": True,
        "binding_groups": len(bindings),
        "numerical_bindings": atom_count,
        "bound_jobs": sorted(jobs),
        "rq4_excluded": True,
        "unchanged_passthrough_tables": len(passthrough),
        "verified_external_tables": verified_external,
        "superseded_external_tables": superseded_external,
        "tex_sha256": hashlib.sha256(tex_path.read_bytes()).hexdigest(),
        "values_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
        "bindings_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    args = parser.parse_args()
    print(json.dumps(verify_bindings(_resolve(args.bindings)), indent=2))


if __name__ == "__main__":
    main()
