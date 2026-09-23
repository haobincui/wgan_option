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


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL)
    for item in passthrough:
        matching = [table for table in tables if "\\label{" + item["label"] + "}" in table]
        if len(matching) != 1:
            raise ValueError(f"Missing or duplicate descriptive table: {item['label']}")
        digest = hashlib.sha256(matching[0].encode("utf-8")).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Excluded/descriptive table changed: {item['label']}")
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
