"""Bind the current thesis's three main MAE tables to original-sample results.

This scoped companion leaves the historical full-chapter bindings unchanged.
It reads frozen v2 summaries; it does not run models, predictions or bootstrap
resampling. Verify its JSON with ``verify_chapter3_bootstrap_bindings``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.rq123 import build_chapter3_bootstrap_bindings as base
except ModuleNotFoundError:
    import build_chapter3_bootstrap_bindings as base


DEFAULT_TEX = base.ROOT.parent / "PhdThesis/Chapter3/chapter3.tex"
DEFAULT_OUTPUT = base.ROOT / "docs/chapter3_observed_mae_bindings.json"
TABLES = (
    "tab:ch3:direct_model_results",
    "tab:ch3:rq2_direct_results",
    "tab:ch3:rq3_rolling_results",
)


def build_payload(tex_path: Path, values_path: Path) -> dict:
    tex = tex_path.read_text(encoding="utf-8")
    values = json.loads(values_path.read_text(encoding="utf-8"))
    if values.get("kind") != "chapter3_shared_market_panel_bootstrap_v2":
        raise ValueError("Expected the frozen corrected v2 result archive")
    builder = base.BindingBuilder(tex, values)
    for label, rows in (
        (TABLES[0], (("FiLM-CNN, matched LP", "lp_matched"),
                     ("Pure-CNN, no text", "no_text"))),
        (TABLES[1], (("BoW", "bow"), ("Text embedding (matched LP)", "lp_matched"),
                     ("Sentiment", "sentiment"))),
    ):
        if "Panel A: MAE" not in base._panel(builder.tables[label], "A"):
            raise ValueError(f"{label} must identify the MAE panel")
        for marker, condition in (*rows, ("Persistence", "persistence")):
            mae_atoms = []
            improvement_atoms = []
            for job, fold in base.FOLDS:
                mae_atoms.append((base.arm(job, condition, "observed_mean_mae"), base.fixed(9)))
                if condition != "persistence":
                    cid = base.direct_contrast(job, fold, condition)
                    mae_atoms.extend(base._star_and_stat(values, job, cid))
                source = base.ratio(job, condition, "persistence", "observed_improvement_percent")
                improvement_atoms.append((source, base._math_if_negative(values, source, 4)))
            builder.row(f"{label}_{condition}_mae", label, marker, mae_atoms, panel="A")
            builder.row(f"{label}_{condition}_improvement", label, marker,
                        improvement_atoms, panel="B")

    label = TABLES[2]
    if "Four-fold MAE" not in builder.tables[label]:
        raise ValueError("The common-parent table must identify four-fold MAEs")
    job = "rq3_full_absolute_mae"
    rows = (
        ("FiLM, matched LP", "film_lp_matched"),
        ("FiLM, zero text", "film_zero_text"),
        ("FiLM, shuffled LP", "film_lp_shuffle"),
        ("FiLM, BoW", "film_bow"),
        ("FiLM, sentiment", "film_sentiment"),
        ("Pure-CNN continuation", "pure_cnn_continue_no_text"),
        ("Pure-CNN parent", "pure_cnn_parent"),
        ("Persistence", "persistence"),
    )
    for marker, condition in rows:
        builder.row(
            f"rq3_{condition}_observed_mae", label, marker,
            [
                (base.arm(job, condition, "observed_mean_mae"), base.fixed(10)),
                (base.ratio(job, condition, "persistence", "observed_improvement_percent"),
                 base.fixed(4)),
                (base.ratio(job, condition, "pure_cnn_continue_no_text", "observed_mae_ratio"),
                 base.fixed(6)),
            ],
        )

    def prose(binding_id: str, start: str, end: str, atoms: list) -> None:
        builder.prose(binding_id,
                      {"kind": "unique_text", "value": start, "window_lines": 30},
                      start, end, atoms)

    atoms = []
    for condition in ("lp_matched", "no_text"):
        for index, (fold_job, _) in enumerate(base.FOLDS):
            source = base.ratio(fold_job, condition, "persistence", "observed_improvement_percent")
            if condition == "lp_matched" and index == 2:
                atoms.append((None, {"operator": "negate", "operands": [source]}, base.fixed(4)))
            else:
                atoms.append((source, base.fixed(4)))
    prose("rq1_observed_quarterly_prose",
          "Relative to persistence, FiLM-CNN reduces its MAE by ",
          "performance is therefore period-dependent rather than uniformly better than\npersistence.",
          atoms)
    prose("rq1_observed_overall_prose",
          "Giving the four folds equal weight, the overall",
          "Pure-CNN, compared with persistence at 0.0020292830.",
          [(base.arm("direct_overall", condition, "observed_mean_mae"), base.fixed(10))
           for condition in ("lp_matched", "no_text", "persistence")])
    prose("rq3_observed_leaders_prose",
          "Matched LP has the lowest MAE,",
          "gain is attributable to textual information.",
          [(base.arm(job, condition, "observed_mean_mae"), base.fixed(10))
           for condition in ("film_lp_matched", "film_zero_text", "pure_cnn_continue_no_text")]
          + [(base.ratio(job, "film_lp_matched", "pure_cnn_continue_no_text",
                        "observed_mae_ratio"), base.fixed(6)),
             (base.ratio(job, "film_lp_matched", "pure_cnn_continue_no_text",
                         "observed_improvement_percent"), base.fixed(4))])
    atoms = []
    for condition in ("film_lp_matched", "film_zero_text", "film_lp_shuffle"):
        source = base.ratio(job, condition, "pure_cnn_continue_no_text", "observed_improvement_percent")
        atoms.append((None, {"operator": "negate", "operands": [source]}, base.fixed(4))
                     if condition == "film_lp_shuffle" else (source, base.fixed(4)))
    prose("rq3_observed_relative_prose",
          "Relative to the Pure-CNN continuation, matched LP and zero text reduce the",
          "MAE than persistence.", atoms)
    prose("rq3_matched_same_checkpoint_mae",
          "Both intervention estimates are positive. The matched-input",
          "under wrong input.",
          [(base.arm("rq3_intervention", condition, "observed_mean_mae"), base.fixed(10))
           for condition in ("matched_input", "zero_input", "wrong_input")])

    jobs = sorted({
        source["job_id"]
        for binding in builder.bindings
        for item in binding["values"]
        for source in ([item["source"]] if "source" in item
                       else item["source_expression"]["operands"])
    })
    return {
        "schema_version": 1,
        "kind": "chapter3_bootstrap_tex_bindings",
        "status": "complete",
        "tex_path": os.path.relpath(tex_path.resolve(), base.ROOT),
        "values_path": os.path.relpath(values_path.resolve(), base.ROOT),
        "scope": {
            "included": ["rq1_panels_a_b", "rq2_panels_a_b", "rq3_absolute_mae",
                         "associated_mae_prose"],
            "excluded": ["other_result_tables", "other_prose", "rq4"],
            "excluded_anchor_range": {
                "start": "\\label{subsec:ch3:rq4_conditional_text_value}",
                "end": "\\subsection*{Robustness Tests}",
            },
        },
        "coverage": {
            "required_result_tables": list(TABLES),
            "bound_jobs": jobs,
            "binding_groups": len(builder.bindings),
            "numerical_bindings": sum(len(b["values"]) for b in builder.bindings),
            "interpretation": (
                "Scoped verification of original-sample MAEs and their arithmetic "
                "comparisons in the three main result tables and associated prose. "
                "Existing stars and standard-error ratios in their MAE panels are "
                "bound to the unchanged paired contrasts. Other results and the "
                "historical full-chapter mirror are outside this verification."
            ),
        },
        "bindings": builder.bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--values", type=Path, default=base.DEFAULT_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.tex, args.values)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != serialized:
            raise SystemExit("Original-sample MAE bindings are stale; regenerate with --write")
    else:
        args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": "current" if args.check else "written",
                      "path": str(args.output), **payload["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
