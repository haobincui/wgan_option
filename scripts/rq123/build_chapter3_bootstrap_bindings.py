"""Build explicit Chapter 3 TeX-to-v2 numerical bindings.

This module is intentionally analysis-only.  It never searches the v2 archive
for a number that happens to match the manuscript.  Instead, every table row
and prose fragment below names its formal ``job / collection / item / field``
source first.  The displayed literal is then rendered from that named source
and required to occur in the explicitly selected TeX fragment.

The resulting JSON is checked independently by
``verify_chapter3_bootstrap_bindings.py``.  RQ4 and the two descriptive tables
whose source is outside the v2 bootstrap archive are deliberately excluded.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

try:
    from scripts.rq123.verify_chapter3_bootstrap_bindings import render
except ModuleNotFoundError:  # direct ``python scripts/rq123/...py`` invocation
    from verify_chapter3_bootstrap_bindings import render


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEX = ROOT / "docs/chapter3.tex"
DEFAULT_VALUES = (
    ROOT
    / "outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2"
    / "chapter3_values.json"
)
DEFAULT_OUTPUT = ROOT / "docs/chapter3_bootstrap_bindings.json"

LEGACY_CAPACITY_TABLE_LABEL = (
    "tab:ch3:legacy_full_wgan_capacity_vs_fixed_pure_cnn"
)
F4_CAPACITY_TABLE_LABEL = "tab:ch3:f4_film_pure_capacity_robustness"
F4_CAPACITY_ANALYSIS_KIND = "f4_film_pure_capacity_3seed_analysis_v1"
DEFAULT_F4_CAPACITY_ANALYSIS_DIR = (
    ROOT
    / "outputs/experiments"
    / "rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1"
    / "analysis"
)
F4_CAPACITY_SUMMARY_JSON = "f4_capacity_summary.json"
F4_CAPACITY_SUMMARY_CSV = "f4_capacity_summary.csv"
F4_CAPACITY_PAIR_METRICS = "f4_pair_metrics.csv.gz"
F4_CAPACITY_TABLE_TEX = "f4_capacity_table.tex"

PASSTHROUGH_LABELS = (
    "tab:ch3:baseline_training_diagnostics",
    "tab:ch3:rq2_training_diagnostics",
    "tab:ch3:rq3_training_diagnostics",
    "tab:ch3:rq4_conditional_results",
    "tab:ch3:rq4_fold_pooled_combined",
    "tab:ch3:rq4_fold_pooled_oos",
)

REQUIRED_RESULT_TABLES = (
    "tab:ch3:direct_model_results",
    "tab:ch3:rq2_direct_results",
    "tab:ch3:rq3_rolling_results",
    "tab:ch3:rq3_validation_trajectory",
    "tab:ch3:rq3_branch_inference",
    "tab:ch3:rq3_intervention_inference",
    "tab:ch3:generator_architecture_robustness",
    "tab:ch3:generator_architecture_secondary",
    "tab:ch3:alignment_window_common_panel",
    "tab:ch3:alignment_window_within_model",
    "tab:ch3:alignment_window_coverage",
)

REQUIRED_JOBS = (
    "direct_f1_2023q1",
    "direct_f2_2023q2",
    "direct_f3_2023q3",
    "direct_f4_2023q4",
    "direct_overall",
    "rq3_full_absolute_mae",
    "rq3_branch",
    "rq3_intervention",
    "rq3_validation_epoch0_epoch30_best",
    "generator_architecture_robustness",
    "alignment_common_5m_panel",
    "alignment_own_panel_10m",
    "alignment_own_panel_15m",
    "alignment_own_panel_20m",
    "alignment_own_panel_30m",
)

FOLDS = (
    ("direct_f1_2023q1", "f1_2023q1"),
    ("direct_f2_2023q2", "f2_2023q2"),
    ("direct_f3_2023q3", "f3_2023q3"),
    ("direct_f4_2023q4", "f4_2023q4"),
)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _source(job_id: str, collection: str, item_id: str, field: str) -> dict[str, str]:
    return {
        "job_id": job_id,
        "collection": collection,
        "item_id": item_id,
        "field": field,
    }


def arm(job_id: str, condition: str, field: str) -> dict[str, str]:
    return _source(job_id, "arms", condition, field)


def contrast(job_id: str, contrast_id: str, field: str) -> dict[str, str]:
    return _source(job_id, "contrasts", contrast_id, field)


def ratio(job_id: str, focal: str, reference: str, field: str) -> dict[str, str]:
    return _source(job_id, "ratios", f"{focal}__over__{reference}", field)


def fixed(digits: int, *, math_mode: bool = False, explicit_plus: bool = False) -> dict:
    return {
        "style": "fixed",
        "digits": digits,
        **({"math_mode": True} if math_mode else {}),
        **({"explicit_plus": True} if explicit_plus else {}),
    }


def scientific(
    digits: int, *, math_mode: bool = False, explicit_plus: bool = False
) -> dict:
    return {
        "style": "latex_scientific",
        "digits": digits,
        **({"math_mode": True} if math_mode else {}),
        **({"explicit_plus": True} if explicit_plus else {}),
    }


STARS = {"style": "stars"}


def _resolve(values: dict, source: dict) -> Any:
    try:
        return values["jobs"][source["job_id"]][source["collection"]][
            source["item_id"]
        ][source["field"]]
    except KeyError as exc:
        raise ValueError(f"Missing formal source: {source}") from exc


def _expression_value(values: dict, expression: dict) -> float:
    operands = [_resolve(values, item) for item in expression["operands"]]
    operator = expression["operator"]
    if operator == "difference" and len(operands) == 2:
        return operands[0] - operands[1]
    if operator == "ratio" and len(operands) == 2:
        return operands[0] / operands[1]
    if operator == "one_minus_ratio_percent" and len(operands) == 2:
        return 100.0 * (1.0 - operands[0] / operands[1])
    if operator == "negate" and len(operands) == 1:
        return -operands[0]
    raise ValueError(f"Unsupported source expression in builder: {expression}")


def _atom(
    values: dict,
    source: dict | None,
    format_spec: dict,
    *,
    expression: dict | None = None,
) -> dict:
    if (source is None) == (expression is None):
        raise ValueError("Exactly one source or expression is required")
    value = _resolve(values, source) if source is not None else _expression_value(values, expression)
    item = {"literal": render(value, format_spec), "format": format_spec}
    if source is not None:
        item["source"] = source
    else:
        item["source_expression"] = expression
    return item


def direct_contrast(job_id: str, fold: str, condition: str) -> str:
    return f"direct_{fold}_{condition}_vs_persistence"


def rq1_contrast(fold: str) -> str:
    return f"rq1_{fold}_lp_matched_vs_no_text"


def rq2_contrast(fold: str, reference: str) -> str:
    return f"rq2_{fold}_lp_matched_vs_{reference}"


def _tables(tex: str) -> dict[str, str]:
    blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.DOTALL)
    result: dict[str, str] = {}
    for block in blocks:
        for label in re.findall(r"\\label\{([^}]+)\}", block):
            if label in result:
                raise ValueError(f"Duplicate table label: {label}")
            result[label] = block
    return result


def _panel(table: str, panel: str | None) -> str:
    if panel is None:
        return table
    marker = f"Panel {panel}:"
    starts = [match.start() for match in re.finditer(re.escape(marker), table)]
    if len(starts) != 1:
        raise ValueError(f"Expected one {marker!r}, found {len(starts)}")
    start = starts[0]
    next_panel = re.search(r"Panel [A-Z]:", table[start + len(marker) :])
    end = len(table) if next_panel is None else start + len(marker) + next_panel.start()
    return table[start:end]


def _row(table: str, row_marker: str, panel: str | None = None) -> str:
    section = _panel(table, panel)
    pattern = re.compile(
        r"(?m)^" + re.escape(row_marker) + r"(?:[ \t]|\n).*?\\\\[ \t]*(?:\n|$)",
        re.DOTALL,
    )
    matches = [match.group(0).rstrip("\n") for match in pattern.finditer(section)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row {row_marker!r} in panel {panel!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


class BindingBuilder:
    def __init__(self, tex: str, values: dict):
        self.tex = tex
        self.values = values
        self.tables = _tables(tex)
        self.bindings: list[dict] = []
        self._ids: set[str] = set()

    def _append(
        self,
        binding_id: str,
        anchor: dict,
        fragment: str,
        atom_specs: Iterable[tuple[dict, dict] | tuple[None, dict, dict]],
    ) -> None:
        if binding_id in self._ids:
            raise ValueError(f"Duplicate binding id: {binding_id}")
        self._ids.add(binding_id)
        atoms: list[dict] = []
        for spec in atom_specs:
            if len(spec) == 2:
                source, format_spec = spec
                atoms.append(_atom(self.values, source, format_spec))
            elif len(spec) == 3 and spec[0] is None:
                _, expression, format_spec = spec
                atoms.append(
                    _atom(self.values, None, format_spec, expression=expression)
                )
            else:
                raise ValueError(f"Invalid atom specification: {spec}")
        if not atoms:
            raise ValueError(f"Binding {binding_id} has no atoms")
        missing = [item["literal"] for item in atoms if item["literal"] and item["literal"] not in fragment]
        if missing:
            raise ValueError(
                f"Formal literal(s) absent from explicit fragment {binding_id}: {missing}; "
                f"fragment={fragment!r}"
            )
        self.bindings.append(
            {
                "binding_id": binding_id,
                "anchor": anchor,
                "expected_substring": fragment,
                "values": atoms,
            }
        )

    def row(
        self,
        binding_id: str,
        label: str,
        row_marker: str,
        atom_specs: Iterable[tuple],
        *,
        panel: str | None = None,
        window_lines: int = 250,
    ) -> None:
        if label not in self.tables:
            raise ValueError(f"Missing target table: {label}")
        fragment = _row(self.tables[label], row_marker, panel)
        self._append(
            binding_id,
            {"kind": "latex_label", "value": label, "window_lines": window_lines},
            fragment,
            atom_specs,
        )

    def prose(
        self,
        binding_id: str,
        anchor: dict,
        start_marker: str,
        end_marker: str,
        atom_specs: Iterable[tuple],
    ) -> None:
        starts = [match.start() for match in re.finditer(re.escape(start_marker), self.tex)]
        if len(starts) != 1:
            raise ValueError(
                f"Expected one prose start marker {start_marker!r}, found {len(starts)}"
            )
        start = starts[0]
        end = self.tex.find(end_marker, start + len(start_marker))
        if end < 0:
            raise ValueError(f"Missing prose end marker after {start_marker!r}: {end_marker!r}")
        fragment = self.tex[start : end + len(end_marker)]
        self._append(binding_id, anchor, fragment, atom_specs)


def _math_if_negative(values: dict, source: dict, digits: int) -> dict:
    return fixed(digits, math_mode=float(_resolve(values, source)) < 0)


def _star_and_stat(values: dict, job: str, cid: str) -> list[tuple]:
    return [
        (contrast(job, cid, "significance_stars"), STARS),
        (contrast(job, cid, "statistic"), fixed(2)),
    ]


def _theta_gain_p(values: dict, job: str, cid: str, theta_digits: int = 6) -> list[tuple]:
    gain = contrast(job, cid, "geometric_gain_percent")
    return [
        (contrast(job, cid, "point"), fixed(theta_digits)),
        (contrast(job, cid, "significance_stars"), STARS),
        (contrast(job, cid, "statistic"), fixed(2)),
        (gain, _math_if_negative(values, gain, 4)),
        (contrast(job, cid, "reported_p"), fixed(4)),
    ]


def _build_direct_rq1_table(builder: BindingBuilder) -> None:
    """Bind every numerical result cell in the quarterly RQ1 table."""
    values = builder.values
    label = "tab:ch3:direct_model_results"
    if "Panel A: Mean MAE across bootstrap draws" not in _panel(
        builder.tables[label], "A"
    ):
        raise ValueError("RQ1 Panel A must explicitly report bootstrap mean MAE")

    for row_id, row_marker, condition in (
        ("film", "FiLM-CNN, matched LP", "lp_matched"),
        ("pure", "Pure-CNN, no text", "no_text"),
    ):
        atoms: list[tuple] = []
        for job, fold in FOLDS:
            cid = direct_contrast(job, fold, condition)
            atoms.append((arm(job, condition, "bootstrap_mean_mae"), fixed(9)))
            atoms.extend(_star_and_stat(values, job, cid))
        builder.row(
            f"rq1_quarterly_mae_{row_id}", label, row_marker, atoms, panel="A"
        )
    builder.row(
        "rq1_quarterly_mae_persistence",
        label,
        "Persistence",
        [(arm(job, "persistence", "bootstrap_mean_mae"), fixed(9)) for job, _ in FOLDS],
        panel="A",
    )

    for row_id, row_marker, condition in (
        ("film", "FiLM-CNN, matched LP", "lp_matched"),
        ("pure", "Pure-CNN, no text", "no_text"),
        ("persistence", "Persistence", "persistence"),
    ):
        atoms = []
        for job, _ in FOLDS:
            source = ratio(
                job,
                condition,
                "persistence",
                "bootstrap_improvement_percent",
            )
            atoms.append((source, _math_if_negative(values, source, 4)))
        builder.row(
            f"rq1_quarterly_improvement_{row_id}",
            label,
            row_marker,
            atoms,
            panel="B",
        )

    for row_id, row_marker, condition in (
        ("film", "FiLM-CNN, matched LP", "lp_matched"),
        ("pure", "Pure-CNN, no text", "no_text"),
    ):
        atoms = []
        for job, fold in FOLDS:
            cid = direct_contrast(job, fold, condition)
            atoms.append((contrast(job, cid, "reported_p"), fixed(4)))
        builder.row(
            f"rq1_quarterly_p_{row_id}", label, row_marker, atoms, panel="C"
        )

    theta_atoms: list[tuple] = []
    gain_atoms: list[tuple] = []
    p_atoms: list[tuple] = []
    for job, fold in FOLDS:
        cid = rq1_contrast(fold)
        theta_atoms.extend(
            [
                (contrast(job, cid, "point"), fixed(6)),
                (contrast(job, cid, "significance_stars"), STARS),
                (contrast(job, cid, "statistic"), fixed(2)),
            ]
        )
        gain = contrast(job, cid, "geometric_gain_percent")
        gain_atoms.append((gain, _math_if_negative(values, gain, 4)))
        p_atoms.append((contrast(job, cid, "reported_p"), fixed(4)))
    builder.row(
        "rq1_quarterly_film_vs_pure_theta",
        label,
        "$\\widehat{\\theta}$",
        theta_atoms,
        panel="D",
    )
    builder.row(
        "rq1_quarterly_film_vs_pure_gain",
        label,
        "Geometric gain (\\%)",
        gain_atoms,
        panel="D",
    )
    builder.row(
        "rq1_quarterly_film_vs_pure_p",
        label,
        "$p_{\\mathrm{Holm}}$",
        p_atoms,
        panel="D",
    )


def _build_direct_rq2_table(builder: BindingBuilder) -> None:
    """Bind every numerical result cell in the direct RQ2 table."""
    values = builder.values
    label = "tab:ch3:rq2_direct_results"
    if "Panel A: Mean MAE across bootstrap draws" not in _panel(
        builder.tables[label], "A"
    ):
        raise ValueError("RQ2 Panel A must explicitly report bootstrap mean MAE")
    rows = (
        ("bow", "BoW", "bow"),
        ("lp", "Text embedding (matched LP)", "lp_matched"),
        ("sentiment", "Sentiment", "sentiment"),
    )
    for row_id, row_marker, condition in rows:
        atoms: list[tuple] = []
        for job, fold in FOLDS:
            cid = direct_contrast(job, fold, condition)
            atoms.append((arm(job, condition, "bootstrap_mean_mae"), fixed(9)))
            atoms.extend(_star_and_stat(values, job, cid))
        builder.row(
            f"rq2_quarterly_mae_{row_id}", label, row_marker, atoms, panel="A"
        )
    builder.row(
        "rq2_quarterly_mae_persistence",
        label,
        "Persistence",
        [(arm(job, "persistence", "bootstrap_mean_mae"), fixed(9)) for job, _ in FOLDS],
        panel="A",
    )

    for row_id, row_marker, condition in rows + (("persistence", "Persistence", "persistence"),):
        atoms = []
        for job, _ in FOLDS:
            source = ratio(
                job,
                condition,
                "persistence",
                "bootstrap_improvement_percent",
            )
            atoms.append((source, _math_if_negative(values, source, 4)))
        builder.row(
            f"rq2_quarterly_improvement_{row_id}",
            label,
            row_marker,
            atoms,
            panel="B",
        )

    for row_id, row_marker, condition in rows + (("persistence", "Persistence", "persistence"),):
        atoms = []
        for job, _ in FOLDS:
            source = ratio(
                job, condition, "lp_matched", "bootstrap_mae_ratio"
            )
            atoms.append((source, fixed(6)))
        builder.row(
            f"rq2_quarterly_normalization_{row_id}",
            label,
            row_marker,
            atoms,
            panel="C",
        )

    for row_id, row_marker, condition in rows:
        atoms = []
        for job, fold in FOLDS:
            cid = direct_contrast(job, fold, condition)
            atoms.append((contrast(job, cid, "reported_p"), fixed(4)))
        builder.row(
            f"rq2_quarterly_p_{row_id}", label, row_marker, atoms, panel="D"
        )

    for row_id, row_marker, reference in (
        ("bow", "Text embedding vs BoW", "bow"),
        ("sentiment", "Text embedding vs sentiment", "sentiment"),
    ):
        cid = rq2_contrast("overall", reference)
        builder.row(
            f"rq2_overall_{row_id}",
            label,
            row_marker,
            _theta_gain_p(values, "direct_overall", cid),
            panel="E",
        )


def _build_rq3_summary_table(builder: BindingBuilder) -> None:
    """Bind the full RQ3 MAE panel and both pre-specified families."""
    values = builder.values
    label = "tab:ch3:rq3_rolling_results"
    if "Panel A: Four-fold mean MAE across bootstrap draws" not in _panel(
        builder.tables[label], "A"
    ):
        raise ValueError("RQ3 Panel A must explicitly report bootstrap mean MAE")
    job = "rq3_full_absolute_mae"
    rows = (
        ("matched", "FiLM, matched LP", "film_lp_matched"),
        ("zero", "FiLM, zero text", "film_zero_text"),
        ("shuffle", "FiLM, shuffled LP", "film_lp_shuffle"),
        ("bow", "FiLM, BoW", "film_bow"),
        ("sentiment", "FiLM, sentiment", "film_sentiment"),
        ("pure_continue", "Pure-CNN continuation", "pure_cnn_continue_no_text"),
        ("pure_parent", "Pure-CNN parent", "pure_cnn_parent"),
        ("persistence", "Persistence", "persistence"),
    )
    for row_id, row_marker, condition in rows:
        improvement = ratio(
            job, condition, "persistence", "bootstrap_improvement_percent"
        )
        normalized = ratio(
            job,
            condition,
            "pure_cnn_continue_no_text",
            "bootstrap_mae_ratio",
        )
        builder.row(
            f"rq3_full_mae_{row_id}",
            label,
            row_marker,
            [
                (arm(job, condition, "bootstrap_mean_mae"), fixed(10)),
                (improvement, _math_if_negative(values, improvement, 4)),
                (normalized, fixed(6)),
            ],
            panel="A",
        )

    comparisons = (
        (
            "branch_shuffle",
            "Matched LP vs shuffled LP",
            "rq3_branch",
            "matched_vs_film_lp_shuffle",
            6,
        ),
        (
            "branch_zero",
            "Matched LP vs zero text",
            "rq3_branch",
            "matched_vs_film_zero_text",
            7,
        ),
        (
            "intervention_zero",
            "Matched input vs zero input",
            "rq3_intervention",
            "matched_input_vs_zero_input",
            6,
        ),
        (
            "intervention_wrong",
            "Matched input vs wrong input",
            "rq3_intervention",
            "matched_input_vs_wrong_input",
            6,
        ),
    )
    for row_id, row_marker, source_job, cid, digits in comparisons:
        builder.row(
            f"rq3_summary_{row_id}",
            label,
            row_marker,
            [
                (contrast(source_job, cid, "point"), fixed(digits)),
                (contrast(source_job, cid, "significance_stars"), STARS),
                (contrast(source_job, cid, "statistic"), fixed(2)),
                (contrast(source_job, cid, "reported_p"), fixed(4)),
            ],
            panel="B",
        )


def _build_rq3_validation_table(builder: BindingBuilder) -> None:
    """Bind observed validation levels, changes, and paired diagnostics."""
    label = "tab:ch3:rq3_validation_trajectory"
    job = "rq3_validation_epoch0_epoch30_best"
    conditions = (
        ("matched", "FiLM, matched LP", "film_lp_matched"),
        ("zero", "FiLM, zero text", "film_zero_text"),
        ("shuffle", "FiLM, shuffled LP", "film_lp_shuffle"),
        (
            "pure_continue",
            "Pure-CNN continuation",
            "pure_cnn_continue_no_text",
        ),
    )
    stages = (
        ("A", "epoch0_to_epoch30", "epoch_0", "epoch_30"),
        ("B", "epoch30_to_best", "epoch_30", "best"),
    )
    for panel, stage_id, reference_stage, focal_stage in stages:
        for row_id, row_marker, condition in conditions:
            reference = f"{condition}::{reference_stage}"
            focal = f"{condition}::{focal_stage}"
            cid = f"validation_{condition}_{stage_id}"
            difference = ratio(job, focal, reference, "observed_mae_difference")
            improvement = ratio(
                job, focal, reference, "observed_improvement_percent"
            )
            builder.row(
                f"rq3_validation_{stage_id}_{row_id}",
                label,
                row_marker,
                [
                    (arm(job, reference, "observed_mean_mae"), fixed(10)),
                    (arm(job, focal, "observed_mean_mae"), fixed(10)),
                    (contrast(job, cid, "significance_stars"), STARS),
                    (contrast(job, cid, "statistic"), fixed(2)),
                    (difference, scientific(4, math_mode=True, explicit_plus=True)),
                    (
                        None,
                        {"operator": "negate", "operands": [improvement]},
                        fixed(4, math_mode=True, explicit_plus=True),
                    ),
                ],
                panel=panel,
            )


def _build_rq3_inference_tables(builder: BindingBuilder) -> None:
    values = builder.values
    for row_id, row_marker, cid, digits in (
        ("zero", "Matched LP vs zero text", "matched_vs_film_zero_text", 7),
        ("shuffle", "Matched LP vs shuffled LP", "matched_vs_film_lp_shuffle", 6),
    ):
        builder.row(
            f"rq3_branch_table_{row_id}",
            "tab:ch3:rq3_branch_inference",
            row_marker,
            _theta_gain_p(values, "rq3_branch", cid, digits),
        )

    for row_id, row_marker, cid in (
        ("zero", "Matched input vs zero input", "matched_input_vs_zero_input"),
        ("wrong", "Matched input vs wrong input", "matched_input_vs_wrong_input"),
    ):
        builder.row(
            f"rq3_intervention_table_{row_id}",
            "tab:ch3:rq3_intervention_inference",
            row_marker,
            _theta_gain_p(values, "rq3_intervention", cid),
        )


ARCHITECTURE_ARMS = (
    (
        "crossattn",
        "Cross-attention U-Net",
        "formal:crossattn_unet_mask_coords_v1::train_05m::common_5m_primary::text_matched",
        "crossattn_unet_mask_coords_v1",
    ),
    (
        "transformer",
        "Transformer-token model",
        "formal:transformer_tokens_mask_coords_v1::train_05m::common_5m_primary::text_matched",
        "transformer_tokens_mask_coords_v1",
    ),
    (
        "stylemod",
        "StyleMod U-Net",
        "formal:stylemod_unet_mask_coords_v1::train_05m::common_5m_primary::text_matched",
        "stylemod_unet_mask_coords_v1",
    ),
)


def _build_architecture_tables(builder: BindingBuilder) -> None:
    """Bind primary, secondary, and persistence architecture table rows."""
    values = builder.values
    job = "generator_architecture_robustness"
    primary_label = "tab:ch3:generator_architecture_robustness"
    secondary_label = "tab:ch3:generator_architecture_secondary"

    for row_id, display, condition, prefix in ARCHITECTURE_ARMS:
        cid = f"{prefix}_vs_film_reference"
        builder.row(
            f"architecture_primary_{row_id}",
            primary_label,
            f"{display} vs FiLM-CNN",
            [(arm(job, condition, "observed_mean_mae"), fixed(10))]
            + _theta_gain_p(values, job, cid),
        )

    for row_id, display, condition, prefix in ARCHITECTURE_ARMS:
        cid = f"{prefix}_vs_pure_cnn_reference"
        builder.row(
            f"architecture_secondary_{row_id}",
            secondary_label,
            f"{display} vs Pure-CNN",
            [(arm(job, condition, "observed_mean_mae"), fixed(10))]
            + _theta_gain_p(values, job, cid),
            panel="A",
        )

    persistence_rows = (
        (
            "film",
            "FiLM-CNN",
            "film_reference::train_05m::common_5m_primary::text_matched",
        ),
        (
            "pure",
            "Pure-CNN",
            "pure_cnn_reference::train_05m::common_5m_primary::text_zero",
        ),
    ) + tuple((row_id, display, condition) for row_id, display, condition, _ in ARCHITECTURE_ARMS)
    for row_id, display, condition in persistence_rows:
        improvement = ratio(
            job, condition, "persistence", "observed_improvement_percent"
        )
        builder.row(
            f"architecture_persistence_{row_id}",
            secondary_label,
            display,
            [
                (arm(job, condition, "observed_mean_mae"), fixed(10)),
                (improvement, _math_if_negative(values, improvement, 4)),
            ],
            panel="B",
        )


WINDOW_ROWS = (
    ("05", "5 min"),
    ("10", "10 min"),
    ("15", "15 min"),
    ("20", "20 min"),
    ("30", "30 min"),
)


def _common_window_condition(model: str, tolerance: str) -> str:
    if model == "film":
        return (
            f"film_lp_matched::train_{tolerance}m::common_5m_primary::text_matched"
        )
    if model == "pure":
        return (
            f"pure_cnn_no_text::train_{tolerance}m::common_5m_primary::text_zero"
        )
    raise ValueError(model)


def _build_window_tables(builder: BindingBuilder) -> None:
    """Bind common-panel between-model and within-model window rows."""
    values = builder.values
    job = "alignment_common_5m_panel"
    common_label = "tab:ch3:alignment_window_common_panel"
    within_label = "tab:ch3:alignment_window_within_model"

    for tolerance, row_marker in WINDOW_ROWS:
        film = _common_window_condition("film", tolerance)
        pure = _common_window_condition("pure", tolerance)
        cid = f"common5_train{tolerance}_film_vs_pure_cnn"
        builder.row(
            f"window_common_{tolerance}",
            common_label,
            row_marker,
            [
                (arm(job, film, "observed_mean_mae"), fixed(10)),
                (arm(job, pure, "observed_mean_mae"), fixed(10)),
            ]
            + _theta_gain_p(values, job, cid),
        )

    for model_id, row_marker, cid_prefix in (
        ("film", "FiLM-CNN", "common5_film_lp_matched"),
        ("pure", "Pure-CNN", "common5_pure_cnn_no_text"),
    ):
        # The row marker repeats four times, so isolate an explicit model block
        # before extracting each tolerance row.
        table = builder.tables[within_label]
        starts = [match.start() for match in re.finditer(rf"(?m)^{re.escape(row_marker)}$", table)]
        if len(starts) != 4:
            raise ValueError(f"Expected four {row_marker} window rows")
        for index, (tolerance, _) in enumerate(WINDOW_ROWS[1:]):
            row_start = starts[index]
            next_start = starts[index + 1] if index + 1 < len(starts) else len(table)
            model_fragment = table[row_start:next_start]
            # Cut through the first table-row terminator; no nested row break
            # occurs in these macro invocations.
            row_match = re.search(r".*?\\\\[ \t]*(?:\n|$)", model_fragment, re.DOTALL)
            if row_match is None:
                raise ValueError(f"Missing {row_marker} {tolerance} row")
            fragment = row_match.group(0).rstrip("\n")
            cid = f"{cid_prefix}_train{tolerance}_vs_train05"
            atoms = [
                _atom(values, contrast(job, cid, "point"), fixed(6)),
                _atom(values, contrast(job, cid, "significance_stars"), STARS),
                _atom(values, contrast(job, cid, "statistic"), fixed(2)),
            ]
            gain = contrast(job, cid, "geometric_gain_percent")
            atoms.extend(
                [
                    _atom(values, gain, _math_if_negative(values, gain, 4)),
                    _atom(values, contrast(job, cid, "reported_p"), fixed(4)),
                ]
            )
            missing = [a["literal"] for a in atoms if a["literal"] and a["literal"] not in fragment]
            if missing:
                raise ValueError(
                    f"Formal literals absent from window row {model_id}/{tolerance}: {missing}"
                )
            binding_id = f"window_within_{model_id}_{tolerance}"
            if binding_id in builder._ids:
                raise ValueError(f"Duplicate binding id: {binding_id}")
            builder._ids.add(binding_id)
            builder.bindings.append(
                {
                    "binding_id": binding_id,
                    "anchor": {
                        "kind": "latex_label",
                        "value": within_label,
                        "window_lines": 120,
                    },
                    "expected_substring": fragment,
                    "values": atoms,
                }
            )


def _build_window_coverage_mae_bindings(builder: BindingBuilder) -> None:
    """Bind only the v2-sourced pooled MAE columns in the coverage table.

    Pair counts and coverage percentages retain their separate descriptive
    lineage and are intentionally not claimed as bootstrap-archive results.
    """
    label = "tab:ch3:alignment_window_coverage"
    for tolerance, row_marker in WINDOW_ROWS:
        if tolerance == "05":
            job = "alignment_common_5m_panel"
            film = _common_window_condition("film", tolerance)
            pure = _common_window_condition("pure", tolerance)
        else:
            job = f"alignment_own_panel_{int(tolerance)}m"
            film = (
                f"film_lp_matched::train_{tolerance}m::"
                "own_tolerance_secondary::text_matched"
            )
            pure = (
                f"pure_cnn_no_text::train_{tolerance}m::"
                "own_tolerance_secondary::text_zero"
            )
        builder.row(
            f"window_coverage_pooled_mae_{tolerance}",
            label,
            row_marker,
            [
                (arm(job, film, "pair_count"), fixed(0)),
                (arm(job, film, "observed_mean_mae"), fixed(10)),
                (arm(job, pure, "observed_mean_mae"), fixed(10)),
            ],
        )


def _build_prose_bindings(builder: BindingBuilder) -> None:
    """Bind principal interpretation and conclusion statements.

    The selectors in this function deliberately use stable, non-numerical
    sentence boundaries.  They are populated only after the final narrative
    synchronization so a prose edit cannot silently redirect a binding to a
    different result.
    """
    def label_anchor(label: str, window_lines: int = 180) -> dict:
        return {"kind": "latex_label", "value": label, "window_lines": window_lines}

    # RQ1: quarterly persistence comparisons and the aggregate FiLM/Pure result.
    rq1_quarterly: list[tuple] = []
    for index, (job, _) in enumerate(FOLDS):
        source = ratio(
            job, "lp_matched", "persistence", "bootstrap_improvement_percent"
        )
        if index == 2:  # prose says "increases ... by" rather than printing a minus
            rq1_quarterly.append(
                (None, {"operator": "negate", "operands": [source]}, fixed(4))
            )
        else:
            rq1_quarterly.append((source, fixed(4)))
    for job, _ in FOLDS:
        source = ratio(job, "no_text", "persistence", "bootstrap_improvement_percent")
        rq1_quarterly.append((source, fixed(4)))
    rq1_quarterly.append(
        (
            contrast(
                "direct_f4_2023q4",
                "direct_f4_2023q4_lp_matched_vs_persistence",
                "reported_p",
            ),
            fixed(4),
        )
    )
    builder.prose(
        "rq1_interpretation_quarterly",
        label_anchor("tab:ch3:direct_model_results", 100),
        "Relative to persistence, FiLM-CNN reduces its mean-across-draw MAE by ",
        "performance is therefore period-dependent rather than uniformly better than\npersistence.",
        rq1_quarterly,
    )

    rq1_overall_cid = "rq1_overall_lp_matched_vs_no_text"
    rq1_overall_atoms: list[tuple] = []
    for job, fold in FOLDS:
        gain = contrast(job, rq1_contrast(fold), "geometric_gain_percent")
        rq1_overall_atoms.append((gain, fixed(4)))
    rq1_overall_atoms.extend(
        [
            (
                contrast(
                    "direct_f2_2023q2",
                    "rq1_f2_2023q2_lp_matched_vs_no_text",
                    "reported_p",
                ),
                fixed(4),
            ),
            (arm("direct_overall", "lp_matched", "bootstrap_mean_mae"), fixed(10)),
            (arm("direct_overall", "no_text", "bootstrap_mean_mae"), fixed(10)),
            (arm("direct_overall", "persistence", "bootstrap_mean_mae"), fixed(10)),
            (
                contrast("direct_overall", rq1_overall_cid, "point"),
                scientific(4, math_mode=True),
            ),
            (
                contrast(
                    "direct_overall", rq1_overall_cid, "geometric_gain_percent"
                ),
                fixed(4),
            ),
            (contrast("direct_overall", rq1_overall_cid, "statistic"), fixed(2)),
            (contrast("direct_overall", rq1_overall_cid, "reported_p"), fixed(4)),
        ]
    )
    builder.prose(
        "rq1_interpretation_overall",
        label_anchor("tab:ch3:direct_model_results", 100),
        "The direct architecture comparison is favourable to FiLM-CNN in three of the\nfour test folds.",
        "The small\naverage advantage is therefore descriptive and is not statistically\nestablished across the rolling evaluation.",
        rq1_overall_atoms,
    )

    # RQ2: explicit quarterly arm differences and the two aggregate contrasts.
    rq2_difference_atoms: list[tuple] = []
    for job, focal, reference in (
        ("direct_f1_2023q1", "lp_matched", "bow"),
        ("direct_f2_2023q2", "lp_matched", "bow"),
        ("direct_f4_2023q4", "lp_matched", "bow"),
        ("direct_f3_2023q3", "bow", "lp_matched"),
    ):
        rq2_difference_atoms.append(
            (
                ratio(job, focal, reference, "bootstrap_mae_difference"),
                scientific(4, math_mode=True),
            )
        )
    rq2_bow_cid = "rq2_overall_lp_matched_vs_bow"
    rq2_difference_atoms.extend(
        [
            (contrast("direct_overall", rq2_bow_cid, "point"), fixed(6)),
            (
                contrast(
                    "direct_overall", rq2_bow_cid, "geometric_gain_percent"
                ),
                fixed(4),
            ),
            (contrast("direct_overall", rq2_bow_cid, "statistic"), fixed(2)),
            (contrast("direct_overall", rq2_bow_cid, "reported_p"), fixed(4)),
        ]
    )
    builder.prose(
        "rq2_interpretation_bow",
        label_anchor("tab:ch3:rq2_direct_results", 120),
        "The relative ordering of BoW and the matched text embedding also changes over\ntime.",
        "It should not be interpreted as stable\nsuperiority of LP over BoW.",
        rq2_difference_atoms,
    )

    rq2_sentiment_cid = "rq2_overall_lp_matched_vs_sentiment"
    builder.prose(
        "rq2_interpretation_sentiment",
        label_anchor("tab:ch3:rq2_direct_results", 130),
        "The comparison with sentiment is more consistent.",
        "It is therefore suggestive rather than a verified\nrepresentation advantage under the stronger evidence rule.",
        [
            (contrast("direct_overall", rq2_sentiment_cid, "point"), fixed(6)),
            (
                contrast(
                    "direct_overall",
                    rq2_sentiment_cid,
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (contrast("direct_overall", rq2_sentiment_cid, "statistic"), fixed(2)),
            (contrast("direct_overall", rq2_sentiment_cid, "reported_p"), fixed(4)),
        ],
    )

    # RQ3: descriptive full-panel summaries and the frozen paired contrasts.
    rq3_full = "rq3_full_absolute_mae"
    matched_vs_pure = ratio(
        rq3_full,
        "film_lp_matched",
        "pure_cnn_continue_no_text",
        "bootstrap_mae_ratio",
    )
    builder.prose(
        "rq3_interpretation_full_panel_leaders",
        label_anchor("tab:ch3:rq3_rolling_results", 100),
        "Matched LP has the lowest mean-across-draw MAE,",
        "gain is attributable to textual information.",
        [
            (arm(rq3_full, "film_lp_matched", "bootstrap_mean_mae"), fixed(10)),
            (arm(rq3_full, "film_zero_text", "bootstrap_mean_mae"), fixed(10)),
            (
                arm(
                    rq3_full,
                    "pure_cnn_continue_no_text",
                    "bootstrap_mean_mae",
                ),
                fixed(10),
            ),
            (matched_vs_pure, fixed(6)),
            (
                ratio(
                    rq3_full,
                    "film_lp_matched",
                    "pure_cnn_continue_no_text",
                    "bootstrap_improvement_percent",
                ),
                fixed(4),
            ),
        ],
    )

    matched_pure_improvement = ratio(
        rq3_full,
        "film_lp_matched",
        "pure_cnn_continue_no_text",
        "bootstrap_improvement_percent",
    )
    zero_pure_improvement = ratio(
        rq3_full,
        "film_zero_text",
        "pure_cnn_continue_no_text",
        "bootstrap_improvement_percent",
    )
    shuffle_pure_improvement = ratio(
        rq3_full,
        "film_lp_shuffle",
        "pure_cnn_continue_no_text",
        "bootstrap_improvement_percent",
    )
    builder.prose(
        "rq3_interpretation_full_panel_relative",
        label_anchor("tab:ch3:rq3_rolling_results", 80),
        "Relative to the Pure-CNN continuation, matched LP and zero text reduce the",
        "no\ncorresponding four-fold Holm family was frozen.",
        [
            (matched_pure_improvement, fixed(4)),
            (zero_pure_improvement, fixed(4)),
            (
                None,
                {"operator": "negate", "operands": [shuffle_pure_improvement]},
                fixed(4),
            ),
        ],
    )

    branch_shuffle = "matched_vs_film_lp_shuffle"
    branch_zero = "matched_vs_film_zero_text"
    builder.prose(
        "rq3_interpretation_branch_summary",
        label_anchor("tab:ch3:rq3_rolling_results", 100),
        "The branch comparisons frozen before the v2 reanalysis show small descriptive advantages, but",
        "descriptive bootstrap-draw means.",
        [
            (contrast("rq3_branch", branch_shuffle, "point"), fixed(6)),
            (
                contrast(
                    "rq3_branch", branch_shuffle, "geometric_gain_percent"
                ),
                fixed(4),
            ),
            (contrast("rq3_branch", branch_shuffle, "reported_p"), fixed(4)),
            (contrast("rq3_branch", branch_zero, "point"), fixed(7)),
            (contrast("rq3_branch", branch_zero, "reported_p"), fixed(4)),
            (
                contrast("rq3_branch", branch_zero, "geometric_gain_percent"),
                fixed(4),
            ),
            (
                ratio(
                    rq3_full,
                    "film_lp_matched",
                    "film_zero_text",
                    "bootstrap_improvement_percent",
                ),
                fixed(4),
            ),
        ],
    )

    builder.prose(
        "rq3_interpretation_branch_detailed",
        label_anchor("tab:ch3:rq3_branch_inference", 90),
        "Both four-fold point estimates favour matched LP, but neither contrast is",
        "Taken together, the branch results do not verify a stable incremental",
        [
            (contrast("rq3_branch", branch_zero, "point"), fixed(7)),
            (
                contrast("rq3_branch", branch_zero, "geometric_gain_percent"),
                fixed(4),
            ),
            (contrast("rq3_branch", branch_zero, "statistic"), fixed(2)),
            (contrast("rq3_branch", branch_zero, "reported_p"), fixed(4)),
            (contrast("rq3_branch", branch_shuffle, "point"), fixed(6)),
            (
                contrast(
                    "rq3_branch", branch_shuffle, "geometric_gain_percent"
                ),
                fixed(4),
            ),
            (contrast("rq3_branch", branch_shuffle, "statistic"), fixed(2)),
            (contrast("rq3_branch", branch_shuffle, "reported_p"), fixed(4)),
        ],
    )

    intervention_job = "rq3_intervention"
    intervention_zero = "matched_input_vs_zero_input"
    intervention_wrong = "matched_input_vs_wrong_input"
    builder.prose(
        "rq3_interpretation_intervention_detailed",
        label_anchor("tab:ch3:rq3_intervention_inference", 90),
        "Both intervention estimates are positive.",
        "predictions themselves are unchanged.",
        [
            (arm(intervention_job, "matched_input", "observed_mean_mae"), fixed(10)),
            (arm(intervention_job, "zero_input", "observed_mean_mae"), fixed(10)),
            (arm(intervention_job, "wrong_input", "observed_mean_mae"), fixed(10)),
            (contrast(intervention_job, intervention_zero, "point"), fixed(6)),
            (
                contrast(
                    intervention_job,
                    intervention_zero,
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (contrast(intervention_job, intervention_wrong, "point"), fixed(6)),
            (
                contrast(
                    intervention_job,
                    intervention_wrong,
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (contrast(intervention_job, intervention_zero, "statistic"), fixed(2)),
            (contrast(intervention_job, intervention_wrong, "statistic"), fixed(2)),
            (contrast(intervention_job, intervention_zero, "reported_p"), fixed(4)),
            (contrast(intervention_job, intervention_wrong, "reported_p"), fixed(4)),
        ],
    )

    validation_job = "rq3_validation_epoch0_epoch30_best"
    validation_conditions = (
        "film_lp_matched",
        "film_zero_text",
        "film_lp_shuffle",
        "pure_cnn_continue_no_text",
    )
    epoch0_atoms: list[tuple] = [
        (
            arm(validation_job, "film_lp_matched::epoch_0", "observed_mean_mae"),
            fixed(10),
        )
    ]
    for index, condition in enumerate(validation_conditions):
        focal = f"{condition}::epoch_30"
        reference = f"{condition}::epoch_0"
        difference = ratio(
            validation_job, focal, reference, "observed_mae_difference"
        )
        improvement = ratio(
            validation_job, focal, reference, "observed_improvement_percent"
        )
        if index < 2:  # increases are printed as positive magnitudes
            epoch0_atoms.extend(
                [
                    (difference, scientific(4, math_mode=True)),
                    (
                        None,
                        {"operator": "negate", "operands": [improvement]},
                        fixed(4),
                    ),
                ]
            )
        else:  # reductions are also printed as positive magnitudes
            epoch0_atoms.extend(
                [
                    (
                        None,
                        {"operator": "negate", "operands": [difference]},
                        scientific(4, math_mode=True),
                    ),
                    (improvement, fixed(4)),
                ]
            )
    epoch0_atoms.extend(
        [
            (
                None,
                {
                    "operator": "difference",
                    "operands": [
                        arm(
                            validation_job,
                            "film_lp_matched::epoch_30",
                            "observed_mean_mae",
                        ),
                        arm(
                            validation_job,
                            "film_zero_text::epoch_30",
                            "observed_mean_mae",
                        ),
                    ],
                },
                scientific(4, math_mode=True),
            ),
            (
                None,
                {
                    "operator": "difference",
                    "operands": [
                        arm(
                            validation_job,
                            "film_lp_matched::epoch_30",
                            "observed_mean_mae",
                        ),
                        arm(
                            validation_job,
                            "film_lp_shuffle::epoch_30",
                            "observed_mean_mae",
                        ),
                    ],
                },
                scientific(4, math_mode=True),
            ),
        ]
    )
    builder.prose(
        "rq3_validation_interpretation_epoch0_to_epoch30",
        label_anchor("tab:ch3:rq3_validation_trajectory", 130),
        "Across the 40 seed--fold validation cells, all continuation paths have the",
        "Descriptively, the epoch-30 ordering\ndoes not suggest an acceleration specific to matched LP.",
        epoch0_atoms,
    )

    best_atoms: list[tuple] = []
    for condition in validation_conditions:
        focal = f"{condition}::best"
        reference = f"{condition}::epoch_30"
        difference = ratio(
            validation_job, focal, reference, "observed_mae_difference"
        )
        improvement = ratio(
            validation_job, focal, reference, "observed_improvement_percent"
        )
        best_atoms.extend(
            [
                (
                    None,
                    {"operator": "negate", "operands": [difference]},
                    scientific(4, math_mode=True),
                ),
                (improvement, fixed(4)),
            ]
        )
    best_atoms.append(
        (
            contrast(
                validation_job,
                "validation_film_lp_matched_epoch30_to_best",
                "reported_p",
            ),
            fixed(4),
        )
    )
    builder.prose(
        "rq3_validation_interpretation_epoch30_to_best",
        label_anchor("tab:ch3:rq3_validation_trajectory", 120),
        "The validation-selected checkpoints have lower equal-cell mean MAE than their",
        "behaviour and does not constitute independent out-of-sample evidence.",
        best_atoms,
    )

    # Generator-architecture robustness: primary, secondary, persistence and
    # same-checkpoint text summaries all retain the formal pooled estimand.
    architecture_job = "generator_architecture_robustness"
    architecture_primary_atoms: list[tuple] = [
        (
            arm(
                architecture_job,
                "film_reference::train_05m::common_5m_primary::text_matched",
                "observed_mean_mae",
            ),
            fixed(10),
        )
    ]
    for cid in (
        "crossattn_unet_mask_coords_v1_vs_film_reference",
        "stylemod_unet_mask_coords_v1_vs_film_reference",
        "transformer_tokens_mask_coords_v1_vs_film_reference",
    ):
        gain = contrast(architecture_job, cid, "geometric_gain_percent")
        architecture_primary_atoms.append(
            (None, {"operator": "negate", "operands": [gain]}, fixed(4))
        )
    builder.prose(
        "architecture_interpretation_primary",
        label_anchor("tab:ch3:generator_architecture_robustness", 70),
        "Table~\\ref{tab:ch3:generator_architecture_robustness} shows that the",
        "none of the alternative architectures significantly improves upon FiLM-CNN,\nand no significance superscripts appear in the table.",
        architecture_primary_atoms,
    )

    architecture_secondary_atoms: list[tuple] = [
        (
            arm(
                architecture_job,
                "pure_cnn_reference::train_05m::common_5m_primary::text_zero",
                "observed_mean_mae",
            ),
            fixed(10),
        )
    ]
    for cid in (
        "crossattn_unet_mask_coords_v1_vs_pure_cnn_reference",
        "transformer_tokens_mask_coords_v1_vs_pure_cnn_reference",
        "stylemod_unet_mask_coords_v1_vs_pure_cnn_reference",
    ):
        architecture_secondary_atoms.extend(
            [
                (
                    contrast(architecture_job, cid, "geometric_gain_percent"),
                    fixed(4),
                ),
                (contrast(architecture_job, cid, "statistic"), fixed(2)),
            ]
        )
    builder.prose(
        "architecture_interpretation_secondary",
        label_anchor("tab:ch3:generator_architecture_secondary", 70),
        "Table~\\ref{tab:ch3:generator_architecture_secondary} places the primary",
        "These comparisons thus\nprovide no evidence that an alternative architecture outperforms Pure-CNN.",
        architecture_secondary_atoms,
    )

    architecture_persistence_conditions = (
        "film_reference::train_05m::common_5m_primary::text_matched",
        "pure_cnn_reference::train_05m::common_5m_primary::text_zero",
        "formal:crossattn_unet_mask_coords_v1::train_05m::common_5m_primary::text_matched",
        "formal:stylemod_unet_mask_coords_v1::train_05m::common_5m_primary::text_matched",
        "formal:transformer_tokens_mask_coords_v1::train_05m::common_5m_primary::text_matched",
    )
    architecture_persistence_atoms: list[tuple] = [
        (arm(architecture_job, "persistence", "observed_mean_mae"), fixed(10))
    ]
    for condition in architecture_persistence_conditions:
        architecture_persistence_atoms.append(
            (
                ratio(
                    architecture_job,
                    condition,
                    "persistence",
                    "observed_improvement_percent",
                ),
                fixed(4),
            )
        )
    builder.prose(
        "architecture_interpretation_persistence",
        label_anchor("tab:ch3:generator_architecture_secondary", 80),
        "All five models also have lower point MAE than the common persistence benchmark",
        "bootstrap tests and are not members of the architecture testing family.",
        architecture_persistence_atoms,
    )

    architecture_text_min = contrast(
        architecture_job,
        "film_reference_matched_vs_zero",
        "geometric_gain_percent",
    )
    architecture_text_max = contrast(
        architecture_job,
        "film_reference_matched_vs_shuffle",
        "geometric_gain_percent",
    )
    builder.prose(
        "architecture_interpretation_text_reliance",
        label_anchor("tab:ch3:generator_architecture_secondary", 100),
        "The same-checkpoint interventions reinforce this cautious interpretation.",
        "matched LP input improves forecast accuracy.",
        [
            (architecture_text_min, fixed(4)),
            (architecture_text_max, fixed(4)),
            (
                contrast(
                    architecture_job,
                    "formal:stylemod_unet_mask_coords_v1_matched_vs_zero",
                    "reported_p",
                ),
                fixed(4),
            ),
        ],
    )

    # Alignment-window robustness: bind the extrema quoted in the common-panel
    # interpretation and every displayed matched-versus-zero text gain.
    window_job = "alignment_common_5m_panel"
    common5_gain = contrast(
        window_job, "common5_train05_film_vs_pure_cnn", "geometric_gain_percent"
    )
    common5_stat = contrast(
        window_job, "common5_train05_film_vs_pure_cnn", "statistic"
    )
    builder.prose(
        "window_interpretation_common_panel",
        label_anchor("tab:ch3:alignment_window_common_panel", 55),
        "FiLM-CNN achieves the lower point-estimate MAE in three of the five",
        "the reverse directional hypothesis was not tested.",
        [
            (common5_gain, fixed(3)),
            (
                None,
                {"operator": "negate", "operands": [common5_stat]},
                fixed(2),
            ),
            (
                contrast(
                    window_job,
                    "common5_train05_film_vs_pure_cnn",
                    "reported_p",
                ),
                fixed(4),
            ),
        ],
    )

    window_text_atoms: list[tuple] = []
    window_text_sources: list[dict] = []
    for tolerance, _ in WINDOW_ROWS:
        source = contrast(
            window_job,
            f"common5_train{tolerance}_film_matched_vs_zero",
            "geometric_gain_percent",
        )
        window_text_sources.append(source)
        window_text_atoms.append((source, fixed(4)))
    window_text_atoms.extend(
        [
            (window_text_sources[2], fixed(4)),
            (
                None,
                {"operator": "negate", "operands": [window_text_sources[1]]},
                fixed(2),
            ),
            (
                contrast(
                    window_job,
                    "common5_train15_film_matched_vs_zero",
                    "reported_p",
                ),
                fixed(4),
            ),
        ]
    )
    coverage_count_atoms = [
        (
            arm(
                window_job,
                _common_window_condition("film", "05"),
                "pair_count",
            ),
            fixed(0),
        ),
        (
            arm(
                "alignment_own_panel_30m",
                "film_lp_matched::train_30m::own_tolerance_secondary::text_matched",
                "pair_count",
            ),
            fixed(0),
        ),
    ]
    builder.prose(
        "window_interpretation_coverage_counts",
        label_anchor("tab:ch3:alignment_window_coverage", 60),
        "Increasing $w$ from 5 to 30 minutes expands the cumulative test universe from",
        "a descriptive comparison between the two architectures.",
        coverage_count_atoms,
    )
    builder.prose(
        "window_interpretation_text_reliance",
        label_anchor("tab:ch3:alignment_window_coverage", 90),
        "As a separate inference-time diagnostic, each trained FiLM-CNN checkpoint is",
        "statistically supported evidence that wider alignment induces an average MAE\nbenefit from retaining the matched text.",
        window_text_atoms,
    )

    # Conclusion: bind the principal numerical takeaways for RQ1--RQ3.  RQ4
    # remains explicitly excluded even though its prose appears in this section.
    conclusion_anchor = {
        "kind": "unique_text",
        "value": "\\section{Conclusion}",
        "window_lines": 180,
    }
    builder.prose(
        "conclusion_rq1",
        conclusion_anchor,
        "For RQ1, the directly trained FiLM-CNN is competitive with, but is not shown",
        "information.",
        [
            (
                contrast(
                    "direct_overall",
                    "rq1_overall_lp_matched_vs_no_text",
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "direct_overall",
                    "rq1_overall_lp_matched_vs_no_text",
                    "reported_p",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "direct_f4_2023q4",
                    "direct_f4_2023q4_lp_matched_vs_persistence",
                    "reported_p",
                ),
                fixed(4),
            ),
        ],
    )
    builder.prose(
        "conclusion_rq2",
        conclusion_anchor,
        "RQ2 compares representation quality under a common FiLM architecture and",
        "through which contextual text affects forecasts.",
        [
            (
                contrast(
                    "direct_overall",
                    "rq2_overall_lp_matched_vs_sentiment",
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "direct_overall",
                    "rq2_overall_lp_matched_vs_sentiment",
                    "reported_p",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "direct_overall",
                    "rq2_overall_lp_matched_vs_bow",
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "direct_overall",
                    "rq2_overall_lp_matched_vs_bow",
                    "reported_p",
                ),
                fixed(4),
            ),
        ],
    )
    builder.prose(
        "conclusion_rq3",
        conclusion_anchor,
        "RQ3 imposes stronger controls on the incremental value attributed to text.",
        "therefore remains \\texttt{no\\_verified\\_text\\_reliance}.",
        [
            (
                contrast(
                    "rq3_branch",
                    "matched_vs_film_lp_shuffle",
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "rq3_branch", "matched_vs_film_lp_shuffle", "reported_p"
                ),
                fixed(4),
            ),
            (
                contrast(
                    "rq3_branch",
                    "matched_vs_film_zero_text",
                    "geometric_gain_percent",
                ),
                fixed(4),
            ),
            (
                contrast(
                    "rq3_branch", "matched_vs_film_zero_text", "reported_p"
                ),
                fixed(4),
            ),
        ],
    )
    builder.prose(
        "conclusion_alignment_coverage",
        conclusion_anchor,
        "Finally, widening the news--market alignment tolerance from",
        "small. Wider alignment thus offers an operational coverage gain, not verified\nevidence of greater forecasting accuracy or stronger text reliance.",
        coverage_count_atoms,
    )


def _passthrough_records(tex: str, existing: dict) -> list[dict]:
    records = deepcopy(existing.get("passthrough_table_hashes", []))
    if tuple(item.get("label") for item in records) != PASSTHROUGH_LABELS:
        raise ValueError("The six frozen passthrough-table records changed")
    tables = _tables(tex)
    for record in records:
        label = record["label"]
        if label not in tables:
            raise ValueError(f"Missing passthrough table: {label}")
        digest = hashlib.sha256(tables[label].encode("utf-8")).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(
                f"Frozen passthrough table changed: {label}: "
                f"{digest} != {record['sha256']}"
            )
    return records


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path: Path, role: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing F4 capacity {role}: {path}")
    return path


def _summary_artifact_hash(summary: dict, name: str) -> str:
    try:
        digest = summary["artifacts"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"F4 capacity summary is missing artifact hash: {name}") from exc
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Invalid F4 capacity artifact hash for {name}")
    return digest


def _validate_f4_capacity_summary(summary: dict) -> None:
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
            raise ValueError(
                f"F4 capacity summary contract drift at {field}: "
                f"{summary.get(field)!r} != {value!r}"
            )
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("F4 capacity summary must contain six capacity rows")
    if [row.get("capacity_id") for row in rows if isinstance(row, dict)] != expected[
        "capacity_ids"
    ]:
        raise ValueError("F4 capacity summary row order or capacity IDs drifted")


def _resolve_recorded_path(path_value: Any) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("F4 capacity summary contains an invalid source path")
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _f4_capacity_source_record(
    analysis_dir: Path,
    chapter_table: str,
) -> dict[str, str]:
    """Validate the generated F4 table lineage before recording any hashes."""

    analysis_dir = analysis_dir.resolve()
    summary_path = _required_file(
        analysis_dir / F4_CAPACITY_SUMMARY_JSON, "summary JSON"
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid F4 capacity summary JSON: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("F4 capacity summary JSON must be an object")
    _validate_f4_capacity_summary(summary)

    paths = {
        "pair_metrics": _required_file(
            analysis_dir / F4_CAPACITY_PAIR_METRICS, "pair metrics"
        ),
        "summary_csv": _required_file(
            analysis_dir / F4_CAPACITY_SUMMARY_CSV, "summary CSV"
        ),
        "latex_table": _required_file(
            analysis_dir / F4_CAPACITY_TABLE_TEX, "LaTeX table"
        ),
    }
    artifact_names = {
        "pair_metrics": F4_CAPACITY_PAIR_METRICS,
        "summary_csv": F4_CAPACITY_SUMMARY_CSV,
        "latex_table": F4_CAPACITY_TABLE_TEX,
    }
    hashes: dict[str, str] = {}
    for role, path in paths.items():
        digest = _sha256_path(path)
        expected_digest = _summary_artifact_hash(summary, artifact_names[role])
        if digest != expected_digest:
            raise ValueError(
                f"F4 capacity {role} hash drift: {digest} != {expected_digest}"
            )
        hashes[role] = digest

    input_path = _required_file(
        _resolve_recorded_path(summary.get("input_pair_metrics_path")),
        "input pair metrics",
    )
    input_digest = summary.get("input_pair_metrics_sha256")
    if (
        not isinstance(input_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_digest) is None
        or _sha256_path(input_path) != input_digest
    ):
        raise ValueError("F4 capacity input pair-metric source hash drift")

    generated_tex = paths["latex_table"].read_text(encoding="utf-8")
    generated_blocks = re.findall(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}", generated_tex, re.DOTALL
    )
    generated_tables = _tables(generated_tex)
    if len(generated_blocks) != 1:
        raise ValueError("Generated F4 capacity LaTeX must contain exactly one table")
    if set(generated_tables) != {F4_CAPACITY_TABLE_LABEL}:
        raise ValueError(
            "Generated F4 capacity LaTeX must contain exactly its active table label"
        )
    if generated_tables[F4_CAPACITY_TABLE_LABEL] != chapter_table:
        raise ValueError(
            "Chapter F4 capacity table is not byte-identical to the generated table"
        )

    return {
        "summary_json_path": _repo_relative(summary_path),
        "summary_json_sha256": _sha256_path(summary_path),
        "pair_metrics_path": _repo_relative(paths["pair_metrics"]),
        "pair_metrics_sha256": hashes["pair_metrics"],
        "summary_csv_path": _repo_relative(paths["summary_csv"]),
        "summary_csv_sha256": hashes["summary_csv"],
        "latex_table_path": _repo_relative(paths["latex_table"]),
        "latex_table_sha256": hashes["latex_table"],
        "input_pair_metrics_path": _repo_relative(input_path),
        "input_pair_metrics_sha256": input_digest,
    }


def _external_table_records(
    tex: str,
    existing: dict,
    *,
    f4_capacity_analysis_dir: Path,
) -> list[dict]:
    """Update external-table hashes, migrating capacity provenance atomically.

    Until the chapter adopts the new F4 label, the historical record remains
    unchanged and the not-yet-produced analysis artifacts are never opened.
    Once the new label appears, every source artifact is required and verified
    before the historical label is nested as superseded provenance.
    """

    records = deepcopy(existing.get("external_table_changes", []))
    if len(records) != 2:
        raise ValueError("The two externally tracked table records are required")
    labels = [record.get("label") for record in records]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate externally tracked table label")
    tables = _tables(tex)
    legacy_present = LEGACY_CAPACITY_TABLE_LABEL in tables
    active_present = F4_CAPACITY_TABLE_LABEL in tables
    if legacy_present == active_present:
        raise ValueError(
            "Chapter must contain exactly one legacy or active capacity table"
        )
    if active_present and "\\label{" + LEGACY_CAPACITY_TABLE_LABEL + "}" in tex:
        raise ValueError("Superseded legacy capacity label remains in the chapter")

    alignment_label = "tab:ch3:alignment_window_coverage"
    by_label = {record.get("label"): record for record in records}
    if alignment_label not in by_label or alignment_label not in tables:
        raise ValueError("Missing externally tracked alignment-window table")
    alignment = deepcopy(by_label[alignment_label])
    alignment["current_sha256"] = hashlib.sha256(
        tables[alignment_label].encode("utf-8")
    ).hexdigest()
    alignment["status"] = "user_authorized_v2_update_verified"
    alignment["numerical_source_audit"] = (
        "Pair counts and coverage remain externally audited; the two "
        "displayed pooled-MAE columns are additionally bound row by row "
        "to formal v2 arm records in this JSON."
    )

    if legacy_present:
        if LEGACY_CAPACITY_TABLE_LABEL not in by_label:
            raise ValueError("Missing active legacy capacity-table record")
        capacity = deepcopy(by_label[LEGACY_CAPACITY_TABLE_LABEL])
        capacity["current_sha256"] = hashlib.sha256(
            tables[LEGACY_CAPACITY_TABLE_LABEL].encode("utf-8")
        ).hexdigest()
        capacity["status"] = "user_authorized_descriptive_update_verified"
        capacity["numerical_source_audit"] = (
            "Q3/Q4 values were independently reconciled to the pinned capacity "
            "pair CSVs using panel=core, stratum_type=overall, "
            "stratum_value=all, and tolerance_minutes=5; see the integration "
            "notes. This descriptive table is not v2-bootstrap bound."
        )
        return [capacity, alignment]

    if LEGACY_CAPACITY_TABLE_LABEL in by_label:
        previous = deepcopy(by_label[LEGACY_CAPACITY_TABLE_LABEL])
        last_active_sha256 = previous.get("current_sha256")
        baseline_sha256 = previous.get("baseline_sha256")
    elif F4_CAPACITY_TABLE_LABEL in by_label:
        previous_active = by_label[F4_CAPACITY_TABLE_LABEL]
        previous = deepcopy(previous_active.get("supersedes"))
        if not isinstance(previous, dict):
            raise ValueError("Active capacity record lacks superseded provenance")
        last_active_sha256 = previous.get("last_active_sha256")
        baseline_sha256 = previous.get("baseline_sha256")
    else:
        raise ValueError("Missing capacity-table external provenance record")
    if previous.get("label") != LEGACY_CAPACITY_TABLE_LABEL:
        raise ValueError("Superseded capacity-table label drifted")
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (baseline_sha256, last_active_sha256)
    ):
        raise ValueError("Superseded capacity-table hashes are invalid")

    chapter_table = tables[F4_CAPACITY_TABLE_LABEL]
    source = _f4_capacity_source_record(
        f4_capacity_analysis_dir,
        chapter_table,
    )
    capacity = {
        "label": F4_CAPACITY_TABLE_LABEL,
        "current_sha256": hashlib.sha256(chapter_table.encode("utf-8")).hexdigest(),
        "status": "user_authorized_descriptive_update_verified",
        "source": source,
        "supersedes": {
            "label": LEGACY_CAPACITY_TABLE_LABEL,
            "status": "superseded",
            "baseline_sha256": baseline_sha256,
            "last_active_sha256": last_active_sha256,
        },
        "numerical_source_audit": (
            "The F4-only FiLM-CNN/Pure-CNN capacity table was generated from "
            "the frozen three-seed, 143-pair analysis artifacts recorded above; "
            "it is externally source-verified and not v2-bootstrap bound."
        ),
    }
    return [capacity, alignment]


def build_payload(
    tex_path: Path = DEFAULT_TEX,
    values_path: Path = DEFAULT_VALUES,
    existing_path: Path = DEFAULT_OUTPUT,
    *,
    f4_capacity_analysis_dir: Path = DEFAULT_F4_CAPACITY_ANALYSIS_DIR,
) -> dict:
    tex_path = tex_path.resolve()
    values_path = values_path.resolve()
    existing_path = existing_path.resolve()
    tex = tex_path.read_text(encoding="utf-8")
    values = json.loads(values_path.read_text(encoding="utf-8"))
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    if values.get("kind") != "chapter3_shared_market_panel_bootstrap_v2":
        raise ValueError("Only the formal corrected v2 archive may be bound")
    missing_jobs = sorted(set(REQUIRED_JOBS) - set(values.get("jobs", {})))
    if missing_jobs:
        raise ValueError(f"Formal v2 archive is missing jobs: {missing_jobs}")

    builder = BindingBuilder(tex, values)
    missing_tables = sorted(set(REQUIRED_RESULT_TABLES) - set(builder.tables))
    if missing_tables:
        raise ValueError(f"Chapter is missing required result tables: {missing_tables}")
    _build_direct_rq1_table(builder)
    _build_direct_rq2_table(builder)
    _build_rq3_summary_table(builder)
    _build_rq3_validation_table(builder)
    _build_rq3_inference_tables(builder)
    _build_architecture_tables(builder)
    _build_window_tables(builder)
    _build_window_coverage_mae_bindings(builder)
    _build_prose_bindings(builder)

    bound_jobs: set[str] = set()
    atom_count = 0
    for binding in builder.bindings:
        for item in binding["values"]:
            sources = (
                [item["source"]]
                if "source" in item
                else item["source_expression"]["operands"]
            )
            bound_jobs.update(source["job_id"] for source in sources)
            atom_count += 1
    missing_bound_jobs = sorted(set(REQUIRED_JOBS) - bound_jobs)
    if missing_bound_jobs:
        raise ValueError(f"No binding reaches required formal jobs: {missing_bound_jobs}")

    contract = deepcopy(existing["binding_contract"])
    operators = contract["source_expression"]["allowed_operators"]
    if "negate" not in operators:
        operators.append("negate")
    contract["source_expression"]["operand_contract"] = (
        "difference, ratio, and one_minus_ratio_percent require two formal "
        "source operands; negate requires one formal source operand."
    )

    external = _external_table_records(
        tex,
        existing,
        f4_capacity_analysis_dir=f4_capacity_analysis_dir,
    )
    scope = deepcopy(existing["scope"])
    scope["excluded_anchor_range"]["start"] = (
        "\\label{subsec:ch3:rq4_conditional_text_value}"
    )
    coverage_interpretation = (
        "All displayed numerical cells in the listed inferential/result "
        "tables plus principal interpretation and Conclusion statements "
        "are explicitly bound. The pooled-MAE columns in the descriptive "
        "coverage table are bound to v2 arms, while its pair counts and "
        "coverage percentages retain their external descriptive audit. "
        "Training, capacity, constraint, design-count, and RQ4 numbers are "
        "outside this v2 inferential binding scope."
    )
    if F4_CAPACITY_TABLE_LABEL in builder.tables:
        coverage_interpretation = (
            "All displayed numerical cells in the listed inferential/result "
            "tables plus principal interpretation and Conclusion statements "
            "are explicitly bound. The pooled-MAE columns in the descriptive "
            "coverage table are bound to v2 arms, while its pair counts and "
            "coverage percentages retain their external descriptive audit. "
            "The active F4 capacity table is externally source-verified, but "
            "training, capacity, constraint, design-count, and RQ4 numbers "
            "remain outside this v2 inferential binding scope."
        )
    payload = {
        "schema_version": 1,
        "kind": "chapter3_bootstrap_tex_bindings",
        "status": "complete",
        "tex_path": _repo_relative(tex_path),
        "values_path": _repo_relative(values_path),
        "scope": scope,
        "binding_contract": contract,
        "passthrough_table_hash_contract": deepcopy(
            existing["passthrough_table_hash_contract"]
        ),
        "passthrough_table_hashes": _passthrough_records(tex, existing),
        "external_table_changes": external,
        "coverage": {
            "required_result_tables": list(REQUIRED_RESULT_TABLES),
            "bound_jobs": sorted(bound_jobs),
            "binding_groups": len(builder.bindings),
            "numerical_bindings": atom_count,
            "interpretation": coverage_interpretation,
        },
        "bindings": builder.bindings,
    }
    return payload


def _serialized(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write the bindings JSON")
    action.add_argument(
        "--check", action="store_true", help="require the existing JSON to be current"
    )
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--f4-capacity-analysis-dir",
        type=Path,
        default=DEFAULT_F4_CAPACITY_ANALYSIS_DIR,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build_payload(
        args.tex,
        args.values,
        output,
        f4_capacity_analysis_dir=args.f4_capacity_analysis_dir,
    )
    serialized = _serialized(payload)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != serialized:
            raise SystemExit("Chapter 3 bootstrap bindings are stale; run with --write")
    else:
        output.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "current" if args.check else "written",
                "path": _repo_relative(output),
                "binding_groups": len(payload["bindings"]),
                "numerical_bindings": payload["coverage"]["numerical_bindings"],
                "bound_jobs": payload["coverage"]["bound_jobs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
