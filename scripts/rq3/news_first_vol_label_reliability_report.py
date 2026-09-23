"""Self-contained HTML report for the label-reliability experiment."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _table(path: Path, *, maximum_rows: int = 200) -> str:
    if not path.is_file():
        return '<p class="missing">Not materialized.</p>'
    frame = pd.read_csv(path, low_memory=False)
    note = ""
    if len(frame) > maximum_rows:
        note = f"<p>Showing {maximum_rows:,} of {len(frame):,} rows.</p>"
        frame = frame.head(maximum_rows)
    return note + frame.to_html(index=False, border=0, classes="data", escape=True)


def _kv_table(payload: Mapping[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td>{escape(rendered)}</td></tr>"
        )
    return '<table class="kv"><tbody>' + "".join(rows) + "</tbody></table>"


def render_label_reliability_report(experiment_root: str | Path) -> Path:
    """Render one portable report without network assets or Q4 content."""

    root = Path(experiment_root).resolve()
    status = _read_json(root / "label_reliability_stage_status.json")
    selection = _read_json(root / "label_reliability_selection.json")
    audit_manifest = _read_json(root / "bootstrap" / "audit" / "manifest.json")
    selected = selection.get("selected_arm")
    if selection.get("status") == "terminal_no_winner":
        headline = (
            "completed_no_reliable_label_policy: no filtering/weighting arm "
            "passed the predeclared fold-only gate; the experiment stopped before Q3."
        )
    elif selected:
        headline = (
            f"Arm {escape(str(selected))} was selected using four inner validation "
            "folds; Q3 is an exploratory repeated evaluation only."
        )
    else:
        headline = "The experiment is prepared or still running; no arm is selected."
    sections = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>News-first Vol label reliability</title>",
        """<style>
        body{font:14px/1.45 system-ui,sans-serif;margin:2rem;color:#17202a}
        h1,h2{color:#123b5d} .notice{padding:1rem;background:#eef6fb;border-left:4px solid #2980b9}
        .guard{padding:.8rem;background:#fff4e5;border-left:4px solid #f39c12}
        table{border-collapse:collapse;width:100%;margin:.7rem 0 1.5rem}
        th,td{border:1px solid #d8dee4;padding:.35rem .5rem;text-align:right;vertical-align:top}
        th:first-child,td:first-child{text-align:left}.kv th{width:30%;text-align:left}.kv td{text-align:left}
        tr:nth-child(even){background:#f7f9fa}.missing{color:#6b7280;font-style:italic}
        code{background:#f1f3f5;padding:.1rem .25rem}
        </style></head><body>""",
        "<h1>News-first Vol label-reliability experiment</h1>",
        f'<p class="notice">{headline}</p>',
        '<p class="guard"><strong>Evaluation boundary:</strong> Q3 was not used '
        "for arm selection. Q4 was not materialized as a training loader and was "
        "not predicted, selected on, or reported. Bootstrap surfaces were audit-only.</p>",
        "<h2>Experiment state</h2>",
        _kv_table(status),
        "<h2>Immutable selection</h2>",
        _kv_table(selection) if selection else '<p class="missing">Not selected.</p>',
        "<h2>Raw-trade bootstrap lineage</h2>",
        _kv_table(audit_manifest) if audit_manifest else '<p class="missing">Not materialized.</p>',
        "<h2>Reliability profiles and retention</h2>",
        _table(root / "reliability_profiles" / "reliability_profile_manifest.csv"),
        "<h2>Fold-only arm comparisons</h2>",
        _table(root / "analysis" / "label_reliability_comparisons.csv"),
        "<h2>Exploratory repeated Q3 pair metrics</h2>",
        _table(root / "analysis" / "q3_pair_metrics.csv.gz"),
        "<h2>Resource summary</h2>",
        _table(root / "resource_summary.csv"),
        "<h2>Interpretation limits</h2>",
        "<p>The design uses three fixed seeds and four chronological folds. It tests "
        "label treatment at one Small architecture and one learning rate; it does not "
        "establish initialization stability beyond those seeds. Any Q3 result is "
        "exploratory because Q3 has been examined in earlier work. A WGAN that does "
        "not show learnability should be described through reconstruction, critic, GP "
        "and optimization diagnostics rather than as a generative advantage.</p>",
        "</body></html>",
    ]
    output = root / "report" / "label_reliability_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    return output


__all__ = ["render_label_reliability_report"]
