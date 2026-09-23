"""Self-contained report for the FiLM-G + NoLP Critic capacity experiment."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


def _fmt(value: object, digits: int = 9) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    headers = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(row[column]) for column in columns) + " |"
        for _, row in frame.loc[:, list(columns)].iterrows()
    ]
    return "\n".join([headers, rule, *rows])


def _html_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_fmt(row[column]))}</td>" for column in columns)
        + "</tr>"
        for _, row in frame.loc[:, list(columns)].iterrows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report(experiment_root: str | Path) -> Path:
    root = Path(experiment_root).resolve()
    analysis = root / "analysis"
    selection = json.loads(
        (analysis / "film_nolp_capacity_q3_selection.json").read_text(encoding="utf-8")
    )
    q3 = pd.read_csv(analysis / "film_nolp_capacity_q3_scores.csv")
    q3_persistence = pd.read_csv(
        analysis / "film_nolp_capacity_q3_persistence_contrasts.csv"
    )
    q4 = pd.read_csv(analysis / "film_nolp_capacity_q4_scores.csv")
    q4_persistence = pd.read_csv(
        analysis / "film_nolp_capacity_q4_persistence_contrasts.csv"
    )
    q4_summary = json.loads(
        (analysis / "film_nolp_capacity_q4_summary.json").read_text(encoding="utf-8")
    )
    columns = (
        "capacity_profile",
        "parameter_count",
        "geometric_mae_ratio",
        "improvement_fraction",
        "nonworse_seed_count",
    )
    contrast_columns = (
        "capacity_profile",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_holm",
        "statistically_supported",
    )
    markdown = f"""# FiLM G + NoLP Critic：多 Seed 容量实验

## 结论先行

Q3冻结的点估计leader为 `{selection["point_leader"]}`，one-SE候选为
`{selection["one_se_candidate"]}`。候选证据等级为
`{selection["candidate_statistical_support"]}`。容量选择只使用5m训练模型在共同Q3面板上的结果；30m不参与选型。

## Q3容量排名

{_markdown_table(q3, columns)}

### Q3相对persistence

{_markdown_table(q3_persistence, contrast_columns)}

## Q4回顾性评价

{_markdown_table(q4, columns)}

### Q4相对persistence

{_markdown_table(q4_persistence, contrast_columns)}

Q4的解释标签为 `{q4_summary["q4_confirmation_label"]}`。当前143个Q4 pairs与历史预测证据重合
`{q4_summary["historical_q4_exposure"]["pair_overlap_label"]}`，因此该结果不是confirmatory holdout，且不得用于重新选择容量。

## 固定实验边界

- Generator：`film_conv_bottleneck_concat_v1`；Critic：`lp_disabled_same_shape_v1`。
- Seeds：42、202、404；real_text；LR=5e-7；exact-TTM 16×16。
- raw_joint、current_support_masked、identity residual、Gaussian32保持固定。
- Stage B对全部36个cell按Q3冻结epoch和G/D LR轨迹重新训练至Q3末。
- Q4所有模型使用同一common 5m面板；5m训练lane为primary，30m训练lane仅作secondary。
"""
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    md_path = report / "film_nolp_capacity_conclusion.md"
    md_path.write_text(markdown, encoding="utf-8")
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>FiLM G + NoLP Capacity</title>
<style>body{{font:15px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd3df;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1,h2{{color:#153e75}}code{{background:#edf2f7;padding:2px 4px}}</style></head><body>
<h1>FiLM G + NoLP Critic：多 Seed 容量实验</h1>
<p>Q3 point leader：<code>{escape(str(selection["point_leader"]))}</code>；one-SE candidate：<code>{escape(str(selection["one_se_candidate"]))}</code>；证据：<code>{escape(str(selection["candidate_statistical_support"]))}</code>。</p>
<h2>Q3容量排名</h2>{_html_table(q3, columns)}
<h2>Q3相对persistence</h2>{_html_table(q3_persistence, contrast_columns)}
<h2>Q4回顾性评价</h2>{_html_table(q4, columns)}
<h2>Q4相对persistence</h2>{_html_table(q4_persistence, contrast_columns)}
<p>Q4与历史预测pair重合：{escape(str(q4_summary["historical_q4_exposure"]["pair_overlap_label"]))}。这是retrospective frozen exploratory评价，不是confirmatory holdout，不能用于重新选型。</p>
</body></html>"""
    html_path = report / "film_nolp_capacity_conclusion.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


__all__ = ["render_report"]
