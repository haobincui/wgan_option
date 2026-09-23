"""Self-contained Q3 report for the legacy 2x2 architecture experiment."""

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
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(row[column]) for column in columns) + " |"
        for _, row in frame.loc[:, list(columns)].iterrows()
    ]
    return "\n".join([header, rule, *rows])


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
        (analysis / "legacy_architecture_q3_selection.json").read_text(encoding="utf-8")
    )
    scores = pd.read_csv(analysis / "legacy_architecture_q3_cell_scores.csv")
    contrasts = pd.read_csv(
        analysis / "legacy_architecture_q3_candidate_anchor_contrasts.csv"
    )
    persistence = pd.read_csv(
        analysis / "legacy_architecture_q3_persistence_contrasts.csv"
    )
    factorial = pd.read_csv(
        analysis / "legacy_architecture_q3_secondary_factorial_effects.csv"
    )
    score_columns = (
        "architecture_cell",
        "mean_model_mae",
        "mean_persistence_mae",
        "geometric_mae_ratio",
        "improvement_vs_persistence_fraction",
    )
    contrast_columns = (
        "candidate_cell",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_holm",
        "nonworse_seed_count",
        "statistically_supported",
    )
    persistence_columns = (
        "architecture_cell",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_holm",
        "nonworse_seed_count",
        "statistically_supported",
    )
    factorial_columns = (
        "factorial_effect",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_holm",
        "statistically_nonzero",
    )
    supported = selection["statistically_supported_candidates"]
    supported_text = ", ".join(supported) if supported else "无"
    markdown = f"""# Legacy 2×2 Generator×Critic 多 Seed 实验

## 结论先行

Q3共同5m面板上的点估计leader为 `{selection["point_leader"]}`。相对原始
`{selection["anchor_cell"]}` anchor，通过预注册门槛的candidate为：`{supported_text}`。
整体状态为 `{selection["selection_status"]}`。

## 四个架构cell

{_markdown_table(scores, score_columns)}

## Candidate − Anchor 配对检验

{_markdown_table(contrasts, contrast_columns)}

三项比较组成一个Holm-3 family；bootstrap按seed后CME session两级配对重采样10,000次。
通过门槛要求平均MAE差<0、95% CI上界<0、Holm p<0.05且至少2/3 seeds不劣。

## Secondary：各cell相对persistence

{_markdown_table(persistence, persistence_columns)}

这四项构成独立的secondary Holm-4 family，不改变架构选择。

## Secondary：2×2 factorial归因

{_markdown_table(factorial, factorial_columns)}

Generator主效应、Critic主效应和G×D interaction构成独立的secondary Holm-3 family；
它们用于架构归因，不替换candidate−anchor primary family，也不改变选择。

## 固定边界与解释

- legacy widths、real_text、5m、exact-TTM 16×16、LR=5e-7。
- `raw_joint`、`current_support_masked`、identity residual和Gaussian32保持不变。
- FiLM G + NoLP Critic的三个seed来自正式容量实验的精确SHA只读reference；其他9项重新训练。
- legacy来自同一Q3容量筛选的post-selection点估计leader，但当时仅为`descriptive_only`；
  因此这是探索性动机，不是确认性架构先验。
- 本实验只读Q3，属于探索性架构检验；Q4窗口、loader、预测和评价均被禁止。
"""
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / "legacy_architecture_q3_conclusion.md"
    md_path.write_text(markdown, encoding="utf-8")
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Legacy Architecture 2x2</title>
<style>body{{font:15px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #ccd3df;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1,h2{{color:#153e75}}code{{background:#edf2f7;padding:2px 4px}}</style></head><body>
<h1>Legacy 2×2 Generator×Critic 多 Seed 实验</h1>
<p>Q3 point leader：<code>{escape(str(selection["point_leader"]))}</code>；通过门槛的candidate：<code>{escape(supported_text)}</code>；状态：<code>{escape(str(selection["selection_status"]))}</code>。</p>
<h2>四个架构cell</h2>{_html_table(scores, score_columns)}
<h2>Candidate − Anchor</h2>{_html_table(contrasts, contrast_columns)}
<h2>Secondary：各cell相对persistence</h2>{_html_table(persistence, persistence_columns)}
<h2>Secondary：2×2 factorial归因</h2>{_html_table(factorial, factorial_columns)}
<p>Primary inference使用Holm-3与10,000次seed→CME-session paired bootstrap。本实验Q4访问严格为零。</p>
</body></html>"""
    html_path = report_dir / "legacy_architecture_q3_conclusion.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


__all__ = ["render_report"]
