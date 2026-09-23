"""Self-contained Q3 stage report for the news-first LR sweep."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class LearningRateReportError(ValueError):
    """Raised when LR artifacts cannot support a fail-closed report."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LearningRateReportError(f"Expected JSON object: {path}")
    return dict(payload)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "—"
        if numeric != 0.0 and abs(numeric) < 1.0e-3:
            return f"{numeric:.3e}"
        return f"{numeric:.6f}"
    return str(value)


def _table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    empty: str = "无可用记录",
) -> str:
    available = [(key, label) for key, label in columns if key in frame.columns]
    if frame.empty or not available:
        return f'<p class="muted">{escape(empty)}</p>'
    head = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    body: list[str] = []
    for row in frame.to_dict(orient="records"):
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(_fmt(row.get(key)))}</td>" for key, _ in available)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _truthy(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value is True
        or value == 1
        or str(value).strip().lower() in {"true", "yes", "1"}
    )


def _lr_curve_svg(frame: pd.DataFrame) -> str:
    required = {
        "initial_learning_rate",
        "mae_ratio",
        "tolerance_minutes",
        "text_ablation_mode",
    }
    if frame.empty or not required.issubset(frame.columns):
        return '<p class="muted">无可绘制的LR曲线</p>'
    plot = frame.copy()
    for column in ("initial_learning_rate", "mae_ratio", "tolerance_minutes"):
        plot[column] = pd.to_numeric(plot[column], errors="coerce")
    plot = plot[
        np.isfinite(plot["initial_learning_rate"])
        & (plot["initial_learning_rate"] > 0.0)
        & np.isfinite(plot["mae_ratio"])
        & np.isfinite(plot["tolerance_minutes"])
    ].copy()
    if plot.empty:
        return '<p class="muted">无可绘制的LR曲线</p>'
    plot["lr_log10"] = np.log10(plot["initial_learning_rate"])
    width, height = 860.0, 390.0
    left, right, top, bottom = 72.0, 26.0, 30.0, 78.0
    x_min = float(plot["lr_log10"].min())
    x_max = float(plot["lr_log10"].max())
    if math.isclose(x_min, x_max):
        x_min -= 0.2
        x_max += 0.2
    values = plot["mae_ratio"].to_numpy(dtype=float)
    y_min = min(float(values.min()), 1.0)
    y_max = max(float(values.max()), 1.0)
    padding = max(0.0015, (y_max - y_min) * 0.15)
    y_min -= padding
    y_max += padding

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    palette = {
        ("current_only", 5): "#175cd3",
        ("current_only", 30): "#039855",
        ("real_text", 5): "#b54708",
        ("real_text", 30): "#7f56d9",
        ("current_only", 10): "#0e7090",
        ("current_only", 15): "#c11574",
        ("real_text", 10): "#93370d",
        ("real_text", 15): "#5925dc",
    }
    elements = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        'aria-label="Q3 MAE ratio by initial learning rate">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#667085"/>',
    ]
    baseline = y_coord(1.0)
    elements.append(
        f'<line x1="{left}" y1="{baseline:.2f}" x2="{width-right}" '
        f'y2="{baseline:.2f}" stroke="#b42318" stroke-dasharray="5 4"/>'
    )
    elements.append(
        f'<text x="{width-right-3}" y="{baseline-6:.2f}" text-anchor="end" '
        'font-size="11" fill="#b42318">persistence = 1</text>'
    )
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4.0
        y = y_coord(value)
        elements.append(
            f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#667085">{value:.4f}</text>'
        )
    legend_index = 0
    groups = plot.groupby(["text_ablation_mode", "tolerance_minutes"], sort=True)
    for (raw_mode, raw_tolerance), group in groups:
        mode = str(raw_mode)
        tolerance = int(raw_tolerance)
        color = palette.get((mode, tolerance), "#344054")
        ordered = group.sort_values("lr_log10", kind="stable")
        points = " ".join(
            f"{x_coord(float(row.lr_log10)):.2f},{y_coord(float(row.mae_ratio)):.2f}"
            for row in ordered.itertuples()
        )
        dash = ' stroke-dasharray="6 4"' if mode == "real_text" else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        for row in ordered.itertuples():
            label = (
                f"{row.lr_profile}; {mode}; {tolerance}m; "
                f"MAE ratio={float(row.mae_ratio):.6f}"
            )
            elements.append(
                f'<circle cx="{x_coord(float(row.lr_log10)):.2f}" '
                f'cy="{y_coord(float(row.mae_ratio)):.2f}" r="4" fill="{color}">'
                f"<title>{escape(label)}</title></circle>"
            )
        legend_x = left + (legend_index % 4) * 185.0
        legend_y = height - 34.0 + (legend_index // 4) * 17.0
        elements.append(
            f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" '
            f'x2="{legend_x+20:.2f}" y2="{legend_y:.2f}" stroke="{color}"{dash}/>'
            f'<text x="{legend_x+25:.2f}" y="{legend_y+4:.2f}" font-size="10" '
            f'fill="#344054">{escape(mode)} · {tolerance}m</text>'
        )
        legend_index += 1
    profiles = (
        plot[["lr_profile", "initial_learning_rate", "lr_log10"]]
        .drop_duplicates()
        .sort_values("lr_log10", kind="stable")
    )
    for row in profiles.itertuples():
        x = x_coord(float(row.lr_log10))
        elements.append(
            f'<text x="{x:.2f}" y="{height-bottom+18:.2f}" text-anchor="middle" '
            f'font-size="10" fill="#667085">{float(row.initial_learning_rate):.0e}</text>'
        )
    elements.extend(
        (
            f'<text x="{(left+width-right)/2:.2f}" y="{height-48:.2f}" '
            'text-anchor="middle" font-size="12">initial learning rate (log scale)</text>',
            f'<text transform="translate(18 {(top+height-bottom)/2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12">MAE / persistence MAE</text>',
            "</svg>",
        )
    )
    return '<div class="chart-wrap">' + "".join(elements) + "</div>"


def _stage(selection: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    stages = selection.get("stages")
    if not isinstance(stages, Mapping):
        raise LearningRateReportError("lr_selection lacks stages")
    if isinstance(stages.get("lr_confirm"), Mapping):
        return "lr_confirm", dict(stages["lr_confirm"])
    if isinstance(stages.get("lr_screen"), Mapping):
        return "lr_screen", dict(stages["lr_screen"])
    raise LearningRateReportError("No frozen LR stage is available")


def render_lr_sweep_report(experiment_root: str | Path) -> Path:
    """Render a portable report using Q3 selection artifacts only."""

    root = Path(experiment_root).resolve()
    selection = _read_json(root / "lr_selection.json")
    if bool(selection.get("q4_used_for_selection", False)):
        raise LearningRateReportError("LR report refuses selection that accessed Q4")
    stage_name, stage = _stage(selection)
    if stage.get("q4_used_for_selection") is not False:
        raise LearningRateReportError("LR stage must explicitly declare no Q4 access")
    if str(stage.get("selection_panel", "")) != "common_validation_05m":
        raise LearningRateReportError("LR report permits common_validation_05m only")
    comparisons = _read_csv(root / "lr_comparisons.csv")
    manifest = _read_csv(root / "lr_profile_manifest.csv")
    audit = _read_csv(root / "lr_selection_audit.csv")
    current = comparisons[
        comparisons["selection_stage"].astype(str).eq(stage_name)
    ].copy()
    if current.empty:
        raise LearningRateReportError(f"No comparisons for {stage_name}")
    gate = bool(stage.get("gate_passed"))
    selected = str(stage.get("selected_lr_profile", "")).strip()
    formal = str(stage.get("winner_lr_profile", "")).strip()
    leader = str(stage.get("ranked_leader_lr_profile", "")).strip()
    if gate and not selected:
        raise LearningRateReportError("Passing LR stage lacks selected_lr_profile")
    if not gate and selected:
        raise LearningRateReportError("Failed LR stage must not retain a selection")
    if stage_name == "lr_confirm" and gate and not formal:
        raise LearningRateReportError("Passing confirm lacks formal LR winner")

    candidate_frame = pd.DataFrame(stage.get("candidate_lr_profiles", []))
    if candidate_frame.empty:
        raise LearningRateReportError("LR stage lacks candidate summaries")
    diagnostic = current.sort_values(
        ["initial_learning_rate", "text_ablation_mode", "tolerance_minutes"],
        kind="stable",
    )
    epoch0_rate = (
        float(_truthy(current["baseline_inclusive_epoch0_selected"]).mean())
        if "baseline_inclusive_epoch0_selected" in current.columns
        else float("nan")
    )
    if gate:
        if stage_name == "lr_screen":
            headline = f"Q3 screen 通过：暂定 {selected}，需四容差 confirm 后才能成为正式赢家。"
        else:
            headline = f"Q3 confirm 通过：正式学习率赢家为 {formal}。"
    else:
        headline = f"Q3 {stage_name} gate 未通过：没有学习率赢家；" f"{leader or '排名首位候选'} 仅是排名诊断。"
    status_path = root / "registry" / "experiment_status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    if status.get("q4_accessed") is True:
        raise LearningRateReportError("Experiment status reports Q4 access")

    candidate_table = _table(
        candidate_frame,
        (
            ("lr_profile", "LR档"),
            ("initial_learning_rate", "初始LR"),
            ("geometric_mae_ratio", "Current-only MAE ratio"),
            ("mean_improvement_fraction", "平均改善"),
            ("all_tolerances_not_worse", "各容差不差"),
            ("lr_gate_passed", "Gate"),
            ("real_text_geometric_mae_ratio", "Real-text MAE ratio（诊断）"),
            ("real_text_minus_current_log_ratio", "Text−Current log ratio"),
        ),
    )
    diagnostics_table = _table(
        diagnostic,
        (
            ("lr_profile", "LR档"),
            ("initial_learning_rate", "初始LR"),
            ("text_ablation_mode", "模式"),
            ("tolerance_minutes", "容差(min)"),
            ("model_mae", "Best-learned MAE"),
            ("persistence_mae", "Persistence MAE"),
            ("mae_ratio", "MAE ratio"),
            ("best_learned_epoch", "Best epoch"),
            ("best_learned_train_recon", "Train recon"),
            ("best_learned_val_recon", "Val recon"),
            ("best_learned_train_val_gap", "Val−Train gap"),
            ("baseline_inclusive_best_epoch", "Baseline-inclusive epoch"),
            ("runtime_minutes", "Runtime(min)"),
            ("peak_memory_mib", "Peak memory(MiB)"),
        ),
    )
    manifest_table = _table(
        manifest,
        (
            ("lr_profile", "LR档"),
            ("initial_learning_rate", "初始LR"),
            ("scheduler_min_lr", "Scheduler floor"),
            ("capacity_profile", "容量"),
            ("screen_completed_job_count", "Screen完成任务"),
            ("confirm_completed_job_count", "Confirm完成任务"),
            ("lr_profile_sha256", "Profile SHA-256"),
        ),
    )
    audit_table = _table(
        audit,
        (
            ("lr_stage", "Stage"),
            ("evaluated_at_utc", "评估时间"),
            ("candidate_lr_count", "候选数"),
            ("gate_passing_lr_count", "过Gate数"),
            ("selected_lr_profile", "冻结候选"),
            ("winner_lr_profile", "正式赢家"),
            ("ranked_leader_lr_profile", "排名首位"),
            ("q4_rows_passed_to_evaluator", "传入Q4行"),
        ),
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>News-first Vol Learning-rate Sweep — Q3-only</title>
<style>
body{{margin:0;background:#f7f8fa;color:#101828;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 24px 56px}}h1{{font-size:28px;margin:0 0 8px}}h2{{margin-top:34px;font-size:20px}}
.lead{{font-size:17px;background:#fff;border-left:5px solid {'#039855' if gate else '#d92d20'};padding:16px 18px;border-radius:8px;box-shadow:0 1px 3px #10182818}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}}.card{{background:#fff;border:1px solid #e4e7ec;border-radius:9px;padding:14px}}
.label{{color:#667085;font-size:12px;text-transform:uppercase}}.value{{font-size:18px;font-weight:650;margin-top:4px}}.muted{{color:#667085}}
.note{{background:#fffaeb;border:1px solid #fedf89;padding:12px 14px;border-radius:8px}}.table-wrap{{overflow:auto;background:#fff;border:1px solid #e4e7ec;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px 10px;border-bottom:1px solid #eaecf0;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#f9fafb;color:#475467;position:sticky;top:0}}
.chart-wrap{{background:#fff;border:1px solid #e4e7ec;border-radius:8px;padding:8px}}svg{{display:block;width:100%;height:auto}}
code{{background:#eef2f6;padding:2px 5px;border-radius:4px}}footer{{margin-top:36px;color:#667085;font-size:12px}}
</style></head><body><main>
<h1>News-first Vol 学习率 Sweep</h1>
<p class="muted">独立 Q3-only、raw_joint support、large Regression、best-learned epoch ≥ 1</p>
<p class="lead">{escape(headline)}</p>
<div class="cards">
<div class="card"><div class="label">当前阶段</div><div class="value">{escape(stage_name)}</div></div>
<div class="card"><div class="label">选择口径</div><div class="value">current_only</div></div>
<div class="card"><div class="label">选择容差</div><div class="value">{escape(str(stage.get('selection_tolerances', [])))}</div></div>
<div class="card"><div class="label">Baseline-inclusive epoch 0比例</div><div class="value">{escape(_fmt(epoch0_rate))}</div></div>
</div>
<p class="note"><b>数据边界：</b>学习率选择与预测只使用 2023Q3 <code>common_validation_05m</code>；训练 loader 会物化 Q4 test samples，但 Q4 未生成预测、未评估、未参与选择。Real-text 仅作诊断，不参与学习率选择。Gate 要求几何平均改善至少0.5%，且每个选择容差均不差于 persistence。</p>
<h2>LR—MAE ratio 曲线</h2>{_lr_curve_svg(current)}
<h2>候选汇总</h2>{candidate_table}
<h2>Best-learned、过拟合与资源诊断</h2>{diagnostics_table}
<h2>冻结LR血缘</h2>{manifest_table}
<h2>选择审计</h2>{audit_table}
<footer>状态：{escape(str(status.get('status', '未提供')))} · 自包含HTML，无外部资源 · Q4 used for selection = false</footer>
</main></body></html>"""
    output = root / "report" / "lr_sweep_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


render_lr_report = render_lr_sweep_report


__all__ = [
    "LearningRateReportError",
    "render_lr_report",
    "render_lr_sweep_report",
]
