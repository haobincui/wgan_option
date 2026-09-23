"""Portable HTML report for the news-first capacity experiment."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class CapacityReportError(ValueError):
    """Raised when final capacity artifacts cannot support a report."""


NO_LEARNED_CAPACITY_STATUS = "completed_no_learned_capacity"
CAPACITY_STAGES = (
    "regression_screen",
    "regression_confirm",
    "wgan_screen",
    "wgan_confirm",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CapacityReportError(f"Expected JSON object: {path}")
    return dict(payload)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if abs(numeric) < 1.0e-3 and numeric != 0.0:
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
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _stage_cards(selection: Mapping[str, Any]) -> str:
    cards: list[str] = []
    stages = selection.get("stages", {})
    if not isinstance(stages, Mapping):
        return ""
    for name in (
        "regression_screen",
        "regression_confirm",
        "wgan_screen",
        "wgan_confirm",
    ):
        raw = stages.get(name)
        if not isinstance(raw, Mapping):
            continue
        gate = bool(raw.get("gate_passed"))
        candidate_label = "正式赢家" if gate else "排名首位候选（未入选）"
        candidate = (
            raw.get("winner_profile") if gate else raw.get("ranked_leader_profile", "—")
        )
        parameter_count = (
            raw.get("winner_parameter_count")
            if gate
            else raw.get("ranked_leader_parameter_count")
        )
        cards.append(
            '<article class="card">'
            f"<h3>{escape(name)}</h3>"
            f'<div class="status {"pass" if gate else "fail"}">{"PASS" if gate else "FAIL"}</div>'
            f"<p><b>{candidate_label}：</b>{escape(str(candidate or '—'))}</p>"
            f"<p><b>参数量：</b>{escape(_fmt(parameter_count))}</p>"
            f"<p><b>选择模式：</b>{escape(str(raw.get('selection_text_ablation_mode', '')))}</p>"
            f"<p><b>容差：</b>{escape(str(raw.get('selection_tolerances', [])))}</p>"
            "</article>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _truthy_series(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value is True
        or value == 1
        or str(value).strip().lower() in {"true", "yes", "1"}
    )


def _mae_ratio_svg(frame: pd.DataFrame) -> str:
    required = {
        "capacity_profile",
        "parameter_log10",
        "mae_ratio",
        "tolerance_minutes",
    }
    if frame.empty or not required.issubset(frame.columns):
        return '<p class="muted">无可绘制的容量—MAE ratio记录</p>'
    plot = frame.copy()
    plot["parameter_log10"] = pd.to_numeric(plot["parameter_log10"], errors="coerce")
    plot["mae_ratio"] = pd.to_numeric(plot["mae_ratio"], errors="coerce")
    plot["tolerance_minutes"] = pd.to_numeric(
        plot["tolerance_minutes"], errors="coerce"
    )
    plot = plot[
        np.isfinite(plot["parameter_log10"])
        & np.isfinite(plot["mae_ratio"])
        & np.isfinite(plot["tolerance_minutes"])
    ]
    if plot.empty:
        return '<p class="muted">无可绘制的容量—MAE ratio记录</p>'

    width, height = 820.0, 360.0
    left, right, top, bottom = 66.0, 24.0, 28.0, 62.0
    x_min = float(plot["parameter_log10"].min())
    x_max = float(plot["parameter_log10"].max())
    if math.isclose(x_min, x_max):
        x_min -= 0.2
        x_max += 0.2
    ratios = plot["mae_ratio"].to_numpy(dtype=float)
    y_min = min(float(ratios.min()), 1.0)
    y_max = max(float(ratios.max()), 1.0)
    padding = max(0.002, (y_max - y_min) * 0.12)
    y_min -= padding
    y_max += padding

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    palette = ("#275efe", "#087443", "#b54708", "#7f56d9")
    elements = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        'aria-label="MAE ratio versus model capacity">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#667085"/>',
    ]
    baseline_y = y_coord(1.0)
    elements.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{width - right}" '
        f'y2="{baseline_y:.2f}" stroke="#b42318" stroke-dasharray="5 4"/>'
    )
    elements.append(
        f'<text x="{width - right - 4}" y="{baseline_y - 5:.2f}" text-anchor="end" '
        'font-size="11" fill="#b42318">persistence = 1</text>'
    )
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4.0
        y = y_coord(value)
        elements.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#667085">{value:.3f}</text>'
        )
    for color_index, (tolerance, group) in enumerate(
        plot.groupby("tolerance_minutes", sort=True)
    ):
        color = palette[color_index % len(palette)]
        ordered = group.sort_values("parameter_log10", kind="stable")
        points = " ".join(
            f"{x_coord(float(row.parameter_log10)):.2f},{y_coord(float(row.mae_ratio)):.2f}"
            for row in ordered.itertuples()
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2" opacity="0.75"/>'
        )
        for row in ordered.itertuples():
            x = x_coord(float(row.parameter_log10))
            y = y_coord(float(row.mae_ratio))
            label = (
                f"{row.capacity_profile}; {int(tolerance)}m; "
                f"MAE ratio={float(row.mae_ratio):.6f}"
            )
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}">'
                f"<title>{escape(label)}</title></circle>"
            )
        legend_x = left + color_index * 130.0
        elements.append(
            f'<circle cx="{legend_x:.2f}" cy="{height - 18:.2f}" r="4" fill="{color}"/>'
            f'<text x="{legend_x + 8:.2f}" y="{height - 14:.2f}" font-size="11" '
            f'fill="#344054">{int(tolerance)}m</text>'
        )
    profiles = (
        plot[["capacity_profile", "parameter_log10"]]
        .drop_duplicates()
        .sort_values("parameter_log10", kind="stable")
    )
    for row in profiles.itertuples():
        x = x_coord(float(row.parameter_log10))
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 18:.2f}" text-anchor="middle" '
            f'font-size="10" fill="#667085">{escape(str(row.capacity_profile))}</text>'
        )
    elements.extend(
        (
            f'<text x="{(left + width - right) / 2:.2f}" y="{height - 34:.2f}" '
            'text-anchor="middle" font-size="12">log10(parameters)</text>',
            f'<text transform="translate(17 {(top + height - bottom) / 2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12">MAE ratio</text>',
            "</svg>",
        )
    )
    return '<div class="chart-wrap">' + "".join(elements) + "</div>"


def _stage_only_report(root: Path, selection: Mapping[str, Any]) -> Path:
    experiment_status = _read_json(root / "registry" / "experiment_status.json")
    stage_status = _read_json(root / "capacity_stage_status.json")
    if experiment_status.get("status") != NO_LEARNED_CAPACITY_STATUS:
        raise CapacityReportError(
            "Final Q4 artifacts are absent, but experiment status is not "
            f"{NO_LEARNED_CAPACITY_STATUS}"
        )
    if stage_status.get("status") != NO_LEARNED_CAPACITY_STATUS:
        raise CapacityReportError(
            "Experiment and capacity-stage terminal statuses disagree"
        )
    if bool(selection.get("q4_used_for_selection", False)):
        raise CapacityReportError("Capacity selection must not use Q4")

    stages = selection.get("stages")
    status_stages = stage_status.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(status_stages, Mapping):
        raise CapacityReportError("Capacity selection/status lacks stage mappings")
    if any(
        name in stages or name in status_stages
        for name in ("wgan_screen", "wgan_confirm")
    ):
        raise CapacityReportError(
            "No-learned-capacity report requires WGAN to remain unrun"
        )
    terminal_stage = str(experiment_status.get("current_stage", ""))
    if terminal_stage not in {"regression_screen", "regression_confirm"}:
        raise CapacityReportError(
            f"Invalid no-learned-capacity terminal stage: {terminal_stage!r}"
        )
    if str(stage_status.get("current_stage", "")) != terminal_stage:
        raise CapacityReportError("Experiment and capacity current stages disagree")
    terminal_selection = stages.get(terminal_stage)
    terminal_status = status_stages.get(terminal_stage)
    if not isinstance(terminal_selection, Mapping) or not isinstance(
        terminal_status, Mapping
    ):
        raise CapacityReportError("Failed Regression stage is not frozen")
    if bool(terminal_selection.get("gate_passed")):
        raise CapacityReportError(
            "completed_no_learned_capacity requires a failed Regression gate"
        )
    if terminal_status.get("status") != NO_LEARNED_CAPACITY_STATUS:
        raise CapacityReportError("Failed Regression stage is not terminal")
    if any(
        terminal_selection.get(key) not in (None, "", [])
        for key in ("winner_profile", "capacity_profile", "selected_profiles")
    ):
        raise CapacityReportError(
            "Failed gate must not declare a formal capacity winner"
        )
    ranked_leader = str(terminal_selection.get("ranked_leader_profile", "")).strip()
    if not ranked_leader:
        raise CapacityReportError("Failed gate lacks ranked-leader diagnostics")

    comparisons = _read_csv(root / "capacity_comparisons.csv")
    profiles = _read_csv(root / "capacity_profile_manifest.csv")
    resources = _read_csv(root / "resource_summary.csv")
    required_diagnostics = {
        "selection_stage",
        "capacity_profile",
        "model_family",
        "text_ablation_mode",
        "tolerance_minutes",
        "parameter_count",
        "parameter_log10",
        "mae_ratio",
        "best_learned_epoch",
        "best_learned_train_recon",
        "best_learned_val_recon",
        "best_learned_train_val_gap",
        "baseline_inclusive_best_epoch",
        "baseline_inclusive_epoch0_selected",
        "runtime_minutes",
        "peak_memory_mib",
    }
    missing = sorted(required_diagnostics - set(comparisons.columns))
    if missing:
        raise CapacityReportError(
            f"capacity_comparisons lacks stage diagnostics: {missing}"
        )
    profile_required = {
        "capacity_profile",
        "expected_regression_parameters",
        "expected_wgan_parameters",
        "capacity_profile_sha256",
    }
    resource_required = {
        "job_id",
        "model",
        "capacity_stage",
        "capacity_profile",
        "runtime_minutes",
        "peak_memory_mib",
        "status",
    }
    if not profile_required.issubset(
        profiles.columns
    ) or not resource_required.issubset(resources.columns):
        raise CapacityReportError("Profile/resource audit schema is incomplete")
    for frame, label, column in (
        (comparisons, "capacity comparisons", "model_family"),
        (resources, "resource summary", "model"),
    ):
        if (
            column in frame.columns
            and frame[column].astype(str).str.lower().eq("wgan").any()
        ):
            raise CapacityReportError(f"{label} unexpectedly contains WGAN rows")

    terminal_rows = comparisons.copy()
    if "selection_stage" in terminal_rows.columns:
        terminal_rows = terminal_rows[
            terminal_rows["selection_stage"].astype(str).eq(terminal_stage)
        ]
    terminal_rows = terminal_rows[
        terminal_rows["model_family"].astype(str).eq("regression")
    ].copy()
    if terminal_rows.empty:
        raise CapacityReportError("No Regression diagnostics for failed stage")
    selection_mode = str(
        terminal_selection.get("selection_text_ablation_mode", "real_text")
    )
    chart_rows = terminal_rows[
        terminal_rows.get(
            "text_ablation_mode", pd.Series("", index=terminal_rows.index)
        )
        .astype(str)
        .eq(selection_mode)
    ].copy()
    if chart_rows.empty:
        raise CapacityReportError("No selection-mode MAE ratios for failed stage")

    epoch0 = _truthy_series(terminal_rows["baseline_inclusive_epoch0_selected"])
    gap = pd.to_numeric(terminal_rows["best_learned_train_val_gap"], errors="coerce")
    runtime = pd.to_numeric(terminal_rows["runtime_minutes"], errors="coerce")
    memory = pd.to_numeric(terminal_rows["peak_memory_mib"], errors="coerce")
    epoch0_ratio = float(epoch0.mean()) if len(epoch0) else float("nan")
    median_gap = float(gap.dropna().median()) if gap.notna().any() else float("nan")
    runtime_total = (
        float(runtime.dropna().sum()) if runtime.notna().any() else float("nan")
    )
    peak_memory = float(memory.dropna().max()) if memory.notna().any() else float("nan")

    status_table = pd.DataFrame(
        [
            {
                "scope": "experiment",
                "status": experiment_status.get("status"),
                "current_stage": experiment_status.get("current_stage"),
                "completed_at_utc": experiment_status.get("completed_at_utc"),
            },
            {
                "scope": "capacity_stage",
                "status": stage_status.get("status"),
                "current_stage": stage_status.get("current_stage"),
                "completed_at_utc": terminal_status.get("completed_at_utc"),
            },
        ]
    )
    comparison_view = terminal_rows.sort_values(
        [
            column
            for column in (
                "text_ablation_mode",
                "tolerance_minutes",
                "parameter_count",
            )
            if column in terminal_rows.columns
        ],
        kind="stable",
    )
    profile_view = profiles.sort_values("expected_regression_parameters", kind="stable")
    resource_view = resources.copy()
    if "capacity_stage" in resource_view.columns:
        resource_view = resource_view[
            resource_view["capacity_stage"].astype(str).eq(terminal_stage)
        ]

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News-first Vol 容量 Gate 失败终态报告</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--bg:#f5f7fb;--panel:#fff;--blue:#275efe;--green:#087443;--red:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}} h1{{font-size:30px;margin:0 0 8px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{margin:0 0 10px}}
.lead{{font-size:17px;background:#fff1f0;border-left:4px solid var(--red);padding:18px 20px;border-radius:6px}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px #1720330d}}
.status{{display:inline-block;border-radius:999px;padding:3px 9px;font-weight:700}} .status.pass{{background:#dff7ea;color:var(--green)}} .status.fail{{background:#fee4e2;color:var(--red)}}
.table-wrap,.chart-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:#edf1f7}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}} .kpi{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px}} .kpi b{{display:block;font-size:20px}}
code{{background:#edf1f7;padding:1px 4px;border-radius:3px}} footer{{margin-top:36px;color:var(--muted)}} svg{{display:block;width:100%;min-width:680px;height:auto}}
</style>
</head>
<body><main>
<h1>News-first Vol 容量 Gate 失败终态报告</h1>
<p class="muted">实验目录：{escape(str(root))}</p>
<div class="lead"><b>有效终态：</b>{escape(terminal_stage)} 的 Regression 容量 gate 未通过。<b>正式容量赢家：无；WGAN：未运行；Q4：未读取、未评估。</b> {escape(ranked_leader)} 只是最接近预注册目标的排名首位候选，不是赢家。本报告只使用Q3 stage审计产物，未打开任何 <code>final_q4_*</code> 文件。</div>

<h2>Stage状态与选择审计</h2>
{_stage_cards(selection)}
{_table(status_table, (("scope", "Scope"), ("status", "Status"), ("current_stage", "Current stage"), ("completed_at_utc", "Completed UTC")))}

<h2>失败Stage诊断摘要</h2>
<div class="kpis">
<div class="kpi">Baseline-inclusive选择epoch 0比例<b>{escape(_fmt(epoch0_ratio))}</b></div>
<div class="kpi">Best-learned Val−Train gap中位数<b>{escape(_fmt(median_gap))}</b></div>
<div class="kpi">诊断任务总运行分钟<b>{escape(_fmt(runtime_total))}</b></div>
<div class="kpi">观测到的峰值显存MiB<b>{escape(_fmt(peak_memory))}</b></div>
</div>

<h2>MAE ratio 与模型容量</h2>
<p>仅展示失败stage用于选择的 <code>{escape(selection_mode)}</code> 行；低于红色1.0基准线表示优于persistence，但所有正式gate条件未同时满足，因此不产生赢家。</p>
{_mae_ratio_svg(chart_rows)}

<h2>训练与泛化诊断</h2>
{_table(comparison_view, (("capacity_profile", "Profile"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("parameter_count", "Parameters"), ("parameter_log10", "log10(parameters)"), ("best_learned_epoch", "Learned epoch"), ("best_learned_train_recon", "Train recon"), ("best_learned_val_recon", "Val recon"), ("best_learned_train_val_gap", "Val−Train gap"), ("baseline_inclusive_best_epoch", "Inclusive best epoch"), ("baseline_inclusive_epoch0_selected", "Epoch 0 selected"), ("mae_ratio", "MAE ratio"), ("runtime_minutes", "Runtime min"), ("peak_memory_mib", "Peak MiB"), ("profile_gate_passed", "Gate")))}

<h2>容量Profile定义</h2>
{_table(profile_view, (("capacity_profile", "Profile"), ("expected_regression_parameters", "Regression parameters"), ("expected_wgan_parameters", "WGAN parameters"), ("gen_base_channels", "Gen channels"), ("gen_hidden_dim", "Gen hidden"), ("disc_hidden_dim", "Critic hidden"), ("capacity_profile_sha256", "Profile SHA-256")))}

<h2>资源审计</h2>
{_table(resource_view, (("job_id", "Job"), ("capacity_profile", "Profile"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("runtime_minutes", "Runtime min"), ("gpu_hours", "GPU hours"), ("peak_memory_mib", "Peak MiB"), ("mean_utilization_gpu_pct", "Mean GPU %"), ("status", "Status")))}

<h2>解释边界</h2>
<ul>
<li>本终态没有通过预注册的平均改善0.5%且各主容差不劣于persistence的gate，因此不得把排名首位候选称为正式赢家。</li>
<li>没有启动WGAN stage，也没有读取Q4；该报告不能提供样本外测试结论。</li>
<li>Train/validation gap定义为best-learned epoch的 <code>val_recon − train_recon</code>；WGAN若未来运行，其train recon标准化映射自 <code>g_recon</code>。</li>
<li>GPU资源遥测按同卡并行wave采样，峰值显存与利用率不是进程独占测量。</li>
</ul>
<footer>输入：capacity_selection.json、capacity_comparisons.csv、capacity_profile_manifest.csv、resource_summary.csv、capacity_stage_status.json、registry/experiment_status.json。HTML不依赖外部CSS、脚本或网络资源。</footer>
</main></body></html>"""
    output = root / "report" / "capacity_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def render_capacity_report(experiment_root: str | Path) -> Path:
    """Render a self-contained, conclusion-first HTML report."""

    root = Path(experiment_root).resolve()
    selection = _read_json(root / "capacity_selection.json")
    analysis = root / "analysis"
    final_artifacts = sorted(
        path for path in analysis.glob("final_q4_*") if path.is_file()
    )
    validation_path = analysis / "final_q4_validation_summary.json"
    if not validation_path.is_file():
        if final_artifacts:
            raise CapacityReportError(
                "Partial final_q4 artifacts exist without validation summary"
            )
        return _stage_only_report(root, selection)

    required_final_paths = (
        validation_path,
        analysis / "final_q4_pair_metrics.csv.gz",
        analysis / "final_q4_model_comparison.csv",
        analysis / "final_q4_persistence_bootstrap.csv",
        analysis / "final_q4_text_ablation_bootstrap.csv",
    )
    missing_final = [str(path) for path in required_final_paths if not path.is_file()]
    if missing_final:
        raise CapacityReportError(
            f"Final Q4 artifact set is incomplete: {missing_final}"
        )

    validation = _read_json(validation_path)
    if str(validation.get("status", "")).lower() != "pass":
        raise CapacityReportError("Final Q4 validation did not pass")
    if bool(selection.get("q4_used_for_selection", False)):
        raise CapacityReportError("Capacity selection must not use Q4")

    comparisons = _read_csv(root / "capacity_comparisons.csv")
    q4_summary = _read_csv(root / "analysis" / "final_q4_model_comparison.csv")
    persistence = _read_csv(root / "analysis" / "final_q4_persistence_bootstrap.csv")
    text = _read_csv(root / "analysis" / "final_q4_text_ablation_bootstrap.csv")
    for name, frame in {
        "capacity comparisons": comparisons,
        "Q4 model comparison": q4_summary,
        "Q4 persistence bootstrap": persistence,
        "Q4 text bootstrap": text,
    }.items():
        if "capacity_profile" not in frame.columns:
            raise CapacityReportError(f"{name} lacks capacity_profile lineage")

    regression_profile = str(validation["regression_capacity_profile"])
    wgan_status = str(validation.get("wgan_status", "included"))
    wgan_profile = str(validation.get("wgan_capacity_profile", ""))
    persistence_better = persistence[
        pd.to_numeric(persistence.get("mean_diff"), errors="coerce") < 0
    ]
    significant = persistence_better[
        pd.to_numeric(persistence_better.get("p_holm"), errors="coerce") < 0.05
    ]
    text_better = text[pd.to_numeric(text.get("mean_diff"), errors="coerce") < 0]
    text_significant = text_better[
        pd.to_numeric(text_better.get("p_holm"), errors="coerce") < 0.05
    ]

    wgan_conclusion = (
        f"WGAN最终选择 <b>{escape(wgan_profile)}</b>"
        if wgan_status == "included"
        else "WGAN容量gate未通过，因此按预注册规则跳过Q4"
    )
    conclusion = (
        f"Regression最终选择 <b>{escape(regression_profile)}</b>；{wgan_conclusion}。"
        f"Q4中，{len(significant)}/{len(persistence)}个"
        "模型×文本×容差比较在Holm校正后显著优于persistence；"
        f"真实文本相对控制组有{len(text_significant)}/{len(text)}个Holm显著改善。"
    )

    comparison_view = comparisons.sort_values(
        [
            column
            for column in ("selection_stage", "score_rank", "tolerance_minutes")
            if column in comparisons
        ],
        kind="stable",
    )
    overall = q4_summary[
        q4_summary.get("stratum_type", pd.Series("overall", index=q4_summary.index))
        .astype(str)
        .eq("overall")
    ].copy()
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News-first Vol 模型容量选择报告</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--bg:#f5f7fb;--panel:#fff;--blue:#275efe;--green:#087443;--red:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}} h1{{font-size:30px;margin:0 0 8px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{margin:0 0 10px}}
.lead{{font-size:17px;background:#edf3ff;border-left:4px solid var(--blue);padding:18px 20px;border-radius:6px}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 8px #1720330d}}
.status{{display:inline-block;border-radius:999px;padding:3px 9px;font-weight:700}} .status.pass{{background:#dff7ea;color:var(--green)}} .status.fail{{background:#fee4e2;color:var(--red)}}
.table-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:#edf1f7}}
code{{background:#edf1f7;padding:1px 4px;border-radius:3px}} footer{{margin-top:36px;color:var(--muted)}}
</style>
</head>
<body><main>
<h1>News-first Vol 模型容量选择与Q4检验</h1>
<p class="muted">实验目录：{escape(str(root))}</p>
<div class="lead">{conclusion}</div>

<h2>容量选择结果</h2>
{_stage_cards(selection)}
<p>容量只使用固定的Q3 <code>common_validation_05m</code>面板选择；Q4仅在所有容量与gate冻结后开启。分数为各指定容差上 <code>log(best-learned MAE / persistence MAE)</code> 的等权平均。最终容量采用按CME session聚类的one-standard-error规则，选择阈值内参数最少的模型。</p>

<h2>Q3选择审计</h2>
{_table(comparison_view, (("selection_stage", "Stage"), ("model_family", "Model"), ("capacity_profile", "Profile"), ("parameter_count", "Parameters"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("best_learned_epoch", "Learned epoch"), ("mae_ratio", "MAE ratio"), ("mean_improvement_fraction", "Mean improvement"), ("profile_gate_passed", "Gate"), ("score_rank", "Rank"), ("one_se_eligible", "1-SE eligible")))}

<h2>冻结容量的Q4表现</h2>
{_table(overall, (("model", "Model"), ("capacity_profile", "Profile"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("pair_count", "Pairs"), ("session_count", "Sessions"), ("mae", "Model MAE"), ("persistence_mae", "Persistence MAE"), ("gap", "Gap"), ("skill", "Skill"), ("win", "Win rate")))}

<h2>相对Persistence的Q4推断</h2>
<p>负的mean difference表示模型MAE更低。CME session整组重抽样10,000次；本表的Holm族覆盖所有冻结模型×文本模式×容差检验。</p>
{_table(persistence, (("model", "Model"), ("capacity_profile", "Profile"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("mean_diff", "MAE difference"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p"), ("pair_count", "Pairs"), ("session_count", "Sessions")))}

<h2>真实文本消融</h2>
<p>差值定义为real_text MAE减去control MAE；负值表示真实新闻文本更好。current-only与text-shuffle均保留。</p>
{_table(text, (("model", "Model"), ("capacity_profile", "Profile"), ("tolerance_minutes", "Tolerance"), ("control_mode", "Control"), ("mean_diff", "Real-control"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p"), ("pair_count", "Pairs"), ("session_count", "Sessions")))}

<h2>解释边界</h2>
<ul>
<li>容量选择与gate完全基于Q3；Q4结果不能反向用于修改profile、checkpoint或阈值。</li>
<li>Best-learned checkpoint明确排除epoch 0；persistence仍作为独立、可审计基线。</li>
<li>Session bootstrap反映测试session抽样不确定性，不包含训练seed不确定性。</li>
<li>Support mask只评价current和target共同raw-supported格点；结果不代表完整256格曲面的无条件误差。</li>
</ul>
<footer>输入：capacity_selection.json、capacity_comparisons.csv及analysis/final_q4_*正式表。报告不依赖外部CSS、脚本或网络资源。</footer>
</main></body></html>"""
    output = root / "report" / "capacity_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["CapacityReportError", "render_capacity_report"]
