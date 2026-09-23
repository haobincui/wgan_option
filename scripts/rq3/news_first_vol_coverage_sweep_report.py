"""Self-contained technical HTML report for the four-stage Q3 coverage sweep."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_coverage_sweep_analysis import (
    EXPERIMENT_KIND,
    STAGE_CONTRACTS,
    STAGE_ORDER,
    _payload_sha256,
    _sha256,
)


class CoverageSweepReportError(ValueError):
    """Raised when analysis artifacts cannot support the technical report."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CoverageSweepReportError(f"Expected JSON object: {path}")
    return dict(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _fmt(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        if number != 0.0 and abs(number) < 1.0e-4:
            return f"{number:.3e}"
        return f"{number:.{digits}f}"
    return str(value)


def _pct(value: Any, *, digits: int = 4) -> str:
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}%"


def _table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    percent_columns: Sequence[str] = (),
    empty: str = "暂无可用记录",
) -> str:
    available = [(key, label) for key, label in columns if key in frame.columns]
    if frame.empty or not available:
        return f'<p class="muted">{escape(empty)}</p>'
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    rows: list[str] = []
    percent = set(percent_columns)
    for row in frame.to_dict(orient="records"):
        cells = []
        for key, _ in available:
            value = _pct(row.get(key)) if key in percent else _fmt(row.get(key))
            cells.append(f"<td>{escape(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _score_chart(scores: pd.DataFrame, stage_id: str) -> str:
    selected = scores[scores["stage_id"].eq(stage_id)].copy()
    selected = selected[
        selected["text_ablation_mode"].isin(("current_only", "real_text"))
    ]
    if selected.empty:
        return '<p class="muted">无可绘制分数。</p>'
    selected["parameter_count"] = pd.to_numeric(
        selected["parameter_count"], errors="coerce"
    )
    selected["mean_improvement_fraction"] = pd.to_numeric(
        selected["mean_improvement_fraction"], errors="coerce"
    )
    if not np.isfinite(
        selected[["parameter_count", "mean_improvement_fraction"]].to_numpy(dtype=float)
    ).all():
        raise CoverageSweepReportError(f"Non-finite score chart: {stage_id}")

    width, height = 940.0, 380.0
    left, right, top, bottom = 84.0, 24.0, 28.0, 78.0
    x_values = np.log10(selected["parameter_count"].to_numpy(dtype=float))
    y_values = selected["mean_improvement_fraction"].to_numpy(dtype=float) * 100.0
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = min(float(y_values.min()), 0.0), max(float(y_values.max()), 0.0)
    pad = max((y_max - y_min) * 0.14, 1.0e-4)
    y_min, y_max = y_min - pad, y_max + pad

    def x_coord(value: float) -> float:
        if math.isclose(x_min, x_max):
            return (left + width - right) / 2.0
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    elements = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        f'aria-label="{escape(stage_id)} Q3 improvement by capacity">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#667085"/>',
    ]
    zero = y_coord(0.0)
    elements.append(
        f'<line x1="{left}" y1="{zero:.2f}" x2="{width - right}" y2="{zero:.2f}" '
        'stroke="#344054" stroke-dasharray="4 4"/>'
    )
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4.0
        y = y_coord(value)
        elements.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#667085">{value:.4f}%</text>'
        )
    colors = {"current_only": "#b54708", "real_text": "#275efe"}
    rates = sorted(selected["initial_learning_rate"].astype(float).unique())
    dashes = {
        rate: ("" if index == 0 else f"{3 + index} 3")
        for index, rate in enumerate(rates)
    }
    for (mode, rate), group in selected.groupby(
        ["text_ablation_mode", "initial_learning_rate"], sort=True
    ):
        ordered = group.sort_values("parameter_count", kind="stable")
        color = colors[str(mode)]
        dash = dashes[float(rate)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = " ".join(
            f"{x_coord(math.log10(float(row.parameter_count))):.2f},"
            f"{y_coord(float(row.mean_improvement_fraction) * 100.0):.2f}"
            for row in ordered.itertuples()
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="1.7"{dash_attr}/>'
        )
        for row in ordered.itertuples():
            x = x_coord(math.log10(float(row.parameter_count)))
            y = y_coord(float(row.mean_improvement_fraction) * 100.0)
            title = (
                f"{row.capacity_profile}, {mode}, LR={float(rate):.2g}, "
                f"improvement={float(row.mean_improvement_fraction) * 100.0:.6f}%"
            )
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="{color}">'
                f"<title>{escape(title)}</title></circle>"
            )
    profiles = (
        selected[["capacity_profile", "parameter_count"]]
        .drop_duplicates()
        .sort_values("parameter_count")
    )
    for row in profiles.itertuples():
        x = x_coord(math.log10(float(row.parameter_count)))
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 21:.2f}" text-anchor="middle" '
            f'font-size="10" fill="#667085">{escape(str(row.capacity_profile))}</text>'
        )
    elements.extend(
        (
            f'<text x="{(left + width - right) / 2:.2f}" y="{height - 20}" text-anchor="middle" '
            'font-size="12">模型参数量（log10；虚线样式区分学习率）</text>',
            f'<text transform="translate(19 {(top + height - bottom) / 2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12">相对 persistence 改善</text>',
            '<circle cx="690" cy="18" r="4" fill="#275efe"/><text x="700" y="22" font-size="11">real_text</text>',
            '<circle cx="790" cy="18" r="4" fill="#b54708"/><text x="800" y="22" font-size="11">current_only</text>',
            "</svg>",
        )
    )
    return '<div class="chart-wrap">' + "".join(elements) + "</div>"


def _stage_score_table(scores: pd.DataFrame, stage_id: str) -> str:
    frame = scores[scores["stage_id"].eq(stage_id)].copy()
    frame = frame.sort_values(
        ["text_ablation_mode", "mean_log_mae_ratio", "parameter_count"],
        kind="stable",
    )
    return _table(
        frame,
        (
            ("model_family", "模型"),
            ("capacity_profile", "容量"),
            ("parameter_count", "参数量"),
            ("initial_learning_rate", "初始LR"),
            ("text_ablation_mode", "文本模式"),
            ("geometric_mae_ratio", "MAE ratio"),
            ("mean_improvement_fraction", "改善"),
            ("sd_log_mae_ratio_across_seeds", "跨seed SD(log ratio)"),
            ("rank_within_model_and_mode", "lane排名"),
        ),
        percent_columns=("mean_improvement_fraction",),
    )


def _key_bootstrap_table(bootstrap: pd.DataFrame, stage_id: str) -> str:
    frame = bootstrap[
        bootstrap["stage_id"].eq(stage_id)
        & bootstrap["tolerance_scope"].astype(str).str.startswith("combined_")
    ].copy()
    priorities = {
        "capacity_by_learning_rate_interaction": 0,
        "text_mode_difference": 1,
        "capacity_minus_lane_leader": 2,
        "learning_rate_minus_lane_leader": 3,
        "model_minus_persistence": 4,
    }
    frame["priority"] = frame["contrast_family"].map(priorities).fillna(9)
    frame = frame.sort_values(
        ["priority", "p_holm", "mean_difference"], kind="stable"
    ).head(40)
    return _table(
        frame,
        (
            ("contrast_family", "对比"),
            ("candidate_capacity_profile", "候选容量"),
            ("candidate_initial_learning_rate", "候选LR"),
            ("candidate_text_ablation_mode", "候选文本"),
            ("reference_capacity_profile", "参照容量"),
            ("reference_initial_learning_rate", "参照LR"),
            ("reference_text_ablation_mode", "参照文本"),
            ("mean_difference", "MAE差"),
            ("ci_95_lower", "95%CI下界"),
            ("ci_95_upper", "95%CI上界"),
            ("p_holm", "Holm p"),
        ),
    )


def _stage_section(
    stage_id: str,
    validation: Mapping[str, Any],
    scores: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    contract = STAGE_CONTRACTS[stage_id]
    stage_validation = validation["stage_validations"][stage_id]
    selection = validation["stage_selections"][stage_id]
    leader = selection["overall_point_estimate_leader"]
    title = f"Stage {contract.stage_index} · {contract.purpose}"
    interaction_note = ""
    if stage_id == STAGE_ORDER[-1]:
        interaction_note = (
            '<div class="callout"><strong>交互项解释：</strong>'
            "Stage 4 的 capacity×LR 行使用 difference-in-differences："
            "(候选容量在候选LR相对5e-7的变化) − (Large的同一变化)。"
            "负值表示候选容量从该LR获得更大的MAE下降；它不等同于主效应。</div>"
        )
    return f"""
    <section id="stage-{contract.stage_index}">
      <h2>{escape(title)}</h2>
      <div class="cards">
        <div class="card"><span>新增训练任务</span><strong>{stage_validation["new_job_count"]:,}</strong></div>
        <div class="card"><span>完整证据单元</span><strong>{stage_validation["evidence_job_count"]:,}</strong></div>
        <div class="card"><span>每单元Q3 pairs</span><strong>{stage_validation["pair_count_per_job"]:,}</strong></div>
        <div class="card"><span>每单元Q3 sessions</span><strong>{stage_validation["session_count_per_job"]:,}</strong></div>
      </div>
      <p><strong>Q3点估计领先：</strong>{escape(str(leader["model_family"]))} /
      {escape(str(leader["capacity_profile"]))} / LR {_fmt(leader["initial_learning_rate"])} /
      {escape(str(leader["text_ablation_mode"]))}；几何MAE ratio
      {_fmt(leader["geometric_mae_ratio"])}，相对 persistence 改善
      {_pct(leader["mean_improvement_fraction"])}。</p>
      <p class="muted">本阶段未使用0.5%门禁；“领先”只表示共同Q3面板上的三seed点估计排序。</p>
      {_score_chart(scores, stage_id)}
      <h3>完整容量／LR／文本分数</h3>
      {_stage_score_table(scores, stage_id)}
      {interaction_note}
      <h3>关键配对推断（combined tolerance scopes）</h3>
      {_key_bootstrap_table(bootstrap, stage_id)}
    </section>
    """


def _resource_section(root: Path) -> str:
    candidates = (
        root / "resource_summary.csv",
        root / "registry" / "resource_summary.csv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return '<p class="muted">资源汇总尚未物化；训练完成后报告会自动读取。</p>'
    frame = _read_csv(path)
    if frame.empty:
        return '<p class="muted">资源汇总为空。</p>'
    if "stage_id" not in frame.columns:
        return _table(
            frame.head(30),
            (
                ("job_id", "任务"),
                ("gpu_id", "GPU"),
                ("runtime_minutes", "分钟"),
                ("peak_memory_mib", "峰值MiB"),
            ),
        )
    aggregations: dict[str, tuple[str, str]] = {"run_count": ("job_id", "nunique")}
    for output, source, operation in (
        ("process_hours_total", "gpu_hours", "sum"),
        ("shared_peak_memory_mib", "peak_memory_mib", "max"),
        ("job_window_gpu_utilization_pct", "mean_utilization_gpu_pct", "mean"),
    ):
        if source in frame.columns:
            aggregations[output] = (source, operation)
    summary = frame.groupby("stage_id", sort=True).agg(**aggregations).reset_index()

    # ``gpu_hours`` in resource_summary is the wall time of every concurrent
    # process.  Summing it is therefore process-hours, not physical-device
    # GPU-hours.  Derive allocated device-hours from each formal wave's wall
    # interval so a 24-way wave is counted once per assigned GPU.
    registry_path = root / "task_registry.csv"
    usage_path = root / "resource_usage.csv"
    if registry_path.is_file():
        registry = _read_csv(registry_path)
        required = {
            "stage_id",
            "global_wave",
            "gpu_id",
            "started_at_utc",
            "completed_at_utc",
        }
        if required.issubset(registry.columns):
            registry = registry.copy()
            registry["started_at_utc"] = pd.to_datetime(
                registry["started_at_utc"], utc=True, errors="coerce"
            )
            registry["completed_at_utc"] = pd.to_datetime(
                registry["completed_at_utc"], utc=True, errors="coerce"
            )
            registry = registry.dropna(subset=["started_at_utc", "completed_at_utc"])
            wave_rows: list[dict[str, Any]] = []
            for (stage_id, global_wave), rows in registry.groupby(
                ["stage_id", "global_wave"], sort=True
            ):
                start = rows["started_at_utc"].min()
                end = rows["completed_at_utc"].max()
                gpu_count = int(rows["gpu_id"].nunique())
                wave_rows.append(
                    {
                        "stage_id": stage_id,
                        "global_wave": int(global_wave),
                        "start": start,
                        "end": end,
                        "allocated_device_gpu_hours": (
                            (end - start).total_seconds() * gpu_count / 3600.0
                        ),
                    }
                )
            waves = pd.DataFrame(wave_rows)
            if not waves.empty:
                allocated = (
                    waves.groupby("stage_id", sort=True)["allocated_device_gpu_hours"]
                    .sum()
                    .rename("allocated_device_gpu_hours")
                    .reset_index()
                )
                summary = summary.merge(allocated, on="stage_id", how="left")

                if usage_path.is_file():
                    usage = _read_csv(usage_path)
                    usage_required = {
                        "timestamp_utc",
                        "wave",
                        "sample_status",
                        "utilization_gpu_pct",
                        "memory_used_mib",
                    }
                    if usage_required.issubset(usage.columns):
                        usage = usage.copy()
                        usage["timestamp_utc"] = pd.to_datetime(
                            usage["timestamp_utc"], utc=True, errors="coerce"
                        )
                        telemetry_rows: list[pd.DataFrame] = []
                        for wave in wave_rows:
                            selected = usage[
                                (usage["wave"] == wave["global_wave"])
                                & (usage["timestamp_utc"] >= wave["start"])
                                & (usage["timestamp_utc"] <= wave["end"])
                                & (usage["sample_status"] == "ok")
                            ].copy()
                            if not selected.empty:
                                selected["stage_id"] = wave["stage_id"]
                                telemetry_rows.append(selected)
                        if telemetry_rows:
                            telemetry = pd.concat(telemetry_rows, ignore_index=True)
                            device = (
                                telemetry.groupby("stage_id", sort=True)
                                .agg(
                                    allocated_window_gpu_utilization_pct=(
                                        "utilization_gpu_pct",
                                        "mean",
                                    ),
                                    allocated_window_peak_memory_mib=(
                                        "memory_used_mib",
                                        "max",
                                    ),
                                )
                                .reset_index()
                            )
                            summary = summary.merge(device, on="stage_id", how="left")
    return _table(
        summary,
        (
            ("stage_id", "阶段"),
            ("run_count", "任务数"),
            ("process_hours_total", "并发任务时长合计（process-hours）"),
            ("allocated_device_gpu_hours", "两卡分配时长（device GPU-hours）"),
            (
                "allocated_window_gpu_utilization_pct",
                "分配窗口整卡平均利用率",
            ),
            ("allocated_window_peak_memory_mib", "分配窗口整卡峰值MiB"),
        ),
    )


def _validate_inputs(
    root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_path = root / "analysis" / "coverage_validation_summary.json"
    validation = _read_json(validation_path)
    if str(validation.get("experiment_kind")) != EXPERIMENT_KIND:
        raise CoverageSweepReportError("Experiment kind mismatch")
    declared = str(validation.get("payload_sha256", ""))
    unhashed = dict(validation)
    unhashed.pop("payload_sha256", None)
    if declared != _payload_sha256(unhashed):
        raise CoverageSweepReportError("Validation payload hash mismatch")
    for field in (
        "q4_used_for_selection",
        "q4_predictions_generated",
        "q4_evaluation_permitted",
        "minimum_improvement_gate_applied",
    ):
        if bool(validation.get(field, False)):
            raise CoverageSweepReportError(f"Forbidden report state: {field}")
    analysed = tuple(validation.get("analysed_stages", ()))
    if not analysed or tuple(sorted(analysed, key=STAGE_ORDER.index)) != analysed:
        raise CoverageSweepReportError("Analysed stage order is invalid")
    for stage_id in analysed:
        if stage_id not in STAGE_CONTRACTS:
            raise CoverageSweepReportError(f"Unknown report stage: {stage_id}")
        stage = validation["stage_validations"][stage_id]
        for artifact in stage.get("artifacts", {}).values():
            path = Path(str(artifact["path"]))
            if not path.is_file() or _sha256(path) != str(artifact["sha256"]):
                raise CoverageSweepReportError(f"Stage artifact hash mismatch: {path}")
    combined = validation.get("combined_artifacts", {})
    for artifact in combined.values():
        path = Path(str(artifact["path"]))
        if not path.is_file() or _sha256(path) != str(artifact["sha256"]):
            raise CoverageSweepReportError(f"Combined artifact hash mismatch: {path}")
    scores = _read_csv(root / "analysis" / "coverage_q3_scores.csv")
    across = _read_csv(root / "analysis" / "coverage_q3_across_seed_summary.csv")
    bootstrap = _read_csv(root / "analysis" / "coverage_q3_bootstrap.csv")
    if set(scores["stage_id"].astype(str)) != set(analysed):
        raise CoverageSweepReportError("Score stages disagree with validation")
    if (
        not bootstrap["resampling_method"]
        .eq("seed_then_paired_CME_session_cluster")
        .all()
    ):
        raise CoverageSweepReportError("Bootstrap method drifted")
    return validation, scores, across, bootstrap


def render_coverage_sweep_report(experiment_root: str | Path) -> Path:
    """Render a portable, dependency-free HTML technical report."""

    root = Path(experiment_root).resolve()
    validation, scores, across, bootstrap = _validate_inputs(root)
    analysed = tuple(validation["analysed_stages"])
    final = None
    final_path = root / "coverage_final_candidate.json"
    if final_path.is_file():
        final = _read_json(final_path)
        if bool(final.get("q4_used_for_selection", False)) or bool(
            final.get("q4_predictions_generated", False)
        ):
            raise CoverageSweepReportError("Final candidate used Q4")

    if final is None:
        outcome = f"已完成 {len(analysed)}/4 个Q3阶段；Q4保持锁定，尚未冻结最终候选。"
        candidate_html = '<p class="muted">四阶段完成后才生成最终候选。</p>'
    else:
        regression = final["regression_candidate"]
        wgan = final["wgan_candidate"]
        overall = final["overall_point_estimate_candidate"]
        outcome = (
            "四阶段Q3覆盖测试已完成并冻结候选；Q4保持锁定，"
            "本报告仍未生成或读取任何Q4预测。"
        )
        candidate_html = f"""
        <div class="cards">
          <div class="card"><span>Regression候选</span><strong>{escape(str(regression["capacity_profile"]))}</strong><small>LR {_fmt(regression["initial_learning_rate"])} · {escape(str(regression["text_ablation_mode"]))}</small></div>
          <div class="card"><span>WGAN候选</span><strong>{escape(str(wgan["capacity_profile"]))}</strong><small>LR {_fmt(wgan["initial_learning_rate"])} · {escape(str(wgan["text_ablation_mode"]))}</small></div>
          <div class="card"><span>最低Q3点估计</span><strong>{escape(str(overall["model_family"]))} / {escape(str(overall["capacity_profile"]))}</strong><small>MAE ratio {_fmt(overall["geometric_mae_ratio"])}</small></div>
        </div>
        """

    stage_html = "".join(
        _stage_section(stage_id, validation, scores, bootstrap) for stage_id in analysed
    )
    artifact_rows = []
    for stage_id, stage in validation["stage_validations"].items():
        for label, artifact in stage["artifacts"].items():
            artifact_rows.append(
                {
                    "stage_id": stage_id,
                    "artifact": label,
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                }
            )
    for name, reference in validation.get("reference_lineage", {}).items():
        artifact_rows.append(
            {
                "stage_id": "reference",
                "artifact": name,
                "path": reference.get("pair_metrics_path", reference.get("root", "")),
                "sha256": reference.get("pair_metrics_sha256", ""),
            }
        )
    lineage_table = _table(
        pd.DataFrame(artifact_rows),
        (
            ("stage_id", "阶段"),
            ("artifact", "产物"),
            ("path", "路径"),
            ("sha256", "SHA-256"),
        ),
    )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>News-first Vol 四阶段覆盖测试 · Q3技术报告</title>
<style>
:root{{--ink:#101828;--muted:#667085;--line:#d0d5dd;--soft:#f8fafc;--blue:#275efe;--amber:#b54708;--green:#067647}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f6;color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:28px auto;padding:0 22px 60px}} section,header{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:26px;margin:18px 0;box-shadow:0 2px 8px rgba(16,24,40,.05)}}
h1{{font-size:30px;margin:0 0 9px}} h2{{font-size:22px;margin:0 0 14px}} h3{{font-size:16px;margin:24px 0 10px}} p{{max-width:92ch}} .lede{{font-size:17px}} .muted{{color:var(--muted)}}
.badge{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eef4ff;color:#3538cd;font-weight:650;margin-right:7px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:16px 0}} .card{{border:1px solid var(--line);border-radius:10px;padding:13px;background:var(--soft)}} .card span,.card small{{display:block;color:var(--muted)}} .card strong{{display:block;font-size:20px;margin:3px 0}}
.callout{{border-left:4px solid var(--amber);background:#fffaeb;padding:12px 14px;margin:16px 0;border-radius:6px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px;margin:10px 0 18px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th{{position:sticky;top:0;background:#f2f4f7;text-align:left}} th,td{{padding:8px 9px;border-bottom:1px solid #eaecf0;white-space:nowrap}} tr:last-child td{{border-bottom:0}}
.chart-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px;padding:8px;background:#fff}} svg{{min-width:760px;width:100%;height:auto}}
code{{background:#f2f4f7;padding:2px 5px;border-radius:4px}} nav a{{color:var(--blue);margin-right:12px}} footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style></head><body><main>
<header><span class="badge">Q3 only</span><span class="badge">3 seeds</span><span class="badge">session-cluster bootstrap</span>
<h1>News-first Vol 四阶段覆盖测试</h1>
<p class="lede"><strong>{escape(outcome)}</strong></p>
<p>全部模型使用窄16×16网格、<code>raw_joint</code>逐格support mask、identity residual、LP embedding和pair-balanced目标。容量、学习率、文本模式与等待容差只在声明的阶段变化。</p>
<nav>{"".join(f'<a href="#stage-{STAGE_CONTRACTS[s].stage_index}">Stage {STAGE_CONTRACTS[s].stage_index}</a>' for s in analysed)}<a href="#limitations">边界</a><a href="#lineage">血缘</a></nav>
</header>
<section><h2>结论优先</h2>{candidate_html}
<div class="callout"><strong>重要：</strong>本轮按要求不使用0.5%改善门禁。因此候选是Q3点估计最小者，不是已证明优于persistence的模型。置信区间和Holm校正结果仍完整保留。</div>
<p>组合主键固定为 <code>model / capacity / LR / tolerance / mode / seed</code>；先在独立pair层计算，再跨三个训练seed汇总。Bootstrap先抽seed，再在每个seed内整簇抽取CME session。</p>
<p class="muted">Pair win 沿用冻结数值口径：<code>abs(MAE gap) &lt;= 1e-8 IV</code> 记为tie，只有 <code>MAE gap &lt; -1e-8 IV</code> 才记为win。</p>
</section>
{stage_html}
<section><h2>GPU资源</h2>{_resource_section(root)}<p class="muted">process-hours 是所有并发任务时长之和；device GPU-hours 按每个wave从首个正式任务启动到最后任务结束的区间乘以分配GPU数计算。利用率与显存来自该正式分配窗口的整卡遥测，不能归因到单个进程。</p></section>
<section id="limitations"><h2>解释边界</h2>
<ul><li>这是快速探索性容量／LR筛查；只有三个seed，不能声称初始化稳定性。</li>
<li>Q3已被反复用于容量、文本和LR选择，最终Q4只能表述为一次探索性重复评价。</li>
<li>5/10/15/30分钟是累计新闻等待上限，会改变进入训练的数据和窗口语义；它们不是同一任务的四份独立观测。</li>
<li>市场响应horizon始终固定为 current 5分钟 → target 5分钟。</li>
<li>text_shuffle是负对照，不能作为部署文本模式；它用于判断真实文本改善是否超出随机文本路径。</li>
<li>最终候选冻结前，分析代码拒绝Q4 panel；本报告也不包含任何Q4预测或指标。</li></ul>
</section>
<section id="lineage"><h2>产物与SHA-256血缘</h2>{lineage_table}</section>
<footer>生成范围：{escape(str(validation["selection_data_scope"]))} · bootstrap={int(validation["bootstrap_iterations"]):,} · validation payload SHA-256 {escape(str(validation["payload_sha256"]))}</footer>
</main></body></html>"""
    if "https://" in html or "http://" in html:
        raise CoverageSweepReportError("Report must be self-contained")
    output = root / "report" / "coverage_completion_q3_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = [
    "CoverageSweepReportError",
    "render_coverage_sweep_report",
]
