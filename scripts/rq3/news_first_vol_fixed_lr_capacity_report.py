"""Self-contained Q3 report for the fixed-LR capacity-by-seed experiment."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_fixed_lr_capacity_analysis import (
    EXPECTED_PROFILES,
    EXPECTED_SEEDS,
    GATE_MIN_IMPROVEMENT,
    SELECTION_MODE,
    _payload_sha256,
)
from scripts.rq3.news_first_vol_fixed_lr_capacity_sweep import (
    EXPERIMENT_KIND,
    FIXED_LEARNING_RATE,
    FIXED_SCHEDULER_MIN_LR,
)
from scripts.rq3.news_first_vol_training import ROOT_KEY


class FixedLearningRateCapacityReportError(ValueError):
    """Raised when analysis artifacts cannot support the capacity report."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FixedLearningRateCapacityReportError(f"Expected JSON object: {path}")
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
    body = []
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


def _mean_sd(mean: Any, sd: Any) -> str:
    return f"{_fmt(float(mean))} ± {_fmt(float(sd))}"


def _capacity_chart(summary: pd.DataFrame) -> str:
    required = {
        "capacity_profile",
        "parameter_count",
        "text_ablation_mode",
        "tolerance_minutes",
        "mae_ratio_mean",
        "mae_ratio_sd",
    }
    if not required.issubset(summary.columns) or len(summary) != 24:
        raise FixedLearningRateCapacityReportError(
            "Capacity chart requires the exact 24-cell summary"
        )
    plot = summary.copy()
    for column in ("parameter_count", "mae_ratio_mean", "mae_ratio_sd"):
        plot[column] = pd.to_numeric(plot[column], errors="coerce")
    if not np.isfinite(
        plot[["parameter_count", "mae_ratio_mean", "mae_ratio_sd"]].to_numpy(
            dtype=float
        )
    ).all():
        raise FixedLearningRateCapacityReportError(
            "Capacity chart values are non-finite"
        )

    width, height = 940.0, 430.0
    left, right, top, bottom = 82.0, 30.0, 30.0, 90.0
    x_values = np.log10(plot["parameter_count"].to_numpy(dtype=float))
    low = plot["mae_ratio_mean"].to_numpy(dtype=float) - plot["mae_ratio_sd"].to_numpy(
        dtype=float
    )
    high = plot["mae_ratio_mean"].to_numpy(dtype=float) + plot["mae_ratio_sd"].to_numpy(
        dtype=float
    )
    y_min, y_max = min(float(low.min()), 1.0), max(float(high.max()), 1.0)
    pad = max(2.0e-5, (y_max - y_min) * 0.16)
    y_min, y_max = y_min - pad, y_max + pad
    x_min, x_max = float(x_values.min()), float(x_values.max())

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    styles = {
        ("real_text", 5): ("#275efe", ""),
        ("real_text", 30): ("#275efe", "6 4"),
        ("current_only", 5): ("#b54708", ""),
        ("current_only", 30): ("#b54708", "6 4"),
    }
    elements = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        'aria-label="MAE ratio mean plus or minus sample SD across three seeds by parameter count">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#667085"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#667085"/>',
    ]
    baseline = y_coord(1.0)
    elements.append(
        f'<line x1="{left}" y1="{baseline:.2f}" x2="{width - right}" '
        f'y2="{baseline:.2f}" stroke="#344054" stroke-dasharray="3 3"/>'
    )
    elements.append(
        f'<text x="{width - right - 4}" y="{baseline - 6:.2f}" text-anchor="end" '
        'font-size="11" fill="#344054">persistence = 1</text>'
    )
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4.0
        y = y_coord(value)
        elements.append(
            f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#667085">{value:.6f}</text>'
        )
    for (mode, tolerance), group in plot.groupby(
        ["text_ablation_mode", "tolerance_minutes"], sort=True
    ):
        color, dash = styles[(str(mode), int(tolerance))]
        ordered = group.sort_values("parameter_count", kind="stable")
        points = " ".join(
            f"{x_coord(math.log10(float(row.parameter_count))):.2f},"
            f"{y_coord(float(row.mae_ratio_mean)):.2f}"
            for row in ordered.itertuples()
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash_attr}/>'
        )
        for row in ordered.itertuples():
            x = x_coord(math.log10(float(row.parameter_count)))
            mean, sd = float(row.mae_ratio_mean), float(row.mae_ratio_sd)
            low_y, high_y = y_coord(mean - sd), y_coord(mean + sd)
            elements.extend(
                (
                    f'<line x1="{x:.2f}" y1="{low_y:.2f}" x2="{x:.2f}" '
                    f'y2="{high_y:.2f}" stroke="{color}" opacity=".72"/>',
                    f'<line x1="{x - 4:.2f}" y1="{low_y:.2f}" x2="{x + 4:.2f}" '
                    f'y2="{low_y:.2f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.2f}" y1="{high_y:.2f}" x2="{x + 4:.2f}" '
                    f'y2="{high_y:.2f}" stroke="{color}"/>',
                    f'<circle cx="{x:.2f}" cy="{y_coord(mean):.2f}" r="4" fill="{color}">'
                    f"<title>{escape(str(row.capacity_profile))}, {escape(str(mode))}, "
                    f"{int(tolerance)}m, ratio={mean:.8f}, SD={sd:.8f}</title></circle>",
                )
            )
    profiles = (
        plot[["capacity_profile", "parameter_count"]]
        .drop_duplicates()
        .sort_values("parameter_count")
    )
    for row in profiles.itertuples():
        x = x_coord(math.log10(float(row.parameter_count)))
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 21:.2f}" text-anchor="middle" '
            f'font-size="11" fill="#667085">{escape(str(row.capacity_profile))}</text>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 36:.2f}" text-anchor="middle" '
            f'font-size="10" fill="#667085">{int(row.parameter_count):,}</text>'
        )
    legend = (
        ("real_text 5m", "#275efe", ""),
        ("real_text 30m", "#275efe", "6 4"),
        ("current_only 5m", "#b54708", ""),
        ("current_only 30m", "#b54708", "6 4"),
    )
    for index, (label, color, dash) in enumerate(legend):
        x = left + index * 200.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<line x1="{x}" y1="{height - 24}" x2="{x + 24}" y2="{height - 24}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>'
            f'<text x="{x + 31}" y="{height - 20}" font-size="11" fill="#344054">'
            f"{escape(label)}</text>"
        )
    elements.extend(
        (
            f'<text x="{(left + width - right) / 2:.2f}" y="{height - 52}" text-anchor="middle" '
            'font-size="12">Regression parameters (log scale)</text>',
            f'<text transform="translate(20 {(top + height - bottom) / 2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12">MAE ratio, mean ± seed SD</text>',
            "</svg>",
        )
    )
    return '<div class="chart-wrap">' + "".join(elements) + "</div>"


def _validate_inputs(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    analysis = root / "analysis"
    validation = _read_json(analysis / "fixed_lr_capacity_validation_summary.json")
    selection = _read_json(root / "fixed_lr_capacity_selection.json")
    across = _read_csv(analysis / "fixed_lr_capacity_across_seed_summary.csv")
    scores = _read_csv(analysis / "fixed_lr_capacity_scores.csv")
    one_se = _read_csv(analysis / "fixed_lr_capacity_one_se.csv")
    persistence = _read_csv(analysis / "fixed_lr_capacity_persistence_bootstrap.csv")
    text = _read_csv(analysis / "fixed_lr_capacity_text_bootstrap.csv")
    pairwise = _read_csv(analysis / "fixed_lr_capacity_pairwise_bootstrap.csv")
    resources = _read_csv(analysis / "fixed_lr_capacity_resource_summary.csv")

    if str(validation.get("status", "")).lower() != "pass":
        raise FixedLearningRateCapacityReportError("Capacity validation did not pass")
    if (
        str(validation.get("experiment_kind", "")) != EXPERIMENT_KIND
        or str(selection.get("experiment_kind", "")) != EXPERIMENT_KIND
    ):
        raise FixedLearningRateCapacityReportError("Experiment kind drifted")
    if tuple(validation.get("seeds", ())) != EXPECTED_SEEDS:
        raise FixedLearningRateCapacityReportError("Training seed set drifted")
    if tuple(validation.get("capacity_profiles", ())) != EXPECTED_PROFILES:
        raise FixedLearningRateCapacityReportError("Capacity profile set drifted")
    if str(selection.get("selection_mode", "")) != SELECTION_MODE:
        raise FixedLearningRateCapacityReportError(
            "Formal capacity lane must be real_text"
        )
    if bool(validation.get("q4_used_for_selection")) or bool(
        validation.get("q4_predictions_generated")
    ):
        raise FixedLearningRateCapacityReportError("Q4 entered capacity analysis")
    if not math.isclose(
        float(validation.get("fixed_initial_learning_rate")),
        FIXED_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        float(validation.get("fixed_scheduler_min_lr")),
        FIXED_SCHEDULER_MIN_LR,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise FixedLearningRateCapacityReportError("Fixed LR contract drifted")
    supplied_hash = str(selection.get("selection_payload_sha256", ""))
    unhashed = {
        key: value
        for key, value in selection.items()
        if key != "selection_payload_sha256"
    }
    if supplied_hash != _payload_sha256(unhashed) or supplied_hash != str(
        validation.get("selection_payload_sha256", "")
    ):
        raise FixedLearningRateCapacityReportError("Selection payload hash mismatch")

    config_sha = str(selection.get("resolved_config_sha256", ""))
    if config_sha != str(validation.get("resolved_config_sha256", "")):
        raise FixedLearningRateCapacityReportError("Resolved-config lineage mismatch")
    if config_sha != "injected_pair_metrics":
        snapshot_path = root / "resolved_config.yaml"
        hash_path = root / "registry" / "resolved_config.sha256"
        snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        if not isinstance(snapshot, Mapping) or not isinstance(
            snapshot.get(ROOT_KEY), Mapping
        ):
            raise FixedLearningRateCapacityReportError("Resolved config is malformed")
        observed = _payload_sha256(dict(snapshot[ROOT_KEY]))
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if observed != config_sha or recorded != config_sha:
            raise FixedLearningRateCapacityReportError("Resolved-config hash mismatch")

    expected_lengths = {
        "across": (len(across), 24),
        "scores": (len(scores), 12),
        "one-SE": (len(one_se), 6),
        "persistence": (len(persistence), 36),
        "text": (len(text), 18),
        "capacity contrasts": (len(pairwise), 15),
        "resources": (len(resources), 6),
    }
    bad = {
        name: value for name, value in expected_lengths.items() if value[0] != value[1]
    }
    if bad:
        raise FixedLearningRateCapacityReportError(f"Incomplete report matrices: {bad}")
    for frame, label in (
        (one_se, "one-SE"),
        (persistence, "persistence"),
        (text, "text"),
        (pairwise, "capacity"),
    ):
        if (
            not frame["resampling_method"]
            .astype(str)
            .eq("seed_then_paired_CME_session_cluster")
            .all()
        ):
            raise FixedLearningRateCapacityReportError(
                f"{label} inference lacks two-level bootstrap lineage"
            )
    winner = str(selection.get("winner_capacity_profile", ""))
    if bool(selection.get("gate_passed")) != bool(winner):
        raise FixedLearningRateCapacityReportError("Gate and winner fields disagree")
    return (
        validation,
        selection,
        across,
        scores,
        one_se,
        persistence,
        text,
        pairwise,
        resources,
    )


def render_fixed_lr_capacity_report(experiment_root: str | Path) -> Path:
    """Render an answer-first, network-independent fixed-LR capacity report."""

    root = Path(experiment_root).resolve()
    (
        validation,
        selection,
        across,
        scores,
        one_se,
        persistence,
        text,
        pairwise,
        resources,
    ) = _validate_inputs(root)
    leader = str(selection["ranked_leader_capacity_profile"])
    leader_parameters = int(selection["ranked_leader_parameter_count"])
    winner = str(selection.get("winner_capacity_profile", ""))
    gate_passed = bool(selection["gate_passed"])
    lane_score = scores[
        scores["capacity_profile"].astype(str).eq(leader)
        & scores["text_ablation_mode"].astype(str).eq(SELECTION_MODE)
    ].iloc[0]
    leader_persistence = persistence[
        persistence["capacity_profile"].astype(str).eq(leader)
        & persistence["text_ablation_mode"].astype(str).eq(SELECTION_MODE)
        & persistence["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ].iloc[0]
    leader_text = text[
        text["capacity_profile"].astype(str).eq(leader)
        & text["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ].iloc[0]

    if gate_passed:
        lead = (
            f"固定LR <b>{FIXED_LEARNING_RATE:.1e}</b> 下，三seed的real-text容量筛选通过gate；"
            f"one-SE最小容量为 <b>{escape(winner)}</b>。"
        )
    else:
        lead = (
            f"固定LR <b>{FIXED_LEARNING_RATE:.1e}</b> 下，没有容量通过预注册gate；"
            f"<b>{escape(leader)}</b> 只是real-text的Q3排名首位候选，不是正式赢家。"
        )
    lead += (
        f" 排名候选有 {leader_parameters:,} 个参数，跨seed几何MAE ratio为 "
        f"<b>{float(lane_score['geometric_mae_ratio']):.8f}</b>；"
        f"相对persistence的两级bootstrap差值为 "
        f"<b>{float(leader_persistence['mean_difference']):+.3e}</b> "
        f"（95% CI {float(leader_persistence['ci_95_lower']):+.3e} 至 "
        f"{float(leader_persistence['ci_95_upper']):+.3e}）。"
    )

    display = across.copy()
    display["model_mae_mean_sd"] = [
        _mean_sd(mean, sd)
        for mean, sd in zip(display["model_mae_mean"], display["model_mae_sd"])
    ]
    display["mae_ratio_mean_sd"] = [
        _mean_sd(mean, sd)
        for mean, sd in zip(display["mae_ratio_mean"], display["mae_ratio_sd"])
    ]
    display["improvement_mean_sd"] = [
        f"{100 * float(mean):+.5f}% ± {100 * float(sd):.5f}%"
        for mean, sd in zip(
            display["improvement_fraction_mean"],
            display["improvement_fraction_sd"],
        )
    ]
    real_scores = scores[scores["text_ablation_mode"].eq(SELECTION_MODE)].sort_values(
        ["mean_log_mae_ratio", "parameter_count"]
    )
    current_scores = scores[
        scores["text_ablation_mode"].eq("current_only")
    ].sort_values(["mean_log_mae_ratio", "parameter_count"])

    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>News-first Vol 固定LR容量×三Seed Q3报告</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--bg:#f5f7fb;--panel:#fff;--red:#b42318;--green:#087443}}
@media(prefers-color-scheme:dark){{:root{{--ink:#edf2f7;--muted:#aab4c5;--line:#39455a;--bg:#111827;--panel:#1b2434;--red:#ff8b83;--green:#63d29a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1280px;margin:auto;padding:32px}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}h3{{margin-top:22px}}
.lead{{font-size:17px;background:var(--panel);border-left:4px solid {"var(--green)" if gate_passed else "var(--red)"};padding:18px 20px;border-radius:7px}}.muted{{color:var(--muted)}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:12px}}.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}}.kpi b{{display:block;font-size:19px}}
.table-wrap,.chart-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--panel)}}.note{{border:1px solid var(--line);background:var(--panel);padding:12px 15px;border-radius:7px}}code{{background:#7f8caa22;padding:1px 4px;border-radius:3px}}footer{{margin-top:36px;color:var(--muted)}}svg{{display:block;width:100%;min-width:760px;height:auto}}@media(max-width:700px){{main{{padding:18px}}h1{{font-size:24px}}}}
</style></head><body><main>
<h1>News-first Vol 固定LR容量×三Seed Q3报告</h1>
<p class="muted">LR={FIXED_LEARNING_RATE:.1e}；scheduler floor={FIXED_SCHEDULER_MIN_LR:.1e}；Q3共同面板；3 seeds；raw-joint support。</p>

<h2>结论</h2><div class="lead">{lead}</div>
<div class="kpis"><div class="kpi">Real-text排名首位<b>{escape(leader)}</b></div><div class="kpi">参数量<b>{leader_parameters:,}</b></div><div class="kpi">平均改善<b>{100 * float(lane_score["mean_improvement_fraction"]):+.5f}%</b></div><div class="kpi">Real−Current<b>{float(leader_text["mean_difference"]):+.3e}</b></div></div>

<h2>参数量增加没有自动带来更低误差</h2>
<p>点为三个训练seed的pair-balanced MAE ratio均值，误差条为seed间样本SD；1以下优于persistence。横轴是参数量对数，纵轴是明确标注的局部尺度。</p>
{_capacity_chart(across)}

<h2>正式Real-text容量排名与one-SE</h2>
<p>Gate要求5m/30m与全部seed均不差于persistence，且跨seed平均改善至少{100 * GATE_MIN_IMPROVEMENT:.1f}%；只有通过gate的容量才可能被one-SE选中。</p>
{_table(real_scores, (("capacity_profile", "Capacity"), ("parameter_count", "Parameters"), ("geometric_mae_ratio", "Geometric ratio"), ("mean_improvement_fraction", "Improvement"), ("sd_log_mae_ratio_across_seeds", "Seed SD log-ratio"), ("all_seed_tolerances_not_worse", "All seed×tol non-worse"), ("gate_passed", "Gate")))}
{_table(one_se, (("capacity_profile", "Capacity"), ("parameter_count", "Parameters"), ("mean_log_mae_ratio", "Mean log ratio"), ("bootstrap_standard_error", "Two-level SE"), ("formal_one_se_threshold", "Formal threshold"), ("formal_one_se_eligible", "Eligible"), ("selected_capacity", "Selected")))}

<h2>Current-only诊断排名</h2>
<p>Current-only结果保留用于判断容量是否只是在拟合当前曲面；它不参与正式winner冻结。</p>
{_table(current_scores, (("capacity_profile", "Capacity"), ("parameter_count", "Parameters"), ("geometric_mae_ratio", "Geometric ratio"), ("mean_improvement_fraction", "Improvement"), ("sd_log_mae_ratio_across_seeds", "Seed SD log-ratio"), ("gate_passed", "Diagnostic gate")))}

<h2>跨Seed精确结果</h2>
<p>每行先在各seed内对{int(validation["pair_count_per_run"])}个pairs等权，再对seed 42/202/404报告均值±样本SD。</p>
{_table(display, (("capacity_profile", "Capacity"), ("parameter_count", "Parameters"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("model_mae_mean_sd", "Model MAE mean±SD"), ("mae_ratio_mean_sd", "MAE ratio mean±SD"), ("improvement_mean_sd", "Improvement mean±SD"), ("pair_count_per_seed", "Pairs/seed"), ("session_count_per_seed", "Sessions/seed")))}

<h2>两级配对Bootstrap</h2>
<p>每次迭代先有放回抽取三个训练seed，再在每个被抽中的seed内按CME session整簇抽样；所有对比共享相同pair。负值表示左侧模型更好。</p>
<h3>相对Persistence</h3>{_table(persistence, (("capacity_profile", "Capacity"), ("text_ablation_mode", "Text mode"), ("tolerance_scope", "Scope"), ("mean_difference", "Model−Persistence"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}
<h3>Real-text相对Current-only</h3>{_table(text, (("capacity_profile", "Capacity"), ("tolerance_scope", "Scope"), ("mean_difference", "Real−Current"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}
<h3>Real-text容量相对Q3排名首位</h3>{_table(pairwise, (("candidate_capacity_profile", "Candidate"), ("candidate_parameter_count", "Parameters"), ("reference_capacity_profile", "Reference"), ("tolerance_scope", "Scope"), ("mean_difference", "Candidate−Reference"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}

<h2>训练epoch与GPU资源</h2>
<p>资源指标来自assigned-GPU wave telemetry；调度在每个wave/GPU内交叉平衡text mode与tolerance，但同一GPU并发worker仍共享利用率信号，因此适合运维对比，不是单进程因果归因。</p>
{_table(resources, (("capacity_profile", "Capacity"), ("parameter_count", "Parameters"), ("run_count", "Runs"), ("best_learned_epoch_mean", "Best epoch mean"), ("best_learned_epoch_sd", "Best epoch SD"), ("best_learned_train_val_gap_mean", "Val−train gap mean"), ("baseline_inclusive_epoch0_rate", "Epoch-0 rate"), ("runtime_minutes_total", "Runtime min total"), ("gpu_hours_total", "GPU-hours"), ("peak_memory_mib_max", "Peak MiB"), ("mean_utilization_gpu_pct_mean", "GPU util mean")))}

<h2>范围与解释边界</h2>
<ul><li>固定网格、raw-joint逐格mask、identity residual、pair-balanced loss、LP embedding与时间切分均不随容量变化。</li><li>5m与30m训练数据都在同一个2023Q3 5m共同面板上评价，共{int(validation["session_count_per_run"])}个CME sessions。</li><li>容量选择只使用best-learned epoch≥1，不允许epoch 0 persistence起点冒充训练结果。</li><li>GPU调度在wave与物理GPU内交叉平衡mode/tolerance，避免把这两个实验轴系统性绑定到某张卡。</li><li>Holm校正分别在persistence、文本和容量对比族内进行。</li></ul>
<div class="note"><b>Q4边界：</b>统一训练loader可能在内存中物化Q4 test samples；本分析没有生成Q4预测、没有计算Q4指标、没有用Q4选择容量。因此严谨表述是“Q4未预测、未评估、未参与选择”，而不是“Q4行从未读取”。</div>

<h2>限制</h2><ul><li>三个seed只能提供初步的初始化不确定性，不能声称训练稳定性已经充分识别。</li><li>只有33个CME sessions；bootstrap区间仍可能受少数高波动session影响。</li><li>Q3已被多轮探索使用，本报告属于探索性容量筛选，不是全新盲测。</li><li>若所有容量都未过gate，one-SE诊断不能被解释为正式赢家。</li></ul>

<footer>输入：fixed_lr_capacity_selection.json、跨seed结果、one-SE、三组two-level bootstrap及资源汇总CSV。HTML内嵌样式与SVG，不依赖网络资源。</footer>
</main></body></html>"""
    output = root / "report" / "fixed_lr_capacity_q3_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = [
    "FixedLearningRateCapacityReportError",
    "render_fixed_lr_capacity_report",
]
