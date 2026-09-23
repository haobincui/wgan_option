"""Self-contained technical HTML report for the Q3 local-LR seed sweep."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3.news_first_vol_local_lr_analysis import (
    EXPECTED_LEARNING_RATES,
    EXPECTED_SEEDS,
    GATE_MIN_IMPROVEMENT,
    _payload_sha256,
)
from scripts.rq3.news_first_vol_local_lr_sweep import EXPERIMENT_KIND
from scripts.rq3.news_first_vol_training import ROOT_KEY


class LocalLearningRateReportError(ValueError):
    """Raised when local-LR analysis artifacts cannot support the report."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise LocalLearningRateReportError(f"Expected a JSON object: {path}")
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
        numeric = float(value)
        if not math.isfinite(numeric):
            return "—"
        if numeric != 0.0 and abs(numeric) < 1.0e-4:
            return f"{numeric:.3e}"
        return f"{numeric:.{digits}f}"
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


def _mean_sd_label(mean: Any, sd: Any) -> str:
    left = _fmt(float(mean))
    right = _fmt(float(sd))
    return f"{left} ± {right}"


def _ratio_chart(summary: pd.DataFrame) -> str:
    """Focused-scale line-and-interval chart over the five ordered LR values."""

    required = {
        "initial_learning_rate",
        "text_ablation_mode",
        "tolerance_minutes",
        "mae_ratio_mean",
        "mae_ratio_sd",
    }
    if not required.issubset(summary.columns) or summary.empty:
        return '<p class="muted">缺少可绘制的跨-seed MAE ratio。</p>'
    plot = summary.copy()
    for column in ("initial_learning_rate", "mae_ratio_mean", "mae_ratio_sd"):
        plot[column] = pd.to_numeric(plot[column], errors="coerce")
    plot = plot[
        np.isfinite(plot["initial_learning_rate"])
        & np.isfinite(plot["mae_ratio_mean"])
        & np.isfinite(plot["mae_ratio_sd"])
    ]
    if len(plot) != 20:
        raise LocalLearningRateReportError(
            "LR chart requires the complete 20-cell summary"
        )

    width, height = 900.0, 420.0
    left, right, top, bottom = 82.0, 30.0, 30.0, 82.0
    x_values = np.log10(plot["initial_learning_rate"].to_numpy(dtype=float))
    low_values = plot["mae_ratio_mean"].to_numpy(dtype=float) - plot[
        "mae_ratio_sd"
    ].to_numpy(dtype=float)
    high_values = plot["mae_ratio_mean"].to_numpy(dtype=float) + plot[
        "mae_ratio_sd"
    ].to_numpy(dtype=float)
    y_min = min(float(low_values.min()), 1.0)
    y_max = max(float(high_values.max()), 1.0)
    padding = max(2.0e-5, (y_max - y_min) * 0.16)
    y_min -= padding
    y_max += padding
    x_min, x_max = float(x_values.min()), float(x_values.max())

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    styles = {
        ("current_only", 5): ("#275efe", ""),
        ("current_only", 30): ("#275efe", "6 4"),
        ("real_text", 5): ("#b54708", ""),
        ("real_text", 30): ("#b54708", "6 4"),
    }
    elements = [
        f'<svg viewBox="0 0 {int(width)} {int(height)}" role="img" '
        'aria-label="Mean MAE ratio plus or minus one standard deviation across three seeds">',
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
        ordered = group.sort_values("initial_learning_rate", kind="stable")
        points = " ".join(
            f"{x_coord(math.log10(float(row.initial_learning_rate))):.2f},"
            f"{y_coord(float(row.mae_ratio_mean)):.2f}"
            for row in ordered.itertuples()
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash_attr}/>'
        )
        for row in ordered.itertuples():
            x = x_coord(math.log10(float(row.initial_learning_rate)))
            mean = float(row.mae_ratio_mean)
            sd = float(row.mae_ratio_sd)
            y_low, y_high = y_coord(mean - sd), y_coord(mean + sd)
            elements.extend(
                (
                    f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" '
                    f'y2="{y_high:.2f}" stroke="{color}" opacity=".72"/>',
                    f'<line x1="{x - 4:.2f}" y1="{y_low:.2f}" x2="{x + 4:.2f}" '
                    f'y2="{y_low:.2f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.2f}" y1="{y_high:.2f}" x2="{x + 4:.2f}" '
                    f'y2="{y_high:.2f}" stroke="{color}"/>',
                    f'<circle cx="{x:.2f}" cy="{y_coord(mean):.2f}" r="4" '
                    f'fill="{color}"><title>{escape(str(mode))}, {int(tolerance)}m, '
                    f"LR={float(row.initial_learning_rate):.2e}, ratio={mean:.8f}, "
                    f"SD={sd:.8f}</title></circle>",
                )
            )
    for rate in EXPECTED_LEARNING_RATES:
        x = x_coord(math.log10(rate))
        elements.append(
            f'<text x="{x:.2f}" y="{height - bottom + 20:.2f}" text-anchor="middle" '
            f'font-size="11" fill="#667085">{rate:.2g}</text>'
        )
    legend_y = height - 23.0
    legend_items = (
        ("current_only 5m", "#275efe", ""),
        ("current_only 30m", "#275efe", "6 4"),
        ("real_text 5m", "#b54708", ""),
        ("real_text 30m", "#b54708", "6 4"),
    )
    for index, (label, color, dash) in enumerate(legend_items):
        x = left + index * 190.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<line x1="{x:.2f}" y1="{legend_y:.2f}" x2="{x + 24:.2f}" '
            f'y2="{legend_y:.2f}" stroke="{color}" stroke-width="2"{dash_attr}/>'
            f'<text x="{x + 31:.2f}" y="{legend_y + 4:.2f}" font-size="11" '
            f'fill="#344054">{escape(label)}</text>'
        )
    elements.extend(
        (
            f'<text x="{(left + width - right) / 2:.2f}" y="{height - 48:.2f}" '
            'text-anchor="middle" font-size="12">Initial learning rate (log scale)</text>',
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
]:
    analysis = root / "analysis"
    validation = _read_json(analysis / "local_lr_validation_summary.json")
    selection = _read_json(root / "local_lr_selection.json")
    across = _read_csv(analysis / "local_lr_across_seed_summary.csv")
    scores = _read_csv(analysis / "local_lr_scores.csv")
    persistence = _read_csv(analysis / "local_lr_persistence_bootstrap.csv")
    text = _read_csv(analysis / "local_lr_text_bootstrap.csv")
    pairwise = _read_csv(analysis / "local_lr_pairwise_bootstrap.csv")

    if str(validation.get("status", "")).lower() != "pass":
        raise LocalLearningRateReportError("Local-LR validation did not pass")
    if str(selection.get("experiment_kind", "")) != EXPERIMENT_KIND:
        raise LocalLearningRateReportError("Local-LR experiment kind drifted")
    if tuple(validation.get("seeds", ())) != EXPECTED_SEEDS:
        raise LocalLearningRateReportError("Validation seed set drifted")
    if bool(validation.get("q4_used_for_selection")) or bool(
        validation.get("q4_predictions_generated")
    ):
        raise LocalLearningRateReportError("Q4 must not enter local-LR analysis")
    supplied_hash = str(selection.get("selection_payload_sha256", ""))
    payload_without_hash = {
        key: value
        for key, value in selection.items()
        if key != "selection_payload_sha256"
    }
    if not supplied_hash or supplied_hash != _payload_sha256(payload_without_hash):
        raise LocalLearningRateReportError("Local-LR selection payload hash mismatch")
    if supplied_hash != str(validation.get("selection_payload_sha256", "")):
        raise LocalLearningRateReportError("Selection and validation hashes disagree")
    selection_config_sha = str(selection.get("resolved_config_sha256", ""))
    validation_config_sha = str(validation.get("resolved_config_sha256", ""))
    if not selection_config_sha or selection_config_sha != validation_config_sha:
        raise LocalLearningRateReportError("Selection/config lineage hash mismatch")
    if selection_config_sha != "injected_pair_metrics" and (
        len(selection_config_sha) != 64
        or any(
            character not in "0123456789abcdef" for character in selection_config_sha
        )
    ):
        raise LocalLearningRateReportError("Resolved-config SHA-256 is malformed")
    if selection_config_sha != "injected_pair_metrics":
        snapshot_path = root / "resolved_config.yaml"
        hash_path = root / "registry" / "resolved_config.sha256"
        if not snapshot_path.is_file() or not hash_path.is_file():
            raise LocalLearningRateReportError(
                "Resolved-config lineage files are missing"
            )
        snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        if not isinstance(snapshot, Mapping) or not isinstance(
            snapshot.get(ROOT_KEY), Mapping
        ):
            raise LocalLearningRateReportError("Resolved-config snapshot is malformed")
        observed_config_sha = _payload_sha256(dict(snapshot[ROOT_KEY]))
        recorded_config_sha = hash_path.read_text(encoding="utf-8").strip()
        if (
            observed_config_sha != selection_config_sha
            or recorded_config_sha != selection_config_sha
        ):
            raise LocalLearningRateReportError("Resolved-config lineage hash mismatch")
    if len(across) != 20 or len(scores) != 10:
        raise LocalLearningRateReportError("Across-seed summary matrix is incomplete")
    for frame, label in (
        (persistence, "persistence"),
        (text, "text"),
        (pairwise, "pairwise LR"),
    ):
        if (
            frame.empty
            or not frame["resampling_method"]
            .astype(str)
            .eq("seed_then_paired_CME_session_cluster")
            .all()
        ):
            raise LocalLearningRateReportError(
                f"{label} inference lacks the two-level bootstrap contract"
            )
    return validation, selection, across, scores, persistence, text, pairwise


def render_local_lr_report(experiment_root: str | Path) -> Path:
    """Render an answer-first, self-contained Q3 technical report."""

    root = Path(experiment_root).resolve()
    validation, selection, across, scores, persistence, text, pairwise = (
        _validate_inputs(root)
    )
    leader = str(selection["ranked_leader_lr_profile"])
    leader_rate = float(selection["ranked_leader_learning_rate"])
    gate_passed = bool(selection["gate_passed"])
    formal = str(selection.get("winner_lr_profile", ""))
    if gate_passed != bool(formal):
        raise LocalLearningRateReportError("Gate and formal-winner fields disagree")

    leader_combined = persistence[
        persistence["lr_profile"].astype(str).eq(leader)
        & persistence["text_ablation_mode"].astype(str).eq("current_only")
        & persistence["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ]
    leader_text = text[
        text["lr_profile"].astype(str).eq(leader)
        & text["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ]
    if len(leader_combined) != 1 or len(leader_text) != 1:
        raise LocalLearningRateReportError("Leader bootstrap rows are not unique")
    persistence_row = leader_combined.iloc[0]
    text_row = leader_text.iloc[0]
    score_row = scores[
        scores["lr_profile"].astype(str).eq(leader)
        & scores["text_ablation_mode"].astype(str).eq("current_only")
    ]
    if len(score_row) != 1:
        raise LocalLearningRateReportError("Leader score is not unique")
    score = score_row.iloc[0]

    if gate_passed:
        conclusion = (
            f"三seed的Q3筛选通过预注册gate，正式LR为 <b>{leader_rate:.2e}</b>。"
        )
    else:
        conclusion = (
            f"没有LR通过预注册gate；<b>{leader_rate:.2e}</b> 只是Q3排名首位候选，"
            "不是正式赢家。"
        )
    conclusion += (
        f" 排名候选的跨seed几何MAE ratio为 <b>{float(score['geometric_mae_ratio']):.8f}</b>，"
        f"相对persistence的两级bootstrap联合差值为 "
        f"<b>{float(persistence_row['mean_difference']):+.3e}</b> "
        f"（95% CI {float(persistence_row['ci_95_lower']):+.3e} 至 "
        f"{float(persistence_row['ci_95_upper']):+.3e}）。"
    )

    display = across.copy()
    display["model_mae_mean_sd"] = [
        _mean_sd_label(mean, sd)
        for mean, sd in zip(display["model_mae_mean"], display["model_mae_sd"])
    ]
    display["mae_ratio_mean_sd"] = [
        _mean_sd_label(mean, sd)
        for mean, sd in zip(display["mae_ratio_mean"], display["mae_ratio_sd"])
    ]
    display["improvement_pct_mean_sd"] = [
        f"{100 * float(mean):+.5f}% ± {100 * float(sd):.5f}%"
        for mean, sd in zip(
            display["improvement_fraction_mean"],
            display["improvement_fraction_sd"],
        )
    ]
    fixed_center = pairwise[
        pairwise["reference_kind"].astype(str).eq("fixed_prior_center")
        & pairwise["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ].copy()
    ranked_reference = pairwise[
        pairwise["reference_kind"].astype(str).eq("ranked_leader")
        & pairwise["tolerance_scope"].astype(str).eq("combined_05m_30m")
    ].copy()

    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>News-first Vol 本地低学习率多Seed Q3报告</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--bg:#f5f7fb;--panel:#fff;--blue:#275efe;--orange:#b54708;--red:#b42318;--green:#087443}}
@media(prefers-color-scheme:dark){{:root{{--ink:#edf2f7;--muted:#aab4c5;--line:#39455a;--bg:#111827;--panel:#1b2434;--blue:#7aa2ff;--orange:#f2a65a;--red:#ff8b83;--green:#63d29a}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}} h1{{font-size:30px;margin:0 0 8px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{margin-top:22px}}
.lead{{font-size:17px;background:var(--panel);border-left:4px solid {"var(--green)" if gate_passed else "var(--red)"};padding:18px 20px;border-radius:7px}} .muted{{color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:12px}} .kpi{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}} .kpi b{{display:block;font-size:19px}}
.table-wrap,.chart-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:var(--panel)}}
.note{{border:1px solid var(--line);background:var(--panel);padding:12px 15px;border-radius:7px}} code{{background:#7f8caa22;padding:1px 4px;border-radius:3px}} footer{{margin-top:36px;color:var(--muted)}} svg{{display:block;width:100%;min-width:720px;height:auto}}
@media(max-width:700px){{main{{padding:18px}}h1{{font-size:24px}}}}
</style></head><body><main>
<h1>News-first Vol 本地低学习率多Seed Q3报告</h1>
<p class="muted">分析范围：2023Q3；3个训练seed；5/30分钟训练容差；raw-joint support。</p>

<h2>技术摘要</h2>
<div class="lead">{conclusion}</div>
<div class="kpis">
<div class="kpi">排名候选LR<b>{leader_rate:.2e}</b></div>
<div class="kpi">跨seed改善<b>{100 * float(score["mean_improvement_fraction"]):+.5f}%</b></div>
<div class="kpi">Persistence差值p值<b>{float(persistence_row["p_two_sided"]):.4f}</b></div>
<div class="kpi">真实文本差值<b>{float(text_row["mean_difference"]):+.3e}</b></div>
</div>

<h2>五档LR的结果接近persistence，seed波动必须显式保留</h2>
<p>每个点是三个训练seed的pair-balanced MAE ratio均值，误差条为seed间样本SD；1以下表示优于persistence。纵轴使用明确标注的局部尺度，以便观察很小的差异，因此不能从视觉距离推断经济量级。</p>
{_ratio_chart(across)}

<h2>跨Seed精确结果</h2>
<p>每行先在各seed内对{int(validation["pair_count_per_run"])}个Q3 market pairs等权，再对seed 42/202/404报告均值±样本SD；累计新闻行不会重复放大pair。</p>
{_table(display, (("lr_profile", "LR档"), ("initial_learning_rate", "Initial LR"), ("text_ablation_mode", "Text mode"), ("tolerance_minutes", "Tolerance"), ("model_mae_mean_sd", "Model MAE mean±SD"), ("mae_ratio_mean_sd", "MAE ratio mean±SD"), ("improvement_pct_mean_sd", "Improvement mean±SD"), ("pair_count_per_seed", "Pairs/seed"), ("session_count_per_seed", "Sessions/seed")))}

<h2>两级配对Bootstrap没有把训练seed当成普通pair</h2>
<p>每次迭代先有放回抽取三个训练seed，再在每个被抽中的seed内按CME session整簇抽样；模型与persistence、real-text与current-only以及LR之间始终使用相同pair。下表负值分别表示模型或real-text更好。</p>
<h3>相对Persistence</h3>
{_table(persistence, (("lr_profile", "LR档"), ("text_ablation_mode", "Text mode"), ("tolerance_scope", "Scope"), ("mean_difference", "Model−Persistence"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p"), ("seed_count", "Seeds"), ("session_count_per_seed", "Sessions/seed")))}
<h3>真实文本相对Current-only</h3>
{_table(text, (("lr_profile", "LR档"), ("tolerance_scope", "Scope"), ("mean_difference", "Real−Current"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}
<h3>相对先验中心1e-6</h3>
{_table(fixed_center, (("candidate_lr_profile", "Candidate"), ("candidate_learning_rate", "Candidate LR"), ("reference_lr_profile", "Reference"), ("tolerance_scope", "Scope"), ("mean_difference", "Candidate−1e-6"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}
<h3>相对Q3排名首位候选</h3>
{_table(ranked_reference, (("candidate_lr_profile", "Candidate"), ("candidate_learning_rate", "Candidate LR"), ("reference_lr_profile", "Ranked leader"), ("tolerance_scope", "Scope"), ("mean_difference", "Candidate−leader"), ("ci_95_lower", "CI low"), ("ci_95_upper", "CI high"), ("p_two_sided", "Raw p"), ("p_holm", "Holm p")))}

<h2>范围、数据与指标定义</h2>
<ul>
<li>面板固定为 <code>2023-07-01 ≤ effective_origin_utc &lt; 2023-10-01</code> 的共同5m Q3 validation pairs。</li>
<li>模型误差是在每个pair的 <code>raw_joint</code> support cells上计算MAE，再对pair等权。</li>
<li>训练容差5m/30m只改变训练样本；两者都在完全相同的Q3 5m面板上评价。</li>
<li>LR选择分数为5m和30m的 <code>log(model MAE / persistence MAE)</code>等权平均，再对三个seed等权平均。</li>
</ul>

<h2>实验与推断方法</h2>
<p>60个任务覆盖5 LR × 3 seed × 2文本模式 × 2训练容差。所有任务使用large Regression、LP embedding、identity residual、相同loss与scheduler规则。容量、数据、mask和时间切分不随LR变化。Gate要求平均改善至少{100 * GATE_MIN_IMPROVEMENT:.1f}%且所有seed×主容差均不差于persistence。</p>

<h2>限制、稳健性与解释边界</h2>
<div class="note"><b>Q4边界：</b>现有训练loader可能按统一接口在内存中物化Q4 test samples；本分析没有生成Q4预测、没有计算Q4指标、没有用Q4进行选择。因而严谨结论是“Q4未预测、未评估、未参与选择”，不是“Q4行从未被读取”。</div>
<ul>
<li>三个seed只能初步估计初始化不确定性；seed层bootstrap并不替代更多独立训练重复。</li>
<li>Q3只有{int(validation["session_count_per_run"])}个CME sessions，尾部新闻冲击的区间可能仍不稳定。</li>
<li>这是超参数筛选而非因果实验；真实文本差值只表示预测表现关联。</li>
<li>Holm校正分别在persistence、文本和LR对比族内完成。</li>
</ul>

<h2>建议的下一步</h2>
<ol>
<li>只有当某一LR同时通过gate且两级bootstrap区间支持稳定改善时，才冻结它用于后续样本外评价。</li>
<li>若所有档位仍接近persistence，优先测试mask-aware输入、低支持pair权重和shift=0任务，而不是继续细分LR。</li>
<li>任何后续Q4分析必须读取本轮冻结的selection hash，且不得反向修改LR。</li>
</ol>

<h2>仍需回答的问题</h2>
<ul><li>增加训练seed后，文本增益是否仍小于session与seed两层不确定性？</li><li>不同支持格点阈值下，排名候选是否保持一致？</li></ul>
<footer>输入：local_lr_selection.json、analysis/local_lr_across_seed_summary.csv、local_lr_scores.csv及三组two-level bootstrap表。报告为独立HTML，不依赖网络资源。</footer>
</main></body></html>"""
    output = root / "report" / "local_lr_q3_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["LocalLearningRateReportError", "render_local_lr_report"]
