"""Self-contained Q3 report for Generator current-input support masking."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_current_input_analysis import (
    CAPACITY_PROFILE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    EXPECTED_PAIR_COUNT,
    EXPECTED_SESSION_COUNT,
    FIXED_LEARNING_RATE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    PARAMETER_COUNT,
    SEEDS,
    TEXT_MODE,
    TOLERANCES,
    _payload_sha256,
    _read_json,
    _sha256_file,
    _write_json,
)


class CurrentInputReportError(ValueError):
    """Raised when saved Q3 evidence is insufficient for reporting."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _analysis_dir(root: Path) -> Path:
    return root / "analysis" / "current_input_ablation"


def _validate_summary(root: Path) -> dict[str, Any]:
    directory = _analysis_dir(root)
    summary_path = directory / "current_input_analysis_summary.json"
    summary = _read_json(summary_path)
    saved = str(summary.pop("analysis_sha256", ""))
    if not saved or saved != _payload_sha256(summary):
        raise CurrentInputReportError("Current-input analysis summary hash mismatch")
    summary["analysis_sha256"] = saved
    expected = {
        "masked_generator_current_input_mode": (
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        ),
        "reference_generator_current_input_mode": FULL_CURRENT_GENERATOR_INPUT_MODE,
        "generator_input_change": "encoder_surface_only",
        "residual_anchor": "original_unmasked_full_current_surface",
        "critic_and_loss_mask": "raw_joint",
        "mask_channel_added": False,
        "reference_evidence_design": "historical_immutable_coverage_experiment",
        "reference_retrained_concurrently": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise CurrentInputReportError(f"Analysis contract drifted: {field}")
    if any(
        bool(summary.get(field, True))
        for field in (
            "q4_predictions_generated",
            "q4_evaluated",
            "q4_used_for_checkpoint_selection",
        )
    ):
        raise CurrentInputReportError("Q4 isolation contract was violated")
    if (
        int(summary.get("q3_pair_count_per_cell", -1)) != EXPECTED_PAIR_COUNT
        or int(summary.get("q3_session_count_per_cell", -1)) != EXPECTED_SESSION_COUNT
        or int(summary.get("bootstrap_iterations", -1)) != DEFAULT_BOOTSTRAP_ITERATIONS
    ):
        raise CurrentInputReportError("Q3 panel/bootstrap counts drifted")
    for artifact in dict(summary.get("artifacts") or {}).values():
        path = Path(str(artifact["path"]))
        if not path.is_file() or _sha256_file(path) != str(artifact["sha256"]):
            raise CurrentInputReportError(f"Analysis artifact hash drift: {path}")
    validation = _read_json(directory / "current_input_validation_summary.json")
    validation_saved = str(validation.pop("validation_sha256", ""))
    if validation_saved != _payload_sha256(validation):
        raise CurrentInputReportError("Analysis validation hash mismatch")
    if validation.get("status") != "pass" or validation.get("analysis_sha256") != saved:
        raise CurrentInputReportError("Analysis validation did not pass")
    return summary


def _fmt(value: Any, digits: int = 7) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}g}"


def _html_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = []
    for raw in frame.to_dict(orient="records"):
        cells = []
        for name, _ in columns:
            value = raw.get(name, "")
            rendered = (
                _fmt(value)
                if isinstance(value, (int, float, np.number))
                else escape(str(value))
            )
            cells.append(f"<td>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _difference_chart(cell: pd.DataFrame) -> str:
    ordered = cell.sort_values(["tolerance_minutes", "seed"], kind="stable")
    values = ordered["masked_minus_full_current"].to_numpy(dtype=float)
    maximum = max(float(np.max(np.abs(values))), 1.0e-12)
    width, height = 900, 300
    zero_y = 135
    usable = 105
    marks = [
        f'<line x1="45" y1="{zero_y}" x2="870" y2="{zero_y}" '
        'stroke="#64748b" stroke-dasharray="5 5"/>'
    ]
    for index, row in enumerate(ordered.itertuples(index=False)):
        x = 95 + index * 135
        value = float(row.masked_minus_full_current)
        y = zero_y - value / maximum * usable
        color = "#15803d" if value < 0.0 else "#b91c1c"
        marks.append(
            f'<line x1="{x}" y1="{zero_y}" x2="{x}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="5"/>'
            f'<circle cx="{x}" cy="{y:.2f}" r="6" fill="{color}">'
            f"<title>{int(row.tolerance_minutes)}m seed {int(row.seed)}: "
            f"{value:.9g}</title></circle>"
            f'<text x="{x}" y="270" text-anchor="middle" font-size="11">'
            f"{int(row.tolerance_minutes)}m / {int(row.seed)}</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Masked minus full-current MAE by seed and tolerance">'
        + "".join(marks)
        + "</svg>"
    )


def _conclusion(primary: Mapping[str, Any]) -> str:
    mean = float(primary["mean_diff"])
    lower = float(primary["ci_95_lower"])
    upper = float(primary["ci_95_upper"])
    direction = "降低" if mean < 0.0 else "提高"
    decisive = lower > 0.0 or upper < 0.0
    return (
        f"在合并 5m/30m、三 seed 的 Q3 主比较中，逐格 current-support masking "
        f"使 pair-balanced masked MAE {direction} {abs(mean):.7g}；"
        f"seed→CME-session 两级配对 bootstrap 95% CI 为 "
        f"[{lower:.7g}, {upper:.7g}]，区间"
        f"{'不含' if decisive else '包含'} 0。"
    )


def render_current_input_report(experiment_root: str | Path) -> Path:
    """Render the saved Q3 evidence as one portable, self-contained HTML file."""

    root = Path(experiment_root).resolve(strict=False)
    summary = _validate_summary(root)
    directory = _analysis_dir(root)
    cell = pd.read_csv(directory / "current_input_cell_summary.csv")
    cross_seed = pd.read_csv(directory / "current_input_cross_seed_summary.csv")
    combined = pd.read_csv(directory / "current_input_combined_summary.csv")
    comparisons = pd.read_csv(directory / "current_input_masked_vs_full_bootstrap.csv")
    persistence = pd.read_csv(
        directory / "current_input_models_vs_persistence_bootstrap.csv"
    )
    if (
        len(cell) != 6
        or len(cross_seed) != 2
        or len(combined) != 1
        or len(comparisons) != 3
        or len(persistence) != 6
    ):
        raise CurrentInputReportError("Report input cardinality drifted")
    primary = comparisons[comparisons["analysis_role"].eq("primary")]
    secondary = comparisons[comparisons["analysis_role"].eq("secondary_tolerance")]
    if len(primary) != 1 or len(secondary) != 2:
        raise CurrentInputReportError("Report comparison family drifted")
    primary_row = primary.iloc[0].to_dict()
    title = "TY News-first Vol：Generator Current-Support Input Q3 消融"
    css = """
body{font-family:Inter,system-ui,sans-serif;max-width:1160px;margin:28px auto;padding:0 22px;color:#172033;line-height:1.52}
h1,h2{color:#0f172a}.lead{font-size:1.1rem;background:#eff6ff;border-left:5px solid #2563eb;padding:14px 18px}
.meta{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:10px}.card{background:#f8fafc;border:1px solid #e2e8f0;padding:11px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.87rem;margin:12px 0 25px}th,td{border-bottom:1px solid #e2e8f0;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}
.contract{background:#ecfdf5;border:1px solid #a7f3d0;padding:13px}.caveat{background:#fff7ed;border:1px solid #fed7aa;padding:13px}
svg{width:100%;height:auto;border:1px solid #e2e8f0;background:white}code{font-size:.82rem;word-break:break-all}
"""
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{css}</style></head><body>
<h1>{escape(title)}</h1>
<p class="lead"><strong>结论：</strong>{escape(_conclusion(primary_row))} 负的 masked−full 差值表示 masked encoder input 更好。</p>
<div class="meta"><div class="card"><b>架构</b><br>{CAPACITY_PROFILE} WGAN<br>{PARAMETER_COUNT:,} 参数</div><div class="card"><b>固定设置</b><br>real_text · LR {FIXED_LEARNING_RATE:g}<br>seeds {", ".join(str(v) for v in SEEDS)}</div><div class="card"><b>Q3 validation</b><br>{EXPECTED_PAIR_COUNT} pairs / {EXPECTED_SESSION_COUNT} sessions<br>5m 与 30m</div><div class="card"><b>推断</b><br>{DEFAULT_BOOTSTRAP_ITERATIONS:,} draws<br>seed → CME session</div></div>
<h2>被检验的模型输入差异</h2><div class="contract">在模型契约层面，<code>{CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE}</code> 只把 Generator encoder 的输入改为 <code>current_surface × current_support_mask</code>，mask 仅由 current raw-parameter JSON 推导；没有增加 mask channel，网络层数、宽度和参数量不变。<strong>Residual/persistence anchor 仍是原始、未遮蔽的 full current surface</strong>，因此零增量仍严格代表 persistence。Discriminator、gradient penalty、reconstruction、constraints 与评价继续使用 future-aware <code>raw_joint</code> mask；joint/target support 从不进入 Generator encoder。</div>
<h2>六个 seed × tolerance 单元</h2>{_difference_chart(cell)}
{_html_table(cell, [("seed", "Seed"), ("tolerance_minutes", "容差(min)"), ("masked_mae", "Masked MAE"), ("full_current_mae", "Full MAE"), ("persistence_mae", "Persistence"), ("masked_minus_full_current", "Masked−Full"), ("masked_beats_full_pair_rate", "Masked pair win")])}
<h2>跨 seed 汇总</h2>{_html_table(cross_seed, [("tolerance_minutes", "容差(min)"), ("mean_masked_mae", "Masked MAE"), ("mean_full_current_mae", "Full MAE"), ("mean_masked_minus_full_current", "Masked−Full"), ("sd_masked_minus_full_current_across_seeds", "Across-seed SD")])}
<h2>正式比较</h2>{_html_table(comparisons, [("analysis_role", "角色"), ("tolerance_scope", "范围"), ("mean_diff", "Masked−Full"), ("ci_95_lower", "95% CI low"), ("ci_95_upper", "95% CI high"), ("p_two_sided", "p"), ("holm_adjusted_p", "Holm p"), ("seed_count", "Seeds"), ("session_count_per_seed", "Sessions/seed")])}
<p>Primary 只有合并 5m/30m 的一项；5m 与 30m 是 secondary，两项 p 值在同一 family 内做 Holm 校正。</p>
<h2>分别相对 persistence</h2>{_html_table(persistence, [("generator_current_input_mode", "Input mode"), ("analysis_role", "角色"), ("tolerance_scope", "范围"), ("mean_diff", "Model−Persistence"), ("ci_95_lower", "95% CI low"), ("ci_95_upper", "95% CI high"), ("holm_adjusted_p", "Holm p")])}
<h2>边界</h2><div class="caveat"><strong>full_current 对照不是本轮与 masked 模型同期重训</strong>，而是来自既有 coverage experiment 的冻结历史证据。训练配置、epoch-0 tensors、共同 Q3 panel 和所用 artifact hashes 均做了精确配对/校验，但历史代码状态与并发运行环境仍可能形成残余混杂；因此本结果不能表述为严格同期、单因素 RCT。本报告只使用共同 Q3 validation 面板和 best_learned epoch ≥ 1 checkpoint；每个 mode×seed×tolerance 单元严格为 123 pairs / 33 CME sessions，且 persistence 向量逐键一致。Checkpoint 也用同一 Q3 validation 选择，因此结论是探索性的 validation reuse。实现会先 materialize 工作簿整张 sheet，再在 evaluator 前严格筛成 Q3；进入 evaluator 的 Q4 rows 为 0。没有读取既有 Q4 预测、没有生成或评价 Q4，也没有用 Q4 选择模型。这里只比较一种 Small 架构；它不能证明其他容量下结论相同。</div>
<p><small>Modes: <code>{CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE}</code> vs <code>{FULL_CURRENT_GENERATOR_INPUT_MODE}</code><br>Text: <code>{TEXT_MODE}</code>; tolerances: <code>{", ".join(str(v) for v in TOLERANCES)}</code><br>Analysis SHA-256: <code>{escape(str(summary["analysis_sha256"]))}</code></small></p>
</body></html>"""
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "current_input_ablation_q3_report.html"
    path.write_text(html, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "generator_current_input_q3_self_contained_html_report",
        "analysis_sha256": summary["analysis_sha256"],
        "report_path": str(path),
        "report_sha256": _sha256_file(path),
        "self_contained": True,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "created_at_utc": _utc_now(),
    }
    manifest["report_manifest_sha256"] = _payload_sha256(manifest)
    _write_json(report_dir / "current_input_ablation_report_manifest.json", manifest)
    return path


__all__ = ["CurrentInputReportError", "render_current_input_report"]
