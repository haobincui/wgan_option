"""Self-contained HTML report for the Q3 zero-latent-noise ablation."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_zero_noise_ablation import (
    CAPACITY_PROFILE,
    EXPERIMENT_KIND,
    EXPERIMENT_STAGE,
    FROZEN_SEED,
    GENERATOR_NOISE_MODE,
    REFERENCE_NOISE_MODE,
    _payload_sha256,
    _read_json,
    _sha256_file,
    _write_json,
)


class ZeroNoiseReportError(ValueError):
    """Raised when analysis lineage is insufficient for reporting."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "zero_noise_analysis_summary.json"
    summary = _read_json(summary_path)
    saved = str(summary.pop("analysis_sha256", ""))
    if not saved or saved != _payload_sha256(summary):
        raise ZeroNoiseReportError("Analysis summary hash mismatch")
    summary["analysis_sha256"] = saved
    if (
        summary.get("experiment_kind") != EXPERIMENT_KIND
        or summary.get("experiment_stage") != EXPERIMENT_STAGE
    ):
        raise ZeroNoiseReportError("Analysis belongs to another experiment")
    if any(
        bool(summary.get(field, True))
        for field in (
            "q4_predictions_generated",
            "q4_evaluated",
            "q4_used_for_checkpoint_selection",
        )
    ):
        raise ZeroNoiseReportError("Q4 isolation contract was violated")
    if (
        int(summary.get("q3_pair_count", -1)) != 123
        or int(summary.get("q3_session_count", -1)) != 33
    ):
        raise ZeroNoiseReportError("Q3 panel count drifted")
    if (
        int(summary.get("zero_prediction_mc_samples", -1)) != 1
        or int(summary.get("gaussian_reference_prediction_mc_samples", -1)) != 16
    ):
        raise ZeroNoiseReportError("Evaluation draw contract drifted")
    for artifact in dict(summary.get("artifacts") or {}).values():
        path = Path(str(artifact["path"]))
        if not path.is_file() or _sha256_file(path) != str(artifact["sha256"]):
            raise ZeroNoiseReportError(f"Analysis artifact hash drift: {path}")
    validation = _read_json(root / "analysis" / "zero_noise_validation_summary.json")
    validation_saved = str(validation.pop("validation_sha256", ""))
    if validation_saved != _payload_sha256(validation):
        raise ZeroNoiseReportError("Validation summary hash mismatch")
    if validation.get("status") != "pass" or validation.get("analysis_sha256") != saved:
        raise ZeroNoiseReportError("Analysis validation did not pass")
    return summary


def _fmt(value: Any, digits: int = 7) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    return "—" if not np.isfinite(number) else f"{number:.{digits}g}"


def _html_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        cells = []
        for name, _ in columns:
            value = values.get(name, "")
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


def _mae_chart(cell: pd.DataFrame) -> str:
    series = [
        ("zero", "zero_mae", "#2563eb"),
        ("Gaussian", "gaussian_mae", "#ea580c"),
        ("persistence", "persistence_mae", "#64748b"),
    ]
    maximum = max(float(cell[column].max()) for _, column, _ in series)
    maximum = maximum if maximum > 0.0 else 1.0
    width, height = 880, 320
    plot_height = 230
    group_width = 190
    bars = []
    for group_index, row in enumerate(
        cell.sort_values(["text_ablation_mode", "tolerance_minutes"]).itertuples()
    ):
        base_x = 60 + group_index * group_width
        for series_index, (label, column, color) in enumerate(series):
            value = float(getattr(row, column))
            bar_height = value / maximum * plot_height
            x = base_x + series_index * 38
            y = 260 - bar_height
            bars.append(
                f'<rect x="{x}" y="{y:.2f}" width="30" height="{bar_height:.2f}" fill="{color}"/>'
                f"<title>{escape(label)}: {value:.8g}</title>"
            )
        label = f"{row.text_ablation_mode} {int(row.tolerance_minutes)}m"
        bars.append(
            f'<text x="{base_x + 38}" y="285" text-anchor="middle" font-size="11">{escape(label)}</text>'
        )
    legend = "".join(
        f'<rect x="{580 + i * 95}" y="15" width="12" height="12" fill="{color}"/>'
        f'<text x="{597 + i * 95}" y="26" font-size="11">{escape(label)}</text>'
        for i, (label, _, color) in enumerate(series)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pair-balanced masked MAE by cell">'
        f'<line x1="45" y1="260" x2="840" y2="260" stroke="#94a3b8"/>{legend}{"".join(bars)}</svg>'
    )


def _conclusion(primary: pd.DataFrame) -> str:
    pieces = []
    for row in primary.sort_values("text_ablation_mode").itertuples():
        direction = "lower" if float(row.mean_diff) < 0 else "higher"
        decisive = float(row.ci_95_upper) < 0 or float(row.ci_95_lower) > 0
        pieces.append(
            f"{row.text_ablation_mode}: zero-noise MAE was {direction} by "
            f"{abs(float(row.mean_diff)):.7g}; 95% CI "
            f"[{float(row.ci_95_lower):.7g}, {float(row.ci_95_upper):.7g}]"
            f"{' (excludes zero)' if decisive else ' (includes zero)'}"
        )
    return "; ".join(pieces) + "."


def render_zero_noise_report(experiment_root: str | Path) -> Path:
    """Render a compact report whose figures and claims come only from saved CSVs."""

    root = Path(experiment_root).resolve(strict=False)
    summary = _validate_summary(root)
    cell = pd.read_csv(root / "analysis" / "zero_noise_cell_summary.csv")
    comparisons = pd.read_csv(
        root / "analysis" / "zero_noise_vs_gaussian_bootstrap.csv"
    )
    persistence = pd.read_csv(
        root / "analysis" / "noise_models_vs_persistence_bootstrap.csv"
    )
    if len(cell) != 4 or len(comparisons) != 6 or len(persistence) != 12:
        raise ZeroNoiseReportError("Report input cardinality drifted")
    primary = comparisons[comparisons["analysis_role"].eq("primary")]
    secondary = comparisons[comparisons["analysis_role"].eq("secondary_sensitivity")]
    title = "TY News-first Vol：Zero-noise WGAN Q3 配对消融"
    css = """
body{font-family:Inter,system-ui,sans-serif;max-width:1120px;margin:28px auto;padding:0 22px;color:#172033;line-height:1.5}
h1,h2{color:#0f172a} .lead{font-size:1.12rem;background:#eff6ff;border-left:5px solid #2563eb;padding:14px 18px}
.meta{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px}.card{background:#f8fafc;border:1px solid #e2e8f0;padding:10px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:12px 0 24px}th,td{border-bottom:1px solid #e2e8f0;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}
.caveat{background:#fff7ed;border:1px solid #fed7aa;padding:12px}svg{width:100%;height:auto;background:white;border:1px solid #e2e8f0}
code{font-size:.84rem;word-break:break-all}
"""
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{css}</style></head><body>
<h1>{escape(title)}</h1>
<p class="lead"><strong>结论：</strong>{escape(_conclusion(primary))} 负的 zero−Gaussian 差值代表 zero-noise 更好；正式主比较将 5m/30m 合并，并在两个文本模式上做 Holm 校正。</p>
<div class="meta"><div class="card"><b>模型</b><br>{CAPACITY_PROFILE} WGAN<br>149,333 nominal / 145,237 active</div><div class="card"><b>Seed</b><br>{FROZEN_SEED}（单 seed）</div><div class="card"><b>Q3 validation 面板</b><br>123 pairs / 33 sessions</div><div class="card"><b>Bootstrap</b><br>10,000 次 CME-session cluster</div></div>
<h2>四个实验单元的 masked MAE</h2>{_mae_chart(cell)}
{_html_table(cell, [("text_ablation_mode", "文本模式"), ("tolerance_minutes", "容差(min)"), ("zero_mae", "Zero MAE"), ("gaussian_mae", "Gaussian MAE"), ("persistence_mae", "Persistence MAE"), ("zero_minus_gaussian", "Zero−Gaussian"), ("zero_beats_gaussian_pair_rate", "Zero pair win rate")])}
<h2>主比较：合并 5m/30m</h2>
{_html_table(primary, [("text_ablation_mode", "文本模式"), ("mean_diff", "Zero−Gaussian"), ("ci_95_lower", "95% CI low"), ("ci_95_upper", "95% CI high"), ("p_two_sided", "p"), ("holm_adjusted_p", "Holm p"), ("pair_count", "pair-observations"), ("session_count", "sessions")])}
<h2>分容差敏感性</h2>
{_html_table(secondary, [("text_ablation_mode", "文本模式"), ("tolerance_scope", "容差"), ("mean_diff", "Zero−Gaussian"), ("ci_95_lower", "95% CI low"), ("ci_95_upper", "95% CI high"), ("holm_adjusted_p", "Holm p")])}
<h2>相对 persistence（二级证据）</h2>
{_html_table(persistence[persistence["tolerance_scope"].eq("combined_05m_30m")], [("generator_noise_mode", "Noise"), ("text_ablation_mode", "文本模式"), ("mean_diff", "Model−Persistence"), ("ci_95_lower", "95% CI low"), ("ci_95_upper", "95% CI high"), ("holm_adjusted_p", "Holm p")])}
<h2>解释与边界</h2><div class="caveat">Zero 模式保留 noise_dim=32 和同一 Small 名义架构（G 123,472 + D 25,861 = 149,333 参数）；其中 128×32=4,096 个 latent fusion 权重结构性失活并必须从 initial 到 best/final 保持不变，因此有效 active 参数为 145,237。训练时先消耗同形状 Gaussian RNG 再置零，从而尽量保持 Dropout/GP RNG 进程可比。它只移除 latent 输入，Dropout 与 gradient-penalty alpha 仍是随机的。Zero Q3 为单次显式零向量推理，Gaussian 是冻结参考的 MC16。只有一个训练 seed，且 best_learned checkpoint 也用同一 Q3 validation 面板选择，因此 bootstrap 只是对 33 个 validation session 的描述性、探索性重用，不代表初始化稳定性或完全样本外误差。读取原始工作簿和 split loader 时允许先物化 Q4 行，但本实验没有把 Q4 行传给 evaluator、没有生成 Q4 预测、没有评价，也没有用于 checkpoint 选择。</div>
<p><small>Experiment: <code>{EXPERIMENT_KIND}</code> / <code>{EXPERIMENT_STAGE}</code><br>Noise: <code>{GENERATOR_NOISE_MODE}</code> vs immutable <code>{REFERENCE_NOISE_MODE}</code><br>Analysis SHA-256: <code>{escape(str(summary["analysis_sha256"]))}</code></small></p>
</body></html>"""
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "zero_noise_q3_report.html"
    path.write_text(html, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "zero_noise_q3_self_contained_html_report",
        "experiment_kind": EXPERIMENT_KIND,
        "analysis_sha256": summary["analysis_sha256"],
        "report_path": str(path),
        "report_sha256": _sha256_file(path),
        "self_contained": True,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "created_at_utc": _utc_now(),
    }
    manifest["report_manifest_sha256"] = _payload_sha256(manifest)
    _write_json(report_dir / "report_manifest.json", manifest)
    return path


__all__ = ["ZeroNoiseReportError", "render_zero_noise_report"]
