"""Self-contained HTML and Markdown report for the 8x8 WGAN sweep."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scripts.rq3 import news_first_vol_wgan_grid08_sweep as sweep


def _table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    view = frame[columns].copy()
    if limit is not None:
        view = view.head(limit)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    rows = []
    for record in view.to_dict(orient="records"):
        cells = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                rendered = f"{value:.8g}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _resource_summary(root: Path) -> pd.DataFrame:
    registry = sweep._load_registry(root)
    status_rows = []
    for job in registry["jobs"]:
        status = sweep._read_json(sweep._job_status_path(root, job["job_id"]))
        if status.get("status") != "completed":
            continue
        start = pd.Timestamp(status["started_at_utc"])
        end = pd.Timestamp(status["completed_at_utc"])
        status_rows.append(
            {
                "wave": int(job["wave"]),
                "gpu_id": int(job["gpu_id"]),
                "job_hours": (end - start).total_seconds() / 3600.0,
                "start": start,
                "end": end,
            }
        )
    status_frame = pd.DataFrame(status_rows)
    usage_path = root / "resource_usage.csv"
    usage = pd.read_csv(usage_path) if usage_path.is_file() else pd.DataFrame()
    rows = []
    for (wave, gpu), group in status_frame.groupby(["wave", "gpu_id"], sort=True):
        samples = (
            usage[
                pd.to_numeric(usage.get("wave"), errors="coerce").eq(wave)
                & pd.to_numeric(usage.get("gpu_index"), errors="coerce").eq(gpu)
                & usage.get(
                    "sample_status", pd.Series(index=usage.index, dtype=str)
                ).eq("ok")
            ]
            if not usage.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "wave": int(wave),
                "gpu_id": int(gpu),
                "device_gpu_hours": (
                    group["end"].max() - group["start"].min()
                ).total_seconds()
                / 3600.0,
                "process_hours": float(group["job_hours"].sum()),
                "peak_memory_mib": float(samples["memory_used_mib"].max())
                if not samples.empty
                else float("nan"),
                "mean_gpu_utilization_pct": float(samples["utilization_gpu_pct"].mean())
                if not samples.empty
                else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(root / "resource_summary.csv", index=False)
    return frame


def _markdown(
    selection: Mapping[str, Any], scores: pd.DataFrame, q3: pd.DataFrame
) -> str:
    candidate = (
        f"{selection['one_se_candidate_capacity_profile']} × "
        f"{selection['one_se_candidate_lr_profile']}"
    )
    leader = (
        f"{selection['point_leader_capacity_profile']} × "
        f"{selection['point_leader_lr_profile']}"
    )
    best = scores.iloc[0]
    lines = [
        "# News-first Vol 8×8 WGAN 容量 × LR 实验结论",
        "",
        f"- Q2 点估计第一：**{leader}**。",
        f"- one-SE 候选：**{candidate}**。",
        f"- 证据标签：**{selection['selection_label']}**。",
        f"- Q2 第一名 MAE ratio：`{float(best['mae_ratio']):.9f}`，点估计改善 `{100 * float(best['improvement_fraction']):.5f}%`。",
        "- 30m 只作 lagged-news 稳健性；Q3 未用于选择；Q4 未创建 loader、预测或评价。",
        "- 这是 8×8 内部调参。16×16 仅作描述性背景，不能据此声称分辨率变化造成性能差异。",
        "- 8×8 的 raw-joint 有效 pair 比 16×16 少约 19%–21%，因此跨分辨率 raw MAE 不具严格可比性。",
    ]
    if not q3.empty:
        lines.extend(["", "## Q3 探索性评价", ""])
        for row in q3.to_dict(orient="records"):
            lines.append(
                f"- `{row['contrast']}`: 差值 `{float(row['estimate']):+.6e}`，"
                f"95% CI `[{float(row['ci_95_lower']):+.6e}, {float(row['ci_95_upper']):+.6e}]`，"
                f"Holm p=`{float(row['holm_p']):.4g}`。"
            )
    return "\n".join(lines) + "\n"


def _finalize_manifest(root: Path) -> Path:
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "delivery_hashes.csv":
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        candidates.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sweep._sha256_file(path),
            }
        )
    return sweep._write_csv(
        root / "delivery_hashes.csv", candidates, tuple(candidates[0])
    )


def render_grid08_report(experiment_root: str | Path) -> Path:
    root = Path(experiment_root).resolve()
    sweep._validate_root_lineage(root)
    selection = sweep._read_json(root / "analysis" / "grid08_selection.json")
    scores = pd.read_csv(root / "analysis" / "grid08_capacity_lr_scores.csv")
    interactions = pd.read_csv(root / "analysis" / "grid08_interaction_contrasts.csv")
    robustness = pd.read_csv(root / "analysis" / "grid08_30m_robustness_scores.csv")
    q3_path = root / "analysis" / "grid08_q3_primary_contrasts.csv"
    q3 = pd.read_csv(q3_path) if q3_path.is_file() else pd.DataFrame()
    resources = _resource_summary(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md = _markdown(selection, scores, q3)
    md_path = report_dir / "grid08_wgan_capacity_lr_conclusion.md"
    sweep._atomic_write_text(md_path, md)
    q3_section = "<p>Q3 尚未解锁。</p>"
    if not q3.empty:
        q3_section = _table(
            q3,
            ["contrast", "estimate", "ci_95_lower", "ci_95_upper", "holm_p"],
        )
    resource_section = (
        "<p>无正式资源记录。</p>"
        if resources.empty
        else _table(
            resources,
            [
                "wave",
                "gpu_id",
                "device_gpu_hours",
                "process_hours",
                "peak_memory_mib",
                "mean_gpu_utilization_pct",
            ],
        )
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>8×8 WGAN Capacity × LR</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#18212b}}h1,h2{{color:#14365d}}.card{{background:#f4f7fb;border:1px solid #d7e0ec;border-radius:10px;padding:16px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd6e2;padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef2f7;padding:2px 4px}}.warn{{border-left:5px solid #c78200;padding-left:12px}}</style></head>
<body><h1>News-first Vol 8×8 WGAN 容量 × LR 独立实验</h1>
<div class="card"><b>点估计第一：</b>{html.escape(str(selection["point_leader_capacity_profile"]))} × {html.escape(str(selection["point_leader_lr_profile"]))}<br>
<b>one-SE 候选：</b>{html.escape(str(selection["one_se_candidate_capacity_profile"]))} × {html.escape(str(selection["one_se_candidate_lr_profile"]))}<br>
<b>证据标签：</b>{html.escape(str(selection["selection_label"]))}</div>
<p class="warn">本实验只在 8×8 内选型。8×8 相比 16×16 减少约 19%–21% raw-joint 有效 pair；跨分辨率 raw MAE 仅作描述性背景，不能解释为分辨率的因果效果。</p>
<h2>Q2 30配置排名</h2>{_table(scores, ["rank", "capacity_profile", "lr_profile", "expected_wgan_parameters", "mae_ratio", "improvement_fraction", "ci_95_lower", "ci_95_upper", "holm_p"], limit=30)}
<h2>Capacity × LR DiD（small × 5e-7锚点）</h2>{_table(interactions, ["capacity_profile", "lr_profile", "did_mae", "ci_95_lower", "ci_95_upper", "holm_p"])}
<h2>30m lagged-news 稳健性排名</h2>{_table(robustness, ["rank_30m", "rank_5m", "rank_change_30m_minus_5m", "capacity_profile", "lr_profile", "mae_ratio", "improvement_fraction", "ci_95_lower", "ci_95_upper", "non_worse_seed_count"])}
<h2>Q3探索性评价</h2>{q3_section}
<h2>资源口径</h2><p>device GPU-hours 按每个 wave×物理GPU 的占用区间计算；process-hours 是并发任务时长之和，两者不得混用。</p>{resource_section}
<h2>边界</h2><ul><li>Q2 是唯一选择面板；30m不参与选择。</li><li>Q3在selection/checkpoint allowlist冻结后读取，且不重新选型。</li><li>Q4 loader/prediction/evaluation均为0。</li><li>Q3历史上已被重复使用，结果仅属探索性。</li></ul>
<footer>Generated {datetime.now(timezone.utc).isoformat()}</footer></body></html>"""
    path = report_dir / "grid08_wgan_capacity_lr_report.html"
    sweep._atomic_write_text(path, document)
    _finalize_manifest(root)
    return path


__all__ = ["render_grid08_report"]
