"""Self-contained report renderer for the frozen ten-seed RQ1/RQ2/RQ3 bundle."""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
    RETROSPECTIVE_LABEL,
    UnifiedAnalysisError,
    sha256_file,
    validate_frozen_artifact_manifest,
    validate_paired_evidence,
)


REPORT_DISCLAIMER = (
    "These results are retrospective, frozen, and exploratory—not confirmatory. "
    "The evaluated 2023 quarters have been inspected by earlier experiments. "
    "No comparison in this report supports a causal interpretation."
)

REQUIRED_ANALYSIS_ROLES = (
    "pair_metrics_source",
    "prediction_manifest_source",
    "checkpoint_manifest_source",
    "scheduled_events_source",
    "market_jump_source",
    "rq1_rq2_results",
    "rq1_rq2_persistence_secondary",
    "rq3_scheduled_match_plan",
    "rq3_scheduled_details",
    "rq3_scheduled_results",
    "rq3_market_jump_join",
    "rq3_market_jump_match_plan",
    "rq3_market_jump_details",
    "rq3_market_jump_results",
    "analysis_summary",
)


def _require_sha(value: Any) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise UnifiedAnalysisError(
            "analysis_manifest_sha256 must be one lowercase SHA-256 digest"
        )
    return digest


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise UnifiedAnalysisError(f"Existing report output drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[[column for column in columns if column in frame]].copy()
    if selected.empty:
        return "_No rows._"
    headers = list(selected.columns)

    def clean(value: Any) -> str:
        return _format_value(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[[column for column in columns if column in frame]].copy()
    if selected.empty:
        return "<p><em>No rows.</em></p>"
    headers = "".join(
        f"<th>{html.escape(str(column))}</th>" for column in selected.columns
    )
    rows: list[str] = []
    for values in selected.itertuples(index=False, name=None):
        cells = "".join(
            f"<td>{html.escape(_format_value(value))}</td>" for value in values
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _artifact_map(frame: pd.DataFrame) -> dict[str, Path]:
    return {
        str(row.role): Path(str(row.path))
        for row in frame[["role", "path"]].itertuples(index=False)
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise UnifiedAnalysisError(f"Expected JSON object: {path}")
    return payload


def _load_report_inputs(
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Path], dict[str, pd.DataFrame]]:
    expected = _require_sha(manifest_sha256)
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise UnifiedAnalysisError(
            f"Analysis manifest SHA-256 drift: expected={expected}, actual={actual}"
        )
    payload = _read_json(manifest_path)
    if payload.get("interpretation") != RETROSPECTIVE_LABEL:
        raise UnifiedAnalysisError("Analysis manifest retrospective label drift")
    validated = validate_frozen_artifact_manifest(
        manifest_path, expected_roles=REQUIRED_ANALYSIS_ROLES
    )
    artifacts = _artifact_map(validated)
    summary = _read_json(artifacts["analysis_summary"])
    if (
        summary.get("interpretation") != RETROSPECTIVE_LABEL
        or summary.get("confirmatory") is not False
    ):
        raise UnifiedAnalysisError(
            "Analysis summary must remain retrospective and non-confirmatory"
        )
    required_universe = {
        "arms",
        "seeds",
        "folds",
        "tolerances_minutes",
        "arms_by_tolerance",
    }
    missing_universe = sorted(required_universe - set(summary))
    if missing_universe:
        raise UnifiedAnalysisError(
            f"Analysis summary is missing frozen evidence universe: {missing_universe}"
        )
    raw_arm_map = summary["arms_by_tolerance"]
    if not isinstance(raw_arm_map, Mapping):
        raise UnifiedAnalysisError(
            "Analysis summary arms_by_tolerance must be a mapping"
        )
    validate_paired_evidence(
        artifacts["pair_metrics_source"],
        artifacts["prediction_manifest_source"],
        artifacts["checkpoint_manifest_source"],
        expected_arms=tuple(map(str, summary["arms"])),
        expected_seeds=tuple(map(int, summary["seeds"])),
        expected_folds=tuple(map(str, summary["folds"])),
        expected_tolerances=tuple(map(int, summary["tolerances_minutes"])),
        expected_arms_by_tolerance={
            int(tolerance): tuple(map(str, arms))
            for tolerance, arms in raw_arm_map.items()
        },
    )
    tables = {
        role: pd.read_csv(artifacts[role])
        for role in (
            "rq1_rq2_results",
            "rq1_rq2_persistence_secondary",
            "rq3_scheduled_match_plan",
            "rq3_scheduled_results",
            "rq3_market_jump_match_plan",
            "rq3_market_jump_details",
            "rq3_market_jump_results",
        )
    }
    for role in (
        "rq1_rq2_results",
        "rq1_rq2_persistence_secondary",
        "rq3_scheduled_results",
        "rq3_market_jump_results",
    ):
        table = tables[role]
        if (
            "interpretation" not in table
            or not table["interpretation"].eq(RETROSPECTIVE_LABEL).all()
        ):
            raise UnifiedAnalysisError(f"{role} retrospective label drift")
    rq12 = tables["rq1_rq2_results"]
    if "tolerance_minutes" not in rq12:
        raise UnifiedAnalysisError("rq1_rq2_results lacks tolerance_minutes")
    rq12_tolerances = set(
        pd.to_numeric(rq12["tolerance_minutes"], errors="raise").astype(int)
    )
    if rq12_tolerances != {5}:
        raise UnifiedAnalysisError(
            "RQ1/RQ2 report evidence must contain the frozen 5m panel only"
        )
    persistence = tables["rq1_rq2_persistence_secondary"]
    if "tolerance_minutes" not in persistence:
        raise UnifiedAnalysisError(
            "rq1_rq2_persistence_secondary lacks tolerance_minutes"
        )
    persistence_tolerances = set(
        pd.to_numeric(persistence["tolerance_minutes"], errors="raise").astype(int)
    )
    if persistence_tolerances != {5}:
        raise UnifiedAnalysisError(
            "RQ1/RQ2 persistence evidence must contain the frozen 5m panel only"
        )
    return summary, artifacts, tables


def _sections(
    summary: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> list[tuple[str, str, str]]:
    rq12 = tables["rq1_rq2_results"]
    persistence = tables["rq1_rq2_persistence_secondary"]
    scheduled = tables["rq3_scheduled_results"]
    market = tables["rq3_market_jump_results"]
    match_plan = tables["rq3_scheduled_match_plan"]
    market_match_plan = tables["rq3_market_jump_match_plan"]
    market_details = tables["rq3_market_jump_details"]
    persistence_family_sizes = (
        persistence.groupby("research_question", sort=True)["comparison_id"]
        .nunique()
        .to_dict()
    )
    persistence_family_summary = " and ".join(
        f"{research_question} Holm-{int(size)}"
        for research_question, size in persistence_family_sizes.items()
    )
    parent_diagnostic_present = bool(
        persistence["focal_arm"].eq("parent_current_only").any()
        and persistence.loc[
            persistence["focal_arm"].eq("parent_current_only"), "comparison_role"
        ]
        .eq("diagnostic_only")
        .all()
    )
    parent_diagnostic_sentence = (
        " The RQ1 `parent_current_only` comparison is **diagnostic_only**; even if "
        "its statistical gate passes, it cannot support a secondary conclusion."
        if parent_diagnostic_present
        else ""
    )
    parent_diagnostic_html = (
        " The RQ1 <code>parent_current_only</code> comparison is "
        "<strong>diagnostic_only</strong>; even if its statistical gate passes, it "
        "cannot support a secondary conclusion."
        if parent_diagnostic_present
        else ""
    )

    rq12_columns = (
        "research_question",
        "tolerance_minutes",
        "contrast_id",
        "focal_mean_mae",
        "reference_mean_mae",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "passes_full_gate",
        "passes_primary_gate",
        "claim_scope",
    )
    persistence_columns = (
        "research_question",
        "comparison_id",
        "comparison_role",
        "focal_mean_mae",
        "reference_mean_mae",
        "mean_log_mae_ratio",
        "geometric_mae_ratio",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "estimability_status",
        "inference_permitted",
        "passes_secondary_gate",
        "claim_scope",
    )
    scheduled_columns = (
        "window_id",
        "window_role",
        "contrast_id",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "pair_count",
        "candidate_pair_count",
        "unmatched_pair_count",
        "match_rate",
        "event_count",
        "eligible_event_count",
        "session_count",
        "coverage_gate_passes",
        "passes_primary_gate",
        "claim_scope",
    )
    market_columns = (
        "anomaly_tier",
        "analysis_role",
        "contrast_id",
        "pair_count",
        "session_count",
        "mean_jump_text_advantage",
        "mean_control_text_advantage",
        "mean_jump_increment",
        "candidate_pair_count",
        "unmatched_pair_count",
        "match_rate",
        "consistent_seed_count",
        "consistent_fold_count",
        "inference_permitted",
        "coverage_gate_passes",
        "full_seed_panel",
        "full_fold_panel",
        "matched_tier_category_coverage_passes",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "passes_primary_gate",
        "claim_scope",
    )
    market_match_columns = (
        "matched_set_id",
        "fold",
        "anomaly_tier",
        "jump_pair_id",
        "jump_session_id",
        "jump_effective_origin_utc",
        "match_status",
        "unmatched_reason",
        "control_pair_id",
        "control_session_id",
        "control_effective_origin_utc",
        "eligible_control_count",
        "available_control_count_at_assignment",
        "clock_distance_minutes",
        "match_distance",
        "distance_metric",
        "matching_method",
    )
    market_detail_columns = (
        "matched_set_id",
        "fold",
        "seed",
        "anomaly_tier",
        "jump_pair_id",
        "jump_effective_origin_utc",
        "control_pair_id",
        "control_session_id",
        "control_effective_origin_utc",
        "match_distance",
        "contrast_id",
        "focal_arm",
        "reference_arm",
        "jump_text_advantage",
        "control_text_advantage",
        "jump_increment",
        "difference_direction",
    )

    rq12_primary = rq12[
        rq12.get("passes_primary_gate", pd.Series(False, index=rq12.index)).astype(bool)
    ]
    scheduled_primary = scheduled[scheduled["passes_primary_gate"].astype(bool)]
    market_primary = market[market["passes_primary_gate"].astype(bool)]
    passing_labels = [
        f"{row.research_question} {row.contrast_id} (frozen 5m panel)"
        for row in rq12_primary.itertuples(index=False)
    ]
    passing_labels.extend(
        f"RQ3 scheduled {row.contrast_id} ({row.window_id})"
        for row in scheduled_primary.itertuples(index=False)
    )
    passing_labels.extend(
        f"RQ3 market jumps {row.contrast_id} (all tiers)"
        for row in market_primary.itertuples(index=False)
    )
    if passing_labels:
        decision_md = (
            "The following frozen retrospective-primary gates passed: "
            + "; ".join(passing_labels)
            + ". This is exploratory evidence only."
        )
        decision_html = (
            "<p>The following frozen retrospective-primary gates passed: "
            + html.escape("; ".join(passing_labels))
            + ". <strong>This is exploratory evidence only.</strong></p>"
        )
    else:
        decision_md = (
            "No frozen retrospective-primary gate passed. Robustness or descriptive "
            "patterns below do not override that result."
        )
        decision_html = (
            "<p><strong>No frozen retrospective-primary gate passed.</strong> "
            "Robustness or descriptive patterns below do not override that result.</p>"
        )
    all_tier_estimability = set(
        market.loc[market["anomaly_tier"].eq("all"), "estimability_status"].astype(str)
    )
    if all_tier_estimability == {"not_estimable_due_to_coverage"}:
        decision_md += (
            " The frozen all-tier market-jump analysis is "
            "`not_estimable_due_to_coverage`; its matched sample is reported "
            "descriptively and receives no inferential claim."
        )
        decision_html += (
            "<p>The frozen all-tier market-jump analysis is "
            "<code>not_estimable_due_to_coverage</code>; its matched sample is "
            "reported descriptively and receives no inferential claim.</p>"
        )

    methods_md = (
        f"The frozen matrix contains **{int(summary['job_count'])} jobs** and "
        f"**{int(summary['pair_metric_row_count'])} pair-metric rows**. RQ1/RQ2 use "
        "the frozen **5-minute panel only**. Their estimand is "
        "**log(MAE_focal / MAE_reference)**; negative values favor the focal arm, and "
        "`geometric_mae_ratio` is its exponentiated value. "
        f"Inference uses {int(summary['bootstrap_iterations']):,} seed → fold → "
        "paired CME-session bootstrap draws. Holm correction is applied within each "
        "research-question family, followed by the predeclared seed/fold consistency "
        "gate. The arm-vs-persistence secondary analysis uses two independent "
        f"families: **{persistence_family_summary}**. The exact-TTM maturity axis is "
        "non-uniform, while the frozen CNN and adjacent-row smoothness loss retain their "
        "existing index-space semantics. Missing or blank BoW article text is represented "
        "as an empty document—not the token `nan`—and its article/pair counts remain in "
        "the frozen overlay transform." + parent_diagnostic_sentence
    )
    methods_html = (
        f"<p>The frozen matrix contains <strong>{int(summary['job_count'])} jobs</strong> "
        f"and <strong>{int(summary['pair_metric_row_count'])} pair-metric rows</strong>. "
        "RQ1/RQ2 use the frozen <strong>5-minute panel only</strong>. Their estimand is "
        "<strong>log(MAE<sub>focal</sub> / MAE<sub>reference</sub>)</strong>; negative "
        "values favor the focal arm, and <code>geometric_mae_ratio</code> is its "
        f"exponentiated value. Inference uses {int(summary['bootstrap_iterations']):,} "
        "seed → fold → paired CME-session bootstrap draws. Holm correction is "
        "applied within each research-question family, followed by the predeclared "
        "seed/fold consistency gate. The arm-vs-persistence secondary analysis uses "
        f"two independent families: <strong>{html.escape(persistence_family_summary)}</strong>. "
        "The exact-TTM maturity axis is non-uniform, while the frozen CNN and adjacent-row "
        "smoothness loss retain their existing index-space semantics. Missing or blank BoW "
        "article text is represented as an empty document—not the token <code>nan</code>—and "
        "its article/pair counts remain in the frozen overlay transform."
        + parent_diagnostic_html
        + "</p>"
    )
    scheduled_md = (
        "All RQ3 scheduled-news calculations use the frozen **30-minute news-alignment "
        "panel**; `30m` changes the admissible news/market alignment, while the target "
        "remains the surface exactly five minutes after the current surface. The windows "
        "are **[0,+30] primary**, **[-10,+20] robustness**, and "
        "**[0,+5] descriptive**, with inclusive boundaries. Scheduled pairs are matched "
        "to ordinary-news controls with the same fold and weekday and within the "
        "frozen UTC clock caliper, greedily and without replacement. "
        f"The frozen plan contains "
        f"{match_plan.loc[match_plan['match_status'].eq('matched'), 'matched_set_id'].nunique()} "
        f"matched sets and {match_plan['match_status'].eq('unmatched').sum()} explicitly "
        "unmatched scheduled pairs.\n\n" + _markdown_table(scheduled, scheduled_columns)
    )
    scheduled_html = (
        "<p>All RQ3 scheduled-news calculations use the frozen <strong>30-minute "
        "news-alignment panel</strong>; 30m changes the admissible news/market alignment, "
        "while the target remains the surface exactly five minutes after the current "
        "surface. The windows are <strong>[0,+30] primary</strong>, "
        "<strong>[-10,+20] robustness</strong>, and <strong>[0,+5] descriptive</strong>, "
        "with inclusive boundaries. Scheduled pairs are matched to ordinary-news controls "
        "with the same fold and weekday and within the frozen UTC clock caliper, greedily "
        f"and without replacement. The frozen plan contains "
        f"{match_plan.loc[match_plan['match_status'].eq('matched'), 'matched_set_id'].nunique()} "
        f"matched sets and {match_plan['match_status'].eq('unmatched').sum()} explicitly "
        "unmatched scheduled pairs.</p>" + _html_table(scheduled, scheduled_columns)
    )
    market_md = (
        "The frozen market-jump table is joined by exact UTC forecast-origin timestamp. "
        "Each jump is matched 1:1, greedily and without replacement, to a non-jump "
        "control with the same fold and weekday, within the frozen UTC clock caliper, "
        "and nearest on the frozen current-state features. Text advantage is "
        "`MAE_reference - MAE_focal`; the reported DiD (`mean_jump_increment`) is "
        "**jump text advantage minus matched-control text advantage**, so positive "
        "values mean the focal text representation is relatively more valuable at "
        "jumps. The pooled broad/primary/high (`all`) panel is the retrospective "
        "primary analysis only when its frozen pair/session coverage gate passes. The "
        "table records that decision explicitly in `estimability_status`. The "
        "fold-completeness and matched-tier-completeness fields are audit diagnostics, "
        "not additional admission gates. The narrower primary and high tiers remain "
        "descriptive; no causal attribution or "
        "confirmatory decision is permitted.\n\n"
        "### DiD results\n\n"
        + _markdown_table(market, market_columns)
        + "\n\n### Frozen jump-to-control match plan\n\n"
        + _markdown_table(market_match_plan, market_match_columns)
        + "\n\n### Pair-seed DiD details\n\n"
        + _markdown_table(market_details, market_detail_columns)
    )
    market_html = (
        "<p>The frozen market-jump table is joined by exact UTC forecast-origin "
        "timestamp. Each jump is matched 1:1, greedily and without replacement, to a "
        "non-jump control with the same fold and weekday, within the frozen UTC clock "
        "caliper, and nearest on the frozen current-state features. Text advantage is "
        "<code>MAE_reference - MAE_focal</code>; the reported DiD "
        "(<code>mean_jump_increment</code>) is <strong>jump text advantage minus "
        "matched-control text advantage</strong>, so positive values mean the focal "
        "text representation is relatively more valuable at jumps. The pooled "
        "broad/primary/high (<code>all</code>) panel is the retrospective primary "
        "analysis only when its frozen pair/session coverage gate passes. The "
        "table records that decision explicitly in <code>estimability_status</code>. The "
        "fold-completeness and matched-tier-completeness fields are audit diagnostics, "
        "not additional admission gates. The narrower primary and high tiers remain "
        "descriptive; no causal attribution or "
        "confirmatory decision is permitted.</p><h3>DiD results</h3>"
        + _html_table(market, market_columns)
        + "<h3>Frozen jump-to-control match plan</h3>"
        + _html_table(market_match_plan, market_match_columns)
        + "<h3>Pair-seed DiD details</h3>"
        + _html_table(market_details, market_detail_columns)
    )
    return [
        ("Decision summary", decision_md, decision_html),
        ("Frozen method", methods_md, methods_html),
        (
            "RQ1 and RQ2 paired results",
            _markdown_table(rq12, rq12_columns),
            _html_table(rq12, rq12_columns),
        ),
        (
            "RQ1 and RQ2: 5m arm vs persistence secondary",
            _markdown_table(persistence, persistence_columns),
            _html_table(persistence, persistence_columns),
        ),
        ("RQ3 scheduled news", scheduled_md, scheduled_html),
        ("RQ3 frozen market jumps", market_md, market_html),
    ]


def _lineage_markdown(artifacts: Mapping[str, Path]) -> str:
    rows = pd.DataFrame(
        [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in sorted(artifacts.items())
        ]
    )
    return _markdown_table(rows, ("role", "path", "sha256"))


def _lineage_html(artifacts: Mapping[str, Path]) -> str:
    rows = pd.DataFrame(
        [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in sorted(artifacts.items())
        ]
    )
    return _html_table(rows, ("role", "path", "sha256"))


def render_unified_report(
    *,
    analysis_manifest_path: str | Path,
    analysis_manifest_sha256: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render hash-validated, self-contained Markdown and HTML reports."""

    manifest_path = Path(analysis_manifest_path)
    summary, artifacts, tables = _load_report_inputs(
        manifest_path, analysis_manifest_sha256
    )
    sections = _sections(summary, tables)
    title = "News-first Vol FiLM + NoLP Critic: unified ten-seed RQ1/RQ2/RQ3 report"
    markdown_parts = [
        f"# {title}",
        "",
        f"> **Status:** `{RETROSPECTIVE_LABEL}`",
        ">",
        f"> {REPORT_DISCLAIMER}",
    ]
    for heading, markdown_body, _ in sections:
        markdown_parts.extend(["", f"## {heading}", "", markdown_body])
    markdown_parts.extend(
        [
            "",
            "## Frozen lineage",
            "",
            "The renderer validated every source and analysis artifact against the explicit manifest. No `latest` path or directory scan was used.",
            "",
            _lineage_markdown(artifacts),
            "",
        ]
    )

    html_sections = "".join(
        f"<section><h2>{html.escape(heading)}</h2>{body}</section>"
        for heading, _, body in sections
    )
    html_document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#5b6474; --line:#d8dee9; --accent:#174ea6; --warn:#7a3e00; }}
body {{ margin:0; background:#f5f7fb; color:var(--ink); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:32px 24px 64px; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 18px; }} h2 {{ margin-top:34px; }}
.notice {{ border-left:5px solid #d97706; background:#fff7ed; color:var(--warn); padding:14px 16px; }}
section {{ background:white; border:1px solid var(--line); border-radius:10px; margin:20px 0; padding:18px; box-shadow:0 2px 8px rgba(20,32,55,.04); }}
.table-wrap {{ overflow:auto; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; white-space:nowrap; }}
th {{ background:#eef3fb; position:sticky; top:0; }} code {{ overflow-wrap:anywhere; }} .muted {{ color:var(--muted); }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="notice"><strong>Status: {html.escape(RETROSPECTIVE_LABEL)}</strong><br>{html.escape(REPORT_DISCLAIMER)}</div>
{html_sections}
<section><h2>Frozen lineage</h2><p class="muted">Every source and analysis artifact was validated against the explicit manifest. No <code>latest</code> path or directory scan was used.</p>{_lineage_html(artifacts)}</section>
</main></body></html>
"""

    destination = Path(output_dir)
    markdown_path = destination / "news_first_vol_film_nolp_10seed_unified_report.md"
    html_path = destination / "news_first_vol_film_nolp_10seed_unified_report.html"
    _atomic_write(markdown_path, "\n".join(markdown_parts))
    _atomic_write(html_path, html_document)
    report_manifest = {
        "schema_version": 1,
        "interpretation": RETROSPECTIVE_LABEL,
        "analysis_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _require_sha(analysis_manifest_sha256),
        },
        "reports": [
            {
                "path": str(markdown_path.resolve()),
                "size_bytes": markdown_path.stat().st_size,
                "sha256": sha256_file(markdown_path),
            },
            {
                "path": str(html_path.resolve()),
                "size_bytes": html_path.stat().st_size,
                "sha256": sha256_file(html_path),
            },
        ],
    }
    _atomic_write(
        destination / "report_manifest.json",
        json.dumps(report_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return markdown_path, html_path


__all__ = ["REPORT_DISCLAIMER", "render_unified_report"]
