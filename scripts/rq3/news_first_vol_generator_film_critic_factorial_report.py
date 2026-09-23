"""Self-contained report for the Generator-FiLM/Critic-text factorial study."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.rq3 import news_first_vol_generator_film_critic_factorial as factorial
from scripts.rq3 import (
    news_first_vol_generator_film_critic_factorial_analysis as analysis,
)


REPORT_MD = "film_critic_factorial_conclusion.md"
REPORT_HTML = "film_critic_factorial_conclusion.html"


class FilmCriticReportError(ValueError):
    """Raised when report inputs no longer match their frozen hashes."""


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}g}"


def _load_selection(root: Path) -> dict[str, Any]:
    path = root / "analysis" / analysis.Q3_SELECTION
    payload = analysis._mapping(path, "Q3 selection")
    analysis._verify_self_hash(payload, "selection_sha256", "Q3 selection")
    for name, expected in payload.get("analysis_artifact_sha256", {}).items():
        artifact = root / "analysis" / str(name)
        if not artifact.is_file() or factorial._sha256_file(artifact) != str(expected):
            raise FilmCriticReportError(f"Q3 report input hash drift: {artifact}")
    return payload


def _load_q4(root: Path) -> dict[str, Any] | None:
    path = root / "analysis" / analysis.Q4_SUMMARY
    if not path.exists():
        return None
    payload = analysis._mapping(path, "Q4 summary")
    analysis._verify_self_hash(payload, "summary_sha256", "Q4 summary")
    for name, expected in payload.get("artifact_sha256", {}).items():
        artifact = root / "analysis" / str(name)
        if not artifact.is_file() or factorial._sha256_file(artifact) != str(expected):
            raise FilmCriticReportError(f"Q4 report input hash drift: {artifact}")
    exposure = payload.get("historical_q4_exposure") or {}
    if (
        not bool(exposure.get("historically_exposed"))
        or str(exposure.get("pair_overlap_label")) != "143/143"
        or bool(payload.get("confirmatory_claim_permitted"))
    ):
        raise FilmCriticReportError("Q4 historical-exposure contract drift")
    return payload


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in frame[list(columns)].to_dict(orient="records"):
        rows.append(
            "| "
            + " | ".join(str(record[column]).replace("|", "\\|") for column in columns)
            + " |"
        )
    return "\n".join(rows)


def _html_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "<p><em>No rows.</em></p>"
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = []
    for record in frame[list(columns)].to_dict(orient="records"):
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(str(record[column]))}</td>" for column in columns)
            + "</tr>"
        )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _display_contrasts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    output = frame.copy()
    for column in (
        "mean_difference",
        "bootstrap_se",
        "ci_95_lower",
        "ci_95_upper",
        "p_two_sided",
        "holm_p",
    ):
        if column in output.columns:
            output[column] = output[column].map(_fmt)
    return output


def _markdown(
    root: Path,
    selection: Mapping[str, Any],
    architecture: pd.DataFrame,
    text: pd.DataFrame,
    q4: Mapping[str, Any] | None,
    q4_primary: pd.DataFrame,
) -> str:
    winner = selection["winner"]
    anchor = selection["anchor"]
    lines = [
        "# News-first Vol Generator-FiLM × Critic-text Factorial",
        "",
        "## Result first",
        "",
        (
            f"The frozen Q3 decision is **{selection['selection_label']}**. "
            f"The selected architecture uses Generator "
            f"`{winner['generator_conditioning_mode']}` and Critic "
            f"`{winner['critic_conditioning_mode']}` with `real_text`."
        ),
        "",
        (
            f"The anchor is Generator `{anchor['generator_conditioning_mode']}` + "
            f"Critic `{anchor['critic_conditioning_mode']}`. Text alignment at the "
            f"selected architecture is "
            f"**{'supported' if selection['text_alignment_supported'] else 'not supported'}** "
            "under its separate Holm-4 family; it was not used as a hard winner gate."
        ),
        "",
        "## Inference boundary",
        "",
        (
            "This is a **single-seed experiment (seed 42)**. The 10,000-draw "
            "session-cluster bootstrap measures variation across sampled CME sessions "
            "only. It does **not** estimate seed-to-seed or model-initialization uncertainty."
        ),
        "",
        "Q3 is the development/selection panel. All 16 development cells were evaluated "
        "on the same 148-news-row / 135-pair / 33-session 5m panel, and Q4 contributed "
        "zero rows to checkpoint or architecture selection.",
        "",
        "## Q3 architecture contrasts (candidate − anchor MAE)",
        "",
        _markdown_table(
            architecture,
            (
                "candidate_generator_conditioning_mode",
                "candidate_critic_conditioning_mode",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "holm_p",
                "eligible",
            ),
        ),
        "",
        "Negative differences favour the candidate. Eligibility required a negative "
        "mean, a 95% CI upper bound below zero, and Holm-adjusted p < 0.05.",
        "",
        "## Q3 real-text − shuffled-text contrasts",
        "",
        _markdown_table(
            text,
            (
                "generator_conditioning_mode",
                "critic_conditioning_mode",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "holm_p",
                "alignment_supported",
            ),
        ),
    ]
    if q4 is None:
        lines.extend(
            [
                "",
                "## Q4 status",
                "",
                "Q4 remains locked or unevaluated. No Q4 result is included in this report.",
            ]
        )
    else:
        exposure = q4["historical_q4_exposure"]
        lines.extend(
            [
                "",
                "## Q4 frozen exploratory evaluation",
                "",
                (
                    "Q4 is **historically exposed**: all **143/143** exact-grid Q4 pairs "
                    "overlap the prior Q4 evaluation panel. These results are therefore "
                    "**retrospective/frozen exploratory, not confirmatory**, even though "
                    "the current selection and refit recipes were frozen before this run."
                ),
                "",
                (
                    f"The primary panel contains {q4['panel_lineage']['5m_primary']['counts']['pairs']} "
                    "pairs across "
                    f"{q4['panel_lineage']['5m_primary']['counts']['sessions']} sessions; "
                    f"the historical overlap audit is {exposure['pair_overlap_label']} pairs."
                ),
                "",
                "### Primary Holm-4 contrasts",
                "",
                _markdown_table(
                    q4_primary,
                    (
                        "contrast",
                        "mean_difference",
                        "ci_95_lower",
                        "ci_95_upper",
                        "holm_p",
                        "holm_significant",
                        "improvement_supported",
                    ),
                ),
                "",
                "The 30m analysis is secondary lagged-news robustness and cannot change "
                "the frozen winner.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Selection payload SHA: `{selection['selection_sha256']}`",
            f"- Q4 summary SHA: `{q4['summary_sha256'] if q4 else 'not_available'}`",
            f"- Experiment root: `{root}`",
            "",
        ]
    )
    return "\n".join(lines)


def _html(
    root: Path,
    selection: Mapping[str, Any],
    architecture: pd.DataFrame,
    text: pd.DataFrame,
    q4: Mapping[str, Any] | None,
    q4_primary: pd.DataFrame,
) -> str:
    winner = selection["winner"]
    q4_block = "<h2>Q4 status</h2><p>Q4 remains locked or unevaluated.</p>"
    if q4 is not None:
        q4_block = (
            "<h2>Q4 frozen exploratory evaluation</h2>"
            "<div class='warning'><strong>Historically exposed:</strong> all "
            "<strong>143/143</strong> exact-grid Q4 pairs overlap the previous Q4 "
            "evaluation panel. This is retrospective/frozen exploratory and "
            "<strong>not confirmatory</strong>.</div>"
            "<h3>Primary Holm-4 contrasts</h3>"
            + _html_table(
                q4_primary,
                (
                    "contrast",
                    "mean_difference",
                    "ci_95_lower",
                    "ci_95_upper",
                    "holm_p",
                    "holm_significant",
                    "improvement_supported",
                ),
            )
            + "<p>The 30m analysis is secondary lagged-news robustness and cannot "
            "change the frozen winner.</p>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generator-FiLM × Critic-text factorial</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1,h2,h3{{line-height:1.2}} table{{border-collapse:collapse;width:100%;font-size:12px;margin:1rem 0}}
th,td{{border:1px solid #ccd3df;padding:.45rem;text-align:left;vertical-align:top}} th{{background:#eef2f8}}
.warning{{border-left:5px solid #a33;background:#fff0f0;padding:1rem}} code{{overflow-wrap:anywhere}}
</style></head><body>
<h1>News-first Vol Generator-FiLM × Critic-text Factorial</h1>
<h2>Result first</h2>
<p>The frozen Q3 decision is <strong>{escape(str(selection["selection_label"]))}</strong>.
The winner is Generator <code>{escape(str(winner["generator_conditioning_mode"]))}</code> +
Critic <code>{escape(str(winner["critic_conditioning_mode"]))}</code> with real text.</p>
<div class="warning"><strong>Single seed (42):</strong> the 10,000-draw CME-session bootstrap
does not estimate seed-to-seed or model-initialization uncertainty.</div>
<h2>Q3 architecture contrasts</h2>
{_html_table(architecture, ("candidate_generator_conditioning_mode", "candidate_critic_conditioning_mode", "mean_difference", "ci_95_lower", "ci_95_upper", "holm_p", "eligible"))}
<h2>Q3 real-text − shuffled-text contrasts</h2>
{_html_table(text, ("generator_conditioning_mode", "critic_conditioning_mode", "mean_difference", "ci_95_lower", "ci_95_upper", "holm_p", "alignment_supported"))}
{q4_block}
<h2>Reproducibility</h2><p>Selection payload SHA: <code>{escape(str(selection["selection_sha256"]))}</code><br>
Experiment root: <code>{escape(str(root))}</code></p>
</body></html>"""


def render_film_critic_report(root: str | Path) -> Path:
    """Render immutable-input Markdown and self-contained HTML conclusions."""

    experiment_root = Path(root).resolve(strict=False)
    selection = _load_selection(experiment_root)
    architecture = _display_contrasts(
        experiment_root / "analysis" / analysis.Q3_ARCH_CONTRASTS
    )
    text = _display_contrasts(experiment_root / "analysis" / analysis.Q3_TEXT_CONTRASTS)
    q4 = _load_q4(experiment_root)
    if q4 is not None and str(q4.get("selection_sha256")) != str(
        selection["selection_sha256"]
    ):
        raise FilmCriticReportError("Q4 summary is not bound to the Q3 selection")
    q4_primary = (
        _display_contrasts(experiment_root / "analysis" / analysis.Q4_PRIMARY_CONTRASTS)
        if q4 is not None
        else pd.DataFrame()
    )
    directory = experiment_root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / REPORT_MD
    html_path = directory / REPORT_HTML
    factorial._atomic_write_text(
        markdown_path,
        _markdown(experiment_root, selection, architecture, text, q4, q4_primary),
    )
    factorial._atomic_write_text(
        html_path,
        _html(experiment_root, selection, architecture, text, q4, q4_primary),
    )
    return markdown_path


__all__ = ["FilmCriticReportError", "render_film_critic_report"]
