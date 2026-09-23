"""Durable reports for the Pure-CNN parent -> FiLM text-effect experiment.

The renderer is intentionally data-only: it does not read predictions, choose
checkpoints, or rerun statistics.  It accepts the frozen output of
``analyze_backbone_text_effect`` and writes answer-first Markdown plus a
self-contained HTML document with matching values.
"""

from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


INTERPRETATION = "retrospective_rolling_development_text_effectiveness"
REPORT_STEM = "film_unet_pure_cnn_backbone_text_effect_10seed_report"


class TextEffectReportError(ValueError):
    """Raised when report inputs do not match the frozen analysis contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate_result(result: Mapping[str, Any]) -> None:
    required_tables = {
        "main_comparisons",
        "intervention_comparisons",
        "validation_trajectory",
        "validation_trajectory_cells",
        "validation_trajectory_arms",
        "validation_trajectory_comparisons",
        "validation_trajectory_snapshots",
    }
    missing = sorted(required_tables - set(result))
    if missing:
        raise TextEffectReportError(f"Analysis result lacks tables: {missing}")
    for name in required_tables:
        if not isinstance(result[name], pd.DataFrame) or result[name].empty:
            raise TextEffectReportError(f"Analysis table is empty or invalid: {name}")
    conclusion = result.get("conclusion")
    if not isinstance(conclusion, Mapping) or not str(conclusion.get("status", "")):
        raise TextEffectReportError("Analysis conclusion is missing its status")
    for name in ("main_comparisons", "intervention_comparisons"):
        table = result[name]
        if len(table) != 2 or not table["holm_family_size"].eq(2).all():
            raise TextEffectReportError(f"{name} is not a frozen Holm-2 family")
        if not table["bootstrap_iterations"].eq(10_000).all():
            raise TextEffectReportError(
                f"{name} was not computed with 10,000 bootstrap draws"
            )
    trajectory = result["validation_trajectory_comparisons"]
    if len(trajectory) != 2 or not {
        "passes_optimization_gate",
        "comparison_id",
    }.issubset(trajectory.columns):
        raise TextEffectReportError(
            "validation_trajectory_comparisons is not the frozen two-row family"
        )


def _status_sentence(conclusion: Mapping[str, Any]) -> str:
    status = str(conclusion["status"])
    descriptions = {
        "stable_text_mae_increment": (
            "Matched LP text has a stable MAE increment under all three "
            "predeclared evidence layers."
        ),
        "optimization_only": (
            "The evidence is limited to an optimization/training-path effect; "
            "the complete text-effect claim is not supported."
        ),
        "not_generalized": (
            "The observed text signal does not pass the frozen-test "
            "generalization requirement."
        ),
        "no_verified_text_reliance": (
            "The experiment does not verify stable reliance on text for lower MAE."
        ),
    }
    return descriptions.get(status, f"Frozen conclusion status: {status}.")


def _comparison_view(table: pd.DataFrame, *, gate_column: str) -> pd.DataFrame:
    columns = [
        "comparison_id",
        "focal_mean_mae",
        "reference_mean_mae",
        "reference_minus_focal_mae",
        "geometric_improvement_percent",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        gate_column,
    ]
    missing = [column for column in columns if column not in table]
    if missing:
        raise TextEffectReportError(f"Comparison table lacks columns: {missing}")
    result = table[columns].copy()
    for column in (
        "focal_mean_mae",
        "reference_mean_mae",
        "reference_minus_focal_mae",
        "geometric_improvement_percent",
        "ci_95_lower",
        "ci_95_upper",
        "holm_adjusted_p",
    ):
        result[column] = result[column].map(lambda value: f"{float(value):.8g}")
    result[gate_column] = result[gate_column].map(
        lambda value: "PASS" if value else "FAIL"
    )
    return result


def _trajectory_view(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison_id",
        "epoch30_gain_difference_equal_cell_mean",
        "epoch1_30_auc_per_interval_difference_equal_cell_mean",
        "consistent_seed_count",
        "consistent_fold_count",
        "passes_optimization_gate",
    ]
    missing = [column for column in columns if column not in table]
    if missing:
        raise TextEffectReportError(f"Trajectory table lacks columns: {missing}")
    result = table[columns].copy()
    for column in columns[1:3]:
        result[column] = result[column].map(lambda value: f"{float(value):.8g}")
    result["passes_optimization_gate"] = result["passes_optimization_gate"].map(
        lambda value: "PASS" if value else "FAIL"
    )
    return result


def _snapshot_view(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "arm",
        "checkpoint_label",
        "epoch",
        "target_mae_equal_cell_mean",
        "gain_equal_cell_mean",
    ]
    missing = [column for column in columns if column not in table]
    if missing:
        raise TextEffectReportError(f"Snapshot table lacks columns: {missing}")
    primary = table.loc[
        table["arm"].isin({"matched", "film_zero_text", "film_lp_shuffle"})
    ][columns].copy()
    if primary.empty:
        raise TextEffectReportError("Snapshot table lacks primary validation arms")
    for column in ("target_mae_equal_cell_mean", "gain_equal_cell_mean"):
        primary[column] = primary[column].map(lambda value: f"{float(value):.8g}")
    return primary.reset_index(drop=True)


def render_text_effect_markdown(
    result: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
) -> str:
    """Render the frozen result as an answer-first Markdown report."""

    _validate_result(result)
    conclusion = result["conclusion"]
    metadata = dict(metadata or {})
    standard = _comparison_view(
        result["main_comparisons"], gate_column="passes_support_gate"
    )
    interventions = _comparison_view(
        result["intervention_comparisons"], gate_column="passes_support_gate"
    )
    trajectory = _trajectory_view(result["validation_trajectory_comparisons"])
    snapshots = _snapshot_view(result["validation_trajectory_snapshots"])
    lines = [
        "# Pure-CNN Parent → FiLM Text: 10-Seed Text-Effect Report",
        "",
        "## Outcome",
        "",
        f"**Status:** `{conclusion['status']}`",
        "",
        _status_sentence(conclusion),
        "",
        "A stable text-effect claim is made only when matched text beats both "
        "equal-budget zero/shuffled training branches, the same matched checkpoint "
        "gets worse under zero/wrong text, and validation gain is faster than both "
        "controls.",
        "",
        "## Evidence gates",
        "",
        "| Layer | Complete family supported |",
        "|---|---:|",
        "| Validation optimization | "
        f"{bool(conclusion['validation_trajectory_family_all_supported'])} |",
        "| Frozen-test branch comparison | "
        f"{bool(conclusion['main_standard_family_all_supported'])} |",
        "| Same-checkpoint text intervention | "
        f"{bool(conclusion['intervention_family_all_supported'])} |",
        "",
        "## Validation gain trajectory",
        "",
        _markdown_table(snapshots),
        "",
        _markdown_table(trajectory),
        "",
        "`Gain = MAE_parent − MAE_arm`; positive matched-minus-control differences "
        "favour matched text. The epoch-1–30 AUC is a trapezoidal approximation "
        "through the frozen 1/5/10/20/30 snapshots.",
        "",
        "## Frozen-test branch comparisons",
        "",
        _markdown_table(standard),
        "",
        "Negative log-MAE ratios and positive improvement percentages favour "
        "matched text. `reference_minus_focal_mae` is the absolute text increment "
        "(for zero text, exactly MAE_zero − MAE_matched). Both rows form one "
        "Holm-2 family.",
        "",
        "## Same-checkpoint text interventions",
        "",
        _markdown_table(interventions),
        "",
        "The matched checkpoint weights and MC noise are fixed; only the LP input "
        "is kept matched, zeroed, or independently shuffled. Both rows form a "
        "separate Holm-2 family.",
        "",
        "## Design and scope",
        "",
        f"- Interpretation: `{metadata.get('interpretation', INTERPRETATION)}`",
        f"- Seeds: {metadata.get('seed_count', conclusion.get('seed_count', 10))}",
        f"- Rolling folds: {metadata.get('fold_count', conclusion.get('fold_count', 4))}",
        f"- Bootstrap draws per comparison: {conclusion['bootstrap_iterations_per_comparison']}",
        "- Resampling: seed → fold → paired CME-session; frozen-test MC=64.",
        "- This is retrospective rolling development evidence, not a new "
        "confirmatory holdout and not a causal claim.",
        "",
    ]
    if metadata:
        lines.extend(
            [
                "## Artifact lineage",
                "",
                "```json",
                json.dumps(
                    _jsonable(metadata), ensure_ascii=False, sort_keys=True, indent=2
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="metrics", escape=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def render_text_effect_html(
    result: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
) -> str:
    """Render the frozen result as a self-contained HTML report."""

    _validate_result(result)
    conclusion = result["conclusion"]
    metadata = dict(metadata or {})
    standard = _comparison_view(
        result["main_comparisons"], gate_column="passes_support_gate"
    )
    interventions = _comparison_view(
        result["intervention_comparisons"], gate_column="passes_support_gate"
    )
    trajectory = _trajectory_view(result["validation_trajectory_comparisons"])
    snapshots = _snapshot_view(result["validation_trajectory_snapshots"])
    gates = pd.DataFrame(
        [
            {
                "layer": "Validation optimization",
                "complete_family_supported": bool(
                    conclusion["validation_trajectory_family_all_supported"]
                ),
            },
            {
                "layer": "Frozen-test branch comparison",
                "complete_family_supported": bool(
                    conclusion["main_standard_family_all_supported"]
                ),
            },
            {
                "layer": "Same-checkpoint text intervention",
                "complete_family_supported": bool(
                    conclusion["intervention_family_all_supported"]
                ),
            },
        ]
    )
    metadata_json = escape(
        json.dumps(_jsonable(metadata), ensure_ascii=False, sort_keys=True, indent=2)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pure-CNN Parent to FiLM Text Effect</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1180px; margin: 2rem auto;
       padding: 0 1rem; color: #172033; line-height: 1.5; }}
h1, h2 {{ color: #102a43; }}
.outcome {{ border-left: 6px solid #2563eb; background: #eff6ff; padding: 1rem; }}
code, pre {{ background: #f3f4f6; }}
pre {{ padding: 1rem; overflow: auto; }}
table.metrics {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
table.metrics th, table.metrics td {{ border: 1px solid #d8dee9; padding: .45rem;
                                     text-align: right; }}
table.metrics th:first-child, table.metrics td:first-child {{ text-align: left; }}
.caveat {{ color: #5b6472; font-size: .95rem; }}
</style>
</head>
<body>
<h1>Pure-CNN Parent → FiLM Text: 10-Seed Text-Effect Report</h1>
<section class="outcome">
<h2>Outcome</h2>
<p><strong>Status:</strong> <code>{escape(str(conclusion["status"]))}</code></p>
<p>{escape(_status_sentence(conclusion))}</p>
<p>A stable claim requires all three predeclared layers below.</p>
</section>
<h2>Evidence gates</h2>
{_html_table(gates)}
<h2>Validation gain trajectory</h2>
{_html_table(snapshots)}
{_html_table(trajectory)}
<p>Gain = parent MAE − arm MAE. The epoch-1–30 AUC is integrated through the
frozen 1/5/10/20/30 snapshots.</p>
<h2>Frozen-test branch comparisons</h2>
{_html_table(standard)}
<h2>Same-checkpoint text interventions</h2>
{_html_table(interventions)}
<h2>Design and scope</h2>
<ul>
<li>Interpretation: <code>{escape(str(metadata.get("interpretation", INTERPRETATION)))}</code></li>
<li>10,000 seed → fold → paired CME-session bootstrap draws per comparison.</li>
<li>Each test family is corrected with Holm-2.</li>
</ul>
<p class="caveat">This is retrospective rolling development evidence, not a new
confirmatory holdout and not a causal claim.</p>
<h2>Artifact lineage</h2>
<pre>{metadata_json}</pre>
</body>
</html>
"""


def write_analysis_artifacts(
    output_dir: str | Path,
    result: Mapping[str, Any],
    *,
    interpretation: str = INTERPRETATION,
) -> dict[str, Path]:
    """Write all frozen analysis tables, conclusion, and a SHA manifest."""

    _validate_result(result)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            path = root / f"{name}.csv"
            _atomic_text(path, value.to_csv(index=False, lineterminator="\n"))
            paths[name] = path
    conclusion_path = root / "conclusion.json"
    _atomic_text(
        conclusion_path,
        json.dumps(
            _jsonable(result["conclusion"]),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    paths["conclusion"] = conclusion_path
    manifest = {
        "schema_version": 1,
        "kind": "film_unet_pure_cnn_backbone_text_effect_analysis_manifest_v1",
        "interpretation": interpretation,
        "artifacts": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                **(
                    {"row_count": int(len(result[name]))}
                    if isinstance(result.get(name), pd.DataFrame)
                    else {}
                ),
            }
            for name, path in sorted(paths.items())
        },
    }
    manifest_path = root / "analysis_manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    paths["manifest"] = manifest_path
    return paths


def write_text_effect_report(
    output_dir: str | Path,
    result: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write matching Markdown/HTML reports and a hash-bound report artifact."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    markdown_path = root / f"{REPORT_STEM}.md"
    html_path = root / f"{REPORT_STEM}.html"
    _atomic_text(markdown_path, render_text_effect_markdown(result, metadata=metadata))
    _atomic_text(html_path, render_text_effect_html(result, metadata=metadata))
    artifact = {
        "schema_version": 1,
        "kind": "film_unet_pure_cnn_backbone_text_effect_report_artifact_v1",
        "interpretation": str((metadata or {}).get("interpretation", INTERPRETATION)),
        "conclusion_status": str(result["conclusion"]["status"]),
        "reports": {
            "markdown": {
                "path": str(markdown_path),
                "size_bytes": markdown_path.stat().st_size,
                "sha256": _sha256_file(markdown_path),
            },
            "html": {
                "path": str(html_path),
                "size_bytes": html_path.stat().st_size,
                "sha256": _sha256_file(html_path),
                "self_contained": True,
            },
        },
    }
    artifact_path = root / "report_artifact.json"
    _atomic_text(
        artifact_path,
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return {
        "markdown": markdown_path,
        "html": html_path,
        "artifact": artifact_path,
    }


__all__ = [
    "INTERPRETATION",
    "REPORT_STEM",
    "TextEffectReportError",
    "render_text_effect_html",
    "render_text_effect_markdown",
    "write_analysis_artifacts",
    "write_text_effect_report",
]
