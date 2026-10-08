"""Render Chapter 3 paired-effect figures from the frozen bootstrap archive.

No forecasts, bootstrap draws, probabilities, or adjustment families are
recomputed. The geometric transformation is checked against archived values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO_ROOT / "outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2"
DEFAULT_OUTPUT = (REPO_ROOT.parent / "PhdThesis/Chapter3/Chapter3Figs"
                  / "additional_figures/paired_effects")
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
BLUE = "#245a87"
DARK = "#3d434a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Archive member escapes source root: {name}")
    return path


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 12, "axes.titlesize": 12.5,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "text.color": DARK, "axes.labelcolor": DARK,
        "xtick.color": DARK, "ytick.color": DARK,
    })


def _load_archive(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    hashes_path = root / "output_hashes.csv"
    hashes = pd.read_csv(hashes_path)
    if hashes.relative_path.duplicated().any():
        raise ValueError("Duplicate archive checksum entry")
    checksums = hashes.set_index("relative_path").sha256.to_dict()
    checked: dict[str, str] = {}

    def check(name: str) -> Path:
        path = _member(root, name)
        actual = _sha256(path)
        if actual != checksums.get(name):
            raise ValueError(f"Archived SHA256 mismatch or missing checksum: {name}")
        checked[name] = actual
        return path

    table_path = check("analysis/all_contrasts.csv")
    manifest = json.loads(check("analysis_manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads(check("input_manifest.json").read_text(encoding="utf-8"))
    qa = json.loads(check("qa.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != "chapter3_shared_market_panel_bootstrap_v2"
            or manifest.get("iterations") != 10000
            or manifest.get("ci_method") != "percentile_95"
            or manifest.get("bootstrap_method_version")
            != "crossed_seed_shared_fold_occurrence_session_v2"
            or not qa.get("passed")):
        raise ValueError("Unexpected archive method, iteration count, or QA status")
    config_path = check(inputs["config_archive_path"])
    if _sha256(config_path) != inputs["config_sha256"]:
        raise ValueError("Archived configuration/input-manifest hash mismatch")
    for implementation in inputs["implementation"]:
        archived = check(implementation["archived_path"])
        if _sha256(archived) != implementation["sha256"]:
            raise ValueError("Archived implementation/input-manifest hash mismatch")
    wanted = {f"direct_{fold}" for fold in FOLDS} | {"rq3_branch", "rq3_intervention"}
    jobs = [job for job in manifest["jobs"] if job["job_id"] in wanted]
    if {job["job_id"] for job in jobs} != wanted or len(jobs) != 6:
        raise ValueError("Missing or repeated source jobs for paired-effect figures")
    for job in jobs:
        for key in ("draws_path", "panel_path", "schedule_path"):
            check(job[key])
    table = pd.read_csv(table_path, float_precision="round_trip")
    if table.contrast_id.duplicated().any():
        raise ValueError("Duplicate archived contrast_id")
    metadata = {
        "archive_root": str(root), "source_table_path": str(table_path),
        "source_table_sha256": _sha256(table_path),
        "archive_hash_manifest_path": str(hashes_path),
        "archive_hash_manifest_sha256": _sha256(hashes_path),
        "checked_archive_members": checked,
        "bootstrap_method_version": manifest["bootstrap_method_version"],
        "iterations": manifest["iterations"], "ci_method": manifest["ci_method"],
        "p_value_interpretation": manifest["p_value_interpretation"],
        "input_manifest_path": str(root / "input_manifest.json"),
        "archive_qa_passed": bool(qa["passed"]),
        "source_input_hash_records": [
            record for record in inputs["inputs"]
            if record.get("source_id") in {"direct", "rq3_branch", "rq3_intervention"}
        ],
        "verification_scope": "archived contrasts, method, QA, configuration, implementation, panels, draws, and schedules",
        "original_inputs_rehashed": False,
    }
    return table, {job["job_id"]: job for job in jobs}, metadata


def _select_rows(table: pd.DataFrame, jobs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = table.set_index("contrast_id", drop=False)
    rq3_spec = [
        ("matched_vs_film_zero_text", "rq3_branch", "(a) Equal-budget continuation branches", "Zero-text branch", "rq3_branch_holm2"),
        ("matched_vs_film_lp_shuffle", "rq3_branch", "(a) Equal-budget continuation branches", "Shuffled-LP branch", "rq3_branch_holm2"),
        ("matched_input_vs_zero_input", "rq3_intervention", "(b) Same-checkpoint text interventions", "Zero input", "rq3_intervention_holm2"),
        ("matched_input_vs_wrong_input", "rq3_intervention", "(b) Same-checkpoint text interventions", "Shuffled LP input", "rq3_intervention_holm2"),
    ]
    rq12_spec = []
    for panel, reference, label in (
        ("(a) Matched LP versus Pure-CNN", "no_text", "rq1"),
        ("(b) Matched LP versus BoW", "bow", "rq2"),
        ("(c) Matched LP versus sentiment", "sentiment", "rq2"),
    ):
        for number, fold in enumerate(FOLDS, start=1):
            family = ("rq1_lp_matched_vs_no_text_rolling_holm4" if label == "rq1"
                      else f"rq2_representations_{fold}_holm2")
            rq12_spec.append((f"{label}_{fold}_lp_matched_vs_{reference}", f"direct_{fold}",
                              panel, f"2023Q{number}", family))

    def collect(specs: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
        rows = []
        for order, (contrast, job_id, panel, plot_label, family) in enumerate(specs):
            if contrast not in indexed.index:
                raise ValueError(f"Missing archived contrast: {contrast}")
            row = indexed.loc[contrast].copy()
            if row.job_id != job_id or row.family_id != family:
                raise ValueError(f"Frozen comparison/family mismatch: {contrast}")
            declarations = [x for x in jobs[job_id]["contrasts"] if x["contrast_id"] == contrast]
            if len(declarations) != 1:
                raise ValueError(f"Missing manifest comparison declaration: {contrast}")
            for key in ("focal", "reference", "family_id", "alternative", "apply_holm"):
                if row[key] != declarations[0][key]:
                    raise ValueError(f"Archived comparison declaration {key} mismatch: {contrast}")
            if (row.estimand != "equal_cell" or row.seed_count != 10
                    or row.alternative != "one_sided" or not row.apply_holm
                    or not np.isclose(row.reported_p, row.holm_p, atol=1e-15, rtol=0)):
                raise ValueError(f"Unexpected archived estimand or Holm probability: {contrast}")
            if row.family_size != (4 if family.endswith("holm4") else 2):
                raise ValueError(f"Archived family size mismatch: {contrast}")
            for field, value in (
                ("geometric_gain_percent", -100 * np.expm1(row.point)),
                ("gain_ci_lower_percent", -100 * np.expm1(row.ci_upper)),
                ("gain_ci_upper_percent", -100 * np.expm1(row.ci_lower)),
            ):
                if not np.isfinite(row[field]) or not np.isclose(row[field], value, atol=2e-11, rtol=0):
                    raise ValueError(f"Archived geometric transformation mismatch: {contrast}/{field}")
            if not (row.ci_lower <= row.ci_upper and row.gain_ci_lower_percent <= row.gain_ci_upper_percent):
                raise ValueError(f"Interval ordering mismatch: {contrast}")
            row["plot_panel"] = panel
            row["plot_label"] = plot_label
            row["plot_order"] = order
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    return collect(rq3_spec), collect(rq12_spec)


def _save(fig: plt.Figure, root: Path, stem: str, title: str) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (
            box.x0 < boundary.x0 - 1 or box.y0 < boundary.y0 - 1
            or box.x1 > boundary.x1 + 1 or box.y1 > boundary.y1 + 1
        ):
            clipped.append(text.get_text())
    if clipped:
        raise ValueError(f"Figure text extends outside canvas: {clipped}")
    if fig._suptitle is not None:
        heading = fig._suptitle.get_window_extent(renderer)
        if heading.overlaps(fig.axes[0]._left_title.get_window_extent(renderer)):
            raise ValueError("Figure heading overlaps the first panel title")
    if fig.texts:
        footer = fig.texts[-1].get_window_extent(renderer)
        xlabel = fig.axes[-1].xaxis.label.get_window_extent(renderer)
        if footer.overlaps(xlabel):
            raise ValueError("Direction note overlaps the horizontal-axis label")
    fig.savefig(root / f"{stem}.pdf", metadata={"Title": title, "CreationDate": None, "ModDate": None})
    fig.savefig(root / f"{stem}.png", dpi=600)
    result = {"figure_size_inches": fig.get_size_inches().tolist(),
              "text_bounds_checked": True, "heading_and_footer_overlap_checked": True,
              "minimum_font_size_pt": 11.5}
    plt.close(fig)
    return result


def _forest_panels(rows: pd.DataFrame, root: Path, stem: str, title: str,
                   limits: tuple[float, float], tick_interval: float,
                   height: float) -> dict[str, Any]:
    panels = rows.plot_panel.drop_duplicates().tolist()
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, height), sharex=True,
                             squeeze=False)
    fig.subplots_adjust(left=0.255, right=0.815,
                        bottom=0.20 if len(panels) == 2 else 0.145,
                        top=0.82 if len(panels) == 2 else 0.89,
                        hspace=0.62 if len(panels) == 2 else 0.56)
    fig.suptitle(title, y=0.974, fontsize=14)
    for index, panel in enumerate(panels):
        ax = axes[index, 0]
        data = rows.loc[rows.plot_panel.eq(panel)].reset_index(drop=True)
        y = np.arange(len(data))
        ax.set_title(panel, loc="left", pad=11)
        ax.axvline(0, color="#747b83", lw=1, zorder=1)
        ax.hlines(y, data.gain_ci_lower_percent, data.gain_ci_upper_percent,
                  color=DARK, lw=1.4, zorder=2)
        ax.vlines(data.gain_ci_lower_percent, y - .075, y + .075, color=DARK, lw=1.2)
        ax.vlines(data.gain_ci_upper_percent, y - .075, y + .075, color=DARK, lw=1.2)
        marker = "D" if panel.startswith("(b) Same") else "o"
        ax.scatter(data.geometric_gain_percent, y, color=BLUE, marker=marker, s=45, zorder=3)
        ax.set_yticks(y, data.plot_label)
        ax.set_ylim(len(data) - .55, -.55)
        ax.set_xlim(limits)
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.grid(axis="x", color="#e9ebee", linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=11)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#a7acb2")
        ax.text(1.16, 1.12, r"$p_{\mathrm{Holm}}$", transform=ax.transAxes,
                ha="center", va="center", fontsize=11.5)
        for position, probability in enumerate(data.reported_p):
            ax.text(1.16, position, f"{probability:.4f}",
                    transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=11.5)
        if index < len(panels) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1, 0].set_xlabel("Geometric MAE gain for matched LP (%)", labelpad=10)
    fig.text(.535, .038, "Positive gain favours matched LP",
             ha="center", va="center", fontsize=11.5)
    rendered = _save(fig, root, stem, title)
    rendered.update({"x_limits": list(limits), "panels": panels,
                     "row_count": len(rows), "common_x_axis": True,
                     "point_colour": BLUE, "interval_colour": DARK,
                     "significance_colouring": False})
    return rendered


def _captions(root: Path) -> None:
    rq3 = (
        "Paired evidence on incremental text value. Panel (a) compares matched-LP "
        "FiLM continuations with equal-budget zero-text and shuffled-LP branches "
        "initialised from the same parent state. Panel (b) fixes the matched-LP "
        "checkpoints and Monte Carlo noise and replaces text only at inference. "
        "Points show original-sample geometric MAE gains, computed as "
        "$100[1-\\exp(\\widehat{\\theta})]$ from equal-seed--fold mean log-MAE "
        "ratios; positive values favour matched text. Horizontal bars are "
        "unadjusted 95\\% percentile intervals from 10,000 shared-market-panel "
        "bootstrap draws. The two panels retain separate one-sided Holm-2 "
        "families; the displayed probabilities are the archived adjusted "
        "bootstrap sign-tail probabilities. Both panels use a common scale."
    )
    rq12 = (
        "Quarterly paired forecast comparisons under direct training. Panels "
        "(a)--(c) compare matched-LP FiLM-CNN with Pure-CNN, BoW, and sentiment, "
        "respectively. Points show original-sample geometric MAE gains from "
        "equal-seed mean log-MAE ratios within each test quarter; positive "
        "values favour matched LP. Horizontal bars are unadjusted 95\\% "
        "percentile intervals from 10,000 shared-market-panel bootstrap draws. "
        "Displayed adjusted bootstrap sign-tail probabilities retain the "
        "original one-sided families: Holm-4 across quarters for the "
        "FiLM-CNN/Pure-CNN comparison and a separate Holm-2 representation "
        "family within each quarter for matched LP versus BoW and sentiment. "
        "The three panels share the same horizontal scale."
    )
    entries = [
        ("ch3_rq3_paired_effects", "fig:ch3:rq3_paired_effects", "Paired evidence on incremental text value", rq3),
        ("ch3_rq12_quarterly_paired_effects", "fig:ch3:rq12_quarterly_paired_effects", "Quarterly paired forecast comparisons", rq12),
    ]
    note = (
        "Suggested locations: place the RQ1/2 figure after the interpretation "
        "of Table `tab:ch3:rq2_direct_results`, before Incremental Text Value. "
        "Place the RQ3 figure after the same-checkpoint intervention discussion, "
        "before Greater Text Value during Events. These previews are not "
        "included in the existing chapter."
    )
    markdown = ["# Chapter 3 paired-effect figure captions", "", note, ""]
    tex = ["% Standalone insertion preview; not included in Chapter3/chapter3.tex.", ""]
    for stem, label, short, caption in entries:
        markdown.extend([f"## {short}", "", caption, ""])
        tex.extend([
            "\\begin{figure}[htbp]", "\\centering",
            "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/additional_figures/paired_effects/" + stem + ".pdf}",
            f"\\caption[{short}]{{{caption}}}", f"\\label{{{label}}}", "\\end{figure}", "",
        ])
    (root / "captions.md").write_text("\n".join(markdown), encoding="utf-8")
    (root / "insertion_preview.tex").write_text("\n".join(tex), encoding="utf-8")


def render(archive_root: Path = DEFAULT_ARCHIVE,
           output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    archive_root = Path(archive_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    table, jobs, sources = _load_archive(archive_root)
    rq3, rq12 = _select_rows(table, jobs)
    output_root.mkdir(parents=True, exist_ok=True)
    _style()
    rendering = {
        "rq3": _forest_panels(rq3, output_root, "ch3_rq3_paired_effects",
                               "Incremental text value: paired effects",
                               (-.23, .51), .1, 4.9),
        "rq12": _forest_panels(rq12, output_root, "ch3_rq12_quarterly_paired_effects",
                                "Quarterly direct-training paired effects",
                                (-.40, 1.47), .25, 8.0),
    }
    rq3.to_csv(output_root / "ch3_rq3_paired_effects_data.csv", index=False, float_format="%.17g")
    rq12.to_csv(output_root / "ch3_rq12_quarterly_paired_effects_data.csv", index=False, float_format="%.17g")
    _captions(output_root)
    names = [f"{stem}.{extension}"
             for stem in ("ch3_rq3_paired_effects", "ch3_rq12_quarterly_paired_effects")
             for extension in ("pdf", "png")] + [
                 "ch3_rq3_paired_effects_data.csv", "ch3_rq12_quarterly_paired_effects_data.csv",
                 "captions.md", "insertion_preview.tex",
             ]
    provenance = {
        "kind": "chapter3_frozen_paired_effect_figures_v1", "schema_version": 1,
        "source_provenance": sources, "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "output_root": str(output_root), "rendering": rendering,
        "png_dpi": 600, "pdf_fonttype": 42,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "validation": {"archived_rows_preserved": True, "geometric_transformations_checked": True,
                       "ci_endpoints_reversed_under_monotone_gain_transform": True,
                       "adjustment_families_preserved": True, "bootstrap_recomputed": False,
                       "rq3_rows": 4, "rq12_rows": 12},
        "outputs": {name: {"sha256": _sha256(output_root / name),
                            "size_bytes": (output_root / name).stat().st_size}
                    for name in names},
    }
    (output_root / "figure_provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    provenance = render(args.archive_root, args.output_root)
    print(json.dumps({"output_root": provenance["output_root"],
                      "validation": provenance["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
