"""Render separate Chapter 3 figures with the frozen Pure-CNN comparison.

The original two figures remain unchanged. No model is loaded and no
predictions are generated: both forecasts are saved 64-draw conditional means.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq3.chapter3_forecast_pure_comparison_data import (
    DEFAULT_PURE_SOURCE_ROOT,
    load_pure_comparison_inputs,
    with_pure_atm_values,
)
from scripts.rq3.plot_chapter3_forecast_examples import (
    BLUE, RED, MUTED, GRID,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_OUTPUT_ROOT as ORIGINAL_OUTPUT_ROOT,
    _clean_axes, _mesh, _save, _select_smile_example, _sha256, _style,
    _supported_quads,
)

DEFAULT_OUTPUT_ROOT = ORIGINAL_OUTPUT_ROOT / "with_pure_cnn"
SURFACE_STEM = "ch3_realised_film_pure_surfaces"
ATM_STEM = "ch3_atm_realised_film_pure_time_series"
SMILE_STEM = "ch3_realised_film_pure_shortest_maturity_smile"
SELECTION_FILENAME = "ch3_surface_smile_selection_candidates.csv"
GREEN = "#2A7F62"
SERIES = (
    ("realised", "Realised", BLUE, "-", "o"),
    ("predicted", "FiLM-CNN", RED, "--", "x"),
    ("pure_predicted", "Pure-CNN", GREEN, "-.", "^"),
)


def _surface_figure(selected, moneyness, maturities, output_root):
    mask = selected["joint_mask"]
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    display = np.zeros_like(mask)
    display[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1] = True
    if np.any(display & ~selected["target_mask"]):
        raise ValueError("The display rectangle exceeds realised interpolation support.")
    values = {field: 100.0 * selected[field] for field, *_ in SERIES}
    for field, array in values.items():
        if not np.isfinite(array[display]).all() or np.any(array[display] <= 0):
            raise ValueError(f"Invalid frozen IV in the displayed {field} surface.")
    combined = np.concatenate([array[display] for array in values.values()])
    low, high = float(combined.min()), float(combined.max())
    if high <= low:
        raise ValueError("The selected surfaces have no displayed IV variation.")
    norm = Normalize(low, high)
    cmap = matplotlib.colormaps["plasma"]
    quads = _supported_quads(display)
    original_quads = _supported_quads(mask)
    smile_row = int(rows[0])
    smile_mask = mask[smile_row]
    smile_x = moneyness[smile_mask]
    smile_maturity = float(maturities[smile_row])
    xlimits = [float(moneyness[columns[0]] - 0.002), float(moneyness[columns[-1]] + 0.002)]
    ylimits = [float(maturities[rows[0]] - 1), float(maturities[rows[-1]] + 1)]

    fig = plt.figure(figsize=(8, 8.2))
    fig.suptitle("Five-minute-ahead implied volatility", y=0.985, fontsize=15)
    target = selected["target_time_utc"]
    fig.text(0.5, 0.943, "Forecast target: " + target.strftime("%d %B %Y, %H:%M UTC"),
             ha="center", fontsize=12, color=MUTED)
    positions = ((0.075, 0.535, 0.34, 0.335), (0.535, 0.535, 0.34, 0.335),
                 (0.075, 0.085, 0.34, 0.335))
    titles = ("(a) Realised IVS", "(b) FiLM-CNN IVS", "(c) Pure-CNN IVS")
    for position, title, (field, label, colour, linestyle, marker) in zip(positions, titles, SERIES):
        axis = fig.add_axes(position, projection="3d")
        array = values[field]
        faces, colour_values = _mesh(array, quads, moneyness, maturities)
        axis.add_collection3d(Poly3DCollection(
            faces, facecolors=cmap(norm(colour_values)), edgecolors="#505050",
            linewidths=0.22, alpha=1, zsort="average",
        ))
        line, = axis.plot(
            moneyness, np.full_like(moneyness, smile_maturity),
            np.where(smile_mask, array[smile_row], np.nan),
            color=colour, linestyle=linestyle, linewidth=1.8,
            marker=marker, markersize=3.5, zorder=10,
        )
        line.set_path_effects([patheffects.Stroke(linewidth=3.2, foreground="white"),
                               patheffects.Normal()])
        axis.set_title(title, pad=10)
        axis.set_xlabel("$K/F$", labelpad=8)
        axis.yaxis.set_rotate_label(True)
        axis.set_ylabel("Maturity", labelpad=8)
        axis.set_zlabel("Annualised IV (%)", labelpad=7)
        axis.set_xlim(*xlimits)
        axis.set_ylim(*ylimits)
        padding = 0.06 * (high - low)
        axis.set_zlim(low - padding, high + padding)
        ticks = [xlimits[0], 1.0, xlimits[1]]
        axis.set_xticks(ticks, [f"{x:.3f}" for x in ticks])
        axis.set_yticks([10, 15, 21])
        axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(pad=1, labelsize=11.5)
        axis.view_init(elev=32, azim=-105)
        axis.set_box_aspect((1.15, 0.65, 0.80))
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.pane.set_facecolor((1, 1, 1, 1))
            coordinate.pane.set_edgecolor(GRID)
            coordinate._axinfo["grid"].update(color=GRID, linewidth=0.5)

    colour_axis = fig.add_axes((0.925, 0.58, 0.013, 0.235))
    colourbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis)
    colourbar.ax.set_title("IV (%)", fontsize=12, pad=9)
    colourbar.ax.tick_params(labelsize=11.5)
    colourbar.outline.set_linewidth(0.5)
    colourbar.solids.set_rasterized(False)

    axis = fig.add_axes((0.575, 0.155, 0.365, 0.26))
    _clean_axes(axis)
    axis.set_title("(d) Shortest-maturity smile", loc="left", pad=12)
    for field, label, colour, linestyle, marker in SERIES:
        axis.plot(smile_x, values[field][smile_row, smile_mask], color=colour,
                  linestyle=linestyle, linewidth=1.4 if field != "pure_predicted" else 1.1,
                  marker=marker, markersize=5 if field != "pure_predicted" else 4,
                  markerfacecolor="none" if field == "pure_predicted" else colour,
                  markeredgewidth=1 if marker != "o" else 0, label=label)
    ticks = np.linspace(smile_x[0], smile_x[-1], 5)
    axis.set_xticks(ticks, [f"{x:.3f}" for x in ticks])
    axis.set_xlabel("Moneyness, $K/F$")
    axis.set_xlim(*xlimits)
    axis.tick_params(labelsize=11.5)
    axis.margins(y=0.17)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                frameon=False, fontsize=11.5, handlelength=1.8,
                handletextpad=0.35, columnspacing=0.7, borderaxespad=0)
    axis.text(0.02, 0.97, f"{smile_maturity:g} business days", transform=axis.transAxes,
              va="top", fontsize=11.5, color=MUTED)
    _save(fig, output_root, SURFACE_STEM, "Realised, FiLM-CNN and Pure-CNN IV surfaces and smile")

    records = []
    for row, maturity in enumerate(maturities):
        for col, relative_strike in enumerate(moneyness):
            record = {
                "fold": selected["fold"], "pair_id": selected["pair_id"],
                "origin_time_utc": selected["origin_time_utc"].isoformat(),
                "target_time_utc": target.isoformat(),
                "maturity_business_days": float(maturity), "moneyness": float(relative_strike),
                "joint_support": bool(mask[row, col]),
                "target_support": bool(selected["target_mask"][row, col]),
                "display_support": bool(display[row, col]),
                "display_completion": bool(display[row, col] and not mask[row, col]),
            }
            for field, name in (("realised", "realised"), ("predicted", "film_predicted"),
                                ("pure_predicted", "pure_predicted")):
                record[f"{name}_iv_decimal"] = float(selected[field][row, col])
                record[f"{name}_iv_percent"] = float(values[field][row, col])
            records.append(record)
    data = pd.DataFrame(records)
    data.to_csv(output_root / f"{SURFACE_STEM}_data.csv", index=False)
    data.loc[data["maturity_business_days"].eq(smile_maturity) & data["joint_support"]].to_csv(
        output_root / f"{SMILE_STEM}_data.csv", index=False)
    return {
        "fold": str(selected["fold"]), "pair_id": str(selected["pair_id"]),
        "origin_time_utc": selected["origin_time_utc"].isoformat(),
        "target_time_utc": target.isoformat(), "session_id": str(selected["session_id"]),
        "supported_cell_count": int(mask.sum()), "supported_quad_count": len(original_quads),
        "display_node_count": int(display.sum()),
        "display_completion_node_count": int((display & ~mask).sum()),
        "rendered_quad_count_per_surface": len(quads),
        "rendered_triangle_count_per_surface": 2 * len(quads),
        "display_completion": {
            "method": "reuse_frozen_grids_in_realised_supported_rectangle",
            "realised_grid_construction": "original_total_variance_interpolation",
            "predicted_grids": "unchanged_frozen_64_draw_means",
            "target_support_verified": True, "evaluation_support": "original_joint_mask",
        },
        "colour_limits_iv_percent": [low, high], "colormap": "plasma",
        "axis_limits": {"moneyness": xlimits, "maturity_business_days": ylimits},
        "view": {"elevation_degrees": 32, "azimuth_degrees": -105,
                 "box_aspect_moneyness_maturity_iv": [1.15, 0.65, 0.80]},
        "shortest_maturity_smile": {"maturity_business_days": smile_maturity,
                                    "grid_row_index": smile_row,
                                    "supported_node_count": int(smile_mask.sum()),
                                    "moneyness_nodes": smile_x.tolist()},
        "panel_layout": "2_by_2_three_surfaces_one_smile", "figure_size_inches": [8, 8.2],
    }


def _plot_atm_series(axis, frame, marker_size):
    valid = frame.loc[frame["atm_valid"]]
    for field, label, colour, linestyle, marker in SERIES:
        atm_field = {"realised": "atm_realised_iv", "predicted": "atm_predicted_iv",
                     "pure_predicted": "atm_pure_iv"}[field]
        axis.plot(valid["target_time_utc"], 100.0 * valid[atm_field], color=colour,
                  linestyle=linestyle, linewidth=1.1, marker=marker,
                  markersize=float(np.sqrt(marker_size)) if marker != "x" else float(np.sqrt(1.45 * marker_size)),
                  markerfacecolor="none" if marker == "^" else colour,
                  markeredgewidth=0 if marker == "o" else 1, label=label)


def _atm_figure(pairs, detail_date, output_root):
    ordered = pairs.sort_values(["target_time_utc", "pair_id"]).reset_index(drop=True)
    valid = ordered.loc[ordered["atm_valid"]]
    date = pd.Timestamp(detail_date, tz="UTC")
    detail = ordered.loc[ordered["target_time_utc"].dt.normalize().eq(date)].reset_index(drop=True)
    detail_valid = detail.loc[detail["atm_valid"]]
    if detail_valid.empty:
        raise ValueError(f"No valid ATM observations on {detail_date}.")
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.095, hspace=0.55)
    fig.suptitle("Constant-maturity ATM implied volatility", y=0.975, fontsize=15)
    for axis in axes:
        _clean_axes(axis)
        axis.tick_params(labelsize=11.5)
    _plot_atm_series(axes[0], valid, marker_size=13)
    axes[0].set_title("(a) 2023 test observations", loc="left", pad=9)
    axes[0].set_xlabel("Forecast target date (UTC)")
    start, end = valid["target_time_utc"].min(), valid["target_time_utc"].max()
    ticks = [start.normalize()]
    for boundary in pd.date_range("2023-04-01", "2023-10-01", freq="QS", tz="UTC"):
        axes[0].axvline(boundary, color="#BDC3C8", linewidth=0.7, linestyle=(0, (3, 4)))
        if start < boundary < end:
            ticks.append(boundary)
    ticks.append(end.normalize())
    axes[0].set_xticks(ticks)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=timezone.utc))
    axes[0].get_xticklabels()[0].set_horizontalalignment("left")
    axes[0].get_xticklabels()[-1].set_horizontalalignment("right")
    axes[0].set_xlim(start - pd.Timedelta(days=4), end + pd.Timedelta(days=4))
    axes[0].margins(y=0.08)
    _plot_atm_series(axes[1], detail, marker_size=25)
    axes[1].set_title("(b) " + date.strftime("%d %B %Y"), loc="left", pad=9)
    axes[1].set_xlabel("Forecast target time (UTC)")
    first, last = detail["target_time_utc"].min(), detail["target_time_utc"].max()
    ticks = [first, *pd.date_range(first.ceil("h"), last.floor("h"), freq="h"), last]
    axes[1].set_xticks(sorted(set(ticks)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axes[1].get_xticklabels()[-1].set_horizontalalignment("right")
    axes[1].set_xlim(first - pd.Timedelta(minutes=7), last + pd.Timedelta(minutes=7))
    axes[1].margins(y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.932), ncol=3, frameon=False)
    _save(fig, output_root, ATM_STEM, "Realised, FiLM-CNN and Pure-CNN constant-maturity ATM IV")

    exported = ordered[["fold", "pair_id", "session_id", "origin_time_utc", "target_time_utc", "atm_valid"]].copy()
    exported["maturity_business_days"] = 21
    exported["atm_moneyness"] = 1.0
    for field, name in (("atm_realised_iv", "realised"), ("atm_predicted_iv", "film_predicted"),
                        ("atm_pure_iv", "pure_predicted")):
        exported[f"{name}_atm_iv_decimal"] = ordered[field]
        exported[f"{name}_atm_iv_percent"] = 100.0 * ordered[field]
    exported["is_detail_date"] = ordered["target_time_utc"].dt.normalize().eq(date)
    exported["detail_link_from_previous"] = ordered["pair_id"].isin(detail_valid["pair_id"].iloc[1:])
    exported.to_csv(output_root / f"{ATM_STEM}_data.csv", index=False, na_rep="")
    return {
        "maturity_business_days": 21, "moneyness": 1.0,
        "interpolation": "linear_in_variance_between_adjacent_moneyness_nodes",
        "moneyness_bracket": [0.998, 1.002], "variance_weights": [0.5, 0.5],
        "valid_pair_count": int(len(valid)),
        "valid_pairs_by_fold": {str(k): int(v) for k, v in valid.groupby("fold").size().items()},
        "valid_utc_date_count": int(valid["target_time_utc"].dt.normalize().nunique()),
        "valid_cme_session_count": int(valid["session_id"].nunique()),
        "detail_date_utc": detail_date, "detail_pair_count": int(len(detail)),
        "detail_valid_pair_count": int(len(detail_valid)),
        "detail_missing_pair_ids": detail.loc[~detail["atm_valid"], "pair_id"].tolist(),
        "line_connection_policy": "consecutive_valid_observations_on_actual_time_axis",
        "full_year_segment_count_per_series": len(valid) - 1,
        "detail_segment_count_per_series": len(detail_valid) - 1,
        "figure_size_inches": [7.5, 6.5],
    }


def _write_captions(output_root, selected, seed):
    target = selected["target_time_utc"].strftime("%d %B %Y at %H:%M UTC")
    surface = (
        f"Realised and predicted implied-volatility surfaces for the five-minute forecast target on {target}. "
        "The transition is selected for a pronounced realised smile at the shortest jointly supported maturity. "
        f"Panels (a)--(c) show the realised IV surface and the 64-draw conditional-mean forecasts from the "
        f"matched-LP FiLM-CNN and Pure-CNN (both seed {seed}). "
        r"The frozen grids are displayed over $K/F\in[0.986,1.014]$ and 6--21 business days, with common axis and colour scales. "
        "The realised grid follows the original total-variance interpolation; forecast evaluation retains the joint-support mask. "
        "The front-edge curves and panel (d) show the shortest jointly supported smile at 6 business days. "
        "Blue solid lines with circles denote realised IV, red dashed lines with crosses denote FiLM-CNN forecasts, "
        "and green dash-dot lines with triangles denote Pure-CNN forecasts. "
        "Maturity is measured in business days; IV is annualised and expressed as a percentage."
    )
    atm = (
        "Realised and predicted constant-maturity ATM implied volatility at the sampled five-minute forecast targets. "
        "Panel (a) shows the 188 supported test observations in 2023; panel (b) enlarges the 12 observations on 12 July 2023. "
        r"At a maturity of 21 business days, ATM IV is interpolated linearly in variance between $K/F=0.998$ and $K/F=1.002$. "
        f"Forecasts are the 64-draw conditional means from the matched-LP FiLM-CNN and Pure-CNN (both seed {seed}). "
        "Blue solid lines with circles denote realised IV, red dashed lines with crosses denote FiLM-CNN forecasts, "
        "and green dash-dot lines with triangles denote Pure-CNN forecasts. "
        "Markers identify sampling times; lines join consecutive available observations on the actual time axis. "
        "IV is annualised and expressed as a percentage, and all times are UTC."
    )
    placement = (
        "Suggested placement: in `Main Results: FiLM-CNN versus Pure-CNN Baseline`, after the forecast-accuracy table "
        "and its discussion, before the text-representation comparison. These are alternative figure versions; "
        "the existing figures and chapter text are preserved."
    )
    selection = (
        "The original example is retained. From 161 sufficiently supported transitions, select the 23 shortest-maturity "
        "smiles with at least five contiguous nodes spanning K/F=0.990 to 1.010, an IV trough within 0.006 of ATM, "
        "and at least two nodes on each monotone branch. Rank by the smaller endpoint-minus-trough IV difference, "
        "breaking ties by (fold, pair_id). Selection uses realised IV only."
    )
    (output_root / "captions.md").write_text(
        "# Chapter 3 forecast figures with Pure-CNN\n\n" + placement + "\n\n## Surface comparison\n\n" + surface
        + "\n\n## ATM time series\n\n" + atm + "\n\n## Example selection\n\n" + selection + "\n", encoding="utf-8")
    preview = "% Insertion preview; the existing chapter and figures are unchanged.\n"
    for stem, caption, label in ((SURFACE_STEM, surface, "fig:ch3:realised_film_pure_surfaces"),
                                (ATM_STEM, atm, "fig:ch3:atm_film_pure_time_series")):
        preview += (
            "\\begin{figure}[htbp]\n\\centering\n"
            "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/forecast_examples/with_pure_cnn/"
            + stem + ".pdf}\n\\caption{" + caption.replace("%", r"\%")
            + "}\n\\label{" + label + "}\n\\end{figure}\n\n"
        )
    (output_root / "chapter3_forecast_figures_preview.tex").write_text(preview, encoding="utf-8")


def render(film_source_root, pure_source_root, output_root):
    output_root = Path(output_root).resolve()
    original_root = ORIGINAL_OUTPUT_ROOT.resolve()
    if output_root == original_root or original_root.is_relative_to(output_root):
        raise ValueError("The comparison must be saved separately from the original figures.")
    original_hashes = {p.name: _sha256(p) for p in original_root.iterdir() if p.is_file()}
    pairs, moneyness, maturities, provenance = load_pure_comparison_inputs(
        film_source_root, pure_source_root=pure_source_root, seed=42)
    pairs = with_pure_atm_values(pairs, moneyness, maturities, maturity_days=21)
    if (len(pairs), int(pairs["surface_eligible"].sum()), int(pairs["atm_valid"].sum())) != (500, 161, 188):
        raise ValueError("The comparison panel differs from the agreed 500/161/188 counts.")
    selected, candidates, selection = _select_smile_example(pairs, moneyness, maturities)
    old_manifest = json.loads((original_root / "figure_provenance.json").read_text())
    if selected["pair_id"] != old_manifest["surface"]["pair_id"]:
        raise ValueError("The new comparison must retain the original example.")
    detail_date = "2023-07-12"
    date_counts = pairs.loc[pairs["atm_valid"]].groupby(pairs["target_time_utc"].dt.normalize()).size()
    if date_counts.idxmax() != pd.Timestamp(detail_date, tz="UTC") or date_counts.max() != 12:
        raise ValueError("The agreed detail date does not have the most ATM observations.")
    output_root.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_root / SELECTION_FILENAME, index=False)
    _style()
    plt.rcParams.update({"xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    surface = _surface_figure(selected, moneyness, maturities, output_root)
    atm = _atm_figure(pairs, detail_date, output_root)
    _write_captions(output_root, selected, seed=42)
    outputs = {}
    for name in (f"{SURFACE_STEM}.pdf", f"{SURFACE_STEM}.png", f"{SURFACE_STEM}_data.csv",
                 f"{SMILE_STEM}_data.csv", SELECTION_FILENAME,
                 f"{ATM_STEM}.pdf", f"{ATM_STEM}.png", f"{ATM_STEM}_data.csv",
                 "captions.md", "chapter3_forecast_figures_preview.tex"):
        path = output_root / name
        outputs[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    for name, expected in original_hashes.items():
        if _sha256(original_root / name) != expected:
            raise ValueError(f"Original figure asset changed: {name}")
    scripts = (Path(__file__), Path(__file__).with_name("chapter3_forecast_pure_comparison_data.py"),
               Path(__file__).with_name("chapter3_forecast_figure_data.py"),
               Path(__file__).with_name("plot_chapter3_forecast_examples.py"))
    manifest = {
        "schema_version": 1, "kind": "chapter3_frozen_forecast_examples_with_pure_cnn_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(output_root),
        "source_provenance": provenance,
        "prediction": {"film_arm": "lp_matched", "pure_arm": "pure_cnn_no_text", "model_seed": 42, "mc_draw_mean": 64},
        "surface_selection": selection, "surface": surface, "atm": atm,
        "original_assets_preserved": {"root": str(original_root), "sha256": original_hashes},
        "rendering": {
            "iv_scale": "annualised_percent", "source_iv_scale": "annualised_decimal_ACT_365",
            "decimal_to_percent_multiplier": 100, "timezone": "UTC", "png_dpi": 600,
            "pdf_fonttype": 42, "font_family": "DejaVu Sans", "minimum_tick_font_size_pt": 11.5,
            "series": {label: {"colour": colour, "linestyle": linestyle, "marker": marker}
                       for field, label, colour, linestyle, marker in SERIES},
            "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
        },
        "code_sha256": {str(path): _sha256(path) for path in scripts}, "outputs": outputs,
    }
    (output_root / "figure_provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film-source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--pure-source-root", type=Path, default=DEFAULT_PURE_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = render(args.film_source_root.expanduser().resolve(), args.pure_source_root.expanduser().resolve(),
                    args.output_root.expanduser().resolve())
    print(json.dumps({"output_root": result["output_root"], "surface": result["surface"], "atm": result["atm"]}, indent=2))


if __name__ == "__main__":
    main()
