"""Produce the two Chapter 3 figures from frozen, original-protocol forecasts.

No model is loaded: predicted IVs are the saved 64-draw conditional means.
The data helper validates the frozen source hashes, pair alignment and masks.
"""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.rq3.chapter3_forecast_figure_data import (
    load_frozen_inputs,
    with_atm_values,
)


DEFAULT_SOURCE_ROOT = ROOT / (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_10seed_"
    "lr2p5e5_exact_ttm_rolling_v1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT.parent / "PhdThesis/Chapter3/Chapter3Figs/forecast_examples"
)
SURFACE_STEM = "ch3_realised_predicted_surface"
ATM_STEM = "ch3_atm_realised_predicted_time_series"
SMILE_STEM = "ch3_shortest_maturity_smile"
SELECTION_FILENAME = "ch3_surface_smile_selection_candidates.csv"
INK = "#202020"
BLUE = "#2F75B5"
RED = "#C43C39"
MUTED = "#62676D"
GRID = "#E2E5E8"


def _select_smile_example(
    pairs: pd.DataFrame, moneyness: np.ndarray, maturities: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Choose a pronounced realised smile on each pair's shortest supported row."""
    candidates = []
    tolerance = 1e-12
    eligible = pairs.loc[pairs["surface_eligible"]]
    for _, item in eligible.iterrows():
        row = int(np.flatnonzero(item["joint_mask"].any(axis=1))[0])
        columns = np.flatnonzero(item["joint_mask"][row])
        if len(columns) < 5 or not np.all(np.diff(columns) == 1):
            continue
        x = moneyness[columns]
        values = 100.0 * item["realised"][row, columns]
        trough = int(np.argmin(values))
        if x[0] > 0.990 + tolerance or x[-1] < 1.010 - tolerance:
            continue
        if abs(x[trough] - 1.0) > 0.006 + tolerance:
            continue
        if trough < 2 or len(columns) - trough - 1 < 2:
            continue
        if np.any(np.diff(values[: trough + 1]) > tolerance):
            continue
        if np.any(np.diff(values[trough:]) < -tolerance):
            continue
        left_depth = float(values[0] - values[trough])
        right_depth = float(values[-1] - values[trough])
        if min(left_depth, right_depth) <= 0:
            continue
        candidates.append({
            "fold": item["fold"], "pair_id": item["pair_id"],
            "target_time_utc": item["target_time_utc"].isoformat(),
            "maturity_business_days": float(maturities[row]),
            "supported_smile_nodes": int(len(columns)),
            "moneyness_min": float(x[0]), "moneyness_max": float(x[-1]),
            "trough_moneyness": float(x[trough]),
            "left_wing_minus_trough_percentage_points": left_depth,
            "right_wing_minus_trough_percentage_points": right_depth,
            "score_percentage_points": min(left_depth, right_depth),
        })
    if not candidates:
        raise ValueError("No supported shortest-maturity row satisfies the smile-shape criteria.")
    ranked = pd.DataFrame(candidates).sort_values(
        ["score_percentage_points", "fold", "pair_id"],
        ascending=[False, True, True], kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    chosen = pairs.loc[pairs["pair_id"].eq(ranked.iloc[0]["pair_id"])].iloc[0].copy()
    selection = {
        "method": "maximum_bilateral_realised_smile_depth",
        "surface_eligible_count": int(len(eligible)),
        "smile_eligible_count": int(len(ranked)),
        "maturity_rule": "shortest_jointly_supported_grid_row",
        "minimum_contiguous_nodes": 5,
        "required_moneyness_coverage": [0.990, 1.010],
        "maximum_trough_distance_from_atm": 0.006,
        "minimum_nodes_on_each_side_of_trough": 2,
        "branch_rule": "realised_left_nonincreasing_right_nondecreasing",
        "numerical_tolerance": tolerance,
        "score": "min(left_endpoint_minus_trough, right_endpoint_minus_trough)",
        "score_units": "annualised_IV_percentage_points",
        "ordering": ["descending_score", "fold", "pair_id"],
        "selected_pair_id": str(chosen["pair_id"]),
        "selected_score_percentage_points": float(ranked.iloc[0]["score_percentage_points"]),
        "candidate_table": SELECTION_FILENAME,
    }
    return chosen, ranked, selection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_root: Path, stem: str, title: str) -> None:
    fig.savefig(
        output_root / f"{stem}.pdf",
        format="pdf",
        metadata={"Title": title, "CreationDate": None, "ModDate": None},
        facecolor="white",
    )
    fig.savefig(output_root / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def _supported_quads(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(
        mask[:-1, :-1]
        & mask[:-1, 1:]
        & mask[1:, :-1]
        & mask[1:, 1:]
    )


def _mesh(
    values_percent: np.ndarray,
    quad_indices: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
) -> tuple[list[list[list[float]]], list[float]]:
    """Split each supported quad into triangles without spanning mask holes."""

    faces: list[list[list[float]]] = []
    colours: list[float] = []
    for row, col in quad_indices:
        vertices = [
            [moneyness[col], maturities[row], values_percent[row, col]],
            [moneyness[col + 1], maturities[row], values_percent[row, col + 1]],
            [moneyness[col + 1], maturities[row + 1], values_percent[row + 1, col + 1]],
            [moneyness[col], maturities[row + 1], values_percent[row + 1, col]],
        ]
        for indices in ((0, 1, 2), (0, 2, 3)):
            triangle = [[float(x) for x in vertices[index]] for index in indices]
            faces.append(triangle)
            colours.append(float(np.mean([vertex[2] for vertex in triangle])))
    return faces, colours


def _surface_figure(
    selected: pd.Series,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    output_root: Path,
) -> dict:
    mask = selected["joint_mask"]
    realised = 100.0 * selected["realised"]
    predicted = 100.0 * selected["predicted"]
    supported_columns = np.flatnonzero(mask.any(axis=0))
    supported_rows = np.flatnonzero(mask.any(axis=1))
    display_mask = np.zeros_like(mask)
    display_mask[
        supported_rows[0] : supported_rows[-1] + 1,
        supported_columns[0] : supported_columns[-1] + 1,
    ] = True
    if np.any(display_mask & ~selected["target_mask"]):
        raise ValueError("Display rectangle extends beyond the realised surface's interpolation support.")
    for values in (realised, predicted):
        if not np.isfinite(values[display_mask]).all() or np.any(values[display_mask] <= 0):
            raise ValueError("Display rectangle contains invalid frozen IV values.")
    supported_values = np.concatenate((realised[display_mask], predicted[display_mask]))
    low, high = float(supported_values.min()), float(supported_values.max())
    if high <= low:
        raise ValueError("The selected surface has no displayed IV variation.")
    norm = Normalize(vmin=low, vmax=high)
    cmap = matplotlib.colormaps["plasma"]
    original_quad_indices = _supported_quads(mask)
    quad_indices = _supported_quads(display_mask)

    smile_row = int(np.flatnonzero(mask.any(axis=1))[0])
    smile_mask = mask[smile_row]
    smile_maturity = float(maturities[smile_row])
    smile_moneyness = moneyness[smile_mask]
    x_limits = [float(moneyness[supported_columns[0]] - 0.002),
                float(moneyness[supported_columns[-1]] + 0.002)]
    y_limits = [float(maturities[supported_rows[0]] - 1),
                float(maturities[supported_rows[-1]] + 1)]

    fig = plt.figure(figsize=(8.0, 7.4))
    fig.suptitle("Five-minute-ahead implied volatility", y=0.983, fontsize=15)
    target = selected["target_time_utc"]
    fig.text(
        0.5,
        0.935,
        "Forecast target: " + target.strftime("%d %B %Y, %H:%M UTC"),
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    fig.subplots_adjust(left=0.085, right=0.85, bottom=0.48, top=0.845, wspace=0.35)

    for axis, values, title, line_colour, line_style, marker in zip(
        axes, (realised, predicted), ("(a) Realised IVS", "(b) Predicted IVS"),
        (BLUE, RED), ("-", "--"), ("o", "x"),
    ):
        faces, colour_values = _mesh(values, quad_indices, moneyness, maturities)
        collection = Poly3DCollection(
            faces,
            facecolors=cmap(norm(colour_values)),
            edgecolors="#505050",
            linewidths=0.22,
            alpha=1.0,
            zsort="average",
        )
        axis.add_collection3d(collection)
        front_curve, = axis.plot(
            moneyness,
            np.full_like(moneyness, smile_maturity),
            np.where(smile_mask, values[smile_row], np.nan),
            color=line_colour,
            linestyle=line_style,
            linewidth=1.8,
            marker=marker,
            markersize=3.5,
            zorder=10,
        )
        front_curve.set_path_effects([
            patheffects.Stroke(linewidth=3.2, foreground="white"),
            patheffects.Normal(),
        ])
        axis.set_title(title, pad=10)
        axis.set_xlabel("$K/F$", labelpad=8)
        axis.yaxis.set_rotate_label(True)
        axis.set_ylabel("Maturity", labelpad=8)
        axis.set_zlabel("Annualised IV (%)", labelpad=7)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        padding = (high - low) * 0.06
        axis.set_zlim(low - padding, high + padding)
        x_ticks = [x_limits[0], 1.0, x_limits[1]]
        axis.set_xticks(x_ticks, [f"{value:.3f}" for value in x_ticks])
        axis.set_yticks([10, 15, 21])
        axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(pad=1, labelsize=11.5)
        axis.view_init(elev=32, azim=-105)
        axis.set_box_aspect((1.15, 0.65, 0.80))
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.pane.set_facecolor((1, 1, 1, 1))
            coordinate_axis.pane.set_edgecolor(GRID)
            coordinate_axis._axinfo["grid"]["color"] = GRID
            coordinate_axis._axinfo["grid"]["linewidth"] = 0.5

    colour_axis = fig.add_axes((0.90, 0.535, 0.014, 0.245))
    colourbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis
    )
    colourbar.ax.set_title("IV (%)", fontsize=12, pad=9)
    colourbar.ax.tick_params(labelsize=11.5)
    colourbar.outline.set_linewidth(0.5)
    colourbar.solids.set_rasterized(False)

    smile_axis = fig.add_axes((0.11, 0.10, 0.82, 0.265))
    _clean_axes(smile_axis)
    smile_axis.set_title(
        f"(c) Shortest-maturity smile ({smile_maturity:g} business days)",
        loc="left",
        pad=12,
    )
    smile_axis.plot(
        moneyness,
        np.where(smile_mask, realised[smile_row], np.nan),
        color=BLUE,
        linewidth=1.4,
        marker="o",
        markersize=5,
        markeredgewidth=0,
        label="Realised",
    )
    smile_axis.plot(
        moneyness,
        np.where(smile_mask, predicted[smile_row], np.nan),
        color=RED,
        linewidth=1.4,
        linestyle=(0, (4, 2)),
        marker="x",
        markersize=6,
        markeredgewidth=1,
        label="Predicted",
    )
    smile_axis.set_xlabel("Moneyness, $K/F$")
    smile_axis.set_xticks(smile_moneyness, [f"{node:.3f}" for node in smile_moneyness])
    smile_axis.set_xlim(float(smile_moneyness[0] - 0.002), float(smile_moneyness[-1] + 0.002))
    smile_axis.margins(y=0.15)
    smile_axis.legend(loc="lower right", frameon=False, fontsize=11)
    _save(fig, output_root, SURFACE_STEM, "Realised and predicted IV surfaces and shortest-maturity smile")

    in_mesh = np.zeros_like(mask)
    for row, col in quad_indices:
        in_mesh[row : row + 2, col : col + 2] = True
    in_original_mesh = np.zeros_like(mask)
    for row, col in original_quad_indices:
        in_original_mesh[row : row + 2, col : col + 2] = True
    records = []
    for row, maturity in enumerate(maturities):
        for col, relative_strike in enumerate(moneyness):
            records.append(
                {
                    "fold": selected["fold"],
                    "pair_id": selected["pair_id"],
                    "target_time_utc": target.isoformat(),
                    "maturity_business_days": float(maturity),
                    "moneyness": float(relative_strike),
                    "joint_support": bool(mask[row, col]),
                    "target_support": bool(selected["target_mask"][row, col]),
                    "display_support": bool(display_mask[row, col]),
                    "display_completion": bool(display_mask[row, col] and not mask[row, col]),
                    "included_in_mesh": bool(in_mesh[row, col]),
                    "included_in_original_joint_mesh": bool(in_original_mesh[row, col]),
                    "realised_iv_decimal": float(selected["realised"][row, col]),
                    "predicted_iv_decimal": float(selected["predicted"][row, col]),
                    "realised_iv_percent": float(realised[row, col]),
                    "predicted_iv_percent": float(predicted[row, col]),
                }
            )
    pd.DataFrame(records).to_csv(output_root / f"{SURFACE_STEM}_data.csv", index=False)
    smile_data = pd.DataFrame(records)
    smile_data = smile_data.loc[
        smile_data["maturity_business_days"].eq(smile_maturity) & smile_data["joint_support"]
    ]
    smile_data.to_csv(output_root / f"{SMILE_STEM}_data.csv", index=False)
    return {
        "fold": str(selected["fold"]),
        "pair_id": str(selected["pair_id"]),
        "origin_time_utc": selected["origin_time_utc"].isoformat(),
        "target_time_utc": target.isoformat(),
        "session_id": str(selected["session_id"]),
        "supported_cell_count": int(mask.sum()),
        "supported_quad_count": len(original_quad_indices),
        "rendered_quad_count": len(quad_indices),
        "rendered_triangle_count": 2 * len(quad_indices),
        "display_node_count": int(display_mask.sum()),
        "display_completion_node_count": int((display_mask & ~mask).sum()),
        "display_completion": {
            "method": "reuse_frozen_grids_in_realised_supported_rectangle",
            "realised_grid_construction": "original_total_variance_interpolation",
            "predicted_grid": "unchanged_frozen_64_draw_mean",
            "target_support_verified": True,
            "evaluation_support": "original_joint_mask",
        },
        "colour_limits_iv_percent": [low, high],
        "colormap": "plasma",
        "axis_limits": {"moneyness": x_limits, "maturity_business_days": y_limits},
        "view": {
            "elevation_degrees": 32, "azimuth_degrees": -105,
            "box_aspect_moneyness_maturity_iv": [1.15, 0.65, 0.80],
        },
        "shortest_maturity_smile": {
            "maturity_business_days": smile_maturity,
            "grid_row_index": smile_row,
            "supported_node_count": int(smile_mask.sum()),
            "moneyness_nodes": smile_moneyness.tolist(),
            "connection_policy": "adjacent_joint_supported_grid_nodes",
            "highlighted_on_3d_surface": True,
        },
        "figure_size_inches": [8.0, 7.4],
    }


def _clean_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color("#A5ABB1")
        axis.spines[name].set_linewidth(0.7)
    axis.grid(axis="y", color=GRID, linewidth=0.65)
    axis.set_axisbelow(True)
    axis.set_ylabel("Annualised IV (%)")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _plot_atm_series(axis: plt.Axes, frame: pd.DataFrame, marker_size: float) -> None:
    valid = frame.loc[frame["atm_valid"]]
    axis.plot(
        valid["target_time_utc"],
        100.0 * valid["atm_realised_iv"],
        color=BLUE,
        marker="o",
        markersize=float(np.sqrt(marker_size)),
        markeredgewidth=0,
        linewidth=1.1,
        label="Realised",
        zorder=3,
    )
    axis.plot(
        valid["target_time_utc"],
        100.0 * valid["atm_predicted_iv"],
        color=RED,
        marker="x",
        markersize=float(np.sqrt(1.45 * marker_size)),
        markeredgewidth=1.0,
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label="Predicted",
        zorder=4,
    )


def _atm_figure(pairs: pd.DataFrame, detail_date: str, output_root: Path) -> dict:
    ordered = pairs.sort_values(["target_time_utc", "pair_id"]).reset_index(drop=True)
    valid = ordered.loc[ordered["atm_valid"]]
    date = pd.Timestamp(detail_date, tz="UTC")
    detail = ordered.loc[ordered["target_time_utc"].dt.normalize().eq(date)].copy()
    detail = detail.reset_index(drop=True)
    if detail.empty:
        raise ValueError(f"No frozen test observations on {detail_date} UTC.")
    detail_valid = detail.loc[detail["atm_valid"]]
    links = list(zip(detail_valid.index[:-1], detail_valid.index[1:]))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.095, hspace=0.55)
    fig.suptitle("Constant-maturity ATM implied volatility", y=0.975, fontsize=15)
    for axis in axes:
        _clean_axes(axis)
    _plot_atm_series(axes[0], valid, marker_size=13)
    axes[0].set_title("(a) 2023 test observations", loc="left", pad=9)
    axes[0].set_xlabel("Forecast target date (UTC)")
    start, end = valid["target_time_utc"].min(), valid["target_time_utc"].max()
    year_ticks = [start.normalize()]
    for boundary in pd.date_range("2023-04-01", "2023-10-01", freq="QS", tz="UTC"):
        axes[0].axvline(boundary, color="#BDC3C8", linewidth=0.7, linestyle=(0, (3, 4)))
        if start < boundary < end:
            year_ticks.append(boundary)
    year_ticks.append(end.normalize())
    axes[0].set_xticks(year_ticks)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=timezone.utc))
    axes[0].set_xlim(start - pd.Timedelta(days=4), end + pd.Timedelta(days=4))
    axes[0].margins(y=0.08)

    _plot_atm_series(axes[1], detail, marker_size=25)
    axes[1].set_title("(b) " + date.strftime("%d %B %Y"), loc="left", pad=9)
    axes[1].set_xlabel("Forecast target time (UTC)")
    first, last = detail["target_time_utc"].min(), detail["target_time_utc"].max()
    hour_ticks = [first]
    hour_ticks.extend(pd.date_range(first.ceil("h"), last.floor("h"), freq="h"))
    hour_ticks.append(last)
    axes[1].set_xticks(sorted(set(hour_ticks)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axes[1].set_xlim(first - pd.Timedelta(minutes=7), last + pd.Timedelta(minutes=7))
    axes[1].margins(y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.932), ncol=2, frameon=False)
    _save(fig, output_root, ATM_STEM, "Realised and predicted constant-maturity ATM IV")

    detail_predecessors = {str(detail.iloc[right]["pair_id"]) for _, right in links}
    exported = ordered[
        ["fold", "pair_id", "session_id", "origin_time_utc", "target_time_utc", "atm_valid"]
    ].copy()
    exported["maturity_business_days"] = 21
    exported["atm_moneyness"] = 1.0
    exported["realised_atm_iv_decimal"] = ordered["atm_realised_iv"]
    exported["predicted_atm_iv_decimal"] = ordered["atm_predicted_iv"]
    exported["realised_atm_iv_percent"] = 100.0 * ordered["atm_realised_iv"]
    exported["predicted_atm_iv_percent"] = 100.0 * ordered["atm_predicted_iv"]
    exported["is_detail_date"] = ordered["target_time_utc"].dt.normalize().eq(date)
    exported["detail_link_from_previous"] = ordered["pair_id"].astype(str).isin(detail_predecessors)
    exported.to_csv(output_root / f"{ATM_STEM}_data.csv", index=False, na_rep="")
    return {
        "maturity_business_days": 21,
        "moneyness": 1.0,
        "interpolation": "linear_in_variance_between_adjacent_moneyness_nodes",
        "moneyness_bracket": [0.998, 1.002],
        "variance_weights": [0.5, 0.5],
        "valid_pair_count": int(len(valid)),
        "valid_pairs_by_fold": {str(k): int(v) for k, v in valid.groupby("fold").size().items()},
        "valid_utc_date_count": int(valid["target_time_utc"].dt.normalize().nunique()),
        "valid_cme_session_count": int(valid["session_id"].nunique()),
        "detail_date_utc": detail_date,
        "detail_pair_count": int(len(detail)),
        "detail_valid_pair_count": int(len(detail_valid)),
        "detail_missing_pair_ids": detail.loc[~detail["atm_valid"], "pair_id"].astype(str).tolist(),
        "line_connection_policy": "consecutive_valid_observations_on_actual_time_axis",
        "full_year_segment_count": max(0, len(valid) - 1),
        "detail_segment_count": len(links),
        "detail_segments": [
            {
                "from_pair_id": str(detail.iloc[left]["pair_id"]),
                "to_pair_id": str(detail.iloc[right]["pair_id"]),
                "from_target_time_utc": detail.iloc[left]["target_time_utc"].isoformat(),
                "to_target_time_utc": detail.iloc[right]["target_time_utc"].isoformat(),
            }
            for left, right in links
        ],
        "figure_size_inches": [7.5, 6.5],
    }


def _write_captions(output_root: Path, selected: pd.Series, seed: int) -> None:
    target = selected["target_time_utc"].strftime("%d %B %Y at %H:%M UTC")
    surface_caption = (
        f"Realised and predicted implied-volatility surfaces for the five-minute "
        f"forecast target on {target}. The transition is selected for a "
        "pronounced realised volatility smile at the shortest jointly "
        "supported maturity. "
        f"Predictions are the 64-draw conditional mean of the matched-LP FiLM-CNN "
        f"(seed {seed}). Panels (a) and (b) display the complete frozen grids "
        r"over $K/F\in[0.986,1.014]$ and 6--21 business days, with common axis "
        "and colour scales. The realised grid follows the original total-"
        "variance interpolation, and forecast evaluation retains the joint-"
        "support mask. The coloured front-edge "
        "curves and Panel (c) show the volatility smile at the shortest jointly "
        "supported grid maturity "
        f"({selected['shortest_smile_maturity']:g} business days). "
        "Realised IV is shown by a blue solid line and predicted IV by a red "
        "dashed line. Maturity is measured in business days; implied volatility is annualised "
        "and expressed as a percentage."
    )
    atm_caption = (
        "Realised and predicted constant-maturity ATM implied volatility at "
        "the sampled five-minute forecast targets. Panel (a) shows the 188 "
        "supported test observations in 2023; panel (b) enlarges 12 July 2023, "
        "the date with the most supported observations. At 21 business days, "
        "ATM IV is obtained by linear interpolation in variance between "
        r"$K/F=0.998$ and $K/F=1.002$. "
        f"Predictions are the 64-draw conditional mean of the matched-LP "
        f"FiLM-CNN (seed {seed}). Markers identify sampling times; lines join "
        "consecutive available observations on the actual time axis. "
        "Realised IV is shown by blue solid lines and predicted IV by red "
        "dashed lines. "
        "Implied volatility is annualised and expressed as "
        "a percentage, and all times are UTC."
    )
    placement = (
        "Suggested placement: in `Main Results: FiLM-CNN versus Pure-CNN Baseline`, "
        "after the forecast-accuracy table and its discussion, before the text-"
        "representation comparison. These figures illustrate the fitted "
        "forecast and the sampled ATM path for one fixed seed."
    )
    captions = (
        "# Chapter 3 forecast figures\n\n"
        + placement
        + "\n\n## Figure 1\n\n"
        + surface_caption.replace("%", r"\%")
        + "\n\n## Figure 2\n\n"
        + atm_caption.replace("%", r"\%")
        + "\n\n## Surface example selection\n\n"
        "The initial pool has 161 transitions with at least 32 supported cells, "
        "four maturity rows, four moneyness columns and 12 complete grid faces. "
        "Each transition's shortest jointly supported row must have at least "
        "five contiguous nodes spanning K/F=0.990 to 1.010, with a realised "
        "IV trough within 0.006 of ATM and at least two nodes on each side. "
        "The left branch must be nonincreasing and the right branch "
        "nondecreasing. Rank qualifying rows by the smaller of the two "
        "endpoint-minus-trough IV differences, breaking ties by (fold, pair_id). "
        "The selection uses realised IV only. The ranked candidates are "
        f"saved in `{SELECTION_FILENAME}`.\n"
    )
    (output_root / "captions.md").write_text(captions, encoding="utf-8")
    preview = (
        "% Standalone insertion preview; the existing chapter is not edited.\n"
        "% Suggested placement: after the RQ1 forecast-accuracy discussion.\n"
        "\\begin{figure}[htbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/forecast_examples/"
        + SURFACE_STEM
        + ".pdf}\n"
        "\\caption[Realised and predicted IV surfaces and shortest-maturity smile]{"
        + surface_caption.replace("%", r"\%")
        + "}\n\\label{fig:ch3:realised_predicted_surface}\n\\end{figure}\n\n"
        "\\begin{figure}[htbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/forecast_examples/"
        + ATM_STEM
        + ".pdf}\n"
        "\\caption[Realised and predicted constant-maturity ATM implied volatility]{"
        + atm_caption.replace("%", r"\%")
        + "}\n\\label{fig:ch3:atm_forecast_time_series}\n\\end{figure}\n"
    )
    (output_root / "chapter3_forecast_figures_preview.tex").write_text(preview, encoding="utf-8")


def render(source_root: Path, output_root: Path) -> dict:
    pairs, moneyness, maturities, provenance = load_frozen_inputs(source_root, seed=42)
    pairs = with_atm_values(pairs, moneyness, maturities, maturity_days=21)
    if (len(pairs), int(pairs["surface_eligible"].sum()), int(pairs["atm_valid"].sum())) != (500, 161, 188):
        raise ValueError("Frozen-panel counts differ from the agreed 500/161/188 figure contract.")
    selected, smile_candidates, surface_selection = _select_smile_example(pairs, moneyness, maturities)
    selected["shortest_smile_maturity"] = float(
        maturities[np.flatnonzero(selected["joint_mask"].any(axis=1))[0]]
    )
    detail_date = "2023-07-12"
    date_counts = pairs.loc[pairs["atm_valid"]].groupby(pairs["target_time_utc"].dt.normalize()).size()
    if date_counts.idxmax() != pd.Timestamp(detail_date, tz="UTC") or int(date_counts.max()) != 12:
        raise ValueError("The agreed detail date is not the date with most supported ATM observations.")

    output_root.mkdir(parents=True, exist_ok=True)
    smile_candidates.to_csv(output_root / SELECTION_FILENAME, index=False)
    _style()
    surface_info = _surface_figure(selected, moneyness, maturities, output_root)
    atm_info = _atm_figure(pairs, detail_date, output_root)
    _write_captions(output_root, selected, seed=42)
    outputs = {}
    for name in (
        f"{SURFACE_STEM}.pdf", f"{SURFACE_STEM}.png", f"{SURFACE_STEM}_data.csv",
        f"{SMILE_STEM}_data.csv",
        SELECTION_FILENAME,
        f"{ATM_STEM}.pdf", f"{ATM_STEM}.png", f"{ATM_STEM}_data.csv",
        "captions.md", "chapter3_forecast_figures_preview.tex",
    ):
        path = output_root / name
        outputs[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "kind": "chapter3_frozen_forecast_examples_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_provenance": provenance,
        "prediction": {"arm": "lp_matched", "model_seed": 42, "mc_draw_mean": 64},
        "surface_selection": surface_selection,
        "surface": surface_info,
        "atm": atm_info,
        "rendering": {
            "iv_scale": "annualised_percent",
            "source_iv_scale": "annualised_decimal_ACT_365",
            "decimal_to_percent_multiplier": 100,
            "timezone": "UTC",
            "png_dpi": 600,
            "pdf_fonttype": 42,
            "font_family": "DejaVu Sans",
            "series": {
                "realised": {"colour": BLUE, "marker": "o", "linestyle": "solid"},
                "predicted": {"colour": RED, "marker": "x", "linestyle": "dashed"},
            },
            "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "plotting_script_sha256": _sha256(Path(__file__)),
        "data_helper_sha256": _sha256(Path(__file__).with_name("chapter3_forecast_figure_data.py")),
        "outputs": outputs,
    }
    (output_root / "figure_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = render(args.source_root.expanduser().resolve(), args.output_root.expanduser().resolve())
    print(json.dumps({"output_root": manifest["output_root"], "surface": manifest["surface"], "atm": manifest["atm"]}, indent=2))


if __name__ == "__main__":
    main()
