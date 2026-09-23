"""Plot the frozen 60-minute rolling-skewness market-jump screen.

This is a presentation-only renderer.  It reads the rolling Fisher--Pearson
skewness, adjacent-valid-estimate changes, and anomaly tiers already frozen by
``market_jump_detection.py``; it does not recompute any statistic or label.
The illustrative time series is restricted to exactly one CME session, option
maturity, and underlying futures contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#263238"
MUTED = "#66727A"
GRID = "#D9E0E4"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#DCEAF4"
ORANGE = "#D9822B"
GOLD = "#C5A03D"
WHITE = "#FFFFFF"

DEFAULT_SOURCE = Path(
    "outputs/rq3/atm_skew_jumps_20260810_final/pair_slice_metrics.csv.gz"
)
DEFAULT_RANKINGS = Path(
    "outputs/rq3/atm_skew_jumps_20260810_final/metric_rankings.csv.gz"
)
DEFAULT_OUTPUT_STEM = Path("docs/figures/ch3_atm_jump_rolling_skewness_screen_v1")

# This group contains the largest number of frozen broad rolling-skew flags
# (14) while retaining 185 valid 60-minute estimates.  The choice is disclosed
# in the figure and manifest; it is illustrative, not an inferential subset.
DEFAULT_SESSION_ID = "cme_ty_20230712T2200Z"
DEFAULT_MATURITY_DATE = "2023-07-21"
DEFAULT_UNDERLYING_CONTRACT_ID = "TYU3"

GROUP_COLUMNS = ("session_id", "maturity_date", "underlying_contract_id")
SKEW_COLUMN = "rolling_jump_skew_60m"
CHANGE_COLUMN = "delta_rolling_jump_skew_60m"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources(source: Path, rankings: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "slice_pair_id",
        "origin_time_utc",
        *GROUP_COLUMNS,
        SKEW_COLUMN,
        CHANGE_COLUMN,
    ]
    rank_columns = [
        "slice_pair_id",
        "metric_name",
        "anomaly_tier",
        "robust_z",
        "abs_empirical_percentile",
    ]
    metrics = pd.read_csv(source, usecols=metric_columns, low_memory=False)
    rank = pd.read_csv(rankings, usecols=rank_columns, low_memory=False)
    if metrics["slice_pair_id"].duplicated().any():
        raise ValueError("pair_slice_metrics contains duplicate slice_pair_id values")
    rolling_rank = rank.loc[
        rank["metric_name"].astype(str).eq("rolling_jump_skew_change")
    ].copy()
    if rolling_rank["slice_pair_id"].duplicated().any():
        raise ValueError("rolling-skew rankings contain duplicate slice_pair_id values")
    unexpected = set(
        rolling_rank.loc[
            rolling_rank["anomaly_tier"].fillna("none").astype(str).ne("none"),
            "anomaly_tier",
        ].astype(str)
    ) - {"broad"}
    if unexpected:
        raise ValueError(
            "Frozen rolling-skew rankings must be capped at broad; found "
            f"{sorted(unexpected)}"
        )
    metrics["origin_time_utc"] = pd.to_datetime(
        metrics["origin_time_utc"], utc=True, errors="raise"
    )
    return metrics, rolling_rank


def _select_group(
    metrics: pd.DataFrame,
    rolling_rank: pd.DataFrame,
    *,
    session_id: str,
    maturity_date: str,
    underlying_contract_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[
        metrics["session_id"].astype(str).eq(session_id)
        & metrics["maturity_date"].astype(str).eq(maturity_date)
        & metrics["underlying_contract_id"].astype(str).eq(underlying_contract_id)
    ].copy()
    selected = selected.sort_values("origin_time_utc", kind="stable").reset_index(
        drop=True
    )
    if selected.empty:
        raise ValueError(
            "No pair-slice rows match the requested session/maturity/contract"
        )
    if selected["origin_time_utc"].duplicated().any():
        raise ValueError("Selected group contains duplicate origin timestamps")
    if selected[SKEW_COLUMN].notna().sum() == 0:
        raise ValueError("Selected group has no valid 60-minute skewness estimates")

    flagged_rank = rolling_rank.loc[
        rolling_rank["anomaly_tier"].fillna("none").astype(str).eq("broad")
    ].copy()
    flagged = selected.merge(
        flagged_rank,
        on="slice_pair_id",
        how="inner",
        validate="one_to_one",
    )
    flagged = flagged.loc[
        flagged[SKEW_COLUMN].notna() & flagged[CHANGE_COLUMN].notna()
    ].copy()
    return selected, flagged


def _configure_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.set_facecolor(WHITE)
    axis.tick_params(colors=INK, labelsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)


def _plot_valid_segments(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    column: str,
    color: str,
    label: str,
) -> None:
    valid = frame.loc[frame[column].notna(), ["origin_time_utc", column]].copy()
    gaps = valid["origin_time_utc"].diff().dt.total_seconds().div(60.0)
    valid["_run"] = (gaps.isna() | gaps.gt(5.0)).cumsum()
    first = True
    for _, run in valid.groupby("_run", sort=False):
        axis.plot(
            run["origin_time_utc"],
            run[column],
            color=color,
            linewidth=1.55,
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
            alpha=0.94,
            label=label if first else None,
        )
        first = False


def _time_ticks(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    interior = [
        tick
        for tick in pd.date_range(start.ceil("30min"), end.floor("30min"), freq="30min")
        if tick - start >= pd.Timedelta(minutes=15)
        and end - tick >= pd.Timedelta(minutes=15)
    ]
    ticks = [start, *interior, end]
    return list(dict.fromkeys(ticks))


def render_figure(
    metrics: pd.DataFrame,
    rolling_rank: pd.DataFrame,
    selected: pd.DataFrame,
    flagged: pd.DataFrame,
    *,
    output_stem: Path,
    session_id: str,
    maturity_date: str,
    underlying_contract_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    global_skew = pd.to_numeric(metrics[SKEW_COLUMN], errors="coerce").dropna()
    global_change = pd.to_numeric(metrics[CHANGE_COLUMN], errors="coerce").dropna()
    global_flags = rolling_rank.loc[
        rolling_rank["anomaly_tier"].fillna("none").astype(str).eq("broad")
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "svg.hashsalt": "ch3_atm_jump_rolling_skewness_screen_v1",
        }
    )
    fig = plt.figure(figsize=(13.5, 7.1), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.25, 1.0),
        height_ratios=(1.55, 1.0),
        left=0.075,
        right=0.985,
        bottom=0.145,
        top=0.80,
        hspace=0.13,
        wspace=0.26,
    )
    skew_axis = fig.add_subplot(grid[0, 0])
    change_axis = fig.add_subplot(grid[1, 0], sharex=skew_axis)
    distribution_axis = fig.add_subplot(grid[:, 1])

    fig.suptitle(
        "60-minute rolling skewness screen for five-minute ATM-IV jumps",
        x=0.075,
        y=0.965,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.917,
        (
            "Frozen RQ3 archive | bias-corrected Fisher–Pearson skewness of "
            "signed 5-min ATM-IV changes | 60-min clock window, n≥30"
        ),
        ha="left",
        va="top",
        fontsize=9.6,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.878,
        (
            f"Illustrative group with most broad flags: {underlying_contract_id}, "
            f"maturity {maturity_date}, {session_id}"
        ),
        ha="left",
        va="top",
        fontsize=9.0,
        color=MUTED,
    )

    _plot_valid_segments(
        skew_axis,
        selected,
        column=SKEW_COLUMN,
        color=BLUE,
        label="60-min rolling skewness",
    )
    skew_axis.axhline(0.0, color=INK, linewidth=0.95, linestyle=":")
    if not flagged.empty:
        skew_axis.scatter(
            flagged["origin_time_utc"],
            flagged[SKEW_COLUMN],
            marker="^",
            s=48,
            facecolor=WHITE,
            edgecolor=GOLD,
            linewidth=1.35,
            zorder=5,
            label=f"Broad flag (n={len(flagged)})",
        )
    skew_axis.set_ylabel("Rolling Fisher–Pearson skewness")
    skew_axis.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.6)
    skew_axis.tick_params(axis="x", labelbottom=False)
    _configure_axis(skew_axis)

    _plot_valid_segments(
        change_axis,
        selected,
        column=CHANGE_COLUMN,
        color=ORANGE,
        label="Current valid estimate − previous valid estimate",
    )
    change_axis.axhline(0.0, color=INK, linewidth=0.95, linestyle=":")
    if not flagged.empty:
        change_axis.scatter(
            flagged["origin_time_utc"],
            flagged[CHANGE_COLUMN],
            marker="^",
            s=48,
            facecolor=WHITE,
            edgecolor=GOLD,
            linewidth=1.35,
            zorder=5,
        )
    change_axis.set_ylabel("Adjacent-valid change")
    change_axis.set_xlabel("Market-pair origin (UTC)")
    change_axis.legend(frameon=False, loc="upper left", fontsize=8.2)
    _configure_axis(change_axis)

    valid_times = selected.loc[selected[SKEW_COLUMN].notna(), "origin_time_utc"]
    start, end = valid_times.min(), valid_times.max()
    skew_axis.set_xlim(start, end)
    change_axis.set_xticks(_time_ticks(start, end))
    change_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))

    bins = np.linspace(float(global_skew.min()), float(global_skew.max()), 48)
    distribution_axis.hist(
        global_skew,
        bins=bins,
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.7,
    )
    distribution_axis.axvline(0.0, color=INK, linewidth=0.95, linestyle=":")
    distribution_axis.axvline(
        float(global_skew.median()),
        color=ORANGE,
        linewidth=1.2,
        linestyle="--",
        label=f"Median = {global_skew.median():.3f}",
    )
    distribution_axis.set_title(
        "Frozen 60-min skewness distribution", loc="left", fontsize=10.4
    )
    distribution_axis.set_xlabel("Rolling Fisher–Pearson skewness")
    distribution_axis.set_ylabel("Eligible slice-pair observations")
    distribution_axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    _configure_axis(distribution_axis)
    flag_rate = len(global_flags) / len(global_change) if len(global_change) else np.nan
    distribution_axis.text(
        0.98,
        0.97,
        (
            f"Skew estimates: n={len(global_skew):,}\n"
            f"Adjacent-valid changes: n={len(global_change):,}\n"
            f"Broad flags: n={len(global_flags):,} ({flag_rate:.2%})"
        ),
        transform=distribution_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": WHITE,
            "edgecolor": GRID,
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )

    fig.text(
        0.075,
        0.045,
        (
            "The plotted change is read directly from the frozen adjacent-valid "
            "screen. Lines never cross session, maturity, or contract; a gap "
            ">5 minutes starts a new segment. Rolling-skew signals are broad-only."
        ),
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Date": None},
    )
    plt.close(fig)

    metadata = {
        "global_rolling_skew_60m_n": int(len(global_skew)),
        "global_adjacent_valid_change_n": int(len(global_change)),
        "global_broad_flag_n": int(len(global_flags)),
        "global_broad_flag_rate_among_valid_changes": float(flag_rate),
        "selected_rows": int(len(selected)),
        "selected_rolling_skew_60m_n": int(selected[SKEW_COLUMN].notna().sum()),
        "selected_adjacent_valid_change_n": int(selected[CHANGE_COLUMN].notna().sum()),
        "selected_broad_flag_n": int(len(flagged)),
        "selected_time_min_utc": str(start),
        "selected_time_max_utc": str(end),
    }
    return png_path, svg_path, metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the frozen ATM-jump 60-minute rolling-skewness screen."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--maturity-date", default=DEFAULT_MATURITY_DATE)
    parser.add_argument(
        "--underlying-contract-id", default=DEFAULT_UNDERLYING_CONTRACT_ID
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.resolve()
    rankings = args.rankings.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not rankings.is_file():
        raise FileNotFoundError(rankings)
    output_stem = args.output_stem
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")

    metrics, rolling_rank = _load_sources(source, rankings)
    selected, flagged = _select_group(
        metrics,
        rolling_rank,
        session_id=args.session_id,
        maturity_date=args.maturity_date,
        underlying_contract_id=args.underlying_contract_id,
    )
    png_path, svg_path, metadata = render_figure(
        metrics,
        rolling_rank,
        selected,
        flagged,
        output_stem=output_stem,
        session_id=args.session_id,
        maturity_date=args.maturity_date,
        underlying_contract_id=args.underlying_contract_id,
    )
    manifest: dict[str, Any] = {
        "schema_version": "atm_jump_rolling_skewness_screen_figure_v1",
        "calculation_policy": (
            "Read frozen rolling_jump_skew_60m and "
            "delta_rolling_jump_skew_60m; do not recompute."
        ),
        "grouping_policy": (
            "One session_id x maturity_date x underlying_contract_id; break "
            "display lines when consecutive valid timestamps differ by >5 minutes."
        ),
        "selection_policy": (
            "Frozen group with the largest number of broad "
            "rolling_jump_skew_change flags; illustrative only."
        ),
        "source": str(source),
        "source_sha256": _sha256(source),
        "rankings": str(rankings),
        "rankings_sha256": _sha256(rankings),
        "selection": {
            "session_id": args.session_id,
            "maturity_date": args.maturity_date,
            "underlying_contract_id": args.underlying_contract_id,
        },
        **metadata,
        "outputs": {
            "png": str(png_path),
            "png_sha256": _sha256(png_path),
            "svg": str(svg_path),
            "svg_sha256": _sha256(svg_path),
        },
    }
    manifest_path = output_stem.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
