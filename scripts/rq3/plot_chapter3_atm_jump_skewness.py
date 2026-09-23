"""Render the Chapter 3 rolling ATM-IV-jump skewness figure.

The chart is deliberately regenerated from the frozen market-first outputs.  It
does not join observations belonging to different maturity, underlying, or CME
session sequences, and it does not invent a global screening threshold: the
formal thresholds vary by year and business-day maturity bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "outputs/rq3/atm_skew_jumps_20260810_final"
DEFAULT_OUTPUT_PREFIX = ROOT / "docs/figures/ch3_atm_jump_rolling_skewness"

INK = "#26343D"
MUTED = "#66737D"
GRID = "#DDE4E8"
BLUE = "#2F75B5"
BLUE_LIGHT = "#AFCBE1"
GOLD = "#C99A2E"
GROUP_COLUMNS = ["maturity_date", "underlying_contract_id", "session_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative manifest path while allowing temp outputs."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _finite_rows(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    result = frame.loc[finite].copy()
    result[column] = values.loc[finite].to_numpy(dtype=float)
    return result


def _time_ticks(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    year_start = pd.Timestamp(year=start.year, month=1, day=1, tz="UTC")
    calendar_ticks = pd.date_range(year_start, end, freq="4MS")
    interior = [tick for tick in calendar_ticks if start < tick < end]
    return [start, *interior, end]


def _set_symmetric_y_limits(axis: plt.Axes, values: pd.Series) -> None:
    limit = float(np.abs(values.to_numpy(dtype=float)).max())
    padding = max(0.05 * limit, 0.05)
    axis.set_ylim(-(limit + padding), limit + padding)


def _load(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    metrics_path = source_root / "pair_slice_metrics.csv.gz"
    rankings_path = source_root / "metric_rankings.csv.gz"
    metrics = pd.read_csv(metrics_path)
    rankings = pd.read_csv(rankings_path)

    required_metrics = {
        "slice_pair_id",
        "pair_id",
        "origin_time_utc",
        "maturity_date",
        "underlying_contract_id",
        "session_id",
        "rolling_jump_skew_60m",
        "delta_rolling_jump_skew_60m",
    }
    required_rankings = {
        "slice_pair_id",
        "pair_id",
        "origin_time_utc",
        "metric_name",
        "metric_change",
        "anomaly_tier",
    }
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    missing_rankings = sorted(required_rankings - set(rankings.columns))
    if missing_metrics or missing_rankings:
        raise ValueError(
            "Frozen inputs do not satisfy the plotting contract: "
            f"metrics={missing_metrics}, rankings={missing_rankings}"
        )
    return metrics, rankings, metrics_path, rankings_path


def render(source_root: Path, output_prefix: Path) -> dict[str, object]:
    metrics, rankings, metrics_path, rankings_path = _load(source_root)

    levels = metrics[
        [
            "pair_id",
            "origin_time_utc",
            "maturity_date",
            "underlying_contract_id",
            "session_id",
            "rolling_jump_skew_60m",
        ]
    ].copy()
    levels["origin_time_utc"] = pd.to_datetime(
        levels["origin_time_utc"], utc=True, errors="raise"
    )
    levels = _finite_rows(levels, "rolling_jump_skew_60m")

    changes = metrics[
        [
            "slice_pair_id",
            "pair_id",
            "origin_time_utc",
            "delta_rolling_jump_skew_60m",
        ]
    ].copy()
    changes["origin_time_utc"] = pd.to_datetime(
        changes["origin_time_utc"], utc=True, errors="raise"
    )
    changes = _finite_rows(changes, "delta_rolling_jump_skew_60m")

    rolling_rankings = rankings.loc[
        rankings["metric_name"].astype(str).eq("rolling_jump_skew_change"),
        [
            "slice_pair_id",
            "pair_id",
            "origin_time_utc",
            "metric_change",
            "anomaly_tier",
        ],
    ].copy()
    rolling_rankings["origin_time_utc"] = pd.to_datetime(
        rolling_rankings["origin_time_utc"], utc=True, errors="raise"
    )
    rolling_rankings = _finite_rows(rolling_rankings, "metric_change")
    rolling_rankings["anomaly_tier"] = (
        rolling_rankings["anomaly_tier"].fillna("none").astype(str)
    )

    unexpected_tiers = set(rolling_rankings["anomaly_tier"]) - {"none", "broad"}
    if unexpected_tiers:
        raise ValueError(
            "Rolling-skew-only rankings must be capped at broad; found "
            f"{sorted(unexpected_tiers)}"
        )

    ranked_changes = rolling_rankings.merge(
        changes[
            [
                "slice_pair_id",
                "origin_time_utc",
                "delta_rolling_jump_skew_60m",
            ]
        ],
        on="slice_pair_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_ranking", "_metrics"),
    )
    timestamps_match = ranked_changes["origin_time_utc_ranking"].eq(
        ranked_changes["origin_time_utc_metrics"]
    )
    values_match = np.isclose(
        ranked_changes["metric_change"].to_numpy(dtype=float),
        ranked_changes["delta_rolling_jump_skew_60m"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    if (
        not ranked_changes["_merge"].eq("both").all()
        or not timestamps_match.all()
        or not values_match.all()
    ):
        raise ValueError(
            "Frozen rolling-skew rankings do not reconcile one-to-one with "
            "pair_slice_metrics"
        )

    flagged = rolling_rankings.loc[rolling_rankings["anomaly_tier"].eq("broad")].copy()

    group_count = int(levels[GROUP_COLUMNS].drop_duplicates().shape[0])
    tier_counts = {
        str(key): int(value)
        for key, value in flagged["anomaly_tier"].value_counts().sort_index().items()
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "svg.hashsalt": "chapter3_atm_jump_rolling_skewness_v1",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.24, top=0.84, hspace=0.22)
    fig.suptitle(
        "Rolling skewness of signed five-minute ATM-IV changes",
        x=0.085,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.085,
        0.91,
        "Bias-corrected Fisher–Pearson statistic; main window = 60 clock minutes with at least 30 observations",
        ha="left",
        fontsize=11,
        color=MUTED,
    )

    axes[0].scatter(
        levels["origin_time_utc"],
        levels["rolling_jump_skew_60m"],
        s=11,
        color=BLUE,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    axes[0].axhline(0.0, color=INK, linewidth=0.9, linestyle=(0, (2, 2)))
    axes[0].set_ylabel("Rolling skewness")
    axes[0].set_title(
        (
            f"A. Valid 60-minute estimates (n={len(levels):,}; "
            f"{group_count} session–maturity–contract groups)"
        ),
        loc="left",
        fontsize=11,
        pad=8,
    )
    _set_symmetric_y_limits(axes[0], levels["rolling_jump_skew_60m"])

    axes[1].scatter(
        changes["origin_time_utc"],
        changes["delta_rolling_jump_skew_60m"],
        s=10,
        color=BLUE_LIGHT,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        label=f"All consecutive valid changes (n={len(changes):,})",
    )
    axes[1].scatter(
        flagged["origin_time_utc"],
        flagged["metric_change"],
        s=34,
        marker="^",
        facecolors="white",
        edgecolors=GOLD,
        linewidths=1.1,
        alpha=0.96,
        label=(
            f"Broad-tier flags (n={len(flagged):,}; "
            f"{flagged['pair_id'].nunique():,} unique pairs)"
        ),
    )
    axes[1].axhline(0.0, color=INK, linewidth=0.9, linestyle=(0, (2, 2)))
    axes[1].set_ylabel(r"Change in rolling skewness, $\Delta\widehat{\gamma}_{60}$")
    axes[1].set_xlabel("Market-pair origin (UTC)")
    axes[1].set_title(
        "B. Consecutive changes and observations retained by the anomaly screen",
        loc="left",
        fontsize=11,
        pad=8,
    )
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    _set_symmetric_y_limits(axes[1], changes["delta_rolling_jump_skew_60m"])

    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID)
        axis.spines["bottom"].set_color(GRID)
    time_start = min(levels["origin_time_utc"].min(), changes["origin_time_utc"].min())
    time_end = max(levels["origin_time_utc"].max(), changes["origin_time_utc"].max())
    axes[1].set_xlim(time_start - pd.Timedelta(days=8), time_end + pd.Timedelta(days=8))
    axes[1].set_xticks(_time_ticks(time_start, time_end))
    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b\n%Y", tz=timezone.utc)
    )

    fig.text(
        0.085,
        0.035,
        (
            "All observations are shown as points rather than joined lines; the "
            f"{group_count} maturity × underlying-contract × CME-session groups "
            "are kept separate.\n"
            "Gaps longer than five minutes reset the rolling calculation. Formal "
            "screening is stratified by year and business-day maturity bucket;\n"
            "therefore no single global cutoff is shown."
        ),
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
        linespacing=1.35,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    json_path = output_prefix.with_suffix(".json")
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(fig)

    payload: dict[str, object] = {
        "schema_version": "chapter3_atm_jump_rolling_skewness_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_root": _display_path(source_root),
        "inputs": {
            _display_path(metrics_path): _sha256(metrics_path),
            _display_path(rankings_path): _sha256(rankings_path),
        },
        "definition": (
            "Bias-corrected Fisher-Pearson skewness of A/B-quality signed "
            "five-minute ATM-IV changes within maturity_date x "
            "underlying_contract_id x CME session; 60 clock minutes and at "
            "least 30 observations; reset after a gap longer than five minutes."
        ),
        "screening_metric": "delta_rolling_jump_skew_60m",
        "rolling_level_rows": int(len(levels)),
        "rolling_change_rows": int(len(changes)),
        "session_maturity_contract_group_count": group_count,
        "flagged_slice_metric_rows": int(len(flagged)),
        "flagged_unique_market_pairs": int(flagged["pair_id"].nunique()),
        "flagged_tiers": tier_counts,
        "display_policy": (
            "All finite 60-minute levels and changes are displayed without "
            "clipping. Full-sample observations are plotted as points rather "
            "than joined across groups or gaps longer than five minutes."
        ),
        "research_role": (
            "Auxiliary market-first anomaly label for exploratory conditional "
            "analysis; not a filter for the primary news-first training sample."
        ),
        "outputs": {
            _display_path(png_path): _sha256(png_path),
            _display_path(svg_path): _sha256(svg_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    print(json.dumps(render(source_root, output_prefix), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
