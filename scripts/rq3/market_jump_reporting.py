"""Figures and notebook report for the RQ3 market-jump archive.

This module is deliberately presentation-only.  It reads the frozen tables
written by :mod:`scripts.rq3.market_jump_detection`; it never reopens the raw
market index or reruns anomaly selection.  Keeping that boundary explicit
makes every number in the notebook and HTML report traceable to a formal CSV
or JSON output in the same archive.
"""

from __future__ import annotations

import json
import math
import os
import re
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
BLUE_LIGHT = "#DCEAF5"
ORANGE = "#D9822B"
ORANGE_LIGHT = "#F7E4D1"
GOLD = "#C5A03D"
OLIVE = "#7A8450"
PINK = "#B56576"
NEUTRAL = "#A9B2B8"
WHITE = "#FFFFFF"

TIER_ORDER = ("high", "primary", "broad")
TIER_COLORS = {"high": BLUE, "primary": ORANGE, "broad": GOLD, "none": NEUTRAL}
TIER_HATCHES = {"high": "", "primary": "//", "broad": "..", "none": "xx"}
METRIC_LABELS = {
    "atm_iv_jump": "ATM IV jump",
    "smile_skew_jump": "Smile-skew jump",
    "rolling_jump_skew_change": "Rolling-skew change",
}
METRIC_COLORS = {
    "atm_iv_jump": BLUE,
    "smile_skew_jump": ORANGE,
    "rolling_jump_skew_change": GOLD,
}

CORE_REPORT_FILES = (
    "data_quality_summary.csv",
    "validation_summary.json",
    "atm_observations.csv.gz",
    "all_surface_points.csv.gz",
    "pair_slice_metrics.csv.gz",
    "metric_rankings.csv.gz",
    "market_pair_rankings.csv.gz",
    "candidate_pairs.csv",
    "candidate_episodes.csv",
    "episode_pair_members.csv",
    "field_dictionary.json",
)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a formal output while treating a headerless empty CSV as empty."""

    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _require_report_archive(output_root: Path) -> None:
    if not output_root.is_dir():
        raise FileNotFoundError(f"Market-jump output directory does not exist: {output_root}")
    missing = [name for name in CORE_REPORT_FILES if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Market-jump report archive is incomplete; missing formal outputs: "
            + ", ".join(missing)
        )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _configure_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.set_facecolor(WHITE)
    axis.tick_params(colors=INK, labelsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)


def _set_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.08, y=0.985, ha="left", va="top", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.08, 0.94, subtitle, ha="left", va="top", color=MUTED, fontsize=9)


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(WHITE)
    outputs = []
    for suffix in (".png", ".svg"):
        path = figures_dir / f"{stem}{suffix}"
        fig.savefig(
            path,
            dpi=170 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor=WHITE,
            metadata={"Creator": "wgan_option RQ3 market jump reporting"},
        )
        outputs.append(path)
    plt.close(fig)
    return outputs


def _annotate_bars(axis: plt.Axes, bars: Iterable[Any], *, horizontal: bool = False) -> None:
    for bar in bars:
        value = float(bar.get_width() if horizontal else bar.get_height())
        if not math.isfinite(value):
            continue
        if horizontal:
            axis.annotate(
                f"{value:,.2f}" if not float(value).is_integer() else f"{int(value):,}",
                (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=INK,
                fontsize=8,
            )
        else:
            axis.annotate(
                f"{value:,.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=8,
            )


def _placeholder(axis: plt.Axes, message: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=11,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_atm_quality(atm: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [0.9, 1.1]})
    fig.subplots_adjust(top=0.80, wspace=0.27)
    _set_title(
        fig,
        "Approximate ATM observation coverage",
        "One nearest observed strike per snapshot and real expiry; distance is |K/F - 1|.",
    )

    quality = _text(atm, "atm_quality", "unknown")
    preferred_order = ["A", "B", "C", "outside_5pct", "unusable", "unknown"]
    counts = quality.value_counts()
    labels = [label for label in preferred_order if label in counts.index]
    labels.extend(sorted(set(counts.index) - set(labels)))
    values = [int(counts.get(label, 0)) for label in labels]
    colors = [BLUE, ORANGE, GOLD, NEUTRAL, MUTED, NEUTRAL]
    if values:
        bars = axes[0].bar(
            np.arange(len(labels)),
            values,
            color=colors[: len(labels)],
            edgecolor=INK,
            linewidth=0.6,
        )
        for bar, label in zip(bars, labels):
            bar.set_hatch({"A": "", "B": "//", "C": ".."}.get(label, "xx"))
        axes[0].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        axes[0].set_ylabel("Snapshot-expiry observations")
        axes[0].set_title("Quality tier counts", loc="left", color=INK, fontsize=11)
        _annotate_bars(axes[0], bars)
        _configure_axis(axes[0])
    else:
        _placeholder(axes[0], "No ATM observations")

    distance = _numeric(atm, "abs_moneyness_distance").dropna() * 100.0
    if not distance.empty:
        upper = max(5.0, float(distance.quantile(0.995)))
        clipped = distance.clip(upper=upper)
        axes[1].hist(
            clipped,
            bins=45,
            color=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=0.7,
        )
        for threshold, label, style in ((1.0, "A: 1%", "-"), (2.0, "B: 2%", "--"), (5.0, "C: 5%", ":")):
            axes[1].axvline(threshold, color=ORANGE, linestyle=style, linewidth=1.5, label=label)
        axes[1].set_xlabel("Absolute ATM distance (%)")
        axes[1].set_ylabel("Observation count")
        axes[1].set_title("Distance distribution (99.5th percentile clipped)", loc="left", color=INK, fontsize=11)
        axes[1].legend(frameon=False, fontsize=8)
        _configure_axis(axes[1])
    else:
        _placeholder(axes[1], "ATM distance is unavailable")

    paths = _save_figure(fig, figures_dir, "01_atm_quality_coverage")
    metadata = {
        "id": "atm_quality_coverage",
        "question": "How much of the full snapshot-expiry population is close enough to ATM?",
        "family": "comparison_and_distribution",
        "source": "atm_observations.csv.gz",
        "rows": int(len(atm)),
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _rank_score(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    score = _numeric(frame, "abs_robust_z")
    if score.notna().any():
        return score, "Absolute robust z-score"
    percentile = _numeric(frame, "abs_empirical_percentile") * 100.0
    if percentile.notna().any():
        return percentile, "Absolute empirical percentile (%)"
    return _numeric(frame, "abs_metric_change"), "Absolute metric change"


def _plot_anomaly_ranking(rankings: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    flagged = rankings[_text(rankings, "anomaly_tier", "none").ne("none")].copy()
    score, score_label = _rank_score(flagged)
    flagged["_score"] = score
    flagged["_tier_order"] = _numeric(flagged, "anomaly_tier_order").fillna(0)
    flagged = flagged.sort_values(["_tier_order", "_score"], ascending=[False, False], kind="stable").head(20)
    fig, axis = plt.subplots(figsize=(12.0, 7.0))
    fig.subplots_adjust(top=0.84, left=0.30)
    _set_title(
        fig,
        "Top ranked ATM and skew anomalies",
        "Up to 20 flagged slice metrics, ordered by anomaly tier and within-bucket robust severity.",
    )
    if flagged.empty:
        _placeholder(axis, "No high, primary, or broad anomaly passed the configured thresholds")
    else:
        timestamps = pd.to_datetime(flagged.get("origin_time_utc"), utc=True, errors="coerce")
        metric_names = _text(flagged, "metric_name", "metric")
        maturity = _text(flagged, "maturity_bucket", "")
        labels = [
            f"{METRIC_LABELS.get(metric, metric)} | {time.strftime('%Y-%m-%d %H:%M') if not pd.isna(time) else 'unknown'} | {bucket}"
            for metric, time, bucket in zip(metric_names, timestamps, maturity)
        ]
        colors = [METRIC_COLORS.get(metric, BLUE) for metric in metric_names]
        positions = np.arange(len(flagged))
        bars = axis.barh(
            positions,
            flagged["_score"].fillna(0).to_numpy(),
            color=colors,
            edgecolor=INK,
            linewidth=0.6,
        )
        for bar, tier in zip(bars, _text(flagged, "anomaly_tier", "none")):
            bar.set_hatch(TIER_HATCHES.get(tier, "xx"))
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel(score_label)
        axis.set_title("Metric, UTC origin, and maturity bucket", loc="left", color=INK, fontsize=10)
        _annotate_bars(axis, bars, horizontal=True)
        _configure_axis(axis, grid_axis="x")
    paths = _save_figure(fig, figures_dir, "02_anomaly_ranking")
    metadata = {
        "id": "anomaly_ranking",
        "question": "Which detected slice-level changes have the highest robust severity?",
        "family": "ranking",
        "source": "metric_rankings.csv.gz",
        "rows": int(len(flagged)),
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _plot_anomaly_timeline(pair_rankings: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    work = pair_rankings[_text(pair_rankings, "anomaly_tier", "none").ne("none")].copy()
    work["_time"] = pd.to_datetime(work.get("origin_time_utc"), utc=True, errors="coerce")
    work["_score"] = _numeric(work, "max_abs_robust_z")
    fallback = _numeric(work, "max_abs_empirical_percentile") * 100.0
    score_label = "Maximum absolute robust z-score"
    if not work["_score"].notna().any():
        work["_score"] = fallback
        score_label = "Maximum absolute empirical percentile (%)"
    work = work[work["_time"].notna() & work["_score"].notna()].copy()

    fig, axis = plt.subplots(figsize=(12.0, 5.2))
    fig.subplots_adjust(top=0.82)
    _set_title(
        fig,
        "Market-pair anomaly timeline",
        "Each point is one unique five-minute market pair; news rows never duplicate market observations.",
    )
    if work.empty:
        _placeholder(axis, "No ranked market-pair anomaly is available")
    else:
        markers = {"high": "o", "primary": "s", "broad": "^"}
        for tier in TIER_ORDER:
            subset = work[_text(work, "anomaly_tier").eq(tier)]
            if subset.empty:
                continue
            axis.scatter(
                subset["_time"],
                subset["_score"],
                s={"high": 55, "primary": 42, "broad": 30}[tier],
                marker=markers[tier],
                facecolor=TIER_COLORS[tier] if tier != "broad" else WHITE,
                edgecolor=TIER_COLORS[tier] if tier != "broad" else GOLD,
                linewidth=1.0,
                alpha=0.88,
                label=tier,
            )
        axis.set_xlabel("Market-pair origin (UTC)")
        axis.set_ylabel(score_label)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
        axis.legend(frameon=False, ncol=3, loc="upper left", title="Anomaly tier")
        _configure_axis(axis)
    paths = _save_figure(fig, figures_dir, "03_anomaly_timeline")
    metadata = {
        "id": "anomaly_timeline",
        "question": "When do unique five-minute anomaly pairs occur?",
        "family": "trend_scatter",
        "source": "market_pair_rankings.csv.gz",
        "rows": int(len(work)),
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _plot_metric_tiers(rankings: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    work = rankings[_text(rankings, "anomaly_tier", "none").isin(TIER_ORDER)].copy()
    metric_names = list(METRIC_LABELS)
    observed = [name for name in _text(work, "metric_name").unique() if name not in metric_names]
    metric_names.extend(sorted(observed))
    counts = pd.crosstab(_text(work, "metric_name"), _text(work, "anomaly_tier"))
    counts = counts.reindex(index=metric_names, columns=list(TIER_ORDER), fill_value=0)

    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.20)
    _set_title(
        fig,
        "Flagged metrics by anomaly tier",
        "Counts are slice-level rankings; rolling-skew-only signals are capped at broad by design.",
    )
    if int(counts.to_numpy().sum()) == 0:
        _placeholder(axis, "No metric crossed the configured anomaly thresholds")
    else:
        positions = np.arange(len(counts))
        bottom = np.zeros(len(counts), dtype=float)
        for tier in TIER_ORDER:
            values = counts[tier].to_numpy(dtype=float)
            bars = axis.bar(
                positions,
                values,
                bottom=bottom,
                label=tier,
                color=TIER_COLORS[tier],
                edgecolor=INK,
                linewidth=0.6,
                hatch=TIER_HATCHES[tier],
            )
            for bar, value, offset in zip(bars, values, bottom):
                if value > 0:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        offset + value / 2,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=INK,
                    )
            bottom += values
        labels = [METRIC_LABELS.get(name, name) for name in counts.index]
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylabel("Flagged slice metrics")
        axis.legend(frameon=False, ncol=3, title="Anomaly tier")
        _configure_axis(axis)
    paths = _save_figure(fig, figures_dir, "04_metric_tier_distribution")
    metadata = {
        "id": "metric_tier_distribution",
        "question": "Which metrics and severity tiers contribute detected candidates?",
        "family": "composition",
        "source": "metric_rankings.csv.gz",
        "rows": int(len(work)),
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _plot_rolling_skew_diagnostic(
    pair_metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    rolling = _numeric(pair_metrics, "rolling_jump_skew_60m").dropna()
    flagged = rankings[
        _text(rankings, "metric_name").eq("rolling_jump_skew_change")
        & _text(rankings, "anomaly_tier", "none").ne("none")
    ].copy()
    flagged["_time"] = pd.to_datetime(
        flagged.get("origin_time_utc"), utc=True, errors="coerce"
    )
    flagged["_change"] = _numeric(flagged, "metric_change")
    flagged = flagged[flagged["_time"].notna() & flagged["_change"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [0.9, 1.1]})
    fig.subplots_adjust(top=0.79, wspace=0.28)
    _set_title(
        fig,
        "Rolling ATM-jump skewness diagnostic",
        "Bias-corrected Fisher–Pearson skew over the 60-minute window; skew and its change are dimensionless.",
    )
    if rolling.empty:
        _placeholder(axes[0], "rolling_jump_skew_60m is unavailable")
    else:
        lower, upper = rolling.quantile([0.005, 0.995]).tolist()
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            lower, upper = float(rolling.min()), float(rolling.max())
        clipped = rolling.clip(lower=lower, upper=upper) if lower < upper else rolling
        axes[0].hist(
            clipped,
            bins=45,
            color=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=0.7,
        )
        axes[0].axvline(0.0, color=INK, linewidth=1.0, linestyle=":")
        axes[0].set_xlabel("Rolling Fisher–Pearson skewness")
        axes[0].set_ylabel("Eligible slice-pair observations")
        axes[0].set_title(
            "60-minute rolling skew distribution (0.5–99.5% clipped)",
            loc="left",
            color=INK,
            fontsize=10,
        )
        _configure_axis(axes[0])

    if flagged.empty:
        _placeholder(axes[1], "No rolling-skew change crossed the broad threshold")
    else:
        markers = {"high": "o", "primary": "s", "broad": "^"}
        use_categorical_time = len(flagged) < 4
        if use_categorical_time:
            flagged = flagged.sort_values("_time", kind="stable").copy()
            flagged["_time_position"] = np.arange(len(flagged), dtype=float)
        for tier in TIER_ORDER:
            subset = flagged[_text(flagged, "anomaly_tier").eq(tier)]
            if subset.empty:
                continue
            axes[1].scatter(
                subset["_time_position"] if use_categorical_time else subset["_time"],
                subset["_change"],
                s={"high": 52, "primary": 40, "broad": 30}[tier],
                marker=markers[tier],
                facecolor=TIER_COLORS[tier] if tier != "broad" else WHITE,
                edgecolor=TIER_COLORS[tier],
                linewidth=1.0,
                alpha=0.88,
                label=tier,
            )
        axes[1].axhline(0.0, color=INK, linewidth=1.0, linestyle=":")
        axes[1].set_xlabel("Market-pair origin (UTC)")
        axes[1].set_ylabel("Change in rolling skewness")
        axes[1].set_title(
            "Flagged rolling-skew changes",
            loc="left",
            color=INK,
            fontsize=10,
        )
        if use_categorical_time:
            labels = [timestamp.strftime("%Y-%m-%d\n%H:%M") for timestamp in flagged["_time"]]
            axes[1].set_xticks(flagged["_time_position"], labels)
            axes[1].margins(x=0.20)
        else:
            axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
            axes[1].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator())
            )
        axes[1].legend(frameon=False, ncol=3, title="Anomaly tier")
        _configure_axis(axes[1])
    paths = _save_figure(fig, figures_dir, "06_rolling_skew_diagnostic")
    metadata = {
        "id": "rolling_skew_diagnostic",
        "question": "What is the distribution of rolling ATM-jump skewness, and when do its flagged changes occur?",
        "family": "distribution_and_time_scatter",
        "source": "pair_slice_metrics.csv.gz; metric_rankings.csv.gz",
        "rows": int(len(rolling)),
        "flagged_rows": int(len(flagged)),
        "unit": "dimensionless Fisher-Pearson skewness",
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _plot_metric_change_distributions(
    rankings: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Show the signed ATM-IV and local-smile-skew change distributions."""

    specifications = (
        (
            "atm_iv_jump",
            "ATM IV change distribution",
            "Target − current ATM IV (percentage points)",
            BLUE,
            100.0,
        ),
        (
            "smile_skew_jump",
            "Local smile-skew change distribution",
            "Target − current IV/log(K/F) slope",
            ORANGE,
            1.0,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    fig.subplots_adjust(top=0.79, wspace=0.28)
    _set_title(
        fig,
        "ATM and local smile-skew change distributions",
        "All quality-eligible ranked slice pairs; tails are clipped only for display, never for scoring.",
    )
    row_counts: dict[str, int] = {}
    for axis, (metric_name, title, xlabel, color, multiplier) in zip(
        axes, specifications
    ):
        values = _numeric(
            rankings[_text(rankings, "metric_name").eq(metric_name)],
            "metric_change",
        ).dropna() * float(multiplier)
        row_counts[metric_name] = int(len(values))
        if values.empty:
            _placeholder(axis, f"{metric_name} is unavailable")
            continue
        lower, upper = values.quantile([0.005, 0.995]).tolist()
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            lower, upper = float(values.min()), float(values.max())
        clipped = values.clip(lower=lower, upper=upper) if lower < upper else values
        axis.hist(
            clipped,
            bins=50,
            color=color,
            alpha=0.22,
            edgecolor=color,
            linewidth=0.75,
        )
        axis.axvline(0.0, color=INK, linewidth=1.0, linestyle=":", label="zero")
        axis.axvline(
            float(values.median()),
            color=color,
            linewidth=1.2,
            linestyle="--",
            label="median",
        )
        axis.set_title(title, loc="left", color=INK, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Eligible slice-pair observations")
        axis.legend(frameon=False)
        _configure_axis(axis)
    paths = _save_figure(fig, figures_dir, "07_metric_change_distributions")
    metadata = {
        "id": "metric_change_distributions",
        "question": "What are the signed ATM-IV and local smile-skew change distributions?",
        "family": "distribution",
        "source": "metric_rankings.csv.gz",
        "rows_by_metric": row_counts,
        "display_tail_policy": "0.5th to 99.5th percentile clipping only",
        "outputs": [path.name for path in paths],
    }
    return paths, metadata


def _bridge_episode_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "episode_id" not in frame.columns:
        return set()
    return {value for value in frame["episode_id"].fillna("").astype(str) if value}


def _plot_evidence_matches(
    episodes: pd.DataFrame,
    official_bridge: pd.DataFrame,
    news_bridge: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    episode_ids = _bridge_episode_ids(episodes)
    official_ids = _bridge_episode_ids(official_bridge)
    news_ids = _bridge_episode_ids(news_bridge)
    counts = {
        "Both": len(episode_ids & official_ids & news_ids),
        "Official only": len((episode_ids & official_ids) - news_ids),
        "Factiva only": len((episode_ids & news_ids) - official_ids),
        "Neither": len(episode_ids - official_ids - news_ids),
    }
    labels = list(counts)
    values = list(counts.values())
    colors = [BLUE, ORANGE, GOLD, NEUTRAL]
    hatches = ["", "//", "..", "xx"]

    fig, axis = plt.subplots(figsize=(9.5, 4.9))
    fig.subplots_adjust(top=0.80)
    _set_title(
        fig,
        "Post-hoc evidence coverage for candidate episodes",
        "Official releases and Factiva availability are independent evidence lanes, not causal labels.",
    )
    if not episode_ids:
        _placeholder(axis, "No candidate episode is available for evidence matching")
    else:
        bars = axis.bar(np.arange(len(labels)), values, color=colors, edgecolor=INK, linewidth=0.7)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylabel("Unique candidate episodes")
        _annotate_bars(axis, bars)
        _configure_axis(axis)
    paths = _save_figure(fig, figures_dir, "05_episode_evidence_coverage")
    metadata = {
        "id": "episode_evidence_coverage",
        "question": "How many market-first episodes have nearby official releases or Factiva articles?",
        "family": "composition_comparison",
        "source": "candidate_episodes.csv; episode_official_event_bridge.csv; episode_news_bridge.csv",
        "rows": int(len(episode_ids)),
        "outputs": [path.name for path in paths],
        "counts": counts,
    }
    return paths, metadata


def _safe_file_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return token[:80] or "unknown"


def _select_case_specs(
    episodes: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if episodes.empty or candidate_pairs.empty or pair_metrics.empty or limit <= 0:
        return []
    episode_work = episodes.copy()
    episode_work["_episode_rank"] = _numeric(episode_work, "episode_rank").fillna(np.inf)
    episode_work = episode_work.sort_values(
        ["_episode_rank", "max_abs_robust_z"],
        ascending=[True, False],
        kind="stable",
    )
    pairs_by_id = candidate_pairs.set_index("pair_id", drop=False) if "pair_id" in candidate_pairs.columns else pd.DataFrame()
    cases: list[dict[str, Any]] = []
    for episode in episode_work.itertuples(index=False):
        pair_id = str(getattr(episode, "peak_pair_id", ""))
        if not pair_id or pair_id not in pairs_by_id.index:
            continue
        pair = pairs_by_id.loc[pair_id]
        if isinstance(pair, pd.DataFrame):
            pair = pair.iloc[0]
        maturity = str(pair.get("peak_maturity_date", ""))
        metrics = pair_metrics[_text(pair_metrics, "pair_id").eq(pair_id)].copy()
        if maturity:
            selected = metrics[_text(metrics, "maturity_date").eq(maturity)]
        else:
            selected = pd.DataFrame()
        if selected.empty and not metrics.empty:
            atm_score = _numeric(metrics, "abs_delta_atm_iv").fillna(-np.inf)
            skew_score = _numeric(metrics, "abs_delta_atm_iv_skew_secant").fillna(-np.inf)
            selected = metrics.loc[[np.maximum(atm_score, skew_score).idxmax()]]
        if selected.empty:
            continue
        row = selected.iloc[0]
        cases.append(
            {
                "episode_id": str(getattr(episode, "episode_id")),
                "episode_rank": int(getattr(episode, "episode_rank", len(cases) + 1)),
                "anomaly_tier": str(getattr(episode, "anomaly_tier", "")),
                "peak_metric_name": str(getattr(episode, "peak_metric_name", "")),
                "pair_id": pair_id,
                "origin_time_utc": str(row.get("origin_time_utc", pair.get("origin_time_utc", ""))),
                "target_time_utc": str(row.get("target_time_utc", pair.get("target_time_utc", ""))),
                "maturity_date": str(row.get("maturity_date", maturity)),
                "pair_atm_strike": pd.to_numeric(pd.Series([row.get("pair_atm_strike")]), errors="coerce").iloc[0],
            }
        )
        if len(cases) >= int(limit):
            break
    return cases


def _load_case_points(path: Path, cases: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame()
    anchors = {
        str(case[time_key])
        for case in cases
        for time_key in ("origin_time_utc", "target_time_utc")
    }
    maturities = {str(case["maturity_date"]) for case in cases}
    columns = [
        "anchor_time_utc",
        "anchor_time_london",
        "maturity_date",
        "underlying_contract_id",
        "strike",
        "strike_over_forward",
        "log_moneyness",
        "implied_vol",
        "implied_vol_pct",
        "weight_sum",
    ]
    chunks: list[pd.DataFrame] = []
    try:
        iterator = pd.read_csv(path, usecols=lambda column: column in columns, chunksize=100_000, low_memory=False)
        for chunk in iterator:
            mask = _text(chunk, "anchor_time_utc").isin(anchors) & _text(chunk, "maturity_date").isin(maturities)
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _plot_case_smiles(
    cases: Sequence[Mapping[str, Any]],
    case_points: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    if case_points.empty:
        return paths, manifests
    for sequence, case in enumerate(cases, start=1):
        current = case_points[
            _text(case_points, "anchor_time_utc").eq(str(case["origin_time_utc"]))
            & _text(case_points, "maturity_date").eq(str(case["maturity_date"]))
        ].copy()
        target = case_points[
            _text(case_points, "anchor_time_utc").eq(str(case["target_time_utc"]))
            & _text(case_points, "maturity_date").eq(str(case["maturity_date"]))
        ].copy()
        if current.empty or target.empty:
            continue
        for frame in (current, target):
            frame["_x"] = (_numeric(frame, "strike_over_forward") - 1.0) * 100.0
            frame["_y"] = _numeric(frame, "implied_vol") * 100.0
            frame.sort_values("_x", inplace=True, kind="stable")
        fig, axis = plt.subplots(figsize=(9.5, 5.1))
        fig.subplots_adjust(top=0.78)
        _set_title(
            fig,
            f"Case smile {sequence}: episode {case['episode_rank']} ({case['anomaly_tier']})",
            f"Real expiry {case['maturity_date']}; current and target are five minutes apart, IV in percent.",
        )
        axis.plot(
            current["_x"],
            current["_y"],
            color=BLUE,
            marker="o",
            markerfacecolor=WHITE,
            markeredgecolor=BLUE,
            linewidth=1.8,
            label="Current",
        )
        axis.plot(
            target["_x"],
            target["_y"],
            color=ORANGE,
            marker="s",
            markerfacecolor=ORANGE_LIGHT,
            markeredgecolor=ORANGE,
            linestyle="--",
            linewidth=1.8,
            label="Target (+5 min)",
        )
        axis.axvline(0.0, color=INK, linewidth=1.0, linestyle=":", label="K/F = 1")
        axis.set_xlabel("Observed moneyness, 100 × (K/F - 1) (%)")
        axis.set_ylabel("Annualized implied volatility (%)")
        axis.legend(frameon=False, ncol=3, loc="best")
        _configure_axis(axis)
        stem = f"case_smile_{sequence:02d}_{_safe_file_token(case['episode_id'])}"
        case_paths = _save_figure(fig, figures_dir, stem)
        paths.extend(case_paths)
        manifests.append(
            {
                "id": stem,
                "question": "How did the observed raw-IV smile change across the peak five-minute pair?",
                "family": "comparison_line",
                "source": "all_surface_points.csv.gz; pair_slice_metrics.csv.gz; candidate_episodes.csv",
                "episode_id": str(case["episode_id"]),
                "pair_id": str(case["pair_id"]),
                "maturity_date": str(case["maturity_date"]),
                "outputs": [path.name for path in case_paths],
            }
        )
    return paths, manifests


def generate_figures(
    output_root: str | Path,
    max_case_events: int = 12,
) -> list[Path]:
    """Create report-ready PNG and SVG figures from formal archive tables."""

    root = Path(output_root).resolve()
    _require_report_archive(root)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    atm = _read_csv(root / "atm_observations.csv.gz")
    rankings = _read_csv(root / "metric_rankings.csv.gz")
    pair_rankings = _read_csv(root / "market_pair_rankings.csv.gz")
    candidate_pairs = _read_csv(root / "candidate_pairs.csv")
    episodes = _read_csv(root / "candidate_episodes.csv")
    pair_metrics = _read_csv(root / "pair_slice_metrics.csv.gz")
    official_bridge = _read_csv(root / "episode_official_event_bridge.csv")
    news_bridge = _read_csv(root / "episode_news_bridge.csv")

    outputs: list[Path] = []
    chart_manifest: list[dict[str, Any]] = []
    for builder, args in (
        (_plot_atm_quality, (atm, figures_dir)),
        (_plot_anomaly_ranking, (rankings, figures_dir)),
        (_plot_anomaly_timeline, (pair_rankings, figures_dir)),
        (_plot_metric_tiers, (rankings, figures_dir)),
        (_plot_evidence_matches, (episodes, official_bridge, news_bridge, figures_dir)),
        (_plot_rolling_skew_diagnostic, (pair_metrics, rankings, figures_dir)),
        (_plot_metric_change_distributions, (rankings, figures_dir)),
    ):
        figure_paths, metadata = builder(*args)
        outputs.extend(figure_paths)
        chart_manifest.append(metadata)

    cases = _select_case_specs(
        episodes,
        candidate_pairs,
        pair_metrics,
        limit=max(0, int(max_case_events)),
    )
    case_points = _load_case_points(root / "all_surface_points.csv.gz", cases)
    case_paths, case_manifest = _plot_case_smiles(cases, case_points, figures_dir)
    outputs.extend(case_paths)
    chart_manifest.extend(case_manifest)

    manifest = {
        "renderer": "matplotlib",
        "background": "white",
        "palette_policy": "approved blue/orange/gold/neutral roots; no red-green semantics",
        "formal_inputs_only": True,
        "charts": chart_manifest,
    }
    (figures_dir / "chart_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return outputs


def _code_cell_source() -> str:
    return """from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

output_root = Path.cwd().resolve()

def read_csv(name, **kwargs):
    path = output_root / name
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

quality = read_csv('data_quality_summary.csv')
coverage = read_csv('dataset_coverage.csv')
exclusions = read_csv('exclusion_reason_summary.csv')
fields = read_csv('field_dictionary.csv')
episodes = read_csv('candidate_episodes.csv')
candidate_pairs = read_csv('candidate_pairs.csv')
official_bridge = read_csv('episode_official_event_bridge.csv')
news_bridge = read_csv('episode_news_bridge.csv')
"""


def _summary_cell_source() -> str:
    return """def compact_count(value):
    try:
        return f'{int(value):,}'
    except (TypeError, ValueError):
        return str(value)

status = 'pass' if not quality.empty and quality['status'].astype(str).eq('pass').all() else 'needs_review'
candidate_count = len(candidate_pairs)
episode_count = len(episodes)
official_rows = len(official_bridge)
news_rows = len(news_bridge)
counts = quality.set_index('check')['actual'].to_dict() if not quality.empty else {}
atm_count = counts.get('reconstructed_slice_count', 'unknown')
point_count = counts.get('reconstructed_surface_point_count', 'unknown')
display(Markdown(f'''**Technical summary.** The frozen archive has validation status **{status}**. It reconstructed **{compact_count(point_count)}** observed strike-IV points and **{compact_count(atm_count)}** approximate-ATM snapshot-expiry observations, then identified **{compact_count(candidate_count)}** candidate five-minute market pairs grouped into **{compact_count(episode_count)}** episodes. Post-hoc evidence bridges contain **{compact_count(official_rows)}** official-release links and **{compact_count(news_rows)}** Factiva links. These are time-associated candidates, not causal attributions.'''))
"""


def _data_cell_source() -> str:
    return """display(Markdown('### Frozen validation checks'))
display(quality.head(20).style.hide(axis='index'))

display(Markdown('### Row counts and date coverage'))
display(coverage.style.hide(axis='index'))

display(Markdown('### Explicit exclusions and audit-only differences'))
display(exclusions.style.hide(axis='index'))

display(Markdown('### Bounded previews of formal outputs'))
preview_files = [
    'atm_observations.csv.gz',
    'pair_slice_metrics.csv.gz',
    'metric_rankings.csv.gz',
    'market_pair_rankings.csv.gz',
]
for name in preview_files:
    preview = read_csv(name, nrows=5)
    display(Markdown(f'**{name}** — first {len(preview)} rows only'))
    display(preview)
"""


def _episode_cell_source() -> str:
    return """if episodes.empty:
    display(Markdown('No episode crossed the configured anomaly thresholds.'))
else:
    preferred = [
        'episode_rank', 'episode_start_utc', 'episode_end_utc', 'anomaly_tier',
        'peak_metric_name', 'peak_metric_quality', 'peak_signed_direction',
        'peak_metric_change', 'peak_maturity_date', 'max_abs_robust_z',
        'market_pair_count',
    ]
    columns = [column for column in preferred if column in episodes.columns]
    display(episodes.loc[:, columns].head(20).style.hide(axis='index'))
"""


def _evidence_cell_source() -> str:
    return """display(Markdown('### Official releases'))
if official_bridge.empty:
    display(Markdown('No official release fell in the configured evidence windows, or the optional bridge was not supplied.'))
else:
    preferred = ['episode_id', 'window_relation', 'event_id', 'event_name', 'event_type', 'release_time_utc']
    display(official_bridge[[column for column in preferred if column in official_bridge.columns]].head(20))

display(Markdown('### Factiva availability'))
if news_bridge.empty:
    display(Markdown('No Factiva article fell in the configured evidence windows, or the optional bridge was not supplied.'))
else:
    preferred = ['episode_id', 'window_relation', 'news_available_time_utc', 'headline', 'article_id', 'alignment_match_method']
    display(news_bridge[[column for column in preferred if column in news_bridge.columns]].head(20))
"""


def _drift_limitation_cell_source() -> str:
    return """drift = exclusions[exclusions.get('reason', pd.Series(dtype=str)).astype(str).eq('frozen_params_vs_precalib_drift')]
if not drift.empty:
    row = drift.iloc[0]
    rate = 100.0 * float(row['count']) / float(row['denominator']) if float(row['denominator']) else float('nan')
    display(Markdown(
        f"**Frozen-surface reconciliation.** {int(row['count']):,} of {int(row['denominator']):,} "
        f"slices ({rate:.2f}%) differ from the persisted precalibration reconstruction above numerical tolerance. "
        f"The maximum absolute q difference is {float(row['max_abs_q_difference']):.6g}; the maximum absolute IV difference is "
        f"{float(row['max_abs_iv_difference']):.6g}. Frozen `surface_anchor.params_json` q/IV values are authoritative; "
        "precalibration values remain lineage/audit fields."
    ))
"""


def _takeaways_cell_source() -> str:
    return """failed = quality.loc[quality.get('status', pd.Series(dtype=str)).astype(str).eq('fail')] if not quality.empty and 'status' in quality else pd.DataFrame()
official_ids = set(official_bridge.get('episode_id', pd.Series(dtype=str)).dropna().astype(str))
news_ids = set(news_bridge.get('episode_id', pd.Series(dtype=str)).dropna().astype(str))
episode_ids = set(episodes.get('episode_id', pd.Series(dtype=str)).dropna().astype(str))
matched = len(episode_ids & (official_ids | news_ids))
if episode_ids:
    evidence_text = f'{matched:,} of {len(episode_ids):,} candidate episodes have at least one nearby official or Factiva item in the configured windows.'
else:
    evidence_text = 'No episode crossed the configured threshold, so there is no event-evidence association to interpret.'
validation_text = 'All frozen count checks passed.' if failed.empty else f'{len(failed)} frozen count check(s) require review before relying on the rankings.'
display(Markdown(f'''- **Use the episode table as the investigation queue.** It is deduplicated at the market-pair level before news is joined.
- **Treat evidence as temporal context only.** {evidence_text}
- **Respect data-quality gates.** {validation_text}
- **Next step.** Review the highest-ranked case smiles together with all one-to-many news candidates; do not select a headline solely because it is convenient or prominent.'''))
"""


def _display_figure_cell(relative_path: str) -> str:
    return (
        "path = output_root / " + repr(relative_path) + "\n"
        "if path.is_file():\n"
        "    display(Image(filename=str(path)))\n"
        "else:\n"
        "    display(Markdown(f'Figure unavailable: `{path.name}`'))\n"
    )


def build_companion_notebook(output_root: str | Path) -> Path:
    """Build the formal-output-only technical notebook companion."""

    root = Path(output_root).resolve()
    _require_report_archive(root)
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError("Notebook reporting requires nbformat (install nbformat>=5.9).") from exc

    cells = [
        nbformat.v4.new_markdown_cell(
            "# TY 期权近似 ATM Vol、Skew 突变与新闻关联分析\n\n"
            "完整市场样本、五分钟异常检测和事后新闻证据的可审计技术报告。"
        ),
        nbformat.v4.new_markdown_cell("## tl;dr"),
        nbformat.v4.new_code_cell(_code_cell_source()),
        nbformat.v4.new_code_cell(_summary_cell_source()),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "近似 ATM 是每个真实到期切片中 `|log(K/F)|` 最小的实际观测点，并不要求 `K/F=1`。"
            "五分钟 ATM 变化在 current/target 两端使用同一名义执行价。局部 smile skew 是共同左右翼之间的 "
            "IV 对 `log(K/F)` 的 secant slope；滚动 skewness 是合格 signed ATM jumps 的 Fisher–Pearson skew。\n\n"
            "### Key Assumptions\n\n"
            "- 发现样本先覆盖完整市场索引，官方事件和 Factiva 只在排名完成后连接。\n"
            "- A/B/C ATM 质量分别对应两端距离不超过 1%/2%/5%。\n"
            "- high、primary、broad 是稳健统计分级，不表示新闻造成了价格变化。\n"
            "- 五分钟收益存在重叠，因此 rolling-skew-only 信号最多进入 broad。"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n所有计算表和预览均直接读取本目录的正式 CSV；大表只显示前五行。"),
        nbformat.v4.new_code_cell(_data_cell_source()),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n"
            "以下图形分别检查近 ATM 覆盖、异常排序、时间聚集、指标构成和事后证据覆盖。"
            "颜色之外同时使用 marker、hatch 或线型区分类别。"
        ),
    ]

    figure_notes = {
        "01_atm_quality_coverage.png": "ATM 覆盖图显示观测点距离与质量层级；超过 5% 的点仍在全集中，但不进入正式异常排名。",
        "02_anomaly_ranking.png": "排名图使用分层后的 robust severity；不同指标的原始单位不可直接比较，因此优先读稳健分数。",
        "03_anomaly_timeline.png": "时间图以唯一五分钟 market pair 为粒度，避免同分钟多篇新闻扩大异常数量。",
        "04_metric_tier_distribution.png": "层级构成图用于检查异常是否集中在单一指标，并提醒 rolling skewness 只是辅助证据。",
        "05_episode_evidence_coverage.png": "证据覆盖图区分官方事件和 Factiva，不匹配不代表没有市场信息冲击。",
        "06_rolling_skew_diagnostic.png": "Rolling skew 图左侧展示 60 分钟 Fisher–Pearson skewness 的分布，右侧只标出越过阈值的变化；该指标无量纲且只作为辅助诊断。",
        "07_metric_change_distributions.png": "ATM IV 与局部 smile-skew 的 signed change 分布分别使用自身单位；尾部仅为绘图裁剪，异常评分始终使用未裁剪数据。",
    }
    for filename, note in figure_notes.items():
        if (root / "figures" / filename).is_file():
            cells.append(nbformat.v4.new_markdown_cell(f"### {filename.removesuffix('.png').replace('_', ' ').title()}\n\n{note}"))
            cells.append(nbformat.v4.new_code_cell(_display_figure_cell(f"figures/{filename}")))

    cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "### Highest-ranked candidate episodes\n\n"
                "最多展示前 20 个 episode；精确明细和全部期限保留在正式 CSV。"
            ),
            nbformat.v4.new_code_cell(_episode_cell_source()),
            nbformat.v4.new_markdown_cell(
                "### Time-associated official and Factiva evidence\n\n"
                "同一 episode 可以对应多条证据；下表有界展示前 20 条，不强制选择唯一解释。"
            ),
            nbformat.v4.new_code_cell(_evidence_cell_source()),
        ]
    )

    case_files = sorted((root / "figures").glob("case_smile_*.png"))
    if case_files:
        cells.append(
            nbformat.v4.new_markdown_cell(
                "### Peak-event observed smile cases\n\n"
                "每张图比较同一真实到期日在 current 与五分钟 target 的实际 raw-IV 点。"
                "曲线仅连接观测点以辅助阅读，不代表额外插值。"
            )
        )
        for case_path in case_files:
            cells.append(nbformat.v4.new_code_cell(_display_figure_cell(f"figures/{case_path.name}")))

    cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "## Limitations, uncertainty, and robustness\n\n"
                "- ATM 与 smile skew 都受可交易执行价密度和报价质量影响；质量层级必须与异常分数一起解释。\n"
                "- Smile secant 要求两端共有且始终位于 ATM 两侧的执行价，并设置最小 log-moneyness span；这减少爆炸值，但牺牲覆盖。\n"
                "- Robust z-score 和经验分位数是在年份与期限桶内计算，结果依赖预设阈值；0/10 分钟 episode 聚类敏感性保存在独立表中。\n"
                "- 新闻关联仅说明发布时间与异常窗口接近，不识别未报道信息，也不建立因果关系。"
            ),
            nbformat.v4.new_code_cell(_drift_limitation_cell_source()),
            nbformat.v4.new_markdown_cell("## Takeaways"),
            nbformat.v4.new_code_cell(_takeaways_cell_source()),
            nbformat.v4.new_markdown_cell(
                "### Further questions\n\n"
                "- 高等级异常在不同期限上是否方向一致，还是由单一稀疏期限驱动？\n"
                "- 官方事件窗口内的异常相对同一时段无事件基线是否更集中？\n"
                "- 在加入非重叠收益和成交量/流动性控制后，rolling skewness 信号是否仍然存在？"
            ),
        ]
    )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "title": "TY 期权近似 ATM Vol、Skew 突变与新闻关联分析",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "rq3_report": {
                "formal_inputs_only": True,
                "required_files": list(CORE_REPORT_FILES),
                "report_shape": "technical",
            },
        },
    )
    notebook_path = root / "analysis.ipynb"
    nbformat.write(notebook, notebook_path)
    return notebook_path


def render_notebook_html(
    notebook_path: str | Path,
    output_html: str | Path,
    *,
    timeout_seconds: int = 600,
) -> Path:
    """Execute a companion notebook and export a self-contained HTML report."""

    source = Path(notebook_path).resolve()
    target = Path(output_html).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Companion notebook does not exist: {source}")
    # Avoid jupyter_core's legacy-path deprecation and make notebook execution
    # deterministic across clean CI and research workstations.
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    try:
        import nbformat
        from nbconvert import HTMLExporter
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise RuntimeError(
            "Notebook rendering requires nbformat, nbconvert, and an installed Python kernel."
        ) from exc

    notebook = nbformat.read(source, as_version=4)
    executor = ExecutePreprocessor(
        timeout=int(timeout_seconds),
        kernel_name="python3",
        allow_errors=False,
    )
    executor.preprocess(notebook, resources={"metadata": {"path": str(source.parent)}})
    nbformat.write(notebook, source)

    exporter = HTMLExporter()
    exporter.embed_images = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.mathjax_url = ""
    exporter.require_js_url = ""
    report_title = str(
        notebook.metadata.get(
            "title", "TY 期权近似 ATM Vol、Skew 突变与新闻关联分析"
        )
    )
    body, _ = exporter.from_notebook_node(
        notebook,
        resources={"metadata": {"name": report_title}},
    )
    # nbconvert emits an empty script element when remote RequireJS is
    # disabled.  Removing it avoids a browser request to the current document
    # while keeping all report content and embedded images self-contained.
    body = body.replace('<script src=""></script>', "")
    body = re.sub(
        r"<title>.*?</title>",
        f"<title>{escape(report_title)}</title>",
        body,
        count=1,
        flags=re.DOTALL,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


__all__ = [
    "build_companion_notebook",
    "generate_figures",
    "render_notebook_html",
]
