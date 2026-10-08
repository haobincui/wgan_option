"""Draw the Chapter 3 news/market timing schematic without editing the thesis."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq3.plot_chapter3_forecast_examples import _save, _sha256, _style

THESIS_ROOT = ROOT.parent / "PhdThesis"
SOURCE = THESIS_ROOT / "Chapter3/chapter3.tex"
DEFAULT_OUTPUT_ROOT = THESIS_ROOT / "Chapter3/Chapter3Figs/additional_figures/timing_design"
STEM = "ch3_news_market_timing"
INK = "#202020"
BLUE = "#2F75B5"
GOLD = "#D6A23D"
PALE_BLUE = "#E5EEF7"
PALE_GOLD = "#F7EED8"


def _arrow(axis, start, end, y, colour=INK):
    axis.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="<->",
                                  mutation_scale=9, linewidth=1, color=colour))


def _panel(axis, delay, title):
    tau = -float(delay)
    axis.set_xlim(-5.85, 5.85)
    axis.set_ylim(-0.90, 2.05)
    axis.set_axis_off()
    axis.set_title(title, loc="left", fontsize=13, pad=13)
    axis.add_patch(Rectangle((-5, 0.56), 5, 0.45, facecolor=PALE_BLUE,
                             edgecolor=BLUE, linewidth=1))
    axis.add_patch(Rectangle((0, 0.56), 5, 0.45, facecolor=PALE_GOLD,
                             edgecolor=GOLD, linewidth=1))
    if delay:
        axis.add_patch(Rectangle((tau, 0.56), delay, 0.45, facecolor="none",
                                 edgecolor="#8796A5", hatch="///", linewidth=0))
    # The recorded availability marker identifies a minute proxy, not a
    # verified message-delivery timestamp within that minute.
    axis.plot([tau, tau], [0.23, 1.10], color=INK, linewidth=1,
              linestyle=(0, (3, 3)), zorder=2)
    axis.scatter([tau], [1.10], marker="D", s=28, color=INK, zorder=3)
    current = axis.text(-2.5, 0.785, "Current IVS", ha="center", va="center", fontsize=12)
    target = axis.text(2.5, 0.785, "Realised target IVS", ha="center", va="center", fontsize=12)
    current.set_bbox(dict(facecolor=PALE_BLUE, edgecolor="none", alpha=0.95, pad=1))
    target.set_bbox(dict(facecolor=PALE_GOLD, edgecolor="none", alpha=0.95, pad=1))
    axis.add_patch(FancyArrowPatch((-5.55, 0.22), (5.55, 0.22), arrowstyle="->",
                                  mutation_scale=10, linewidth=1, color=INK))
    for x, label in ((-5, r"$t_a-5$"), (0, r"$t_a$ (forecast origin)"), (5, r"$t_a+5$")):
        axis.plot([x, x], [0.17, 0.28], color=INK, linewidth=1)
        axis.text(x, 0.04, label, ha="center", va="top", fontsize=12)
    _arrow(axis, 0, 5, -0.40, colour=GOLD)
    axis.text(2.5, -0.55, "Forecast horizon: 5 minutes", ha="center", va="top", fontsize=11.5)
    if delay == 0:
        axis.text(tau, 1.24, r"$\tau_a=t_a$", ha="center", va="bottom", fontsize=12)
    else:
        axis.text(tau, 1.22, r"$\tau_a$", ha="center", va="bottom", fontsize=12)
        _arrow(axis, tau, 0, 1.60)
        axis.text((tau + 0) / 2, 1.76, "Alignment delay: 3 minutes", ha="center", fontsize=11.5)


def render(output_root):
    source_text = SOURCE.read_text(encoding="utf-8")
    for label in ("eq:ch3:news_market_origin", "eq:ch3:news_market_windows"):
        if f"\\label{{{label}}}" not in source_text:
            raise ValueError(f"The timing source no longer defines {label}.")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _style()
    fig = plt.figure(figsize=(8, 6.6))
    axes = [fig.add_axes((0.065, bottom, 0.895, 0.26)) for bottom in (0.49, 0.10)]
    fig.suptitle("News availability and five-minute market transitions", y=0.975, fontsize=15)
    legend = [Patch(facecolor=PALE_BLUE, edgecolor=BLUE, label="Current-surface window"),
              Patch(facecolor=PALE_GOLD, edgecolor=GOLD, label="Target-surface window"),
              Patch(facecolor="white", edgecolor="#8796A5", hatch="///",
                    label="Current observations after news"),
              Line2D([], [], marker="D", color=INK, linestyle="none", markersize=4,
                     label="Recorded news availability")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=2, frameon=False, fontsize=11.5)
    _panel(axes[0], 0, "(a) Origin at the recorded news-availability minute")
    _panel(axes[1], 3, "(b) Origin after the recorded news-availability minute")
    fig.text(0.5, 0.025, "Time is shown relative to the forecast origin; timestamps follow UTC.",
             ha="center", fontsize=11.5, color="#62676D")
    _save(fig, output_root, STEM, "News availability and five-minute surface windows")
    records = []
    for panel, delay in (("a", 0), ("b", 3)):
        for element, start, end in (("current_window", -5, 0), ("target_window", 0, 5),
                                    ("alignment_delay", -delay, 0), ("forecast_horizon", 0, 5)):
            records.append({"panel": panel, "element": element,
                            "start_minutes_relative_to_origin": start,
                            "end_minutes_relative_to_origin": end,
                            "interval_convention": "left_closed_right_open" if "window" in element else "duration",
                            "news_availability_minutes_relative_to_origin": -delay,
                            "illustrative_example": panel == "b"})
    pd.DataFrame(records).to_csv(output_root / f"{STEM}_data.csv", index=False)
    caption = (
        r"Timing of news availability and the five-minute IVS forecast. The current surface uses $[t_a-5,t_a)$ "
        r"and the realised target uses $[t_a,t_a+5)$. Panel (a) shows an origin coinciding with the "
        r"recorded availability minute $\tau_a$. Panel (b) illustrates a three-minute delay to the first "
        "eligible origin; hatching identifies current-window observations after news availability. "
        "The primary alignment permits a delay of at most five minutes, while the forecast horizon remains "
        "five minutes. Availability is proxied by the start of the recorded UTC minute."
    )
    (output_root / "captions.md").write_text(
        "# News–market timing figure\n\nSuggested placement: after Eq. `eq:ch3:news_market_windows` and "
        "the primary matching-count paragraph, before `Target construction and the maturity-clock distinction`.\n\n"
        + caption + "\n", encoding="utf-8")
    (output_root / "insertion_preview.tex").write_text(
        "% Standalone preview; the chapter is not edited or linked to this file.\n"
        "\\begin{figure}[htbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/additional_figures/timing_design/"
        + STEM + ".pdf}\n\\caption[News and market timing]{" + caption
        + "}\n\\label{fig:ch3:news_market_timing}\n\\end{figure}\n", encoding="utf-8")
    outputs = {}
    for name in (f"{STEM}.pdf", f"{STEM}.png", f"{STEM}_data.csv", "captions.md", "insertion_preview.tex"):
        p = output_root / name
        outputs[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    provenance = {
        "kind": "chapter3_news_market_timing_schematic_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(SOURCE), "sha256": _sha256(SOURCE),
                   "labels": ["eq:ch3:news_market_origin", "eq:ch3:news_market_windows"]},
        "panel_a": {"news_delay_minutes": 0},
        "panel_b": {"news_delay_minutes": 3, "illustrative_not_observed_sample": True},
        "time_units": "minutes_relative_to_UTC_forecast_origin",
        "availability_convention": "start_of_recorded_UTC_minute",
        "primary_maximum_alignment_delay_minutes": 5, "forecast_horizon_minutes": 5,
        "same_CME_trading_day_required": True, "market_closure_crossing_allowed": False,
        "figure_size_inches": [8, 6.6], "png_dpi": 600, "pdf_fonttype": 42,
        "minimum_font_size_pt": 11.5,
        "code_sha256": {str(Path(__file__)): _sha256(Path(__file__)),
                        str(Path(__file__).with_name("plot_chapter3_forecast_examples.py")):
                        _sha256(Path(__file__).with_name("plot_chapter3_forecast_examples.py"))},
        "outputs": outputs,
    }
    (output_root / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = render(args.output_root.expanduser())
    print(json.dumps({"output_root": str(args.output_root), "outputs": list(result["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
