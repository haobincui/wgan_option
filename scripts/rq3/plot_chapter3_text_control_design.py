"""Render the Chapter 3 common-parent and same-checkpoint control schematic.

The current thesis text is the design authority. This script draws no model
results and leaves the chapter and existing forecast figures unchanged.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq3.plot_chapter3_forecast_examples import (
    BLUE, GRID, INK, MUTED, _save, _sha256, _style,
)

THESIS_ROOT = ROOT.parent / "PhdThesis"
DEFAULT_CHAPTER = THESIS_ROOT / "Chapter3/chapter3.tex"
DEFAULT_OUTPUT = THESIS_ROOT / "Chapter3/Chapter3Figs/additional_figures/text_control_design"
STEM = "ch3_text_control_design"
FIGURE_LABEL = "fig:ch3:text_control_design"


def _source(chapter: Path) -> tuple[dict, str]:
    text = chapter.read_text(encoding="utf-8")
    anchor = r"\label{subsec:ch3:rq3_incremental_text_value}"
    start = text.index(anchor)
    next_section = re.search(r"\\subsubsection\*?\{", text[start + len(anchor):])
    end = start + len(anchor) + next_section.start() if next_section else len(text)
    excerpt = text[start:end]
    normalised = " ".join(excerpt.split())
    requirements = {
        "six_continuation_branches": "Each frozen parent defines six equal-protocol continuation branches",
        "identity_graft": "reproduce the Pure-CNN parent output exactly",
        "independent_checkpoint_selection": "Each continuation branch applies the same validation checkpoint-selection rule independently",
        "descriptive_representations": "BoW and sentiment are retained as descriptive representation diagnostics",
        "shared_training_noise": "all six branches share the prescribed batch ordering and training-noise sequence",
        "fixed_inference_noise": "fixes each matched-LP checkpoint and its Monte Carlo noise and changes only the inference-time text input",
        "branch_shuffle_seed": "master seed of 20260905",
        "intervention_shuffle_seed": "master seed 20260906",
        "intervention_cross_day": "Donors must come from a different CME trading day",
    }
    checks = {key: phrase in normalised for key, phrase in requirements.items()}
    if not all(checks.values()):
        raise ValueError(f"Chapter design no longer matches the schematic: {checks}")
    return {
        "path": str(chapter.resolve()), "sha256": _sha256(chapter),
        "section_label": "subsec:ch3:rq3_incremental_text_value",
        "start_line": text[:start].count("\n") + 1,
        "end_line": text[:end].count("\n") + 1,
        "semantic_checks": checks,
    }, excerpt


def _nodes() -> list[dict]:
    def node(node_id, panel, role, label, x, y, width, height, detail):
        return dict(node_id=node_id, panel=panel, role=role, label=label,
                    x=x, y=y, width=width, height=height, semantic_detail=detail)

    nodes = [
        node("parent", "A", "parent", "Pure-CNN parent\nSurface-only\nValidation selected",
             0.16, 0.66, 0.25, 0.11, "Frozen validation-best Pure-CNN checkpoint for one seed--fold cell."),
        node("graft", "A", "graft", "Identity FiLM graft\nCopied Generator\nand Critic",
             0.445, 0.66, 0.245, 0.11,
             "Shared Generator and Critic weights copied exactly; zero-initialised FiLM reproduces the parent forecast."),
    ]
    for node_id, label, y, role, detail in (
        ("matched_branch", "FiLM-CNN\nMatched LP", 0.855, "primary",
         "Continuation training with correctly matched LP embeddings."),
        ("zero_branch", "FiLM-CNN\nZero text", 0.765, "primary",
         "Equal-protocol continuation training with a zero text vector."),
        ("shuffled_branch", "FiLM-CNN\nShuffled LP", 0.675, "primary",
         "Fixed one-to-one within-partition derangement; same-day donors allowed; master seed 20260905."),
        ("bow_branch", "FiLM-CNN\nBoW (descriptive)", 0.585, "descriptive",
         "Bag-of-words continuation retained as a descriptive representation diagnostic."),
        ("sentiment_branch", "FiLM-CNN\nSentiment (descriptive)", 0.495, "descriptive",
         "Sentiment continuation retained as a descriptive representation diagnostic."),
        ("pure_branch", "Pure-CNN continuation\nNo text", 0.395, "surface_control",
         "Additional Pure-CNN training starts directly from the frozen Pure-CNN parent."),
    ):
        nodes.append(node(node_id, "A", role, label, 0.815, y, 0.315, 0.07, detail))
    nodes.append(node(
        "matched_checkpoint", "B", "checkpoint",
        "Selected matched-LP\nFiLM-CNN checkpoint\nWeights + MC noise fixed",
        0.23, 0.125, 0.35, 0.14,
        "The validation-selected matched-LP continuation checkpoint is reused for all three inference conditions; no retraining.",
    ))
    for node_id, label, y, detail in (
        ("matched_input", "Matched LP\nOriginal input", 0.205,
         "Original correctly matched LP input under the fixed checkpoint and noise bank."),
        ("zero_input", "Zero text\nInference input", 0.125,
         "Only the inference-time text input is replaced by a zero vector."),
        ("wrong_input", "Wrong LP\nSeparate permutation", 0.045,
         "Separate test-partition derangement with cross-day donors, distinct from the branch-shuffle donors; master seed 20260906."),
    ):
        nodes.append(node(node_id, "B", "intervention", label,
                          0.815, y, 0.315, 0.07, detail))
    return nodes


def _edges(nodes: list[dict]) -> list[dict]:
    lookup = {node["node_id"]: node for node in nodes}
    records = []

    def edge(source, target, meaning, vertices):
        records.append(dict(source_node=source, target_node=target,
                            semantic_detail=meaning, vertices=vertices))

    parent, graft = lookup["parent"], lookup["graft"]
    edge("parent", "graft", "Copy the Generator and Critic and add an identity FiLM graft.",
         [(parent["x"] + parent["width"] / 2, parent["y"]),
          (graft["x"] - graft["width"] / 2, graft["y"])])
    for target in ("matched_branch", "zero_branch", "shuffled_branch", "bow_branch", "sentiment_branch"):
        node = lookup[target]
        edge("graft", target, "Same graft initialisation; independently train and select the validation-best checkpoint.",
             [(graft["x"] + graft["width"] / 2, graft["y"]),
              (0.62, graft["y"]), (0.62, node["y"]),
              (node["x"] - node["width"] / 2, node["y"])])
    pure = lookup["pure_branch"]
    edge("parent", "pure_branch", "Continue the Pure-CNN directly from the frozen parent, without a FiLM graft.",
         [(parent["x"], parent["y"] - parent["height"] / 2),
          (parent["x"], pure["y"]), (pure["x"] - pure["width"] / 2, pure["y"])])
    checkpoint = lookup["matched_checkpoint"]
    for target in ("matched_input", "zero_input", "wrong_input"):
        node = lookup[target]
        edge("matched_checkpoint", target, "Fixed model weights and Monte Carlo noise; vary inference text only.",
             [(checkpoint["x"] + checkpoint["width"] / 2, checkpoint["y"]),
              (0.62, checkpoint["y"]), (0.62, node["y"]),
              (node["x"] - node["width"] / 2, node["y"])])
    return records


def _draw(nodes: list[dict], edges: list[dict]) -> tuple[plt.Figure, dict]:
    _style()
    fig = plt.figure(figsize=(8, 8.4))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    texts = [
        axis.text(0.035, 0.975, "(a) Common-parent continuation", fontsize=13.5, va="top"),
        axis.text(0.5, 0.935, "Four folds × ten seeds; one parent per seed–fold cell",
                  fontsize=11.5, ha="center", color=MUTED),
        axis.text(0.5, 0.313,
                  "Equal continuation protocol; shared batch order and training noise\n"
                  "Validation checkpoint selected separately in every branch",
                  fontsize=11.5, ha="center", va="center", color=MUTED, linespacing=1.4),
        axis.text(0.035, 0.26, "(b) Same-checkpoint text interventions", fontsize=13.5, va="center"),
    ]
    axis.plot([0.035, 0.9725], [0.28, 0.28], color=GRID, linewidth=0.8)
    for edge in edges:
        route = PlotPath(edge["vertices"], [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(edge["vertices"]) - 1))
        axis.add_patch(FancyArrowPatch(path=route, arrowstyle="-|>", mutation_scale=11,
                                       linewidth=1, color=MUTED, zorder=1))
    node_artists = []
    for node in nodes:
        role = node["role"]
        descriptive = role == "descriptive"
        neutral = role in ("parent", "surface_control", "checkpoint")
        fill = "#F7F8F9" if descriptive else ("#F0F2F4" if neutral else "#EDF4FA")
        stroke = "#A1A8AE" if descriptive else (MUTED if neutral else BLUE)
        patch = Rectangle((node["x"] - node["width"] / 2, node["y"] - node["height"] / 2),
                          node["width"], node["height"], facecolor=fill, edgecolor=stroke,
                          linewidth=1, linestyle="--" if descriptive else "-", zorder=2)
        axis.add_patch(patch)
        text = axis.text(node["x"], node["y"], node["label"], fontsize=11.5,
                         ha="center", va="center", color=INK, linespacing=1.25, zorder=3)
        texts.append(text)
        node_artists.append((node["node_id"], text, patch))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = [text.get_text() for text in texts
               if not fig.bbox.contains(*text.get_window_extent(renderer).min)
               or not fig.bbox.contains(*text.get_window_extent(renderer).max)]
    overflow = [node_id for node_id, text, patch in node_artists
                if not patch.get_window_extent(renderer).contains(*text.get_window_extent(renderer).min)
                or not patch.get_window_extent(renderer).contains(*text.get_window_extent(renderer).max)]
    collisions = []
    for index, text in enumerate(texts):
        for other in texts[index + 1:]:
            if text.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)):
                collisions.append([text.get_text(), other.get_text()])
    if outside or overflow or collisions:
        raise ValueError(f"Layout failure: outside={outside}; node_overflow={overflow}; collisions={collisions}")
    return fig, {"text_inside_canvas": True, "node_text_inside_boxes": True,
                 "text_overlaps": 0, "minimum_source_font_points": 11.5,
                 "minimum_font_at_14_5cm_width_points": 11.5 * (14.5 / 2.54) / 8}


def _write_companions(output_root: Path) -> None:
    caption = (
        "Experimental controls for incremental text value. Panel (a) shows how a "
        "validation-selected Pure-CNN parent initialises six continuation branches. "
        "The FiLM graft initially reproduces the parent forecast. Matched LP, zero "
        "text and shuffled LP form the principal text comparisons; BoW and sentiment are "
        "descriptive. Each branch selects its own validation checkpoint under the "
        "shared continuation protocol. Panel (b) evaluates the selected matched-LP "
        "checkpoint with matched, zero and wrong text while model weights and Monte "
        "Carlo noise remain fixed. Wrong text uses a separate cross-day permutation "
        "from the continuation-branch shuffle."
    )
    (output_root / "captions.md").write_text(
        "# Common-parent and same-checkpoint text controls\n\n"
        "Suggested placement: at the end of the four-fold common-parent design "
        "in `Incremental Text Value`, after the paragraph introducing branch "
        "comparisons and same-checkpoint interventions and before the training "
        "diagnostics table. This is an insertion preview; chapter text is unchanged.\n\n"
        + caption + "\n", encoding="utf-8")
    relative = "Chapter3/Chapter3Figs/additional_figures/text_control_design/" + STEM + ".pdf"
    preview = (
        "% Standalone insertion preview; not included by the thesis.\n"
        "\\begin{figure}[htbp]\n"
        "\\captionsetup{font=footnotesize,labelfont=bf,textfont=normalfont}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{relative}}}\n"
        "\\caption[Common-parent and same-checkpoint text controls]{" + caption + "}\n"
        f"\\label{{{FIGURE_LABEL}}}\n"
        "\\end{figure}\n"
    )
    (output_root / "insertion_preview.tex").write_text(preview, encoding="utf-8")


def render(chapter: Path, output_root: Path) -> dict:
    source, _ = _source(chapter)
    nodes = _nodes()
    edges = _edges(nodes)
    fig, layout_checks = _draw(nodes, edges)
    output_root.mkdir(parents=True, exist_ok=True)
    _save(fig, output_root, STEM, "Common-parent and same-checkpoint text controls")
    with (output_root / f"{STEM}_nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nodes[0]))
        writer.writeheader()
        writer.writerows(nodes)
    with (output_root / f"{STEM}_edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_node", "target_node", "semantic_detail", "route_json"])
        writer.writeheader()
        writer.writerows({**{k: edge[k] for k in ("source_node", "target_node", "semantic_detail")},
                         "route_json": json.dumps(edge["vertices"])} for edge in edges)
    _write_companions(output_root)
    output_files = [output_root / f"{STEM}.{extension}" for extension in ("pdf", "png")]
    output_files += [output_root / f"{STEM}_{kind}.csv" for kind in ("nodes", "edges")]
    output_files += [output_root / filename for filename in ("captions.md", "insertion_preview.tex")]
    manifest = {
        "schema_version": 1,
        "artifact_kind": "experimental_design_schematic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)),
                 "shared_rendering_helper_path": str(ROOT / "scripts/rq3/plot_chapter3_forecast_examples.py"),
                 "shared_rendering_helper_sha256": _sha256(ROOT / "scripts/rq3/plot_chapter3_forecast_examples.py")},
        "design": {"continuation_branch_count": 6, "film_branch_count": 5,
                   "same_checkpoint_input_count": 3, "training_branches_select_checkpoints_independently": True,
                   "intervention_changes_text_only": True, "training_shuffle_master_seed": 20260905,
                   "intervention_shuffle_master_seed": 20260906,
                   "wrong_text_requires_cross_day_donors": True},
        "layout_checks": layout_checks,
        "rendering": {"matplotlib_version": matplotlib.__version__, "font_family": "DejaVu Sans",
                      "figure_size_inches": [8, 8.4], "png_dpi": 600, "pdf_fonttype": 42},
        "figure_label": FIGURE_LABEL,
        "outputs": [{"filename": path.name, "sha256": _sha256(path)} for path in output_files],
    }
    (output_root / "figure_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter-source", type=Path, default=DEFAULT_CHAPTER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(args.chapter_source, args.output_root)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "layout_checks": result["layout_checks"]}, indent=2))


if __name__ == "__main__":
    main()
