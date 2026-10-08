"""Verify and package the five additional Chapter 3 figures for review.

Use --regenerate to rerun the four local plotting scripts first. This command
never edits a manuscript, includes a preview, trains a model, or reruns tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = ROOT.parent / "PhdThesis"
DEFAULT_OUTPUT = THESIS_ROOT / "Chapter3/Chapter3Figs/additional_figures"
SCRIPTS = {
    "text_control_design": "plot_chapter3_text_control_design.py",
    "paired_effects": "plot_chapter3_paired_effects.py",
    "timing_design": "plot_chapter3_timing_design.py",
    "vega_weights": "plot_chapter3_vega_weights.py",
}
FIGURES = (
    ("text_control_design", "ch3_text_control_design", "fig:ch3:text_control_design",
     "Common-parent continuation and same-checkpoint text interventions",
     "After the common-parent design, before the training-diagnostics table."),
    ("paired_effects", "ch3_rq3_paired_effects", "fig:ch3:rq3_paired_effects",
     "Paired evidence on incremental text value",
     "After the same-checkpoint intervention discussion, before the event analysis."),
    ("timing_design", "ch3_news_market_timing", "fig:ch3:news_market_timing",
     "News availability and five-minute market transitions",
     "After the news-market window equation and matching-count paragraph, before target construction."),
    ("paired_effects", "ch3_rq12_quarterly_paired_effects", "fig:ch3:rq12_quarterly_paired_effects",
     "Quarterly paired forecast comparisons under direct training",
     "After the representation-results interpretation, before Incremental Text Value."),
    ("vega_weights", "ch3_vega_weight_distribution", "fig:ch3:vega_weight_distribution",
     "Current-window Vega weight distribution",
     "In the appendix, with a reference after the Vega-weighted MAE definition if retained."),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_files():
    active = set()

    def walk(path):
        path = path.resolve()
        if path in active:
            return
        active.add(path)
        content = "\n".join(re.sub(r"(?<!\\)%.*", "", line)
                            for line in path.read_text(encoding="utf-8").splitlines())
        for target in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", content):
            child = THESIS_ROOT / target
            walk(child if child.suffix else child.with_suffix(".tex"))

    walk(THESIS_ROOT / "thesis.tex")
    old_assets = {p for p in (THESIS_ROOT / "Chapter3/Chapter3Figs/forecast_examples").rglob("*") if p.is_file()}
    return {str(p): sha256(p) for p in sorted(active | old_assets)}, len(active)


def caption_text(block):
    match = re.search(r"\\caption(?:\[[^\]]*\])?\s*\{", block)
    if match is None:
        raise ValueError("Figure preview lacks a caption")
    start = match.end()
    depth = 1
    for index in range(start, len(block)):
        if block[index] in "{}" and block[index - 1] != "\\":
            depth += 1 if block[index] == "{" else -1
            if depth == 0:
                return " ".join(block[start:index].replace("{%\n", "{").lstrip("%\n").split())
    raise ValueError("Unclosed caption in figure preview")


def build(output_root, regenerate=False):
    output_root = Path(output_root).resolve()
    baseline, manuscript_count = protected_files()
    if regenerate:
        for group, script in SCRIPTS.items():
            subprocess.run([sys.executable, str(ROOT / "scripts/rq3" / script),
                            "--output-root", str(output_root / group)], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    blocks = {}
    for group in SCRIPTS:
        folder = output_root / group
        path = folder / "figure_provenance.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        entries = record["outputs"]
        if isinstance(entries, list):
            entries = {item["filename"]: item for item in entries}
        for name, item in entries.items():
            if sha256(folder / name) != item["sha256"]:
                raise ValueError(f"Output SHA256 mismatch: {group}/{name}")
        if "code_sha256" in record:
            for name, expected in record["code_sha256"].items():
                if sha256(name) != expected:
                    raise ValueError(f"Plotting-code SHA256 mismatch: {name}")
        if "script_path" in record and sha256(record["script_path"]) != record["script_sha256"]:
            raise ValueError(f"Plotting-script SHA256 mismatch: {group}")
        if "code" in record:
            code = record["code"]
            if (sha256(code["path"]) != code["sha256"] or
                    sha256(code["shared_rendering_helper_path"]) != code["shared_rendering_helper_sha256"]):
                raise ValueError("Text-control plotting-code SHA256 mismatch")
        manifests[group] = {"path": str(path), "sha256": sha256(path)}
        preview = (folder / "insertion_preview.tex").read_text(encoding="utf-8")
        for block in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", preview, re.DOTALL):
            label = re.search(r"\\label\{([^}]+)\}", block)
            if label is None or label.group(1) in blocks:
                raise ValueError("Missing or repeated figure-preview label")
            blocks[label.group(1)] = block
    if set(blocks) != {item[2] for item in FIGURES}:
        raise ValueError("The standalone previews must define exactly the five agreed figures")
    pdfs = []
    catalogue = []
    captions = ["# Additional Chapter 3 figures\n\nFive separately generated figures for review. The manuscript is unchanged.\n"]
    preview = ["% Independent insertion previews. This file is not included in thesis.tex.\n"
               "% Insert individual figures at their suggested locations.\n"]
    for number, (group, stem, label, title, placement) in enumerate(FIGURES, start=1):
        folder = output_root / group
        pdfs.append(folder / f"{stem}.pdf")
        catalogue.append({"number": number, "title": title, "figure_label": label,
                          "pdf": str(folder / f"{stem}.pdf"), "png": str(folder / f"{stem}.png"),
                          "suggested_placement": placement})
        captions.append(f"\n## {number}. {title}\n\nSuggested placement: {placement}\n\n"
                        + caption_text(blocks[label]) + "\n")
        preview.append("\n% Suggested placement: " + placement + "\n" + blocks[label] + "\n")
    (output_root / "captions.md").write_text("".join(captions), encoding="utf-8")
    (output_root / "insertion_preview.tex").write_text("".join(preview), encoding="utf-8")
    merger = shutil.which("pdfunite")
    if merger is None:
        raise RuntimeError("pdfunite is needed to assemble the review PDF")
    merged = output_root / "chapter3_additional_figures_review.pdf"
    subprocess.run([merger, *map(str, pdfs), str(merged)], check=True)
    for name, expected in baseline.items():
        if sha256(name) != expected:
            raise ValueError(f"A protected manuscript or previous figure changed: {name}")
    packaged_outputs = {name: {"sha256": sha256(output_root / name),
                              "bytes": (output_root / name).stat().st_size}
                        for name in ("captions.md", "insertion_preview.tex", merged.name)}
    record = {"kind": "chapter3_additional_figure_review_bundle_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "figure_count": len(FIGURES), "figures": catalogue,
              "component_manifests": manifests, "outputs": packaged_outputs,
              "preservation": {"active_manuscript_files": manuscript_count,
                               "all_protected_paths_unchanged": True, "sha256": baseline},
              "preview_included_in_manuscript": False,
              "bootstrap_or_training_recomputed": False,
              "bundle_script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))}}
    (output_root / "figure_catalog.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    result = build(args.output_root, regenerate=args.regenerate)
    print(json.dumps({"figure_count": result["figure_count"],
                      "review_pdf": str(args.output_root / "chapter3_additional_figures_review.pdf"),
                      "manuscript_unchanged": result["preservation"]["all_protected_paths_unchanged"]}))


if __name__ == "__main__":
    main()
