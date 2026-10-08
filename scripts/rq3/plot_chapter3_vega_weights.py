"""Display the frozen current-window Vega weights used in Chapter 3.

Weights are normalised over the original pair-level joint support. This script
does not price options, train models, impute unsupported nodes or alter IVs.
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

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq3.chapter3_forecast_figure_data import (
    _raw_support_module,
    _validate_grids,
)
from scripts.rq3.plot_chapter3_forecast_examples import (
    BLUE, GRID, INK, MUTED, _save, _sha256, _style,
)

DEFAULT_EXPERIMENT_ROOT = ROOT / "outputs/experiments/rq12_direct_vega_mae_only_20261007_v2"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "PhdThesis/Chapter3/Chapter3Figs/additional_figures/vega_weights"
STEM = "ch3_vega_weight_distribution"
PAIR_ID = "pair_153aaac6a519efc67020"
FOLD = "f2_2023q2"


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def _signed_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _payload_hash({key: value for key, value in payload.items()
                      if key != "payload_sha256"}) != payload.get("payload_sha256"):
        raise ValueError(f"Source manifest payload hash mismatch: {path}")
    return payload


def _checked(path: Path, expected: str, role: str, sources: list[dict]) -> Path:
    path = path.resolve()
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Source hash mismatch for {role}: {path}")
    sources.append({"role": role, "path": str(path), "sha256": actual,
                    "bytes": path.stat().st_size})
    return path


def _array(value: str, dtype=float) -> np.ndarray:
    result = np.asarray(json.loads(value), dtype=dtype)
    if result.size != 256:
        raise ValueError("A frozen surface must contain 256 nodes.")
    return result.reshape(16, 16)


def load_example(experiment_root: Path) -> dict:
    """Bind the selected cache row to the frozen v2 experiment and raw masks."""
    experiment_root = experiment_root.resolve()
    sources: list[dict] = []
    rescore_path = experiment_root / "analysis/control_new_dual_rescore_manifest.json"
    rescore = _signed_json(rescore_path)
    sources.append({"role": "frozen_rescore_manifest", "path": str(rescore_path),
                    "sha256": _sha256(rescore_path), "bytes": rescore_path.stat().st_size})
    index_path = _checked(Path(rescore["input_hashes_path"]),
                          rescore["input_hashes_sha256"], "rescore_source_hash_index", sources)
    index = pd.read_csv(index_path)
    prediction_manifest = (experiment_root / "film/evaluation/predictions/tolerance_05m"
                           / FOLD / "seed_42/lp_matched.csv.manifest.json")
    records = index.loc[index.path.eq(str(prediction_manifest))]
    if len(records) != 1:
        raise ValueError("Frozen rescore index must pin the selected prediction manifest once.")
    _checked(prediction_manifest, str(records.iloc[0].sha256), "v2_prediction_manifest", sources)
    prediction = _signed_json(prediction_manifest)
    for key, expected in {"kind": "rq123_prediction_manifest_v1", "arm": "lp_matched",
                          "fold": FOLD, "seed": 42, "prediction_mc_samples": 64,
                          "cell_weight_mode": "black76_current_vega_v1"}.items():
        if prediction.get(key) != expected:
            raise ValueError(f"Frozen v2 prediction contract differs for {key}.")
    if prediction["cell_weight_manifest_sha256"] != rescore["cell_weight_manifest_sha256"]:
        raise ValueError("Prediction and completed rescore use different weight manifests.")
    panel_path = _checked(Path(prediction["panel_path"]), prediction["panel_sha256"],
                          "v2_test_pair_panel", sources)
    prediction_path = _checked(Path(prediction["prediction_path"]), prediction["prediction_sha256"],
                               "v2_prediction", sources)
    panel = pd.read_csv(panel_path)
    forecasts = pd.read_csv(prediction_path)
    if panel.pair_id.duplicated().any() or forecasts.pair_id.duplicated().any():
        raise ValueError("Panel and predictions must contain unique pair IDs.")
    pair_rows = panel.loc[panel.pair_id.eq(PAIR_ID)]
    forecast_rows = forecasts.loc[forecasts.pair_id.eq(PAIR_ID)]
    if len(pair_rows) != 1 or len(forecast_rows) != 1:
        raise ValueError("The requested existing illustration is absent or duplicated.")
    pair, forecast = pair_rows.iloc[0], forecast_rows.iloc[0]
    moneyness, maturities = _validate_grids(pair.strike_grid, pair.maturity_days_grid)
    origin = pd.Timestamp(pair.current_snapshot_time_utc)
    target = pd.Timestamp(pair.target_snapshot_time_utc)
    if origin != pd.Timestamp(pair.effective_origin_utc) or target - origin != pd.Timedelta(minutes=5):
        raise ValueError("Frozen origin/target five-minute alignment differs.")
    anchor = origin.isoformat().replace("+00:00", "Z")
    support_module = _raw_support_module()
    current_params = support_module.parse_raw_surface_params(pair.current_surface_param_json)
    current_mask = support_module.raw_support_mask(
        current_params, strike_grid=moneyness, maturity_days_grid=maturities)
    target_mask = support_module.raw_support_mask(
        support_module.parse_raw_surface_params(pair.target_surface_param_json),
        strike_grid=moneyness, maturity_days_grid=maturities)
    joint = current_mask & target_mask
    if not np.array_equal(joint, _array(forecast.support_mask_flat, bool)):
        raise ValueError("Frozen forecast joint mask differs from recomputed raw support.")
    manifest_path = _checked(Path(rescore["cell_weight_manifest_path"]),
                             rescore["cell_weight_manifest_sha256"], "current_vega_manifest", sources)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "schema_version": 1, "mode": "black76_current_vega_v1",
        "cell_weight_mode": "black76_current_vega_v1",
        "pricing_clock": "quote_expiry_minus_trade_utc_nanoseconds_ACT365",
        "support_policy": "current_raw_bracket_intersection_v1_outside_zero",
        "grid_interpolation": "linear_vega_in_percent_strike_then_linear_vega_in_business_days",
        "loss_normalization": "apply_existing_raw_joint_mask_then_normalize_vega_sum_per_sample",
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Current-window Vega metadata mismatch: {key}")
    if manifest["qc"]["status"] != "pass" or not manifest["qc"]["current_only_quote_provenance"]:
        raise ValueError("Current-window Vega cache provenance QA did not pass.")
    for field in ("input_workbook", "precalib_csv", "raw_source_manifest"):
        _checked(Path(manifest[field]), manifest[field + "_sha256"], field, sources)
    for filename, expected in manifest["audit_sha256"].items():
        _checked(manifest_path.parent / filename, expected, "vega_audit:" + filename, sources)
    _checked(manifest_path.parent / "qc_summary.json", manifest["qc_summary_sha256"], "vega_cache_qa", sources)
    cache_path = _checked(manifest_path.parent / manifest["weights_path"],
                          manifest["weights_sha256"], "frozen_current_vega_cache", sources)
    with np.load(cache_path, allow_pickle=False) as cache:
        anchors = cache["current_snapshot_time_utc"].astype(str).tolist()
        if len(anchors) != len(set(anchors)) or anchors != sorted(anchors):
            raise ValueError("Current Vega cache anchors are duplicated or unordered.")
        indices = [index for index, value in enumerate(anchors) if value == anchor]
        if len(indices) != 1:
            raise ValueError("The current origin is absent from the frozen Vega cache.")
        cache_index = indices[0]
        if not np.array_equal(cache["strike_grid"], moneyness) or not np.array_equal(cache["maturity_grid_days"], maturities):
            raise ValueError("Frozen pair/cache grids differ.")
        raw = cache["vega_weights"][cache_index].copy()
        cached_current_mask = cache["current_strict_support"][cache_index]
        fingerprint = str(cache["cell_weight_fingerprints"][cache_index])
    if raw.dtype != np.float32 or not np.array_equal(cached_current_mask, current_mask):
        raise ValueError("Current Vega cache dtype/support differs from the frozen raw pair.")
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw[~current_mask] != 0) or np.any(raw[current_mask] <= 0):
        raise ValueError("Current Vega must be finite, positive on support, and zero elsewhere.")
    grid_hash = _payload_hash({"strike_grid": moneyness.tolist(), "maturity_grid_days": maturities.tolist()})
    digest = hashlib.sha256()
    digest.update(anchor.encode())
    digest.update(grid_hash.encode())
    digest.update(np.asarray(raw, dtype="<f4", order="C").tobytes())
    if grid_hash != manifest["grid_fingerprint"] or digest.hexdigest() != fingerprint:
        raise ValueError("Frozen grid or current-Vega cell fingerprint differs.")
    if str(forecast.cell_weight_fingerprint) != fingerprint:
        raise ValueError("Completed v2 prediction used another Vega cache row.")
    audit = pd.read_csv(manifest_path.parent / "anchor_audit.csv")
    anchor_rows = audit.loc[audit.current_snapshot_time_utc.eq(anchor)]
    if len(anchor_rows) != 1:
        raise ValueError("Current origin must have one Vega anchor audit.")
    anchor_audit = anchor_rows.iloc[0]
    if (anchor_audit.cell_weight_fingerprint != fingerprint
            or anchor_audit.canonical_params_sha256 != _payload_hash(current_params)
            or int(anchor_audit.current_strict_support_cells) != int(current_mask.sum())):
        raise ValueError("Vega anchor audit differs from selected current raw surface.")
    quotes = pd.read_csv(manifest_path.parent / "quote_restoration_audit.csv")
    quotes = quotes.loc[quotes.current_snapshot_time_utc.eq(anchor)]
    quote_times = pd.to_datetime(quotes.trade_datetime_utc, utc=True)
    if (len(quotes) != int(anchor_audit.accepted_quote_rows)
            or not quote_times.between(origin - pd.Timedelta(minutes=5), origin, inclusive="left").all()
            or int(quotes.multiplicity.sum()) != int(anchor_audit.restored_quote_occurrences)):
        raise ValueError("Selected quote audit violates the accepted current window.")
    raw = raw.astype(np.float64)
    normalised = np.where(joint, raw, 0.0)
    denominator = float(normalised.sum())
    if denominator <= 0:
        raise ValueError("Joint support has no positive Vega mass.")
    normalised /= denominator
    if not np.isclose(normalised.sum(), 1.0, rtol=0, atol=1e-15) or np.any(normalised[~joint] != 0):
        raise ValueError("Whole-pair Vega normalisation failed.")
    q21 = np.flatnonzero(maturities == 21)
    slice_row = int(q21[0]) if len(q21) and joint[q21[0]].any() else int(np.flatnonzero(joint.any(axis=1))[0])
    return {
        "origin": origin, "target": target, "moneyness": moneyness,
        "maturities": maturities, "raw": raw, "joint": joint,
        "current_mask": current_mask, "target_mask": target_mask,
        "normalised": normalised, "denominator": denominator, "slice_row": slice_row,
        "cache_index": cache_index, "fingerprint": fingerprint, "sources": sources,
        "source_metadata": expected_metadata,
        "anchor_audit": {key: value.item() if isinstance(value, np.generic) else value
                         for key, value in anchor_audit.items()},
    }


def _edges(values: np.ndarray) -> np.ndarray:
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("Heatmap coordinates require at least two increasing physical nodes.")
    return np.concatenate(([values[0] - (values[1] - values[0]) / 2],
                           (values[:-1] + values[1:]) / 2,
                           [values[-1] + (values[-1] - values[-2]) / 2]))


def render(experiment_root: Path, output_root: Path) -> dict:
    example = load_example(experiment_root)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    m, q, joint = example["moneyness"], example["maturities"], example["joint"]
    weights = example["normalised"]
    rows, columns = np.flatnonzero(joint.any(axis=1)), np.flatnonzero(joint.any(axis=0))
    row_indices = np.arange(rows[0], rows[-1] + 1)
    column_indices = np.arange(columns[0], columns[-1] + 1)
    displayed = np.ma.masked_where(~joint[np.ix_(row_indices, column_indices)],
                                  100 * weights[np.ix_(row_indices, column_indices)])
    x_edges, y_edges = _edges(m[column_indices]), _edges(q[row_indices])
    _style()
    plt.rcParams.update({"xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    fig = plt.figure(figsize=(8, 4.8))
    fig.suptitle("Current Vega weights", y=0.97, fontsize=15)
    fig.text(0.5, 0.905, "Forecast origin: " + example["origin"].strftime("%d %B %Y, %H:%M UTC"),
             ha="center", fontsize=12, color=MUTED)
    map_axis = fig.add_axes((0.095, 0.20, 0.34, 0.57))
    slice_axis = fig.add_axes((0.675, 0.20, 0.29, 0.57))
    colour_axis = fig.add_axes((0.46, 0.20, 0.017, 0.57))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("white")
    maximum = float(100 * weights[joint].max())
    colour_maximum = float(np.ceil(maximum * 2) / 2)
    heatmap = map_axis.pcolormesh(x_edges, y_edges, displayed, cmap=cmap,
                                  norm=Normalize(0, colour_maximum), shading="flat",
                                  edgecolors="none", rasterized=False)
    map_axis.axvline(1.0, color="white", linewidth=1.4, linestyle=(0, (3, 3)))
    map_axis.axvline(1.0, color=INK, linewidth=0.8, linestyle=(0, (3, 3)))
    map_axis.set_title("(a) Joint-support weight map", loc="left", pad=13)
    map_axis.set_xlabel("Moneyness, $K/F$", labelpad=9)
    map_axis.set_ylabel("Maturity (business days)", labelpad=8)
    map_axis.set_xlim(x_edges[0], x_edges[-1])
    map_axis.set_ylim(y_edges[0], y_edges[-1])
    map_axis.set_xticks([x_edges[0], 1.0, x_edges[-1]], [f"{x:.3f}" for x in [x_edges[0], 1.0, x_edges[-1]]])
    map_axis.set_yticks([6, 10, 14, 17, 21])
    map_axis.tick_params(labelsize=11.5)
    for spine in map_axis.spines.values():
        spine.set_color(GRID)
    colourbar = fig.colorbar(heatmap, cax=colour_axis)
    colourbar.set_label("Normalised Vega weight (%)", labelpad=9, fontsize=12)
    colourbar.ax.tick_params(labelsize=11.5)
    colourbar.outline.set_linewidth(0.5)
    colourbar.solids.set_rasterized(False)
    slice_row = example["slice_row"]
    slice_mask = joint[slice_row]
    slice_axis.plot(m, np.where(slice_mask, 100 * weights[slice_row], np.nan),
                    color=BLUE, linewidth=1.4, marker="o", markersize=5,
                    markeredgewidth=0)
    slice_axis.axvline(1.0, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.8)
    slice_axis.set_title(f"(b) {q[slice_row]:g}-business-day slice", loc="left", pad=13)
    slice_axis.set_xlabel("Moneyness, $K/F$", labelpad=9)
    slice_axis.set_ylabel("Weight (%)", labelpad=8)
    slice_axis.set_xlim(x_edges[0], x_edges[-1])
    slice_axis.set_ylim(0, 1.08 * maximum)
    slice_axis.set_xticks([x_edges[0], 1.0, x_edges[-1]], [f"{x:.3f}" for x in [x_edges[0], 1.0, x_edges[-1]]])
    slice_axis.set_yticks(np.linspace(0, colour_maximum, 4))
    slice_axis.tick_params(labelsize=11.5)
    slice_axis.spines[["top", "right"]].set_visible(False)
    slice_axis.spines[["left", "bottom"]].set_color(GRID)
    slice_axis.grid(axis="y", color=GRID, linewidth=0.6)
    slice_axis.set_axisbelow(True)
    fig.text(0.5, 0.048, "Weights sum to 100% over the full joint support; the slice retains these weights.",
             ha="center", fontsize=11.5, color=MUTED)
    _save(fig, output_root, STEM, "Current-window normalised Vega weights")
    records = []
    for row, maturity in enumerate(q):
        for col, strike in enumerate(m):
            records.append({
                "fold": FOLD, "pair_id": PAIR_ID,
                "current_snapshot_time_utc": example["origin"].isoformat(),
                "target_snapshot_time_utc": example["target"].isoformat(),
                "maturity_business_days": float(maturity), "moneyness": float(strike),
                "current_support": bool(example["current_mask"][row, col]),
                "target_support": bool(example["target_mask"][row, col]),
                "joint_support": bool(joint[row, col]),
                "raw_current_black76_vega": float(example["raw"][row, col]),
                "normalised_vega_weight": float(weights[row, col]),
                "normalised_vega_weight_percent": float(100 * weights[row, col]),
                "is_slice": bool(row == slice_row),
            })
    pd.DataFrame(records).to_csv(output_root / f"{STEM}_data.csv", index=False)
    caption = (
        "Current-window Vega weights for the forecast origin on "
        + example["origin"].strftime("%d %B %Y at %H:%M UTC") + ". "
        "Panel (a) displays the recorded Black--76 Vega matrix after restriction to the original joint support "
        "and normalisation by its within-surface sum. Vega is computed from accepted transactions in the preceding "
        "five-minute window, aggregated within the existing strike--maturity buckets and interpolated onto the IV grid. "
        "The vertical axis retains the non-uniform business-day maturity spacing; unsupported nodes are left blank. "
        f"Panel (b) shows the {q[slice_row]:g}-business-day slice using the same whole-surface normalisation. "
        "Colours and vertical values express each node's weight as a percentage of the total joint-support Vega; "
        r"the dashed line marks ATM ($K/F=1$)."
    )
    placement = ("Suggested insertion: as a supplementary figure in the appendix. "
                 "The main text may refer to this figure after Eq. (eq:ch3:vega_weighted_mae) "
                 "and the explanation of within-pair normalisation.")
    (output_root / "caption.md").write_text(caption + "\n\n" + placement + "\n", encoding="utf-8")
    (output_root / "insertion_preview.tex").write_text(
        "% Appendix insertion preview only; this file is not included in the thesis.\n"
        "% The main text may refer to this figure after Eq. (eq:ch3:vega_weighted_mae).\n"
        "\\begin{figure}[htbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{Chapter3/Chapter3Figs/additional_figures/vega_weights/"
        + STEM + ".pdf}\n\\caption{" + caption.replace("%", r"\%")
        + "}\n\\label{fig:ch3:vega_weight_distribution}\n\\end{figure}\n", encoding="utf-8")
    outputs = {}
    for name in (f"{STEM}.pdf", f"{STEM}.png", f"{STEM}_data.csv", "caption.md", "insertion_preview.tex"):
        path = output_root / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    provenance = {
        "schema_version": 1, "kind": "chapter3_current_vega_weight_figure_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": FOLD, "pair_id": PAIR_ID,
        "selection": "retain_existing_frozen_surface_illustration_pair",
        "current_snapshot_time_utc": example["origin"].isoformat(),
        "target_snapshot_time_utc": example["target"].isoformat(),
        "quote_window_start_utc": (example["origin"] - pd.Timedelta(minutes=5)).isoformat(),
        "quote_window_end_utc": example["origin"].isoformat(),
        "quote_window_end_exclusive": True,
        "cache_anchor_index": example["cache_index"],
        "cell_weight_fingerprint": example["fingerprint"],
        "weight_metadata": example["source_metadata"],
        "raw_current_vega_units": "option_price_units_per_unit_annualised_IV",
        "selected_anchor_audit": example["anchor_audit"],
        "normalisation": {
            "formula": "M_joint*V_current/sum(M_joint*V_current)",
            "joint_node_count": int(joint.sum()),
            "current_vega_denominator": example["denominator"],
            "normalised_weight_sum": float(weights.sum()),
            "slice_business_days": float(q[slice_row]),
            "slice_node_count": int(slice_mask.sum()),
            "slice_weight_sum": float(weights[slice_row].sum()),
            "slice_renormalised": False, "unsupported_normalised_weight": 0,
        },
        "grid": {"moneyness": m.tolist(), "maturity_business_days": q.tolist(),
                 "orientation": "rows_maturity_columns_moneyness",
                 "heatmap_moneyness_edges": x_edges.tolist(),
                 "heatmap_maturity_edges": y_edges.tolist(),
                 "heatmap_cell_area_used_in_normalisation": False},
        "rendering": {"figure_size_inches": [8, 4.8], "png_dpi": 600,
                      "minimum_tick_font_size_pt": 11.5, "font_family": "DejaVu Sans",
                      "colour_scale": "viridis", "weight_percent_multiplier": 100,
                      "heatmap_colour_limits_percent": [0, colour_maximum], "timezone": "UTC",
                      "pdf_fonttype": 42, "matplotlib_version": matplotlib.__version__,
                      "unsupported_display": "blank", "imputation_performed": False},
        "sources": example["sources"],
        "code_sha256": {
            str(path.resolve()): _sha256(path) for path in (
                Path(__file__), Path(__file__).with_name("chapter3_forecast_figure_data.py"),
                Path(__file__).with_name("plot_chapter3_forecast_examples.py"),
                ROOT / "src/film_wgan/support.py",
            )
        },
        "outputs": outputs, "training_or_inference_performed": False,
    }
    (output_root / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = render(args.experiment_root, args.output_root)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "pair_id": result["pair_id"],
                      "normalisation": result["normalisation"]}, indent=2))


if __name__ == "__main__":
    main()
