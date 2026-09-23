"""Recompute Chapter 3 robustness summaries from frozen pair errors.

This analysis does not train models or overwrite historical experiments. MAE
point estimates are original seed--fold cell means with equal cell weights;
contrasts are averages of the original cell log-MAE ratios. Bootstrap draws
only estimate uncertainty, using the existing shared-market bootstrap core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from scripts.rq123.shared_panel_bootstrap_core import (
    DEFAULT_ITERATIONS,
    DEFAULT_RNG_SEED,
    METHOD_VERSION,
    load_schedule,
    make_schedule,
    prepare_panel,
    run_bootstrap,
    save_schedule,
)


REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "outputs/experiments"
ARCHITECTURE = EXPERIMENTS / "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1"
ALIGNMENT = EXPERIMENTS / "rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1"
CAPACITY = EXPERIMENTS / "rq3_news_first_vol_film_nolp_capacity_seed_exact_ttm_v1"
DEFAULT_OUTPUT = REPO / "outputs/analysis/chapter3_robustness_equal_cell_v1"
KIND = "chapter3_robustness_equal_cell_v1"
EXPECTED_FOLDS = {"f1_2023q1": 110, "f2_2023q2": 112,
                  "f3_2023q3": 135, "f4_2023q4": 143}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, float_format="%.17g", compression=compression)


def load_pair_files(root: Path, inputs: list[dict[str, str]]) -> pd.DataFrame:
    files = sorted((root / "predictions/cells").glob("*.pair_metrics.csv"))
    if not files:
        raise ValueError(f"No frozen pair metrics: {root}")
    frames = []
    for path in files:
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    return pd.concat(frames, ignore_index=True)


def condition(model: str, text: str, tolerance: int | None = None) -> str:
    prefix = "" if tolerance is None else f"t{tolerance:02d}::"
    return f"{prefix}{model}::{text}"


def canonical(frame: pd.DataFrame, *, alignment: bool) -> pd.DataFrame:
    result = frame.copy()
    result["condition"] = [condition(row.model_id, row.text_condition,
        int(row.train_tolerance_minutes) if alignment else None)
        for row in result.itertuples(index=False)]
    result = result.rename(columns={"target_mae": "value"})
    return result[["condition", "seed", "fold", "pair_id", "session_id",
                   "value", "persistence_mae"]]


def point_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use equal seed--fold weights, never pooled rows or bootstrap averages."""
    panel = prepare_panel(frame)
    cells = panel.frame.groupby(["condition", "seed", "fold"], sort=True).agg(
        observed_mae=("value", "mean"), persistence_mae=("persistence_mae", "mean"),
        pair_count=("pair_id", "size"), session_count=("session_id", "nunique"),
    ).reset_index()
    means = cells.groupby("condition", sort=True).agg(
        observed_mean_mae=("observed_mae", "mean"),
        persistence_mae=("persistence_mae", "mean"),
        cell_count=("observed_mae", "size"),
    ).reset_index()
    means["arithmetic_improvement_vs_persistence_percent"] = 100 * (
        1 - means.observed_mean_mae / means.persistence_mae)
    means["seed_count"] = len(panel.seeds)
    means["fold_count"] = len(panel.folds)
    means["pair_count"] = frame[["fold", "pair_id"]].drop_duplicates().shape[0]
    means["session_count"] = frame[["fold", "session_id"]].drop_duplicates().shape[0]
    np.testing.assert_allclose(means.observed_mean_mae, panel.observed_means("equal_cell"),
                               rtol=1e-13, atol=1e-15)
    return means, cells


def contrast_recipes(legacy: pd.DataFrame, *, alignment: bool) -> list[dict[str, Any]]:
    """Preserve historical contrast identities and multiple-testing families."""
    mapping: dict[str, tuple[str, str]] = {}
    if alignment:
        legacy = legacy[~legacy.contrast_id.str.startswith("own_panel")]
        for tolerance in (5, 10, 15, 20, 30):
            film = condition("film_lp_matched", "matched", tolerance)
            pure = condition("pure_cnn_no_text", "zero", tolerance)
            mapping[f"common5_train{tolerance:02d}_film_vs_pure_cnn"] = (film, pure)
            mapping[f"common5_train{tolerance:02d}_film_matched_vs_zero"] = (
                film, condition("film_lp_matched", "zero", tolerance))
            if tolerance > 5:
                for model, text in (("film_lp_matched", "matched"),
                                    ("pure_cnn_no_text", "zero")):
                    mapping[f"common5_{model}_train{tolerance:02d}_vs_train05"] = (
                        condition(model, text, tolerance), condition(model, text, 5))
    else:
        alternatives = ("crossattn_unet_mask_coords_v1", "transformer_tokens_mask_coords_v1",
                        "stylemod_unet_mask_coords_v1")
        models = ["film_reference", *(f"formal:{name}" for name in alternatives),
                  "scaled:film_reference_scaled", "scaled:validation_point_leader_scaled"]
        for name in alternatives:
            for reference, text in (("film_reference", "matched"), ("pure_cnn_reference", "zero")):
                mapping[f"{name}_vs_{reference}"] = (
                    condition(f"formal:{name}", "matched"), condition(reference, text))
        mapping["scaled_point_leader_vs_scaled_film"] = (
            condition("scaled:validation_point_leader_scaled", "matched"),
            condition("scaled:film_reference_scaled", "matched"))
        for model in models:
            for intervention in ("zero", "shuffle"):
                mapping[f"{model}_matched_vs_{intervention}"] = (
                    condition(model, "matched"), condition(model, intervention))
    ids = set(legacy.contrast_id)
    if ids != set(mapping) or legacy.contrast_id.duplicated().any():
        raise ValueError(f"Historical recipe mismatch: {ids.symmetric_difference(mapping)}")
    recipes = []
    for row in legacy.itertuples(index=False):
        focal, reference = mapping[row.contrast_id]
        recipes.append({"contrast_id": row.contrast_id, "focal": focal, "reference": reference,
                        "family_id": row.family_id, "apply_holm": bool(row.apply_holm),
                        "scope": row.scope})
    return recipes


def apply_holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["holm_p"] = np.nan
    result["family_size"] = 0
    for _, group in result[result.apply_holm].groupby("family_id", sort=True):
        ordered = group.sort_values(["p_one", "contrast_id"], kind="stable")
        adjusted = np.minimum(1, np.maximum.accumulate(
            ordered.p_one.to_numpy() * np.arange(len(ordered), 0, -1)))
        result.loc[ordered.index, "holm_p"] = adjusted
        result.loc[ordered.index, "family_size"] = len(ordered)
    result["reported_p"] = result.holm_p.where(result.apply_holm, result.p_one)
    return result


def summarize_contrasts(result: Any, recipes: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray]:
    rows, draws = [], []
    for recipe in recipes:
        stats = result.contrast(recipe["focal"], recipe["reference"])
        draws.append(stats.pop("draws"))
        rows.append({**recipe, **stats,
                     "geometric_gain_percent": 100 * (1 - math.exp(stats["point"]))})
    frame = pd.DataFrame(rows)
    frame["statistic"] = pd.to_numeric(frame.statistic, errors="coerce")
    return apply_holm(frame), np.column_stack(draws)


def verify_inputs(inputs: list[dict[str, str]]) -> None:
    for record in inputs:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {record['path']}")


def verify(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["kind"] != KIND:
        raise ValueError("Unexpected analysis kind")
    verify_inputs(manifest["inputs"])
    verify_inputs(manifest["implementation"])
    for record in manifest["outputs"]:
        if sha256(output / record["path"]) != record["sha256"]:
            raise ValueError(f"Output changed: {record['path']}")
    for job in manifest["jobs"]:
        panel = prepare_panel(pd.read_csv(output / job["panel"], float_precision="round_trip"))
        schedule = load_schedule(output / job["schedule"])
        result = run_bootstrap(panel, schedule, estimand="equal_cell")
        contrasts, log_draws = summarize_contrasts(result, job["recipes"])
        saved = pd.read_csv(output / job["contrasts"], float_precision="round_trip")
        pd.testing.assert_frame_equal(contrasts.loc[:, saved.columns], saved, check_dtype=False, check_exact=False,
                                      rtol=1e-13, atol=1e-15)
        with np.load(output / job["draws"], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["mean_draws"], result.mean_draws)
            np.testing.assert_array_equal(archive["log_ratio_draws"], log_draws)
            np.testing.assert_array_equal(archive["observed_mean_mae"], result.observed_means)
    return {"passed": True, "replayed_jobs": len(manifest["jobs"]),
            "iterations": manifest["iterations"], "inputs_unchanged": True}


def run(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for source in (ARCHITECTURE, ALIGNMENT, CAPACITY):
        if output == source.resolve() or source.resolve() in output.parents:
            raise ValueError("Output must not be inside a historical experiment")
    inputs: list[dict[str, str]] = []
    architecture = load_pair_files(ARCHITECTURE, inputs)
    alignment = load_pair_files(ALIGNMENT, inputs)
    frames = {"architecture": canonical(architecture[architecture.panel_role.eq("common_5m_primary")], alignment=False),
              "alignment": canonical(alignment[alignment.panel_role.eq("common_5m_primary")], alignment=True)}
    panels = {name: prepare_panel(frame) for name, frame in frames.items()}
    for panel in panels.values():
        first = panel.frame[panel.frame.condition.eq(panel.conditions[0]) & panel.frame.seed.eq(panel.seeds[0])]
        if first.groupby("fold").pair_id.nunique().to_dict() != EXPECTED_FOLDS:
            raise ValueError("Four-fold common-panel counts changed")
        if panel.seeds != (42, 202, 404):
            raise ValueError("Three-seed robustness universe changed")
    if panels["architecture"].market_fingerprint != panels["alignment"].market_fingerprint:
        raise ValueError("Architecture and alignment must share the market panel")
    schedule = make_schedule(panels["architecture"], iterations=DEFAULT_ITERATIONS, rng_seed=DEFAULT_RNG_SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.attempt-", dir=output.parent))
    save_schedule(schedule, staging / "shared_schedule.npz")
    jobs = []
    for name, root in (("architecture", ARCHITECTURE), ("alignment", ALIGNMENT)):
        legacy_path = root / "analysis/bootstrap_results.csv"
        inputs.append({"path": str(legacy_path), "sha256": sha256(legacy_path)})
        recipes = contrast_recipes(pd.read_csv(legacy_path), alignment=name == "alignment")
        result = run_bootstrap(panels[name], schedule, estimand="equal_cell")
        means, cells = point_summary(frames[name])
        means["bootstrap_mae_se"] = result.mean_draws.std(axis=0, ddof=1)
        means["mae_ci_lower"], means["mae_ci_upper"] = np.quantile(result.mean_draws, (.025, .975), axis=0)
        means["bootstrap_mean_mae_audit_only"] = result.mean_draws.mean(axis=0)
        contrasts, log_draws = summarize_contrasts(result, recipes)
        csv_write(staging / f"{name}_panel.csv.gz", frames[name])
        csv_write(staging / f"{name}_arm_summary.csv", means)
        csv_write(staging / f"{name}_cell_summary.csv", cells)
        csv_write(staging / f"{name}_contrasts.csv", contrasts)
        np.savez_compressed(staging / f"{name}_draws.npz", conditions=np.asarray(result.conditions),
                            observed_mean_mae=result.observed_means, mean_draws=result.mean_draws,
                            contrast_ids=np.asarray(contrasts.contrast_id), log_ratio_draws=log_draws)
        jobs.append({"name": name, "panel": f"{name}_panel.csv.gz", "schedule": "shared_schedule.npz",
                     "contrasts": f"{name}_contrasts.csv", "draws": f"{name}_draws.npz", "recipes": recipes})
        print(f"Completed {name}: {len(result.conditions)} arms, {len(recipes)} contrasts", flush=True)
    own_summaries = []
    for tolerance in (5, 10, 15, 20, 30):
        role = "common_5m_primary" if tolerance == 5 else "own_tolerance_secondary"
        subset = alignment[alignment.train_tolerance_minutes.eq(tolerance) & alignment.panel_role.eq(role)]
        summary, _ = point_summary(canonical(subset, alignment=True))
        summary["tolerance_minutes"] = tolerance
        summary["panel_role"] = "own_tolerance_secondary"
        summary["five_minute_reuses_common_panel"] = tolerance == 5
        own_summaries.append(summary)
    csv_write(staging / "alignment_own_panel_arm_summary.csv", pd.concat(own_summaries, ignore_index=True))
    capacity_rows = []
    for quarter, expected_pairs in (("q3", 135), ("q4", 143)):
        path = CAPACITY / f"analysis/film_nolp_capacity_{quarter}_pair_metrics.csv.gz"
        inputs.append({"path": str(path), "sha256": sha256(path)})
        frame = pd.read_csv(path, float_precision="round_trip")
        frame = frame[frame.tolerance_minutes.eq(5) & frame.panel.eq("core") & frame.stratum_type.eq("overall")]
        if frame.duplicated(["capacity_profile", "seed", "pair_id"]).any():
            raise ValueError("Capacity diagnostic contains duplicate pair rows")
        for profile, group in frame.groupby("capacity_profile", sort=True):
            counts = group.groupby("seed").pair_id.nunique()
            if list(counts.index) != [42, 202, 404] or not counts.eq(expected_pairs).all():
                raise ValueError("Historical capacity seed/pair panel changed")
            cells = group.groupby("seed")[["model_mae", "persistence_mae"]].mean().mean()
            capacity_rows.append({"quarter": f"2023{quarter.upper()}", "capacity_profile": profile,
                "observed_mean_mae": float(cells.model_mae), "persistence_mae": float(cells.persistence_mae),
                "seed_count": 3, "pair_count": expected_pairs, "fold_count": 1,
                "scope": "historical_development_diagnostic_not_main_four_fold_result"})
    csv_write(staging / "capacity_historical_summary.csv", pd.DataFrame(capacity_rows))
    verify_inputs(inputs)
    outputs = [{"path": str(path.relative_to(staging)), "sha256": sha256(path)}
               for path in sorted(staging.iterdir()) if path.is_file()]
    implementation = [{"path": str(path), "sha256": sha256(path)} for path in
        (Path(__file__).resolve(), REPO / "scripts/rq123/shared_panel_bootstrap_core.py")]
    json_write(staging / "manifest.json", {"kind": KIND, "method_version": METHOD_VERSION,
        "iterations": DEFAULT_ITERATIONS, "rng_seed": str(DEFAULT_RNG_SEED),
        "point_mae": "arithmetic mean of original pair MAE within each seed-fold, then equal seed-fold mean",
        "contrast": "equal seed-fold mean of log(candidate cell MAE/reference cell MAE)",
        "bootstrap_role": "uncertainty_only; mean-of-draws is audit-only",
        "bootstrap_sharing": "one seed draw and common fold-occurrence/session schedule across all models and seeds",
        "p_interpretation": "approximate uncentered plus-one one-sided bootstrap sign-tail probability",
        "holm": "same identities, families and adjustment scope as historical contrast definitions",
        "own_panel_inference": "not recomputed; own-tolerance panels report descriptive original MAE only",
        "common_fold_pair_counts": EXPECTED_FOLDS, "inputs": inputs, "implementation": implementation,
        "outputs": outputs, "jobs": jobs})
    verification = verify(staging)
    json_write(staging / "qa.json", verification)
    if output.exists():
        raise FileExistsError(f"Output appeared during calculation: {output}")
    os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output_root.resolve()), sort_keys=True))
    else:
        print(run(args.output_root))


if __name__ == "__main__":
    main()
