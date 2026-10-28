"""CPU dual scoring of frozen old/new predictions; no fitting or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts._path_setup  # noqa: E402,F401

import numpy as np
import pandas as pd

from film_wgan.support import parse_raw_surface_params, raw_support_mask
from wgan_option.utils.vega_weights import (
    CELL_WEIGHT_MODE, load_vega_weight_grid, sha256_file, utc_anchor,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_ROOT = ROOT / "outputs/experiments/rq12_direct_vega_20261007_v1"
DEFAULT_OLD_FILM = ROOT / "outputs/experiments/rq12_news_first_vol_film_unet_text128_c32_nolp_direct_text_10seed_lr2p5e5_exact_ttm_rolling_v1"
DEFAULT_OLD_PURE = ROOT / "outputs/experiments/rq12_news_first_vol_cnn_unet_c32_nolp_pure_no_text_10seed_exact_ttm_rolling_v1"
METRICS = ("plain_mae", "vega_mae", "persistence_plain_mae", "persistence_vega_mae")
PAIR_KEY = ["architecture", "arm", "fold", "seed", "tolerance_minutes", "pair_id"]


def _payload_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


@lru_cache(maxsize=1024)
def _file_sha(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return sha256_file(path)


def checked_file(path: str | Path, expected_sha: str, expected_size: Any = None) -> Path:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    if not expected_sha or _file_sha(str(resolved), stat.st_mtime_ns, stat.st_size) != expected_sha:
        raise ValueError(f"Frozen input SHA drift: {resolved}")
    if expected_size not in (None, "") and stat.st_size != int(expected_size):
        raise ValueError(f"Frozen input size drift: {resolved}")
    return resolved


def _read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("payload_sha256"):
        unsigned = {k: v for k, v in value.items() if k != "payload_sha256"}
        if _payload_sha(unsigned) != value["payload_sha256"]:
            raise ValueError(f"Frozen prediction manifest payload drift: {path}")
    return value


def frozen_prediction_entries(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Check stored-file provenance directly, without mutable source-code pins."""
    registry_path = root / "registry/task_registry.json"
    registry = json.loads(registry_path.read_text())
    if not registry.get("predictions_frozen"):
        raise ValueError(f"Prediction registry is not frozen yet: {root}")
    manifest_path = checked_file(registry["prediction_manifest_path"], registry["prediction_manifest_sha256"])
    entries = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"job_id", "arm", "fold", "seed", "tolerance_minutes", "prediction_path", "prediction_sha256", "prediction_manifest_path", "prediction_manifest_sha256"}
    if not required.issubset(entries.columns) or entries.empty or entries["job_id"].duplicated().any():
        raise ValueError(f"Invalid frozen prediction index: {manifest_path}")
    if "expected_prediction_cells" in registry and len(entries) != int(registry["expected_prediction_cells"]):
        raise ValueError("Frozen prediction index does not contain every expected model job.")
    return entries, [{"role": "prediction_registry", "path": str(registry_path), "sha256": sha256_file(registry_path)}, {"role": "prediction_index", "path": str(manifest_path), "sha256": sha256_file(manifest_path)}]


def _array(value: Any, shape: tuple[int, int], *, mask: bool = False) -> np.ndarray:
    array = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64)
    if array.size != math.prod(shape) or np.any(~np.isfinite(array)):
        raise ValueError("Stored surface/mask does not match its finite frozen grid.")
    array = array.reshape(shape)
    if mask:
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Stored support mask is not binary.")
        return array.astype(bool)
    # Training and stored MC predictions use float32 IV; use that same input
    # representation, then CPU float64 reductions for the two scoring rules.
    return array.astype(np.float32).astype(np.float64)


def score_arrays(predicted: np.ndarray, target: np.ndarray, current: np.ndarray, support: np.ndarray, vegas: np.ndarray) -> dict[str, float]:
    arrays = [np.asarray(x) for x in (predicted, target, current, support, vegas)]
    if len({x.shape for x in arrays}) != 1 or any(np.any(~np.isfinite(x)) for x in arrays):
        raise ValueError("Scoring arrays must share a finite surface shape.")
    support = np.asarray(support, dtype=bool)
    if not np.any(support) or np.any(vegas < 0) or np.any(vegas[support] <= 0):
        raise ValueError("Scoring requires positive current Vega on nonempty raw_joint support.")
    weight = np.asarray(vegas, dtype=np.float64)[support]
    error = np.abs(np.asarray(predicted, dtype=np.float64)[support] - target[support])
    persistence = np.abs(np.asarray(current, dtype=np.float64)[support] - target[support])
    return {
        "plain_mae": float(error.mean()),
        "vega_mae": float(np.dot(error, weight) / weight.sum()),
        "persistence_plain_mae": float(persistence.mean()),
        "persistence_vega_mae": float(np.dot(persistence, weight) / weight.sum()),
    }


def _add_skills(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for prefix in ("plain", "vega"):
        baseline = result[f"persistence_{prefix}_mae"]
        result[f"{prefix}_skill_vs_persistence"] = np.where(baseline > 0, 1 - result[f"{prefix}_mae"] / baseline, np.nan)
        result[f"{prefix}_persistence_zero"] = baseline <= 0
    return result


@lru_cache(maxsize=16)
def load_panel(path: str, expected_sha: str, manifest_path: str, manifest_sha: str) -> dict[str, dict[str, Any]]:
    panel_path = checked_file(path, expected_sha)
    frame = pd.read_csv(panel_path, keep_default_na=False)
    if frame.empty or frame["pair_id"].duplicated().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Frozen scoring panel must contain one row per pair/sample.")
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        q = json.loads(row["strike_grid"])
        days = json.loads(row["maturity_days_grid"])
        shape = tuple(int(x) for x in json.loads(row["surface_shape"]))
        if shape != (len(days), len(q)) or row["surface_model"] != "raw":
            raise ValueError("Frozen panel is not on the requested raw IV grid.")
        current_support = raw_support_mask(parse_raw_surface_params(row["current_surface_param_json"]), strike_grid=q, maturity_days_grid=days)
        target_support = raw_support_mask(parse_raw_surface_params(row["target_surface_param_json"]), strike_grid=q, maturity_days_grid=days)
        support = current_support & target_support
        if not support.any():
            raise ValueError("Frozen test panel contains a pair excluded by original raw_joint support.")
        anchor = utc_anchor(row["current_snapshot_time_utc"])
        vegas, metadata = load_vega_weight_grid(manifest_path, anchor, q, days, manifest_sha)
        current, target = (_array(row[field], shape) for field in ("current_surface_flat", "target_surface_flat"))
        support_sha = _payload_sha({"shape": shape, "mask": support.astype(int).reshape(-1).tolist(), "strike_grid": q, "maturity_grid_days": days})
        truth_sha = _payload_sha({"pair_id": str(row["pair_id"]), "current_anchor": anchor, "target_anchor": utc_anchor(row["target_snapshot_time_utc"]), "current": current.reshape(-1).tolist(), "target": target.reshape(-1).tolist(), "support_sha256": support_sha, "cell_weight_fingerprint": metadata["cell_weight_fingerprint"]})
        result[str(row["pair_id"])] = {
            "sample_id": str(row["sample_id"]), "current_snapshot_time_utc": anchor,
            "target_snapshot_time_utc": utc_anchor(row["target_snapshot_time_utc"]),
            "session_id": str(row.get("session_id", "")), "shape": shape,
            "current": current, "target": target, "support": support,
            "vegas": vegas.astype(np.float64), "support_mask_fingerprint": support_sha,
            "pair_truth_fingerprint": truth_sha, **metadata,
        }
    return result


def verify_shared_panel(reference: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]]) -> None:
    if set(reference) != set(candidate):
        raise ValueError("Old/new stored panels have different pair universes.")
    for pair_id in reference:
        for field in ("pair_truth_fingerprint", "support_mask_fingerprint", "cell_weight_fingerprint"):
            if reference[pair_id][field] != candidate[pair_id][field]:
                raise ValueError(f"Old/new panels differ in current/truth/support/weights: {pair_id}, {field}")


def score_prediction_job(
    entry: Mapping[str, Any], *, version: str, architecture: str,
    weight_manifest: Path, weight_manifest_sha: str,
    shared_panels: dict[tuple[int, str], dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_path = checked_file(entry["prediction_path"], entry["prediction_sha256"], entry.get("size_bytes"))
    prediction_manifest = checked_file(entry["prediction_manifest_path"], entry["prediction_manifest_sha256"])
    provenance = _read_signed(prediction_manifest)
    for field in ("job_id", "arm", "fold", "seed", "tolerance_minutes"):
        if str(provenance[field]) != str(entry[field]):
            raise ValueError(f"Prediction index and per-job manifest disagree: {field}")
    if provenance["prediction_sha256"] != entry["prediction_sha256"]:
        raise ValueError("Prediction CSV hash disagrees with its frozen per-job manifest.")
    if Path(provenance["prediction_path"]).resolve() != prediction_path:
        raise ValueError("Prediction CSV path disagrees with its frozen manifest.")
    panel = load_panel(str(Path(provenance["panel_path"]).resolve()), provenance["panel_sha256"], str(weight_manifest), weight_manifest_sha)
    key = (int(entry["tolerance_minutes"]), str(entry["fold"]))
    if key in shared_panels:
        verify_shared_panel(shared_panels[key], panel)
    else:
        shared_panels[key] = panel
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    if len(predictions) != int(provenance["row_count"]) or predictions["pair_id"].duplicated().any() or set(predictions["pair_id"].astype(str)) != set(panel):
        raise ValueError("Stored predictions do not match the complete frozen pair panel.")
    scored: list[dict[str, Any]] = []
    for row in predictions.to_dict("records"):
        truth = panel[str(row["pair_id"])]
        if str(row["sample_id"]) != truth["sample_id"] or str(row["prediction_status"]) != "ok" or row.get("support_mask_mode") != "raw_joint":
            raise ValueError("Stored prediction sample/status/support policy differs from its panel.")
        for field in ("job_id", "arm", "fold", "seed", "tolerance_minutes"):
            if str(row[field]) != str(entry[field]):
                raise ValueError(f"Stored prediction row job provenance differs: {field}")
        if str(row.get("checkpoint_sha256", "")) != str(provenance.get("checkpoint_sha256", "")):
            raise ValueError("Stored prediction checkpoint identity differs from its manifest.")
        stored_support = _array(row["support_mask_flat"], truth["shape"], mask=True)
        if not np.array_equal(stored_support, truth["support"]):
            raise ValueError("Stored prediction raw_joint mask differs from recomputed panel support.")
        if version == "vega_retrained":
            if row.get("cell_weight_mode") != CELL_WEIGHT_MODE or row.get("cell_weight_fingerprint") != truth["cell_weight_fingerprint"] or row.get("cell_weight_manifest_sha256") != weight_manifest_sha:
                raise ValueError("Retrained prediction does not carry the shared current Vega identity.")
        predicted = _array(row["predicted_surface_flat"], truth["shape"])
        metrics = score_arrays(predicted, truth["target"], truth["current"], truth["support"], truth["vegas"])
        scored.append({
            "experiment_version": version, "architecture": architecture,
            "arm": str(entry["arm"]), "fold": str(entry["fold"]), "seed": int(entry["seed"]),
            "tolerance_minutes": int(entry["tolerance_minutes"]), "job_id": str(entry["job_id"]),
            "pair_id": str(row["pair_id"]), "sample_id": truth["sample_id"],
            "session_id": truth["session_id"], "current_snapshot_time_utc": truth["current_snapshot_time_utc"],
            "target_snapshot_time_utc": truth["target_snapshot_time_utc"],
            "supported_cell_count": int(truth["support"].sum()),
            "support_mask_fingerprint": truth["support_mask_fingerprint"],
            "cell_weight_fingerprint": truth["cell_weight_fingerprint"],
            "pair_truth_fingerprint": truth["pair_truth_fingerprint"],
            "cell_weight_manifest_sha256": weight_manifest_sha,
            "prediction_sha256": entry["prediction_sha256"], **metrics,
        })
    inputs = [
        {"role": "prediction", "path": str(prediction_path), "sha256": entry["prediction_sha256"]},
        {"role": "prediction_manifest", "path": str(prediction_manifest), "sha256": entry["prediction_manifest_sha256"]},
        {"role": "test_panel", "path": str(Path(provenance["panel_path"]).resolve()), "sha256": provenance["panel_sha256"]},
    ]
    return scored, inputs


def summarize_pair_scores(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Equal pairs within fold, equal folds within seed, equal seeds per arm."""
    group = ["experiment_version", "architecture", "arm", "tolerance_minutes"]
    fold = pairs.groupby(group + ["seed", "fold"], sort=True)[list(METRICS)].mean().reset_index()
    count = pairs.groupby(group + ["seed", "fold"], sort=True).size().reset_index(name="pair_count")
    fold = _add_skills(fold.merge(count, on=group + ["seed", "fold"], validate="one_to_one"))
    seed = fold.groupby(group + ["seed"], sort=True)[list(METRICS)].mean().reset_index()
    seed["fold_count"] = fold.groupby(group + ["seed"], sort=True).size().to_numpy()
    seed = _add_skills(seed)
    overall = seed.groupby(group, sort=True)[list(METRICS)].mean().reset_index()
    overall["seed_count"] = seed.groupby(group, sort=True).size().to_numpy()
    overall = _add_skills(overall)
    for metric in ("plain_mae", "vega_mae", "plain_skill_vs_persistence", "vega_skill_vs_persistence"):
        overall[f"{metric}_seed_std"] = seed.groupby(group, sort=True)[metric].std(ddof=1).to_numpy()
    return fold, seed, overall


def control_new_deltas(pairs: pd.DataFrame) -> pd.DataFrame:
    old = pairs[pairs["experiment_version"] == "unweighted_control"]
    new = pairs[pairs["experiment_version"] == "vega_retrained"]
    if old.empty or new.empty:
        return pd.DataFrame()
    if old[PAIR_KEY].duplicated().any() or new[PAIR_KEY].duplicated().any():
        raise ValueError("Control/new pair keys are not unique.")
    result = old.merge(new, on=PAIR_KEY, suffixes=("_control", "_new"), how="outer", indicator=True, validate="one_to_one")
    if not result["_merge"].eq("both").all():
        raise ValueError("Control/new job or pair coverage differs; no partial comparison is allowed.")
    for field in ("pair_truth_fingerprint", "support_mask_fingerprint", "cell_weight_fingerprint"):
        if not result[f"{field}_control"].eq(result[f"{field}_new"]).all():
            raise ValueError(f"Control/new paired comparison has different {field}.")
    output = result[PAIR_KEY].copy()
    for metric in METRICS + ("plain_skill_vs_persistence", "vega_skill_vs_persistence"):
        output[f"{metric}_control"] = result[f"{metric}_control"]
        output[f"{metric}_new"] = result[f"{metric}_new"]
        output[f"delta_{metric}_new_minus_control"] = result[f"{metric}_new"] - result[f"{metric}_control"]
    return output


def summary_control_new_deltas(summary: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Subtract matching summaries after computing skill at that summary level."""
    old = summary[summary["experiment_version"] == "unweighted_control"]
    new = summary[summary["experiment_version"] == "vega_retrained"]
    if old.empty or new.empty or old[list(keys)].duplicated().any() or new[list(keys)].duplicated().any():
        raise ValueError("Control/new summary keys are incomplete or duplicated.")
    result = old.merge(new, on=list(keys), suffixes=("_control", "_new"), how="outer", indicator=True, validate="one_to_one")
    if not result["_merge"].eq("both").all():
        raise ValueError("Control/new summary coverage differs.")
    output = result[list(keys)].copy()
    for metric in METRICS + ("plain_skill_vs_persistence", "vega_skill_vs_persistence"):
        output[f"{metric}_control"] = result[f"{metric}_control"]
        output[f"{metric}_new"] = result[f"{metric}_new"]
        output[f"delta_{metric}_new_minus_control"] = result[f"{metric}_new"] - result[f"{metric}_control"]
    return output


def rescore(
    main_root: str | Path = DEFAULT_MAIN_ROOT, *, controls_only: bool = False,
    old_film_root: str | Path = DEFAULT_OLD_FILM, old_pure_root: str | Path = DEFAULT_OLD_PURE,
    weight_manifest_path: str | Path | None = None, expected_weight_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    main = Path(main_root).resolve()
    manifest = Path(weight_manifest_path or main / "weights/manifest.json").resolve()
    weight_sha = sha256_file(manifest)
    if expected_weight_manifest_sha256 and weight_sha != expected_weight_manifest_sha256:
        raise ValueError("Rescoring weight manifest does not match the requested experiment pin.")
    weights_meta = json.loads(manifest.read_text())
    weights_path = checked_file(manifest.parent / weights_meta["weights_path"], weights_meta["weights_sha256"])
    inputs = [{"role": "current_weight_manifest", "path": str(manifest), "sha256": weight_sha}, {"role": "current_weight_grid", "path": str(weights_path), "sha256": weights_meta["weights_sha256"]}]
    roots = [("unweighted_control", "film", Path(old_film_root).resolve()), ("unweighted_control", "pure", Path(old_pure_root).resolve())]
    if not controls_only:
        roots.extend(("vega_retrained", architecture, main / architecture) for architecture in ("film", "pure"))
    shared_panels: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    all_pairs: list[dict[str, Any]] = []
    job_counts: dict[str, int] = {}
    for version, architecture, root in roots:
        entries, provenance = frozen_prediction_entries(root)
        inputs.extend(provenance)
        entries = entries[entries["tolerance_minutes"].astype(int) == 5]
        job_counts[f"{version}/{architecture}"] = len(entries)
        for index, entry in enumerate(entries.to_dict("records"), 1):
            pairs, provenance = score_prediction_job(entry, version=version, architecture=architecture, weight_manifest=manifest, weight_manifest_sha=weight_sha, shared_panels=shared_panels)
            all_pairs.extend(pairs)
            inputs.extend(provenance)
            if index == 1 or index % 20 == 0 or index == len(entries):
                print(f"CPU rescored frozen predictions {version}/{architecture}: {index}/{len(entries)}", flush=True)
    pairs = _add_skills(pd.DataFrame(all_pairs))
    fold, seed, overall = summarize_pair_scores(pairs)
    deltas = control_new_deltas(pairs)
    output = main / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    prefix = "controls_dual" if controls_only else "control_new_dual"
    files = {
        f"{prefix}_pair_metrics.csv.gz": pairs,
        f"{prefix}_seed_fold_arm_summary.csv": fold,
        f"{prefix}_seed_arm_summary.csv": seed,
        f"{prefix}_arm_summary.csv": overall,
    }
    if not deltas.empty:
        files[f"{prefix}_pair_deltas.csv.gz"] = deltas
        group = ["architecture", "arm", "tolerance_minutes"]
        files[f"{prefix}_seed_fold_deltas.csv"] = summary_control_new_deltas(fold, group + ["seed", "fold"])
        files[f"{prefix}_seed_deltas.csv"] = summary_control_new_deltas(seed, group + ["seed"])
        files[f"{prefix}_equal_fold_seed_deltas.csv"] = summary_control_new_deltas(overall, group)
    for filename, frame in files.items():
        frame.to_csv(output / filename, index=False)
    input_frame = pd.DataFrame(inputs).drop_duplicates(["role", "path", "sha256"]).sort_values(["role", "path"])
    input_path = output / f"{prefix}_input_hashes.csv"
    input_frame.to_csv(input_path, index=False)
    summary = {
        "schema_version": 1, "kind": "rq12_current_vega_frozen_prediction_rescore_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "controls_only": controls_only, "execution": "CPU_stored_predictions_only_no_training_or_inference",
        "cell_weight_manifest_path": str(manifest), "cell_weight_manifest_sha256": weight_sha,
        "job_counts": job_counts, "pair_metric_rows": len(pairs),
        "shared_test_panels": {f"{minutes}m/{fold_name}": len(panel) for (minutes, fold_name), panel in shared_panels.items()},
        "aggregation": "equal_pairs_within_fold_equal_folds_within_seed_equal_seeds_within_arm",
        "skill_definition": "1-minus-aggregated-model-MAE-divided-by-aggregated-persistence-MAE; zero persistence is undefined",
        "scoring": {"units": "decimal_IV", "surface_representation": "frozen_panel_and_prediction_float32", "reduction_dtype": "float64_CPU", "support": "frozen_panel_current_and_target_raw_strict_support_intersection", "plain_mae": "mean(abs(prediction-target))_on_raw_joint", "vega_mae": "sum(current_Vega*abs(prediction-target))_on_raw_joint/sum(current_Vega)_on_raw_joint", "persistence": "frozen_panel_current_surface", "deltas": "new_minus_control_at_each_reported_aggregation_level"},
        "input_hashes_path": str(input_path), "input_hashes_sha256": sha256_file(input_path),
        "rescore_source_path": str(Path(__file__).resolve()), "rescore_source_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": [{"path": str(output / filename), "sha256": sha256_file(output / filename)} for filename in files],
    }
    summary["payload_sha256"] = _payload_sha(summary)
    (output / f"{prefix}_rescore_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--old-film-root", type=Path, default=DEFAULT_OLD_FILM)
    parser.add_argument("--old-pure-root", type=Path, default=DEFAULT_OLD_PURE)
    parser.add_argument("--weight-manifest", type=Path)
    parser.add_argument("--weight-manifest-sha256")
    args = parser.parse_args(argv)
    rescore(args.main_root, controls_only=args.controls_only, old_film_root=args.old_film_root, old_pure_root=args.old_pure_root, weight_manifest_path=args.weight_manifest, expected_weight_manifest_sha256=args.weight_manifest_sha256)


def run_rescore(global_root: str | Path, *, controls_only: bool = False) -> dict[str, Any]:
    """Supervisor entry point using the frozen default control experiment roots."""
    return rescore(global_root, controls_only=controls_only)


if __name__ == "__main__":
    main()
