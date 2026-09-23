"""Replay frozen RQ1--RQ3 inference without importing a mutable source adapter.

The archived source pipeline's implementation drift is recorded, not ignored
or retrospectively repaired. This audit binds the frozen panels, recipes and
market schedules to raw prediction hashes, independently checked observed
point estimates, and the stable bootstrap core used for numerical replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from scripts.rq123 import shared_panel_bootstrap_core as core


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_point_audit_v3"
DEFAULT_POINTS = ROOT / "outputs/analysis/chapter3_observed_point_estimates_v1"
DEFAULT_OUTPUT = ROOT / "outputs/analysis/chapter3_main_inference_verified_v1"
KIND = "chapter3_main_inference_independent_replay_v1"
JOBS = tuple(f"direct_f{i}_2023q{i}" for i in range(1, 5)) + (
    "direct_overall", "rq3_full_absolute_mae", "rq3_branch",
    "rq3_intervention", "rq3_validation_epoch0_epoch30_best",
)
RAW_IDS = {"direct", "rq3_full", "rq3_branch", "rq3_intervention", "rq3_validation"}
KEYS = ["condition", "seed", "fold", "pair_id", "session_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _check_hashes(records: list[dict], base: Path | None = None) -> None:
    for item in records:
        path = base / item["path"] if base else _resolve(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen audit input changed: {path}")


def _holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["p_selected"] = np.where(result.alternative.eq("two_sided"), result.p_two, result.p_one)
    result["holm_p"] = np.nan
    result["family_size"] = 0
    for _, group in result[result.apply_holm].groupby("family_id", sort=True):
        group = group.sort_values(["p_selected", "job_id", "contrast_id"], kind="stable")
        adjusted = np.minimum(1.0, np.maximum.accumulate(
            group.p_selected.to_numpy() * np.arange(len(group), 0, -1)))
        result.loc[group.index, "holm_p"] = adjusted
        result.loc[group.index, "family_size"] = len(group)
    result["reported_p"] = result.holm_p.where(result.apply_holm, result.p_selected)
    result["significance_stars"] = [
        "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
        for p in result.reported_p
    ]
    return result


def _raw_id(job_id: str) -> str:
    if job_id.startswith("direct_"):
        return "direct"
    return {"rq3_full_absolute_mae": "rq3_full", "rq3_branch": "rq3_branch",
            "rq3_intervention": "rq3_intervention",
            "rq3_validation_epoch0_epoch30_best": "rq3_validation"}[job_id]


def _canonical_from_raw(raw: pd.DataFrame, source_id: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Independently bind archived pair values to the raw prediction table."""
    result = raw.copy()
    if source_id == "rq3_validation":
        result["condition"] = result.arm + "::" + result.checkpoint_label
    elif source_id == "rq3_intervention":
        result["condition"] = result.input_condition
    else:
        result["condition"] = result.arm.replace({"matched": "film_lp_matched"})
    result = result[result.fold.isin(panel.fold.unique())].copy()
    result = result[result.condition.isin(set(panel.condition) - {"persistence"})]
    result = result.rename(columns={"target_mae": "value"})[KEYS + ["value", "persistence_mae"]]
    if "persistence" in set(panel.condition):
        market_keys = ["seed", "fold", "pair_id", "session_id"]
        if result.groupby(market_keys).persistence_mae.nunique().gt(1).any():
            raise ValueError("Persistence differs between raw model arms")
        persistence = result.drop_duplicates(market_keys).copy()
        persistence["condition"] = "persistence"
        persistence["value"] = persistence.persistence_mae
        result = pd.concat([result, persistence], ignore_index=True)
    return result.sort_values(KEYS).reset_index(drop=True)


def _check_point_estimates(job_id: str, panel: pd.DataFrame, point_root: Path) -> list[dict]:
    """Check every observed cell, then fold and overall arithmetic averages."""
    if job_id == "rq3_validation_epoch0_epoch30_best":
        # This independent point-audit predates validation coverage. Validation
        # is instead bound pair by pair to its raw validation predictions.
        return []
    dataset = "rq12" if job_id.startswith("direct_") else "rq3_standard"
    aliases = {}
    if job_id == "rq3_intervention":
        dataset = "rq3_intervention"
        aliases = {"matched_input": "film_lp_matched", "zero_input": "film_lp_matched__zero_input",
                   "wrong_input": "film_lp_matched__wrong_input"}
    cells = panel.groupby(["condition", "seed", "fold"], as_index=False).value.mean()
    cells["arm"] = cells.condition.replace(aliases)
    records = []
    for filename, keys, actual in (
        ("cell_summary.csv", ["arm", "seed", "fold"], cells),
        ("fold_summary.csv", ["arm", "fold"], cells.groupby(["arm", "fold"], as_index=False).value.mean()),
        ("overall_summary.csv", ["arm"], cells.groupby("arm", as_index=False).value.mean()),
    ):
        if filename == "overall_summary.csv" and panel.fold.nunique() != 4:
            continue
        expected = _csv(point_root / filename)
        expected = expected[expected.dataset.eq(dataset)]
        joined = actual.merge(expected[keys + ["observed_mean_mae"]], on=keys,
                              how="left", validate="one_to_one")
        if joined.observed_mean_mae.isna().any():
            raise ValueError(f"Missing observed point-audit cells for {job_id}")
        np.testing.assert_allclose(joined.value, joined.observed_mean_mae, rtol=0, atol=1e-14)
        for row in joined.to_dict("records"):
            records.append({"job_id": job_id, "level": filename.removesuffix("_summary.csv"),
                            **{key: row[key] for key in keys}, "replayed_mean_mae": row["value"],
                            "point_audit_mean_mae": row["observed_mean_mae"]})
    return records


def _replay(root: Path, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_arms = _csv(root / "source_archive/all_arm_summary.csv")
    original_contrasts = _csv(root / "source_archive/all_contrasts.csv")
    raw = {record["source_id"]: _csv(_resolve(record["path"])) for record in manifest["raw_sources"]}
    arm_rows, contrast_rows, point_checks = [], [], []
    for job in manifest["jobs"]:
        job_id = job["job_id"]
        frame = _csv(root / job["panel_path"])
        raw_panel = _canonical_from_raw(raw[_raw_id(job_id)], _raw_id(job_id), frame)
        ordered = frame.sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(raw_panel[KEYS], ordered[KEYS], check_dtype=False)
        np.testing.assert_allclose(raw_panel[["value", "persistence_mae"]],
                                   ordered[["value", "persistence_mae"]], rtol=0, atol=1e-14)
        point_checks += _check_point_estimates(job_id, frame, root / "point_audit")
        panel = core.prepare_panel(frame)
        schedule = core.load_schedule(root / job["schedule_path"])
        if schedule.iterations != 10_000 or job["estimand"] != "equal_cell":
            raise ValueError("Main inference requires 10,000 equal-cell draws")
        result = core.run_bootstrap(panel, schedule, estimand="equal_cell")
        old_arms = original_arms[original_arms.job_id.eq(job_id)].set_index("condition")
        with np.load(root / job["draws_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["condition_names"], np.asarray(result.conditions))
            np.testing.assert_allclose(saved["mean_draws"], result.mean_draws, rtol=0, atol=1e-14)
            for index, condition in enumerate(result.conditions):
                row = old_arms.loc[condition].to_dict()
                mean = float(result.observed_means[index])
                np.testing.assert_allclose(row["observed_mean_mae"], mean, rtol=0, atol=1e-14)
                row.update(condition=condition, observed_mean_mae=mean)
                arm_rows.append(row)
            for index, recipe in enumerate(job["contrasts"]):
                selected = original_contrasts[original_contrasts.job_id.eq(job_id)
                    & original_contrasts.contrast_id.eq(recipe["contrast_id"])]
                if len(selected) != 1:
                    raise ValueError("Missing or duplicate archived contrast")
                row = selected.iloc[0].to_dict()
                for key in ("family_id", "focal", "reference"):
                    if row[key] != recipe[key]:
                        raise ValueError("Frozen contrast recipe disagrees with source summary")
                stats = result.contrast(recipe["focal"], recipe["reference"])
                np.testing.assert_allclose(saved["log_ratio_draws"][:, index], stats.pop("draws"),
                                           rtol=0, atol=1e-13)
                for key in ("point", "bootstrap_se", "ci_lower", "ci_upper", "p_one", "p_two"):
                    np.testing.assert_allclose(row[key], stats[key], rtol=0, atol=1e-12)
                row.update(stats)
                row["geometric_gain_percent"] = 100 * (1 - np.exp(stats["point"]))
                contrast_rows.append(row)
    contrasts = _holm(pd.DataFrame(contrast_rows))
    # All Holm families are closed within the selected main-analysis jobs.
    expected = original_contrasts.set_index(["job_id", "contrast_id"])
    for row in contrasts.itertuples(index=False):
        old = expected.loc[(row.job_id, row.contrast_id)]
        np.testing.assert_allclose(row.reported_p, old.reported_p, rtol=0, atol=1e-15)
        if int(row.family_size) != int(old.family_size):
            raise ValueError("Selected main jobs truncate an archived Holm family")
    return pd.DataFrame(arm_rows), contrasts, pd.DataFrame(point_checks)


def verify_audit(root: Path) -> dict:
    manifest = _read(root / "audit_manifest.json")
    if manifest["kind"] != KIND:
        raise ValueError("Not an independent main-inference audit")
    _check_hashes(manifest["implementation"])
    _check_hashes(manifest["raw_sources"])
    hashes = _csv(root / "output_hashes.csv").to_dict("records")
    _check_hashes(hashes, root)
    arms, contrasts, points = _replay(root, manifest)
    for name, frame, keys in (
        ("all_arm_summary.csv", arms, ["job_id", "condition"]),
        ("all_contrasts.csv", contrasts, ["job_id", "contrast_id"]),
        ("observed_point_checks.csv", points, ["job_id", "level", "arm", "seed", "fold"]),
    ):
        stored = _csv(root / "analysis" / name).fillna("")
        replayed = frame.fillna("")
        pd.testing.assert_frame_equal(stored.sort_values(keys).reset_index(drop=True),
                                      replayed.sort_values(keys).reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    return {"kind": KIND, "passed": True, "jobs": len(manifest["jobs"]),
            "contrasts": len(contrasts), "arm_summaries": len(arms), "point_checks": len(points),
            "raw_prediction_binding_verified": True, "source_pipeline_drift_disclosed": True}


def run_audit(source: Path, points: Path, output: Path) -> Path:
    source, points, output = source.resolve(), points.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_manifest = _read(source / "input_manifest.json")
    source_analysis = _read(source / "analysis_manifest.json")
    source_hashes = _csv(source / "output_hashes.csv")
    _check_hashes([{"path": row.relative_path, "sha256": row.sha256}
                   for row in source_hashes.itertuples(index=False)], source)
    raw_sources = [dict(row, path=str(_resolve(row["path"]))) for row in source_manifest["inputs"]
                   if row.get("source_id") in RAW_IDS and row.get("role") == "data"]
    if {row["source_id"] for row in raw_sources} != RAW_IDS:
        raise ValueError("Missing required raw prediction sources")
    _check_hashes(raw_sources)
    jobs = [job for job in source_analysis["jobs"] if job["job_id"] in JOBS]
    if {job["job_id"] for job in jobs} != set(JOBS) or len(jobs) != len(JOBS):
        raise ValueError("Incomplete or duplicate RQ1--RQ3 job coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.attempt-", dir=output.parent))
    for name in ("analysis", "source_archive", "point_audit", "implementation"):
        (staging / name).mkdir()
    source_records = []

    def copy(original: Path, relative: str) -> None:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        digest = sha256(original)
        if sha256(target) != digest:
            raise ValueError(f"Source changed during copy: {original}")
        source_records.append({"path": str(original), "sha256": digest, "archived_path": relative})

    for name in ("input_manifest.json", "analysis_manifest.json", "output_hashes.csv"):
        copy(source / name, f"source_archive/{name}")
    for name in ("all_arm_summary.csv", "all_contrasts.csv"):
        copy(source / "analysis" / name, f"source_archive/{name}")
    for path in sorted({job[field] for job in jobs for field in ("panel_path", "schedule_path", "draws_path")}):
        copy(source / path, path)
    for name in ("audit_manifest.json", "cell_summary.csv", "fold_summary.csv", "overall_summary.csv"):
        copy(points / name, f"point_audit/{name}")
    _check_hashes(_read(points / "audit_manifest.json")["sources"])
    implementation = []
    for path in (Path(__file__).resolve(), Path(core.__file__).resolve()):
        copy(path, f"implementation/{path.name}")
        implementation.append({"path": str(path), "sha256": sha256(path)})
    drift = []
    for record in source_manifest["implementation"]:
        path = _resolve(record["path"])
        current = sha256(path) if path.is_file() else None
        drift.append({"path": str(path), "source_recorded_sha256": record["sha256"],
                      "current_sha256_at_audit": current, "changed_since_source_run": current != record["sha256"]})
        if path.name == Path(core.__file__).name and current != record["sha256"]:
            raise ValueError("The bootstrap core changed; this is not the same numerical replay")
    manifest = {"kind": KIND, "schema_version": 1, "source_root": str(source),
                "source_pipeline_implementation_drift": drift,
                "drift_policy": "Source driver/adapter drift is disclosed; replay uses frozen canonical panels, recipes and schedules, independently bound to raw prediction values and observed point estimates.",
                "raw_sources": raw_sources, "source_file_hashes": source_records,
                "implementation": implementation, "jobs": jobs, "iterations": 10000,
                "point_estimand": "observed_equal_seed_fold_cell_mean; paired mean cell log ratio",
                "uncertainty": "shared_market_panel_crossed_seed_percentile_and_plus_one_sign_tail",
                "scope": "RQ1, RQ2, RQ3 and RQ3 validation; excludes RQ4 and robustness",
                "validation_point_check": "Raw validation pair values verified; not covered by the earlier observed-point audit."}
    _json(staging / "audit_manifest.json", manifest)
    arms, contrasts, point_checks = _replay(staging, manifest)
    _save_csv(staging / "analysis/all_arm_summary.csv", arms)
    _save_csv(staging / "analysis/all_contrasts.csv", contrasts)
    _save_csv(staging / "analysis/observed_point_checks.csv", point_checks)
    _json(staging / "qa.json", {"passed": True, "jobs": len(jobs), "contrasts": len(contrasts),
          "arm_summaries": len(arms), "observed_point_checks": len(point_checks),
          "raw_prediction_binding_verified": True, "source_pipeline_drift_disclosed": True})
    records = [{"path": str(path.relative_to(staging)), "sha256": sha256(path)}
               for path in sorted(staging.rglob("*")) if path.is_file()]
    _save_csv(staging / "output_hashes.csv", pd.DataFrame(records))
    verify_audit(staging)
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--point-root", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", "--verify", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_audit(args.output_root.resolve()), sort_keys=True))
    else:
        print(run_audit(args.source_root, args.point_root, args.output_root))


if __name__ == "__main__":
    main()
