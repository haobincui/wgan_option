"""Paired Vega-MAE robustness analysis of frozen direct RQ1/RQ2 predictions.

The five same-model retrained/control contrasts reuse the Chapter 3 archived
shared-market schedule.  Ordinary MAE is descriptive.  This script performs
CPU analysis only and never changes the predictions, rescore manifest, or
bootstrap archive from which its inputs are read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rq123 import shared_panel_bootstrap_core as core
from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import holm_adjust


KIND = "rq12_vega_mae_paired_robustness_v1"
ARMS = ("lp_matched", "lp_shuffle", "pure_cnn_no_text", "bow", "sentiment")
VERSIONS = ("unweighted_control", "vega_retrained")
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
SEEDS = (42, 202, 404, 382624741, 779214671, 1343862330, 1607127774,
         1662128673, 2014889368, 2041145538)
FOLD_PAIR_COUNTS = (110, 112, 135, 143)
FOLD_SESSION_COUNTS = (34, 36, 33, 45)
DEFAULT_MAIN_ROOT = REPO_ROOT / "outputs/experiments/rq12_direct_vega_mae_only_20261007_v2"
DEFAULT_BOOTSTRAP_ROOT = REPO_ROOT / "outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2"
DEFAULT_OUTPUT_SUBDIR = "analysis/vega_robustness"
METRICS = ("plain_mae", "vega_mae", "persistence_plain_mae", "persistence_vega_mae")
SHARED_LINEAGE = (
    "sample_id", "session_id", "current_snapshot_time_utc", "target_snapshot_time_utc",
    "supported_cell_count", "support_mask_fingerprint", "cell_weight_fingerprint",
    "pair_truth_fingerprint", "cell_weight_manifest_sha256",
    "persistence_plain_mae", "persistence_vega_mae",
)
REQUIRED_COLUMNS = (
    "experiment_version", "architecture", "arm", "fold", "seed", "tolerance_minutes",
    "pair_id", *SHARED_LINEAGE, "plain_mae", "vega_mae",
)


class VegaRobustnessError(ValueError):
    """Frozen input, provenance, or analysis-contract validation failed."""


def _condition(version: str, arm: str) -> str:
    return f"{version}::{arm}"


def prepare_input_panel(frame: pd.DataFrame, *, require_frozen_counts: bool = True) -> core.Panel:
    """Validate paired scoring lineage and adapt the ten conditions to the core.

    ``require_frozen_counts=False`` permits small synthetic panels for semantic
    tests; it retains the five-model, two-version, exact-pairing and common
    truth/support/current-weight/persistence contracts.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise VegaRobustnessError(f"missing pair-metric columns: {missing}")
    data = frame.loc[:, list(dict.fromkeys(REQUIRED_COLUMNS))].copy(deep=True)
    if data.empty or data.isna().any().any():
        raise VegaRobustnessError("pair metrics must be nonempty without null required fields")
    for column in set(data.columns) - set(METRICS) - {"seed", "tolerance_minutes", "supported_cell_count"}:
        data[column] = data[column].astype(str)
        if data[column].str.strip().eq("").any():
            raise VegaRobustnessError(f"empty pair-metric field: {column}")
    for column in ("seed", "tolerance_minutes", "supported_cell_count"):
        numeric = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise VegaRobustnessError(f"{column} must contain finite integers")
        data[column] = numeric.astype(np.int64)
    for column in METRICS:
        numeric = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
            raise VegaRobustnessError(f"{column} must contain finite strictly positive losses")
        data[column] = numeric
    if set(data.experiment_version) != set(VERSIONS) or set(data.arm) != set(ARMS):
        raise VegaRobustnessError("expected exactly two versions and the five declared model arms")
    expected_architecture = data.arm.map(lambda arm: "pure" if arm == "pure_cnn_no_text" else "film")
    if not data.architecture.eq(expected_architecture).all():
        raise VegaRobustnessError("model arm/architecture mismatch")
    if set(data.tolerance_minutes) != {5}:
        raise VegaRobustnessError("only the common five-minute test panel is admissible")
    if not data.supported_cell_count.between(1, 256).all():
        raise VegaRobustnessError("supported_cell_count must be between 1 and 256")
    keys = ["arm", "seed", "fold", "pair_id"]
    if data.duplicated(["experiment_version", *keys]).any():
        raise VegaRobustnessError("duplicate version/model/seed/fold/pair primary key")
    old = data.loc[data.experiment_version.eq(VERSIONS[0])].set_index(keys).sort_index()
    new = data.loc[data.experiment_version.eq(VERSIONS[1])].set_index(keys).sort_index()
    if not old.index.equals(new.index):
        raise VegaRobustnessError("control/retrained primary keys do not pair one-to-one")
    for column in SHARED_LINEAGE:
        if not np.array_equal(old[column].to_numpy(), new[column].to_numpy()):
            raise VegaRobustnessError(f"paired scoring lineage drift: {column}")
    # The market is shared across model seeds and architectures, not repeated
    # independent observations.  Fail even if both versions drift together.
    market_groups = data.groupby(["fold", "pair_id"], sort=True, dropna=False)
    for column in SHARED_LINEAGE:
        if not market_groups[column].nunique(dropna=False).eq(1).all():
            raise VegaRobustnessError(f"shared market-panel lineage drift: {column}")
    if data.cell_weight_manifest_sha256.nunique() != 1:
        raise VegaRobustnessError("the common current-Vega manifest must be identical for every row")
    canonical = data[["seed", "fold", "pair_id", "session_id"]].copy()
    canonical["condition"] = data.experiment_version + "::" + data.arm
    canonical["value"] = data.vega_mae
    canonical["persistence_mae"] = data.persistence_vega_mae
    panel = core.prepare_panel(canonical)
    if require_frozen_counts:
        market = panel.frame[["fold", "pair_id", "session_id"]].drop_duplicates()
        if len(data) != 50_000 or len(old) != 25_000 or len(market) != 500:
            raise VegaRobustnessError("frozen universe requires 50,000 rows and 25,000 paired keys over 500 market pairs")
        if panel.seeds != SEEDS or panel.folds != FOLDS:
            raise VegaRobustnessError("frozen ten-seed/four-fold universe drift")
        if market.session_id.nunique() != 148:
            raise VegaRobustnessError("frozen market panel requires 148 CME sessions")
        for fold, pair_count, session_count in zip(FOLDS, FOLD_PAIR_COUNTS, FOLD_SESSION_COUNTS, strict=True):
            selected = market.loc[market.fold.eq(fold)]
            if len(selected) != pair_count or selected.session_id.nunique() != session_count:
                raise VegaRobustnessError(f"frozen pair/session counts drift in {fold}")
        for column in ("support_mask_fingerprint", "cell_weight_fingerprint", "pair_truth_fingerprint", "cell_weight_manifest_sha256"):
            if not data[column].str.fullmatch(r"[0-9a-f]{64}").all():
                raise VegaRobustnessError(f"invalid SHA-256 lineage field: {column}")
    return panel


def analyze_frame(
    frame: pd.DataFrame, schedule: core.Schedule, *, require_frozen_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Compute five two-sided Holm-adjusted Vega contrasts on one shared schedule."""
    panel = prepare_input_panel(frame, require_frozen_counts=require_frozen_counts)
    result = core.run_bootstrap(panel, schedule, estimand="equal_cell")
    cells = frame.groupby(["experiment_version", "arm", "seed", "fold"], sort=True)[list(METRICS)].mean()
    means = cells.groupby(["experiment_version", "arm"], sort=True).mean()
    rows: list[dict[str, Any]] = []
    contrast_draws: list[np.ndarray] = []
    point_errors: list[float] = []
    for arm in ARMS:
        focal, reference = _condition(VERSIONS[1], arm), _condition(VERSIONS[0], arm)
        contrast = result.contrast(focal, reference)
        old, new = means.loc[(VERSIONS[0], arm)], means.loc[(VERSIONS[1], arm)]
        old_cells = cells.loc[(VERSIONS[0], arm), "vega_mae"].sort_index()
        new_cells = cells.loc[(VERSIONS[1], arm), "vega_mae"].sort_index()
        independent_theta = float(np.log(new_cells.to_numpy() / old_cells.to_numpy()).mean())
        point_errors.append(abs(independent_theta - contrast["point"]))
        if not math.isclose(independent_theta, contrast["point"], rel_tol=1e-10, abs_tol=5e-14):
            raise VegaRobustnessError(f"equal-cell log-ratio preservation failed: {arm}")
        row = {
            "arm": arm, "architecture": "pure" if arm == "pure_cnn_no_text" else "film",
            "focal_condition": focal, "reference_condition": reference,
            "metric": "vega_mae", "estimand": "equal_cell",
            "family_id": "same_model_vega_retraining_two_sided_holm5",
            "old_plain_mae": float(old.plain_mae), "new_plain_mae": float(new.plain_mae),
            "persistence_plain_mae": float(old.persistence_plain_mae),
            "old_vega_mae": float(old.vega_mae), "new_vega_mae": float(new.vega_mae),
            "persistence_vega_mae": float(old.persistence_vega_mae),
            "ordinary_arithmetic_gain_percent": 100.0 * (1.0 - new.plain_mae / old.plain_mae),
            "vega_arithmetic_gain_percent": 100.0 * (1.0 - new.vega_mae / old.vega_mae),
            "old_plain_skill_vs_persistence": 1.0 - old.plain_mae / old.persistence_plain_mae,
            "new_plain_skill_vs_persistence": 1.0 - new.plain_mae / old.persistence_plain_mae,
            "old_vega_skill_vs_persistence": 1.0 - old.vega_mae / old.persistence_vega_mae,
            "new_vega_skill_vs_persistence": 1.0 - new.vega_mae / old.persistence_vega_mae,
            "mean_log_ratio": contrast["point"], "bootstrap_se": contrast["bootstrap_se"],
            "ci_lower": contrast["ci_lower"], "ci_upper": contrast["ci_upper"],
            "geometric_gain_percent": -100.0 * math.expm1(contrast["point"]),
            # The gain transform is decreasing, so CI endpoints reverse.
            "gain_ci_lower_percent": -100.0 * math.expm1(contrast["ci_upper"]),
            "gain_ci_upper_percent": -100.0 * math.expm1(contrast["ci_lower"]),
            "p_two": contrast["p_two"], "seed_consistent": contrast["seed_consistent"],
            "fold_consistent": contrast["fold_consistent"],
            "seed_count": len(panel.seeds), "fold_count": len(panel.folds),
            "pair_count": panel.frame.pair_id.nunique(),
            "session_count": sum(map(len, panel.sessions_by_fold)),
            "iterations": len(schedule.draw_id), "bootstrap_method_version": core.METHOD_VERSION,
        }
        rows.append(row)
        contrast_draws.append(contrast["draws"])
    adjusted = holm_adjust({row["arm"]: row["p_two"] for row in rows})
    for row in rows:
        row["p_holm"] = adjusted[row["arm"]]
        row["reject_two_sided_holm_0_05"] = bool(row["p_holm"] < 0.05)
    summary = pd.DataFrame(rows)
    draws = {
        "draw_id": np.array(schedule.draw_id, copy=True),
        "arm_names": np.asarray(ARMS), "condition_names": np.asarray(result.conditions),
        "observed_means": np.array(result.observed_means, copy=True),
        "mean_draws": np.array(result.mean_draws, copy=True),
        "log_ratio_draws": np.column_stack(contrast_draws),
    }
    if not np.isfinite(draws["mean_draws"]).all() or not np.isfinite(draws["log_ratio_draws"]).all():
        raise VegaRobustnessError("bootstrap produced nonfinite draws")
    qa = {
        "passed": True, "pair_metric_rows": len(frame), "paired_primary_keys": len(frame) // 2,
        "model_count": 5, "condition_count": len(panel.conditions), "holm_family_size": 5,
        "iterations_per_comparison": len(schedule.draw_id), "seed_count": len(panel.seeds),
        "fold_count": len(panel.folds), "pair_count": panel.frame.pair_id.nunique(),
        "session_count": sum(map(len, panel.sessions_by_fold)),
        "market_fingerprint": panel.market_fingerprint, "panel_fingerprint": panel.panel_fingerprint,
        "paired_truth_support_current_weight_persistence_equal": True,
        "shared_market_across_all_conditions_and_seeds": True,
        "observed_equal_cell_log_ratios_preserved": True,
        "maximum_observed_point_discrepancy": max(point_errors),
        "ordinary_mae_descriptive_only": True, "geometric_gain_ci_endpoints_reversed": True,
        "p_value_interpretation": "approximate_uncentered_plus_one_bootstrap_sign_tail",
    }
    return summary, draws, qa


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VegaRobustnessError(f"JSON object required: {path}")
    return value


def _signed_json(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    digest = value.get("payload_sha256")
    if not digest or digest != _payload_sha({key: item for key, item in value.items() if key != "payload_sha256"}):
        raise VegaRobustnessError(f"signed payload SHA-256 mismatch: {path}")
    return value


class _InputVerifier:
    """Hash each distinct input and reject conflicting pins or mid-run changes."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def check(self, path: Path, expected: str | None = None, *, role: str) -> str:
        source = path.resolve()
        if not source.is_file():
            raise VegaRobustnessError(f"missing frozen input: {source}")
        stat = source.stat()
        key = str(source)
        prior = self.records.get(key)
        if prior is None:
            digest = _sha256(source)
            after = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise VegaRobustnessError(f"input changed during hashing: {source}")
            prior = {"path": key, "sha256": digest, "size_bytes": stat.st_size,
                     "mtime_ns": stat.st_mtime_ns, "roles": []}
            self.records[key] = prior
        elif (stat.st_size, stat.st_mtime_ns) != (prior["size_bytes"], prior["mtime_ns"]):
            raise VegaRobustnessError(f"input changed during analysis: {source}")
        if expected is not None and str(expected) != prior["sha256"]:
            raise VegaRobustnessError(f"frozen SHA-256 mismatch: {source}")
        if role not in prior["roles"]:
            prior["roles"].append(role)
        return str(prior["sha256"])

    def verify_unchanged(self) -> None:
        for record in self.records.values():
            source = Path(record["path"])
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
                raise VegaRobustnessError(f"input changed during analysis: {source}")
            if _sha256(source) != record["sha256"]:
                raise VegaRobustnessError(f"input content changed during analysis: {source}")


def _archive_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise VegaRobustnessError(f"archive path escapes root: {relative_path}")
    return candidate


def _load_frozen_rescore(main_root: Path, verifier: _InputVerifier) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis = main_root / "analysis"
    manifest_path = analysis / "control_new_dual_rescore_manifest.json"
    verifier.check(manifest_path, role="frozen_rescore_manifest")
    manifest = _signed_json(manifest_path)
    if manifest.get("kind") != "rq12_current_vega_frozen_prediction_rescore_v1" or manifest.get("controls_only") is not False:
        raise VegaRobustnessError("the complete control/new frozen rescore manifest is required")
    if manifest.get("pair_metric_rows") != 50_000:
        raise VegaRobustnessError("rescore manifest must declare 50,000 pair metric rows")
    expected_jobs = {f"{version}/{architecture}": count for version in VERSIONS for architecture, count in (("film", 160), ("pure", 40))}
    if manifest.get("job_counts") != expected_jobs:
        raise VegaRobustnessError("rescore manifest must pin 200 control and 200 retrained prediction jobs")
    verifier.check(Path(manifest["rescore_source_path"]), manifest["rescore_source_sha256"], role="frozen_rescore_implementation")
    input_hashes = Path(manifest["input_hashes_path"])
    if input_hashes.resolve() != (analysis / "control_new_dual_input_hashes.csv").resolve():
        raise VegaRobustnessError("rescore input-hash table path does not belong to the requested main root")
    verifier.check(input_hashes, manifest["input_hashes_sha256"], role="frozen_rescore_input_hash_table")
    frozen_inputs = pd.read_csv(input_hashes, dtype=str)
    if set(frozen_inputs.columns) != {"role", "path", "sha256"} or frozen_inputs.empty or frozen_inputs.isna().any().any():
        raise VegaRobustnessError("malformed frozen rescore input-hash table")
    if not frozen_inputs.sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise VegaRobustnessError("invalid rescore input SHA-256 pin")
    for item in frozen_inputs.itertuples(index=False):
        verifier.check(Path(item.path), item.sha256, role=f"rescore_input:{item.role}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 8:
        raise VegaRobustnessError("expected the eight frozen dual-rescore output pins")
    pair_path = analysis / "control_new_dual_pair_metrics.csv.gz"
    if not any(Path(item["path"]).resolve() == pair_path.resolve() for item in outputs):
        raise VegaRobustnessError("pair-level metrics are absent from the frozen rescore output pins")
    for item in outputs:
        output = Path(item["path"])
        if output.resolve().parent != analysis.resolve():
            raise VegaRobustnessError("frozen rescore output is outside the requested main root analysis")
        verifier.check(output, item["sha256"], role="frozen_rescore_output")
    weight_path = Path(manifest["cell_weight_manifest_path"])
    if weight_path.resolve() != (main_root / "weights/manifest.json").resolve():
        raise VegaRobustnessError("common current-Vega manifest belongs to a different main root")
    verifier.check(weight_path, manifest["cell_weight_manifest_sha256"], role="common_current_vega_manifest")
    weights = _read_json(weight_path)
    if weights.get("cell_weight_mode") != "black76_current_vega_v1" or weights.get("qc", {}).get("status") != "pass":
        raise VegaRobustnessError("common current-Vega weight cache QA did not pass")
    grid_path = _archive_path(weight_path.parent, weights["weights_path"])
    verifier.check(grid_path, weights["weights_sha256"], role="common_current_vega_grid")
    for name, digest in weights.get("audit_sha256", {}).items():
        verifier.check(_archive_path(weight_path.parent, name), digest, role="current_vega_provenance_audit")
    verifier.check(weight_path.parent / "qc_summary.json", weights["qc_summary_sha256"], role="current_vega_cache_qa")
    frame = pd.read_csv(pair_path, float_precision="round_trip")
    if set(frame.cell_weight_manifest_sha256) != {manifest["cell_weight_manifest_sha256"]}:
        raise VegaRobustnessError("pair metrics do not reference the pinned common current-Vega manifest")
    if "prediction_sha256" not in frame or not set(frame.prediction_sha256).issubset(set(frozen_inputs.loc[frozen_inputs.role.eq("prediction"), "sha256"])):
        raise VegaRobustnessError("pair metrics reference an unpinned frozen prediction")
    return frame, manifest


def _load_archived_schedule(bootstrap_root: Path, verifier: _InputVerifier) -> tuple[core.Schedule, dict[str, Any]]:
    hash_path = bootstrap_root / "output_hashes.csv"
    verifier.check(hash_path, role="bootstrap_archive_output_hash_table")
    hashes = pd.read_csv(hash_path, dtype=str)
    if set(hashes.columns) != {"relative_path", "sha256"} or hashes.empty or hashes.isna().any().any() or hashes.relative_path.duplicated().any():
        raise VegaRobustnessError("malformed archived bootstrap output-hash table")
    for item in hashes.itertuples(index=False):
        verifier.check(_archive_path(bootstrap_root, item.relative_path), item.sha256, role="bootstrap_archive_output")
    for mandatory in ("analysis_manifest.json", "input_manifest.json", "qa.json"):
        if mandatory not in set(hashes.relative_path):
            raise VegaRobustnessError(f"bootstrap archive does not pin {mandatory}")
    manifest = _read_json(bootstrap_root / "analysis_manifest.json")
    qa = _read_json(bootstrap_root / "qa.json")
    if not all(qa.get(key) is True for key in ("passed", "input_hashes_unchanged", "observed_estimands_preserved")):
        raise VegaRobustnessError("archived shared-market bootstrap QA did not pass")
    if manifest.get("bootstrap_method_version") != core.METHOD_VERSION or manifest.get("iterations") != 10_000 or int(manifest.get("rng_seed", -1)) != 20260904:
        raise VegaRobustnessError("archived schedule method/iteration/RNG contract drift")
    if qa.get("iterations_per_comparison") != 10_000:
        raise VegaRobustnessError("archived bootstrap QA iteration count drift")
    source_manifest = _read_json(bootstrap_root / "input_manifest.json")
    for item in source_manifest["inputs"]:
        source = Path(item["path"])
        if not source.is_absolute():
            source = REPO_ROOT / source
        verifier.check(source, item["sha256"], role=f"bootstrap_archive_input:{item['role']}")
        if "json_require" in item:
            data = _read_json(source)
            for key, expected in item["json_require"].items():
                if data.get(key) != expected:
                    raise VegaRobustnessError(f"archived source QA drift: {source}: {key}")
    core_file = Path(core.__file__).resolve()
    core_items = [item for item in source_manifest["implementation"] if Path(item["path"]).resolve() == core_file]
    if len(core_items) != 1:
        raise VegaRobustnessError("bootstrap archive must pin the shared-panel core implementation")
    verifier.check(core_file, core_items[0]["sha256"], role="shared_panel_core_implementation")
    verifier.check(_archive_path(bootstrap_root, core_items[0]["archived_path"]), core_items[0]["sha256"], role="archived_shared_panel_core_implementation")
    jobs = [job for job in manifest.get("jobs", []) if job.get("job_id") == "direct_overall"]
    if len(jobs) != 1:
        raise VegaRobustnessError("expected exactly one archived direct_overall schedule")
    job = jobs[0]
    if job.get("estimand") != "equal_cell" or job.get("pair_count") != 500 or job.get("session_count") != 148:
        raise VegaRobustnessError("archived direct_overall market universe/estimand drift")
    schedule_path = _archive_path(bootstrap_root, job["schedule_path"])
    if job["schedule_path"] not in set(hashes.relative_path):
        raise VegaRobustnessError("the selected direct_overall schedule has no archived file pin")
    digest = verifier.check(schedule_path, role="reused_direct_overall_schedule")
    schedule = core.load_schedule(schedule_path)
    metadata = schedule.metadata
    if metadata["iterations"] != 10_000 or int(metadata["rngseed"]) != 20260904 or metadata["method_version"] != core.METHOD_VERSION:
        raise VegaRobustnessError("loaded schedule metadata differs from the archived contract")
    if tuple(metadata["seeds"]) != SEEDS or tuple(metadata["folds"]) != FOLDS or metadata["market_fingerprint"] != job["market_fingerprint"]:
        raise VegaRobustnessError("loaded schedule differs from the declared direct_overall job")
    return schedule, {"job_id": "direct_overall", "schedule_id": job["schedule_id"],
                      "schedule_path": str(schedule_path), "schedule_sha256": digest,
                      "schedule_metadata": metadata, "archive_qa": qa}


def _write_json(path: Path, payload: dict[str, Any], *, signed: bool = False) -> None:
    value = dict(payload)
    if signed:
        value["payload_sha256"] = _payload_sha(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def run_analysis(
    main_root: Path = DEFAULT_MAIN_ROOT, bootstrap_root: Path = DEFAULT_BOOTSTRAP_ROOT,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
) -> dict[str, Any]:
    """Validate frozen sources, replay 10,000 archived draws, and publish a sidecar."""
    main_root, bootstrap_root = Path(main_root).resolve(), Path(bootstrap_root).resolve()
    output = _archive_path(main_root, output_subdir)
    if output == main_root or output.exists():
        raise VegaRobustnessError(f"analysis destination must be new: {output}")
    verifier = _InputVerifier()
    print("Validating frozen dual-rescore inputs and prediction SHA-256 pins...", flush=True)
    frame, rescore_manifest = _load_frozen_rescore(main_root, verifier)
    print("Validating the immutable Chapter 3 archive and loading direct_overall schedule...", flush=True)
    schedule, schedule_info = _load_archived_schedule(bootstrap_root, verifier)
    own_source = Path(__file__).resolve()
    verifier.check(own_source, role="vega_robustness_implementation")
    from scripts.rq123 import news_first_vol_film_nolp_10seed_analysis as holm_module
    verifier.check(Path(holm_module.__file__), role="holm_adjust_implementation")
    print("Computing five equal-cell paired Vega contrasts on the archived 10,000 draws...", flush=True)
    summary, draws, qa = analyze_frame(frame, schedule)
    qa.update({"kind": KIND, "schema_version": 1, "frozen_input_sha256_verified": True,
               "bootstrap_archive_output_sha256_verified": True,
               "bootstrap_archive_source_sha256_verified": True,
               "archived_schedule_reused_without_new_rng_draws": True,
               "schedule_sha256": schedule_info["schedule_sha256"],
               "schedule_weights_sha256": schedule.metadata["weights_sha256"]})
    canonical_panel = prepare_input_panel(frame).frame
    cell_summary = frame.groupby(["experiment_version", "architecture", "arm", "seed", "fold"], sort=True)[list(METRICS)].mean().reset_index()
    # A fresh staging directory keeps partially generated analysis out of the
    # published sidecar.  Existing immutable evidence is only read.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        provenance = staging / "provenance"
        provenance.mkdir()
        shutil.copy2(own_source, provenance / own_source.name)
        shutil.copy2(Path(core.__file__), provenance / "shared_panel_bootstrap_core.py")
        shutil.copy2(Path(holm_module.__file__), provenance / "news_first_vol_film_nolp_10seed_analysis.py")
        summary.to_csv(staging / "summary.csv", index=False, float_format="%.17g")
        cell_summary.to_csv(staging / "seed_fold_descriptive_metrics.csv", index=False, float_format="%.17g")
        canonical_panel.to_csv(staging / "canonical_panel.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
        np.savez_compressed(staging / "bootstrap_draws.npz", **draws)
        # Preserve the exact archived weights so the sidecar can be inspected
        # without regenerating or replacing the original archive.
        shutil.copy2(schedule_info["schedule_path"], staging / "reused_schedule.npz")
        if _sha256(staging / "reused_schedule.npz") != schedule_info["schedule_sha256"]:
            raise VegaRobustnessError("copied archived schedule SHA-256 mismatch")
        verifier.verify_unchanged()
        qa["input_hashes_unchanged"] = True
        inputs = sorted(verifier.records.values(), key=lambda item: item["path"])
        pd.DataFrame([{key: value for key, value in item.items() if key != "roles"} | {"roles": "|".join(sorted(item["roles"]))} for item in inputs]).to_csv(staging / "input_hashes.csv", index=False)
        _write_json(staging / "input_manifest.json", {
            "kind": KIND, "schema_version": 1, "inputs": inputs,
            "main_root": str(main_root), "bootstrap_root": str(bootstrap_root),
            "frozen_rescore_payload_sha256": rescore_manifest["payload_sha256"],
            "schedule": schedule_info,
        }, signed=True)
        _write_json(staging / "qa.json", qa, signed=True)
        _write_json(staging / "analysis_manifest.json", {
            "kind": KIND, "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output), "bootstrap_method_version": core.METHOD_VERSION,
            "iterations": 10_000, "rng_seed": 20260904, "schedule_id": schedule_info["schedule_id"],
            "schedule_sha256": schedule_info["schedule_sha256"],
            "schedule_weights_sha256": schedule.metadata["weights_sha256"],
            "market_fingerprint": qa["market_fingerprint"], "panel_fingerprint": qa["panel_fingerprint"],
            "primary_metric": "common_current_vega_mae", "primary_family": list(ARMS),
            "primary_family_size": 5, "alternative": "two_sided", "multiplicity": "Holm_5",
            "estimand": "equal_cell", "point_estimate": "mean_over_40_seed_fold_cells_of_log(new_vega_mae/control_vega_mae)",
            "observed_mae_aggregation": "equal_pairs_within_fold_equal_folds_and_seeds",
            "ordinary_mae_role": "descriptive_only_no_bootstrap_or_formal_test",
            "ci_method": "percentile_95", "gain_transform": "100*(1-exp(theta)); interval endpoints reversed",
            "p_value_interpretation": "approximate_uncentered_plus_one_bootstrap_sign_tail",
            "resampling_unit": "crossed_model_seeds_and_shared_fold_occurrences_with_nested_CME_sessions",
            "confirmatory": False, "interpretation": "retrospective_rolling_development_robustness",
            "training_or_inference_performed": False, "immutable_inputs_modified": False,
        }, signed=True)
        output_hashes = [{"relative_path": str(path.relative_to(staging)), "sha256": _sha256(path)}
                         for path in sorted(staging.rglob("*")) if path.is_file()]
        pd.DataFrame(output_hashes).to_csv(staging / "output_hashes.csv", index=False)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Published verified paired robustness analysis: {output}", flush=True)
    print(summary[["arm", "mean_log_ratio", "geometric_gain_percent", "gain_ci_lower_percent", "gain_ci_upper_percent", "p_two", "p_holm", "seed_consistent", "fold_consistent"]].to_string(index=False), flush=True)
    return {"output_root": str(output), "qa": qa, "summary": summary.to_dict("records")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    arguments = parser.parse_args(argv)
    run_analysis(arguments.main_root, arguments.bootstrap_root, arguments.output_subdir)


if __name__ == "__main__":
    main()
