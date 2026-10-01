"""Recompute frozen Chapter 3 RQ4 uncertainty with shared market draws.

The scheduled matched-set differences are signed, so the positive-MAE
``run_bootstrap`` statistic is not applicable.  This module uses its validated
seed/fold/session schedule, then applies those weights to the original RQ4
equal-seed-fold mean of signed event-minus-control increments.

Run from the repository root::

    python -m scripts.rq123.chapter3_rq4_shared_market_bootstrap
    python -m scripts.rq123.chapter3_rq4_shared_market_bootstrap --verify-only

The original experiment outputs and thesis text are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.rq123.shared_panel_bootstrap_core import (
    METHOD_VERSION as SCHEDULE_METHOD_VERSION,
    Schedule,
    load_schedule,
    make_schedule,
    prepare_panel,
    save_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    REPO_ROOT
    / "outputs/experiments/"
    "rq123_news_first_vol_film_unet_text128_c32_nolp_10seed_parent30_cont240_exact_ttm_rolling_v1/"
    "analysis/unified"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/chapter3_rq4_shared_market_bootstrap_10000_v1"
SOURCE_FILES = (
    "rq3_scheduled_matched_differences.csv",
    "rq3_scheduled_match_plan.csv",
    "rq3_scheduled_results.csv",
)
METHOD_VERSION = "rq4_equal_cell_crossed_seed_shared_market_session_v1"
DEFAULT_ITERATIONS = 10_000
DEFAULT_RNG_SEED = 20_260_904
_KEY = ["window_id", "contrast_id", "seed", "fold", "matched_set_id"]
_RESULT_COLUMNS = [
    "window_id",
    "window_role",
    "contrast_id",
    "focal_arm",
    "reference_arm",
    "mean_difference",
    "bootstrap_se",
    "ci_95_lower",
    "ci_95_upper",
    "p_value_one_sided",
    "holm_adjusted_p",
    "consistent_seed_count",
    "consistent_fold_count",
    "matched_set_count",
    "event_session_count",
    "seed_count",
    "fold_count",
    "bootstrap_iterations",
    "bootstrap_seed",
    "resampling_method",
    "historical_ci_95_lower",
    "historical_ci_95_upper",
    "historical_p_value_one_sided",
    "historical_holm_adjusted_p",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.17g")
    return buffer.getvalue()


def _load_inputs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [source_dir / name for name in SOURCE_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    details, plan, historical = (pd.read_csv(path) for path in paths)
    required_details = set(_KEY) | {
        "window_role",
        "session_id",
        "event_pair_id",
        "control_pair_id",
        "event_text_advantage",
        "control_text_advantage",
        "scheduled_increment",
        "focal_arm",
        "reference_arm",
    }
    required_plan = {
        "window_id",
        "matched_set_id",
        "match_status",
        "fold",
        "event_session_id",
        "event_pair_id",
        "control_pair_id",
    }
    required_historical = {
        "window_id",
        "contrast_id",
        "mean_difference",
        "ci_95_lower",
        "ci_95_upper",
        "p_value_one_sided",
        "holm_adjusted_p",
        "consistent_seed_count",
        "consistent_fold_count",
        "pair_count",
        "session_count",
    }
    for label, frame, required in (
        ("details", details, required_details),
        ("match plan", plan, required_plan),
        ("historical results", historical, required_historical),
    ):
        missing = sorted(required - set(frame))
        if frame.empty or missing:
            raise ValueError(f"{label} is empty or missing columns: {missing}")
    if details.duplicated(_KEY).any():
        raise ValueError("duplicate window/contrast/seed/fold/matched-set detail rows")
    if historical.duplicated(["window_id", "contrast_id"]).any():
        raise ValueError("duplicate historical window/contrast rows")
    if plan.duplicated(["window_id", "matched_set_id"]).any():
        raise ValueError("duplicate match-plan window/matched-set rows")
    for name in ("event_text_advantage", "control_text_advantage", "scheduled_increment"):
        if not np.isfinite(pd.to_numeric(details[name], errors="coerce")).all():
            raise ValueError(f"{name} contains non-finite values")
    residual = (
        details["event_text_advantage"].to_numpy(float)
        - details["control_text_advantage"].to_numpy(float)
        - details["scheduled_increment"].to_numpy(float)
    )
    if np.max(np.abs(residual)) > 1e-15:
        raise ValueError("event-minus-control arithmetic differs from frozen increment")
    matched = plan.loc[plan["match_status"].eq("matched")].copy()
    observed = details[
        ["window_id", "matched_set_id", "fold", "session_id", "event_pair_id", "control_pair_id"]
    ].drop_duplicates()
    merged = observed.merge(
        matched[
            ["window_id", "matched_set_id", "fold", "event_session_id", "event_pair_id", "control_pair_id"]
        ].rename(columns={"event_session_id": "session_id"}),
        on=["window_id", "matched_set_id", "fold", "session_id", "event_pair_id", "control_pair_id"],
        how="outer",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(matched):
        raise ValueError("frozen details differ from the matched event/control plan")
    detail_pairs = set(zip(details["window_id"], details["contrast_id"], strict=True))
    historical_pairs = set(zip(historical["window_id"], historical["contrast_id"], strict=True))
    if detail_pairs != historical_pairs:
        raise ValueError("detail and historical window/contrast coverage differ")
    return details, plan, historical


def _window_arrays(
    frame: pd.DataFrame,
) -> tuple[Any, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Validate crossed lineage and aggregate signed values by event session."""

    carrier = frame[
        ["contrast_id", "seed", "fold", "matched_set_id", "session_id"]
    ].rename(columns={"contrast_id": "condition", "matched_set_id": "pair_id"}).copy()
    carrier["value"] = 1.0  # Schedule metadata only; signed inference uses sums below.
    panel = prepare_panel(carrier)
    if len(panel.conditions) != 2 or len(panel.seeds) != 10:
        raise ValueError("RQ4 requires two contrasts and ten crossed model seeds per window")
    aggregate = frame.groupby(
        ["contrast_id", "seed", "fold", "session_id"], sort=False
    )["scheduled_increment"].sum()
    market = panel.frame
    reference = market.loc[
        market["condition"].eq(panel.conditions[0]) & market["seed"].eq(panel.seeds[0])
    ]
    counts_by_fold: list[np.ndarray] = []
    sums_by_fold: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    for fold, sessions in zip(panel.folds, panel.sessions_by_fold, strict=True):
        counts = np.array(
            [int((reference["fold"].eq(fold) & reference["session_id"].eq(session)).sum())
             for session in sessions],
            dtype=np.int64,
        )
        sums = np.empty((len(panel.conditions), len(panel.seeds), len(sessions)), dtype=float)
        for ci, condition in enumerate(panel.conditions):
            for si, seed in enumerate(panel.seeds):
                sums[ci, si] = [
                    float(aggregate.loc[(condition, seed, fold, session)])
                    for session in sessions
                ]
        counts_by_fold.append(counts)
        sums_by_fold.append(sums)
        cells.append(sums.sum(axis=2) / float(counts.sum()))
    cell_points = np.stack(cells, axis=2)
    return panel, tuple(sums_by_fold), tuple(counts_by_fold), cell_points


def _draw_values(
    schedule: Schedule,
    sums_by_fold: tuple[np.ndarray, ...],
    counts_by_fold: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Apply one fold/session occurrence to all sampled seeds and contrasts."""

    iterations = schedule.iterations
    seed_count = len(schedule.seeds)
    fold_count = len(schedule.folds)
    condition_count = sums_by_fold[0].shape[0]
    draws = np.zeros((iterations, condition_count), dtype=np.float64)
    for occurrence in range(fold_count):
        for fold_index, (sums, counts) in enumerate(
            zip(sums_by_fold, counts_by_fold, strict=True)
        ):
            mask = schedule.fold_indices[:, occurrence] == fold_index
            if not np.any(mask):
                continue
            weights = schedule.session_weights[
                mask, occurrence, : len(counts)
            ].astype(np.float64, copy=False)
            numerator = np.einsum("bm,csm->bcs", weights, sums, optimize=True)
            denominator = weights @ counts.astype(np.float64, copy=False)
            cell_means = numerator / denominator[:, None, None]
            seed_weights = schedule.seed_weights[mask].astype(np.float64, copy=False)
            draws[mask] += np.einsum(
                "bs,bcs->bc", seed_weights, cell_means, optimize=True
            ) / float(seed_count * fold_count)
    if not np.isfinite(draws).all():
        raise ValueError("non-finite RQ4 bootstrap draws")
    return draws


def _holm_two(values: np.ndarray) -> np.ndarray:
    if values.shape != (2,):
        raise ValueError("each RQ4 window needs exactly two directional comparisons")
    order = np.argsort(values, kind="stable")
    result = np.empty(2, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (2 - rank) * float(values[index])))
        result[index] = running
    return result


def _window_results(
    window: str,
    frame: pd.DataFrame,
    historical: pd.DataFrame,
    panel: Any,
    cell_points: np.ndarray,
    draws: np.ndarray,
    rng_seed: int,
) -> list[dict[str, Any]]:
    points = cell_points.mean(axis=(1, 2))
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=0)
    p_one = (1.0 + np.count_nonzero(draws <= 0.0, axis=0)) / (len(draws) + 1.0)
    holm = _holm_two(p_one)
    seed_votes = (cell_points.mean(axis=2) >= 0.0).sum(axis=1)
    fold_votes = (cell_points.mean(axis=1) >= 0.0).sum(axis=1)
    rows: list[dict[str, Any]] = []
    for ci, contrast in enumerate(panel.conditions):
        source = historical.loc[
            historical["window_id"].eq(window) & historical["contrast_id"].eq(contrast)
        ]
        if len(source) != 1:
            raise ValueError(f"historical result missing: {window}/{contrast}")
        old = source.iloc[0]
        if (
            abs(float(points[ci]) - float(old["mean_difference"])) > 1e-18
            or int(seed_votes[ci]) != int(old["consistent_seed_count"])
            or int(fold_votes[ci]) != int(old["consistent_fold_count"])
            or int(frame["matched_set_id"].nunique()) != int(old["pair_count"])
            or int(frame["session_id"].nunique()) != int(old["session_count"])
        ):
            raise ValueError(f"frozen point/direction/coverage drift: {window}/{contrast}")
        part = frame.loc[frame["contrast_id"].eq(contrast)]
        first = part.iloc[0]
        rows.append(
            {
                "window_id": window,
                "window_role": str(first["window_role"]),
                "contrast_id": contrast,
                "focal_arm": str(first["focal_arm"]),
                "reference_arm": str(first["reference_arm"]),
                "mean_difference": float(points[ci]),
                "bootstrap_se": float(draws[:, ci].std(ddof=1)),
                "ci_95_lower": float(lower[ci]),
                "ci_95_upper": float(upper[ci]),
                "p_value_one_sided": float(p_one[ci]),
                "holm_adjusted_p": float(holm[ci]),
                "consistent_seed_count": int(seed_votes[ci]),
                "consistent_fold_count": int(fold_votes[ci]),
                "matched_set_count": int(frame["matched_set_id"].nunique()),
                "event_session_count": int(frame["session_id"].nunique()),
                "seed_count": len(panel.seeds),
                "fold_count": len(panel.folds),
                "bootstrap_iterations": len(draws),
                "bootstrap_seed": int(rng_seed),
                "resampling_method": METHOD_VERSION,
                "historical_ci_95_lower": float(old["ci_95_lower"]),
                "historical_ci_95_upper": float(old["ci_95_upper"]),
                "historical_p_value_one_sided": float(old["p_value_one_sided"]),
                "historical_holm_adjusted_p": float(old["holm_adjusted_p"]),
            }
        )
    return rows


def _calculate(
    details: pd.DataFrame,
    historical: pd.DataFrame,
    iterations: int,
    rng_seed: int,
    *,
    archive: Path | None = None,
    verify: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for window, frame in details.groupby("window_id", sort=True):
        panel, sums, counts, cell_points = _window_arrays(frame)
        schedule = make_schedule(panel, iterations=iterations, rng_seed=rng_seed)
        schedule_path = f"schedules/{window}.npz"
        draws_path = f"draws/{window}.npz"
        if archive is not None:
            if verify:
                stored = load_schedule(archive / schedule_path)
                for name in ("seed_weights", "fold_indices", "session_weights"):
                    if not np.array_equal(getattr(stored, name), getattr(schedule, name)):
                        raise ValueError(f"archived schedule drift: {window}/{name}")
            else:
                save_schedule(schedule, archive / schedule_path)
        draws = _draw_values(schedule, sums, counts)
        if archive is not None:
            if verify:
                with np.load(archive / draws_path, allow_pickle=False) as saved:
                    if set(saved.files) != {"draws", "contrasts"}:
                        raise ValueError(f"archived draw members drift: {window}")
                    if not np.array_equal(saved["draws"], draws) or not np.array_equal(
                        saved["contrasts"], np.asarray(panel.conditions)
                    ):
                        raise ValueError(f"archived draws drift: {window}")
            else:
                np.savez_compressed(
                    archive / draws_path,
                    draws=draws,
                    contrasts=np.asarray(panel.conditions),
                )
        result_rows.extend(
            _window_results(window, frame, historical, panel, cell_points, draws, rng_seed)
        )
        windows.append(
            {
                "window_id": window,
                "contrast_ids": list(panel.conditions),
                "seed_count": len(panel.seeds),
                "fold_count": len(panel.folds),
                "event_session_count": int(frame["session_id"].nunique()),
                "matched_set_count": int(frame["matched_set_id"].nunique()),
                "market_fingerprint": panel.market_fingerprint,
                "schedule_path": schedule_path,
                "draws_path": draws_path,
            }
        )
    result = pd.DataFrame(result_rows, columns=_RESULT_COLUMNS)
    if len(result) != 6:
        raise ValueError(f"expected six scheduled RQ4 results, found {len(result)}")
    return result, windows


def run_analysis(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> Path:
    if iterations < 2 or rng_seed < 0:
        raise ValueError("bootstrap iterations must be >=2 and rng seed non-negative")
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output exists; use --verify-only: {output_dir}")
    details, plan, historical = _load_inputs(source_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.attempt-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for folder in ("inputs", "implementation", "schedules", "draws"):
            (staging / folder).mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(source_dir / name, staging / "inputs" / name)
        implementation = {
            "script": Path(__file__).resolve(),
            "shared_schedule_core": REPO_ROOT / "scripts/rq123/shared_panel_bootstrap_core.py",
        }
        for key, path in implementation.items():
            shutil.copy2(path, staging / "implementation" / f"{key}.py")
        results, windows = _calculate(
            details, historical, int(iterations), int(rng_seed), archive=staging
        )
        (staging / "rq4_shared_market_results.csv").write_text(_csv_text(results), encoding="utf-8")
        (staging / "README.md").write_text(
            "# Chapter 3 RQ4 shared-market bootstrap\n\n"
            "Frozen scheduled-event matched differences are reanalysed without retraining, "
            "rematching, or editing thesis text. Each 10,000-draw replicate samples seed "
            "multiplicities once, fold occurrences once, and event-side CME sessions once "
            "per fold occurrence. The same market schedule applies to every selected seed "
            "and both paired comparisons. Each sampled seed-fold cell is a matched-set "
            "mean; cells have equal weight. The signed statistic is the event-minus-control "
            "increment in matched-LP advantage. The 95% interval is percentile based; "
            "one-sided probabilities are +1-corrected bootstrap sign-tail areas for a "
            "positive increment and adjusted within each window by Holm-2.\n\n"
            "`rq4_shared_market_results.csv` holds the new values beside historical values. "
            "`inputs/` preserves frozen sources; `schedules/` and `draws/` support exact "
            "replay. `manifest.json` records SHA-256 fingerprints. Run the script with "
            "`--verify-only` to regenerate and check every archived draw.\n",
            encoding="utf-8",
        )
        files = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        manifest = {
            "kind": METHOD_VERSION,
            "schema_version": 1,
            "schedule_method_version": SCHEDULE_METHOD_VERSION,
            "bootstrap_iterations": int(iterations),
            "bootstrap_rng_seed": int(rng_seed),
            "point_estimand": "equal_seed_fold_mean_of_matched_set_increment",
            "positive_direction": "greater_matched_LP_value_in_event",
            "source_directory": str(source_dir),
            "windows": windows,
            "files_sha256": {name: _sha256(staging / name) for name in files},
            "source_sha256": {name: _sha256(source_dir / name) for name in SOURCE_FILES},
            "implementation_sha256": {
                key: _sha256(path) for key, path in implementation.items()
            },
            "input_rows": {
                "matched_differences": len(details),
                "match_plan": len(plan),
                "historical_results": len(historical),
            },
        }
        _write_json(staging / "manifest.json", manifest)
        verify_archive(staging, source_dir=source_dir, check_code=True)
        staging.rename(output_dir)
    return output_dir


def verify_archive(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    source_dir: Path | None = None,
    check_code: bool = True,
) -> pd.DataFrame:
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != METHOD_VERSION or manifest.get("schedule_method_version") != SCHEDULE_METHOD_VERSION:
        raise ValueError("archive method version drift")
    for name, expected in manifest["files_sha256"].items():
        if _sha256(output_dir / name) != expected:
            raise ValueError(f"archived file hash drift: {name}")
    source_dir = Path(source_dir or manifest["source_directory"]).resolve()
    for name in SOURCE_FILES:
        if _sha256(source_dir / name) != manifest["source_sha256"][name]:
            raise ValueError(f"source file hash drift: {name}")
    if check_code:
        current = {
            "script": Path(__file__).resolve(),
            "shared_schedule_core": REPO_ROOT / "scripts/rq123/shared_panel_bootstrap_core.py",
        }
        for key, path in current.items():
            if _sha256(path) != manifest["implementation_sha256"][key]:
                raise ValueError(f"implementation hash drift: {key}")
    details, _, historical = _load_inputs(output_dir / "inputs")
    regenerated, windows = _calculate(
        details,
        historical,
        int(manifest["bootstrap_iterations"]),
        int(manifest["bootstrap_rng_seed"]),
        archive=output_dir,
        verify=True,
    )
    if windows != manifest["windows"]:
        raise ValueError("archive window metadata drift")
    if _csv_text(regenerated) != (output_dir / "rq4_shared_market_results.csv").read_text(encoding="utf-8"):
        raise ValueError("archived RQ4 result rows drift")
    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        results = verify_archive(args.output_dir, source_dir=args.source_dir)
        print(f"Verified {len(results)} RQ4 contrasts in {args.output_dir}")
    else:
        path = run_analysis(
            args.source_dir,
            args.output_dir,
            iterations=args.iterations,
            rng_seed=args.rng_seed,
        )
        print(f"Wrote and verified RQ4 shared-market archive: {path}")


if __name__ == "__main__":
    main()
