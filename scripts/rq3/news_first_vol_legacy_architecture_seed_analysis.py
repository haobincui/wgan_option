"""Q3-only paired analysis for the legacy-width architecture experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_legacy_architecture_seed as sweep
from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)


SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260822


class LegacyArchitectureAnalysisError(ValueError):
    pass


def _gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return path


def _csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].astype(str).nunique()),
        "sessions": int(frame["session_id"].astype(str).nunique()),
    }


def _panel(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = sweep._validate_root(root)
    workbook = root / "data_windows/tolerance_05m_pre_q4.xlsx"
    raw = pd.read_excel(workbook, sheet_name=str(resolved["datasets"]["sheet_name"]))
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    start = pd.Timestamp(resolved["split"]["development_train_end_utc"])
    end = pd.Timestamp(resolved["split"]["development_validation_end_utc"])
    selected = raw.loc[(timestamps >= start) & (timestamps < end)].copy()
    panel, lineage, _ = _load_panel_source(
        selected,
        sheet_name=str(resolved["datasets"]["sheet_name"]),
        panel_name="legacy_architecture_q3_common_05m",
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode="raw_joint",
    )
    expected = {"rows": 148, "pairs": 135, "sessions": 33}
    if _counts(panel) != expected:
        raise LegacyArchitectureAnalysisError(
            f"Q3 panel count drift: {_counts(panel)} != {expected}"
        )
    universe = sweep._payload_sha256(
        sorted(
            tuple(map(str, row))
            for row in panel[["session_id", "pair_id"]].itertuples(
                index=False, name=None
            )
        )
    )
    reference_contract = sweep._sweep(resolved)["external_reference"]
    if universe != reference_contract["q3_panel_universe_sha256"]:
        raise LegacyArchitectureAnalysisError("Q3 panel universe SHA drift")
    lineage.update(
        source_path=str(workbook.resolve()),
        source_sha256=sweep._sha256_file(workbook),
        interval_start_utc=str(start),
        interval_end_utc_exclusive=str(end),
        q4_rows_read=0,
        q4_loader_created=False,
        panel_universe_sha256=universe,
    )
    return panel, lineage


def _local_cells(root: Path) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in sweep._registry(root)["jobs"]]
    if len(jobs) != sweep.EXPECTED_LOCAL_JOBS:
        raise LegacyArchitectureAnalysisError("Local job matrix drift")
    rows = []
    for job in jobs:
        status = sweep._read_json(sweep._job_status_path(root, job["job_id"]))
        if not sweep._completed_local_job_valid(root, job, status):
            raise LegacyArchitectureAnalysisError(
                f"Incomplete local job: {job['job_id']}"
            )
        rows.append(
            {**job, "run_dir": status["run_dir"], "artifacts": status["artifacts"]}
        )
    return rows


def _reference_cells(root: Path) -> list[dict[str, Any]]:
    rows = sweep._validate_references(root)
    return [
        {
            **row,
            "job_spec_sha256": row["reference_spec_sha256"],
            "capacity_profile": "legacy",
            "support_mask_mode": "raw_joint",
        }
        for row in rows
    ]


def evaluation_cells(root: Path) -> list[dict[str, Any]]:
    rows = [*_local_cells(root), *_reference_cells(root)]
    keys = {
        (
            row["generator_conditioning_mode"],
            row["critic_conditioning_mode"],
            int(row["seed"]),
        )
        for row in rows
    }
    expected = {
        (generator, critic, seed)
        for generator in sweep.GENERATOR_MODES
        for critic in sweep.CRITIC_MODES
        for seed in sweep.SEEDS
    }
    if len(rows) != sweep.EXPECTED_CELLS or keys != expected:
        raise LegacyArchitectureAnalysisError("Combined 2x2x3 matrix drift")
    return rows


def _artifact(cell: Mapping[str, Any], role: str) -> tuple[Path, str]:
    artifacts = cell.get("artifacts") or {}
    if isinstance(artifacts, Mapping):
        rows = [artifacts[role]] if role in artifacts else []
    else:
        rows = [row for row in artifacts if row.get("artifact_role") == role]
    if len(rows) != 1:
        raise LegacyArchitectureAnalysisError(f"Expected one {role}: {cell['job_id']}")
    row = rows[0]
    path = Path(str(row["path"]))
    if not path.is_file() or sweep._sha256_file(path) != row["sha256"]:
        raise LegacyArchitectureAnalysisError(f"Artifact drift: {path}")
    return path, str(row["sha256"])


def _run_spec(cell: Mapping[str, Any]) -> tuple[RunSpec, str]:
    checkpoint, checkpoint_sha = _artifact(cell, "generator_best_learned")
    return (
        RunSpec(
            run_id=str(cell["job_id"]),
            run_dir=Path(str(cell["run_dir"])),
            model="wgan",
            tolerance_minutes=5,
            seed=int(cell["seed"]),
            checkpoint_path=checkpoint,
            text_ablation_mode="real_text",
            support_mask_mode="raw_joint",
            generator_current_input_mode="current_support_masked",
            metadata={
                "capacity_profile": "legacy",
                "generator_conditioning_mode": cell["generator_conditioning_mode"],
                "critic_conditioning_mode": cell["critic_conditioning_mode"],
                "external_reference": bool(cell.get("external_reference")),
            },
        ),
        checkpoint_sha,
    )


def _prediction_cache(
    root: Path,
    *,
    cell: Mapping[str, Any],
    spec: RunSpec,
    checkpoint_sha: str,
    panel: pd.DataFrame,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    directory = root / "analysis/predictions/q3"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cell['job_id']}.csv.gz"
    manifest_path = directory / f"{cell['job_id']}.manifest.json"
    universe = sweep._payload_sha256(
        sorted(
            tuple(map(str, row))
            for row in panel[["session_id", "pair_id"]].itertuples(
                index=False, name=None
            )
        )
    )
    contract = {
        "schema_version": 1,
        "job_id": cell["job_id"],
        "job_spec_sha256": cell["job_spec_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "panel_universe_sha256": universe,
        "mc_samples": 16,
        "generator_conditioning_mode": cell["generator_conditioning_mode"],
        "critic_conditioning_mode": cell["critic_conditioning_mode"],
        "seed": int(cell["seed"]),
        "external_reference": bool(cell.get("external_reference")),
        "q4_rows_read": 0,
    }
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise LegacyArchitectureAnalysisError("Partial prediction cache")
        manifest = sweep._read_json(manifest_path)
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "prediction_path",
                "prediction_sha256",
                "row_count",
                "manifest_sha256",
            }
        }
        if unsigned != contract:
            raise LegacyArchitectureAnalysisError("Prediction contract drift")
        signed = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if manifest.get("manifest_sha256") != sweep._payload_sha256(signed):
            raise LegacyArchitectureAnalysisError("Prediction manifest SHA drift")
        if sweep._sha256_file(path) != manifest["prediction_sha256"]:
            raise LegacyArchitectureAnalysisError("Prediction output SHA drift")
        return pd.read_csv(path, low_memory=False)
    predictions = evaluator(spec, "core", panel.copy())
    _gzip_csv(path, predictions)
    manifest = {
        **contract,
        "prediction_path": str(path.resolve()),
        "prediction_sha256": sweep._sha256_file(path),
        "row_count": len(predictions),
    }
    manifest["manifest_sha256"] = sweep._payload_sha256(manifest)
    sweep._write_json(manifest_path, manifest)
    return predictions


def _evaluate(
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    panel: pd.DataFrame,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    parts = []
    coverage = None
    for cell in cells:
        if bool(cell.get("external_reference")):
            prediction = sweep._require_mapping(
                cell.get("prediction"), "reference prediction"
            )
            prediction_path = Path(str(prediction["path"]))
            if sweep._sha256_file(prediction_path) != prediction["sha256"]:
                raise LegacyArchitectureAnalysisError(
                    f"Reference prediction drift: {cell['job_id']}"
                )
            predictions = pd.read_csv(prediction_path, low_memory=False)
            if len(predictions) != int(prediction["row_count"]):
                raise LegacyArchitectureAnalysisError(
                    f"Reference prediction row drift: {cell['job_id']}"
                )
            checkpoint_sha = str(cell["artifacts"]["generator_best_learned"]["sha256"])
            spec = RunSpec(
                run_id=str(cell["job_id"]),
                run_dir=Path(str(cell["source_run_dir"])),
                model="wgan",
                tolerance_minutes=5,
                seed=int(cell["seed"]),
                checkpoint_path=Path(
                    cell["artifacts"]["generator_best_learned"]["path"]
                ),
                text_ablation_mode="real_text",
                support_mask_mode="raw_joint",
                generator_current_input_mode="current_support_masked",
                metadata={"external_reference": True},
            )
        else:
            spec, checkpoint_sha = _run_spec(cell)
            predictions = _prediction_cache(
                root,
                cell=cell,
                spec=spec,
                checkpoint_sha=checkpoint_sha,
                panel=panel,
                evaluator=production,
            )
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            "core",
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise LegacyArchitectureAnalysisError(
                f"Prediction exclusions: {cell['job_id']}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        keys = frozenset(
            pairs[["session_id", "pair_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        coverage = keys if coverage is None else coverage
        if keys != coverage:
            raise LegacyArchitectureAnalysisError("Pair coverage differs across cells")
        pairs.insert(0, "job_id", cell["job_id"])
        pairs.insert(
            1, "generator_conditioning_mode", cell["generator_conditioning_mode"]
        )
        pairs.insert(2, "critic_conditioning_mode", cell["critic_conditioning_mode"])
        pairs.insert(
            3,
            "architecture_cell",
            sweep._architecture_slug(
                str(cell["generator_conditioning_mode"]),
                str(cell["critic_conditioning_mode"]),
            ),
        )
        pairs.insert(4, "external_reference", bool(cell.get("external_reference")))
        parts.append(pairs)
    result = pd.concat(parts, ignore_index=True)
    key = ["architecture_cell", "seed", "pair_id"]
    if (
        result.duplicated(key).any()
        or result[key[:-1]].drop_duplicates().shape[0] != 12
    ):
        raise LegacyArchitectureAnalysisError("Pair metric matrix drift")
    return result


def _two_level_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    value_column: str = "difference",
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    seeds = tuple(sorted(frame["seed"].astype(int).unique()))
    if seeds != sweep.SEEDS:
        raise LegacyArchitectureAnalysisError("Bootstrap requires all three seeds")
    rng = np.random.default_rng(seed)
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    points = []
    for train_seed in seeds:
        selected = frame[frame["seed"].astype(int).eq(train_seed)]
        grouped = selected.groupby("session_id", sort=True)[value_column].agg(
            ["sum", "count"]
        )
        sums = grouped["sum"].to_numpy(float)
        counts = grouped["count"].to_numpy(float)
        if len(sums) == 0:
            raise LegacyArchitectureAnalysisError("Empty session cluster")
        arrays[train_seed] = (sums, counts)
        points.append(float(sums.sum() / counts.sum()))
    draws = np.empty(int(replicates), dtype=float)
    for draw_index in range(int(replicates)):
        seed_means = []
        for chosen in rng.integers(0, len(seeds), size=len(seeds)):
            sums, counts = arrays[seeds[int(chosen)]]
            sampled = rng.integers(0, len(sums), size=len(sums))
            seed_means.append(float(sums[sampled].sum() / counts[sampled].sum()))
        draws[draw_index] = float(np.mean(seed_means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (np.sum(draws <= 0) + 1) / (len(draws) + 1)
    p_upper = (np.sum(draws >= 0) + 1) / (len(draws) + 1)
    return {
        "mean_difference": float(np.mean(points)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2 * min(p_lower, p_upper))),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _score_and_contrasts(
    pair_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed_cells = (
        pair_metrics.groupby(
            [
                "architecture_cell",
                "generator_conditioning_mode",
                "critic_conditioning_mode",
                "seed",
            ],
            sort=True,
        )
        .agg(
            model_mae=("model_mae", "mean"), persistence_mae=("persistence_mae", "mean")
        )
        .reset_index()
    )
    seed_cells["log_mae_ratio"] = np.log(
        seed_cells["model_mae"] / seed_cells["persistence_mae"]
    )
    scores = (
        seed_cells.groupby(
            [
                "architecture_cell",
                "generator_conditioning_mode",
                "critic_conditioning_mode",
            ],
            sort=True,
        )
        .agg(
            mean_model_mae=("model_mae", "mean"),
            mean_persistence_mae=("persistence_mae", "mean"),
            mean_log_mae_ratio=("log_mae_ratio", "mean"),
        )
        .reset_index()
    )
    scores["geometric_mae_ratio"] = np.exp(scores["mean_log_mae_ratio"])
    scores["improvement_vs_persistence_fraction"] = 1.0 - scores["geometric_mae_ratio"]
    scores = scores.sort_values(
        ["mean_log_mae_ratio", "architecture_cell"], kind="stable"
    ).reset_index(drop=True)
    anchor_slug = sweep._architecture_slug(*sweep.ANCHOR)
    keys = ["seed", "session_id", "pair_id"]
    anchor = pair_metrics[pair_metrics["architecture_cell"].eq(anchor_slug)][
        keys + ["model_mae"]
    ].rename(columns={"model_mae": "anchor_mae"})
    rows = []
    candidates = [
        sweep._architecture_slug(generator, critic)
        for generator in sweep.GENERATOR_MODES
        for critic in sweep.CRITIC_MODES
        if (generator, critic) != sweep.ANCHOR
    ]
    for index, candidate in enumerate(candidates):
        selected = pair_metrics[pair_metrics["architecture_cell"].eq(candidate)][
            keys + ["model_mae"]
        ].rename(columns={"model_mae": "candidate_mae"})
        paired = selected.merge(anchor, on=keys, validate="one_to_one")
        paired["difference"] = paired["candidate_mae"] - paired["anchor_mae"]
        by_seed = paired.groupby("seed", sort=True)[
            ["candidate_mae", "anchor_mae"]
        ].mean()
        nonworse = int((by_seed["candidate_mae"] <= by_seed["anchor_mae"]).sum())
        row = {
            "candidate_cell": candidate,
            "anchor_cell": anchor_slug,
            "nonworse_seed_count": nonworse,
            **_two_level_paired_bootstrap(
                paired,
                seed=BOOTSTRAP_SEED + index,
                replicates=BOOTSTRAP_REPLICATES,
            ),
        }
        rows.append(row)
    contrasts = pd.DataFrame(rows)
    contrasts["p_holm"] = holm_adjust(contrasts["p_two_sided"].tolist())
    contrasts["statistically_supported"] = (
        contrasts["mean_difference"].lt(0)
        & contrasts["ci_95_upper"].lt(0)
        & contrasts["p_holm"].lt(0.05)
        & contrasts["nonworse_seed_count"].ge(2)
    )
    leader = str(scores.iloc[0]["architecture_cell"])
    supported = contrasts.loc[
        contrasts["statistically_supported"], "candidate_cell"
    ].tolist()
    selection = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "selection_window": "2023Q3_common_5m",
        "anchor_cell": anchor_slug,
        "point_leader": leader,
        "statistically_supported_candidates": supported,
        "selection_status": "supported_candidate_exists"
        if supported
        else "completed_no_supported_architecture_change",
        "success_rule": "mean_lt_0_and_ci_upper_lt_0_and_holm_p_lt_0.05_and_at_least_2_of_3_seeds_nonworse",
        "holm_family_size": 3,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_method": "seed_then_paired_CME_session_cluster",
        "q4_read": False,
        "q4_used_for_selection": False,
    }
    return scores, contrasts, selection


def _persistence_contrasts(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = sorted(pair_metrics["architecture_cell"].astype(str).unique())
    if len(cells) != 4:
        raise LegacyArchitectureAnalysisError("Persistence family requires four cells")
    for index, cell in enumerate(cells):
        selected = pair_metrics[pair_metrics["architecture_cell"].eq(cell)].copy()
        selected["difference"] = selected["model_mae"] - selected["persistence_mae"]
        by_seed = selected.groupby("seed", sort=True)[
            ["model_mae", "persistence_mae"]
        ].mean()
        rows.append(
            {
                "architecture_cell": cell,
                "nonworse_seed_count": int(
                    (by_seed["model_mae"] <= by_seed["persistence_mae"]).sum()
                ),
                **_two_level_paired_bootstrap(
                    selected,
                    seed=BOOTSTRAP_SEED + 100 + index,
                    replicates=BOOTSTRAP_REPLICATES,
                ),
            }
        )
    output = pd.DataFrame(rows)
    output["p_holm"] = holm_adjust(output["p_two_sided"].tolist())
    output["statistically_supported"] = (
        output["mean_difference"].lt(0)
        & output["ci_95_upper"].lt(0)
        & output["p_holm"].lt(0.05)
        & output["nonworse_seed_count"].ge(2)
    )
    output["inference_role"] = "secondary_cell_vs_persistence_holm4"
    return output


def _secondary_factorial_effects(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "session_id", "pair_id"]
    values: dict[tuple[str, str], pd.DataFrame] = {}
    for generator in sweep.GENERATOR_MODES:
        for critic in sweep.CRITIC_MODES:
            cell = sweep._architecture_slug(generator, critic)
            selected = pair_metrics[pair_metrics["architecture_cell"].eq(cell)][
                keys + ["model_mae"]
            ].rename(columns={"model_mae": cell})
            values[(generator, critic)] = selected
    combined = values[(sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[0])]
    for key in (
        (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[1]),
        (sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[0]),
        (sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[1]),
    ):
        combined = combined.merge(values[key], on=keys, validate="one_to_one")
    y00 = combined[
        sweep._architecture_slug(sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[0])
    ]
    y01 = combined[
        sweep._architecture_slug(sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[1])
    ]
    y10 = combined[
        sweep._architecture_slug(sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[0])
    ]
    y11 = combined[
        sweep._architecture_slug(sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[1])
    ]
    effect_values = {
        "generator_main_film_minus_concat": 0.5 * ((y10 + y11) - (y00 + y01)),
        "critic_main_nolp_minus_lp": 0.5 * ((y01 + y11) - (y00 + y10)),
        "generator_by_critic_interaction": y11 - y10 - y01 + y00,
    }
    rows = []
    for index, (effect, values_array) in enumerate(effect_values.items()):
        selected = combined[keys].copy()
        selected["difference"] = values_array
        rows.append(
            {
                "factorial_effect": effect,
                **_two_level_paired_bootstrap(
                    selected,
                    seed=BOOTSTRAP_SEED + 200 + index,
                    replicates=BOOTSTRAP_REPLICATES,
                ),
            }
        )
    output = pd.DataFrame(rows)
    output["p_holm"] = holm_adjust(output["p_two_sided"].tolist())
    output["statistically_nonzero"] = (
        output["ci_95_lower"].gt(0) | output["ci_95_upper"].lt(0)
    ) & output["p_holm"].lt(0.05)
    output["inference_role"] = "secondary_factorial_effect_holm3"
    return output


def _training_diagnostics(cells: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        metrics_path, _ = _artifact(cell, "training_metrics_csv")
        metadata_path, _ = _artifact(cell, "best_learned_checkpoint")
        metrics = pd.read_csv(metrics_path)
        metadata = sweep._read_json(metadata_path)
        best_epoch = int(
            metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch", 0))
        )
        selected = metrics[metrics["epoch"].astype(int).eq(best_epoch)]
        learned = metrics[metrics["epoch"].astype(int).ge(1)]
        if best_epoch < 1 or len(selected) != 1 or learned.empty:
            raise LegacyArchitectureAnalysisError(f"Best epoch drift: {cell['job_id']}")
        row = selected.iloc[0]
        rows.append(
            {
                "job_id": cell["job_id"],
                "architecture_cell": sweep._architecture_slug(
                    str(cell["generator_conditioning_mode"]),
                    str(cell["critic_conditioning_mode"]),
                ),
                "seed": int(cell["seed"]),
                "external_reference": bool(cell.get("external_reference")),
                "best_learned_epoch": best_epoch,
                "best_g_lr": float(row["g_lr"]),
                "best_d_lr": float(row["d_lr"]),
                "best_gp": float(row["gp"]),
                "best_d_real_minus_fake": float(row["d_real"] - row["d_fake"]),
                "all_epoch_d_real_gt_fake_fraction": float(
                    (learned["d_real"] > learned["d_fake"]).mean()
                ),
                "best_g_adv": float(row["g_adv"]),
                "best_g_recon": float(row["g_recon"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["architecture_cell", "seed"], kind="stable")


def run_q3_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    root = Path(experiment_root).resolve()
    sweep._validate_root(root)
    panel, lineage = _panel(root)
    cells = evaluation_cells(root)
    pair_metrics = _evaluate(root, cells, panel, evaluator=evaluator)
    scores, contrasts, selection = _score_and_contrasts(pair_metrics)
    persistence = _persistence_contrasts(pair_metrics)
    factorial_effects = _secondary_factorial_effects(pair_metrics)
    diagnostics = _training_diagnostics(cells)
    analysis = root / "analysis"
    pair_path = _gzip_csv(
        analysis / "legacy_architecture_q3_pair_metrics.csv.gz", pair_metrics
    )
    score_path = _csv(analysis / "legacy_architecture_q3_cell_scores.csv", scores)
    contrast_path = _csv(
        analysis / "legacy_architecture_q3_candidate_anchor_contrasts.csv", contrasts
    )
    persistence_path = _csv(
        analysis / "legacy_architecture_q3_persistence_contrasts.csv", persistence
    )
    factorial_path = _csv(
        analysis / "legacy_architecture_q3_secondary_factorial_effects.csv",
        factorial_effects,
    )
    diagnostic_path = _csv(
        analysis / "legacy_architecture_training_diagnostics.csv", diagnostics
    )
    selection.update(
        {
            "panel_lineage": lineage,
            "panel_counts": _counts(panel),
            "local_training_job_count": sweep.EXPECTED_LOCAL_JOBS,
            "external_reference_count": sweep.EXPECTED_REFERENCES,
            "output_sha256": {
                pair_path.name: sweep._sha256_file(pair_path),
                score_path.name: sweep._sha256_file(score_path),
                contrast_path.name: sweep._sha256_file(contrast_path),
                persistence_path.name: sweep._sha256_file(persistence_path),
                factorial_path.name: sweep._sha256_file(factorial_path),
                diagnostic_path.name: sweep._sha256_file(diagnostic_path),
            },
        }
    )
    selection["payload_sha256"] = sweep._payload_sha256(selection)
    return sweep._write_json(
        analysis / "legacy_architecture_q3_selection.json", selection
    )


__all__ = [
    "LegacyArchitectureAnalysisError",
    "_score_and_contrasts",
    "_persistence_contrasts",
    "_secondary_factorial_effects",
    "_two_level_paired_bootstrap",
    "evaluation_cells",
    "run_q3_analysis",
]
