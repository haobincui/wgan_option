"""Three-seed capacity selection and frozen retrospective Q4 analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_film_nolp_capacity_seed as sweep
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
HISTORICAL_Q4_PREDICTIONS = (
    sweep.REPO_ROOT
    / "outputs/experiments/rq3_news_first_vol_training_q097_103_ttm07_38_v1"
    / "analysis/predictions/regression_05m_core.csv.gz"
)


class CapacitySeedAnalysisError(ValueError):
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


def _stage_jobs(root: Path, stage: str) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in sweep._registry(root)["jobs"]
        if job["experiment_stage"] == stage
    ]
    if len(jobs) != 36:
        raise CapacitySeedAnalysisError(f"Expected 36 {stage} jobs")
    keys = {
        (job["capacity_profile"], int(job["seed"]), int(job["tolerance_minutes"]))
        for job in jobs
    }
    expected = {
        (profile, seed, tolerance)
        for profile in sweep.PROFILES
        for seed in sweep.SEEDS
        for tolerance in sweep.TOLERANCES
    }
    if keys != expected:
        raise CapacitySeedAnalysisError(f"Incomplete {stage} matrix")
    sweep._validate_stage(root, stage)
    return jobs


def _panel(root: Path, *, q4: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = sweep._validate_root(root)
    if q4:
        registry = sweep._registry(root)
        workbook = Path(str(registry.get("q4_window_path", "")))
        expected = {"rows": 167, "pairs": 143, "sessions": 45}
        start, end = resolved["split"]["q4_start_utc"], resolved["split"]["q4_end_utc"]
        name = "film_nolp_capacity_q4_common_05m"
        if not registry.get("q4_gate_open") or not registry.get(
            "q4_window_materialized"
        ):
            raise CapacitySeedAnalysisError("Q4 gate is closed")
    else:
        workbook = root / "data_windows/pre_q4/tolerance_05m_pre_q4.xlsx"
        expected = {"rows": 148, "pairs": 135, "sessions": 33}
        start = resolved["split"]["development_train_end_utc"]
        end = resolved["split"]["development_validation_end_utc"]
        name = "film_nolp_capacity_q3_common_05m"
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    raw = pd.read_excel(workbook, sheet_name=str(resolved["datasets"]["sheet_name"]))
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    selected = raw.loc[
        (timestamps >= pd.Timestamp(start)) & (timestamps < pd.Timestamp(end))
    ].copy()
    panel, lineage, _ = _load_panel_source(
        selected,
        sheet_name=str(resolved["datasets"]["sheet_name"]),
        panel_name=name,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode="raw_joint",
    )
    if _counts(panel) != expected:
        raise CapacitySeedAnalysisError(
            f"Panel count drift: {_counts(panel)} != {expected}"
        )
    lineage.update(
        source_path=str(workbook.resolve()),
        source_sha256=sweep._sha256_file(workbook),
        interval_start_utc=start,
        interval_end_utc_exclusive=end,
        q4_rows_read_by_selection=0 if not q4 else len(panel),
    )
    return panel, lineage


def _artifact(status: Mapping[str, Any], role: str) -> tuple[Path, str]:
    rows = [
        row for row in status.get("artifacts", []) if row.get("artifact_role") == role
    ]
    if len(rows) != 1:
        raise CapacitySeedAnalysisError(f"Expected one {role}")
    path = Path(rows[0]["path"])
    if not path.is_file() or sweep._sha256_file(path) != rows[0]["sha256"]:
        raise CapacitySeedAnalysisError(f"Artifact drift: {path}")
    return path, str(rows[0]["sha256"])


def _run_spec(
    root: Path, job: Mapping[str, Any], checkpoint_role: str
) -> tuple[RunSpec, str]:
    status = sweep._read_json(sweep._job_status_path(root, str(job["job_id"])))
    checkpoint, checkpoint_sha = _artifact(status, checkpoint_role)
    return (
        RunSpec(
            run_id=str(job["job_id"]),
            run_dir=Path(status["run_dir"]),
            model="wgan",
            tolerance_minutes=int(job["tolerance_minutes"]),
            seed=int(job["seed"]),
            checkpoint_path=checkpoint,
            text_ablation_mode="real_text",
            support_mask_mode="raw_joint",
            generator_current_input_mode="current_support_masked",
            metadata={
                "capacity_profile": job["capacity_profile"],
                "generator_conditioning_mode": sweep.GENERATOR_MODE,
                "critic_conditioning_mode": sweep.CRITIC_MODE,
                "model_contract_sha256": job["model_contract_sha256"],
            },
        ),
        checkpoint_sha,
    )


def _prediction_cache(
    root: Path,
    *,
    stage_name: str,
    job: Mapping[str, Any],
    spec: RunSpec,
    checkpoint_sha: str,
    panel: pd.DataFrame,
    mc_samples: int,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    directory = root / "analysis/predictions" / stage_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job['job_id']}.csv.gz"
    manifest_path = directory / f"{job['job_id']}.manifest.json"
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
        "job_id": job["job_id"],
        "job_spec_sha256": job["job_spec_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "panel_universe_sha256": universe,
        "mc_samples": int(mc_samples),
        "capacity_profile": job["capacity_profile"],
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
    }
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise CapacitySeedAnalysisError("Partial prediction cache")
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
        if unsigned != contract or sweep._payload_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ) != manifest.get("manifest_sha256"):
            raise CapacitySeedAnalysisError("Prediction manifest contract drift")
        if sweep._sha256_file(path) != manifest["prediction_sha256"]:
            raise CapacitySeedAnalysisError("Prediction cache SHA drift")
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
    jobs: Sequence[Mapping[str, Any]],
    *,
    panel: pd.DataFrame,
    stage_name: str,
    checkpoint_role: str,
    mc_samples: int,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    production = evaluator or TrainedRunEvaluator(mc_samples=mc_samples)
    parts = []
    coverage = None
    for job in jobs:
        spec, checkpoint_sha = _run_spec(root, job, checkpoint_role)
        predictions = _prediction_cache(
            root,
            stage_name=stage_name,
            job=job,
            spec=spec,
            checkpoint_sha=checkpoint_sha,
            panel=panel,
            mc_samples=mc_samples,
            evaluator=production,
        )
        samples, exclusions, _ = compute_sample_metrics(
            spec, "core", panel, predictions, evaluate_embedded_atm_skew=False
        )
        if not exclusions.empty:
            raise CapacitySeedAnalysisError(f"Prediction exclusions: {job['job_id']}")
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
            raise CapacitySeedAnalysisError("Pair coverage differs across jobs")
        pairs.insert(0, "job_id", job["job_id"])
        pairs.insert(1, "capacity_profile", job["capacity_profile"])
        pairs.insert(2, "parameter_count", int(job["expected_wgan_parameters"]))
        pairs.insert(3, "checkpoint_sha256", checkpoint_sha)
        parts.append(pairs)
    output = pd.concat(parts, ignore_index=True)
    key = ["capacity_profile", "seed", "tolerance_minutes", "pair_id"]
    if (
        output.duplicated(key).any()
        or output[key[:-1]].drop_duplicates().shape[0] != 36
    ):
        raise CapacitySeedAnalysisError("Pair metric matrix drift")
    return output


def _training_diagnostics(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    import torch

    rows = []
    for job in jobs:
        status = sweep._read_json(sweep._job_status_path(root, str(job["job_id"])))
        metrics_path, _ = _artifact(status, "training_metrics_csv")
        metadata_path, _ = _artifact(status, "best_learned_checkpoint")
        generator_path, _ = _artifact(status, "generator_best_learned")
        metrics = pd.read_csv(metrics_path)
        metadata = sweep._read_json(metadata_path)
        best_epoch = int(
            metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch", 0))
        )
        learned = metrics[metrics["epoch"].astype(int).ge(1)].copy()
        selected = metrics[metrics["epoch"].astype(int).eq(best_epoch)]
        if best_epoch < 1 or len(selected) != 1 or learned.empty:
            raise CapacitySeedAnalysisError(
                f"Training diagnostics lost best epoch: {job['job_id']}"
            )
        checkpoint = torch.load(generator_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, Mapping):
            state = checkpoint.get(
                "state_dict", checkpoint.get("model_state_dict", checkpoint)
            )
        else:
            state = {}
        film_tensors = [
            tensor.detach().float().reshape(-1)
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and "film" in str(name).lower()
        ]
        film_rms = (
            float(torch.cat(film_tensors).square().mean().sqrt())
            if film_tensors
            else math.nan
        )
        row = selected.iloc[0]
        rows.append(
            {
                "job_id": job["job_id"],
                "capacity_profile": job["capacity_profile"],
                "parameter_count": int(job["expected_wgan_parameters"]),
                "seed": int(job["seed"]),
                "tolerance_minutes": int(job["tolerance_minutes"]),
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
                "best_g_calendar": float(row["g_calendar"]),
                "best_g_butterfly": float(row["g_butterfly"]),
                "best_g_smooth": float(row["g_smooth"]),
                "film_parameter_rms": film_rms,
                "critic_lp_feature_contract": "disabled_same_shape_exact_zero",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["parameter_count", "seed", "tolerance_minutes"], kind="stable"
    )


def _two_level(
    frame: pd.DataFrame,
    *,
    value_column: str,
    seed: int,
) -> dict[str, Any]:
    seeds = tuple(sorted(frame["seed"].astype(int).unique()))
    if seeds != sweep.SEEDS:
        raise CapacitySeedAnalysisError("Bootstrap lost a seed")
    rng = np.random.default_rng(seed)
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    point = []
    for train_seed in seeds:
        selected = frame[frame["seed"].astype(int).eq(train_seed)]
        grouped = selected.groupby("session_id", sort=True)[value_column].agg(
            ["sum", "count"]
        )
        arrays[train_seed] = (
            grouped["sum"].to_numpy(float),
            grouped["count"].to_numpy(float),
        )
        point.append(float(grouped["sum"].sum() / grouped["count"].sum()))
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        seed_means = []
        for chosen_index in rng.integers(0, len(seeds), size=len(seeds)):
            sums, counts = arrays[seeds[int(chosen_index)]]
            sampled = rng.integers(0, len(sums), size=len(sums))
            seed_means.append(float(sums[sampled].sum() / counts[sampled].sum()))
        draws[index] = float(np.mean(seed_means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (np.sum(draws <= 0) + 1) / (len(draws) + 1)
    p_upper = (np.sum(draws >= 0) + 1) / (len(draws) + 1)
    return {
        "mean_difference": float(np.mean(point)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2 * min(p_lower, p_upper))),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": int(seed),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _log_ratio_standard_error(frame: pd.DataFrame, *, seed: int) -> float:
    rng = np.random.default_rng(seed)
    arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for train_seed in sweep.SEEDS:
        selected = frame[frame["seed"].astype(int).eq(train_seed)]
        grouped = selected.groupby("session_id", sort=True).agg(
            model_sum=("model_mae", "sum"),
            persistence_sum=("persistence_mae", "sum"),
            pair_count=("pair_id", "size"),
        )
        arrays[train_seed] = (
            grouped["model_sum"].to_numpy(float),
            grouped["persistence_sum"].to_numpy(float),
            grouped["pair_count"].to_numpy(float),
        )
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        terms = []
        for chosen_index in rng.integers(0, len(sweep.SEEDS), size=len(sweep.SEEDS)):
            model, persistence, counts = arrays[sweep.SEEDS[int(chosen_index)]]
            sampled = rng.integers(0, len(model), size=len(model))
            model_mean = model[sampled].sum() / counts[sampled].sum()
            persistence_mean = persistence[sampled].sum() / counts[sampled].sum()
            terms.append(math.log(float(model_mean / persistence_mean)))
        draws[index] = float(np.mean(terms))
    return float(draws.std(ddof=1))


def _capacity_tables(
    pair_metrics: pd.DataFrame, *, primary_tolerance: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lane = pair_metrics[
        pair_metrics["tolerance_minutes"].astype(int).eq(primary_tolerance)
    ].copy()
    seed_cells = (
        lane.groupby(["capacity_profile", "parameter_count", "seed"], sort=True)
        .agg(
            model_mae=("model_mae", "mean"), persistence_mae=("persistence_mae", "mean")
        )
        .reset_index()
    )
    seed_cells["log_mae_ratio"] = np.log(
        seed_cells["model_mae"] / seed_cells["persistence_mae"]
    )
    score_rows = []
    persistence_rows = []
    for offset, profile in enumerate(sweep.PROFILES):
        selected = seed_cells[seed_cells["capacity_profile"].eq(profile)]
        mean_log = float(selected["log_mae_ratio"].mean())
        raw = lane[lane["capacity_profile"].eq(profile)].copy()
        raw["difference"] = raw["model_mae"] - raw["persistence_mae"]
        inference = _two_level(
            raw, value_column="difference", seed=BOOTSTRAP_SEED + offset
        )
        nonworse = int((selected["model_mae"] <= selected["persistence_mae"]).sum())
        score_rows.append(
            {
                "capacity_profile": profile,
                "parameter_count": int(selected["parameter_count"].iloc[0]),
                "mean_log_mae_ratio": mean_log,
                "geometric_mae_ratio": math.exp(mean_log),
                "improvement_fraction": 1.0 - math.exp(mean_log),
                "seed_count": 3,
                "nonworse_seed_count": nonworse,
            }
        )
        persistence_rows.append({"capacity_profile": profile, **inference})
    scores = (
        pd.DataFrame(score_rows)
        .sort_values(["mean_log_mae_ratio", "parameter_count"], kind="stable")
        .reset_index(drop=True)
    )
    persistence = pd.DataFrame(persistence_rows)
    persistence["p_holm"] = holm_adjust(persistence["p_two_sided"].tolist())
    persistence["statistically_supported"] = (
        persistence["mean_difference"].lt(0)
        & persistence["ci_95_upper"].lt(0)
        & persistence["p_holm"].lt(0.05)
        & persistence["capacity_profile"]
        .map(dict(zip(scores["capacity_profile"], scores["nonworse_seed_count"])))
        .ge(2)
    )
    pairwise_rows = []
    for left_index, left in enumerate(sweep.PROFILES):
        for right_index, right in enumerate(
            sweep.PROFILES[left_index + 1 :], start=left_index + 1
        ):
            keys = ["seed", "session_id", "pair_id"]
            lhs = lane[lane["capacity_profile"].eq(left)][keys + ["model_mae"]].rename(
                columns={"model_mae": "left"}
            )
            rhs = lane[lane["capacity_profile"].eq(right)][keys + ["model_mae"]].rename(
                columns={"model_mae": "right"}
            )
            paired = lhs.merge(rhs, on=keys, validate="one_to_one")
            paired["difference"] = paired["left"] - paired["right"]
            pairwise_rows.append(
                {
                    "left_capacity_profile": left,
                    "right_capacity_profile": right,
                    **_two_level(
                        paired,
                        value_column="difference",
                        seed=BOOTSTRAP_SEED + 100 + 10 * left_index + right_index,
                    ),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["p_holm"] = holm_adjust(pairwise["p_two_sided"].tolist())
    leader = str(scores.iloc[0]["capacity_profile"])
    best_score = float(scores.iloc[0]["mean_log_mae_ratio"])
    # The leader SE uses the same log(ratio of pair-balanced MAEs) estimand as
    # the point score, resampling seeds then whole CME sessions.
    leader_raw = lane[lane["capacity_profile"].eq(leader)].copy()
    leader_se = _log_ratio_standard_error(leader_raw, seed=BOOTSTRAP_SEED + 999)
    threshold = best_score + leader_se
    eligible = scores[scores["mean_log_mae_ratio"].le(threshold + 1e-15)]
    candidate = str(
        eligible.sort_values("parameter_count", kind="stable").iloc[0][
            "capacity_profile"
        ]
    )
    support = bool(
        persistence.loc[
            persistence["capacity_profile"].eq(candidate), "statistically_supported"
        ].iloc[0]
    )
    selection = {
        "point_leader": leader,
        "one_se_candidate": candidate,
        "best_mean_log_mae_ratio": best_score,
        "leader_bootstrap_se": leader_se,
        "one_se_threshold": threshold,
        "candidate_statistical_support": "statistically_supported"
        if support
        else "descriptive_only",
        "q3_used_for_capacity_selection": True,
        "q4_used_for_capacity_selection": False,
    }
    return scores, persistence, pairwise, selection


def run_q3_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    root = Path(experiment_root).resolve()
    registry = sweep._registry(root)
    if registry.get("selection_frozen"):
        raise CapacitySeedAnalysisError("Selection is already frozen")
    jobs = _stage_jobs(root, sweep.DEVELOPMENT_STAGE)
    panel, lineage = _panel(root, q4=False)
    pairs = _evaluate(
        root,
        jobs,
        panel=panel,
        stage_name="q3_development_best_learned",
        checkpoint_role="generator_best_learned",
        mc_samples=16,
        evaluator=evaluator,
    )
    scores, persistence, pairwise, selection = _capacity_tables(
        pairs, primary_tolerance=5
    )
    diagnostics = _training_diagnostics(root, jobs)
    analysis = root / "analysis"
    pair_path = _gzip_csv(analysis / "film_nolp_capacity_q3_pair_metrics.csv.gz", pairs)
    score_path = _csv(analysis / "film_nolp_capacity_q3_scores.csv", scores)
    persistence_path = _csv(
        analysis / "film_nolp_capacity_q3_persistence_contrasts.csv", persistence
    )
    pairwise_path = _csv(
        analysis / "film_nolp_capacity_q3_pairwise_contrasts.csv", pairwise
    )
    diagnostics_path = _csv(
        analysis / "film_nolp_capacity_training_diagnostics.csv", diagnostics
    )
    payload = {
        "schema_version": 1,
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "selection_panel": "common_05m_q3",
        "selection_tolerance_minutes": 5,
        "training_seeds": list(sweep.SEEDS),
        "capacity_profiles": list(sweep.PROFILES),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lineage": lineage,
        **selection,
        "artifact_sha256": {
            pair_path.name: sweep._sha256_file(pair_path),
            score_path.name: sweep._sha256_file(score_path),
            persistence_path.name: sweep._sha256_file(persistence_path),
            pairwise_path.name: sweep._sha256_file(pairwise_path),
            diagnostics_path.name: sweep._sha256_file(diagnostics_path),
        },
        "created_at_utc": sweep._utc_now(),
    }
    payload["selection_sha256"] = sweep._payload_sha256(payload)
    return sweep._write_json(analysis / "film_nolp_capacity_q3_selection.json", payload)


def _recipe_from_job(
    root: Path, job: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = sweep._read_json(sweep._job_status_path(root, str(job["job_id"])))
    metadata_path, metadata_sha = _artifact(status, "best_learned_checkpoint")
    generator_path, generator_sha = _artifact(status, "generator_best_learned")
    discriminator_path, discriminator_sha = _artifact(
        status, "discriminator_best_learned"
    )
    metadata = sweep._read_json(metadata_path)
    epoch = int(metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch", 0)))
    if epoch < 1:
        raise CapacitySeedAnalysisError(f"Invalid best learned epoch: {job['job_id']}")

    def trace(name: str) -> list[dict[str, Any]]:
        rows = [
            {"epoch": int(row["epoch"]), "lr": float(row["lr"])}
            for row in metadata.get(name, [])
            if 1 <= int(row["epoch"]) <= epoch
        ]
        if [row["epoch"] for row in rows] != list(range(1, epoch + 1)):
            raise CapacitySeedAnalysisError(f"Incomplete {name}: {job['job_id']}")
        return rows

    recipe = {
        "schema_version": 1,
        "refit_mode": sweep.REFIT_MODE,
        "num_epochs": epoch,
        "generator_lr_trace": trace("generator_lr_trace"),
        "discriminator_lr_trace": trace("discriminator_lr_trace"),
    }
    sweep.factorial._validate_refit_recipe(recipe)
    lineage = {
        "development_job_id": job["job_id"],
        "capacity_profile": job["capacity_profile"],
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "num_epochs": epoch,
        "best_learned_metadata_path": str(metadata_path.resolve()),
        "best_learned_metadata_sha256": metadata_sha,
        "best_learned_generator_sha256": generator_sha,
        "best_learned_discriminator_sha256": discriminator_sha,
        "training_config_sha256": job["config_sha256"],
        "job_spec_sha256": job["job_spec_sha256"],
    }
    return recipe, lineage


def freeze_refit_recipes(experiment_root: str | Path, selection: str | Path) -> Path:
    root = Path(experiment_root).resolve()
    selection_path = Path(selection).resolve()
    payload = sweep._read_json(selection_path)
    unsigned = {
        key: value for key, value in payload.items() if key != "selection_sha256"
    }
    if sweep._payload_sha256(unsigned) != payload.get("selection_sha256"):
        raise CapacitySeedAnalysisError("Selection self-hash drift")
    directory = root / "analysis/refit_recipes"
    directory.mkdir(parents=True, exist_ok=True)
    existing = {path.stem: path for path in directory.glob("*.json")}
    expected_ids = {
        str(job["job_id"]) for job in _stage_jobs(root, sweep.DEVELOPMENT_STAGE)
    }
    if set(existing) - expected_ids:
        raise CapacitySeedAnalysisError(
            "Refit recipe directory contains an unknown job"
        )
    rows = []
    for job in _stage_jobs(root, sweep.DEVELOPMENT_STAGE):
        recipe, lineage = _recipe_from_job(root, job)
        path = directory / f"{job['job_id']}.json"
        if path.is_file():
            if sweep._read_json(path) != recipe:
                raise CapacitySeedAnalysisError(
                    f"Existing refit recipe differs: {job['job_id']}"
                )
        else:
            sweep._write_json(path, recipe)
        rows.append(
            {
                **lineage,
                "recipe_path": str(path.resolve()),
                "recipe_sha256": sweep._sha256_file(path),
                "generator_lr_trace_sha256": sweep._payload_sha256(
                    recipe["generator_lr_trace"]
                ),
                "discriminator_lr_trace_sha256": sweep._payload_sha256(
                    recipe["discriminator_lr_trace"]
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "refit_mode": sweep.REFIT_MODE,
        "selection_path": str(selection_path),
        "selection_sha256": sweep._sha256_file(selection_path),
        "selection_payload_sha256": payload["selection_sha256"],
        "recipe_count": len(rows),
        "recipes": rows,
    }
    manifest["manifest_sha256"] = sweep._payload_sha256(manifest)
    return sweep._write_json(root / "analysis/refit_recipe_manifest.json", manifest)


def _historical_q4_overlap(pairs: pd.DataFrame) -> dict[str, Any]:
    if not HISTORICAL_Q4_PREDICTIONS.is_file():
        raise FileNotFoundError(HISTORICAL_Q4_PREDICTIONS)
    historical = pd.read_csv(HISTORICAL_Q4_PREDICTIONS, low_memory=False)
    current_ids = set(pairs["pair_id"].astype(str))
    if "pair_id" in historical:
        historical_ids = set(historical["pair_id"].astype(str))
    else:
        historical_ids = set()
    overlap = current_ids & historical_ids
    if len(current_ids) != 143 or len(overlap) != 143:
        raise CapacitySeedAnalysisError("Historical Q4 overlap must be 143/143")
    return {
        "historical_prediction_path": str(HISTORICAL_Q4_PREDICTIONS.resolve()),
        "historical_prediction_sha256": sweep._sha256_file(HISTORICAL_Q4_PREDICTIONS),
        "current_pair_count": len(current_ids),
        "overlapping_pair_count": len(overlap),
        "pair_overlap_label": "143/143",
        "historically_exposed": True,
        "interpretation": "retrospective_frozen_exploratory_not_confirmatory",
    }


def run_q4_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    root = Path(experiment_root).resolve()
    registry = sweep._registry(root)
    if (
        not registry.get("selection_frozen")
        or not registry.get("refit_complete")
        or not registry.get("q4_gate_open")
    ):
        raise CapacitySeedAnalysisError("Q4 upstream freeze is incomplete")
    sweep._validate_frozen_selection(root)
    sweep._validate_allowlist(root)
    jobs = _stage_jobs(root, sweep.REFIT_STAGE)
    panel, lineage = _panel(root, q4=True)
    pairs = _evaluate(
        root,
        jobs,
        panel=panel,
        stage_name="q4_refit_final",
        checkpoint_role="generator_final",
        mc_samples=64,
        evaluator=evaluator,
    )
    primary = pairs[pairs["tolerance_minutes"].astype(int).eq(5)].copy()
    secondary = pairs[pairs["tolerance_minutes"].astype(int).eq(30)].copy()
    scores, persistence, pairwise, q4_ranking = _capacity_tables(
        primary, primary_tolerance=5
    )
    secondary_scores, secondary_persistence, secondary_pairwise, _ = _capacity_tables(
        secondary.assign(tolerance_minutes=5), primary_tolerance=5
    )
    analysis = root / "analysis"
    pair_path = _gzip_csv(analysis / "film_nolp_capacity_q4_pair_metrics.csv.gz", pairs)
    score_path = _csv(analysis / "film_nolp_capacity_q4_scores.csv", scores)
    persistence_path = _csv(
        analysis / "film_nolp_capacity_q4_persistence_contrasts.csv", persistence
    )
    pairwise_path = _csv(
        analysis / "film_nolp_capacity_q4_pairwise_contrasts.csv", pairwise
    )
    secondary_path = _csv(
        analysis / "film_nolp_capacity_q4_30m_secondary_scores.csv", secondary_scores
    )
    secondary_persistence_path = _csv(
        analysis / "film_nolp_capacity_q4_30m_secondary_persistence.csv",
        secondary_persistence,
    )
    secondary_pairwise_path = _csv(
        analysis / "film_nolp_capacity_q4_30m_secondary_pairwise.csv",
        secondary_pairwise,
    )
    summary = {
        "schema_version": 1,
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "q4_loader_created": True,
        "q4_predictions_generated": True,
        "q4_evaluated": True,
        "q4_used_for_selection": False,
        "q4_confirmation_label": "retrospective_frozen_exploratory",
        "frozen_q3_selection_sha256": registry["selection_sha256"],
        "q4_descriptive_ranking": q4_ranking,
        "panel_lineage": lineage,
        "historical_q4_exposure": _historical_q4_overlap(pairs),
        "artifact_sha256": {
            path.name: sweep._sha256_file(path)
            for path in (
                pair_path,
                score_path,
                persistence_path,
                pairwise_path,
                secondary_path,
                secondary_persistence_path,
                secondary_pairwise_path,
            )
        },
        "created_at_utc": sweep._utc_now(),
    }
    summary["summary_sha256"] = sweep._payload_sha256(summary)
    return sweep._write_json(analysis / "film_nolp_capacity_q4_summary.json", summary)


__all__ = ["freeze_refit_recipes", "run_q3_analysis", "run_q4_analysis"]
