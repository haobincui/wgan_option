"""Q2 selection and frozen-Q3 analysis for the independent 8x8 WGAN sweep."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_wgan_grid08_sweep as sweep
from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)
from wgan_option.utils.text_ablation import REAL_TEXT


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260820


class Grid08AnalysisError(ValueError):
    pass


def _gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return path


def _jobs(root: Path, *, stage: str | None = None) -> list[dict[str, Any]]:
    registry = sweep._load_registry(root)
    jobs = [dict(job) for job in registry["jobs"]]
    if stage is not None:
        jobs = [job for job in jobs if job["experiment_stage"] == stage]
    return jobs


def _status(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    value = sweep._read_json(sweep._job_status_path(root, job["job_id"]))
    if not sweep._completed_valid(job, value):
        raise Grid08AnalysisError(f"Incomplete or hash-invalid job: {job['job_id']}")
    return value


def _run_spec(root: Path, job: Mapping[str, Any]) -> RunSpec:
    status = _status(root, job)
    run_dir = Path(status["run_dir"])
    checkpoint = run_dir / "checkpoints" / "generator_best_learned.pt"
    if not checkpoint.is_file():
        raise Grid08AnalysisError(f"Missing best-learned generator: {job['job_id']}")
    return RunSpec(
        run_id=str(job["job_id"]),
        run_dir=run_dir,
        model="wgan",
        tolerance_minutes=int(job["tolerance_minutes"]),
        seed=int(job["seed"]),
        checkpoint_path=checkpoint,
        text_ablation_mode=str(job["text_ablation_mode"]),
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={
            "capacity_profile": job["capacity_profile"],
            "lr_profile": job["lr_profile"],
            "surface_grid_sha256": job["surface_grid_sha256"],
        },
    )


def _panel(root: Path, *, panel: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = sweep._validate_root_lineage(root)
    workbook = (
        root / "data_windows" / "tolerance_05m_train_q2_only.xlsx"
        if panel == "q2"
        else sweep._dataset_path(resolved, 5)
    )
    raw = pd.read_excel(workbook, sheet_name=resolved["datasets"]["sheet_name"])
    timestamps = pd.to_datetime(raw["effective_origin_utc"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise Grid08AnalysisError("Panel contains invalid timestamps")
    if panel == "q2":
        start = pd.Timestamp(resolved["split"]["train_end_utc"])
        end = pd.Timestamp(resolved["split"]["validation_end_utc"])
        expected = sweep.EXPECTED_PANEL_COUNTS["common_q2"]
    elif panel == "q3":
        start = pd.Timestamp(resolved["split"]["q3_start_utc"])
        end = pd.Timestamp(resolved["split"]["q3_end_utc"])
        expected = sweep.EXPECTED_PANEL_COUNTS["common_q3"]
    else:
        raise Grid08AnalysisError(f"Unknown panel: {panel}")
    selected = raw.loc[(timestamps >= start) & (timestamps < end)].copy()
    output, lineage, _ = _load_panel_source(
        selected,
        sheet_name=str(resolved["datasets"]["sheet_name"]),
        panel_name=f"grid08_common_{panel}",
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode="raw_joint",
    )
    pairs = int(output["pair_id"].astype(str).nunique())
    sessions = int(output["session_id"].astype(str).nunique())
    if (pairs, sessions) != (expected["pairs"], expected["sessions"]):
        raise Grid08AnalysisError(
            f"{panel} panel drift: {(pairs, sessions)} != "
            f"{(expected['pairs'], expected['sessions'])}"
        )
    lineage.update(
        {
            "path": str(workbook.resolve()),
            "sha256": sweep._sha256_file(workbook),
            "interval_start_utc": start.isoformat(),
            "interval_end_utc_exclusive": end.isoformat(),
            "pair_count": pairs,
            "session_count": sessions,
            "q3_used_for_selection": False,
            "q4_rows_passed_to_evaluator": 0,
        }
    )
    return output, lineage


def _evaluate_jobs(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    panel_name: str,
    mc_samples: int,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, lineage = _panel(root, panel=panel_name)
    production = evaluator or TrainedRunEvaluator(mc_samples=mc_samples)
    pair_parts = []
    prediction_dir = root / "analysis" / f"{panel_name}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for raw in jobs:
        job = dict(raw)
        spec = _run_spec(root, job)
        prediction_path = prediction_dir / f"{job['job_id']}.csv.gz"
        if prediction_path.is_file():
            predictions = pd.read_csv(prediction_path, low_memory=False)
        else:
            predictions = production(spec, "core", panel.copy())
            _gzip_csv(prediction_path, predictions)
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            f"grid08_common_{panel_name}",
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise Grid08AnalysisError(
                f"Prediction exclusions for {job['job_id']}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        pairs.insert(0, "capacity_profile", job["capacity_profile"])
        pairs.insert(1, "lr_profile", job["lr_profile"])
        pairs.insert(
            2, "architecture_profile_sha256", job["architecture_profile_sha256"]
        )
        pairs.insert(3, "model_contract_sha256", job["model_contract_sha256"])
        pairs.insert(4, "surface_grid_sha256", job["surface_grid_sha256"])
        pair_parts.append(pairs)
    output = pd.concat(pair_parts, ignore_index=True)
    keys = [
        "capacity_profile",
        "lr_profile",
        "seed",
        "text_ablation_mode",
        "tolerance_minutes",
        "pair_id",
    ]
    if output.duplicated(keys).any():
        raise Grid08AnalysisError("Duplicate pair metrics key")
    coverage = {
        frozenset(
            group[["pair_id", "session_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        for _, group in output.groupby(keys[:-1], sort=False)
    }
    if len(coverage) != 1:
        raise Grid08AnalysisError("All evaluated jobs must share paired panel coverage")
    lineage["evaluated_run_count"] = len(jobs)
    return output.reset_index(drop=True), lineage


def _session_arrays(
    group: pd.DataFrame, columns: Sequence[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sessions = sorted(group["session_id"].astype(str).unique())
    sums = np.zeros((len(sessions), len(columns)), dtype=np.float64)
    counts = np.zeros(len(sessions), dtype=np.float64)
    for index, session in enumerate(sessions):
        part = group[group["session_id"].astype(str).eq(session)]
        values = part[list(columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise Grid08AnalysisError("Non-finite pair metric in bootstrap")
        sums[index] = values.sum(axis=0)
        counts[index] = len(part)
    return sessions, sums, counts


def _two_level_draws(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    statistic: Callable[[np.ndarray], np.ndarray],
    seed: int,
    iterations: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, np.ndarray]:
    seeds = sorted(frame["seed"].astype(int).unique())
    if seeds != list(sweep.SEEDS):
        raise Grid08AnalysisError(f"Bootstrap requires all three seeds: {seeds}")
    rng = np.random.default_rng(int(seed))
    observed_by_seed = []
    draws_by_seed = []
    for model_seed in seeds:
        group = frame[frame["seed"].astype(int).eq(model_seed)]
        _, sums, counts = _session_arrays(group, columns)
        observed_by_seed.append(
            statistic(sums.sum(axis=0, keepdims=True) / counts.sum())[0]
        )
        selected = rng.integers(0, len(sums), size=(iterations, len(sums)))
        means = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]
        draws_by_seed.append(statistic(means))
    observed = float(np.mean(observed_by_seed))
    matrix = np.stack(draws_by_seed, axis=1)
    seed_selection = rng.integers(0, len(seeds), size=(iterations, len(seeds)))
    draws = np.take_along_axis(matrix, seed_selection, axis=1).mean(axis=1)
    return observed, draws


def _summary(observed: float, draws: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(draws, [0.025, 0.975])
    centered = draws - observed
    p = (float(np.sum(np.abs(centered) >= abs(observed))) + 1.0) / (len(draws) + 1.0)
    return {
        "estimate": observed,
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": p,
    }


def _config_bootstrap(group: pd.DataFrame, *, offset: int) -> dict[str, Any]:
    def score(values: np.ndarray) -> np.ndarray:
        return np.log(values[:, 0] / values[:, 1])

    def gap(values: np.ndarray) -> np.ndarray:
        return values[:, 0] - values[:, 1]

    observed_score, score_draws = _two_level_draws(
        group,
        columns=("model_mae", "persistence_mae"),
        statistic=score,
        seed=BOOTSTRAP_SEED + offset,
    )
    observed_gap, gap_draws = _two_level_draws(
        group,
        columns=("model_mae", "persistence_mae"),
        statistic=gap,
        seed=BOOTSTRAP_SEED + 1000 + offset,
    )
    score_summary = _summary(observed_score, score_draws)
    gap_summary = _summary(observed_gap, gap_draws)
    return {
        "score": score_summary["estimate"],
        "score_bootstrap_se": score_summary["bootstrap_se"],
        "score_ci_95_lower": score_summary["ci_95_lower"],
        "score_ci_95_upper": score_summary["ci_95_upper"],
        "mae_gap": gap_summary["estimate"],
        "ci_95_lower": gap_summary["ci_95_lower"],
        "ci_95_upper": gap_summary["ci_95_upper"],
        "p_two_sided": gap_summary["p_two_sided"],
        "mae_ratio": math.exp(observed_score),
        "improvement_fraction": 1.0 - math.exp(observed_score),
    }


def _training_diagnostics(
    root: Path, jobs: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    rows = []
    for job in jobs:
        status = _status(root, job)
        run_dir = Path(status["run_dir"])
        metrics = sweep._read_json(run_dir / "metrics" / "training_metrics.json")
        best = sweep._read_json(run_dir / "metrics" / "best_learned_checkpoint.json")
        epoch = int(best.get("best_epoch", best.get("best_learned_epoch_ge_1", 0)))
        learned = next(row for row in metrics if int(row["epoch"]) == epoch)
        trained = [row for row in metrics if int(row["epoch"]) >= 1]
        correct = [float(row["d_real"]) > float(row["d_fake"]) for row in trained]
        rows.append(
            {
                "job_id": job["job_id"],
                "capacity_profile": job["capacity_profile"],
                "lr_profile": job["lr_profile"],
                "seed": job["seed"],
                "tolerance_minutes": job["tolerance_minutes"],
                "best_learned_epoch": epoch,
                "baseline_selected_epoch0": int(
                    sweep._read_json(run_dir / "metrics" / "best_checkpoint.json").get(
                        "best_epoch", -1
                    )
                )
                == 0,
                "best_gp": float(learned.get("gp", float("nan"))),
                "best_d_real": float(learned.get("d_real", float("nan"))),
                "best_d_fake": float(learned.get("d_fake", float("nan"))),
                "critic_correct_order_epoch_fraction": float(np.mean(correct)),
                "best_g_adv": float(learned.get("g_adv", float("nan"))),
                "best_g_recon": float(learned.get("g_recon", float("nan"))),
                "best_g_calendar": float(learned.get("g_calendar", float("nan"))),
                "best_g_butterfly": float(learned.get("g_butterfly", float("nan"))),
                "best_g_smooth": float(learned.get("g_smooth", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _interaction_table(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    anchor_capacity = "small"
    anchor_lr = "lr_5e_07"
    rows = []
    offset = 0
    for capacity in sweep.PROFILES:
        if capacity == anchor_capacity:
            continue
        for lr_profile in sweep.LR_PROFILES:
            if lr_profile == anchor_lr:
                continue
            selected = {}
            for key, cap, lr in (
                ("xy", capacity, lr_profile),
                ("x0", capacity, anchor_lr),
                ("0y", anchor_capacity, lr_profile),
                ("00", anchor_capacity, anchor_lr),
            ):
                part = pair_metrics[
                    pair_metrics["capacity_profile"].eq(cap)
                    & pair_metrics["lr_profile"].eq(lr)
                ][["seed", "session_id", "pair_id", "model_mae"]].copy()
                selected[key] = part.rename(columns={"model_mae": key})
            merged = selected["xy"]
            for key in ("x0", "0y", "00"):
                merged = merged.merge(
                    selected[key],
                    on=["seed", "session_id", "pair_id"],
                    validate="one_to_one",
                )

            def did(values: np.ndarray) -> np.ndarray:
                return values[:, 0] - values[:, 1] - values[:, 2] + values[:, 3]

            observed, draws = _two_level_draws(
                merged,
                columns=("xy", "x0", "0y", "00"),
                statistic=did,
                seed=BOOTSTRAP_SEED + 3000 + offset,
            )
            summary = _summary(observed, draws)
            rows.append(
                {
                    "capacity_profile": capacity,
                    "lr_profile": lr_profile,
                    "anchor_capacity_profile": anchor_capacity,
                    "anchor_lr_profile": anchor_lr,
                    "did_mae": observed,
                    **{
                        key: value
                        for key, value in summary.items()
                        if key != "estimate"
                    },
                }
            )
            offset += 1
    if len(rows) != 20:
        raise AssertionError(
            "Capacity x LR interaction family must contain 20 contrasts"
        )
    adjusted = holm_adjust([row["p_two_sided"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
    return pd.DataFrame(rows)


def _selection_from_scores(scores: pd.DataFrame, root: Path) -> dict[str, Any]:
    ordered = scores.sort_values(
        ["score", "expected_wgan_parameters", "initial_learning_rate"], kind="stable"
    ).reset_index(drop=True)
    leader = ordered.iloc[0]
    threshold = float(leader["score"] + leader["score_bootstrap_se"])
    one_se = ordered[ordered["score"] <= threshold].sort_values(
        ["expected_wgan_parameters", "initial_learning_rate", "score"], kind="stable"
    )
    candidate = one_se.iloc[0]
    support = bool(
        float(candidate["ci_95_upper"]) < 0.0
        and float(candidate["holm_p"]) < 0.05
        and int(candidate["non_worse_seed_count"]) >= 2
    )
    payload = {
        "schema_version": 1,
        "selection_panel": "grid08_common_q2_05m_real_text",
        "selection_uses_tolerance_minutes": [5],
        "q3_used_for_selection": False,
        "q4_used_for_selection": False,
        "point_leader_capacity_profile": str(leader["capacity_profile"]),
        "point_leader_lr_profile": str(leader["lr_profile"]),
        "point_leader_score": float(leader["score"]),
        "one_se_threshold": threshold,
        "one_se_candidate_capacity_profile": str(candidate["capacity_profile"]),
        "one_se_candidate_lr_profile": str(candidate["lr_profile"]),
        "one_se_candidate_score": float(candidate["score"]),
        "one_se_candidate_statistical_support": support,
        "selection_label": "statistically_supported" if support else "descriptive_only",
        "primary_job_count": 180,
        "selection_input_sha256": sweep._sha256_file(
            root / "analysis" / "grid08_capacity_lr_scores.csv"
        ),
    }
    payload["selection_sha256"] = sweep._payload_sha256(payload)
    sweep._write_json(root / "analysis" / "grid08_selection.json", payload)
    return payload


def run_grid08_q2_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    resolved = sweep._validate_root_lineage(root)
    registry = sweep._load_registry(root)
    if registry.get("q3_unlocked"):
        raise Grid08AnalysisError("Q2 selection cannot be recomputed after Q3 unlock")
    jobs = _jobs(root, stage=sweep.PRIMARY_STAGE)
    if len(jobs) != 180:
        raise Grid08AnalysisError(f"Expected 180 primary jobs, got {len(jobs)}")
    pair_metrics, lineage = _evaluate_jobs(
        root, jobs, panel_name="q2", mc_samples=16, evaluator=evaluator
    )
    _gzip_csv(root / "analysis" / "grid08_q2_pair_metrics.csv.gz", pair_metrics)
    primary = pair_metrics[
        pair_metrics["tolerance_minutes"].astype(int).eq(5)
        & pair_metrics["text_ablation_mode"].eq(REAL_TEXT)
    ].copy()
    score_rows = []
    for offset, ((capacity, lr_profile), group) in enumerate(
        primary.groupby(["capacity_profile", "lr_profile"], sort=False)
    ):
        summary = _config_bootstrap(group, offset=offset)
        per_seed = group.groupby("seed", sort=True)[
            ["model_mae", "persistence_mae"]
        ].mean()
        raw_profile = sweep._sweep(resolved)["profiles"][capacity]
        score_rows.append(
            {
                "capacity_profile": capacity,
                "lr_profile": lr_profile,
                "initial_learning_rate": sweep.LRS[lr_profile],
                "expected_wgan_parameters": int(
                    raw_profile["expected_wgan_parameters"]
                ),
                "pair_count_per_seed": int(
                    group.groupby("seed")["pair_id"].nunique().min()
                ),
                "session_count_per_seed": int(
                    group.groupby("seed")["session_id"].nunique().min()
                ),
                "non_worse_seed_count": int(
                    (per_seed["model_mae"] <= per_seed["persistence_mae"]).sum()
                ),
                **summary,
            }
        )
    scores = (
        pd.DataFrame(score_rows)
        .sort_values("score", kind="stable")
        .reset_index(drop=True)
    )
    if len(scores) != 30:
        raise Grid08AnalysisError("Q2 score table must contain 30 configurations")
    scores["holm_p"] = holm_adjust(scores["p_two_sided"].tolist())
    scores["rank"] = np.arange(1, len(scores) + 1)
    scores.to_csv(root / "analysis" / "grid08_capacity_lr_scores.csv", index=False)
    interactions = _interaction_table(primary)
    interactions.to_csv(
        root / "analysis" / "grid08_interaction_contrasts.csv", index=False
    )
    diagnostics = _training_diagnostics(root, jobs)
    diagnostics.to_csv(
        root / "analysis" / "grid08_training_diagnostics.csv", index=False
    )
    selection = _selection_from_scores(scores, root)
    robustness_rows = []
    secondary = pair_metrics[
        pair_metrics["tolerance_minutes"].astype(int).eq(30)
        & pair_metrics["text_ablation_mode"].eq(REAL_TEXT)
    ]
    for offset, ((capacity, lr_profile), group) in enumerate(
        secondary.groupby(["capacity_profile", "lr_profile"], sort=False)
    ):
        per_seed = group.groupby("seed")[["model_mae", "persistence_mae"]].mean()
        robustness_rows.append(
            {
                "capacity_profile": capacity,
                "lr_profile": lr_profile,
                "non_worse_seed_count": int(
                    (per_seed["model_mae"] <= per_seed["persistence_mae"]).sum()
                ),
                **_config_bootstrap(group, offset=9000 + offset),
            }
        )
    robustness_frame = pd.DataFrame(robustness_rows).sort_values("score", kind="stable")
    robustness_frame["rank_30m"] = np.arange(1, len(robustness_frame) + 1)
    q2_ranks = scores.set_index(["capacity_profile", "lr_profile"])["rank"]
    robustness_frame["rank_5m"] = [
        int(q2_ranks.loc[(row.capacity_profile, row.lr_profile)])
        for row in robustness_frame.itertuples()
    ]
    robustness_frame["rank_change_30m_minus_5m"] = (
        robustness_frame["rank_30m"] - robustness_frame["rank_5m"]
    )
    robustness_frame.to_csv(
        root / "analysis" / "grid08_30m_robustness_scores.csv", index=False
    )
    candidate_row = robustness_frame[
        robustness_frame["capacity_profile"].eq(
            selection["one_se_candidate_capacity_profile"]
        )
        & robustness_frame["lr_profile"].eq(selection["one_se_candidate_lr_profile"])
    ].iloc[0]
    robustness = candidate_row.to_dict()
    sweep._write_json(
        root / "analysis" / "grid08_q2_validation_summary.json",
        {
            "schema_version": 1,
            "selection_sha256": selection["selection_sha256"],
            "panel_lineage": lineage,
            "q2_pair_metrics_sha256": sweep._sha256_file(
                root / "analysis" / "grid08_q2_pair_metrics.csv.gz"
            ),
            "score_sha256": sweep._sha256_file(
                root / "analysis" / "grid08_capacity_lr_scores.csv"
            ),
            "interaction_sha256": sweep._sha256_file(
                root / "analysis" / "grid08_interaction_contrasts.csv"
            ),
            "robustness_30m_sha256": sweep._sha256_file(
                root / "analysis" / "grid08_30m_robustness_scores.csv"
            ),
            "candidate_30m_robustness": robustness,
            "q3_rows_passed_to_evaluator": 0,
            "q4_loader_created": False,
            "q4_rows_passed_to_evaluator": 0,
        },
    )
    return selection


def _paired_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    label: str,
    offset: int,
) -> dict[str, Any]:
    keys = ["seed", "session_id", "pair_id"]
    lhs = left[keys + ["model_mae", "persistence_mae"]].copy()
    if right is None:
        frame = lhs

        def difference(values: np.ndarray) -> np.ndarray:
            return values[:, 0] - values[:, 1]

        columns = ("model_mae", "persistence_mae")
    else:
        rhs = right[keys + ["model_mae"]].rename(columns={"model_mae": "right_mae"})
        frame = lhs.merge(rhs, on=keys, validate="one_to_one")

        def difference(values: np.ndarray) -> np.ndarray:
            return values[:, 0] - values[:, 1]

        columns = ("model_mae", "right_mae")
    observed, draws = _two_level_draws(
        frame,
        columns=columns,
        statistic=difference,
        seed=BOOTSTRAP_SEED + 20_000 + offset,
    )
    return {"contrast": label, **_summary(observed, draws)}


def run_grid08_q3_analysis(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    sweep._validate_root_lineage(root)
    registry = sweep._load_registry(root)
    allowlist_path = root / "q3_checkpoint_allowlist.csv"
    selection_path = root / "analysis" / "grid08_selection.json"
    if not registry.get("q3_unlocked") or not allowlist_path.is_file():
        raise Grid08AnalysisError(
            "Q3 remains locked until diagnostics and allowlist freeze"
        )
    if sweep._sha256_file(allowlist_path) != registry.get(
        "q3_checkpoint_allowlist_sha256"
    ):
        raise Grid08AnalysisError("Q3 checkpoint allowlist hash drift")
    selection = sweep._read_json(selection_path)
    if (
        sweep._payload_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "selection_sha256"
            }
        )
        != selection["selection_sha256"]
    ):
        raise Grid08AnalysisError("Selection self-hash mismatch")
    with allowlist_path.open("r", encoding="utf-8", newline="") as handle:
        allowed_ids = {row["job_id"] for row in csv.DictReader(handle)}
    jobs = [job for job in _jobs(root) if job["job_id"] in allowed_ids]
    if len(jobs) != len(allowed_ids):
        raise Grid08AnalysisError("Q3 allowlist references missing jobs")
    pair_metrics, lineage = _evaluate_jobs(
        root, jobs, panel_name="q3", mc_samples=64, evaluator=evaluator
    )
    _gzip_csv(root / "analysis" / "grid08_q3_pair_metrics.csv.gz", pair_metrics)
    primary = pair_metrics[pair_metrics["tolerance_minutes"].astype(int).eq(5)].copy()
    candidate_key = (
        selection["one_se_candidate_capacity_profile"],
        selection["one_se_candidate_lr_profile"],
    )
    leader_key = (
        selection["point_leader_capacity_profile"],
        selection["point_leader_lr_profile"],
    )

    def cell(capacity: str, lr: str, mode: str) -> pd.DataFrame:
        result = primary[
            primary["capacity_profile"].eq(capacity)
            & primary["lr_profile"].eq(lr)
            & primary["text_ablation_mode"].eq(mode)
        ].copy()
        if (
            result[["seed", "pair_id"]].drop_duplicates().shape[0]
            != 3 * sweep.EXPECTED_PANEL_COUNTS["common_q3"]["pairs"]
        ):
            raise Grid08AnalysisError(f"Incomplete Q3 cell: {capacity}/{lr}/{mode}")
        return result

    candidate_real = cell(*candidate_key, REAL_TEXT)
    candidate_current = cell(*candidate_key, "current_only")
    leader_real = cell(*leader_key, REAL_TEXT)
    anchor_real = cell("small", "lr_5e_07", REAL_TEXT)
    contrast_rows = [
        _paired_contrast(
            candidate_real, None, label="candidate_vs_persistence", offset=0
        ),
    ]
    if leader_key != candidate_key:
        contrast_rows.append(
            _paired_contrast(
                leader_real, None, label="point_leader_vs_persistence", offset=1
            )
        )
    contrast_rows.append(
        _paired_contrast(
            candidate_real, anchor_real, label="candidate_vs_anchor", offset=2
        )
    )
    contrast_rows.append(
        _paired_contrast(
            candidate_real,
            candidate_current,
            label="candidate_real_text_vs_current_only",
            offset=3,
        )
    )
    adjusted = holm_adjust([row["p_two_sided"] for row in contrast_rows])
    for row, value in zip(contrast_rows, adjusted):
        row["holm_p"] = value
        row["family_size"] = len(contrast_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(root / "analysis" / "grid08_q3_primary_contrasts.csv", index=False)
    secondary_rows = []
    for (capacity, lr, mode), group in pair_metrics[
        pair_metrics["tolerance_minutes"].astype(int).eq(30)
    ].groupby(["capacity_profile", "lr_profile", "text_ablation_mode"]):
        secondary_rows.append(
            {
                "capacity_profile": capacity,
                "lr_profile": lr,
                "text_ablation_mode": mode,
                **_config_bootstrap(group, offset=30_000 + len(secondary_rows)),
            }
        )
    pd.DataFrame(secondary_rows).to_csv(
        root / "analysis" / "grid08_q3_30m_secondary.csv", index=False
    )
    summary = {
        "schema_version": 1,
        "status": "completed_q3_only",
        "selection_sha256": selection["selection_sha256"],
        "allowlist_sha256": sweep._sha256_file(allowlist_path),
        "q3_pair_metrics_sha256": sweep._sha256_file(
            root / "analysis" / "grid08_q3_pair_metrics.csv.gz"
        ),
        "primary_contrasts_sha256": sweep._sha256_file(
            root / "analysis" / "grid08_q3_primary_contrasts.csv"
        ),
        "panel_lineage": lineage,
        "primary_family_size": len(contrast_rows),
        "q3_used_for_selection": False,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "q4_rows_passed_to_evaluator": 0,
    }
    summary["payload_sha256"] = sweep._payload_sha256(summary)
    sweep._write_json(root / "analysis" / "grid08_q3_validation_summary.json", summary)
    return summary


__all__ = ["run_grid08_q2_analysis", "run_grid08_q3_analysis"]
