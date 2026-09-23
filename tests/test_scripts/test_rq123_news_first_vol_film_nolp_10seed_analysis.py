from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq123.news_first_vol_film_nolp_10seed_analysis import (
    CANONICAL_ARMS,
    DEFAULT_PERSISTENCE_COMPARISONS,
    DEFAULT_RQ3_CONTRASTS,
    RETROSPECTIVE_LABEL,
    UnifiedAnalysisError,
    analyze_rq1_rq2,
    analyze_persistence_secondary,
    analyze_rq3_market_jumps,
    analyze_rq3_scheduled,
    apply_holm_and_consistency_gate,
    build_market_jump_match_plan,
    build_ordinary_match_plan,
    join_frozen_market_jumps,
    run_unified_analysis,
    seed_fold_session_paired_bootstrap,
    sha256_file,
    validate_paired_evidence,
)
from scripts.rq123.news_first_vol_film_nolp_10seed_report import (
    REPORT_DISCLAIMER,
    render_unified_report,
)


TEST_ARMS = ("continuation_no_text", "lp_matched")
TEST_SEEDS = (42, 202)
TEST_FOLDS = ("f1_2023q1", "f2_2023q2")
TEST_TOLERANCES = (5, 30)
_ASSERTIONS = unittest.TestCase()


def _pair_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(TEST_FOLDS):
        release = pd.Timestamp("2023-01-10T13:30:00Z") + pd.DateOffset(
            months=3 * fold_index
        )
        deltas = (
            -10,
            0,
            5,
            20,
            30,
            60,
            120,
            180,
            240,
            300,
            7 * 24 * 60 - 10,
            7 * 24 * 60,
            7 * 24 * 60 + 5,
            7 * 24 * 60 + 20,
            7 * 24 * 60 + 30,
            24 * 60,
        )
        for pair_index, delta in enumerate(deltas):
            origin = release + pd.Timedelta(minutes=delta)
            pair_id = f"{fold}_pair_{pair_index:02d}"
            session_id = f"{fold}_session_{pair_index // 2:02d}"
            exposed = -10 <= delta <= 30
            for tolerance in TEST_TOLERANCES:
                for seed in TEST_SEEDS:
                    for arm in TEST_ARMS:
                        job_id = f"{arm}_{fold}_{seed}_{tolerance}m"
                        if arm == "continuation_no_text":
                            target_mae = 1.0 + pair_index * 0.001
                        else:
                            # LP is uniformly better and especially useful around a
                            # scheduled release, making the RQ3 increment positive.
                            target_mae = (
                                0.50 if exposed else 0.90
                            ) + pair_index * 0.001
                        rows.append(
                            {
                                "job_id": job_id,
                                "tolerance_minutes": tolerance,
                                "fold": fold,
                                "seed": seed,
                                "arm": arm,
                                "pair_id": pair_id,
                                "session_id": session_id,
                                "effective_origin_utc": origin.isoformat(),
                                "target_mae": target_mae,
                                "persistence_mae": 1.2 + pair_index * 0.001,
                                "checkpoint_sha256": "0" * 64,
                                "prediction_sha256": "1" * 64,
                                "current_surface_mean": 0.20
                                + (pair_index % 10) * 0.001,
                                "current_surface_std": 0.02
                                + (pair_index % 10) * 0.0001,
                                "current_short_atm_mean": 0.18
                                + (pair_index % 10) * 0.001,
                                "current_strike_slope": -0.01
                                + (pair_index % 10) * 0.0001,
                                "current_term_slope": 0.015
                                + (pair_index % 10) * 0.0001,
                                "current_curvature": 0.003
                                + (pair_index % 10) * 0.00001,
                                "current_supported_cell_fraction": 0.80
                                + (pair_index % 10) * 0.001,
                            }
                        )
    return pd.DataFrame(rows)


def _events() -> pd.DataFrame:
    rows = []
    for fold_index, fold in enumerate(TEST_FOLDS):
        release = pd.Timestamp("2023-01-10T13:30:00Z") + pd.DateOffset(
            months=3 * fold_index
        )
        rows.append(
            {
                "event_id": f"event_{fold}",
                "release_time_utc": release.isoformat(),
                "scheduled_or_unscheduled": "scheduled",
            }
        )
    return pd.DataFrame(rows)


def _market_jumps(pair_rows: pd.DataFrame) -> pd.DataFrame:
    base = (
        pair_rows[pair_rows["tolerance_minutes"].eq(30)][
            ["pair_id", "session_id", "effective_origin_utc"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    selected = [
        base[base["pair_id"].eq(f"{TEST_FOLDS[0]}_pair_00")].iloc[0],
        base[base["pair_id"].eq(f"{TEST_FOLDS[0]}_pair_01")].iloc[0],
        base[base["pair_id"].eq(f"{TEST_FOLDS[1]}_pair_00")].iloc[0],
    ]
    base = pd.DataFrame(selected).reset_index(drop=True)
    base["anomaly_tier"] = ["broad", "primary", "high"]
    return (
        base.rename(columns={"pair_id": "market_pair_id"})
        .assign(pair_id=lambda frame: frame["market_pair_id"])[
            ["pair_id", "session_id", "effective_origin_utc", "anomaly_tier"]
        ]
        .rename(columns={"effective_origin_utc": "origin_time_utc"})
    )


def _write_job_artifacts(
    tmp_path: Path, pair_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pair_rows.copy()
    prediction_rows = []
    checkpoint_rows = []
    for job_id in sorted(pairs["job_id"].unique()):
        prediction = tmp_path / "predictions" / f"{job_id}.csv"
        checkpoint = tmp_path / "checkpoints" / f"{job_id}.pt"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text(f"job_id,value\n{job_id},1\n", encoding="utf-8")
        checkpoint.write_bytes(f"checkpoint:{job_id}".encode())
        prediction_sha = sha256_file(prediction)
        checkpoint_sha = sha256_file(checkpoint)
        pairs.loc[pairs["job_id"].eq(job_id), "prediction_sha256"] = prediction_sha
        pairs.loc[pairs["job_id"].eq(job_id), "checkpoint_sha256"] = checkpoint_sha
        prediction_rows.append(
            {
                "job_id": job_id,
                "prediction_path": str(prediction),
                "prediction_sha256": prediction_sha,
                "size_bytes": prediction.stat().st_size,
            }
        )
        checkpoint_rows.append(
            {
                "job_id": job_id,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "size_bytes": checkpoint.stat().st_size,
            }
        )
    return pairs, pd.DataFrame(prediction_rows), pd.DataFrame(checkpoint_rows)


def _check_seed_fold_session_bootstrap_is_deterministic_and_gated() -> None:
    frame = _pair_rows()
    selected = frame[frame["tolerance_minutes"].eq(5)]
    first = seed_fold_session_paired_bootstrap(
        selected,
        focal_arm="lp_matched",
        reference_arm="continuation_no_text",
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=250,
        rng_seed=71,
    )
    second = seed_fold_session_paired_bootstrap(
        selected,
        focal_arm="lp_matched",
        reference_arm="continuation_no_text",
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=250,
        rng_seed=71,
    )
    assert first == second
    assert first["mean_log_mae_ratio"] < 0
    assert first["geometric_mae_ratio"] < 1
    assert math.isclose(
        first["geometric_mae_ratio"],
        math.exp(first["mean_log_mae_ratio"]),
    )
    assert first["ci_95_upper"] < 0
    assert (
        first["resampling_method"]
        == "seed_then_fold_then_paired_session_cluster_recompute_cell_log_mae_ratio"
    )

    result = pd.DataFrame(
        [
            {
                **first,
                "research_question": "RQ1",
                "tolerance_minutes": 5,
                "contrast_id": "lp_minus_no_text",
            }
        ]
    )
    gated = apply_holm_and_consistency_gate(
        result,
        min_nonworse_seed_fraction=1.0,
        min_nonworse_fold_fraction=1.0,
    )
    assert bool(gated.loc[0, "passes_full_gate"])
    assert gated.loc[0, "holm_adjusted_p"] == first["p_value_one_sided"]


def _check_holm_family_and_ten_seed_four_fold_consistency_thresholds() -> None:
    results = pd.DataFrame(
        [
            {
                "research_question": "RQ2",
                "tolerance_minutes": 5,
                "contrast_id": "lp_minus_bow",
                "mean_difference": -0.2,
                "ci_95_upper": -0.01,
                "p_value_one_sided": 0.01,
                "consistent_seed_count": 7,
                "consistent_fold_count": 3,
                "seed_count": 10,
                "fold_count": 4,
            },
            {
                "research_question": "RQ2",
                "tolerance_minutes": 5,
                "contrast_id": "lp_minus_sentiment",
                "mean_difference": -0.1,
                "ci_95_upper": -0.01,
                "p_value_one_sided": 0.04,
                "consistent_seed_count": 6,
                "consistent_fold_count": 3,
                "seed_count": 10,
                "fold_count": 4,
            },
        ]
    )
    gated = apply_holm_and_consistency_gate(results)
    by_contrast = gated.set_index("contrast_id")
    assert abs(by_contrast.loc["lp_minus_bow", "holm_adjusted_p"] - 0.02) < 1e-12
    assert bool(by_contrast.loc["lp_minus_bow", "passes_full_gate"])
    assert by_contrast.loc["lp_minus_bow", "required_consistent_seed_count"] == 7
    assert by_contrast.loc["lp_minus_bow", "required_consistent_fold_count"] == 3
    assert not bool(by_contrast.loc["lp_minus_sentiment", "passes_full_gate"])


def _check_rq1_rq2_is_log_mae_ratio_on_5m_only() -> None:
    results = analyze_rq1_rq2(
        _pair_rows(),
        contrast_specs=(
            {
                "research_question": "RQ1",
                "contrast_id": "lp_minus_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=100,
        rng_seed=72,
    )
    assert set(results["tolerance_minutes"]) == {5}
    assert set(results["gate_estimate_column"]) == {"mean_log_mae_ratio"}
    assert results["mean_log_mae_ratio"].lt(0).all()
    assert results["geometric_mae_ratio"].lt(1).all()
    assert "mean_difference" not in results

    with _ASSERTIONS.assertRaisesRegex(UnifiedAnalysisError, "frozen to 5m"):
        analyze_rq1_rq2(
            _pair_rows(),
            tolerance_minutes=30,
            contrast_specs=(
                {
                    "research_question": "RQ1",
                    "contrast_id": "lp_minus_no_text",
                    "focal_arm": "lp_matched",
                    "reference_arm": "continuation_no_text",
                },
            ),
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            iterations=20,
        )


def _check_persistence_secondary_uses_two_independent_holm_families() -> None:
    default = pd.DataFrame(DEFAULT_PERSISTENCE_COMPARISONS)
    assert default.groupby("research_question").size().to_dict() == {"RQ1": 4, "RQ2": 4}
    assert set(default.loc[default["research_question"].eq("RQ1"), "arm"]) == {
        "parent_current_only",
        "continuation_no_text",
        "lp_matched",
        "lp_shuffle",
    }
    assert set(default.loc[default["research_question"].eq("RQ2"), "arm"]) == {
        "continuation_no_text",
        "lp_matched",
        "bow",
        "sentiment",
    }
    parent = default[
        default["arm"].eq("parent_current_only")
        & default["research_question"].eq("RQ1")
    ].iloc[0]
    assert parent["comparison_role"] == "diagnostic_only"

    results = analyze_persistence_secondary(
        _pair_rows(),
        comparison_specs=(
            {
                "research_question": "RQ1",
                "comparison_id": "no_text_vs_persistence",
                "arm": "continuation_no_text",
                "comparison_role": "diagnostic_only",
            },
            {
                "research_question": "RQ1",
                "comparison_id": "lp_vs_persistence",
                "arm": "lp_matched",
                "comparison_role": "secondary",
            },
            {
                "research_question": "RQ2",
                "comparison_id": "no_text_vs_persistence",
                "arm": "continuation_no_text",
                "comparison_role": "secondary",
            },
            {
                "research_question": "RQ2",
                "comparison_id": "lp_vs_persistence",
                "arm": "lp_matched",
                "comparison_role": "secondary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=100,
        rng_seed=73,
    )
    assert set(results["tolerance_minutes"]) == {5}
    assert results.groupby("research_question").size().to_dict() == {"RQ1": 2, "RQ2": 2}
    assert results["mean_log_mae_ratio"].lt(0).all()
    diagnostic = results[results["comparison_role"].eq("diagnostic_only")].iloc[0]
    assert math.isfinite(float(diagnostic["holm_adjusted_p"]))
    assert bool(diagnostic["passes_statistical_gate"])
    assert not bool(diagnostic["inference_permitted"])
    assert not bool(diagnostic["passes_secondary_gate"])
    assert diagnostic["claim_scope"] == "retrospective_diagnostic_only"
    assert results.groupby("research_question")["holm_adjusted_p"].nunique().ge(1).all()


def _check_validate_paired_evidence_checks_files_hashes_and_pair_universe(
    tmp_path: Path,
) -> None:
    pairs, predictions, checkpoints = _write_job_artifacts(tmp_path, _pair_rows())
    evidence = validate_paired_evidence(
        pairs,
        predictions,
        checkpoints,
        expected_arms=TEST_ARMS,
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        expected_tolerances=TEST_TOLERANCES,
    )
    assert evidence.pair_metrics["job_id"].nunique() == 16

    state_drift = pairs.copy()
    state_drift.loc[
        state_drift["job_id"].eq("lp_matched_f1_2023q1_42_30m")
        & state_drift["pair_id"].eq("f1_2023q1_pair_00"),
        "current_surface_mean",
    ] += 0.01
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "current-state universe differs across arms"
    ):
        validate_paired_evidence(
            state_drift,
            predictions,
            checkpoints,
            expected_arms=TEST_ARMS,
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            expected_tolerances=TEST_TOLERANCES,
        )

    cross_seed_drift = pairs.copy()
    cross_seed_drift.loc[
        cross_seed_drift["seed"].eq(42)
        & cross_seed_drift["fold"].eq("f1_2023q1")
        & cross_seed_drift["tolerance_minutes"].eq(30)
        & cross_seed_drift["pair_id"].eq("f1_2023q1_pair_00"),
        "current_term_slope",
    ] += 0.01
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "differs across arms/seeds"
    ):
        validate_paired_evidence(
            cross_seed_drift,
            predictions,
            checkpoints,
            expected_arms=TEST_ARMS,
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            expected_tolerances=TEST_TOLERANCES,
        )

    nonfinite = pairs.copy()
    nonfinite.loc[nonfinite.index[0], "current_surface_std"] = float("nan")
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "finite current-state values"
    ):
        validate_paired_evidence(
            nonfinite,
            predictions,
            checkpoints,
            expected_arms=TEST_ARMS,
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            expected_tolerances=TEST_TOLERANCES,
        )

    drifted = pairs.copy()
    mask = drifted["arm"].eq("lp_matched") & drifted["pair_id"].str.endswith("00")
    drifted.loc[mask, "session_id"] = "tampered_session"
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "universe differs across arms"
    ):
        validate_paired_evidence(
            drifted,
            predictions,
            checkpoints,
            expected_arms=TEST_ARMS,
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            expected_tolerances=TEST_TOLERANCES,
        )


def _check_canonical_arm_by_tolerance_matrix_is_400_cell_compatible(
    tmp_path: Path,
) -> None:
    rows = []
    arms_by_tolerance = {
        5: CANONICAL_ARMS,
        30: (
            "parent_current_only",
            "continuation_no_text",
            "lp_matched",
            "lp_shuffle",
        ),
    }
    for tolerance, arms in arms_by_tolerance.items():
        for arm_index, arm in enumerate(arms):
            job_id = f"{arm}_f1_2023q1_42_{tolerance}m"
            for pair_index in range(2):
                rows.append(
                    {
                        "job_id": job_id,
                        "tolerance_minutes": tolerance,
                        "fold": "f1_2023q1",
                        "seed": 42,
                        "arm": arm,
                        "pair_id": f"pair_{tolerance}_{pair_index}",
                        "session_id": f"session_{pair_index}",
                        "effective_origin_utc": f"2023-01-0{pair_index + 2}T13:30:00Z",
                        "target_mae": 1.0 + arm_index * 0.01,
                        "persistence_mae": 1.5,
                        "checkpoint_sha256": "0" * 64,
                        "prediction_sha256": "1" * 64,
                        "current_surface_mean": 0.20 + pair_index * 0.001,
                        "current_surface_std": 0.02,
                        "current_short_atm_mean": 0.19 + pair_index * 0.001,
                        "current_strike_slope": -0.01,
                        "current_term_slope": 0.015,
                        "current_curvature": 0.003,
                        "current_supported_cell_fraction": 0.8,
                    }
                )
    pairs, predictions, checkpoints = _write_job_artifacts(tmp_path, pd.DataFrame(rows))
    evidence = validate_paired_evidence(
        pairs,
        predictions,
        checkpoints,
        expected_arms=CANONICAL_ARMS,
        expected_seeds=(42,),
        expected_folds=("f1_2023q1",),
        expected_tolerances=(5, 30),
    )
    assert evidence.pair_metrics["job_id"].nunique() == 10

    Path(predictions.loc[0, "prediction_path"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    with _ASSERTIONS.assertRaisesRegex(UnifiedAnalysisError, "SHA-256 drift"):
        validate_paired_evidence(
            pairs,
            predictions,
            checkpoints,
            expected_arms=CANONICAL_ARMS,
            expected_seeds=(42,),
            expected_folds=("f1_2023q1",),
            expected_tolerances=(5, 30),
        )


def _check_scheduled_windows_are_exact_and_use_ordinary_matching() -> None:
    assert {spec["contrast_role"] for spec in DEFAULT_RQ3_CONTRASTS} == {"primary"}
    assert len(DEFAULT_RQ3_CONTRASTS) == 2
    pairs = _pair_rows()
    events = _events()
    exact_clock_plan = build_ordinary_match_plan(pairs, events, clock_caliper_minutes=0)
    release_matches = exact_clock_plan[
        exact_clock_plan["window_id"].eq("scheduled_0_plus30_primary")
        & exact_clock_plan["event_pair_id"].str.endswith("_pair_01")
        & exact_clock_plan["match_status"].eq("matched")
    ]
    assert set(release_matches["control_pair_id"].str[-8:]) == {"_pair_11"}
    assert not exact_clock_plan["control_pair_id"].str.endswith("_pair_15").any()
    plan = build_ordinary_match_plan(pairs, events, clock_caliper_minutes=360)
    counts = plan.groupby("window_id")["event_pair_id"].nunique().to_dict()
    assert counts == {
        "scheduled_0_plus30_primary": 8,
        "scheduled_0_plus5_descriptive": 4,
        "scheduled_minus10_plus20_robustness": 8,
    }
    assert (
        plan.groupby("window_id")["control_pair_id"]
        .apply(lambda values: values.is_unique)
        .all()
    )
    matched_plan = plan[plan["match_status"].eq("matched")]
    event_times = pd.to_datetime(matched_plan["event_effective_origin_utc"], utc=True)
    control_times = pd.to_datetime(
        matched_plan["control_effective_origin_utc"], utc=True
    )
    assert all(
        event.weekday() == control.weekday()
        for event, control in zip(event_times, control_times, strict=True)
    )
    assert not matched_plan["control_pair_id"].str.endswith("_pair_15").any()
    assert (
        matched_plan["matching_method"]
        .eq(
            "fold_weekday_exact_clock_caliper_greedy_no_replacement_calendar_distance_v1"
        )
        .all()
    )
    primary_deltas = set(
        plan.loc[plan["window_role"].eq("primary"), "event_delta_minutes"]
    )
    assert primary_deltas == {0.0, 5.0, 20.0, 30.0}
    robustness_deltas = set(
        plan.loc[plan["window_role"].eq("robustness"), "event_delta_minutes"]
    )
    assert robustness_deltas == {-10.0, 0.0, 5.0, 20.0}

    frozen_plan, details, results = analyze_rq3_scheduled(
        pairs,
        events,
        match_plan=plan,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=150,
        rng_seed=99,
        clock_caliper_minutes=360,
        minimum_primary_pairs=1,
        minimum_primary_releases=1,
        minimum_primary_sessions=1,
    )
    assert frozen_plan.equals(plan)
    assert details["scheduled_increment"].gt(0).all()
    primary = results[results["window_role"].eq("primary")].iloc[0]
    assert primary["mean_difference"] > 0
    assert primary["ci_95_lower"] > 0
    assert bool(primary["passes_primary_gate"])
    descriptive = results[results["window_role"].eq("descriptive")].iloc[0]
    assert not bool(descriptive["passes_primary_gate"])
    assert descriptive["claim_scope"] == "retrospective_descriptive_only"

    overlapping_events = pd.concat(
        [
            events,
            events.iloc[[0]].assign(event_id="overlapping_release_audit_only"),
        ],
        ignore_index=True,
    )
    _, _, overlap_results = analyze_rq3_scheduled(
        pairs,
        overlapping_events,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=20,
        rng_seed=98,
        clock_caliper_minutes=360,
        minimum_primary_pairs=1,
        minimum_primary_releases=1,
        minimum_primary_sessions=1,
    )
    overlap_primary = overlap_results[
        overlap_results["window_role"].eq("primary")
    ].iloc[0]
    assert overlap_primary["event_count"] == len(TEST_FOLDS)
    assert overlap_primary["eligible_event_count"] == len(TEST_FOLDS) + 1

    tampered_plan = plan.copy()
    tampered_plan.loc[
        tampered_plan["match_status"].eq("matched").idxmax(), "control_pair_id"
    ] = "tampered_control"
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "differs from deterministic replay"
    ):
        analyze_rq3_scheduled(
            pairs,
            events,
            match_plan=tampered_plan,
            contrast_specs=(
                {
                    "contrast_id": "lp_vs_no_text",
                    "focal_arm": "lp_matched",
                    "reference_arm": "continuation_no_text",
                    "contrast_role": "primary",
                },
            ),
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            iterations=20,
            clock_caliper_minutes=360,
        )

    _, _, undercovered = analyze_rq3_scheduled(
        pairs,
        events,
        match_plan=plan,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=50,
        rng_seed=100,
        clock_caliper_minutes=360,
    )
    undercovered_primary = undercovered[undercovered["window_role"].eq("primary")].iloc[
        0
    ]
    assert not bool(undercovered_primary["coverage_gate_passes"])
    assert not bool(undercovered_primary["passes_primary_gate"])
    assert (
        undercovered_primary["claim_scope"]
        == "retrospective_undercovered_descriptive_only"
    )

    sparse_pairs = pairs.copy()
    sparse_fold = TEST_FOLDS[0]
    sparse_release = pd.Timestamp("2023-01-10T13:30:00Z")
    sparse_mask = sparse_pairs["fold"].eq(sparse_fold) & sparse_pairs[
        "pair_id"
    ].str.endswith(("_pair_01", "_pair_02"))
    sparse_pairs.loc[sparse_mask, "effective_origin_utc"] = sparse_pairs.loc[
        sparse_mask, "pair_id"
    ].map(
        {
            f"{sparse_fold}_pair_01": (
                sparse_release + pd.Timedelta(minutes=45)
            ).isoformat(),
            f"{sparse_fold}_pair_02": (
                sparse_release + pd.Timedelta(minutes=50)
            ).isoformat(),
        }
    )
    _, _, sparse_results = analyze_rq3_scheduled(
        sparse_pairs,
        events,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=50,
        rng_seed=101,
        clock_caliper_minutes=360,
        minimum_primary_pairs=1,
        minimum_primary_releases=1,
        minimum_primary_sessions=1,
    )
    sparse_descriptive = sparse_results[
        sparse_results["window_role"].eq("descriptive")
    ].iloc[0]
    assert sparse_descriptive["fold_count"] == 1
    assert sparse_descriptive["observed_fold_count"] == 1
    assert not bool(sparse_descriptive["full_fold_panel"])
    assert json.loads(sparse_descriptive["missing_folds_json"]) == [sparse_fold]
    assert sparse_descriptive["claim_scope"] == "retrospective_descriptive_only"


def _check_market_jump_join_is_exact_all_tier_and_descriptive() -> None:
    pairs = _pair_rows()
    jumps = _market_jumps(pairs)
    joined = join_frozen_market_jumps(pairs, jumps)
    assert set(joined["anomaly_tier"]) == {"broad", "primary", "high"}
    assert joined["all_tier_coverage_gate_passes"].all()
    assert (
        joined["join_method"]
        .eq("exact_effective_origin_utc_to_frozen_origin_time_utc")
        .all()
    )
    match_plan = build_market_jump_match_plan(pairs, joined)
    matched = match_plan[match_plan["match_status"].eq("matched")]
    assert len(matched) == 3
    assert matched["control_pair_id"].is_unique
    assert not matched["control_pair_id"].str.endswith("_pair_15").any()
    assert matched["match_distance"].eq(0.0).all()
    assert matched["distance_metric"].eq("robust_scale_euclidean").all()
    jump_times = pd.to_datetime(matched["jump_effective_origin_utc"], utc=True)
    control_times = pd.to_datetime(matched["control_effective_origin_utc"], utc=True)
    assert all(
        jump.weekday() == control.weekday()
        for jump, control in zip(jump_times, control_times, strict=True)
    )
    assert all(
        (jump.hour, jump.minute) == (control.hour, control.minute)
        for jump, control in zip(jump_times, control_times, strict=True)
    )
    frozen_plan, details, results = analyze_rq3_market_jumps(
        pairs,
        joined,
        match_plan=match_plan,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=100,
        rng_seed=101,
        minimum_pairs=1,
        minimum_sessions=1,
    )
    assert frozen_plan.equals(match_plan)
    assert details["jump_increment"].gt(0).all()
    primary_all = results[results["anomaly_tier"].eq("all")].iloc[0]
    assert primary_all["mean_jump_increment"] > 0
    assert (
        primary_all["mean_jump_text_advantage"]
        > primary_all["mean_control_text_advantage"]
    )
    assert primary_all["ci_95_lower"] > 0
    assert bool(
        results.loc[results["anomaly_tier"].eq("all"), "inference_permitted"].iloc[0]
    )
    assert primary_all["estimability_status"] == "estimable"
    assert not results.loc[
        results["anomaly_tier"].isin(["primary", "high"]), "inference_permitted"
    ].any()
    assert set(
        results.loc[
            results["anomaly_tier"].isin(["primary", "high"]),
            "estimability_status",
        ]
    ) == {"descriptive_severity_only"}
    assert results["interpretation"].eq(RETROSPECTIVE_LABEL).all()
    assert set(
        results.loc[results["anomaly_tier"].isin(["primary", "high"]), "analysis_role"]
    ) == {"descriptive_primary_high"}

    _, _, not_estimable = analyze_rq3_market_jumps(
        pairs,
        joined,
        match_plan=match_plan,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=20,
        rng_seed=102,
        minimum_pairs=999,
        minimum_sessions=999,
    )
    not_estimable_all = not_estimable[not_estimable["anomaly_tier"].eq("all")].iloc[0]
    assert not_estimable_all["estimability_status"] == "not_estimable_due_to_coverage"
    assert not bool(not_estimable_all["inference_permitted"])
    assert not bool(not_estimable_all["passes_primary_gate"])

    state_drift = pairs.copy()
    state_drift.loc[
        state_drift["job_id"].eq("lp_matched_f1_2023q1_42_30m")
        & state_drift["pair_id"].eq(f"{TEST_FOLDS[0]}_pair_00"),
        "current_surface_mean",
    ] += 0.1
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "differs across arms/seeds"
    ):
        build_market_jump_match_plan(state_drift, joined)

    fold_two_control_ids = {
        f"{TEST_FOLDS[1]}_pair_{index:02d}" for index in range(10, 15)
    }
    undercovered_pairs = pairs[~pairs["pair_id"].isin(fold_two_control_ids)].copy()
    partial_tier_plan, _, partial_tier_results = analyze_rq3_market_jumps(
        undercovered_pairs,
        joined,
        contrast_specs=(
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        expected_seeds=TEST_SEEDS,
        expected_folds=TEST_FOLDS,
        iterations=20,
        minimum_pairs=2,
        minimum_sessions=1,
    )
    assert (
        partial_tier_plan.loc[
            partial_tier_plan["anomaly_tier"].eq("high"), "match_status"
        ]
        .eq("unmatched")
        .all()
    )
    partial_tier_all = partial_tier_results[
        partial_tier_results["anomaly_tier"].eq("all")
    ].iloc[0]
    assert bool(partial_tier_all["coverage_gate_passes"])
    assert bool(partial_tier_all["inference_permitted"])
    assert not bool(partial_tier_all["full_fold_panel"])
    assert not bool(partial_tier_all["matched_tier_category_coverage_passes"])
    assert partial_tier_all["fold_count"] == 1
    assert partial_tier_all["bootstrap_iterations"] == 20

    tampered_plan = match_plan.copy()
    tampered_plan.loc[
        tampered_plan["match_status"].eq("matched").idxmax(), "control_pair_id"
    ] = "tampered_control"
    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "differs from deterministic replay"
    ):
        analyze_rq3_market_jumps(
            pairs,
            joined,
            match_plan=tampered_plan,
            contrast_specs=(
                {
                    "contrast_id": "lp_vs_no_text",
                    "focal_arm": "lp_matched",
                    "reference_arm": "continuation_no_text",
                    "contrast_role": "primary",
                },
            ),
            expected_seeds=TEST_SEEDS,
            expected_folds=TEST_FOLDS,
            iterations=20,
            minimum_pairs=1,
            minimum_sessions=1,
        )

    with _ASSERTIONS.assertRaisesRegex(
        UnifiedAnalysisError, "all-tier coverage gate failed"
    ):
        join_frozen_market_jumps(pairs, jumps[jumps["anomaly_tier"].ne("high")])


def _check_run_and_report_are_explicit_hashed_and_self_contained(
    tmp_path: Path,
) -> None:
    pairs, predictions, checkpoints = _write_job_artifacts(tmp_path, _pair_rows())
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pair_path = inputs / "pair_metrics.csv"
    prediction_path = inputs / "prediction_manifest.csv"
    checkpoint_path = inputs / "checkpoint_manifest.csv"
    events_path = inputs / "scheduled_events.csv"
    jumps_path = inputs / "candidate_pairs.csv"
    pairs.to_csv(pair_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    checkpoints.to_csv(checkpoint_path, index=False)
    _events().to_csv(events_path, index=False)
    _market_jumps(pairs).to_csv(jumps_path, index=False)

    output = tmp_path / "analysis"
    run_kwargs = {
        "pair_metrics_path": pair_path,
        "pair_metrics_sha256": sha256_file(pair_path),
        "prediction_manifest_path": prediction_path,
        "prediction_manifest_sha256": sha256_file(prediction_path),
        "checkpoint_manifest_path": checkpoint_path,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "scheduled_events_path": events_path,
        "scheduled_events_sha256": sha256_file(events_path),
        "market_jump_path": jumps_path,
        "market_jump_sha256": sha256_file(jumps_path),
        "output_dir": output,
        "expected_arms": TEST_ARMS,
        "expected_seeds": TEST_SEEDS,
        "expected_folds": TEST_FOLDS,
        "expected_tolerances": TEST_TOLERANCES,
        "rq1_rq2_contrast_specs": (
            {
                "research_question": "RQ1",
                "contrast_id": "lp_minus_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
            },
        ),
        "persistence_comparison_specs": (
            {
                "research_question": "RQ1",
                "comparison_id": "no_text_vs_persistence",
                "arm": "continuation_no_text",
                "comparison_role": "diagnostic_only",
            },
            {
                "research_question": "RQ1",
                "comparison_id": "lp_vs_persistence",
                "arm": "lp_matched",
                "comparison_role": "secondary",
            },
            {
                "research_question": "RQ2",
                "comparison_id": "no_text_vs_persistence",
                "arm": "continuation_no_text",
                "comparison_role": "secondary",
            },
            {
                "research_question": "RQ2",
                "comparison_id": "lp_vs_persistence",
                "arm": "lp_matched",
                "comparison_role": "secondary",
            },
        ),
        "rq3_contrast_specs": (
            {
                "contrast_id": "lp_vs_no_text",
                "focal_arm": "lp_matched",
                "reference_arm": "continuation_no_text",
                "contrast_role": "primary",
            },
        ),
        "bootstrap_iterations": 100,
        "bootstrap_seed": 5,
        "scheduled_clock_caliper_minutes": 360,
        "scheduled_minimum_primary_pairs": 1,
        "scheduled_minimum_primary_releases": 1,
        "scheduled_minimum_primary_sessions": 1,
        "market_jump_minimum_pairs": 999,
        "market_jump_minimum_sessions": 999,
        "required_market_root": None,
    }
    paths = run_unified_analysis(**run_kwargs)
    summary = json.loads(paths["analysis_summary"].read_text(encoding="utf-8"))
    assert summary["market_jump_estimability_status"] == "not_estimable_due_to_coverage"
    analysis_mtimes = {path: path.stat().st_mtime_ns for path in paths.values()}
    resumed_paths = run_unified_analysis(**run_kwargs)
    assert resumed_paths == paths
    assert {path: path.stat().st_mtime_ns for path in paths.values()} == analysis_mtimes
    manifest = paths["analysis_manifest"]
    report_kwargs = {
        "analysis_manifest_path": manifest,
        "analysis_manifest_sha256": sha256_file(manifest),
        "output_dir": tmp_path / "report",
    }
    markdown, html_report = render_unified_report(**report_kwargs)
    report_mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            markdown,
            html_report,
            Path(report_kwargs["output_dir"]) / "report_manifest.json",
        )
    }
    assert render_unified_report(**report_kwargs) == (markdown, html_report)
    assert {path: path.stat().st_mtime_ns for path in report_mtimes} == report_mtimes
    markdown_text = markdown.read_text(encoding="utf-8")
    html_text = html_report.read_text(encoding="utf-8")
    assert RETROSPECTIVE_LABEL in markdown_text
    assert REPORT_DISCLAIMER in markdown_text
    assert "[0,+30] primary" in markdown_text
    assert "target remains the surface exactly five minutes" in markdown_text
    assert "not_estimable_due_to_coverage" in markdown_text
    assert "<style>" in html_text
    assert "http://" not in html_text and "https://" not in html_text

    checkpoint_artifact = Path(checkpoints.loc[0, "checkpoint_path"])
    checkpoint_bytes = checkpoint_artifact.read_bytes()
    checkpoint_artifact.write_bytes(b"post-analysis checkpoint tamper")
    with _ASSERTIONS.assertRaisesRegex(UnifiedAnalysisError, "SHA-256 drift"):
        render_unified_report(
            analysis_manifest_path=manifest,
            analysis_manifest_sha256=sha256_file(manifest),
            output_dir=tmp_path / "checkpoint_tampered_report",
        )
    checkpoint_artifact.write_bytes(checkpoint_bytes)

    summary_path = paths["analysis_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["confirmatory"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with _ASSERTIONS.assertRaisesRegex(UnifiedAnalysisError, "SHA-256 drift"):
        render_unified_report(
            analysis_manifest_path=manifest,
            analysis_manifest_sha256=sha256_file(manifest),
            output_dir=tmp_path / "tampered_report",
        )


class TestNewsFirstVolFilmNoLP10SeedAnalysis(unittest.TestCase):
    def test_seed_fold_session_bootstrap_is_deterministic_and_gated(self) -> None:
        _check_seed_fold_session_bootstrap_is_deterministic_and_gated()

    def test_holm_family_and_ten_seed_four_fold_consistency_thresholds(
        self,
    ) -> None:
        _check_holm_family_and_ten_seed_four_fold_consistency_thresholds()

    def test_rq1_rq2_is_log_mae_ratio_on_5m_only(self) -> None:
        _check_rq1_rq2_is_log_mae_ratio_on_5m_only()

    def test_persistence_secondary_uses_two_independent_holm_families(self) -> None:
        _check_persistence_secondary_uses_two_independent_holm_families()

    def test_validate_paired_evidence_checks_files_hashes_and_pair_universe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _check_validate_paired_evidence_checks_files_hashes_and_pair_universe(
                Path(directory)
            )

    def test_canonical_arm_by_tolerance_matrix_is_400_cell_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _check_canonical_arm_by_tolerance_matrix_is_400_cell_compatible(
                Path(directory)
            )

    def test_scheduled_windows_are_exact_and_use_ordinary_matching(self) -> None:
        _check_scheduled_windows_are_exact_and_use_ordinary_matching()

    def test_market_jump_join_is_exact_all_tier_and_descriptive(self) -> None:
        _check_market_jump_join_is_exact_all_tier_and_descriptive()

    def test_run_and_report_are_explicit_hashed_and_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _check_run_and_report_are_explicit_hashed_and_self_contained(
                Path(directory)
            )


if __name__ == "__main__":
    unittest.main()
