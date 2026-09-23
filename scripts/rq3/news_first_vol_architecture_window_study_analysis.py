"""Fail-closed analysis for the architecture and alignment-window studies."""

from __future__ import annotations

from collections.abc import Mapping
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_architecture_window_study as study
from scripts.rq3.news_first_vol_experiment_supervisor_helpers import utc_now


PAIR_METRIC_COLUMNS = {
    "evaluation_id",
    "source_cell_id",
    "model_id",
    "generator_mode",
    "train_tolerance_minutes",
    "panel_tolerance_minutes",
    "panel_role",
    "text_condition",
    "fold",
    "seed",
    "pair_id",
    "session_id",
    "window_relation",
    "post_news_current_overlap_minutes",
    "target_mae",
    "persistence_mae",
    "persistence_skill",
    "predicted_calendar_violation_rate",
    "target_calendar_violation_rate",
    "calendar_violation_rate_gap",
    "predicted_butterfly_violation_rate",
    "target_butterfly_violation_rate",
    "butterfly_violation_rate_gap",
    "checkpoint_sha256",
    "prediction_sha256",
    "noise_bank_profile_sha256",
}
FINITE_COLUMNS = {
    "target_mae",
    "persistence_mae",
    "persistence_skill",
    "predicted_calendar_violation_rate",
    "target_calendar_violation_rate",
    "calendar_violation_rate_gap",
    "predicted_butterfly_violation_rate",
    "target_butterfly_violation_rate",
    "butterfly_violation_rate_gap",
    "post_news_current_overlap_minutes",
}


def _config(root: Path) -> dict[str, Any]:
    raw = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    return study._mapping(raw, "resolved config")


def _metrics(root: Path) -> pd.DataFrame:
    path = root / "predictions/pair_metrics.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"pair_id": str, "session_id": str})
    missing = sorted(PAIR_METRIC_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Pair metrics missing required diagnostics: {missing}")
    if frame.empty:
        raise ValueError("Pair metrics are empty")
    for column in FINITE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Pair metric {column} contains missing/non-finite values")
    if (pd.to_numeric(frame["target_mae"]) < 0).any() or (
        pd.to_numeric(frame["persistence_mae"]) <= 0
    ).any():
        raise ValueError("MAE values are outside the valid domain")
    uniqueness = ["evaluation_id", "pair_id"]
    if frame.duplicated(uniqueness).any():
        raise ValueError("Duplicate evaluation/pair rows")
    line = str(_config(root)["line"])
    expected_conditions = (
        {"matched", "zero", "shuffle"}
        if line == "architecture"
        else {"matched", "zero"}
    )
    if set(frame["text_condition"].astype(str)) != expected_conditions:
        raise ValueError(
            f"Text-intervention evidence drift for {line}: {sorted(expected_conditions)}"
        )
    if set(frame["window_relation"].astype(str)) != set(study.WINDOW_RELATIONS):
        raise ValueError("Three window-relation strata are not all represented")
    return frame


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_id",
        "generator_mode",
        "train_tolerance_minutes",
        "panel_tolerance_minutes",
        "panel_role",
        "text_condition",
        "seed",
        "fold",
        "window_relation",
    ]
    return (
        frame.groupby(keys, as_index=False)
        .agg(
            mean_mae=("target_mae", "mean"),
            mean_persistence_mae=("persistence_mae", "mean"),
            mean_persistence_skill=("persistence_skill", "mean"),
            predicted_calendar_violation_rate=(
                "predicted_calendar_violation_rate",
                "mean",
            ),
            target_calendar_violation_rate=("target_calendar_violation_rate", "mean"),
            predicted_butterfly_violation_rate=(
                "predicted_butterfly_violation_rate",
                "mean",
            ),
            target_butterfly_violation_rate=("target_butterfly_violation_rate", "mean"),
            pair_count=("pair_id", "nunique"),
            session_count=("session_id", "nunique"),
            mean_post_news_current_overlap_minutes=(
                "post_news_current_overlap_minutes",
                "mean",
            ),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def analyze(root_or_path: str | Path, registry: Mapping[str, Any]) -> Path:
    root = Path(root_or_path).resolve()
    if not registry.get("evaluation_frozen") or not registry.get("predictions_frozen"):
        raise RuntimeError("Analysis gate is not frozen")
    frame = _metrics(root)
    summary = _summary(frame)
    output = root / "analysis/cell_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    manifest = root / "analysis/analysis_manifest.json"
    study._write_signed(
        manifest,
        {
            "schema_version": 1,
            "kind": "architecture_window_analysis_manifest_v1",
            "study_kind": registry["study_kind"],
            "pair_metrics_path": str(
                (root / "predictions/pair_metrics.csv.gz").resolve()
            ),
            "pair_metrics_sha256": study._sha256_file(
                root / "predictions/pair_metrics.csv.gz"
            ),
            "summary_path": str(output.resolve()),
            "summary_sha256": study._sha256_file(output),
            "pair_metric_rows": len(frame),
            "persistence_skill_present": True,
            "calendar_diagnostics_present": True,
            "butterfly_diagnostics_present": True,
            "text_conditions": sorted(set(frame["text_condition"].astype(str))),
            "completed_at_utc": utc_now(),
        },
    )
    return manifest


def _selector(
    *, model: str, train_tolerance: int, panel_role: str, condition: str
) -> dict[str, Any]:
    return {
        "model_id": model,
        "train_tolerance_minutes": int(train_tolerance),
        "panel_role": panel_role,
        "text_condition": condition,
    }


def _contrast(
    contrast_id: str,
    family_id: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    holm: bool,
    scope: str,
) -> dict[str, Any]:
    candidate_tolerance = int(candidate["train_tolerance_minutes"])
    reference_tolerance = int(reference["train_tolerance_minutes"])
    panel_role = str(candidate["panel_role"])
    if candidate_tolerance != reference_tolerance:
        comparison_relation = "expanded_training_window_vs_5m_on_common5"
    elif panel_role == "own_tolerance_secondary":
        comparison_relation = "same_training_tolerance_own_panel"
    elif candidate.get("text_condition") != reference.get("text_condition"):
        comparison_relation = "same_checkpoint_text_intervention_common5"
    else:
        comparison_relation = "same_training_tolerance_common5"
    return {
        "contrast_id": contrast_id,
        "family_id": family_id,
        "apply_holm": bool(holm),
        "scope": scope,
        "comparison_relation": comparison_relation,
        "candidate": dict(candidate),
        "reference": dict(reference),
    }


def _architecture_contrasts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary = "architecture_primary_new_modes_vs_film_holm3"
    for mode in study.ARCHITECTURE_MODES:
        candidate = _selector(
            model=f"formal:{mode}",
            train_tolerance=5,
            panel_role="common_5m_primary",
            condition="matched",
        )
        rows.append(
            _contrast(
                f"{mode}_vs_film_reference",
                primary,
                candidate,
                _selector(
                    model="film_reference",
                    train_tolerance=5,
                    panel_role="common_5m_primary",
                    condition="matched",
                ),
                holm=True,
                scope="primary",
            )
        )
        rows.append(
            _contrast(
                f"{mode}_vs_pure_cnn_reference",
                "architecture_vs_pure_cnn_descriptive",
                candidate,
                _selector(
                    model="pure_cnn_reference",
                    train_tolerance=5,
                    panel_role="common_5m_primary",
                    condition="zero",
                ),
                holm=False,
                scope="secondary_descriptive",
            )
        )
    rows.append(
        _contrast(
            "scaled_point_leader_vs_scaled_film",
            "architecture_scaled_descriptive",
            _selector(
                model="scaled:validation_point_leader_scaled",
                train_tolerance=5,
                panel_role="common_5m_primary",
                condition="matched",
            ),
            _selector(
                model="scaled:film_reference_scaled",
                train_tolerance=5,
                panel_role="common_5m_primary",
                condition="matched",
            ),
            holm=False,
            scope="secondary_descriptive",
        )
    )
    conditional_models = sorted(
        set(
            frame.loc[
                frame["generator_mode"].astype(str).ne("cnn_unet_mask_coords_v1"),
                "model_id",
            ].astype(str)
        )
    )
    for model in conditional_models:
        for reference_condition in ("zero", "shuffle"):
            rows.append(
                _contrast(
                    f"{model}_matched_vs_{reference_condition}",
                    f"architecture_text_reliance_{model}_holm2",
                    _selector(
                        model=model,
                        train_tolerance=5,
                        panel_role="common_5m_primary",
                        condition="matched",
                    ),
                    _selector(
                        model=model,
                        train_tolerance=5,
                        panel_role="common_5m_primary",
                        condition=reference_condition,
                    ),
                    holm=True,
                    scope="text_intervention",
                )
            )
    return rows


def _window_contrasts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tolerance in study.WINDOW_TOLERANCES:
        rows.append(
            _contrast(
                f"common5_train{tolerance:02d}_film_vs_pure_cnn",
                "window_between_models_common5_holm5",
                _selector(
                    model="film_lp_matched",
                    train_tolerance=tolerance,
                    panel_role="common_5m_primary",
                    condition="matched",
                ),
                _selector(
                    model="pure_cnn_no_text",
                    train_tolerance=tolerance,
                    panel_role="common_5m_primary",
                    condition="zero",
                ),
                holm=True,
                scope="primary",
            )
        )
        rows.append(
            _contrast(
                f"common5_train{tolerance:02d}_film_matched_vs_zero",
                "window_matched_vs_zero_descriptive",
                _selector(
                    model="film_lp_matched",
                    train_tolerance=tolerance,
                    panel_role="common_5m_primary",
                    condition="matched",
                ),
                _selector(
                    model="film_lp_matched",
                    train_tolerance=tolerance,
                    panel_role="common_5m_primary",
                    condition="zero",
                ),
                holm=False,
                scope="text_intervention_descriptive",
            )
        )
    for model, condition in (
        ("film_lp_matched", "matched"),
        ("pure_cnn_no_text", "zero"),
    ):
        for tolerance in (10, 15, 20, 30):
            rows.append(
                _contrast(
                    f"common5_{model}_train{tolerance:02d}_vs_train05",
                    f"window_within_{model}_common5_holm4",
                    _selector(
                        model=model,
                        train_tolerance=tolerance,
                        panel_role="common_5m_primary",
                        condition=condition,
                    ),
                    _selector(
                        model=model,
                        train_tolerance=5,
                        panel_role="common_5m_primary",
                        condition=condition,
                    ),
                    holm=True,
                    scope="primary_tolerance_sensitivity",
                )
            )
    for tolerance in (10, 15, 20, 30):
        rows.append(
            _contrast(
                f"own_panel{tolerance:02d}_film_vs_pure_cnn",
                "window_own_panel_between_models_descriptive",
                _selector(
                    model="film_lp_matched",
                    train_tolerance=tolerance,
                    panel_role="own_tolerance_secondary",
                    condition="matched",
                ),
                _selector(
                    model="pure_cnn_no_text",
                    train_tolerance=tolerance,
                    panel_role="own_tolerance_secondary",
                    condition="zero",
                ),
                holm=False,
                scope="secondary_own_panel",
            )
        )
    return rows


def _select(frame: pd.DataFrame, selector: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame.copy()
    for key, value in selector.items():
        if key not in selected:
            raise ValueError(f"Contrast selector column missing: {key}")
        if key.endswith("_minutes"):
            selected = selected.loc[pd.to_numeric(selected[key]).eq(int(value))]
        else:
            selected = selected.loc[selected[key].astype(str).eq(str(value))]
    return selected


def _paired_pairs(
    frame: pd.DataFrame, *, candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> pd.DataFrame:
    left = _select(frame, candidate)
    right = _select(frame, reference)
    keys = ["seed", "fold", "pair_id"]
    for label, selected in (("candidate", left), ("reference", right)):
        if selected.empty or selected.duplicated(keys).any():
            raise ValueError(
                f"{label} selector is empty or non-unique: {candidate}/{reference}"
            )
    merged = left[keys + ["session_id", "target_mae"]].merge(
        right[keys + ["session_id", "target_mae"]],
        on=keys,
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"Contrast pair universe is not exact: {len(left)}/{len(right)}/{len(merged)}"
        )
    if (
        not merged["session_id_candidate"]
        .astype(str)
        .equals(merged["session_id_reference"].astype(str))
    ):
        raise ValueError("Paired contrast has CME-session drift")
    return merged.rename(columns={"session_id_candidate": "session_id"}).drop(
        columns="session_id_reference"
    )


def _hierarchical_draws(
    paired: pd.DataFrame, *, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized-within-cluster seed -> fold -> session paired bootstrap.

    Pair-level candidate/reference errors are pre-aggregated once per CME
    session.  Each draw then samples only compact NumPy arrays, preserving the
    original cluster multiplicities and pair weighting without repeatedly
    slicing/concatenating pandas frames.
    """

    seeds = np.asarray(sorted(pd.to_numeric(paired["seed"]).astype(int).unique()))
    if len(seeds) != 3:
        raise ValueError(f"Bootstrap requires exactly three seeds, got {seeds}")
    working = paired.assign(
        seed=pd.to_numeric(paired["seed"], errors="raise").astype(int),
        fold=paired["fold"].astype(str),
        session_id=paired["session_id"].astype(str),
        target_mae_candidate=pd.to_numeric(
            paired["target_mae_candidate"], errors="raise"
        ),
        target_mae_reference=pd.to_numeric(
            paired["target_mae_reference"], errors="raise"
        ),
    )
    if (
        not np.isfinite(
            working[["target_mae_candidate", "target_mae_reference"]].to_numpy(
                dtype=float
            )
        ).all()
        or (working[["target_mae_candidate", "target_mae_reference"]] <= 0)
        .to_numpy()
        .any()
    ):
        raise ValueError("Bootstrap MAE values must be finite and positive")
    hierarchy: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for seed in seeds:
        seed_rows = working.loc[working["seed"].eq(int(seed))]
        folds = sorted(seed_rows["fold"].unique())
        if len(folds) != 4:
            raise ValueError("Bootstrap requires four folds within every seed")
        fold_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for fold in folds:
            sessions = (
                seed_rows.loc[seed_rows["fold"].eq(fold)]
                .groupby("session_id", sort=True, as_index=False)
                .agg(
                    candidate_sum=("target_mae_candidate", "sum"),
                    reference_sum=("target_mae_reference", "sum"),
                    pair_count=("target_mae_candidate", "size"),
                )
            )
            if len(sessions) < 2:
                raise ValueError("Bootstrap fold contains fewer than two CME sessions")
            fold_blocks.append(
                (
                    sessions["candidate_sum"].to_numpy(dtype=float),
                    sessions["reference_sum"].to_numpy(dtype=float),
                    sessions["pair_count"].to_numpy(dtype=np.int64),
                )
            )
        hierarchy.append(fold_blocks)

    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        candidate_sum = 0.0
        reference_sum = 0.0
        pair_count = 0
        for seed_index in rng.integers(0, len(hierarchy), size=len(hierarchy)):
            fold_blocks = hierarchy[int(seed_index)]
            for fold_index in rng.integers(0, len(fold_blocks), size=len(fold_blocks)):
                candidate, reference, counts = fold_blocks[int(fold_index)]
                session_indexes = rng.integers(0, len(counts), size=len(counts))
                candidate_sum += float(candidate[session_indexes].sum())
                reference_sum += float(reference[session_indexes].sum())
                pair_count += int(counts[session_indexes].sum())
        if pair_count <= 0 or candidate_sum <= 0 or reference_sum <= 0:
            raise ValueError("Bootstrap draw has an invalid aggregate")
        # Candidate and reference share the same sampled pair count, so the
        # ratio of their means is exactly the ratio of their sums.
        draws[index] = math.log(candidate_sum / reference_sum)
    return draws


def _holm_family(rows: list[dict[str, Any]]) -> None:
    by_family: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["apply_holm"]:
            by_family.setdefault(str(row["family_id"]), []).append(index)
        else:
            row["holm_p_value"] = float("nan")
    for indexes in by_family.values():
        ordered = sorted(
            indexes, key=lambda index: float(rows[index]["p_value_one_sided"])
        )
        running = 0.0
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0, (len(ordered) - rank) * float(rows[index]["p_value_one_sided"])
            )
            running = max(running, adjusted)
            rows[index]["holm_p_value"] = running


def bootstrap(root_or_path: str | Path, registry: Mapping[str, Any]) -> Path:
    root = Path(root_or_path).resolve()
    frame = _metrics(root)
    config = _config(root)
    analysis = study._mapping(config["analysis"], "analysis")
    replicates = int(analysis["bootstrap_replicates"])
    if replicates != 10_000:
        raise ValueError("Formal bootstrap must use exactly 10,000 replicates")
    confidence = float(analysis["confidence_level"])
    contrasts = (
        _architecture_contrasts(frame)
        if config["line"] == "architecture"
        else _window_contrasts()
    )
    rows: list[dict[str, Any]] = []
    base_seed = int(analysis["bootstrap_seed"])
    for index, contrast in enumerate(contrasts):
        paired = _paired_pairs(
            frame, candidate=contrast["candidate"], reference=contrast["reference"]
        )
        candidate_mae = float(paired["target_mae_candidate"].mean())
        reference_mae = float(paired["target_mae_reference"].mean())
        point = math.log(candidate_mae / reference_mae)
        draws = _hierarchical_draws(
            paired,
            replicates=replicates,
            rng=np.random.default_rng(base_seed + index * 1009),
        )
        alpha = 1.0 - confidence
        rows.append(
            {
                "contrast_id": contrast["contrast_id"],
                "family_id": contrast["family_id"],
                "apply_holm": contrast["apply_holm"],
                "scope": contrast["scope"],
                "comparison_relation": contrast["comparison_relation"],
                "candidate_selector_sha256": study._payload_sha256(
                    contrast["candidate"]
                ),
                "reference_selector_sha256": study._payload_sha256(
                    contrast["reference"]
                ),
                "paired_pairs": len(paired),
                "paired_sessions": int(paired["session_id"].nunique()),
                "candidate_mae": candidate_mae,
                "reference_mae": reference_mae,
                "log_mae_ratio": point,
                "relative_improvement": 1.0 - math.exp(point),
                "ci_lower": float(np.quantile(draws, alpha / 2.0)),
                "ci_upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
                "p_value_one_sided": float(
                    (1 + np.count_nonzero(draws >= 0)) / (replicates + 1)
                ),
            }
        )
    _holm_family(rows)
    output = root / "analysis/bootstrap_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    family_counts = (
        pd.Series([row["family_id"] for row in rows if row["apply_holm"]])
        .value_counts()
        .to_dict()
    )
    required_counts = (
        {"architecture_primary_new_modes_vs_film_holm3": 3}
        if config["line"] == "architecture"
        else {
            "window_between_models_common5_holm5": 5,
            "window_within_film_lp_matched_common5_holm4": 4,
            "window_within_pure_cnn_no_text_common5_holm4": 4,
        }
    )
    for family, expected in required_counts.items():
        if int(family_counts.get(family, 0)) != expected:
            raise ValueError(f"Holm family size drift: {family}")
    if config["line"] == "architecture":
        text_families = {
            family: int(count)
            for family, count in family_counts.items()
            if str(family).startswith("architecture_text_reliance_")
        }
        if len(text_families) != 6 or set(text_families.values()) != {2}:
            raise ValueError(
                f"Architecture text-reliance Holm-2 family drift: {text_families}"
            )
    manifest = root / "analysis/bootstrap_manifest.json"
    study._write_signed(
        manifest,
        {
            "schema_version": 1,
            "kind": "architecture_window_bootstrap_manifest_v1",
            "study_kind": registry["study_kind"],
            "replicates": replicates,
            "resampling_hierarchy": analysis["resampling_hierarchy"],
            "holm_family_counts": family_counts,
            "result_path": str(output.resolve()),
            "result_sha256": study._sha256_file(output),
            "completed_at_utc": utc_now(),
        },
    )
    return manifest


def report(root_or_path: str | Path, registry: Mapping[str, Any]) -> Path:
    root = Path(root_or_path).resolve()
    summary_path = root / "analysis/cell_diagnostics.csv"
    bootstrap_path = root / "analysis/bootstrap_results.csv"
    if not summary_path.is_file() or not bootstrap_path.is_file():
        raise RuntimeError("Report requires diagnostic and bootstrap outputs")
    results = pd.read_csv(bootstrap_path)
    metrics = _metrics(root)
    lines = [
        f"# {registry['study_kind']}",
        "",
        f"Interpretation: `{registry['interpretation']}`.",
        "",
        "This three-seed study is exploratory and is not evidence of stable cross-seed performance.",
        "",
        "The window is the maximum news-to-market alignment waiting tolerance; every model still predicts the next five-minute surface. Own-panel raw MAE values use different pair universes and therefore must not be compared directly across windows. Additional post-news samples at wider tolerances are neither a longer prediction horizon nor isolated causal evidence of a text effect.",
        "",
        "All selection used validation MAE only. Test surfaces, persistence skill, arbitrage diagnostics, and the design-frozen text interventions were opened only after checkpoint and prediction-plan freeze.",
        "",
        "## Frozen design",
        "",
    ]
    config = _config(root)
    design_html: list[str] = []
    if registry["line"] == "architecture":
        screen = study._read_signed(
            root / "registry/screen_selection.json",
            kind="architecture_screen_selection_v1",
        )
        architecture = study._read_signed(
            root / "registry/architecture_selection.json",
            kind="architecture_formal_selection_v1",
        )
        scaled = study._read_signed(
            root / "registry/scaled_capacity_profiles.json",
            kind="architecture_scaled_capacity_profiles_v1",
        )
        lines.append(
            "Frozen conditioning LR by mode: "
            + ", ".join(
                f"`{mode}`={row['conditioning_learning_rate']:.6g}"
                for mode, row in sorted(screen["selected_by_mode"].items())
            )
            + "."
        )
        lines.append(f"Validation point leader: `{architecture['point_leader_mode']}`.")
        lines.append(
            "Scaled Generator parameters: "
            + ", ".join(
                f"`{variant}`={int(profile['generator_parameters']):,}"
                for variant, profile in sorted(scaled["profiles"].items())
            )
            + "."
        )
        design_html.extend(
            [
                "<h2>Frozen design</h2>",
                "<p>Conditioning LR by mode: "
                + ", ".join(
                    f"<code>{html.escape(mode)}</code>={row['conditioning_learning_rate']:.6g}"
                    for mode, row in sorted(screen["selected_by_mode"].items())
                )
                + ".</p>",
                f"<p>Validation point leader: <code>{html.escape(str(architecture['point_leader_mode']))}</code>.</p>",
                "<p>Scaled Generator parameters: "
                + ", ".join(
                    f"<code>{html.escape(variant)}</code>={int(profile['generator_parameters']):,}"
                    for variant, profile in sorted(scaled["profiles"].items())
                )
                + ".</p>",
            ]
        )
    else:
        training = study._mapping(config["training"], "training")
        lines.append(
            "Frozen LRs: "
            f"backbone={float(training['backbone_learning_rate']):.6g}, "
            f"text={float(training['text_learning_rate']):.6g}, "
            f"conditioning={float(training['conditioning_learning_rate']):.6g}, "
            f"critic={float(training['critic_learning_rate']):.6g}."
        )
        lines.append("Generator parameters: FiLM c32=827,745; Pure-CNN c32=416,353.")
        design_html.extend(
            [
                "<h2>Frozen design</h2>",
                "<p>Frozen LRs: "
                f"backbone={float(training['backbone_learning_rate']):.6g}, "
                f"text={float(training['text_learning_rate']):.6g}, "
                f"conditioning={float(training['conditioning_learning_rate']):.6g}, "
                f"critic={float(training['critic_learning_rate']):.6g}.</p>",
                "<p>Generator parameters: FiLM c32=827,745; Pure-CNN c32=416,353.</p>",
            ]
        )
    lines.extend(
        [
            "",
            "| Scope | Family | Contrast | Candidate MAE | Reference MAE | Improvement | log(MAE ratio) | 95% CI | Holm p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results.to_dict(orient="records"):
        holm = "—" if pd.isna(row["holm_p_value"]) else f"{row['holm_p_value']:.6g}"
        lines.append(
            f"| {row['scope']} | {row['family_id']} | {row['contrast_id']} | "
            f"{row['candidate_mae']:.8g} | {row['reference_mae']:.8g} | "
            f"{100.0 * row['relative_improvement']:.4g}% | {row['log_mae_ratio']:.6g} | "
            f"[{row['ci_lower']:.6g}, {row['ci_upper']:.6g}] | {holm} |"
        )
    relation_summary = (
        metrics.groupby(
            [
                "model_id",
                "train_tolerance_minutes",
                "panel_role",
                "text_condition",
                "window_relation",
            ],
            as_index=False,
        )
        .agg(
            mae=("target_mae", "mean"),
            persistence_mae=("persistence_mae", "mean"),
            persistence_skill=("persistence_skill", "mean"),
            calendar_violation=("predicted_calendar_violation_rate", "mean"),
            butterfly_violation=("predicted_butterfly_violation_rate", "mean"),
            pair_count=("pair_id", "nunique"),
            session_count=("session_id", "nunique"),
            evaluation_pair_rows=("pair_id", "size"),
        )
        .sort_values(
            ["panel_role", "train_tolerance_minutes", "model_id", "window_relation"],
            kind="stable",
        )
    )
    lines.extend(
        [
            "",
            "## Window-relation diagnostics",
            "",
            "| Model | Train window | Panel | Text | Relation | MAE | Persistence MAE | Skill | Calendar | Butterfly | Unique pairs | Unique sessions | Evaluation rows |",
            "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in relation_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['model_id']} | {int(row['train_tolerance_minutes'])}m | "
            f"{row['panel_role']} | {row['text_condition']} | {row['window_relation']} | "
            f"{row['mae']:.8g} | {row['persistence_mae']:.8g} | "
            f"{row['persistence_skill']:.8g} | {row['calendar_violation']:.8g} | "
            f"{row['butterfly_violation']:.8g} | {int(row['pair_count'])} | "
            f"{int(row['session_count'])} | {int(row['evaluation_pair_rows'])} |"
        )
    if registry["line"] == "window":
        own = metrics.loc[
            metrics["panel_role"].astype(str).eq("own_tolerance_secondary")
        ]
        own_summary = (
            own.groupby(
                ["model_id", "train_tolerance_minutes", "text_condition"],
                as_index=False,
            )
            .agg(
                mae=("target_mae", "mean"),
                persistence_mae=("persistence_mae", "mean"),
                persistence_skill=("persistence_skill", "mean"),
                pair_count=("pair_id", "nunique"),
                session_count=("session_id", "nunique"),
                evaluation_pair_rows=("pair_id", "size"),
            )
            .sort_values(["train_tolerance_minutes", "model_id"], kind="stable")
        )
        widest = own_summary.loc[
            pd.to_numeric(own_summary["train_tolerance_minutes"]).eq(30),
            ["model_id", "text_condition", "pair_count"],
        ].rename(columns={"pair_count": "widest_30m_pair_count"})
        own_summary = own_summary.merge(
            widest,
            on=["model_id", "text_condition"],
            how="left",
            validate="many_to_one",
        )
        own_summary["coverage_fraction_vs_30m"] = (
            own_summary["pair_count"] / own_summary["widest_30m_pair_count"]
        )
        if (
            own_summary["widest_30m_pair_count"].isna().any()
            or not own_summary["coverage_fraction_vs_30m"].between(0, 1).all()
        ):
            raise ValueError("Own-panel 30m coverage denominator drift")
        lines.extend(
            [
                "",
                "## Own-panel coverage and persistence",
                "",
                "Coverage is unique own-panel pairs divided by the same model/text condition's 30m own-panel unique-pair count.",
                "",
                "| Model | Train/panel window | Text | MAE | Persistence MAE | Skill | Unique pairs | 30m denominator | Coverage fraction | Unique sessions | Evaluation rows |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in own_summary.to_dict(orient="records"):
            lines.append(
                f"| {row['model_id']} | {int(row['train_tolerance_minutes'])}m | "
                f"{row['text_condition']} | {row['mae']:.8g} | "
                f"{row['persistence_mae']:.8g} | {row['persistence_skill']:.8g} | "
                f"{int(row['pair_count'])} | {int(row['widest_30m_pair_count'])} | "
                f"{row['coverage_fraction_vs_30m']:.6g} | {int(row['session_count'])} | "
                f"{int(row['evaluation_pair_rows'])} |"
            )
    markdown = root / "report/conclusion.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result_columns = [
        "scope",
        "family_id",
        "contrast_id",
        "comparison_relation",
        "candidate_mae",
        "reference_mae",
        "relative_improvement",
        "log_mae_ratio",
        "ci_lower",
        "ci_upper",
        "holm_p_value",
    ]
    table_blocks = [
        *design_html,
        "<h2>Bootstrap contrasts</h2>",
        results[result_columns].to_html(index=False, border=0, classes="dataframe"),
        "<h2>Window-relation diagnostics</h2>",
        relation_summary.to_html(index=False, border=0, classes="dataframe"),
    ]
    if registry["line"] == "window":
        table_blocks.extend(
            [
                "<h2>Own-panel coverage and persistence</h2>",
                "<p>Coverage is unique own-panel pairs divided by the same model/text condition's 30m own-panel unique-pair count.</p>",
                own_summary.to_html(index=False, border=0, classes="dataframe"),
            ]
        )
    html_path = root / "report/report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;}"
        "table{border-collapse:collapse;width:100%;font-size:.86rem;margin-bottom:2rem;}"
        "th,td{border:1px solid #d0d7de;padding:.4rem;text-align:right;}"
        "th{background:#f6f8fa;}th:first-child,td:first-child{text-align:left;}"
        "</style></head><body>"
        f"<h1>{html.escape(str(registry['study_kind']))}</h1>"
        f"<p>Interpretation: <code>{html.escape(str(registry['interpretation']))}</code>.</p>"
        "<p>This three-seed study is exploratory and is not evidence of stable cross-seed performance.</p>"
        "<p>The window is the maximum news-to-market alignment waiting tolerance; every model still predicts the next five-minute surface. Own-panel raw MAE values use different pair universes and must not be compared directly across windows. Wider-window post-news additions are neither a longer prediction horizon nor isolated causal evidence of a text effect.</p>"
        "<p>Selection used validation MAE only; test metrics were opened only after the checkpoint and prediction-plan freeze.</p>"
        + "".join(table_blocks)
        + "</body></html>\n",
        encoding="utf-8",
    )
    manifest = root / "report/report_manifest.json"
    study._write_signed(
        manifest,
        {
            "schema_version": 1,
            "kind": "architecture_window_report_manifest_v1",
            "markdown_path": str(markdown.resolve()),
            "markdown_sha256": study._sha256_file(markdown),
            "html_path": str(html_path.resolve()),
            "html_sha256": study._sha256_file(html_path),
            "completed_at_utc": utc_now(),
        },
    )
    return manifest


__all__ = ["analyze", "bootstrap", "report"]
