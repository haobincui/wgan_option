"""Read-only reconciliation of Chapter 3 observed MAE point estimates.

Run ``python -m scripts.rq123.audit_evaluation_estimands`` for JSON on stdout.
The command reads frozen pair metrics; it never trains, predicts, bootstraps,
or rewrites experiment outputs. Bootstrap means are not point estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path("outputs/experiments")
DIRECT = EXPERIMENTS / "rq12_news_first_vol_text_10seed_composite_lr2p5e5_descriptive_no_bootstrap_exact_ttm_rolling_v1"
RQ3 = EXPERIMENTS / "rq12_news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_5m_v1"
SOURCES = {
    "direct": (DIRECT / "analysis/text_10seed_composite_pair_metrics.csv", "arm", "d832b3b057ea440845bdd97312744bf5f75e706c80c2057a44d8f88e64f38754"),
    "rq3_full": (RQ3 / "evaluation/standard_pair_metrics.csv.gz", "arm", "87e70dc0842c9fa99dcf98c40c9ec48c23ce1841a6e38f08559d5ea8e2db90e5"),
    "rq3_branch": (RQ3 / "analysis/standard_primary_panel.csv.gz", "arm", "705783913e5252d8754cf9820ba9ca3c6a361dcc4fe6defec0fada40817cf041"),
    "rq3_intervention": (RQ3 / "analysis/intervention_primary_panel.csv.gz", "input_condition", "4898c79b51db2d7aacee8c749e14b2edf18e8920ce50c9c04d59375b4d77009b"),
}
EXPECTED_COUNTS = {"f1_2023q1": (110, 34), "f2_2023q2": (112, 36), "f3_2023q3": (135, 33), "f4_2023q4": (143, 45)}
DEFAULT_SAVED_SUMMARY = Path("outputs/analysis/chapter3_shared_market_panel_bootstrap_10000_v2/analysis/all_arm_summary.csv")
KEYS = ["seed", "fold", "pair_id", "session_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_panel(frame: pd.DataFrame, condition_column: str) -> pd.DataFrame:
    """Validate equal market panels and retain one row per seed/arm/pair."""
    needed = [condition_column, *KEYS, "target_mae", "persistence_mae"]
    if not set(needed).issubset(frame.columns):
        raise ValueError("Missing pair-metric columns")
    result = frame[needed].rename(columns={condition_column: "condition"}).copy()
    if result.empty or result.isna().any().any():
        raise ValueError("Empty panel or missing values")
    for column in ["condition", "fold", "pair_id", "session_id"]:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError("Blank panel identifiers")
    seeds = pd.to_numeric(result["seed"], errors="raise").to_numpy(float)
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Seeds must be finite integers")
    result["seed"] = seeds.astype(np.int64)
    for column in ["target_mae", "persistence_mae"]:
        values = pd.to_numeric(result[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("MAEs must be finite and strictly positive")
        result[column] = values
    if result.duplicated(["condition", "seed", "fold", "pair_id"]).any():
        raise ValueError("Duplicate condition/seed/fold/pair rows")
    lineage_columns = ["fold", "pair_id", "session_id"]
    first = result.iloc[0]
    reference = result[result.condition.eq(first.condition) & result.seed.eq(first.seed)]
    lineage = set(reference[lineage_columns].itertuples(index=False, name=None))
    for condition in result.condition.unique():
        for seed in result.seed.unique():
            selected = result[result.condition.eq(condition) & result.seed.eq(seed)]
            if set(selected[lineage_columns].itertuples(index=False, name=None)) != lineage:
                raise ValueError("Unequal market lineage across conditions/seeds")
    if reference.pair_id.duplicated().any():
        raise ValueError("A pair occurs in multiple folds or sessions")
    if reference.groupby("session_id").fold.nunique().gt(1).any():
        raise ValueError("Sessions must be nested in folds")
    if result.groupby(lineage_columns).persistence_mae.nunique().ne(1).any():
        raise ValueError("Persistence differs across conditions/seeds")
    return result


def cell_means(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby(["condition", "seed", "fold"], sort=True)[["target_mae", "persistence_mae"]].mean()


def summarize(panel: pd.DataFrame, *, by_fold: bool, reference: str) -> list[dict]:
    """Average pairs, then seeds, then folds equally for a four-fold aggregate."""
    cells = cell_means(panel)
    means = cells.groupby(["condition", "fold"] if by_fold else ["condition"]).mean()
    rows = []
    folds = sorted(panel.fold.unique()) if by_fold else ["all_four_folds"]
    for fold in folds:
        group = means.xs(fold, level="fold") if by_fold else means
        persistence = float(group.persistence_mae.iloc[0])
        reference_mae = float(group.loc[reference, "target_mae"])
        for condition, mae in [*(group.target_mae.items()), ("persistence", persistence)]:
            rows.append({"condition": condition, "fold": fold, "observed_mean_mae": float(mae),
                         "improvement_vs_persistence_percent": 100 * (1 - float(mae) / persistence),
                         "normalized_mae": float(mae) / reference_mae, "normalization_reference": reference})
    return rows


def require_same_predictions(left: pd.DataFrame, left_condition: str,
                             right: pd.DataFrame, right_condition: str) -> None:
    def selected(panel, condition):
        return panel[panel.condition.eq(condition)].set_index(KEYS)[["target_mae", "persistence_mae"]].sort_index()
    a, b = selected(left, left_condition), selected(right, right_condition)
    if a.empty or not a.index.equals(b.index) or not np.array_equal(a.to_numpy(), b.to_numpy()):
        raise ValueError("Matched-condition predictions or lineage differ")


def contrast_points(panel: pd.DataFrame, focal: str, references: list[str]) -> list[dict]:
    values = cell_means(panel).target_mae.unstack("condition")
    return [{"focal": focal, "reference": reference,
             "mean_cell_log_mae_ratio": float(np.log(values[focal] / values[reference]).mean()),
             "ratio_of_arithmetic_maes": float(values[focal].mean() / values[reference].mean())}
            for reference in references]


def verify_saved_summary(frame: pd.DataFrame, table_rows: list[dict]) -> int:
    """Check available 10-seed primary arm summaries, not their uncertainty."""
    required = {"condition", "seed_count", "fold_count", "pair_count", "observed_mean_mae"}
    if not required.issubset(frame.columns):
        raise ValueError("Saved arm summary lacks required columns")
    expected = {}
    for row in table_rows:
        fold = row["fold"]
        key = (row["condition"], 4 if fold == "all_four_folds" else 1,
               500 if fold == "all_four_folds" else EXPECTED_COUNTS[fold][0])
        old = expected.setdefault(key, row["observed_mean_mae"])
        if not np.isclose(old, row["observed_mean_mae"], rtol=0, atol=2e-15):
            raise ValueError("Inconsistent duplicated table point estimates")
    checked = 0
    for row in frame.to_dict("records"):
        if row["seed_count"] != 10 or row.get("estimand", "equal_cell") != "equal_cell":
            continue
        key = (row["condition"], row["fold_count"], row["pair_count"])
        if key in expected:
            if not np.isclose(row["observed_mean_mae"], expected[key], rtol=0, atol=2e-15):
                raise ValueError(f"Saved observed mean differs: {key}")
            checked += 1
    if checked == 0:
        raise ValueError("No primary observed arm means found in saved summary")
    return checked


def run_audit(repo_root: Path = REPO_ROOT, saved_summary: Path | None = None) -> dict:
    panels, manifests = {}, []
    for name, (relative_path, condition, expected_hash) in SOURCES.items():
        path = repo_root / relative_path
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen source checksum mismatch: {path}")
        panel = canonical_panel(pd.read_csv(path), condition)
        if panel.seed.nunique() != 10 or set(panel.fold.unique()) != set(EXPECTED_COUNTS):
            raise ValueError("Primary seed/fold universe drift")
        for fold, (pairs, sessions) in EXPECTED_COUNTS.items():
            selected = panel[panel.fold.eq(fold)]
            if selected.pair_id.nunique() != pairs or selected.session_id.nunique() != sessions:
                raise ValueError("Primary pair/session counts drift")
        panels[name] = panel
        manifests.append({"source": name, "path": str(relative_path), "sha256": actual_hash,
                          "rows": len(panel), "unique_pairs": panel.pair_id.nunique(),
                          "sessions": panel.session_id.nunique()})
    require_same_predictions(panels["rq3_full"], "film_lp_matched", panels["rq3_branch"], "matched")
    require_same_predictions(panels["rq3_full"], "film_lp_matched", panels["rq3_intervention"], "matched_input")
    direct = panels["direct"]
    rq1 = direct[direct.condition.isin(["lp_matched", "no_text"])]
    rq2 = direct[direct.condition.isin(["lp_matched", "bow", "sentiment"])]
    require_same_predictions(rq1, "lp_matched", rq2, "lp_matched")
    tables = {"rq1_quarter": summarize(rq1, by_fold=True, reference="lp_matched"),
              "rq2_quarter": summarize(rq2, by_fold=True, reference="lp_matched"),
              "direct_all_arms_quarter": summarize(direct, by_fold=True, reference="lp_matched"),
              "direct_four_fold": summarize(direct, by_fold=False, reference="lp_matched"),
              "rq3_four_fold": summarize(panels["rq3_full"], by_fold=False, reference="pure_cnn_continue_no_text"),
              "rq3_intervention": summarize(panels["rq3_intervention"], by_fold=False, reference="matched_input")}
    summary_path = saved_summary or repo_root / DEFAULT_SAVED_SUMMARY
    summary_check = {"path": str(summary_path), "status": "not_available_not_verified"}
    if summary_path.is_file():
        summary_hash = sha256(summary_path)
        checked = verify_saved_summary(pd.read_csv(summary_path), [row for rows in tables.values() for row in rows])
        if sha256(summary_path) != summary_hash:
            raise ValueError("Saved summary changed during audit")
        summary_check.update(status="observed_means_verified_only", rows_checked=checked, sha256=summary_hash)
    elif saved_summary is not None:
        raise FileNotFoundError(summary_path)
    for manifest in manifests:
        if sha256(repo_root / manifest["path"]) != manifest["sha256"]:
            raise ValueError("Source changed during audit")
    matched = next(row["observed_mean_mae"] for row in tables["rq3_four_fold"] if row["condition"] == "film_lp_matched")
    return {"status": "observed_point_estimates_verified", "writes_experiment_outputs": False,
            "weighting": "equal pairs within seed-fold; equal seeds; equal folds for aggregate",
            "fold_pair_session_counts": EXPECTED_COUNTS, "source_manifest": manifests,
            "same_matched_predictions_across_branch_and_intervention": True,
            "rq1_rq2_share_matched_and_persistence_inputs": True, "tables": tables,
            "contrasts": {"rq1": contrast_points(direct, "lp_matched", ["no_text"]),
                          "rq2": contrast_points(direct, "lp_matched", ["bow", "sentiment"]),
                          "rq3": contrast_points(panels["rq3_full"], "film_lp_matched", ["film_zero_text", "film_lp_shuffle"])},
            "rq3_historical_display_reconciliation": {"previous_tex_bootstrap_mean_rounded": 0.002040977,
                 "observed_mean_mae": matched, "difference_from_rounded_tex_mean": 0.002040977 - matched,
                 "previous_bootstrap_mean_reproduced": False,
                 "note": "Historical TeX number is provenance context, not a re-executed bootstrap result."},
            "saved_summary_check": summary_check,
            "limitations": ["Pair MAEs are frozen inputs; prediction tensors and model checkpoints are not recomputed.",
                            "No bootstrap draws, standard errors, p-values, or confidence intervals are verified by this audit."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--saved-arm-summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.repo_root, args.saved_arm_summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
