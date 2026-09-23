"""Audit Chapter 3 observed MAEs from frozen predictions, without bootstrapping.

Run ``python -m scripts.rq123.chapter3_point_estimate_audit``. Every reported
point gives pairs equal weight within a seed/fold, seeds equal weight within
a fold, and all four folds equal weight. Historical artifacts are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/chapter3_observed_point_estimates_v1"
RQ12_ROOT = REPO_ROOT / (
    "outputs/experiments/rq12_news_first_vol_text_10seed_composite_"
    "lr2p5e5_descriptive_no_bootstrap_exact_ttm_rolling_v1"
)
RQ3_ROOT = REPO_ROOT / (
    "outputs/experiments/rq12_news_first_vol_film_unet_pure_cnn_backbone_"
    "text_effect_10seed_5m_v1"
)
DEFAULT_SOURCES = {
    "rq12": RQ12_ROOT / "analysis/text_10seed_composite_pair_metrics.csv",
    "rq3_standard": RQ3_ROOT / "evaluation/standard_pair_metrics.csv.gz",
    "rq3_intervention": RQ3_ROOT / "analysis/intervention_primary_panel.csv.gz",
}
FOLDS = ("f1_2023q1", "f2_2023q2", "f3_2023q3", "f4_2023q4")
SEEDS = (42, 202, 404, 382624741, 1607127774, 1662128673,
         2041145538, 2014889368, 1343862330, 779214671)
ARMS = {
    "rq12": ("lp_matched", "lp_shuffle", "no_text", "bow", "sentiment"),
    "rq3_standard": ("film_lp_matched", "film_zero_text", "film_lp_shuffle",
                     "film_bow", "film_sentiment", "pure_cnn_continue_no_text",
                     "pure_cnn_parent"),
    "rq3_intervention": ("film_lp_matched", "film_lp_matched__wrong_input",
                         "film_lp_matched__zero_input"),
}
PRIMARY_REFERENCE = {
    "rq12": "lp_matched", "rq3_standard": "pure_cnn_continue_no_text",
    "rq3_intervention": "film_lp_matched",
}
PAIRED_CONTRASTS = {
    "rq12": (("lp_matched", "no_text"), ("lp_matched", "bow"),
             ("lp_matched", "sentiment")),
    "rq3_standard": (("film_lp_matched", "film_lp_shuffle"),
                     ("film_lp_matched", "film_zero_text")),
    "rq3_intervention": (("film_lp_matched", "film_lp_matched__zero_input"),
                         ("film_lp_matched", "film_lp_matched__wrong_input")),
}
PAIR_KEYS = ["seed", "fold", "pair_id"]
IDENTITY_COLUMNS = ["session_id", "effective_origin_utc", "target_mae",
                    "persistence_mae", "checkpoint_sha256", "prediction_sha256",
                    "noise_bank_profile_sha256", "inference_determinism_contract_sha256"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_panel(frame: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    """Reject incomplete or unpaired four-fold, ten-seed prediction panels."""
    required = set(PAIR_KEYS + IDENTITY_COLUMNS + ["arm", "tolerance_minutes"])
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing pair-level columns: {sorted(missing)}")
    panel = frame.copy()
    if panel[list(required)].isna().any().any():
        raise ValueError("Missing pair-level values")
    if set(panel.arm) != set(arms):
        raise ValueError("Incomplete or unexpected arm set")
    if set(panel.fold) != set(FOLDS) or set(panel.seed) != set(SEEDS):
        raise ValueError("Expected exactly four rolling folds and ten canonical seeds")
    if not panel.tolerance_minutes.eq(5).all():
        raise ValueError("Expected the frozen five-minute tolerance panel")
    if panel.duplicated(["arm", *PAIR_KEYS]).any():
        raise ValueError("Duplicate arm/seed/fold/pair row")
    expected = pd.MultiIndex.from_product([arms, FOLDS, SEEDS], names=["arm", "fold", "seed"])
    cells = panel.groupby(["arm", "fold", "seed"]).size()
    if len(cells) != len(expected) or not expected.isin(cells.index).all():
        raise ValueError("Missing arm/seed/fold cell")
    losses = panel[["target_mae", "persistence_mae"]].to_numpy(float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("MAEs must be finite and nonnegative")
    for fold, group in panel.groupby("fold"):
        reference = None
        for _, cell in group.groupby(["arm", "seed"]):
            market = cell.set_index("pair_id")[["session_id", "effective_origin_utc",
                                                "persistence_mae"]].sort_index()
            if reference is None:
                reference = market
            elif not reference.equals(market):
                raise ValueError(f"Unpaired market observations in {fold}")
            if cell.checkpoint_sha256.nunique() != 1 or cell.prediction_sha256.nunique() != 1:
                raise ValueError("Multiple prediction/checkpoint artifacts within one cell")
    return panel


def matched_identity(standard: pd.DataFrame, intervention: pd.DataFrame) -> dict:
    """Require the unchanged matched input to be the same frozen prediction."""
    compared = []
    for panel in (standard, intervention):
        compared.append(panel.loc[panel.arm.eq("film_lp_matched")]
                        .set_index(PAIR_KEYS)[IDENTITY_COLUMNS].sort_index())
    if compared[0].empty or not compared[0].equals(compared[1]):
        raise ValueError("RQ3 matched checkpoint/prediction identity failed")
    return {"passed": True, "pair_seed_rows": len(compared[0]),
            "compared_columns": [*PAIR_KEYS, *IDENTITY_COLUMNS]}


def _derived(frame: pd.DataFrame, reference: str, keys: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["improvement_vs_persistence_percent"] = (
        100 * (1 - result.observed_mean_mae / result.persistence_mean_mae)
    )
    reference_values = result.loc[result.arm.eq(reference), [*keys, "observed_mean_mae"]]
    reference_values = reference_values.rename(columns={"observed_mean_mae": "reference_mean_mae"})
    result = result.merge(reference_values, on=keys, how="left", validate="many_to_one")
    result["reference_arm"] = reference
    result["mae_divided_by_reference"] = result.observed_mean_mae / result.reference_mean_mae
    return result


def summarise_panel(panel: pd.DataFrame, dataset: str) -> dict[str, pd.DataFrame]:
    """Summarise raw pair losses; no draw means or archived summaries are read."""
    cells = panel.groupby(["arm", "seed", "fold"], as_index=False).agg(
        observed_mean_mae=("target_mae", "mean"),
        persistence_mean_mae=("persistence_mae", "mean"),
        pair_count=("pair_id", "nunique"), session_count=("session_id", "nunique"),
    )
    if (cells[["observed_mean_mae", "persistence_mean_mae"]] <= 0).any().any():
        raise ValueError("Positive cell MAEs required for log-ratio contrasts")
    persistence = cells.drop_duplicates(["seed", "fold"]).copy()
    persistence["arm"] = "persistence"
    persistence["observed_mean_mae"] = persistence.persistence_mean_mae
    cells = pd.concat([cells, persistence], ignore_index=True)
    cells["dataset"] = dataset
    folds = cells.groupby(["dataset", "arm", "fold"], as_index=False).agg(
        observed_mean_mae=("observed_mean_mae", "mean"),
        persistence_mean_mae=("persistence_mean_mae", "mean"),
        seed_count=("seed", "nunique"), pair_count=("pair_count", "first"),
        session_count=("session_count", "first"),
    )
    overall = folds.groupby(["dataset", "arm"], as_index=False).agg(
        observed_mean_mae=("observed_mean_mae", "mean"),
        persistence_mean_mae=("persistence_mean_mae", "mean"),
        fold_count=("fold", "nunique"), seed_count=("seed_count", "first"),
        pair_count=("pair_count", "sum"), session_count=("session_count", "sum"),
    )
    reference = PRIMARY_REFERENCE[dataset]
    result = {
        "cell_summary": _derived(cells, reference, ["dataset", "seed", "fold"]),
        "fold_summary": _derived(folds, reference, ["dataset", "fold"]),
        "overall_summary": _derived(overall, reference, ["dataset"]),
    }
    wide = cells.pivot(index=["seed", "fold"], columns="arm", values="observed_mean_mae")
    contrasts = []
    recipes = [(arm, "persistence") for arm in ARMS[dataset]] + list(PAIRED_CONTRASTS[dataset])
    for focal, comparator in recipes:
        log_ratio = np.log(wide[focal] / wide[comparator])
        for fold in (*FOLDS, "overall"):
            selected = log_ratio if fold == "overall" else log_ratio.xs(fold, level="fold")
            point = float(selected.mean())
            contrasts.append({"dataset": dataset, "fold": fold, "focal": focal,
                              "reference": comparator, "mean_cell_log_mae_ratio": point,
                              "geometric_gain_percent": float(100 * -np.expm1(point)),
                              "seed_fold_cell_count": len(selected)})
    result["contrasts"] = pd.DataFrame(contrasts)
    return result


def run_audit(sources: dict[str, Path] | None = None, output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    sources = DEFAULT_SOURCES if sources is None else sources
    if set(sources) != set(DEFAULT_SOURCES):
        raise ValueError("Expected rq12, rq3_standard and rq3_intervention sources")
    manifest = []
    panels = {}
    for name, path in sources.items():
        path = path.resolve()
        digest = sha256_file(path)
        panels[name] = validate_panel(pd.read_csv(path), ARMS[name])
        if digest != sha256_file(path):
            raise ValueError(f"Source changed while being read: {path}")
        manifest.append({"dataset": name, "path": str(path), "sha256": digest,
                         "rows": len(panels[name])})
    identity = matched_identity(panels["rq3_standard"], panels["rq3_intervention"])
    tables = [summarise_panel(panels[name], name) for name in DEFAULT_SOURCES]
    combined = {name: pd.concat([part[name] for part in tables], ignore_index=True)
                for name in tables[0]}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.attempt-", dir=output.parent))
    for name, table in combined.items():
        table.to_csv(staging / f"{name}.csv", index=False, float_format="%.17g")
    payload = {"schema_version": 1, "kind": "chapter3_observed_point_estimate_audit",
               "sources": manifest, "matched_prediction_identity": identity,
               "weighting": "equal pairs within seed-fold; equal seeds; equal four folds",
               "bootstrap_inputs_used": False, "inference_recomputed": False,
               "implementation_sha256": sha256_file(Path(__file__))}
    (staging / "audit_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Chapter 3 observed point-estimate audit", "",
             "All results use four rolling folds and ten seeds. Pair losses are averaged "
             "within seed/fold, seeds equally within fold, then the four folds equally. "
             "Different fold sample sizes do not change each fold's one-quarter weight.", "",
             "The input consists exclusively of frozen pair-level prediction losses. "
             "Bootstrap means are not point estimates and are not inputs to this audit. "
             "No model, prediction, bootstrap draw, confidence interval or p-value is recomputed.", "",
             f"RQ3 matched input reproduces the standard matched branch exactly across "
             f"{identity['pair_seed_rows']:,} pair/seed rows, including checkpoint and prediction hashes.", "",
             "Arithmetic improvements are ratios of the observed MAEs. Paired log contrasts "
             "average log ratios of original seed/fold cell MAEs; their geometric gains "
             "need not equal arithmetic improvements.", "",
             "| Dataset | Arm | Observed four-fold MAE | Improvement vs persistence (%) |",
             "|---|---|---:|---:|"]
    for row in combined["overall_summary"].itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.arm} | {row.observed_mean_mae:.13f} | "
                     f"{row.improvement_vs_persistence_percent:.8f} |")
    lines += ["", "Source paths and SHA-256 hashes are in `audit_manifest.json`. "
              "`cell_summary.csv`, `fold_summary.csv`, `overall_summary.csv` and "
              "`contrasts.csv` retain unrounded observed estimates."]
    (staging / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if output.exists():
        raise FileExistsError(f"Audit output appeared during processing: {output}")
    staging.rename(output)
    print(f"Audit complete: {output}; RQ3 matched identity: {identity['pair_seed_rows']} rows")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for name, path in DEFAULT_SOURCES.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path)
    args = parser.parse_args()
    run_audit({name: getattr(args, name) for name in DEFAULT_SOURCES}, args.output)


if __name__ == "__main__":
    main()
