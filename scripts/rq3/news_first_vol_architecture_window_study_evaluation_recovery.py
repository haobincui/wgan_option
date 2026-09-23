"""Audited recovery for zero-constraint architecture/window diagnostics.

The active Architecture and Window roots froze their orchestration and
analysis sources before prediction started.  The frozen evaluator rejects any
non-finite pair-level value, even though the upstream raw-joint support
contract intentionally reports calendar/butterfly violation rates as ``NaN``
when a pair has no supported constraints.  Editing either frozen source would
invalidate all completed training lineage.

This sidecar therefore keeps the frozen sources byte-for-byte unchanged and
temporarily installs two narrowly scoped corrections while resuming only the
post-training evaluation pipeline:

* pair evidence retains all four predicted/target constraint denominators;
* rates and gaps must be finite exactly when their denominator is positive,
  and must be ``NaN`` when the denominator is zero; and
* the downstream analysis loader enforces the same conditional contract.

The recovery source and both frozen sources are hash-attested in a signed
manifest inside each formal root.  Training jobs and checkpoints are verified
but never launched or rewritten by this runner.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_architecture_window_study as study
from scripts.rq3 import news_first_vol_architecture_window_study_analysis as analysis


RECOVERY_KIND = "architecture_window_zero_constraint_evaluation_recovery_v1"
RECOVERY_PATH = "registry/evaluation_metric_recovery.json"
MISSINGNESS_CONTRACT = "zero_supported_constraints_implies_nan_rate_v1"
FROZEN_SOURCE_BINDINGS = {
    "scripts/rq3/news_first_vol_architecture_window_study.py": {
        "size_bytes": 233_682,
        "sha256": "ade87b2d47d117af110df188c45044c8365116f4f75c7ab75491f8ed4de52a07",
    },
    "scripts/rq3/news_first_vol_architecture_window_study_analysis.py": {
        "size_bytes": 37_291,
        "sha256": "55b39bba8eedf19d872353fb175eef0ce7b53a0ce6173d5ce93b666d59f7284d",
    },
    "scripts/rq3/news_first_vol_comparison_analysis.py": {
        "size_bytes": 201_766,
        "sha256": "c846394184485a1bd4a0cc10c550b7e4604d766af8f2eee4bd7494be65b73d27",
    },
}
FORMAL_CONFIGS = {
    "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1": (
        study.DEFAULT_ARCHITECTURE_CONFIG
    ),
    "rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1": (
        study.DEFAULT_WINDOW_CONFIG
    ),
}
PRE_RECOVERY_REGISTRY_SHA256 = {
    "rq3_news_first_vol_generator_architecture_3seed_5m_exact_ttm_v1": (
        "6d7c8a369f13d20ef2cc6f285c25ee48f74836aa283b8f1c700ed7befa026774"
    ),
    "rq3_news_first_vol_alignment_tolerance_3seed_exact_ttm_v1": (
        "97bbbabd2ca31f3e99ef755d87e9a0e81646a7abe88a70e8ab73e9cfa98b517e"
    ),
}
CONSTRAINT_FAMILIES = (
    (
        "calendar",
        "predicted_calendar_constraint_count",
        "target_calendar_constraint_count",
        "predicted_calendar_violation_rate",
        "target_calendar_violation_rate",
        "calendar_violation_rate_gap",
    ),
    (
        "butterfly",
        "predicted_butterfly_constraint_count",
        "target_butterfly_constraint_count",
        "predicted_butterfly_violation_rate",
        "target_butterfly_violation_rate",
        "butterfly_violation_rate_gap",
    ),
)
CORE_FINITE_COLUMNS = {
    "post_news_current_overlap_minutes",
    "target_mae",
    "persistence_mae",
    "persistence_skill",
}
CONSTRAINT_COUNT_COLUMNS = {
    column
    for _family, predicted, target, _predicted_rate, _target_rate, _gap in (
        CONSTRAINT_FAMILIES
    )
    for column in (predicted, target)
}


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Pair metric {column} is missing or non-numeric") from exc


def validate_pair_metric_contract(frame: pd.DataFrame) -> None:
    """Fail closed while allowing only structurally undefined diagnostics."""

    required = CORE_FINITE_COLUMNS | CONSTRAINT_COUNT_COLUMNS
    required.update(
        column
        for _family, _predicted, _target, predicted_rate, target_rate, gap in (
            CONSTRAINT_FAMILIES
        )
        for column in (predicted_rate, target_rate, gap)
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair metrics missing recovery diagnostics: {missing}")
    if frame.empty:
        raise ValueError("Pair metrics are empty")

    for column in sorted(CORE_FINITE_COLUMNS | CONSTRAINT_COUNT_COLUMNS):
        values = _numeric(frame, column)
        if not np.isfinite(values).all():
            raise ValueError(f"Pair metric {column} contains missing/non-finite values")

    target_mae = _numeric(frame, "target_mae")
    persistence_mae = _numeric(frame, "persistence_mae")
    if (target_mae < 0.0).any() or (persistence_mae <= 0.0).any():
        raise ValueError("MAE values are outside the valid domain")

    for (
        family,
        predicted_count_column,
        target_count_column,
        predicted_rate_column,
        target_rate_column,
        gap_column,
    ) in CONSTRAINT_FAMILIES:
        predicted_count = _numeric(frame, predicted_count_column)
        target_count = _numeric(frame, target_count_column)
        if (predicted_count < 0.0).any() or (target_count < 0.0).any():
            raise ValueError(f"{family} constraint counts must be non-negative")
        if not np.equal(predicted_count, target_count).all():
            raise ValueError(f"Predicted/target {family} constraint counts differ")
        if not (
            np.isclose(predicted_count, np.rint(predicted_count), rtol=0.0, atol=1e-12)
        ).all():
            raise ValueError(f"{family} constraint counts must be integer-valued")

        predicted_rate = _numeric(frame, predicted_rate_column)
        target_rate = _numeric(frame, target_rate_column)
        gap = _numeric(frame, gap_column)
        positive = predicted_count > 0.0
        zero = ~positive
        for column, values in (
            (predicted_rate_column, predicted_rate),
            (target_rate_column, target_rate),
            (gap_column, gap),
        ):
            if positive.any() and not np.isfinite(values[positive]).all():
                raise ValueError(
                    f"Pair metric {column} must be finite when {family} constraints exist"
                )
            if zero.any() and not np.isnan(values[zero]).all():
                raise ValueError(
                    f"Pair metric {column} must be NaN when {family} constraints are absent"
                )
        for column, values in (
            (predicted_rate_column, predicted_rate),
            (target_rate_column, target_rate),
        ):
            if positive.any() and (
                (values[positive] < 0.0).any() or (values[positive] > 1.0).any()
            ):
                raise ValueError(f"Pair metric {column} is outside [0, 1]")
        if positive.any() and not np.allclose(
            gap[positive],
            predicted_rate[positive] - target_rate[positive],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"{family} violation-rate gap is inconsistent")


def _frozen_source_attestation(root: Path) -> dict[str, Any]:
    """Prove that the recovery did not rewrite the frozen implementation."""

    registry = study._load_registry(root)
    if root.name not in FORMAL_CONFIGS:
        raise ValueError(f"Recovery refused a non-formal root: {root}")
    study._verify_prepare_lineage(root, registry)
    bindings = {
        str(Path(str(row["path"])).resolve()): dict(row)
        for row in registry.get("source_code_bindings", [])
    }
    rows: list[dict[str, Any]] = []
    for relative, expected in FROZEN_SOURCE_BINDINGS.items():
        path = (study.REPO_ROOT / relative).resolve()
        row = bindings.get(str(path))
        if row is None:
            raise ValueError(f"Frozen source binding is missing: {relative}")
        observed = {
            "size_bytes": path.stat().st_size,
            "sha256": study._sha256_file(path),
        }
        frozen = {
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        if observed != expected or frozen != expected:
            raise ValueError(
                f"Frozen source drift; recovery is not permitted: {relative}"
            )
        rows.append({"path": str(path), **expected})

    recovery_source = Path(__file__).resolve()
    if str(recovery_source) in bindings:
        raise ValueError("Recovery source unexpectedly belongs to frozen bindings")
    return {
        "formal_root": str(root),
        "study_kind": str(registry["study_kind"]),
        "source_config_sha256": str(registry["source_config_sha256"]),
        "frozen_sources": rows,
        "recovery_source_path": str(recovery_source),
        "recovery_source_size_bytes": recovery_source.stat().st_size,
        "recovery_source_sha256": study._sha256_file(recovery_source),
        "frozen_sources_modified": False,
    }


def _write_recovery_manifest(
    root: Path, *, status: str, details: Mapping[str, Any] | None = None
) -> Path:
    path = root / RECOVERY_PATH
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = study._read_signed(path, kind=RECOVERY_KIND)
    attestation = _frozen_source_attestation(root)
    if previous and previous.get("source_attestation") != attestation:
        raise ValueError("Evaluation recovery source attestation drift")
    history = [dict(row) for row in previous.get("history", [])]
    history.append(
        {
            "sequence": len(history),
            "status": str(status),
            "pid": os.getpid(),
            "updated_at_utc": study.utc_now(),
            "details": dict(details or {}),
        }
    )
    initial_registry_sha = previous.get("pre_recovery_registry_sha256")
    if initial_registry_sha is None:
        initial_registry_sha = study._sha256_file(study._registry_path(root))
        expected_registry_sha = PRE_RECOVERY_REGISTRY_SHA256[root.name]
        if initial_registry_sha != expected_registry_sha:
            raise ValueError(
                "Pre-recovery task registry drift; explicit re-audit is required"
            )
    return study._write_signed(
        path,
        {
            "schema_version": 1,
            "kind": RECOVERY_KIND,
            "reason": (
                "the frozen blanket finite check rejected intentionally undefined "
                "calendar/butterfly rates for zero supported constraints"
            ),
            "repair_scope": "post_training_prediction_metric_validation_only",
            "missingness_contract": MISSINGNESS_CONTRACT,
            "scientific_metric_definition_changed": False,
            "training_jobs_launched": False,
            "training_or_checkpoint_artifacts_rewritten": False,
            "pair_metric_schema_additions": sorted(CONSTRAINT_COUNT_COLUMNS),
            "pre_recovery_registry_sha256": initial_registry_sha,
            "source_attestation": attestation,
            "history": history,
            "status": str(status),
        },
    )


def _evaluate_prediction_cell(root: Path, cell: Mapping[str, Any]) -> pd.DataFrame:
    """Frozen evaluator with denominator-aware evidence validation."""

    from scripts.rq123.news_first_vol_film_nolp_10seed import (
        _configure_prediction_determinism,
    )
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        _enforce_formal_run_coverage,
        _prediction_export_frame,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    _configure_prediction_determinism(int(cell["seed"]))
    panel = study._panel_with_text(
        root,
        tolerance=int(cell["panel_tolerance_minutes"]),
        fold=str(cell["fold"]),
        condition=str(cell["text_condition"]),
    )
    run = RunSpec(
        run_id=str(cell["evaluation_id"]),
        run_dir=Path(str(cell["run_dir"])),
        model="wgan",
        tolerance_minutes=int(cell["train_tolerance_minutes"]),
        seed=int(cell["seed"]),
        checkpoint_path=Path(str(cell["checkpoint_path"])),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={"fold": cell["fold"], "model_id": cell["model_id"]},
    )
    panel_name = (
        f"{cell['panel_role']}__{int(cell['panel_tolerance_minutes']):02d}m__"
        f"{cell['fold']}"
    )
    evaluator = TrainedRunEvaluator(
        mc_samples=64,
        sample_batch_size=32,
        draw_batch_size=64,
        device=f"cuda:{int(cell['gpu_id'])}",
    )
    predictions = evaluator(run, panel_name, panel)
    samples, exclusions, metric_exclusions = compute_sample_metrics(
        run, panel_name, panel, predictions, evaluate_embedded_atm_skew=False
    )
    export = _prediction_export_frame(run, panel_name, panel, predictions, samples)
    _enforce_formal_run_coverage(run, panel_name, panel, samples, exclusions, export)
    prediction_path = root / "predictions/cells" / f"{cell['evaluation_id']}.csv.gz"
    study._write_frame(prediction_path, export, gzip=True)
    overall = aggregate_pair_metrics(samples)
    overall = overall.loc[
        overall["stratum_type"].eq("overall") & overall["stratum_value"].eq("all")
    ].copy()
    required = {
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
        "skill",
        "predicted_calendar_constraint_count",
        "target_calendar_constraint_count",
        "predicted_calendar_violation_rate",
        "target_calendar_violation_rate",
        "calendar_violation_rate_gap",
        "predicted_butterfly_constraint_count",
        "target_butterfly_constraint_count",
        "predicted_butterfly_violation_rate",
        "target_butterfly_violation_rate",
        "butterfly_violation_rate_gap",
    }
    missing = sorted(required - set(overall.columns))
    if missing:
        raise ValueError(f"Prediction diagnostics missing columns: {missing}")
    noise_sha = study._prediction_noise_sha(
        cell, panel["sample_id"].astype(str).tolist()
    )
    relation_by_pair = panel.assign(pair_id=panel["pair_id"].astype(str)).set_index(
        "pair_id"
    )[["window_relation", "post_news_current_overlap_minutes"]]
    metric_pair_ids = overall["pair_id"].astype(str)
    if set(metric_pair_ids) != set(relation_by_pair.index):
        raise ValueError("Prediction metrics lost window-relation lineage")
    evidence = pd.DataFrame(
        {
            "evaluation_id": str(cell["evaluation_id"]),
            "source_cell_id": str(cell["source_cell_id"]),
            "model_id": str(cell["model_id"]),
            "generator_mode": str(cell["generator_mode"]),
            "train_tolerance_minutes": int(cell["train_tolerance_minutes"]),
            "panel_tolerance_minutes": int(cell["panel_tolerance_minutes"]),
            "tolerance_minutes": int(cell["panel_tolerance_minutes"]),
            "panel_role": str(cell["panel_role"]),
            "text_condition": str(cell["text_condition"]),
            "fold": str(cell["fold"]),
            "seed": int(cell["seed"]),
            "pair_id": overall["pair_id"].astype(str),
            "session_id": overall["session_id"].astype(str),
            "window_relation": metric_pair_ids.map(
                relation_by_pair["window_relation"]
            ).astype(str),
            "post_news_current_overlap_minutes": pd.to_numeric(
                metric_pair_ids.map(
                    relation_by_pair["post_news_current_overlap_minutes"]
                ),
                errors="raise",
            ),
            "target_mae": pd.to_numeric(overall["model_mae"], errors="raise"),
            "persistence_mae": pd.to_numeric(
                overall["persistence_mae"], errors="raise"
            ),
            "persistence_skill": pd.to_numeric(overall["skill"], errors="raise"),
            "predicted_calendar_constraint_count": pd.to_numeric(
                overall["predicted_calendar_constraint_count"], errors="raise"
            ),
            "target_calendar_constraint_count": pd.to_numeric(
                overall["target_calendar_constraint_count"], errors="raise"
            ),
            "predicted_calendar_violation_rate": pd.to_numeric(
                overall["predicted_calendar_violation_rate"], errors="raise"
            ),
            "target_calendar_violation_rate": pd.to_numeric(
                overall["target_calendar_violation_rate"], errors="raise"
            ),
            "calendar_violation_rate_gap": pd.to_numeric(
                overall["calendar_violation_rate_gap"], errors="raise"
            ),
            "predicted_butterfly_constraint_count": pd.to_numeric(
                overall["predicted_butterfly_constraint_count"], errors="raise"
            ),
            "target_butterfly_constraint_count": pd.to_numeric(
                overall["target_butterfly_constraint_count"], errors="raise"
            ),
            "predicted_butterfly_violation_rate": pd.to_numeric(
                overall["predicted_butterfly_violation_rate"], errors="raise"
            ),
            "target_butterfly_violation_rate": pd.to_numeric(
                overall["target_butterfly_violation_rate"], errors="raise"
            ),
            "butterfly_violation_rate_gap": pd.to_numeric(
                overall["butterfly_violation_rate_gap"], errors="raise"
            ),
            "checkpoint_sha256": str(cell["checkpoint_sha256"]),
            "prediction_sha256": study._sha256_file(prediction_path),
            "noise_bank_profile_sha256": noise_sha,
        }
    )
    if len(evidence) != len(panel):
        raise ValueError(f"Incomplete prediction evidence: {cell['evaluation_id']}")
    validate_pair_metric_contract(evidence)
    evidence_path = (
        root / "predictions/cells" / f"{cell['evaluation_id']}.pair_metrics.csv"
    )
    study._write_frame(evidence_path, evidence)
    manifest_path = (
        root / "predictions/cells" / f"{cell['evaluation_id']}.manifest.json"
    )
    study._write_signed(
        manifest_path,
        {
            "schema_version": 2,
            "kind": "architecture_window_prediction_cell_v1",
            "evaluation_spec_sha256": cell["evaluation_spec_sha256"],
            "checkpoint_path": str(Path(str(cell["checkpoint_path"])).resolve()),
            "checkpoint_sha256": str(cell["checkpoint_sha256"]),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": study._sha256_file(prediction_path),
            "pair_metrics_path": str(evidence_path.resolve()),
            "pair_metrics_sha256": study._sha256_file(evidence_path),
            "row_count": len(evidence),
            "noise_bank_profile_sha256": noise_sha,
            "optional_metric_exclusion_count": len(metric_exclusions),
            "evaluation_recovery_kind": RECOVERY_KIND,
            "missingness_contract": MISSINGNESS_CONTRACT,
        },
    )
    return evidence


def _analysis_metrics(root: Path) -> pd.DataFrame:
    frame = _ORIGINAL_ANALYSIS_METRICS(root)
    validate_pair_metric_contract(frame)
    return frame


_ORIGINAL_ANALYSIS_METRICS = analysis._metrics


@contextmanager
def corrected_evaluation_contract(root: Path) -> Iterator[None]:
    """Install and reliably restore the two recovery hooks."""

    _frozen_source_attestation(root)
    original_evaluator = study._evaluate_prediction_cell
    original_metrics = analysis._metrics
    original_required = analysis.PAIR_METRIC_COLUMNS
    original_finite = analysis.FINITE_COLUMNS
    analysis.PAIR_METRIC_COLUMNS = set(original_required) | CONSTRAINT_COUNT_COLUMNS
    analysis.FINITE_COLUMNS = set(CORE_FINITE_COLUMNS) | CONSTRAINT_COUNT_COLUMNS
    study._evaluate_prediction_cell = _evaluate_prediction_cell
    analysis._metrics = _analysis_metrics
    try:
        yield
    finally:
        analysis._metrics = original_metrics
        study._evaluate_prediction_cell = original_evaluator
        analysis.PAIR_METRIC_COLUMNS = original_required
        analysis.FINITE_COLUMNS = original_finite


def _stage(root: Path, name: str, operation: Any) -> Any:
    control = root.with_name(root.name + "_control")
    study.append_stage_journal(
        control,
        experiment_kind=str(study._load_registry(root)["study_kind"]),
        stage=name,
        status="running",
        details={"evaluation_recovery": True},
    )
    result = operation()
    study.append_stage_journal(
        control,
        experiment_kind=str(study._load_registry(root)["study_kind"]),
        stage=name,
        status="completed",
        details={"result": str(result), "evaluation_recovery": True},
    )
    return result


def _verify_terminal_bundles(root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a terminal commit without relying on its final SHA snapshot."""

    study._verify_prepare_lineage(root, registry)
    if not study._all_complete(root, registry["jobs"]):
        raise ValueError("Terminal recovery found incomplete training jobs")
    study._verify_prediction_bundle(root, registry)
    study._verify_analysis_bundle(root, registry)
    study._verify_bootstrap_bundle(root, registry)
    study._verify_report_bundle(root, registry)
    qa_path = Path(str(registry["terminal_qa_path"])).resolve()
    study._verify_file(qa_path, str(registry["terminal_qa_sha256"]), "terminal QA")
    qa_payload = study._read_signed(qa_path, kind="architecture_window_terminal_qa_v1")
    if qa_payload.get("status") != "passed":
        raise ValueError("Terminal QA status is not passed")
    recovery_manifest = study._read_signed(root / RECOVERY_PATH, kind=RECOVERY_KIND)
    if recovery_manifest.get("status") != "completed" or recovery_manifest.get(
        "source_attestation"
    ) != _frozen_source_attestation(root):
        raise ValueError("Terminal root has incomplete recovery lineage")
    return recovery_manifest


def run_evaluation_pipeline(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    _finalize_control_manifest: bool = True,
) -> Path:
    """Resume prediction through terminal QA without touching training."""

    root = _root(output_dir)
    selected_config = str(config_path or FORMAL_CONFIGS.get(root.name, ""))
    if not selected_config:
        raise ValueError(f"No recovery config is registered for {root}")
    config = study.load_config(selected_config)
    control = root.with_name(root.name + "_control")
    with study.SupervisorLock(control, name="pipeline"):
        # Re-read every mutable gate under the same lock used by the frozen
        # pipeline. This prevents a stale pre-lock view from modifying a root
        # that another process has just made terminal.
        registry = study._load_registry(root)
        if registry.get("source_config_sha256") != config["source_config_sha256"]:
            raise ValueError("Recovery source config drift")
        if registry.get("terminal_complete"):
            _verify_terminal_bundles(root, registry)
            if study._output_manifest_path(root).is_file():
                study._verify_output_sha_manifest(root)
            else:
                # qa() saves terminal_complete immediately before creating the
                # root SHA snapshot. Complete only that narrowly defined
                # interrupted terminal commit after every bundle re-verifies.
                study._write_output_sha_manifest(root)
                study._verify_output_sha_manifest(root)
            # A crash after QA can leave only the external control snapshot
            # stale. Re-freezing that append-only external universe does not
            # mutate the terminal formal root.
            if _finalize_control_manifest:
                study._write_control_output_sha_manifest(control)
                study._verify_control_output_sha_manifest(control)
            return root
        if not registry.get("evaluation_frozen"):
            raise RuntimeError("Recovery requires a frozen evaluation allowlist")
        if not study._all_complete(root, registry["jobs"]):
            raise RuntimeError("Evaluation recovery refuses incomplete training jobs")
        study._verify_evaluation_bundle(root, registry)
        _write_recovery_manifest(root, status="running", details={"phase": "start"})
        study._append_resource_snapshot(control, phase="evaluation_recovery_start")
        study.append_stage_journal(
            control,
            experiment_kind=str(registry["study_kind"]),
            stage="evaluation_recovery",
            status="running",
            details={"output_root": str(root), "resume": True},
        )
        try:
            with corrected_evaluation_contract(root):
                _stage(
                    root,
                    "prediction_recovered",
                    lambda: study.predict(selected_config, root, resume=True),
                )
                _stage(root, "analysis_recovered", lambda: study.analyze(root))
                _stage(root, "bootstrap", lambda: study.bootstrap(root))
                _stage(root, "report", lambda: study.report(root))
                _write_recovery_manifest(
                    root,
                    status="completed",
                    details={"phase": "terminal_qa_next"},
                )
                _stage(root, "qa", lambda: study.qa(root))
            study.append_stage_journal(
                control,
                experiment_kind=str(registry["study_kind"]),
                stage="evaluation_recovery",
                status="completed",
                details={"output_root": str(root)},
            )
            study._append_resource_snapshot(
                control, phase="evaluation_recovery_complete"
            )
            if _finalize_control_manifest:
                study._write_control_output_sha_manifest(control)
            return root
        except BaseException as exc:
            current = study._load_registry(root)
            if not current.get("terminal_complete"):
                _write_recovery_manifest(
                    root,
                    status="failed",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            study._append_resource_snapshot(control, phase="evaluation_recovery_failed")
            study.append_stage_journal(
                control,
                experiment_kind=str(registry["study_kind"]),
                stage="evaluation_recovery",
                status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise


def status(output_dir: str | Path) -> dict[str, Any]:
    root = _root(output_dir)
    payload = study.status(FORMAL_CONFIGS.get(root.name, ""), root)
    path = root / RECOVERY_PATH
    if path.is_file():
        payload["evaluation_recovery"] = study._read_signed(path, kind=RECOVERY_KIND)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        result: Any = run_evaluation_pipeline(
            args.output_dir,
            config_path=args.config,
            _finalize_control_manifest=False,
        )
    else:
        result = status(args.output_dir)
    if isinstance(result, Mapping):
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    else:
        print(result, flush=True)
    if args.action == "run":
        root = _root(args.output_dir)
        if study._load_registry(root).get("terminal_complete"):
            # The final stdout line may itself be redirected into the external
            # control tree, so freeze that tree only after the flush above.
            study._write_control_output_sha_manifest(
                root.with_name(root.name + "_control")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONSTRAINT_COUNT_COLUMNS",
    "MISSINGNESS_CONTRACT",
    "corrected_evaluation_contract",
    "run_evaluation_pipeline",
    "status",
    "validate_pair_metric_contract",
]
