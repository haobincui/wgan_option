"""Optional Vega contracts shared by the existing direct experiment wrappers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

VEGA_MODE = "black76_current_vega_v1"
PARALLEL_MODE = "rq12_vega_shared_slots_v1"
OVERRIDE_KIND = "rq12_direct_vega_user_concurrency_override_v1"
WEIGHT_FIELDS = (
    "cell_weight_mode",
    "cell_weight_manifest_path",
    "cell_weight_manifest_sha256",
)


def cell_weight_fields(config: Mapping[str, Any]) -> dict[str, str]:
    """Return no extra fields for historical, unweighted experiment payloads."""
    data = config.get("data", {})
    mode = str(data.get("cell_weight_mode", "none") or "none").strip().lower()
    if mode == "none":
        return {}
    if mode != VEGA_MODE:
        raise ValueError(f"Unsupported cell weighting: {mode}")
    path = Path(str(data.get("cell_weight_manifest_path", ""))).expanduser().resolve()
    expected = str(data.get("cell_weight_manifest_sha256", "")).lower()
    if not path.is_file() or not expected:
        raise ValueError("Vega experiment requires a hash-bound weight manifest")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Vega weight manifest drift")
    payload = json.loads(path.read_text())
    if payload.get("mode", payload.get("cell_weight_mode")) != VEGA_MODE:
        raise ValueError("Vega weight manifest mode drift")
    return {
        "cell_weight_mode": VEGA_MODE,
        "cell_weight_manifest_path": str(path),
        "cell_weight_manifest_sha256": actual,
    }


def validate_weighted_experiment(config: Mapping[str, Any]) -> None:
    fields = cell_weight_fields(config)
    training = config.get("training", {})
    scope = str(training.get("vega_weight_scope", "") or "").strip().lower()
    if scope not in {"", "mae_only"}:
        raise ValueError(f"Unsupported Vega weight scope: {scope}")
    if scope == "mae_only" and not fields:
        raise ValueError("mae_only scope requires Vega cell weighting")
    if fields:
        if config["data"].get("support_mask_mode") != "raw_joint":
            raise ValueError("Vega experiment requires raw joint support")
        monitor = training.get("best_checkpoint_metric")
        if monitor not in {"val_vega_recon", "val_vega_hybrid_score"}:
            raise ValueError("Vega experiment must select Vega validation MAE or its legacy hybrid score")
        if scope == "mae_only":
            if monitor != "val_vega_recon":
                raise ValueError("mae_only scope must select val_vega_recon")
            try:
                baseline_penalty = float(training.get("baseline_penalty_weight"))
            except (TypeError, ValueError) as exc:
                raise ValueError("mae_only scope requires baseline_penalty_weight=0") from exc
            if not math.isfinite(baseline_penalty) or baseline_penalty != 0.0:
                raise ValueError("mae_only scope requires baseline_penalty_weight=0")


def weight_source_paths(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    fields = cell_weight_fields(config)
    if not fields:
        return []
    path = Path(fields["cell_weight_manifest_path"])
    payload = json.loads(path.read_text())
    cache = Path(str(payload["weights_path"]))
    if not cache.is_absolute():
        cache = path.parent / cache
    if not cache.is_file() or hashlib.sha256(cache.read_bytes()).hexdigest() != payload["weights_sha256"]:
        raise ValueError("Vega weight cache drift")
    return [("cell_weight_manifest", path), ("cell_weight_cache", cache.resolve())]


def validate_parallel_runtime(runtime: Mapping[str, Any]) -> None:
    primary = int(runtime.get("benchmark_workers_per_gpu", -1))
    fallback = int(runtime.get("fallback_workers_per_gpu", -1))
    if primary < 1 or fallback < 1 or fallback > primary:
        raise ValueError("Invalid benchmark/fallback concurrency")
    formal = runtime.get("formal_workers_per_gpu")
    if runtime.get("parallel_scheduler_mode") == PARALLEL_MODE:
        candidates = runtime.get("parallel_benchmark_candidates_per_gpu", ())
        fallbacks = runtime.get("parallel_fallback_candidates_per_gpu", ())
        if not candidates or not fallbacks or any(int(n) < 1 for n in (*candidates, *fallbacks)):
            raise ValueError("Parallel runtime requires positive benchmark candidates")
        if formal is not None and int(formal) < 1:
            raise ValueError("Formal parallel concurrency must be positive")
    elif formal is not None and int(formal) not in (primary, fallback):
        raise ValueError("Frozen formal concurrency was not benchmarked")


def validate_global_evidence(config: Mapping[str, Any], evidence: Mapping[str, Any], workers: int) -> None:
    if config["runtime"].get("parallel_scheduler_mode") != PARALLEL_MODE:
        return
    benchmark_passed = evidence.get("kind") == "rq12_direct_vega_global_benchmark_v1" and evidence.get("status") == "passed"
    user_override = evidence.get("kind") == OVERRIDE_KIND and evidence.get("status") == "authorized_by_user"
    if user_override and (evidence.get("authorization") != "explicit_user_instruction" or not evidence.get("user_instruction")):
        raise ValueError("Concurrency override is missing the explicit user instruction")
    if not benchmark_passed and not user_override:
        raise ValueError("Parallel formal prepare requires measured evidence or an explicit user concurrency override")
    if int(evidence.get("selected_workers_per_gpu", -1)) != workers:
        raise ValueError("Formal concurrency differs from the signed global benchmark")
    unsigned = {k: v for k, v in evidence.items() if k != "payload_sha256"}
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if actual != evidence.get("payload_sha256"):
        raise ValueError("Global benchmark evidence self-hash drift")
    sources = evidence.get("source_configs", {})
    expected_source = str(config.get("source_config_path", ""))
    expected_hash = str(config.get("source_config_sha256", ""))
    if not any(str(row.get("path")) == expected_source and row.get("sha256") == expected_hash
               for row in sources.values()):
        raise ValueError("Global benchmark does not bind this source configuration")
    scheduler = Path(__file__).with_name("rq12_direct_vega_parallel.py")
    if evidence.get("supervisor_code_sha256") != hashlib.sha256(scheduler.read_bytes()).hexdigest():
        raise ValueError("Global benchmark supervisor code drift")
