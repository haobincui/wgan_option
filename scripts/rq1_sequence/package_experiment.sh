#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" && -f outputs/experiments/rq1_intraday_sequence_latest.txt ]]; then
  EXPERIMENT_ROOT="$(<outputs/experiments/rq1_intraday_sequence_latest.txt)"
fi
PACKAGE_ARGS=(
  --experiment-root "${EXPERIMENT_ROOT}"
)
if [[ -n "${OUTPUT_ZIP:-}" ]]; then
  PACKAGE_ARGS+=(--output "${OUTPUT_ZIP}")
fi
conda run --no-capture-output -n "${ENV_NAME:-py312}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py package \
  "${PACKAGE_ARGS[@]}"
