#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
EXPERIMENT_ROOT="${1:-${EXPERIMENT_ROOT:-}}"
ARGS=()
[[ -n "${EXPERIMENT_ROOT}" ]] && ARGS=(--experiment-root "${EXPERIMENT_ROOT}")
conda run -n "${ENV_NAME:-py312}" python scripts/rq1/rq1_experiment.py package "${ARGS[@]}"
