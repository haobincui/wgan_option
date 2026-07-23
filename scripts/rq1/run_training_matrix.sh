#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
ENV_NAME="${ENV_NAME:-py312}"
EXPERIMENT_ROOT="${1:-${EXPERIMENT_ROOT:-}}"
if [[ -n "${EXPERIMENT_ROOT}" ]]; then
  shift || true
  ROOT_ARGS=(--experiment-root "${EXPERIMENT_ROOT}")
else
  ROOT_ARGS=()
fi

conda run -n "${ENV_NAME}" python scripts/rq1/rq1_experiment.py train-matrix \
  "${ROOT_ARGS[@]}" --env-name "${ENV_NAME}" "$@"
