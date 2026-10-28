#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" && -f outputs/experiments/rq1_intraday_sequence_latest.txt ]]; then
  EXPERIMENT_ROOT="$(<outputs/experiments/rq1_intraday_sequence_latest.txt)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ1 sequence experiment found." >&2
  exit 1
fi

conda run --no-capture-output -n "${ENV_NAME:-py312}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py monitor \
  --experiment-root "${EXPERIMENT_ROOT}"

LATEST_LOG="$(find "${EXPERIMENT_ROOT}/logs/background" -type f -name 'training_matrix_*.log' -printf '%p\n' 2>/dev/null | sort | tail -n 1)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo "latest_background_log=${LATEST_LOG}"
  tail -n 30 "${LATEST_LOG}"
fi

