#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
EXPERIMENT_ROOT="${1:-${EXPERIMENT_ROOT:-}}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(find outputs/experiments -maxdepth 1 -type d -name 'rq1_incremental_text_*' | sort | tail -1)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ1 experiment found." >&2
  exit 1
fi
shift || true

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_FILE="${EXPERIMENT_ROOT}/logs/training_matrix_${RUN_TS}.log"
PID_FILE="${EXPERIMENT_ROOT}/logs/training_matrix.pid"
mkdir -p "$(dirname "${LOG_FILE}")"

nohup bash scripts/rq1/run_training_matrix.sh "${EXPERIMENT_ROOT}" "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"
printf 'RQ1 training matrix started\nPID: %s\nLog: %s\nExperiment: %s\n' \
  "${PID}" "${LOG_FILE}" "${EXPERIMENT_ROOT}"
