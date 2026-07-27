#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(find outputs/experiments -maxdepth 1 -type d -name 'rq2_pair_representation_raw_vol_continuation_*' -printf '%p\n' | sort | tail -n 1)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ2 continuation experiment found. Run prepare_experiment.sh first." >&2
  exit 1
fi

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LOG_PATH="${LOG_DIR}/training_matrix_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/training_matrix_${RUN_TS}.pid"
mkdir -p "${LOG_DIR}"

nohup setsid bash scripts/rq2_pair/resume_training_matrix.sh \
  --experiment-root "${EXPERIMENT_ROOT}" \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_PATH}"

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
