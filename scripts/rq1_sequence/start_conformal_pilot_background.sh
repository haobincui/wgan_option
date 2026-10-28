#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "EXPERIMENT_ROOT must point to the completed RQ1 sequence experiment." >&2
  exit 1
fi

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LOG_PATH="${LOG_DIR}/conformal_pilot_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/conformal_pilot_${RUN_TS}.pid"
mkdir -p "${LOG_DIR}"

nohup setsid env \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  GPU_ID="${GPU_ID:-0}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-3}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  VARIANT="${VARIANT:-history_matched_lp}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq1_sequence/run_conformal_pilot.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_PATH}"

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
