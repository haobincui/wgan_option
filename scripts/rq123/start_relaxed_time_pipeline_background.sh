#!/usr/bin/env bash
# Start the relaxed-time RQ1-RQ3 pipeline in a detached process group.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/rq123}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/rq123_relaxed_time_${RUN_TS}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/rq123_relaxed_time_${RUN_TS}.pid}"
mkdir -p "${LOG_DIR}"

nohup setsid env \
  RUN_TS="${RUN_TS}" \
  INDEX_ID="${INDEX_ID:-raw_2022_2023_london_itm5_v1}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-32}" \
  GPU_IDS="${GPU_IDS:-0 1}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-2}" \
  RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}" \
  bash scripts/rq123/run_relaxed_time_pipeline.sh \
  >"${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"

echo "Started RQ1-RQ3 relaxed-time pipeline"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
