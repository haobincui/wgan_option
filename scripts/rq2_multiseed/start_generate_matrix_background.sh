#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/rq2_multiseed_textbase_20260627}"
DRIVER_DIR="${EXP_ROOT}/logs/driver"
DRIVER_LOG="${DRIVER_LOG:-${DRIVER_DIR}/generate_matrix_driver.log}"
DRIVER_PID_FILE="${DRIVER_PID_FILE:-${DRIVER_DIR}/generate_matrix_driver.pid}"

mkdir -p "${DRIVER_DIR}"

if [[ -f "${DRIVER_PID_FILE}" ]]; then
  old_pid="$(cat "${DRIVER_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Generate matrix driver is already running: pid=${old_pid}"
    echo "Log: ${DRIVER_LOG}"
    exit 0
  fi
fi

{
  echo "============================================================"
  echo "Starting RQ2 multi-seed generate-result matrix driver"
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EXP_ROOT=${EXP_ROOT}"
  echo "SEEDS_CSV=${SEEDS_CSV:-42,101,202,303,404}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}"
  echo "FORCE_GENERATE=${FORCE_GENERATE:-0}"
  echo "============================================================"
} >>"${DRIVER_LOG}"

EXP_ROOT="${EXP_ROOT}" \
SEEDS_CSV="${SEEDS_CSV:-42,101,202,303,404}" \
ENV_NAME="${ENV_NAME:-py312}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
SKIP_COLLECT="${SKIP_COLLECT:-0}" \
FORCE_GENERATE="${FORCE_GENERATE:-0}" \
nohup bash scripts/rq2_multiseed/run_generate_matrix.sh >>"${DRIVER_LOG}" 2>&1 &
driver_pid=$!
echo "${driver_pid}" >"${DRIVER_PID_FILE}"
disown "${driver_pid}" 2>/dev/null || true

echo "Generate matrix driver started in background."
echo "PID file: ${DRIVER_PID_FILE}"
echo "Log file: ${DRIVER_LOG}"
echo "Monitor:  bash scripts/rq2_multiseed/monitor_training.sh"
