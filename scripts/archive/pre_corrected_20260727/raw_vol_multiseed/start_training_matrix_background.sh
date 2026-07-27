#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
DRIVER_DIR="${EXP_ROOT}/logs/driver"
DRIVER_LOG="${DRIVER_LOG:-${DRIVER_DIR}/training_matrix_driver.log}"
DRIVER_PID_FILE="${DRIVER_PID_FILE:-${DRIVER_DIR}/training_matrix_driver.pid}"
mkdir -p "${DRIVER_DIR}"

if [[ -f "${DRIVER_PID_FILE}" ]]; then
  old_pid="$(cat "${DRIVER_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Raw-vol training matrix driver already running: pid=${old_pid}"
    echo "Log: ${DRIVER_LOG}"
    exit 0
  fi
fi

EXP_ROOT="${EXP_ROOT}" \
CONFIG_PATH="${CONFIG_PATH:-configs/film_wgan/train_raw_vol_textbase.yaml}" \
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}" \
SEEDS="${SEEDS:-42 101 202 303 404}" \
MODELS_TO_RUN="${MODELS_TO_RUN:-text no_text bow llm_sentiment}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
ENV_NAME="${ENV_NAME:-py312}" \
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}" \
POLL_SECONDS="${POLL_SECONDS:-60}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
SKIP_PREPARE="${SKIP_PREPARE:-0}" \
nohup bash scripts/raw_vol_multiseed/run_training_matrix.sh >>"${DRIVER_LOG}" 2>&1 &
driver_pid=$!
echo "${driver_pid}" >"${DRIVER_PID_FILE}"
disown "${driver_pid}" 2>/dev/null || true

echo "Raw-vol training matrix driver started."
echo "PID file: ${DRIVER_PID_FILE}"
echo "Log file: ${DRIVER_LOG}"
