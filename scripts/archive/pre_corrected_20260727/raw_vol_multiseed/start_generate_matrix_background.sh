#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
DRIVER_DIR="${EXP_ROOT}/logs/driver"
DRIVER_LOG="${DRIVER_LOG:-${DRIVER_DIR}/generate_matrix_driver.log}"
DRIVER_PID_FILE="${DRIVER_PID_FILE:-${DRIVER_DIR}/generate_matrix_driver.pid}"
mkdir -p "${DRIVER_DIR}"

EXP_ROOT="${EXP_ROOT}" \
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}" \
SEEDS_CSV="${SEEDS_CSV:-42,101,202,303,404}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
ENV_NAME="${ENV_NAME:-py312}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
SKIP_COLLECT="${SKIP_COLLECT:-0}" \
FORCE_GENERATE="${FORCE_GENERATE:-0}" \
nohup bash scripts/raw_vol_multiseed/run_generate_matrix.sh >>"${DRIVER_LOG}" 2>&1 &
driver_pid=$!
echo "${driver_pid}" >"${DRIVER_PID_FILE}"
disown "${driver_pid}" 2>/dev/null || true

echo "Raw-vol generate matrix driver started."
echo "PID file: ${DRIVER_PID_FILE}"
echo "Log file: ${DRIVER_LOG}"
