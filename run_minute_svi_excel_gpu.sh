#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/minute_svi_excel_gpu_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/minute_svi_excel_gpu.pid"

nohup python scripts/generate_surface/main.py \
  minute-svi-excel \
  --device gpu \
  --config configs/surface_builder/minute-svi-excel.yaml \
  "$@" >"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started minute-svi-excel in background."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
