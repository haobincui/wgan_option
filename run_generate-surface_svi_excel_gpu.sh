#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/generate_surface_excel_gpu_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/generate_surface_excel_gpu.pid"

nohup python scripts/generate_surface/main.py \
  generate_surface \
  --device gpu \
  --model svi \
  --data_range excel \
  --config configs/surface_builder/svi/generate_surface-svi-excel.yaml \
  "$@" >"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started generate_surface excel job in background."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
