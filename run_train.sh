#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"

# Default subcommand: vol-xlsx
SUBCMD="${1:-vol-xlsx}"
shift 2>/dev/null || true

LOG_FILE="${LOG_DIR}/train_${SUBCMD}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${SUBCMD}.pid"

nohup python scripts/train/main.py \
  "${SUBCMD}" \
  "$@" >"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "=========================================="
echo "  Train job launched"
echo "=========================================="
echo "  Subcommand : ${SUBCMD}"
echo "  PID        : ${PID}"
echo "  PID file   : ${PID_FILE}"
echo "  Log file   : ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Follow logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check status:"
echo "  kill -0 ${PID} 2>/dev/null && echo running || echo stopped"
echo ""
echo "Stop training:"
echo "  kill ${PID}"
