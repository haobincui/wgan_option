#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
SUBCMD="${1:-both}"
shift 1 2>/dev/null || true

launch_job() {
  local mode="$1"
  shift

  local log_file="${LOG_DIR}/analyze_error_${mode}_${TIMESTAMP}.log"
  local pid_file="${LOG_DIR}/analyze_error_${mode}.pid"

  nohup python scripts/analyze_error/main.py \
    "${mode}" \
    "$@" >"${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${pid_file}"

  echo "=========================================="
  echo "  Analyze-error job launched"
  echo "=========================================="
  echo "  Mode      : ${mode}"
  echo "  PID       : ${pid}"
  echo "  PID file  : ${pid_file}"
  echo "  Log file  : ${log_file}"
  echo "=========================================="
  echo ""
  echo "Follow logs:"
  echo "  tail -f ${log_file}"
  echo ""
  echo "Check status:"
  echo "  kill -0 ${pid} 2>/dev/null && echo running || echo stopped"
  echo ""
  echo "Stop job:"
  echo "  kill ${pid}"
  echo ""
}

case "${SUBCMD}" in
  vol|svi)
    launch_job "${SUBCMD}" "$@"
    ;;
  both)
    launch_job "vol" "$@"
    launch_job "svi" "$@"
    ;;
  *)
    echo "Usage: bash run_analyze_error.sh [vol|svi|both] [extra args...]"
    echo ""
    echo "Examples:"
    echo "  bash run_analyze_error.sh"
    echo "  bash run_analyze_error.sh vol"
    echo "  bash run_analyze_error.sh svi --config configs/analyze_error/svi.yaml"
    echo "  bash run_analyze_error.sh vol --set bootstrap_samples=2000"
    exit 1
    ;;
esac
