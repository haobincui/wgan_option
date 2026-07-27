#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/logs/rq123_london}"
LOG_PATH="${LOG_DIR}/rq123_london_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/rq123_london_${RUN_TS}.pid"
LATEST_PATH="${LOG_DIR}/latest.env"
CONTROL_ROOT="${CONTROL_ROOT:-outputs/experiments/rq123_london_recalculation_${RUN_TS}}"
mkdir -p "${LOG_DIR}"

if [[ -f "${LATEST_PATH}" ]]; then
  old_pid_path="$(
    awk -F= '$1 == "PID_PATH" {print substr($0, index($0, "=") + 1)}' "${LATEST_PATH}"
  )"
  if [[ -n "${old_pid_path}" && -s "${old_pid_path}" ]]; then
    existing_pid="$(cat "${old_pid_path}")"
    if kill -0 "${existing_pid}" 2>/dev/null; then
      echo "RQ1-RQ3 London pipeline is already running: pid=${existing_pid}" >&2
      exit 1
    fi
  fi
fi

LOG_PATH="${LOG_DIR}/rq123_london_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/rq123_london_${RUN_TS}.pid"
LATEST_PATH="${LOG_DIR}/latest.env"

nohup setsid env \
  RUN_TS="${RUN_TS}" \
  CONTROL_ROOT="${CONTROL_ROOT}" \
  SOURCE_TIMEZONE="Europe/London" \
  GPU_IDS="${GPU_IDS:-0 1}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-2}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq123_london/run_full_recalculation.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_PATH}"

{
  printf 'RUN_TS=%q\n' "${RUN_TS}"
  printf 'PID_PATH=%q\n' "${PID_PATH}"
  printf 'LOG_PATH=%q\n' "${LOG_PATH}"
  printf 'CONTROL_ROOT=%q\n' "${CONTROL_ROOT}"
} >"${LATEST_PATH}"

sleep 2
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Pipeline exited during startup." >&2
  tail -80 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "RQ1-RQ3 London-time recalculation started."
echo "pid=${pid}"
echo "log=${LOG_PATH}"
echo "control_root=${CONTROL_ROOT}"
echo "monitor=bash scripts/rq123_london/monitor_full_recalculation.sh"
