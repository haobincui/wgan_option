#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SERIES_TS="${SERIES_TS:-$(date -u +%Y%m%d-%H%M%S)}"
SERIES_ROOT="${SERIES_ROOT:-outputs/experiments/rq123_availability_lag_${SERIES_TS}}"
LOG_PATH="${SERIES_ROOT}/logs/availability_lag_driver.log"
PID_PATH="${SERIES_ROOT}/logs/availability_lag_driver.pid"
mkdir -p "${SERIES_ROOT}/logs"

nohup setsid env \
  SERIES_TS="${SERIES_TS}" \
  SERIES_ROOT="${SERIES_ROOT}" \
  LAGS="${LAGS:-1 2 5}" \
  GPU_IDS="${GPU_IDS:-0 1}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-2}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq123/run_availability_lag_sensitivity.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_PATH}"

echo "availability-lag sensitivity started"
echo "series_root=${SERIES_ROOT}"
echo "pid=${pid}"
echo "log=${LOG_PATH}"
