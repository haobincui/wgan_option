#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
PIPELINE_ROOT="${PIPELINE_ROOT:-outputs/experiments/rq123_corrected_raw_${RUN_TS}}"
LOG_DIR="${PIPELINE_ROOT}/logs/driver"
LOG_PATH="${LOG_DIR}/corrected_pipeline_${RUN_TS}.log"
PID_PATH="${PIPELINE_ROOT}/registry/pipeline.pid"
LATEST_PATH="outputs/experiments/rq123_corrected_latest.env"
mkdir -p "${LOG_DIR}" "$(dirname "${PID_PATH}")"

if [[ -s "${PID_PATH}" ]]; then
  existing_pid="$(tr -d '[:space:]' <"${PID_PATH}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Corrected RQ1-RQ3 pipeline is already running: pid=${existing_pid}" >&2
    echo "log=${LOG_PATH}" >&2
    exit 1
  fi
fi

nohup setsid env \
  RUN_TS="${RUN_TS}" \
  ENV_NAME="${ENV_NAME}" \
  PIPELINE_ROOT="${PIPELINE_ROOT}" \
  DATASET_DIR="${DATASET_DIR:-}" \
  DATASET_RUN_TS="${DATASET_RUN_TS:-}" \
  SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}" \
  PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}" \
  WINDOW_MINUTES="${WINDOW_MINUTES:-5}" \
  MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}" \
  OPTION_FILTER_MODE="${OPTION_FILTER_MODE:-otm_preferred_itm_fallback}" \
  MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE:-0.05}" \
  GPU_IDS="${GPU_IDS:-0 1}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-2}" \
  BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}" \
  BUILD_DATASET="${BUILD_DATASET:-1}" \
  REFRESH_MARKET_REFERENCES="${REFRESH_MARKET_REFERENCES:-0}" \
  PACKAGE_EXPERIMENT="${PACKAGE_EXPERIMENT:-0}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq123/run_corrected_pipeline.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_PATH}"
cat >"${LATEST_PATH}" <<EOF
RUN_TS=${RUN_TS}
PIPELINE_ROOT=${PIPELINE_ROOT}
PID=${pid}
PID_PATH=${PID_PATH}
LOG_PATH=${LOG_PATH}
EOF

sleep 2
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Corrected pipeline exited during startup." >&2
  tail -80 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "Corrected RQ1-RQ3 pipeline started."
echo "pipeline_root=${PIPELINE_ROOT}"
echo "pid=${pid}"
echo "log=${LOG_PATH}"
echo "status=${PIPELINE_ROOT}/registry/pipeline_status.json"
echo "monitor=PIPELINE_ROOT=${PIPELINE_ROOT} bash scripts/rq123/monitor_corrected_pipeline.sh"
