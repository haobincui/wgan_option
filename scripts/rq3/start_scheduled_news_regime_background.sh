#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
CONFIG_PATH="${CONFIG_PATH:-configs/rq3/scheduled_news_regime_raw_vol.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/rq3_scheduled_news_regime_raw_vol_${RUN_TS}}"
LOG_DIR="${LOG_DIR:-outputs/rq3/logs}"
LOG_PATH="${LOG_DIR}/rq3_scheduled_news_regime_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/rq3_scheduled_news_regime_${RUN_TS}.pid"
LATEST_PATH="${LOG_DIR}/rq3_scheduled_news_regime_latest.env"

mkdir -p "${LOG_DIR}"

if [[ -f "${LATEST_PATH}" ]]; then
  OLD_PID_PATH="$(awk -F= '$1 == "PID_PATH" {print substr($0, index($0, "=") + 1)}' "${LATEST_PATH}")"
  OLD_LOG_PATH="$(awk -F= '$1 == "LOG_PATH" {print substr($0, index($0, "=") + 1)}' "${LATEST_PATH}")"
  if [[ -n "${OLD_PID_PATH}" && -s "${OLD_PID_PATH}" ]]; then
    EXISTING_PID="$(cat "${OLD_PID_PATH}")"
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "RQ3 scheduled-news analysis is already running: pid=${EXISTING_PID}" >&2
      echo "log=${OLD_LOG_PATH}" >&2
      exit 1
    fi
  fi
fi

LOG_PATH="${LOG_DIR}/rq3_scheduled_news_regime_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/rq3_scheduled_news_regime_${RUN_TS}.pid"

nohup setsid env \
  ENV_NAME="${ENV_NAME}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq3/run_scheduled_news_regime.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!

printf '%s\n' "${PID}" >"${PID_PATH}"
cat >"${LATEST_PATH}" <<EOF
RUN_TS=${RUN_TS}
OUTPUT_DIR=${OUTPUT_DIR}
LOG_PATH=${LOG_PATH}
PID_PATH=${PID_PATH}
EOF

sleep 2
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "RQ3 scheduled-news analysis exited during startup." >&2
  tail -60 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "RQ3 scheduled-news analysis started in background."
echo "pid=${PID}"
echo "output_dir=${OUTPUT_DIR}"
echo "log=${LOG_PATH}"
echo "status=${OUTPUT_DIR}/registry/pipeline_status.json"
echo "monitor=bash scripts/rq3/monitor_scheduled_news_regime.sh"
