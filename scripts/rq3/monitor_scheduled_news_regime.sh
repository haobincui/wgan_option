#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LATEST_PATH="${LATEST_PATH:-outputs/rq3/logs/rq3_scheduled_news_regime_latest.env}"
if [[ ! -f "${LATEST_PATH}" ]]; then
  echo "No RQ3 scheduled-news background run has been registered." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${LATEST_PATH}"

echo "output_dir=${OUTPUT_DIR}"
echo "log=${LOG_PATH}"
if [[ -s "${PID_PATH}" ]]; then
  PID="$(cat "${PID_PATH}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "process=running pid=${PID}"
    ps -o pid,ppid,pgid,stat,etime,cmd -p "${PID}"
  else
    echo "process=not_running last_pid=${PID}"
  fi
else
  echo "process=unknown"
fi

STATUS_PATH="${OUTPUT_DIR}/registry/pipeline_status.json"
if [[ -f "${STATUS_PATH}" ]]; then
  echo "pipeline_status:"
  cat "${STATUS_PATH}"
else
  echo "pipeline_status=not_created"
fi

VALIDATION_PATH="${OUTPUT_DIR}/validation_summary.json"
if [[ -f "${VALIDATION_PATH}" ]]; then
  echo "validation_summary:"
  cat "${VALIDATION_PATH}"
elif [[ -f "${LOG_PATH}" ]]; then
  echo "latest_log_tail:"
  tail -50 "${LOG_PATH}"
fi
