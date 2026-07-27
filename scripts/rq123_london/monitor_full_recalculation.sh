#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LATEST_PATH="${LATEST_PATH:-outputs/logs/rq123_london/latest.env}"
if [[ ! -f "${LATEST_PATH}" ]]; then
  echo "No RQ1-RQ3 London pipeline registry found: ${LATEST_PATH}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${LATEST_PATH}"

pid="$(cat "${PID_PATH}")"
if kill -0 "${pid}" 2>/dev/null; then
  echo "process_status=running"
else
  echo "process_status=stopped"
fi
echo "pid=${pid}"
echo "log=${LOG_PATH}"
echo "control_root=${CONTROL_ROOT}"
if [[ -f "${CONTROL_ROOT}/registry/pipeline_status.env" ]]; then
  echo "--- pipeline status ---"
  cat "${CONTROL_ROOT}/registry/pipeline_status.env"
fi
echo "--- completed stages ---"
find "${CONTROL_ROOT}/registry/stages" -maxdepth 1 -type f -name '*.done' \
  -printf '%f\n' 2>/dev/null | sort || true
echo "--- recent log ---"
tail -80 "${LOG_PATH}" 2>/dev/null || true
