#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
EXPERIMENT_ROOT="${1:-${EXPERIMENT_ROOT:-}}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(find outputs/experiments -maxdepth 1 -type d -name 'rq1_incremental_text_*' | sort | tail -1)"
fi

PID_FILE="${EXPERIMENT_ROOT}/logs/training_matrix.pid"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "matrix_status=running pid=${PID}"
  else
    echo "matrix_status=stopped pid=${PID}"
  fi
else
  echo "matrix_status=not_started"
fi

REGISTRY="${EXPERIMENT_ROOT}/registry/run_registry.csv"
if [[ -f "${REGISTRY}" ]]; then
  conda run -n "${ENV_NAME:-py312}" python -c \
    "import pandas as pd; d=pd.read_csv('${REGISTRY}'); print(d.groupby(['variant','status']).size().to_string())"
fi
LATEST_LOG="$(find "${EXPERIMENT_ROOT}/logs" -type f -name '*.log' | sort | tail -1 || true)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo "latest_log=${LATEST_LOG}"
  tail -40 "${LATEST_LOG}"
fi
