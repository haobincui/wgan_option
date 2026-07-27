#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(
    find outputs/experiments -maxdepth 1 -type d \
      -name 'rq1_pair_text_raw_vol_continuation_*' -printf '%p\n' |
      sort |
      tail -n 1
  )"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No completed raw-vol RQ1 pair experiment found." >&2
  exit 1
fi

LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LATEST_PID_PATH="${LOG_DIR}/results_pipeline_latest.pid"
mkdir -p "${LOG_DIR}"

if [[ -s "${LATEST_PID_PATH}" ]]; then
  EXISTING_PID="$(cat "${LATEST_PID_PATH}")"
  if kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "RQ1 result pipeline is already running with PID ${EXISTING_PID}." >&2
    exit 1
  fi
fi

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.pid"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}"

nohup setsid env \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq1_pair/run_results_pipeline.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!

printf '%s\n' "${PID}" >"${PID_PATH}"
printf '%s\n' "${PID}" >"${LATEST_PID_PATH}"
ln -sfn "$(basename "${LOG_PATH}")" "${LOG_DIR}/results_pipeline_latest.log"

sleep 2
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "RQ1 result pipeline exited during startup. Log tail:" >&2
  tail -40 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
echo "status=${EXPERIMENT_ROOT}/registry/results_pipeline_status.json"
echo "result_summary=${EXPERIMENT_ROOT}/final_tables/development_rq1_result_summary.json"
