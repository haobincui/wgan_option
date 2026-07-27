#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(find outputs/experiments -maxdepth 1 -type d -name 'rq2_pair_representation_raw_vol_continuation_*' -printf '%p\n' | sort | tail -n 1)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No RQ2 continuation experiment found." >&2
  exit 1
fi

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LOG_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.pid"
LATEST_PATH="${LOG_DIR}/results_pipeline_latest.txt"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}"
mkdir -p "${LOG_DIR}"

shopt -s nullglob
for EXISTING_PID_PATH in "${LOG_DIR}"/results_pipeline_[0-9]*.pid; do
  EXISTING_PID="$(tr -d '[:space:]' < "${EXISTING_PID_PATH}")"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    EXISTING_COMMAND="$(ps -o args= -p "${EXISTING_PID}" || true)"
    if [[ "${EXISTING_COMMAND}" == *"rq2_pair/run_results_pipeline.sh"* ]]; then
      echo "RQ2 results pipeline is already running." >&2
      echo "pid=${EXISTING_PID}" >&2
      echo "pid_file=${EXISTING_PID_PATH}" >&2
      exit 1
    fi
  fi
done

nohup setsid bash scripts/rq2_pair/run_results_pipeline.sh \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_PATH}"
printf 'run_ts=%s\nexperiment_root=%s\npid=%s\nlog=%s\npid_file=%s\nbootstrap_iterations=%s\nbootstrap_seed=%s\n' \
  "${RUN_TS}" \
  "${EXPERIMENT_ROOT}" \
  "${PID}" \
  "${LOG_PATH}" \
  "${PID_PATH}" \
  "${BOOTSTRAP_ITERATIONS}" \
  "${BOOTSTRAP_SEED}" \
  >"${LATEST_PATH}"

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
echo "status_file=${EXPERIMENT_ROOT}/registry/results_pipeline_status.json"
echo "bootstrap_iterations=${BOOTSTRAP_ITERATIONS}"
echo "bootstrap_seed=${BOOTSTRAP_SEED}"
