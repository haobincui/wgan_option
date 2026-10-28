#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" && -f outputs/experiments/rq1_intraday_sequence_latest.txt ]]; then
  EXPERIMENT_ROOT="$(<outputs/experiments/rq1_intraday_sequence_latest.txt)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ1 sequence experiment found." >&2
  exit 1
fi

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LOG_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/results_pipeline_${RUN_TS}.pid"
mkdir -p "${LOG_DIR}"

nohup setsid env \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  GPU_IDS="${GPU_IDS:-0 1}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-1}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq1_sequence/run_results_pipeline.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_PATH}"

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"

