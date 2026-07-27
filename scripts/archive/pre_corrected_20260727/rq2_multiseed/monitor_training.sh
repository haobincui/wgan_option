#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/rq2_multiseed_textbase_20260627}"
REGISTRY="${EXP_ROOT}/registry/launch_registry.csv"
DRIVER_DIR="${EXP_ROOT}/logs/driver"
TAIL_LINES="${TAIL_LINES:-20}"
SHOW_TAIL="${SHOW_TAIL:-0}"

print_driver_status() {
  local name="$1"
  local pid_file="${DRIVER_DIR}/${name}.pid"
  local log_file="${DRIVER_DIR}/${name}.log"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: not started"
    return
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: running pid=${pid} log=${log_file}"
  else
    echo "${name}: stopped pid=${pid:-unknown} log=${log_file}"
  fi
  if [[ "${SHOW_TAIL}" == "1" && -f "${log_file}" ]]; then
    tail -n "${TAIL_LINES}" "${log_file}"
  fi
}

print_driver_status "training_matrix_driver"
print_driver_status "generate_matrix_driver"

if [[ ! -f "${REGISTRY}" ]]; then
  echo "Launch registry not found yet: ${REGISTRY}" >&2
  exit 0
fi

echo "---- launched training jobs ----"

tail -n +2 "${REGISTRY}" | while IFS=, read -r launch_ts model seed text_mode normalize output_root work_dir pid_file log_file pid status; do
  if [[ -z "${pid}" ]]; then
    continue
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    runtime_status="running"
  else
    runtime_status="stopped"
  fi
  echo "${model} seed_${seed} pid=${pid} status=${runtime_status} log=${log_file}"
  if [[ "${SHOW_TAIL}" == "1" && -f "${log_file}" ]]; then
    tail -n "${TAIL_LINES}" "${log_file}"
  fi
done
