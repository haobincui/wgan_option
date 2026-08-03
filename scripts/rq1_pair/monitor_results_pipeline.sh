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
  echo "No raw-vol RQ1 pair experiment found." >&2
  exit 1
fi

LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
PID_PATH="${LOG_DIR}/results_pipeline_latest.pid"
STATUS_PATH="${EXPERIMENT_ROOT}/registry/results_pipeline_status.json"
REGISTRY_PATH="${EXPERIMENT_ROOT}/registry/generate_registry.csv"
SUMMARY_PATH="${EXPERIMENT_ROOT}/final_tables/development_rq1_result_summary.json"

echo "experiment_root=${EXPERIMENT_ROOT}"
if [[ -s "${PID_PATH}" ]]; then
  PID="$(cat "${PID_PATH}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "process=running pid=${PID}"
    ps -o pid,ppid,pgid,stat,etime,cmd -p "${PID}"
  else
    echo "process=not_running last_pid=${PID}"
  fi
else
  echo "process=not_started"
fi

if [[ -f "${REGISTRY_PATH}" ]]; then
  GENERATED_RUNS="$(( $(wc -l < "${REGISTRY_PATH}") - 1 ))"
  GENERATED_SAMPLES="$(awk -F, 'NR>1 {sum += $6} END {print sum + 0}' "${REGISTRY_PATH}")"
  EXPECTED_RUNS="$(python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for relative in (
    "inputs/experiment_design.json",
    "final_tables/development_rq1_result_summary.json",
    "validation_summary.json",
):
    path = root / relative
    if not path.is_file():
        continue
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("expected_training_runs"):
        print(payload["expected_training_runs"])
        break
    if payload.get("seeds"):
        print(4 * len(payload["seeds"]) * 7)
        break
else:
    print("unknown")
' "${EXPERIMENT_ROOT}")"
  echo "generated_runs=${GENERATED_RUNS}/${EXPECTED_RUNS}"
  echo "generated_samples=${GENERATED_SAMPLES}"
fi

if [[ -f "${STATUS_PATH}" ]]; then
  echo "pipeline_status:"
  cat "${STATUS_PATH}"
fi

if [[ -f "${SUMMARY_PATH}" ]]; then
  echo "result_summary:"
  cat "${SUMMARY_PATH}"
elif [[ -L "${LOG_DIR}/results_pipeline_latest.log" ]]; then
  echo "latest_log_tail:"
  tail -30 "${LOG_DIR}/results_pipeline_latest.log"
fi
