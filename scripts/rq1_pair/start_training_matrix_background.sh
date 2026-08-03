#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(find outputs/experiments -maxdepth 1 -type d -name 'rq1_pair_text_raw_vol_continuation_*' -printf '%p\n' | sort | tail -n 1)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared raw-vol RQ1 continuation experiment found. Run scripts/rq1_pair/prepare_experiment.sh first." >&2
  exit 1
fi

SEED_COUNT="$(python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for relative in (
    "inputs/experiment_design.json",
    "final_tables/development_rq1_result_summary.json",
    "validation_summary.json",
):
    path = root / relative
    if path.is_file() and json.load(open(path, encoding="utf-8")).get("seeds"):
        print(len(json.load(open(path, encoding="utf-8"))["seeds"]))
        break
else:
    print(15)
' "${EXPERIMENT_ROOT}")"
EXPECTED_RUNS="$((4 * SEED_COUNT * 7))"

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${EXPERIMENT_ROOT}/logs/background"
LOG_PATH="${LOG_DIR}/training_matrix_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/training_matrix_${RUN_TS}.pid"
mkdir -p "${LOG_DIR}"

nohup setsid env \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  GPU_IDS="${GPU_IDS:-0}" \
  RUNS_PER_GPU="${RUNS_PER_GPU:-13}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  PYTHONUNBUFFERED=1 \
  bash scripts/rq1_pair/run_training_matrix_parallel.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_PATH}"

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
echo "gpu_ids=${GPU_IDS:-0}"
echo "runs_per_gpu=${RUNS_PER_GPU:-13}"
echo "seed_count=${SEED_COUNT}"
echo "expected_training_runs=${EXPECTED_RUNS}"
