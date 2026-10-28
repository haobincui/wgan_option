#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
GPU_ID="${GPU_ID:-0}"
RUNS_PER_GPU="${RUNS_PER_GPU:-3}"
VARIANT="${VARIANT:-history_matched_lp}"
SEEDS=(42 202 404)
FOLDS=(2023Q1 2023Q2 2023Q3 2023Q4)

if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "EXPERIMENT_ROOT must point to the completed RQ1 sequence experiment." >&2
  exit 1
fi
if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_GPU must be a positive integer." >&2
  exit 1
fi

TASK_ROOT="${EXPERIMENT_ROOT}/registry/conformal_tasks/${VARIANT}"
mkdir -p "${TASK_ROOT}"
for ((worker=0; worker<RUNS_PER_GPU; worker++)); do
  : >"${TASK_ROOT}/worker_${worker}.tsv"
done
task_index=0
for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    printf '%s\t%s\n' "${fold}" "${seed}" \
      >>"${TASK_ROOT}/worker_$((task_index % RUNS_PER_GPU)).tsv"
    task_index=$((task_index + 1))
  done
done

pids=()
for ((worker=0; worker<RUNS_PER_GPU; worker++)); do
  (
    while IFS=$'\t' read -r fold seed; do
      [[ -n "${fold}" ]] || continue
      CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/rq1_sequence/conformal_calibration.py generate-validation-one \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --fold "${fold}" \
        --seed "${seed}" \
        --variant "${VARIANT}"
    done <"${TASK_ROOT}/worker_${worker}.tsv"
  ) >"${TASK_ROOT}/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "Conformal validation generation failed; inspect ${TASK_ROOT}/worker_*.log" >&2
  exit 1
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_sequence/conformal_calibration.py calibrate \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --variant "${VARIANT}" \
  --seeds "${SEEDS[@]}"

echo "RQ1 conformal pilot completed: ${EXPERIMENT_ROOT}/conformal_calibration/${VARIANT}"
