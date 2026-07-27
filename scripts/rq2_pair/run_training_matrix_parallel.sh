#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
ENV_NAME="${ENV_NAME:-py312}"
TASKS_ONLY="${TASKS_ONLY:-0}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(
    find outputs/experiments -mindepth 1 -maxdepth 1 -type d \
      -name 'rq2_pair_representation_raw_vol_continuation_*' -printf '%p\n' |
      sort | tail -n 1
  )"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ2 continuation experiment found." >&2
  exit 1
fi

read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one CUDA device id." >&2
  exit 1
fi
if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_GPU must be a positive integer." >&2
  exit 1
fi

SLOT_GPUS=()
SLOT_NAMES=()
for gpu in "${GPUS[@]}"; do
  for ((slot = 0; slot < RUNS_PER_GPU; slot++)); do
    SLOT_GPUS+=("${gpu}")
    SLOT_NAMES+=("gpu_${gpu}_slot_${slot}")
  done
done

FOLDS=(2023Q1 2023Q2 2023Q3 2023Q4)
SEEDS=(42 202 404)
VARIANTS=(
  pair_pca_bow_residual_pretrained
  pair_sentiment_residual_pretrained
)
TASK_ROOT="${EXPERIMENT_ROOT}/registry/parallel_tasks"
mkdir -p "${TASK_ROOT}"

task_files=()
for slot_name in "${SLOT_NAMES[@]}"; do
  task_file="${TASK_ROOT}/rq2_${slot_name}.tsv"
  : >"${task_file}"
  task_files+=("${task_file}")
done

task_index=0
for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      printf '%s\t%s\t%s\n' "${fold}" "${seed}" "${variant}" \
        >>"${task_files[$((task_index % ${#SLOT_GPUS[@]}))]}"
      task_index=$((task_index + 1))
    done
  done
done

if [[ "${TASKS_ONLY}" == "1" ]]; then
  wc -l "${task_files[@]}"
  exit 0
fi

pids=()
for worker_index in "${!SLOT_GPUS[@]}"; do
  gpu="${SLOT_GPUS[$worker_index]}"
  slot_name="${SLOT_NAMES[$worker_index]}"
  task_file="${task_files[$worker_index]}"
  (
    while IFS=$'\t' read -r fold seed variant; do
      [[ -n "${fold}" ]] || continue
      echo "[gpu=${gpu}] ${fold} seed=${seed} variant=${variant}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/rq2_pair/rq2_pair_experiment.py train-matrix \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --resume \
        --fold "${fold}" \
        --seed "${seed}" \
        --variant "${variant}" \
        --no-registry-write
    done <"${task_file}"
  ) >"${TASK_ROOT}/rq2_${slot_name}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "RQ2 parallel training failed; inspect ${TASK_ROOT}/rq2_gpu_*_slot_*.log" >&2
  exit 1
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq2_pair/rq2_pair_experiment.py train-matrix \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --resume

echo "RQ2 dual-GPU training matrix completed: ${EXPERIMENT_ROOT}"
