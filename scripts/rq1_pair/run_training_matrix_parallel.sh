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
      -name 'rq1_pair_text_raw_vol_continuation_*' -printf '%p\n' |
      sort | tail -n 1
  )"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ1 continuation experiment found." >&2
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
PARENT_VARIANT=pair_pca_no_text_residual
STAGE_B_VARIANTS=(
  pair_pca_no_text_continued
  pair_pca_text_residual_pretrained
  pair_pca_shuffled_residual_pretrained
  pair_pca_text_full_film
  pair_pca_text_concat
  pair_l2_text_full_film
)
TASK_ROOT="${EXPERIMENT_ROOT}/registry/parallel_tasks"
mkdir -p "${TASK_ROOT}"

run_phase() {
  local phase="$1"
  shift
  local variants=("$@")
  local slot_index=0
  local task_files=()
  local slot_name
  for slot_name in "${SLOT_NAMES[@]}"; do
    local task_file="${TASK_ROOT}/${phase}_${slot_name}.tsv"
    : >"${task_file}"
    task_files+=("${task_file}")
  done
  local fold seed variant
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for variant in "${variants[@]}"; do
        printf '%s\t%s\t%s\n' "${fold}" "${seed}" "${variant}" \
          >>"${task_files[$((slot_index % ${#SLOT_GPUS[@]}))]}"
        slot_index=$((slot_index + 1))
      done
    done
  done

  echo "Starting RQ1 ${phase}: gpus=${GPUS[*]}, runs_per_gpu=${RUNS_PER_GPU}, workers=${#SLOT_GPUS[@]}"
  if [[ "${TASKS_ONLY}" == "1" ]]; then
    wc -l "${task_files[@]}"
    return
  fi
  local pids=()
  local worker_index gpu task_file
  for worker_index in "${!SLOT_GPUS[@]}"; do
    gpu="${SLOT_GPUS[$worker_index]}"
    slot_name="${SLOT_NAMES[$worker_index]}"
    task_file="${task_files[$worker_index]}"
    (
      while IFS=$'\t' read -r fold seed variant; do
        [[ -n "${fold}" ]] || continue
        echo "[gpu=${gpu}] ${phase} ${fold} seed=${seed} variant=${variant}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        conda run --no-capture-output -n "${ENV_NAME}" \
          python scripts/rq1_pair/rq1_pair_experiment.py train-matrix \
          --experiment-root "${EXPERIMENT_ROOT}" \
          --resume \
          --fold "${fold}" \
          --seed "${seed}" \
          --variant "${variant}" \
          --no-registry-write
      done <"${task_file}"
    ) >"${TASK_ROOT}/${phase}_${slot_name}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "RQ1 ${phase} failed; inspect ${TASK_ROOT}/${phase}_gpu_*_slot_*.log" >&2
    return 1
  fi
}

run_phase stage_a_parent "${PARENT_VARIANT}"
run_phase stage_b_and_ablations "${STAGE_B_VARIANTS[@]}"
if [[ "${TASKS_ONLY}" == "1" ]]; then
  exit 0
fi

# All runs now exist, so this pass only consolidates the shared registry.
conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_pair/rq1_pair_experiment.py train-matrix \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --resume

echo "RQ1 dual-GPU training matrix completed: ${EXPERIMENT_ROOT}"
