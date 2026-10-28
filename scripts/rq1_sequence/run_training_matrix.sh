#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
TASKS_ONLY="${TASKS_ONLY:-0}"

if [[ -z "${EXPERIMENT_ROOT}" && -f outputs/experiments/rq1_intraday_sequence_latest.txt ]]; then
  EXPERIMENT_ROOT="$(<outputs/experiments/rq1_intraday_sequence_latest.txt)"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No prepared RQ1 sequence experiment found." >&2
  exit 1
fi
if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_GPU must be a positive integer." >&2
  exit 1
fi

read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one CUDA device." >&2
  exit 1
fi
SEED_LIST="$(conda run -n "${ENV_NAME}" python -c '
import json, sys
print(" ".join(str(v) for v in json.load(open(sys.argv[1], encoding="utf-8"))["seeds"]))
' "${EXPERIMENT_ROOT}/inputs/experiment_design.json")"
read -r -a SEEDS <<<"${SEED_LIST}"
FOLDS=(2023Q1 2023Q2 2023Q3 2023Q4)
PARENT_VARIANT=history_no_text_parent
SECOND_PHASE_VARIANTS=(
  history_no_text_continued
  current_only_no_text
  history_news_count
  history_matched_lp
  history_shuffled_lp
)

SLOT_GPUS=()
SLOT_NAMES=()
for gpu in "${GPUS[@]}"; do
  for ((slot=0; slot<RUNS_PER_GPU; slot++)); do
    SLOT_GPUS+=("${gpu}")
    SLOT_NAMES+=("gpu_${gpu}_slot_${slot}")
  done
done

TASK_ROOT="${EXPERIMENT_ROOT}/registry/parallel_tasks"
mkdir -p "${TASK_ROOT}"

run_phase() {
  local phase="$1"
  shift
  local variants=("$@")
  local task_files=()
  local slot_name
  for slot_name in "${SLOT_NAMES[@]}"; do
    local task_file="${TASK_ROOT}/${phase}_${slot_name}.tsv"
    : >"${task_file}"
    task_files+=("${task_file}")
  done
  local task_index=0
  local fold seed variant
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for variant in "${variants[@]}"; do
        printf '%s\t%s\t%s\n' "${fold}" "${seed}" "${variant}" \
          >>"${task_files[$((task_index % ${#task_files[@]}))]}"
        task_index=$((task_index + 1))
      done
    done
  done
  echo "phase=${phase} tasks=${task_index} gpu_ids=${GPUS[*]} runs_per_gpu=${RUNS_PER_GPU}"
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
          python scripts/rq1_sequence/rq1_sequence_experiment.py train-one \
          --experiment-root "${EXPERIMENT_ROOT}" \
          --fold "${fold}" \
          --seed "${seed}" \
          --variant "${variant}" \
          --resume
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
    echo "Phase ${phase} failed; inspect ${TASK_ROOT}/${phase}_gpu_*_slot_*.log" >&2
    return 1
  fi
}

run_phase stage_a_parent "${PARENT_VARIANT}"
run_phase stage_b_and_controls "${SECOND_PHASE_VARIANTS[@]}"
if [[ "${TASKS_ONLY}" == "1" ]]; then
  exit 0
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py monitor \
  --experiment-root "${EXPERIMENT_ROOT}"
echo "RQ1 sequence training matrix completed: ${EXPERIMENT_ROOT}"

