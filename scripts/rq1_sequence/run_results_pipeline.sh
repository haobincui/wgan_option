#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
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

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py collect-checkpoints \
  --experiment-root "${EXPERIMENT_ROOT}"

read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one CUDA device." >&2
  exit 1
fi
SLOT_GPUS=()
for gpu in "${GPUS[@]}"; do
  for ((slot=0; slot<RUNS_PER_GPU; slot++)); do
    SLOT_GPUS+=("${gpu}")
  done
done
SEED_LIST="$(conda run -n "${ENV_NAME}" python -c '
import json, sys
print(" ".join(str(v) for v in json.load(open(sys.argv[1], encoding="utf-8"))["seeds"]))
' "${EXPERIMENT_ROOT}/inputs/experiment_design.json")"
read -r -a SEEDS <<<"${SEED_LIST}"
FOLDS=(2023Q1 2023Q2 2023Q3 2023Q4)
VARIANTS=(
  history_no_text_parent
  history_no_text_continued
  current_only_no_text
  history_news_count
  history_matched_lp
  history_shuffled_lp
)
TASK_ROOT="${EXPERIMENT_ROOT}/registry/generate_tasks"
mkdir -p "${TASK_ROOT}"
TASK_FILES=()
for index in "${!SLOT_GPUS[@]}"; do
  TASK_FILES+=("${TASK_ROOT}/worker_${index}.tsv")
  : >"${TASK_FILES[$index]}"
done
task_index=0
for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      printf '%s\t%s\t%s\n' "${fold}" "${seed}" "${variant}" \
        >>"${TASK_FILES[$((task_index % ${#TASK_FILES[@]}))]}"
      task_index=$((task_index + 1))
    done
  done
done

pids=()
for index in "${!SLOT_GPUS[@]}"; do
  gpu="${SLOT_GPUS[$index]}"
  (
    while IFS=$'\t' read -r fold seed variant; do
      [[ -n "${fold}" ]] || continue
      CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run --no-capture-output -n "${ENV_NAME}" \
        python scripts/rq1_sequence/rq1_sequence_experiment.py generate-one \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --fold "${fold}" \
        --seed "${seed}" \
        --variant "${variant}"
    done <"${TASK_FILES[$index]}"
  ) >"${TASK_ROOT}/worker_${index}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "Generate matrix failed; inspect ${TASK_ROOT}/worker_*.log" >&2
  exit 1
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py compare \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-20260722}"
echo "RQ1 sequence result pipeline completed: ${EXPERIMENT_ROOT}"
