#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${FILM_WGAN_CONDA_ENV:-py312}"
MAX_PARALLEL="${FILM_WGAN_MAX_PARALLEL:-5}"
EXPERIMENT_ROOT="${1:-outputs/experiments/rq1_film_wgan_v3_symmetric_negative_pilot}"
CONFIG_PATH="configs/film_wgan/train_rq1_pair_textbase_v3_pilot.yaml"
ORCHESTRATOR="scripts/rq1_pair/rq1_pair_experiment.py"
FOLD="2023Q1"
SEEDS=(42 202 404)
PARENT_VARIANT="pair_pca_no_text_residual"
CONTINUATION_VARIANT="pair_pca_no_text_continued"
MATCHED_VARIANT="pair_pca_text_residual_pretrained"
SHUFFLED_VARIANT="pair_pca_shuffled_residual_pretrained"
ACTIVE_PIDS=()
ACTIVE_LABELS=()

cd "${REPO_ROOT}"

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FILM_WGAN_MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi
run_cli() {
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${ORCHESTRATOR}" "$@"
}

cleanup_children() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_children INT TERM

wait_next_job() {
  local finished_pid=""
  local exit_code=0
  local finished_label="unknown"
  local index
  set +e
  wait -n -p finished_pid "${ACTIVE_PIDS[@]}"
  exit_code=$?
  set -e
  for index in "${!ACTIVE_PIDS[@]}"; do
    if [[ "${ACTIVE_PIDS[${index}]}" == "${finished_pid}" ]]; then
      finished_label="${ACTIVE_LABELS[${index}]}"
      unset 'ACTIVE_PIDS[index]'
      unset 'ACTIVE_LABELS[index]'
      ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
      ACTIVE_LABELS=("${ACTIVE_LABELS[@]}")
      break
    fi
  done
  if (( exit_code != 0 )); then
    echo "v3 pilot job failed: ${finished_label} (exit=${exit_code})" >&2
    cleanup_children
    exit 1
  fi
}

launch_training_job() {
  local seed="$1"
  local variant="$2"
  local label="${FOLD}/seed_${seed}/${variant}"
  while (( ${#ACTIVE_PIDS[@]} >= MAX_PARALLEL )); do
    wait_next_job
  done
  echo "Launching on physical GPU0: ${label}"
  run_cli train-matrix \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --fold "${FOLD}" \
    --seed "${seed}" \
    --variant "${variant}" \
    --resume &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${label}")
}

finish_phase() {
  while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    wait_next_job
  done
}

if [[ "${FILM_WGAN_LAUNCHER_WAIT_SELF_TEST:-0}" == "1" ]]; then
  (sleep 2) &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("slow-success")
  (exit 7) &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("fast-failure")
  wait_next_job
  echo "wait self-test unexpectedly continued after a failed child" >&2
  exit 99
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to start the v3 pilot from a dirty worktree." >&2
  exit 1
fi

if [[ -f "${EXPERIMENT_ROOT}/inputs/experiment_design.json" ]]; then
  run_cli verify-existing --experiment-root "${EXPERIMENT_ROOT}"
else
  if [[ -d "${EXPERIMENT_ROOT}" ]] && find "${EXPERIMENT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "Experiment root is non-empty but not a verified v3 root: ${EXPERIMENT_ROOT}" >&2
    exit 1
  fi
  run_cli prepare \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --workbook data/processed/raw-excel-session/rq123_cme_session_20260729-131219/merged_vol_rq2_text.xlsx \
    --seeds 42 202 404 \
    --run-folds "${FOLD}" \
    --run-variants \
      "${PARENT_VARIANT}" \
      "${CONTINUATION_VARIANT}" \
      "${MATCHED_VARIANT}" \
      "${SHUFFLED_VARIANT}" \
    --matrix-profile validation_pilot
  run_cli verify-existing --experiment-root "${EXPERIMENT_ROOT}"
fi

echo "Phase 1/3: Stage-A parents"
for seed in "${SEEDS[@]}"; do
  launch_training_job "${seed}" "${PARENT_VARIANT}"
done
finish_phase

echo "Phase 2/3: continuation anchors"
for seed in "${SEEDS[@]}"; do
  launch_training_job "${seed}" "${CONTINUATION_VARIANT}"
done
finish_phase

echo "Phase 3/3: matched and shuffled arms (max parallel=${MAX_PARALLEL})"
for seed in "${SEEDS[@]}"; do
  launch_training_job "${seed}" "${MATCHED_VARIANT}"
  launch_training_job "${seed}" "${SHUFFLED_VARIANT}"
done
finish_phase

run_cli monitor --experiment-root "${EXPERIMENT_ROOT}"
run_cli collect-checkpoints --experiment-root "${EXPERIMENT_ROOT}"
run_cli generate --experiment-root "${EXPERIMENT_ROOT}"
run_cli summarize-validation-pilot --experiment-root "${EXPERIMENT_ROOT}"

echo "RQ1 v3 validation pilot completed on physical GPU0: ${EXPERIMENT_ROOT}"
echo "No outer-test generation or formal results pipeline was run."
