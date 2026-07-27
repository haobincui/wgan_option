#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
SHARED_DATA_REPO="${SHARED_DATA_REPO:-$(cd "${REPO_ROOT}/.." && pwd)/wgan_option}"
FEATURE_DIR="${FEATURE_DIR:-data/processed/text_features/rq2/20260625-075653}"
WINDOW_CANDIDATES="${WINDOW_CANDIDATES:-3 5 10 15 30 60}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
PRIMARY_GPU="${GPU_IDS%% *}"

CONTROL_ROOT="${CONTROL_ROOT:-outputs/experiments/rq123_london_recalculation_${RUN_TS}}"
STATE_DIR="${CONTROL_ROOT}/registry/stages"
STATUS_PATH="${CONTROL_ROOT}/registry/pipeline_status.env"
RQ1_ROOT="${RQ1_ROOT:-outputs/experiments/rq1_pair_text_raw_vol_continuation_london_${RUN_TS}}"
RQ2_ROOT="${RQ2_ROOT:-outputs/experiments/rq2_pair_representation_raw_vol_continuation_london_${RUN_TS}}"
RQ3_ROOT="${RQ3_ROOT:-outputs/experiments/rq3_scheduled_news_regime_raw_vol_london_${RUN_TS}}"
SCAN_TS="${SCAN_TS:-london_${RUN_TS}}"
SCAN_ROOT="${SCAN_ROOT:-data/processed/raw-excel/coverage_scan_${SCAN_TS}}"
BEST_JSON="${BEST_JSON:-${SCAN_ROOT}/raw_vol_selected_dataset.json}"

mkdir -p "${STATE_DIR}" "${CONTROL_ROOT}/logs"

write_status() {
  local status="$1"
  local stage="$2"
  local message="${3:-}"
  {
    printf 'RUN_TS=%s\n' "${RUN_TS}"
    printf 'STATUS=%s\n' "${status}"
    printf 'STAGE=%s\n' "${stage}"
    printf 'MESSAGE=%q\n' "${message}"
    printf 'UPDATED_AT_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'SOURCE_TIMEZONE=%s\n' "${SOURCE_TIMEZONE}"
    printf 'GPU_IDS=%q\n' "${GPU_IDS}"
    printf 'RUNS_PER_GPU=%s\n' "${RUNS_PER_GPU}"
    printf 'RQ1_ROOT=%s\n' "${RQ1_ROOT}"
    printf 'RQ2_ROOT=%s\n' "${RQ2_ROOT}"
    printf 'RQ3_ROOT=%s\n' "${RQ3_ROOT}"
    printf 'BEST_JSON=%s\n' "${BEST_JSON}"
  } >"${STATUS_PATH}"
}

stage_done() {
  [[ -f "${STATE_DIR}/$1.done" ]]
}

mark_done() {
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATE_DIR}/$1.done"
}

on_error() {
  local exit_code=$?
  write_status failed "${CURRENT_STAGE:-unknown}" "exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

run_stage() {
  local stage="$1"
  shift
  CURRENT_STAGE="${stage}"
  if stage_done "${stage}"; then
    echo "[skip] ${stage}"
    return
  fi
  write_status running "${stage}"
  echo "[start] ${stage}"
  "$@"
  mark_done "${stage}"
  echo "[done] ${stage}"
  if [[ -n "${STOP_AFTER_STAGE}" && "${STOP_AFTER_STAGE}" == "${stage}" ]]; then
    write_status stopped_after_stage "${stage}"
    echo "Stopped after requested stage: ${stage}"
    exit 0
  fi
}

setup_shared_raw_data() {
  if [[ ! -e data/raw ]]; then
    if [[ ! -d "${SHARED_DATA_REPO}/data/raw" ]]; then
      echo "Shared raw data not found: ${SHARED_DATA_REPO}/data/raw" >&2
      return 1
    fi
    ln -s "${SHARED_DATA_REPO}/data/raw" data/raw
  fi
  test -f data/raw/text_embedding/news_with_openai_embeddings_large.xlsx
  find data/raw/option_data -type f -name '*.csv.gz' -print -quit | grep -q .
  test -f "${FEATURE_DIR}/llm_sentiment_features.xlsx"
}

audit_news_timestamps() {
  conda run -n "${ENV_NAME}" python \
    scripts/rq123_london/audit_news_timestamps.py \
    --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \
    --output-dir "${CONTROL_ROOT}/inputs/news_timestamp_audit" \
    --source-timezone "${SOURCE_TIMEZONE}"
}

run_coverage_scan() {
  SOURCE_TIMEZONE="${SOURCE_TIMEZONE}" \
  SCAN_TS="${SCAN_TS}" \
  SCAN_ROOT="${SCAN_ROOT}" \
  BEST_JSON="${BEST_JSON}" \
  WINDOW_CANDIDATES="${WINDOW_CANDIDATES}" \
  UPDATE_SELECTED_LINK=1 \
  bash scripts/raw_vol/coverage_scan_raw_vol.sh
}

selected_dataset() {
  conda run -n "${ENV_NAME}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_dataset_dir"])' \
    "${BEST_JSON}"
}

enrich_selected_dataset() {
  local dataset
  dataset="$(selected_dataset)"
  FEATURE_DIR="${FEATURE_DIR}" \
  bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh "${dataset}"
}

prepare_rq1() {
  local dataset
  dataset="$(selected_dataset)"
  conda run -n "${ENV_NAME}" python scripts/rq1_pair/rq1_pair_experiment.py prepare \
    --experiment-root "${RQ1_ROOT}" \
    --workbook "${dataset}/merged_vol_rq2_text.xlsx"
}

train_rq1() {
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
  EXPERIMENT_ROOT="${RQ1_ROOT}" \
  bash scripts/rq1_pair/run_training_matrix_parallel.sh
}

evaluate_rq1() {
  CUDA_VISIBLE_DEVICES="${PRIMARY_GPU}" \
  EXPERIMENT_ROOT="${RQ1_ROOT}" \
  bash scripts/rq1_pair/run_results_pipeline.sh
}

prepare_rq2() {
  conda run -n "${ENV_NAME}" python scripts/rq2_pair/rq2_pair_experiment.py prepare \
    --experiment-root "${RQ2_ROOT}" \
    --source-rq1 "${RQ1_ROOT}" \
    --feature-root "${FEATURE_DIR}"
}

train_rq2() {
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
  EXPERIMENT_ROOT="${RQ2_ROOT}" \
  bash scripts/rq2_pair/run_training_matrix_parallel.sh
}

evaluate_rq2() {
  CUDA_VISIBLE_DEVICES="${PRIMARY_GPU}" \
  bash scripts/rq2_pair/run_results_pipeline.sh \
    --experiment-root "${RQ2_ROOT}"
}

run_rq3() {
  ENV_NAME="${ENV_NAME}" \
  RQ1_EXPERIMENT="${RQ1_ROOT}" \
  RQ2_EXPERIMENT="${RQ2_ROOT}" \
  OUTPUT_DIR="${RQ3_ROOT}" \
  bash scripts/rq3/run_scheduled_news_regime.sh
}

write_manifest() {
  local dataset
  dataset="$(selected_dataset)"
  {
    printf 'run_ts=%s\n' "${RUN_TS}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'source_timezone=%s\n' "${SOURCE_TIMEZONE}"
    printf 'gpu_ids=%s\n' "${GPU_IDS}"
    printf 'runs_per_gpu=%s\n' "${RUNS_PER_GPU}"
    printf 'shared_data_repo=%s\n' "${SHARED_DATA_REPO}"
    printf 'selected_dataset=%s\n' "${dataset}"
    printf 'rq1_experiment=%s\n' "${RQ1_ROOT}"
    printf 'rq2_experiment=%s\n' "${RQ2_ROOT}"
    printf 'rq3_experiment=%s\n' "${RQ3_ROOT}"
  } >"${CONTROL_ROOT}/run_manifest.txt"
}

if [[ "${SOURCE_TIMEZONE}" != "Europe/London" ]]; then
  echo "Canonical RQ1-RQ3 recalculation requires SOURCE_TIMEZONE=Europe/London." >&2
  exit 1
fi

write_status running initializing
run_stage setup_shared_raw_data setup_shared_raw_data
run_stage audit_news_timestamps audit_news_timestamps
run_stage raw_vol_coverage_scan run_coverage_scan
run_stage enrich_raw_vol_rq2 enrich_selected_dataset
run_stage rq1_prepare prepare_rq1
run_stage rq1_train train_rq1
run_stage rq1_evaluate evaluate_rq1
run_stage rq2_prepare prepare_rq2
run_stage rq2_train train_rq2
run_stage rq2_evaluate evaluate_rq2
run_stage rq3_scheduled_news run_rq3
run_stage write_manifest write_manifest
write_status completed completed
echo "RQ1-RQ3 London-time recalculation completed: ${CONTROL_ROOT}"
