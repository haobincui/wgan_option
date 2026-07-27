#!/usr/bin/env bash
# Run the full raw-vol RQ1/RQ2 pipeline in the background.
#
# Default behavior:
#   bash scripts/raw_vol/run_full_raw_vol_rq_background.sh
#
# The first invocation starts a background driver and exits.  The child driver
# then runs coverage scan, RQ2 enrichment, multi-seed training, generate-result,
# comparison archive build, and packaging in sequence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

FULL_RUN_TS="${FULL_RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_${FULL_RUN_TS}}"
DRIVER_DIR="${EXP_ROOT}/logs/driver"
DRIVER_LOG="${DRIVER_LOG:-${DRIVER_DIR}/full_raw_vol_pipeline.log}"
DRIVER_PID_FILE="${DRIVER_PID_FILE:-${DRIVER_DIR}/full_raw_vol_pipeline.pid}"
STATUS_FILE="${STATUS_FILE:-${DRIVER_DIR}/full_raw_vol_pipeline.status}"

if [[ "${RAW_VOL_FULL_PIPELINE_CHILD:-0}" != "1" ]]; then
  mkdir -p "${DRIVER_DIR}"
  if [[ -f "${DRIVER_PID_FILE}" ]]; then
    old_pid="$(cat "${DRIVER_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Raw-vol full pipeline is already running: pid=${old_pid}"
      echo "Log: ${DRIVER_LOG}"
      exit 0
    fi
  fi

  RAW_VOL_FULL_PIPELINE_CHILD=1 \
  FULL_RUN_TS="${FULL_RUN_TS}" \
  EXP_ROOT="${EXP_ROOT}" \
  DRIVER_LOG="${DRIVER_LOG}" \
  DRIVER_PID_FILE="${DRIVER_PID_FILE}" \
  STATUS_FILE="${STATUS_FILE}" \
  nohup bash "$0" "$@" >"${DRIVER_LOG}" 2>&1 &
  driver_pid=$!
  echo "${driver_pid}" >"${DRIVER_PID_FILE}"
  disown "${driver_pid}" 2>/dev/null || true

  echo "Raw-vol full pipeline started in background."
  echo "PID file: ${DRIVER_PID_FILE}"
  echo "Log file: ${DRIVER_LOG}"
  echo "Status file: ${STATUS_FILE}"
  echo "Monitor:"
  echo "  tail -f ${DRIVER_LOG}"
  echo "  EXP_ROOT=${EXP_ROOT} bash scripts/raw_vol_multiseed/monitor_training.sh"
  exit 0
fi

log_step() {
  local message="$1"
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$stamp] ${message}" | tee -a "${STATUS_FILE}"
}

fail_step() {
  local exit_code=$?
  log_step "FAILED exit_code=${exit_code} line=${BASH_LINENO[0]}"
  exit "${exit_code}"
}
trap fail_step ERR

ENV_NAME="${ENV_NAME:-py312}"
DEVICE="${DEVICE:-cpu}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CONFIG_PATH="${CONFIG_PATH:-configs/film_wgan/train_raw_vol_textbase.yaml}"
SURFACE_CONFIG_PATH="${SURFACE_CONFIG_PATH:-configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml}"
FEATURE_DIR="${FEATURE_DIR:-}"
SCAN_TS="${SCAN_TS:-${FULL_RUN_TS}}"
SCAN_ROOT="${SCAN_ROOT:-data/processed/raw-excel/coverage_scan_${SCAN_TS}}"
WINDOW_CANDIDATES="${WINDOW_CANDIDATES:-3 5 10 15 30 60}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
SEEDS="${SEEDS:-42 101 202 303 404}"
SEEDS_CSV="${SEEDS_CSV:-$(printf '%s' "${SEEDS}" | tr ' ' ',')}"
MODELS_TO_RUN="${MODELS_TO_RUN:-text no_text bow llm_sentiment}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PACKAGE_EXPERIMENT="${PACKAGE_EXPERIMENT:-1}"

SKIP_COVERAGE_SCAN="${SKIP_COVERAGE_SCAN:-0}"
SKIP_ENRICH="${SKIP_ENRICH:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_COMPARISON="${SKIP_COMPARISON:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
FORCE_ENRICH="${FORCE_ENRICH:-0}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"

mkdir -p "${DRIVER_DIR}"
: >"${STATUS_FILE}"

log_step "START full raw-vol RQ1/RQ2 pipeline"
log_step "EXP_ROOT=${EXP_ROOT}"
log_step "FULL_RUN_TS=${FULL_RUN_TS}"
log_step "ENV_NAME=${ENV_NAME} DEVICE=${DEVICE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_step "SEEDS=${SEEDS}"
log_step "MODELS_TO_RUN=${MODELS_TO_RUN}"

if [[ "${SKIP_COVERAGE_SCAN}" != "1" ]]; then
  log_step "STEP 1/6 coverage scan"
  ENV_NAME="${ENV_NAME}" \
  DEVICE="${DEVICE}" \
  CONFIG_PATH="${SURFACE_CONFIG_PATH}" \
  SCAN_TS="${SCAN_TS}" \
  SCAN_ROOT="${SCAN_ROOT}" \
  WINDOW_CANDIDATES="${WINDOW_CANDIDATES}" \
  MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS}" \
  WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS}" \
  UPDATE_SELECTED_LINK=1 \
  bash scripts/raw_vol/coverage_scan_raw_vol.sh
else
  log_step "STEP 1/6 coverage scan skipped"
fi

SELECTED_LINK="data/processed/raw-excel/rq_raw_vol_selected"
if [[ -z "${DATASET_DIR:-}" ]]; then
  BEST_JSON="${BEST_JSON:-${SCAN_ROOT}/raw_vol_selected_dataset.json}"
  if [[ -f "${BEST_JSON}" ]]; then
    DATASET_DIR="$(python - <<PY
import json
print(json.load(open("${BEST_JSON}", "r", encoding="utf-8"))["selected_dataset_dir"])
PY
)"
  elif [[ -e "${SELECTED_LINK}" ]]; then
    DATASET_DIR="${SELECTED_LINK}"
  else
    echo "Cannot determine DATASET_DIR. Run coverage scan or set DATASET_DIR explicitly." >&2
    exit 1
  fi
fi

if [[ -e "${SELECTED_LINK}" ]]; then
  DATA_PATH="${DATA_PATH:-${SELECTED_LINK}/merged_vol_rq2_text.xlsx}"
else
  DATA_PATH="${DATA_PATH:-${DATASET_DIR}/merged_vol_rq2_text.xlsx}"
fi

log_step "Selected DATASET_DIR=${DATASET_DIR}"
log_step "Training DATA_PATH=${DATA_PATH}"

if [[ "${SKIP_ENRICH}" != "1" ]]; then
  if [[ "${FORCE_ENRICH}" == "1" || ! -f "${DATA_PATH}" ]]; then
    log_step "STEP 2/6 enrich raw-vol workbook with RQ2 text features"
    FEATURE_DIR="${FEATURE_DIR}" ENV_NAME="${ENV_NAME}" bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh "${DATASET_DIR}"
  else
    log_step "STEP 2/6 enrich skipped because DATA_PATH exists"
  fi
else
  log_step "STEP 2/6 enrich skipped"
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Expected enriched raw-vol workbook missing: ${DATA_PATH}" >&2
  exit 1
fi

log_step "STEP 3/6 prepare experiment archive"
EXP_ROOT="${EXP_ROOT}" \
CONFIG_PATH="${CONFIG_PATH}" \
DATA_PATH="${DATA_PATH}" \
FEATURE_DIR="${FEATURE_DIR:-data/processed/text_features/rq2/20260625-075653}" \
bash scripts/raw_vol_multiseed/prepare_experiment.sh

if [[ "${SKIP_TRAINING}" != "1" ]]; then
  log_step "STEP 4/6 multi-seed training matrix"
  EXP_ROOT="${EXP_ROOT}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  DATA_PATH="${DATA_PATH}" \
  SEEDS="${SEEDS}" \
  MODELS_TO_RUN="${MODELS_TO_RUN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  ENV_NAME="${ENV_NAME}" \
  WAIT_FOR_COMPLETION=1 \
  POLL_SECONDS="${POLL_SECONDS}" \
  SKIP_PREPARE=1 \
  bash scripts/raw_vol_multiseed/run_training_matrix.sh
else
  log_step "STEP 4/6 training skipped"
fi

log_step "Collect checkpoint registry"
EXP_ROOT="${EXP_ROOT}" DATA_PATH="${DATA_PATH}" SEEDS_CSV="${SEEDS_CSV}" bash scripts/raw_vol_multiseed/collect_run_registry.sh

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  log_step "STEP 5/6 generate all-split JSON"
  EXP_ROOT="${EXP_ROOT}" \
  DATA_PATH="${DATA_PATH}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  ENV_NAME="${ENV_NAME}" \
  SKIP_COLLECT=1 \
  FORCE_GENERATE="${FORCE_GENERATE}" \
  bash scripts/raw_vol_multiseed/run_generate_matrix.sh
else
  log_step "STEP 5/6 generate skipped"
fi

if [[ "${SKIP_COMPARISON}" != "1" ]]; then
  log_step "STEP 6/6 build comparison archive"
  EXP_ROOT="${EXP_ROOT}" SEEDS_CSV="${SEEDS_CSV}" bash scripts/raw_vol_multiseed/build_comparison_archive.sh
else
  log_step "STEP 6/6 comparison skipped"
fi

if [[ "${PACKAGE_EXPERIMENT}" == "1" && "${SKIP_PACKAGE}" != "1" ]]; then
  log_step "Package experiment zip"
  EXP_ROOT="${EXP_ROOT}" bash scripts/raw_vol_multiseed/package_experiment.sh
fi

log_step "DONE full raw-vol RQ1/RQ2 pipeline"
