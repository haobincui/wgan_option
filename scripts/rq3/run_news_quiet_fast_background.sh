#!/usr/bin/env bash
# Fast RQ3 news-event vs no-news quiet workflow.
#
# Default:
#   RQ3_SURFACE_WORKERS=32 \
#   QUIET_MAX_SAMPLES=3711 \
#   QUIET_CANDIDATE_COUNT=12000 \
#   QUIET_SAMPLE_SEED=20260625 \
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/rq3/run_news_quiet_fast_background.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/rq3/rq3_news_quiet_fast_${RUN_TS}}"
LOG_DIR="${LOG_DIR:-outputs/rq3/logs}"
DRIVER_LOG="${DRIVER_LOG:-${LOG_DIR}/rq3_news_quiet_fast_${RUN_TS}.log}"
DRIVER_PID_FILE="${DRIVER_PID_FILE:-${LOG_DIR}/rq3_news_quiet_fast_${RUN_TS}.pid}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/rq3_news_quiet_fast_${RUN_TS}.status}"

if [[ "${RQ3_NEWS_QUIET_FAST_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  if [[ -f "${DRIVER_PID_FILE}" ]]; then
    old_pid="$(cat "${DRIVER_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "RQ3 fast news/quiet workflow is already running: pid=${old_pid}"
      echo "Log: ${DRIVER_LOG}"
      exit 0
    fi
  fi

  RQ3_NEWS_QUIET_FAST_CHILD=1 \
  RUN_TS="${RUN_TS}" \
  OUT_ROOT="${OUT_ROOT}" \
  LOG_DIR="${LOG_DIR}" \
  DRIVER_LOG="${DRIVER_LOG}" \
  DRIVER_PID_FILE="${DRIVER_PID_FILE}" \
  STATUS_FILE="${STATUS_FILE}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  RQ3_SURFACE_WORKERS="${RQ3_SURFACE_WORKERS:-32}" \
  QUIET_MAX_SAMPLES="${QUIET_MAX_SAMPLES:-3711}" \
  QUIET_CANDIDATE_COUNT="${QUIET_CANDIDATE_COUNT:-12000}" \
  QUIET_SAMPLE_SEED="${QUIET_SAMPLE_SEED:-20260625}" \
  FORCE_SURFACE="${FORCE_SURFACE:-0}" \
  FORCE_GENERATE="${FORCE_GENERATE:-0}" \
  FORCE_TARGETS="${FORCE_TARGETS:-0}" \
  nohup bash "$0" "$@" >"${DRIVER_LOG}" 2>&1 &
  driver_pid=$!
  echo "${driver_pid}" >"${DRIVER_PID_FILE}"
  disown "${driver_pid}" 2>/dev/null || true

  echo "RQ3 fast news/quiet workflow started in background."
  echo "PID file: ${DRIVER_PID_FILE}"
  echo "Log file: ${DRIVER_LOG}"
  echo "Status file: ${STATUS_FILE}"
  echo "Output root: ${OUT_ROOT}"
  echo "Monitor:"
  echo "  tail -f ${STATUS_FILE}"
  exit 0
fi

log_step() {
  local message="$1"
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${stamp}] ${message}" | tee -a "${STATUS_FILE}"
}

fail_step() {
  local exit_code=$?
  log_step "FAILED exit_code=${exit_code} line=${BASH_LINENO[0]}"
  exit "${exit_code}"
}
trap fail_step ERR

ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RQ3_SURFACE_WORKERS="${RQ3_SURFACE_WORKERS:-32}"
QUIET_MAX_SAMPLES="${QUIET_MAX_SAMPLES:-3711}"
QUIET_CANDIDATE_COUNT="${QUIET_CANDIDATE_COUNT:-12000}"
QUIET_SAMPLE_SEED="${QUIET_SAMPLE_SEED:-20260625}"
FORCE_SURFACE="${FORCE_SURFACE:-0}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"
FORCE_TARGETS="${FORCE_TARGETS:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
: >"${STATUS_FILE}"

SVI_SOURCE_WORKBOOK="${SVI_SOURCE_WORKBOOK:-data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx}"
RAW_SOURCE_WORKBOOK="${RAW_SOURCE_WORKBOOK:-outputs/experiments/raw_vol_rq1_rq2_20260629-144428/inputs/data/merged_vol_rq2_text.xlsx}"
NEWS_XLSX="${NEWS_XLSX:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"

QUIET_TARGET_DIR="${OUT_ROOT}/quiet_targets"
QUIET_TARGETS="${QUIET_TARGET_DIR}/quiet_targets.txt"
SVI_SURFACE_RUN_TS="rq3_news_quiet_fast_svi_window_${RUN_TS}"
RAW_SURFACE_RUN_TS="rq3_news_quiet_fast_raw_window_${RUN_TS}"
SVI_WINDOW_JSON="data/processed/svi-window/${SVI_SURFACE_RUN_TS}/surface-svi-window.json"
RAW_WINDOW_JSON="data/processed/raw-window/${RAW_SURFACE_RUN_TS}/surface-raw-window.json"
SVI_WORKBOOK="${OUT_ROOT}/svi_news_quiet_eval_workbook.xlsx"
RAW_WORKBOOK="${OUT_ROOT}/raw_vol_news_quiet_eval_workbook.xlsx"
GEN_DIR="rq3_news_quiet_fast_${RUN_TS}_all_json"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

run_generate() {
  local label="$1"
  local run_dir="$2"
  local workbook_path="$3"
  local config_path="${run_dir}/metrics/training_resolved_config.yaml"
  local checkpoint_path="${run_dir}/checkpoints/film_wgan_best_val_mae_gap_vs_current.pt"
  local summary_path="${run_dir}/${GEN_DIR}/summary.csv"

  require_file "${config_path}" "${label} resolved config"
  require_file "${checkpoint_path}" "${label} checkpoint"
  require_file "${workbook_path}" "${label} news/quiet workbook"

  if [[ -f "${summary_path}" && "${FORCE_GENERATE}" != "1" ]]; then
    log_step "Reuse generate-result ${label}: ${summary_path}"
    return
  fi

  log_step "Run generate-result ${label}"
  conda run -n "${ENV_NAME}" python scripts/film_wgan/main.py generate-result \
    --config "${config_path}" \
    --checkpoint "${checkpoint_path}" \
    --output-dir "${GEN_DIR}" \
    --split all \
    --selection-mode all \
    --selection-count 0 \
    --no-plot \
    --set "data_path=${workbook_path}"

  require_file "${summary_path}" "${label} generated summary"
}

log_step "START fast RQ3 news-event vs no-news quiet workflow"
log_step "RUN_TS=${RUN_TS}"
log_step "OUT_ROOT=${OUT_ROOT}"
log_step "ENV_NAME=${ENV_NAME}"
log_step "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_step "RQ3_SURFACE_WORKERS=${RQ3_SURFACE_WORKERS}"
log_step "QUIET_MAX_SAMPLES=${QUIET_MAX_SAMPLES}"
log_step "QUIET_CANDIDATE_COUNT=${QUIET_CANDIDATE_COUNT}"
log_step "QUIET_SAMPLE_SEED=${QUIET_SAMPLE_SEED}"

require_file "${SVI_SOURCE_WORKBOOK}" "SVI source workbook"
require_file "${RAW_SOURCE_WORKBOOK}" "raw-vol source workbook"
require_file "${NEWS_XLSX}" "news workbook"

if [[ ! -f "${QUIET_TARGETS}" || "${FORCE_TARGETS}" == "1" ]]; then
  log_step "Prepare no-news quiet target timestamps"
  conda run -n "${ENV_NAME}" python scripts/rq3/main.py prepare-news-quiet-targets \
    --source-merged-vol "${SVI_SOURCE_WORKBOOK}" \
    --news-xlsx "${NEWS_XLSX}" \
    --output-dir "${QUIET_TARGET_DIR}" \
    --horizon-minutes 5 \
    --quiet-grid-minutes 5 \
    --quiet-buffer-minutes 60 \
    --candidate-count "${QUIET_CANDIDATE_COUNT}" \
    --sample-seed "${QUIET_SAMPLE_SEED}"
else
  log_step "Reuse quiet target timestamps: ${QUIET_TARGETS}"
fi
require_file "${QUIET_TARGETS}" "quiet target timestamps"

if [[ ! -f "${SVI_WINDOW_JSON}" || "${FORCE_SURFACE}" == "1" ]]; then
  log_step "Generate SVI quiet window surfaces"
  conda run -n "${ENV_NAME}" python scripts/generate_surface/main.py generate_surface \
    --config configs/surface_builder/svi/generate_surface-svi-window.yaml \
    --model svi \
    --data_range window \
    --run-ts "${SVI_SURFACE_RUN_TS}" \
    --target-datetimes-file "${QUIET_TARGETS}" \
    --window-minutes 5 \
    --calibration-workers "${RQ3_SURFACE_WORKERS}"
else
  log_step "Reuse SVI quiet window surfaces: ${SVI_WINDOW_JSON}"
fi
require_file "${SVI_WINDOW_JSON}" "SVI quiet window JSON"

if [[ ! -f "${RAW_WINDOW_JSON}" || "${FORCE_SURFACE}" == "1" ]]; then
  log_step "Generate raw-vol quiet window surfaces"
  conda run -n "${ENV_NAME}" python scripts/generate_surface/main.py generate_surface \
    --config configs/surface_builder/raw/generate_surface-raw-window.yaml \
    --model raw \
    --data_range window \
    --run-ts "${RAW_SURFACE_RUN_TS}" \
    --target-datetimes-file "${QUIET_TARGETS}" \
    --window-minutes 5
else
  log_step "Reuse raw-vol quiet window surfaces: ${RAW_WINDOW_JSON}"
fi
require_file "${RAW_WINDOW_JSON}" "raw-vol quiet window JSON"

log_step "Build SVI news/quiet workbook from window surfaces"
conda run -n "${ENV_NAME}" python scripts/rq3/main.py build-news-quiet-workbook-from-window \
  --source-merged-vol "${SVI_SOURCE_WORKBOOK}" \
  --window-surface-json "${SVI_WINDOW_JSON}" \
  --output-workbook "${SVI_WORKBOOK}" \
  --horizon-minutes 5 \
  --quiet-grid-minutes 5 \
  --quiet-buffer-minutes 60 \
  --quiet-max-samples "${QUIET_MAX_SAMPLES}" \
  --quiet-sample-seed "${QUIET_SAMPLE_SEED}"

log_step "Build raw-vol news/quiet workbook from window surfaces"
conda run -n "${ENV_NAME}" python scripts/rq3/main.py build-news-quiet-workbook-from-window \
  --source-merged-vol "${RAW_SOURCE_WORKBOOK}" \
  --window-surface-json "${RAW_WINDOW_JSON}" \
  --output-workbook "${RAW_WORKBOOK}" \
  --horizon-minutes 5 \
  --quiet-grid-minutes 5 \
  --quiet-buffer-minutes 60 \
  --quiet-max-samples "${QUIET_MAX_SAMPLES}" \
  --quiet-sample-seed "${QUIET_SAMPLE_SEED}"

log_step "Analyze workbook IVS jumps"
conda run -n "${ENV_NAME}" python scripts/rq3/main.py news-quiet-workbook \
  --workbook "${SVI_WORKBOOK}" \
  --output-dir "${OUT_ROOT}/svi_workbook_analysis" \
  --split all
conda run -n "${ENV_NAME}" python scripts/rq3/main.py news-quiet-workbook \
  --workbook "${RAW_WORKBOOK}" \
  --output-dir "${OUT_ROOT}/raw_vol_workbook_analysis" \
  --split all

SVI_TEXT_RUN="outputs/experiments/rq2_multiseed_textbase_20260627/training_runs/text/seed_404/20260627_103354"
SVI_NOTEXT_RUN="outputs/experiments/rq2_multiseed_textbase_20260627/training_runs/no_text/seed_101/20260627_105956"
SVI_BOW_RUN="outputs/experiments/rq2_multiseed_textbase_20260627/training_runs/bow/seed_404/20260627_130500"
SVI_LLM_RUN="outputs/experiments/rq2_multiseed_textbase_20260627/training_runs/llm_sentiment/seed_303/20260627_140802"

RAW_TEXT_RUN="outputs/experiments/raw_vol_rq1_rq2_20260629-144428/training_runs/text/seed_202/20260629_190920"
RAW_NOTEXT_RUN="outputs/experiments/raw_vol_rq1_rq2_20260629-144428/training_runs/no_text/seed_303/20260629_204422"
RAW_BOW_RUN="outputs/experiments/raw_vol_rq1_rq2_20260629-144428/training_runs/bow/seed_202/20260629_214824"
RAW_LLM_RUN="outputs/experiments/raw_vol_rq1_rq2_20260629-144428/training_runs/llm_sentiment/seed_42/20260629_223725"

run_generate "svi_text_seed404" "${SVI_TEXT_RUN}" "${SVI_WORKBOOK}"
run_generate "svi_no_text_seed101" "${SVI_NOTEXT_RUN}" "${SVI_WORKBOOK}"
run_generate "svi_bow_seed404" "${SVI_BOW_RUN}" "${SVI_WORKBOOK}"
run_generate "svi_llm_sentiment_seed303" "${SVI_LLM_RUN}" "${SVI_WORKBOOK}"

run_generate "raw_text_seed202" "${RAW_TEXT_RUN}" "${RAW_WORKBOOK}"
run_generate "raw_no_text_seed303" "${RAW_NOTEXT_RUN}" "${RAW_WORKBOOK}"
run_generate "raw_bow_seed202" "${RAW_BOW_RUN}" "${RAW_WORKBOOK}"
run_generate "raw_llm_sentiment_seed42" "${RAW_LLM_RUN}" "${RAW_WORKBOOK}"

log_step "Analyze SVI model result groups and DiD"
conda run -n "${ENV_NAME}" python scripts/rq3/main.py news-quiet-result \
  --output-dir "${OUT_ROOT}/svi_result_analysis" \
  --text-label text_seed404 \
  --result "text_seed404=${SVI_TEXT_RUN}/${GEN_DIR}/summary.csv" \
  --result "no_text_seed101=${SVI_NOTEXT_RUN}/${GEN_DIR}/summary.csv" \
  --result "bow_seed404=${SVI_BOW_RUN}/${GEN_DIR}/summary.csv" \
  --result "llm_sentiment_seed303=${SVI_LLM_RUN}/${GEN_DIR}/summary.csv"

log_step "Analyze raw-vol model result groups and DiD"
conda run -n "${ENV_NAME}" python scripts/rq3/main.py news-quiet-result \
  --output-dir "${OUT_ROOT}/raw_vol_result_analysis" \
  --text-label text_seed202 \
  --result "text_seed202=${RAW_TEXT_RUN}/${GEN_DIR}/summary.csv" \
  --result "no_text_seed303=${RAW_NOTEXT_RUN}/${GEN_DIR}/summary.csv" \
  --result "bow_seed202=${RAW_BOW_RUN}/${GEN_DIR}/summary.csv" \
  --result "llm_sentiment_seed42=${RAW_LLM_RUN}/${GEN_DIR}/summary.csv"

cat >"${OUT_ROOT}/run_manifest.txt" <<EOF
run_ts=${RUN_TS}
out_root=${OUT_ROOT}
quiet_targets=${QUIET_TARGETS}
quiet_candidate_count=${QUIET_CANDIDATE_COUNT}
quiet_max_samples=${QUIET_MAX_SAMPLES}
quiet_sample_seed=${QUIET_SAMPLE_SEED}
rq3_surface_workers=${RQ3_SURFACE_WORKERS}
svi_window_json=${SVI_WINDOW_JSON}
raw_window_json=${RAW_WINDOW_JSON}
svi_workbook=${SVI_WORKBOOK}
raw_workbook=${RAW_WORKBOOK}
gen_dir=${GEN_DIR}
log=${DRIVER_LOG}
status=${STATUS_FILE}
EOF

log_step "DONE OUT_ROOT=${OUT_ROOT}"
