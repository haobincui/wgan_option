#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
PIPELINE_ROOT="${PIPELINE_ROOT:-outputs/experiments/rq123_corrected_raw_${RUN_TS}}"
DATASET_RUN_TS="${DATASET_RUN_TS:-rq123_corrected_lag0_${RUN_TS}}"
DATASET_DIR="${DATASET_DIR:-data/processed/raw-excel/${DATASET_RUN_TS}}"
RQ1_ROOT="${RQ1_ROOT:-${PIPELINE_ROOT}/rq1}"
RQ2_ROOT="${RQ2_ROOT:-${PIPELINE_ROOT}/rq2}"
RQ3_ROOT="${RQ3_ROOT:-${PIPELINE_ROOT}/rq3}"

SOURCE_OPTION_DATA_GLOB="${OPTION_DATA_GLOB:-data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz}"
SOURCE_NEWS_WORKBOOK="${NEWS_WORKBOOK:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
SOURCE_FEATURE_ROOT="${FEATURE_ROOT:-data/processed/text_features/rq2/20260625-075653}"
SOURCE_RQ1_CONFIG="${RQ1_CONFIG:-configs/film_wgan/train_rq1_pair_textbase.yaml}"
SOURCE_RQ2_CONFIG="${RQ2_CONFIG:-configs/film_wgan/train_rq2_pair_textbase.yaml}"
SOURCE_RQ3_CONFIG="${RQ3_CONFIG:-configs/rq3/scheduled_news_regime_raw_vol.yaml}"
SOURCE_RAW_CONFIG="${RAW_CONFIG:-configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml}"
SOURCE_RATE_CURVE="${RATE_CURVE_PATH:-data/reference/us_treasury_par_yield_curve_2022_2023.csv}"
SOURCE_EXPIRY_REFERENCE="${EXPIRY_REFERENCE_PATH:-data/reference/cme_ty_monthly_expirations_2022_2024.csv}"
SOURCE_EVENT_CALENDAR="${EVENT_CALENDAR_PATH:-data/reference/rq3_scheduled_macro_events_2023.csv}"

SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}"
OPTION_FILTER_MODE="${OPTION_FILTER_MODE:-otm_preferred_itm_fallback}"
MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE:-0.05}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}"
BUILD_DATASET="${BUILD_DATASET:-1}"
REFRESH_MARKET_REFERENCES="${REFRESH_MARKET_REFERENCES:-0}"
PACKAGE_EXPERIMENT="${PACKAGE_EXPERIMENT:-0}"

PYTHON=(conda run --no-capture-output -n "${ENV_NAME}" python)
STATUS_TOOL=("${PYTHON[@]}" scripts/rq123/corrected_pipeline.py)
STAGE_DIR="${PIPELINE_ROOT}/registry/stages"
DRIVER_LOG_DIR="${PIPELINE_ROOT}/logs/driver"
mkdir -p "${STAGE_DIR}" "${DRIVER_LOG_DIR}" "${PIPELINE_ROOT}/inputs"

write_status() {
  local status="$1"
  local phase="$2"
  local message="${3:-}"
  "${STATUS_TOOL[@]}" write-status \
    --pipeline-root "${PIPELINE_ROOT}" \
    --status "${status}" \
    --phase "${phase}" \
    --message "${message}"
}

mark_done() {
  local stage="$1"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${STAGE_DIR}/${stage}.done"
}

is_done() {
  [[ -f "${STAGE_DIR}/$1.done" ]]
}

fail_pipeline() {
  local exit_code=$?
  local line="${BASH_LINENO[0]:-unknown}"
  write_status failed failed "exit_code=${exit_code}; line=${line}" || true
  exit "${exit_code}"
}
trap fail_pipeline ERR

write_status running initialize "Corrected raw-vol RQ1-RQ3 pipeline"
{
  printf 'run_ts=%s\n' "${RUN_TS}"
  printf 'pipeline_root=%s\n' "${PIPELINE_ROOT}"
  printf 'dataset_dir=%s\n' "${DATASET_DIR}"
  printf 'rq1_root=%s\n' "${RQ1_ROOT}"
  printf 'rq2_root=%s\n' "${RQ2_ROOT}"
  printf 'rq3_root=%s\n' "${RQ3_ROOT}"
  printf 'source_timezone=%s\n' "${SOURCE_TIMEZONE}"
  printf 'publication_availability_lag_minutes=%s\n' \
    "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"
  printf 'option_filter_mode=%s\n' "${OPTION_FILTER_MODE}"
  printf 'max_itm_moneyness_distance=%s\n' \
    "${MAX_ITM_MONEYNESS_DISTANCE}"
  printf 'gpu_ids=%s\n' "${GPU_IDS}"
  printf 'runs_per_gpu=%s\n' "${RUNS_PER_GPU}"
} >"${PIPELINE_ROOT}/inputs/run_parameters.env"
{
  git rev-parse HEAD
  git status --short --branch
} >"${PIPELINE_ROOT}/inputs/git_state.txt"
git diff --binary >"${PIPELINE_ROOT}/inputs/uncommitted_changes.patch"
git ls-files --others --exclude-standard \
  >"${PIPELINE_ROOT}/inputs/untracked_files.txt"

if ! is_done market_references; then
  write_status running market_references "Validate frozen rates and CME expiry references"
  if [[ "${REFRESH_MARKET_REFERENCES}" == "1" ]]; then
    "${PYTHON[@]}" scripts/raw_vol/build_reference_data.py
  fi
  test -f data/reference/rq123_market_reference_manifest.json
  test -f data/reference/us_treasury_par_yield_curve_2022_2023.csv
  test -f data/reference/cme_ty_monthly_expirations_2022_2024.csv
  mark_done market_references
fi

if ! is_done source_snapshot; then
  write_status running source_snapshot "Hard-link/copy and hash all external inputs"
  "${STATUS_TOOL[@]}" snapshot-inputs \
    --pipeline-root "${PIPELINE_ROOT}" \
    --input "raw_option_data=${SOURCE_OPTION_DATA_GLOB}" \
    --input "news_workbook=${SOURCE_NEWS_WORKBOOK}" \
    --input "text_features=${SOURCE_FEATURE_ROOT}" \
    --input "configs=${SOURCE_RQ1_CONFIG}" \
    --input "configs=${SOURCE_RQ2_CONFIG}" \
    --input "configs=${SOURCE_RQ3_CONFIG}" \
    --input "configs=${SOURCE_RAW_CONFIG}" \
    --input "market_references=${SOURCE_RATE_CURVE}" \
    --input "market_references=${SOURCE_EXPIRY_REFERENCE}" \
    --input "market_references=data/reference/rq123_market_reference_manifest.json" \
    --input "event_calendar=${SOURCE_EVENT_CALENDAR}"
  mark_done source_snapshot
fi

OPTION_DATA_GLOB="${PIPELINE_ROOT}/inputs/source_snapshot/raw_option_data/0#TY+_202[23]-*.csv.gz"
NEWS_WORKBOOK="${PIPELINE_ROOT}/inputs/source_snapshot/news_workbook/$(basename "${SOURCE_NEWS_WORKBOOK}")"
FEATURE_ROOT="${PIPELINE_ROOT}/inputs/source_snapshot/text_features"
RQ1_CONFIG="${PIPELINE_ROOT}/inputs/source_snapshot/configs/$(basename "${SOURCE_RQ1_CONFIG}")"
RQ2_CONFIG="${PIPELINE_ROOT}/inputs/source_snapshot/configs/$(basename "${SOURCE_RQ2_CONFIG}")"
RQ3_CONFIG="${PIPELINE_ROOT}/inputs/source_snapshot/configs/$(basename "${SOURCE_RQ3_CONFIG}")"
RAW_CONFIG="${PIPELINE_ROOT}/inputs/source_snapshot/configs/$(basename "${SOURCE_RAW_CONFIG}")"
RATE_CURVE_PATH="${PIPELINE_ROOT}/inputs/source_snapshot/market_references/$(basename "${SOURCE_RATE_CURVE}")"
EVENT_CALENDAR_PATH="${PIPELINE_ROOT}/inputs/source_snapshot/event_calendar/$(basename "${SOURCE_EVENT_CALENDAR}")"

for feature_name in \
  bow_features.xlsx \
  bow_manifest.json \
  bow_vocabulary.json \
  llm_sentiment_features.xlsx \
  llm_sentiment_manifest.json \
  openai_sentiment_cache.jsonl
do
  if [[ ! -f "${FEATURE_ROOT}/${feature_name}" ]]; then
    write_status failed input_preflight \
      "Missing frozen text feature artifact: ${FEATURE_ROOT}/${feature_name}"
    exit 1
  fi
done

if ! is_done corrected_dataset; then
  write_status running corrected_dataset "Build and audit corrected raw-vol labels"
  if [[ "${BUILD_DATASET}" == "1" && ! -f "${DATASET_DIR}/merged_vol.xlsx" ]]; then
    ENV_NAME="${ENV_NAME}" \
    CONFIG_PATH="${RAW_CONFIG}" \
    OPTION_DATA_GLOB="${OPTION_DATA_GLOB}" \
    TARGET_XLSX="${NEWS_WORKBOOK}" \
    RATE_CURVE_PATH="${RATE_CURVE_PATH}" \
    SOURCE_TIMEZONE="${SOURCE_TIMEZONE}" \
    PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
    RUN_TS="${DATASET_RUN_TS}" \
    WINDOW_MINUTES="${WINDOW_MINUTES}" \
    MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY}" \
    OPTION_FILTER_MODE="${OPTION_FILTER_MODE}" \
    MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE}" \
    MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS}" \
    WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS}" \
      bash scripts/raw_vol/prepare_raw_vol_dataset.sh
  fi
  test -f "${DATASET_DIR}/merged_vol.xlsx"
  if [[ ! -f "${DATASET_DIR}/merged_vol_rq2_text.xlsx" ]]; then
    ENV_NAME="${ENV_NAME}" \
    FEATURE_DIR="${FEATURE_ROOT}" \
    SOURCE_TIMEZONE="${SOURCE_TIMEZONE}" \
    WINDOW_MINUTES="${WINDOW_MINUTES}" \
    MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY}" \
    PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
      bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh "${DATASET_DIR}"
  fi
  "${PYTHON[@]}" scripts/raw_vol/raw_vol_pipeline.py validate \
    --dataset-dir "${DATASET_DIR}" \
    --workbook "${DATASET_DIR}/merged_vol_rq2_text.xlsx" \
    --window-minutes "${WINDOW_MINUTES}" \
    --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
    --source-timezone "${SOURCE_TIMEZONE}" \
    --publication-availability-lag-minutes "${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
    --min-usable-pairs "${MIN_USABLE_PAIRS}" \
    --warn-usable-pairs "${WARN_USABLE_PAIRS}"
  mark_done corrected_dataset
fi

if ! is_done text_lineage; then
  write_status running text_lineage "Freeze LP lineage and duplicate audit"
  "${PYTHON[@]}" scripts/rq123/audit_text_lineage.py \
    --news-workbook "${NEWS_WORKBOOK}" \
    --output-dir "${PIPELINE_ROOT}/inputs/text_lineage"
  mark_done text_lineage
fi

if ! is_done sentiment_audit; then
  write_status running sentiment_audit "Freeze manual and repeat-scoring audit templates"
  "${PYTHON[@]}" scripts/rq123/audit_sentiment_scores.py prepare \
    --news-workbook "${NEWS_WORKBOOK}" \
    --sentiment-features "${FEATURE_ROOT}/llm_sentiment_features.xlsx" \
    --sentiment-manifest "${FEATURE_ROOT}/llm_sentiment_manifest.json" \
    --sentiment-cache "${FEATURE_ROOT}/openai_sentiment_cache.jsonl" \
    --output-dir "${PIPELINE_ROOT}/inputs/sentiment_audit" \
    --seed "${BOOTSTRAP_SEED}"
  mark_done sentiment_audit
fi

if ! is_done rq1_prepare; then
  write_status running rq1_prepare "Prepare rolling RQ1 support grids and transforms"
  "${PYTHON[@]}" scripts/rq1_pair/rq1_pair_experiment.py prepare \
    --experiment-root "${RQ1_ROOT}" \
    --config "${RQ1_CONFIG}" \
    --workbook "${DATASET_DIR}/merged_vol_rq2_text.xlsx" \
    --news-workbook "${NEWS_WORKBOOK}" \
    --reuse
  mark_done rq1_prepare
fi

if ! is_done rq1_training; then
  write_status running rq1_training "Run paired Stage-A/Stage-B RQ1 matrix"
  EXPERIMENT_ROOT="${RQ1_ROOT}" \
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
  ENV_NAME="${ENV_NAME}" \
    bash scripts/rq1_pair/run_training_matrix_parallel.sh
  mark_done rq1_training
fi

if ! is_done rq1_results; then
  write_status running rq1_results "Freeze checkpoints and generate RQ1 OOS results"
  "${PYTHON[@]}" scripts/rq1_pair/rq1_pair_experiment.py results-pipeline \
    --experiment-root "${RQ1_ROOT}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
    --bootstrap-seed "${BOOTSTRAP_SEED}"
  mark_done rq1_results
fi

if ! is_done rq2_prepare; then
  write_status running rq2_prepare "Import RQ1 parents and build train-only RQ2 representations"
  "${PYTHON[@]}" scripts/rq2_pair/rq2_pair_experiment.py prepare \
    --experiment-root "${RQ2_ROOT}" \
    --config "${RQ2_CONFIG}" \
    --source-rq1 "${RQ1_ROOT}" \
    --feature-root "${FEATURE_ROOT}" \
    --reuse
  mark_done rq2_prepare
fi

if ! is_done rq2_training; then
  write_status running rq2_training "Run BoW and ChatGPT-sentiment continuation branches"
  EXPERIMENT_ROOT="${RQ2_ROOT}" \
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
  ENV_NAME="${ENV_NAME}" \
    bash scripts/rq2_pair/run_training_matrix_parallel.sh
  mark_done rq2_training
fi

if ! is_done rq2_results; then
  write_status running rq2_results "Freeze checkpoints and generate RQ2 OOS results"
  "${PYTHON[@]}" scripts/rq2_pair/rq2_pair_experiment.py results-pipeline \
    --experiment-root "${RQ2_ROOT}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
    --bootstrap-seed "${BOOTSTRAP_SEED}"
  mark_done rq2_results
fi

if ! is_done rq3_results; then
  write_status running rq3_results "Run frozen all-OOS conditional predictive robustness"
  if [[ -d "${RQ3_ROOT}" && -n "$(find "${RQ3_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ "$("${PYTHON[@]}" -c \
      "import json; from pathlib import Path; p=Path('${RQ3_ROOT}/validation_summary.json'); print(json.loads(p.read_text()).get('status','') if p.is_file() else '')")" != "ok" ]]; then
      archive="${PIPELINE_ROOT}/archive/rq3_incomplete_$(date -u +%Y%m%d-%H%M%S)"
      mkdir -p "$(dirname "${archive}")"
      mv "${RQ3_ROOT}" "${archive}"
    fi
  fi
  if [[ ! -f "${RQ3_ROOT}/validation_summary.json" ]]; then
    RQ1_EXPERIMENT="${RQ1_ROOT}" \
    RQ2_EXPERIMENT="${RQ2_ROOT}" \
    OUTPUT_DIR="${RQ3_ROOT}" \
    CONFIG_PATH="${RQ3_CONFIG}" \
    EVENT_CALENDAR_PATH="${EVENT_CALENDAR_PATH}" \
    ENV_NAME="${ENV_NAME}" \
      bash scripts/rq3/run_scheduled_news_regime.sh
  fi
  mark_done rq3_results
fi

write_status running final_validation "Cross-check RQ1/RQ2 support and all result archives"
"${STATUS_TOOL[@]}" validate-final \
  --pipeline-root "${PIPELINE_ROOT}" \
  --dataset-dir "${DATASET_DIR}" \
  --rq1-experiment "${RQ1_ROOT}" \
  --rq2-experiment "${RQ2_ROOT}" \
  --rq3-experiment "${RQ3_ROOT}"
"${STATUS_TOOL[@]}" build-manifest --pipeline-root "${PIPELINE_ROOT}"
mark_done final_validation

if [[ "${PACKAGE_EXPERIMENT}" == "1" ]]; then
  write_status running package "Package corrected experiment"
  "${STATUS_TOOL[@]}" package --pipeline-root "${PIPELINE_ROOT}"
  mark_done package
fi

write_status completed completed "Corrected raw-vol RQ1-RQ3 pipeline completed"
echo "pipeline_root=${PIPELINE_ROOT}"
echo "validation=${PIPELINE_ROOT}/validation_summary.json"
echo "rq1_primary=${RQ1_ROOT}/final_tables/development_rq1_primary_controlled_incremental_text.csv"
echo "rq2_primary=${RQ2_ROOT}/final_tables/development_rq2_primary_lp_vs_baselines.csv"
echo "rq3_primary=${RQ3_ROOT}/final_tables/development_rq3_primary_conditional_robustness.csv"
