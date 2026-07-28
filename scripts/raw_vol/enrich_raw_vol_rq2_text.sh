#!/usr/bin/env bash
# Add RQ2 BoW and ChatGPT sentiment feature columns to a raw-vol workbook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-3}"
DATASET_DIR="${1:-${DATASET_DIR:-}}"
FEATURE_DIR="${FEATURE_DIR:-}"

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 <data/processed/raw-excel/run_dir>" >&2
  exit 1
fi

if [[ -z "${FEATURE_DIR}" ]]; then
  FEATURE_DIR="$(find data/processed/text_features/rq2 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
fi
if [[ -z "${FEATURE_DIR}" || ! -f "${FEATURE_DIR}/bow_features.xlsx" || ! -f "${FEATURE_DIR}/llm_sentiment_features.xlsx" ]]; then
  echo "Feature artifacts not found. Set FEATURE_DIR to an RQ2 text feature directory." >&2
  exit 1
fi

run_python() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n "${ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}

run_python scripts/rq2/enrich_merged_vol.py \
  --merged-vol "${DATASET_DIR}/merged_vol.xlsx" \
  --bow-features "${FEATURE_DIR}/bow_features.xlsx" \
  --sentiment-features "${FEATURE_DIR}/llm_sentiment_features.xlsx" \
  --output "${DATASET_DIR}/merged_vol_rq2_text.xlsx"

run_python scripts/raw_vol/raw_vol_pipeline.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --workbook "${DATASET_DIR}/merged_vol_rq2_text.xlsx" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"

echo "Raw-vol RQ2 workbook ready: ${DATASET_DIR}/merged_vol_rq2_text.xlsx"
