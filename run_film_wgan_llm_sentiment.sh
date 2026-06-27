#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_film_wgan_svi_excel.sh"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/film_wgan/train_lp_exp_F5.yaml}"
DATA_PATH="${DATA_PATH:-data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/film_wgan/llm_sentiment}"

ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_TS="${RUN_TS:-llm_sentiment_$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/film_wgan/${RUN_TS}}"

export ENV_NAME CUDA_VISIBLE_DEVICES RUN_TS WORK_DIR

if [[ ! -f "${BASE_RUNNER}" ]]; then
  echo "Base runner not found: ${BASE_RUNNER}" >&2
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  cat >&2 <<EOF
RQ2 enriched workbook not found: ${DATA_PATH}

Generate it first, for example:
  RQ2_TS=\$(date -u +%Y%m%d-%H%M%S)
  FEATURE_DIR=data/processed/text_features/rq2/\${RQ2_TS}

  python scripts/generate_rq2_text_features.py \\
    --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx \\
    --output-dir "\${FEATURE_DIR}" \\
    --text-column LP \\
    --target-dim 1024 \\
    --model gpt-5.5

  python scripts/rq2/enrich_merged_vol.py \\
    --merged-vol data/processed/svi-excel/20260410-174929/merged_vol.xlsx \\
    --bow-features "\${FEATURE_DIR}/bow_features.xlsx" \\
    --sentiment-features "\${FEATURE_DIR}/llm_sentiment_features.xlsx" \\
    --output ${DATA_PATH}
EOF
  exit 1
fi

# RQ2 sentiment baseline: same FiLM WGAN downstream, Sun-style ChatGPT text representation.
bash "${BASE_RUNNER}" "${TRAIN_CONFIG_PATH}" \
  --set "data_path=${DATA_PATH}" \
  --set "text_embedding_mode=llm_sentiment" \
  --set "normalize_text_embedding=true" \
  --set "output_root=${OUTPUT_ROOT}" \
  "$@"
