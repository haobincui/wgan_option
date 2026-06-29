#!/usr/bin/env bash
# Launch the RQ2 n-gram frequency BoW FiLM WGAN baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/film_wgan/train_lp_exp_F5.yaml}"
DATA_PATH="${DATA_PATH:-data/processed/svi-excel/20260410-174929/merged_vol_rq2_text.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/film_wgan/bow}"

ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_TS="${RUN_TS:-bow_$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/film_wgan/${RUN_TS}}"
LOG_FILE="${WORK_DIR}/run.log"
PID_FILE="${WORK_DIR}/run.pid"

export ENV_NAME CUDA_VISIBLE_DEVICES RUN_TS WORK_DIR

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    return
  fi
  # shellcheck disable=SC1091
  source activate "${ENV_NAME}"
}

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG_PATH}" >&2
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

mkdir -p "${WORK_DIR}"

# RQ2 BoW baseline: same FiLM WGAN downstream, n-gram frequency text representation.
activate_env
python -V >/dev/null

nohup python scripts/film_wgan/main.py train --config "${TRAIN_CONFIG_PATH}" \
  --set "data_path=${DATA_PATH}" \
  --set "text_embedding_mode=bow" \
  --set "normalize_text_embedding=true" \
  --set "output_root=${OUTPUT_ROOT}" \
  "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "=========================================="
echo "  RQ2 BoW FiLM WGAN job launched"
echo "=========================================="
echo "  Train config : ${TRAIN_CONFIG_PATH}"
echo "  Data path    : ${DATA_PATH}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  CUDA device  : ${CUDA_VISIBLE_DEVICES}"
echo "  PID          : ${PID}"
echo "  PID file     : ${PID_FILE}"
echo "  Log file     : ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Follow logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check status:"
echo "  kill -0 ${PID} 2>/dev/null && echo running || echo stopped"
echo ""
echo "Stop training:"
echo "  kill ${PID}"
