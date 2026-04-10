#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_merge_vol.sh <input_dir> [extra args...]
  bash run_merge_vol.sh --input-dir <input_dir> [extra args...]

Examples:
  bash run_merge_vol.sh data/processed/svi-excel/20260410-174929
  bash run_merge_vol.sh data/processed/svi-excel/20260410-174929 --offset-minutes 10
  bash run_merge_vol.sh --input-dir data/processed/svi-excel/20260410-174929 \
    --news-xlsx data/raw/text_embedding/news_with_openai_embeddings_large.xlsx

This wrapper launches `scripts/merge_file/merge_vol.py` in the background and
writes logs under `logs/`.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit $([[ $# -eq 0 ]] && echo 1 || echo 0)
fi

FORWARDED_ARGS=()
INPUT_DIR=""

if [[ "${1:-}" != -* ]]; then
  INPUT_DIR="$1"
  shift
  FORWARDED_ARGS+=(--input-dir "${INPUT_DIR}")
fi

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--input-dir" ]]; then
    if [[ $# -lt 2 ]]; then
      echo "Error: --input-dir requires a value." >&2
      exit 1
    fi
    INPUT_DIR="$2"
  fi
  FORWARDED_ARGS+=("$1")
  shift
done

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Error: input_dir is required." >&2
  echo "" >&2
  usage >&2
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/merge_vol_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/merge_vol.pid"

nohup python scripts/merge_file/merge_vol.py \
  "${FORWARDED_ARGS[@]}" >"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "=========================================="
echo "  merge_vol job launched"
echo "=========================================="
echo "  Input dir   : ${INPUT_DIR}"
echo "  Output file : ${INPUT_DIR}/merged_vol.xlsx"
echo "  PID         : ${PID}"
echo "  PID file    : ${PID_FILE}"
echo "  Log file    : ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Follow logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check status:"
echo "  kill -0 ${PID} 2>/dev/null && echo running || echo stopped"
echo ""
echo "Stop job:"
echo "  kill ${PID}"
