#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/rq1_pair_text_raw_vol_continuation_shortatm21_15seed_${RUN_TS}}"
CONFIG="${CONFIG:-configs/film_wgan/train_rq1_pair_textbase_15seed.yaml}"
WORKBOOK="${WORKBOOK:-data/processed/raw-excel-session/rq123_cme_session_20260729-131219/merged_vol_rq2_text.xlsx}"
NEWS_WORKBOOK="${NEWS_WORKBOOK:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
ENV_NAME="${ENV_NAME:-py312}"
GPU_IDS="${GPU_IDS:-0}"
RUNS_PER_GPU="${RUNS_PER_GPU:-13}"
MIN_FREE_GB="${MIN_FREE_GB:-220}"

NORMALIZED_GPU_IDS="$(xargs <<<"${GPU_IDS}")"
if [[ "${NORMALIZED_GPU_IDS}" != "0" ]]; then
  echo "This launcher is restricted to physical GPU 0; received GPU_IDS=${GPU_IDS}." >&2
  echo "GPU 1 is reserved for another workload." >&2
  exit 1
fi
if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_GPU must be a positive integer." >&2
  exit 1
fi

GPU0_FREE_MB="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
ESTIMATED_RUN_MB="${ESTIMATED_RUN_MB:-1500}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-3000}"
REQUIRED_GPU_MB="$((RUNS_PER_GPU * ESTIMATED_RUN_MB + GPU_RESERVE_MB))"
if (( GPU0_FREE_MB < REQUIRED_GPU_MB )); then
  echo "Insufficient GPU 0 memory for ${RUNS_PER_GPU} concurrent runs: " \
       "free=${GPU0_FREE_MB} MiB, estimated_required=${REQUIRED_GPU_MB} MiB." >&2
  echo "Lower RUNS_PER_GPU and restart." >&2
  exit 1
fi

AVAILABLE_KB="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((MIN_FREE_GB * 1024 * 1024))"
if (( AVAILABLE_KB < REQUIRED_KB )); then
  AVAILABLE_GB="$((AVAILABLE_KB / 1024 / 1024))"
  echo "Insufficient free space: ${AVAILABLE_GB} GiB available; ${MIN_FREE_GB} GiB required." >&2
  echo "Free disk space or set EXPERIMENT_ROOT to a larger filesystem." >&2
  exit 1
fi

if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_pair/rq1_pair_experiment.py prepare \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --config "${CONFIG}" \
  --workbook "${WORKBOOK}" \
  --news-workbook "${NEWS_WORKBOOK}"

SEED_COUNT="$(python -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["seed_count"])
' "${EXPERIMENT_ROOT}/inputs/experiment_design.json")"
EXPECTED_RUNS="$(python -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["expected_training_runs"])
' "${EXPERIMENT_ROOT}/inputs/experiment_design.json")"
if [[ "${SEED_COUNT}" != "15" || "${EXPECTED_RUNS}" != "420" ]]; then
  echo "Unexpected experiment design: seeds=${SEED_COUNT}, runs=${EXPECTED_RUNS}." >&2
  exit 1
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
GPU_IDS="${GPU_IDS}" \
RUNS_PER_GPU="${RUNS_PER_GPU}" \
ENV_NAME="${ENV_NAME}" \
bash scripts/rq1_pair/start_training_matrix_background.sh

echo "seed_count=${SEED_COUNT}"
echo "expected_training_runs=${EXPECTED_RUNS}"
echo "checkpoint_retention=best_and_final"
echo "gpu_ids=${GPU_IDS}"
echo "runs_per_gpu=${RUNS_PER_GPU}"
echo "gpu0_free_at_launch_mb=${GPU0_FREE_MB}"
