#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
SEEDS="${SEEDS_CSV:-42,101,202,303,404}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-0}"
TRAIN_COUNT="${TRAIN_COUNT:-0}"

"${PYTHON_BIN}" scripts/rq2_multiseed/rq2_multiseed.py build-comparison \
  --exp-root "${EXP_ROOT}" \
  --seeds "${SEEDS}" \
  --expected-sample-count "${EXPECTED_SAMPLE_COUNT}" \
  --train-count "${TRAIN_COUNT}"

cp "${EXP_ROOT}/final_tables/thesis_eval_main_metrics.csv" "${EXP_ROOT}/final_tables/raw_vol_eval_main_metrics_by_seed.csv"
cp "${EXP_ROOT}/comparisons/seed_level_tests.csv" "${EXP_ROOT}/final_tables/raw_vol_seed_level_tests.csv"
cp "${EXP_ROOT}/final_tables/thesis_text_vs_baselines.csv" "${EXP_ROOT}/final_tables/raw_vol_text_vs_baselines.csv"
cp "${EXP_ROOT}/final_tables/best_seed_pairwise_text_vs_baselines_eval_p_values.csv" "${EXP_ROOT}/final_tables/raw_vol_best_seed_eval_comparison.csv"
