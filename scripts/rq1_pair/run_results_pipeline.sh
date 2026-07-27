#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  EXPERIMENT_ROOT="$(
    find outputs/experiments -maxdepth 1 -type d \
      -name 'rq1_pair_text_raw_vol_rolling_*' -printf '%p\n' |
      sort |
      tail -n 1
  )"
fi
if [[ -z "${EXPERIMENT_ROOT}" || ! -d "${EXPERIMENT_ROOT}" ]]; then
  echo "No completed raw-vol RQ1 pair experiment found." >&2
  exit 1
fi

BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260722}"
LOCK_PATH="${EXPERIMENT_ROOT}/registry/results_pipeline.lock"
mkdir -p "$(dirname "${LOCK_PATH}")"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "Another RQ1 result pipeline already holds ${LOCK_PATH}." >&2
  exit 1
fi

echo "experiment_root=${EXPERIMENT_ROOT}"
echo "bootstrap_iterations=${BOOTSTRAP_ITERATIONS}"
echo "bootstrap_seed=${BOOTSTRAP_SEED}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-not_set}"

exec conda run --no-capture-output -n py312 \
  python scripts/rq1_pair/rq1_pair_experiment.py results-pipeline \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"
