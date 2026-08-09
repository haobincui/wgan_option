#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${FILM_WGAN_CONDA_ENV:-py312}"
PILOT_FOLD="${FILM_WGAN_PILOT_FOLD:-2023Q1}"
EXPERIMENT_ROOT="${1:-outputs/experiments/rq1_film_wgan_v2_pilot}"
CONFIG_PATH="configs/film_wgan/train_rq1_pair_textbase_v2_pilot.yaml"

cd "${REPO_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to start the v2 pilot from a dirty worktree." >&2
  exit 1
fi

if [[ ! -f "${EXPERIMENT_ROOT}/inputs/experiment_design.json" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/rq1_pair/rq1_pair_experiment.py prepare \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --seeds 42 202 404
fi

variants=(
  pair_pca_no_text_residual
  pair_pca_no_text_continued
  pair_pca_text_residual_pretrained
  pair_pca_shuffled_residual_pretrained
)

for seed in 42 202 404; do
  for variant in "${variants[@]}"; do
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/rq1_pair/rq1_pair_experiment.py train-matrix \
      --experiment-root "${EXPERIMENT_ROOT}" \
      --fold "${PILOT_FOLD}" \
      --seed "${seed}" \
      --variant "${variant}" \
      --resume
  done
done

echo "V2 validation pilot training finished: ${EXPERIMENT_ROOT}"
echo "No outer-test generation or results pipeline was run."
