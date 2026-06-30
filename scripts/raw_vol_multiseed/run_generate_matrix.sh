#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}"
SEEDS="${SEEDS_CSV:-42,101,202,303,404}"
ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"

export CUDA_VISIBLE_DEVICES

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/rq2_multiseed/rq2_multiseed.py collect \
    --exp-root "${EXP_ROOT}" \
    --seeds "${SEEDS}" \
    --data-path "${DATA_PATH}"
fi

SELECTED="${EXP_ROOT}/checkpoint_selection/selected_checkpoints.csv"
[[ -f "${SELECTED}" ]] || { echo "Selected checkpoint registry not found: ${SELECTED}" >&2; exit 1; }

"${PYTHON_BIN}" - "${SELECTED}" <<'PY' | while IFS=$'\t' read -r model seed run_dir checkpoint_path; do
import pandas as pd
import sys

selected = pd.read_csv(sys.argv[1])
for _, row in selected[selected["status"].eq("ok")].iterrows():
    print(f"{row['model']}\t{int(row['seed'])}\t{row['run_dir']}\t{row['checkpoint_path']}")
PY
  run_dir_abs="${REPO_ROOT}/${run_dir}"
  checkpoint_abs="${REPO_ROOT}/${checkpoint_path}"
  config_path="${run_dir_abs}/metrics/training_resolved_config.yaml"
  output_dir="${run_dir_abs}/after10_val_mae_all_json"
  sample_dir="${output_dir}/samples"
  log_dir="${EXP_ROOT}/logs/${model}/seed_${seed}"
  log_file="${log_dir}/generate_after10_val_mae_all_json.log"
  mkdir -p "${log_dir}"

  sample_count=0
  if [[ -d "${sample_dir}" ]]; then
    sample_count="$(find "${sample_dir}" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
  fi
  if [[ "${FORCE_GENERATE}" != "1" && "${sample_count}" -gt 0 && -f "${output_dir}/summary.csv" && -f "${output_dir}/run_metadata.json" && -f "${output_dir}/generate_resolved_config.yaml" ]]; then
    echo "Reusing existing generate-result output for ${model} seed ${seed}: ${output_dir}"
    continue
  fi

  echo "Generating raw-vol all-split JSON for ${model} seed ${seed}"
  conda run -n "${ENV_NAME}" python scripts/film_wgan/main.py generate-result \
    --config "${config_path}" \
    --checkpoint "${checkpoint_abs}" \
    --output-dir after10_val_mae_all_json \
    --split all \
    --selection-mode all \
    --selection-count 0 \
    --no-plot \
    >"${log_file}" 2>&1
done
