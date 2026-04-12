#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-py312}"
TRAIN_CONFIG_TEMPLATE="${TRAIN_CONFIG_TEMPLATE:-configs/volgan/train_lp.yaml}"
DATA_PATH_OVERRIDE="${DATA_PATH_OVERRIDE:-data/processed/svi-excel/20260410-174929/merged_vol.xlsx}"

MC_SAMPLES="${MC_SAMPLES:-64}"
REWEIGHT_BETA_MODE="${REWEIGHT_BETA_MODE:-fixed}"
REWEIGHT_BETA="${REWEIGHT_BETA:-25.0}"
SPLIT="${SPLIT:-val}"
SELECTION_MODE="${SELECTION_MODE:-all}"
SELECTION_COUNT="${SELECTION_COUNT:-0}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/volgan/${RUN_TS}}"
TRAIN_CONFIG_PATH="${WORK_DIR}/train_volgan.yaml"
GENERATE_CONFIG_PATH="${WORK_DIR}/generate_result_volgan.yaml"
LOG_PATH="${WORK_DIR}/run.log"

mkdir -p "$WORK_DIR"

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    return
  fi
  # shellcheck disable=SC1091
  source activate "$ENV_NAME"
}

load_train_template_vars() {
  eval "$(
    python - "$TRAIN_CONFIG_TEMPLATE" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

def emit(name, value):
    if isinstance(value, bool):
        serialized = "true" if value else "false"
    elif value is None:
        serialized = ""
    else:
        serialized = str(value)
    print(f"{name}={shlex.quote(serialized)}")

emit("DATA_PATH", payload.get("data_path", ""))
emit("SHEET_NAME", payload.get("sheet_name", "gan_input_ready"))
emit("TEXT_MODE", payload.get("text_embedding_mode", "hd"))
emit("TRAIN_RATIO", payload.get("train_ratio", 0.8))
emit("SEED", payload.get("seed", 42))
emit("CUDA_FLAG", payload.get("cuda", True))
emit("TRAIN_OUTPUT_ROOT_TEMPLATE", payload.get("output_root", ""))
PY
  )"
}

write_train_config() {
  python - "$TRAIN_CONFIG_TEMPLATE" "$TRAIN_CONFIG_PATH" "$DATA_PATH" "$TRAIN_OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
data_path = sys.argv[3]
output_root = sys.argv[4]

payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
payload["data_path"] = data_path
payload["output_root"] = output_root
output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
PY
}

write_generate_config() {
  local checkpoint_path="$1"
  cat >"$GENERATE_CONFIG_PATH" <<EOF
data_path: ${DATA_PATH}
sheet_name: ${SHEET_NAME}
text_embedding_mode: ${TEXT_MODE}
train_ratio: ${TRAIN_RATIO}

checkpoint_path: ${checkpoint_path}
seed: ${SEED}
cuda: ${CUDA_FLAG}

mc_samples: ${MC_SAMPLES}
reweight_beta_mode: ${REWEIGHT_BETA_MODE}
reweight_beta: ${REWEIGHT_BETA}
quantiles:
  - 0.05
  - 0.5
  - 0.95

split: ${SPLIT}
selection_mode: ${SELECTION_MODE}
selection_count: ${SELECTION_COUNT}
output_dir: ${GENERATE_OUTPUT_ROOT}
EOF
}

find_latest_run_dir() {
  local root="$1"
  python - "$root" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pattern = re.compile(r"\d{8}_\d{6}$")
run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and pattern.fullmatch(p.name)])
if not run_dirs:
    raise SystemExit(1)
print(run_dirs[-1])
PY
}

activate_env
python -V >/dev/null
load_train_template_vars

if [[ -n "$DATA_PATH_OVERRIDE" ]]; then
  DATA_PATH="$DATA_PATH_OVERRIDE"
fi

DATASET_FAMILY="${DATASET_FAMILY:-$(basename "$(dirname "$(dirname "$DATA_PATH")")")}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${TRAIN_OUTPUT_ROOT_TEMPLATE:-outputs/training/volgan/${DATASET_FAMILY}}}"
if [[ -z "$TRAIN_OUTPUT_ROOT" ]]; then
  TRAIN_OUTPUT_ROOT="outputs/training/volgan/${DATASET_FAMILY}"
fi
GENERATE_OUTPUT_ROOT="${GENERATE_OUTPUT_ROOT:-outputs/generate_result/volgan/${DATASET_FAMILY}}"

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Root dir: ${ROOT_DIR}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Train config template: ${TRAIN_CONFIG_TEMPLATE}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Data path: ${DATA_PATH}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Dataset family: ${DATASET_FAMILY}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Train output root: ${TRAIN_OUTPUT_ROOT}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generate output root: ${GENERATE_OUTPUT_ROOT}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Text mode: ${TEXT_MODE}"
} | tee "$LOG_PATH"

python -V | tee -a "$LOG_PATH"

write_train_config
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Training config: ${TRAIN_CONFIG_PATH}" | tee -a "$LOG_PATH"
python scripts/volgan/main.py train --config "$TRAIN_CONFIG_PATH" | tee -a "$LOG_PATH"

LATEST_TRAIN_RUN_DIR="$(find_latest_run_dir "$TRAIN_OUTPUT_ROOT")"
BEST_CHECKPOINT_PATH="${LATEST_TRAIN_RUN_DIR}/checkpoints/volgan_best.pt"
if [[ ! -f "$BEST_CHECKPOINT_PATH" ]]; then
  echo "Best checkpoint not found: ${BEST_CHECKPOINT_PATH}" >&2
  exit 1
fi

write_generate_config "$BEST_CHECKPOINT_PATH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generate-result config: ${GENERATE_CONFIG_PATH}" | tee -a "$LOG_PATH"
python scripts/volgan/main.py generate-result --config "$GENERATE_CONFIG_PATH" | tee -a "$LOG_PATH"

LATEST_GENERATE_RUN_DIR="$(find_latest_run_dir "$GENERATE_OUTPUT_ROOT")"

echo "=========================================="
echo "VolGAN run complete"
echo "=========================================="
echo "Train run dir     : ${LATEST_TRAIN_RUN_DIR}"
echo "Best checkpoint   : ${BEST_CHECKPOINT_PATH}"
echo "Generate run dir  : ${LATEST_GENERATE_RUN_DIR}"
echo "Work dir          : ${WORK_DIR}"
echo "Log file          : ${LOG_PATH}"
echo "=========================================="
