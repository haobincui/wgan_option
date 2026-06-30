#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
CONFIG_PATH="${CONFIG_PATH:-configs/film_wgan/train_raw_vol_textbase.yaml}"
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}"
SEEDS="${SEEDS:-42 101 202 303 404}"
MODELS_TO_RUN="${MODELS_TO_RUN:-text no_text bow llm_sentiment}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ENV_NAME="${ENV_NAME:-py312}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

export CUDA_VISIBLE_DEVICES ENV_NAME

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  EXP_ROOT="${EXP_ROOT}" CONFIG_PATH="${CONFIG_PATH}" DATA_PATH="${DATA_PATH}" \
    "${SCRIPT_DIR}/prepare_experiment.sh"
fi

mkdir -p "${EXP_ROOT}/registry"
LAUNCH_REGISTRY="${EXP_ROOT}/registry/launch_registry.csv"
if [[ ! -f "${LAUNCH_REGISTRY}" ]]; then
  printf 'launch_ts_utc,model,seed,text_embedding_mode,normalize_text_embedding,output_root,work_dir,pid_file,log_file,pid,status\n' >"${LAUNCH_REGISTRY}"
fi

model_mode() {
  case "$1" in
    text) printf 'lp true' ;;
    no_text) printf 'none false' ;;
    bow) printf 'bow true' ;;
    llm_sentiment) printf 'llm_sentiment true' ;;
    *) echo "Unknown model: $1" >&2; return 1 ;;
  esac
}

for model in ${MODELS_TO_RUN}; do
  read -r text_mode normalize_text <<<"$(model_mode "${model}")"
  for seed in ${SEEDS}; do
    launch_ts="$(date -u +%Y%m%d-%H%M%S)"
    run_id="raw_vol_${model}_seed_${seed}_${launch_ts}"
    output_root="${EXP_ROOT}/training_runs/${model}/seed_${seed}"
    work_dir="${EXP_ROOT}/logs/${model}/seed_${seed}/${run_id}"
    mkdir -p "${output_root}" "${work_dir}"

    echo "Launching raw-vol ${model} seed ${seed}"
    RUN_TS="${run_id}" WORK_DIR="${work_dir}" ./run_film_wgan_svi_excel.sh "${CONFIG_PATH}" \
      --train-only \
      --set "data_path=${DATA_PATH}" \
      --set "text_embedding_mode=${text_mode}" \
      --set "normalize_text_embedding=${normalize_text}" \
      --set "seed=${seed}" \
      --set "output_root=${output_root}"

    pid_file="${work_dir}/run.pid"
    log_file="${work_dir}/run.log"
    pid="$(cat "${pid_file}")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${launch_ts}" "${model}" "${seed}" "${text_mode}" "${normalize_text}" \
      "${output_root}" "${work_dir}" "${pid_file}" "${log_file}" "${pid}" "launched" \
      >>"${LAUNCH_REGISTRY}"

    if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
      while kill -0 "${pid}" 2>/dev/null; do sleep "${POLL_SECONDS}"; done
      latest_run="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
      if [[ -z "${latest_run}" || ! -f "${latest_run}/metrics/training_resolved_config.yaml" ]]; then
        echo "Training output missing for ${model} seed ${seed}. See ${log_file}" >&2
        tail -n 80 "${log_file}" >&2 || true
        exit 1
      fi
    fi
  done
done

echo "Launch registry written to ${LAUNCH_REGISTRY}"
