#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-configs/surface_builder/svi/generate_surface-svi-excel.yaml}"
CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-48}"
MAX_TARGET_DATETIMES="${MAX_TARGET_DATETIMES:-0}"
MAX_FILES="${MAX_FILES:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
SAVE_PRECALIB_CSV="${SAVE_PRECALIB_CSV:-1}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-data/processed/svi-excel/${RUN_TS}}"
SURFACE_LOG="${OUTPUT_DIR}/surface-svi-excel.log"
PID_FILE="${PID_FILE:-${OUTPUT_DIR}/launcher.pid}"

mkdir -p "$OUTPUT_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "A launcher process is already recorded in $PID_FILE (pid=$existing_pid)." >&2
    exit 1
  fi
fi

if ! [[ "$CALIBRATION_WORKERS" =~ ^[0-9]+$ ]] || ! [[ "$MAX_TARGET_DATETIMES" =~ ^[0-9]+$ ]] || ! [[ "$MAX_FILES" =~ ^[0-9]+$ ]] || ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
  echo "CALIBRATION_WORKERS, MAX_TARGET_DATETIMES, MAX_FILES, and CHUNK_SIZE must be non-negative integers." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cmd=(
  python scripts/generate_surface/main.py generate_surface
  --device cpu
  --model svi
  --data_range excel
  --config "$CONFIG_PATH"
  --run-ts "$RUN_TS"
  --calibration-workers "$CALIBRATION_WORKERS"
)

if (( MAX_TARGET_DATETIMES > 0 )); then
  cmd+=(--max-target-datetimes "$MAX_TARGET_DATETIMES")
fi

if (( MAX_FILES > 0 )); then
  cmd+=(--max-files "$MAX_FILES")
fi

if (( CHUNK_SIZE > 0 )); then
  cmd+=(--chunk-size "$CHUNK_SIZE")
fi

if [[ "$SAVE_PRECALIB_CSV" == "1" ]]; then
  cmd+=(--save-precalib-csv)
else
  cmd+=(--no-save-precalib-csv)
fi

if (( $# > 0 )); then
  cmd+=("$@")
fi

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launch directory: $ROOT_DIR"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Run timestamp: $RUN_TS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Output directory: $OUTPUT_DIR"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Calibration workers: $CALIBRATION_WORKERS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Command: ${cmd[*]}"
} >>"$LAUNCH_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, command not started."
  echo "Command: ${cmd[*]}"
  exit 0
fi

nohup "${cmd[@]}" >>"$LAUNCH_LOG" 2>&1 &
launcher_pid=$!
echo "$launcher_pid" >"$PID_FILE"

echo "Started background job."
echo "PID: $launcher_pid"
echo "Run dir: $OUTPUT_DIR"
echo "Launcher log: $LAUNCH_LOG"
echo "PID file: $PID_FILE"
echo "Surface log: ${OUTPUT_DIR}/surface-svi-excel.log"
echo "Tail command: tail -f ${OUTPUT_DIR}/surface-svi-excel.log"
