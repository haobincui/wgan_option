#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

activate_py312() {
  if source activate py312 2>/dev/null; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available, cannot activate py312." >&2
    return 1
  fi

  local conda_base
  conda_base="$(conda info --base)"

  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  source activate py312
}

activate_py312

export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python -m unittest discover -s tests -p 'test*.py' "$@"
