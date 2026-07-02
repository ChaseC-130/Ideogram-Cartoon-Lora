#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOSR_ROOT="${PROJECT_ROOT}/external/VOSR"
PYTHON="${VOSR_ROOT}/.venv/bin/python"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_vosr.sh INPUT_IMAGE_OR_DIR [OUTPUT_DIR] [extra VOSR args...]"
  echo
  echo "Environment overrides:"
  echo "  VOSR_UPSCALE=1"
  echo "  VOSR_TILE_SIZE=512"
  echo "  VOSR_TILE_OVERLAP=32"
  echo "  VOSR_ALIGN_METHOD=wavelet"
  echo "  VOSR_SEED=42"
  echo "  VOSR_DEVICE=mps"
  exit 2
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing VOSR Python environment at ${PYTHON}" >&2
  echo "Run the VOSR setup in external/VOSR first." >&2
  exit 1
fi

abspath() {
  case "$1" in
    /*) printf "%s\n" "$1" ;;
    *) printf "%s\n" "${PROJECT_ROOT}/$1" ;;
  esac
}

INPUT_PATH="$(abspath "$1")"
shift

OUTPUT_DIR="${PROJECT_ROOT}/output/vosr_assets"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_DIR="$(abspath "$1")"
  shift
fi

CHECKPOINT="${VOSR_CHECKPOINT:-${VOSR_ROOT}/preset/ckpts/VOSR_0.5B_os}"
UPSCALE="${VOSR_UPSCALE:-1}"
TILE_SIZE="${VOSR_TILE_SIZE:-512}"
TILE_OVERLAP="${VOSR_TILE_OVERLAP:-32}"
ALIGN_METHOD="${VOSR_ALIGN_METHOD:-wavelet}"
SEED="${VOSR_SEED:-42}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export VOSR_DISABLE_COMPILE="${VOSR_DISABLE_COMPILE:-1}"

cd "${VOSR_ROOT}"
exec "${PYTHON}" inference_vosr_onestep.py \
  -c "${CHECKPOINT}" \
  -i "${INPUT_PATH}" \
  -o "${OUTPUT_DIR}" \
  -u "${UPSCALE}" \
  --tile_size "${TILE_SIZE}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --align_method "${ALIGN_METHOD}" \
  --seed "${SEED}" \
  "$@"
