#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_VOSR="${PROJECT_ROOT}/scripts/run_vosr.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_vosr_recursive.sh INPUT_DIR [OUTPUT_DIR] [extra VOSR args...]"
  echo
  echo "Example:"
  echo "  scripts/run_vosr_recursive.sh output/img2img_assets_refined output/vosr_assets"
  exit 2
fi

abspath() {
  case "$1" in
    /*) printf "%s\n" "$1" ;;
    *) printf "%s\n" "${PROJECT_ROOT}/$1" ;;
  esac
}

INPUT_ROOT="$(abspath "$1")"
shift

OUTPUT_ROOT="${PROJECT_ROOT}/output/vosr_assets"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_ROOT="$(abspath "$1")"
  shift
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input directory not found: ${INPUT_ROOT}" >&2
  exit 1
fi

find "${INPUT_ROOT}" -type f \( \
  -iname '*.png' -o \
  -iname '*.jpg' -o \
  -iname '*.jpeg' -o \
  -iname '*.bmp' -o \
  -iname '*.tif' -o \
  -iname '*.tiff' -o \
  -iname '*.webp' \
\) -exec dirname {} \; | sort -u | while IFS= read -r input_dir; do
  rel="${input_dir#${INPUT_ROOT}}"
  rel="${rel#/}"
  if [[ -z "${rel}" ]]; then
    out_dir="${OUTPUT_ROOT}"
  else
    out_dir="${OUTPUT_ROOT}/${rel}"
  fi

  echo "VOSR: ${input_dir} -> ${out_dir}"
  "${RUN_VOSR}" "${input_dir}" "${out_dir}" "$@"
done
