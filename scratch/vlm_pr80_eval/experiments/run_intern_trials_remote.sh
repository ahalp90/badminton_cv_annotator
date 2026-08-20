#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MANIFEST OUTPUT_DIR [runner arguments...]" >&2
  exit 2
fi

: "${VLM_WORK_ROOT:?set VLM_WORK_ROOT to the experiment workspace}"
: "${VLM_INTERN_IMAGE:?set VLM_INTERN_IMAGE to the InternVideo3 Apptainer image}"
: "${VLM_INTERN_ENV_ROOT:?set VLM_INTERN_ENV_ROOT to the InternVideo3 Python environment root}"
: "${VLM_INTERN_PYTHON:?set VLM_INTERN_PYTHON to its Python executable path}"
: "${VLM_INTERN_FFMPEG_PREFIX:?set VLM_INTERN_FFMPEG_PREFIX to the ffmpeg environment root}"
: "${VLM_HF_CACHE:?set VLM_HF_CACHE to the Hugging Face cache}"

EXPERIMENTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(dirname -- "$EXPERIMENTS_DIR")
VLM_WORK_ROOT=$(realpath -- "$VLM_WORK_ROOT")
MANIFEST=$(realpath -- "$1")
OUTPUT_DIR=$(realpath -m -- "$2")
readonly EXPERIMENTS_DIR PACKAGE_ROOT VLM_WORK_ROOT MANIFEST OUTPUT_DIR
readonly CACHE_ROOT="$VLM_WORK_ROOT/cache/internvideo3"
shift 2

case "$MANIFEST" in
  "$VLM_WORK_ROOT"/*) ;;
  *) echo "manifest must be under VLM_WORK_ROOT" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$VLM_WORK_ROOT"/*) ;;
  *) echo "output directory must be under VLM_WORK_ROOT" >&2; exit 2 ;;
esac
if [[ ! -f $VLM_INTERN_IMAGE || ! -x $VLM_INTERN_PYTHON ]]; then
  echo "InternVideo3 image or Python executable is missing" >&2
  exit 2
fi
if [[ ! -d $VLM_INTERN_FFMPEG_PREFIX ]]; then
  echo "InternVideo3 ffmpeg environment is missing" >&2
  exit 2
fi

active_processes=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
if [[ -n $active_processes ]]; then
  echo "The GPU is already in use: $active_processes" >&2
  exit 75
fi

mkdir -p \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/xdg" \
  "$OUTPUT_DIR" \
  "$VLM_WORK_ROOT/tmp"
MANIFEST_DIR=$(dirname -- "$MANIFEST")
readonly MANIFEST_DIR

record_gpu_state() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    >"$OUTPUT_DIR/internvideo3-gpu-after.txt"
}
trap record_gpu_state EXIT

timeout --kill-after=30s 25m apptainer exec \
  --nv \
  --no-mount home,cwd \
  --bind "$PACKAGE_ROOT:$PACKAGE_ROOT:ro" \
  --bind "$VLM_INTERN_ENV_ROOT:$VLM_INTERN_ENV_ROOT:ro" \
  --bind "$VLM_INTERN_FFMPEG_PREFIX:$VLM_INTERN_FFMPEG_PREFIX:ro" \
  --bind "$VLM_HF_CACHE:$VLM_HF_CACHE" \
  --bind "$CACHE_ROOT:$CACHE_ROOT" \
  --bind "$VLM_WORK_ROOT/tmp:$VLM_WORK_ROOT/tmp" \
  --bind "$MANIFEST_DIR:$MANIFEST_DIR:ro" \
  --bind "$OUTPUT_DIR:$OUTPUT_DIR" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH="$PACKAGE_ROOT" \
  --env HF_HOME="$VLM_HF_CACHE" \
  --env TORCH_HOME="$CACHE_ROOT/torch" \
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
  --env TMPDIR="$VLM_WORK_ROOT/tmp" \
  --env LD_LIBRARY_PATH="$VLM_INTERN_FFMPEG_PREFIX/lib" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VLM_INTERN_IMAGE" \
  "$VLM_INTERN_PYTHON" -u -m experiments.run_trials \
  --backend internvideo3 \
  --manifest "$MANIFEST" \
  --out "$OUTPUT_DIR" \
  "$@"
