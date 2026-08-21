#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 MANIFEST ARM VIDEO_ID OUTPUT_DIR [LIMIT]" >&2
  exit 2
fi

: "${VLM_WORK_ROOT:?set VLM_WORK_ROOT to the experiment workspace}"
: "${VLM_SOURCE_ROOT:?set VLM_SOURCE_ROOT to the transferred source root}"
: "${VLM_HF_CACHE:?set VLM_HF_CACHE to the pinned Hugging Face cache}"
: "${VLM_INTERN_IMAGE:?set VLM_INTERN_IMAGE}"
: "${VLM_INTERN_PYTHON:?set VLM_INTERN_PYTHON}"

MANIFEST=$(realpath -- "$1")
ARM=$2
VIDEO_ID=$3
OUTPUT_DIR=$(realpath -m -- "$4")
LIMIT=${5:-}
EXPERIMENTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(dirname -- "$EXPERIMENTS_DIR")
VLM_WORK_ROOT=$(realpath -- "$VLM_WORK_ROOT")
readonly MANIFEST ARM VIDEO_ID OUTPUT_DIR EXPERIMENTS_DIR PACKAGE_ROOT VLM_WORK_ROOT
readonly CACHE_ROOT="$VLM_WORK_ROOT/cache/internvideo3"

runner_args=()
if [[ -n $LIMIT ]]; then
  runner_args+=(--limit "$LIMIT")
fi

for path in "$MANIFEST" "$OUTPUT_DIR"; do
  case "$path" in
    "$VLM_WORK_ROOT"/runs/*) ;;
    *) echo "trial paths must sit under VLM_WORK_ROOT/runs" >&2; exit 2 ;;
  esac
done
if [[ -e $OUTPUT_DIR ]]; then
  echo "output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/xdg" \
  "$VLM_WORK_ROOT/tmp" \
  "$OUTPUT_DIR"
exec >"$OUTPUT_DIR/run.log" 2>&1
started_at=$(date --iso-8601=seconds)

finish() {
  status=$?
  finished_at=$(date --iso-8601=seconds)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    >"$OUTPUT_DIR/gpu-after.txt" || true
  if [[ $status -eq 0 ]]; then
    state="done"
    touch "$OUTPUT_DIR/DONE"
  else
    state=failed
    touch "$OUTPUT_DIR/FAILED"
  fi
  printf '{"arm":"%s","video_id":"%s","started_at":"%s","finished_at":"%s","state":"%s","exit_code":%d}\n' \
    "$ARM" "$VIDEO_ID" "$started_at" "$finished_at" "$state" "$status" \
    >"$OUTPUT_DIR/status.json"
}
trap finish EXIT

active_processes=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
if [[ -n $active_processes ]]; then
  echo "The GPU is already in use: $active_processes" >&2
  exit 75
fi

environment_root=$(dirname -- "$(dirname -- "$VLM_INTERN_PYTHON")")
container_args=(
  --nv
  --no-mount "home,cwd"
  --bind "$PACKAGE_ROOT:$PACKAGE_ROOT:ro"
  --bind "$VLM_SOURCE_ROOT:$VLM_SOURCE_ROOT:ro"
  --bind "$VLM_WORK_ROOT/runs:$VLM_WORK_ROOT/runs"
  --bind "$VLM_HF_CACHE:$VLM_HF_CACHE"
  --bind "$CACHE_ROOT:$CACHE_ROOT"
  --bind "$VLM_WORK_ROOT/tmp:$VLM_WORK_ROOT/tmp"
  --bind "$environment_root:$environment_root:ro"
  --env PYTHONNOUSERSITE=1
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH="$PACKAGE_ROOT:$VLM_SOURCE_ROOT"
  --env HF_HOME="$VLM_HF_CACHE"
  --env TORCH_HOME="$CACHE_ROOT/torch"
  --env TRITON_CACHE_DIR="$CACHE_ROOT/triton"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env TMPDIR="$VLM_WORK_ROOT/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

timeout --kill-after=30s 45m apptainer exec \
  "${container_args[@]}" \
  "$VLM_INTERN_IMAGE" \
  "$VLM_INTERN_PYTHON" -u -m experiments.rally_opening_trials run \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR/attempts" \
  --arm "$ARM" \
  --video-id "$VIDEO_ID" \
  "${runner_args[@]}"
