#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 BACKEND MANIFEST OUTPUT_DIR [runner arguments...]" >&2
  exit 2
fi

: "${VLM_WORK_ROOT:?set VLM_WORK_ROOT to the experiment workspace}"
: "${VLM_SOURCE_ROOT:?set VLM_SOURCE_ROOT to the transferred source root}"
: "${VLM_HF_CACHE:?set VLM_HF_CACHE to the pinned Hugging Face cache}"

BACKEND=$1
MANIFEST=$(realpath -- "$2")
OUTPUT_DIR=$(realpath -m -- "$3")
shift 3
EXPERIMENTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(dirname -- "$EXPERIMENTS_DIR")
VLM_WORK_ROOT=$(realpath -- "$VLM_WORK_ROOT")
readonly BACKEND MANIFEST OUTPUT_DIR EXPERIMENTS_DIR PACKAGE_ROOT VLM_WORK_ROOT
readonly CACHE_ROOT="$VLM_WORK_ROOT/cache/$BACKEND"

case "$MANIFEST" in
  "$VLM_WORK_ROOT"/*) ;;
  *) echo "manifest must be under VLM_WORK_ROOT" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$VLM_WORK_ROOT"/*) ;;
  *) echo "output directory must be under VLM_WORK_ROOT" >&2; exit 2 ;;
esac
if [[ -e $OUTPUT_DIR ]]; then
  echo "output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p \
  "$CACHE_ROOT/flashinfer" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/vllm" \
  "$CACHE_ROOT/vllm-config" \
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
  printf '{"backend":"%s","started_at":"%s","finished_at":"%s","state":"%s","exit_code":%d}\n' \
    "$BACKEND" "$started_at" "$finished_at" "$state" "$status" \
    >"$OUTPUT_DIR/status.json"
}
trap finish EXIT

active_processes=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
if [[ -n $active_processes ]]; then
  echo "The GPU is already in use: $active_processes" >&2
  exit 75
fi

container_args=(
  --nv
  --no-mount "home,cwd"
  --bind "$PACKAGE_ROOT:$PACKAGE_ROOT:ro"
  --bind "$VLM_SOURCE_ROOT:$VLM_SOURCE_ROOT:ro"
  --bind "$VLM_WORK_ROOT/runs:$VLM_WORK_ROOT/runs"
  --bind "$VLM_HF_CACHE:$VLM_HF_CACHE"
  --bind "$CACHE_ROOT:$CACHE_ROOT"
  --bind "$VLM_WORK_ROOT/tmp:$VLM_WORK_ROOT/tmp"
  --env PYTHONNOUSERSITE=1
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPATH="$PACKAGE_ROOT:$VLM_SOURCE_ROOT"
  --env HF_HOME="$VLM_HF_CACHE"
  --env TORCH_HOME="$CACHE_ROOT/torch"
  --env TRITON_CACHE_DIR="$CACHE_ROOT/triton"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env TMPDIR="$VLM_WORK_ROOT/tmp"
)

case "$BACKEND" in
  qwen3-vl)
    : "${VLM_QWEN_IMAGE:?set VLM_QWEN_IMAGE}"
    : "${VLM_QWEN_PYTHON:?set VLM_QWEN_PYTHON}"
    image=$VLM_QWEN_IMAGE
    python_path=$VLM_QWEN_PYTHON
    container_args+=(
      --env FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
      --env VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
      --env VLLM_CONFIG_ROOT="$CACHE_ROOT/vllm-config"
      --env VLLM_NO_USAGE_STATS=1
    )
    ;;
  internvideo3)
    : "${VLM_INTERN_IMAGE:?set VLM_INTERN_IMAGE}"
    : "${VLM_INTERN_PYTHON:?set VLM_INTERN_PYTHON}"
    image=$VLM_INTERN_IMAGE
    python_path=$VLM_INTERN_PYTHON
    container_args+=(--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
    ;;
  *)
    echo "unknown backend: $BACKEND" >&2
    exit 2
    ;;
esac

environment_root=$(dirname -- "$(dirname -- "$python_path")")
container_args+=(--bind "$environment_root:$environment_root:ro")

apptainer exec \
  "${container_args[@]}" \
  "$image" \
  "$python_path" -u -m experiments.run_multiscale_trials \
  --backend "$BACKEND" \
  --manifest "$MANIFEST" \
  --out "$OUTPUT_DIR" \
  "$@"
