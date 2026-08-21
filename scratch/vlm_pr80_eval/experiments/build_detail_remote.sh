#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

: "${VLM_WORK_ROOT:?set VLM_WORK_ROOT to the experiment workspace}"
: "${VLM_SOURCE_ROOT:?set VLM_SOURCE_ROOT to the transferred source root}"
: "${VLM_PIPELINE_PYTHON:?set VLM_PIPELINE_PYTHON to the CPU pipeline Python}"
: "${VLM_CONTEXT_CASES:?set VLM_CONTEXT_CASES to the frozen context cases}"
: "${VLM_CONTEXT_ATTEMPTS:?set VLM_CONTEXT_ATTEMPTS to the selected broad attempts}"
: "${VLM_CONTEXT_BACKEND:?set VLM_CONTEXT_BACKEND to the selected broad backend}"
: "${VLM_CONTEXT_SECONDS:?set VLM_CONTEXT_SECONDS to 90 or 120}"
: "${VLM_SOURCE_VIDEO_01:?set VLM_SOURCE_VIDEO_01}"
: "${VLM_SOURCE_VIDEO_15:?set VLM_SOURCE_VIDEO_15}"
: "${VLM_SOURCE_VIDEO_21:?set VLM_SOURCE_VIDEO_21}"

RUN_DIR=$(realpath -m -- "$1")
VLM_WORK_ROOT=$(realpath -- "$VLM_WORK_ROOT")
EXPERIMENTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(dirname -- "$EXPERIMENTS_DIR")
readonly RUN_DIR VLM_WORK_ROOT EXPERIMENTS_DIR PACKAGE_ROOT

case "$RUN_DIR" in
  "$VLM_WORK_ROOT"/runs/*) ;;
  *) echo "run directory must sit under VLM_WORK_ROOT/runs" >&2; exit 2 ;;
esac
if [[ -e $RUN_DIR ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
exec >"$RUN_DIR/build.log" 2>&1
started_at=$(date --iso-8601=seconds)

finish() {
  status=$?
  finished_at=$(date --iso-8601=seconds)
  if [[ $status -eq 0 ]]; then
    state="done"
    touch "$RUN_DIR/DONE"
  else
    state=failed
    touch "$RUN_DIR/FAILED"
  fi
  printf '{"started_at":"%s","finished_at":"%s","state":"%s","exit_code":%d}\n' \
    "$started_at" "$finished_at" "$state" "$status" \
    >"$RUN_DIR/status.json"
}
trap finish EXIT

PYTHONPATH="$PACKAGE_ROOT:$VLM_SOURCE_ROOT" \
  "$VLM_PIPELINE_PYTHON" -u -m experiments.build_detail_from_context \
  --context-cases "$VLM_CONTEXT_CASES" \
  --context-attempts "$VLM_CONTEXT_ATTEMPTS" \
  --context-seconds "$VLM_CONTEXT_SECONDS" \
  --backend "$VLM_CONTEXT_BACKEND" \
  --source-video "sset_01=$VLM_SOURCE_VIDEO_01" \
  --source-video "sset_15=$VLM_SOURCE_VIDEO_15" \
  --source-video "sset_21=$VLM_SOURCE_VIDEO_21" \
  --out "$RUN_DIR/detail"
