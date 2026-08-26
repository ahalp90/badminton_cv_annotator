#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [--all-eligible]" >&2
  exit 2
fi

: "${VLM_WORK_ROOT:?set VLM_WORK_ROOT to the experiment workspace}"
: "${VLM_SOURCE_ROOT:?set VLM_SOURCE_ROOT to the transferred source root}"
: "${VLM_PIPELINE_PYTHON:?set VLM_PIPELINE_PYTHON to the CPU pipeline Python}"
: "${VLM_ARTIFACTS_ROOT:?set VLM_ARTIFACTS_ROOT to the annotator artefacts}"
: "${VLM_REPO_ROOT:?set VLM_REPO_ROOT to the repository with ShuttleSet truth}"
: "${VLM_SCENE_LABELS:?set VLM_SCENE_LABELS to the human timeline directory}"

RUN_DIR=$(realpath -m -- "$1")
selection_args=(--pilot-cases 12)
if [[ $# -eq 2 ]]; then
  if [[ $2 != --all-eligible ]]; then
    echo "unknown case mode: $2" >&2
    exit 2
  fi
  selection_args=(--all-eligible)
fi
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
    state="failed"
    touch "$RUN_DIR/FAILED"
  fi
  printf '{"started_at":"%s","finished_at":"%s","state":"%s","exit_code":%d}\n' \
    "$started_at" "$finished_at" "$state" "$status" \
    >"$RUN_DIR/status.json"
}
trap finish EXIT

PYTHONPATH="$PACKAGE_ROOT:$VLM_SOURCE_ROOT" \
  "$VLM_PIPELINE_PYTHON" -u -m experiments.build_multiscale_trials \
  --artifacts-root "$VLM_ARTIFACTS_ROOT" \
  --repo-root "$VLM_REPO_ROOT" \
  --scene-labels-dir "$VLM_SCENE_LABELS" \
  --video sset_01 \
  --video sset_15 \
  --video sset_21 \
  "${selection_args[@]}" \
  --context-seconds 90 \
  --context-seconds 120 \
  --max-frames 96 \
  --out "$RUN_DIR/context/cases"
