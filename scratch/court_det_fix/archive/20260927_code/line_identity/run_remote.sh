#!/usr/bin/env bash
# Run one experiment stage from the remote experiment root with durable log, PID and exit receipts.
# Usage (from anywhere on the compute host):
#   bash <root>/line_identity/run_remote.sh <run> <label> <script-stem> [script args...]
# The script runs with the packet's single-thread environment and the sibling experiment
# directories on PYTHONPATH. Tool caches stay inside the run directory.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
run=$1
label=$2
script=$3
shift 3
here=line_identity
out="$here/runs/$run"
mkdir -p "$out/logs" "$out/receipts" "$out/cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The stage scripts prove replays with assert; an inherited optimise flag would strip them.
unset PYTHONOPTIMIZE
export XDG_CACHE_HOME="$PWD/$out/cache"
# This folder goes first so no sibling directory can shadow an experiment module.
export PYTHONPATH=$here:src:.:vp_pruning_20260914:marking_diagnosis_20260914:axis_matching_20260914:automatic_axes_20260914:automatic_axes_20260914/svd_fixed
python="${REMOTE_PYTHON:-$HOME/.venvs/venv-rtmlib/bin/python}"
printf '%s\n' "$$" > "$out/receipts/$label.pid"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$python" "$here/$script.py" --root . --run "$run" "$@" > "$out/logs/$label.log" 2>&1
status=$?
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$status" > "$out/receipts/${label}_exit_code.txt"
printf 'started=%s\nfinished=%s\nscript=%s\nargs=%s\n' "$started" "$finished" "$script" "$*" > "$out/receipts/${label}_times.txt"
rm -f "$out/receipts/$label.pid"
exit "$status"
