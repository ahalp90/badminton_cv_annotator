#!/usr/bin/env bash
# Run one W5 stage from the remote experiment root with durable receipts.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
run=$1
label=$2
script=$3
shift 3
here=w5_holistic
out="$here/runs/$run"
mkdir -p "$out/logs" "$out/receipts" "$out/cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONOPTIMIZE
export XDG_CACHE_HOME="$PWD/$out/cache"
export PYTHONPATH=$here:next_steps_20260916/webui_seed/source:frozen_helpers_20260914/marking_diagnosis:frozen_helpers_20260914/vp_pruning:frozen_helpers_20260914/axis_matching:frozen_helpers_20260914/legacy:src:.
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
