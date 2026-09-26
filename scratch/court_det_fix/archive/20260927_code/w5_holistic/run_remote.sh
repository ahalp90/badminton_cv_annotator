#!/usr/bin/env bash
# Run one W5 stage from the remote experiment root with durable receipts.
set -uo pipefail
if (( $# < 3 )); then
  printf 'usage: %s RUN LABEL SCRIPT [ARGS...]\n' "$0" >&2
  exit 2
fi
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
run=$1
label=$2
script=$3
shift 3
here=w5_holistic
out="$here/runs/$run"
if [[ -e "$out/logs/$label.log" ]]; then
  printf 'W5 stage log already exists: %s; use a new run name\n' "$out/logs/$label.log" >&2
  exit 1
fi
repo_root="$(git -C "$here" rev-parse --show-toplevel)"
mkdir -p "$out/logs" "$out/receipts" "$out/cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONOPTIMIZE
export XDG_CACHE_HOME="$PWD/$out/cache"
export PYTHONPATH="$repo_root:$repo_root/src:$PWD/$here:$PWD/next_steps_20260916/webui_seed/source:$PWD/frozen_helpers_20260914/marking_diagnosis:$PWD/frozen_helpers_20260914/vp_pruning:$PWD/frozen_helpers_20260914/axis_matching:$PWD/frozen_helpers_20260914/legacy:$PWD/src:$PWD"
python="${REMOTE_PYTHON:-$HOME/.venvs/venv-pipeline/bin/python}"
home_prefix="\$HOME/"
if [[ "$python" == "$home_prefix"* ]]; then
  python="$HOME/${python#\$HOME/}"
fi
printf '%s\n' "$$" > "$out/receipts/$label.pid"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$python" -u "$here/$script.py" --root . --run "$run" "$@" 2>&1 | tee "$out/logs/$label.log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 )); then
  status=${pipeline_status[1]}
fi
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$status" > "$out/receipts/${label}_exit_code.txt"
printf 'started=%s\nfinished=%s\nscript=%s\nargs=%s\n' "$started" "$finished" "$script" "$*" > "$out/receipts/${label}_times.txt"
rm -f "$out/receipts/$label.pid"
exit "$status"
