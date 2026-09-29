#!/usr/bin/env bash
# Stage beside run_temporal_union.py and L3_temporal/ on Carmack.
set -u
if [ "$#" -lt 2 ]; then
    echo "usage: $0 <gx|am3> <target-case> [target-case ...]" >&2
    exit 2
fi
cohort=$1
shift
first_target=$1
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
output="$here/results/$cohort"
mkdir -p "$here/logs" "$here/receipts" "$output"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export COURT_DET_FIX_LINE_IDENTITY="$root/line_identity_eval_20260922"
export COURT_DET_FIX_L3_TEMPORAL="$here/L3_temporal/run_l3_temporal.py"
export COURT_DET_FIX_VIEW_ALIGNMENT="$here/L3_temporal/view_alignment.json"
python="${REMOTE_PYTHON:-$HOME/.venvs/venv-rtmlib/bin/python}"
date -u +%Y-%m-%dT%H:%M:%SZ > "$here/receipts/$first_target.started.txt"
"$python" -u "$here/run_temporal_union.py" \
    --cohort "$cohort" \
    --population-root "$here/populations" \
    --output "$output" \
    --targets "$@" > "$here/logs/$first_target.log" 2>&1
status=$?
printf '%s\n' "$status" > "$here/receipts/$first_target.exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$here/receipts/$first_target.finished.txt"
exit "$status"
