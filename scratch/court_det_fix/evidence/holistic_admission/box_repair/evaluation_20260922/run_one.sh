#!/usr/bin/env bash
# Run one repaired person-observation case with durable logs and receipts.
# Usage: bash run_one.sh <run> <case-id>
set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <run> <case-id>" >&2
    exit 2
fi

run=$1
case_id=$2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${COURT_DET_FIX_ROOT:-}" ]; then
    root="$COURT_DET_FIX_ROOT"
elif [ -d "$here/../frozen_views" ]; then
    root="$(cd "$here/.." && pwd)"
else
    root="$(cd "$here/../../../../" && pwd)"
fi
output="$here/runs/$run"
log="$output/logs/person_observations_$case_id.log"
receipt_prefix="$output/receipts/person_observations_$case_id"
mkdir -p "$output/logs" "$output/receipts" "$output/cache"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$output/cache"
unset PYTHONOPTIMIZE

python="${REMOTE_PYTHON:-$HOME/.venvs/venv-rtmlib/bin/python}"
export PYTHONPATH="$here:$root/src:$root:$root/frozen_helpers_20260914/vp_pruning:$root/frozen_helpers_20260914/marking_diagnosis:$root/frozen_helpers_20260914/axis_matching:$root/frozen_helpers_20260914/automatic_axes:$root/frozen_helpers_20260914/automatic_axes/svd_fixed"

printf '%s\n' "$$" > "$receipt_prefix.pid"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$python" "$here/launch_matcher.py" \
    --root "$root" \
    --run "$run" \
    --arm person_observations \
    --ids "$case_id" \
    --inputs-dir "$here/repair/inputs" \
    --repair-manifest "$here/repair/manifest.json.gz" \
    > "$log" 2>&1
status=$?
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$status" > "$receipt_prefix.exit_code.txt"
printf 'started=%s\nfinished=%s\nscript=launch_matcher.py\ncase=%s\n' \
    "$started" "$finished" "$case_id" > "$receipt_prefix.times.txt"
rm -f "$receipt_prefix.pid"
exit "$status"
