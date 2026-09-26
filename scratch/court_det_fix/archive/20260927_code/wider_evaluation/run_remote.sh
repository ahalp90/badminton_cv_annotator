#!/usr/bin/env bash
# Continue the historical full-16 evaluation; use run_cases.py with a new output for SVD runs.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
repo="$(git -C "$here" rev-parse --show-toplevel)"
label=${1:?Supply a new dispatch label}
shift
output="$here/runs/20260922/measured"
mkdir -p "$output/receipts" "$output/logs"
log="$output/logs/$label.log"
if [[ -e "$log" ]]; then
  printf 'Log already exists: %s\n' "$log" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo:$repo/src"
unset PYTHONOPTIMIZE
printf '%s\n' "$$" > "$output/receipts/$label.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/receipts/$label.started"
python="${REMOTE_PYTHON:-$HOME/.venvs/venv-rtmlib/bin/python}"
"$python" -u "$here/run_cases.py" \
  --root "$root" --output "$output" --workers 6 \
  --manifest "$here/runs/20260922/manifest.json.gz" \
  --control-pack "$here/runs/20260922/control_inputs.json.gz" "$@" \
  --direction-budget 16 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if (( status == 0 )); then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$output/receipts/$label.exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/receipts/$label.finished"
exit "$status"
