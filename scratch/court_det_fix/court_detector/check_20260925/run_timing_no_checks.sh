#!/bin/bash
# After run_all.sh finishes: timing with self-checks off, one process per view, at most 8 at a time.
# Usage: run_timing_no_checks.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
until [ -e "$OUT/all.done" ]; do sleep 60; done
cd "$CHECKOUT" || exit 1
mkdir -p "$OUT/timing_no_checks/logs"
cut -d " " -f 2- "$OUT/correctness/groups.txt" | tr " " "\n" | xargs -P 8 -L 1 bash -c '
    VIEW=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/timing_no_checks" --timing --no-self-checks "$VIEW" > "$OUT/timing_no_checks/logs/$VIEW.log" 2>&1
    echo $? > "$OUT/timing_no_checks/logs/$VIEW.exit"'
touch "$OUT/timing_no_checks/done"
