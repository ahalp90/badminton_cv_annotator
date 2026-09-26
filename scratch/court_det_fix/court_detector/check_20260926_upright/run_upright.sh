#!/bin/bash
# The upright-camera filter on the 28 D17 views, one stage after another, at most 8 processes each:
#   1. artefacts and self-checks on, in check_20260925's 8 groups
#   2. timing with self-checks off, one process per view, to compare with check_20260925's
#      timing_no_checks run. The code with the filter off is unchanged since that run.
# Usage: run_upright.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
cd "$CHECKOUT" || exit 1
GROUPS_FILE=scratch/court_det_fix/court_detector/check_20260925/correctness/groups.txt
mkdir -p "$OUT/upright/logs" "$OUT/timing_upright/logs"

xargs -P 8 -L 1 bash -c '
    GROUP=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/upright" --artefacts "$@" > "$OUT/upright/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/upright/logs/$GROUP.exit"' < "$GROUPS_FILE"
touch "$OUT/upright/done"

cut -d " " -f 2- "$GROUPS_FILE" | tr " " "\n" | xargs -P 8 -L 1 bash -c '
    VIEW=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/timing_upright" --timing --no-self-checks "$VIEW" > "$OUT/timing_upright/logs/$VIEW.log" 2>&1
    echo $? > "$OUT/timing_upright/logs/$VIEW.exit"'
touch "$OUT/timing_upright/done"
touch "$OUT/all.done"
