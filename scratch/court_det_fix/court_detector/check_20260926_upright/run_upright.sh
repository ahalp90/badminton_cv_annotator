#!/bin/bash
# The upright-camera filter on the 28 D17 views, one stage after another, at most 8 processes each:
#   1. filter on, artefacts and self-checks on, in check_20260925's 8 groups
#   2. timing, filter on, self-checks off, one process per view
#   3. timing, filter off (--any-camera-roll), self-checks off: the same-day comparison for stage 2
# Usage: run_upright.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
cd "$CHECKOUT" || exit 1
GROUPS_FILE=scratch/court_det_fix/court_detector/check_20260925/correctness/groups.txt
mkdir -p "$OUT/upright/logs" "$OUT/timing_upright/logs" "$OUT/timing_any_roll/logs"

xargs -P 8 -L 1 bash -c '
    GROUP=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/upright" --artefacts "$@" > "$OUT/upright/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/upright/logs/$GROUP.exit"' < "$GROUPS_FILE"
touch "$OUT/upright/done"

for ARM in timing_upright timing_any_roll; do
    export ARM
    cut -d " " -f 2- "$GROUPS_FILE" | tr " " "\n" | xargs -P 8 -L 1 bash -c '
        VIEW=$0
        FLAG=$([ "$ARM" = timing_any_roll ] && echo --any-camera-roll)
        "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
            --output "$OUT/$ARM" --timing --no-self-checks $FLAG "$VIEW" > "$OUT/$ARM/logs/$VIEW.log" 2>&1
        echo $? > "$OUT/$ARM/logs/$VIEW.exit"'
    touch "$OUT/$ARM/done"
done
touch "$OUT/all.done"
