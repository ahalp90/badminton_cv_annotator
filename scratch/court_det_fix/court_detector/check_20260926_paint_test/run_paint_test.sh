#!/bin/bash
# The gap-bounded paint test on the 28 D17 views: artefacts and self-checks on, in check_20260925's
# 8 groups, at most 8 processes, each group stopped after 2 hours.
# Usage: run_paint_test.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
cd "$CHECKOUT" || exit 1
GROUPS_FILE=scratch/court_det_fix/court_detector/check_20260925/correctness/groups.txt
mkdir -p "$OUT/paint_test/logs"

xargs -P 8 -L 1 bash -c '
    GROUP=$0
    timeout 2h "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/paint_test" --artefacts "$@" > "$OUT/paint_test/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/paint_test/logs/$GROUP.exit"' < "$GROUPS_FILE"
touch "$OUT/paint_test/done"
