#!/bin/bash
# The default detector, with the 10% geometry blend, on the 28 test views: artefacts and self-checks on,
# in check_20260925's 8 groups, at most 8 processes, each group stopped after 2 hours.
# Usage: run_blend_default.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
cd "$CHECKOUT" || exit 1
GROUPS_FILE=scratch/court_det_fix/court_detector/check_20260925/correctness/groups.txt
mkdir -p "$OUT/blend_default/logs"

xargs -P 8 -L 1 bash -c '
    GROUP=$0
    timeout 2h "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/blend_default" --artefacts "$@" > "$OUT/blend_default/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/blend_default/logs/$GROUP.exit"' < "$GROUPS_FILE"
touch "$OUT/blend_default/done"
