#!/bin/bash
# The final run on the 28 test views, two arms: the default detector (10% geometry blend), and the
# default with line-averaged paint. Artefacts and self-checks on, in check_20260925's 8 groups per arm,
# at most 8 processes at once, each group stopped after 2 hours.
# Usage: run_carmack.sh CHECKOUT OUT PYTHON FRESH_FEET
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4
cd "$CHECKOUT" || exit 1
GROUPS_FILE=scratch/court_det_fix/court_detector/check_20260925/correctness/groups.txt
mkdir -p "$OUT/blend_default/logs" "$OUT/line_paint/logs"

# One line per arm and group: the arm, its extra run_views flag, the group name, then the group's views.
{ sed 's/^/blend_default none /' "$GROUPS_FILE"; sed 's/^/line_paint --line-paint /' "$GROUPS_FILE"; } |
xargs -P 8 -L 1 bash -c '
    ARM=$0 FLAG=$1 GROUP=$2
    shift 2
    EXTRA=()
    [ "$FLAG" != none ] && EXTRA=("$FLAG")
    timeout 2h "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/$ARM" --artefacts "${EXTRA[@]}" "$@" > "$OUT/$ARM/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/$ARM/logs/$GROUP.exit"'
touch "$OUT/done"
