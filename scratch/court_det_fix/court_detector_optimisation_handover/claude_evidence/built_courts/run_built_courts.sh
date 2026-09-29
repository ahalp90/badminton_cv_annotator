#!/bin/bash
# Rebuild every direction pair's courts on the test views (local_built_courts.py), with and without the
# upright-camera filter, at most 8 views at once, each stopped after an hour. Writes one record per view into
# OUT/filtered/ and OUT/unfiltered/, logs and exit codes into OUT/logs/, then done, or failed listing the runs
# that failed or wrote no exit code.
# Usage: run_built_courts.sh CHECKOUT ARTEFACTS OUT PYTHON
#   ARTEFACTS: a joined-detector run's artefacts, one per view (the final 26 September Carmack run's blend_default)
set -u
export CHECKOUT=$1 ARTEFACTS=$2 OUT=$3 PYTHON=$4
cd "$CHECKOUT" || exit 1
mkdir -p "$OUT/logs"
rm -f "$OUT/done" "$OUT/failed"
views=$(ls "$ARTEFACTS" | sed -n 's/\.json\.gz$//p')

for arm in filtered unfiltered; do for view in $views; do echo "$arm $view"; done; done |
xargs -P 8 -L 1 bash -c '
    FLAG=()
    [ "$0" = unfiltered ] && FLAG=(--without-upright-filter)
    PYTHONPATH=.:src timeout 1h "$PYTHON" \
        scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/built_courts/local_built_courts.py \
        "$ARTEFACTS" "$1" "$OUT/$0" "${FLAG[@]}" > "$OUT/logs/$0.$1.log" 2>&1
    echo $? > "$OUT/logs/$0.$1.exit"'

exits=("$OUT"/logs/*.exit)
expected=$((2 * $(echo "$views" | wc -w)))
failed=$(grep -Lx 0 "${exits[@]}" 2> /dev/null)
if [ -n "$failed" ] || [ "${#exits[@]}" -ne "$expected" ]; then
    { echo "$failed"; echo "${#exits[@]} exit files, expected $expected"; } > "$OUT/failed"
    exit 1
fi
touch "$OUT/done"
