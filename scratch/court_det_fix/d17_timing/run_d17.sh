#!/bin/bash
# Run the fresh D17 chain on the accepted 20-case gallery and the eight labelled non-court
# controls, at direction budgets 16 and 12. One process per view and budget.
# Usage: run_d17.sh OUTPUT_DIR PYTHON PARALLEL_JOBS
set -u
if [ $# -ne 3 ]; then
    echo "usage: $0 OUTPUT_DIR PYTHON PARALLEL_JOBS" >&2
    exit 2
fi
export D17_OUT=$1
export D17_PYTHON=$2
JOBS=$3
export D17_HERE
D17_HERE=$(cd "$(dirname "$0")" && pwd)

# Slowest views first, so long GX and amateur jobs do not finish last.
GALLERY="gxBQ_window_00_frame_0 gxBQ_window_00_frame_5 gxBQ_window_00_frame_689 gxBQ_window_03_frame_77876
am2_window_01_frame_28019 am2_window_00_frame_150 am3_window_00_frame_0 am3_window_02_frame_17174
am1_window_00_frame_54 letterboxed_short_frame_78 shuttleset_03_scene_0017 shuttleset_03_scene_0019
shuttleset_03_scene_0034 shuttleset_03_scene_0023 shuttleset_03_scene_0016 shuttleset_03_scene_0032
shuttleset_21_scene_0034 shuttleset_21_scene_0039 shuttleset_21_scene_0044 sset_21_gloiZ_gTJaE_frame_00000001"
NON_COURT="sset_21_gloiZ_gTJaE_frame_00000300 sset_21_gloiZ_gTJaE_frame_00009558 sset_21_gloiZ_gTJaE_frame_00014336
sset_21_gloiZ_gTJaE_frame_00023893 sset_21_gloiZ_gTJaE_frame_00038228 sset_21_gloiZ_gTJaE_frame_00062120
sset_21_gloiZ_gTJaE_frame_00090790 sset_21_gloiZ_gTJaE_frame_00100347"

mkdir -p "$D17_OUT/logs"
for CASE in $GALLERY $NON_COURT; do
    for BUDGET in 16 12; do
        echo "$CASE $BUDGET"
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    LOG="$D17_OUT/logs/$0.budget$1"
    D17_LAUNCH=$(date +%s.%N) "$D17_PYTHON" "$D17_HERE/run_d17.py" \
        --case "$0" --direction-budget "$1" --output "$D17_OUT/budget$1" > "$LOG.log" 2>&1
    echo $? > "$LOG.exit"'
# all.done means the batch finished; it lists any job that exited non-zero.
grep -L '^0$' "$D17_OUT"/logs/*.exit > "$D17_OUT/all.done"
