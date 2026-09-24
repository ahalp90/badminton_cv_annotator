#!/bin/bash
# Prefilter measurement: the full-search D17 chain per view under measure_prefilter.py's
# observation-only hooks, slowest views first. One JSON-lines file of per-pair rows per view.
# Usage: run_prefilter.sh CHECKOUT OUT FEET_FILE PYTHON HOOK JOBS
set -u
if [ $# -ne 6 ]; then
    echo "usage: $0 CHECKOUT OUT FEET_FILE PYTHON HOOK JOBS" >&2
    exit 2
fi
export CHECKOUT=$1 OUT=$2 FEET=$3 PYTHON=$4 HOOK=$5
JOBS=$6
VIEWS="gxBQ_window_00_frame_0 gxBQ_window_00_frame_5 gxBQ_window_00_frame_689 gxBQ_window_03_frame_77876
am2_window_01_frame_28019 am2_window_00_frame_150 am3_window_00_frame_0 am3_window_02_frame_17174
am1_window_00_frame_54 letterboxed_short_frame_78 shuttleset_03_scene_0017 shuttleset_03_scene_0019
shuttleset_03_scene_0034 shuttleset_03_scene_0023 shuttleset_03_scene_0016 shuttleset_03_scene_0032
shuttleset_21_scene_0034 shuttleset_21_scene_0039 shuttleset_21_scene_0044 sset_21_gloiZ_gTJaE_frame_00000001
sset_21_gloiZ_gTJaE_frame_00000300 sset_21_gloiZ_gTJaE_frame_00009558 sset_21_gloiZ_gTJaE_frame_00014336
sset_21_gloiZ_gTJaE_frame_00023893 sset_21_gloiZ_gTJaE_frame_00038228 sset_21_gloiZ_gTJaE_frame_00062120
sset_21_gloiZ_gTJaE_frame_00090790 sset_21_gloiZ_gTJaE_frame_00100347"
mkdir -p "$OUT/logs" "$OUT/rows"
for CASE in $VIEWS; do
    echo "$CASE"
done | xargs -P "$JOBS" -L 1 bash -c '
    # The hook imports run_d17 from the working directory.
    cd "$CHECKOUT/scratch/court_det_fix/d17_timing" || exit 1
    PREFILTER_OUT="$OUT/rows/$0.jsonl" D17_LAUNCH=$(date +%s.%N) "$PYTHON" "$HOOK" --case "$0" \
        --direction-budget 16 --output "$OUT/budget16" --feet "$FEET" > "$OUT/logs/$0.log" 2>&1
    echo $? > "$OUT/logs/$0.exit"'
# all.done means the batch finished; it lists any job that exited non-zero.
grep -L '^0$' "$OUT"/logs/*.exit > "$OUT/all.done"
