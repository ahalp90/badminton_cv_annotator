#!/bin/bash
# Full-search D17 run per feet variant, all variants in one job pool, slowest views first.
# Usage: run_feet_variants.sh CHECKOUT OUT FEET_DIR PYTHON JOBS VARIANT...
set -u
export CHECKOUT=$1 OUT=$2 FEET=$3 PYTHON=$4
JOBS=$5
shift 5
VIEWS="gxBQ_window_00_frame_0 gxBQ_window_00_frame_5 gxBQ_window_00_frame_689 gxBQ_window_03_frame_77876
am2_window_01_frame_28019 am2_window_00_frame_150 am3_window_00_frame_0 am3_window_02_frame_17174
am1_window_00_frame_54 letterboxed_short_frame_78 shuttleset_03_scene_0017 shuttleset_03_scene_0019
shuttleset_03_scene_0034 shuttleset_03_scene_0023 shuttleset_03_scene_0016 shuttleset_03_scene_0032
shuttleset_21_scene_0034 shuttleset_21_scene_0039 shuttleset_21_scene_0044 sset_21_gloiZ_gTJaE_frame_00000001
sset_21_gloiZ_gTJaE_frame_00000300 sset_21_gloiZ_gTJaE_frame_00009558 sset_21_gloiZ_gTJaE_frame_00014336
sset_21_gloiZ_gTJaE_frame_00023893 sset_21_gloiZ_gTJaE_frame_00038228 sset_21_gloiZ_gTJaE_frame_00062120
sset_21_gloiZ_gTJaE_frame_00090790 sset_21_gloiZ_gTJaE_frame_00100347"
mkdir -p "$OUT/logs"
for CASE in $VIEWS; do
    for VARIANT in "$@"; do
        echo "$VARIANT $CASE"
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    LOG="$OUT/logs/$0.$1"
    D17_LAUNCH=$(date +%s.%N) "$PYTHON" "$CHECKOUT/scratch/court_det_fix/d17_timing/run_d17.py" --case "$1" \
        --direction-budget 16 --output "$OUT/$0/budget16" --feet "$FEET/feet_$0.json.gz" > "$LOG.log" 2>&1
    echo $? > "$LOG.exit"'
grep -L '^0$' "$OUT"/logs/*.exit > "$OUT/all.done"
