#!/bin/bash
# Court-detector checks on the 28 D17 views, one after another, at most 8 processes each:
#   1. correctness: run_views.py against the 24 Sept baseline, 8 groups of 2-6 views
#   2. run_d17.py rerun at this checkout, to compare with the baseline bit for bit
#   3. timing: run_views.py, one process per view, artefacts off
# Usage: run_all.sh CHECKOUT OUT PYTHON FRESH_FEET BASELINE
# --any-camera-roll (added 26 September) keeps the detector as it was on 25 September.
set -u
export CHECKOUT=$1 OUT=$2 PYTHON=$3 FRESH_FEET=$4 BASELINE=$5
cd "$CHECKOUT" || exit 1
mkdir -p "$OUT/correctness/logs" "$OUT/timing/logs"

# Groups balanced on the baseline's per-view seconds. The 12 views from the long sset21 video
# sit in the last two groups, so that video is decoded twice rather than eight times.
cat > "$OUT/correctness/groups.txt" <<'GROUPS'
g1 gxBQ_window_03_frame_77876 am1_window_00_frame_54
g2 gxBQ_window_00_frame_689 am3_window_02_frame_17174 am2_window_01_frame_28019
g3 gxBQ_window_00_frame_0 shuttleset_03_scene_0019
g4 gxBQ_window_00_frame_5 shuttleset_03_scene_0034 letterboxed_short_frame_78
g5 shuttleset_03_scene_0017 shuttleset_03_scene_0032 am2_window_00_frame_150
g6 shuttleset_03_scene_0016 shuttleset_03_scene_0023 am3_window_00_frame_0
g7 shuttleset_21_scene_0039 sset_21_gloiZ_gTJaE_frame_00000001 sset_21_gloiZ_gTJaE_frame_00014336 sset_21_gloiZ_gTJaE_frame_00062120 sset_21_gloiZ_gTJaE_frame_00009558 sset_21_gloiZ_gTJaE_frame_00090790
g8 shuttleset_21_scene_0034 shuttleset_21_scene_0044 sset_21_gloiZ_gTJaE_frame_00100347 sset_21_gloiZ_gTJaE_frame_00000300 sset_21_gloiZ_gTJaE_frame_00023893 sset_21_gloiZ_gTJaE_frame_00038228
GROUPS

xargs -P 8 -L 1 bash -c '
    GROUP=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/correctness" --baseline "$BASELINE" --feet "$FRESH_FEET/feet_standing.json.gz" \
        --artefacts --any-camera-roll "$@" > "$OUT/correctness/logs/$GROUP.log" 2>&1
    echo $? > "$OUT/correctness/logs/$GROUP.exit"' < "$OUT/correctness/groups.txt"
touch "$OUT/correctness/done"

bash scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/fresh_feet/run_feet_variants.sh \
    "$CHECKOUT" "$OUT/run_d17" "$FRESH_FEET" "$PYTHON" 8 standing
touch "$OUT/run_d17/done"

cut -d " " -f 2- "$OUT/correctness/groups.txt" | tr " " "\n" | xargs -P 8 -L 1 bash -c '
    VIEW=$0
    "$PYTHON" -m scratch.court_det_fix.court_detector.run_views --people "$FRESH_FEET/people" \
        --output "$OUT/timing" --timing --any-camera-roll "$VIEW" > "$OUT/timing/logs/$VIEW.log" 2>&1
    echo $? > "$OUT/timing/logs/$VIEW.exit"'
touch "$OUT/all.done"
