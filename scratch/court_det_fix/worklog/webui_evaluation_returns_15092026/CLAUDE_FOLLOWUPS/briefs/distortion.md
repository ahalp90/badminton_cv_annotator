# Brief: lens distortion on the GX footage

Read-only on every input. Write only under `OUT`.

## Why

Apparent line bowing on the GX video has no verified cause. If the footage has barrel distortion, straight-line and vanishing-point fitting are systematically off on that video, and undistortion before extraction would be the fix. Test it from the cached line fragments, with the amateur and broadcast views as controls.

## Inputs

All paths relative to the repository root `<repo>`.

- Fragments per case, native pixels, field `segments_px` under `cases[*]` with `id`:
  - GX: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/inputs.json.gz`, cases `gxBQ_window_00_frame_0` and `gxBQ_window_00_frame_5` (1920 by 1080).
  - Amateur: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260908/marking_refit/marking_inputs.json.gz`, cases `am2_window_00_frame_150`, `am2_window_01_frame_28019`, `am3_window_00_frame_0` (1920 by 1080).
  - Broadcast: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/broadcast_extension/inputs.json.gz`, cases `shuttleset_03_scene_0016`, `shuttleset_03_scene_0019`, `shuttleset_21_scene_0020` (960 by 540).
- Merged rows with member fragments: `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e0/<case_id>.json.gz`, list `membership` with `line_id`, `member_raw_fragment_ids`, `covered_length_px`. Raw fragment ID is the index into `segments_px`.
- Frames, for a picture only: GX at `.../20260909/gx_extension/people/images/`, amateur at `.../20260908/marking_refit/images/` if present, broadcast at `.../20260909/broadcast_extension/images/` if present. Skip pictures where a frame is missing.

## Do

For each case, take every merged row with at least four member fragments whose endpoints span at least 40 percent of the image width or height. For each such row:

1. Collect all member endpoints. Fit a straight line by total least squares. Record the RMS perpendicular residual.
2. Fit a quadratic in the coordinate along the line direction to the perpendicular residuals. Record the sagitta: the quadratic's deviation from the chord at the middle of the span, in pixels, with a sign that is positive when the bow points away from the image centre.
3. Record the row's mean distance from the image centre.

Report per case: number of rows tested, median and maximum absolute sagitta, the fraction of rows with positive sagitta, and the Spearman correlation between absolute sagitta and distance from centre. Normalise sagitta by image width so the 960-wide broadcast views compare with the 1920-wide others. Draw one figure per case with the tested rows coloured by sagitta sign, saved as `OUT/<case_id>.png`, only where a frame exists.

## Gate

The three broadcast views are the control. If their median absolute sagitta is not small compared with GX's, the test is not measuring lens distortion; say so.

## Write

`OUT/table.csv` with one row per case and `OUT/note.md`: the per-case table, then three or four sentences: does GX show a consistent outward bow that grows with radius, do the amateur views show it, do the broadcast views not. Give the largest GX sagitta in native pixels. No fix proposed.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/distortion/`. Use `~/.venvs/badminton-cicd/bin/python` (numpy, scipy, headless cv2, matplotlib if present; fall back to cv2 drawing if not).
