# Brief: lens distortion on GX, measured on the painted ridge

Second attempt. The first attempt (`distortion/`) fitted quadratics to merged-row endpoints and could not separate lens distortion from merge noise, because merged rows mix paint with mat edges and people. Leave that folder as it is. This attempt samples the paint itself.

Read-only on every input. Write only under `OUT`.

## Inputs

All paths relative to the repository root `<repo>`. Three views, each with a frame and a frozen control court whose corners are in the E3 record under `control.corners_native_px`, ordered far-left, far-right, near-right, near-left. Assert the frame's decoded size equals `control.native_size`.

| View | Frame | E3 record |
| --- | --- | --- |
| GX0, test | `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/people/images/gxBQ_window_00_frame_00000000.png` | `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e3/gxBQ_window_00_frame_0.json.gz` |
| Amateur-3 frame 0, control | `scratch/court_det_fix/worklog/checks/independent/player_guided/20260908/people_original/images/am3_window_00_frame_00000000.png` | `.../e3/am3_window_00_frame_0.json.gz` |
| ShuttleSet 03 scene 16, control | `scratch/court_det_fix/worklog/checks/independent/people_job/images/shuttleset_03_scene_0016.png` | `.../e3/shuttleset_03_scene_0016.json.gz` |

The Amateur-3 and ShuttleSet controls are approved or reference courts of ordinary cameras; they are the comparison, not the target.

## Do

For each view and each of the four court edges (far baseline, near baseline, left sideline, right sideline), clip the edge to the image with a 20-pixel margin and skip it when the clipped length is under 300 native pixels (150 for the 960-wide view).

1. Along the clipped edge take 60 evenly spaced positions. At each, sample the greyscale intensity along the perpendicular over plus and minus 12 native pixels (6 for the 960-wide view) at half-pixel steps with bilinear interpolation.
2. Find the ridge: the brightest sample after subtracting the profile's median, refined by parabolic interpolation over its two neighbours. Discard the position when the peak is less than 25 grey levels above the median or when the peak sits at either end of the window. Record the signed offset of the ridge from the projected straight edge.
3. Fit a straight line to the surviving offsets against position along the edge, then a quadratic. Report: number of surviving positions, RMS residual of the line fit, and the sagitta, the quadratic's deviation from its chord at the middle of the span, signed positive when the bow points away from the image centre.
4. Draw the frame with the sampled ridge points in green, the discarded positions in grey, and the projected control edge in yellow; save as `OUT/<case_id>.png`, plus one close crop per edge.

## Gate

The two control views must have every edge's absolute sagitta under 1.5 native pixels. If they do not, say the method is not clean enough and report what you saw; do not tune the thresholds to make it pass.

## Write

`OUT/table.csv` with one row per view and edge, and `OUT/note.md`: the gate outcome, the table, then three or four sentences: does GX show a bow larger than the controls, is its sign consistent with barrel distortion (outward on every edge) or pincushion (inward), and how large is the largest GX sagitta in native pixels. No fix proposed.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/distortion_ridge/`. Use `~/.venvs/badminton-cicd/bin/python` (numpy, scipy, headless cv2).
