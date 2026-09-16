# Brief: do later GX frames keep the precise directions

Read-only on every input. Write only under `OUT`.

## Why

Automatic direction selection on GX frame 0 discards the two precise directions (candidates 1183 for x and 122 for y) that fit the approved court to half a pixel. A per-view detector could only recover them if some other frame's automatic selection keeps them. Seven GX frames have cached selections.

## Inputs

All paths relative to the repository root `<repo>`.

- Cached selections: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_records.json.gz`. `rows` holds 13 records; keep the seven with `video_group == "gxBQ"`. Each has `direction_estimator.points_working` (16 homogeneous 3-vectors in 960 by 540 working pixels), `direction_estimator.retained_candidate_ids`, `direction_estimator.normalised_to_working`, and `frame_index`.
- Bank reconstruction for the control pair's points: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/automatic_axes/diagnose_direction_bank.py` has `reconstruct_bank`; the saved GX0 estimator is `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/vp_pruning/coverage/results/gxBQ_window_00_frame_0.json.gz`. Read `reconstruct_bank` to learn its arguments and the coordinate space of its output. The E2 record `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e2/gxBQ_window_00_frame_0.json.gz` gives `arms.B.points_normalised` and `arms.B.points_working` for the 16 baseline directions, which fixes the conversion.
- Bound code: `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/_bundles/badminton_direction_audit_7299ff3/direction_audit_7299ff3/geometry_certificate.py`. `set_bound(corners, points)` returns the lower bound on maximum corner error for the best ordered pair in a direction set, corners ordered FL, FR, NR, NL. Copy the file into `OUT`, do not edit it.
- Approved GX0 control corners: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/gx0_control_measurements.json.gz`, field `approved_corners_native_px`; divide by 2 for working pixels. Check the corner order against the E3 record `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e3/gxBQ_window_00_frame_0.json.gz`, which stores the control corners in working pixels.

## Gate, run first

1. Reconstruct the GX0 bank and confirm its retained IDs and working points match the temporal record's frame 0 row and the E2 record's B arm.
2. `set_bound` over B's 16 GX0 working points against the approved control corners in working pixels must print 6.022569 to six decimals.
3. `set_bound` over the two-point set {1183, 122} must print 0.423931.

If either number differs, the coordinate handling is wrong. Stop and report what you got.

## Do

For each of the seven GX frames:

- For each of its 16 directions, compute the angle in degrees between the unit-normalised homogeneous 3-vector and candidate 1183's, and likewise 122's. Report the minimum per control direction and which retained ID achieves it.
- Compute `set_bound` over the frame's 16 directions against the GX0 approved control corners. Label it "valid" for frames 0 and 5 and "indicative only, camera may have moved" for the others.
- Report the frame's nominal seconds from the record.

Write `OUT/table.csv` with one row per frame and `OUT/note.md`: the gate output, the table, then two or three sentences answering: does any frame keep both control directions within about one degree, and does any frame's bound fall under 3.5 working pixels? Do not interpret beyond that.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/gx_directions/`. Use `~/.venvs/badminton-cicd/bin/python`. Put `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/automatic_axes`, `.../20260914/vp_pruning`, `src` and the repository root on `PYTHONPATH` if the bank reconstruction needs them.
