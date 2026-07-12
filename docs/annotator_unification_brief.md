# Annotator unification brief: what to graft from the tour tool before it retires

For the agent working Curtis's side. Direction agreed 2026-07-12: the off-frame annotator (`annotate_court_corners_offframe.py`) becomes the canonical tool; this doc lists everything `annotate_court_corners.py` has that it should absorb, with where each piece lives and why it exists. After the graft, the tour GUI retires (see the retirement notes at the end). Both tools sit merged on `feature/courtkeynet` at 1630fd1.

## How the tour tool works, in one paragraph

One cv2 window scrubs the video (trackbar, `,`/`.` step, `<`/`>` jump 25). 'c' captures the four corners directly; 'i' walks a named tour of the 30 painted-line intersections for frames with off-frame corners. Every point is placed with two clicks: a rough click opens a zoom loupe (source pixels scaled 8x with NEAREST so pixels stay blocky), and one click inside the loupe pins the point to sub-source-pixel precision. On commit it writes the standard corners CSV (`video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation`) plus a `<stem>_points.csv` sidecar naming each clicked intersection and the fit RMS. Any commit failure prints AND displays its reason, and writes nothing.

## Graft list, in priority order

1. **Point-count floor of 5** (`IntersectionSession._commit`). A 4-point homography is exact by construction: zero residual even when a landmark is mislabelled, so no residual gate can see the error. Your `_dlt_conditioning` rank check catches degenerate GEOMETRY but not a wrong-name click with valid geometry. The floor is one comparison plus a clear message ("need the 4 corners or at least 5 points; only N clicked").

2. **Post-fit reprojection gate, calibrated to point count and to what the data is FOR** (`fit_corner_quad`, `FIT_MAX_REPROJ_PX = 3.0`). Ours gates on the WORST point, not the mean, because least squares smears one bad click across the good ones (a 30 px misidentification among six clicks leaves the mean near 1 px). The gate's job is catching mislabelled crossings, which reproject 20 px or more; and the ground truth only has to out-resolve auto reads whose quality cliff sits at 10 refpx at the 1280x720 reference (the session-18 eyeball of the vid-3 fallback renders: below 10 the quads are fine, from 10 they almost consistently latch a wrong line), which is about 15 px native on 1920-wide footage. Honest loupe clicks jitter 1-2 px, and the real 18-25 point frames already collected peaked at 8.8 px worst while corner repeatability stayed 2-4 px. So the recommended form: at 6 points or fewer, hard-fail over 3 px (with that little redundancy the fit absorbs jitter, honest residuals sit near zero, and 3 px there is a real anomaly); above 6 points, warn-and-confirm from 10 px and hard-fail over 15 px, showing worst and RMS. Do not adopt the bare 3 px constant at high counts; it would have rejected most of the already-collected frames.

3. **Projected-quad sanity check** (`_quad_is_camera_valid`). Convexity (four edge turns share a sign) plus behind-baseline order (near baseline below the far one in image y). Catches a fit that converged to a mirror or a crossed quad, which residuals alone pass. Small, pure, directly liftable.

4. **Loupe two-click precision** (`loupe_origin` / `loupe_to_source` / `source_to_loupe` / `render_loupe`). Sub-source-pixel placement from a zoomed NEAREST crop, with exact round-trip maths (unit-tested). If your nudge/type-exact editing already gives equivalent precision, skip; otherwise this is what makes 1-2 px hand clicks routine. If you take it, take these too:
   - the 0.35 s click guard after the loupe opens (`LOUPE_CLICK_GUARD_S`): a double-click's second press lands inside the freshly opened loupe and pins a point the user never chose
   - open the loupe at a predictable spot away from the click (we place it beside the main window, falling back to the screen corner)

5. **Undo key ('u'), two-tier**. One press undoes one thing: a pending (loupe-open) click cancels first and re-poses the same point; otherwise the last placed point un-places. In-capture only; committed frames are edited by your upsert instead, which is strictly better than what we had (we only warned about duplicate rows).

6. **Help panel at launch plus a stateful HUD**. The single biggest complaint driver was silence: keys that did nothing visible, no statement of what the tool expected next, toolbar mystery-buttons. What fixed it: a help panel at launch (toggle 'h', any key dismisses, and the panel must state that OpenCV's toolbar disk icon saves a screenshot, not annotations); a HUD strip with a mode banner, a per-step instruction ("click near corner 3 of 4; a zoom window will open"), live counts, the commit requirement ("saves on d: all 4 corners or 5+ points, N so far"), and every abort/failure reason shown in the window, not just the terminal. Font scale derived from frame width or it becomes unreadable on resized windows.

7. **Window and quit robustness** (`window_visible`, the seen-once latches). Two real, non-obvious bugs to avoid reintroducing:
   - closing the LAST cv2 window via the WM tears down Qt's receiver and `getWindowProperty` then RAISES `cv2.error` ("NULL guiReceiver") instead of returning 0; wrap the visibility check, fold the raise into "closed"
   - a not-yet-mapped window also reads as not-visible, so a visibility-based quit fires at launch on a busy WM; arm close-detection only after the window has reported visible once
   Both were hit live on Ariel's stack (OpenCV 5.0 QT5, `~/.venvs/court-annotator`).

8. **Torch-free import discipline**. The tool must run in a venv with no torch (`~/.venvs/court-annotator`); the courtkeynet package `__init__` lazy-loads the wrapper, and `tests/test_courtkeynet_annotation.py::test_annotator_imports_without_torch` pins the property with a torch-blocked subprocess. Your scripts are already clean here; keep them that way and consider an equivalent pin for your entry point.

## Tests worth porting with their features

From `tests/test_courtkeynet_annotation.py`: the loupe round-trip trio (exact source-pixel recovery, edge clamping, sub-pixel mapping); the undo transitions (six tests, both machines); the misclick gates (minimal 5-point fit AND six-spread-points where the mean would pass but the worst point fails); `test_collinear_catches_diagonal_grid_triples` (the court grid's even spacing makes DIAGONAL triples exactly collinear; float fuzz sits ~1e-15, real non-collinear triples ~1e-4; check your conditioning floor sees this case; you have a hidden-diagonal test already, so likely covered).

## Known scorer follow-up (our side of the ledger)

`score_hand_corners.canonicalise_quad` re-derives TL TR BR BL geometrically, which can rotate the ring and mislabel slots when ground truth has far off-frame corners (your vid-1 BR at x=2157, BL at y=1147). When scoring those frames, match by your `corner_label` column instead. This is queued on our side; saying it here so nobody "fixes" the scorer blind.

## Retirement notes for the tour tool

Do not delete the FILE outright: `build_point_table`, the loupe maths, the CSV helpers, and `fit_corner_quad` are imported by `tests/test_courtkeynet_annotation.py`, and the torch-free regression test imports `build_point_table` specifically. Retire the GUI shell (`run_annotation_tool` and the draw functions) once the graft lands; move or re-home the pure helpers if your tool absorbs them. The corners-CSV contract and the sidecar format are consumed by `score_hand_corners.py` and the eval; keep both byte-compatible.

## Withdrawn from our earlier findings

The "no collinearity/general-position check" claim was wrong: `court_landmarks._dlt_conditioning` (floor 1e-8) is exactly that, and three tests pin it. The gap list above is what actually stands: the 4-point floor, the residual gate (calibrated, not the bare 3 px), and the quad sanity check.
