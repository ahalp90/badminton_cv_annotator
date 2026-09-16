# Brief: which structures the three GX0 merged rows lie on

Read-only on every input. Write only under `OUT`.

## Why

An external audit showed that GX0 merged rows 31, 51 and 61 (zero-based merged-line indices) are the rows that make the capped-angle direction score prefer the original direction pair (candidates 737 for x, 4104 for y) over the closer control pair. Nobody has looked at what those rows are in the image.

## Inputs

All paths relative to the repository root `<repo>`.

- Frame: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/people/images/gxBQ_window_00_frame_00000000.png` (native 1920 by 1080).
- Fragments: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/inputs.json.gz`. Find the case with `id == "gxBQ_window_00_frame_0"`; `segments_px` is a list of 751 fragments `[x1, y1, x2, y2]` in native pixels. Fragment index in that list is the raw fragment ID.
- Membership: `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e0/gxBQ_window_00_frame_0.json.gz`. `membership` is a list of 106 merged rows with `line_id`, `member_raw_fragment_ids`, `covered_length_px`. `line_id` is the merged-line index.
- Support masks: `scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e2/gxBQ_window_00_frame_0.json.gz`. `arms.B.leader_candidate_ids` lists 16 candidate IDs; `arms.B.support_masks` is the matching list of boolean masks over the 106 merged rows. Take the masks for candidates 737 and 4104.
- Approved control court: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/gx0_control_measurements.json.gz`, field `approved_corners_native_px`.

## Do

1. Print the member fragment IDs and covered length for rows 31, 51 and 61, and check that row 31 and row 61 are in candidate 737's mask and row 51 in candidate 4104's mask. If any is not, stop and report; the brief's premise would be wrong.
2. Draw on the frame: the approved control court in yellow (thin), every fragment in candidate 737's other support rows in green, every fragment in candidate 4104's other support rows in blue, and the member fragments of rows 31, 51 and 61 in red (thick). Label each red row with its number near its midpoint.
3. Save the full overlay as `OUT/overlay.png` and a crop of about 500 by 350 native pixels around each red row as `OUT/row_31.png`, `OUT/row_51.png`, `OUT/row_61.png`. Save the same crops without any drawing as `OUT/row_31_clean.png` and so on, so the underlying structure is visible.
4. Look at the crops. For each red row write one or two sentences: what it follows (court paint, a mat edge, a wall or floor seam, the net or a post, a person, equipment, something else), whether it is on the playing floor, and how it relates to the nearest court line of the yellow control.
5. Write `OUT/note.md`: a table with row, member count, covered length, your read; then the rows' relation to the control court; then one paragraph on whether these rows are court paint or not. Say plainly that the read is your visual judgement.

## Gate

The member counts and covered lengths you print must come from the E0 record, and step 1's mask check must pass. State both in the note.

## Do not

Do not change any input. Do not fit anything. Do not speculate beyond what the crops show.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/gx0_rows/`. Use `~/.venvs/badminton-cicd/bin/python` (has numpy and headless cv2).
