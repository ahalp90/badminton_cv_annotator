# Brief: mask fragments inside person boxes and replay GX0 direction selection

Read-only on every input. Write only under `OUT`.

## Why

On GX0 the precise directions lose coverage-first allocation to three merged rows that are not court paint. One of them, row 61, lies on a spectator's face inside a person box the input pack already carries. This is the cheapest possible test of "decide which fragments are court paint before fitting": drop fragments inside person boxes, rerun the merge and the direction selection unchanged, and measure whether the selected directions move toward the approved court.

## Inputs

All paths relative to the repository root `<repo>`.

- Pack: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260909/gx_extension/inputs.json.gz`, case `gxBQ_window_00_frame_0`: `segments_px` (751 fragments, native pixels), `bbox_px` (18 person boxes for this frame, native `[x1, y1, x2, y2]`), `dimensions`.
- Saved estimator, the baseline to reproduce: `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/vp_pruning/coverage/results/gxBQ_window_00_frame_0.json.gz`.
- Replay code: `scratch/court_det_fix/direction_agreement/`. `run_arms.py` shows the exact sequence: `membership.merge_lines_with_membership` on the fragments, the first 128 merged lines as direction lines, `vp_pruning.estimate` with the saved settings, foot-anchor residuals, `selection.retain_pencils_with_buckets` for the coverage allocation. `common.py` and `run_remote.sh` show the `PYTHONPATH` the modules expect: `direction_agreement`, `src`, the repository root and the `20260914` helper folders `vp_pruning`, `marking_diagnosis`, `axis_matching`, `automatic_axes`. The E0 record `runs/direction_agreement_20260915_144900/e0/gxBQ_window_00_frame_0.json.gz` lists what the replay must reproduce.
- Control directions and bound: `CLAUDE_FOLLOWUPS/gx_directions/run_gx_directions.py` already reconstructs the bank, reads the control pair's working points and calls `set_bound`; reuse its functions rather than rewriting them. Approved control corners in working pixels: E3 record `runs/direction_agreement_20260915_144900/e3/gxBQ_window_00_frame_0.json.gz`, field `control.corners_working_px`.
- Control fit: `direction_agreement/run_fits.py` has `fit_pairs`, the least-squares fit of every ordered direction pair to the control; the E3 record gives the baseline's best value 6.798 working px.

## Gate, run first

With no masking, your replay must reproduce the saved estimator exactly: retained candidate IDs, working points and support masks equal, as `run_arms.py` asserts. Print the result. Stop if it fails.

## Do

Two masking rules, reported separately:

- Rule A: drop a fragment when both endpoints lie inside any person box.
- Rule B: drop a fragment when its midpoint lies inside any person box.

For each rule: count fragments dropped; run the merge and the coverage allocation unchanged; report how many merged lines result and whether the structures behind baseline rows 31, 51 and 61 survive (match by fragment membership, since row numbers change). Then measure the 16 selected directions three ways: the minimum angle to the control pair's x and y directions, `set_bound` against the approved control corners (baseline 6.023), and the best control fit over all 240 ordered pairs (baseline 6.798). Also run the diagnostics on the fragments dropped: how many lie within 6 working pixels of a projected edge of the approved court, so a reader can see what paint the mask costs.

## Write

`OUT/table.csv` with one row per rule plus the baseline, and `OUT/note.md`: the gate output, the table, which of the three offending structures each rule removes, and two or three sentences on whether masking person boxes alone moves GX0's selection toward the approved court. No threshold proposed; no other cue tried.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/person_mask_replay/`. Use `~/.venvs/badminton-cicd/bin/python`. If an import fails because the local venv lacks a package the replay needs, say which and stop; do not install anything.
