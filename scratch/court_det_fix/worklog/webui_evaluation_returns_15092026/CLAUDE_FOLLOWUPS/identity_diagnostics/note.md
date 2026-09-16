# Identity diagnostics on every exported winner

Read-only on every input. All numbers below come from the copy of
`audit.py` in this folder (unmodified) plus a new script, `extend.py`, that
loops the same functions over every winner in
`experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/ranking_records.json.gz`
and the two GX0 sources
(`gx0_control_measurements.json.gz` and its companion candidate89).

## Gate: reproducing the nine bundle candidates

I ran the copied `audit.py` against the bundle's own inputs
(`_bundles/court_identifiability_extension_7299ff3/inputs/`) and compared the
result to the bundle's committed
`_bundles/court_identifiability_extension_7299ff3/results/audit_results.json`.

| Check | Bundle value | My rerun | Match |
|---|---|---|---|
| Available interval counts (9 candidates, in order) | 12,12,12,12,12,5,12,12,12 | 12,12,12,12,12,5,12,12,12 | yes |
| Constraint ranks (9 candidates, in order) | 8,8,8,8,8,7,8,8,8 | 8,8,8,8,8,7,8,8,8 | yes |
| Amateur-2 `184:4123` separation (near baseline to near long-service) | 0.378 to 0.452 working px | 0.37842643865334613 to 0.45224696291744076 | yes |
| Amateur-2 `30:33` separation (far long-service to far baseline) | 1.86 to 2.24 working px | 1.8615469304044554 to 2.2439383079545183 | yes |
| Scene19 ideal-line-rank progression (5 lines / +centre / +second cross / all 11) | 7, 7, 8, 8 | 7, 7, 8, 8 | yes |

A full byte diff of the two `audit_results.json` files shows differences only
in SVD singular-value arrays and camera-error diagnostics, all at the
15th-to-17th significant figure (about 1e-13 to 1e-16 relative). That is
floating-point noise from the SVD/BLAS backend, not a numeric disagreement.
Every value the brief names for the gate matches exactly. The gate passes;
I extended to all winners.

## Method note on the new "minimum separation" column

`audit.py` only ever ran `line_separation` for two hand-picked marking pairs
(one per flagged candidate). To get a "minimum separation between any two
markings" for every winner, I wrote a new function, `minimal_marking_separation`,
in `extend.py` that calls the unmodified `audit.line_separation` and
`audit.clip_segments` over pairs of markings, and kept the smallest value.

I restricted the pairs to same-family markings only: the five vertical lines
(sidelines and centre) against each other, and the six horizontal lines
(baselines and service lines) against each other. A vertical line and a
horizontal line almost always cross somewhere inside the visible image once
extended to their full projective line, so scanning every pair without this
restriction returned 0 for nearly every candidate — a fact about crossing
lines, not evidence of a shared ridge. Restricting to same-family pairs is
what makes the number mean "how close are two lines that are supposed to run
parallel," which is what the scene19/Amateur-2 mechanism is about, and it
reproduces the two pairs `audit.py` itself picked (both horizontal-family).

## Corner-error source per row

For the 17 automatic winners, "maximum corner error to reference" is the
line/paint corner-error cell recorded in `automatic_axes_results.md`'s results
table (all-camera-eligible column, display pixels at 1280x720). For GX0
controls 1864 and 5144 it is `approved_max_corner_display_px` from
`gx0_control_measurements.json.gz`, measured against approved candidate89 (not
a manual reference). Candidate89 has no such number: it is itself the
reference the other two are measured against, so its cell says so rather than
inventing one.

## Table

Full data, including untruncated visual-ruling quotes, is in `table.csv`.

| Case | Population | Candidate | Role | Avail. intervals | Rank | Min separation (working px) | Between | Max corner error (display px) |
|---|---|---|---|---|---|---|---|---|
| am2_window_00_frame_150 | automatic_all_camera | 30:30 | line | 12 | 8 | 1.936 | far_long_service vs far_baseline | 185.1 |
| am2_window_00_frame_150 | automatic_all_camera | 30:33 | paint | 12 | 8 | 1.862 | far_long_service vs far_baseline | 10.8 |
| am2_window_01_frame_28019 | automatic_all_camera | 16:1800 | line | 12 | 8 | 2.159 | far_long_service vs far_baseline | 193.5 |
| am2_window_01_frame_28019 | automatic_all_camera | 184:4123 | paint | 12 | 8 | 0.369 | near_long_service vs near_baseline | 1189.5 |
| am3_window_00_frame_0 | automatic_all_camera | 43:22603 | line | 12 | 8 | 3.057 | far_long_service vs far_baseline | 7.7 |
| am3_window_00_frame_0 | automatic_all_camera | 43:22627 | paint | 12 | 8 | 3.077 | far_long_service vs far_baseline | 11.3 |
| gxBQ_window_00_frame_0 | automatic_all_camera | 22:4579 | line | 12 | 8 | 1.870 | far_long_service vs far_baseline | 18.8 |
| gxBQ_window_00_frame_0 | automatic_all_camera | 22:4588 | paint | 12 | 8 | 1.835 | far_long_service vs far_baseline | 12.7 |
| gxBQ_window_00_frame_5 | automatic_all_camera | 181:29836 | line | 12 | 8 | 11.075 | near_long_service vs near_baseline | 705.5 |
| gxBQ_window_00_frame_5 | automatic_all_camera | 10:1274 | paint | 12 | 8 | 4.599 | right_singles vs right_doubles | 704.4 |
| shuttleset_03_scene_0016 | automatic_all_camera | 1:60 | line | 12 | 8 | 9.510 | far_long_service vs far_baseline | 8.6 |
| shuttleset_03_scene_0016 | automatic_all_camera | 1:90 | paint | 12 | 8 | 9.474 | far_long_service vs far_baseline | 11.3 |
| shuttleset_03_scene_0017 | automatic_all_camera | 1:80 | line | 12 | 8 | 9.587 | far_baseline vs far_long_service | 11.1 |
| shuttleset_03_scene_0017 | automatic_all_camera | 1:132 | paint | 12 | 8 | 9.653 | far_baseline vs far_long_service | 10.2 |
| shuttleset_03_scene_0019 | automatic_all_camera | 1:60 | line | 12 | 8 | 9.469 | far_baseline vs far_long_service | 12.7 |
| shuttleset_03_scene_0019 | automatic_all_camera | 165:6702 | paint | 5 | 7 | 4.629 | right_doubles vs right_singles | 8390.9 |
| shuttleset_21_scene_0020 | automatic_all_camera | 0:2 | line and paint | 12 | 8 | 9.919 | far_long_service vs far_baseline | 7.2 / 7.2 |
| gxBQ_window_00_frame_0 | gx0_label_guided_bank_control | 1864 | line (GX0 control) | 12 | 8 | 1.826 | far_long_service vs far_baseline | 3.121 |
| gxBQ_window_00_frame_0 | gx0_label_guided_bank_control | 5144 | paint (GX0 control) | 12 | 8 | 1.809 | far_long_service vs far_baseline | 4.050 |
| gxBQ_window_00_frame_0 | GX0 supplied-direction comparator | 89 | comparator (approved reference) | 12 | 8 | 1.858 | far_long_service vs far_baseline | not recorded; 89 is itself the reference for 1864 and 5144 |

Visual rulings (quoted from `automatic_axes_visual_judgements.md` and
`gx0_control_measurements.json.gz`) are in `table.csv`, one per row, since
several run long. Candidate89 has "no ruling recorded" — it was not itself
inspected in the gallery; it is the approved court the two GX0 controls are
measured and judged against.

## Answers

**Does every approved or usable court have rank 8?**

Yes. Every automatic winner the user called usable (Amateur-3 line and paint,
ShuttleSet 03 scene16 line and paint, ShuttleSet 03 scene17 line, ShuttleSet
03 scene19 line, ShuttleSet 21 scene20, and Amateur-2 frame150 paint), both
approved GX0 controls (1864, 5144), and comparator89 all have rank 8. But so
does almost everything else in the table, usable or not: rank 8 is not
special to good courts here. Of the twenty rows, nineteen have rank 8;
only the one already-known algebraically degenerate case (scene19's
`165:6702`) has rank 7.

**Does every false winner have rank under 8 or a separation under one pixel?**

For the two winners the audit names as decisive false paint winners: yes.
`165:6702` has rank 7 (fails the rank test on its own). `184:4123` has rank 8
but a 0.369 working-pixel separation between near_baseline and
near_long_service, under one pixel. Both are caught by "rank under 8 OR
separation under one pixel."

**Which candidates break either pattern?**

None of the two named false winners breaks the pattern. But if the question
is whether "rank 8 required, else check for sub-pixel separation" would flag
every visually rejected automatic winner — not just the two named false
winners — then no, it would not. Several winners the user judged unusable for
ordinary shear/framing reasons have rank 8 and separations well over one
pixel: GX frame0 line (`22:4579`, rank 8, 1.870 px), GX frame0 paint
(`22:4588`, rank 8, 1.835 px), GX frame5 line (`181:29836`, rank 8, 11.075 px),
GX frame5 paint (`10:1274`, rank 8, 4.599 px), Amateur-2 frame28019 line
(`16:1800`, rank 8, 2.159 px), and ShuttleSet 03 scene17 paint (`1:132`, rank
8, 9.653 px). These are not the "background structures pass profile" false
winners the audit's mechanism targets — they are ordinary shear and
mis-framing failures — so they are not a counterexample to the audit's own
claim about false-paint-winner mechanisms. But they show the "rank 8
required, else sub-pixel separation" test only catches the two specific
mechanisms it was built to catch; it gives no signal at all for the other
kind of bad court in this set, and rank 8 by itself does not separate good
courts from bad ones anywhere in this table. No threshold is proposed here;
this is a description of what these two diagnostics do and do not distinguish
across the winners exported so far.

## Files written

- `table.csv` — one row per candidate, full untruncated visual-ruling quotes
- `note.md` — this file
- `rows.json` — the same rows as machine-readable JSON (intermediate, kept for traceability)
- `audit.py` — unmodified copy of the bundle's audit script, imported from (not edited)
- `extend.py` — new script: loads every winner, calls audit.py's functions, writes table.csv/rows.json
- `gate_results/audit_results.json` — fresh rerun of audit.py against the bundle's own inputs, used for the gate comparison above
