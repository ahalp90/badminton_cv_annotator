# Direction agreement: two individual changes on nine views

Run `direction_agreement_20260915_144900` on `fix/court-det` (scientific checkpoint `9fcec94`; stage records produced under commit `0caae0a`, accounting and report under the later commits listed in `runs.md`). Session model Claude Fable 5.1. Written for a colleague who knows the court-detection work but has not followed this branch. Nothing here decides production readiness or asks for visual judgement.

**Bottom line.** The result: each change regresses at least one view and improves others, measured by the paint winner's distance to the frozen control at the generation stage (working pixels; gaps under 0.05 are called the same). The midpoint anchor (M) turns GX5 from an outright failure into a court 9.9 pixels from its approved control (baseline 525.7) and improves both Amateur-2 views, but it regresses GX0 (7.7 to 379.3, approved control) and Amateur-3 (8.5 to 23.8, manual reference). The precise representative (R) improves Amateur-2 frame 28019 and ShuttleSet 03 scene 17 by more than half a pixel (both manual references) and regresses GX0 (7.7 to 13.3), Amateur-3 (8.5 to 58.2; its pool holds no court within 44 pixels) and ShuttleSet 21 (2.0 to 2.8). By the line ranking R improves GX0 (16.5 to 10.4) and regresses Amateur-3 further (5.8 to 136.9). For M, the direction of change agrees with its control-fit potential (E3) on eight of nine views; for R, on five. The rescoring stages rescue nothing: the all-camera paint winner moves from a close camera-first court to hundreds or thousands of pixels away in four of the 27 case-arms, two of them baseline. No usability claim is made.

## What the work tried to learn

Automatic direction selection loses the precise observed directions that the matcher can use. Two separate changes were proposed as possible causes of that loss. The question: does either change, on its own, let the selection keep those directions? This run tests each change alone on nine frozen views:

- **The first hypothesis (H1), the midpoint-anchor arm (M):** measure a merged line's agreement with a candidate direction at the projected midpoint of the line's longest fragment, near the visible evidence. The baseline measures it at the line's foot from the image centre.
- **The second hypothesis (H2), the precise-representative arm (R):** after the coverage rule has allocated its 16 leader groups, pick the most precise member of each leader's suppression bucket as the representative. The baseline keeps the leader itself.

The replayed baseline (B) is the comparator. Both changes combined (MR) and the fixed-support singular value decomposition (SVD) refits of every arm stay diagnostic-only; only M and R reach the matcher. The packet fixed that allocation before any result was seen, so control-fit numbers could not choose which arms got the expensive trial.

The run has five stages. Membership recovery and baseline replay (E0) recovers which fragments each merged line came from and proves the baseline replays exactly. The anchor comparison (E1) measures what the midpoint anchor changes. Arm freezing (E2) fixes the four arms' directions. The control fits (E3) fit a court to each view's control from every ordered pair of directions in each arm, as a diagnostic of what the directions could support. The matcher trial (E4) runs M and R through the unchanged matcher and its two rescoring stages.

## Completion and accounting

| Stage | What | Cases | Status | Exit |
| --- | --- | ---: | --- | ---: |
| E0 | Merge membership recovered; estimator, selection and global-selection replays | 9 | complete, all replays exact | 0 |
| E1 | Foot-anchor and midpoint-anchor residual matrices, membership changes | 9 | complete | 0 |
| E2 | Four frozen arms (B, M, R, MR) with buckets and representatives | 9 | complete; every arm keeps 16 directions | 0 |
| E3 | Control fits for 8 sets per case, 240 ordered pairs each | 9 (72 sets, 17,280 fits) | complete, 0 non-finite, 4 non-converged; 3 saved-record replays exact | 0 |
| E4 | M and R through the unchanged matcher plus two rescoring stages | 18 case-arms + 36 rescores | complete; 72 replays exact; accounting 81 rows (9 views, B/M/R, 3 stages), none missing | 0 |

## How to read the numbers

- **Population:** the nine frozen development views in the table below. Working size is 960 by 540 pixels for all nine.
- **Primary metric:** maximum corner distance between a court and the view's frozen control, in working pixels, allowing the existing 180-degree corner relabelling. Smaller is closer. Gaps under 0.05 pixels are reported as the same; the numbers are still given.
- **Controls:** six controls are visually approved courts. Three are manual references without visual approval. Distances to an unapproved control say how far a court is from a hand-drawn reference, not from a judged-good court.
- **Evidence status:** E3 fits are least-squares diagnostics. Each fits a court to the control given a fixed direction pair, so it measures direction-fit potential and is not a generated court. E4 courts are generated by the unchanged matcher without any control; the control enters only afterwards to measure them. Membership counts describe the merge assignment, not physical stripes. There is no numerical usability threshold.

| View | Source frame | Native size | Control | Approved |
| --- | --- | --- | --- | --- |
| GX0 | GX video, window 0, frame 0 | 1920 by 1080 | approved candidate 89 (supplied-direction line winner) | yes |
| GX5 | GX video, window 0, frame 5 | 1920 by 1080 | approved generated court | yes |
| Am2-150 | Amateur-2, window 0, frame 150 | 1920 by 1080 | approved vanishing-point diagnostic court | yes |
| Am2-28019 | Amateur-2, window 1, frame 28019 | 1920 by 1080 | manual reference | no |
| Am3-0 | Amateur-3, window 0, frame 0 | 1920 by 1080 | manual reference | no |
| SS03-17 | ShuttleSet 03, scene 17 | 960 by 540 | manual reference | no |
| SS03-19 | ShuttleSet 03, scene 19 | 960 by 540 | approved automatic marking refit | yes |
| SS03-16 | ShuttleSet 03, scene 16 | 960 by 540 | approved automatic marking refit | yes |
| SS21-20 | ShuttleSet 21, scene 20 | 960 by 540 | approved automatic marking refit | yes |

## E0: the baseline replays exactly

The instrumented merger returns the same coefficients in the same order as the shared `detector._merge_lines` on all nine views, while carrying each line's member fragments. Replaying the saved estimator reproduces every field: IDs, counts, statuses, masks and order exactly, and lines, points and the transform within 1e-12. The bucket-recording copy of the coverage rule returns the same retained rows and statuses as the original. The packet's global-selection replay passed for all nine views. GX0 and GX5 have 106 and 108 merged lines and SS03-19 has 124, so "128 direction lines" is a cap, not a count.

## E1: the anchor moves, membership barely does

Anchors move a median 137 to 195 working pixels from the foot, up to 497 pixels. Between 40 and 81 lines per view come from a single fragment, so their anchor is that fragment's own midpoint.

Over all candidate-line pairs the residual changes a lot: the median absolute change is 3.9 to 7.7 degrees. Almost all of those pairs are far outside the 1.5-degree membership band under both anchors. Restricted to pairs that are members under at least one anchor, the median change is 0.01 to 0.04 degrees and the 99th percentile 1.5 to 5.4 degrees. Membership changes for 0.8 to 2.7 percent of entries per view. Per view, 2,400 to 5,300 candidates gain at least one supporting line and 970 to 3,300 lose one.

The largest changes are finite candidates sitting at one of the two anchors. A candidate at the foot has an undefined ray there (residual about 90 degrees) and a well-defined one at the midpoint anchor (about 1 degree). A candidate at the midpoint anchor shows the reverse. All 20 largest changes in every view have this form. The midpoint anchor therefore moves the blind spot from an arbitrary point to where the visible evidence is. That is a property of the anchor rule, not a code fault.

The number of candidates with at least two supporting lines is identical under both anchors. A pair candidate always lies on both lines that generate it, and infinity candidates ignore the anchor.

## E2: allocation is anchor-sensitive; the precision rule changes most buckets

M shares 0 to 5 of 16 leaders with B by candidate identifier (ID). By angle the overlap is larger. The median M direction lies 0.9 degrees from the nearest B direction, but 34 of the 144 M directions lie more than 5 degrees from every B direction.

The precision rule changes the representative in 83 of 144 buckets for R and 87 for MR. Where R changes it, the new representative supports at most as many lines as the leader in 80 of the 83 buckets (median one line fewer). R's directions stay within 0.08 degrees of B's at the median. Only 2 of 144 are more than 5 degrees away. Fourteen buckets are singletons and cannot change.

## E3: control-fit potential moves both ways

Best finite control fit of any ordered direction pair, working pixels. Every set had 240 finite attempts.

| View | Control approved | B | B+SVD | M | M+SVD | R | R+SVD | MR | MR+SVD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GX0 | yes | 6.80 | 3.96 | 19.65 | 19.91 | 6.18 | 4.22 | 19.75 | 19.81 |
| GX5 | yes | 23.80 | 22.50 | 10.59 | 10.10 | 23.24 | 23.44 | 11.26 | 10.84 |
| Am2-150 | yes | 3.26 | 2.98 | 1.89 | 3.87 | 2.80 | 2.42 | 2.25 | 3.87 |
| Am2-28019 | no | 2.97 | 1.63 | 2.27 | 2.13 | 1.42 | 1.41 | 1.40 | 1.86 |
| Am3-0 | no | 3.75 | 4.39 | 4.18 | 4.33 | 4.27 | 4.38 | 5.00 | 4.32 |
| SS03-17 | no | 1.98 | 0.86 | 2.83 | 0.86 | 0.62 | 0.76 | 0.59 | 0.68 |
| SS03-19 | yes | 2.27 | 2.18 | 2.29 | 2.26 | 1.92 | 1.34 | 1.98 | 1.42 |
| SS03-16 | yes | 1.50 | 0.81 | 1.82 | 0.81 | 0.59 | 0.64 | 1.00 | 0.81 |
| SS21-20 | yes | 1.03 | 1.03 | 1.03 | 1.03 | 1.37 | 1.23 | 1.03 | 1.03 |

- **M against B:** better on GX5 (23.8 to 10.6), Am2-150 and Am2-28019. Worse on GX0 (6.8 to 19.7), Am3-0, SS03-17 and SS03-16. Same on SS03-19 (2.29 against 2.27, a 0.02 gap) and SS21-20 (identical). GX0 and GX5 move in opposite directions.
- **R against B:** better on seven views. Worse on Am3-0 (3.7 to 4.3) and SS21-20 (1.0 to 1.4). In every view R's best pair uses the same two groups as B's best pair.
- **SVD:** moves directions a median 0.1 to 0.4 degrees, at most 3.3. It helps B on eight views (two of them by under 0.1 pixels) and hurts on Am3-0. For the other arms the pattern is uneven: M+SVD is worse than M on Am2-150, 3.9 against 1.9. The best finite pair converged in every one of the 72 sets, so best-finite and best-converged values coincide.
- **Replays:** the B and B+SVD fits for GX0, GX5 and Am2-28019 reproduce the earlier `svd_fixed` records to 0.0 pixels.

## E4: what the unchanged matcher made of M and R

All 18 case-arms generated: 240 ordered direction pairs each, 100 to 152 of them rejected by the camera bound, no basis failures, no arm reusing the baseline by identity. Every replay at both rescoring stages passed. Errors below are maximum corner distances to the frozen control in working pixels. B is the saved baseline diagnosed with the same control. Both rankings are shown because the earlier sessions found usable courts under either.

Winners at the generation stage (256 entries after the global cap):

| View | Approved | B line | B paint | M line | M paint | R line | R paint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GX0 | yes | 16.5 | 7.7 | 577.6 | 379.3 | 10.4 | 13.3 |
| GX5 | yes | 513.6 | 525.7 | 9.9 | 9.9 | 365.1 | 378.5 |
| Am2-150 | yes | 140.9 | 12.4 | 141.7 | 8.8 | 138.8 | 12.6 |
| Am2-28019 | no | none | none | 142.2 | 7.0 | 143.4 | 6.1 |
| Am3-0 | no | 5.8 | 8.5 | 28.6 | 23.8 | 136.9 | 58.2 |
| SS03-17 | no | 8.4 | 7.6 | 9.1 | 8.4 | 6.9 | 6.9 |
| SS03-19 | yes | 3.9 | 3.9 | 3.9 | 3.9 | 3.7 | 3.8 |
| SS03-16 | yes | 2.6 | 3.9 | 3.0 | 3.3 | 3.3 | 3.9 |
| SS21-20 | yes | 2.0 | 2.0 | 2.0 | 2.0 | 2.8 | 2.8 |

B has no winner at generation on Am2-28019 because none of its 256 entries passes the camera gate; its camera-first winners are 145.1 (line) and 17.2 (paint). On SS21-20, M's winners are the same court as B's (identical corners). A small distance is not a usability verdict: B's GX0 winners (16.5 and 7.7) were judged sheared and not usable in the earlier session, and the gallery carries those rulings.

- **M against B (paint winner):** better on GX5 (525.7 to 9.9), Am2-150 (12.4 to 8.8), Am2-28019 (none to 7.0) and SS03-16 (3.9 to 3.3). Worse on GX0 (7.7 to 379.3), Am3-0 (8.5 to 23.8) and SS03-17 (7.6 to 8.4). Same on SS03-19 (3.93 against 3.92, a 0.007 gap) and SS21-20 (identical court). The direction of change agrees with E3 on eight views; SS03-16 moves the opposite way. By the line ranking M improves GX5 (513.6 to 9.9) and regresses GX0 (16.5 to 577.6), Am3-0 (5.8 to 28.6), SS03-17 (8.4 to 9.1) and SS03-16 (2.6 to 3.0).
- **R against B (paint winner):** better on Am2-28019 (none to 6.1), SS03-17 (7.6 to 6.9) and SS03-19 (3.93 to 3.80). Same on SS03-16 (3.94 against 3.90). Worse on GX0 (7.7 to 13.3), Am2-150 (12.4 to 12.6), Am3-0 (8.5 to 58.2) and SS21-20 (2.0 to 2.8). GX5 fails under both (525.7 and 378.5). By the line ranking R improves GX0 (16.5 to 10.4), SS03-17 (8.4 to 6.9) and SS03-19 (3.9 to 3.7) and regresses Am3-0 (5.8 to 136.9), SS03-16 (2.6 to 3.3) and SS21-20 (2.0 to 2.8). E3 had R better on seven views, five of them by under a pixel. The direction of change agrees with E3 on five views (Am2-28019, SS03-17, SS03-19, Am3-0, SS21-20), is opposite on GX0 and Am2-150, flat on SS03-16, and GX5 fails under both.

**Where the error enters.** The packet asks to keep direction-fit potential, candidate availability, filtering and ranking apart. Availability is the nearest court in any matched pair's shortlist before the global cap (the pool, 17,677 to 31,731 courts per case-arm):

| View | B fit | B pool | M fit | M pool | R fit | R pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GX0 | 6.8 | 7.7 | 19.7 | 33.8 | 6.2 | 8.7 |
| GX5 | 23.8 | 23.7 | 10.6 | 9.8 | 23.2 | 22.8 |
| Am2-150 | 3.3 | 8.3 | 1.9 | 4.3 | 2.8 | 8.9 |
| Am2-28019 | 3.0 | 15.3 | 2.3 | 2.1 | 1.4 | 2.4 |
| Am3-0 | 3.7 | 4.3 | 4.2 | 20.5 | 4.3 | 44.2 |
| SS03-17 | 2.0 | 3.6 | 2.8 | 4.3 | 0.6 | 2.4 |
| SS03-19 | 2.3 | 3.1 | 2.3 | 3.1 | 1.9 | 2.7 |
| SS03-16 | 1.5 | 2.3 | 1.8 | 2.9 | 0.6 | 2.7 |
| SS21-20 | 1.0 | 2.0 | 1.0 | 2.0 | 1.4 | 2.2 |

A pooled court can sit closer to the control than the E3 fit (GX5 under B, M and R; Am2-28019 under M). The two are different constructions: the E3 fit is a least-squares fit from one fixed direction pair, and a pooled court is a matcher proposal, so the columns are not a nested bound. Availability losses above 10 pixels: M on GX0 (19.7 to 33.8), M on Am3-0 (4.2 to 20.5), R on Am3-0 (4.3 to 44.2) and B on Am2-28019 (3.0 to 15.3). B and R lose about 5 to 6 pixels on Am2-150 (3.3 to 8.3, 2.8 to 8.9). Pre-cap geometry is not saved, so none of these losses can be placed between proposal and the per-pair cap.

Filtering: the generation stage's global cap selects from all pooled courts, camera-eligible or not, and the nearest pooled court survives it in 21 of 27 case-arms. The camera-first stage keeps only camera-eligible courts, and the nearest camera-eligible pooled court survives its cap in 23 of 27. The exceptions:

| Case-arm | Nearest pooled | Nearest final (generation) | Nearest camera-eligible pooled | Nearest final (camera-first) |
| --- | ---: | ---: | ---: | ---: |
| GX0 M | 33.8 | 314.3 | 34.6 | 339.1 |
| GX0 R | 8.7 | 10.1 | 8.7 | 10.1 |
| GX5 B | 23.7 | 460.5 | 89.9 | 342.1 |
| GX5 R | 22.8 | 355.8 | 89.3 | 281.6 |
| Am2-28019 B | 15.3 | 82.8 | 16.0 | 16.0 |
| Am3-0 R | 44.2 | 58.0 | 55.4 | 55.4 |

On GX5 the camera bound removes every court within 24 pixels of the control for B and R (nearest camera-eligible 89.9 and 89.3), while M's pool has a camera-eligible court at 9.9.

Ranking: the winners sit 0 to 5 pixels beyond the nearest final court on most views (for example SS03-17: nearest 3.6, 4.3 and 2.4 against paint winners 7.6, 8.4 and 6.9). The line ranking picks a court 137 to 143 pixels from the control on Am2-150 for all three arms and on Am2-28019 for M and R, with paint winners of 6 to 13 pixels in the same pools.

**Rescoring stages.** Camera-first rescoring changed no winner's error except on the views that were already failing (GX0 M, GX5 B and R), on Am3-0 R (paint 58.2 to 74.3), and on Am2-28019 B, which gains its first winners there (145.1 and 17.2). The all-camera stage keeps every pooled court whose prefilter camera error is within the bound (724 to 3,265 entries). One of M's 1,303 entries on Am2-28019 (candidate 74:4750, prefilter error 0.065) carries no camera error in its final gates because its geometry check failed, so the accounting counts 1,302 camera-eligible there. Its line winners equal the camera-first line winners in all 27 case-arms. Its paint winner moved from a close camera-first court to hundreds or thousands of pixels away in four case-arms, and shifted between failing values in two more (GX0 M, GX5 B):

| Case-arm | Paint winner, camera-first | Paint winner, all-camera | All-camera entries |
| --- | ---: | ---: | ---: |
| SS03-19 B | 3.9 | 6,297.0 | 2,757 |
| SS03-19 M | 3.9 | 702.8 | 2,594 |
| Am2-28019 B | 17.2 | 892.1 | 1,434 |
| GX5 R | 448.1 | 2,237.9 | 2,216 |

The floor gate passes no entry on GX0, GX5 or Am2-28019 in any arm or stage, as in the baseline; it passes 3 to 145 entries on the other six views.

## Negative findings

- M regresses GX0 by both rankings (paint 7.7 to 379.3, line 16.5 to 577.6; no pooled court within 33 pixels), Am3-0 by both (paint 8.5 to 23.8, line 5.8 to 28.6), SS03-17 by both (paint 7.6 to 8.4, line 8.4 to 9.1) and SS03-16 by the line ranking (2.6 to 3.0).
- R regresses GX0 by paint (7.7 to 13.3), Am3-0 by both rankings (paint 8.5 to 58.2, line 5.8 to 136.9; B's line court there was judged essentially perfect earlier; no pooled court within 44 pixels), SS21-20 by both (2.0 to 2.8), Am2-150 by paint (12.4 to 12.6) and SS03-16 by line (2.6 to 3.3), and leaves GX5 failing (378.5).
- E3's sub-pixel R gains on GX0, GX5, Am2-150 and SS03-16 do not reach the paint winners; GX0's reaches the line winner (16.5 to 10.4).
- The line ranking selects a court 137 to 143 pixels from the control on both Amateur-2 views under every arm that has a winner there.
- The all-camera paint winner moves from a close camera-first court to hundreds or thousands of pixels away in four of 27 case-arms, two of them baseline (SS03-19, Am2-28019); two more case-arms shift between failing values.
- No entry passes the floor gate on GX0, GX5 or Am2-28019 under any arm; this is inherited from the baseline and unchanged by M or R.
- M rejects more direction pairs at the camera bound on six views (up to 138 of 240 on Am2-150 against 112 for B) and pools fewer courts there.

## Costs

E0 to E2 take about 6 seconds per view on the remote host (single thread). The residual matrices take 0.02 to 0.04 seconds each and each coverage allocation 0.04 seconds. E3 takes 3 to 7 seconds per view; the 16 SVD refits take 0.6 to 2.0 milliseconds per set.

E4 generation is the expensive part. Seconds per case-arm as recorded by the matcher's own timer (`elapsed_s` in each generation record; the stream logs' wrapper timer, which includes writing the record, reads 2 to 5 seconds more). Single-threaded, four processes running at once on an otherwise idle 32-core host; B's times come from the earlier baseline run under three concurrent groups:

| View | B | M | R |
| --- | ---: | ---: | ---: |
| GX0 | 2,293 | 2,141 | 2,306 |
| GX5 | 1,920 | 1,729 | 1,838 |
| Am2-150 | 1,250 | 854 | 1,152 |
| Am2-28019 | 1,416 | 1,541 | 1,364 |
| Am3-0 | 1,265 | 1,021 | 1,228 |
| SS03-17 | 254 | 236 | 213 |
| SS03-19 | 216 | 167 | 197 |
| SS03-16 | 245 | 225 | 212 |
| SS21-20 | 174 | 183 | 143 |

Camera-first rescoring takes 21 to 59 seconds per case-arm and all-camera rescoring 40 to 340 seconds. The 17 non-smoke case-arms ran as four streams and finished 1 hour 43 minutes after the first launch (06:10:44 to 07:53:26 UTC); the full accounting took 96 seconds. Concurrent single-run timings on one host; not a speed claim.

## Limitations

- The experiment cannot rescue a direction outside the allocated buckets, and the precision rule can only pick among candidates the coverage rule already grouped.
- Three controls are unapproved manual references.
- Pre-per-pair-cap geometry is unavailable, so per-pair losses inside the matcher are not measured.
- The anchor rule and the precision score are declared hypotheses; no weights, refits, backfills or objective sweeps were tried.
- Single-run timings on one host; no end-to-end speed claim.

## Unresolved questions

- Why does M's GX0 pool hold no court within 33 pixels of the control when B's holds one at 7.7? The E2 record for GX0 (`nearest_baseline_direction_deg`) shows two of M's 16 directions more than 10 degrees from any B direction; whether the pair that produced B's court lost a direction under M can be checked from that record and the pair provenance of candidate 22:4588.
- Why do R's Am3-0 representatives fit the control to 4.3 pixels in E3 yet yield no pooled court within 44? The per-pair cap keeps 256 courts per pair and the pre-cap geometry is not saved, so the loss cannot be placed between proposal and cap with these records.
- On GX5, B's and R's nearest pooled courts (about 23 pixels) fail the camera bound while M's 9.8-pixel court passes. Whether the bound rejects usable courts there or those courts are geometrically implausible needs the camera error of the nearest courts, which can be looked up in the stage records by candidate ID.
- The all-camera paint ranking fails on four case-arms with pools of 1,434 to 2,757 courts, while most pools above 1,300 keep a close paint winner, so pool size alone does not explain it. Whether those four pools share a wrong-court family is untested.
- The line ranking's 140-pixel courts on the Amateur-2 views appear under every arm. Whether they are one recurring wrong court cannot be read from pair IDs, which are not comparable across arms.
- Whether M's GX5 court is visually usable is not decided here; it is 9.9 pixels from an approved control, and B's GX0 winner at 7.7 pixels was judged unusable earlier.

## Links

- Protocol: `../worklog/CLAUDE_DIRECTION_EXPERIMENTS.md` (untracked worklog tree).
- Records: `worklog.md` (resume block and log), `runs.md` (commands, hashes, timings, exit codes), `evidence.md` (provenance and replays), `hypotheses.md` (per-arm evidence).
- Code: `membership.py`, `selection.py`, `run_arms.py`, `run_fits.py`, `run_matcher.py`, `diagnose_matrix.py`, `manifest.py`, `summarise.py`, `render_gallery.py`; tests under `tests/`.
- Run outputs: `runs/direction_agreement_20260915_144900/` with `manifest.json.gz`, `summary.md`, `e0/`, `e1/` (matrices as `.npy.xz`, untracked), `e2/`, `e3/`, `e4/`, `logs/` and `receipts/`.
- Gallery: `visual_check.html`, nine views with the B, M and R line and paint winners from the all-camera pools, controls labelled with their approval status, and the earlier session's visual rulings on the B winners joined by candidate ID. It references the frames by relative path in the untracked worklog tree, so it renders only beside that tree.
