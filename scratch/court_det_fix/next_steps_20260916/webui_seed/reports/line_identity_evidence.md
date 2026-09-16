# Evidence: inputs, replays and gates

Terms are defined in `worklog.md`. The main tables come from a file in `runs/`
or `prior_checks/`, produced by the script named beside it. The C2 identity
trace below comes from the saved raw records and its dedicated replay in
`../next_steps_20260916/C2_traces/`.

## Inputs and their identity

- Nine frozen views, packs and native frames as the direction experiment used them (`shared.CASES`, `shared.frame_path`); their MD5s are in the direction experiment's run manifest (`../direction_agreement/runs/direction_agreement_20260915_144900/manifest.json.gz`).
- Direction-experiment records read: `e2/<case>.json.gz` (frozen selections per arm), `e3/<case>.json.gz` (control fits and the frozen control per view), `e0/<case>.json.gz` (merge membership).
- Courts-before-the-gate generation records, run `pregate_20260915_115949` (gitignored, 530 MB with side files): GX0 under M `078199bb97162a6fd5d5068aafd1491a`, Amateur-3 under R `dabcb61c2c38dd1384dfe3d111bdd7a4`.
- Baseline generation records pulled from `R/automatic_axes_20260914/results/` (gitignored cruft): Amateur-3 `d0acd5a240be59960d627f122c1af438`, GX0 `654f7003d7f48c7e0351d2f418bdbdea`.
- The broadcast frames (`*_cached_view.png`) are single-channel, so saturation is 0 for every ShuttleSet fragment; the paint rule reduces to its contrast term there.

## Courts before the player gate (`prior_checks/pregate_loss/`)

Gate 1 (`gate1.py`, output `gate1_output.txt`): both new generation records equal the saved ones from the direction experiment in every pair status (240 of 240), every shortlist candidate ID list and corners (134 and 100 matched pairs, largest corner gap 0.0), the 256 final entry IDs, both winner IDs and the pooled count.

Gate 2 (`analyse.py`, output `gate2_analysis_output.txt`): for every matched pair the usable-court corners the cap-loss run recorded equal the combined corners at the usable indices exactly; the record's three counts equal the array length and the mask sums; every usable court passes the geometry mask; the player mask recomputes from the stored fractions. The nearest usable court per pair equals the cap-loss table's nearest proposed court to 0.0 px with the same index in all 116 and 95 pairs that have one. One script fix was needed first: the cross-check had compared an index into the post-mask array with an index into the pre-mask array.

## Axis-matching replay (`axis_replay.py`, `runs/axis_replay/`)

For each of six pairs the replay rebuilt the pair's basis and reran the axis matching with no per-direction cap. Gates, all in `run.log`:

1. The basis equals the record's `basis_working` (atol 1e-9).
2. The direction-fit homography projects to the control with the same maximum corner error the fit record stores (1e-6), and the court rebuilt from the decomposed per-direction scale and offset reproduces that error (1e-6). The off-diagonal leak of the decomposition is near zero by construction (the fit and the basis share the direction columns) and is printed, not counted as evidence.
3. The 512 kept matchings per direction equal the record's entries by ID and parameters (atol 1e-9), and the enumerated and distinct counts equal the record's diagnostics.
4. Every kept horizontal matching combined with every kept vertical one equals the pair's recorded pre-mask nearest court within 1e-3 px for arms M and R (33.822, 46.100 and 40.140 from `prior_checks/pregate_loss/table.csv`); for arm B, which has no pre-mask recording, it lies at or below the pair's own shortlist nearest, and in fact equals it (4.2708 against 4.2709, 7.6621 and 34.3268 against the same), so on these three pairs the per-pair cap kept the nearest combined court. Asserted in the script since audit 1.

Result table (working px, maximum corner distance to the control; "best" is the best court reachable when this direction takes its best surviving matching and the other direction takes the ideal one; rank is the 0-based position in the score order, which the cap truncates after position 511; "best distinct score" is the score of the matching in the "best distinct" column):

| View, selection | Pair | Direction fit | Direction | Ideal markings supported | Best enumerated | Best after support and player rules | Best distinct (rank) | Best kept (rank) | Cap threshold score | Best distinct score | Kept by kept | Distinct by distinct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GX0, M | 23 | 19.65 | horizontal | 2 of 5 | 23.43 | 23.43 | 23.43 (2228) | 33.54 (399) | 0.584 | 0.405 | 33.82 | 23.24 |
| GX0, M | 23 | 19.65 | vertical | 3 of 6 | 18.93 | 18.93 | 18.93 (1999) | 24.67 (245) | 0.533 | 0.333 | 33.82 | 23.24 |
| Am3-0, R | 43 | 4.27 | horizontal | 5 of 5 | 4.28 | 4.28 | 6.24 (11518) | 49.65 (228) | 0.885 | 0.581 | 46.10 | 6.19 |
| Am3-0, R | 43 | 4.27 | vertical | 5 of 6 | 3.92 | 3.92 | 4.50 (103) | 4.50 (103) | 0.545 | 0.650 | 46.10 | 6.19 |
| Am3-0, R | 30 | 6.87 | horizontal | 4 of 5 | 7.25 | 7.25 | 7.25 (23510) | 46.37 (267) | 0.868 | 0.435 | 40.14 | 8.73 |
| Am3-0, R | 30 | 6.87 | vertical | 4 of 6 | 7.60 | 7.60 | 9.98 (618) | 11.04 (315) | 0.541 | 0.518 | 40.14 | 8.73 |
| Am3-0, B | 43 | 3.75 | horizontal | 5 of 5 | 5.08 | 5.08 | 5.23 (6584) | 5.39 (367) | 0.884 | 0.661 | 4.27 | 4.27 |
| Am3-0, B | 43 | 3.75 | vertical | 5 of 6 | 3.43 | 3.43 | 3.92 (14) | 3.92 (14) | 0.541 | 0.754 | 4.27 | 4.27 |
| GX0, B | 22 | 7.82 | horizontal | 4 of 5 | 6.84 | 6.84 | 6.84 (292) | 6.84 (292) | 0.561 | 0.588 | 7.66 | 7.54 |
| GX0, B | 22 | 7.82 | vertical | 5 of 6 | 7.52 | 7.52 | 7.80 (399) | 7.80 (399) | 0.523 | 0.552 | 7.66 | 7.54 |
| GX0, B | 143 | 6.80 | horizontal | 4 of 5 | 6.28 | 6.28 | 6.28 (1459) | 34.40 (178) | 0.591 | 0.480 | 34.33 | 6.40 |
| GX0, B | 143 | 6.80 | vertical | 3 of 6 | 7.19 | 7.19 | 7.35 (504) | 7.35 (504) | 0.526 | 0.526 | 34.33 | 6.40 |

Reading: on Amateur-3 under R the closest eligible horizontal matching is enumeration 29,686, with the fit-side error 4.2783 px, score 0.6579, support 5 of 5 and player compatibility true. Its assignment tuple is `(7, 197, 32, 3, 19)`. The actual representative of that tuple is enumeration 29,665 at 8.5405 px and score 0.7844, with distinct score rank 1,936; the representative has the higher score required by the stable descending-score deduplication. The closest distinct assignment is a different tuple, `(6, 10, 32, 3, 19)`, at enumeration 30,886, 6.2409 px, score 0.5811 and distinct score rank 11,518. The closest kept row is enumeration 18,981 at 49.6472 px and kept rank 228. Every one of the 512 kept horizontal matchings supports all five markings. The template's symmetry enumerates every physical matching twice, so a rank counts both twins. Pair 30 tells the same cap-ordering story at rank 23,510. Under the baseline selection the same pair keeps a 5.39 px horizontal matching at rank 367, which is why the baseline pool holds a 4.27 px court. On GX0 the baseline's best-fit pair (143) loses its horizontal matching to the cap in the same way (rank 1,459, best kept 34.40) while pair 22 survives at rank 292; under M the ideal matching supports only 2 of 5 horizontal markings, so the enumeration ceiling (23.43) sits above the fit and the cap adds ten more pixels. With no per-direction cap the nearest court from the pair would be 6.19 px (Am3-0 R pair 43), 8.73 (pair 30), 6.40 (GX0 B pair 143) and 23.24 (GX0 M pair 23).

### C2 raw identity trace

The dedicated replay is in `../next_steps_20260916/C2_traces/check_traces.py`; its raw witnesses are in `witnesses.json`. It reruns the frozen `projective_seed.match_axis` helper against the R pair-43 input and checks the saved first 512 IDs, parameters and diagnostics. The GX0 rows use the saved pair shortlists for the full pre-global pool and a 512-by-512 product replay for pair 143. The control corners, source paths, working/native scale and helper hashes are recorded with the witnesses.

## Paint-profile study (`paint_profiles.py`, `runs/paint_profiles/`)

Descriptive only; the labels come from the control and are not used by any filter. Per fragment: ridge contrast (brightest pixel within 2.5 working px of the fragment minus the brighter of the two flank means, with two flank windows tried, 3 to 6 and 6 to 10 working px on each side, and the larger contrast kept) and saturation at that pixel, medians over 12 samples along the fragment. Fragments with both endpoints within 6 working px of any painted marking projected through the control are "marking"; the band can also catch fragments between two close markings at the far end, so the label is a study aid. All 5,370 raw pack fragments are measured, including the few the matcher's observation step clips away. Median ridge contrast, marking against other: GX0 21 against 12, GX5 22 against 12, Am2-150 51 against 16, Am2-28019 45 against 18, Am3-0 41 against 8, SS03-17 71 against 23, SS03-19 59 against 23, SS03-16 66 against 25, SS21-20 69 against 18. Median peak saturation, marking against other: GX0 59 against 46, GX5 51 against 48, Am2-150 56 against 48, Am2-28019 48 against 47, Am3-0 26 against 92; the broadcast frames are single-channel, so saturation is 0 there. The saturation cut-off of 90 therefore removes no GX marking fragment that the contrast term would keep at the median, but the Amateur-2 marking saturation p75 is 95 and 96, so on those two views about a quarter of the marking fragments fall to the saturation term alone; 90 was a round number near GX0's marking p75 of 85 and nothing else recommends it. GX0's six offending fragments measure: row 31 (wall seam) 16, 7, 11 and 7; row 51 (floor strip) 11; row 61 (face) -2. The lowest marking values on GX0 come from thin far lines; which ones is the session's reading of the profiles, not a recorded attribution.

## Filter replay (`filter_replay.py`, `runs/filter_replay/`)

Gate: on every view the unfiltered replay reproduces the saved baseline direction record exactly (retained candidate IDs, working points, support masks; `run.log`, "gate passed" per view). The baseline arm's selection is then identical to the saved one by construction (`selection_identical_to_baseline` true in `table.csv`).

Arms and rules: person (drop a fragment whose midpoint lies inside any person box in the pack; the pack's boxes come from every sampled frame of a three-second window, so a fragment is masked wherever any player stood in that window), paint (keep a fragment with ridge contrast at least 20 and peak saturation at most 90), paint_person (both), paint at contrast 15 and 25, and two observation-only arms (paint_observations, person_observations) that keep the baseline directions and filter only the fragments the axis matching sees. How the paint rule was arrived at: the contrast measure went through three versions, each changed after reading GX0 results (the study's numbers for the tan strip, then a GX0 smoke replay of the filter that lost the y-direction); the thresholds were then read off the study's quantiles, which use control-derived labels on the same nine views (GX0's six offending fragments measure at most 16; marking medians sit at 21 and above). No view is held out, GX0 shaped the measure, and the rule was fixed before the nine-view replay and not changed after it.

Stage 1, direction selection. The incidence bound (`set_bound`) and the best direction fit (`fit_pairs`, best finite ordered pair) against the control, working px:


| View | baseline: bound / fit | person: bound / fit | paint: bound / fit | paint_person: bound / fit | paint15: bound / fit | paint25: bound / fit |
| --- | --- | --- | --- | --- | --- | --- |
| GX0 | 6.0 / 6.8 | 1.2 / 1.5 | 15.4 / 32.8 | 15.4 / 32.8 | 8.1 / 17.6 | 15.1 / 37.9 |
| GX5 | 9.7 / 23.8 | 3.1 / 11.5 | 4.4 / 8.4 | 6.1 / 18.8 | 8.4 / 19.1 | 3.3 / 6.3 |
| Am2-150 | 2.9 / 3.3 | 6.1 / 8.9 | 8.7 / 12.0 | 12.1 / 14.3 | 7.4 / 14.3 | 3.9 / 4.9 |
| Am2-28019 | 2.6 / 3.0 | 2.1 / 5.2 | 2.2 / 2.7 | 0.3 / 0.6 | 0.5 / 0.5 | 0.7 / 0.8 |
| Am3-0 | 3.1 / 3.7 | 3.7 / 4.2 | 6.1 / 9.0 | 4.0 / 4.7 | 3.7 / 4.3 | 3.7 / 4.3 |
| SS03-17 | 1.5 / 2.0 | 1.5 / 2.0 | 1.9 / 2.1 | 1.3 / 2.1 | 1.9 / 2.0 | 1.7 / 1.9 |
| SS03-19 | 2.2 / 2.3 | 4.2 / 4.5 | 3.5 / 3.7 | 3.2 / 3.2 | 1.1 / 1.3 | 6.2 / 6.2 |
| SS03-16 | 1.0 / 1.5 | 4.9 / 5.1 | 1.6 / 2.0 | 1.6 / 2.0 | 1.6 / 2.0 | 0.5 / 0.6 |
| SS21-20 | 0.9 / 1.0 | 4.4 / 4.5 | 11.6 / 12.3 | 11.6 / 12.3 | 3.6 / 3.8 | 6.1 / 6.4 |

| View | person: dropped (on markings) | paint: dropped (on markings) | paint_person: dropped (on markings) | paint15: dropped (on markings) | paint25: dropped (on markings) | fragments (on markings) |
| --- | --- | --- | --- | --- | --- | --- |
| GX0 | 69 (0) | 543 (17) | 565 (17) | 441 (12) | 603 (19) | 751 (36) |
| GX5 | 70 (1) | 585 (18) | 606 (18) | 471 (10) | 639 (21) | 790 (40) |
| Am2-150 | 48 (5) | 243 (14) | 261 (15) | 226 (14) | 260 (14) | 397 (36) |
| Am2-28019 | 116 (3) | 257 (12) | 305 (15) | 242 (12) | 272 (12) | 416 (32) |
| Am3-0 | 50 (0) | 381 (13) | 395 (13) | 365 (13) | 391 (16) | 507 (52) |
| SS03-17 | 49 (13) | 286 (3) | 319 (15) | 237 (3) | 325 (4) | 675 (58) |
| SS03-19 | 39 (3) | 289 (13) | 314 (16) | 239 (11) | 328 (13) | 678 (71) |
| SS03-16 | 32 (5) | 281 (11) | 300 (16) | 236 (10) | 315 (11) | 669 (62) |
| SS21-20 | 32 (4) | 229 (0) | 243 (4) | 196 (0) | 248 (1) | 487 (57) |

Reading: no arm improves every view. The person-box rule takes both GX views a long way toward their controls (GX0 6.0 to 1.2 bound, 6.8 to 1.5 fit; GX5 9.7 to 3.1 and 23.8 to 11.5) while removing no fragment on GX0's markings and one on GX5's, but it moves Am2-150 from 2.9 and 3.3 to 6.1 and 8.9 and three ShuttleSet views (SS03-19, SS03-16, SS21-20) from 0.9 to 2.3 out to 4.2 to 5.1 after removing 32 to 48 fragments there. The paint rule at contrast 20 removes 42 to 75 percent of all fragments and 0 to 18 of the 32 to 71 fragments on markings; it improves GX5 (bound 4.4, fit 8.4) and Am2-28019 (2.2, 2.7), leaves SS03-17 and SS03-16 within half a pixel, and regresses SS03-19 (2.2 to 3.5), GX0 (32.8), Am2-150 (12.0), Am3-0 (9.0) and SS21-20 (12.3, where the nearest selected direction to the control's x-direction moves from 1.4 to 8.2 degrees in the normalised chart although no marking fragment was removed). The contrast sensitivities are monotone on some views (SS03-19 bounds 1.1, 3.5, 6.2 at 15, 20, 25) and not on others (Am2-28019 fits 0.5, 2.7, 0.8; Am3-0 4.3, 9.0, 4.3; SS21-20 3.8, 12.3, 6.4). The coverage selection reacts to the fragment set in a way that is not monotone in how much clutter is removed, as the direction experiment found for its two changes.

Stage 2, axis matching on the arm's best-fit pair with the filtered observations and no cap (the ladder of `axis_replay.py`; ranks are 0-based positions in the score order; "kept by kept" is the nearest court from the 512 kept matchings per direction, "no cap" from every distinct matching):


| View | Arm | Pair | Direction fit | Horizontal best distinct (rank) | Horizontal best kept (rank) | Vertical best distinct (rank) | Vertical best kept (rank) | Kept by kept | No cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GX0 | baseline | 143 | 6.8 | 6.3 (1459 of 2344) | 34.4 (178) | 7.3 (504 of 1944) | 7.3 (504) | 34.3 | 6.4 |
| GX0 | paint_observations | 143 | 6.8 | 6.3 (181 of 220) | 6.3 (181) | 24.5 (108 of 190) | 24.5 (108) | 12.6 | 12.6 |
| GX0 | person_observations | 143 | 6.8 | 6.3 (773 of 1364) | 7.6 (376) | 7.3 (376 of 1522) | 7.3 (376) | 8.3 | 6.4 |
| GX0 | person | 20 | 1.5 | 1.8 (20 of 1740) | 1.8 (20) | 1.6 (87 of 1834) | 1.6 (87) | 1.6 | 1.6 |
| GX0 | paint | 21 | 32.8 | 857.3 (0 of 2) | 857.3 (0) | 46.1 (58 of 310) | 46.1 (58) | 605.2 | 605.2 |
| GX0 | paint_person | 19 | 32.8 | none | none | 46.1 (44 of 180) | 46.1 (44) | none | none |
| GX5 | baseline | 17 | 23.8 | 21.4 (1676 of 1768) | 22.3 (211) | 21.0 (874 of 2302) | 25.2 (134) | 23.7 | 20.7 |
| GX5 | paint_observations | 17 | 23.8 | 23.7 (16 of 44) | 23.7 (16) | 25.2 (58 of 198) | 25.2 (58) | 25.1 | 25.1 |
| GX5 | person_observations | 17 | 23.8 | 22.3 (117 of 1152) | 22.3 (117) | 21.0 (758 of 2038) | 25.2 (102) | 23.7 | 21.2 |
| GX5 | person | 17 | 11.5 | 11.2 (5 of 2230) | 11.2 (5) | 10.3 (169 of 1890) | 10.3 (169) | 11.1 | 11.1 |
| GX5 | paint | 21 | 8.4 | 9.0 (15 of 72) | 9.0 (15) | 11.6 (106 of 288) | 11.6 (106) | 10.0 | 10.0 |
| GX5 | paint_person | 19 | 18.8 | 18.3 (1 of 68) | 18.3 (1) | 19.8 (141 of 306) | 19.8 (141) | 19.4 | 19.4 |
| Am2-150 | baseline | 30 | 3.3 | 5.2 (6 of 4172) | 5.2 (6) | 3.2 (702 of 3376) | 8.0 (36) | 8.7 | 5.4 |
| Am2-150 | paint_observations | 30 | 3.3 | 8.9 (2 of 1162) | 8.9 (2) | 3.3 (238 of 1240) | 3.3 (238) | 6.6 | 6.6 |
| Am2-150 | person_observations | 30 | 3.3 | 5.2 (6 of 4922) | 5.2 (6) | 3.2 (568 of 3078) | 4.6 (429) | 8.7 | 5.4 |
| Am2-150 | person | 32 | 8.9 | 14.3 (1790 of 5990) | 20.4 (262) | 15.5 (1142 of 1192) | 64.0 (390) | 57.5 | 17.4 |
| Am2-150 | paint | 28 | 12.0 | 16.3 (1351 of 2010) | 16.3 (103) | 16.0 (1200 of 1350) | 32.3 (109) | 40.5 | 12.9 |
| Am2-150 | paint_person | 15 | 14.3 | 16.3 (1741 of 2390) | 22.7 (30) | 26.9 (1049 of 1274) | 32.7 (374) | 46.2 | 22.6 |
| Am2-28019 | baseline | 15 | 3.0 | 4.9 (3 of 7102) | 4.9 (3) | 3.9 (729 of 3958) | 60.9 (345) | 61.6 | 5.9 |
| Am2-28019 | paint_observations | 15 | 3.0 | 5.0 (1 of 298) | 5.0 (1) | 5.5 (1 of 764) | 5.5 (1) | 7.6 | 7.6 |
| Am2-28019 | person_observations | 15 | 3.0 | 4.9 (2 of 4366) | 4.9 (2) | 3.9 (190 of 2632) | 3.9 (190) | 5.9 | 5.9 |
| Am2-28019 | person | 15 | 5.2 | 4.8 (78 of 6308) | 4.8 (78) | 4.8 (3 of 2582) | 4.8 (3) | 4.6 | 4.6 |
| Am2-28019 | paint | 15 | 2.7 | 5.7 (43 of 1504) | 5.7 (43) | 3.1 (413 of 774) | 3.1 (413) | 5.2 | 5.2 |
| Am2-28019 | paint_person | 27 | 0.6 | 4.4 (5 of 604) | 4.4 (5) | 4.0 (10 of 386) | 4.0 (10) | 3.5 | 3.5 |
| Am3-0 | baseline | 43 | 3.7 | 5.2 (6585 of 23758) | 5.4 (366) | 3.9 (15 of 1930) | 3.9 (15) | 4.3 | 4.3 |
| Am3-0 | paint_observations | 43 | 3.7 | 6.3 (1 of 66) | 6.3 (1) | 3.9 (8 of 290) | 3.9 (8) | 6.2 | 6.2 |
| Am3-0 | person_observations | 43 | 3.7 | 5.2 (6584 of 23758) | 5.4 (367) | 3.9 (14 of 1930) | 3.9 (14) | 4.3 | 4.3 |
| Am3-0 | person | 42 | 4.2 | 6.2 (2660 of 23980) | 48.0 (238) | 5.1 (370 of 1898) | 5.1 (370) | 35.5 | 5.3 |
| Am3-0 | paint | 16 | 9.0 | 9.3 (65 of 98) | 9.3 (65) | 9.5 (35 of 274) | 9.5 (35) | 11.1 | 11.1 |
| Am3-0 | paint_person | 24 | 4.7 | 9.7 (1 of 92) | 9.7 (1) | 4.8 (28 of 288) | 4.8 (28) | 8.8 | 8.8 |
| SS03-17 | baseline | 1 | 2.0 | 3.4 (506 of 1064) | 3.4 (506) | 2.2 (684 of 10254) | 3.0 (214) | 3.5 | 3.5 |
| SS03-17 | paint_observations | 1 | 2.0 | 3.4 (134 of 272) | 3.4 (134) | 2.2 (146 of 5376) | 2.2 (146) | 3.5 | 3.5 |
| SS03-17 | person_observations | 1 | 2.0 | 3.4 (506 of 1064) | 3.4 (506) | 3.0 (172 of 9484) | 3.0 (172) | 3.5 | 3.5 |
| SS03-17 | person | 1 | 2.0 | 3.4 (506 of 1064) | 3.4 (506) | 3.0 (78 of 9484) | 3.0 (78) | 3.5 | 3.5 |
| SS03-17 | paint | 1 | 2.1 | 2.3 (8 of 192) | 2.3 (8) | 3.0 (27 of 5886) | 3.0 (27) | 3.1 | 3.1 |
| SS03-17 | paint_person | 1 | 2.1 | 2.3 (129 of 234) | 2.3 (129) | 2.1 (24 of 4820) | 2.1 (24) | 2.3 | 2.3 |
| SS03-19 | baseline | 1 | 2.3 | 3.2 (7 of 758) | 3.2 (7) | 2.9 (325 of 9469) | 2.9 (325) | 3.1 | 3.1 |
| SS03-19 | paint_observations | 1 | 2.3 | 3.2 (4 of 150) | 3.2 (4) | 2.9 (70 of 5640) | 2.9 (70) | 3.1 | 3.1 |
| SS03-19 | person_observations | 1 | 2.3 | 3.2 (6 of 584) | 3.2 (6) | 2.9 (1466 of 8761) | 2.9 (368) | 3.1 | 3.1 |
| SS03-19 | person | 137 | 4.5 | 5.4 (0 of 564) | 5.4 (0) | 4.6 (2447 of 3396) | 8.5 (486) | 7.9 | 5.1 |
| SS03-19 | paint | 2 | 3.7 | 4.5 (3 of 136) | 4.5 (3) | 3.7 (37 of 3586) | 3.7 (37) | 4.5 | 4.5 |
| SS03-19 | paint_person | 1 | 3.2 | 3.8 (5 of 154) | 3.8 (5) | 3.4 (541 of 4800) | 6.4 (24) | 6.3 | 3.7 |
| SS03-16 | baseline | 1 | 1.5 | 2.4 (32 of 962) | 2.4 (32) | 1.5 (167 of 8358) | 1.5 (167) | 2.3 | 2.3 |
| SS03-16 | paint_observations | 1 | 1.5 | 2.4 (8 of 234) | 2.4 (8) | 1.5 (59 of 5002) | 1.5 (59) | 2.3 | 2.3 |
| SS03-16 | person_observations | 1 | 1.5 | 2.4 (32 of 962) | 2.4 (32) | 1.6 (523 of 8060) | 3.7 (196) | 3.7 | 2.3 |
| SS03-16 | person | 1 | 5.1 | 5.1 (7 of 1036) | 5.1 (7) | 5.2 (857 of 6672) | 28.5 (417) | 28.1 | 5.2 |
| SS03-16 | paint | 1 | 2.0 | 3.0 (7 of 358) | 3.0 (7) | 2.2 (6 of 6134) | 2.2 (6) | 2.7 | 2.7 |
| SS03-16 | paint_person | 1 | 2.0 | 3.0 (7 of 360) | 3.0 (7) | 2.5 (2 of 5660) | 2.5 (2) | 3.1 | 3.1 |
| SS21-20 | baseline | 0 | 1.0 | 1.4 (0 of 1230) | 1.4 (0) | 1.8 (26 of 9162) | 1.8 (26) | 1.6 | 1.6 |
| SS21-20 | paint_observations | 0 | 1.0 | 1.4 (0 of 158) | 1.4 (0) | 1.7 (6 of 3868) | 1.7 (6) | 1.5 | 1.5 |
| SS21-20 | person_observations | 0 | 1.0 | 1.4 (0 of 1230) | 1.4 (0) | 1.8 (30 of 8642) | 1.8 (30) | 1.6 | 1.6 |
| SS21-20 | person | 0 | 4.5 | 4.7 (6 of 1228) | 4.7 (6) | 4.8 (1345 of 7064) | 28.3 (364) | 27.9 | 4.8 |
| SS21-20 | paint | 31 | 12.3 | 12.0 (95 of 138) | 12.0 (95) | 14.2 (42 of 266) | 14.2 (42) | 13.6 | 13.6 |
| SS21-20 | paint_person | 31 | 12.3 | 12.1 (90 of 138) | 12.1 (90) | 14.2 (43 of 266) | 14.2 (43) | 13.6 | 13.6 |

Reading: holding the baseline directions and filtering only the observations the axis matching sees (paint_observations) helps where the cap had cut a near-fit matching under the baseline: GX0 pair 143's horizontal moves from position 1459 to 181 and the kept-by-kept nearest from 34.3 to 12.6 px (its vertical side worsens from 7.3 to 24.5; why is not measured), and Amateur-2 frame 28019 pair 15 from 61.6 to 7.6. Where the baseline had not been cut it costs a little: Amateur-3 pair 43's horizontal moves from position 6585 to 1 of 66, but the kept-by-kept nearest goes from 4.3 to 6.2 px, and GX5 from 23.7 to 25.1. On the four ShuttleSet views and Amateur-2 frame 150 it changes the nearest court by at most 2.1 px. The arms that also reselect directions (paint, paint_person) inherit stage 1's reallocation: better on GX5 (23.7 to 10.0), Amateur-2 frame 28019 (61.6 to 5.2 and 3.5) and SS03-17, worse on GX0 (605), Amateur-2 frame 150 (40.5), Amateur-3 (11.1) and SS21-20 (13.6). The person-box rule behaves the same way: with the baseline directions and person-filtered observations (person_observations) the kept-by-kept nearest moves from 34.3 to 8.3 on GX0 and from 61.6 to 5.9 on Amateur-2 frame 28019, stays within 0.1 px on six views and worsens by 1.4 on SS03-16; with its own directions (person) GX0 reaches 1.6 and GX5 11.1, but Amateur-2 frame 150 (57.5), Amateur-3 (35.5), SS03-16 (28.1) and SS21-20 (27.9) lose the court to the cap on one direction. One caveat for the matcher run: with a filtered pack entry, `generate` also builds its line families, stripe weights and distance maps from the filtered fragments, so the floor, stripe and paint scores of the observation-only arms see filtered evidence too; the nearest pooled court is the number closest to the axis matching, the winners are not attributable to it alone.

The baseline axis rows for GX0 and Amateur-3 equal the gated `axis_replay` values (34.3268 and 4.2708 kept by kept); the other seven baseline rows have no matcher record to gate against. The Amateur-3 baseline rank differs by one between the two scripts (6584 against 6585, kept 367 against 366): the two scripts take the fit homography from different sources (the stored E3 record against a fresh `fit_pairs`), and each physical matching has a mirror twin with equal error, so `argmin` can land on either twin; that reading is inferred, not tested.

The matcher inputs (`inputs/<arm>/`) are written with a fixed gzip header time, so rerunning the replay on unchanged code reproduces their MD5s. `account.py` gates every matcher record against them before accounting: input hashes, case fields, the kept fragment indices, and the direction record (equal to the saved baseline estimator for the observation-only arms; reproduced by the unchanged selection on the filtered fragments for the own-direction arms).

## Accounting gate (`account.py`)

Before any matcher record existed, the accounting core was run on the direction experiment's own generation record for GX0 under M with this folder's construction of the control and reference (the E3 control scaled to native pixels; the pack's reference entry). It reproduced that experiment's accounting row exactly: status diagnosed, nearest pooled 33.822484574184806 px, line winner 577.6346368531953, paint winner 379.2568320601342, pooled 29696, final 256 (session transcript, 2026-09-16 09:55). The same functions (`diagnose_automatic.diagnose`, `diagnose_matrix.accounting`) measure the filter arms, so their rows are comparable with the baseline rows read from `e4/accounting.csv.gz`.


## Matcher run (`run_matcher.py` on the compute host, `account.py` locally; `runs/<run>/matcher/`)

Gate: `account.py` exit 0 on the 108 stage records of the 36 jobs (four arms, nine views, three stages). Per record it checked the input hashes the record carries against the files in `inputs/`, that the case entry equals the pack's in every field but the fragment list and that the list is the pack's fragments at the recorded kept indices, that the observation-only arms' direction record equals the saved baseline estimator field for field, and that the own-direction arms' directions are what the unchanged selection produces from the filtered fragments (retained IDs equal, points within 1e-12). The measures are the direction experiment's own (`diagnose_automatic.diagnose`, `diagnose_matrix.accounting`), which reproduced that experiment's GX0-under-M row exactly in the dry run above.

The generation-stage table is in `results.md`; `comparison.md` also holds the all-camera stage. Agreement between the local axis-stage proxy (kept-matching nearest on the best-fit pair) and the matcher's nearest pooled court, working px: paint arm, local against matcher, GX0 605.2 against 151.9, GX5 10.0 against 10.0, Am2-150 40.5 against 37.2, Am2-28019 5.2 against 5.2, Am3-0 11.1 against 11.1, SS03-17 3.1 against 3.1, SS03-19 4.5 against 4.5, SS03-16 2.7 against 2.7, SS21-20 13.6 against 13.6; person arm, GX0 1.6 against 1.9, GX5 11.1 against 11.1, Am2-150 57.5 against 73.3, Am2-28019 4.6 against 4.6, Am3-0 35.5 against 23.6, SS03-17 3.5 against 3.5, SS03-19 7.9 against 7.9, SS03-16 28.1 against 28.1, SS21-20 27.9 against 17.3; paint_observations, GX0 12.6 against 9.0, GX5 25.1 against 25.1, Am2-150 6.6 against 6.6, Am2-28019 7.6 against 7.6, Am3-0 6.2 against 5.7, broadcast views within 0.1; person_observations, GX0 8.3 against 7.7, GX5 23.7 against 23.7, Am2-150 8.7 against 8.3, Am2-28019 5.9 against 6.0, Am3-0 4.3 against 4.3, SS03-16 3.7 against 3.7, the other broadcast views within 0.1. Within 0.1 px on 26 of the 36 view-arms. Where the matcher is closer than the proxy, a direction pair other than the best-fit one supplied the court (a deduction: a pair's pool court cannot be closer than its kept-by-kept nearest); on GX0 that holds in every arm, so the proxy's direction of change there is wrong for paint_observations (proxy 34.3 to 12.6, matcher 7.7 to 9.0). Where the proxy is closer (Am2-150 under person), a step after the axis matching removed the pair's court, and which step was not instrumented.
