# Evidence: inputs, replays and gates

Terms are defined in `worklog.md`. Every number here comes from a file in `runs/` or `prior_checks/`, produced by the script named beside it.

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

Reading: on Amateur-3 under R a horizontal matching that reaches the fit is enumerated and passes the support and player rules (4.28 px), the duplicate removal keeps a different matching with the same marking assignment at 6.24 px, and that one sits at position 11,518 of 23,864 in the score order against a cap threshold of 0.885; every one of the 512 kept horizontal matchings supports all five markings. The rank shown is that of the most accurate survivor; a less accurate near-fit matching may rank higher, which does not change the conclusion because the best kept court is 49.65 px. The template's symmetry enumerates every physical matching twice, so position 11,518 is about the 5,760th physical matching and the cap keeps about 256 physical matchings per direction. Pair 30 tells the same story at rank 23,510. Under the baseline selection the same pair keeps a 5.39 px horizontal matching at rank 367, which is why the baseline pool holds a 4.27 px court. On GX0 the baseline's best-fit pair (143) loses its horizontal matching to the cap in the same way (rank 1,459, best kept 34.40) while pair 22 survives at rank 292; under M the ideal matching supports only 2 of 5 horizontal markings, so the enumeration ceiling (23.43) sits above the fit and the cap adds ten more pixels. With no per-direction cap the nearest court from the pair would be 6.19 px (Am3-0 R pair 43), 8.73 (pair 30), 6.40 (GX0 B pair 143) and 23.24 (GX0 M pair 23).

## Paint-profile study (`paint_profiles.py`, `runs/paint_profiles/`)

Descriptive only; the labels come from the control and are not used by any filter. Per fragment: ridge contrast (brightest pixel within 2.5 working px of the fragment minus the brighter of the two flank means, with two flank windows tried, 3 to 6 and 6 to 10 working px on each side, and the larger contrast kept) and saturation at that pixel, medians over 12 samples along the fragment. Fragments with both endpoints within 6 working px of any painted marking projected through the control are "marking"; the band can also catch fragments between two close markings at the far end, so the label is a study aid. All 5,370 raw pack fragments are measured, including the few the matcher's observation step clips away. Median ridge contrast, marking against other: GX0 21 against 12, GX5 22 against 12, Am2-150 51 against 16, Am2-28019 45 against 18, Am3-0 41 against 8, SS03-17 71 against 23, SS03-19 59 against 23, SS03-16 66 against 25, SS21-20 69 against 18. Median peak saturation, marking against other: GX0 59 against 46, GX5 51 against 48, Am2-150 56 against 48, Am2-28019 48 against 47, Am3-0 26 against 92; the broadcast frames are single-channel, so saturation is 0 there. On the GX and Amateur-2 views the saturation term therefore does no work and the contrast term carries the rule. GX0's six offending fragments measure: row 31 (wall seam) 16, 7, 11 and 7; row 51 (floor strip) 11; row 61 (face) -2. The lowest marking values on GX0 come from thin far lines; which ones is the session's reading of the profiles, not a recorded attribution.
