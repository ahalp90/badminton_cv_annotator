# Follow-up 3 — fixed-B SVD pruning screen

## Result and decision

**The 12-direction budget preserves the exact best cached B ordered pair in all nine views.** It keeps 132 of the original 240 ordered pairs per view, a **45% reduction in pair count**, with **zero change in best finite or best converged corner error**. It is the smallest tested budget with this property. Budgets 8 and 10 lose the best pair on seven and six views, respectively.

**GO only for a bounded downstream trial of K=12 using the exact frozen normalization and unchanged B directions. NO-GO for K=8 or K=10 on this population. NO-GO for treating algebraic-residual ranking as a normalization-independent rule or a production-ready detector change.** The coordinate checks below are material to that distinction.

This run completed the numerical comparison that the previous `return.md` could not execute. No detector, matcher or fitting optimiser was run, no B_svd direction was substituted, and nothing was pushed to GitHub.

## Fixed comparison

The supplied snapshot is labelled `ahalp90/badminton_cv_annotator` revision `b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea`. The population is exactly the nine cases in `direction_agreement/common.py`: six visually approved controls and three manual references. Every view is evaluated at 960 × 540 working pixels. Manual references are not promoted to approved courts.

For every baseline B support mask, the code first computes

```text
normalised_lines = direction_lines @ normalised_to_working
support_rows = normalised_lines[original_B_support_mask]
unit_normal_rows = support_rows / norm(support_rows[:, :2], axis=1)
```

It then calls the frozen producer's SVD diagnostic: full SVD, a unit homogeneous null-space point, singular values padded to three entries, `(sigma_2 - sigma_3) / sigma_1`, and `sqrt(mean((unit_normal_rows @ point)**2))`. The actual functions `svd_direction` and `fit_groups` are loaded in isolation, without importing the producer's optimiser dependencies.

Groups are ordered lexicographically by **algebraic RMS ascending, support count descending, original group index ascending**. No tolerance, nullspace-gap threshold, control error, or refitted direction enters admission. At each fixed budget 8, 10, 12, 14 or 16, a cached **B** ordered pair survives exactly when both original group IDs survive. Groups are never renumbered. `B_svd.groups` is used only to validate the recomputed diagnostics; `B_svd` court fits and directions are not used for evaluation.

The outcome is the saved maximum corner error, with the frozen experiment's existing 180-degree relabelling. It measures least-squares direction-fit potential, not a generated court, ranked detector winner, or usability ruling. In particular, preserving GX5's best fit does not improve its 23.800169 px error.

## Budget accounting

“Best retained” means the actual original best pair ID survives, not merely a different pair passing a loose error threshold. The 1e-8 px comparison tolerance is inherited from the E3 producer's replay check; all zero deltas reported here are exactly zero in the cached floating-point values.

| Directions | Pairs/view | Pairs removed | Best retained, all | Approved | Manual | Largest error increase, px |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 56 | 76.67% | 2/9 | 2/6 | 0/3 | 263.693881 |
| 10 | 90 | 62.50% | 3/9 | 2/6 | 1/3 | 263.693881 |
| 12 | 132 | 45.00% | 9/9 | 6/6 | 3/3 | 0.000000 |
| 14 | 182 | 24.17% | 9/9 | 6/6 | 3/3 | 0.000000 |
| 16 | 240 | 0.00% | 9/9 | 6/6 | 3/3 | 0.000000 |

K=14 gives the same best-pair retention as K=12, but removes only 24.17% of pairs. These are combinatorial reductions, **not measured wall-clock speedups**. The cost of SVD screening and the downstream candidate workload have not been benchmarked.

### Best retained fit by view

A = visually approved control; M = manual reference. All errors are maximum corner error in working pixels. Best-finite and best-converged values are identical for every row/budget in this run.

| View | Control | Full B / K=16 | K=8 | K=10 | K=12 | K=14 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| GX0 | A | 6.798367 | 6.798367 | 6.798367 | 6.798367 | 6.798367 |
| GX5 | A | 23.800169 | 23.800169 | 23.800169 | 23.800169 | 23.800169 |
| Am2-150 | A | 3.256384 | 28.343385 | 28.343385 | 3.256384 | 3.256384 |
| Am2-28019 | M | 2.972162 | 108.431838 | 108.431838 | 2.972162 | 2.972162 |
| Am3-0 | M | 3.749206 | 267.443087 | 267.443087 | 3.749206 | 3.749206 |
| SS03-17 | M | 1.982868 | 76.514861 | 1.982868 | 1.982868 | 1.982868 |
| SS03-19 | A | 2.268884 | 147.404758 | 142.988233 | 2.268884 | 2.268884 |
| SS03-16 | A | 1.501147 | 89.767705 | 88.101770 | 1.501147 | 1.501147 |
| SS21-20 | A | 1.033503 | 99.119436 | 99.119436 | 1.033503 | 1.033503 |

### Exact K=12 admissions and retained witnesses

All indices in this table are the original zero-based B group IDs; group ranks in the detailed records are one-based. “Best pair” is ordered and is never reconstructed using new indices. All other budgets' selected IDs, representative candidate IDs and exact support line IDs are in the machine-readable outputs.

| View | K=12 groups, in residual-rank order | Discarded groups | Best pair ID | Best pair groups |
| --- | --- | --- | ---: | --- |
| GX0 | `9, 12, 1, 11, 8, 4, 3, 6, 10, 5, 13, 14` | `0, 2, 7, 15` | 143 | `[9, 8]` |
| GX5 | `8, 6, 7, 1, 11, 3, 10, 5, 9, 2, 14, 13` | `0, 4, 12, 15` | 17 | `[1, 3]` |
| Am2-150 | `11, 14, 15, 0, 3, 12, 10, 7, 4, 6, 2, 13` | `1, 5, 8, 9` | 30 | `[2, 0]` |
| Am2-28019 | `15, 12, 14, 9, 7, 10, 0, 2, 13, 3, 11, 1` | `4, 5, 6, 8` | 15 | `[1, 0]` |
| Am3-0 | `11, 9, 10, 12, 6, 14, 5, 15, 3, 13, 2, 8` | `0, 1, 4, 7` | 43 | `[2, 14]` |
| SS03-17 | `15, 11, 14, 2, 8, 7, 9, 6, 0, 5, 3, 12` | `1, 4, 10, 13` | 1 | `[0, 2]` |
| SS03-19 | `13, 12, 11, 8, 2, 6, 7, 14, 3, 9, 0, 5` | `1, 4, 10, 15` | 1 | `[0, 2]` |
| SS03-16 | `12, 13, 9, 11, 2, 7, 15, 6, 3, 14, 0, 5` | `1, 4, 8, 10` | 1 | `[0, 2]` |
| SS21-20 | `8, 15, 13, 10, 1, 4, 11, 7, 6, 5, 14, 0` | `2, 3, 9, 12` | 0 | `[0, 1]` |

## Concrete regression witnesses

Two examples show why the smaller budgets are stopped rather than justified by average error or pair-count savings. The full cached records, including homographies, are preserved under each case's `regression_witnesses` in `results/results.json.gz`.

| View and budget | Dropped full-pool best | Why it was dropped | Best surviving pair | Error change, px |
| --- | --- | --- | --- | ---: |
| Am3-0, K=10 | pair 43, `[2, 14]` | group 2 has residual rank 11 | pair 190, `[12, 10]` | 3.749206 → 267.443087 |
| SS03-19, K=10 | pair 1, `[0, 2]` | group 0 has residual rank 11 | pair 98, `[6, 9]` | 2.268884 → 142.988233 |

The first is a manual-reference view; the second is a visually approved control. K=12 restores both pairs without changing their B directions or rerunning their fits.

## Failed and unconverged fits

Across the original 2,160 cached B ordered-pair fits there are **zero failed fits, 2,159 converged fits, and one finite but unconverged fit**. The exception is `am2_window_01_frame_28019`, pair **44**, groups **[2, 15]**, with saved error **361.6190928255014 px**. It survives every tested budget and never wins the best-finite comparison. It is counted as unconverged, not failed.

Consequently the aggregate counts `(failed, unconverged, converged)` are `(0, 1, 503)`, `(0, 1, 809)`, `(0, 1, 1187)`, `(0, 1, 1637)` and `(0, 1, 2159)` for K=8, 10, 12, 14 and 16. The same one cached exception appears in multiple nested budget evaluations; these are not additional solver attempts.

## Coordinate-description checks

### Equivalent descriptions with the same metric: passed

For working-image point coordinates `x' = H x`, row lines become `L' = L H^-1` and the saved normalization map becomes `T' = H T`. Thus the rows sent to the SVD remain `L' T' = L T`, apart from floating-point roundoff. This distinction prevents pixel units or a shifted image origin from accidentally changing the screen.

All **54 case/test combinations** preserved the **entire ranking**, and therefore every budget's selected group set. The six tests per case were working-pixel scale ×2; working translation (+320, −180); an invertible affine working-coordinate change; signed homogeneous row scalings with magnitudes from 1e-3 to 1e3; reversed line rows with identically reversed mask columns; and a 30-degree orthogonal rotation of the normalized frame. The matrices are fixed in the script, and per-case outcomes are retained.

The largest absolute diagnostic differences across these tests were 3.552714e-15 for singular values, 1.554312e-15 for normalized nullspace gap, and 3.642919e-17 for algebraic RMS. The diagnostic comparison tolerances were atol=1e-12 and rtol=1e-10; neither tolerance changes admission.

### Changing the normalization metric: not invariant

A different test changes the normalized frame itself and performs the unit-homogeneous-point SVD in that new frame, without mapping back to the frozen metric. This changes the objective's Euclidean unit-point constraint. Unit 2D line-normal scaling does **not** make algebraic RMS invariant to arbitrary changes of that frame's origin or scale.

The following are negative controls, not alternative arms selected or tuned for deployment:

| Description used for the SVD objective | Identical full rankings | K=12 baseline best pairs retained |
| --- | ---: | ---: |
| Normalized-frame translation (+0.5, −0.25) | 0/9 | 7/9 |
| Normalized-frame scale ×2 | 0/9 | 9/9 |
| Normalized-frame scale ×0.5 | 1/9 | 4/9 |
| Incorrect shortcut: omit saved transform; use working-pixel rows | 0/9 | 5/9 |

An exact approved-control counterexample is **SS03-19**: translating the normalized frame by (+0.5, −0.25) moves group 0 from residual rank 11 to rank 13. K=12 then discards baseline pair 1, groups [0, 2], at **2.2688838662554693 px**. The best surviving cached B fit is pair 95, groups [6, 5], at **142.28774540858052 px**. No direction or fit was changed; only the ranking metric changed.

Am2-28019 is a second example: the same normalized-frame translation moves group 1 from rank 12 to rank 14, losing pair 15 at **2.9721620031633305 px**. Its best surviving pair is 217 at **108.43183817409118 px**.

Therefore the K=12 result is conditional on the saved normalization. It must not be described as a coordinate-free SVD pruning rule. The deliberately incorrect working-pixel shortcut also loses four baseline best pairs at K=12, demonstrating why the transform-before-row-normalization order matters.

## Reproduction, ties and verification

All **144** baseline support groups reproduced the saved singular values, normalized nullspace gaps and algebraic RMS values **bit-for-bit in this environment**; the maximum replay difference was zero. Support masks, support IDs/counts, original representative candidate IDs and B working directions were also checked against the frozen records. E3's stored MD5 links to E2 and the frozen estimator matched for all nine cases.

There were **no exact residual ties and no adjacent ties within 1e-12**. The smallest adjacent gap was **1.896156011987546e-7**, between groups 10 and 7 in Am2-150 (ranks 7 and 8). Thus no actual admission depended on either tie-break key, and the observed coordinate-metric counterexamples are not floating-point tie accidents. Synthetic exact ties verified descending support count, then ascending original group index, including shuffled input order.

All **eight focused tests passed**. They cover the full nine-view main path, full-budget equality, exact tie handling, a two-line nullspace, nonfinite residuals, zero line normals/insufficient support, separate failed versus unconverged fit accounting, incomplete pair caches, changed B directions, missing inputs, an incorrect revision marker and a nonempty output destination. Some test methods cover multiple closely related failure cases. See `verification.txt` for the actual run log. The analysis ran with Python **3.13.5** and NumPy **2.3.5**.

## Boundaries of the conclusion

K=12 is a **development-set-selected budget**: the ranking does not read controls, but the budget recommendation follows inspection of these nine labelled outcomes. No independent views, matcher-generated candidate pools, downstream ranking, temporal behavior or runtime measurements were evaluated. Exact best-fit retention here does not establish that a later matcher preserves its full-pool ranked winner. Fixed support membership is not evidence that support lines are physical court stripes, and this test does not repair incorrectly assigned membership.

The next bounded experiment justified by this result is a fixed-normalization K=12 matcher/full-pool retention comparison against K=16, preserving the existing controls and checking ranked winners. That experiment has **not** been run here. No production integration is recommended on this evidence alone.

## Source and deliverable map

The input archive contained 27 compressed records, five reference files and a revision marker. The numerical inputs are only the named nine E2 files, nine E3 files and nine frozen baseline-direction files under `scratch/court_det_fix/`. Case IDs, paths and input MD5 values are retained in `results/results.json.gz`.

The SVD implementation is the supplied `frozen_helpers_20260914/automatic_axes/svd_fixed/run_svd_fixed.py`, lines 17–44. The transform order and distinction between B and B_svd are established by `direction_agreement/run_fits.py`, lines 80–116. Control interpretation and the warning that E3 measures fit potential rather than generated courts are in `evidence/direction_search/README.md`. The supplied `.github/AGENTS.md` governs the compressed outputs and minimal read-only implementation.

**Task-document access limitation:** `WEBUI_FOLLOWUPS.md` itself was not in the uploaded archive, and it could not be reopened through the browser or direct HTTPS in this run. The fixed comparison, ranking, budgets and required checks above were carried forward from the prior `return.md` and verified against the supplied producers. This run does not claim a fresh reading of that remote task document or independent verification of the current GitHub branch. The supplied revision marker and the records' internal cross-links were checked; the revision's authenticity was not independently re-fetched.

The complete deliverables are `run_screen.py`, `test_screen.py`, `verification.txt`, this report, and the four compressed files in `results/`. `summary.csv.gz` has 45 case/budget rows; `groups.csv.gz` has 144 original-group rows; `aggregate.csv.gz` has five budget rows. `results.json.gz` preserves every selection, diagnostic, coordinate test and exact cached regression witness. Reproduction commands and file descriptions are in `README.md`.
