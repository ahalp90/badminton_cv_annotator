# SVD court-detection evaluation — supplementary evidence

## Scope

Read-only re-analysis of the task3_inputs.tar.gz supplied in the conversation, labelled revision b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea. The current remote branch could not be independently read. No detector, matcher, optimizer, new direction refit, or new court fit was run. B_svd values below are **previously cached E3 control-fit potentials**, not newly measured detection outcomes.

The population remains nine development views: six visually approved controls and three manual references. The latter are not upgraded to approved courts. Errors use the existing maximum-corner metric and 180-degree relabelling. Pair retention means retaining the exact original cached best B ordered pair. This is not ranked-winner retention.

## Additional diagnostic comparison

All rankings retain original B group identities and original B directions. A pair survives only when both group IDs survive. Singular values use the frozen normalization, with sigma1 >= sigma2 >= sigma3. Four untuned metric definitions were compared; the additional budgets of 4 and 6 are exploratory, not modifications to the earlier follow-up 3 result.

| Ranking | K=8 | K=10 | K=12 |
|---|---:|---:|---:|
| Algebraic RMS ascending | 2/9 | 3/9 | 9/9 |
| Support count descending | 7/9 | 8/9 | 8/9 |
| (sigma2-sigma3)/sigma1 descending | 2/9 | 2/9 | 2/9 |
| sigma3/sigma2 ascending | 2/9 | 2/9 | 4/9 |

Residual and singular-spectrum rankings break exact ties by descending support count and then original group ID; support-only breaks ties by group ID. The complete outputs contain every selection and error delta. These are same-population explorations, not evidence that any screen generalizes.

Changing to a singular-gap ranking is not a supported remedy. At K=12, the worst error increase is 263.693881 px for descending gap, and 140.018862 px for sigma3/sigma2. Support-only K=12 still loses GX0's original best pair; its best retained error increases by 12.732955 px.

The cached B_svd best fit improves on B in eight views and regresses Amateur-3 (3.749206 to 4.388916 px). GX0 improves from 6.798367 to 3.961121 px; GX5 only from 23.800169 to 22.502448 px. These values establish only the old control-fit diagnostic, not automatic-court improvements.

The saved time for computing all 16 B SVD diagnostics is 0.676839–1.084412 ms across views (median 0.721200 ms). These are existing producer timings, not a new benchmark or a detector-speedup result.

## Engineering assessment

The preferred next experiment is **assignment-conditioned fitting**, not a stronger global singular-value gate:

1. Group equivalent line-to-template assignments with the same upstream geometry, observations and weights.
2. Fit each distinct assignment once using all of its assigned observations, using the existing linear/projective parameterization and a stable least-squares solver.
3. Recompute its score before choosing a duplicate representative and applying the per-axis cap.
4. For surviving complete court hypotheses, compare an additive normalized line-based homography refit, followed by geometric refinement and the existing full-pool image scoring.

The pre-cap part addresses a documented loss mechanism; the late homography refinement cannot recover assignments that were already discarded. Neither proposed change has been implemented or validated here. Keep original contenders in the first experiment so geometry changes can be traced; adding contenders is not a guarantee that the final scored winner will improve.

The frozen direction-search note documents the Amateur-3 R duplicate/cap witness: the closest eligible enumeration is 4.2783 px, its actual higher-scoring duplicate representative is 8.5405 px, and the closest retained under the 512 cap is 49.6472 px. The 6.2409 px row is a distinct assignment, not that duplicate representative. This evidence identifies where to investigate; it does not prove that SVD solves the cap problem.

Use singular spectra for conditional degeneracy diagnostics. Test stability of the **predicted court geometry** in image pixels (and downstream court-plane coordinates where relevant), rather than equating precise vanishing-point coordinates with a reliable court. A well-conditioned fit can still describe the wrong court. A low-residual minimal fit has no redundant observations to test its validity.

Exact production function names are not claimed: the uploaded files include the diagnostic producers but not the matcher implementation. The proposed integration boundary is semantic, before representative selection/cap and after complete assignment for joint refinement. `run_fits.fit_case`, `fit_groups`, and `svd_direction` are diagnostic references, not automatically the correct production insertion points.

## Reproduction

Put these scripts next to the supplied input archive and extract it. Requires Python and NumPy; no network or optimizer dependencies are used.

```sh
tar -xzf task3_inputs.tar.gz
python inspect_cached.py
python compare_rankings.py
```

The scripts write `cached_extension.json.gz` and `exploratory_rankings.json.gz` beside themselves. They only read the input files. Verification checks reproduce the earlier RMS screen's retention counts, full-budget equality for all four metrics, the identity RMS=sigma3/sqrt(n), and that the SVD optimum does not exceed the original direction's algebraic residual under the same fixed metric.

`frozen_direction_search_README.md` is an unchanged copy of the evidence note from the uploaded archive. It is a source note, not a new experiment. The archive inputs are deliberately not duplicated in this supplementary bundle.
