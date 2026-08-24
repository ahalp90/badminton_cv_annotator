# Contact detection feasibility

## Decision

Use histogram gradient boosting as the cheap baseline, then test BST-X on the same frozen search region. Keep X3D-S for later, when a measured failure shows that RGB would add evidence the shuttle and pose inputs cannot provide.

The current raw proposals are too narrow for any sliding classifier. At ±10 base-30 frames, they cover 83.8% of non-serves and 66.1% of serves. Region v2 raises the pooled operational ceiling to 98.4% and 97.9%. This ceiling means that the frozen search surface contains a centre within ±10 of the contact; it is not model recall. `sset_21` remains the limiting fixture at 93.7% and 94.7%.

The best tree uses physical features plus validity masks. Its pooled leave-one-fixture-out result is 84.5% precision, 90.5% recall and 87.4% F1 at ±10. BST-X should use the same region, event matching and temporal NMS. Its acceptance target is 89.9% F1 with at least 87.5% precision.

## Results at a glance

| Question | Answer | Main evidence |
| --- | --- | --- |
| Do the current proposals find every contact? | No | Raw coverage at ±10: 83.8% non-serve, 66.1% serve |
| Does a broader search built without GT fix the pooled ceiling? | Almost | Region v2 at ±10: 98.4% non-serve, 97.9% serve |
| Is player-side geometry the main attribution problem? | No | Current and ankle rules are both 89.0% accurate on directly matched contacts |
| Which tree is worth keeping? | Histogram boosting | 84.5% precision, 90.5% recall, 87.4% F1 at ±10 |
| What should run next? | BST-X | It already uses the available pose, shuttle and court inputs |

“Serve” means the first contact in a rally. “Non-serve” means every later contact. A ±10 base-30 tolerance is eight frames in the 25 fps fixtures and ten frames in the 30 fps fixture.

The geometry result needs one caveat. Direct side calls are strong, but the inferred final-hitter side is correct in only 112 of 228 answered rallies, or 49.1%, under the current rule. The gap is consistent with missed contacts disrupting the strict Top/Bottom alternation. The current evidence does not isolate how much comes from missing contacts versus the alternation rule itself.

## What remains unsolved

Region v2 still misses 37 non-serves in `sset_21`. Every one has a visible shuttle within ±10, but none has usable tracked-player analysis or a detected rally span. That is now a separate search outside court-view tracking, with missing player inputs.

A deliberately broad shuttle-only diagnostic finds every non-serve and 291 of 292 serves at ±10. Its regions contain 366,048 of 404,229 source frames, or 90.6% of the broadcasts. It does not exclude replay or cutaway footage, so it is a ceiling check rather than a useful model boundary.

## Next experiment

1. Train one bounded BST-X pilot on the frozen region-v2 centres.
2. Compare it directly with HGB using the same leave-one-fixture-out folds, event matching and temporal NMS.
3. Test ShuttleSet22 as a separate same-format generalisation set.
4. Handle the uncovered `sset_21` contacts through a separate live close-up or shuttle-only search path.

No further random-forest sweep is warranted. Build X3D-S only when the BST-X failures show a clear need for racket, body-motion or broadcast-view evidence from RGB.

## Detailed reports

- [Contact recall and player geometry](contact_recall_and_player_geometry.md) contains the exact raw/final tables, PR88 result and ankle-rule evaluation.
- [Tree contact detector trial](tree_contact_detector_results.md) contains region-v2 coverage, held-out tree results, controls and failure analysis.
- [Binary contact detector options](binary_contact_detector_options.md) compares the tree, BST-X and X3D-S data paths.
- [BST-X contact detector plan](bst_x_contact_detector_plan.md) specifies the next experiment.
