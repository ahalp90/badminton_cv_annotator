# Paint-side experiment

Checking which side contains white paint catches an inner/outer edge mistake
and partly reduces the recurring left-side inset. It does not solve the whole
fit. The seven-case run is complete. Detector search and ranking are unchanged.

The current fitter assigns each observed fragment to a court marking and its
centre or edge using geometry. It then freezes that interpretation. This test
asks whether observed brightness can correct the edge labels before fitting.
It uses six previously examined ShuttleSet video-03 scenes (SS03 in the table)
and one previously approved fit from the amateur clip identified as GX.
These are development cases, not independent test data.

## Result

The comparison below changes only edge labels. Both arms start from the same
parent court and fit the same points with the same weights. Movement is polarity
minus original, in working pixels: negative x means left, negative y means up.
The paint score is the existing W5 whole-court score, weighted by visible marking length;
larger scores are better. It is not a visual ruling.

| Case | Flipped / fitted fragments | Upper-left movement (x, y) | Paint score, original → polarity |
| --- | ---: | ---: | ---: |
| SS03-16 | 9 / 42 | −2.244, −0.220 | 0.816990 → 0.817729 |
| SS03-17 | 18 / 42 | −2.394, −0.539 | 0.857938 → 0.858660 |
| SS03-19 | 15 / 37 | +0.062, +0.332 | 0.814663 → 0.806984 |
| SS03-29 | 10 / 41 | −1.245, −0.466 | 0.860242 → 0.907194 |
| SS03-34 | 15 / 34 | −1.078, −0.529 | 0.847227 → 0.838148 |
| SS03-38 | 10 / 45 | −0.048, +0.103 | 0.843741 → 0.855990 |
| GX frame 5 | 1 / 38 | +0.017, +0.011 | 0.428568 → 0.428446 |

All seven modified fits converge and retain the existing full-court and camera
gates. The approved GX fit moves at most 0.0483 working pixels at any corner.
For SS03-16 and SS03-38, the current selector chooses the parent, not its refit.
Those selected parents score 0.859603 and 0.869039 respectively, above both
refit arms. This experiment does not establish a better automatic selection.

SS03-34 fragment 236 has a strong contradiction: its measured signed contrast
is −73.74 grey levels at ±1 pixel, while its assigned outer edge expects positive
contrast. All 14 interior sample pairs are available. Its label changes to the
inner edge. At ±2 pixels the diagnostic contrast is also negative, −96.19.
The resulting upper-left movement recovers only part of the user's preferred
diagnostic shift of −3, −2 pixels. Its existing paint score falls slightly.

Against the single shared video-03 grid, median signed edge offsets across the
six scenes change as follows. Positive means inset towards the court interior.
The grid is a static per-video reference, not independent scene-level truth.

| Edge | Original refit | Polarity refit |
| --- | ---: | ---: |
| Left | +3.745 | +1.799 |
| Right | +1.682 | +1.353 |
| Far | −0.534 | −0.388 |
| Near | +1.555 | +1.621 |

The left-side agreement improves substantially; near-edge agreement does not.
Paint score improves in four of seven cases. No images were reviewed and no
new visual acceptance judgement was made.

## Meaning and next step

Paint-side evidence is cheap and resolves a demonstrated mistake in a fitted
fragment's edge identity. That mistake contributes to the inset. The remaining
error and mixed scores show that changing these labels alone is insufficient.

Before changing the assignment algorithm, inspect the residuals of the corrected
SS03-34 fit: identify which markings still pull it inward and compare their
measured paint edges with the projected stripe width. Keep the same frozen
case and points. This should distinguish remaining assignment mistakes from
stripe-width or image-blur effects without another broad fitting sweep.

The result supports using polarity as assignment evidence. It does not yet
justify replacing the detector or changing its automatic acceptance rules.

## Fixed experiment

Use the current full-source selection on SS03 scenes 16, 17, 19, 29, 34 and 38,
plus GX frame 5 as a previously approved fit. Start from each selection's
parent. First reproduce the saved refit. Then change only contradicted edge
labels, keeping marking identity, fitted points and weights fixed.

Sample greyscale at +/-1 working pixel along the perpendicular to each raw
fragment. Use its interior samples, excluding invalid or available same-image
person masks. Require at least half of those sample pairs. A median signed
difference of at least 10 grey levels supplies an edge direction; weaker
evidence remains unresolved. The contrast threshold is borrowed from the
existing ridge check, not calibrated or tuned against these cases. Also record
the +/-2-pixel contrast as a diagnostic; it does not choose the labels.

For edge positions 1 and 2, project the positive court-coordinate direction at
the fragment midpoint. This defines which image side should contain paint.
Swap 1 and 2 if strong observed brightness points the other way. Leave centre
assignments unchanged. Keep the original fragment samples in the refitter so
that a change in sample membership cannot explain the result.

Use six processes and one native numerical thread each. Reuse one greyscale
conversion per case. Report fragment decisions, corner movements, existing
scores and gates. References and the user-approved scene-34 diagnostic enter
only the post-fit comparison. No new visual rulings or deployment threshold.

## Limits

Contrast can be affected by crossings, occlusion, blur and adjacent markings.
This is a fixed-label counterfactual, not a finished assignment algorithm.
Success would support using polarity in assignment; it would not validate a
universal correction or establish production acceptance.

## Reproduction and checks

The full [results](results.json.gz) preserve every retained fragment decision,
both fits, the current selection, scores, gates and all four corner movements.
The [reference comparison](reference_comparison.json.gz) is computed afterwards
by [summarise.py](summarise.py); reference coordinates never enter label choice.
The [single-case smoke result](scene34_smoke.json.gz) preceded the full run.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/edge_polarity/run_probe.py --workers 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/badminton-cicd/bin/python \
  scratch/court_det_fix/edge_polarity/summarise.py
PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/badminton-cicd/bin/pytest \
  scratch/court_det_fix/edge_polarity/test_probe.py -q
```

The seven-case run and post-fit summary exit 0. All saved baseline refits
reproduce within 1.1e-9 native pixels, below the declared 1e-4 tolerance.
Ten synthetic tests pass (exit 0), covering both court axes, both image
orientations, endpoint reversal, fixed constraint membership and masked paint.
Scoped Ruff and `git diff --check` also exit 0.
The whole-project Pyrefly check exits 0 with no errors (39 suppressed).

Polarity measurement and relabelling take 0.054–0.078 seconds per case in this
six-process run. These are individual timings, not a robust runtime benchmark;
they exclude fitting, scoring and loading. An assertion confirms one greyscale
conversion per case across polarity measurement and scoring. The existing
prepared-measurement helper supplies that reuse; the general runtime path still
needs the recorded greyscale cleanup.
