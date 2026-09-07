# Independent court fitting experiment

This experiment tests whether image lines and the badminton court layout can
locate courts without CourtKeyNet. It is an additive research prototype;
the annotation pipeline does not use its outputs.

`detector.py` extracts full-frame OpenCV fragments, groups them into two line
families and proposes perspective transforms from possible line identities.
It scores the visible portions of the finite painted markings. The net is
excluded from the floor template. Court dimensions and markings reuse the
existing project constants; no neural predictions or manual regions enter
the detector.

The search keeps multiple spatially distinct candidates. It permits off-screen
corners and rejects close-scoring alternatives. The `accepted` flag means that
the experimental support and score-gap rules passed. It is not a measured
probability of correctness or permission to use that court in production.

## Scope and limits

- Upright views from behind a baseline. The initial line-family split favours
  a nearly horizontal baseline and can fail on oblique or rolled cameras.
- Fixed working resolution, finite line coverage and a minimum visible span
  limit the effect of small image structures. These are engineering settings,
  not fitted confidence thresholds.
- A fixed seed samples at most 4,096 image-line rectangles from up to 32 lines
  in each family. A missing competitor can make a score gap misleading.
- The line scorer requires at least four distinct observed lines in each
  direction. Some valid partial courts have insufficient evidence for this rule.
- Separate supported courts in different image regions leave the target
  unresolved, even when their support scores differ.
- Background lines and neighbouring courts remain plausible competing fits.
  Synthetic tests alone do not establish rejection accuracy on real footage.

The candidate-search idea draws on
[tennis-court-detection](https://github.com/gchlebus/tennis-court-detection)
and MonoTrack's badminton adaptation. This implementation uses custom Python
code and the repository's existing OpenCV extraction and badminton template.

Run the synthetic checks from the repository root:

```bash
pytest -q tests/test_independent_court.py
```

Development comparisons use original ShuttleSet and the committed
[amateur references](../../../data/amateur_court_corners/README.md).
Reference geometry is used only for scoring. Extrapolated reference corners
are distinguished from the clicked visible landmarks that support them.
ShuttleSet22 supplies no tuning or feature selection for this experiment.
