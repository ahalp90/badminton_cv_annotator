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

## Run a comparison

Use a gzip JSON manifest with a `cases` list. Each case needs a unique `id`,
an `image` path relative to the manifest, and `reference_status`:
`matching_view`, `view_unverified`, `unlabelled` or `non_court`.
Optional `corners_px` are four native-image points in TL/TR/BR/BL order.
Optional `landmarks` contain `court_m: [x, y]` and `image_px: [x, y]`.
Court coordinates use width along x and length along y.

```bash
python -m experiments.annotator.independent_court.evaluate \
  --manifest inputs/manifest.json.gz --output results/independent
python -m experiments.annotator.independent_court.baseline \
  --manifest inputs/manifest.json.gz --output results/baseline.json.gz --device cpu
```

The evaluator needs the repository's NumPy/OpenCV environment. The baseline
also needs the existing CourtKeyNet weights and PyTorch. It records raw model
validity and availability of the existing model-plus-line proposal separately.
It does not run the production scene acceptance, repair or sharing stages.

The evaluator writes `results.json.gz` and cyan/orange overlays. Metrics use
1280x720 coordinates and only `matching_view` references. Corner errors include
off-screen corners. Landmark RMS measures Euclidean reprojection error over
visible reference clicks. Invalid projections receive an explicit status.
Reference labels never influence detector acceptance.

`--extractor lsd` selects OpenCV's line segment detector. It is a comparison
hook, not a validated improvement; DeepLSD has not been evaluated.

## Recorded development evidence

The [replacement assessment](../../../docs/courtkeynet/fallback_evaluation/independent_detector.md)
explains the populations, results and decision. The three `recorded/*.json.gz`
bundles contain exact case IDs, references, all retained candidates, decisions
and fresh baseline outputs where measured. `inputs.cases` can be written as a
manifest for the commands above once its relative image paths are populated.

Original broadcast inputs are cached grayscale medians of three samples;
`sampled_frame_indices` records all three. The first frame number in the image
filename does not identify a raw input frame. Amateur inputs are full native
frames; controls are resized raw frames. Full input images remain external to
these bundles. The report includes three representative overlays.

The final spatial-ambiguity correction was replayed over unchanged saved
candidates. A fresh search on the affected scene reproduced every candidate,
score and final decision exactly. Saved timing covers the original search,
excluding that acceptance-only replay, and concurrent CPU jobs affected it.
