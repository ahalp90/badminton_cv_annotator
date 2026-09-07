# Independent court fitting experiment

This experiment tests whether image lines and the badminton court layout can
locate courts without CourtKeyNet. It is an additive research prototype;
the annotation pipeline does not use its outputs.

`detector.py` extracts full-frame OpenCV fragments, groups them into two line
families and proposes perspective transforms from possible line identities.
It scores the visible portions of the finite painted markings. The net is
excluded from the floor template. Court dimensions and markings reuse the
existing project constants. The geometry search accepts image-line evidence;
manual regions and reference corners never enter the detector.

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

`--extractor lsd` selects OpenCV's line segment detector. Neural comparisons use
cached DeepLSD or LINEA fragments through the same geometry search.

### Compare cached lines

`--line-cache` replaces OpenCV extraction with saved native-image line segments.
The cache is a gzip JSON object containing a `variant` name and a `cases` list.
Each record contains `id`, `dimensions: {width, height}`, `image_file_md5` and
`segments_px`: one `[x1, y1, x2, y2]` row per line in native image pixels.
Empty line lists are valid. Non-finite and zero-length lines are rejected.
Records join by ID; one cache can serve several smaller evaluation manifests.
The evaluator checks the PNG hash and decoded dimensions before using the lines.

```bash
python -m experiments.annotator.independent_court.evaluate \
  --manifest inputs/manifest.json.gz --line-cache lines/model.json.gz \
  --output results/model
```

The result records the cache file's SHA-256 digest, extractor name and metadata.
Model and source hashes are retained when supplied. Search timings exclude neural inference and cache
validation. The cache does not supply candidate courts or acceptance decisions.
Direct callers can pass native fragments as `detect(image, segments_px=lines)`.

### Export neural lines

`export_lines.py` runs frozen upstream models in a separate inference environment.
It records the checkpoint SHA-256, the source checkout's Git commit, image hashes,
model settings and native coordinates. Its `--help` needs no model dependencies.
Install each upstream model's inference dependencies before running it.

```bash
python -m experiments.annotator.independent_court.export_lines \
  --model deeplsd-md --source third_party/DeepLSD \
  --weights weights/deeplsd_md.tar --manifest inputs/manifest.json.gz \
  --output lines
python -m experiments.annotator.independent_court.export_lines \
  --model deeplsd-wireframe --source third_party/DeepLSD \
  --weights weights/deeplsd_wireframe.tar --manifest inputs/manifest.json.gz \
  --output lines
python -m experiments.annotator.independent_court.export_lines \
  --model linea-large --source third_party/LINEA \
  --weights weights/linea_hgnetv2_l.pth --manifest inputs/manifest.json.gz \
  --output lines
```

DeepLSD uses the inference-only implementation at upstream commit
`f7d9d6258c0cd25d4f6eea882853565403d289be` in this comparison. MegaDepth emits two
caches: gradient checking disabled (`hard`) and enabled (`default`). Wireframe
emits the `hard` variant. Images are grey, with longest dimension capped at 960.
The full Ceres-based refinement package is not used. See the
[DeepLSD instructions and weights](https://github.com/cvg/DeepLSD#usage).

LINEA uses upstream commit `475c5ceea64114a48495c15888094e12f1a2d267`, the large
checkpoint and the authors' RGB 640x640 preprocessing. It retains scores strictly
greater than 0.2. The model postprocessor restores native image coordinates.
See the [LINEA source and checkpoints](https://github.com/SebastianJanampa/LINEA).

The portable exporter reproduced the original private inference outputs exactly
on four images covering every input resolution. This check included all four
model variants and their line coordinates, scores, image hashes and dimensions.

### Painted stripes

`--extractor ridge` retains Hough fragments that look like bright painted
stripes, with darker pixels on both sides. It supports white and yellow paint.
The optional filter leaves candidate search and acceptance rules unchanged.
On the same development inputs it preserves 15/18 accurate top broadcast
proposals and increases accurate accepted fits from 1/18 to 10/18. It rejects
all eight non-court controls, compared with seven for ordinary Hough.
Amateur accuracy remains 0/11, with four wrong accepted fits. The filter is
therefore useful evidence for further experiments, not a production solution.
The exact settings and 58 outputs are in `recorded/ridge.json.gz`, including
three subsequently downloaded static-camera gameplay samples with no numerical
reference labels. The original recorded bundles remain the frozen first pass.

## Recorded development evidence

The [replacement assessment](../../../docs/courtkeynet/fallback_evaluation/independent_detector.md)
explains the populations, results and decision. The original, amateur and control
bundles in `recorded/` contain exact case IDs, references, all retained candidates, decisions
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
