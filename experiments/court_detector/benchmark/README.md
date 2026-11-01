# Court detector comparison

[The comparison report](court_det_benchmark.md) covers our court detector,
MonoTrack and the fixed SAM 3 court-marking baseline.

## Broadcast rallies

The [SAM 3 guide](sam3/README.md#shuttleset22-rally-benchmark) describes detection
and fitting on the 3,327 MonoTrack benchmark rally midpoints. The two existing
per-rally court sets are rescored with the same metrics:

```bash
PYTHONPATH=. python -m experiments.court_detector.benchmark.compare_detectors \
    --sam-results <SAM results>/rallies.json.gz \
    --inputs experiments/annotator/monotrack_benchmark/benchmark_results/inputs \
    --out <comparison results>
```

The output contains per-rally predictions and scores, aggregate statistics,
per-video statistics and the subset where all three detectors returned a court.
Our stored courts use rally video mode, including the player check. MonoTrack
and SAM 3 use the midpoint image.

## Manually annotated courts

The inputs are [27 labelled frames from eight videos](../../../data/amateur_court_corners/README.md).
The runner evaluates those frames directly and also prepares each static video's
middle frame. Three clip midpoints already have exact labels; the other four use
the mean of their video's annotated corners. Labelled frames and midpoints have
separate summaries. C1jR4vZmrkI has a moving camera, so its midpoint is omitted
by default; its three labelled frames remain included.

A local JSON path map connects each annotation's video stem to its source video.
For example, `C6NrJyBwn6c` names the original amateur video, while
`Cb-xs5rPyxI_gameplay` names the short committed clip. Runtime path maps stay
outside the published results. The three downloaded clips use the decoded frame
counts in `additional_examples/manifest.json.gz` and are read sequentially;
their container frame counts include discarded preroll.

```bash
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts prepare \
    --labels data/amateur_court_corners --video-map <local video map.json> --out <work directory>

bash experiments/court_detector/benchmark/build_monotrack.sh <MonoTrack image binary>

PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts ours \
    --out <work directory> --deeplsd-source <DeepLSD source> --deeplsd-weights <DeepLSD weights>
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts monotrack \
    --out <work directory> --monotrack-binary <MonoTrack image binary> \
    --reuse-monotrack experiments/court_detector/benchmark/results/manual/monotrack_labelled_baseline.json.gz
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts sam3 \
    --out <work directory>
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts score \
    --out <work directory>
```

The runners resume completed images. MonoTrack reuses 18 existing predictions on
native labelled frames. The nine downloaded-clip frames use explicit decoded
indices, and decoded counts determine the three clip midpoints. Four independent
CPU processes handle the remaining images.
Each neural model loads once for the batch. The `ours` stage preserves the
single-image comparison with full line search and no person detector. MonoTrack's
image adapter reuses its released detection stages.
SAM 3 uses its [fixed settings](sam3/README.md#method), with the fitted boundary
at the outer perimeter of the retained markings.

Reported corner errors use 1080p-equivalent pixels: native distance multiplied
by `1080 / image height`. This preserves aspect ratio, including for 4:3 clips.
ShuttleSet22 is already 1920×1080 throughout. The result files retain both
`corner_error_px` (native) and `corner_error_1080p_px`.

The main comparison uses `robust_courts.py` for our detector:

```bash
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.robust_courts \
    --out <work directory> --video-map <full-source map.json> \
    --deeplsd-source <DeepLSD source> --deeplsd-weights <DeepLSD weights>
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts score \
    --out <work directory> --result-prefix image_only
PYTHONPATH=.:src python -m experiments.court_detector.benchmark.manual_courts score \
    --out <work directory> --ours-source ours_robust
```

This runs the committed player checks and three-frame scene composition around
each labelled frame or midpoint. It compares the aligned refit with the accepted
individual fits and keeps the best. At a video start, the two context images
come from the middle and end of the pose window, giving three distinct images.
Each target is evaluated independently.
Existing pose samples are reused; missing samples are extracted only from the
requested 31-frame windows, with overlap cached. Full source videos provide the
context around the short clips' labels. Source frames and saved poses are scaled
to the benchmark image size before fitting.

The full-source map uses a video path or an object containing `path`,
`frame_offset` (label index to full-source index), and optional `pose_directory`
or `pose_cache_directory`. Dense pose arrays must cover every requested sample;
sparse per-frame caches can fall back to inference. Each `frame_#########.npy.xz`
cache addresses a full-source frame and contains one row per person: four box
coordinates followed by 17 pairs of keypoint coordinates, all in source pixels.
`ours_robust.json.gz` stores
the new predictions separately from the original `ours.json.gz`. The two score
commands preserve the single-image results as `image_only_*` and write the main
comparison as `comparison_*`.

Corner error uses reference vertices within the image. The `visible` annotation
flag records whether a corner could be clicked; it differs from being inside the
image. Occluded in-frame corners projected from clicked landmarks still count.
The IoU clips both polygons to the image and counts absent predictions as zero.

`comparison_frames.json.gz` contains portable frame identifiers, references,
predictions and scores. `comparison_summary.json.gz` separates labelled frames
from midpoints, includes per-video summaries and counts the exact and averaged
references. The three exactly labelled midpoints also appear in the labelled set;
the two groups are not independent samples. Raw frames and per-method runtime
files can be kept in the work directory for inspection.

## Example images

`render_examples.py` draws a raw broadcast midpoint and one panel per detector,
reusing the MonoTrack report's outline drawing helpers:

```bash
PYTHONPATH=. python -m experiments.court_detector.benchmark.render_examples \
    --results <comparison results>/comparison_rallies.json.gz \
    --manifest experiments/annotator/monotrack_benchmark/benchmark_results/inputs/manifest.json.gz \
    --videos <video directory> --keys 17:47 --out <images>
```

For manual frames, `manual_courts render --out <work directory> --keys <image keys>`
uses the saved inputs and results. Fluorescent pink, yellow and red identify our
detector,
MonoTrack and SAM 3. Their 2 px outlines have matching panel labels and distinct
dash patterns. White marks the reference court.

`render_examples --hide-reference` omits the reference and numeric scores when
the midpoint shows another camera view with no corresponding labels.
