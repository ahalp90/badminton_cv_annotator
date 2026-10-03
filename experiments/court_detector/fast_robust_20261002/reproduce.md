# Reproduce the court evaluation

Run from the repository root in an environment with NumPy, pandas and OpenCV.
The [published dataset](court_sharing_patched/README.md) contains the final
per-scene courts. Source videos and ShuttleSet22 annotations are supplied
separately; no remote access setup is required to read the predictions.

## Analyse the published courts

Set `SSET22_LABELS` to the ShuttleSet22 annotation `set/` directory.
The evaluator uses the release manifests to select the 86 eligible videos.

```bash
BASE=experiments/court_detector/fast_robust_20261002
PATCHED="$BASE/court_sharing_patched"

PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py analyse \
  --input-root "$PATCHED" \
  --shuttleset-root data/shuttleset/set \
  --shuttleset22-root "$SSET22_LABELS" --output "$PATCHED"
PYTHONPATH=.:src python scripts/summarise_court_rally_views.py \
  --input "$PATCHED" --output "$PATCHED/rally_views"
PYTHONPATH=.:src python scripts/compare_court_sharing.py \
  --original "$BASE" --patched "$PATCHED"
```

The paired comparison reads the committed original evaluation tables. It checks
the cohort, scene and rally intervals, and unchanged before-sharing errors.
It fixes the original main-view scene population for both methods.

Reference corners come from the supplied inverse homography. Template coordinates
are x=25..325 for ShuttleSet 1–20, x=27.4..327.6 for later ShuttleSet and all
ShuttleSet22, and y=150..810 throughout. Those are annotation template units.
Predictions and references are compared at 1280 × 720.

## Render review frames

The committed `render_requests.json.gz` uses source-video filenames. Resolve
`source_video` against your video directory in a working copy of the requests.
Then run the existing `fetch-frames` and `render` commands:

```bash
PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py fetch-frames \
  --requests "$REQUESTS" --output "$FRAMES"
PYTHONPATH=.:src python scripts/evaluate_courts_fast_robust.py render \
  --requests "$REQUESTS" --frames "$FRAMES" --output "$GALLERY"
```

Frame indices are zero-based. The extractor verifies the decoded frame position.
Outlines use the 40 mm painted-stripe model, drawn with 1 px strokes and a
7-pixel dash followed by a 5-pixel gap. Run each PNG through pngquant, then
oxipng, before publishing it.

The final rally sample table records one midpoint per scene/rally overlap,
including courtless scenes. Courtless frames are copied without an outline.
These samples use the final saved court after video-wide sharing.

## Export ordering and metadata

The published files come from the completed saved-fit replay. Later detector
logic changes were not rerun over the corpus. To prepare another copy of that
replay, apply `court_detector.geometry.normalise_output_corners` to each scene's
non-null `corners_native_px` before running the evaluation above. This only rolls
a court's four corners by two when its first baseline has the greater mean image
y. Leave `scene_corners_native_px` and group diagnostics unchanged.

In the cohort, retain the basename of `video` and omit `people`. In video outputs,
omit `replay.detector_source`. Evaluation and render metadata should likewise use
source-video basenames. These changes remove machine-specific locations without
changing court geometry or scene intervals. Source videos, pose arrays and raw
replay outputs are separate inputs; the published predictions can be used directly.
