# SAM 3 court baseline

SAM 3 segments court markings with the prompt **`court markings`** and an instance
confidence threshold of **0.5**. OpenCV fits an enclosing quadrilateral to the
selected markings.

## Method

Masks scoring above 0.5 are grouped where their pixels overlap or touch
(8-connectivity). Each SAM instance stays intact, including disconnected strokes.
The group with the most foreground pixels represents the selected court.

Its foreground coordinates go directly into `cv2.approxPolyN(points, 4)` at the
original image resolution. OpenCV builds the convex hull internally and reduces
it to four sides by greedily choosing contractions that add the least area. The
other arguments use their defaults: `ensure_convex=True` and
`epsilon_percentage=-1`.

The fit encloses the selected markings, including any stray pixels. Cropped
courts can retain an image-edge side, and fitted corners can lie outside the
image. Empty masks and hulls with fewer than four vertices return no prediction.

## Settings trialled

The initial prompt trials included `court lines`, `court markings`,
`court boundary lines`, `badminton court`, `badminton doubles court`,
`sport court` and `court floor`. The three line-based prompts picked out the
painted markings most consistently. The badminton-court prompts mostly scored
below 0.1, while the broader floor prompts tended to segment the whole floor.

`court lines`, `court markings` and `court boundary lines` were compared across
nine trial frames at instance confidence cutoffs of **0.3 and 0.5**. A later
sweep used `court markings` at **0.5, 0.6, 0.7 and 0.8**, comparing direct
mask-to-quadrilateral fits with Hough-based fits. Higher cutoffs progressively
removed useful markings; the selected setting remained `court markings` at 0.5.

The early caches retained predictions above **0.1**, allowing these cutoffs to
be applied without rerunning SAM 3. This was the cache threshold; the comparison
cutoffs listed above used scores strictly greater than each value.

## Running

### ShuttleSet22 rally benchmark

`benchmark_rallies.py` runs detection, fitting and numeric evaluation in one process.
It uses the MonoTrack benchmark's committed input manifest: **3,327 rally windows
from 46 videos**, with video 15 already excluded by that benchmark. Each input is
the zero-based frame `(start_frame + end_frame) // 2`; the window end is exclusive.
The SAM model loads once, and each video is opened once. Frames and masks stay in
memory; only numerical predictions and progress are saved.

From the repository root, with the SAM 3 environment and checkpoint available:

```bash
PYTHONPATH=. python experiments/court_detector/benchmark/sam3/benchmark_rallies.py \
    --manifest experiments/annotator/monotrack_benchmark/benchmark_results/inputs/manifest.json.gz \
    --references data/shuttleset22/set/homography.csv.gz \
    --videos <ShuttleSet22 video directory> \
    --out <new results directory>
```

Reference corners come from ShuttleSet22's official homography table, scaled from
1280×720 to each video's native resolution. These are main-camera references;
an alternate-camera midpoint can score poorly even if its detected court is right.
All manifest rallies remain in the evaluation.

- **Corner error:** mean Euclidean distance in native pixels at true reference
  corners inside the image, minimised over the eight cyclic order/orientation
  matches. A matched predicted corner can be outside the image. Frame-edge
  intersections are not reference corners. Each rally contributes one mean.
- **Resolution:** reports use 1080p-equivalent corner error (`native error ×
  1080 / image height`), preserving aspect ratio. All ShuttleSet22 videos are
  1920×1080, so their native and normalised errors are identical.
- **IoU:** intersection area divided by union area after clipping both court
  polygons to `[0, width] × [0, height]`. Missing predictions receive zero IoU.
- **Missing predictions:** counted separately. Corner-error statistics include
  only predictions with visible reference corners. IoU is reported both over all
  rallies and conditionally on a prediction. Aggregation gives each rally equal
  weight; repeated views are not independent samples of camera geometry.

`rallies.json.gz` contains every rally's frame, reference and predicted corners,
retained-pixel count and scores. It is checkpointed after each video.
`summary.json.gz` contains aggregate scores, settings and package versions.
`status.json.gz` updates after every rally and records completion or failure.
Existing predictions are protected from accidental overwrite; use a new output
directory for a fresh run.

### Individual images

Segmentation requires a CUDA GPU, Meta's
[SAM 3 package](https://github.com/facebookresearch/sam3) and the `facebook/sam3`
checkpoint:

```bash
python experiments/court_detector/benchmark/sam3/segment_courts.py \
    <frame images...> --out-dir <detections>
```

Fitting and rendering run on CPU with NumPy and OpenCV 4.11 or newer:

```bash
python experiments/court_detector/benchmark/sam3/merge_pieces.py \
    <detections> --out-dir <results>
```

These commands run from the repository root. Relative saved frame paths resolve
from the working directory. The fitter reads `court_markings` files and applies the
0.5 cutoff, including to older caches saved with a lower threshold.

Each comparison shows cyan markings on the left and the orange quadrilateral on
the right, at the same scale and crop. Markings are thickened for display.
`quadrilaterals.json.gz` contains image dimensions, selected pixel counts and
four cyclic `(x, y)` corners in original-image pixels, or `null` for no prediction.
The saved corners are unclipped.

The [comparison report](../court_det_benchmark.md) gives the completed broadcast
and manual-court results for this method, our detector and MonoTrack.

## References and comparisons

[Wang et al., CVPR 2025, §5.1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Layered_Image_Vectorization_via_Semantic_Simplification_CVPR_2025_paper.pdf)
simplify SAM mask contours with Douglas–Peucker to initialise vector curves.
[Bommes et al., Progress in Photovoltaics, 2021, §4.4](https://doi.org/10.1002/pip.3448)
fit minimum-perimeter enclosing quadrilaterals to Mask R-CNN masks. These are
examples of geometric fitting after segmentation. This baseline uses OpenCV's
[greedy enclosing-polygon approximation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).

[Alternate fits](alternate_fits/README.md) contains the trial contact sheets,
polygon results and the script that reproduces them.
