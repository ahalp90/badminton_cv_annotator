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

The planned benchmark measures IoU within the image and mean pixel error at true
reference corners visible onscreen. Corner order is matched to the reference;
corresponding predictions can be offscreen. Missing predictions count as detection
failures.

## References and comparisons

[Wang et al., CVPR 2025, §5.1](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Layered_Image_Vectorization_via_Semantic_Simplification_CVPR_2025_paper.pdf)
simplify SAM mask contours with Douglas–Peucker to initialise vector curves.
[Bommes et al., Progress in Photovoltaics, 2021, §4.4](https://doi.org/10.1002/pip.3448)
fit minimum-perimeter enclosing quadrilaterals to Mask R-CNN masks. These are
examples of geometric fitting after segmentation. This baseline uses OpenCV's
[greedy enclosing-polygon approximation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).

[Alternate fits](alternate_fits/README.md) contains the trial contact sheets,
polygon results and the script that reproduces them.
