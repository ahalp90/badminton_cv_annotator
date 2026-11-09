# Alternate polygon fits

The contact sheets compare ten OpenCV conversions of SAM 3 court-marking masks
across nine trial images. Each sheet starts with the selected markings, followed
by the fits in the order below. `results.json.gz` contains their vertices in
original-image coordinates and the displayed status labels.

| Method | Conversion |
| --- | --- |
| `mask_polyn` | Mask pixels → `approxPolyN(4)` |
| `mask_enclosing` | Mask hull → `minEnclosingConvexPolygon(4)` |
| `mask_polydp` | Mask hull → `approxPolyDP`, increasing tolerance until at most four vertices remain |
| `hough_polyn` | Hough endpoints → `approxPolyN(4)` |
| `hough_enclosing` | Hough endpoint hull → `minEnclosingConvexPolygon(4)` |
| `hough_polydp` | Hough endpoint hull → `approxPolyDP`, increasing tolerance until at most four vertices remain |
| `contour_polydp` | Largest external contour → `approxPolyDP` at 2% of its perimeter |
| `closed_contour_polydp` | 5×5 morphological closing → largest external contour → `approxPolyDP` at 2% |
| `contour_polydp_four` | Largest external contour → `approxPolyDP`, increasing tolerance until at most four vertices remain |
| `closed_contour_polydp_four` | 5×5 closing → largest external contour → `approxPolyDP`, increasing tolerance until at most four vertices remain |

## Settings

The prompt is `court markings`, with instance scores strictly above 0.5. Grouping
uses the same whole-mask rule as the final baseline. These comparisons resize the
selected mask to 640 pixels wide with area interpolation, retaining every pixel
with a positive value. The final baseline fits at the original resolution.

Hough uses `rho=1`, `theta=π/180`, `threshold=64`, `minLineLength=64` and
`maxLineGap=19`. The tolerance sweep runs from 0.1% to 30% of contour perimeter in
0.1% steps, stopping at the first result with at most four vertices. Triangles,
lines and non-convex results remain visible in the comparisons. `approxPolyN`
uses its defaults apart from the four-side argument.

## Reproducing the sheets

The script uses NumPy and OpenCV 5.0.0. From the repository root:

```bash
python -m experiments.court_detector.benchmark.sam3.alternate_fits.compare \
    <detections> --out-dir <output>
```

Inputs are the original `*__court_markings.npz` caches, containing `masks`,
`scores`, `prompt` and `frame_path`, plus the frame images referenced by those
paths. Relative frame paths resolve from the working directory. The caches and
original frames are separate from this folder.

The filename prefix before `__` supplies the image label, such as `am1`; each
input has a distinct label. The script writes one contact sheet per image and
`results.json.gz`. All fitting uses the segmentation masks; frame images provide
the display background.
