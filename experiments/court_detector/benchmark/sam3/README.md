# SAM 3 court baseline

A zero-shot SAM 3 baseline for comparison with the court detector. Given a short text prompt, SAM 3 returns the court's painted lines as several scored pieces. `merge_pieces.py` turns those pieces into one court region using a few fixed rules.

## Running

Segmentation needs a CUDA GPU, Meta's [`sam3`](https://github.com/facebookresearch/sam3) package and access to the gated `facebook/sam3` checkpoint:

```bash
python experiments/court_detector/benchmark/sam3/segment_courts.py <frame images...> --out-dir <detections>
```

Merging and rendering run anywhere with numpy, SciPy and OpenCV, and take seconds:

```bash
python experiments/court_detector/benchmark/sam3/merge_pieces.py <detections> --out-dir <renders> --cutoff 0.5
```

The segmentation step saves pieces scoring above 0.1, so the merge step can try other cutoffs without rerunning SAM 3. It also records each frame's path as given, so run both scripts from the repository root.

Two `sam3` quirks: the PyPI release (0.1.4) is missing its tokenizer file, so install from the git repository; and inference must run under bf16 autocast, which `segment_courts.py` does.

## Prompts

`court lines`, `court markings` and `court boundary lines`. In a quick look at nine saved frames, these were the only prompts that consistently picked out the painted lines. Prompts naming the court itself ("badminton court", "badminton doubles court") mostly scored below 0.1. Broader ones ("sport court", "court floor") masked the whole floor.

## Merge methods

| Method | Rule |
| --- | --- |
| `top_piece_hull` | Filled convex hull of the highest-scoring piece |
| `union_bbox` | Bounding box of all pieces together |
| `union_hull` | Filled convex hull of all pieces together |
| `central_bbox` | Bounding box of the line network nearest the frame centre |
| `central_hull` | Filled convex hull of that network |
| `largest_network_hull` | Filled convex hull of the line network with the most line pixels |
| `largest_network_quad` | That hull fitted to a quadrilateral, clipped to the frame |

The network methods first join pieces across small breaks, up to about 0.6% of frame width. A court's lines form one connected network, and neighbouring courts sit across a gap of floor, so each network approximates one court. "Nearest the frame centre" means the network holding the line pixel closest to the centre. That approximates "the most central court" without per-video hints.

Hulls follow a perspective court better than axis-aligned boxes. The two box methods are kept for comparison.

`largest_network_quad` fits the hull to four corners in two steps:

1. Where the frame cuts a corner off the court, the hull gets an extra side along the frame edge. The method extends the court sides either side of it until they meet beyond the frame, and uses that point as the corner. Straight lines stay straight under perspective, so this finds the true corner whenever both of its sides are visible.
2. `cv2.approxPolyN` (OpenCV 4.11+) removes any remaining sides beyond four, such as kinks where a player hides the lines. It extends neighbouring sides until they meet, choosing the removal that adds the least area, so the quadrilateral always encloses the hull.

Filling the quadrilateral clips it to the frame. When a whole court side is out of frame, the frame edge stands in for it. The visible lines can't pin down that side: many perspective views of a 13.4 × 6.1 m court share the same visible part. For IoU against the detector's court clipped to the frame, the clipped region is the fair comparison, and corners outside the frame shouldn't count as measured.

The quadrilateral can only be as good as the hull. If the largest network is the wrong court, or only part of one, the quadrilateral fits that.

## Outputs

For each frame, prompt and method, `merge_pieces.py` writes a render with the region tinted red. It also writes one grid per frame, with a row per prompt and a column for the raw pieces plus each method, and a `summary.csv` of piece counts, top scores and region areas.
