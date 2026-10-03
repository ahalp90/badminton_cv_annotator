# Extracted court detections: ShuttleSet and ShuttleSet22

This dataset contains predicted court corners for **86 videos and 44,810 scenes**:
40 ShuttleSet videos and 46 ShuttleSet22 videos. Use it to locate players or
measure distances in court coordinates. The files contain model predictions;
the official court labels are in `data/shuttleset/set/homography.csv` and
`data/shuttleset22/set/homography.csv.gz`. Each labels the main camera view of one
match at 1280 × 720. The cohort’s `source_id` identifies that match. There are
no official per-scene court labels for other camera views.

| File | What it contains |
| --- | --- |
| `cohort.json.gz` | Video IDs, source filenames, source-dataset names and original match IDs |
| `videos/<video_id>.json.gz` | Video timing, scene boundaries and the final court assigned to each scene |

## Load a court

Run from the repository root:

```python
import gzip
import json
from pathlib import Path

root = Path("data/court_detections/sset_and_sset22/extractions_20261003")
with gzip.open(root / "videos/sset_36.json.gz", "rt") as handle:
    video = json.load(handle)

scene = video["scenes"][383]
corners = scene["corners_native_px"]
```

| Field | Meaning |
| --- | --- |
| `fps`, `frame_count`, `native_size` | Video timing and `[width, height]` |
| `scenes` | Ordered scenes covering the whole video without gaps or overlaps |
| `start_frame`, `end_frame` | Zero-based interval `[start_frame, end_frame)`; end is excluded |
| `frame_index` | Sampled frame in which the court coordinates were established |
| `status` | `court`, `no_court`, or `scene_too_short_for_feet` |
| `corners_native_px` | Final court: four `[x, y]` points in native video pixels, clockwise, with the upper baseline first; `null` means no court |
| `scene_corners_native_px` | Individual court before sharing, where saved; `null` denotes a recovered courtless scene; absent means the final court is the scene's own fit |
| `view_pool` | Details of the court borrowed from matching camera views, where applicable |

The scene's **`corners_native_px`** contains the court used downstream. These corners already
follow the viewer-facing convention; no further reordering is needed. The upper
baseline is the pair with the smaller mean image y. Separately saved individual
courts and group diagnostics retain their internal ordering.

There are 12,024 scenes with courts, 18,196 without courts and 14,590 too short
for the player-sampling window. A missing court has no usable corner coordinates. Courts are
estimated at a sampled frame and applied across its scene, rather than tracked
frame by frame. Coordinates can lie outside the image. Source videos and pose
arrays are supplied separately.

## Quality and release

This release applies improved court sharing to saved fast-robust detections,
including recovery of previously courtless scenes. The saved replay records
source commit `17a50b57`. Later changes to three-frame
composition and image-only search were **not** rerun to produce these files.

Agreement with the supplied default-camera references is within 10 px mean corner
error for 85/86 video representatives and 6,171/6,833 rally representatives.
Alternate camera views need separate assessment.

- [Results and limitations](../../../../experiments/court_detector/report.md)
- [Evaluation tables and final gallery](../../../../experiments/court_detector/comparisons/release.md)
- [Reproduce the evaluation](../../../../experiments/court_detector/tools/README.md)
