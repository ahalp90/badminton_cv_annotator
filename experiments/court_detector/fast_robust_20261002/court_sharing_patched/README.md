# Court dataset for downstream use

Use `videos/<video_id>.json.gz` for the final court assigned to every scene.
The dataset covers 86 released videos: 40 ShuttleSet and 46 ShuttleSet22.
All 44,810 scenes are retained, including scenes without a court.

These are detector predictions. The supplied default-camera references agree
within 10 px mean corner error for 85/86 video representatives and 6,171/6,833
rally representatives. Alternate camera views need separate assessment.
The [main report](../player_tiebreak_trial/search_results/README.md) explains
the evaluation, alternatives and limitations.

## Read a court

```python
import gzip
import json
from pathlib import Path

root = Path("experiments/court_detector/fast_robust_20261002/court_sharing_patched")
with gzip.open(root / "videos/sset_36.json.gz", "rt") as handle:
    video = json.load(handle)

scene = video["scenes"][383]
corners = scene["corners_native_px"]
```

`cohort.json.gz` maps each `video_id` to its dataset and original `source_id`.
Video filenames are portable names; source videos and pose arrays are supplied
separately. Machine-specific paths are omitted from the committed metadata.

| Field | Meaning |
| --- | --- |
| `fps`, `frame_count`, `native_size` | Video timing and `[width, height]` |
| `scenes` | Ordered scenes covering the whole video without gaps or overlaps |
| `start_frame`, `end_frame` | Zero-based interval `[start_frame, end_frame)`; end is excluded |
| `frame_index` | Sampled frame in which the court coordinates were established |
| `status` | `court`, `no_court`, or `scene_too_short_for_feet` in this release |
| `corners_native_px` | Final court after sharing: four `[x, y]` points in native video pixels, ordered TL, TR, BR, BL; `null` means no court |
| `scene_corners_native_px` | Original individual court before sharing, where present; a recovered courtless scene has `null` here |
| `view_pool` | Shared court's source, alignment and acceptance details, where applicable |

Use the top-level scene's `corners_native_px` for downstream work. Courts are
estimates at a sampled frame, applied across that scene; they are not per-frame
camera tracking. Coordinates may lie outside the image. Preserve missing courts
as missing, rather than filling them with another scene's court automatically.

Final corner lists use the image's upper baseline first. The export reordered
190 courts by 180° to enforce this convention; every coordinate and fitted
geometry stayed unchanged. Original individual courts and group diagnostics
retain their internal ordering. Use the final scene field for court coordinates.

There are 12,024 scenes with courts, 18,196 without a court and 14,590 too short
for the player-sampling window. No scene rows were dropped. The replay recovered
108 previously courtless scenes. Original individual courts remain unchanged.

## Tables and version

`per_scene.csv.gz` contains the final corners and evaluation fields.
`rally_views/per_rally_view.csv.gz` chooses the camera group occupying the most
time in each labelled rally, then a representative court from that group.
`per_rally.csv.gz` instead uses the single longest-overlapping scene.
Use the former for the report's rally-view numbers. All reported corner errors
are scaled to the references' 1280 × 720 resolution; native corner fields retain
the source video resolution.

This release replays court sharing over saved fast-robust fits, with the
courtless-receiver patch. It uses the original player-required search policy.
Later changes to three-frame composition and image-only search are separate
code changes; they were not used to generate this dataset. Replay timings are
additional processing over saved fits, not complete extraction timings.

All 86 replay jobs completed successfully. Validation checked scene intervals,
finite corner arrays, original individual courts and the paired evaluation.
The replay reproduced 7,572 stored source scores and 178 pooled fits.
