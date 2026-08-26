# ShuttleSet22 extraction handoff

Issue [#106](https://github.com/ahalp90/badminton_cv_annotator/issues/106)
prepared whole-video perception inputs for the binary shot-classifier work.

## Result

- 58 ShuttleSet22 annotation records were reviewed.
- 47 unique public sources were downloaded and extracted successfully.
- Eight records overlap ShuttleSet and are excluded from this extraction set.
- Three records have no frame-aligned public source: 14, 45, and 56.

Each completed source directory contains:

- `*_ball.csv.gz`: TrackNet frame, pixel-coordinate, and visibility output.
- `shuttle_track.npy.xz`: normalized shuttle `(x, y, visible)` rows.
- `pose_kps.npy.xz`: RTMLib 17-keypoint coordinates.
- `pose_kp_scores.npy.xz`: confidence for each keypoint.
- `pose_bboxes.npy.xz`: detected-person bounding boxes.
- `pose_scores.npy.xz`: confidence for each person detection.
- `pose_ndet.npy.xz`: detected-person count for each frame.

The six NPY archives use LZMA preset 9. The TrackNet CSV uses gzip level 9.

## Data locations

The active Bourbaki workspace is:

```text
/scratch/cmarti56/issue106-shuttleset22-data/
```

Its data directories are:

```text
/scratch/cmarti56/issue106-shuttleset22-data/annotations/
/scratch/cmarti56/issue106-shuttleset22-data/sources/
/scratch/cmarti56/issue106-shuttleset22-data/extracted-simple/
```

`annotations/` holds the 5.7 MB ShuttleSet22 annotation corpus. `sources/`
holds the 47 newly downloaded videos. `extracted-simple/` holds the 4.7 GB of
published outputs in 47 match directories. These are host-local scratch paths.

## Reuse boundary

`configs/shuttleset22/sources.toml` is the reviewed mapping of annotation IDs,
public source URLs, overlap records, and unavailable records. Use ShuttleSet22
annotations when training on these outputs. Do not combine the eight overlap
records as independent examples with ShuttleSet.

Raw broadcaster videos and extracted arrays are intentionally outside Git. The
annotations are from the MIT-licensed CoachAI Projects repository; the videos
remain subject to broadcaster rights.
