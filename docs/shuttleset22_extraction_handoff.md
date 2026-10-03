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

The official annotations are checked in at `data/shuttleset22/set/`. The
142 compressed CSV files occupy 1.6 MB and preserve the original 5.7 MB of CSV
bytes.

Source videos and extraction arrays are supplied separately. Under the external
dataset root, `sources/` holds the 47 downloaded videos and `extracted-simple/`
holds 4.7 GB of published outputs in 47 match directories. Machine-specific
storage locations belong in the project's local remote-access notes.

## Reuse boundary

`configs/shuttleset22/sources.toml` is the reviewed mapping of annotation IDs,
public source URLs, overlap records, and unavailable records. Use ShuttleSet22
annotations when training on these outputs. Do not combine the eight overlap
records as independent examples with ShuttleSet.

Raw broadcaster videos and extracted arrays are intentionally outside Git. The
annotations are from the MIT-licensed CoachAI Projects repository; the videos
remain subject to broadcaster rights.
