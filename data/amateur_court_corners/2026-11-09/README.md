# Court annotations — 8–9 November 2026

These manual court labels cover 21 full amateur videos: the 20 videos in the
[amateur annotation set](../../amateur_videos/README.md) plus the excluded
doubles match `C6NrJyBwn6c`. Every frame number indexes the full source video,
so these labels need no clip offsets or coordinate scaling. The annotation set
used exactly these courts for its manual-court outputs.

The labels were made with the [court annotation tool](../../../scripts/court_annotation/README.md)
(`annotate_amateur.sh`). There are 44 labelled frames: 63 corners were clicked
and 113 were extrapolated from a fit to the clicked court-line intersections.
Fit error ranged from 0.4 to 3.8 px.

- [hand_corners.csv.gz](hand_corners.csv.gz): four corner rows per labelled
  frame, in the same columns as the parent dataset. The `video` column gives the
  source filename; its stem is the YouTube ID.
- [hand_corners_landmarks.csv.gz](hand_corners_landmarks.csv.gz): the clicked
  intersections, with court coordinates in metres.
- [camera_reviews/](camera_reviews/): one review file per video. Each lists the
  candidate camera changes found by an automatic scan and the decision for each:
  `annotated`, `unchanged`, `changed` or `pending` (not reviewed). Some also
  record manually marked play periods or a note.

A label applies from its frame until the next label for that video. Frames
before a video's first label have no court. Normalised coordinates are
unclamped: values outside 0–1 mark corners that lie outside the picture.

`BkjErIAsZu4` and `C1jR4vZmrkI` were labelled on the original MKV files. The
annotation run used MP4 copies remuxed without re-encoding, so the frame
numbers are unchanged.
