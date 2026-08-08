# `sset_15` broadcast timeline labelling record

## Result

Curtis completed the annotation and multiple visual checking passes on 8 August
2026. The canonical human timeline is
[`data/sset_15_broadcast_timeline_labels.csv.gz`](data/sset_15_broadcast_timeline_labels.csv.gz).
It contains 403 zero-based, half-open intervals covering every source frame in
`[0, 149487)`.

This timeline was prepared independently for
[Issue 71](https://github.com/ahalp90/badminton_cv_annotator/issues/71). It does
not depend on the separate `sset_21` review.

## Source pin

- Stable ID: `sset_15`, ShuttleSet video ID 15
- Match: Anthony Sinisuka Ginting vs Anders Antonsen, Indonesia Masters 2020 Final
- Source mapping: `https://www.youtube.com/watch?v=yu9oyMXRGHY`
- Downloaded source: 1920x1080 MP4 at 25 FPS with 149,487 decoded frames
- Review copy: 512x288 MP4 at 25 FPS with the same 149,487 decoded frames
- Derived duration: 5,979.48 seconds
- Review-copy MD5: `39c693db594e850399e3a8cae34ffdde`
- Review copy location: Curtis's laptop scratch directory; no video is committed

The review-copy digest pins the exact encode used for frame numbering. The
source URL and ShuttleSet metadata identify the match independently.

## Labelling workflow

The five classes are `live`, `live-non-standard`, `replay`, `cutaway`, and
`other`. Curtis reviewed the complete video with
`annotator.manual_broadcast_timeline_annotator`.

The tool loaded 751 PySceneDetect intervals generated from the exact review
copy. The detector used a content threshold of `27.0` and the shared 25 FPS
minimum-scene-length configuration. The raw scene CSV has SHA-256
`02c45b0fe74f5b9aa0998ab8d3355ca48b4b169fd4e70a05d98487a2444beee2`.
It remains outside the tracked repository because it is annotation scaffolding,
not human truth.

Scene cuts supplied navigation and initial bounds only. Curtis assigned the
classes and visually checked the complete video. No VLM proposal file was used.

## Human review record

Four complete handoff files were retained outside the repository during review:

| Handoff | Time on 8 August 2026 AEST | Rows |
| --- | --- | ---: |
| First pass | 20:12 | 751 |
| Second pass | 20:40 | 751 |
| Reviewed | 21:00 | 751 |
| Final | 22:14 | 751 |

The second pass changed 16 scene rows covering 4,740 frames. This included 13
class changes and four note changes. The next checking pass changed 15 scene
rows covering 947 frames from `cutaway` to `live-non-standard`.

The final handoff applied two further human decisions:

- `[145678, 146328)` changed from `replay` to `other`. It is a doubles replay
  from another match, not repeated footage from this video.
- `[147580, 148007)` changed from `replay` to `cutaway`.

The final reviewed handoff has SHA-256
`0054db150c78cd0efdf4f29d7887bb599fda7666caa91cdac75fa548cb3aa8c3`.
Active review time was not recorded. The handoff timestamps include breaks and
are not a measure of annotation speed.

## Canonicalization and validation

The final handoff preserved all 751 scene boundaries. Of those boundaries, 348
separated adjacent rows with the same class and note. Removing them changed no
frame label and left 403 canonical intervals. The deterministic tracked gzip
has SHA-256
`fb68449e3ae0513af5368e3082f7b49d6ad6f6be95598dbe7230dc299c57c022`.

The repository label-contract reader accepted the final handoff and the
canonical gzip round trip. Both cover `[0, 149487)` without gaps or overlaps.
The 751-row handoff matches the raw scene bounds exactly, and its shortest
interval is 14 frames.

All 824 ShuttleSet stroke frames across 104 rallies were checked. No stroke or
rally extent overlaps a final `replay`, `cutaway`, or `other` interval.

| Class | Intervals | Frames | Seconds | Source share |
| --- | ---: | ---: | ---: | ---: |
| `live` | 104 | 29,319 | 1,172.76 | 19.61% |
| `live-non-standard` | 18 | 3,555 | 142.20 | 2.38% |
| `replay` | 72 | 16,304 | 652.16 | 10.91% |
| `cutaway` | 185 | 78,075 | 3,123.00 | 52.23% |
| `other` | 24 | 22,234 | 889.36 | 14.87% |

## Limits

- This is one human-labelled broadcast with no independent second-labeler
  agreement measurement.
- Active review time and per-scene decision events were not recorded.
- The source video is not committed. Reproduction requires a review copy that
  matches the recorded MD5, FPS, dimensions, and frame count.
- These labels add recording-only ground truth. They do not change replay,
  segmentation, serve-selection, or production thresholds.
