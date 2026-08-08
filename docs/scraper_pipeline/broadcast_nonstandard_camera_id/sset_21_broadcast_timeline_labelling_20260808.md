# `sset_21` broadcast timeline labelling record

## Result

Curtis completed a full first pass and human checking pass on 8 August 2026.
The canonical human timeline is
[`data/sset_21_broadcast_timeline_labels.csv.gz`](data/sset_21_broadcast_timeline_labels.csv.gz).
It contains 258 zero-based, half-open intervals covering every source frame in
`[0, 100349)`.

This timeline was reviewed separately from `sset_15`. Both timelines are
prepared in the same repository worktree for review.

## Source pin

- Stable ID: `sset_21`, ShuttleSet video ID 21
- Match: An Se Young vs Ratchanok Intanon, YONEX Thailand Open 2021 Quarter-finals
- Source mapping: `https://www.youtube.com/watch?v=gloiZ_gTJaE`
- Downloaded source: 1920x1080 MP4 at 30 FPS with 100,349 decoded frames
- Review copy: 512x288 MP4 at 30 FPS with the same 100,349 decoded frames
- Derived duration: 3,344.97 seconds
- Reported review-copy MD5: `a07863d2acae6353ef158cf3576a1a9d`
- Review copy location: Curtis's laptop scratch directory; no video is committed

The review-copy digest pins the encode used for frame numbering. The source URL
and ShuttleSet metadata identify the match independently.

## Labelling workflow

The five classes are `live`, `live-non-standard`, `replay`, `cutaway`, and
`other`. Curtis reviewed the complete video with
`annotator.manual_broadcast_timeline_annotator`.

The tool loaded 454 PySceneDetect intervals generated from the exact review
copy. The raw scene CSV has SHA-256
`a6611f34eadb8f0ef3db2087fbebd0570c6f4f1611bd8dc84a8219de3edcce6d`.
Its 453 internal boundaries exactly match the uploaded cut-frame file. These
files remain outside the tracked repository because they are navigation
scaffolding rather than human truth. No proposal CSV was used.

## Human review record

The first-pass and final-review handoffs each contain 454 intervals with
identical scene bounds. The checking pass changed 24 class rows covering 7,127
frames and made two note-only changes. Both handoffs were uploaded at 22:03
AEST on 8 August 2026. The upload time does not measure active review time.

The GUI could not split eight mixed scenes during review. Curtis recorded these
explicit class transitions separately:

- `live` starts at frame 25,475 within `[25132, 25762)`.
- `live` starts at frame 41,816 within `[41416, 42423)`.
- `live` starts at frame 45,218 within `[44950, 45780)`.
- `live` starts at frame 61,218 within `[61204, 61515)`.
- `live` starts at frame 64,590 within `[64321, 64888)`.
- `live-non-standard` starts at frame 70,917 within `[70662, 70986)`.
- `live` starts at frame 74,892 within `[74623, 75079)`.
- `live` starts at frame 76,399 within `[76199, 76637)`.

Two targeted visual checks supplied the remaining corrections:

- `[27103, 27465)` is `live-non-standard` because the live player and racket
  are mostly outside the frame.
- The final checked sequence is `[91200, 91482)` `cutaway`, `[91482, 91534)`
  `live-non-standard`, `[91534, 91722)` `live`, `[91722, 92120)` `cutaway`,
  and `[92120, 92602)` `live`.

The seven notes embedded in the final-review CSV were attached to incorrect
scene rows while the split operation was failing. They were cleared from the
canonical timeline. The separate boundary record above is the source for the
repairs.

## Canonicalization and validation

The final-review handoff has SHA-256
`bc4de2db840ed9bd1f34f1ea2cd4fe2235b93dc3de5b95207d423815d3361820`.
Applying the reviewed split decisions changes 3,219 frame labels and yields 465
segments before canonicalization. Removing adjacent boundaries with identical
class and note information leaves 258 canonical intervals. The deterministic
gzip has SHA-256
`06812dbd11f60540920b435bf37db08327d8aac042960749a17fc05a74a9a2c7`.

The repository label-contract reader accepted the final gzip round trip. It
covers `[0, 100349)` without gaps or overlaps, and its shortest interval is 14
frames. All 663 ShuttleSet stroke frames across 75 rallies were checked. No
stroke or rally extent overlaps a final `replay`, `cutaway`, or `other`
interval.

| Class | Intervals | Frames | Seconds | Source share |
| --- | ---: | ---: | ---: | ---: |
| `live` | 77 | 32,648 | 1,088.27 | 32.53% |
| `live-non-standard` | 32 | 4,219 | 140.63 | 4.20% |
| `replay` | 45 | 19,671 | 655.70 | 19.60% |
| `cutaway` | 91 | 38,681 | 1,289.37 | 38.55% |
| `other` | 13 | 5,130 | 171.00 | 5.11% |

## Limits

- This is one human-labelled broadcast with no independent second-labeler
  agreement measurement.
- Curtis described the `live-non-standard` transition as approximately frame
  91,482. The canonical timeline uses 91,482 as the selected frame boundary.
- Active review time and per-scene decision events were not recorded.
- The source video is not committed. Reproduction requires a review copy that
  matches the recorded MD5, FPS, dimensions, and frame count.
- These labels add recording-only ground truth. They do not change replay,
  segmentation, serve-selection, or production thresholds.
