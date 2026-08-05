# `sset_01` broadcast timeline labelling record

## Result

Curtis completed a full first pass on 4 August 2026 and a visual checking pass
on 5 August 2026. The canonical human timeline is
[`data/sset_01_broadcast_timeline_labels.csv.gz`](data/sset_01_broadcast_timeline_labels.csv.gz).
It contains 491 zero-based, half-open intervals covering every source frame in
`[0, 154393)`.

## Source pin

- Stable ID: `sset_01`, ShuttleSet video ID 1
- Match: Kento Momota vs Chou Tien Chen, Fuzhou Open 2019 Finals
- Source mapping: `https://www.youtube.com/watch?v=O669aZhH0LI`
- Review copy: 512x288 H.264 MP4 without audio
- Verified rate and extent: 25 FPS and 154,393 decoded frames
- Derived duration: 6,175.72 seconds
- Review copy location: local laptop scratch only; no video is committed
- Encoded-file hash: not recorded because the review copy is unavailable on
  the server. The stable ShuttleSet ID, source URL, dimensions, rate, and frame
  count are the source-identity pin. Exact reviewed frame-stream
  reproducibility is unavailable without a digest of the laptop copy.

## Labelling workflow

The five classes are `live`, `live-non-standard`, `replay`, `cutaway`, and
`other`. Curtis reviewed the full video with
`annotator.manual_broadcast_timeline_annotator`. Normal court footage remained `live` between
rallies. Unusual views became `live-non-standard` only when they showed actual
live action or warm-up activity. Graphics and broadcast stings were `other`.

Gemini 3.1 Pro supplied coarse review proposals through the web dashboard. The
video was sent as eleven nominal 600-second shards with two-second overlaps.
The request was made at about 17:27 Australia/Sydney on 4 August 2026. Raw
responses are retained outside the shared repository.

The model did not follow the requested timestamp contract. The ten later
responses returned shard-relative timestamps, the first nine described
footage beyond their uploads, and six of ten overlapping joins disagreed. The
review guide therefore added each shard origin, discarded out-of-range
content, gave the later shard ownership of each overlap, and treated every
proposal as advisory.

## Human review record

The first-pass upload contained 532 intervals and covered the complete source.
The checking pass changed two frame ranges:

- `[23820, 24351)` from `cutaway` to `live`
- `[127226, 127770)` from `live` to `cutaway`

It also added one note to `[10496, 10924)`:
`practice/warm up, no legitimate play.`

The first pass was uploaded at 23:48 AEST on 4 August. The reviewed file was
uploaded at 12:39 AEST on 5 August. Active review time was not recorded, so
the elapsed human-review metric is unavailable. Upload timestamps are not a
valid substitute because they include breaks and off-task time.

The GUI did not record accepted, moved, split, merged, and rejected proposal
actions. Those disposition counts cannot be reconstructed reliably from the
final partition. A post-hoc comparison is retained only as a workflow signal:

- Gemini and the final labels agree on 70,409 of 154,375 proposal-covered
  frames, or 45.61%.
- No proposal interval exactly matches a final interval in both bounds and
  class.
- 63 of 231 proposal boundaries fall within 25 frames of a final boundary.
- 79 of 490 final boundaries fall within 25 frames of a proposal boundary.

These figures do not measure model accuracy independently. The human labels
were created while the proposals were visible, and the comparison excludes
the final 18 frames that had no proposal.

## Canonicalization and validation

The reviewed plain CSV has SHA-256
`d7a5a60d545695c7c454237ccfccc67a6ed489d415eabdc833f6269962d0832e`.
Forty-one adjacent boundaries carried identical class and note information.
They were removed without changing any frame label, leaving 491 canonical
intervals. The deterministic gzip has SHA-256
`b65082468aa1635d177028b46367ebc643013892854aa45798b8b96062532bad`.

The label-contract reader accepted the reviewed input and the gzip round trip.
Both cover `[0, 154393)` without gaps or overlaps. No reviewed interval is
shorter than five frames. All 113 tracked ShuttleSet rally extents were checked;
none overlaps a final `replay`, `cutaway`, or `other` interval.

| Class | Intervals | Frames | Seconds | Source share |
| --- | ---: | ---: | ---: | ---: |
| `live` | 122 | 48,497 | 1,939.88 | 31.41% |
| `live-non-standard` | 15 | 1,362 | 54.48 | 0.88% |
| `replay` | 62 | 16,713 | 668.52 | 10.82% |
| `cutaway` | 167 | 75,665 | 3,026.60 | 49.01% |
| `other` | 125 | 12,156 | 486.24 | 7.87% |

## Limits

- This is one human-labelled broadcast and has no independent second-labeler
  agreement measurement.
- VLM proposal-disposition counts and active review time were not captured by
  the first GUI version.
- The source review-copy hash is unavailable. Stable source identity and exact
  decoded video metadata are recorded instead.
- The labels are recording-only evidence for Issue 29. They do not change
  replay, segmentation, serve-selection, or production thresholds.
