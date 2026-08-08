# CourtKeyNet fallback evaluation

## Result

The current court-corner chain is useful on standard fixed-camera broadcast
footage and deliberately fails closed on the amateur sample tested so far.

CourtKeyNet supplies frame-level corner candidates. A scene median uses only
unflagged detections. When two or three corners are confident, the classical-CV
fallback uses court-line geometry to recover the missing corners. An optional
video-level consensus pass repairs a measured advertising-board alias that the
per-scene geometry gate cannot recognise.

This record separates the measured result from the wider interpretation. It
does not claim that the fallback is a general amateur-court detector.

## Current path

`src/courtkeynet/wrapper.py` resizes frames with aspect-preserving padding and
returns corners in `TL, TR, BR, BL` order. The frame gate uses:

- minimum peak confidence `0.02`;
- maximum entropy `0.8`;
- quad area fraction from `0.01` to `0.95`;
- convexity and expected-corner-quadrant checks.

`src/courtkeynet/court_corners.py` keeps geometry-clean detections. Four
confident median corners use the model result. Two or three trigger the
classical-CV fallback. Zero or one returns `None`; from-zero court detection is
outside the current fallback's scope.

The fallback pools line evidence over a static-camera scene and fits the BWF
`6.10 m x 13.40 m` court model. It accepts only when line reprojection is at
most `0.010` of the frame diagonal and anchor reprojection is at most `0.020`.
It also needs at least two lines in each court-line family. A failed gate
returns `None`.

## Standard-broadcast measurements

The CourtKeyNet wrapper threshold was selected on 400 court-view frames from 10
ShuttleSet broadcasts and 63 hand-checked non-court frames. At peak floor
`0.02`, 88% of court frames passed, including 100% on seven videos. All 63
non-court frames were rejected. The recorded median error was 3.5 pixels in the
1280x720 reference space, with per-video medians from 2.2 to 5.5 pixels.

The narrower fallback ship check contains 46 recovered scenes from video 3 and
44 from video 21. It also records one fail-closed scene for each video. The
video-3 failure had only one confident corner. The video-21 failure narrowly
missed the line gate at `0.0103 > 0.0100`.

## Why cross-scene repair exists

The fallback can mistake an advertising-board edge for the far baseline when
both far corners are missing. The resulting quad can be internally consistent
and pass the per-scene reprojection gate while both far corners are badly
displaced.

`consensus_repair()` uses the fixed-camera assumption across one video's scene
quads. It takes the per-corner median, measures each scene by its worst corner's
distance from that median, and flags distances strictly above 55 pixels.

The recorded 1280x720 data separates clean and aliased scenes clearly:

| Recorded check | Result |
| --- | ---: |
| Video 3 good-scene maximum distance | 18.1 px |
| Video 3 aliased-scene minimum distance | 177.3 px |
| Video 21 control maximum distance | 8.2 px |
| Video 3 flagged scenes | 7 of 46 |
| Video 21 flagged scenes | 0 of 44 |

For video 3, whole-quad repair changed mean scene error from 23.1724 to 5.3698
reference pixels. P90 error changed from 107.2800 to 10.9483. Video 21 was
unchanged at 4.7757 mean and 5.0847 p90.

## Whole-quad policy

A flagged scene's complete quad is replaced with the consensus quad. Unflagged
quads remain byte-for-byte values from the input array.

The measured corners-only alternative produced almost the same aggregate result
on these scenes: 5.3686 mean reference-pixel error for video 3. The current code
still replaces the full quad because the boards alias is a homography-level
failure. Patching selected corners could combine two incompatible geometries.

The pass raises when half or more scenes are flagged, because the median no
longer represents a trustworthy majority. It also rejects empty or malformed
inputs. The caller must supply one fixed-camera video's quads in one shared
pixel space.

## Amateur-footage limit

The later amateur check covered 11 labelled frames across eight scenes in four
videos. The shipped chain produced no scoreable quad: `0/11` frames passed and
all eight scenes failed closed. Most scenes had at most one confident corner.
One scene had no geometry-clean frame and one failed the anchor gate at
`0.063 > 0.020`.

The raw ungated median errors were 849, 453, 457 and 376 pixels across the four
videos. Relaxing the gate would therefore have accepted very inaccurate
corners. Consensus could not help because no scene produced a quad to vote on.

This is a measured limit of the current chain. GitHub issue
[#24](https://github.com/ahalp90/badminton_cv_annotator/issues/24) tracks a
standalone amateur detector as optional future work.

## Colour-prior result

A colour study checked 30 video-3 frames and 11 amateur ground-truth frames.
Lab chroma did not separate the true far baseline from the advertising-board
edge. Their video-3 medians were only 7.8 `ab` units apart and every scene's
10-90% ranges overlapped.

Lightness separated that particular video-3 pair. A threshold of `L >= 200`
kept 93.5% of true-baseline samples and 1.7% of the competing edge samples. The
result did not generalise into a safe absolute line-colour rule: line paint and
venue colour varied substantially, and thin 4:2:0-compressed lines retained
only 17-49% of their chroma signal.

The current fallback therefore does not use a chroma or absolute white-line
prior.

## Recorded evaluation inputs

The `recorded_inputs/` directory contains the two compact fallback CSVs used by
the deterministic ship check:

- `fb5_3.csv.gz`
- `fb5_21.csv.gz`

Each row records one scene quad and its diagnostic measurements. The check also
reads the tracked ShuttleSet reference corners from
`data/shuttleset/set/homography.csv`. It does not open video, model weights,
credentials or environment files.

## Reproduce the check

From the repository root:

```bash
~/.venvs/badminton-cicd/bin/python \
  docs/courtkeynet/fallback_evaluation/check_consensus_repair.py
```

The script calls the current `consensus_repair()` implementation. It asserts
the seven video-3 flags, zero video-21 flags, the recorded separation values
and the mean consensus error against ground truth.

## Limits

- The wrapper measurement covers professional ShuttleSet broadcasts, not broad
  venue or amateur diversity.
- The consensus result covers two fixed-camera videos at 1280x720. The 55-pixel
  threshold needs remeasurement before use at 4K or with moving cameras.
- Line curvature is diagnosed but not corrected. Recorded fallback scenes
  include material sagitta values.
- Consensus repair cannot recover a video where most scenes are wrong or where
  the frame-level chain produces no quads.
- The evaluation records localisation and fail-closed behaviour. They do not
  measure downstream rally or feature accuracy.

Research and measurements were checked against the current implementation on
8 August 2026.

