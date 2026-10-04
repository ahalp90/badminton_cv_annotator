# Sharing one court across frames

Two early experiments tested whether several frames from the same camera view could share one court calibration. The first fitted one court across three short clips. The second collected 13 cached frames to inspect how stable the available line and court measurements looked over longer spans.

The three-frame fit improved two clips relative to floor scoring alone.
The longer-span cache left camera continuity unmeasured.

The current detector uses separate multi-frame composition and court-sharing code. Its behaviour is documented in the [court-detector design](../../../docs/court_detector/design.md).

## Three-frame trial

The earlier player-guided trial used one court across three frames from each of three clips. Adding net measurements and then refitting from pooled line fragments changed the worst corner error as follows:

| Clip | Median floor scoring | Floor plus net | Pooled-fragment refit |
| --- | ---: | ---: | ---: |
| Yellow | 221.19 | 26.01 | 21.22 |
| Letterboxed | 10.06 | 17.38 | 14.75 |
| Centre | 4.04 | 4.04 | 3.20 |

Yellow, Letterboxed and Centre are the three saved clip names. Floor scoring
measures agreement with court markings; the second method adds projected-net
evidence, and the third refits the court using line fragments from all three
frames.

Values are the worst corner error in pixels across the three references at 1280 × 720. The historical 15-pixel cutoff belongs to this experiment only.

All three columns already use one court across the three frames. The table compares changes to scoring and geometry, not shared calibration against independent single-frame fitting. The Yellow clip improved sharply but still remained above the experiment's 15-pixel cutoff; Letterboxed remained slightly worse than floor scoring alone; Centre improved slightly.

The [earlier-approaches summary](../../court_detector/comparisons/earlier_approaches.md) covers the broader investigation. The committed [replay archive](../independent_court/recorded/player_guided/replay.zip) contains the observations, proposals, references, code and saved results; the [summary](../independent_court/recorded/player_guided/summary.json.gz) contains the headline measurements.

Inside the archive, `joint_short.py` combines retained per-frame proposals, removes median-image proposals, suppresses near-duplicates, scores one selected player pair across frames and pools line fragments for the final refit. `joint_short/results.json.gz` contains the per-frame scores, candidate corners and reference errors.

## Thirteen cached frames

The later cache contains these frames:

| Video group | Cached frame indices | Span |
| --- | --- | --- |
| GX | 0, 5, 689, 5111, 5766, 77876, 86088 | a close pair, another frame about 11.5 s later, then sparse samples across roughly 24 min |
| Amateur-2 | 150, 28019 | two distant samples |
| Amateur-3 | 0, 10514, 17174, 24515 | four samples from one video |

The [frame gallery](../independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_view.html) and [cached records](../independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_records.json.gz) preserve frame indices, dimensions, line segments, direction hypotheses, merged-line masks, transforms, settings and source hashes.

GX uses the cached source rate of 59.885 frames/second, so its quoted times are frame index divided by that rate rather than presentation timestamps. The Amateur-2 and Amateur-3 records do not contain frame rates.

Some rows also carry older candidate corners and scores. Those are copied historical outputs, not fresh results from the later matcher. Modern line/paint winners exist for GX frames 0 and 5, both Amateur-2 samples and Amateur-3 frame 0 in the [ranking records](../independent_court/recorded/player_guided/projective_patterns/evaluation/ranking_records.json.gz).

## Camera continuity

No registration or motion analysis was run between the cached frames. Without those measurements, the 13-frame set cannot establish that a camera stayed fixed across any of the long gaps or that one projection remained valid for an entire group.

Three separate questions were mixed together in the early work:

- whether line measurements from several frames improve a fitted court;
- whether a court can be scored more reliably across several frames;
- whether an accepted court can be reused while the camera view remains unchanged.

The three-frame trial changed more than one of these at once. The 13-frame cache added longer time spans, but not the continuous camera measurements needed to answer the reuse question on its own.

The later detector work separated these jobs into scene composition, view matching and court sharing. See [`docs/court_detector/`](../../../docs/court_detector/README.md) and [`experiments/court_detector/`](../../court_detector/README.md).
