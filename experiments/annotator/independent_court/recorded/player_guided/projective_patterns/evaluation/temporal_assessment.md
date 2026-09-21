# Assess automatic calibration once per stable camera view

The practical question is whether one automatically chosen court per stable camera view
could provide a useful CourtKeyNet-free detector sooner than reliable single-frame detection.
This packet supplies existing observations for that assessment. It does not establish
camera stability, demonstrate a temporal improvement, or change the pending experiments.

## Available evidence

[Frame gallery](temporal_view.html) · [cached records](temporal_records.json.gz)

| Video group | Cached frame indices | What the sample offers |
| --- | --- | --- |
| GX | 0, 5, 689, 5111, 5766, 77876, 86088 | A close pair, a frame about 11.5 seconds later, and sparse later samples across roughly 24 minutes |
| Amateur-2 | 150, 28019 | Two distant samples with contrasting modern automatic results; too sparse to establish temporal continuity |
| Amateur-3 | 0, 10514, 17174, 24515 | Four samples from a video whose first detailed modern result was usable |

All 13 records preserve original frame indices, native dimensions, cached native-pixel
line segments and the original selected direction hypotheses in working coordinates.
They include masks over merged lines, the coordinate transform, saved settings and MD5s.
The transform and direction semantics match [the evaluation guide](README.md).
Actual fragment-to-merged-line assignments remain unavailable.

GX timing uses the cached source rate of 59.885 frames/second. Seconds are nominal
frame-index/rate values, not presentation timestamps. The selected Amateur-2/3 input
pack does not provide frame rates, so their seconds are left unknown. Do not infer them.

Earlier candidate corners, original scores/ranks and source variants are copied where
present in the amateur input pack. They are not outputs of the modern matcher. An empty
candidate list means that pack provides none, not that a new search failed. Modern
line/paint winners are already in [ranking records](ranking_records.json.gz) for GX0,
GX5, both Amateur-2 samples and Amateur-3 frame0. They were not generated for every
frame here. Keep pipeline and population identity in comparisons.

Existing annotations are included as `saved_reference` for diagnosis only. Similar
annotations do not verify that the camera was fixed. The gallery reuses committed
images: GX0/5 are native frames; later GX images contain historical overlays. Amateur
images are published frame renderings. Labels and overlays are not stability evidence.
Each record identifies its image kind and hash. No image was added or modified.

These observations permit inspection of apparent camera changes, visible line differences
and persistent background structures at the sampled instants. They cannot rule out cuts,
pans, zooms or camera motion between those instants. No registration or motion detector
was run. No group in this export has a verified shared projection.

## An earlier shared-court experiment already exists

The [earlier player-guided report](../../../../../../../scratch/court_det_fix/evidence/independent_proposals/README.md) describes
one court chosen across three frames for each of three short clips. Its committed
[replay archive](../../replay.zip) is about 281 KB and already contains the observations,
proposals, references, code and complete saved results. Do not duplicate or rerun it for
this assessment. The [summary](../../summary.json.gz) preserves these headline values:

| Clip | Median floor scoring | Floor plus net | Pooled-fragment refinement then rescoring |
| --- | ---: | ---: | ---: |
| Yellow | 221.19 | 26.01 | 21.22 |
| Letterboxed | 10.06 | 17.38 | 14.75 |
| Centre | 4.04 | 4.04 | 3.20 |

Values are worst corner error across the three references at 1280 × 720. The historical
15-pixel cutoff is not a current usability criterion. All three stages already choose
one court across the frames; this table does **not** isolate aggregation versus independent
single-frame selection. The net stage adds a cue and the final stage also changes geometry.

For exact accounting, inspect `joint_short.py` inside the archive. It unions retained
per-frame proposals, excludes median-image proposals and suppresses near-duplicates.
It uses a selected persistent player pair, takes median floor/net scores across frames,
and pools fragments for the refinement stage. `joint_short/results.json.gz` retains
per-frame floor/net scores, candidate corners and reference metrics for each stage.
`fits_short_all/results.json.gz` preserves earlier proposals; `lines_short/deeplsd_md_default.json.gz`
preserves cached fragments. The `people_short` records identify the anchors and dimensions.

This is evidence that a shared-court workflow has been implemented and produced mixed
results with an older pipeline. It is not a controlled temporal ablation of the newer
spacing matcher, a whole-video stability test, or an automatic acceptance rule.

## The assessment to make

Evaluate whether the observations justify prioritising calibration per stable view.
Distinguish (1) pooling line evidence to generate a shared court, (2) scoring existing
candidates across frames, and (3) reusing a selected court while the camera stays fixed.
Separate what can be inferred from these cached samples from what remains a hypothesis.

Camera-view grouping must be automatic at deployment. Analyse circular reasoning if
court agreement itself decides which frames share a view. Persistent background lines
can agree across frames too. Frame0/5 are only about 0.084 seconds apart and offer little
temporal separation. Distant samples offer variety but weak continuity evidence.
A whole-video maximum score is not the same as sustained support for a shared projection.

The pending B/M/R/MR direction experiments remain fixed. No new annotation is required
for this assessment. Request at most one bounded follow-up comparison if a consequential
question cannot be answered from the cached evidence. Specify what new observation that
comparison would provide, rather than treating more frames alone as proof of improvement.
