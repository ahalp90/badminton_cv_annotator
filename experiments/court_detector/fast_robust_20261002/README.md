# Fast-robust court evaluation: ShuttleSet and ShuttleSet22

The detector usually finds a close court fit, but sharing a court across scenes
sometimes replaces accurate fits with a badly wrong court. **74 of 86 videos
have a representative court within 10 pixels of the supplied homography. Twelve
exceed 20 pixels.** The saved individual fits show that most of these errors
appear during sharing. Fix that selection step before using the full extracts
for player positions or distances in court metres.

This evaluation covers the completed Carmack `fast-robust` extracts at detector
commit `17a50b57`. These are descriptive results on the project's release corpus,
not a test on unseen venues. The main results below use the original predictions;
the later sharing-fix pilot is reported separately.

## What was measured

The corpus contains **40 ShuttleSet and 46 ShuttleSet22 videos**, with 6,833
usable labelled rallies. A rally interval runs from its first labelled contact
through its last contact. It excludes the unlabelled flight after the last shot.

Each video contributes one representative prediction. A view group contains
scenes the detector considers to show the same camera view. The group
with the greatest total overlap with rally intervals supplies that prediction.
Within the group, choose the actual scene court closest to the other courts,
weighted by rally overlap. Its sampled frame must lie within a rally when such a
frame is available; all 86 selected frames do. Official court geometry has no
part in this choice. The largest group covers a median 94.7% of labelled rally
time per video, although the minimum is 33.9%.

**Corner error** is the mean Euclidean distance over the four outer doubles-court
corners. Predictions are scaled to the annotations' **1280 × 720 pixels**. The
reference court comes from inverting the supplied homography at the dataset's
court-template corners. Cyclic corner rotations are allowed; mirrored order is
not. The 5, 10 and 20 pixel thresholds describe agreement, not validated limits
for downstream use.

## Results

| Measure | ShuttleSet | ShuttleSet22 | Combined |
| --- | ---: | ---: | ---: |
| Videos | 40 | 46 | 86 |
| Usable rallies | 3,182 | 3,651 | 6,833 |
| Median representative corner error | 2.82 px | 2.37 px | 2.73 px |
| 90th percentile representative error | 9.42 px | 136.58 px | 91.15 px |
| Representative error ≤5 px | 34/40 (85.0%) | 38/46 (82.6%) | 72/86 (83.7%) |
| Representative error ≤10 px | 36/40 (90.0%) | 38/46 (82.6%) | 74/86 (86.0%) |
| Mean per-video rally-time detection coverage | 97.3% | 96.4% | 96.8% |
| Rallies whose longest-overlap scene has a court | 3,011/3,182 | 3,451/3,651 | 6,462/6,833 (94.6%) |
| Rallies whose longest-overlap court is within 10 px | 2,631/3,182 | 2,597/3,651 | 5,228/6,833 (76.5%) |

Coverage counts the presence of a prediction, including wrong predictions.
A missing court counts against both rally fractions. The rally comparisons use
the scene with the greatest overlap even when another, shorter scene has a court.
The 10 px agreement rate across rallies is lower than the per-video rate because
it includes other views and local failures. A static homography cannot label
camera changes correctly, so that rate is agreement with the reference rather
than verified frame-by-frame accuracy.

The mean representative error is **45.26 px**, with a 95% bootstrap interval of
**17.56–79.80 px**. A few large failures raise it far above the median. Mean
per-video detection coverage is **96.81%**, with interval **96.04–97.48%**.
Intervals use 2,000 resamples of whole videos, seed 20261002. Videos from the same
venue can share failure modes; these intervals do not establish uncertainty on
new venues. The combined median court intersection-over-union is **0.985**.

## Where sharing goes wrong

The twelve high-error representative courts are listed in the
[gallery](gallery.md). Their errors range from 32.9 to 973.6 px. Visual checks of
selected main-view frames confirm that several fit the wrong court markings,
produce excessive width, or put baselines outside the visible court.

A comparison of saved fits isolates a large regression. Among **5,977 detected
scenes that overlap usable rallies in the dominant view groups**, every corner
was within 20 px in **5,945 before sharing**, versus **4,924 afterwards**.
Sharing moved **1,024** scenes outside that tolerance and moved **3** inside.
Here the threshold applies to the worst corner; the main results use mean error.
This uses retained predictions, not a new extraction or an ablation. It measures
the selected groups only. Counts are in [pooling_diagnostic.csv.gz](pooling_diagnostic.csv.gz).

The code and saved group records explain the failure in the five groups checked
closely: ShuttleSet 26, 30 and 36, and ShuttleSet22 43 and 44.

- A candidate individual court is discarded if any group member rejects it.
  In each checked group, the accurate candidates fail a player-position check.
- An oversized wrong candidate survives. It is compared with the pooled fit,
  which is also poor.
- The surviving candidate replaces the group's scene courts. The accurate
  candidates have already been removed from the comparison.

For example, all 119 accurate candidates in ShuttleSet 30's dominant group are
rejected at the same member scene. The survivor's mean score is 0.479; the
members' own courts average 0.844. It nevertheless replaces those courts.
These scores explain the selection; they are not accuracy measurements.
See [pool_group and scene_candidate](../../../src/court_detector/view_pool.py).

ShuttleSet 30's veto comes from a between-points scene, frames 26,011–26,145.
One player stands on court while the other prepares at the sideline. Replaying
the player check on its 31 samples gives one player in the correct court in
31/31 samples, but players in both halves in 0/31. The check requires both halves
in at least half the samples, so it rejects the correct court. The oversized
court passes in 31/31 by including people outside the real court, including
line judges mistaken for standing players. A valid
between-points view therefore vetoes geometry supported by the rally scenes.
The [scene clip](veto_scene.mp4) shows the full 4.5 seconds without overlays.

The fix in `84bbba1e` chooses one best shared court, then handles failed transfers
per scene. Player absence must not disqualify a court for the group. A scene
whose transfer fails geometry or camera checks keeps its own fit. The retained
`scene_corners_native_px` allow sharing to be rerun without repeating the
expensive scene searches. A pilot on ShuttleSet 30's 123-member main-view group
reproduced every saved source score exactly. After sharing, all 123 courts were
within 10 px at every corner, versus none within 20 px before the fix. This
includes the 101 scenes overlapping labelled rallies and scene 0094.
The pilot took 16.9 minutes; it does not establish the result for other videos.
The [pilot measurements](view_checks/sharing_pilot_summary.json.gz) and
[per-scene comparison](view_checks/sharing_pilot_comparison.csv.gz) retain the results.
Keeping individual fits alone is not sufficient: scene 0094's original fit is
already oversized, as its [overlay](view_checks/sset30_scene0094.png) shows.

## View grouping follow-up

Keep the hash threshold at 0.30 for now. In a diagnostic sample of 988 frames
across all 86 videos, tightening it to 0.25 would exclude 84 of 340 existing
member-to-reference matches. None of the sampled pairs with low whole-image
similarity passed 0.30. Similarity is only a proxy for camera view, so this does
not prove every group is correct. The [follow-up](view_checks/README.md) records
the measurements, limitations and exact player-check replay for ShuttleSet 30.

Fast-robust searches each scene's middle frame. Groups with fewer than three
contributing scenes retain those individual fits; they do not receive a pooled
fit from multiple frames within the scene. The multi-frame player check still
runs.

## Exclusions and reference limits

The cohort matches both release manifests. ShuttleSet 09, 10, 12 and 27 and
ShuttleSet22 15 are excluded. ShuttleSet22 14, 45 and 56 have no resolved source;
eight overlapping broadcasts are counted under ShuttleSet only.

Of 7,210 labelled rallies, 177 ShuttleSet and 200 ShuttleSet22 rallies have
non-increasing contact frames after ordering by shot number. They are excluded
using the release loader's timing rule. There are no nonfinite or out-of-range
contact frames in this cohort. A flaw flag alone does not exclude an otherwise
usable rally interval.

ShuttleSet22 reverses the bottom-corner column names in rows 1–7 and 34; only 34
is in this cohort. Using the homography avoids this ordering issue. Its explicit
corner columns also differ from its homography by up to **3.4 px at one corner**.
The original annotations do not establish whether their points follow paint
centres or outer edges. Small pixel differences should be read with that limit.

## Files and reproduction

- [Gallery](gallery.md): one native-resolution court PNG for each of 86 videos.
- [Eight hard samples](hard_samples/README.md): four per dataset, selected by
  large reference disagreement from different videos. Selection requires a
  sampled frame inside a usable rally and a scene that contains most of at least
  one rally. These are diagnostic examples, not a random sample.
- [Per-video table](per_video.csv.gz), [per-rally table](per_rally.csv.gz),
  [per-scene table](per_scene.csv.gz), and [summary](summary.json.gz).
- [Frame requests](render_requests.json.gz) and [captions](render_captions.csv.gz)
  retain source paths, frame numbers and the actual predicted corners.

The PNGs show one-pixel dashed red borders around the predicted white stripes,
including internal markings. They preserve the saved predictions, including
wrong ones. Compression uses `pngquant --speed 3 --nofs 256`, then `oxipng -o 2`.

The [evaluation script](../../../scripts/evaluate_courts_fast_robust.py) provides
`analyse`, `fetch-frames` and `render`. See [reproduce.md](reproduce.md) for the
input locations and commands. Analysis, full-image rendering, targeted lint and
whole-project type checking completed successfully for the evaluation script.
The measurements and galleries describe the original extracts; they do not
measure the subsequent sharing fix.
