# Court detector evaluation

The detector runs end to end on real videos. Development checks support
three-frame composition and the optional video-robust mode, with the limits
below. There is no measured accuracy rate for the final detector on unseen
footage. The full GPU evaluation of video-robust mode on video 040 is
**pending**, so this record does not establish that it is ready to merge.

The [operator guide](../../src/court_detector/README.md) explains how to run it;
[design.md](design.md) explains the choices behind it.

## Four kinds of evidence

Keep these apart when reading any number below.

| Kind | What it measures | What it does not measure |
| --- | --- | --- |
| **Detector objective score** | The combined score, 0.9 × paint + 0.1 × geometry + 0.04 × net reward, from 0 to about 1.04 | Whether the court matches the real court |
| **Labelled accuracy** | Error against hand-marked court corners, in working pixels or floor metres | Anything outside the few labelled views |
| **Visual review** | A person's judgement of a drawn outline | A rate across footage; tiny offsets |
| **Controlled timing** | Paired runs on the same hardware and input | Accuracy; results on other hardware |

The combined score ranks courts inside the detector. Strong paint-like evidence
can belong to the wrong marking or to something outside the court. A good court
with worn or obscured paint can score lower.

## Evidence sources

| Source | Scope | Status |
| --- | --- | --- |
| Development views | 28 saved views: 20 with courts, 8 controls; earlier 27 and 47-view sets | Complete; used to build the method |
| Controlled timing | One five-minute interval, L40 GPU, at most eight CPU cores | Complete; single runs before composition |
| Broad output run | 97 videos, 28 September, commit `5b98b617` | Complete; before composition and pooling |
| Scene checks | 24 videos, 533 scenes | Complete |
| Composition checks | Eight selected scenes; first 218 of 405 scenes of video 040 | Complete; visual review plus consistency |
| Cached pooling checks | Video 003 (two scenes) and video 040 (11 scenes) | Complete |
| Full video-robust GPU run of video 040 | Whole 69-minute match | **Pending** |

Representative image backgrounds come from the match videos named by project
video ID and scene number in each caption. The underlying broadcast footage
remains under its source rights; see [data attribution](../../data/ATTRIBUTION.md).

## Development views

These views were used to develop the method. Their counts show how choices
compared; they do not estimate accuracy on new footage.

- Of 27 development cases, 21 choices were judged visually usable. Applying
  the existing player-position check retrospectively to saved rankings raised
  this to 24. These were qualitative whole-frame judgements.
- On the net-reward trial, median per-frame errors across 45 labelled frames
  were 3.40, 2.75, 2.75 and 2.76 working pixels at net weights 0, 0.02, 0.04 and
  0.08. The comparison did not establish a practically meaningful advantage
  of 0.04 over 0.02. It used 45 referenced frames from 71 saved candidate pools,
  including a replacement pool with extra candidates for one amateur frame.
  It did not rerun the later complete detector.
- The 10% line-support share cut one view's largest floor error after refit
  from 0.94 m to 0.32 m. Another view still slipped at 1.06 m.
- The versions tested accepted one or two of eight non-court control views.
- The joined detector reproduced the research chain's chosen courts exactly
  on all 28 views, under the older settings, on 25 September.

"Working pixels" are pixels in the detector's shrunken image, with its longest
side at most 960 pixels. They differ from the source video's native pixels.

## Controlled timing

Single runs on 28 September processed one five-minute interval on an NVIDIA L40
with at most eight CPU cores. The input was Momota / Chou, Fuzhou Open 2019
final, source frames `[11273, 18773)` at 25 fps. Both used live GPU neural
inference and chronological court reuse. Compiler caches were already warm;
model loading and worker startup/shutdown were included. Earlier frame-seek
validation was excluded.

| Timing | GPU templates | CPU templates |
| --- | ---: | ---: |
| Court detector, including DeepLSD line detection | 173.226 s | 219.878 s |
| People detection setup and calls | 35.006 s | 36.554 s |
| Scene detection | 19.163 s | 18.453 s |
| Whole run | 227.396 s | 274.885 s |

All 33 scene outcomes and corner arrays matched between the two runs. The
target of about 30 s per five minutes (90 s upper end) is not met. These runs
predate three-frame composition, which adds two endpoint searches per fresh
scene.

The per-scene `stage_seconds` values cover detector stages. Video-robust
grouping and pooling run after those stages. The video-level `processing_seconds`
includes both scene processing and pooling, so use it to account for the full
video-mode cost.

The broad run and the cached checks below are **not** controlled timing
comparisons. Their times include different inputs, sharing and settings.

## Broad output run, 28 September

The detector ran over 97 videos with live GPU line, pose and template work.
It used commit `5b98b617`, before composition and pooling.

- 93 of 97 videos completed. Three short clips failed a decoded-frame coverage
  check. One video stopped when a refit returned no corners.
- The completed videos had 2,405 scenes: 769 courts, 823 no-court and 813 too
  short for the foot window.
- Successful per-video times sum to 14,194.74 s. The three batch times sum to
  14,237.71 s, including overhead and time spent on failed videos.

These counts describe outputs. A "court" row is not a verified correct court.

## Scene checks, 24 videos

Of 76 accepted reuse pairs, an older perceptual-hash grouping put 74 in the same
group. This supports checked reuse agreeing with an independent view grouping.
It does not show each reused court is accurate. The same-shot thumbnail check
shortened 244 of 359 foot windows. That is disagreement with cut-only windows,
not proof of missed cuts.

## Composition: visual review and consistency

**Visual review supports composition.** In an eight-scene selected sample, the
project owner judged the video 005 composite perfect while all three
single-frame fits had serious faults. They also reviewed the composites
produced from the first 218 scenes of video 040. The reported exception was
scene 0152, where every tested method produced a false court.

**Consistency supports it too.** On nine scenes of one returning camera view in
video 040, composite corners scattered 0.47 native pixels around their median
after image alignment. The paint-selected single-frame courts scattered
1.38 pixels. These are root-mean-square distances at 1920 × 1080 resolution.
The nine scenes had complete first/middle/last observations. Scatter measures
agreement between scenes; a consistently misplaced court could also agree well.

**Score-only selection missed useful composites.** The video 040 prefix
produced 13 valid fits in the research comparison. Its selector kept an
original in all 13 cases. Eleven had complete scores for their common-frame
comparison, and none of those composites won. Two lacked an original's score;
the false court in scene 0152 actually beat both available originals. A valid
fit in this record does not establish that the production player checks passed.

For video 005, paint support fell from 0.507 to 0.461 while line support rose
from 0.537 to 0.566. The paint weight outweighed the line gain. This is why the
detector accepts a composite that passes its checks without requiring a score win.

| Single-frame court, video 005 | Three-frame composite, video 005 |
| --- | --- |
| ![Single-frame fit](assets/video005_13361_single_frame.png) | ![Composite fit](assets/video005_13361_composite.png) |

Both outlines use the same last-sampled image of video 005, scene anchored at
frame 13361. The left court comes from that single frame; the right combines
three observations. Outlines are 1 px red dashes; open the full images to inspect
them. The dashes distinguish the outline from the solid court markings.

![False court on a close-up](assets/video040_0152_composite.png)

Scene 0152 of video 040 is a close-up. Every tested method, composite included,
draws a false court across the advertising boards.

## Cached pooling checks

These checks used the production composition and pooling functions on saved
images, lines and people boxes. They were generated during development of the
whole-scene fallback, shortly before commit `6fa3bf44`; the files do not embed
an exact source revision. They did not rerun line detection, pose inference or
the original searches. The saved boxes have no foot samples, so this comparison
does not exercise the production player checks.

All scores are mean combined scores over the same member middle frames. The
pooled mean is diagnostic and can include a member where the pooled court fails
that member's check; the member keeps its own scene court. A donor court enters
the comparison only if it has every score term and passes every member's check.

| Group | Pooled court | Best whole-scene court | Scene's own courts | Middle-frame courts | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| Video 003, 2 scenes | 0.489324 | 0.597417 (scene 3076) | 0.548500 | 0.570873 | Scene 3076's court in both scenes |
| Video 040, 11 scenes | 0.956844 | 0.952697 (scene 0080) | 0.929183 | 0.947985 | Pooled court |

On video 040, the pooled court beat each scene's own court in 11 of 11 frames
and the middle-frame court in 8 of 11. Its fit used 757 stripe samples. The
pooled and whole-scene courts are one fixed court across the group; the last
two columns describe a different court per scene.

**Why video 003 failed pooling.** Scene 3076 and scene 16208 named one physical
stripe differently. After alignment into scene 3076, the samples donated as
the right doubles sideline lay 2.7 native pixels on average from that court's
right singles line. They were at least 14.8 pixels from its right doubles line.
The pooled fit compromised between incompatible marking names.

| Video 003 scene 16208: its own court | After pooling: scene 3076's court |
| --- | --- |
| ![Scene court](assets/video003_16208_scene_court.png) | ![Final court](assets/video003_16208_final_court.png) |

In the final court, the right boundary follows the outer yellow stripe. The
scene's own court misses the visible doubles stripe.

## Pending: full video-robust GPU run

A complete run of video 040 (Denmark Open 2019 quarter-final, 103,476 frames at
25 fps) started on 29 September from commit `6fa3bf44`. It uses fresh scene
detection, live GPU line and pose inference, GPU templates, eight CPU workers
and `--court-mode video-robust`, with chronological reuse off. It is the first
test of pooling with real foot checks and the first timing of whole-group
scoring. Its outcome is not known at the time of writing.

Before a merge decision, record process and per-video completion, scene
coverage, court/no-court/failure counts, full wall time, pooling time and group
winners. Inspect a few scene/final pairs on identical images, including any
member rejections and whole-scene fallback. This run is an end-to-end check,
not a controlled speed comparison or a labelled accuracy benchmark.

## Limits of this evaluation

- Most views are development views; there is no held-out labelled test set.
- Labelled references mix stripe centres and edges, so small pixel differences
  can be ambiguous.
- ShuttleSet's supplied homography is fixed per video. After a camera change it
  can be wrong for the current shot, so check the visible lines instead.
- Visual reviews include early qualitative development rulings and the project
  owner's later composition review. They were not a blinded, independent
  accuracy assessment.
- Cached pooling covers two groups. Video 040's group is 11 scenes of one
  camera view, not 11 independent tests.

## Reproducing the numbers

The committed files in [data/](data/) support the pooling scores, corner
scatter, scene-window counts and broad-run output totals. From the repository
root, this example rechecks the video 040 score comparison:

```python
import gzip
import json

with gzip.open("docs/court_detector/data/pool_fallback_video040.json.gz", "rt") as stream:
    run = json.load(stream)
group = run["groups"][0]
print(group["mean_combined_scores"])  # pooled, scene and middle means
wins = 0
for row in run["rows"]:
    scores = row["view_pool"]["scores"]
    wins += scores["pooled"]["combined_score"] > scores["scene"]["combined_score"]
print(wins, "of", len(run["rows"]))
```

Rerunning the historical experiments needs external videos and saved model
outputs. Some comparison scripts also remain in private working directories;
the retained data supports auditing the reported results without those files.
For a fresh detector run, use the committed operator guide. For numerical
comparisons, use one environment for both versions. The recorded server
environment was Python 3.12.13, NumPy 2.5.3, SciPy 1.17.1 and OpenCV 5.0.0.93.

## Retained evidence

The [provenance manifest](data/provenance.json.gz) records each file's exact
historical source path and content MD5. Compressed files retain their original
contents; all five PNGs are unchanged source outlines. The images total about
9.7 MiB. Historical paths in the manifest are provenance, not required inputs.

| Files in [data/](data/) | Purpose |
| --- | --- |
| `pool_fallback_video003.json.gz`, `pool_fallback_video040.json.gz` | Group membership, court alternatives, scores and final rows for the cached fallback checks |
| `label_conflict_video003.json.gz` | Distances that exposed the singles/doubles naming conflict |
| `pool_reference_frame_scores.json.gz` | Earlier unconditional-pool comparison on scene reference frames; distinct from the final middle-frame comparison |
| `composite_scatter_video040.json.gz` | Registered corners, scene IDs and scatter calculations |
| `composition_prefix_video040.csv.gz` | Outcomes for the 218-scene composition prefix, including missing comparisons |
| `scene_checks_24_videos.json.gz` | Reuse-group agreement and player-window counts |
| `broad_run_batch1_summary.json.gz` through `batch3` | Broad-run statuses, scene counts and timings |
| `interval_cpu_*.json.gz`, `interval_cuda_*.json.gz` | Timing, source-interval validation and all 33 scene results for the paired backend check |
| `net_reference_comparison.json.gz` | The labelled, retrospective net-weight comparison |

The earlier development choices are supported by the committed
[decision record](../../scratch/court_det_fix/DETECTOR_DECISIONS.md),
[CourtKeyNet retirement evidence](../../scratch/court_det_fix/evidence/retirement/README.md)
and [paired net-weight report](../../scratch/court_det_fix/net_recovery/statistics/paired_reference_report.md).
The [historical performance record](../../scratch/court_det_fix/court_detector/PERFORMANCE.md)
explains the optimisation comparisons. These are supporting records; this
handover preserves the conclusions needed to operate and maintain the detector.
