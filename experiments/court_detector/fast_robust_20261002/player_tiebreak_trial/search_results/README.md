# Three court methods tie on the main view; broader search adds unjudged courts

**On the eight trial videos, the court-sharing patch, score-first selection and
broader search give the same final main-view result.** All three put all
726 fixed main-view scenes within 10 px at every corner. Their representative
courts differ by at most 0.04 px of mean error.

Before sharing, both score-first methods bring about 2 percentage points more
main-view fits within 20 px, but sharing already fixes those scenes. Broader
search gives a court to 13 rallies where the court-sharing patch has none. Only two of
those agree with the reference; the other 11 are 284–1,220 px away and have not
been judged. Its paired jobs took 48.5% longer in total. The user prefers the
simple court-sharing patch on the evidence reviewed so far.

## Question and sample

How much do score-first selection and broader search change court agreement
and coverage, compared with the court-sharing patch and with each other, on the same
videos?

| Method | Search and selection |
| --- | --- |
| Patched court sharing | Original search and player checks; failed sharing affects only the receiving scene |
| Score-first selection | Original search; final choice uses court score, with player support only for exact ties |
| Broader search | Search without player rejection, plus score-first selection; geometry, camera and search limits remain |

All three include the court-sharing patch. The tables also keep the original extracts
and an auxiliary `search_player_veto` arm for context; neither is one of the
three methods.

The sample is ShuttleSet 11, 21, 30 and 36, and ShuttleSet22 27, 43, 44 and 51:
4,615 scenes and 668 usable rallies. These purpose-selected videos are the only
ones with all three methods. The numbers describe these eight videos; they do
not estimate how the new methods would do across the 86-video corpus. The
[86-video rally-view statistics](../../rally_views/README.md) measure the
original extracts only.

## How the comparison works

**The video is the unit.** Each method gets one value per video, and each video
counts equally in a mean. Scenes and rallies within a video share a camera,
a venue and often a shared court, so they are not independent trials. Pooled
counts sit beside the means with their denominators; they weight videos by
their number of scenes, rallies or frames.

**Differences are paired.** Each contrast subtracts one method's value from
another's within the same video, then averages over the eight videos.

**Intervals show reweighting, not generalisation.** Each interval runs from the
2.5th to the 97.5th percentile of the mean over 2,000 resamples of whole videos,
seed 20261003. One draw matrix serves every method, contrast and runtime ratio.
The interval shows how much a mean moves when these eight videos are
reweighted. It is not a confidence interval for the 86 videos or new venues:
the sample is small and chosen on purpose, and shared venues limit what it
says about new settings. A zero-width interval means no video changed; it does
not prove the methods are equivalent. No significance tests are made.

The measures:

- **Representative error.** Each video's representative court is the medoid
  of its main view group: the court closest to the group's other courts. Error
  is its mean four-corner distance from the supplied default-camera homography
  at 1280 × 720. Polygon overlap (IoU) is descriptive only.
- **Fixed main view.** The same 726 scenes for every method: the court-sharing-patched
  run's view group with the most labelled rally frames in each video. Of these,
  610 overlap rallies. Each video contributes the percentage of its scenes
  whose **worst corner** is within 10 px after sharing, and within 20 px for the
  scene's own fit before sharing.
- **Rally-time coverage.** The percentage of labelled rally frames inside a
  scene with any court. This measures presence, not accuracy.
- **Rally view.** For each rally, take the view group with the most rally
  frames, then that group's medoid court, as in the
  [rally-view method](../../rally_views/README.md#measure-the-dominant-view-of-each-rally).
  Does the group have a court, and is it within 10 px mean corner error of the
  reference? This is agreement with the default-view reference across all
  rallies, including those dominated by another detector group or an ungrouped
  scene. Such a group may or may not show a different camera view, so this
  does not score accuracy in alternate views. A missing court counts as a miss.

## Results by method

| Per-video mean [95% reweighting interval] | Patched court sharing | Score-first selection | Broader search |
| --- | ---: | ---: | ---: |
| Representative error, px | 2.84 [2.00, 3.87] | 2.84 [2.00, 3.87] | 2.84 [2.01, 3.88] |
| Representative error median / p90, px | 2.60 / 4.15 | 2.60 / 4.15 | 2.60 / 4.15 |
| Representative IoU | 0.984 | 0.984 | 0.984 |
| Main view within 10 px after sharing | 100% (726/726) | 100% (726/726) | 100% (726/726) |
| Main-view own fits within 20 px before sharing | 97.3% [96.0, 98.7] | 99.3% [98.6, 99.8] | 99.6% [99.2, 99.9] |
| — pooled scenes | 706/726 | 721/726 | 723/726 |
| Rally-time coverage | 97.2% [94.9, 98.8] | 97.5% [95.3, 99.1] | 98.2% [96.6, 99.3] |
| — pooled rally frames | 97.0% | 97.3% | 98.0% |
| Rally view has a court | 96.0% [94.3, 97.5] | 96.4% [94.4, 98.3] | 97.6% [96.6, 98.8] |
| — pooled rallies | 638/668 | 640/668 | 650/668 |
| Rally view within 10 px | 94.2% [92.0, 96.3] | 94.4% [92.0, 96.8] | 94.5% [92.2, 96.9] |
| — pooled rallies | 625/668 | 626/668 | 627/668 |

These method intervals overlap heavily because they mostly reflect how much
the videos differ from each other. The paired differences below remove that
spread.

## Paired differences

Each cell gives the mean within-video difference, its 95% reweighting interval,
then how many videos went higher, lower or stayed equal. Percentages differ in
percentage points (pp).

| Measure | Score-first − court-sharing patch | Broader − court-sharing patch | Broader − score-first |
| --- | ---: | ---: | ---: |
| Representative error, px | 0 [0, 0]; 0/0/8 | +0.005 [0.000, 0.016]; 2/0/6 | +0.005 [0.000, 0.016]; 2/0/6 |
| Main view within 10 px after sharing, pp | 0 [0, 0]; 0/0/8 | 0 [0, 0]; 0/0/8 | 0 [0, 0]; 0/0/8 |
| Own fits within 20 px before sharing, pp | +1.95 [0.75, 3.04]; 6/1/1 | +2.28 [0.88, 3.57]; 6/1/1 | +0.33 [0.00, 0.66]; 2/0/6 |
| Rally-time coverage, pp | +0.33 [0.08, 0.65]; 5/1/2 | +0.95 [0.36, 1.74]; 8/0/0 | +0.62 [0.13, 1.33]; 7/0/1 |
| Rally view has a court, pp | +0.47 [−0.14, 1.08]; 3/1/4 | +1.64 [0.89, 2.59]; 7/0/1 | +1.17 [0.25, 2.34]; 4/0/4 |
| Rally view within 10 px, pp | +0.20 [0.00, 0.61]; 1/0/7 | +0.32 [0.00, 0.75]; 2/0/6 | +0.11 [0.00, 0.34]; 1/0/7 |

**The representative courts are effectively identical.** The court-sharing patch and
score-first selection choose identical corners in all eight videos. Broader
search raises the mean error by 0.0001 px in ShuttleSet22 44 and 0.04 px in
ShuttleSet22 27. Those count as "higher" in the table, but they are far too
small to see.

**Score-first repairs individual fits that sharing already fixes.** Before
sharing, score-first selection repairs 16 main-view fits at 20 px and damages
one. Broader search repairs 18 and damages the same one, compared with the
court-sharing patch. Compared with score-first selection, it repairs two and damages
none; at 10 px it repairs three and damages one. After sharing, all three
methods have all 726 scenes within 10 px, so none of these changes moves the
final count. The before-sharing gain is real for individual fits, but the
final result hides it.

The damaged fit is ShuttleSet 11, scene 154, with 189 labelled rally frames.
Under both score-first methods its worst corner grows from 9.4 to 1,418.6 px;
sharing then restores it to within 2.3 px. A bad individual court can hide
behind a successful shared fit. The [selection report](../selection_results/README.md)
records the case.

**Broader search adds rally courts but little agreement.** It gives a court
to 13 rallies where the court-sharing patch has none, and drops one. Two of the 13
agree with the reference within 10 px; the other 11 sit 284–1,220 px away.
[Matched PNGs of all 13 cases](../rally_review/added_rally_courts/) show the
same frame under each method.
Against the court-sharing patch, score-first selection adds three and drops the same
one: ShuttleSet22 44, set 3, rally 5, where the court-sharing patch's court is 468.5 px
from the reference.

In every method, every rally court outside 10 px, and every rally without a
court, sits in a rally dominated by a group other than that run's main group.
In rallies dominated by the main group, all three methods have a court within
10 px. If a non-main group shows the default camera, a court that far off is
wrong; if it shows another view, the reference cannot judge it. Nobody has
judged these courts.

**Agreement within 10 px barely moves.** The methods agree with the reference
in 625, 626 and 627 of 668 rallies. The lower bounds sit at zero because only
one or two videos change, and many resamples leave those videos out.

**Coverage rises, but coverage is not accuracy.** Broader search raises
rally-time coverage above the court-sharing patch in every video. A frame counts as
covered when its scene has any court, right or wrong.

Each dataset has four videos, so these are descriptive means only. Under the
court-sharing patch, ShuttleSet videos have higher representative error than
ShuttleSet22 (3.59 px against 2.08 px) and lower rally-time coverage (95.5%
against 98.9%). Broader search's extra rally courts come mostly from
ShuttleSet (329 → 337 of 348 rallies) rather than ShuttleSet22 (309 → 313 of
320). Its within-10 px count rises by one rally in each dataset.

## Per-video results

Each cell lists court-sharing patch / score-first selection / broader search.

| Video (rallies) | Representative error, px | Main-view own fits within 20 px | Rally-time coverage, % | Rallies with a court | Rally courts within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShuttleSet 11 (108) | 1.81 / 1.81 / 1.81 | 90 / 89 / 89 of 90 | 89.6 / 90.2 / 93.1 | 98 / 98 / 103 | 96 / 96 / 96 |
| ShuttleSet 21 (75) | 3.20 / 3.20 / 3.20 | 73 / 75 / 76 of 76 | 98.1 / 98.3 / 98.6 | 71 / 72 / 72 | 70 / 70 / 70 |
| ShuttleSet 30 (104) | 6.00 / 6.00 / 6.00 | 119 / 123 / 123 of 123 | 97.6 / 97.7 / 98.2 | 100 / 100 / 101 | 98 / 98 / 98 |
| ShuttleSet 36 (61) | 3.36 / 3.36 / 3.36 | 72 / 74 / 75 of 76 | 96.7 / 97.1 / 97.3 | 60 / 61 / 61 | 60 / 61 / 61 |
| ShuttleSet22 27 (59) | 1.86 / 1.86 / 1.91 | 62 / 62 / 62 of 62 | 98.7 / 100.0 / 100.0 | 58 / 59 / 59 | 58 / 58 / 58 |
| ShuttleSet22 43 (96) | 2.67 / 2.67 / 2.67 | 133 / 138 / 138 of 139 | 98.6 / 98.6 / 98.6 | 93 / 93 / 94 | 92 / 92 / 92 |
| ShuttleSet22 44 (109) | 2.53 / 2.53 / 2.53 | 104 / 105 / 105 of 105 | 98.7 / 98.7 / 99.6 | 103 / 102 / 105 | 98 / 98 / 99 |
| ShuttleSet22 51 (56) | 1.26 / 1.26 / 1.26 | 53 / 55 / 55 of 55 | 99.8 / 99.8 / 99.8 | 55 / 55 / 55 | 53 / 53 / 53 |

ShuttleSet 11 shows the pattern most clearly: broader search adds five rally
courts there, and none agrees with the reference.

## Courts outside the main view need visual judgement

Scenes with any court rise from 1,263 (court-sharing patch) to 1,317 (score-first
selection) to 1,900 (broader search) of 4,615. Compared with score-first
selection, broader search adds 669 scenes with courts and removes 86. Of the
additions, 620 are outside labelled rallies and 49 overlap rallies outside the
fixed main view. Within rally intervals, the additions cover 1,256 frames and
the removals 69. Manual review found separate detection groups showing the same
court and camera view after the arena surroundings changed. A scene outside the
fixed group therefore need not show a different camera view. Abstaining can be
the right result.

The user's visual judgements so far:

- **Earlier image review.** All 16 court-bearing images were wrong and all four
  abstentions reasonable. On those 16 scenes, broader search keeps six courts
  identical to the rejected geometry, has no court in seven, and changes three
  to unjudged courts. All four abstentions remain. The images were chosen to
  show changes, so they do not estimate a false-court rate. The
  [scene table](reviewed_scenes.csv.gz) records each comparison.
- **A rejected court reused in rallies.** In ShuttleSet22 43, both score-first
  methods share the court from scene 483 (review image 02, judged wrong) into
  scenes 89 and 176, covering 171 labelled rally frames. The
  [receiving frames](../rally_review/shared_court_check/) have now been reviewed.
  Both score-first methods fit the lines well but orient the court vertically
  within the middle band of the horizontally visible court. The court-sharing patch
  also gives a wrong court: its orientation and alignment are nearly right,
  but it is inset by one vertical segment and one horizontal segment.
  The user suspects face-on seated judges were mistaken for standing people,
  including line judges whose towers make them appear unusually tall. That
  explanation has not been verified. Improving those person detections is
  outside the remaining project scope.
- **First scene-still sample.** The court-sharing patch and score-first selection were
  equivalent and good. Broader search produced one malformed court at frame
  144569 in ShuttleSet22 43. Less than a quarter of the court was visible; many
  lines aligned, but the court's extent and orientation were wrong. The user
  was undecided whether that court is better than an abstention.
- **Second scene-still sample.** All three methods were equivalent across seven
  matching frames from three more videos.

The [nine full rally clips](../rally_review/clips/) show three rallies under
all three methods. They were picked by scene-cut count, not by error. The
[review notes](../rally_review/README.md) list the PNG folders for both samples.

## Runtime

Stage A ran the court-sharing patch and score-first selection in one job per video,
sharing the search work. Stage B did the same for broader search and the
auxiliary player-veto arm. These timings compare paired stage jobs, not three
separately timed methods.

| Paired stage jobs, 8 videos | Stage B against stage A |
| --- | --- |
| Summed job time | 16.95 h → 25.17 h; ratio 1.485, so 48.5% longer |
| Mean extra time per video | +61.6 min [49.7, 74.6]; range 36.8–89.7 min |
| Per-video time ratio | 1.415–1.654; geometric mean 1.504 [1.451, 1.563] |

The summed ratio weights long videos more. The geometric mean weights each
video equally. It comes out higher because the ShuttleSet22 jobs, mostly the
shorter ones, had larger ratios (1.53–1.65, against 1.42–1.46 for ShuttleSet).
Videos ran concurrently, so the summed hours are not waiting time.

The logs do not separate the court-sharing patch from score-first selection. In stage
A, the extra score-first choice and refit step took 1.4–5.1 min per video. That
timer excludes sharing and reuses the baseline refit where possible, so it is
not a standalone speed comparison.

## What this means for the choice

The main-view measures cannot separate the methods: all three reach the same
ceiling with the same representative courts. The differences lie outside the
main view group, where the reference cannot always judge. There, the earlier
image review found all 16 selected courts wrong. Broader search costs about
half as much again in compute and adds unjudged courts. In the first
scene-still sample, it also produced the malformed court described above.

The user prefers the simple court-sharing patch on the evidence reviewed so far. That
is a practical choice from small samples, not a ranking across all views. No
full-corpus repair has been launched. The project needs reliable courts to
measure player positions and distances in court metres.

## Evidence and reproduction

Tables added for the statistics:

- [Per-video metrics](video_metrics.csv.gz): one row per method and video,
  with each count and its denominator. The summary and paired effects average
  these rows.
- [Method summary](method_summary.csv.gz): all-video means with intervals,
  medians, p10 and p90, pooled counts, and descriptive rows for each dataset.
- [Paired effects](paired_effects.csv.gz): the three method contrasts and the
  stage-job timings.
- [Rally views](rally_views.csv.gz): one row per method and rally, from
  `rally_views` in `scripts/summarise_court_rally_views.py`.

The earlier tables are unchanged:
[per-video results](per_video.csv.gz), [paired scenes](paired_scenes.csv.gz),
[main-view transitions](main_view_transitions.csv.gz),
[detection changes](detection_changes.csv.gz), [view groups](view_groups.csv.gz),
[court sources](court_paths.csv.gz), [run checks](audit.csv.gz) and
[reviewed scenes](reviewed_scenes.csv.gz).

Run the [comparison script](compare_search.py) from the repository root:

```bash
PYTHONPATH=src:. ~/.venvs/badminton-cicd/bin/python \
  experiments/court_detector/fast_robust_20261002/player_tiebreak_trial/search_results/compare_search.py
```

Inputs are the downloaded directories under
`local_scratch/court_evaluation/player_tiebreak_results/{carmack,bourbaki}/`.
The three methods are `stage_a/videos`, `stage_a/trial_videos` and
`stage_b/trial_videos`, respectively. The auxiliary arm is `stage_b/videos`.

Each run checks that both hosts finished both stages with exit 0, that all
arms share the same scene partition, and that saved individual fits match the
choice records. No scene or group failures were recorded. The statistics rerun
of 3 October finished with exit 0 and reproduced the eight earlier tables
exactly after decompression. Ruff and an explicit Pyrefly check of the script
passed with exit 0. Before the statistics were added, an independent Opus 5.5
xhigh review reproduced the counts and checked a randomly selected rally. No
dedicated tests or detector reruns were added. A separate check of the statistical
extension reproduced the method means, rally counts and a paired interval.
All 18 paired method estimates and intervals were also independently recalculated.
