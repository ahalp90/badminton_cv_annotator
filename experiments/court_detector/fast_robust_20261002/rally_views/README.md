# Court agreement in the view occupying most of each rally

**The original extracts agree with the default homography within 10 px in
5,228 of 6,833 rallies (76.5%).** Selecting the view group with the most time
in each rally gives the same count as the earlier longest-scene calculation.
This measures agreement with the supplied default view, not accuracy during
camera changes. The three later trial methods have results for eight videos
and must be assessed separately.

Does the median court proposed per rally match the video's supplied homography?

These results cover the original fast-robust extracts at `17a50b57`: 40
ShuttleSet videos and 46 ShuttleSet22 videos. The release exclusions include
ShuttleSet 9, 10, 12 and 27; ShuttleSet22 15; unresolved ShuttleSet22 14, 45 and
56; and eight overlaps counted under ShuttleSet only. Invalid contact ordering
excludes 377 rallies. Valid flaw-marked rallies remain included.

## Measure the dominant view of each rally

A rally interval runs from its first labelled contact through its last contact.
For each rally, add the overlap durations of scenes belonging to each detector
view group. Choose the group occupying the most frames, including scenes with
no court. Ties go to the group appearing first. Within the winning group, use
the actual detected court closest to its other courts, weighted by overlap
duration. Prefer a court whose sampled frame is inside the rally. A winning
group without a court counts as missing.

This chooses the court without consulting the homography. Corner error then
averages the four corner distances at 1280 × 720. We set 10 px mean corner error
as the upper sanity bound for a usable court in the default camera view.
Detection groups collect scenes with identical or near-enough camera views.
During manual review, some scenes showed the same court and camera view but
changed arena surroundings. These scenes received separate court fits, so
separate detection groups do not always mean different camera views.

| Measure | ShuttleSet | ShuttleSet22 | Combined |
| --- | ---: | ---: | ---: |
| Usable rallies | 3,182 | 3,651 | 6,833 |
| Winning group has a court | 3,011 | 3,451 | 6,462 (94.6%) |
| Court within 10 px mean corner error | 2,631 (82.7%) | 2,597 (71.1%) | 5,228 (76.5%) |
| Winning group is also the video's main group | 2,828 | 3,055 | 5,883 |
| Rally contains >1 detection group | 682 | 540 | 1,222 |

The winning group occupies more than half the rally in 6,829/6,833 cases.
Only two representative scenes differ from the longest-scene selection, with
no change to the 10 px count. Missing courts stay in the denominator.

Giving each video equal weight yields a mean rally agreement rate of **78.0%**,
with a 95% bootstrap interval of **70.8–84.1%**. The corresponding per-video
representative-court agreement is **74/86 (86.0%)**, with interval **79.1–93.0%**.
These intervals resample whole videos 2,000 times, seed 20261002. They describe
variation within this corpus. Shared venues limit inference to new settings.
They do not quantify errors in the static homography labels.

## Other views and the rest of the broadcast

Of 16.78 hours inside usable rally intervals, 1.89 hours fall outside the
video's main detector view group. Predictions cover 1.36 of those hours
(72.1% of that time; the mean per-video rate is 43.9%). Time outside the main
group includes alternate views, missed courts and separately grouped scenes
from the same camera view. This total therefore does not measure alternate-view
accuracy alone.

Outside labelled rally intervals, predictions cover 28.57 of 81.59 hours
(35.0%). That time includes unlabelled play and between-point footage as well
as replays, close-ups and other broadcast material. Neither detection rate is
an accuracy score. A missing court can be the correct response.

The user's review of the earlier trial's 20 selected images reinforces this
limit: all 16 predicted courts were wrong, while all four abstentions were
reasonable. Those judgements concern selected changes in the eight-video
trial, not a random sample of these original extracts.

## Use and reproduce these results

The [original report](../README.md) and [86 representative PNGs](../courts/)
describe the full corpus before repairs. The trial's
[selection report](../player_tiebreak_trial/selection_results/README.md)
records the corrected human review. Compare full rally clips before choosing
a method for a full-corpus rerun. Check whether each method follows camera
changes and abstains when it cannot fit a usable court. The project needs these
courts to measure badminton players' positions and distances in court metres.
Wrong geometry distorts those measurements.

The [per-rally table](per_rally_view.csv.gz),
[view-duration table](view_populations.csv.gz) and [summary](summary.json.gz)
retain the counts and denominators. Reproduce from the repository root:

```bash
PYTHONPATH=src:. ~/.venvs/badminton-cicd/bin/python \
  scripts/summarise_court_rally_views.py \
  --input experiments/court_detector/fast_robust_20261002 \
  --output experiments/court_detector/fast_robust_20261002/rally_views
```

The original evaluation was rerun on 3 October. Its per-video, per-scene,
per-rally and sharing-diagnostic tables reproduced exactly. The supplementary
calculation ran over all 86 videos and all 6,833 usable rallies.
