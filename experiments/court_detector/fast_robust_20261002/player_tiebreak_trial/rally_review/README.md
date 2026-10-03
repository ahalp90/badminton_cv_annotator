# Watch the same rallies under all three methods

These nine clips show three rallies from different videos, each rendered with
the court-sharing patch, score-first selection and search without player rejection.
The court outline follows each method's saved result for the current scene.
A frame without an outline means that method supplied no court.

| Rally | Cuts within labelled play | Patched court sharing | Score-first selection | Search without player rejection |
| --- | ---: | --- | --- | --- |
| ShuttleSet22 43, set 3, rally 19 | 2 | [Clip](clips/ss22_43_set3_rally19_court_sharing_patched.mp4) | [Clip](clips/ss22_43_set3_rally19_score_first.mp4) | [Clip](clips/ss22_43_set3_rally19_search_without_player_rejection.mp4) |
| ShuttleSet 11, set 3, rally 2 | 2 | [Clip](clips/sset_11_set3_rally02_court_sharing_patched.mp4) | [Clip](clips/sset_11_set3_rally02_score_first.mp4) | [Clip](clips/sset_11_set3_rally02_search_without_player_rejection.mp4) |
| ShuttleSet 36, set 1, rally 9 | 1 | [Clip](clips/sset_36_set1_rally09_court_sharing_patched.mp4) | [Clip](clips/sset_36_set1_rally09_score_first.mp4) | [Clip](clips/sset_36_set1_rally09_search_without_player_rejection.mp4) |

The rallies have the most scene cuts among the eight trial videos, taking
one rally per video. Ties use longer frame intervals, then video ID. No third
video has a rally with two cuts. Selection does not use prediction errors.

Each clip includes one second before the first labelled contact and two seconds
after the last. Durations are 28.33, 16.72 and 41.27 seconds respectively.
The clips retain the source's 1920 × 1080 resolution and frame rate, without
audio. Dashed red 1 px lines trace the outer borders of the predicted white
stripes. Video compression can soften them; the PNGs preserve the precise
overlay positions.

## PNG folders

The [full SS22 27 scene](../../../../../local_scratch/court_evaluation/easy_court_misses/clips/ss22_27_scene0027.mp4) lasts
56.0 seconds; its detector sample is at 28.0 seconds. The
[full SS22 44 scene](../../../../../local_scratch/court_evaluation/easy_court_misses/clips/ss22_44_scene0349.mp4) lasts
20.3 seconds; its detector sample is at 10.13 seconds. Both clips are plain,
silent 1080p video covering the complete detected scene, without overlays.
These two clips are local review files and are not committed.
The [actual detection-frame PNGs](easy_court_misses/) compare all three methods
at the sampled frames for these two scenes and ShuttleSet 36 scene 383.

Each representative folder contains one court per trial video, selected with
the same method as the original gallery: choose the view group with the most
rally time, then the actual court closest to the group's other courts,
weighted by rally overlap. The reference homography does not choose the court.

- **Eight representative courts per method:** [court-sharing patch](representative_courts/court_sharing_patched/), [score-first selection](representative_courts/score_first/), [search without player rejection](representative_courts/search_without_player_rejection/).
- **One still per scene in the padded clips:** [court-sharing patch](scene_stills/court_sharing_patched/), [score-first selection](scene_stills/score_first/), [search without player rejection](scene_stills/search_without_player_rejection/). Each folder has 12 PNGs.
- **Shared-court check:** [six PNGs](shared_court_check/) compare two rally frames in ShuttleSet22 43. Reviewed: both score-first methods fit a wrongly oriented court; the court-sharing patch is correctly oriented but inset by one segment in each direction. All three are wrong. The [trial report](../search_results/README.md#visual-evidence-and-remaining-misses) records the details.
- **Second scene sample:** [court-sharing patch](second_scene_stills/court_sharing_patched/), [score-first selection](second_scene_stills/score_first/), [search without player rejection](second_scene_stills/search_without_player_rejection/). Seven matching frames per method, from ShuttleSet 30 set 1 rally 18, ShuttleSet 21 set 2 rally 11 and ShuttleSet22 51 set 1 rally 12. Each rally contains one cut; the same lead and tail apply.
- **13 rallies gaining a court over the court-sharing patch:** [court-sharing patch](added_rally_courts/court_sharing_patched/), [score-first selection](added_rally_courts/score_first/), [search without player rejection](added_rally_courts/search_without_player_rejection/). Thirteen matching frames per method, covering every added rally court in the statistical comparison.
- **Original full corpus:** [86 representative courts](../../courts/). These remain the original extracts, before the fixes. Only the eight trial videos have all three methods available.

The [trial report](../search_results/README.md) gives the statistics. These
selected rallies show camera transitions; they do not estimate error frequency.
The trial report records the user's judgements of both samples. The second
sample uses the next three videos by the same cut-count rule, excluding the
first set's videos.

The 13 added-court stills use the broader-search representative scene for each
rally. Each frame lies inside both that scene and the labelled rally interval;
it is the detector's sampled frame where possible, otherwise the midpoint of
their overlap. The court-sharing patch supplies no court in any of these scenes.
ShuttleSet22 43 set 3 rally 3 and ShuttleSet22 44 set 2 rally 34 / set 3 rally 31
each have a single labelled contact, so their evaluation intervals are one frame.
The [added-court manifest](added_rally_courts/manifest.json.gz) records all 13
frame choices and the three methods' saved courts. All 39 PNGs use the same
1 px dashed red outlines and were compressed with pngquant and oxipng.

## Reproduce

The [rendering script](../../../../../scripts/render_court_trial_rallies.py)
has `plan`, `fetch` and `render` commands. Run from the repository root with
`PYTHONPATH=src:.` and the project's Python environment. `fetch` runs on the
host holding the source videos. It checks the decoded frame positions.

The [manifest](manifest.json.gz) records the source frames, saved courts,
selected scenes and encoded clip checks. The raw fetched frames and clips
remain in `local_scratch/court_evaluation/rally_review/fetched/`.

```bash
PYTHONPATH=src:. ~/.venvs/badminton-cicd/bin/python \
  scripts/render_court_trial_rallies.py render \
  --plan local_scratch/court_evaluation/rally_review/plan.json.gz \
  --fetched local_scratch/court_evaluation/rally_review/fetched \
  --output experiments/court_detector/fast_robust_20261002/player_tiebreak_trial/rally_review
```

All nine clips have the expected frame counts and dimensions. All 60 planned
PNGs were rendered and compressed with pngquant and oxipng. The six additional
sharing-check PNGs have their own [frame requests](shared_court_check/requests.json.gz).
