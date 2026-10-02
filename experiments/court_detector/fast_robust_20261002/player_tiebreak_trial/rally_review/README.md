# Watch the same rallies under all three methods

These nine clips show three rallies from different videos, each rendered with
the sharing fix, score-first selection and search without player rejection.
The court outline follows each method's saved result for the current scene.
A frame without an outline means that method supplied no court.

| Rally | Cuts within labelled play | Sharing fix | Score-first selection | Search without player rejection |
| --- | ---: | --- | --- | --- |
| ShuttleSet22 43, set 3, rally 19 | 2 | [Clip](clips/ss22_43_set3_rally19_sharing_fix.mp4) | [Clip](clips/ss22_43_set3_rally19_score_first.mp4) | [Clip](clips/ss22_43_set3_rally19_search_without_player_rejection.mp4) |
| ShuttleSet 11, set 3, rally 2 | 2 | [Clip](clips/sset_11_set3_rally02_sharing_fix.mp4) | [Clip](clips/sset_11_set3_rally02_score_first.mp4) | [Clip](clips/sset_11_set3_rally02_search_without_player_rejection.mp4) |
| ShuttleSet 36, set 1, rally 9 | 1 | [Clip](clips/sset_36_set1_rally09_sharing_fix.mp4) | [Clip](clips/sset_36_set1_rally09_score_first.mp4) | [Clip](clips/sset_36_set1_rally09_search_without_player_rejection.mp4) |

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

Each representative folder contains one court per trial video, selected with
the same method as the original gallery: choose the view group with the most
rally time, then the actual court closest to the group's other courts,
weighted by rally overlap. The reference homography does not choose the court.

- **Eight representative courts per method:** [sharing fix](representative_courts/sharing_fix/), [score-first selection](representative_courts/score_first/), [search without player rejection](representative_courts/search_without_player_rejection/).
- **One still per scene in the padded clips:** [sharing fix](scene_stills/sharing_fix/), [score-first selection](scene_stills/score_first/), [search without player rejection](scene_stills/search_without_player_rejection/). Each folder has 12 PNGs.
- **Shared-court check:** [six PNGs](shared_court_check/) compare two rally frames in ShuttleSet22 43. Both score-first methods receive their court from scene 483, whose source image the user judged wrong. Judge these receiving frames separately.
- **Original full corpus:** [86 representative courts](../../courts/). These remain the original extracts, before the fixes. Only the eight trial videos have all three methods available.

The [trial report](../search_results/README.md) gives the statistics. These
selected rallies show camera transitions; they do not estimate error frequency.

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
