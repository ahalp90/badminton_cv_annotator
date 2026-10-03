# Evidence behind the court comparisons

This directory contains the saved material used by the reports in
[comparisons/](../comparisons/). The reports explain the findings and link to
individual files; the folders here preserve the inputs and outputs needed to
check them.

| Folder | Contents | Report |
| --- | --- | --- |
| [development/](development/) | Earlier timing, composition and pooling checks, with comparison images | [Development evaluation](../comparisons/development_evaluation.md) |
| [release/](release/) | Measurements and images of the released 86-video extraction | [Release evaluation](../comparisons/release.md) |
| [baseline/](baseline/) | Original measurements, court images and sharing diagnostics | [Original extraction](../comparisons/baseline.md) |
| [search/](search/) | Search-trial comparisons and reviewed images and clips | [Player-check trial](../comparisons/search.md) |
| [inputs/](inputs/) | Saved predictions used to rebuild those comparisons | Described below |

The internal groupings keep related machine outputs together. The two named
worker folders under `inputs/player_tiebreak_results/` record where the trial
jobs ran; their names have no bearing on the findings.

## Saved predictions

These outputs allow the [comparison tools](../tools/README.md) to recalculate
results without running court detection again.

| Location under `inputs/` | Contents |
| --- | --- |
| `cohort.json.gz`, `videos/` | Original detections for the 86 videos, before the sharing repair |
| `player_tiebreak_results/` | Eight-video search-trial outputs, grouped by worker and stage; each stage contains two methods' predictions, choices and timings |
| `choices/*.jsonl.gz` within each trial stage | Selected candidates, scores, player measurements and search timings |

The [trial's file guide](../comparisons/search.md#files) maps the stage and folder
names to the methods. Stage A used detector commit `928ed398`; stage B used
`cd776fb2`.

## Larger optional inputs

Two kinds of evidence are stored separately from Git:

- `inputs/candidate_pools/<worker>/<stage>/*.jsonl.gz`: scores for every candidate
  court, allowing alternative selection rules to be tried without repeating the search
- `search/rally_review/scene_clips/`: the two complete scenes where detection
  sampled a transition or a sliver of court; the clearer view appears later

The numerical comparisons use the checked-in predictions and choices. Rendering
also needs the source videos at their local locations; saved source-video fields
contain filenames.
