# Predictions used in the comparisons

These saved outputs let the comparison scripts recalculate results without
running court detection again. The [reproduction guide](../reproduce.md) has
the commands.

| Location | Contents |
| --- | --- |
| `cohort.json.gz`, `videos/` | Original detections for the 86 videos, before the sharing repair. |
| `player_tiebreak_results/` | Eight-video search-trial outputs, grouped by job and stage. Each stage contains two methods' predictions, choices and elapsed times. |
| `choices/*.jsonl.gz` inside each trial stage | Candidates chosen by each method, their scores, player measurements and search timings. |

The [trial README](../search_policy_trial/README.md#files) maps the stage and
folder names to the three methods. Stage A used detector commit `928ed398`;
stage B used `cd776fb2`.

## Larger optional inputs

Two kinds of input are stored separately from Git:

- `candidate_pools/<worker>/<stage>/*.jsonl.gz`: scores for every candidate
  court. These allow other selection rules to be tried without repeating the
  search. The checked-in choice records contain the selected candidates only.
- `../search_policy_trial/rally_review/scene_clips/`: the two complete scenes
  where the detector sampled a transition or a sliver of court. They show the
  clear view that appears later.

The numerical comparisons use the checked-in predictions and choice records;
they do not need these larger files. Source-video fields contain filenames,
so rendering also needs the source videos at their local locations.
