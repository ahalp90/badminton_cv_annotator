# Test player support as a tie-breaker

Both trial stages completed on 3 October. The
[selection comparison](selection_results/README.md) finds
identical representative courts after sharing. The user judged all 16
court-bearing review images incorrect and all four abstentions reasonable.
That review does not support an overall preference for either method.

See the [completed search comparison](search_results/README.md) for results and
runtime, and the [rally clips and PNG folders](rally_review/README.md) for review.

Run the same eight videos twice. Stage A changes final selection and refit
acceptance. Stage B also removes player-based rejection during search. Both
stages include the court-sharing patch. Each stage runs from its own committed checkout
and writes to its own directory. Original extracts remain untouched.

The trial picks the highest combined court score. On an exact tie, it prefers
candidates that pass the current player rule, then candidates containing a
person in at least half the samples, then the rest. Further ties keep the
existing ranking order. There is no score bonus or tolerance. Final refitting
retains geometry and camera checks without requiring player presence.

Stage A keeps existing search pruning. Stage B retains geometric checks and
search limits while admitting courts without player support. Full-court player
measurements can break ties after axes are combined. Early axis and template
ranking stays score-only where those measurements are not yet available.

## Videos and outputs

ShuttleSet 30 and 36 had poor final courts; 11 and 21 had low-error final courts.
ShuttleSet22 43 and 44 had poor courts; 27 and 51 had low-error courts.
The sample includes individual-scene errors inside otherwise good videos.
Jobs ran with three workers per video. See the main report for measured timings.

`trial_run_video.py` runs two choices from each scoring record. The checkout's
normal choice is saved in `videos/`; score-first selection is in `trial_videos/`.
Both use separate view pools. Candidate scores, player measurements and corners
are saved in `choices/`. A failed scene, group or missing arm fails the video.

**Use stage A's `videos/` as the fixed-sharing baseline.** Stage B's `videos/`
uses the changed search but the original final player veto, so it is an
auxiliary result rather than the original control. The three main comparisons
are A/videos, A/trial_videos and B/trial_videos.

The selection change is isolated in the experiment entry point. The trial
did not justify replacing the production selection default. Its scripts remain
here to explain how the alternative results were produced.

## Interpretation

Timings cover paired work; they do not estimate the standalone cost of either
choice rule. The temporary deployment and monitoring scripts are kept in local
investigation history rather than this evidence collection.

Compare individual courts before sharing and final courts afterwards. Split
by rally overlap and main/other view. The static dataset homography only labels
the default view; alternate-view disagreement is not a measured error. Count
changed group assignments and new false-court risks outside rallies. Keep the
original cohort exclusions. Eight selected videos are a screening trial, not a
claim about accuracy across the entire corpus.
