# Search trial: do player checks discard useful courts?

Relaxing player checks recovered some individual courts but introduced false courts.
After sharing between matching camera views, the three methods had the same main-camera
result. This eight-video development trial supports retaining the existing
player-required search.

The question arose because player checks can reject a visible court when people are
missed or partly out of view. The detector also scores how well court lines match the
image. The trial tested whether choosing by score or searching more broadly would
recover useful courts.

## Comparison and results

The trial ran fresh detections on eight selected videos: ShuttleSet 11, 21, 30 and 36,
and ShuttleSet22 27, 43, 44 and 51. These included known failures and otherwise good
videos with difficult scenes.

The trial compared three ways of finding and choosing a court:

| Method | Change from the original search |
| --- | --- |
| Original player checks | Candidates must satisfy the existing checks on where people stand. |
| Score-first selection | The highest court score wins; player support breaks exact ties. Earlier search filters stay in place. |
| Broader search | Player-based rejection is also removed during the search. Geometry and camera checks remain. |

Errors compare corners with the supplied main-camera references at 1280 × 720. The
comparison uses a fixed set of 726 main-camera scenes. All three methods then shared
courts between matching views. After sharing, all three had every corner within 10
pixels of the reference in all 726 main-camera scenes being compared. The search changes
did not improve that final count.

Before sharing, the alternatives did repair some of those 726 individual fits. With
every corner required to be within 20 pixels, score-first repaired 16 and broke one.
Broader search repaired 18 and broke the same one. That failure was ShuttleSet 11 scene
154: its worst corner went from 9.4 pixels away to 1,418.6 pixels away. Sharing later
replaced it with a court within 2.3 pixels. The good final result therefore concealed a
bad individual detection.

Broader search supplied courts to 13 rallies that previously had none and removed one
elsewhere. Only two of those additions matched the main-camera reference within 10
pixels. Across all 668 trial rallies, the count with a representative court averaging at
most 10 pixels of corner error rose from 625 to 627; score-first reached 626.

The separate score-first review checked eight detections it added and eight it removed.
None of those 16 images contained a real court.

A separate review of the 13 added-rally images found three clear courts, two fragments
where leaving the court missing was preferable, and eight images with no court. All 13
rally images show a different frame from the one the detector analysed; the
[detection-frame images](rally_review/easy_court_misses/) show what it actually saw in
the three clear-court cases.

The trial ran before the later change that lets courtless scenes receive a shared court.
Its results apply to these eight development videos.

## Selection rules

Score-first removes the final player veto, including after the court refit. Its earlier
search filters remain. Broader search also removes player-based rejection during search.
When court scores tie, these alternatives prefer candidates passing the original player
rule, then candidates with a person in at least half the samples, then the rest. There
is no player-score bonus.

There were seven exact score ties among 1,660 recorded score-first choices. Player
support did not change any of them. The observed changes therefore came from relaxing
the acceptance rule, rather than the tie-breaker.

## Processing time

The search trial ran two methods together in each job so they could reuse search work.
The original/score-first pair took 16.95 summed hours. The broader-search pair included
an extra version that kept the final player check. Together they took 25.17 hours, 48.5%
more. The logs do not separate the individual methods’ times, so the comparison measures
paired-job cost.

## What to take forward

The sampled frames explained several apparent search failures. The three clear-court
rally images led to two scenes sampled during a transition or partial view, and one
scene rejected by the player check. The later sharing repair recovered the third case.
The [release
evaluation](../released_dataset_evaluation/README.md#where-courts-are-still-wrong-or-missing)
records those outcomes.

This trial favours work on sampling, partial views and false-court rejection before
widening the search by default. Its selected videos and reviewed images do not establish
an accuracy rate on unfamiliar footage.

## Files

- [Selection comparison](selection_results/README.md): detailed score-first
  counts and its 20-image review.
- `search_results/`: tables comparing all three methods, including processing time.
- [Review images and clips](rally_review/README.md): the courts and missed scenes
  checked by hand.
- `trial_run_video.py`: the experiment runner. It records two choices from each
  search so the alternatives can reuse the same work.
- [`../inputs/player_tiebreak_results/`](../inputs/player_tiebreak_results/): saved
  predictions, choices and timings used by the comparison scripts.

The saved files use these original job names:

| Stage | Folder | Method |
| --- | --- | --- |
| A | `videos/` | Original player checks |
| A | `trial_videos/` | Score-first |
| B | `trial_videos/` | Broader search |
| B | `videos/` | Extra comparison: broader search with the original final player veto |

The [reproduction guide](../reproduce.md#compare-the-three-search-methods) rebuilds the
tables from these saved files.
