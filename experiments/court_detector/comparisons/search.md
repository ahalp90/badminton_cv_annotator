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
[detection-frame images](../evidence/search/rally_review/easy_court_misses) show what it actually saw in
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
evaluation](release.md#where-courts-are-still-wrong-or-missing)
records those outcomes.

This trial favours work on sampling, partial views and false-court rejection before
widening the search by default. Its selected videos and reviewed images do not establish
an accuracy rate on unfamiliar footage.

## Choosing courts by score instead of player checks

Choosing the highest-scoring court improved some individual detections but
made no difference to the final main-camera courts after sharing. It also
produced false courts in frames where there was no court to find.

This comparison used eight videos: ShuttleSet 11, 21, 30 and 36, and ShuttleSet22
27, 43, 44 and 51. Both methods used the same search results. The alternative
selected by court score and used player positions only to break exact ties;
it also removed the player check after the final adjustment of the court.

### What changed

Errors below compare corners with the official main-camera labels at 1280 × 720.
Both methods are measured on the same 726 scenes from the main camera views.

| Result | Original player checks | Score-first |
| --- | ---: | ---: |
| Individual scene fits with every corner within 20 px, before sharing | 706 / 726 | 721 / 726 |
| Final scene fits with every corner within 10 px, after sharing | 726 / 726 | 726 / 726 |
| Scenes with any detected court, across all 4,615 scenes | 1,263 | 1,317 |

Before sharing, score-first repaired 16 fits and broke one. That failure was
ShuttleSet 11 scene 154: the worst corner difference rose from 9.4 to 1,418.6
pixels. Its oversized court scored 0.807, above the accurate court's 0.687.
Sharing then restored an accurate court. Looking only at the final results
would hide that mistake.

There were seven exact score ties among 1,660 recorded choices. Player support
did not change any of them, so this trial found no benefit from that tie-breaker.

Score-first added detections in 151 scenes and removed them in 97. Most changes
were outside labelled rallies: 142 additions and 93 removals. Two newly detected
courts were also shared with other scenes, spreading the effect of those choices.

### What the review images show

None of the 16 images with detections contained a real court.
The four other review images had no detected court; leaving them that way was
judged reasonable.

The [20 images](../evidence/search/selection_results/review_frames) contain:

- 01–08: a court added by score-first, one example per video.
- 09–16: a court retained by the original method but removed by score-first,
  one example per video.
- 17–20: frames within labelled rallies where neither method detected a court.

All 16 false-court examples were outside labelled rallies. They were selected
to inspect changes, so their count cannot estimate how often either method
makes this mistake across a whole broadcast.

### Results files

[Per-video results](../evidence/search/selection_results/per_video.csv.gz) and [paired scenes](../evidence/search/selection_results/paired_scenes.csv.gz)
contain the numerical comparisons. The [release evaluation](release.md)
covers the later sharing repair across all 86 videos. The
[reproduction guide](../tools/README.md#compare-the-three-search-methods)
rebuilds the trial tables.

## Images and clips from the search trial

These examples show what changed when player checks were relaxed, including
false courts and views that the detector missed. The
[trial report](#comparison-and-results) gives the numerical results.

### Three rallies crossing camera cuts

| Rally | Cuts within labelled play | Original player checks | Score-first selection | Search without player rejection |
| --- | ---: | --- | --- | --- |
| ShuttleSet22 43, set 3, rally 19 | 2 | [Clip](../evidence/search/rally_review/clips/ss22_43_set3_rally19_court_sharing_patched.mp4) | [Clip](../evidence/search/rally_review/clips/ss22_43_set3_rally19_score_first.mp4) | [Clip](../evidence/search/rally_review/clips/ss22_43_set3_rally19_search_without_player_rejection.mp4) |
| ShuttleSet 11, set 3, rally 2 | 2 | [Clip](../evidence/search/rally_review/clips/sset_11_set3_rally02_court_sharing_patched.mp4) | [Clip](../evidence/search/rally_review/clips/sset_11_set3_rally02_score_first.mp4) | [Clip](../evidence/search/rally_review/clips/sset_11_set3_rally02_search_without_player_rejection.mp4) |
| ShuttleSet 36, set 1, rally 9 | 1 | [Clip](../evidence/search/rally_review/clips/sset_36_set1_rally09_court_sharing_patched.mp4) | [Clip](../evidence/search/rally_review/clips/sset_36_set1_rally09_score_first.mp4) | [Clip](../evidence/search/rally_review/clips/sset_36_set1_rally09_search_without_player_rejection.mp4) |

The nine links contain four distinct clips. All three methods produce identical
clips for ShuttleSet 11 and 36. The original-player-check and score-first clips are
also identical for ShuttleSet22 43.

The rallies were chosen for their camera cuts, one per video. Each clip includes
one second before the first labelled contact and two seconds after the last.
Dashed lines show the predicted court stripes. The still images below preserve
small line offsets more clearly than the compressed videos.

### Still images and missed scenes

| Folder | What it shows |
| --- | --- |
| [representative_courts/](../evidence/search/rally_review/representative_courts) | One typical court from each trial video's main camera, for each method. |
| [scene_stills/](../evidence/search/rally_review/scene_stills) | Twelve frames per method, covering the scenes in the three clips above. |
| [second_scene_stills/](../evidence/search/rally_review/second_scene_stills) | Seven more frames per method from ShuttleSet 30 and 21, and ShuttleSet22 51. |
| [shared_court_check/](../evidence/search/rally_review/shared_court_check) | Two frames from ShuttleSet22 43 where all three methods fit the wrong court. The original-player-check result has roughly the right orientation but sits one line inward. |
| [added_rally_courts/](../evidence/search/rally_review/added_rally_courts) | All 13 rallies gaining a court under broader search. Manual review found three clear courts, two fragments and eight frames with no court. |
| [easy_court_misses/](../evidence/search/rally_review/easy_court_misses) | The actual frames used for detection in three apparently easy missed-court cases. Two show transitions or a court sliver; the third failed the player check. |

Each added-rally image shows a different frame from the one the detector used.
The detection-frame images show what it saw in the three clear-court cases. The complete
[ShuttleSet22 27 scene](../evidence/search/rally_review/scene_clips/ss22_27_scene0027.mp4) and
[ShuttleSet22 44 scene](../evidence/search/rally_review/scene_clips/ss22_44_scene0349.mp4) show the court appearing
after the sampled moment. Those two longer clips are stored separately from
Git, as described in the [input guide](../evidence/README.md#larger-optional-inputs).

The [original 86-video images](../evidence/baseline/courts) show the courts before
the sharing repair. The examples in this folder were chosen to investigate
changes, so they do not measure the frequency of errors across a whole broadcast.

The [reproduction guide](../tools/README.md#search-trial-images-and-clips)
contains the commands for these images and clips.

## Files

- [Selection comparison](#choosing-courts-by-score-instead-of-player-checks): detailed score-first
  counts and its 20-image review.
- [Search tables](../evidence/search/search_results/): tables comparing all three methods, including processing time.
- [Review images and clips](#images-and-clips-from-the-search-trial): the courts and missed scenes
  checked by hand.
- [Trial runner](../tools/trial_run_video.py): the experiment runner. It records two choices from each
  search so the alternatives can reuse the same work.
- [Saved trial predictions](../evidence/inputs/player_tiebreak_results): saved
  predictions, choices and timings used by the comparison scripts.

The saved files use these original job names:

| Stage | Folder | Method |
| --- | --- | --- |
| A | `videos/` | Original player checks |
| A | `trial_videos/` | Score-first |
| B | `trial_videos/` | Broader search |
| B | `videos/` | Extra comparison: broader search with the original final player veto |

The [reproduction guide](../tools/README.md#compare-the-three-search-methods) rebuilds the
tables from these saved files.
