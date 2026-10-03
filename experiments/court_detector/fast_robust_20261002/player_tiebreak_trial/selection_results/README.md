# Both choices recover the main view; the reviewed courts outside rallies are wrong

**Both choices recover the eight representative main-view courts, but the
review images do not establish which choice is better overall.** Choosing
individual courts by score repairs some fits and damages one good fit that
sharing then rescues. On 3 October, the user judged all 16 court-bearing review
images incorrect: eight from each choice. The four missing-court examples were
reasonable abstentions. This supersedes the earlier recommendation to prefer
the court-sharing patch across the extracts. The broader-search comparison is reported
separately when complete.

## What was compared

The **court-sharing patch** chooses the shared court first. A failed transfer affects
only the receiving scene, which keeps its own court. Individual court selection
still uses the original player-position checks.

**Score-first selection** uses the same search and court-sharing patch. It chooses the
highest-scoring candidate and uses player positions only to break exact ties.
It also removes the player-position veto after refining the chosen court.

Both choices ran from the same search records at commit `928ed398`. The sample
contains videos 11, 21, 30 and 36 from ShuttleSet, and videos 27, 43, 44 and 51
from ShuttleSet22.
These eight purpose-selected videos contain 4,615 scenes and 668 usable labelled
rallies. They provide a screening comparison, not an estimate for all 86 videos.

Errors below use the supplied default-camera court annotation at 1280 × 720.
The representative error averages the four corner distances. The scene checks
require **every corner** to meet the stated tolerance.

For each video, select the court-sharing-patched run's view group containing the most
labelled rally frames. Compare the same 726 scenes from those eight groups
across both choices. Of these scenes, 610 overlap labelled rallies.
The groups identify likely camera views; they are detector outputs, not
independent labels. Other views cannot be judged against the default-camera
annotation.

## Results

| Measure | Patched court sharing | Patched court sharing plus score-first selection |
| --- | ---: | ---: |
| Representative court within 6 px mean corner error | 8/8 videos | 8/8 videos |
| Individual main-view court within 20 px at every corner, before sharing | 706/726 scenes | 721/726 scenes |
| Main-view court within 10 px at every corner, after sharing | 726/726 scenes | 726/726 scenes |
| Scenes with any detected court, across all views | 1,263/4,615 | 1,317/4,615 |
| Mean per-video fraction of rally time with a detected court | 97.21% | 97.54% |

The representative corners are identical between the two choices in all eight
videos. Both choose the same source court for each main-view group. Score-first
selection adds four main-group members, including one with 75 rally frames;
these additions are outside the fixed 726-scene comparison above.

Before sharing, score-first selection repairs **16 individual main-view courts
and damages one**, using the 20 px threshold. Four repairs and the one regression
overlap labelled rallies. The remaining 12 repairs occur outside rallies.

The regression is ShuttleSet 11, scene 154, with 189 labelled rally frames.
The largest corner error grows from **9.4 to 1,418.6 px**. The oversized candidate
scores 0.807, above the accurate candidate's 0.687. Sharing subsequently restores
the court to within 2.3 px at every corner. Final results alone hide this mistake.

There are seven exact top-score ties among 1,660 recorded choices. Player support
changes none of those choices, so this run provides no evidence that the
tie-breaking rule improves results.

## More detections do not establish better coverage of usable courts

Score-first selection adds 151 detected scenes and removes 97. Of these changes,
**142 additions and 93 removals occur outside labelled rallies**. Within rally
intervals, it adds 508 frames with a court and removes 15. Only 75 added frames
belong to the newly recovered main-group scene. The other added detections need
view-specific judgement before they count as an accuracy gain.

Two newly detected courts become sources for sharing in other view groups.
One serves five scenes outside rallies. The other serves 14 scenes, including
two with 171 rally frames. The user judged both source courts wrong; the
receiving frames still need their own visual judgement. After sharing, no
regression crosses the 10 px threshold in the fixed main-view groups.

The [20 review PNGs](review_frames/) contain:

- **01–08, `court_from_score_first_selection`:** newly detected courts, with
  the score-first result outlined
- **09–16, `court_from_court_sharing_patched_dropped_by_score_first`:** courts the
  court-sharing-patched run retained, with that result outlined
- **17–20, `no_court_in_either_run_during_rally`:** missed courts, with no overlay;
  each sampled frame is inside a labelled rally

The first 16 contain one addition and one removal per video. Two additions were
selected because they supply shared courts; the other 14 were drawn with seed
20261002. The four misses cover three failure reasons across three videos.
This sample illustrates changes, not their frequency. The user judged all
16 predicted courts incorrect and all four abstentions reasonable. All 16
court-bearing samples are outside labelled rallies. Score-first selection
therefore introduces eight reviewed false courts and removes eight others;
these selected examples do not estimate either method's overall false-court rate.
Full-resolution overlays use dashed red 1 px lines along the
boundaries of the painted court stripes.

## What to do next

The **search without player rejection** trial is now complete. It allows more
candidates through the search, then applies score-first selection. See the
[completed comparison](../search_results/README.md) before selecting a lasting
policy.

The useful statistical check is the paired count of repaired and damaged courts,
before and after sharing. It has exposed a failure hidden by the headline
results. Significance tests on these eight selected videos would not settle
the unlabelled-view question. Shared courts also make scene errors dependent;
726 scenes are not 726 independent tests of the selection rule.

Restoring the retained individual fits and rerunning sharing remains a candidate
repair for the main-view sharing damage. The main-view equality and reviewed
images do not establish that it is as good as or better than score-first
selection overall. Assess the broader-search results before choosing a repair.
Any repaired outputs need the full-corpus checks, with originals preserved.
Reliable courts are needed before deriving player positions and distances in
court metres. No corpus repair has been launched from this comparison.

## Evidence and checks

[Per-video results](per_video.csv.gz), [paired scenes](paired_scenes.csv.gz) and
[review sample records](review_samples.csv.gz) retain the numerical evidence.
The per-video table includes the earlier original extraction for context; it
is a separate run, not a paired control for selection. Only the court-sharing-patched and
score-first outputs use the same search records.

The [saved inputs](../../inputs/README.md) retain stage-A predictions, choices
and timings. `videos/` contains the court-sharing-patched results and
`trial_videos/` contains score-first results. The maintained
[comparison script](../search_results/compare_search.py) now compares these
methods alongside the broader-search trial. It reuses the same evaluator and
fixed main-view population; the [reproduction guide](../../reproduce.md) gives
the command.

All eight jobs completed with exit 0. Scene partitions match, saved individual
fits match the choice records, and no scene or group errors were recorded.
Paired jobs took 56 minutes to 3 hours 36 minutes per video; these times include
both choices and cannot measure their standalone cost. Analysis, frame retrieval,
PNG rendering and compression checks passed with exit 0.

One independent Opus 5.5 xhigh review reproduced the choices and main results.
Its extra analysis chose scenes using their measured errors; those counts are
excluded from the fixed-population comparison here. A claimed sub-pixel bound
was checked separately: the largest corner shift between shared results is
0.854 px after cyclic corner alignment. Representative corners remain identical.
No additional source images were inspected by the reviewing agents.
