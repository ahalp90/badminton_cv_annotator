# The sharing fix repairs the selected videos; score-first selection needs more evidence

**Use the sharing fix as the leading option for repairing the extracts.**
Across eight selected videos, choosing individual courts by score adds no
improvement to the representative courts after sharing. It repairs some
individual fits, but also damages a good fit that sharing then rescues.
The search without player rejection is still running.

## What was compared

The **sharing fix** chooses the shared court first. A failed transfer affects
only the receiving scene, which keeps its own court. Individual court selection
still uses the original player-position checks.

**Score-first selection** uses the same search and sharing fix. It chooses the
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

For each video, select the sharing-fix run's view group containing the most
labelled rally frames. Compare the same 726 scenes from those eight groups
across both choices. Of these scenes, 610 overlap labelled rallies.
The groups identify likely camera views; they are detector outputs, not
independent labels. Other views cannot be judged against the default-camera
annotation.

## Results

| Measure | Sharing fix | Sharing fix plus score-first selection |
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
two with 171 rally frames. Their accuracy remains unresolved. No regression
crosses the 10 px threshold in the fixed main-view groups.

The [20 review PNGs](review_frames/) contain:

- **01–08, `court_from_score_first_selection`:** newly detected courts, with
  the score-first result outlined
- **09–16, `court_from_sharing_fix_dropped_by_score_first`:** courts the
  sharing-fix run retained, with that result outlined
- **17–20, `no_court_in_either_run_during_rally`:** missed courts, with no overlay;
  each sampled frame is inside a labelled rally

The first 16 contain one addition and one removal per video. Two additions were
selected because they supply shared courts; the other 14 were drawn with seed
20261002. The four misses cover three failure reasons across three videos.
This sample illustrates changes, not their frequency. The user reviewed it and
accepted the sampled skips. The other images have no recorded per-image accuracy
judgements. Full-resolution overlays use dashed red 1 px lines along the
boundaries of the painted court stripes.

## What to do next

Finish the existing **search without player rejection** trial. It allows more
candidates through the search, then applies score-first selection. Compare it
with both completed choices before selecting a lasting policy.

The useful statistical check is the paired count of repaired and damaged courts,
before and after sharing. It has exposed a failure hidden by the headline
results. Significance tests on these eight selected videos would not settle
the unlabelled-view question. Shared courts also make scene errors dependent;
726 scenes are not 726 independent tests of the selection rule.

If broader search provides no material benefit, restore the retained individual
fits and rerun sharing across the 86-video corpus. Preserve the original extracts
and repeat the paired checks over the complete repaired outputs. This would
address the known sharing damage without paying for another full search.
Reliable courts are needed before deriving player positions and distances in
court metres. No corpus repair has been launched from this comparison.

## Evidence and checks

[Per-video results](per_video.csv.gz), [paired scenes](paired_scenes.csv.gz) and
[review sample records](review_samples.csv.gz) retain the numerical evidence.
The per-video table includes the earlier original extraction for context; it
is a separate run, not a paired control for selection. Only the sharing-fix and
score-first outputs use the same search records.

Raw inputs are under
`local_scratch/court_evaluation/player_tiebreak_results/{carmack,bourbaki}/stage_a/`.
`videos/` contains the sharing-fix results, `trial_videos/` contains score-first
results, and `choices/` contains the individual fits and candidate measurements.
The analysis reuses `analyse_video`, `video_row` and `score_courts` from
`scripts/evaluate_courts_fast_robust.py`. The saved reproduction script is
`local_scratch/court_evaluation/player_tiebreak_results/analysis_a/reproduce.py`.

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
