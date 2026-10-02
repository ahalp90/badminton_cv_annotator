# Broader search preserves the main-view result and adds mostly unjudged courts

**All three methods recover the measured main-view courts in the eight-video
trial. Broader search adds many predictions elsewhere, but the data do not
establish that those predictions are useful.** It retains six of the courts
the user rejected in the earlier image review. Review the full rally clips
before choosing a method for a full-corpus run.

## What was compared

Does removing player-based rejection from court search improve on changing
only sharing or final selection?

| Method | Search and selection |
| --- | --- |
| Sharing fix | Original search and player checks; failed sharing affects only the receiving scene |
| Score-first selection | Original search; final choice uses court score, with player support only for exact ties |
| Search without player rejection | Broader search plus score-first selection; geometry, camera and search limits remain |

All three include the sharing fix. The sample contains ShuttleSet 11, 21, 30
and 36, and ShuttleSet22 27, 43, 44 and 51. These purpose-selected videos have
4,615 scenes and 668 usable rallies. They are a screening sample, not an
estimate of performance over the complete 86-video corpus.

The main-view comparison keeps the same 726 scenes selected from the sharing-fix
run's groups. Of these, 610 overlap labelled rallies. A detector group is a
proxy for a camera view, not an independent label. Corners are compared with
the default homography at 1280 × 720. Scene thresholds apply to the **worst
corner**; representative-video errors use the **mean of four corners**.

## Main views remain accurate against the supplied reference

| Measure | Sharing fix | Score-first selection | Search without player rejection |
| --- | ---: | ---: | ---: |
| Representative court within 6 px mean error | 8/8 videos | 8/8 | 8/8 |
| Main-view court within 10 px at every corner, after sharing | 726/726 scenes | 726/726 | 726/726 |
| Main-view court within 20 px at every corner, before sharing | 706/726 scenes | 721/726 | 723/726 |
| Scenes with any final court, all views | 1,263/4,615 | 1,317/4,615 | 1,900/4,615 |
| Mean per-video rally-time prediction coverage | 97.21% | 97.54% | 98.16% |

Compared with score-first selection, broader search repairs two individual
main-view fits at the 20 px threshold and damages none. One repair overlaps
a rally; the other does not. At 10 px, it repairs three and damages one.
Sharing already makes every court in this fixed main-view population meet
10 px, so these individual changes do not improve the final pass count.

The score-first selection regression in ShuttleSet 11, scene 154, remains
relevant: a bad individual court can be hidden by a successful shared fit.
The [selection report](../selection_results/README.md) records that case.
Agreement in the main view does not establish performance during camera changes.

## More detections need visual judgement

Compared with score-first selection, broader search adds 669 scenes with
courts and removes 86. Of the additions, **620 are outside labelled rallies**;
49 overlap rallies outside the fixed main-view group. It removes five courts
from rally-overlapping scenes. Within rally intervals, additions cover 1,256
frames and removals cover 69 frames. Frames are counted at their native rate;
the per-video coverage table above accounts for different video lengths.

These counts show where the methods add or remove predictions. They do not
show whether those predictions are correct.
The static homography cannot judge alternate camera views. Some scenes outside
the fixed group may still show the default camera, because detector grouping
is imperfect. Abstaining can be the right result.

The user judged all 16 court-bearing images in the earlier review wrong and
all four abstentions reasonable. On those same 16 scenes, broader search has:

- six courts identical to the rejected geometry;
- seven scenes with no court;
- three changed courts whose correctness has not been judged.

All four reviewed abstentions remain abstentions. These counts describe the
current outputs at the reviewed scenes; they are not seven newly repaired
errors relative to score-first selection. The images were selected to show
changes, so they cannot estimate either method's overall false-court rate.
The [scene table](reviewed_scenes.csv.gz) records the exact comparisons.

One rejected court is also reused during rallies. In ShuttleSet22
43, both score-first methods use scene 483 (review image 02) as the source for
scenes 89 and 176, covering 171 labelled rally frames. The source court remains
identical to the one the user rejected. The receiving scenes need their own
visual judgement; the source review alone does not label their geometry.
Their [comparison PNGs](../rally_review/shared_court_check/) show all three
methods at the same two rally frames.

## Runtime and next decision

The broader-search paired jobs took **1.42–1.65 times as long** as the earlier
paired jobs on the same hosts: roughly 1.55–5.10 hours per video, versus
0.94–3.60 hours. Each job includes two selection passes and shared search work.
These timings do not measure the standalone cost of any one method.

Summing the eight video jobs gives **16.95 hours with the original search and
25.17 hours with broader search: 48.5% longer**. Videos ran concurrently, so
these totals are not elapsed waiting time. The logs do not separate end-to-end
runtime for the sharing fix and score-first selection. The extra score-first
choice/refit step took 1.4–5.1 minutes per video in the original-search trial.
That timer excludes sharing and reuses the baseline refit when possible;
it is not a standalone speed comparison.

The [nine full rally clips](../rally_review/clips/) show three rallies from
different videos under all three methods. They were selected by scene-cut
count, not by measured court errors. The [review notes](../rally_review/README.md)
explain the intervals and link the representative PNG folders. Check whether
each method follows camera changes and abstains when a usable court cannot
be fitted. The project needs reliable courts to measure player positions and
distances in court metres.

There is no demonstrated overall winner yet. The sharing fix addresses a
known main-view failure, but the earlier images do not justify preferring it
across all views. Broader search's extra detections do not justify preferring
that method either. No full-corpus repair has been launched. The
[86-video baseline and rally-view statistics](../../rally_views/README.md)
remain measurements of the original extracts.

## Evidence and reproduction

Both hosts completed both stages with exit 0. All eight video outputs have
matching scene partitions and no recorded scene or group failures. The
comparison checks saved individual fits against the choice records.

The full comparison and original-corpus evaluation reproduced their tables
exactly. Ruff, whole-project Pyrefly, an explicit type check of the comparison
script, rendering, media checks and report-link checks passed with exit 0.
An independent Opus 5.5 xhigh review also reproduced the measurements and
checked a randomly selected rally. No dedicated tests or detector reruns were
added for this evaluation.

- [Per-video results](per_video.csv.gz) and [paired scenes](paired_scenes.csv.gz)
- [Main-view transitions](main_view_transitions.csv.gz) and [detection changes](detection_changes.csv.gz)
- [View groups](view_groups.csv.gz), [court sources](court_paths.csv.gz) and [run checks](audit.csv.gz)
- [Reproduction script](compare_search.py)

Run the comparison from the repository root with `PYTHONPATH=src:.`:

```bash
PYTHONPATH=src:. ~/.venvs/badminton-cicd/bin/python \
  experiments/court_detector/fast_robust_20261002/player_tiebreak_trial/search_results/compare_search.py
```

Inputs are the downloaded directories under
`local_scratch/court_evaluation/player_tiebreak_results/{carmack,bourbaki}/`.
The three methods are `stage_a/videos`, `stage_a/trial_videos` and
`stage_b/trial_videos`, respectively. The tables also retain the original
extracts and the auxiliary `stage_b/videos` result. That auxiliary result keeps
the final player veto and is not the broader-search method compared above.

The eight selected videos and their shared scene predictions do not support
treating thousands of scenes as independent trials. Paired counts are the
useful statistical comparison here; no significance claim is made.
