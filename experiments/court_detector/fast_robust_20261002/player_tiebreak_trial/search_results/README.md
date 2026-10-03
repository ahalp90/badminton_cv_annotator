# Court detection: full-corpus repair and search trials

**Use patched court sharing as the default.** Across the 86-video corpus, it raises default-view agreement within 10 px from 74 to 85 video representatives and from 5,228 to 6,171 rally representatives. No representative or rally crosses from within 10 px to outside it. The eight-video trials give no compelling final main-view benefit from score-first selection or broader search. Broader search's paired jobs take 48.5% longer.

The [released court dataset](../../court_sharing_patched/README.md) retains all 44,810 scenes from 40 ShuttleSet and 46 ShuttleSet22 videos. The replay changes 4,237 existing final courts and recovers 108 courtless scenes. It preserves every saved individual fit. Final corner lists use the viewer-facing far-baseline-first convention; 190 lists were turned by 180° without changing their geometry. These courts support downstream player positions and distances in court metres, subject to the view and sampling limits below.

## What was compared

There are two comparisons with different populations:

- **Full86:** original extraction against patched sharing on the same 86 videos. The replay uses saved individual fits and includes the later patch that lets matching courtless scenes receive an established shared court.
- **Trial8:** fresh searches on ShuttleSet 11, 21, 30 and 36, and ShuttleSet22 27, 43, 44 and 51. These purpose-selected videos contain 4,615 scenes and 668 usable rallies. Their three methods predate the courtless-receiver patch.

Trial court sharing uses the original search and player checks. Score-first uses the same search but selects by court score, with player support only for exact ties. Broader search also removes player rejection; geometry, camera and search limits remain. All three use the earlier sharing patch, which isolates a sharing failure to its receiving scene.

Compare columns within each population. The full86 replay and trial8 searches do not measure the same change, and their values cannot establish an effect between populations.

## How to read the results

All corner errors use the supplied default-camera reference at 1280 × 720. A video's representative is the medoid of its main view group: the court closest to the group's other courts. The main group occupies the most labelled rally frames. Representative error averages the four corner distances.

The fixed main-view scenes hold the comparison population constant: 6,748 original main-view scenes in full86 and 726 scenes in trial8. The final main-view measure requires the **worst corner** to be within 10 px. The before-sharing measure instead uses 20 px on each scene's own fit.

A rally representative comes from the view group occupying the most time in that rally. Its 10 px measure uses **mean corner error**, with missing courts counted as misses. It measures agreement with the default-camera reference. It cannot judge the accuracy of a different camera view. Rally-time coverage counts labelled frames whose scene has any court; it measures presence, including wrong courts.

Full86 includes 6,833 valid rallies. Another 377 were excluded because their contact-frame sequence did not strictly increase.

Means give each video equal weight. Pooled counts give each scene or rally equal weight. These are descriptive results for the evaluated videos. Paired differences compare methods within the same video.

| Measure | Original86 | Patched86 | Trial8 court sharing | Trial8 score-first | Trial8 broader search |
| --- | ---: | ---: | ---: | ---: | ---: |
| Representative mean error, px | 45.260 | 2.855 | 2.84 | 2.84 | 2.84 |
| Representative error median / p90, px | 2.73 / 91.15 | 2.58 / 4.17 | 2.60 / 4.15 | 2.60 / 4.15 | 2.60 / 4.15 |
| Representatives within 10 px | 74/86 | 85/86 | 8/8 | 8/8 | 8/8 |
| Final main-view scenes within 10 px, pooled | 5,400/6,748 | 6,522/6,748 | 726/726 | 726/726 | 726/726 |
| Final main-view percentage, mean across videos | 84.02% | 97.64% | 100% | 100% | 100% |
| Own main-view fits within 20 px before sharing, pooled | 6,673/6,748 | 6,673/6,748 | 706/726 | 721/726 | 723/726 |
| Rally-time coverage, mean across videos | 96.81% | 96.87% | 97.2% | 97.5% | 98.2% |
| Rally views with a court, pooled | 6,462/6,833 | 6,474/6,833 | 638/668 | 640/668 | 650/668 |
| Rally views within 10 px, pooled | 5,228/6,833 | 6,171/6,833 | 625/668 | 626/668 | 627/668 |

## Full86: substantial repair, concentrated in badly shared courts

Representative mean error falls by 42.405 px, from 45.260 to 2.855 px. The median moves only from 2.73 to 2.58 px, while the 90th percentile falls from 91.15 to 4.17 px. The main gain is repairing large errors rather than refining already good courts. Eleven additional video representatives meet 10 px. The remaining miss is ShuttleSet 32 at 10.814 px.

The fixed main-view count rises by 1,122 scenes, from 5,400 to 6,522 of 6,748. The equal-video improvement is 13.62 percentage points. Before-sharing fits are unchanged, so this result comes from the shared assignments. Representative error does rise in 24 videos, by at most 2.20 px; the 10 px representative count has no regressions.

Rally agreement gains 943 cases, from 76.51% to 90.31% of 6,833 rallies. All 943 cross into the 10 px threshold; none crosses out. There are 12 additional rally views with a court and no losses. Pooled rally-time coverage rises from 1,695,124 to 1,696,307 of 1,750,729 labelled frames. Most of the benefit therefore comes from better courts where courts already existed.

ShuttleSet gains 223 agreeing rallies, reaching 2,854/3,182. ShuttleSet22 gains 720, reaching 3,317/3,651. The final representative counts are 39/40 and 46/46, with mean errors of 3.242 and 2.519 px respectively. Rally agreement improves in 32 videos. The largest gains are ShuttleSet22 52 (+101), ShuttleSet 30 and ShuttleSet22 44 (+98 each), ShuttleSet22 43 (+92) and ShuttleSet22 40 (+91).

The paired checks passed: all 86 outputs completed, all scene rows were retained, and original individual corners matched exactly before export-order normalisation. The replay reproduced 7,572 stored source scores and 178 pooled fits, with no receiver failures. This establishes that the replay preserved its inputs and completed consistently; it does not validate every final court visually.

## Trial8: better individual fits, little final gain

All three trial methods finish with all 726 fixed main-view scenes within 10 px at every corner. Court sharing and score-first choose identical representative corners in all eight videos. Broader search changes representative mean error by at most 0.04 px in a video.

Before sharing, score-first repairs 16 main-view fits at 20 px and damages one. Broader search repairs 18 and damages the same one. Their equal-video gains over trial court sharing are 1.95 and 2.28 percentage points. Sharing already repairs those differences, leaving the final main-view count unchanged.

The damaged fit is ShuttleSet 11 scene 154, covering 189 labelled rally frames. Its worst-corner error grows from 9.4 to 1,418.6 px under both score-first methods; sharing restores it to within 2.3 px. Successful sharing can conceal a bad individual fit. The [selection report](../selection_results/README.md) records this case.

Broader search adds courts to 13 rallies where trial court sharing has none, and drops one. Two additions agree within 10 px; the other 11 are 284–1,220 px from the default-camera reference. The net agreeing count rises by only two rallies, from 625 to 627 of 668. Its equal-video gain is 0.32 percentage points, compared with 0.20 for score-first. Broader search raises rally-time coverage in every trial video, by 0.95 percentage points on average.

Trial intervals resample whole videos 2,000 times. For broader search against court sharing, the 95% reweighting intervals are 0.00–0.75 percentage points for rally agreement and 0.36–1.74 for rally-time coverage. These show sensitivity to reweighting the eight chosen videos. They do not estimate performance across the corpus or new venues. Shared venues and the small, selected sample limit generalisation.

## Visual evidence and remaining misses

The user's earlier review judged all 16 selected court-bearing images wrong and four abstentions reasonable. The images were chosen to show changes, so they cannot estimate a false-court rate. Broader search retained six rejected courts unchanged, abstained in seven cases and produced three changed courts that were unjudged. The [reviewed scene table](reviewed_scenes.csv.gz) records the comparisons.

The user also judged the [shared courts in ShuttleSet22 43 scenes 89 and 176](../rally_review/shared_court_check/) wrong. Both score-first methods orient the court within the wrong band of visible lines. Patched sharing gives roughly the right orientation but insets the court by one segment in each direction. That patched court remains in the final release, covering 171 labelled rally frames. Mistaking seated judges for standing players is an unverified explanation.

The first [scene-still sample](../rally_review/scene_stills/) showed equivalent good courts for patched sharing and score-first. Broader search produced one malformed court at ShuttleSet22 43 frame 144569; the user was undecided whether it was preferable to abstention. All three methods were equivalent in the [second sample](../rally_review/second_scene_stills/), covering seven matching frames from three other videos.

For the [13 added rally courts](../rally_review/added_rally_courts/), the user judged three images to show clear courts, two to show fragments where abstention was preferred, and eight to contain no court. Those rally frames differ from the detection frames. The three clear-court misses explain two different limitations:

- ShuttleSet22 27 scene 27 samples a tiny court sliver; the usable view starts after the scene midpoint. Its selected fit is rank deficient.
- ShuttleSet22 44 scene 349 samples a transition overlay; the player-count check stops the search. A usable view starts a handful of frames later.
- ShuttleSet 36 scene 383 shows a clear court, but its refit passes the both-halves player rule in only 15/31 samples.

The later receiver patch recovers three courtless scenes in ShuttleSet 36, including scene 383. That recovered court has 3.364 px mean error and 4.470 px worst-corner error. This change is included in full86 but absent from trial8. The two ShuttleSet22 transition cases remain unresolved.

The [rally sample folders](../../court_sharing_patched/rally_samples/) contain three final PNG-only rally samples, covering 21 sub-scenes. The [complete gallery](../../court_sharing_patched/gallery.md) provides the wider visual record. The user's judgements above remain sample-specific; default-camera agreement cannot settle alternate-view accuracy.

## Runtime and decision

The full86 sharing replay takes 89,166.487 summed seconds, or 24.768 compute hours. Median time per video is 948.099 seconds, about 15.8 minutes. The concurrent launch completes 85 videos in 70 minutes 33 seconds, with up to 27 slots. ShuttleSet 36 completed separately beforehand in about 15.5 minutes; its runtime is included in the sum and median. These are incremental replay times over saved fits, rather than end-to-end extraction times.

The trial timers measure paired jobs. Stage A runs court sharing and score-first together, reusing search work. Stage B runs broader search and an auxiliary player-veto arm together. Summed job time rises from 16.95 to 25.17 hours: 48.5% longer, with 61.6 extra minutes per video on average. Concurrent execution means summed hours are not elapsed waiting time. The logs do not separately time the three algorithms.

Patched sharing gives the strongest supported improvement for the released corpus. The trial alternatives improve individual fits and court presence, but the final main-view result is unchanged and the added courts include clear failures. The evidence supports the user's preferred simple default, while leaving alternate views and transitions for further work.

Separate detector changes restore the selected player-required search policy, make scheduled scene searches independent, retain the best supported individual or combined court, and expose `--fast` and `--full` no-player search options. Those changes did not generate this dataset. No new full-corpus search is part of this comparison.

## Evidence tables

The full86 [summary](../../court_sharing_patched/comparison_summary.csv.gz), [paired effects](../../court_sharing_patched/comparison_effects.csv.gz) and [per-video metrics](../../court_sharing_patched/comparison_video_metrics.csv.gz) provide the exact values and denominators. Trial [method summaries](method_summary.csv.gz), [paired effects](paired_effects.csv.gz), [per-video metrics](video_metrics.csv.gz) and [rally views](rally_views.csv.gz) retain the detailed results and intervals.
