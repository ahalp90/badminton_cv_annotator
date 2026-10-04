# Detector audit: why the remaining failures happen

**The audit traced the three largest failures to specific code paths.** Each was found by a read-only audit of the current code and then rechecked against the saved run.

- **The court stage judges each scene on one 3-second sample, and in rejected scenes that sample mostly falls between rallies.** 888 missed hits sit in 145 scenes dropped before any court search. Another 286 sit in scenes where the detector accepted a court far larger than the picture, which the two-player vote then rejected.
- **Nothing stops a false hit after the rally ends.** The contact model scores it at 0.9 or above in 231 of the 233 rallies, the sequence chooser keeps it, and the landing search starts after it.
- **Most wrong sides follow from a missed hit.** Sides alternate within a rally, so one missed hit flips every hit on one side of it: 837 of the 1,043 wrong sides. Video 42 also takes its net position from a broken court.

**Contents**  
[The court stage](#the-court-stage)  
[The false hit after the rally ends](#the-false-hit-after-the-rally-ends)  
[Wrong sides](#wrong-sides)  
[Corrections to the audit](#corrections-to-the-audit)  
[Method and evidence](#method-and-evidence)

## The court stage

### One 3-second window decides each scene

The detector checks for players at 31 frames, ten per second, around each scene's middle frame (`src/court_detector/run_video.py:177`, `feet.py:45`). Every one of those frames must show a standing person in the picture, and at least half must show two (`feet.py:173–181`). Otherwise the detector stops before searching for a court (`detect.py:265–268`).

The court search applies the same rule to each candidate court (`measurements.py:221–231`), so the early stop only saves time. The evaluated courts were made with this rule: the release records name commit `17a50b57`, which has the same stop, and the rejected scenes record only the first two detector stages.

| Scenes | Count | Median length | Foot window overlaps a labelled rally |
|---|---:|---:|---:|
| Accepted, with labelled hits | 3,160 | 386 frames | 92.9% |
| Stopped before any court search, with missed hits | 145 | 643 frames | 12.4% |

In the rejected scenes, the window mostly samples the break between rallies rather than play. These scenes hold **888 of the 1,620** rejected-scene misses. They also hold 362 of the 588 hits that the new courts newly reject.

Some windows that miss a labelled rally may still hold play: cleaning dropped 524 of the 3,851 official rallies. That cannot close a gap of 12% against 93%.

The [earlier search trial](../../experiments/court_detector/comparisons/search.md) relaxed the player checks and found false courts. This cause is different. The checks can stay; they are applied to the wrong three seconds. Placing the window on active play, or trying several windows, keeps the checks intact.

### Oversized courts pass the detector, then fail the vote

The detector's final check accepts a court when it is convex, the implied camera is plausible and the players stand inside it, with a 15% margin (`measurements.py:221–231`, `players.py:17`). Nothing compares the court with the frame or with the video's other courts. A court much larger than the picture holds the players easily, so it passes.

The annotator's two-player vote then fails these scenes by a wide margin:

| Vote-failed scenes | Scenes | Missed hits | Median vote |
|---|---:|---:|---:|
| Court reaches more than 10% outside the frame, or is over twice the video's usual area | 48 | **286** | 17% |
| Other courts | 28 | 171 | 39% |

The vote needs exactly two people inside the court, so umpires, line judges and spectators inside an oversized court are the likely cause. The pose arrays needed to confirm this are not in the saved results.

The footage confirms one case. Video 52's checked frame shows all four court corners in view, but the detector's court runs off the frame at 2.4 times the usual area. The other two checked vote failures, in videos 26 and 38, have ordinary courts and votes just under 50%.

### Scenes too short to check

98 misses sit in 84 scenes of 90 frames or fewer. The 31-frame foot window needs 91 frames at 30 fps, so these scenes are never searched (`run_video.py:184–189`).

## The false hit after the rally ends

[233 rallies](video_checks.md#extra-hit-after-the-last-label) fail only on one extra hit after the last real one. The audit followed that hit through three stages:

| Stage | What happens to the false hit | Rallies |
|---|---|---:|
| Contact model (`contacts/model.py:125–133`) | Scored at or above the 0.9 cutoff for keeping a hit; median 0.94 | 231 of 233 |
| Sequence chooser (`sequence/choices.py:24–56`) | Could delete it, but keeps it as the rally's last hit | 233 of 233 |
| Landing search (`outcomes/point_winner.py:231`) | Starts after the last kept hit, so any landing it finds comes after the false hit | 67 found, all after |

The rally's end is detected only after the hits are fixed, and nothing feeds it back. A visible end cue, such as a landing, a settled shuttle or a catch, needs to reach the hit choice before the last hit is accepted. That differs from the rejected [endpoint deletion](../annotator_wrapup_evaluation/last_followups.md#endpoint-deletion), which removed hits by their position after the last label.

One end cue is already computed. The shuttle track's guard flags 110 of these false hits on their own frame, and the landing check already distrusts a flagged last hit. The [shuttle guard addendum](shuttle_guard_addendum.md) measures it.

The false hit also becomes the rally's recorded striker in all 233. It does not cost the predicted point winner, which is read from who serves next. That winner agrees with the official rows in 80.1% of these rallies, against 74.2% of fully correct ones.

## Wrong sides

### One missed hit flips a run of sides

The model gives each rally's hits alternating sides. It picks whichever of the two alternating patterns agrees with more of its raw guesses (`sequence/sides.py:17–42`). When a hit inside the rally is missed, the hits before and after the gap need opposite patterns, so one run is wrong whichever pattern wins.

| Where the rally's misses fall | Rallies | Matched hits | Wrong sides | Share |
|---|---:|---:|---:|---:|
| No miss, no extra | 1,841 | 20,276 | 74 | 0.4% |
| First or last hit missed | 551 | 4,957 | 52 | 1.0% |
| Extra hit only | 326 | 3,354 | 80 | 2.4% |
| A hit inside the rally missed | 609 | 5,613 | **837** | 14.9% |

117 rallies with exactly one miss and no extra have wrong sides. In 115 of them, every hit on one side of the gap is wrong.

These rallies already fail on the missed hit, so the flipped sides rarely cost a rally. They do lower the per-hit player score. They would also mislabel hits in any partly correct rally used downstream.

### Video 42 takes its net position from a broken court

The selected model splits near and far with one net position per video, taken from the longest accepted scene (`courts/evidence.py:588–594`). A foot above that band reads as far (`point_winner.py:129–130`).

In video 42 the longest accepted scene is scene 495. Its court reaches x = 27,991 px, yet it passed the vote at 52.8%. Its net band sits at y = 880–936 px, while the scenes holding the hits have it at 693–727 px. A near player standing between those heights reads as far.

Video 42's wrong sides run 66 near-to-far against 26 far-to-near. In its rallies with no miss or extra, 15 of 166 hits get the wrong side (9.0%), against 0.4% overall.

### Video 17: one 40-minute scene with an odd court

Video 17's first scene runs 73,167 frames. Its court puts the far baseline at y ≈ 270 px, against ≈ 418 px in the median of the video's 60 other accepted courts. As the longest scene, it also supplies the video-wide net position. 208 of the video's 209 wrong sides sit in this scene.

The errors arrive through missed hits: 203 of the 209 are in rallies with a miss, and its clean rallies have 6 wrong sides in 357. The odd court suggests it was fitted to a different camera view from most of the scene. Frames from across scene 0 would show whether the camera changes.

## Corrections to the audit

The rechecks changed four of the audit's statements:

- It put all 21 vote-failed scenes with ordinary courts at 42–50%. They range from 5% to 50%, with a median of 42%.
- Its oversized-court count, 339, included seven scenes flagged only for a small area. Four are one recurring wide view in video 38. The table above uses the stricter count, 286.
- It said nothing downstream can remove a trailing hit. The sequence chooser can; it chose not to in all 233.
- It predicted the false hit would cost the point winner. It does not: 80.1% against 74.2%.

## Method and evidence

The audit ran once, read-only, on `main` at `d4994bad`. Since the evaluated run, the annotator has changed only in evaluation files and progress messages. The court gate matches the release commit. Line numbers match `main` at `8b65a268`. The brief named the three failures and the code paths. Each claim needed a call chain with quoted lines and a prediction written before its data check. The official rally endings and old-court decisions were held back to score its untested predictions.

Every finding above was then rechecked: the quoted lines against the code, the counts with `scripts/audit_checks.py`, and the untested predictions against the held-back data.

```bash
PYTHONPATH=src python scratch/annotator_closeout_new_courts/scripts/audit_checks.py
```

The script reads `results/` and the Git-ignored `raw/run/` pull. Shuttle tracks, pose arrays and footage were not available to the audit.

- `results/contexts.csv.gz` and `results/contacts.csv.gz` — scene and match state at each labelled hit
- `results/tail_rallies.csv.gz` — the 233 rallies that fail only on the false hit
- `raw/run/courts/videos/` — the court release, with the detector stages each scene ran (local, ignored)
- `raw/run/output/` — saved sequences, landings and court evidence (local, ignored)
