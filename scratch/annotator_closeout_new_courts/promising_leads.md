# Promising leads: what remains to investigate

First find out why the model adds hits after rallies end. Then check court coverage, player-side errors and serves. Closed ideas are in [last_followups.md](last_followups.md). The [earlier backlog](../annotator_wrapup_evaluation/promising_leads.md) holds the label-sweep and release-inventory items that this work did not change.

**Contents**  
[Priority map](#priority-map)  
[1. Can the model tell when a rally has ended?](#1-can-the-model-tell-when-a-rally-has-ended)  
[2. Why does the new detector decline scenes with play?](#2-why-does-the-new-detector-decline-scenes-with-play)  
[3. Why do player sides fail in videos 17 and 42?](#3-why-do-player-sides-fail-in-videos-17-and-42)  
[4. Why are serves weaker?](#4-why-are-serves-weaker)  
[5. What still fails with good inputs?](#5-what-still-fails-with-good-inputs)  
[Carried over unchanged](#carried-over-unchanged)

## Priority map

| Question | Why now | Next check |
|---|---|---|
| Can the model tell when a rally has ended? | 233 rallies fail only on one false hit after the rally ends | Feed a visible end cue into the hit choice; the landing search now starts after the false hit |
| Why does the detector decline scenes with play? | 1,620 of 2,984 misses are in rejected scenes; outside video 53 that count barely moved | Place the 3-second foot check on active play; reject courts far larger than the frame |
| Why do player sides fail in videos 17 and 42? | Most wrong sides follow a missed hit; video 42's net position comes from a broken court | Take the net position from a checked court; check camera changes inside video 17's 40-minute scene |
| Why are serves weaker? | Serve timing matches fell from 2,766 to 2,722 | Directly check a small sample of accepted-scene serve misses |
| What remains with good inputs? | 1,335 misses already have an accepted court and both players | Diagnose after the four above |

## 1. Can the model tell when a rally has ended?

**233 rallies** fail only because the model adds one hit after the last labelled hit. Every labelled hit in those rallies is matched with the right player. That is 14.7% of failed rallies and the largest single group of near-misses. Another 29 rallies have the same late hit plus other extras.

The sampled added hits are false. In all 16 sampled rallies, the shuttle lands untouched or a player handles it after the rally. The official rows record an ending that rules out another touch for 230 of the 233. The false hit always goes to the receiver, 10–60 frames after the last real hit. Shots into the net and misjudged landings produce it most often.

The [detector audit](detector_audit.md#the-false-hit-after-the-rally-ends) traced the path. The contact model scores the false hit at 0.9 or above in 231 of the 233 rallies, and the sequence chooser keeps it. The landing search then starts after it, so a landing found in 67 of these rallies always comes too late to matter.

The earlier follow-up rejected deleting events “after the last label”. That rule needs the labels, so it cannot run on new footage. The fix needs a cue the model can see before it accepts the last hit:

- the shuttle track settling on the floor or stopping near the net;
- the receiver catching or collecting the shuttle rather than swinging;
- players relaxing or walking after the last real hit.

Test any rule on the development videos first. Count both sides: rallies rescued, and real final hits removed. A real final touch, such as a failed return into the net, must survive.

## 2. Why does the new detector decline scenes with play?

The new detector fixed the two earlier geometry failures. It did not raise coverage much outside video 53:

- 919 missed hits are in scenes where no candidate court passed the detector's checks;
- 457 are in scenes where a court was found but the two-player vote failed, usually by a wide margin;
- 588 labelled hits are in scenes the old courts accepted and the new courts reject; all 588 are now missed.

The newly rejected scenes are the best starting point. The old pipeline already found a workable court there, so they are known to contain one. Video 38 has the most (73 hits), then 11 (44), 23 and 54 (36 each) and 36 (32).

The [detector audit](detector_audit.md#the-court-stage) found two causes:

- **888 misses** sit in 145 scenes dropped before any court search. The detector checks players only in the three seconds around each scene's middle frame. In these scenes that window overlaps a labelled rally 12% of the time, against 93% in accepted scenes. The same scenes hold 362 of the 588 newly rejected hits.
- **286 misses** sit in vote-failed scenes whose court reaches well outside the frame or is over twice the video's usual size. The detector has no check against that.

Try placing the foot window on active play, or several windows per scene, before relaxing any player check. The [earlier search trial](../../experiments/court_detector/comparisons/search.md) found that relaxing them adds false courts. Add a size and position check against the frame and the video's other courts.

[What the footage shows](court_checks.md#what-the-footage-shows) records the sampled rejected scenes.

After a fix, rerun the same evaluation. Compare fully correct rallies, contact P/R/F1, player-aware P/R/F1, serves and the selected review queue. Sample newly accepted scenes from both sides: rescued live play and footage that should still be rejected.

## 3. Why do player sides fail in videos 17 and 42?

Most wrong sides follow from a missed hit. The model alternates sides within a rally, so a hit missed inside the rally flips every hit on one side of the gap. That accounts for 837 of the 1,043 wrong sides, in rallies that already fail on the miss. The [detector audit](detector_audit.md#wrong-sides) has the breakdown.

Video 17 still has 209 matched hits with a wrong or missing side, and 208 of them are in one scene record covering its first 40 minutes. The checked labels are right: in five video 17 rallies where nearly every side was wrong, the footage shows the labelled player hitting. One court and one net position serve that whole span, and that court puts the far baseline about 150 px higher than the video's other courts. Check whether the camera view changes inside the scene.

Video 42 has 99, spread across many scenes. It is also the largest loss against the fresh old-court refit: 28 → 15 complete rallies, with nine of the 14 losses involving side errors. Its video-wide net position comes from scene 495, whose court reaches about 28,000 px off the frame. The resulting band sits about 190 px below the net in the scenes with hits, so near players read as far. Take the net position from a court that passes a size check, or use each scene's own position, then refit and compare.

## 4. Why are serves weaker?

Serve timing matches fell from 2,766 to 2,722. In accepted scenes the new model misses 10.4% of serves, against 9.0% before. Serve timing is also loose: at ±5 frames, 29.9% of serves miss.

Original ShuttleSet is weaker again: only 384 of 668 validation serves have the right timing and player. [#77](https://github.com/ahalp90/badminton_cv_annotator/issues/77) already records first-stroke labels that do not mark the actual serve contact there.

Check a small sample of accepted-scene serve misses directly. Split label timing problems from detector misses before changing either.

## 5. What still fails with good inputs?

**1,335** missed labels have an accepted court and both players available at the labelled frame, up from 1,163. Rescued scenes bring 89 misses of their own, and scenes both courts accepted have 71 more misses than before.

The plan from the earlier backlog still applies. Take a small directly checked sample across serves, middle contacts and final contacts. Separate:

- real detector misses;
- wrong or ambiguous labels;
- hits that never enter the candidate pool;
- available candidates chosen badly by the final sequence;
- boundary or scene-transition cases.

Then pick the intervention that matches the dominant residual error. Do not fit another model just because 1,335 is a large number.

## Carried over unchanged

- **Bad-label sweep ([#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147)).** Video 15 stays excluded. No second wrong-rally video exists: every game now matches at least 72.9% of its labelled hits. They found one new timing error: four video 12 rallies with official timestamps about 15 frames off. Consider correcting or dropping those four rallies in the release.
- **Full release inventory ([#133](https://github.com/ahalp90/badminton_cv_annotator/issues/133)).** Evaluation still covers 46 ShuttleSet22 videos plus 40 original ShuttleSet videos.
- **Review cutoff.** The historical cutoff was carried over to the new confidence model. Its queue precision is similar, 83.5% against 84.4%, so retuning is low priority.
- **Noise-aware training.** Still deferred until a small verified contact set exists.
