# Next investigations

Start by testing the flagged-final-contact rule on development videos: a saved-output simulation rescues 110 rallies and breaks 12. Then address court coverage and the two identified player-geometry cases. Missed contacts in accepted scenes need a checked sample before choosing further model changes.

The order below follows these practical next steps. Serve misses are treated separately from the other accepted-scene misses because their labels and timing need particular attention.

Completed trials are in [Completed experiments](last_followups.md). The [earlier backlog](../../../scratch/annotator_wrapup_evaluation/promising_leads.md) still holds the label-sweep and release-inventory items that this work did not change.

**Contents**

[Priorities and next checks](#priorities-and-next-checks)\
[1. Can the model tell when a rally has ended?](#1-can-the-model-tell-when-a-rally-has-ended)  
[2. Why does the new detector decline scenes with play?](#2-why-does-the-new-detector-decline-scenes-with-play)  
[3. Why do player sides fail in videos 17 and 42?](#3-why-do-player-sides-fail-in-videos-17-and-42)  
[4. Why are serves weaker?](#4-why-are-serves-weaker)  
[5. What still fails with good inputs?](#5-what-still-fails-with-good-inputs)  
[Carried over unchanged](#carried-over-unchanged)

## Priorities and next checks

| Question | Why now | Most useful next check |
|---|---|---|
| Can the model tell when a rally has ended? | 233 rallies fail only on one false hit after the rally ends | Development-set test of dropping a final hit when the shuttle guard flags it; then a visible end cue in hit selection |
| Why does the detector decline scenes with play? | 1,620 of 2,984 misses are in rejected scenes; outside video 53 that count barely moved | Move the 3-second foot check onto active play; reject courts far larger than the frame |
| Why do player sides fail in videos 17 and 42? | Most wrong sides follow a missed hit; video 42's net position comes from a broken court | Net position from a checked court; camera-change check inside video 17's 40-minute scene |
| Why are serves weaker? | Serve timing matches fell from 2,766 to 2,722 | Small direct sample of accepted-scene serve misses |
| What remains with good inputs? | 1,335 misses already have an accepted court and both players | Checked sample of serves, middle contacts and final contacts |

## 1. Can the model tell when a rally has ended?

**233 rallies** fail only because the model adds one hit after the last labelled hit. Every labelled hit in those rallies is matched with the right player. That is 14.7% of failed rallies and the largest single group of near-misses. Another 29 have the same late contact plus further extras or a clip shared with a neighbouring rally.

The sampled added hits are false. In all 16 sampled rallies, the shuttle lands untouched or a player handles it after the rally. The official rows record an ending that rules out another touch for 230 of the 233. The false hit always goes to the receiver, 10–60 frames after the last real hit. Shots into the net and misjudged landings produce it most often.

The [detector audit](detector_audit.md#the-false-hit-after-the-rally-ends) traced the path. The contact model scores the false hit at 0.9 or above in 231 of the 233 rallies, and the sequence chooser keeps it. The landing search then starts after it, so a landing found in 67 of these rallies always comes too late to affect that choice.

The shuttle guard, a per-frame flag for unreliable tracked positions, already flags 110 of the 232 checkable false hits on their own frame ([shuttle guard analysis](shuttle_guard_addendum.md); [summary figure](figures/final_hit_guard.png)). It flags 2 of the real hits just before them, and 12 of 1,721 real final hits in fully correct rallies. Dropping a flagged final hit would rescue 110 rallies and break 12 on the test videos. Because that rule was found on the test results, the development videos are the appropriate first check.

The earlier follow-up rejected deleting events “after the last label”. That rule needs the labels, so it cannot run on new footage; the guard rule can. The other 122 false hits need a cue available before the final contact is accepted. Plausible cues are:

- the shuttle track settling on the floor or stopping near the net;
- the receiver catching or collecting the shuttle rather than swinging;
- players relaxing or walking after the last real hit.

Any candidate rule needs a development-set comparison of rallies rescued against real final hits removed. Failed returns into the net are an important counterexample: the final real touch must survive.

## 2. Why does the new detector decline scenes with play?

The new detector fixed the two checked geometry failures. Outside video 53, rejected-scene misses remain close to the historical count:

- 919 missed hits are in scenes where no candidate court passed the detector's checks;
- 457 are in scenes where a court was found but the two-player vote failed, usually by a wide margin;
- 588 labelled hits are in scenes the old courts accepted and the new courts reject; all 588 are now missed.

The newly rejected scenes are the cleanest comparison because the old pipeline had already accepted a court there. Video 38 has the most newly rejected labels (73 hits), then 11 (44), 23 and 54 (36 each) and 36 (32).

The [detector audit](detector_audit.md#the-court-stage) found two causes:

- **888 misses** sit in 145 scenes dropped before any court search. The detector checks players only in the three seconds around each scene's middle frame. In these scenes that window overlaps a labelled rally 12% of the time, against 93% in accepted scenes. The same scenes hold 362 of the 588 newly rejected hits.
- **286 misses** sit in vote-failed scenes whose court reaches well outside the frame or is over twice the video's usual size. The detector has no check against that.

A better next experiment is to place the foot window on active play, or to sample several windows per scene, without relaxing the player check itself. The [earlier search trial](../../court_detector/comparisons/search.md) found that relaxing those checks adds false courts. A separate size-and-position check against the frame and the video's other courts would address the oversized-court cases.

[What the footage shows](court_checks.md#what-the-footage-shows) records the sampled rejected scenes.

Evaluate a court change using fully correct rallies, contact precision/recall/F1, the same metrics with player correctness, serve recovery and the high-confidence clips. Newly accepted scenes also need a small footage check covering both rescued live play and material that should still be rejected.

## 3. Why do player sides fail in videos 17 and 42?

Most wrong sides follow from a missed hit. The model alternates sides within a rally, so a hit missed inside the rally flips every hit on one side of the gap. That accounts for 837 of the 1,043 wrong sides, in rallies that already fail on the miss. The [detector audit](detector_audit.md#wrong-sides) has the breakdown.

Video 17 still has 209 matched hits with a wrong or missing side, and 208 of them are in one scene record covering its first 40 minutes. The checked labels are right: in five video 17 rallies where nearly every side was wrong, the footage shows the labelled player hitting. One court and one net position serve that whole span, and that court puts the far baseline about 150 px higher than the video's other courts. The remaining question is whether the camera view changes inside that scene.

Video 42 has 99 wrong or missing sides spread across many scenes. It is also the largest loss against the old-court refit: 28 → 15 complete rallies, with nine of the 14 losses involving side errors. Its video-wide net position comes from scene 495, whose court reaches about 28,000 px off the frame. The resulting band sits about 190 px below the net in the scenes with hits, so near players read as far. A net position taken from a court that passes a size check, or from each scene's own court, is the obvious comparison for a refit.

## 4. Why are serves weaker?

Compared with the historical model, serve timing matches fell from 2,766 to 2,722. In accepted scenes the new model misses 10.4% of serves, against 9.0% before. Serve timing is also loose: at ±5 frames, 29.9% of serves miss.

Original ShuttleSet is weaker again: only 384 of 668 validation serves have the right timing and player. [#77](https://github.com/ahalp90/badminton_cv_annotator/issues/77) already records first-stroke labels that do not mark the actual serve contact there.

A small direct sample of accepted-scene serve misses would separate label timing problems from detector misses before either part is changed.

## 5. What still fails with good inputs?

**1,335** missed labels have an accepted court and both players available at the labelled frame, compared with 1,163 in the historical model. Rescued scenes bring 89 misses of their own, and scenes both courts accepted have 71 more misses than before.

The plan from the earlier backlog still applies: a small directly checked sample across serves, middle contacts and final contacts can separate:

- real detector misses;
- wrong or ambiguous labels;
- hits that never enter the candidate pool;
- available candidates chosen badly by the final sequence;
- boundary or scene-transition cases.

Use that sample to decide which errors a contact-model change could address.

## Carried over unchanged

- **Bad-label sweep ([#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147)).** Video 15 stays excluded. No other video shows the same whole-video misalignment: every game now matches at least 72.9% of its labelled hits. This closeout found one additional timing problem, four video 12 rallies with official timestamps about 15 frames off. The release may need those four rallies corrected or dropped.
- **Full release inventory ([#133](https://github.com/ahalp90/badminton_cv_annotator/issues/133)).** Evaluation still covers 46 ShuttleSet22 videos plus 40 original ShuttleSet videos.
- **Review cutoff.** The historical cutoff was carried over to the new confidence model. Its queue precision is similar, 83.5% against 84.4%, so retuning remains low priority.
- **Noise-aware training.** Still deferred until a small verified contact set exists.
