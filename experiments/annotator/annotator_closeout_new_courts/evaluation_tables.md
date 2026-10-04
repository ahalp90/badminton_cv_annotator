# Evaluation numbers and definitions

Use this reference for exact counts, error breakdowns and differences between scoring populations. The [main report](README.md) explains the findings with figures. Earlier-model counts come from the [earlier evaluation tables](../../../scratch/annotator_wrapup_evaluation/evaluation_tables.md) and use the same 46 videos unless labelled otherwise.

**Contents**  
[Population](#population)  
[Definitions](#definitions)  
[New-court output](#new-court-output)  
[Selected review queue](#selected-review-queue)  
[Proposal overlap and rally coverage](#proposal-overlap-and-rally-coverage)  
[What selection discards](#what-selection-discards)  
[Errors inside selected clips](#errors-inside-selected-clips)  
[Where misses occur](#where-misses-occur)  
[Rally position](#rally-position)  
[Timing of matched contacts](#timing-of-matched-contacts)  
[Player assignment](#player-assignment)  
[Video-to-video variation](#video-to-video-variation)  
[Weak videos](#weak-videos)  
[Rallies that fail only on an extra final hit](#rallies-that-fail-only-on-an-extra-final-hit)  
[Rally length](#rally-length)  
[Tighter timing: ±5 frames](#tighter-timing-5-frames)  
[Contact precision and recall](#contact-precision-and-recall)  
[Original ShuttleSet](#original-shuttleset)  
[Reproduce](#reproduce)

## Population

- **46 ShuttleSet22 videos**; video 15 stays excluded.
- **3,327 cleaned rallies / 37,184 contacts.**
- **±10 frames at 30 fps** is the main timing allowance.
- The selected new-court base model and its saved output are unchanged.

The 45-video column without video 53 is a sensitivity check only.

## Definitions

**Fully correct rally** — one proposed clip contains one complete labelled rally; every labelled contact matches once; there are no extra predicted contacts; every matched contact has the correct near/far player.

**Exact whole-rally contact sequence** — same check without player correctness.

**Contact timing match** — a predicted contact lies within the allowed frame distance under complete-video one-to-one matching.

**Selected clip** — a proposal kept by the historical review cutoff, 0.757; “selected” does not mean correct. The cutoff was carried over unchanged to the new confidence model.

**Cleaned labels** — the label subset used for the main evaluation; old result files call them `retained`.

**Court rejected** — the labelled frame lies in a scene the court stage rejected: either no usable court was found, or the two-player vote failed.

## New-court output

| Cleaned labels, ±10 frames | Earlier model, 46 videos | **New courts, 46 videos** | New courts without video 53: sensitivity |
|---|---:|---:|---:|
| Labelled rallies | 3,327 | **3,327** | 3,251 |
| Labelled contacts | 37,184 | **37,184** | 36,247 |
| Exact whole-rally contact sequence | 1,777 (53.4%) | **1,767 (53.1%)** | 1,735 (53.4%) |
| Fully correct rally, including players | 1,763 (53.0%) | **1,744 (52.4%)** | 1,712 (52.7%) |
| Contact timing match | 33,551 (90.2%) | **34,200 (92.0%)** | 33,320 (91.9%) |
| Contact timing + correct player | 32,586 (87.6%) | **33,156 (89.2%)** | 32,309 (89.1%) |
| Serve timing match | 2,766 (83.1%) | **2,722 (81.8%)** | 2,664 (81.9%) |
| Serve timing + correct player | 2,642 (79.4%) | **2,587 (77.8%)** | 2,532 (77.9%) |
| A clip contains the whole rally interval | 2,989 (89.8%) | **2,983 (89.7%)** | 2,920 (89.8%) |

These contact percentages are label-recovery rates. A fresh refit on the old courts gets 1,734 fully correct; [court comparison](court_comparison.md#how-much-refitting-alone-moves-the-totals) compares these totals with refit noise.

## Selected review queue

| Selected clips | Earlier model, 46 videos | **New courts, 46 videos** | Without video 53: sensitivity |
|---|---:|---:|---:|
| Total | 747 | **725** | 714 |
| Known correct | 616 | **591** | 585 |
| Known wrong | 114 | **117** | 112 |
| Labels cannot judge | 17 | **17** | 17 |

For the 46-video queue:

| Exact selected annotation | Result |
|---|---:|
| Precision among judgeable clips | **591 / 708 = 83.5%** |
| Recall across labelled rallies | **591 / 3,327 = 17.8%** |
| F1 | **29.3%** |
| Known-correct share if unknowns get no credit | **591 / 725 = 81.5%** |

“Unknown” means the labels cannot settle exact correctness.

## Proposal overlap and rally coverage

Across the 46 videos:

- 2,983 labelled rallies fit inside at least one proposal;
- 2,840 fit inside a proposal that overlaps exactly one labelled rally;
- 1,004 of the 4,021 proposals overlap no cleaned rally;
- 49 overlap more than one;
- 2,968 overlap exactly one.

Best available rally output:

| Best available output | Earlier model | New courts | New courts without video 53 |
|---|---:|---:|---:|
| Fully correct | 1,763 | **1,744** | 1,712 |
| Contains all labels but has another error | 1,226 | **1,239** | 1,208 |
| Reaches some labels but not the whole rally | 113 | **128** | 120 |
| Reaches no labelled contact | 225 | **216** | 211 |

A clip can contain every labelled contact and still fail because it has extras, another rally's contacts or wrong players. Only five of the 216 unreached rallies are in video 53.

## What selection discards

Selection keeps 725 of 4,021 proposals. Outside the queue are:

- 1,153 correct clips;
- 1,156 wrong clips;
- 987 unjudgeable clips.

Selection keeps 591 of the 1,744 available correct clips (**33.9%**), covering **17.8%** of the 3,327 cleaned rallies.

Examples from the saved queue:

| Video | Correct before selection | Correct selected | Wrong selected | Unjudgeable selected |
|---|---:|---:|---:|---:|
| 33 | 62 | 28 | 6 | 0 |
| 31 | 50 | 24 | 2 | 1 |
| 52 | 59 | 23 | 6 | 1 |
| 34 | 44 | 23 | 3 | 0 |
| 42 | 15 | 0 | 0 | 0 |

Video 42 is the only video with no selected clip.

## Errors inside selected clips

The 46-video queue has 117 known-wrong clips:

| Problem | Clips affected |
|---|---:|
| Extra contacts | **88** |
| Missing contacts | 51 |
| Wrong player on a matched contact | 7 |
| Rally cut off | 2 |

Categories overlap. One clip fails only because of player assignment.

### Extra events

| Position | Events |
|---|---:|
| Before first label | 5 |
| Between first and last labels | 33 |
| After final label | **65** |
| **Total** | **103** |

### Missed labelled events

| Position | Events |
|---|---:|
| Serve | 21 |
| Middle | 21 |
| Final | **27** |
| **Total** | **69** |

These are event counts matched within each selected clip, not full-video contact counts.

The largest exclusive combinations are 65 extras-only, 22 misses-only and 21 with both. The earlier model had 85 extra and 67 missed events across 114 wrong clips. Serves made up 35 of those misses; now 21.

“After the final label” is not proof that a physical hit is false. Earlier deletion work could not separate real tail contacts from bad extras reliably.

## Where misses occur

| State at the labelled frame | Labelled contacts | Matched | Missed | Miss rate |
|---|---:|---:|---:|---:|
| Court rejected | 1,835 | 215 | **1,620** | 88.3% |
| Court accepted; a player pick missing | 60 | 31 | **29** | 48.3% |
| Court accepted; both players picked | 35,289 | 33,954 | **1,335** | 3.8% |
| **Total** | **37,184** | **34,200** | **2,984** | **8.0%** |

A label can match even when its exact frame is rejected because the ±10-frame window can reach a nearby event.

The 1,620 court-rejected misses are **54.3%** of all misses; the earlier model had 2,374 (65.3%). Without video 53, 1,602 of 2,927 misses are still in rejected scenes (**54.7%**); the earlier figure was 1,640 of 2,891.

| Rejected misses by cause | Missed |
|---|---:|
| No usable court found | 1,163 |
| Court found; two-player vote failed | 457 |

[Court checks](court_checks.md#why-scenes-are-still-rejected) breaks the first row down further.

## Rally position

Among court-accepted frames:

| Contact position | Missed | Total | Miss rate | Earlier model |
|---|---:|---:|---:|---:|
| Serve | 293 | 2,826 | 10.4% | 9.0% |
| Middle | 721 | 29,441 | 2.4% | 2.3% |
| Final | 350 | 3,082 | 11.4% | 11.2% |

Single-contact rallies count as serves only.

Across all frames, the new model matches 2,722/3,327 serves, 28,746/30,570 middle contacts and 2,732/3,287 finals. Those counts include court-rejected frames, so they answer a different question from the accepted-scene table above.

## Timing of matched contacts

The rows are cumulative over the 34,200 matches.

| Distance from label | Matches | Share |
|---|---:|---:|
| Exact frame | 9,033 | 26.4% |
| Within 2 frames | 28,460 | 83.2% |
| Within 5 frames | 33,504 | 98.0% |

Median offset is zero; mean offset is about half a frame early. The earlier model was 26.9%, 83.6% and 98.0%.

This describes only contacts that already match; the 2,984 misses are outside the timing-offset distribution. It does not support globally shifting predictions.

## Player assignment

34,199 timing matches have a known target side. The new model gets 33,156 right (**96.9%**):

| Labelled side | Predicted far | Predicted near | No player |
|---|---:|---:|---:|
| Far | 16,468 | 373 | 5 |
| Near | 653 | 16,688 | 12 |

That leaves 1,026 near/far confusions and 17 unassigned predictions. Near/far is image position, not persistent athlete identity across camera or end changes.

Video 17 accounts for 209 of the wrong or missing sides and video 42 for 99.

## Video-to-video variation

The median video has a 92.8% timing-match rate and a 52.2% fully correct-rally rate when each video gets equal weight.

At the high end:

- video 18: 45/60 fully correct (75.0%);
- video 31: 50/68 (73.5%);
- video 41: 55/77 (71.4%).

At the weak end, causes differ: videos 17 and 42 fail mostly on player sides; videos 11, 21, 25 and 38 still lose many hits to court rejection. The [weak-video table](#weak-videos) gives the counts.

## Weak videos

Low score tells us where to look, not what the cause is.

| Video | Timing matches | Fully correct rallies | Misses in rejected scenes | Earlier model: timing / fully correct |
|---|---:|---:|---:|---|
| 42 | 564/624 | 15/58 | 13/60 | 586 / 30 |
| 17 | 876/976 | 20/73 | 0/100 | 842 / 17 |
| 11 | 737/860 | 16/44 | 108/123 | 779 / 16 |
| 25 | 749/889 | 28/85 | 92/140 | 760 / 34 |
| 38 | 949/1,161 | 28/82 | 140/212 | 1,022 / 29 |
| 53 | 880/937 | 32/76 | 18/57 | 195 / 7 |
| 21 | 658/806 | 33/75 | 114/148 | 583 / 31 |
| 12 | 653/781 | 29/61 | 6/128 | 469 / 23 |
| 20 | 452/537 | 19/43 | 80/85 | 324 / 15 |
| 24 | 277/330 | 14/31 | 41/53 | 248 / 11 |
| 39 | 683/717 | 46/75 | 12/34 | 577 / 38 |

Videos 42, 11 and 25 were not in the earlier weak-video table. They and video 38 match fewer hits than the earlier model did. Video 38 has 73 labelled hits in newly rejected scenes, more than any other video.

Video 12's misses are mostly label errors. 101 of its 128 sit in four rallies whose official timestamps are about 15 frames off the visible hits; see [label checks](video_checks.md#video-12-four-rallies-with-shifted-timestamps).

## Rallies that fail only on an extra final hit

363 failed rallies have every labelled hit matched with the right player. In 262, the model adds a hit by the other player 10–60 frames after the last label. In **233**, that is the rally's only error: its clip covers this rally alone and holds exactly one extra hit. Of the other 29, 23 have more extras and 6 share a clip with a neighbouring rally.

The official rows record how each rally ended:

| Official ending | Fails on the extra hit alone | All rallies with that ending | Share |
|---|---:|---:|---:|
| Shot into the net | 78 | 779 | 10.0% |
| Shot out | 59 | 1,085 | 5.4% |
| Opponent's winner landed | 60 | 975 | 6.2% |
| Misjudged landing | 19 | 141 | 13.5% |
| Shot short of the net | 14 | 336 | 4.2% |
| Fault | 1 | 6 | — |
| Not recorded | 2 | 5 | — |
| **Total** | **233** | **3,327** | **7.0%** |

All 16 rallies sampled from the 262 showed no hit at the extra frame, and all 16 are among the 233; see [label checks](video_checks.md#extra-hit-after-the-last-label).

## Rally length

| Rally length | Fully correct | Rate | Earlier model, 47 videos |
|---|---:|---:|---:|
| 1–5 contacts | 486/967 | 50.3% | 52.3% |
| 6–10 | 533/1,004 | 53.1% | 51.6% |
| 11–20 | 515/936 | 55.0% | 52.2% |
| More than 20 | 210/420 | 50.0% | 48.0% |

Length is still not the main explanation for the current failures.

## Tighter timing: ±5 frames

| Cleaned labels, ±5 frames | New courts, 46 videos |
|---|---:|
| Exact whole-rally contact sequence | 1,410 (42.4%) |
| Fully correct rally | 1,394 (41.9%) |
| Contact timing match | 33,504 (90.1%) |
| Contact timing + correct player | 32,526 (87.5%) |
| Serve timing + correct player | 2,224 (66.8%) |
| Selected: correct / wrong / unjudgeable | 520 / 188 / 17 |

Serves lose the most when the allowance tightens: serve timing matches fall from 2,722 to 2,331. The historical 47-video model got 1,430 rallies fully correct at ±5, including video 15.

## Contact precision and recall

| Population, ±10 | Matched predictions | Predictions | Labels | Precision | Recall | F1 | Player-aware F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| New courts, 46 videos | 34,200 | 40,246 | 37,184 | 84.98% | 91.98% | 88.34% | 85.64% |
| Without video 53 | 33,320 | 39,333 | 36,247 | 84.71% | 91.92% | 88.17% | 85.50% |
| Fresh old-court refit, 46 videos | 33,529 | 39,428 | 37,184 | 85.04% | 90.17% | 87.53% | — |

The historical model's 47-video figures were 81.04% precision and 88.22% recall, including video 15.

## Original ShuttleSet

| New-court model, ±10 | Validation: 8 videos | Development: 32 videos |
|---|---:|---:|
| Labelled rallies / contacts | 668 / 5,696 | 2,691 / 27,571 |
| Fully correct rally | 273 (40.9%) | 1,229 (45.7%) |
| Exact whole-rally sequence | 306 | 1,247 |
| Contact timing match | 5,214 (91.5%) | 25,535 (92.6%) |
| Contact timing + correct player | 4,889 (85.8%) | 24,917 (90.4%) |
| Serve timing + correct player | 384 (57.5%) | 1,835 (68.2%) |
| Fully correct at ±5 | 212 | 931 |

Validation videos are scored with a contact tree trained on the other 32. Each development video is scored with the contact tree fitted while its group was held out. The earlier recount of saved development outputs gave 1,209 fully correct, 24,285 timing + player and 1,790 serves + player.

## Reproduce

[Methods, saved files and commands](evaluation_reproduction.md) explain how these counts were produced.
