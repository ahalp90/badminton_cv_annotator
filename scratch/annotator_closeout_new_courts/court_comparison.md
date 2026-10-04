# Old courts versus new courts

**Better courts gain rallies where the court decision changed, but new rejections and refit churn largely cancel that in the total.** On the 46 videos, the new-court model gets **1,744 / 3,327 rallies fully correct (52.4%)**. A fresh refit on the old courts gets **1,734 (52.1%)**; the historical model gets **1,763 (53.0%)**.

**Contents**  
[What is being compared](#what-is-being-compared)  
[How much refitting alone moves the totals](#how-much-refitting-alone-moves-the-totals)  
[Complete rallies](#complete-rallies)  
[Where rallies are gained and lost](#where-rallies-are-gained-and-lost)  
[Same-label contact comparison](#same-label-contact-comparison)  
[Which videos moved](#which-videos-moved)  
[Where contacts are lost](#where-contacts-are-lost)  
[Player assignment](#player-assignment)  
[What this comparison tells us](#what-this-comparison-tells-us)  
[Evidence](#evidence)

## What is being compared

Three saved outputs are scored against the same labels:

- **historical model** — old courts, fitted with scikit-learn 1.6.1; the output the earlier investigation scored;
- **fresh old-court refit** — old courts, refitted with the current code and scikit-learn 1.9.1;
- **new-court model** — the selected base model, fitted the same way as the fresh refit on the new court release.

The fresh refit is the fair pairing for the court change: it shares the new model's training code and library. The historical model is kept for lineage. Contact-level comparisons below use the historical model because the earlier report saved its per-label results.

Both refits track players within each scene and use one net position per video to decide near and far.

## How much refitting alone moves the totals

Two refits with different random seeds differed by **33 complete rallies** ([refit regression](../../experiments/annotator/reports/refit_regression.md)). The new model sits 10 above the fresh refit and 19 below the historical model. Both gaps are smaller than that seed difference.

Refitting alone also moves many individual rallies. Going from the historical model to the fresh old-court refit turned 170 complete rallies incorrect, for a net loss of 29. Fit row order and library version each shifted the total as well. So the totals alone cannot separate a court effect of a few dozen rallies from refit churn. The court-decision groups below can.

## Complete rallies

| Measure, ±10 frames | Historical | Fresh old-court refit | New courts |
|---|---:|---:|---:|
| Fully correct rallies / 3,327 | 1,763 (53.0%) | 1,734 (52.1%) | **1,744 (52.4%)** |
| Matched contacts / 37,184 | 33,551 (90.2%) | 33,529 (90.2%) | **34,200 (92.0%)** |
| Predicted contacts | — | 39,428 | 40,246 |
| Contact precision | — | 85.04% | 84.98% |

The new model finds 671 more labelled hits than the fresh refit while adding 818 predictions, so precision barely moves. More hits found does not mean more rallies complete: one missed, extra or wrong-sided hit still fails the whole rally.

At ±5 frames the new model gets 1,394 rallies fully correct. The historical 47-video figure was 1,430, including video 15.

## Where rallies are gained and lost

Against the fresh refit, the new model gains **223** complete rallies and loses **213**. Against the historical model it gains 224 and loses 243.

Each rally is grouped by the court decision at its labelled hits. A rally counts as rejected under a court version when any of its labelled hits falls in a scene that version rejected.

| Court decision | Rallies | Gained / lost vs fresh refit | Gained / lost vs historical |
|---|---:|---:|---:|
| Accepted under both courts | 2,677 | 154 / 137 | 150 / 164 |
| Rescued by the new courts | 114 | **48 / 1** | 49 / 1 |
| Rejected under both courts | 443 | 21 / 42 | 25 / 45 |
| Rejected only under the new courts | 93 | 0 / **33** | 0 / 33 |

The court effect shows up where the court decision changed: +47 in rescued scenes and −33 in newly rejected ones. Where both versions accepted the scene, gains and losses are about as large as ordinary refit churn and roughly cancel.

## Same-label contact comparison

Both outputs are scored against the same 37,184 labelled hits. The table groups them by court decision at the labelled frame:

| Court decision at the labelled frame | Labels | Historical matches | New matches |
|---|---:|---:|---:|
| Accepted under both courts | 33,982 | 32,778 | 32,707 |
| Rescued: rejected old, accepted new | 1,367 | 8 | **1,278** |
| Rejected under both courts | 1,247 | 232 | 215 |
| Newly rejected: accepted old, rejected new | 588 | 533 | **0** |
| **Total** | **37,184** | **33,551** | **34,200** |

The rescued hits are concentrated: 717 are in video 53 and the rest in 14 other videos. The newly rejected hits are spread across 33 videos. In 451 of them the new detector found no usable court; in 137 it found one but the two-player check failed.

In scenes both versions accepted, matched hits fell slightly, from 32,778 to 32,707.

## Which videos moved

| Video | Fresh old-court refit | New courts | Gained / lost |
|---|---:|---:|---:|
| 53: earlier corrupted-court case | 9 | 32 | 26 / 3 |
| 12 | 21 | 29 | 10 / 2 |
| 39 | 38 | 46 | 12 / 4 |
| 17: earlier far-player case | 18 | 20 | 2 / 0 |
| 19 | 56 | 47 | 3 / 12 |
| 42: largest loss | 28 | 15 | 1 / 14 |

Video 53 alone adds 23 rallies; the other 45 videos together lose 13. Video 42's losses come mostly from missed hits and wrong player sides, not court rejection: only 13 of its 60 misses are in rejected scenes. Video 19's 12 losses include nine with missed hits and two with no overlapping clip.

## Where contacts are lost

| Contact position | Historical missed | New missed |
|---|---:|---:|
| Serve | 561 / 3,327 (16.9%) | 605 / 3,327 (18.2%) |
| Middle | 8.2% | 1,824 / 30,570 (6.0%) |
| Final | 17.3% | 555 / 3,287 (16.9%) |

Middle contacts gain the most. Serves get slightly worse: 44 more serve misses. Final contacts barely move.

| State for missed labels | Historical misses | New misses |
|---|---:|---:|
| Court rejected | 2,374 | **1,620** |
| Court accepted; a player pick missing | 96 | 29 |
| Court accepted; both players picked | 1,163 | **1,335** |
| **Total** | **3,633** | **2,984** |

Court rejection falls from 65.3% to 54.3% of misses. The accepted-scene residual grows, partly because rescued scenes bring 89 misses of their own into accepted territory. Scenes both versions accepted have 71 more misses than before.

## Player assignment

The new model gets 33,156 of 34,199 known-side timing matches right (**96.9%**), the same rate as the historical 47-video result.

| Labelled side | Predicted far | Predicted near | No player |
|---|---:|---:|---:|
| Far | 16,468 | 373 | 5 |
| Near | 653 | 16,688 | 12 |

Video 17 alone has 209 wrong or missing sides, and video 42 has 99. The far player in video 17 is picked again, but its side errors did not go away; [court checks](court_checks.md#video-17-the-far-player-is-back-but-sides-still-fail) has the details.

## What this comparison tells us

Better courts did what a court fix can do: they rescued scenes the old outline lost and fixed both regression cases. They did not raise matched hits in scenes that were already accepted: 32,778 before, 32,707 now.

The flat total comes from two smaller court effects pulling in opposite directions, on top of refit churn. The new detector rejects some scenes the old one accepted, and over half the remaining misses are still in rejected scenes. The next court question is coverage, not geometry.

## Evidence

- `results/court_change_groups.json.gz` — label and rally groups by old and new court decision
- `results/summary.json.gz` — `court_comparison` holds per-video gained and lost counts
- `results/contexts.csv.gz` — new court and player state at each labelled frame
- `../annotator_wrapup_evaluation/results/contexts.csv.gz` — old court state at each labelled frame
- `experiments/annotator/good_court_refit/evidence/saved-stream-analysis.json.gz` — gained and lost rally IDs against both old models
- [model_selection.md](../../experiments/annotator/reports/model_selection.md) — fresh-refit totals and per-video regressions

Commands: [evaluation_reproduction.md](evaluation_reproduction.md).
