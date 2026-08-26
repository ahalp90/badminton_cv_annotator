# Contact detector experiments: what we tested and what we learned

This is the technical report for the contact-detector exploration.

It keeps enough detail to reproduce and interpret the work, but the main story is organised around five questions:

1. What were we trying to learn?
2. What did we test?
3. What happened?
4. What does that result mean?
5. What is worth testing again on the larger dataset?

The two tree models are random forest (RF) and histogram gradient boosting (HGB). HGB is the stronger model in this pilot.

For annotator-level output, see [`auto_annotator_progress.md`](auto_annotator_progress.md). For the short version, see [`README.md`](README.md).

## Table of contents

- [What the pilot was trying to learn](#what-the-pilot-was-trying-to-learn)
- [1. Search region: can we avoid searching the whole video?](#1-search-region-can-we-avoid-searching-the-whole-video)
- [2. Tree model: can simple physical features beat the current heuristics?](#2-tree-model-can-simple-physical-features-beat-the-current-heuristics)
- [3. Complete rallies: does better contact F1 turn into usable output?](#3-complete-rallies-does-better-contact-f1-turn-into-usable-output)
- [4. Miss audit: are missed contacts outside the search or being discarded later?](#4-miss-audit-are-missed-contacts-outside-the-search-or-being-discarded-later)
- [5. Frame-rate motion check](#5-frame-rate-motion-check)
- [6. Cheap decision-layer check](#6-cheap-decision-layer-check)
- [7. Selected-stream player-side check](#7-selected-stream-player-side-check)
- [8. Serve-lookback threshold check](#8-serve-lookback-threshold-check)
- [9. Broad nearby-alternative shortlist](#9-broad-nearby-alternative-shortlist)
- [10. Full candidate-union rally ceiling](#10-full-candidate-union-rally-ceiling)
- [11. Compact serve-prefix candidate check](#11-compact-serve-prefix-candidate-check)
- [12. The main warning: serves in sset_21](#12-the-main-warning-serves-in-sset_21)
- [13. What to carry into the larger dataset](#13-what-to-carry-into-the-larger-dataset)
- [Technical reference](#technical-reference)
- [Compact reference for old experiment codes](#compact-reference-for-old-experiment-codes)
- [Reproduction paths](#reproduction-paths)
- [Pilot limits](#pilot-limits)

## What the pilot was trying to learn

The end goal is not “find every contact at any cost.”

The useful system should return a worthwhile number of rallies that are correct from beginning to end, and abstain when it is not confident enough.

The contact work therefore separates three jobs:

```text
label-blind search region
→ learned contact timing score
→ small decision layer that turns scores into events
→ existing Top/Bottom player-side attribution
```

The three-video pilot was meant to choose a sensible design, not final fitted settings.

The results below should answer whether an idea deserves a larger test. They should not be used to freeze a threshold, a duplicate-removal distance, or a frame-rate convention for production.

## 1. Search region: can we avoid searching the whole video?

### What we were trying to learn

The old raw proposals make too many real contacts unreachable by any later classifier. At ±10 base-30 frames they cover only:

- **83.8% of non-serves**;
- **66.1% of serves**.

The question was whether a broader deterministic region could keep almost every real contact available without scoring almost every frame.

### What we tested

Region version 2 is label-blind. It combines neighbourhoods around:

- current raw contact proposals;
- relaxed shuttle impulse and direction-change peaks;
- local shuttle-to-wrist minima;
- shuttle visibility changes;
- detected rally starts;
- scene starts;
- a 45-base-30-frame look-back before eligible court-view intervals.

That last piece exists because some serves happen in the close-up immediately before the broadcast returns to the full court.

Ground-truth contacts are loaded only after the region is built.

### What happened

Region version 2 scores about **31.9% of source frames**.

At ±10:

| Search surface | All contacts | Non-serves | Serves |
| --- | ---: | ---: | ---: |
| Court-view intervals only | 97.9% | 98.3% | 93.2% |
| **Region version 2** | **98.3%** | **98.4%** | **97.9%** |

Per fixture:

| Fixture | Non-serve coverage | Serve coverage | All-contact coverage |
| --- | ---: | ---: | ---: |
| `sset_01` | 99.4% | 98.2% | 99.3% |
| `sset_15` | 100.0% | 100.0% | 100.0% |
| `sset_21` | 93.7% | 94.7% | 93.8% |

These are **search coverage** numbers. They say that a candidate exists nearby. They do not say the classifier accepts it.

### What the result means

Region version 2 is broad enough to use as the pilot search surface.

Its job is to avoid making contacts impossible to recover. It is therefore reasonable for it to be more permissive than a final detector.

The `sset_21` non-serve coverage remains a warning for whole-video claims, but most contact-detector misses in the later audit occur **inside** the available candidate surface.

### What is worth testing on the larger dataset

Keep the design idea and remeasure coverage on the larger, more varied set.

Only build a separate off-court rescue path if misses outside the region actually destroy a useful number of otherwise-complete rallies.

## 2. Tree model: can simple physical features beat the current heuristics?

### What we tested

The main HGB feature set has **85 columns**:

- 60 physical values sampled at five time offsets;
- 25 validity/missingness flags at the same offsets.

The physical signals include shuttle velocity, speed and impulse, wrist distance, nearest-wrist direction, and player ankle motion.

We also tested:

- the same physical inputs plus 20 context features;
- context only;
- missingness only;
- random forest versions of the main feature sets.

The timing scorer uses leave-one-fixture-out evaluation. Threshold choice happens inside the training side; the held-out fixture does not choose its own cut-off.

### What happened

At ±10, using each model's original decisions:

| Event stream | Precision | Recall | F1 | Non-serve recall | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 66.9% | 79.3% | 72.6% | 81.2% | 61.0% |
| **HGB physical + validity** | **84.5%** | **90.5%** | **87.4%** | **92.9%** | **67.5%** |
| RF physical + validity | 84.1% | 85.2% | 84.6% | 89.6% | 42.8% |
| HGB + context | 81.7% | 89.8% | 85.5% | 92.5% | 63.4% |
| RF + context | 83.9% | 86.5% | 85.2% | 90.4% | 47.9% |

Controls:

| Control | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| HGB context only | 47.0% | 79.6% | 59.1% |
| RF context only | 37.1% | 82.2% | 51.1% |
| HGB missingness only | 16.1% | 96.9% | 27.5% |
| RF missingness only | 16.0% | 96.9% | 27.5% |

### What the result means

HGB with physical values plus validity flags is the useful simple model.

Adding the tested context block did not help HGB. Missingness by itself can get high recall only by predicting far too many events.

There is no strong reason to spend more pilot time tuning random forest.

### Player-side result on the original HGB event stream

The tree predicts contact timing only. The existing Top/Bottom rule is applied afterwards.

At ±10:

| Event stream | Timing recall | Side accuracy on timing matches | Timing + correct-side recall | Joint event+side F1 |
| --- | ---: | ---: | ---: | ---: |
| Current final heuristics | 79.3% | **89.0%** | 70.6% | 64.6% |
| **HGB physical + validity** | **90.5%** | 83.7% | **75.7%** | **73.1%** |
| RF physical + validity | 85.2% | 85.8% | 73.1% | 72.6% |

The side rule is less accurate on HGB timing matches than on the old heuristic timing matches. We have measured that difference, but not established why it occurs.

The selected wider-duplicate-removal event stream has now been scored directly for player side; that result is reported below.

## 3. Complete rallies: does better contact F1 turn into usable output?

### What we were trying to learn

An event-level F1 score can look good while a rally is unusable.

A single missed contact, extra event, or wrong player side can make the whole rally wrong.

The strict rally score asks the product question directly: **how many predicted rallies can we keep without repairing them?**

### The fixed meaning of a fully correct kept rally

A kept span is fully correct only when:

- it maps to exactly one real rally;
- every labelled contact is found within the evaluation tolerance;
- there are no extra predicted contact events;
- every event has a Top/Bottom answer;
- every Top/Bottom answer is correct.

A missing side answer rejects the whole rally.

This meaning is fixed across the follow-up experiments.

The main timing tolerance is ±10 base-30 frames. A ±5 result is a sensitivity check.

The whole-rally timing confidence is the **lowest** retained HGB score in that span. Raising the confidence cut rejects the whole span; it does not quietly remove one weak contact.

### What happened with the original HGB decisions

| Minimum whole-rally timing score | Spans kept | Fully correct at ±10 | Fully correct among kept |
| --- | ---: | ---: | ---: |
| 0.00 | 291 | 21 | 7.2% |
| 0.80 | 216 | 17 | 7.9% |
| 0.85 | 123 | 13 | 10.6% |
| 0.90 | 51 | 9 | **17.6%** |
| 0.95 | 11 | 1 | 9.1% |

### What the result means

Confidence filtering alone does not produce a useful clean subset yet.

The 0.90 cut throws away most output and still leaves only nine fully correct spans.

This is why complete-rally yield should stay the primary project score, with contact precision/recall/F1 used to explain why it moved.

![Whole-rally confidence versus yield for the original and wider duplicate-removal decisions.](figures/followup_rally_yield_curve.png)

## 4. Miss audit: are missed contacts outside the search or being discarded later?

### What we were trying to learn

The original HGB event stream misses **296 of 3,128** contacts at ±10.

Before widening search or building another model, we needed to know whether those contacts were:

- never available to HGB;
- present as lower-scoring candidates;
- lost when nearby peaks were collapsed;
- or lost in the one-to-one event match.

### What happened

The 296 misses contain:

- **95 serves**;
- **201 ordinary exchanges**.

A candidate from the fixed search surface exists near **244 / 296** misses.

For serves, a candidate from the fixed search surface exists near **89 / 95** misses.

The strongest nearby candidate for each missed contact falls into these groups:

| What happened to the strongest nearby candidate | Missed contacts | Missed serves |
| --- | ---: | ---: |
| It scored below the HGB cut-off | **207** | **84** |
| It was removed as a nearby duplicate | 19 | 2 |
| It was retained but lost the one-to-one match | 18 | 3 |
| No candidate from the fixed search surface was present in the window | 52 | 6 |

The evidence streams are usually present somewhere near a miss:

- shuttle evidence near **240 / 296** misses;
- pose evidence near **230 / 296**;
- wrist evidence near **230 / 296**.

The filtered handcrafted rule finds **103** of the HGB misses.

Most importantly for the strict rally goal, only **13** spans are otherwise exact apart from one missing contact. All 13 have a region-v2 candidate nearby.

![What the original HGB missed-contact audit found near each miss.](figures/followup_missed_contact_audit.png)

### What the result means

The search region is not the first problem to expand.

Most misses already have a plausible candidate in the fixed score surface. That makes decision-layer tests and better candidate ranking more useful than immediately searching much more of the broadcast.

The handcrafted overlap is interesting, but it is not enough by itself to justify a merge. The later shortlist gate had to show that a deployable, label-blind candidate list had enough compact headroom first.

### What is worth testing on the larger dataset

Repeat the same readable miss audit after refitting.

If many near-perfect rallies are then broken by contacts genuinely outside the search region, test one bounded rescue search. Otherwise keep the search simple.

## 5. Frame-rate motion check

### What we were trying to learn

`sset_01` and `sset_15` are 25 fps. `sset_21` is 30 fps.

The original motion features use movement per video frame. The same physical movement can therefore produce a different number at a different frame rate.

The question was whether that convention was hurting HGB.

### What we tested

Three otherwise matched HGB trials:

1. **Existing raw motion** — keep the original per-frame velocity, impulse and ankle-speed values.
2. **No frame-rate-sensitive motion** — remove those motion columns.
3. **Common 30 fps motion scale** — convert first differences by `fps / 30` and the shuttle second-difference impulse by the square of that factor.

Search regions, folds, model settings, threshold selection and duplicate-removal rules stay the same.

### What happened

| Motion treatment | Precision | Recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: |
| **Existing raw motion** | **84.5%** | **90.5%** | **87.4%** | 67.5% |
| Remove frame-rate-sensitive motion | 82.7% | 87.1% | 84.8% | 47.9% |
| Common 30 fps scale | 84.0% | 90.2% | 87.0% | **68.2%** |

Strict rallies:

| Minimum whole-rally timing score | Raw motion | Remove motion | Common 30 fps scale |
| --- | ---: | ---: | ---: |
| 0.00 | **21 / 291** | 16 / 293 | 15 / 295 |
| 0.85 | **13 / 123** | 9 / 156 | 10 / 124 |
| 0.90 | **9 / 51** | 6 / 67 | 6 / 58 |

![Pilot frame-rate motion feature check.](figures/followup_motion_feature_check.png)

### What the result means

Raw motion beat both alternatives in this pilot.

The common-scale trial found two extra serves overall, but lost pooled timing F1 and complete rallies.

Do **not** turn this into “raw motion is better in general.” There are only three videos and only one 30 fps fixture. The larger dataset should test the convention again.

## 6. Cheap decision-layer check

### What we were trying to learn

The miss audit showed that many real contacts already had nearby scored candidates.

Before refitting the model, we tested whether a very small change in how scores become events could improve whole-rally output.

HGB itself was not refit.

### What we tested

Five fixed variants:

| Plain-language variant | What changed |
| --- | --- |
| Original decisions | original score cut-off; duplicate-removal distance 5 |
| Lower score cut-off everywhere | one lower score point; distance still 5 |
| Smaller duplicate distance | original score cut-off; distance 4 |
| Wider duplicate distance | original score cut-off; distance 6 |
| Lower score cut-off near rally starts only | lower cut-off only in the existing label-blind rally-start region |

Distances are in base-30 frames and are scaled for the source FPS.

### What happened

| Decision variant | Predicted events | Precision | Recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original decisions | 3,350 | 84.5% | 90.5% | 87.4% | 67.5% |
| Lower score cut-off everywhere | 3,559 | 81.2% | **92.4%** | 86.4% | **73.6%** |
| Smaller duplicate distance | 3,704 | 76.6% | 90.7% | 83.1% | 67.5% |
| **Wider duplicate distance** | **3,238** | **87.2%** | 90.3% | **88.8%** | 67.1% |
| Lower score cut-off near rally starts only | 3,386 | 84.0% | 90.9% | 87.3% | 70.5% |

Strict fully-correct rallies:

| Decision variant | No timing cut | 0.80 | 0.85 | 0.90 | 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original decisions | 21 / 291 | 17 / 216 | 13 / 123 | 9 / 51 | 1 / 11 |
| Lower score cut-off everywhere | 14 / 295 | 10 / 118 | 7 / 72 | 5 / 33 | 0 / 5 |
| Smaller duplicate distance | 7 / 291 | 5 / 187 | 4 / 84 | 4 / 31 | 0 / 7 |
| **Wider duplicate distance** | **27 / 291** | **23 / 231** | **19 / 146** | **13 / 68** | **1 / 14** |
| Lower score cut-off near rally starts only | 20 / 292 | 16 / 194 | 12 / 109 | 8 / 44 | 1 / 8 |

![Timing F1 and serve recall for the five cheap decision-layer variants.](figures/followup_decision_layer_tradeoff.png)

### What the result means

The wider duplicate-removal variant is the best pilot decision layer.

It removes more close-together duplicate peaks, improves precision, leaves recall nearly unchanged, and gives more fully correct rallies.

Lowering the score cut-off finds more contacts and serves, but the extra events are costly at the rally level.

Again: **6 base-30 frames is a tested pilot setting, not a decided constant**.

### What is worth testing on the larger dataset

Reselect the score cut-off and duplicate-removal distance from scratch.

Keep the decision search small and judge it by the strict rally curve first.

Only use separate start handling if the larger error table shows a repeatable serve/start problem.

## 7. Selected-stream player-side check

The direct player-side table has now been run for the selected wider-duplicate-removal event stream.

At ±10:

| Event stream | Timing recall | Side accuracy on answered timing matches | Timing + correct-side recall | Joint event-and-side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 79.3% | **89.0%** | 70.6% | 64.6% | 46.2% |
| Original HGB decisions | **90.5%** | 83.7% | **75.7%** | 73.1% | **56.2%** |
| **Selected wider-duplicate-removal stream** | 90.3% | 83.4% | 75.2% | **73.9%** | **56.2%** |

### What the result means

The selected event stream still improves the complete contact-and-side output over the old heuristics.

It trades 0.5 points of timing-plus-correct-side recall against the original HGB decisions for fewer events and a slightly higher joint event-and-side F1.

The separate rally-level alternation fit has still **not** been rerun on the new contact events.

## 8. Serve-lookback threshold check

### What we were trying to learn

Serves remain weak, particularly on `sset_21`.

Before inventing another model, we tested the cheapest possible change: lower the score threshold only in the existing label-blind serve-lookback region. The same change was also tested together with the already-tested rally-start threshold lowering.

### What happened

| Plain-language decision | Events | Serve timing matches | Correctly sided serves | Joint event-and-side F1 | Fully correct with no timing cut |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original decisions | 3,350 | 197 | 164 | 73.1% | 21 |
| Lower threshold in serve look-back | 3,355 | 199 | 164 | 73.0% | 21 |
| Lower threshold near rally starts | 3,386 | 206 | 169 | 72.9% | 20 |
| Lower threshold in both places | 3,391 | 208 | 169 | 72.9% | 20 |
| **Selected wider-duplicate-removal stream** | **3,238** | 196 | 164 | **73.9%** | **27** |

The serve-lookback change adds five events and two serve timing matches.

It adds:

- **no correctly sided serve**;
- **no fully correct rally**.

All five added events were already present in the later compact serve-prefix candidate lists, and none was the oracle choice for a newly recovered serve.

### What the result means

Close this exact threshold idea.

Another nearby threshold on these same fixtures would only tune the same five-event effect.

## 9. Broad nearby-alternative shortlist

### What we were trying to learn

A second-stage model only makes sense if the first stage can produce a short list that contains substantially more real contacts than the final event list.

The miss audit was ground-truth-centred, so it could not answer that deployable question by itself.

We therefore built one label-blind shortlist and gave it a pass/fail gate before doing any merge, hard-negative refit or cleanup tree.

### What we tested

Start with the selected wider-duplicate-removal event stream.

For each selected event:

- keep the selected event;
- look in the same fixture and search interval within ±10 base-30 frames;
- add the strongest alternative outside the selected event's duplicate-removal distance;
- allow that alternative to be below the normal score cut-off;
- deduplicate the combined list.

Labels are used only after the list is frozen.

### What happened

| Measure | Selected event stream | Shortlist |
| --- | ---: | ---: |
| Candidates | 3,238 | 6,305 |
| Matched contacts at ±10 | 2,825 | 2,922 |
| Contact coverage at ±10 | 90.3% | 93.4% |
| Unmatched candidates at ±10 | 413 | 3,383 |
| Serve coverage at ±10 | 67.1% | 73.3% |
| Contact coverage at ±5 | 86.6% | 90.2% |
| Serve coverage at ±5 | 55.5% | 65.1% |

At ±10:

- the shortlist recovers **97 of the selected stream's 303 misses**;
- it adds **3,067** candidates;
- **2,970** of those added rows remain unmatched;
- that is about **31.6 added candidates per recovered contact**.

The gate required:

- at least **152** recovered contacts;
- total shortlist size no more than twice the selected event stream.

The size condition passed. The recovery condition failed.

![The tested shortlist added many candidates for a modest coverage gain.](figures/followup_shortlist_tradeoff.png)

### What the result means

Stop this exact shortlist as a practical second-stage input on the three-video pilot.

Do not spend more pilot time on the proposed handcrafted merge, hard-negative refit or cleanup tree fed by this list.

The fuller conclusion is:

> **The broad shortlist failed its recovery requirement, and the separate serve-prefix chooser also failed. The frozen candidate union still contains theoretical complete-rally headroom, but no validated label-blind selector has realised it.**

The next two sections show that headroom directly.

## 10. Full candidate-union rally ceiling

### What we were trying to learn

The broad shortlist failed because its event-coverage gain was too small for its size.

That still leaves a different question:

> Does the same frozen union contain combinations that could make more whole rallies correct, even if we do not know how to choose them?

### What we tested

An exact non-deployable oracle chooses from the frozen 6,305-candidate union **after**:

- candidate identities are fixed;
- existing half-open span membership is fixed;
- the shipped Top/Bottom answer for every candidate is fixed.

Of the 6,305 candidates, **6,252** fall inside the unchanged rally spans. The other 53 remain unassigned and are excluded.

Two ceilings are measured:

1. **timing-only feasible** — some exact event subset can match every labelled contact;
2. **timing + side feasible** — some exact subset can also make every selected event's fixed Top/Bottom answer correct.

### What happened

| Tolerance | Selected stream fully correct | Timing-only feasible | Timing + side feasible | Full gain |
| --- | ---: | ---: | ---: | ---: |
| ±10 | 27 | 144 | 42 | **+15** |
| ±5 | 24 | 105 | 37 | **+13** |

At ±10, the +15 full gains split as:

- +2 rallies on `sset_01`;
- +8 on `sset_15`;
- +5 on `sset_21`.

No fully correct selected-stream rally is lost.

The predeclared gate required at least **10** new fully correct rallies. The ceiling passes it.

![Candidate-union rally ceiling.](figures/followup_candidate_union_ceiling.png)

### What the result means

There is real selector headroom inside the candidate union.

But the oracle is **not a selector** and the 42-rally result is **not achieved performance**.

The timing-only ceiling is much higher than the timing-plus-side ceiling: **144 versus 42** at ±10. That makes the shipped player-side answers the larger remaining evidence limit inside this union.

This does **not** reopen the failed handcrafted merge, hard-negative refit or cleanup tree on these fixtures.

A practical selector must be tested on fresh whole videos or through properly nested cross-fitting.

## 11. Compact serve-prefix candidate check

### What we were trying to learn

The general shortlist is too large.

The serve-specific check asks a narrower question:

> Can the frames before a detected span's first selected event supply one missed serve without changing the rest of the rally?

### What we tested

Each detected span gets at most five frozen candidates:

- the three strongest raw HGB peaks in the prefix;
- the best filtered heuristic contact;
- the original selected event as the anchor.

Exact duplicates are merged.

In practice, the 295 anchored spans have three or four candidates each.

The list, one fixed hand-written action, and all Top/Bottom answers are frozen before timing labels are loaded.

### Candidate headroom

The selected stream misses **96 of 292 serves** at ±10.

Of those 96 misses:

- **60** have a frozen prefix candidate within ±10;
- the counts are 18 on `sset_01`, 12 on `sset_15`, and 30 on `sset_21`;
- a timing oracle recovers **58** new serve matches;
- no existing serve match is lost;
- fully correct rallies rise **27 → 29** with no timing-score cut.

![Compact serve-prefix candidate headroom.](figures/followup_serve_prefix_headroom.png)

The compact list therefore passes its timing-headroom gate.

### Fixed chooser result

The fixed rule adds an earlier filtered-heuristic contact when it satisfies the rule.

| ±10 result | Selected event stream | Fixed serve chooser |
| --- | ---: | ---: |
| Predicted contacts | 3,238 | 3,317 |
| Matched serves | 196 | 204 |
| Serve recall | 67.1% | 69.9% |
| Unmatched added events | — | 70 |
| Fully correct at score 0.00 | 27 / 291 | 16 / 290 |
| Fully correct at score 0.90 | 13 / 68 | 9 / 49 |

The rule finds only **8** of the oracle's **58** recoverable serves.

With no timing-score cut it gains one fully correct rally but breaks 12 previously correct rallies.

![The fixed serve chooser damages whole-rally output despite the candidate list's oracle headroom.](figures/followup_serve_prefix_rally_effect.png)

### What the result means

Keep the **candidate-list idea** as a fresh-data research lead.

Do **not** use or tune the tested fixed chooser.

The current stored held-out HGB scores cannot be used to train a selector on these same outer test fixtures without leakage.

The compact prefix construction was chosen after inspecting aggregate results from these three fixtures. Treat it as development evidence that justifies a fresh-data test, not as a generalisation result.

## 12. The main warning: serves in `sset_21`

The weakest fixture should stay visible because it is easy for pooled scores to hide it.

With the original HGB decisions:

| Fixture | Timing F1 | Timing + correct-side recall | Serve timing recall | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 92.6% | 77.3% | 74.3% | 60.2% |
| `sset_15` | 84.2% | 76.5% | 76.9% | 67.3% |
| `sset_21` | **79.6%** | **70.7%** | **44.0%** | **34.7%** |

Region version 2 contains **71 / 75 = 94.7%** of `sset_21` serves.

The original HGB event decisions find **33 / 75**. The wider duplicate-removal decisions find **32 / 75**.

So the serve problem is mostly not “the region never looked there.”

With only three videos, we cannot tell whether `sset_21` represents a general broadcast/serve failure or simply a difficult fixture. The larger dataset is where that question becomes answerable.

## 13. What to carry into the larger dataset

Carry forward the **ideas**:

- region version 2 as the search-surface design;
- HGB with physical values plus validity flags as the simple baseline;
- whole-video splitting;
- the fixed strict definition of a fully correct kept rally;
- the lesson that decision-layer choices can matter;
- direct selected-stream player-side scoring;
- the lesson that candidate-list economy and oracle headroom answer different questions;
- the compact serve-prefix candidate construction as a bounded selector lead.

Choose again on the larger data:

- the HGB fit;
- score cut-offs;
- class and serve weights;
- frame-rate motion convention;
- negative sampling;
- duplicate-removal distance;
- any start-specific handling;
- any second-stage selector or player-side model.

Do not carry the fitted pilot tree, the 6-base-30-frame duplicate distance, or the failed hand-written serve chooser forward as production settings.

## Technical reference

### Main feature set

The physical + validity input has **85 columns**.

Physical values sampled at offsets −10, −5, 0, +5 and +10 base-30 frames:

- `shuttle_vx`
- `shuttle_vy`
- `shuttle_speed`
- `shuttle_impulse`
- `shuttle_impulse_ratio`
- `wrist_gap_min`
- `wrist_gap_top`
- `wrist_gap_bot`
- `nearest_wrist_dx`
- `nearest_wrist_dy`
- `ankle_speed_top`
- `ankle_speed_bot`

Validity flags at the same offsets:

- `shuttle_visible`
- `pose_valid_top`
- `pose_valid_bot`
- `wrist_valid_top`
- `wrist_valid_bot`

The optional context block has 20 columns. It did not improve HGB in this pilot.

### Training shape

The timing scorer uses three outer leave-one-fixture-out folds.

For each held-out fixture:

- the other two fixtures form the training side;
- inner out-of-fold predictions choose the probability threshold;
- the held-out fixture does not choose its threshold.

Training rows:

- positive through ±1 base-30 frame of a labelled contact;
- ignored through ±4;
- hard negatives through ±15;
- easy negatives sampled toward at most 12:1 negatives to positives.

### Region version 1 versus version 2

Using each region's original decisions:

| HGB physical metric | Region version 1 | Region version 2 |
| --- | ---: | ---: |
| Serve search coverage | 91.4% | **97.9%** |
| Timing precision | **87.5%** | 84.5% |
| Timing recall | 88.4% | **90.5%** |
| Timing F1 | **87.9%** | 87.4% |
| Timing + correct-side recall | 74.6% | **75.7%** |
| Joint event+side F1 | **74.1%** | 73.1% |
| Serve timing + correct-side recall | 52.7% | **56.2%** |

Region version 1 is a little more selective. Region version 2 keeps more real contacts, especially serves, available to the classifier. Since the region is a search surface rather than a final detector, use version 2.

### Boundary sensitivity

A court-view-only HGB refit reaches 87.7% timing F1 versus 87.4% for the main region-v2 HGB.

Only nine of the main HGB's 3,350 detections fall in the extra before-court rows.

The before-court rows mainly help **search coverage**, especially for serves; they are not secretly driving the HGB score.

## Compact reference for old experiment codes

Keep these only for reading old logs and result filenames.

| Code | Plain-language meaning |
| --- | --- |
| `B0` | Original score cut-off and 5-base-30-frame duplicate-removal distance |
| `T−` | One lower score cut-off everywhere |
| `N−` | Original score cut-off with duplicate-removal distance 4 |
| `N+` | Original score cut-off with duplicate-removal distance 6 |
| `S−` | Lower score cut-off only near label-blind detected rally starts |
| `L−` | Lower score cut-off only in the existing serve-lookback region |
| `SL−` | Lower score cut-off in both rally-start and serve-lookback regions |

Use the plain-language names in prose.

## Reproduction paths

Feature freezer:

```text
scratch/contact_det/scripts/freeze_tree_contact_features.py
```

Timing scorer:

```text
scratch/contact_det/scripts/score_tree_contact_detector.py
```

Player-side scorer:

```text
scratch/contact_det/scripts/score_contact_player_attribution.py
```

Strict rally scorer:

```text
scratch/contact_det/scripts/score_contact_rallies.py
```

Missed-contact audit:

```text
scratch/contact_det/scripts/analyse_contact_failures.py
```

Decision-layer scorer:

```text
scratch/contact_det/scripts/score_contact_decision_trials.py
```

Broad shortlist scorer:

```text
scratch/contact_det/scripts/score_contact_shortlist.py
```

Serve-lookback scorer:

```text
scratch/contact_det/scripts/score_contact_lookback_trials.py
```

Candidate-union ceiling scorer:

```text
scratch/contact_det/scripts/score_contact_candidate_union_ceiling.py
```

Serve-prefix scorer:

```text
scratch/contact_det/scripts/score_contact_serve_prefix.py
```

Key retained follow-up outputs:

```text
scratch/contact_det/raw/followups/phase1/
scratch/contact_det/raw/followups/phase2/
scratch/contact_det/raw/followups/phase3/
scratch/contact_det/raw/followups/lookback_trials/
scratch/contact_det/raw/followups/candidate_union_ceiling/
scratch/contact_det/raw/followups/serve_prefix/
```

## Pilot limits

The three fixtures are:

- `sset_01`
- `sset_15`
- `sset_21`

Together they contain:

- **292 rallies**
- **3,128 contacts**
- **292 serves**
- **2,836 non-serves**

All three fixtures come from the same dataset. This is a small, low-diversity pilot.

The results can tell us that an idea looks promising, unpromising, or worth measuring again. They cannot establish a final production threshold, frame-rate convention, duplicate-removal distance, selector quality, or generalisation result.

The compact serve-prefix construction and candidate-union oracle are development evidence. Any learned or tuned selector should be assessed on fresh whole videos or with a properly nested cross-fitting design.
