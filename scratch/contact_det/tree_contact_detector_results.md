# Contact detector experiments: what we tested and what we learned

This is the technical report for the contact-detector exploration.

It is written to answer the questions you normally have when returning to an experiment after doing several other things:

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
- [6. Cheap event-selection check](#6-cheap-event-selection-check)
- [7. Selected-stream player-side check](#7-selected-stream-player-side-check)
- [8. Serve-lookback threshold check](#8-serve-lookback-threshold-check)
- [9. Broad nearby-alternative shortlist](#9-broad-nearby-alternative-shortlist)
- [10. Full candidate-union rally ceiling](#10-full-candidate-union-rally-ceiling)
- [11. Compact serve-prefix candidate check](#11-compact-serve-prefix-candidate-check)
- [12. Cleanup headroom: are extra events a useful second-stage target?](#12-cleanup-headroom-are-extra-events-a-useful-second-stage-target)
- [13. Rally acceptance: is the weakest HGB score enough?](#13-rally-acceptance-is-the-weakest-hgb-score-enough)
- [14. The main warning: serves in sset_21](#14-the-main-warning-serves-in-sset_21)
- [15. What to carry into the larger dataset](#15-what-to-carry-into-the-larger-dataset)
- [Technical reference](#technical-reference)
- [Reproduction paths](#reproduction-paths)
- [Pilot limits](#pilot-limits)

## What the pilot was trying to learn

The end goal is not “find every contact at any cost.”

The useful system should return a worthwhile number of rallies that are correct from beginning to end, and abstain when it is not confident enough.

The contact work separates several jobs:

```text
label-blind search region
→ learned contact timing score
→ small event-selection layer
→ player-side attribution
→ optional cleanup / bounded rescue
→ whole-rally acceptance
```

The three-video pilot was meant to choose a sensible design. It was not meant to freeze a final tree, threshold, duplicate-removal distance, frame-rate convention or acceptance rule.

## 1. Search region: can we avoid searching the whole video?

### What were we trying to learn?

The old raw proposals make too many real contacts unreachable by any later classifier. At ±10 frames on a 30 fps-equivalent clock, or about one-third of a second, they cover only:

- **83.8% of non-serves**;
- **66.1% of serves**.

We wanted a broad deterministic search surface that kept almost every real contact available without scoring almost every video frame.

### What did we test?

Region version 2 is label-blind. It combines neighbourhoods around:

- current raw contact proposals;
- relaxed shuttle impulse and direction-change peaks;
- local shuttle-to-wrist minima;
- shuttle visibility changes;
- detected rally starts;
- scene starts;
- a 45-base-30-frame look-back before eligible court-view intervals.

The look-back exists because some serves happen in the close-up immediately before the broadcast returns to full court.

Ground-truth contacts are loaded only after the region is built.

### What happened?

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

These are search-coverage numbers. They say that a candidate exists nearby. They do not say the classifier accepts it.

### What does it mean?

Region version 2 is broad enough to use as the pilot search surface.

Most later misses happen **inside** the available candidate surface, so expanding search again is not the first obvious move.

### What should be tested on the larger dataset?

Remeasure the same region design on more varied videos.

Only build a separate off-court rescue search if outside-region misses are actually breaking a useful number of otherwise-complete rallies.

## 2. Tree model: can simple physical features beat the current heuristics?

### What did we test?

The main HGB feature set has **85 columns**:

- 60 physical values sampled at five time offsets;
- 25 validity/missingness flags at the same offsets.

The physical inputs include shuttle velocity, speed and impulse, wrist distance, nearest-wrist direction and player ankle motion.

We also tested:

- the same physical inputs plus 20 context features;
- context only;
- missingness only;
- random forest versions of the main feature sets.

The timing scorer uses leave-one-fixture-out evaluation. Threshold choice happens on the training side; the held-out fixture does not choose its own threshold.

### What happened?

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

### What does it mean?

HGB with physical values plus validity flags is the useful simple model.

Adding the tested context block did not help HGB. Missingness alone can get high recall only by predicting far too many events.

There is no strong reason to spend more pilot time tuning random forest.

### What should be tested on the larger dataset?

Refit HGB from scratch with whole-video splits, then use it as the simple baseline for any more complex detector.

## 3. Complete rallies: does better contact F1 turn into usable output?

### What were we trying to learn?

An event-level F1 score can look good while a rally is unusable.

One missed contact, one extra event or one wrong player side can spoil the whole rally.

### Fixed meaning of a fully correct kept rally

A kept span is fully correct only when:

- it maps to exactly one real rally;
- every labelled contact is found within the evaluation tolerance;
- there are no extra predicted contact events;
- every event has a Top/Bottom answer;
- every Top/Bottom answer is correct.

A missing side answer rejects the whole rally.

The main tolerance is ±10 base-30 frames. ±5 is a sensitivity check.

The simple whole-rally timing confidence used here is the **lowest** retained HGB score in the span.

### What happened with the original HGB decisions?

| Minimum whole-rally timing score | Spans kept | Fully correct at ±10 | Fully correct among kept |
| --- | ---: | ---: | ---: |
| 0.00 | 291 | 21 | 7.2% |
| 0.80 | 216 | 17 | 7.9% |
| 0.85 | 123 | 13 | 10.6% |
| 0.90 | 51 | 9 | **17.6%** |
| 0.95 | 11 | 1 | 9.1% |

### What does it mean?

Better contact timing has not yet turned into a clean automatic dataset.

Confidence filtering helps somewhat but does not produce a high-precision kept subset.

Complete-rally yield should stay the primary product score. Contact precision/recall/F1 are the diagnostics that explain why it moves.

![Whole-rally confidence versus yield.](figures/followup_rally_yield_curve.png)

## 4. Miss audit: are missed contacts outside the search or being discarded later?

### What were we trying to learn?

The original HGB event stream misses **296 of 3,128** contacts at ±10.

Before widening search or building another model, we wanted to know whether those contacts were:

- never available to HGB;
- present as lower-scoring candidates;
- lost when nearby peaks were collapsed;
- or lost in one-to-one event matching.

### What happened?

The 296 misses contain:

- **95 serves**;
- **201 ordinary exchanges**.

A candidate from the fixed search surface exists near **244 / 296** misses.

For serves, a candidate exists near **89 / 95** misses.

The strongest nearby candidate for each missed contact falls into these groups:

| What happened to the strongest nearby candidate | Missed contacts | Missed serves |
| --- | ---: | ---: |
| It scored below the HGB cut-off | **207** | **84** |
| It was removed as a nearby duplicate | 19 | 2 |
| It was retained but lost the one-to-one match | 18 | 3 |
| No candidate from the fixed search surface was present | 52 | 6 |

The underlying evidence is also usually present near a miss:

- shuttle evidence is nearby for **240 / 296** misses;
- pose evidence is nearby for **230 / 296**;
- wrist evidence is nearby for **230 / 296**.

The filtered contact stream from the old heuristic detector finds **103** of the HGB misses.

Most importantly for strict whole-rally output, only **13** spans are otherwise exact apart from one missing contact. All 13 have a region-v2 candidate nearby.

![Original HGB missed-contact audit.](figures/followup_missed_contact_audit.png)

### What does it mean?

The search region is not the first problem to expand.

Most misses already have a plausible candidate in the fixed score surface. Better selection is more interesting than immediately searching much more of the broadcast.

The 103-contact overlap made the old heuristic stream worth testing as a candidate source. The later broad-shortlist test was the useful decision point, and that shortlist was too noisy.

### Larger dataset

Repeat the same miss audit after refitting. Only add a separate rescue search if outside-region misses repeatedly break otherwise-good rallies.

## 5. Frame-rate motion check

### What were we trying to learn?

`sset_01` and `sset_15` are 25 fps. `sset_21` is 30 fps.

The original motion features use movement per video frame, so the same physical movement can produce a different number at a different frame rate.

### What did we test?

Three otherwise-matched HGB trials:

1. existing raw per-frame motion;
2. frame-rate-sensitive motion removed;
3. first and second differences converted to a common 30 fps scale.

The search regions, data folds, model settings, score-cut-off selection and duplicate-removal rule stayed the same.

### What happened?

| Motion treatment | Precision | Recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: |
| **Existing raw motion** | **84.5%** | **90.5%** | **87.4%** | 67.5% |
| Remove frame-rate-sensitive motion | 82.7% | 87.1% | 84.8% | 47.9% |
| Common 30 fps scale | 84.0% | 90.2% | 87.0% | **68.2%** |

Strict rallies:

| Minimum score | Raw motion | Remove motion | Common 30 fps scale |
| --- | ---: | ---: | ---: |
| 0.00 | **21 / 291** | 16 / 293 | 15 / 295 |
| 0.85 | **13 / 123** | 9 / 156 | 10 / 124 |
| 0.90 | **9 / 51** | 6 / 67 | 6 / 58 |

### What does it mean?

Raw motion won this pilot.

The common-scale trial found two extra serves overall, but lost pooled timing F1 and complete rallies.

Do **not** turn this into “raw motion is better in general.” There are only three videos and only one 30 fps fixture.

![Pilot frame-rate motion feature check.](figures/followup_motion_feature_check.png)

### Larger dataset

Retest the convention with more frame rates and more broadcast diversity.

## 6. Cheap event-selection check

### What were we trying to learn?

The miss audit showed that many real contacts already had nearby scored candidates.

Before refitting the model, we tested whether a tiny change in how scores become events could improve whole-rally output.

HGB itself was not refit.

### What did we test?

| Plain-language variant | What changed |
| --- | --- |
| Original decisions | original score cut-off; duplicate-removal distance 5 |
| Lower score cut-off everywhere | one lower score point; distance still 5 |
| Smaller duplicate distance | original score cut-off; distance 4 |
| Wider duplicate distance | original score cut-off; distance 6 |
| Lower score cut-off near rally starts only | lower cut-off only in the existing label-blind rally-start region |

Distances are in base-30 frames and scaled for source FPS.

### What happened?

| Decision variant | Predicted events | Precision | Recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original decisions | 3,350 | 84.5% | 90.5% | 87.4% | 67.5% |
| Lower score cut-off everywhere | 3,559 | 81.2% | **92.4%** | 86.4% | **73.6%** |
| Smaller duplicate distance | 3,704 | 76.6% | 90.7% | 83.1% | 67.5% |
| **Wider duplicate distance** | **3,238** | **87.2%** | 90.3% | **88.8%** | 67.1% |
| Lower score cut-off near rally starts only | 3,386 | 84.0% | 90.9% | 87.3% | 70.5% |

Strict fully-correct rallies:

| Decision variant | No minimum-score filter | 0.80 | 0.85 | 0.90 | 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original decisions | 21 / 291 | 17 / 216 | 13 / 123 | 9 / 51 | 1 / 11 |
| Lower score cut-off everywhere | 14 / 295 | 10 / 118 | 7 / 72 | 5 / 33 | 0 / 5 |
| Smaller duplicate distance | 7 / 291 | 5 / 187 | 4 / 84 | 4 / 31 | 0 / 7 |
| **Wider duplicate distance** | **27 / 291** | **23 / 231** | **19 / 146** | **13 / 68** | **1 / 14** |
| Lower score cut-off near rally starts only | 20 / 292 | 16 / 194 | 12 / 109 | 8 / 44 | 1 / 8 |

### What does it mean?

Wider duplicate removal is the best pilot event rule.

It removes more close-together peaks, improves precision, leaves recall almost unchanged and gives more fully correct rallies.

The exact six-base-30-frame distance is **not** a production constant.

![Cheap event-selection variants.](figures/followup_decision_layer_tradeoff.png)

### Larger dataset

Reselect the score cut-off and duplicate-removal distance from scratch. Judge the choice by the strict whole-rally curve, not timing F1 alone.

## 7. Selected-stream player-side check

### Why did we run it?

The contact tree predicts timing only. The existing Top/Bottom rule still has to be correct for a rally to be usable.

### What happened?

At ±10:

| Event stream | Timing recall | Side accuracy on answered timing matches | Timing + correct-side recall | Joint event-and-side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 79.3% | **89.0%** | 70.6% | 64.6% | 46.2% |
| Original HGB decisions | **90.5%** | 83.7% | **75.7%** | 73.1% | **56.2%** |
| **Selected wider-duplicate-removal stream** | 90.3% | 83.4% | 75.2% | **73.9%** | **56.2%** |

### What does it mean?

The selected event stream still improves combined timing-and-side output over the old heuristics.

But player-side attribution is not “done.” Its importance becomes more obvious in the later candidate-union oracle.

The separate model that checks whether Top/Bottom contacts alternate through a rally has **not** been rerun on the new contact events.

## 8. Serve-lookback threshold check

### What were we trying to learn?

Serves remain weak, particularly on `sset_21`.

Before inventing another model, we tested the cheapest possible change: lower the score threshold only in the existing label-blind serve-lookback region.

### What happened?

| Plain-language decision | Events | Serve timing matches | Correctly sided serves | Joint event-and-side F1 | Fully correct with no minimum-score filter |
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

All five events already appear in the later compact serve-prefix lists. Even with access to the labels, the oracle does not use any of them to recover a new serve.

### What does it mean?

Close this exact threshold idea. Another nearby cut on these same fixtures would just tune the same tiny effect.

## 9. Broad nearby-alternative shortlist

### What were we trying to learn?

A second-stage model only makes sense if the first stage can give it a reasonably compact list containing meaningfully more real contacts than the final event stream.

### What did we test?

Start with the selected wider-duplicate-removal stream.

For each selected event:

- keep the selected event;
- search the same interval within ±10 base-30 frames;
- add the strongest alternative outside the selected event's duplicate-removal distance;
- allow the alternative to be below the normal score cut-off;
- deduplicate the combined list.

Labels are used only after the list is frozen.

### What happened?

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

- the shortlist recovers **97 of 303** selected-stream misses;
- it adds **3,067** candidates;
- **2,970** of those added rows remain unmatched;
- that is about **31.6 added candidates per recovered contact**.

Before running the shortlist, we decided it needed to:

- recover at least **152** contacts;
- total shortlist size no more than twice the selected event stream.

The size condition passed. The recovery condition failed.

![Broad shortlist candidate cost.](figures/followup_shortlist_tradeoff.png)

### What does it mean?

Stop this exact shortlist as a practical second-stage input on the pilot.

The later cleanup analysis points to a better first job for a second stage: delete bad events before trying to add a broad set of new candidates.

## 10. Full candidate-union rally ceiling

### What were we trying to learn?

The broad shortlist failed its practical gate. We still wanted to know whether the same frozen union contained combinations that could make more whole rallies correct.

### What did we test?

An exact non-deployable oracle chooses from the frozen 6,305-candidate union after:

- candidate identities are fixed;
- unchanged rally-span membership is fixed;
- the shipped Top/Bottom answer for every candidate is fixed.

Of the 6,305 candidates, **6,252** fall inside the unchanged rally spans. The other 53 are excluded.

Two ceilings are measured:

1. timing-only feasible;
2. timing plus the fixed player-side answers feasible.

### What happened?

| Tolerance | Selected stream fully correct | Timing-only feasible | Timing + side feasible | Full gain |
| --- | ---: | ---: | ---: | ---: |
| ±10 | 27 | 144 | 42 | **+15** |
| ±5 | 24 | 105 | 37 | **+13** |

At ±10, the +15 full gains split as:

- +2 rallies on `sset_01`;
- +8 on `sset_15`;
- +5 on `sset_21`.

No fully correct selected-stream rally is lost.

We had decided in advance that the union needed to add at least ten fully correct rallies to be worth carrying forward. The oracle adds fifteen, so it clears that bar.

### What does it mean?

There is real selector headroom inside the candidate union.

But **42 is not achieved performance**. It is an upper bound.

The difference between **144 timing-feasible** and **42 timing-and-side-feasible** rallies makes the current player-side answers a major evidence limit.

![Candidate-union rally ceiling.](figures/followup_candidate_union_ceiling.png)

### Larger dataset

Any practical selector must be assessed on fresh whole videos or through properly nested cross-fitting.

## 11. Compact serve-prefix candidate check

### What were we trying to learn?

The general shortlist is noisy. The serve-specific test asks a narrower question:

> Can the frames before a detected span's first selected event supply one missed serve without changing the rest of the rally?

### What did we test?

Each detected span gets at most five frozen candidates:

- the three strongest raw HGB peaks in the prefix;
- the best filtered heuristic contact;
- the original selected event as the anchor.

Exact duplicates are merged.

The candidate list, hand-written choice and Top/Bottom answers were fixed before the timing labels were loaded.

### Candidate headroom

The selected stream misses **96 of 292 serves** at ±10.

Of those 96 misses:

- **60** have a frozen prefix candidate within ±10;
- a label-informed upper bound says that a perfect chooser could recover **58** new serve matches;
- no existing serve match is lost;
- fully correct rallies could rise **27 → 29** with no minimum-score filter.

### Fixed chooser result

| ±10 result | Selected event stream | Fixed serve chooser |
| --- | ---: | ---: |
| Predicted contacts | 3,238 | 3,317 |
| Matched serves | 196 | 204 |
| Serve recall | 67.1% | 69.9% |
| Fully correct at score 0.00 | 27 / 291 | 16 / 290 |
| Fully correct at score 0.90 | 13 / 68 | 9 / 49 |

The rule finds only **8** of the oracle's **58** recoverable serves and leaves 70 added events unmatched.

### Span-boundary check

The oracle chooses 61 serve candidates. Nineteen are earlier than the original detected-span start. The fixed chooser selects 79 candidates, and none are earlier.

The serves that make two extra rallies fully correct are both already inside their detected spans.

The serve search still runs once from the original detected-span start. For the 19 earlier choices, the saved output span starts at the serve while the original detected bounds stay unchanged.

Scoring those output spans leaves the oracle result at **29 fully correct rallies**.

### What does it mean?

Keep the **candidate-list idea** as a fresh-data research lead.

Do not use or tune the tested fixed chooser.

Train any serve selector on fresh videos. Using these three again would leak the test labels.

The compact prefix construction was designed after looking at this three-video pilot. Treat it as development evidence, not a generalisation result.

## 12. Cleanup headroom: are extra events a useful second-stage target?

### What were we trying to learn?

The broad shortlist experiment asked whether adding more candidates could recover enough missed contacts.

That leaves another, simpler question:

> How many rallies are already almost right and would become correct if we only deleted bad extra events?

This matters because the wider duplicate-removal rule made the output cleaner overall.

### What did we check?

This follow-up uses the saved strict span records for the **original HGB decision rule**.

For the wider duplicate-removal rule, we use the saved overall totals. The detailed span records are not needed for this comparison.

### What happened?

The two event lists differ as follows:

- **112 fewer** predicted events overall;
- **7 fewer** timing matches;
- **105 fewer** unmatched predictions;
- timing F1 **87.4% → 88.8%**;
- fully correct rallies **21 → 27**.

Most of the improvement is from the much larger fall in unmatched predictions.

Then the original HGB strict spans give this upper-bound picture.

#### Timing only

| Timing question | Predicted spans |
| --- | ---: |
| Already have exactly the right event count and all contacts matched | 50 |
| Could become timing-exact if an ideal selector only removed extra events | 108 |
| Could also repair one otherwise-exact missing event | up to 121 |

#### Complete timing plus player side

| Complete-output question | Predicted spans |
| --- | ---: |
| Fully correct now | 21 |
| Could become fully correct by removing extra events only | 38 |
| Could also repair the 13 otherwise-exact one-missing spans, if the new event has the correct side | up to 51 |

![Where the original HGB output has cleanup headroom.](figures/followup_cleanup_headroom.png)

The extra-event cleanup set contains **17** additional rallies beyond the current 21.

The one-missing set contains another **13**. All 13 already have a region-v2 candidate near the missing contact; six are also found by the filtered contact stream from the old heuristic detector.

Concrete target rows are retained at [`raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz`](raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz).

### What does it mean?

The broad shortlist is too noisy for a practical second stage. The cleanup results point to two clearer jobs:

1. delete extra events;
2. separately rescue a small set of otherwise-exact one-missing rallies.

The **up to 51** number is deliberately loose. It is not directly comparable with the **42-rally** candidate-union oracle because the two checks use different streams and constraints.

### Larger dataset

After cross-fitting the first-stage model, test a small cleanup stage first. Only then add one bounded rescue source if the one-missing pattern persists.

## 13. Rally acceptance: is the weakest HGB score enough?

### What were we trying to learn?

A high-precision standalone annotator needs to know when to abstain.

The simplest possible rule is “reject a whole rally if its weakest retained HGB event score is too low.”

We wanted to know whether that was enough.

### What happened?

For the original HGB output:

| Diagnostic slice | Spans in slice | Fully correct | Fully correct within slice |
| --- | ---: | ---: | ---: |
| All scorable predicted spans | 291 | 21 | 7.2% |
| Minimum HGB score at least 0.85 | 123 | 13 | 10.6% |
| Minimum HGB score at least 0.90 | 51 | 9 | 17.6% |
| At most 7 predicted events | 126 | 19 | 15.1% |
| At most 4 predicted events | 67 | 13 | 19.4% |
| Score at least 0.90 and at most 5 events | 35 | 8 | 22.9% |

These are **diagnostic slices, not proposed acceptance rules**.

Short predicted event lists are cleaner. This suggests that errors build up over longer rallies, but three videos are too little to separate rally length from fixture difficulty.

The same score filter also behaves differently across fixtures. A 0.90 filter keeps all six fully correct `sset_01` spans but only two of the fourteen fully correct `sset_15` spans.

### What does it mean?

One raw minimum score is not a universal rally-confidence rule.

Do **not** solve this by hard-rejecting long rallies. That would create an artificially short-rally dataset.

Instead, a rally-level acceptance model should interpret confidence in context.

Candidate inputs to test on held-out predictions include:

- minimum, median and lower-tail contact scores;
- number of predicted contacts in the rally;
- span duration;
- gaps or ambiguity between nearby candidate scores;
- how cleanly the predicted span maps to one real rally;
- player-side confidence and missing side answers.

Use the true labelled rally length only when reporting results. Show **correctness versus retention** separately for short, medium and long rallies.

### Larger dataset

Train and calibrate the acceptance stage only from held-out or cross-fitted first-stage predictions. Do not carry any numerical cut from this three-video diagnostic into the larger fit.

## 14. The main warning: serves in `sset_21`

With the original HGB decisions:

| Fixture | Timing F1 | Timing + correct-side recall | Serve timing recall | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 92.6% | 77.3% | 74.3% | 60.2% |
| `sset_15` | 84.2% | 76.5% | 76.9% | 67.3% |
| `sset_21` | **79.6%** | **70.7%** | **44.0%** | **34.7%** |

Region version 2 contains **71 / 75 = 94.7%** of `sset_21` serves.

The original HGB decisions find **33 / 75**. The wider duplicate-removal decisions find **32 / 75**.

So the serve problem is mostly not “the region never looked there.”

With only three videos, we cannot tell whether `sset_21` represents a repeatable broadcast/serve failure or simply a difficult fixture.

That question belongs on the larger dataset.

## 15. What to carry into the larger dataset

Carry forward the **ideas**:

- region version 2 as the search-surface design;
- HGB with physical values plus validity flags as the simple baseline;
- whole-video splitting;
- the fixed strict meaning of a fully correct kept rally;
- the lesson that event-selection choices matter;
- direct player-side scoring on the selected stream;
- the lesson that deleting extras is a substantial second-stage opportunity;
- one bounded rescue path for otherwise-exact one-missing rallies;
- the compact serve-prefix construction as a bounded selector lead;
- a rally-level acceptance model that uses the number of predicted contacts in the rally;
- correctness-versus-retention reporting across true labelled rally lengths.

Choose again on the larger data:

- the HGB fit;
- score cut-offs;
- class and serve weights;
- frame-rate motion convention;
- negative sampling;
- duplicate-removal distance;
- start-specific handling;
- any cleanup model;
- any rescue selector;
- any player-side model;
- any rally-acceptance threshold.

A clean experimental order is:

1. train and cross-fit the first-stage contact model;
2. choose the basic event-selection rule;
3. test a small cleanup stage that removes extras;
4. test one bounded rescue source for otherwise-exact one-missing rallies;
5. improve player side;
6. learn rally acceptance from held-out predictions;
7. test on genuinely different held-out videos or broadcast groups.

Do not carry the fitted pilot tree, the six-base-30-frame duplicate distance, the failed hand-written serve chooser or any numerical rally-confidence cut forward as production settings.

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

A court-view-only HGB refit reaches **87.7% timing F1** versus **87.4%** for the main region-v2 HGB.

Only nine of the main HGB's 3,350 detections fall in the extra before-court rows.

The before-court rows mainly help search coverage, especially for serves. They are not secretly driving the HGB score.

### Old experiment codes

Old logs and filenames still contain short codes. Use the plain-language descriptions in prose.

| Code | Plain-language meaning |
| --- | --- |
| `B0` | Original score cut-off and 5-base-30-frame duplicate-removal distance |
| `T−` | Lower score cut-off everywhere |
| `N−` | Original score cut-off with duplicate-removal distance 4 |
| `N+` | Original score cut-off with duplicate-removal distance 6 |
| `S−` | Lower score cut-off only near detected rally starts |
| `L−` | Lower score cut-off only in the existing serve-lookback region |
| `SL−` | Lower score cut-off in both rally-start and serve-lookback regions |

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

The rally-repair target builder is:

```text
scratch/contact_det/scripts/build_contact_followup_rally_targets.py
```

It writes:

```text
scratch/contact_det/raw/followups/rally_cleanup_targets/contact_followup_rally_targets.csv.gz
```

The source is the original HGB strict span record in `raw/followups/phase1/run_a/contact_rally_score.json.gz`. The output is an inspection and experiment-target list, not a training set.

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

The results can tell us that an idea looks promising, unpromising or worth measuring again. They cannot establish:

- near-100% kept-rally precision;
- a production threshold;
- a production duplicate-removal distance;
- a final frame-rate convention;
- selector quality on fresh data;
- generalisation to different broadcast conventions.

Any learned or tuned cleanup, rescue, side or acceptance stage should be assessed on fresh whole videos or with a properly nested cross-fitting design.
