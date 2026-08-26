# Random-forest and histogram-gradient-boosting contact detector experiments

This report contains the technical detail behind the contact-detector results.

The two learned models are random forest (RF) and histogram gradient boosting
(HGB).

It does not restate rally-segmentation results or the overall annotator scorecard. Those belong in [`auto_annotator_progress.md`](auto_annotator_progress.md).

## Table of contents

- [Experiment structure](#experiment-structure)
- [Search-region experiment](#search-region-experiment)
- [Exact tree inputs](#exact-tree-inputs)
- [Training and timing evaluation](#training-and-timing-evaluation)
- [Timing-only model sweep](#timing-only-model-sweep)
- [Player-side scoring](#player-side-scoring)
- [Strict complete-rally evaluation](#strict-complete-rally-evaluation)
- [Missed-contact audit](#missed-contact-audit)
- [Frame-rate motion check](#frame-rate-motion-check)
- [Decision-layer check](#decision-layer-check)
- [Serve-lookback threshold closure](#serve-lookback-threshold-closure)
- [Full candidate-union rally ceiling](#full-candidate-union-rally-ceiling)
- [Structured serve-prefix check](#structured-serve-prefix-check)
- [Region v1 versus region v2](#region-v1-versus-region-v2)
- [Boundary sensitivity](#boundary-sensitivity)
- [Where it still fails](#where-it-still-fails)
- [Recommendation](#recommendation)
- [Reproduction](#reproduction)

## Experiment structure

The work has three stages:

```text
deterministic search-region construction
→ RF/HGB contact scoring
→ existing Top/Bottom player-side attribution on frozen predicted events
```

The current **final** heuristic event list is only a comparison point; HGB does not use it as an input.

The current **raw** proposals are one of several inputs used to build the search region.

## Search-region experiment

### Why a new region was needed

The current raw proposals make too many real contacts unreachable by a later classifier.

At ±10 base-30 frames they cover only:

- **83.8% of non-serves**
- **66.1% of serves**

The search-region experiment therefore asks:

> **How much of the video can we discard while still leaving almost every contact available to the classifier?**

The three evaluation fixtures are the videos identified as `sset_01`,
`sset_15` and `sset_21`.

### Region-v2 construction

Region v2 is deterministic and label-blind.

It takes temporal neighbourhoods around:

- current raw contact proposals;
- relaxed shuttle impulse / direction-change peaks;
- local shuttle-to-wrist minima;
- shuttle visibility changes;
- detected rally starts;
- scene starts;
- a 45-base-30-frame look-back immediately before eligible court-view intervals.

The final item exists because some serves happen during the close-up immediately before the broadcast returns to the full court.

Ground truth is loaded only after a region has been constructed, to measure coverage.

### Coverage

Region v2 scores **128,824 candidate centres**, about **31.9% of 404,229 source frames**.

At ±10:

| Search surface | All contacts | Non-serves | Serves |
| --- | ---: | ---: | ---: |
| Court-view intervals only | 97.9% | 98.3% | 93.2% |
| **Region v2** | **98.3%** | **98.4%** | **97.9%** |

Per fixture:

| Fixture | Non-serve coverage | Serve coverage | All-contact coverage |
| --- | ---: | ---: | ---: |
| `sset_01` | 99.4% | 98.2% | 99.3% |
| `sset_15` | 100.0% | 100.0% | 100.0% |
| `sset_21` | 93.7% | 94.7% | 93.8% |

These are **search coverage** numbers, not model recall.

![Region v2 reduces the search workload while keeping almost every labelled contact reachable.](figures/region_v2_search_tradeoff.png)

Region v2 is intentionally permissive. An extra candidate frame only gives the classifier more work; a real contact left outside the region cannot be recovered.

## Exact tree inputs

The main `physics` feature set contains **85 columns**.

### Physical signals: 60 columns

Twelve signals sampled at offsets −10, −5, 0, +5 and +10 base-30 frames:

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

### Validity / missingness: 25 columns

Five signals at the same five offsets:

- `shuttle_visible`
- `pose_valid_top`
- `pose_valid_bot`
- `wrist_valid_top`
- `wrist_valid_bot`

### Optional context: 20 columns

The context block contains centre-frame absolute positions, ankle positions, bbox heights, standing count, interval timing, scene timing and seven region-source flags.

The four tested feature sets are therefore:

| Feature set | Columns |
| --- | ---: |
| `physics` | **85** |
| `physics_context` | **105** |
| `context_only` | **20** |
| `missingness_only` | **25** |

### PySceneDetect

PySceneDetect affects where candidates can occur through scene/court interval construction and the scene-start seed.

Scene timing is also present in the optional context block.

The winning 85-column HGB `physics` model does **not** receive scene timing or region-source flags as explicit inputs.

## Training and timing evaluation

The scorer uses three outer leave-one-fixture-out folds.

For each held-out fixture:

- the other two fixtures are the training side;
- inner out-of-fold predictions choose the probability threshold;
- the held-out fixture does not choose its threshold.

Training rows:

- positive through ±1 base-30 frame of a labelled contact;
- ignored through ±4;
- hard negatives through ±15;
- easy negatives sampled toward at most 12:1 negatives to positives.

The baseline temporal duplicate-removal distance is **5 base-30 frames**. This
operation is also called non-maximum suppression (NMS): when high-scoring
candidates are close together, it keeps the strongest one.

Metrics are reported at ±5, ±10 and ±15. The main comparison is ±10.

## Heuristic proposal and cleanup

Before the learned models, the existing heuristic path moves from a broad raw proposal stream to a cleaner final event list:

| Timing-only event stream at ±10 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Current raw proposals | 41.7% | **82.2%** | 55.3% |
| Current final heuristic cleanup | **66.9%** | 79.3% | **72.6%** |

The wrist/proximity cleanup greatly improves precision, but it also removes some real contacts. That is why the new path keeps the search region relaxed and lets the classifier do the cleanup.

## Timing-only model sweep

This table is **timing-only**.

| Event stream at ±10 | Precision | Recall | F1 | Non-serve recall | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 66.9% | 79.3% | 72.6% | 81.2% | 61.0% |
| **HGB physical** | **84.5%** | **90.5%** | **87.4%** | **92.9%** | **67.5%** |
| RF physical | 84.1% | 85.2% | 84.6% | 89.6% | 42.8% |
| HGB + context | 81.7% | 89.8% | 85.5% | 92.5% | 63.4% |
| RF + context | 83.9% | 86.5% | 85.2% | 90.4% | 47.9% |

Controls:

| Timing-only control | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| HGB context only | 47.0% | 79.6% | 59.1% |
| RF context only | 37.1% | 82.2% | 51.1% |
| HGB missingness only | 16.1% | 96.9% | 27.5% |
| RF missingness only | 16.0% | 96.9% | 27.5% |

The main conclusions from the sweep are:

- HGB is the strongest timing model;
- context hurts HGB;
- missingness alone only achieves high recall by producing far too many events;
- there is no clear reason to spend more time tuning RF.

## Player-side scoring

The tree does not predict player side.

For the existing heuristic, region-v1 and region-v2 event streams, the player-side scorer leaves every prediction frame, threshold and NMS decision unchanged, then applies `attribute_half` at each predicted frame.

The **eligible-court-only HGB** row is the one exception: it is a new full HGB refit on the eligible-court rows, including threshold selection and NMS. That entire event stream is still completed and frozen **before** player-side ground truth is loaded.

Before scoring side, the script checks that the original timing counts are unchanged.

At ±10:

| Event stream | Timing recall | Side accuracy given timing match | Timing + correct-side recall | Joint event+side precision | Joint event+side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 79.3% | **89.0%** | 70.6% | 59.6% | 64.6% | 46.2% |
| **HGB physical** | **90.5%** | 83.7% | **75.7%** | 70.7% | **73.1%** | **56.2%** |
| HGB + context | 89.8% | 81.9% | 73.4% | 66.8% | 69.9% | 46.2% |
| RF physical | 85.2% | 85.8% | 73.1% | **72.1%** | 72.6% | 37.0% |
| Eligible-court-only HGB | 89.1% | 84.1% | 74.8% | 72.5% | **73.7%** | 53.4% |

Adding side correctness narrows the gap:

- HGB is still the region-v2 tree model we would use;
- RF nearly catches HGB once correct player side is required;
- on timing matches, the side rule is right **83.7%** of the time for HGB and **89.0%** for the current final heuristics. We do not know whether the extra recovered contacts explain that difference.

For serves, the side rule is right on **84.1%** of HGB timing matches versus **75.8%** of current-final timing matches, while serve timing recall rises from **61.0% to 67.5%**.

A simpler check using centre-frame `wrist_gap_top` and `wrist_gap_bot` gives essentially the same HGB side result.

## Strict complete-rally evaluation

The event-level result does not tell us how often a whole rally is usable. One
missed contact, one extra event, or one wrong player side rejects the complete
output.

This section reports the original HGB decision settings, called B0 in the later
decision-layer check. The strict evaluator keeps the current 311 predicted
rally spans unchanged.
These time windows come from the existing upstream rally segmentation. The
evaluator uses the current region-v2 HGB events and replays the shipped
Top/Bottom rule at those fixed event frames. Timing and player-side labels load
only after the event scores, score cut-offs, NMS decisions, spans and side
predictions are fixed.

A kept predicted span is fully correct only when it maps to one real rally,
finds every contact, adds no extra event, and gives the correct side for every
contact. The main timing tolerance is ±10 base-30 frames. A ±5 result is kept
as a sensitivity check.

Timing confidence is the weakest retained HGB score in the span. A score cut
abstains on the whole span. It never drops one weak event and keeps the rest of
the rally.

| Minimum timing score | Kept spans | Fully correct at ±10 | Accuracy among kept | Fully correct at ±5 |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 291 | 21 | 7.2% | 19 |
| 0.80 | 216 | 17 | 7.9% | 15 |
| 0.85 | 123 | 13 | 10.6% | 11 |
| 0.90 | 51 | 9 | **17.6%** | 7 |
| 0.95 | 11 | 1 | 9.1% | 0 |

Confidence filtering does not produce a useful clean subset yet. The 0.90 cut
keeps only 51 predicted spans and nine are fully correct. The highest observed
accuracy is still below one in five.

For an exclusive summary, each span is assigned to the first checkpoint it
fails in the order shown below:

| First failed checkpoint | Predicted spans |
| --- | ---: |
| No predicted event | 16 |
| Events but no real rally | 31 |
| More than one real rally | 4 |
| Contact timing or event count | **210** |
| Player side after exact timing | 29 |
| Fully correct | 21 |

The 210 timing failures contain 101 spans with both missing and extra events,
58 with extra events only, and 51 with missing contacts only. A side answer is
missing in four spans. Three are timing failures and one contains no real
rally. Thirteen retained HGB events fall outside every current span and are
reported separately. These priority buckets are derived from overlapping
rejection flags. They are not causal labels emitted by the scorer.

## Missed-contact audit

The primary ±10 timing score matches 2,832 of 3,128 labelled contacts. The 296
misses split into 95 serves and 201 ordinary exchanges.

| Contact type | Labelled | Missed | Miss rate | Misses with a seeded candidate nearby |
| --- | ---: | ---: | ---: | ---: |
| Serve | 292 | 95 | **32.5%** | 89 |
| Ordinary exchange | 2,836 | 201 | 7.1% | 155 |
| **All contacts** | **3,128** | **296** | **9.5%** | **244** |

For each missed contact, the audit inspects every held-out HGB score inside the
fixed ±10 window. It ranks up to three peaks after applying the current
within-interval NMS distance, but it does not apply the score cut-off. This is
an observational, ground-truth-centred table. It does not define a deployable
shortlist.

The strongest nearby candidate for each miss is:

| Strongest nearby candidate | Missed contacts | Missed serves |
| --- | ---: | ---: |
| Below the per-fold probability cut-off | **207** | **84** |
| Removed as a nearby duplicate | 19 | 2 |
| Retained, but lost the one-to-one match | 18 | 3 |
| No seeded candidate in the window | 52 | 6 |

The raw nearby-decision flags overlap. For example, one missed contact may
have both a retained event and lower-scoring rows in the same window. The table
above avoids double counting by classifying only the strongest candidate.

Most misses have usable frozen evidence somewhere in the audit window. Shuttle
evidence is present for 240 of 296 misses. Pose and wrist evidence are each
present for 230. The evidence check is deliberately simple: it reports whether
each stream is valid on any frozen row in the window. It does not prove that
all streams are valid together at the strongest peak.

The filtered handcrafted rule finds 103 of the 296 HGB misses. That overlap
could support one bounded merge test after the shortlist gate. It is not
evidence for replacing HGB because the handcrafted stream also has many misses
and false alarms of its own.

Only 13 spans are otherwise exact apart from one missing contact. All 13
missing contacts already have a seeded region-v2 candidate nearby. Eleven are
best represented by a below-cut-off row, one was removed as a duplicate, and
one loses the one-to-one match despite a retained event. Eight are serves and
six are found by the handcrafted rule.

This supported three Phase 2 decisions:

- test frame-rate-normalised motion features because the fixtures mix 25 and 30 fps
- test a small decision-layer set, including start-specific score handling
- keep physical values plus validity as the baseline for focused feature checks

The next two sections report the completed frame-rate and decision checks.

Search outside the region-v2 candidate surface stays deferred. Among those 13
otherwise-exact spans, no missing contact is outside region v2. A second
learned cleanup stage also stays deferred until a label-blind shortlist has
measured coverage and false-alarm cost.

If that later shortlist gate passes, the 103 HGB misses found by the
handcrafted rule support one simple merge before any cleanup tree.

## Frame-rate motion check

Two fixtures run at 25 frames per second (fps), while `sset_21` runs at 30 fps.
The original motion values measure movement per video frame. This means that
the same movement in the scene can have a different numeric value at different
frame rates.

Two focused trials tested whether that difference was hurting the selected
HGB model:

1. remove shuttle velocity, shuttle speed, shuttle impulse and ankle speed;
2. express those values on a common 30 fps scale.

The corrected freeze multiplies first differences, such as velocity, by
`fps / 30`. Shuttle impulse is a second difference, so it uses the square of
that factor. The dimensionless impulse ratio stays unchanged. All search
regions, folds, fitting settings, score selection and duplicate removal stay
the same.

| HGB physical trial | Timing precision | Timing recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: |
| **Existing raw motion** | **84.5%** | **90.5%** | **87.4%** | 67.5% |
| Remove raw motion | 82.7% | 87.1% | 84.8% | 47.9% |
| Common 30 fps scale | 84.0% | 90.2% | 87.0% | **68.2%** |

The common-scale trial finds two more serves overall, but its pooled timing F1
is lower. The complete-rally result is also lower at every reported confidence
setting:

| Minimum timing score | Raw motion, fully correct / kept | Remove motion | Common 30 fps scale |
| --- | ---: | ---: | ---: |
| 0.00 | **21 / 291** | 16 / 293 | 15 / 295 |
| 0.85 | **13 / 123** | 9 / 156 | 10 / 124 |
| 0.90 | **9 / 51** | 6 / 67 | 6 / 58 |

The existing raw-motion model therefore remains the pilot baseline. The three
fixtures do not support removing or rescaling these values. This does not prove
that raw per-frame motion is generally best. The feature convention must be
retested when the planned larger video set is available.

## Decision-layer check

The decision layer turns held-out HGB scores into contact events. This check
replayed five choices without refitting the model. Every event stream and
Top/Bottom answer was fixed before timing or player-side labels loaded.

The set contains one baseline and four one-choice variants:

- **B0:** original per-fold score cut-off and duplicate distance 5;
- **T−:** the next lower point in the existing score grid;
- **N−:** original score cut-off and duplicate distance 4;
- **N+:** original score cut-off and duplicate distance 6;
- **S−:** the lower score point only inside `region_rally_start`, the existing
  label-blind window around detected rally starts.

The duplicate distances use base-30 frames. They become 4, 5 and 6 frames for
N−, B0 and N+ on 30 fps video. On 25 fps video they become 3, 4 and 5 frames.

| Decision row | Pooled predicted events | Timing precision | Timing recall | Timing F1 | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 3,350 | 84.5% | 90.5% | 87.4% | 67.5% |
| T− | 3,559 | 81.2% | **92.4%** | 86.4% | **73.6%** |
| N− | 3,704 | 76.6% | 90.7% | 83.1% | 67.5% |
| **N+** | **3,238** | **87.2%** | 90.3% | **88.8%** | 67.1% |
| S− | 3,386 | 84.0% | 90.9% | 87.3% | 70.5% |

The strict full-rally result tells the same story:

| Decision row | Fully correct / kept at 0.00 | At 0.80 | At 0.85 | At 0.90 | At 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 21 / 291 | 17 / 216 | 13 / 123 | 9 / 51 | 1 / 11 |
| T− | 14 / 295 | 10 / 118 | 7 / 72 | 5 / 33 | 0 / 5 |
| N− | 7 / 291 | 5 / 187 | 4 / 84 | 4 / 31 | 0 / 7 |
| **N+** | **27 / 291** | **23 / 231** | **19 / 146** | **13 / 68** | **1 / 14** |
| S− | 20 / 292 | 16 / 194 | 12 / 109 | 8 / 44 | 1 / 8 |

Use N+ for the rest of this pilot. It has the best pooled timing F1 and the
most fully correct rallies through the 0.90 confidence setting. The lower
score rows recover more contacts and serves, but their extra events make more
complete rallies fail.

This is a three-fixture pilot choice. N+ slightly lowers pooled serve recall
from 67.5% to 67.1%. On `sset_21`, serve recall falls from 44.0% to 42.7%, so
the wider duplicate distance does not solve that fixture's serve problem.

## Serve-lookback threshold closure

The serve-lookback follow-up lowered the score only inside the existing
`region_serve_lookback` rows. L− applies that change to B0. SL− combines it
with the existing S− rally-start lowering. This isolates whether the lookback
region adds useful serve evidence without changing HGB or trying a new rule.

Every event stream and Top/Bottom answer was frozen before timing and side
labels loaded. The two comparisons add the same five events:

| Decision row | Events | Serve timing matches | Correctly sided serves | Joint event-and-side F1 | Fully correct at 0.00 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 3,350 | 197 | 164 | 73.1% | 21 |
| L− | 3,355 | 199 | 164 | 73.0% | 21 |
| S− | 3,386 | 206 | 169 | 72.9% | 20 |
| SL− | 3,391 | 208 | 169 | 72.9% | 20 |
| **N+** | **3,238** | 196 | 164 | **73.9%** | **27** |

L− and SL− each gain two serve timing matches. Neither gains a correctly
sided serve or a fully correct rally. The extra events lower joint
event-and-side F1 slightly. All five additions were already present in the
serve-prefix candidate lists, and none was the oracle choice for a newly
recovered serve.

Close this threshold idea. Another threshold point would tune the same
five-event effect on these fixtures.

This run also supplies the previously missing N+ event-side table at ±10. N+
has **83.4%** side accuracy on answered timing matches, **75.2%**
timing-plus-correct-side recall, **73.9%** joint event-and-side F1 and **56.2%**
serve timing-plus-correct-side recall.

## Label-blind shortlist check

The shortlist tested whether the held-out score surface contains enough nearby
alternatives to justify a second stage. Every identity was fixed before timing
labels loaded.

For each N+ event, the shortlist kept that event and the strongest row from the
same fixture and search interval within ±10 base-30 frames. The alternative had
to lie strictly outside N+'s duplicate-removal distance. It could fall below
the score cut-off. Equal scores used the earlier frame. The union was deduplicated
by fixture, interval and frame.

This produced 6,305 candidates from 3,238 N+ events. The list added 3,067 rows,
or 1.95 times the N+ size in total.

| Measure | N+ | Shortlist |
| --- | ---: | ---: |
| Candidates | 3,238 | 6,305 |
| Matched contacts at ±10 | 2,825 | 2,922 |
| Contact coverage at ±10 | 90.3% | 93.4% |
| Unmatched candidates at ±10 | 413 | 3,383 |
| Serve coverage at ±10 | 67.1% | 73.3% |
| Matched contacts at ±5 | 2,710 | 2,823 |
| Contact coverage at ±5 | 86.6% | 90.2% |
| Unmatched candidates at ±5 | 528 | 3,482 |
| Serve coverage at ±5 | 55.5% | 65.1% |

At the primary ±10 tolerance, the shortlist recovered 97 of N+'s 303 missed
contacts. It lost none of N+'s existing matches. The gains were 48 on
`sset_01`, 32 on `sset_15` and 17 on `sset_21`.

The rule fixed before scoring required at least 152 recovered contacts and a
list no larger than twice N+. The size passed. The 97-contact gain did not.
Each recovery cost 31.6 added candidates, and 2,970 of the added rows remained
unmatched. Those rows measure shortlist burden. They are not detector false
positives because the shortlist is deliberately overcomplete.

Stop Phase 3 here. Do not test a handcrafted merge, hard-negative refit or
cleanup tree on these three fixtures. The shortlist does not show enough compact
headroom to justify them.

## Full candidate-union rally ceiling

The shortlist result above measures event coverage. A separate exact oracle
asks whether some subset of the same frozen union could make a complete rally
correct. This is a ceiling on the evidence already present, not a deployable
merge or selector.

Candidate identities, existing half-open span membership and shipped
Top/Bottom answers were fixed before labels loaded. The union contains 6,305
distinct candidates. Existing spans contain 6,252 of them. The remaining 53
stay unassigned and cannot be used by this check.

For each clean one-rally span, the oracle searches candidate subsets exactly.
One mode requires exact event count and a timing match for every labelled
contact. The full mode also requires the unchanged production matcher to pair
every selected event with the correct shipped Top/Bottom answer. Timing-linked
components are searched separately, then the combined choice is checked again
with the unchanged full-span evaluator.

| Tolerance | N+ fully correct | Timing-only feasible | Timing + side feasible | Full gain |
| --- | ---: | ---: | ---: | ---: |
| ±10 | 27 | 144 | 42 | **+15** |
| ±5 | 24 | 105 | 37 | **+13** |

At ±10, the 15 full gains split as two on `sset_01`, eight on `sset_15` and
five on `sset_21`. No fully correct N+ identity is lost. This passes the
predeclared gate of at least ten new fully correct rallies.

The timing-only ceiling is much higher than the full ceiling: 144 versus 42
at ±10. This makes shipped player attribution the main evidence limit inside
the union. It does not prove that a practical selector can reach 42. A selector
would need fresh whole-video evaluation or nested cross-fitting. The result
does not reopen the failed handcrafted merge, hard-negative refit or cleanup
tree on these fixtures.

## Structured serve-prefix check

The general shortlist above is too large. A separate serve-only check asked a
narrower question:

> Can the frames before a detected span's first N+ event supply one missed
> serve without changing the rest of the rally?

This does not reopen the general merge. Each detected span gets a list of at
most five frozen candidates. In practice, the 295 anchored spans had three or
four candidates each. The list contains the three strongest raw HGB peaks in
the prefix, the best filtered heuristic contact and the original N+ anchor.
Exact duplicates are merged.

The candidate list, one fixed rule and every Top/Bottom answer were frozen
before timing labels loaded. The fixed rule adds the best filtered heuristic
contact when it is earlier than the anchor and outside N+'s duplicate distance.
A timing oracle then measures the best possible use of the same frozen list. It
only acts when N+ misses the serve. The oracle is a headroom check, not a model
that can be deployed.

The compact list does contain useful serve candidates:

- N+ misses 96 of the 292 serves at ±10;
- 60 of those 96 misses have a frozen prefix candidate within ±10;
- the counts are 18 on `sset_01`, 12 on `sset_15` and 30 on `sset_21`;
- the timing oracle recovers 58 new serve matches and loses none;
- the oracle raises fully correct rallies from 27 to 29 with no timing-score
  cut, without losing a previously correct rally.

That passes the predeclared headroom gate. It shows that a compact label-blind
serve chooser is possible in principle.

The fixed heuristic rule fails:

| ±10 result | N+ | Fixed serve rule |
| --- | ---: | ---: |
| Predicted contacts | 3,238 | 3,317 |
| Matched serves | 196 | 204 |
| Serve recall | 67.1% | 69.9% |
| Unmatched added events | — | 70 |
| Fully correct at score 0.00 | 27 / 291 | 16 / 290 |
| Fully correct at score 0.90 | 13 / 68 | 9 / 49 |

The rule finds eight new serves, or 13.8% of the oracle's 58. It gains one
fully correct rally and makes 12 previously correct rallies fail at a zero
score requirement. At 0.90 it loses four and gains none. Accuracy among kept
rallies falls at both settings.

Do not use or tune this fixed rule. The candidate list has measured headroom,
but choosing from it remains the problem. A learned chooser would need fresh
whole-video confirmation or nested HGB cross-fitting. The current stored
held-out scores cannot train that chooser without leaking information from the
outer test fixture.

This remains development evidence. The prefix was chosen after aggregate
inspection of these three fixtures. The result justifies a separately scoped
fresh-video test. It does not establish generalisation.

## Region v1 versus region v2

This comparison uses each region's original decision settings. The region
choice is a simple trade-off: v1 is more selective; v2 keeps more contacts
reachable.

| HGB physical metric | Region v1 | Region v2 |
| --- | ---: | ---: |
| Serve search coverage | 91.4% | **97.9%** |
| Timing precision | **87.5%** | 84.5% |
| Timing recall | 88.4% | **90.5%** |
| Timing F1 | **87.9%** | 87.4% |
| Timing + correct-side recall | 74.6% | **75.7%** |
| Joint event+side F1 | **74.1%** | 73.1% |
| Serve timing + correct-side recall | 52.7% | **56.2%** |

Region v1 is slightly more selective.

Region v2 keeps more contacts, especially serves, available to the classifier. Since the region's job is search-space reduction rather than final event prediction, use v2 as the search surface.

## Boundary sensitivity

This check also uses the original B0 duplicate distance. We refit HGB using
eligible court-view rows only, to check whether the extra rows just before
court view were somehow driving the result.

| HGB physical search | Timing precision | Timing recall | Timing F1 | Timing + correct-side recall | Joint event+side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main region v2 | 84.5% | **90.5%** | 87.4% | **75.7%** | 73.1% | **56.2%** |
| Eligible court only | **86.3%** | 89.1% | **87.7%** | 74.8% | **73.7%** | 53.4% |

Only nine of the main HGB's 3,350 detections are in the extra before-court rows.

Those rows help **search coverage**, especially for serves, but HGB scores almost the same without them.

## Where it still fails

### `sset_21`

HGB physical with the original B0 decisions at ±10:

| Fixture | Timing F1 | Timing + correct-side recall | Serve timing recall | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 92.6% | 77.3% | 74.3% | 60.2% |
| `sset_15` | 84.2% | 76.5% | 76.9% | 67.3% |
| `sset_21` | **79.6%** | **70.7%** | **44.0%** | **34.7%** |

Region v2 contains **71 / 75 = 94.7%** of `sset_21` serves, while HGB finds **33 / 75 = 44.0%**. Four serves sit outside region v2, but most misses happen after search. With only three fixtures, we cannot tell whether `sset_21` is simply different from the other two or whether the model has fit the other two too closely.

The selected N+ decision layer finds 32 of the 75 serves, or 42.7%. Its pooled
timing result is stronger, but it does not improve the main fixture warning.

### Contacts outside region v2

The 37 `sset_21` non-serves outside region v2 all have visible shuttle evidence near the labelled contact, but none has sticky player analysis or a player pick and none lies inside a detected rally span.

That is an off-court/live-close-up search problem.

### Very broad shuttle-only check

A deliberately broad shuttle-only check searches **90.6% of the broadcasts** and covers every non-serve plus 291 of 292 serves at ±10.

That shows the shuttle evidence exists, but the search is far too broad to use and often lacks usable player evidence.

## Other useful experiments

Three other decisions are worth keeping.

**Ankle side attribution.** On current-final timing matches, the shipped box/net rule scores **2,207 / 2,480 = 89.0%** correct Top/Bottom answers. Replacing the final half test with ankle height gives **2,208 / 2,481 = 89.0%**. That is effectively no change.

**Pull request 88 (PR88) server rule.** PR88 used shuttle motion before and after a chosen contact to decide whether it looked like a serve or return. When that was unclear, it fell back to the older pull request 82 (PR82) alternation answer. It got the server side right on **170 / 239 = 71.1%** of an older fixed 239-rally development set. This is not directly comparable with the 292-rally evaluation used here. On a later held-out test it tied the older rule overall and got worse on one video, so we did not keep it.

**X3D Small (X3D-S) video model.** Leave it aside for now. RGB may contain useful contact clues that pose and shuttle do not. Revisit it only if the failure cases point to missing visual information.

## Recommendation

Use:

- **region v2** as the search region;
- **HGB physical + validity** with the existing raw-motion features;
- **N+**, the 6-base-30-frame duplicate-removal distance, as the pilot decision
  layer.

Do not add the tested context block to HGB.

The raw model's timing F1 is 87.4%. The selected decision layer raises it to
88.8% without refitting. The strict rally result remains too weak for
deployment by confidence filtering alone. The label-blind shortlist missed its
coverage gate, so do not add the simple merge or cleanup tree on this pilot.
The serve-lookback threshold adds no correctly sided serves or complete
rallies, so close it. The compact serve-prefix list and full candidate-union
ceiling both pass their headroom gates, but neither supplies a validated
chooser. Carry that evidence into a fresh-video or nested cross-fitted selector
experiment. Do not tune another rule on these three fixtures.

For selected N+, the **88.8% timing F1** is only the timing score. Its complete
event-and-side results are:

- **75.2% timing + correct-side recall**
- **73.9% joint event-and-side F1**
- **56.2% serve timing + correct-side recall**

For the annotator-level result, see [`auto_annotator_progress.md`](auto_annotator_progress.md).

## Reproduction

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

Decision-layer scorer:

```text
scratch/contact_det/scripts/score_contact_decision_trials.py
```

Serve-lookback scorer:

```text
scratch/contact_det/scripts/score_contact_lookback_trials.py
```

Saved serve-lookback results:

```text
scratch/contact_det/raw/followups/lookback_trials/contact_lookback_trials_a.json.gz
scratch/contact_det/raw/followups/lookback_trials/contact_lookback_trials_b.json.gz
```

Candidate-union ceiling scorer:

```text
scratch/contact_det/scripts/score_contact_candidate_union_ceiling.py
```

Saved candidate-union ceiling results:

```text
scratch/contact_det/raw/followups/candidate_union_ceiling/contact_candidate_union_ceiling_a.json.gz
scratch/contact_det/raw/followups/candidate_union_ceiling/contact_candidate_union_ceiling_b.json.gz
```

Serve-prefix scorer:

```text
scratch/contact_det/scripts/score_contact_serve_prefix.py
```

Saved serve-prefix result:

```text
scratch/contact_det/raw/followups/serve_prefix/contact_serve_prefix_score_a.json.gz
```

Missed-contact audit:

```text
scratch/contact_det/scripts/analyse_contact_failures.py
```

Summary figures:

```text
scratch/contact_det/scripts/plot_summary_figures.py
```

Saved region-v2 timing result:

```text
scratch/contact_det/raw/region_v2/tree_contact_results.json.gz
```

Saved player-side result:

```text
scratch/contact_det/raw/contact_player_attribution_score.json.gz
```
