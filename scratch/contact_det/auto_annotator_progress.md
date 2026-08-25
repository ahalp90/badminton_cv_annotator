# Auto-annotator progress after the contact-detector experiments

This report is organised by **annotator output**.

It does not explain the RF/HGB feature engineering or training procedure. Those details live in [`tree_contact_detector_results.md`](tree_contact_detector_results.md).

## Table of contents

- [Current scorecard](#current-scorecard)
- [Rally identification](#rally-identification)
- [Strict complete-rally result](#strict-complete-rally-result)
- [Contact timing and contact player side](#contact-timing-and-contact-player-side)
- [Serve timing and serving side](#serve-timing-and-serving-side)
- [Rally-level server attribution](#rally-level-server-attribution)
- [What the new contact stream changes](#what-the-new-contact-stream-changes)
- [Recommendation](#recommendation)
- [Metric definitions](#metric-definitions)
- [Data and limits](#data-and-limits)

## Current scorecard

The outputs should stay separate.

| Output | Current result | Best result from this work | What changed |
| --- | ---: | ---: | --- |
| Rally segmentation | 77.3% clean one-rally F1 with no padding cap | unchanged | RF/HGB not rerun through span finder |
| Fully correct kept rally | not previously measured | **21 / 291 at no score cut; 9 / 51 at a 0.90 cut** | fixed HGB events and side answers scored end to end at ±10 base-30 frames |
| Contact timing | 72.6% F1 | **87.9% region-v1 HGB F1** | chosen region-v2 HGB reaches 87.4% |
| Contact timing + correct player side | 70.6% recall | **75.7% HGB recall** | side rule scored on frozen events |
| Serve timing | 61.0% recall | **67.5% HGB recall** | first-contact subset |
| Serve timing + correct serving side | 46.2% recall | **56.2% HGB recall** | side rule scored on frozen serves |
| Rally-level server side | 64.9% accuracy on answered rallies | unchanged | not rerun on HGB events |

The biggest improvement is contact timing. The complete contact+side output improves too, although the side rule is less accurate on HGB's timing-matched contacts than on the current-final matches.


## Rally identification

Nothing in the tree work changed the rally-span finder.

Across the three fixtures:

- **292 labelled rallies**
- **311 predicted spans**
- **233 clean one-to-one matches** with no cap on extra padding

A clean match is still strict: the predicted span must contain **every labelled contact from exactly one rally and no labelled contact from another rally**. The seconds below do not relax that rule. They only cap how much extra time the predicted span may include before the rally's first contact and after its last contact.

| Maximum extra padding beyond the first/last contact | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 1 second | 1.6% | 1.7% | 1.7% |
| 2 seconds | 15.1% | 16.1% | 15.6% |
| 3 seconds | 44.4% | 47.3% | 45.8% |
| 5 seconds | 62.4% | 66.4% | 64.3% |
| No padding cap | 74.9% | 79.8% | 77.3% |

With no padding cap, the same clean one-rally containment rule gives **77.3% F1**.

![Rally-span quality by extra padding cap.](figures/rally_segmentation_quality.png)

The new contact stream has now been scored inside these fixed spans. The span finder itself is still unchanged.

## Strict complete-rally result

The earlier 77.3% rally-span F1 answers a boundary question. It checks whether
a predicted span contains one complete labelled rally and no contact from
another rally. It does not check the predicted contact list or player sides.

The new strict score checks the complete output. A predicted span is fully
correct only when:

- it maps to exactly one real rally
- every contact is found within ±10 base-30 frames
- no extra event remains
- every event has a Top/Bottom answer
- every Top/Bottom answer is correct

The system may abstain on a whole predicted span. Its timing confidence is the
lowest held-out HGB score among the retained events in that span. Player-side
confidence remains the current binary answered-or-missing signal.

| Minimum timing score | Predicted spans kept | Fully correct | Fully correct among kept |
| --- | ---: | ---: | ---: |
| 0.00 | 291 | 21 | 7.2% |
| 0.80 | 216 | 17 | 7.9% |
| 0.85 | 123 | 13 | 10.6% |
| 0.90 | 51 | 9 | **17.6%** |
| 0.95 | 11 | 1 | 9.1% |

These are kept predicted spans, not automatically valid rallies. At a zero
score cut, 34 kept spans do not map to one real rally. The curve also shows
that timing confidence alone is a weak rejection rule. A 0.90 cut discards
most output and leaves only nine fully correct spans.

At the stricter ±5 timing tolerance, 19 spans are fully correct at a zero
score cut and seven are fully correct at 0.90.

For an exclusive summary, each span is assigned to the first checkpoint it
fails in the order shown below:

| First failed checkpoint at ±10 | Predicted spans |
| --- | ---: |
| No predicted event | 16 |
| Events but no real rally | 31 |
| More than one real rally | 4 |
| Contact timing or event count | **210** |
| Player side after exact timing | 29 |
| Fully correct | 21 |

Among the 210 timing failures, 101 have both missing and extra events. Another
58 have extra events only, while 51 have missing contacts only. These groups
are derived from overlapping rejection flags. They describe where the
evaluation rules first fail, not a physical cause for the model error.

## Contact timing and contact player side

These are two different questions:

1. **Did we find the contact at the right time?**
2. **For a timing-matched contact, did the existing Top/Bottom rule identify the correct player side?**

At ±10:

| Event stream | Timing precision | Timing recall | Timing F1 | Player-side accuracy given timing match | Timing + correct-side recall | Joint event+side F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 66.9% | 79.3% | 72.6% | **89.0%** | 70.6% | 64.6% |
| **Region-v2 HGB** | **84.5%** | **90.5%** | **87.4%** | 83.7% | **75.7%** | **73.1%** |
| Region-v2 RF | 84.1% | 85.2% | 84.6% | 85.8% | 73.1% | 72.6% |

![Contact timing recall versus the fraction of all contacts that are both found and attributed correctly.](figures/contact_output_recall.png)

On timing-matched contacts, the side rule is right **83.7%** of the time for HGB and **89.0%** for the current final heuristics.

For non-serves:

- current side accuracy on timing matches: **90.0%**
- HGB side accuracy on timing matches: **83.7%**

But HGB raises non-serve timing recall from **81.2% to 92.9%**, so the share of all non-serves with both time and side right still rises from **73.1% to 77.7%**.

That leaves player-side attribution as the next obvious weak point.

## Serve timing and serving side

A serve here means the **first labelled contact in a rally**. The tree does not predict a special serve class.

At ±10:

| Event stream | Serve timing recall | Serving-side accuracy given timing match | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: |
| Current final heuristics | 61.0% | 75.8% | 46.2% |
| **Region-v2 HGB** | **67.5%** | 84.1% | **56.2%** |
| Region-v2 RF | 42.8% | **86.4%** | 37.0% |

![Serve timing recall versus the fraction of all serves that are both found and attributed correctly.](figures/serve_output_recall.png)

HGB finds more serves, and the side rule is also more accurate on the serves it finds.

The fixture split shows the remaining problem:

| Fixture | HGB serve timing recall | Serving-side accuracy when matched | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: |
| `sset_01` | 74.3% | 81.0% | 60.2% |
| `sset_15` | 76.9% | 87.5% | 67.3% |
| `sset_21` | **44.0%** | 83.9% | **34.7%** |

On `sset_21`, the side rule is reasonable once a serve is found; the main problem is that HGB finds only **44.0%** of the serves.

## Rally-level server attribution

This is separate from the per-serve side table above.

The per-serve table asks:

> “For this detected first contact, did the direct side rule identify the correct serving side?”

The rally-level logic asks:

> “After fitting one strict Top/Bottom alternation phase across the whole rally, what server side does that imply?”

On the current final-heuristic event stream:

- final hitter side: **112 / 228 = 49.1%**
- server side: **148 / 228 = 64.9%**
- 228 of 292 rallies answered

The alternation fit has **not** been rerun on HGB contacts.

Better contact recall may help because a missed contact can flip the parity of every later stroke. We have not rerun the alternation fit yet.


## What the new contact stream changes

The main change is simple: many fewer contacts are missed.

Before the tree experiment, many errors came from **not finding the contact at all**.

HGB removes a substantial part of that problem:

- contact timing recall: **79.3% → 90.5%**
- non-serve timing recall: **81.2% → 92.9%**
- serve timing recall: **61.0% → 67.5%**

The side rule is right less often on HGB's timing-matched contacts: **83.7%** versus **89.0%** for the current final heuristics.

So contact timing and player-side attribution should keep being measured separately.

## Recommendation

For the current auto-annotator:

- use **region v2** as the search region;
- use **HGB physical + validity** as the simple learned contact model;
- treat HGB as the Phase 2 baseline, not as a ready complete-rally output;
- always report timing and timing+correct-side metrics together;
- do not treat **87.4% timing F1** as the score for the complete contact+side output;
- test the cheap contact decision layer and rally-start handling before a separate side model;
- keep rescue search deferred because all 13 otherwise-exact one-missing spans already have a region-v2 candidate;
- do not claim rally-level server attribution improved until the alternation fit is rerun.

The exact region, features, controls and failure cases are in [`tree_contact_detector_results.md`](tree_contact_detector_results.md).

The proposed BST-X detector is specified separately in [`bst_x_contact_detector_plan.md`](bst_x_contact_detector_plan.md).

**Leave X3D-S aside for now.** RGB may show racket motion, shuttle blur or other clues that pose and shuttle features miss. Revisit it only if the failure cases point to useful visual information that is absent from the numeric streams.

## Metric definitions

**Contact timing precision / recall / F1**

One-to-one temporal matching between predicted contact events and labelled contact frames.

**Player-side accuracy given timing match**

Among timing-matched events for which the side rule answers Top/Bottom, the fraction with the correct labelled player side.

**Timing + correct-side recall**

Fraction of all labelled contacts for which both the event time and player side are correct.

**Joint event+side F1**

A prediction counts as correct only if it matches a labelled contact in time and gives the correct player side.

**Serve timing recall**

Contact timing recall restricted to the first labelled contact in each rally.

**Serve timing + correct-side recall**

Fraction of labelled serves for which both the serve time and serving side are correct.

## Data and limits

The measured fixtures are:

- `sset_01`
- `sset_15`
- `sset_21`

Together they contain:

- **292 rallies**
- **3,128 contacts**
- **292 serves**
- **2,836 non-serves**

The tree timing experiment is evaluated whole-fixture leave-one-out.

For player-side scoring, the event streams are frozen before player-side ground truth is loaded.

All three fixtures come from the same dataset, and there are only three of them. `sset_21` is the clearest reminder that the pooled score is not the whole story.

The strict rally curve uses the current fixed spans. It does not claim that a
new span finder would give the same result. The score cut-offs are pilot
measurements on three videos and must be refitted on the planned larger set.
