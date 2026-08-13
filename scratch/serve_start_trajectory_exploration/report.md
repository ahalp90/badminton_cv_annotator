# What the earliest accepted contact tells us about who served

## Main takeaway

All server results here use the same 239 one-to-one rallies. The released alternating-fit method gets **124** right. If we assume that the player at the earliest accepted contact served, that rises to **152**.

Only **24/239** rallies have usable shuttle motion before that contact, so motion can only help in a small number of cases. It is a correction to the basic method, not something we can use for every rally. Applying the motion correction directly raises the result to **163** correct. The prepend/refit method uses the same earliest-contact fallback and the same 15 motion triggers, but gets **159** correct. On those 15 triggered rallies, the direct method gets 13 right and prepend/refit gets 9.

Rerunning the full fit after adding the motion-based guess does not help in this sample. The bigger priorities are making sure we start from the right contact and increasing the number of rallies with usable motion paths.

## Why getting the first contact right matters

With the normal ±10 timing tolerance, **97 of 239** earliest accepted contacts do not match any ShuttleSet stroke. Looking at later accepted contacts recovers the serve in 49 of those rallies and the first return in another 36.

Many errors happen before we try to classify the shuttle motion. Sometimes an earlier ordinary contact candidate is accepted first. In other cases, the accepted contact sequence misses the serve but still includes the return.

## What should we try next?

Improve how we choose the starting contact and increase the number of usable motion paths before adding a more complicated trajectory classifier.

Keep the 0.05-BH rule unchanged while testing it on new videos. The two main limits of the current method are: the wrong contact can be chosen as the starting point, and only 24/239 rallies have usable motion evidence.

<!-- TOC START -->
## Contents

- [Main takeaway](#main-takeaway)
- [Why getting the first contact right matters](#why-getting-the-first-contact-right-matters)
- [What should we try next?](#what-should-we-try-next)
- [Why do we use 292, 249 and 239 rallies in different places?](#why-do-we-use-292-249-and-239-rallies-in-different-places)
- [Is the first accepted contact the serve?](#is-the-first-accepted-contact-the-serve)
- [What happens when the first contact does not match?](#what-happens-when-the-first-contact-does-not-match)
- [How does the motion correction work?](#how-does-the-motion-correction-work)
- [How often do we have usable motion?](#how-often-do-we-have-usable-motion)
- [Does excluding producer-marked interpolation help?](#does-excluding-producer-marked-interpolation-help)
- [Does prepend/refit improve the motion-based guess?](#does-prependrefit-improve-the-motion-based-guess)
- [Extra diagnostics and rule comparisons (optional)](#extra-diagnostics-and-rule-comparisons-optional)
  - [Do the diagnostics point to a better rule?](#do-the-diagnostics-point-to-a-better-rule)
  - [How does the older 0.25-BH rule compare?](#how-does-the-older-025-bh-rule-compare)
  - [Which usable paths give the wrong answer?](#which-usable-paths-give-the-wrong-answer)
- [Extra breakdowns (optional)](#extra-breakdowns-optional)
  - [Segmentation by video](#segmentation-by-video)
  - [Contact alignment by video](#contact-alignment-by-video)
  - [Follow-up for unmatched earliest contacts](#follow-up-for-unmatched-earliest-contacts)
  - [Motion availability by video](#motion-availability-by-video)
  - [Fixed motion rules by video](#fixed-motion-rules-by-video)
  - [Server results and broader checks](#server-results-and-broader-checks)
  - [The four triggered rallies where direct and prepend/refit disagree](#the-four-triggered-rallies-where-direct-and-prependrefit-disagree)
- [Note about an earlier exploratory comparison](#note-about-an-earlier-exploratory-comparison)
- [Limits](#limits)
- [Output files](#output-files)
<!-- TOC END -->

## Why do we use 292, 249 and 239 rallies in different places?

The main comparison uses 239 rallies because each of these has exactly one predicted span and one contact sequence for one ground-truth rally. The 249-rally and 292-rally results show how the findings change when we include broader sets of rallies. They are useful checks, but they do not replace the main 239-rally comparison.

| Rally group | All videos | sset_01 | sset_15 | sset_21 | What it is used for |
|---|---:|---:|---:|---:|---|
| All ground-truth (GT) rallies | 292 | 113 | 104 | 75 | End-to-end view, including segmentation failures |
| Covered rallies | 249 | 110 | 84 | 55 | Check how results change under the current COVERED definition, including merged rallies |
| One-to-one rallies | 239 | 104 | 84 | 51 | Analyses that need one predicted rally for each GT rally |

**How the groups narrow down:** 292 ground-truth rallies → 249 covered rallies → 239 one-to-one rallies.

The 249 covered rows come from 244 predicted spans. Of those spans, 239 cover one ground-truth rally each, while five spans each cover two ground-truth rallies. Those merged cases stay in the 249-rally results, but the main analysis does not score the same shared contact sequence twice.

The analysis separates five questions:

1. Did segmentation map the ground-truth rally to a predicted span?
2. Does the earliest accepted contact match a plausible stroke?
3. Is there a usable continuous shuttle path before that contact?
4. If there is a usable path, is its incoming-motion measure above or below the fixed threshold?
5. Does the final guess about who served turn out to be correct?

## Is the first accepted contact the serve?

Often, but not reliably enough to call it a detected serve. The serve is the largest single category, at **119 of 239** rallies, but **97 of 239** earliest contacts do not match any annotated stroke at the main ±10 tolerance.

The earliest accepted contact is the first output accepted by the released contact detector; it is not produced by a dedicated serve detector. The detector begins with shuttle impulses and player proximity, then applies wrist, suppression and exclusion checks. For this analysis, we independently measure which player is nearest at the accepted frame rather than relying on the released alternating fit.

The timing offset is:

`(accepted contact frame - GT stroke frame) × 30 / source fps`

A negative value means the accepted contact happens earlier than the ground-truth stroke. At each tolerance, we keep the nearest stroke as the match even if several strokes fall inside the window. The final column in the results reports those ambiguous cases separately.

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

The small “multiple” number shows when the timing window contains more than one possible GT stroke. We still use whichever stroke is closest to the accepted contact. This is rare at ±10 (5 rallies), but happens in 117 rallies at ±30, so the wider window is too ambiguous to tell us reliably which stroke the contact belongs to.

We use ±10 as the main tolerance. The stricter ±5 result and the broad ±30 check show how much the answer changes when the tolerance changes.

## What happens when the first contact does not match?

Later accepted contacts recover either the serve or the first return in **85 of the 97** rallies where the earliest contact does not match. This suggests that many bad starting contacts come from an early candidate being accepted first, or from the serve being missed even though later contacts are still present.

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Each later accepted contact is checked independently against every annotated stroke using the same ±10 tolerance. A stroke is allowed to match more than one accepted contact. The first later match appears at contact rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12. Ranks count from the start of the full accepted sequence, so the first later contact is rank 2.

Four of these first matches have more than one annotated stroke inside the ±10 window. In 27 sequences, the same stroke number matches more than one accepted contact. These cases are flagged; they do not change the result categories.

There are still 55 earliest contacts with no match even at ±30. We describe them as **GT-incompatible candidates under the ±30 sanity check**. That means they do not match the existing ground truth within ±30; it does not mean we manually inspected them and proved they were false contacts.

## How does the motion correction work?

The motion check asks whether the shuttle is moving towards that player before the earliest accepted contact.

If it is, the contact is more likely to be the first return rather than the serve, which means the other player probably served. If the path does not meet the incoming-motion threshold, we leave the earliest-contact guess unchanged.

To build the motion path, we look back by at most 30 frames on a 30-fps base timeline, staying within the same court scene. We choose the continuous run closest to the contact. A path is usable only if it has at least five samples, ends close enough to the contact, has recurrence guard `NO_FLAG`, has valid player-distance and body-height measurements, and contains no extremely large single-step jump.

For the trend measure, we calculate the slope between every pair of shuttle-to-player distance samples and take the median slope. Time is scaled from zero to one across the path. We then use the negative slope as the fitted decrease in distance. If that decrease is at least **0.05 apparent player body heights (BH)**, we call the path incoming.

The 0.05-BH threshold was chosen in advance as an engineering judgement. It is not a calibrated physical constant.

The direct method changes the basic earliest-contact guess when the shuttle meets the incoming-motion threshold for the contact player. The prepend/refit method instead adds the inferred server to the contact sequence and reruns the full alternating fit. Motion affects few rallies because usable motion evidence is rare.

## How often do we have usable motion?

Under the main recurrence check, only **24 of 239** one-to-one rallies have usable motion before the contact. The stricter check that also excludes producer-marked filled or interpolated points leaves 14. For most rallies, the method keeps the earliest-contact guess.

![Usable motion evidence under both TrackNet source checks](outputs/plots/motion_evidence_and_inpaint.png)

We can judge the motion rule against **135 earliest contacts** where the ±10 ground truth identifies either serve or first return without ambiguity: 118 serves and 17 first returns. Nineteen of those 135 have usable paths under the recurrence check.

The fixed rule marks 13 of those 19 paths as incoming. Nine are genuine first returns, while four are serves and therefore false return calls.

Looking specifically at the 17 ground-truth first returns: 9 are correctly called incoming, 4 have usable paths but stay below the 0.05-BH threshold, and 4 do not have a usable path at all. The distinction matters: "measured but below threshold" is different from "we had no usable motion evidence."

Across all 239 rallies, there are 24 usable paths and 15 incoming calls. Five of those usable paths belong to contacts that are unmatched or match a later stroke, so they cannot be included in the 135-rally serve-versus-return scoring set.

## Does excluding producer-marked interpolation help?

It removes the four false return calls, but it also removes useful evidence. With the same fixed 0.05-BH rule, the number of correctly found returns drops from 9 to 7.

This is a trade-off: fewer false calls, but also fewer rallies where we can make a useful motion-based call. The rule itself has not been retuned.

| Track source check | Labelled paths with usable motion | Correct return calls | False return calls | Returns missed |
|---|---:|---:|---:|---:|
| Exclude recurrence-flagged points | 19/135 | 9/17 | 4/118 | 8/17 |
| Also exclude producer-marked inpainted points | 10/135 | 7/17 | 0/118 | 10/17 |

The threshold and all other motion decisions stay the same in both rows. The number of labelled usable paths falls from 19 to 10. With the stricter source check, one missed return has usable motion but stays below 0.05 BH, while nine have no usable path. Every video loses some usable evidence.

## Does prepend/refit improve the motion-based guess?

No, not in this sample. The direct motion method gets **163/239** rallies right. Prepend/refit gets **159/239**, even though both use the same fallback and the same set of motion triggers. The entire four-answer difference comes from the 15 rallies where motion triggers.

![Four central server-attribution results](outputs/plots/server_attribution.png)

When motion does not trigger, both methods choose the player at the earliest accepted contact. When motion does trigger, the direct method chooses the other player as server.

Prepend/refit adds that inferred server to the contact sequence, reruns the alternating fit, and falls back to the earliest-contact player if the fit ties.

| Motion group | Rallies | Earliest-contact baseline | Direct correction | Prepend/refit |
|---|---:|---:|---:|---:|
| No incoming-motion trigger | 224 | 150 correct; 224 answers | 150 correct; 224 answers | 150 correct; 224 answers |
| Incoming-motion trigger | 15 | 2 correct; 15 answers | 13 correct; 15 answers | 9 correct; 15 answers |
| All primary rallies | 239 | 152 correct; 239 answers | 163 correct; 239 answers | 159 correct; 239 answers |

For the 15 triggered rallies, direct inference and prepend/refit agree in 11 cases. In two cases, later contact votes overturn a correct direct guess. In two more, the alternating fit ties.

These four cases show how using the whole contact sequence can sometimes weaken a good local motion clue. Fifteen triggered rallies is too small a sample to know whether this pattern will continue on new videos, but in this sample the extra refitting step does not improve the result.

## Extra diagnostics and rule comparisons (optional)

The sections below contain the diagnostic results, comparisons with the older rule, and individual failure cases.

### Do the diagnostics point to a better rule?

No. There are too few usable paths, and the groups overlap too much. Path length, residual scatter and trend-to-jitter do not show a clear reason to add another cutoff.

The decision itself uses only the 0.05-BH fitted decrease. Residual RMS tells us how much the points scatter around the fitted trend. Trend-to-jitter is the fitted decrease divided by that scatter. These are diagnostics; they do not decide whether a path is usable and they are not separate classifiers.

| Group | Paths | Median fitted decrease (BH) | Median residual scatter (BH) | Median trend-to-jitter |
|---|---:|---:|---:|---:|
| GT serves | 6 | 0.383 | 0.137 | 1.147 |
| GT first returns | 13 | 0.386 | 0.091 | 5.321 |

| Group | Paths | Median fitted decrease (BH) | Median residual scatter (BH) | Median trend-to-jitter |
|---|---:|---:|---:|---:|
| Correct calls | 11 | 0.394 | 0.089 | 5.749 |
| Incorrect calls | 8 | 0.013 | 0.107 | -0.275 |

| Observed path length | Paths | Median fitted decrease (BH) | Median residual scatter (BH) |
|---|---:|---:|---:|
| 5 points | 2 | 0.776 | 0.134 |
| 6-9 points | 4 | 0.153 | 0.051 |
| 10+ points | 13 | 0.316 | 0.122 |

These path-length groups summarise what we observed. They were not used to choose or adjust the rule.

![Continuous trend and jitter diagnostics](outputs/plots/trend_and_jitter_diagnostics.png)

In this small set, serves and first returns have almost the same median fitted decrease. Correct calls have a larger median fitted decrease and a higher trend-to-jitter value than incorrect calls. Incorrect calls also have slightly more residual scatter. These are observations about this sample, not new decision rules.

### How does the older 0.25-BH rule compare?

The older rule requires all three of the following: at least 0.25 BH of total shuttle movement, at least 0.25 BH of net movement towards the player, and at least 55% of steps moving towards the player.

The 0.05-BH trend rule instead checks whether the fitted decrease in shuttle-to-player distance reaches 0.05 BH across the observed path.

Both rules use the same checks for sample count, distance from the final path point to the contact, recurrence flags, valid measurements and large jumps. Because the older rule also requires 0.25 BH of total movement, it leaves 18 eligible paths instead of 19 with the recurrence check, and 9 instead of 10 with the producer mask added.

| Fixed comparison | Paths eligible for this rule | Correct return calls | False return calls | Returns missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Historical absolute-closure rule; recurrence check | 18 | 9 | 3 | 8 | 75.0% | 52.9% |
| 0.05-BH trend rule; recurrence check | 19 | 9 | 4 | 8 | 69.2% | 52.9% |
| Historical rule; recurrence plus producer mask | 9 | 7 | 0 | 10 | 100.0% | 41.2% |
| 0.05-BH trend rule; recurrence plus producer mask | 10 | 7 | 0 | 10 | 100.0% | 41.2% |

All four rows use the same 135 earliest contacts with unambiguous ±10 labels. "Returns missed" includes both returns with a usable path that falls below the threshold and returns with no usable motion evidence.

Neither rule was chosen because of the scores in this table. The 0.25-BH values come from the older analysis. The 55% step threshold was chosen using the older ±5/249 scoring setup. The 0.05-BH value was set in advance as an engineering judgement. None of these values has been independently calibrated as a physical threshold.

### Which usable paths give the wrong answer?

There are eight errors among the usable paths: four false return calls on ground-truth serves and four missed ground-truth returns. The traces below show the rally-level evidence for those cases.

The cases are sset_15 set1 rally 25, sset_01 set2 rally 30, sset_01 set1 rally 9, sset_21 set1 rally 40, sset_01 set1 rally 2, sset_15 set1 rally 3, sset_15 set2 rally 6, sset_01 set3 rally 13.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Extra breakdowns (optional)

The tables below show the per-video results and the broader checks.

### Segmentation by video

| Video | GT rallies | Covered | Split across spans | Missed by segmentation |
|---|---:|---:|---:|---:|
| All | 292 | 249 | 24 | 19 |
| sset_01 | 113 | 110 | 1 | 2 |
| sset_15 | 104 | 84 | 4 | 16 |
| sset_21 | 75 | 55 | 19 | 1 |

### Contact alignment by video

| Tolerance | GT serve | GT first return | Later GT stroke | No GT stroke in window | More than one GT stroke in window |
|---|---:|---:|---:|---:|---:|
| ±5 | 87 | 15 | 3 | 134 | 1 |
| ±10 | 119 | 19 | 4 | 97 | 5 |
| ±30 | 156 | 24 | 4 | 55 | 117 |

| Video | Tolerance | Rallies | GT serve | GT first return | Later GT stroke | No GT stroke in window | Multiple in window |
|---|---|---:|---:|---:|---:|---:|---:|
| sset_01 | ±5 | 104 | 30 | 8 | 0 | 66 | 1 |
| sset_01 | ±10 | 104 | 45 | 10 | 0 | 49 | 3 |
| sset_01 | ±30 | 104 | 62 | 14 | 0 | 28 | 48 |
| sset_15 | ±5 | 84 | 36 | 4 | 3 | 41 | 0 |
| sset_15 | ±10 | 84 | 50 | 5 | 4 | 25 | 2 |
| sset_15 | ±30 | 84 | 63 | 5 | 4 | 12 | 53 |
| sset_21 | ±5 | 51 | 21 | 3 | 0 | 27 | 0 |
| sset_21 | ±10 | 51 | 24 | 4 | 0 | 23 | 0 |
| sset_21 | ±30 | 51 | 31 | 5 | 0 | 15 | 16 |

At ±10, the broader 249-row view has 119 nearest serves, 21 nearest first returns, 4 later strokes and 105 unmatched earliest contacts. It also has 5 windows containing more than one stroke.

That similarity does not make the merged rows suitable for trajectory scoring that assumes one predicted rally corresponds to one ground-truth rally.

### Follow-up for unmatched earliest contacts

| Video | Unmatched anchors | Later contact matches serve | No serve match, but return matches | First match is another GT stroke | No later GT match |
|---|---:|---:|---:|---:|---:|
| All | 97 | 49 | 36 | 9 | 3 |
| sset_01 | 49 | 22 | 22 | 3 | 2 |
| sset_15 | 25 | 11 | 8 | 5 | 1 |
| sset_21 | 23 | 16 | 6 | 1 | 0 |

### Motion availability by video

| Track source check | Rallies | Continuous run selected | At least 5 points and close enough to contact | Passes the shared jump check | 0.05-BH incoming calls |
|---|---:|---:|---:|---:|---:|
| Exclude recurrence-flagged points | 239 | 57 | 31 | 24 | 15 |
| Also exclude producer-marked inpainted points | 239 | 48 | 17 | 14 | 10 |

"Continuous run selected" means there is at least one source point in the selected run. "At least 5 points and close enough" applies the minimum sample count and contact-gap checks. "Passes the shared jump check" is the final count of paths considered usable for the 0.05-BH decision. Rallies outside that count do not get a motion-based answer.

| Video | One-to-one rallies | Usable paths, recurrence check | Incoming calls | Usable paths, plus producer mask | Incoming calls |
|---|---:|---:|---:|---:|---:|
| sset_01 | 104 | 8 | 6 | 5 | 4 |
| sset_15 | 84 | 9 | 5 | 5 | 5 |
| sset_21 | 51 | 7 | 4 | 4 | 1 |

### Fixed motion rules by video

| Video | Unique ±10 truth | GT returns | Usable paths | Correct return calls | False return calls | Returns missed |
|---|---:|---:|---:|---:|---:|---:|
| sset_01 | 52 | 8 | 7 | 3 | 2 | 5 |
| sset_15 | 55 | 5 | 6 | 3 | 1 | 2 |
| sset_21 | 28 | 4 | 6 | 3 | 1 | 1 |

### Server results and broader checks

| Server method | Correct | Answers made | Overall accuracy (n=239) |
|---|---:|---:|---:|
| Released alternating fit | 124/239 | 217/239 | 51.9% |
| Assume the earliest contact player served | 152/239 | 239/239 | 63.6% |
| Flip player when the historical rule says incoming | 162/239 | 239/239 | 67.8% |
| Use earliest-contact player; flip when the 0.05-BH trend says incoming | 163/239 | 239/239 | 68.2% |
| Earliest-contact fallback; prepend inferred server and refit on incoming triggers | 159/239 | 239/239 | 66.5% |
| Same fallback and 0.05-BH flip; also mask producer inpaint | 160/239 | 239/239 | 66.9% |
| Motion answer only; abstain without usable evidence | 20/239 | 24/239 | 8.4% |
| Prepend one unknown contact before alternating fit | 125/239 | 217/239 | 52.3% |

The accuracy percentage always uses all 239 rallies as the denominator. "Answers made" tells us whether the method supplied Top or Bottom. The direct method and prepend/refit both fall back to the earliest-contact player, so they give an answer for all 239 rallies.

| Rally group | Released fit | Earliest-contact player | Earliest-contact fallback plus 0.05-BH flip |
|---|---:|---:|---:|
| 239 one-to-one | 124/239 (51.9%) | 152/239 (63.6%) | 163/239 (68.2%) |
| 249 covered, including merges | 128/249 (51.4%) | 154/249 (61.8%) | 165/249 (66.3%) |
| 292 end-to-end, including segmentation failures | 128/292 (43.8%) | 154/292 (52.7%) | 165/292 (56.5%) |

The 292-rally view includes all 43 segmentation failures. Those rallies have no contact to use for an earliest-contact answer. The 249-rally view includes ten ground-truth rows that belong to merged spans. These broader results are useful checks, but neither replaces the main 239-rally result.

| Video | Rallies | Released fit | Earliest-contact player | Direct motion correction | Prepend/refit, same fallback |
|---|---:|---:|---:|---:|---:|
| sset_01 | 104 | 53 | 52 | 58 | 56 |
| sset_15 | 84 | 42 | 64 | 67 | 66 |
| sset_21 | 51 | 29 | 36 | 38 | 37 |

### The four triggered rallies where direct and prepend/refit disagree

| Rally | What the augmented fit does | Direct inference | Prepend/refit result | GT server |
|---|---|---|---|---|
| sset_01 set1 rally 18 | Later votes override to Bot | Top | Bot | Top |
| sset_01 set2 rally 30 | Later votes override to Top | Bot | Top | Bot |
| sset_15 set2 rally 18 | Ties; retain earliest-contact fallback | Top | Bot | Top |
| sset_21 set1 rally 40 | Ties; retain earliest-contact fallback | Bot | Top | Bot |

## Note about an earlier exploratory comparison

One exploratory calculation combined prepend/refit on triggered rallies with the released alternating-fit method as the fallback on non-triggered rallies. It scored 127/239.

That result is not a like-for-like comparison with the direct method because the fallback is different. Of the 36-correct-answer gap between that calculation and the direct method, 32 answers come from the different fallback and only four come from the refitting on triggered rallies.

The row-level output keeps this calculation so it can still be checked later, but the report's main comparison uses the same earliest-contact fallback for both methods.

## Limits

- Only 17 earliest contacts with unique ±10 ground truth are first returns. Only 19 contacts with unique ground truth also have usable motion paths under the recurrence-only check.
- Body-height normalisation is based on apparent height in the image. It is not a physical distance on the court, and it can change with player scale and camera geometry.
- Paths with five points are allowed. The 0.05-BH threshold is modest for that reason, but it is still not calibrated.
- TrackNet residual scatter is measured from the observed path itself. We do not have separate ground truth for TrackNet position error.
- The ±30 view often contains several possible ground-truth strokes. It is a broad sanity check, not clean evidence of which stroke a contact represents.
- We did not add new manual labels. "GT-incompatible" means that a contact does not match the existing ground truth within the stated tolerance; it does not mean we visually checked it and proved it false.
- The three videos are the same videos used in the earlier exploration. The thresholds reported here were fixed before this scoring, but these results are not an independent external validation.

## Output files

- `outputs/rallies.csv.gz`: one checked row for each of 292 ground-truth rallies.
- `outputs/spans.csv.gz`: all 344 half-open predicted spans.
- `outputs/path_points.csv.gz`: the 1,012 sampled path points used to rebuild the motion measurements.
- `outputs/fixed_rules.csv.gz`: the four fixed rule/mask comparisons, both overall and by video.
- `outputs/trend_diagnostics.csv.gz`: continuous trend and jitter values for the 135 contacts with unique ±10 ground truth, under both masks.
- `outputs/metrics.json.gz`: checked summaries for rally counts, alignment, filtering steps and server results.
