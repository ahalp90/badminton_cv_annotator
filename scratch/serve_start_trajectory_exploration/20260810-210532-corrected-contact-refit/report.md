# What the earliest accepted contact tells us about the serve

## Summary

This investigation asks whether the shuttle clearly approaches the player at the earliest accepted contact. If it does, the contact is probably the first return rather than the serve. That signal can identify the other player as server. The corrected result is useful, but limited by contact timing and scarce trajectory evidence.

Three rally counts answer different questions. ShuttleSet contains **292 GT rallies**. This end-to-end population includes 43 rallies without a covering predicted span. The current scoring calls **249 rallies covered**. Ten belong to five predicted spans that each cover two GT rallies, so 249 remains a sensitivity view of `COVERED`. The main downstream population is **239 one-to-one rallies**. Each has one predicted span mapped to one GT rally. Contact identity, trajectory classification and server attribution use 239 when they require an unambiguous match.

The **earliest accepted contact** is the first ordinary contact-detector candidate that survives the released filters inside a predicted rally. It is not a serve detector. It comes from the usual shuttle impulse and player-proximity process. The wrist check, ordinary suppression and definitive exclusion mask can reject it. Nothing requires serve-like motion. The contact player's court half is measured directly at that frame, without using the alternating server fit.

On the 239 one-to-one rallies, the strict ±5 view gives 87 nearest serves, 15 first returns, 3 later strokes and 134 unmatched anchors. At the main ±10 baseline, the counts are **119 serve, 19 first return, 4 later and 97 unmatched**. Five ±10 windows contain several GT strokes. Keeping nearest identity while flagging ambiguity leaves unique truth for 118 serves and 17 returns. At ±30, the labels are 156 serve, 24 return, 4 later and 55 unmatched. However, 117 windows contain several GT strokes. This is a sanity check, not clean identity truth.

The 97 unmatched ±10 anchors reveal two patterns. In 49 rallies, a later accepted contact matches the serve. This is consistent with an earlier ordinary candidate taking the anchor position. In 36, no later contact matches the serve but one matches the first return. The serve appears missing while the return was detected. Nine first match another GT stroke. Three never match later GT. The first later match has rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12.

Usable motion evidence is rare. The recurrence-only check finds 57 continuous pre-contact runs. Thirty-one have at least five points and end close enough to contact. Only **24/239** pass the shared jump check. The unique ±10 truth set contains 19 usable paths. The fixed 0.05-BH rule makes 13 return calls: 9 correct and 4 false. Of 17 GT returns, 4 have usable paths below 0.05 BH and 4 have no usable path. A negative decision is therefore separate from missing evidence.

The unchanged historical rule adds a 0.25-BH total-movement eligibility floor. Its incoming call then requires 0.25 BH of net closure and 55% of steps towards the player. The 0.25 values came from the old analysis, while 55% was selected under old ±5/249 scoring. The new rule calls incoming from a robust fitted decrease of at least **0.05 apparent body heights**. This engineering judgement was fixed before corrected scoring and never swept. Both correctly identify 9 of 17 GT returns. The new rule adds one false call. The result shows what removing the strong, path-length-dependent 0.25-BH floor changes; it does not show a score advantage.

The inpaint comparison holds both rules fixed. Its only added restriction removes producer-marked filled or interpolated points. For the 0.05-BH rule, usable unique-truth paths fall from 19 to 10. Correct return calls fall from 9 to 7, false calls from 4 to 0, and missed returns rise from 8 to 10. Every video loses evidence.

Residual scatter and trend-to-jitter remain diagnostics, not decision cutoffs. Incorrect usable-path calls have slightly higher median scatter and a much weaker median trend-to-jitter signal. The sample is too small to turn that pattern into another rule.

On 239 one-to-one rallies, the released alternating fit is correct in 124/239. Using the earliest contact player gives 152/239. Using that player as the fallback, then flipping to the other player on 0.05-BH incoming evidence, reaches **163/239 (68.2%)**. The producer mask gives 160/239. Prepending a hypothetical contact reaches only 125/239 or 127/239. The useful result is the direct incoming-motion clue, not recursive refitting.

The next step is to improve anchor and path availability before building a richer classifier. Many failures start earlier: an ordinary candidate takes the anchor position, or the accepted sequence lacks the serve. Future work should keep contact and segmentation failures separate from trajectory classification. The unchanged 0.05-BH rule needs testing on new videos before its score can be treated as general performance.


## Rally groups and failure stages

| Rally group | All videos | sset_01 | sset_15 | sset_21 | Used for |
|---|---:|---:|---:|---:|---|
| All GT rallies | 292 | 113 | 104 | 75 | End-to-end view, including segmentation failures |
| Covered rallies | 249 | 110 | 84 | 55 | Sensitivity to the current COVERED definition, including merges |
| One-to-one rallies | 239 | 104 | 84 | 51 | Analyses that need one predicted rally per GT rally |

| Video | GT rallies | Covered | Split across spans | Missed by segmentation |
|---|---:|---:|---:|---:|
| All | 292 | 249 | 24 | 19 |
| sset_01 | 113 | 110 | 1 | 2 |
| sset_15 | 104 | 84 | 4 | 16 |
| sset_21 | 75 | 55 | 19 | 1 |

**Main denominator trail:** 292 GT rallies → 249 covered rallies → 239 one-to-one rallies → 135 unique ±10 serve/return anchors → 19 usable unique-truth paths under the recurrence-only check.

The 249 covered rows use 244 predicted spans. There are 239 one-rally spans and five spans that each cover two GT rallies. The merged rows remain visible in the covered sensitivity results, but they are never double-scored in the primary motion comparison.

The investigation keeps these stages separate:

1. Segmentation either maps a GT rally to a predicted span or fails.
2. The earliest accepted contact either matches a plausible GT stroke or does not.
3. A continuous pre-contact path is either unavailable, rejected by the shared quality checks, or usable.
4. A usable path falls above or below the fixed incoming threshold.
5. The resulting server attribution is either correct or incorrect.

## Earliest-contact alignment

The offset is `(accepted contact frame - GT stroke frame) × 30 / source fps`. Negative values mean the accepted contact occurs earlier. Each tolerance uses the nearest GT stroke even when several strokes fall inside the window. The last column reports that ambiguity separately.

| Tolerance | GT serve | GT first return | Later GT stroke | No GT stroke in window | More than one GT stroke in window |
|---|---:|---:|---:|---:|---:|
| ±5 | 87 | 15 | 3 | 134 | 1 |
| ±10 | 119 | 19 | 4 | 97 | 5 |
| ±30 | 156 | 24 | 4 | 55 | 117 |

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

All three tolerances by video are:

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

The merge-sensitive 249-row view is close to the primary result at ±10: 119 nearest serves, 21 nearest first returns, 4 later strokes and 105 unmatched anchors. It has 5 multiple-stroke windows. This similarity does not make merged rows suitable for one-rally trajectory scoring.

## What follows an unmatched anchor

Later contacts are checked independently against every GT stroke. A GT stroke is not consumed after one match. The rank is one-based in the full accepted-contact sequence, so the first later contact has rank 2.

| Video | Unmatched anchors | Later contact matches serve | No serve match, but return matches | First match is another GT stroke | No later GT match |
|---|---:|---:|---:|---:|---:|
| All | 97 | 49 | 36 | 9 | 3 |
| sset_01 | 49 | 22 | 22 | 3 | 2 |
| sset_15 | 25 | 11 | 8 | 5 | 1 |
| sset_21 | 23 | 16 | 6 | 1 | 0 |

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Four first matches have more than one GT stroke inside ±10. Twenty-seven sequences reuse a GT ordinal for more than one accepted contact. Those flags make the non-consuming check explicit. They do not change the category order in the table.

The 55 anchors still unmatched at ±30 are best described as **GT-incompatible candidates under the ±30 sanity criterion**. That wording does not claim a visually verified false contact.

## Motion evidence before the contact

The path searches back at most 30 base-30fps frames within the same court scene. It uses the continuous run closest to contact. Both fixed rules require at least five samples, no gap larger than two base-30fps frames before contact, recurrence guard `NO_FLAG`, finite contact-player distance and body-height evidence, and a largest-step to median-step ratio no greater than 4.

| Track source check | Rallies | Continuous run selected | At least 5 points and close enough to contact | Passes the shared jump check | 0.05-BH incoming calls |
|---|---:|---:|---:|---:|---:|
| Exclude recurrence-flagged points | 239 | 57 | 31 | 24 | 15 |
| Also exclude producer-marked inpainted points | 239 | 48 | 17 | 14 | 10 |

| Video | One-to-one rallies | Usable paths, recurrence check | Incoming calls | Usable paths, plus producer mask | Incoming calls |
|---|---:|---:|---:|---:|---:|
| sset_01 | 104 | 8 | 6 | 5 | 4 |
| sset_15 | 84 | 9 | 5 | 5 | 5 |
| sset_21 | 51 | 7 | 4 | 4 | 1 |

![Motion evidence availability and fixed inpaint comparison](outputs/plots/motion_evidence_and_inpaint.png)

“Continuous run selected” only means that at least one usable source point exists. “At least 5 points and close enough” applies the sample-count and contact-gap requirements. “Passes the shared jump check” is the usable-evidence count for the 0.05-BH decision. A rally outside that final count has no usable answer from the motion rule.

The 24 usable paths and 15 incoming calls above cover all 239 one-to-one rallies. Requiring unique ±10 serve/return truth leaves 19 usable paths and 13 incoming calls. The remaining five usable paths have another or unmatched anchor identity, so they cannot enter the serve-versus-return classification score.

## Historical absolute closure versus the 0.05-BH trend

The robust trend takes the median slope between every pair of shuttle-to-player distance samples. Time is normalised from zero to one across the observed path. The fitted decrease is the negative slope. The call is “incoming” only when that decrease reaches 0.05 BH.

Both rules first use the shared five-point, contact-gap, recurrence, finite-evidence and jump checks. The historical rule then adds its 0.25-BH total-movement eligibility floor. The trend rule does not. This is why the historical row has 18 eligible paths rather than 19 under the recurrence mask, and 9 rather than 10 under the producer mask. Net closure and the 55% approaching-step condition decide the historical incoming call after that eligibility check.

| Fixed comparison | Paths eligible for this rule | Correct return calls | False return calls | Returns missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Historical absolute-closure rule; recurrence check | 18 | 9 | 3 | 8 | 75.0% | 52.9% |
| 0.05-BH trend rule; recurrence check | 19 | 9 | 4 | 8 | 69.2% | 52.9% |
| Historical rule; recurrence plus producer mask | 9 | 7 | 0 | 10 | 100.0% | 41.2% |
| 0.05-BH trend rule; recurrence plus producer mask | 10 | 7 | 0 | 10 | 100.0% | 41.2% |

All four rows use the same 135 one-to-one anchors with unique ±10 truth: 118 GT serves and 17 GT first returns. “Returns missed” includes both usable paths below the threshold and returns without usable evidence. Under the recurrence-only 0.05-BH rule, four misses have usable negative paths and four have no usable path. Under the producer mask, one missed return has a usable negative path and nine have no usable path. The distinction matters because only the usable negative cases are trajectory decisions.

| Video | Unique ±10 truth | GT returns | Usable paths | Correct return calls | False return calls | Returns missed |
|---|---:|---:|---:|---:|---:|---:|
| sset_01 | 52 | 8 | 7 | 3 | 2 | 5 |
| sset_15 | 55 | 5 | 6 | 3 | 1 | 2 |
| sset_21 | 28 | 4 | 6 | 3 | 1 | 1 |

## What trend and jitter show

The 0.05-BH threshold alone makes the call. Residual RMS measures scatter around the robust trend. Trend-to-jitter divides the fitted decrease by that residual scatter. Neither diagnostic is an eligibility test or another classifier.

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

The path-length groups are descriptive summaries only. They were not used to make or tune the call.

![Continuous trend and jitter diagnostics](outputs/plots/trend_and_jitter_diagnostics.png)

GT serves and first returns have similar median fitted decreases in this small usable set. Correct calls show a much larger median fitted decrease and trend-to-jitter than incorrect calls. Incorrect calls also have slightly more residual scatter. These are descriptive patterns after applying the fixed rule. They do not justify another cutoff.

The error plot shows all eight mistakes with usable recurrence-mask paths: four false return calls and four missed returns. The cases are sset_15 set1 rally 25, sset_01 set2 rally 30, sset_01 set1 rally 9, sset_21 set1 rally 40, sset_01 set1 rally 2, sset_15 set1 rally 3, sset_15 set2 rally 6, sset_01 set3 rally 13.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Server attribution

The main server table uses only the 239 one-to-one rallies. Accuracy keeps abstentions in the denominator. “Answers made” shows whether a method supplied Top or Bottom.

| Server method | Correct over all rallies | Answers made | Accuracy |
|---|---:|---:|---:|
| Released alternating fit | 124/239 | 217/239 | 51.9% |
| Assume the earliest contact player served | 152/239 | 239/239 | 63.6% |
| Flip player when the historical rule says incoming | 162/239 | 239/239 | 67.8% |
| Use earliest-contact player; flip when the 0.05-BH trend says incoming | 163/239 | 239/239 | 68.2% |
| Same fallback and 0.05-BH flip; also mask producer inpaint | 160/239 | 239/239 | 66.9% |
| Motion answer only; abstain without usable evidence | 20/239 | 24/239 | 8.4% |
| Prepend one unknown contact before alternating fit | 125/239 | 217/239 | 52.3% |
| Prepend inferred server before alternating fit | 127/239 | 217/239 | 53.1% |

![Server attribution on the 239 one-to-one rallies](outputs/plots/server_attribution.png)

The direct 0.05-BH rule uses the anchor player when the path is usable but not incoming. It also uses the anchor player when motion evidence is unavailable. The evidence-only row abstains in the second case. Its 24/239 answers show the actual evidence coverage.

The same main methods under the two sensitivity populations are:

| Rally group | Released fit | Earliest-contact player | Earliest-contact fallback plus 0.05-BH flip |
|---|---:|---:|---:|
| 239 one-to-one | 124/239 (51.9%) | 152/239 (63.6%) | 163/239 (68.2%) |
| 249 covered, including merges | 128/249 (51.4%) | 154/249 (61.8%) | 165/249 (66.3%) |
| 292 end-to-end, including segmentation failures | 128/292 (43.8%) | 154/292 (52.7%) | 165/292 (56.5%) |

The end-to-end 292-row accuracy includes all 43 segmentation failures. Those failures have no anchor-based server answer. The 249-row result includes ten merged GT rows. Neither sensitivity view replaces the 239-row primary result.

Primary results by video show that no one video supplies the full improvement:

| Video | Rallies | Released fit | Earliest-contact player | 0.05-BH direct motion rule |
|---|---:|---:|---:|---:|
| sset_01 | 104 | 53 | 52 | 58 |
| sset_15 | 84 | 42 | 64 | 67 |
| sset_21 | 51 | 29 | 36 | 38 |

## Limits

- Only 17 first-return anchors have unique ±10 truth. Only 19 unique-truth anchors have usable recurrence-only motion paths.
- Apparent body-height normalisation is image based. It is not a physical court distance, and its meaning can change with player scale and camera geometry.
- A five-point path is allowed. The 0.05-BH engineering threshold is therefore deliberately modest, but it is still uncalibrated.
- TrackNet residual scatter is measured from the observed path. We do not have independent ground truth for TrackNet positional error.
- The ±30 view often contains several GT strokes. It is a sanity check, not clean stroke identity.
- No new manual labels were added. “GT-incompatible” means unmatched to existing GT under the stated tolerance, not visually proven false.
- The three videos are the same videos used in the historical exploration. The corrected thresholds were fixed before scoring, but the reported scores are not external validation.

## Output files

- `outputs/rallies.csv.gz`: one checked row for each of 292 GT rallies.
- `outputs/spans.csv.gz`: all 344 half-open predicted spans.
- `outputs/path_points.csv.gz`: the 1,012 sampled path points used to rebuild motion measurements.
- `outputs/fixed_rules.csv.gz`: the four fixed rule/mask comparisons globally and by video.
- `outputs/trend_diagnostics.csv.gz`: continuous trend and jitter values for the 135 unique ±10 truth anchors under both masks.
- `outputs/metrics.json.gz`: checked population, alignment, funnel and server summaries.
