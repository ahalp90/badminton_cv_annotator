# What the earliest accepted contact tells us about the serve

## Bottom line

All server scores in this paragraph use the same 239 one-to-one rallies. The released alternating fit gets **124** right. Using the player at the earliest accepted contact gets **152** right. Usable pre-contact motion exists in only **24/239** rallies, so motion can only be a small correction. Using the earliest-contact player by default, then changing the answer when motion clearly approaches that player, reaches **163**. The new information helps when used directly. Prepending the inferred missing serve and rerunning the alternating fit falls to **127**. The old fit therefore loses most of the direct gain. The practical priority is to improve which accepted contact becomes the anchor and how often a clean motion path exists. More trajectory complexity cannot help rallies that never reach a trustworthy contact or usable path.

## Why anchor selection comes first

At the normal ±10 timing tolerance, **97 of 239** earliest contacts do not match a ShuttleSet stroke. Later accepted contacts recover the serve in 49 of those rallies and the first return in another 36. This means many failures occur before motion classification. An early ordinary candidate often takes the anchor position, or the accepted sequence misses the serve while retaining the return.

## What should we do next?

Improve anchor selection and motion-path availability before adding a richer trajectory classifier. Keep the 0.05-BH rule unchanged while testing it on new videos. This focuses work on the two bottlenecks that limit the current method: the wrong contact can become the anchor, and only 24/239 rallies contain usable motion evidence.

## Extended summary (optional)

This investigation asks whether the shuttle approaches the player at the earliest accepted contact. Incoming motion suggests that this contact is the first return, not the serve. The other player is then the likely server. The useful version of this idea is modest: start with the player measured at the earliest contact and use motion only to correct a small number of calls.

Three rally counts describe different parts of the evaluation. ShuttleSet contains **292 ground-truth rallies**. That number shows end-to-end performance and includes segmentation failures. The current segmentation marks **249 rallies as covered**, but ten of those rallies share five predicted spans. They remain a sensitivity view of the existing `COVERED` definition. The main result uses **239 one-to-one rallies**, where one predicted span maps to one ground-truth rally. This avoids scoring the same contact sequence twice.

These populations are not interchangeable. Using all 292 rallies for trajectory scoring would mix missing predicted spans with motion mistakes. Using all 249 covered rallies would reuse one accepted-contact sequence for both ground-truth rallies inside each merged span. The 239-rally set removes those two problems. The broader populations remain useful later, where they show how segmentation failures and the current merge definition affect end-to-end accuracy.

The earliest accepted contact is an ordinary output of the released contact detector. It is not designed to find serves. The detector starts from shuttle impulses and player proximity, then applies its usual wrist, suppression and exclusion checks. The analysis measures which player is nearest at that accepted frame. It does not use the released alternating server fit to choose that player.

This direct player measurement gives a simple baseline. If the contact is the serve, its player is the server. If the contact is the first return, the other player served. The baseline works only as well as the chosen contact. The timing comparison therefore comes before any motion classification or server score.

The practical timing check allows ±10 base-30fps frames between the accepted contact and an annotated stroke. On the 239 one-to-one rallies, the earliest contact is nearest the serve in 119 cases, the first return in 19, and a later stroke in 4. In **97 rallies**, no annotated stroke lies inside the window. Five windows contain more than one annotated stroke, which the analysis flags separately. Stricter ±5 and broader ±30 views are reported later as supporting checks, but ±10 is the main baseline.

The 97 unmatched earliest contacts show that many failures happen before trajectory classification. A later accepted contact matches the serve in 49 rallies. In another 36, no later contact matches the serve, but one matches the first return. The first pattern is consistent with an early ordinary candidate taking the anchor position. The second is consistent with the serve being absent from the accepted sequence while the return remains detectable. Only nine first match another later stroke, and three have no later match. Better anchor selection is therefore a central finding, not a side issue.

Motion coverage is the next limitation. Only **24 of 239 rallies** have a continuous pre-contact path that passes the shared quality checks. The direct server method does not make 239 trajectory-based decisions. It uses the earliest-contact player as its answer whenever motion is unavailable or does not say incoming. Motion changes the answer only when the fixed 0.05-body-height trend rule finds a clear approach towards that player.

This separation matters when reading the final accuracy. A rally without usable motion is not evidence that the shuttle moved away from the player. It is a rally where the trajectory method cannot answer. The full server method still answers because it falls back to the directly measured contact player.

That limited correction still helps. The released alternating fit gets 124/239 rallies right. The earliest-contact player alone gets 152/239. Adding the motion-backed correction reaches **163/239**. The gain is not evidence that trajectory solves server attribution generally. It shows that a small amount of usable incoming-motion evidence can correct some otherwise direct contact-player calls.

The prepend experiment gives an important negative result. Supplying an inferred missing serve and rerunning the old alternating fit reaches only 127/239. The new information is useful when applied directly, but the alternating refit largely throws that improvement away. This result argues against recursive refitting as the next step.

The practical next step is to improve which accepted contact becomes the anchor and how often a clean pre-contact path exists. The fixed 0.05-body-height rule should remain unchanged until it is tested on new videos. More complicated motion classification would add machinery while the larger contact-selection and evidence-availability failures remain unresolved.


## What are the 292, 249 and 239 rallies?

The main result uses 239 rallies because each has one predicted span and one contact sequence for one ground-truth rally. The 249-rally and 292-rally views answer broader sensitivity questions; they do not replace that primary comparison.

| Rally group | All videos | sset_01 | sset_15 | sset_21 | Used for |
|---|---:|---:|---:|---:|---|
| All GT rallies | 292 | 113 | 104 | 75 | End-to-end view, including segmentation failures |
| Covered rallies | 249 | 110 | 84 | 55 | Sensitivity to the current COVERED definition, including merges |
| One-to-one rallies | 239 | 104 | 84 | 51 | Analyses that need one predicted rally per GT rally |

**Population trail:** 292 ground-truth rallies → 249 covered rallies → 239 one-to-one rallies.

The 249 covered rows use 244 predicted spans. There are 239 one-rally spans and five spans that each cover two ground-truth rallies. The merged rows remain visible in the covered sensitivity results, but the primary analysis never scores their shared contact sequence twice.

The investigation keeps five stages separate:

1. Segmentation maps a ground-truth rally to a predicted span or fails.
2. The earliest accepted contact matches a plausible stroke or does not.
3. A continuous pre-contact path is unavailable, rejected by the quality checks, or usable.
4. A usable path falls above or below the fixed incoming threshold.
5. The resulting server attribution is correct or incorrect.

## Is the first accepted contact actually the serve?

Usually it is closest to the serve, but **97 of 239** earliest contacts do not match any annotated stroke at the main ±10 tolerance. The anchor is therefore useful, but too unreliable to treat as a detected serve.

The offset is `(accepted contact frame - GT stroke frame) × 30 / source fps`. Negative values mean the accepted contact occurs earlier. Each tolerance keeps the nearest stroke identity even when several strokes lie inside the window. The last column reports that ambiguity separately.

| Tolerance | GT serve | GT first return | Later GT stroke | No GT stroke in window | More than one GT stroke in window |
|---|---:|---:|---:|---:|---:|
| ±5 | 87 | 15 | 3 | 134 | 1 |
| ±10 | 119 | 19 | 4 | 97 | 5 |
| ±30 | 156 | 24 | 4 | 55 | 117 |

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

The ±10 bar is the practical baseline. The ±5 strict view and ±30 sanity check show how the result changes with tolerance. The broad ±30 window contains several strokes in 117 rallies, so it is not clean identity truth.

## What happens when the first contact is wrong?

Later accepted contacts recover the serve or first return in **85 of the 97** unmatched rallies. Many unmatched anchors therefore reflect an early candidate taking the anchor position or a missing serve, rather than a failure of the later sequence as a whole.

| Video | Unmatched anchors | Later contact matches serve | No serve match, but return matches | First match is another GT stroke | No later GT match |
|---|---:|---:|---:|---:|---:|
| All | 97 | 49 | 36 | 9 | 3 |
| sset_01 | 49 | 22 | 22 | 3 | 2 |
| sset_15 | 25 | 11 | 8 | 5 | 1 |
| sset_21 | 23 | 16 | 6 | 1 | 0 |

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Later contacts are checked independently against every annotated stroke at the same ±10 tolerance. A stroke is not consumed after one match. The first later match occurs at accepted-contact rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12. Rank is one-based in the full accepted sequence, so the first later contact has rank 2.

Four first matches have more than one annotated stroke inside ±10. Twenty-seven sequences reuse one stroke ordinal for more than one accepted contact. These flags make the non-consuming check explicit and do not change the outcome categories.

The 55 anchors still unmatched at ±30 are best described as **GT-incompatible candidates under the ±30 sanity criterion**. This wording does not claim a visually verified false contact.

## Can incoming motion help?

Yes, but usable pre-contact motion exists in only **24 of 239** one-to-one rallies. The motion result is a small correction to the earliest-contact fallback, not a stand-alone answer for every rally.

![Usable motion evidence under both TrackNet source checks](outputs/plots/motion_evidence_and_inpaint.png)

The path search looks back at most 30 base-30fps frames within the same court scene and uses the continuous run closest to contact. Both fixed rules require at least five samples, a final sample close to the contact, recurrence guard `NO_FLAG`, finite player-distance and body-height evidence, and no gross single-step jump.

| Track source check | Rallies | Continuous run selected | At least 5 points and close enough to contact | Passes the shared jump check | 0.05-BH incoming calls |
|---|---:|---:|---:|---:|---:|
| Exclude recurrence-flagged points | 239 | 57 | 31 | 24 | 15 |
| Also exclude producer-marked inpainted points | 239 | 48 | 17 | 14 | 10 |

“Continuous run selected” means that at least one source point exists. “At least 5 points and close enough” applies the sample-count and contact-gap checks. “Passes the shared jump check” is the usable-evidence count for the 0.05-BH decision. A rally outside that count has no motion answer.

To judge the motion call itself, the analysis needs an anchor that can be labelled confidently as either the serve or first return. There are **135 such rallies** at ±10: 118 serves and 17 first returns. Nineteen have usable recurrence-checked paths. The fixed 0.05-BH rule correctly identifies 9 returns, makes 4 false return calls on serves, and misses 8 returns. Four of those misses have usable motion below the threshold; four have no usable path.

The 24 usable paths and 15 incoming calls over all 239 rallies are broader availability counts. Restricting to the 135 labelled rallies leaves 19 usable paths and 13 incoming calls. The other five usable paths have an unmatched or later-stroke anchor and cannot enter serve-versus-return scoring.

## Does removing inpainted TrackNet points help?

Removing producer-marked filled or interpolated points eliminates the four false return calls, but it also removes useful evidence. The same fixed 0.05-BH rule then finds 7 returns instead of 9. This is a precision-coverage trade-off, not a retuned comparison.

| Track source check | Labelled paths with usable motion | Correct return calls | False return calls | Returns missed |
|---|---:|---:|---:|---:|
| Exclude recurrence-flagged points | 19/135 | 9/17 | 4/118 | 8/17 |
| Also exclude producer-marked inpainted points | 10/135 | 7/17 | 0/118 | 10/17 |

The threshold and every other motion decision remain unchanged between rows. Usable labelled paths fall from 19 to 10. Under the stricter source check, one missed return has usable motion below 0.05 BH and nine have no usable path. Every video loses evidence.

## Does the inferred missing serve improve server identification?

The new information helps when used directly. Feeding it back into the released alternating fit mostly loses the improvement: **163/239 falls to 127/239**.

![Four central server-attribution results](outputs/plots/server_attribution.png)

The direct method starts with the earliest-contact player. It changes that answer only for the 15 rallies where usable motion says the shuttle is incoming. When evidence is unavailable or the path does not say incoming, the earliest-contact player remains the answer.

The full primary table keeps the historical rule, stricter producer mask, evidence-only result and both prepend variants visible:

| Server method | Correct | Answers made | Overall accuracy (n=239) |
|---|---:|---:|---:|
| Released alternating fit | 124/239 | 217/239 | 51.9% |
| Assume the earliest contact player served | 152/239 | 239/239 | 63.6% |
| Flip player when the historical rule says incoming | 162/239 | 239/239 | 67.8% |
| Use earliest-contact player; flip when the 0.05-BH trend says incoming | 163/239 | 239/239 | 68.2% |
| Same fallback and 0.05-BH flip; also mask producer inpaint | 160/239 | 239/239 | 66.9% |
| Motion answer only; abstain without usable evidence | 20/239 | 24/239 | 8.4% |
| Prepend one unknown contact before alternating fit | 125/239 | 217/239 | 52.3% |
| Prepend inferred server before alternating fit | 127/239 | 217/239 | 53.1% |

Accuracy retains all 239 rallies in the denominator. “Answers made” shows whether a method supplied Top or Bottom. The inferred-player prepend answers 217 rallies, while the direct fallback answers all 239.

## Detailed motion methods and diagnostics

The detail below explains the fixed rule comparison and track-noise measurements. It supports the main result, but neither diagnostic changes a call.

### Historical absolute closure versus the 0.05-BH trend

The robust trend takes the median slope between every pair of shuttle-to-player distance samples. Time is normalised from zero to one across the observed path. The fitted decrease is the negative slope. The call is “incoming” only when that decrease reaches 0.05 apparent player body heights.

Both rules first use the shared sample-count, contact-gap, recurrence, finite-evidence and jump checks. The historical rule then adds its 0.25-BH total-movement eligibility floor. The trend rule does not. This is why the historical row has 18 eligible paths rather than 19 under the recurrence check, and 9 rather than 10 after removing producer-marked inpaint. Net closure of 0.25 BH and the 55% approaching-step condition then decide the historical call.

| Fixed comparison | Paths eligible for this rule | Correct return calls | False return calls | Returns missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Historical absolute-closure rule; recurrence check | 18 | 9 | 3 | 8 | 75.0% | 52.9% |
| 0.05-BH trend rule; recurrence check | 19 | 9 | 4 | 8 | 69.2% | 52.9% |
| Historical rule; recurrence plus producer mask | 9 | 7 | 0 | 10 | 100.0% | 41.2% |
| 0.05-BH trend rule; recurrence plus producer mask | 10 | 7 | 0 | 10 | 100.0% | 41.2% |

All four rows use the same 135 confidently labelled anchors. “Returns missed” includes both usable paths below threshold and returns without usable evidence. The 0.25 values came from the old analysis, while 55% was selected under old ±5/249 scoring. None is an independently calibrated physical threshold. The 0.05-BH value is an engineering judgement fixed before corrected scoring and was never swept.

| Video | Unique ±10 truth | GT returns | Usable paths | Correct return calls | False return calls | Returns missed |
|---|---:|---:|---:|---:|---:|---:|
| sset_01 | 52 | 8 | 7 | 3 | 2 | 5 |
| sset_15 | 55 | 5 | 6 | 3 | 1 | 2 |
| sset_21 | 28 | 4 | 6 | 3 | 1 | 1 |

### What trend and jitter show

The 0.05-BH fitted decrease alone makes the call. Residual RMS measures scatter around the robust trend. Trend-to-jitter divides fitted decrease by that scatter. Neither diagnostic is an eligibility test or another classifier.

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

Serves and first returns have similar median fitted decreases in this small usable set. Correct calls show a much larger median fitted decrease and trend-to-jitter than incorrect calls. Incorrect calls also have slightly more residual scatter. These patterns do not justify another cutoff.

The error plot shows all eight mistakes with usable recurrence-checked paths: four false return calls and four missed returns. The cases are sset_15 set1 rally 25, sset_01 set2 rally 30, sset_01 set1 rally 9, sset_21 set1 rally 40, sset_01 set1 rally 2, sset_15 set1 rally 3, sset_15 set2 rally 6, sset_01 set3 rally 13.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Supporting breakdowns (optional)

The tables below retain the per-video and sensitivity evidence without placing it in the main reading path.

### Segmentation by video

| Video | GT rallies | Covered | Split across spans | Missed by segmentation |
|---|---:|---:|---:|---:|
| All | 292 | 249 | 24 | 19 |
| sset_01 | 113 | 110 | 1 | 2 |
| sset_15 | 104 | 84 | 4 | 16 |
| sset_21 | 75 | 55 | 19 | 1 |

### Contact alignment by video

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

The merge-sensitive 249-row view is close to the primary result at ±10: 119 nearest serves, 21 nearest first returns, 4 later strokes and 105 unmatched anchors. It has 5 multiple-stroke windows. Similar counts do not make merged rows suitable for one-rally trajectory scoring.

### Motion availability by video

| Video | One-to-one rallies | Usable paths, recurrence check | Incoming calls | Usable paths, plus producer mask | Incoming calls |
|---|---:|---:|---:|---:|---:|
| sset_01 | 104 | 8 | 6 | 5 | 4 |
| sset_15 | 84 | 9 | 5 | 5 | 5 |
| sset_21 | 51 | 7 | 4 | 4 | 1 |

### Server sensitivity and video results

| Rally group | Released fit | Earliest-contact player | Earliest-contact fallback plus 0.05-BH flip |
|---|---:|---:|---:|
| 239 one-to-one | 124/239 (51.9%) | 152/239 (63.6%) | 163/239 (68.2%) |
| 249 covered, including merges | 128/249 (51.4%) | 154/249 (61.8%) | 165/249 (66.3%) |
| 292 end-to-end, including segmentation failures | 128/292 (43.8%) | 154/292 (52.7%) | 165/292 (56.5%) |

The 292-row view includes all 43 segmentation failures. Those failures have no anchor-based answer. The 249-row view includes ten merged ground-truth rows. Neither sensitivity view replaces the 239-rally primary result.

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
