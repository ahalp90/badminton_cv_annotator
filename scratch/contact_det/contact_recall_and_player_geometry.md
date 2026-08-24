# Contact recall and player geometry

## Short answer

No. The current pipeline does not propose every non-serve contact, even before its wrist and suppression filters run.

At the useful 10-frame tolerance, the noisy raw list finds 2,377 of 2,836 later contacts, or 83.8%. It finds 193 of 292 serves, or 66.1%.

At 15 frames, the raw list still misses 376 later contacts and 76 serves. Extra noise is not enough to give complete coverage.

The ankle-height rule does not repair player attribution in these fixtures. It gives almost exactly the same answer as the current box-and-net rule. Direct player-side attribution is already about 89% accurate on matched final contacts at the 10-frame tolerance. The poor rally-level result mainly comes from missed contacts upsetting the expected Top/Bottom alternation.

## What is being measured

The three fixtures are `sset_01`, `sset_15`, and `sset_21`. Together they contain 292 rallies and 3,128 labelled strokes.

Frame tolerances are scaled from a 30 fps base. A tolerance of 10 therefore means 8 frames in the 25 fps fixtures and 10 frames in the 30 fps fixture.

The report uses three tolerances:

- 5 frames: a strong result;
- 10 frames: the main usable result;
- 15 frames: loose, but still potentially useful to a later local search.

The score pairs predictions and real contacts one to one. One noisy prediction cannot count as finding two contacts.

## The current proposal path does not use ground truth

The production path finds rally spans, calculates shuttle direction-change impulses, applies the wrist-distance check, then removes nearby weaker proposals. Ground-truth (GT) stroke frames are not inputs to those steps.

GT is loaded separately by the fixture scorer after prediction. The new measurement code keeps the same boundary: it saves all prediction evidence first, verifies that save, then opens GT.

The raw proposals are not completely unconstrained. They only exist inside detected rally spans and outside the exclusion mask. If the span finder or mask removes a real contact, extra noise in the impulse detector cannot recover it.

## Current full-fixture result

The measurement uses saved artefacts from the current `ad8da4f` pipeline run. The prediction save was made twice and the files were byte-for-byte identical. GT was opened only after the saved evidence and its checksum were verified.

There are 6,170 raw proposals. The current wrist and suppression filters leave 3,706.

### Raw, noisy proposals

| Allowed error | Serves found | Later contacts found | Raw proposals left unmatched |
| --- | ---: | ---: | ---: |
| 5 base-30 frames | 144 / 292 (49.3%) | 2,206 / 2,836 (77.8%) | 3,820 / 6,170 |
| 10 base-30 frames | 193 / 292 (66.1%) | 2,377 / 2,836 (83.8%) | 3,600 / 6,170 |
| 15 base-30 frames | 216 / 292 (74.0%) | 2,460 / 2,836 (86.7%) | 3,494 / 6,170 |

“Left unmatched” means that a proposal did not receive a one-to-one GT match at that tolerance. It is a useful noise count, not a claim that every unmatched proposal is certainly false.

### Current final contacts

| Allowed error | Serves found | Later contacts found | Final contacts left unmatched |
| --- | ---: | ---: | ---: |
| 5 base-30 frames | 132 / 292 (45.2%) | 2,093 / 2,836 (73.8%) | 1,481 / 3,706 |
| 10 base-30 frames | 178 / 292 (61.0%) | 2,303 / 2,836 (81.2%) | 1,225 / 3,706 |
| 15 base-30 frames | 208 / 292 (71.2%) | 2,355 / 2,836 (83.0%) | 1,143 / 3,706 |

The raw list helps, but only by a few percentage points. Most misses occur before the final suppression decision.

A looser serve check asks whether any raw proposal lies near the first stroke. At 10 frames that is true in 195 of 292 rallies, or 66.8%. It does not change the answer: the pipeline is not close to finding every serve.

An older PR98 replay gave slightly lower filtered recall and different proposal counts. It used the same contact code with a different saved mask and court-input profile. The tables above use the newer complete current artefact set and are the main result.

## PR88 result

PR88 was a historical attempt to identify the server from a chosen outgoing contact. It was not added to production.

Its preferred rule was:

```text
if the chosen outgoing contact looks like a return:
    the other side served
elif it looks like a serve:
    the chosen side served
else:
    use the older PR82 answer
```

The frozen PR88 records cover 239 rallies. They contain 239 serve contacts, 2,600 later contacts and 3,200 accepted predictions.

| Allowed error | Serves found | Later contacts found | Predictions left unmatched |
| --- | ---: | ---: | ---: |
| 5 base-30 frames | 122 / 239 (51.1%) | 1,947 / 2,600 (74.9%) | 1,131 |
| 10 base-30 frames | 167 / 239 (69.9%) | 2,149 / 2,600 (82.7%) | 884 |
| 15 base-30 frames | 191 / 239 (79.9%) | 2,203 / 2,600 (84.7%) | 806 |

So this historical accepted set does not find every non-serve contact. At the main 10-frame tolerance it misses 451 of 2,600 later contacts.

A looser question asks whether any prediction is near the serve, without one-to-one competition from later contacts. That finds 123, 170 and 209 of the 239 serves at tolerances 5, 10 and 15.

Those numbers are not PR88's own serve-selection result. At the 10-frame tolerance, PR88 chose the correct visible start in 132 of 239 rallies. It got both that visible start and the server side right in 117. The rule can therefore infer the right server side from the wrong contact, and the two scores must stay separate.

The linked GitHub PR97 later tested the frozen PR88 rule on held-out videos. PR88 tied the older rule on overall server accuracy and became worse on one of the two held-out videos. The useful part of PR97 is its clean separation between saved predictions and later GT scoring, not the serve rule itself.

## How player attribution works now

The raw contact proposal itself has no player label. After the wrist and suppression rules choose the final contacts, the pipeline:

1. picks the tracked player slot whose wrist is closest to the shuttle;
2. reads the bottom of that player's detection box;
3. calls the player Top when the box is above the calibrated net band, Bottom when it is below, and gives no answer inside the band;
4. chooses the Top/Bottom alternation that best fits all contacts in the rally.

So the present code is not blindly assigning the hitter from the court half alone. It uses wrist distance to choose the likely hitter, then uses box position and the net band to name the side. It does not maintain a persistent human identity across the video.

## The ankle-height test

The test keeps the current nearest-wrist hitter choice. Only the Top/Bottom naming rule changes:

```text
if two player detections are usable:
    the player with the smaller mean ankle y is Top
elif one player detection is usable:
    ankle y above the middle of the calibrated net band is Top
    otherwise it is Bottom
else:
    there is no answer
```

Image y grows downwards, so the top-court player has the smaller ankle y. The one-player boundary is the calibrated net-band midpoint, not the middle of the image.

The same rally alternation fit is then applied to the ankle guesses. This makes the comparison fair: contact frames, the likely-hitter choice and the rally-level cleanup remain unchanged.

## Ankle-height result

The rule barely changes the result.

Across all 3,706 final contact candidates:

- the current and ankle rules give the same answer on 3,680;
- they give different Top/Bottom answers on 10;
- the ankle rule fills 16 cases where the current rule has no answer;
- the ankle rule has no missing answer.

On the 2,481 final contacts matched at the 10-frame tolerance:

| Rule | Correct player side | Accuracy | Contacts with an answer |
| --- | ---: | ---: | ---: |
| Current box-and-net rule | 2,207 / 2,480 | 89.0% | 2,480 / 2,481 |
| Ankle-height rule | 2,208 / 2,481 | 89.0% | 2,481 / 2,481 |

Only four matched contacts have one usable player detection. Both rules get two of the four right. That is too small a group to support a strong claim about the one-player fallback.

The final rally fit remains poor:

| Rally-level result | Current rule | Ankle-height rule |
| --- | ---: | ---: |
| Final hitter side | 112 / 228 (49.1%) | 111 / 226 (49.1%) |
| Server side | 148 / 228 (64.9%) | 145 / 226 (64.2%) |
| Rallies with an answer | 228 / 292 (78.1%) | 226 / 292 (77.4%) |

This gap explains why contact-player attribution can look woeful even though the direct side call is usually right. The rally cleanup assumes strict Top/Bottom alternation. If a contact is missed, every later odd/even position can move to the wrong phase. Changing box feet to ankles does not repair that missing-contact problem.

## Scene cuts and serve lookback

A scene change resets player tracking. That could leave a suspected first contact with no usable current-frame player evidence.

It does not happen in the final contact list here. There are 167 final candidates within 15 base-30 frames of a scene start. Both the current rule and the ankle rule have an answer for all 167. Eight use the one-player rule and 159 have two players.

Among the noisy raw proposals in the same scene-start range, only 3 of 272 lack an answer from either rule. None survives into the final list.

A serve lookback therefore has no missing-attribution cases to repair in these three fixtures. Contacts near scene starts are still harder: at the main matching tolerance, both side rules get 52 of 75 matched final contacts right, or 69.3%. A future lookback experiment should try to improve those wrong answers. It should not be presented as a fix for missing geometry that is not present in this sample.

## Limits

An offscreen or broadcast-omitted contact has no exact visible impact to recover. A lookback can infer a likely server or likely region. It cannot turn an unseen event into observed evidence.

The PR88 table is historical and covers 239 of the 292 fixture rallies. It is useful evidence, but it is not a substitute for the current full-fixture raw-proposal measurement.

The direct player-side score covers contacts that received a temporal GT match. It does not say that a side was correct for proposals that missed the real contact.
