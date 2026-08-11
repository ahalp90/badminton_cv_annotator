# Review Feedback

I followed `REVIEW_DIRECTIONS.md` as a fresh reader: I inspected every plot before reading `report.md`, read the report once, and did not use the supplied audit/feedback files or implementation context.

## Plot-first readings

| Plot | My first reading before the report | Do title/labels/categories/denominator support it? |
|---|---|---|
| **anchor_alignment.png** | Across 239 one-to-one rallies, widening the tolerance makes the earliest accepted contact more often nearest a GT serve: 87 at ±5, 119 at ±10, 156 at ±30; ±10 is the main baseline and ±30 becomes highly ambiguous. | **Mostly.** `n=239`, category names, and the baseline/sanity labels are good. But the “multiple GT strokes in window” annotations sit visually inside the gray **No GT stroke** portion. That makes a cross-cutting ambiguity count look like a subset of “no GT,” which is misleading. |
| **unmatched_anchor_followup.png** | Of the 97 ±10-unmatched anchors, later contacts most often eventually recover the serve (49), then the first return (36); when a later GT match exists, it is usually the second accepted contact (56). | **Partly.** Both panels clearly say `n=97`. I had to guess that later matching also uses ±10 and that the left categories are mutually exclusive by priority over the whole later sequence, rather than describing the first later matched contact. |
| **motion_evidence_and_inpaint.png** | Usable motion evidence is rare and becomes rarer when producer-marked inpaint is excluded; the fixed rule loses true return calls but also loses false-return calls. | **Left panel yes; right panel no.** The left denominator `n=239` is clear. The right panel mixes quantities with different natural denominators: usable paths are out of 135 anchors, correct/missed returns are out of 17 GT returns, and false-return calls are out of 118 GT serves. “Returns missed” also hides the important distinction between **no usable evidence** and **usable evidence below threshold**. |
| **trend_and_jitter_diagnostics.png** | The 19 usable paths show substantial serve/return overlap; incorrect calls are not simply the high-jitter tracks, and trend-to-jitter is descriptive rather than another cutoff. | **Mostly.** The 19-path denominator and serve/return counts are explicit. I had to infer that `fitted decrease ≥ 0.05 BH` means “incoming → call first return,” and that “correct/incorrect calls” refer specifically to the 0.05-BH rule. |
| **representative_rule_errors.png** | Four GT serves exhibit enough apparent closure to be called returns, while four GT returns have flat/increasing fitted distance and are missed. | **Not fully.** The individual cases are readable, but “fixed-rule mistakes” does not say **which** of the two fixed rules. I had to infer that these are the eight 0.05-BH trend-rule mistakes. The plot also does not say these are 8 of the 19 usable unique-truth paths. |
| **server_attribution.png** | Earliest-contact player substantially beats the released alternating fit, and adding incoming-motion flips improves 152/239 to 163/239; the producer mask gives slightly less improvement. | **Potentially misleading.** The denominator and answer counts are explicit, but “0.05-BH trend rule — 239/239 answers” initially reads like motion evidence itself supplied 239 answers. After reading the report, I learned that motion exists for only 24/239 rallies; this bar is actually a **hybrid: earliest-contact player by default, flipped when usable motion says incoming**. That distinction is central. |

I did **not** mistake the earliest contact for a serve-specific detector after reading the report, because the report states this unusually clearly. From the plots alone, however, nothing prominently reminds the reader that this is just an ordinary accepted contact candidate.

I also encountered one count I could not reconcile from the supplied report: the historical rule has **18 usable paths versus 19** for the trend rule under the recurrence mask, and **9 versus 10** under the producer mask, even though the report says both rules share the listed path-quality requirements. If the historical rule has one additional eligibility requirement, it needs to be stated; if its 0.25-BH conditions are merely classification conditions, “usable paths” should seemingly be common.

## Comprehension verdict

**Pass on basic comprehension, but not yet “hard to misread.”** After one read I can answer all nine requested questions and the report is substantially self-contained. Its population definitions, the ordinary-contact nature of the anchor, the primary ±10 tolerance, and the limitations are particularly clear.

The main weaknesses are presentation rather than missing conclusions, but several are substantive report defects: the server-attribution figure can make sparse motion evidence look like a 239/239 motion classifier; the inpaint figure mixes denominators and collapses missing evidence with a negative trajectory decision; and the historical-versus-trend comparison contains an unexplained one-path difference in “usable” counts.

## Required fixes, ranked

1. **Rename the server-attribution motion methods to show their fallback explicitly.** For example, “Earliest-contact player; flip on 0.05-BH incoming evidence,” not simply “0.05-BH trend rule.” The figure should also expose that motion is usable on only **24/239** rallies. Otherwise the 163/239 bar is very easy to interpret as a full-coverage motion result.

2. **Fix the mixed denominators in the right side of the inpaint plot.** `Usable paths /135`, `correct returns /17`, `returns missed /17`, and `false returns /118` should not appear as directly comparable bars without those denominators. “Returns missed” should also visibly distinguish a usable below-threshold path from no usable motion evidence.

3. **Explain the 18-vs-19 and 9-vs-10 historical/trend usable-path counts.** The current description of shared eligibility does not explain why the historical rule loses exactly one otherwise usable path under either mask.

4. **Name the 0.05-BH rule and its call direction in the diagnostic/error plots.** `representative_rule_errors.png` should say these are the **eight 0.05-BH trend-rule errors**, ideally `8/19`, and the diagnostics should state that fitted decrease ≥0.05 BH means “incoming / first-return call.”

5. **Move the multiple-GT-window counts out of the gray “No GT stroke” stack in the alignment plot.** They are an overlapping ambiguity flag, not part of the gray category. Their present position visually communicates the opposite.

6. **Make the unmatched-anchor category logic self-contained.** State on the plot that later contacts are tested at ±10 and that the left categories use a priority such as “any later serve match; otherwise return; otherwise other; otherwise none.” This removes the need to read prose to know what the 49 and 36 actually classify.

## Optional improvements

A compact population funnel—**292 GT → 249 covered → 239 one-to-one → 135 unique ±10 serve/return truth → 19 usable unique-truth paths**—would make denominator changes much easier to retain. It would also help to explicitly reconcile the nearby 24-versus-19 usable-path and 15-versus-13 incoming-call counts: the former figures are over all 239 rallies, while the latter require unique ±10 serve/return truth.

I would also change “Both rules find 9 returns” to “Both correctly identify 9 of 17 GT returns,” and put “ordinary contact candidate; not a serve detector” in the first alignment figure or caption. Those are wording improvements rather than changes to the analysis.

## Answers to the nine questions

### 1. What do 292, 249 and 239 rallies mean, and why are different groups used?

292 is the full GT rally population and therefore the end-to-end denominator, including segmentation failures. Of those, 249 are considered covered by the current `COVERED` definition; this still includes ten GT rallies belonging to five predicted spans that each cover two rallies, so it is a merge-sensitive sensitivity population. The primary downstream population is 239 one-to-one rallies, each with one predicted span corresponding to one GT rally. Contact identity, trajectory classification, and primary server attribution use 239 because they need an unambiguous rally mapping.

### 2. What is the earliest accepted contact? Is it already required to look like a serve?

It is the **first ordinary contact-detector candidate inside the predicted rally that survives the released filters**. It comes from the standard shuttle-impulse/player-proximity contact process and may be rejected by wrist, suppression, or exclusion filters. It is **not a serve detector**, and it has no requirement to exhibit serve-like motion. The report is very clear on this point.

### 3. What do the ±5, ±10 and ±30 results say? Which tolerance is primary?

At ±5: 87 serve, 15 first return, 3 later stroke, 134 unmatched. At ±10: **119 serve, 19 first return, 4 later, 97 unmatched**. At ±30: 156 serve, 24 first return, 4 later, 55 unmatched. The wider window recovers more GT matches but also creates much more ambiguity: 117 ±30 windows contain multiple GT strokes. **±10 is the primary baseline**; ±5 is the strict view and ±30 only a sanity check.

### 4. What happened after the 97 anchors unmatched at ±10?

A later accepted contact eventually matched the serve in **49** rallies. In **36**, no later contact matched the serve but one matched the first return. Another GT stroke was the relevant later match in **9**, and **3** had no later GT match. The first later GT-matched contact was accepted-contact rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12. Those latter ranks total 94 because three rallies never acquire a later match.

### 5. How often is usable motion evidence available? Distinguish no evidence from a usable path that says serve.

Across all 239 one-to-one rallies, the recurrence-mask pipeline finds 57 continuous runs, 31 that satisfy point-count/contact-gap requirements, and only **24/239 usable paths** after the shared jump check.

For the 135 anchors with unique ±10 serve/return truth, only **19** have usable paths. Thirteen of those 19 are above the 0.05-BH incoming threshold: 9 are true returns and 4 are false-return calls on GT serves. The remaining **6 usable paths say “not incoming”**; four of those are GT returns that the rule misses and two are GT serves.

For the 17 GT returns specifically, the distinction is clean: **9 are correctly called incoming, 4 have a usable path that falls below threshold, and 4 have no usable path at all**. The latter two cases should not be presented as the same failure mode.

### 6. What changed in the inpaint comparison? Was the motion threshold held fixed?

Only the evidence mask changed: producer-marked filled/interpolated points were additionally excluded. The **0.05-BH threshold was held fixed**.

Overall usable paths fell from 24 to 14. Within unique ±10 truth, usable paths fell **19→10**, correct return calls **9→7**, false-return calls **4→0**, and missed returns **8→10**. Thus the stricter mask removed dubious false positives but also removed substantial useful evidence; every video lost evidence.

### 7. What did the historical 0.25-BH rule do compared with the predeclared 0.05-BH trend rule?

The historical rule requires 0.25 BH total movement, 0.25 BH net closure, and at least 55% of steps toward the player. The new rule uses a robust fitted decrease and calls incoming at **0.05 BH**, a value fixed before the corrected scoring and not swept.

With the recurrence mask, the historical row reports 18 usable paths, **9 correct returns, 3 false returns, 8 missed returns**, for 75% precision and 52.9% recall. The 0.05 trend row reports 19 usable paths, **9 correct, 4 false, 8 missed**, for 69.2% precision and the same 52.9% recall. Under the producer mask both have 7 correct, 0 false and 10 missed, although their stated usable counts are 9 and 10 respectively.

So the new rule does **not** demonstrate a scoring advantage over the historical rule. It shows what happens when the strong, path-length-dependent 0.25-BH floor is removed. The unexplained 18/19 and 9/10 usable-count difference needs clarification.

### 8. Does the server-identification idea help? Does prepending a contact and refitting help?

Yes, the direct server-identification idea helps on this dataset. The released alternating fit has **124/239** correct. Simply using the earliest-contact player reaches **152/239**. Using that player as the default and flipping it when the 0.05-BH motion evidence says incoming reaches **163/239 (68.2%)**; the historical rule reaches 162 and the producer-mask version 160.

The important qualifier is that the 163 result is a **hybrid fallback method**, not 239 motion measurements: usable motion exists on only 24 rallies.

Prepending a contact and recursively refitting does essentially **not** help: the two variants reach only **125/239** and **127/239**, versus 124/239 for the released fit. The useful result is the direct contact-player/incoming-motion clue, not refitting the alternating sequence.

### 9. What are the main caveats and the next practical step?

The strongest caveats are the very small trajectory truth sample—only 17 unique ±10 first-return anchors and 19 usable unique-truth motion paths—the image-based rather than physical body-height normalization, allowance of paths as short as five samples, an uncalibrated 0.05-BH engineering threshold, no independent ground truth for TrackNet positional error, severe ambiguity at ±30, and the fact that “GT-incompatible” is not a manual visual false-contact label. The same three videos also come from the historical exploration, so this is not external validation.

The practical next step stated by the report is to **improve anchor correctness and trajectory-path availability first**, while keeping segmentation/contact failures separate from trajectory-classification failures. The unchanged 0.05-BH rule then needs evaluation on new videos before its score is treated as general performance.
