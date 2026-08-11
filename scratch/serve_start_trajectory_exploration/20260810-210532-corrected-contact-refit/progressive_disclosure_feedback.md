# Progressive-Disclosure Review Feedback

## Suitability verdict

**Yes — this version is suitable for a smart but persistently cognitively overloaded reader.**

It is materially better than the prior version because the report and figures increasingly make the intended interpretation the path of least resistance, rather than merely containing the right qualifications somewhere in the text.

For a rested technical reader: **very good**.

For the target reader described here: **about 8.5/10 overall**, with the plots closer to **9–9.5/10**. The remaining cognitive cost is now concentrated mostly in the prose hierarchy and number density, not ambiguity.

I would be comfortable sending this version as-is, but the changes below would make it more resilient to interruption, skimming, and depleted working memory.

## What improved substantially

### `server_attribution.png`

This is now excellent for the target audience.

“Earliest-contact fallback + motion-backed flips,” the explicit note that it changes only 15 incoming calls, and the reduced set of four outcomes solve the main prior comprehension trap.

A tired reader can now correctly retain:

> **Direct use helps; recursive refitting loses the gain.**

### `motion_evidence_and_inpaint.png`

This is much better.

Reducing the visual to the single question of evidence availability — **24 usable vs. 14 usable** — is the right progressive-disclosure choice.

The prior mixed-denominator classification details are better handled in a table where denominators can be stated explicitly.

### `unmatched_anchor_followup.png`

This is a strong simplification.

The title carries the conclusion, the subtitle states the ±10 rule and category priority, and the unnecessary rank distribution has been removed from the main visual.

That is well suited to information-saturated reading.

### Other ambiguity fixes

The earlier issues are also substantially resolved:

- the anchor plot now says the anchor is an **ordinary contact candidate**
- ambiguity counts are visually separated from **No GT stroke**
- the diagnostic states what **≥0.05 BH** means
- the error plot identifies itself as **all eight trend-rule mistakes**
- the historical **18-vs-19 / 9-vs-10** eligibility discrepancy is now explained

## Remaining cognitive-load issue

The main remaining problem is **document architecture rather than correctness or ambiguity**.

At roughly 3,900 words, the report uses progressive disclosure in principle, but the upper half still asks a depleted reader to retain too many details before reaching the action.

The `Bottom line` paragraph is accurate but number-dense. It asks the reader to process:

- 124/239
- 152/239
- 24/239
- 163/239
- 127/239
- 97/239
- 49
- 36

all in one short span.

A technically strong reader can parse this, but a cognitively saturated reader must repeatedly decide which values are **core memory anchors** and which are supporting evidence.

## Recommended final changes

### 1. Reduce the opening to three durable facts and one action

The opening should make only these points:

1. Earliest-contact player improves server attribution from **124 → 152**
2. Sparse motion correction improves **152 → 163**, but usable motion exists for only **24/239**
3. Recursive refitting falls back to **127**
4. Therefore, prioritize **anchor correctness and path availability**, not additional trajectory complexity

Then move the **97 / 49 / 36** unmatched-anchor explanation immediately below as evidence for that recommendation.

This would sharply reduce working-memory burden.

### 2. Move “What should we do next?” much earlier

For this audience, the action should appear immediately after the opening synthesis.

A reader interrupted after the first screen should already know the decision.

At present, the report makes the reader travel too far before reaching the practical implication.

### 3. Demote supporting tables from the main reading path

The following are useful, but expensive for a tired reader:

- the second population-by-video table
- the nine-row tolerance-by-video table
- the three successive server-attribution tables

Keep the central figure and primary 239-row comparison in the main path.

Move the sensitivity and by-video material under a clearly marked section such as:

> **Supporting breakdowns — optional**

This preserves rigor without forcing every reader through every denominator transition.

## Summary duplication

The `Summary` is somewhat redundant with the later question-driven sections.

For an overloaded reader, use one of these two structures:

### Preferred

**Very short executive summary → question-driven report**

or

### Alternative

**Long explanatory summary → substantially shorter downstream prose**

The current structure makes the reader learn several conclusions twice.

## Core success criterion

The redesign now succeeds at the most important compression task.

A reader can glance at the central figures and recover the correct story:

> **The anchor is often the problem → motion evidence is scarce → use motion as a small direct correction, not as a universal classifier or as input to recursive refitting.**

That is exactly the level of compression needed for a smart reader operating under sustained information saturation.

## Final recommendation

**Suitable to send now.**

If there is time for one final editing pass, prioritize only these three changes:

1. compress the opening
2. move the action earlier
3. demote supporting tables

Those changes would improve cognitive resilience more than further analytical detail or additional explanation.
