Revise the current serve-trajectory report for a technically strong reader who is persistently cognitively overloaded by a high-information workday.

The goal is **not to simplify away technical value**. The goal is to preserve the analysis, caveats, diagnostics, denominator discipline, and reproducibility value while making the document substantially easier to enter, scan, resume after interruption, and understand without holding many facts in working memory at once.

Treat the report as a **durable standalone technical report** that may be read days or months from now.

## Critical framing change

Do **not** organize the report around correcting earlier erroneous measurements.

The prior incorrect or non-comparable results are historical provenance, not part of the main analytical story. A future reader should not need to know those earlier measurements ever existed.

In particular:

- Do not make the main report repeatedly distinguish the corrected prepend/refit result from the earlier `127/239` result.
- Do not put the superseded `127/239` number in the Bottom line or executive summary.
- Avoid language such as “fair variant,” “corrected comparison,” “earlier result,” “old-fit fallback,” or similar wording in the main narrative unless strictly necessary.
- Define the current methods once and compare them directly.
- If provenance must be retained, put it in a short note under Supporting details / Methodological note / Appendix. One concise paragraph is enough.

The durable comparison is:

- released alternating fit: `124/239`
- earliest-contact player baseline: `152/239`
- direct motion-backed correction: `163/239`
- prepend/refit using the same fallback and trigger set: `159/239`

The important local comparison is that on the 15 rallies where motion triggers, direct inference gets 13 correct and prepend/refit gets 9 correct.

The main conclusion is therefore that **recursive refitting does not improve the motion-backed inference in this sample**.

## Preserve the technical value

Do not delete analytical material merely because it is technical.

Preserve, somewhere in the report or supporting material:

- population definitions and denominator transitions
- tolerance sensitivity
- unmatched-anchor follow-up
- motion-path eligibility
- recurrence-mask vs producer-mask evidence comparison
- historical 0.25-BH rule comparison
- 0.05-BH robust trend rule
- threshold provenance
- false-return and missed-return counts
- distinction between no usable evidence and usable evidence below threshold
- path-quality requirements
- diagnostic distributions
- path length / scatter / trend-to-jitter diagnostics where useful
- per-video breakdowns
- representative failure cases
- caveats and limitations
- reproducibility-relevant methodological details

However, technical material does **not** all need to be in the main reading path.

Use progressive disclosure aggressively:

1. main conclusion
2. evidence needed to understand that conclusion
3. optional diagnostics and sensitivity analysis
4. supporting tables / audit detail / provenance

The report should allow a reader to stop after each layer without having misunderstood the result.

## Target reading experience

Assume the reader:

- is technically capable
- can understand statistics and methodological detail
- is not confused by complexity itself
- has very limited working-memory bandwidth
- may be interrupted repeatedly
- may skim before deciding where to read closely
- should not need to remember a denominator introduced several pages earlier
- should not have to infer which numbers are core conclusions versus supporting diagnostics

Optimize for **cognitive accessibility, not intellectual simplification**.

Prefer:

- short sections answering one question each
- explicit local denominators
- descriptive headings that carry conclusions
- one conceptual transition at a time
- compact tables with a clear reason to exist
- optional supporting sections for dense diagnostics
- repeated context where it reduces working-memory burden

Avoid:

- dense paragraphs containing many competing numbers
- several tables in succession without interpretation
- repeated retelling of the same conclusion
- mixing algorithm definition, eligibility, scoring, diagnostics, sensitivity, and error analysis in a single section
- forcing the reader to reconstruct the main analytical story from methodological details

## Rewrite the opening

The opening should contain approximately **three durable facts and one action**, not a catalogue of every result.

The reader should be able to retain something close to:

1. Earliest-contact player improves server attribution from `124 → 152`.
2. Sparse motion evidence can improve this to `163`, but usable motion exists for only `24/239` rallies.
3. Recursive prepend/refit does not improve that signal and reaches `159`.
4. Therefore, prioritize **anchor correctness and usable trajectory-path availability** before adding trajectory complexity.

Immediately after this, explain why anchor correctness matters using the `97` unmatched anchors and the later-contact `49 serve / 36 first-return` recovery result.

Do not overload the opening with every sensitivity count.

## Move the practical conclusion early

Put a section equivalent to **“What should we do next?”** immediately after the opening synthesis and essential evidence.

A reader who stops after the first screen or two should already know:

- what was learned
- what did not help
- what bottleneck matters most
- what practical next step follows

The rest of the report should substantiate that conclusion.

## Rebuild “Detailed motion methods and diagnostics”

The current section is too cognitively dense because it mixes too many reasoning modes.

Do not merely edit its sentences. **Restructure it.**

Replace it with a layered structure similar to:

### How the motion correction works

Keep this short and conceptual.

Explain:

- what pre-contact path is used
- basic eligibility / quality requirements
- robust fitted change in shuttle-to-player distance
- `≥ 0.05 BH` fitted decrease means “incoming”
- incoming evidence flips the earliest-contact player interpretation
- otherwise motion does not change the earliest-contact baseline

State clearly that motion is a **sparse correction**, not a universal classifier.

### How much usable evidence is there?

State the key population locally.

For the `135` anchors with unambiguous ±10 serve/return truth:

- `19` have usable motion under the primary recurrence mask
- `13` are called incoming
- `9` are true returns
- `4` are false-return calls on serves

For the `17` GT returns:

- `9` are correctly called incoming
- `4` have usable paths but fall below threshold
- `4` have no usable path

Preserve this distinction. Do not collapse “no evidence” and “negative motion decision” into one missed-return category without explaining the split.

### Do the diagnostics suggest an obvious better rule?

Summarize the answer first.

The usable paths show substantial overlap, and path length / residual scatter / trend-to-jitter do not currently justify an additional decision cutoff.

Treat these quantities as **descriptive diagnostics**, not hidden classifier rules.

Show only the most useful diagnostic figure(s) in the main report.

The fitted-decrease-by-truth view is the most important. If the four-panel diagnostic is still useful for audit purposes, move the full version to Supporting diagnostics.

### Historical-rule comparison

Put the historical `0.25 BH` comparison in its own optional subsection.

Explain succinctly:

- what the historical rule required
- what the `0.05 BH` trend rule changes conceptually
- their primary-mask precision/recall result
- why usable-path eligibility differs by one path, if applicable

Do not let this comparison interrupt the main explanation of the current method.

### Failure cases

In the main prose, state simply that the usable-path errors comprise:

- 4 false return calls on GT serves
- 4 missed GT returns with usable paths

Move the eight individual rally traces / IDs to a supporting diagnostic section or appendix unless they are needed for a specific argument.

The representative-error figure is valuable audit evidence, but it should not be required reading to understand the main result.

## Figures

Keep figures that carry a clear standalone conclusion.

The strongest main-path figures are those showing:

- anchor alignment
- what happens after unmatched anchors
- scarcity of usable motion evidence
- server-attribution comparison

For diagnostics:

- strongly consider simplifying the main `trend_and_jitter_diagnostics` view to the fitted-decrease distribution with the `0.05 BH` threshold, optionally plus one diagnostic view
- move the full four-panel version to supporting material if it remains useful
- move the eight-case error figure to supporting diagnostics / appendix unless the main prose directly depends on inspecting those examples

Every main-path figure should answer one obvious question.

## Tables

Retain detailed tables where they provide technical value, but demote tables that are primarily:

- by-video breakdowns
- sensitivity breakdowns
- audit detail
- provenance
- secondary diagnostics

Create a clearly marked section such as:

## Supporting breakdowns

This section can contain the full technical tables without making them compulsory reading.

Do not put several dense tables back-to-back in the main narrative unless each is essential to the next conclusion.

## Avoid duplication

The report currently risks teaching the same result in both a long Summary and later question-driven sections.

Choose a clearer architecture:

**Preferred:**

- very short executive synthesis
- practical implication
- question-driven analytical sections
- supporting breakdowns / diagnostics / methodological notes

Do not repeat the full analysis in both the summary and body.

## Standalone story the report should communicate

A reader should be able to leave with this coherent model:

1. We have `239` clean one-to-one rallies for the primary downstream analysis.
2. The earliest accepted contact is an ordinary contact candidate, not a serve detector.
3. Its player identity is already useful for server attribution, improving `124 → 152`.
4. The earliest contact is often not the true serve anchor: `97/239` are unmatched at ±10, and later contacts recover the serve or first return in most of those rallies.
5. Pre-contact motion can correct some of these cases, but usable evidence is scarce: `24/239` overall.
6. Used as a sparse direct correction, motion improves server attribution to `163/239`.
7. Recursive prepend/refit does not improve this signal and reaches `159/239`; on the 15 triggered rallies it gets 9 right versus 13 for direct inference.
8. The practical bottleneck is therefore **anchor selection and trajectory-path availability**, not lack of a more elaborate refitting procedure.
9. Motion thresholds and diagnostics remain preliminary because the usable truth sample is small and requires validation on new videos.

Everything in the report should support, qualify, reproduce, or deepen that story.

## Final quality check

Before finishing, reread the report as if:

- you had not seen any previous version
- you did not know any erroneous early result existed
- you were opening it after a full day of technical meetings
- you could be interrupted after any section

Check that:

- every section has one clear purpose
- important denominators are local
- main conclusions appear before supporting detail
- technical detail remains available without blocking comprehension
- no obsolete comparison dominates the narrative
- the report reads as a cohesive standalone analysis rather than a correction memo
- the reader can recover the main argument by reading only headings, opening paragraphs, central figures, and the final recommendation

Do not reduce rigor. **Reduce the amount of rigor the reader must hold in working memory at the same time.**