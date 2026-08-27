# Follow-up 3: can today's system make a tiny zero-error dataset?

## Bottom line

**Status: complete.** The frozen deterministic ladder could not keep a
non-empty set without observed end-to-end errors. See
[`../3_precision_first_dataset.md`](../3_precision_first_dataset.md).

The experiment tested whether a **very strict automatic rule** could keep a non-empty set of rallies with **zero observed end-to-end errors** on held-out labelled data.

Terrible recall is acceptable.

This does not prove literal 100% reliability on future videos. It tests whether today's signals can already isolate a tiny, extremely high-confidence subset.

## What counts as correct

A retained rally passes only when all required annotations are correct.

At minimum check:

- exact contact count;
- contact timing at the project's accepted tolerance;
- player order / attribution;
- server;
- point outcome.

Use the project's canonical ±5 base-30-frame contact tolerance for the primary
result. Report ±10 and ±15 as sensitivity checks, not as alternative rules.

Landing and hit-height estimates are outside this first completeness predicate.
The current pipeline treats them as experimental outputs. If either becomes a
required dataset field later, run a new evaluation rather than revising this
result.

One error means that retained rally is not perfect.

## Build the rule

Use only automatic signals that exist after Follow-up 2.

Possible inputs include current heuristics and already tested advisory signals from the chosen VLM.

Prefer using observations of court visibility, two detected players,
court-absence runs, shuttle visibility and explicit contact proximity directly
in the rule. Do not pass unexplained internal scores to the model. The evidence
limits are recorded in
[`compact_automatic_evidence.md`](compact_automatic_evidence.md).

The rule may reject almost every rally.

Prefer a simple rule that can be explained in a few sentences. Do not build a large learned meta-model for this test.

Freeze a short monotone rule ladder before opening held-out results. Fit only by
choosing the strictest useful rung on the two development fixtures. Do not fit a
classifier or run a dense threshold sweep.

## Test without leakage

Do not design a rule on all three fixtures and quote its score on those same fixtures as proof.

Use held-out evaluation. A simple option is leave-one-fixture-out:

1. choose thresholds/rules using two fixtures;
2. freeze them;
3. test on the third;
4. rotate if useful.

Report for each held-out run:

- rallies retained;
- rallies rejected;
- retained rallies with any error;
- which field failed when an error occurs.

Counts matter more than percentages when the retained set is tiny.

## Decision

A useful positive result is:

> The frozen automatic rule retained some rallies on held-out data and none had an observed annotation error.

A negative result is also useful:

> Even after sacrificing most recall, today's signals could not isolate a non-empty zero-error subset.

Do not call either result a literal guarantee beyond the tested data.
