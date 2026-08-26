# Follow-up 3: can today's system make a tiny zero-error dataset?

## Bottom line

With the chosen VLM, test whether a **very strict automatic rule** can keep a non-empty set of rallies with **zero observed end-to-end errors** on held-out labelled data.

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

If another field is required for the intended dataset, include it before the rule is frozen.

One error means that retained rally is not perfect.

## Build the rule

Use only automatic signals that exist after Follow-up 2.

Possible inputs include current heuristics and already tested advisory signals from the chosen VLM.

The rule may reject almost every rally.

Prefer a simple rule that can be explained in a few sentences. Do not build a large learned meta-model for this test.

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
