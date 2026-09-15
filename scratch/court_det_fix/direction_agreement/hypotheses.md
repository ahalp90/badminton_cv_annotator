# Hypotheses and per-arm evidence

Neither hypothesis is established. The anchor and the representative score are the packet's declared experimental choices, not geometric truth.

## H1: measuring line agreement near visible evidence improves membership

Change: the agreement angle for a merged line is measured at the projected midpoint of its longest clipped contributing fragment instead of at the line's foot from the image centre. Infinity directions are unaffected by construction.

Arms: M (anchor change alone) and MR (with precision representatives).

Evidence: filled from E1 (residual and membership changes), E3 (control fits) and E4 (matcher results for M).

## H2: choosing a precise representative after coverage allocation improves retention

Change: within each coverage leader's suppression bucket, the representative is the candidate with the lowest `mean(min(angle, 1.5)^2)` over the leader's fixed support lines, ties by candidate ID. Leaders alone drive coverage; representatives never feed back into allocation.

Arms: R (representative change alone) and MR.

Evidence: filled from E2 (how often the choice differs from the leader), E3 and E4 (matcher results for R).

## Fixed-support SVD

Diagnostic only: each arm's representatives refitted by SVD on their own support masks. Not promoted to matching.

## Limitations

- The experiment cannot rescue a direction outside the allocated buckets.
- Control fits are least-squares diagnostics against a control, not generated courts.
- Pre-per-pair-cap geometry is unavailable; per-pair losses are not measured.
