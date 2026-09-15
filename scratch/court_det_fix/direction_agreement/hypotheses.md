# Hypotheses and per-arm evidence

Neither hypothesis is established. The anchor and the representative score are the packet's declared experimental choices, not geometric truth. Numbers below come from `runs/<run>/summary.md`; E4 sections are filled when the matcher stage completes.

## H1: measuring line agreement near visible evidence improves membership

Change: the agreement angle for a merged line is measured at the projected midpoint of its longest clipped contributing fragment instead of at the line's foot from the image centre. Infinity directions are unaffected by construction (verified on every bank).

Arms: M (anchor change alone) and MR (with precision representatives).

E1 evidence (membership, not ground truth):

- Anchors move a median 137-195 working px from the foot, up to 497 px. Between 40 and 81 of each case's lines come from a single fragment, so their anchor is that fragment's own midpoint.
- Residuals change a lot for most candidate-line pairs (median absolute change 3.9-7.7 degrees, 90th percentile 26-34 degrees), but membership at 1.5 degrees changes for only 0.8-2.7 percent of entries. Per case, 2,400-5,300 candidates gain at least one supporting line and 970-3,300 lose one.
- The largest residual changes are finite candidates that sit at one of the two anchors. At the foot the ray is undefined (residual about 90 degrees) and becomes well defined at the midpoint anchor (about 1 degree); a candidate at the midpoint anchor shows the reverse. All 20 largest changes in every case have one of these two forms (7-16 foot-blind and 4-13 midpoint-blind entries per case).
- The number of candidates with at least two supporting lines is identical under both anchors, because a pair candidate always lies on both generating lines and infinity candidates ignore the anchor.
- The coverage allocation is sensitive to the anchor: M shares 0-5 of 16 leaders with B by candidate ID in every case. Geometrically the overlap is larger: across the 144 M directions, the median angle to the nearest B direction is 0.9 degrees, but 34 of 144 lie more than 5 degrees from every B direction (90th percentile 10.1 degrees, maximum 36.2).

E3 evidence (best finite control fit of any ordered pair, working px): M against B is 19.7 vs 6.8 on GX0 (worse), 10.6 vs 23.8 on GX5 (better), 1.9 vs 3.3 on Am2-150 (better), 2.3 vs 3.0 on Am2-28019 (better), 4.2 vs 3.7 on Am3-0 (worse), 2.8 vs 2.0 on SS03-17 (worse), 2.29 vs 2.27 on SS03-19 (worse by 0.02), 1.8 vs 1.5 on SS03-16 (worse), 1.03 vs 1.03 on SS21-20 (identical). Direction-fit potential moves in both directions; GX0 and GX5 move oppositely.

E4 evidence: pending.

## H2: choosing a precise representative after coverage allocation improves retention

Change: within each coverage leader's suppression bucket, the representative is the candidate with the lowest `mean(min(angle, 1.5)^2)` over the leader's fixed support lines, ties by candidate ID. Leaders alone drive coverage; representatives never feed back into allocation.

Arms: R (representative change alone) and MR.

E2 evidence: the precision rule changes the representative in 5-13 of 16 buckets per case for R (83 of 144 in total) and 6-12 for MR (87 of 144). B's bucket sizes range from 1 to 229 members with a median of 10; 14 of the 144 buckets are singletons and cannot change. Where R changes the representative, its own support count is at or below the leader's in 80 of 83 cases (median one line fewer, at most four fewer, at most three more). R's directions stay close to B's: median 0.08 degrees to the nearest B direction, 2 of 144 more than 5 degrees away.

E3 evidence (R against B, working px): 6.2 vs 6.8 on GX0, 23.2 vs 23.8 on GX5, 2.8 vs 3.3 on Am2-150, 1.4 vs 3.0 on Am2-28019, 4.3 vs 3.7 on Am3-0 (worse), 0.6 vs 2.0 on SS03-17, 1.9 vs 2.3 on SS03-19, 0.6 vs 1.5 on SS03-16, 1.4 vs 1.0 on SS21-20 (worse). Seven of nine improve; the best pair of groups is the same as B's in every case.

E4 evidence: pending.

## Fixed-support SVD (diagnostic only)

Each set's representatives refitted by SVD on their own support masks. SVD moves directions a median 0.1-0.4 degrees, at most 3.3 degrees. It helps B on eight cases (SS03-19 and SS21-20 by under 0.1 px) and hurts on Am3-0, and the pattern is uneven for the other arms (for example M+SVD is worse than M on Am2-150, 3.9 vs 1.9). The best finite pair converged in every set. Not promoted to matching.

## Limitations

- The experiment cannot rescue a direction outside the allocated buckets.
- Control fits are least-squares diagnostics against a control, not generated courts; three controls are manual references without visual approval.
- Pre-per-pair-cap geometry is unavailable; per-pair losses are not measured.
- Membership counts describe the merge assignment, not physical stripes.
