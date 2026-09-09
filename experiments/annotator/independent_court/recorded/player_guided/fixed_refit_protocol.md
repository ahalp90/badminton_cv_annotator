# Fixed-assignment stripe refinement, 9 September 2026

The junction comparison still selects 9/20 accurate courts. Complete junction
agreements reduce yellow frames 14 and 156 from errors above 600 pixels to
16.56 and 16.71 pixels. Penalising contradictions alone lets an unobserved
alternative win yellow 156. These results support testing identity evidence
further; they do not establish an acceptance rule.

This next experiment asks whether explicit paint-edge positions improve geometry
when the marking assignment stays fixed. It is a refinement diagnostic, chosen
after the stripe and junction comparisons. It changes geometry for the first
time in this continuation. It does not reuse a parent's eligibility or net
score as if those quantities applied to the new geometry.

## Frozen inputs and controls

Use all 682 eligible starting geometries from the same twenty development
frames. Keep the empty frame in the denominator. Retain every starting geometry.
Use the stripe-exclusive fragment assignments already recorded for each start.
The raw observations, canonical fragment IDs and union-length weights stay fixed.

Keep fragments whose saved mean reverse support is at least 0.55. For each of
their sixteen samples, choose the nearest finite painted interval belonging to
the assigned marking and edge position under the starting homography. Keep
samples within five working pixels. Freeze those sample IDs and finite interval
IDs before optimisation, including the centre line's two separate intervals.
This is a local fitting subset, not a new declaration that other evidence is
irrelevant. No observations are reassigned during optimisation.

Compare two fits from each identical starting homography and fitting subset:

- **Nominal centre:** set each assigned interval's across-stripe offset to zero.
- **Fixed position:** retain each fragment's saved centre or ±0.02-metre edge
  position. The existing metric template is treated as the stripe centre.

Both minimise weighted squared distances from the fixed observed samples to
their assigned finite projected intervals. Distribute each fragment's shared
union-length weight equally among its retained samples. Use one common
eight-parameter homography, with court and image coordinates normalised for
numerical conditioning. Use a central finite-difference Jacobian and a relative
rank tolerance of the square root of machine epsilon. SciPy least-squares uses
at most 100 function evaluations
per unique state. There is no assignment update, robust-loss sweep, threshold
sweep, reference-guided initialisation or best-of-label selection.

An exact key containing the starting coordinates, model and frozen fragment
assignment reuses identical optimisation states within a frame. Record attempted
states, unique states, duplicates and exhausted budgets. This is a finite replay,
not a branching graph search; it needs no approximate state hash.

## Measurement and limits

Record starting and fitted corners, objective before/after, solver status,
evaluation count, Jacobian rank/conditioning, and projective validity. Insufficient
constraints, a rank-deficient fit, a non-converged solver or a projection crossing
the horizon stays an explicit diagnostic failure. Preserve any finite attempted
geometry separately; do not silently substitute it into a ranking.

Attach reference measurements after all fits finish. Report the same worst-corner
error at 1280×720, including off-screen corners, with a 15-pixel accuracy cutoff.
Also report visible-landmark RMS. Compare each fit with its own start, including
regressions among the 78 initially accurate geometries. Separately trace the
already selected stripe and complete-junction winners through each fit; their
identities remain fixed. An empty or failed fit cannot become a success.

Count useful pools descriptively, with original and successful refined geometry
retained. This uses labels to measure availability, not to select a candidate.
New geometry has not passed renewed player/net checks, so these counts are not
end-to-end detector or acceptance results. Renew those checks before any later
selection experiment involving the new geometry.

If the fits improve near variants while preserving identity and accurate starts,
the next step is renewed evidence and selection with all alternatives retained.
If they systematically drift, inspect the frozen fragment/position constraints
before widening proposal search. Report mixed results and their causes rather
than choosing a per-frame fitting mode.

## Follow-up: renewed evidence with every starting fit retained

The diagnostic completed all 1,364 unique fits within budget. Both models retain
9/20 accurate winners when each selected identity is followed through its fit.
They improve yellow 90 but lose letterboxed 58. Nominal-centre fitting loses
10 of the 78 accurate starts; fixed positions lose four. Both raise descriptive
pool availability from 13 to 14 frames when starts are retained.

The next comparison retains each original eligible geometry and each successful
new geometry. Recompute geometry validity, player fractions, camera consistency,
floor gates and net evidence using the exact archived scoring implementation.
Reproduce the archived gates and net scores on all 682 starts. Fail the
comparison if that control differs. Never inherit those quantities from a parent.

For floor evidence, keep every parent's stripe marking/position assignment fixed
for both refinement controls. The nominal-centre fit changes its fitting target;
it is evaluated with the same stripe model and assigned positions as the other
geometries. Recompute forward and reverse responses under the new homography.
An assigned fragment with zero support keeps its identity and contributes zero.
Recompute finite junction observations too; their threshold classifications may
change with geometry. Such a response change is not itself a changed assignment.

Compare stripe-exclusive and complete-agreements-first selection on three pools:
starts alone; starts plus nominal-centre fits; starts plus fixed-position fits.
Retain complete deterministic orders and use candidate ID for exact ties. Keep
the same 3:1 floor/net blend and junction ordering. Do not introduce new weights,
thresholds, acceptance rules, suppression or proposal search. Attach labels only
after all scoring and ranking. Report renewed eligibility losses, the original
9/20 control, changed winners, regressions and useful geometry after renewed gates.
