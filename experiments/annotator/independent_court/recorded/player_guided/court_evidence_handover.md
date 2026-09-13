# Court evidence investigation: handover

The sampling experiments are complete. Keep the original sampler and scorer.
Covered-length ranking improves all seven GX numerical coverage minima versus
random, but accepts no GX court. The user rejects its closest frame-5 court as
a disaster compared with the approved complete-search example.
The [latest report](evidence_ranked_seeds.md) records all 17 runs and replay.

The next prepared experiment estimates two common line directions, prunes the
existing merged pools and enumerates their surviving rectangles/templates.
Rectification and badminton line-pattern matching remain the next branch if
that smaller test is insufficient. Neither generator change is implemented.
Production, original detection, fitting code and annotations remain unchanged
on branch fix/court-det.

## What is established

- **Sampling loses useful geometry.** The approved GX5 complete-search court
  uses pair-product seed 85795, with covered-length ranks [0, 16, 6, 9].
  Random, spread and all tested covered-length weights omit it. Complete-search
  minima are label-guided coverage diagnostics, not automatic selections.
- **Numerical improvement is not visual recovery.** The user called the
  [spread-sampled court](seed_selection.md) a significant regression. The later
  covered-length sample has maximum corner error 16.30 px and paint-inset
  boundary RMS 4.84 px at 1280×720; the user called it a disaster. Other new
  GX geometries have no visual usability judgement.
- **Evidence access and acceptance differ.** The
  [raster-preserving fixed-court experiment](raster_court_matching.md) restores
  support for the usable GX5 court and rejects both known wrong courts. It also
  rejects an established good Amateur-3 fit. It is diagnostic evidence, not a
  validated replacement scorer.
- **False support is real.** In Amateur-3 frame 0, raw fragments 0/62/84 are
  definitely not court lines, probably sunlight through windows. Raw 108 is the
  outer right boundary with a slight inset. None supports the far short-service
  marking. The user resolved this visual question; do not widen direction
  tolerance to recover these matches.
- **Local refinement helps a close start.** From the approved GX5 court,
  legacy/physical refits improve boundary RMS from 1.47 to 1.01/1.35 px.
  Maximum corner error rises from 3.37 to 3.97/5.09 px. Both retain original
  floor rejection. The user judged the right-hand refits marginally better
  and said they perfectly hug the outer white-line boundary. Both starts and
  refits look fine; no preference between paint conventions was stated.
  See the [close-start control](gx_proposal_trace.md#optional-close-start-control--13-september).

The approved complete-search court's mild right-edge inset was acceptable;
left and bottom follow the outside of the paint. This approval is specific to
that court. Frame-0 legacy refit remains useful but imperfect, with top-left
undershoot and bottom-right overshoot; its physical refit was marginally worse.

## Latest experiment and controls

Three predeclared covered-length weights ran on GX0/GX5 and two accepted controls
(12 runs). Only weight 1 qualified for the five-frame GX extension. It improves
both coverage minima on all seven GX frames versus random. Against spread
sampling, corner coverage improves on five, ties on one and worsens on one.

Weight 1 preserves acceptance on Amateur-2 frame 150 and broadcast scene 0019,
with closer emitted courts than random. Its accepted Amateur-2 court is a different
seed, not recovery of the old emitted one. All seven GX cases still reject:
12 proposals pass floor support across the seven cases, then all fail camera
checks. The closest new GX5 proposal instead fails floor support; its separate
camera diagnostic passes. Do not confuse those populations.

The earlier spread rule loses the Amateur-2 output and worsens broadcast selected
corner error from 10.31 to 54.02 px. Neither sampling rule is adopted.
The [seed report](evidence_ranked_seeds.md) distinguishes pre-gate coverage,
emitted controls, independent boundary minima and direct versus traced timings.

## Next experiment: projective patterns

The useful hypothesis is shared evidence across several lines, not merely a new
way to construct a quadrilateral. Two observed lines per direction still fit
any chosen pair of template coordinates exactly. Additional coherent lines,
spacing and finite marking extents must resolve those aliases.

Use separate stages:

1. Estimate competing vanishing-point pairs from cached fragments. Prune the
   existing merged families by directional agreement, then enumerate surviving
   rectangles with the unchanged templates. Test both direction roles.
2. Check known seed retention and frozen controls. Preserve original merging,
   line caps, scoring and gates; record pruning losses and any search truncation.
   Declare a budget/fallback for broad pools, especially broadcast.
3. If this remains insufficient, check homogeneous basis construction and axis
   matching on synthetic courts. Supply directions from the approved GX5 court
   as a label-guided matching diagnostic, then use automatic direction estimates.
4. Compare frozen GX and accepted controls, then wrong-court/background cases.
   Attribute losses before changing ranking or acceptance.

Pruning can remove search clutter but cannot restore lines already lost through
family assignment or merging. A wrong direction estimate creates another loss
point. Retain competing pairs and distinguish estimator scales from acceptance
thresholds. Saved reference/refitted courts need residual diagnostics; they do
not necessarily have an exact original seed identity.

Reuse detector.X_COORDS/Y_COORDS and the existing corner template. Two vanishing
points give affine rectification with aligned axes; the known pattern supplies
the remaining axis scales and offsets. Retain finite marking identities:
unique offsets alone lose the centre line's unpainted middle.

Keep finite and infinite vanishing points, axis roles/signs, raw/group identities,
duplicate-edge ambiguity and competing courts explicit. Preserve conditioning,
residuals and losses at computational limits. Image-angle buckets are not a
substitute for testing common projective directions. Ordering requires a
consistent finite chart; a projective pole can invalidate naive order matching.

Start with single-frame cached observations. No general graph package, broad
scorer rewrite or scene-pooling framework is needed for this test. Measure
selected diagnostic candidates before the old floor gate hides them, while
retaining its outcome. That does not authorise emitting rejected courts.

Declare estimator scales and engineering budgets separately from acceptance.
Use continuous coverage, provenance, runtime and visual assessment without
inventing a new pixel success threshold or required number of wins. Labels must
not choose automatic proposals. A precise failed-stage diagnosis is a valid result.

## Replay, validation and history

- [Evidence-ranked seeds](evidence_ranked_seeds.md): latest result and portable
  archive, including selection smoke checks and exact seed-loss reconstruction.
- [Simple spread selection](seed_selection.md): earlier rejected rule and controls.
- [GX proposal trace](gx_proposal_trace.md): complete-search examples and both
  rejected-start and close-start refinement controls.
- [Raster court matching](raster_court_matching.md): 63 frozen courts on 22 frames,
  access ablations, decisive fragments and raw-186/97 sample-placement attribution.
- [Finite court matching](fixed_court_matching.md): earlier distance-method
  regression; read only when that comparison is relevant.

Latest experiment runs, scoped lint, synthetic smoke, archive replay and content
checks passed (exit 0). The technical review checked selection source and its
baseline behaviour; it did not audit the new scoring outputs or seed-loss helper.
The report records its floating-point ordering limitation. No whole-project
runtime checks were needed because shared implementations did not change.
