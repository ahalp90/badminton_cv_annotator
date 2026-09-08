# Frozen stripe-observation experiment, 9 September 2026

The first assignment experiment reached 5/20 accurate fits. This follow-up asks
whether its raw-group restriction and stripe-position measurement lose useful
evidence. Geometry stays frozen while those two choices are varied separately.

## Fixed inputs and limits

Reuse the same twenty development frames, 714 original/refit geometries and
682 eligible geometries. Keep the extra gallery probe excluded and the empty
pool in all denominators. Preserve saved gates, net evidence and the 3:1
floor/net blend. No extraction, refinement, proposal search, training,
acceptance calibration or production changes. Labels enter after selection.

Use every clipped raw fragment from the first experiment's observation
preparation. The existing collinear groups supply only union-length weights:
divide each group's weight among its fragments in proportion to fragment
length. They do not limit which pieces may support a marking. All candidates
share the same observations and weights, including background fragments.

## Factorial comparison

Compare four selectors using the same eleven named marking identities and
their finite painted intervals:

| Measurement | Evidence may be reused | Each fragment has one marking/position |
|---|---|---|
| Nominal marking centre | Centre independent | Centre exclusive |
| Centre or either paint edge | Stripe independent | Stripe exclusive |

The centre measurement uses the existing metric template. The stripe
measurement adds lines offset by ±0.02 metres across each nominal marking,
representing a 40 mm painted stripe. The nominal line is retained as another
position hypothesis. This uses the template as a stripe-centre convention;
it does not resolve possible differences between nominal court dimensions,
paint boundaries and the manual references.

Each fragment may support one marking/position over its whole length. Choose
that pair by maximum mean reverse support against finite projected intervals.
Then compute forward support from the fragments assigned to that pair. Several
fragments may support one marking. The centre's two disjoint intervals share
one identity and the same across-stripe offset convention.

This assignment is optimal for the separable reverse objective. It is not a
global optimiser of the final forward/reverse mean. Record this limitation;
do not call a local assignment failure a failed exhaustive graph search.
The independent controls allow the same evidence to support other identities
and positions when computing forward coverage. Reverse support is unchanged.

For every model, forward support averages the best compatible fragment response
at each visible marking sample. Reverse support is the weighted mean explained
fragment response. Use their arithmetic mean for floor support. Retain the
five-degree direction gate and two-working-pixel Gaussian distance scale.
No parameter sweep follows the results.

Forward samples correspond to the same court locations across position
hypotheses. Offsets outside the image supply no evidence. Reverse distances
use finite painted intervals, so a line through the unpainted centre gap does
not gain support from an infinite extension.

## Diagnostics and iteration rule

Record complete candidate rankings and fragment identity/position assignments.
Also record the strongest alternative marking per fragment; these are
uncalibrated ambiguities, not acceptance probabilities.

Measure paired-edge support where the projected edges are at least four
working pixels apart (twice the distance scale), both lie in the image, and
the nominal marking is visible. Paired support is the weaker response on the
two edges at the same court location, with each fragment assigned to one edge.
It is a diagnostic only in this first run. Missing paired evidence does not
reject a court. It may identify a later, separately stated selector experiment.

Compare the four selectors with the previously recorded scores and raw-group
assignment. Report all twenty worst-corner errors, visible-landmark error,
candidate counts, runtime and useful-pool availability. Inspect yellow 156,
letterboxed 58 and centre 64 explicitly. The 15-pixel cutoff remains a
historical development metric, including off-screen corners.

Checks cover ideal and thick stripes, fragmented support, evidence exclusivity,
finite centre intervals, unresolved edges, endpoint/input/candidate-order
invariance, empty pools and unchanged geometry/labels. Run focused lint/types
and tests; the user asked to skip the whole-project Pyrefly follow-up.
Use bounded independent Fable review before publishing scientific conclusions.

If these measurements improve the named cases, examine whether improvement
comes from position tolerance or membership before adding a selector. If they
do not, inspect the assigned evidence and paired-edge diagnostic before
choosing the next experiment. Preserve unsuccessful runs and distinguish
post-run diagnostics from prespecified comparisons.

## Follow-up diagnostic: centre-line junctions

The four-way run selected 8/20 with either nominal-centre variant, 8/20 with
independent stripe positions and 9/20 with exclusive stripe positions. Yellow
156 still chose the wrong 653.42-pixel geometry. Paired paint-edge support did
not cleanly separate that winner from the accurate original. The next diagnostic
therefore examines finite marking identity directly.

At each transverse marking, sample the four arms around its intersection with
the centre line. Horizontal arms should contain paint. Vertical arms should
contain paint or lie in an unpainted interval, according to the finite template.
This distinguishes a baseline termination from a long-service crossing.

Each arm samples 0.04–0.24 metres from the junction, excluding the central
painted intersection. Use sixteen samples, the same centre/±0.02-m position
hypotheses, two-pixel distance scale and five-degree direction gate. An arm
needs eight unoccluded in-frame samples spanning eight working pixels.
Saved person boxes exclude samples; their frame offsets remain a limitation.

A site is usable only when all four arms are observable, both transverse arms
have support at least 0.55, and at least one vertical arm does too. On such a
site, support at least 0.55 is present, at most 0.20 is absent, and intermediate
values are unresolved. Record disagreements with the finite template. These
threshold values come from earlier support/absence settings. The earlier absence
setting was a binary coverage fraction; this one thresholds a Gaussian response.
The values were not tuned here.
Absent DeepLSD responses remain imperfect evidence of absent paint.

This is diagnostic only: no new selector or acceptance rule is applied. Inspect
all eligible frozen candidates, including the named yellow, letterboxed and centre
comparisons. A raw detector endpoint is never treated directly as a physical
termination. A later selector needs its own stated comparison and checks.

## Follow-up comparison: junction evidence and selection

The diagnostic found 159 of 682 eligible geometries with a junction disagreement.
None of the 78 geometries within the accuracy tolerance had one. Only 439
geometries had any usable site, so a lack of disagreement can mean a lack of
evidence. These are post-run development observations, used to choose the next
comparison. They do not establish a reliable rejection rule.

Freeze two simple ranking comparisons before computing their outcomes:

- **Contradictions first:** ascending number of disagreeing arms, then descending
  stripe-exclusive score, then candidate ID. Unobserved sites add zero. This
  tests whether the diagnostic can remove known aliases, and exposes escape
  through missing evidence.
- **Complete agreements first:** descending number of usable sites with both
  vertical arms conclusively agreeing, then ascending disagreeing-arm count,
  then descending stripe-exclusive score and candidate ID. This checks the
  effect of rewarding positive identity evidence. It can favour candidates
  that project more junctions into observable image regions.

Keep every eligible candidate in both full orders; change no acceptance flag.
Use the same twenty frames, frozen geometry, gates, stripe scores and recorded
junction responses. Compare with stripe-exclusive and the historical scores.
Read reference metrics only after forming all orders. Report changed winners,
correct candidates demoted, missing-evidence winners and the same 15-pixel
worst-corner tolerance. No weight or threshold sweep follows this comparison.
