# Frozen court-marking assignment experiment, 9 September 2026

Test whether explicit observation-to-marking assignment improves selection on
the existing twenty development frames. No court-graph matcher has been run
before this experiment. The later centred-stripe ledger result is a separate,
combined intervention and is not a graph-matcher baseline.

## Frozen inputs

Read `marking_inputs.json.gz` and `reverse_results.json.gz` directly from
[the existing replay](marking_refit_replay.zip). Preserve all original/refitted
geometries, candidate source IDs and recorded eligibility decisions. Exclude
the labelled extra gallery probe. Keep the empty pool in every denominator.
Do not run extraction, proposal search or refinement.

The shared eligibility rule is the recorded `eligible` flag plus the
`original` scheme's eligibility. Every selector sees this same pool. Retain all
its alternatives; sort complete scores using stable source IDs for ties.
No incremental suppression or acceptance threshold is applied.

## Observation and court representation

Use raw DeepLSD fragments in the replay's working coordinates (maximum image
dimension 960 pixels). Form candidate-independent collinear groups with a
three-degree direction tolerance and a three-pixel perpendicular tolerance.
Use a deterministic longest-first leader rule with coordinate tie-breaking.
Retain raw fragment IDs and finite extents. These are observed line groups:
grouping does not establish that a group is one physical painted stripe.
Stripe-edge uncertainty and distinct unresolved stripes remain limitations.

Project the existing metric court under each frozen homography. Treat the
centre marking as one identity with two painted intervals. The other ten
identities are four sidelines and six transverse markings. Only finite painted
intervals count as support; the centre gap supplies none. Clip to the image.
Use the existing minimum visible length of twelve working pixels.

All observations in the image remain in the shared comparison region. A
candidate cannot remove observations by shrinking its court footprint. Unmatched
observations lower explained coverage; they are not hard contradictions or
proof that a candidate is wrong. No paint classifier is used.

## Selectors and controls

1. **Recorded existing and bidirectional scores:** reuse the saved raw-line
   scores as historical comparators on the shared eligible pool.
2. **Independent group support:** each marking may independently use its best
   observed group, and each group may independently use its best marking.
3. **Partial assignment:** maximise total support with at most one group per
   marking and one marking per group. Unmatched nodes are permitted. This is a
   small bipartite matching problem; finite court structure is supplied by the
   common metric homography. There is no junction classifier or graph-edit
   search in this first implementation.
4. **Raw differential ledger control:** compare explained groups, missing
   visible markings and residual support using these same raw observations and
   frozen geometries. Posts, candidate-dependent centring and refits are
   disabled. This is an adapted control, not a reproduction of the 10/20 run.

For both support selectors, use finite-segment distances, a five-degree
direction gate and a Gaussian distance scale of two working pixels. Measure
forward support per visible marking and reverse support per observed group,
weighted by its union length. The objective is the arithmetic mean of forward
and reverse support. This additive objective permits exact maximum-weight
assignment. The independent control uses the identical components without
exclusivity. Retain the recorded floor/net weighting of 3:1 for both selectors.

For the ledger control, use the existing five-pixel support tolerance, 20%
absence threshold, 50% explanation threshold, and 40/100-pixel group-length
thresholds. Compare long groups explained only by the opponent, then absent
markings, explained groups and continuous forward support. Count pairwise wins;
use stable source IDs for final ties. Report cycles or tied leaders where they
occur. No posts, paint filter, person-box exclusion or candidate-specific court
crop enters these three new selectors.

## Measurements and checks

Freeze the protocol before the full run. Labels enter only after ranking.
Report every case, selected source/stage, objective components, marking
assignments, missing/unmatched evidence, candidate counts and runtime.
Keep the historical worst-corner cutoff of fifteen pixels at 1280×720 and
visible-landmark RMS as separate geometric measurements. Report useful-pool
availability before and after the shared gates. Do not infer semantic accuracy
from corner error alone; inspect the yellow, letterboxed and centre examples.

Meaningful checks cover finite centre intervals, evidence reuse, unmatched
observations outside a proposed court, endpoint/input-order invariance,
candidate-order invariance, empty observations/pools, and unchanged geometry.
An independent review checks the implementation and results before close-out.

No score sweeps, stripe-aware refinement, new proposals, acceptance calibration,
training, production changes or main merge belong to this experiment. A negative
result limits this representation and objective; it does not refute the wider
court-graph proposal.
