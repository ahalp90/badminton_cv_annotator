# Recommended next steps

## Highest-impact bounded follow-up

Add the same-paint measurement as a **diagnostic-only field** in the local
post-assignment verifier, then run it over a deliberately stratified gallery.
Do not change candidate generation or fitting for this experiment.

Recommended strata:

- yellow, white and mixed-colour court markings;
- white and coloured net tapes;
- true full courts and legitimate cropped courts;
- adjacent courts and non-court floor stripes;
- faded, repaired and glossy markings;
- strong player occlusion;
- low-resolution and compressed footage;
- courts where only lengthwise or only transverse reference paint is visible.

Record four outcomes separately:

1. evaluated, no contradiction;
2. evaluated, paint mismatch;
3. untestable because no supported far boundary;
4. untestable because no paint reference.

The untestable states must not become rejection by default.

## Calibration questions

Measure, rather than assume:

- minimum strong samples per fragment;
- minimum fragments per group;
- whether Lab `(a,b)` is preferable to another opponent-colour space;
- whether reference clustering is needed for multicolour courts;
- threshold on absolute separation;
- threshold relative to within-reference spread;
- behaviour under global colour transforms and white-balance changes.

A useful extension is to cluster all usable court-paint signatures and compare
the boundary to the nearest well-supported paint cluster, instead of assuming a
single colour. That extension remains unvalidated here.

## Regression requirements

At minimum, keep these fixed tests:

- Am1: the demonstrated mismatch remains visible;
- GX: no far-baseline assignment remains `untestable`, not rejected;
- changing case IDs does not change the result;
- reference corners are unavailable to the decision;
- off-frame corners alone never reject;
- empty or weak boundary groups never divide by zero or become false mismatch;
- image colour conversion and sampling are deterministic.

## Do not spend further time on these without new evidence

- all-corners-inside rules;
- condition-number-only rejection;
- the current LSD attached-mesh test;
- projected-net support using only retained court fragments;
- greyscale polarity as an object classifier.

## Remaining scientific limit

No single-image colour check can guarantee separation between neutral court
paint and neutral net tape. A future robust solution may need independent
height/floor-plane evidence, temporal motion/parallax, or explicit net
structure. None of those was demonstrated in this bounded audit, so they should
remain separate research leads rather than claims in the current patch.
