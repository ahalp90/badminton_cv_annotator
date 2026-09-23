# Acceptance path and identifiability

## Actual caller path

The refit child does not bypass the existing gates.

1. `run_w5.make_parent_record` applies `verifier.hard_validity` to a parent and
   measures evidence only for a hard-valid parent.
2. `run_w5.attempt_refit` freezes the parent's stripe assignments, creates fixed
   constraints, runs `fixed_stripe_refit.refine`, and checks success, rank 8,
   positive corner denominator, finite corners and convexity.
3. It reconstructs a child homography, calls the imported
   `run_diagnosis.gate_evidence`, reruns `hard_validity`, and remeasures evidence.
4. `process_case` puts hard-valid parents and children into the ranking pools.
5. `verifier.rank_candidates` restricts final R1/R2 ordering to hard-valid,
   `camera_eligible` candidates and ranks by span-weighted paint or geometry
   evidence.

## What the gates establish

| Property | Established? | Mechanism |
|---|---|---|
| finite homography | yes | `hard_validity` |
| positive corner denominators | yes | refit and hard validity |
| convex projected quad | yes | refit and hard validity |
| minimum visible image span | yes, inside `_score` geometry filtering | detector geometry |
| some line-family support scores | measured | `_score` |
| player observations lie inside projected court | measured; used by historical full-court predicate | `zone_net.player_fractions` |
| restricted pinhole-camera plausibility | yes, under square-pixel/centred-principal-point assumptions | `camera_diagnostic.camera` |
| disputed fragment is floor paint | **no** | no physical-object measurement |
| boundary is not net tape | **no** | projected net is not compared with the disputed fragment |
| all court corners are visible | deliberately no | partial views are supported |

## Important gate detail

`run_diagnosis.gate_evidence` calls `detector._score`. If `_score` returns one
geometrically valid projected candidate, `gate_evidence` sets
`geometry_valid = True` even when the returned floor score is `-1.0` because
line-count or family-support thresholds failed.

Both frozen strong-centre cases record `floor_score = -1.0`:

- Am1 line counts `[1, 1]`, family support `[0.425, 0.5521]`;
- GX line counts `[2, 4]`, family support `[0.4306, 0.6597]`.

`hard_validity` consumes `geometry_valid`, not `floor_score`. Final
`camera_eligible` consumes only `camera_error <= 0.1`. Thus a candidate with
`floor_score = -1` can remain in the final camera pool.

The historical full-court predicate additionally checks players, but final R2
selection does not require that predicate; it is reported as a subset ranking.

## How Am1 passes

The Am1 strong-centre fit records:

- 224 fixed samples from 14 retained fragments;
- 96 samples from six fragments labelled `far_baseline`;
- no right-sideline or near-baseline membership;
- objective `4.0102 -> 0.6189`;
- Jacobian rank 8, condition `941.27`;
- minimum corner denominator `0.06005`;
- camera error `0.06926`;
- player fractions `[1.0, 1.0]`;
- two very distant, off-image near corners.

Those values are compatible with all hard-valid and camera-eligible tests while
the first court edge follows the net-bottom band.

## Why fitting cannot validate the label

`fixed_stripe_refit.prepare` freezes each selected sample's finite marking
interval and centre/edge position. The residual is the weighted distance from
that sample to the nearest point on its assigned projected finite segment.

Convergence therefore means:

> a homography exists that places the already-labelled segments near the points.

It does not mean:

> those points physically arose from those court markings.

## Small projective depth counterexample

Let the alleged far baseline be `v = 0` in normalized court coordinates and let
`H` be any image homography. Define

```text
      [1  0  0]
G_t = [0  1  0]
      [0  t  1]

H_t = H G_t
```

Every point `(u, 0, 1)` is fixed by `G_t`, so the complete alleged far baseline
is fixed pointwise. The line `u = 0` is also preserved, although points can slide
along it. A point on the opposite edge `v = 1` maps to

```text
(u / (1+t), 1 / (1+t)).
```

For `t = -0.5, -0.9, -0.99`, the opposite edge expands by factors 2, 10 and 100
while the far baseline remains exact. Am1's other interior constraints break
this exact null family—its final Jacobian is full-rank—but the construction
shows why a boundary plus one-sided arms does not by itself identify court
depth.

## Semantic counterexample

Construct two images with identical retained line geometry, brightness profile,
polarity, player feet and camera fit:

- one contains a 40 mm floor marking;
- the other contains white net tape with the same projected cross-section.

Every existing measurement is identical, yet only one line is a court boundary.
This proves that the current evidence is not semantically identifiable without
an additional object- or material-sensitive measurement.
