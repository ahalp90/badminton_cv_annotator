# Research frontier around net-assisted court localisation

This note records high-information possibilities rather than a fixed plan. The
historical branch work and the local audit suggest that net evidence has real
value, although the useful formulation may differ substantially from the old
projected-line score.

## 1. Independent semantic net observation

A candidate-independent observation could represent several components rather
than one line:

- top-tape polyline or probability field;
- left and right post hypotheses;
- coarse net-region or mesh probability;
- lower-band probability where visible;
- per-component visibility and uncertainty;
- multiple competing net hypotheses in multi-court scenes.

Such an observation adds information that does not originate from the floor
homography. A court candidate can then explain, fail to explain, or remain
agnostic about the observation.

Possible implementations range from classical image structure to learned
segmentation or keypoint models. The branch already contains many labelled or
reviewed court images, overlays, and candidate populations that could support
weak supervision or targeted annotation.

## 2. Court-conditioned association rather than global net selection

A scene may contain several nets, adjacent courts, railings, and wall seams.
The useful latent variable is likely a pairing:

```text
(court candidate, net hypothesis)
```

rather than a single globally selected net. Pairwise compatibility can include:

- agreement between projected and observed top tape;
- post placement and vertical direction;
- expected net width relative to the court;
- visibility-aware component residuals;
- player occupancy on opposite sides of the paired net;
- temporal persistence across a stable camera view.

This formulation naturally preserves uncertainty when several pairings remain
plausible.

## 3. Net/floor evidence ownership

Am1 exposes a deeper issue than candidate ranking: the same physical object can
be assigned as a court marking. A semantic ownership layer could allow a
fragment or pixel region to support one interpretation while contradicting
another.

A simple conceptual state space is:

```text
floor paint | net tape/post/mesh | unrelated structure | unresolved
```

The useful quantity may be a likelihood ratio rather than a hard class. Strong
net ownership would suppress baseline support; unresolved ownership would leave
the original floor evidence unchanged.

This direction connects directly to the same-paint chroma result. Chroma is one
hand-crafted ownership signal. A richer embedding could combine colour,
texture, width, continuity, occlusion, height cues, and temporal behaviour.

## 4. Oracle-net ablation over frozen candidate pools

A high-information diagnostic could separate three questions that the old
replay entangled:

1. whether accurate net observations contain enough information to improve
   court ranking;
2. whether a practical image detector can recover those observations;
3. how the resulting evidence interacts with floor, paint, and player scores.

The oracle version would use manually marked visible top tape and posts while
leaving candidate geometry and base scores frozen. Candidate-independent
annotations would make the comparison less circular than projected generic-line
support.

Interesting outcomes include:

- Am1's false net-band court moving down the ranking;
- GX remaining viable despite cropped or weak net components;
- the historical Yellow improvement surviving;
- the Letterboxed regression disappearing or becoming explainable;
- multi-net scenes resolving through candidate/net association;
- net-only or non-court controls remaining unsupported as courts.

The value of this experiment lies in attribution. A negative oracle result
would reduce the case for detector development; a strong oracle result would
clarify the target representation before model work.

## 5. Joint geometric inference

The regulation net is a known-height structure tied to the court midpoint. An
independent net observation can constrain camera pose and homography jointly,
not merely rerank completed quads.

Possible formulations include:

- a probabilistic factor graph over court homography, focal length, vertical
  direction, and net components;
- robust bundle adjustment with floor markings and net keypoints;
- differentiable rendering of floor paint and net geometry against image
  feature maps;
- a mixture model over multiple courts and nets;
- a scene-level latent camera shared across frames.

The old camera helper supplies a useful initialisation but embeds strong
intrinsic assumptions. A joint model could expose principal point, distortion,
and focal uncertainty rather than hiding them in a scalar residual.

## 6. Stable-view temporal evidence

The branch evidence already favours scene-level reuse and multi-frame proposal
access. Nets are often temporally stable while players and shuttle move across
them. Temporal aggregation may improve:

- top-tape continuity under player occlusion;
- post visibility;
- separation of static net structure from moving bodies;
- association of the played court through player trajectories;
- confidence in a shared court/net pairing.

The same stability can also reinforce a persistent wall or railing, so semantic
ownership remains relevant. Camera-change detection and mixture handling are
part of the open problem rather than assumed preconditions.

## 7. Richer negative evidence

A useful net-aware system may gain more from contradictions than rewards. Some
examples:

- a high-confidence observed net lying far from the candidate's projected
  midcourt;
- a claimed baseline overlapping a net-region probability field;
- posts incompatible with the candidate's vertical direction;
- observed players repeatedly crossing the candidate's alleged outside edge
  while remaining separated by a different net;
- a candidate's projected net landing on a wall structure with no corresponding
  mesh or posts.

Asymmetric evidence fits partial views naturally: a visible contradiction can
be strong, while missing components remain neutral.

## 8. Learned boundary semantics

The branch contains enough diversity for a representation-learning approach to
be plausible. A model need not output the final court. It could supply local
semantic evidence around candidate markings:

- floor-paint probability;
- net-tape probability;
- post probability;
- court-surface probability;
- embedding consistency between boundary and interior markings.

Candidate-conditioned crops may reduce annotation cost and class imbalance.
Hard negatives such as Am1, wall rails, adjacent courts, advertising boards,
and inner/outer paint edges are especially informative.

## 9. Evaluation shape

The historically mixed result suggests several axes of evaluation rather than a
single pooled score:

- correct played-court identity;
- far-boundary accuracy;
- partial-view retention;
- multiple-net association;
- non-court rejection;
- calibration of abstention;
- sensitivity to camera assumptions;
- gains attributable specifically to net evidence;
- runtime and model dependency footprint.

The existing development corpus is valuable for mechanism discovery, while its
own decision documents describe it as non-representative and non-held-out. A
frontier exploration can still extract substantial information from it before a
larger validation set exists.
