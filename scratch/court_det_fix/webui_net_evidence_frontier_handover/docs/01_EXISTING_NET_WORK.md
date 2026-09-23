# Existing net work in the branch

## Summary

The branch contains projected net geometry, synthetic projection tests, a
historical image-support scorer, and a portable three-clip replay. It does not
contain a candidate-independent semantic detector that first identifies a net
from image content and then associates it with a court.

## Historical introduction

Commit `be94d28dbc78289aa21386bbefcf059d416f407e` introduced the experiment under
the message:

> Use net geometry to investigate ambiguous court fits

The commit added a metric net projection helper, synthetic pinhole tests, and a
three-clip temporal replay. The commit text explicitly described the output as
diagnostic and stated that it defined no new acceptance rule.

## Geometry helper

Path:

```text
experiments/annotator/independent_court/net_geometry.py
```

Main entry point:

```text
project_net(corners_px, frame_size) -> NetProjection
```

The helper derives an approximate camera from a candidate court quadrilateral
under these assumptions:

- pinhole camera;
- square pixels;
- principal point at image centre;
- focal search from 0.4 to 4.0 image widths;
- regulation net placement at the court midpoint;
- 1.55 m post height and 1.524 m centre height.

Its output contains two top-tape halves, two posts, a camera residual, and the
selected focal length in image-width units.

The corresponding tests are in:

```text
tests/test_independent_court_net_geometry.py
```

Those tests validate the projection numerically against an independent synthetic
camera, scaling behaviour, and invalid-input handling. They do not detect a net
from pixels.

## Historical image-support scorer

A frozen implementation remains at:

```text
scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py
```

For each floor-supported candidate, this implementation:

1. projects the physical net from the candidate court;
2. discards camera residuals above 0.1;
3. samples 48 positions along each visible projected net component;
4. measures Gaussian distance support from a map built from all raw line
   fragments;
5. blends the result as

```python
scores[index] = (3 * scores[index] + net_support) / 4
```

The measurement is therefore court-conditioned and line-generic. A railing,
wall seam, adjacent net, floor stripe, or other linear structure can provide
support. It is a discriminator between candidate projections rather than an
independent statement that a particular object is a badminton net.

## Historical replay result

The compact replay is represented by:

```text
experiments/annotator/independent_court/recorded/player_guided/replay.zip
experiments/annotator/independent_court/recorded/player_guided/summary.json.gz
experiments/annotator/independent_court/recorded/player_guided/yellow_net.jpg
```

The recorded worst-corner errors at 1280 × 720 were:

| Clip | Floor only | Floor + net | Pooled refinement and rescoring |
| --- | ---: | ---: | ---: |
| Yellow | 221.19 px | 26.01 px | 21.22 px |
| Letterboxed | 10.06 px | 17.38 px | 14.75 px |
| Centre | 4.04 px | 4.04 px | 3.20 px |

The Yellow result demonstrates that net geometry can break a severe ambiguity.
The Letterboxed regression demonstrates that the cue is not automatically
benign. The final stage also changes geometry and pools fragments, so the replay
does not isolate temporal aggregation, net scoring, and refinement as separate
causes.

The detailed account lives at:

```text
experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_assessment.md
scratch/court_det_fix/evidence/independent_proposals/README.md
```

## Current W5 relationship

The modern W5 route still projects net geometry through its camera diagnostic,
but its gate records the camera residual rather than image support along the
net. In the audited path, the projected segments are discarded and only
`camera_error` survives into the gate evidence.

Relevant paths include:

```text
scratch/court_det_fix/w5_holistic/run_w5.py
scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py
scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py
scratch/court_det_fix/w5_holistic/verifier.py
```

The branch decision document explicitly distinguishes the older image cue from
the newer projected-post diagnostic and lists “net-image evidence plus pooled-
fragment refinement” as an older mechanism with renewed interest:

```text
scratch/court_det_fix/DETECTOR_DECISIONS.md
```

## Why the old score appears to have stayed outside the core

The historical record supports several interpretations:

- the cue can be highly informative when competing floor candidates imply
  visibly different physical nets;
- the cue can worsen a previously good candidate;
- candidate-conditioned search is partly circular;
- generic lines are not semantic net evidence;
- player-pair selection and camera stability were separate unresolved inputs;
- multiple-net and partial-view scenes weaken any single global net choice;
- absence of visible net support is not evidence that the court is wrong.

This history makes the net idea active rather than exhausted. It also explains
why the old 3:1 blend never became a general admission rule.
