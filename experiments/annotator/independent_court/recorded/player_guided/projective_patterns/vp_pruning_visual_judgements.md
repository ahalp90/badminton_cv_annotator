# User judgements of the VP-pruning viewer

These judgements refer to the panels in the
[inspected viewer](vp_pruning_visual_check.html), supplied on
14 September after the nine-case scoring run. The viewer is preserved as
inspected; its pending-judgement captions predate this record. Detector decisions
and corner-error measurements remain separate from the user's visual assessment.

Small offsets within the width of the paint were acceptable in several named
examples. This establishes no universal numerical tolerance or preferred paint
edge. No overall usability verdict is inferred where the user gave only detailed
observations. Estimates of centimetres or millimetres below are the user's
descriptions, not newly measured distances.

## GX — frame 0

**Closest generated court:** a gross regression relative to the image shown the
previous night that the user described as a perfect fit of the outer lines.
The exact prior comparison image is not identified by this feedback; do not
silently assign that approval to a frame-0 result.

- The back boundary is placed on the long-service line.
- The right doubles sideline lies halfway between the real doubles and singles
  sidelines.
- The near back boundary slightly overshoots the paint and lies outside the
  court. This particular error is very minor and acceptable.
- The left outer line starts on the inner paint edge and gradually drifts to
  the outer side by its top-left point.

## GX — frame 5

**Closest generated court:** looks good; the earlier approval is reaffirmed.

- The right outer line runs through the middle of the paint rather than its
  outer edge. The user describes the roughly 2 cm difference as inconsequential.
- The left outline hugs the outer paint edge.
- The bottom outline hugs the outer paint edge almost perfectly, with an
  insignificant inset.

## Amateur-2 — frame 150

**Top retained court:** perfectly usable. Borders tend towards the inner paint
edge, which is fairly insignificant.

**Closest generated court:** about the same as the top retained court; the user
cannot really see a difference.

## Amateur-2 — frame 28019

**Closest generated court:** the nearest back boundary and long-service line
are of the same quality as frame 150. Both short-service lines mildly overshoot.
The furthest lines cannot be judged because of perspective. The user assumes
they are roughly fine; this is not confirmed visibility or an overall approval.

## Amateur-3 — frame 0

**Top retained court:** roughly the same issues as Amateur-2 frame 28019, with
the following detail:

- The nearest back boundary and long-service line are fine. The left portion
  of the back boundary tends towards the inner paint edge.
- The left outer line lies on the inner paint edge.
- The right boundary and far back boundary slightly undershoot their true lines.
- Internal lines are roughly accurate on the left. Towards the right, they turn
  inward and undershoot the true lines. This is very noticeable at the vertices.

**Closest generated court:** about the same as the top retained court.

## ShuttleSet 03 — scene 17

**Top retained court:** the nearest three horizontals are essentially perfect,
even though the nearest lies on the inner paint edge. The outer sidelines are
also described as perfect on the inner paint edges; the user calls the roughly
44 mm loss manageable. However, the far short-service, far long-service and
far back lines are all substantially inset.

**Closest generated court:** the same problems as the top retained court.

## ShuttleSet 03 — scene 19

**Top retained court:** the same problems as scene 17: good near horizontals
and outer sidelines, with substantial inset of the three far horizontals.

**Closest generated court:** usable, with noticeable problems:

- The back border overshoots and lies slightly outside the court.
- The first long-service line is slightly inset relative to its true position.
- Other lines are essentially fine. The far back border may overshoot, but
  perspective makes that difficult to judge.
- The outer sidelines follow the inner paint edges and at least find the lines.

## ShuttleSet 03 — scene 16

**Top retained court:** the same issues as scene 17's top retained court.

**Closest generated court:** all horizontals progressively overshoot. The error
is insignificant at the nearest point and becomes a problem at the far back
line. The sidelines hug the inner paint edges.

## ShuttleSet 21 — scene 20

**Top retained court:** basically fine, with a slight but significant issue:

- The outer lines hug the inner paint edges.
- The far back line overshoots at the top left and undershoots at the top right.
  The user describes it as not even a right-angle line. This records the visual
  complaint; it does not impose a 90-degree angle in the projected image.
- The top-left corner is significantly inset from the sideline, with that drift
  appearing only in the final court quadrant.

**Closest generated court:** the user compares it with scene 17's top retained
court, then explicitly describes progressively overshooting horizontals: minor
near the camera, problematic at the far back line. Sidelines hug the inner paint
edges. Preserve this explicit overshoot description; scene 17's earlier inset
description does not silently replace it.

## Interpretation after inspection

The user approves the GX5 geometry and calls Amateur-2 frame 150 perfectly
usable. ShuttleSet 03 scene 19's closest generated court is usable with defects;
that judgement does not transfer to its top retained court.

The GX0 ruling rejects its nearest-by-corner-error proposal. The ShuttleSet
observations also expose substantial marking errors in detector-accepted outputs.
Four detector acceptances therefore cannot be reported as four visually usable
courts. Small maximum corner errors are insufficient evidence of marking accuracy.

These findings motivated the subsequent [marking diagnosis](marking_diagnosis_results.md),
which distinguishes wrong marking identity from poor geometry. Neither the best
corner-error candidate nor these selected panels establish whether better
candidates exist elsewhere in the population.
