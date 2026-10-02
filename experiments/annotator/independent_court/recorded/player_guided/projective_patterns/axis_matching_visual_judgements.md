# User inspection of the supplied-direction spacing and paint gallery

These verdicts apply to `axis_matching_visual_check.html` and the frozen
given-direction candidates. The main
right-hand paint-ranked court is deliberately repeated in the expandable
before/after comparison. Those panels are the same geometry, not independent fits.

Eight of the nine views received feedback. The new ShuttleSet 03 scene 16
candidates remain unjudged. The previous approval of its supplied control stands.
Minor paint-edge differences are accepted in the named views below. The user
prefers outer-edge alignment where practical; this is not a new global tolerance
or an instruction to change the court model.

| View | Line-score candidate | Paint-ranked candidate | Ruling |
| --- | ---: | ---: | --- |
| Amateur-2 frame 150 | 62 | 53 | Line-score winner is a disaster. Paint winner is essentially perfect and hugs paint more tightly than the previously usable control. Right outer line hugs the white inner; non-critical. Far-end judgement is limited by perspective. |
| Amateur-2 frame 28019 | 1195 | 1191 | Paint winner is absolutely usable but slightly worse than the supplied manual reference. Near baseline hugs white inner; right sideline is near white midpoint. Manual reference fits white outer. No separate new ruling on the line-score panel. |
| Amateur-3 frame 0 | 22189 | 22234 | Paint winner is definitely usable. Near baseline stays inside the white outer and may encroach a few millimetres at the right. Short-service lines inset slightly from their midpoints; acceptable guiding lines. Far right may be a few pixels inset. No separate line-score ruling. |
| GX frame 0 | 89 | 350 | Line-score winner is effectively perfect; tiny possible bottom-right inset and far-baseline overshoot are insignificant or hard to judge. Paint winner is usable in a pinch: back boundary overshoots by about one white-line width and left sideline hugs white inner. Strong preference for the line-score winner. |
| GX frame 5 | 89 | 89 | New winner is totally usable and about as good as the approved control. Near baseline hugs white outer. Left sideline drifts towards white midpoint near the service line; apparent termination before the right sideline may be inner-edge placement. No meaningful difference between rankings. |
| ShuttleSet 03 scene 17 | 2 | 53 | Paint winner is definitely usable. Outer horizontals largely hug white outer; far boundary may overshoot marginally. Sidelines hug white inner. Line-score winner looks better at the far line. |
| ShuttleSet 03 scene 19 | 0 | 78 | Paint winner is usable with small errors: sidelines hug white inner, near baseline may sit about 10 mm inward, far baseline about 10 mm outward. Line-score winner is slightly better, though usable court space is similar. |
| ShuttleSet 03 scene 16 | 2 | 329 | New candidates unjudged in this feedback. |
| ShuttleSet 21 scene 20 | 419 | 1327 | Both are functionally equivalent. Paint winner follows the reference at the top and hugs white outer at the bottom, with apparent slight central bowing; sidelines hug white inner. Line-score winner trades endpoint placement for better midpoint alignment at the near baseline. |

## Supplied control courts

- Amateur-2 frame 150: previous control remains fine and usable; new paint winner
  hugs the paint more tightly.
- Amateur-2 frame 28019: manual reference perfectly fits the outer white.
- Amateur-3 frame 0: manual reference is great and absolutely usable. Left
  sideline hugs white outer; near baseline drifts from outer to inner around its
  midpoint. Right sideline is near white inner, with perspective uncertainty.
- GX frame 0: manual reference is great and totally usable. Left sideline starts
  on white outer then moves around the midpoint nearer the camera; near baseline
  is around the white midpoint.
- GX frame 5: previous approved court is essentially perfect. Near baseline and
  left sideline follow white outer; bottom-right may sit around white midpoint.
- ShuttleSet 03 scene 17: manual reference is perfect and follows all outer borders.
- ShuttleSet 03 scene 19: previous approved control is totally usable, sitting
  between the inner edge and midpoint of the white lines.
- ShuttleSet 21 scene 20: previous approved control hugs white inner and is
  otherwise perfect.

## Interpretation after inspection

The supplied-direction matcher has usable generated courts across all eight
judged views. Paint-first ranking has a demonstrated benefit on Amateur-2 and a
demonstrated regression on GX0. Preserve both rankings when supplying automatic
directions; do not introduce a video-specific selector or declare paint-first
ranking the final rule. Scene 16 remains an unjudged generated-case check.

The user observes slight apparent bowing relative to the paint in scene 19's
manual reference and scene 20's near baseline. They suspect lens curvature and
consider correction unlikely to be worthwhile. The overlays use straight
projective segments. The physical cause remains unverified; no curved model,
camera recalibration or annotation edit is implied by this feedback.

Small reference errors do not substitute for these panel-specific visual
judgements. The subsequent [automatic-direction report](../../../../../../scratch/court_det_fix/evidence/direction_search/README.md)
records the completed follow-up.
