# Visual inspection of automatic direction matching

These judgements apply to the [automatic gallery](automatic_axes_visual_check.html)
from the all-camera-eligible comparison. Candidate IDs below match the saved
[measurements](measurements.json.gz). They do not apply to the earlier supplied-
direction gallery or the label-guided observed-bank controls. The subsequent
[GX0 control approval](automatic_axes_results.md#gx0-follow-up-both-control-winners-are-visually-approved)
belongs to two different candidates and leaves these automatic rulings unchanged.

Across nine development views, the user explicitly finds a usable automatic
line-score winner in five views and a usable paint-ranked winner in four.
Six views have a usable winner under at least one ranking. This last count uses
the user's choice between rankings after inspection; no automatic selector
achieves that combined result here. Detector acceptance remains separate.

The line winners are usable on Amateur-3 and all four ShuttleSet views. The paint
winners are usable on Amateur-2 frame150, Amateur-3, ShuttleSet03 scene16 and
ShuttleSet21 scene20. Scene17's paint winner is described as worse with a far
boundary overshoot; no overall usability approval is inferred for that panel.
Neither ranking supplies a usable winner on GX0, GX5 or Amateur-2 frame28019.

## Candidate identity

| View | Automatic line | Automatic paint | Nearest final to inspected control | Closest before camera filtering |
| --- | --- | --- | --- | --- |
| GX frame 0 | 22:4579 | 22:4588 | 22:2055 | Same as nearest final |
| GX frame 5 | 181:29836 | 10:1274 | 108:4216 | 17:2407 |
| Amateur-2 frame 150 | 30:30 | 30:33 | 32:3188 | Same as nearest final |
| Amateur-2 frame 28019 | 16:1800 | 184:4123 | 16:1779 | 16:381 |
| Amateur-3 frame 0 | 43:22603 | 43:22627 | 43:22605 | Same as nearest final |
| ShuttleSet 03 scene 17 | 1:80 | 1:132 | 1:18152 | Same as nearest final |
| ShuttleSet 03 scene 19 | 1:60 | 165:6702 | 1:1278 | Same as nearest final |
| ShuttleSet 03 scene 16 | 1:60 | 1:90 | 1:5587 | Same as nearest final |
| ShuttleSet 21 scene 20 | 0:2 | 0:2 | Same as automatic winner | Same as automatic winner |

Repeated identities are not separate fits or additional judgements. Nearest-to-
control and pre-camera panels are selected using reference/control geometry
only after generation. They are diagnostics, not automatic winners.

## GX, frame 0

- **Line winner:** not really usable. The top-left corner lies up and right of its expected position; the bottom-left lies too far left. The court looks sheared. The bottom-right corner seems perfect. Without the shear, the fit would probably be good
- **Paint winner:** worse, with the same shear and an incorrect bottom-right corner
- **Nearest final:** worse again

The paint winner's smaller measured corner error does not reflect the user's
preference. This is a geometry failure despite relatively small corner errors.

## GX, frame 5

- **Line winner:** rejected; it draws a court on the wall. The user suspects a far-left net post and a person were treated as the two net posts
- **Paint winner:** rejected; it draws a court over the seated children. The user associates its left margin with the space between two net posts
- **Nearest final:** lies over the real court, but is badly undersized and appears sheared in multiple directions
- **Closest before camera filtering:** slightly better sizing and less obvious multidirectional shear, but still terrible

The suspected post/person correspondences are visual interpretations, not
verified explanations of which observations drove the algorithm.

## Amateur-2, frame 150

- **Line winner:** mistakes the blue mat boundary for the court. The sidelines otherwise align with the outer edges of the white paint
- **Paint winner:** described as perfect, including alignment with the outer paint edges. The extreme far end is uncertain because of perspective, but looks suitable
- **Nearest final:** follows the inner paint edges and overshoots the far boundary by a court section. Otherwise suitable, but explicitly not usable

## Amateur-2, frame 28019

- **Line winner:** extends to the blue mat borders and appears sheared in several directions. It roughly follows the right painted sideline, but not adequately
- **Paint winner:** rejected as an incoherent court near the net top
- **Nearest final:** similarly poor shear, but correctly reaches the near painted boundary instead of the mat boundary. The far boundary appears to reach the blue mat
- **Closest before camera filtering:** equally bad

## Amateur-3, frame 0

- **Line winner:** essentially perfect
- **Paint winner:** very usable. The near baseline follows the inner paint edge. The first horizontal in from the net visibly biases downward along its length
- **Nearest final:** very usable overall. That same internal horizontal crosses the painted line, lying above it at the left and meeting the vertex at the right. The user considers this internal line insignificant

The user suspects camera distortion behind the internal-line discrepancy. Its
cause remains unverified; the judgement does not establish an image-warp defect.

## ShuttleSet 03, scene 17

- **Line winner:** very usable with slight skew. Sidelines follow the inner paint edges. Horizontals appear to move from the top side of the paint on the left to the bottom side on the right
- **Paint winner:** similar but worse. The far baseline overshoots by an estimated 20 cm
- **Nearest final:** described as roughly the same as the preceding comparison; no separate overall usability judgement is inferred

The distance estimate is the user's visual description, not a new measurement.

## ShuttleSet 03, scene 19

- **Line winner:** usable. Sidelines are well aligned with the inner paint edges. The near baseline is perfect at the left and slightly inset at the right. The far baseline drifts from the inner paint edge on the left to the outer edge on the right
- **Paint winner:** rejected as an unrelated, hallucinated court
- **Nearest final:** usable, with a small bottom-right defect. The left sideline follows the inner paint edge and the right follows the outer edge. The horizontals have the same drift as the line winner

## ShuttleSet 03, scene 16

- **Line winner:** very usable. Sidelines follow the inner paint edges; the near baseline is essentially perfect. The far baseline drifts from the outer paint edge on the left to the inner edge on the right
- **Paint winner:** usable. Sidelines and near baseline follow the inner paint edges. The far baseline slightly overshoots on the left and slopes towards the outer paint edge at the right
- **Nearest final:** essentially perfect for practical purposes. The near baseline follows the inner paint edge; the lower half of the left sideline is slightly inset. The far baseline drifts from the outer paint edge on the left to the inner edge on the right

This feedback concerns the automatic-direction candidates. The earlier supplied-
direction scene16 candidates remain unjudged.

## ShuttleSet 21, scene 20

**Shared line/paint winner:** very usable. Sidelines follow the inner paint edges.
The near baseline follows the outer edge perfectly. The far baseline slightly
overshoots at the top left and otherwise roughly follows the outer paint edge.

## Implications

The feedback confirms usable automatic proposals on six views under a choice
between the two rankings. It also confirms poor geometry on the three difficult
views, including reference-selected diagnostics. Those displayed diagnostics do
not establish that every candidate in their full pools is unusable.

Preserving direction precision remains the next design question. GX0 now joins
GX5 and Amateur-2 frame28019 as an explicit visual regression case. A subsequent
GX0 control using discarded observed directions has visual approval for both
winners; automatic direction selection remains unresolved. Ranking remains separate: the usable Amateur-2 frame150 paint
winner and rejected ShuttleSet03 scene19 paint winner need opposite choices.

Minor paint-edge differences are acceptable in the named panels. There is no
new universal tolerance, verified lens-distortion explanation, acceptance rule
or production-readiness claim.
