# What GX0 merged rows 31, 51 and 61 lie on

## Gate

The gate passed. The member fragment IDs and covered lengths below come
straight from the E0 record
(`scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e0/gxBQ_window_00_frame_0.json.gz`,
`membership` list, matched by `line_id`). The step 1 mask check also passed:
row 31 and row 61 are both `True` in candidate 737's support mask, and row 51
is `True` in candidate 4104's support mask
(`scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e2/gxBQ_window_00_frame_0.json.gz`,
`arms.B.support_masks`, indexed against `arms.B.leader_candidate_ids`).

## Table

| row | member count | covered length (px) | my read |
|-----|--------------|----------------------|---------|
| 31 | 4 | 153.6 | vertical seam in the corrugated wall cladding, up near a structural post, on the back wall above the playing floor |
| 51 | 1 | 101.8 | a tan/beige floor seam or mat-joint strip running through the green playing surface, near the sideline |
| 61 | 1 | 82.5 | the jawline/cheek contour of a spectator sitting close to the camera in the bottom-right corner of frame, not any court or building structure |

## Relation to the control court

Row 31 sits well outside and above the court, on the shed wall, far from any
of the four yellow control corners or their connecting lines. It does not
run parallel or coincident with any control-court edge in the crop; it is
simply a tall, roughly vertical mark that happens to share the same rough
direction as many other near-vertical wall seams and posts nearby (seen as
the other green/blue support fragments in the wide overlay).

Row 51 sits on the floor near the left sideline of the near court, close to
where the yellow control polygon's edge runs, but it is not on that edge; it
runs at a shallower angle through the mat surface and reads as a mat seam or
patch line rather than a marked court boundary.

Row 61 sits in the bottom-right corner of the frame, near where the green
playing surface meets a wooden floor strip and the crowd is seated close to
the camera. The nearest control-court line is off to the left, not touching
the person or this red mark; row 61 has no spatial relationship to the
control court beyond being roughly in the same corner of the image.

## Is this court paint

None of the three rows is court paint. Row 31 is a wall-cladding seam, well
off the floor entirely. Row 51 looks like a floor mat seam or joint tape,
which is a floor feature but not a painted court line. Row 61 falls on a
person's face, not on any structure at all. This read is my visual judgement
from the crops, not a measurement against any known court-line geometry.

## Checked by the orchestrating session

The orchestrating Claude session viewed row_31.png, row_51.png and row_61.png on 2026-09-15 and agrees with the read above: wall cladding seams, a tan floor strip beside the sideline, and a spectator's face. None is court paint.
