# Distortion ridge check: gate outcome and readings

## Gate outcome: FAIL

The gate requires both control views (Amateur-3 and ShuttleSet-03) to have
every edge's absolute sagitta under 1.5 native pixels. They do not. The
method is not clean enough to trust a GX-vs-control comparison, for two
different reasons on the two controls, and the thresholds have not been
tuned to hide this.

**Amateur-3**: three of its four edges exceed the gate by a wide margin
(4.03, -5.47, 4.04 px). Walking the raw per-position offsets for these
edges shows why: the offset flips sign and magnitude from one sample to
the next (for example, the far baseline runs ...+11.3, -10.2, -9.0, ...
+11.5... across neighbouring positions). That is not a smooth bow; it is
the ridge finder locking onto a different bright feature from one sample
to the next; most likely an adjacent parallel line (the doubles/singles
sideline pair sits close enough that both can fall inside the +/-12 px
window) or a floor reflection. The left sideline also has a roughly
300-pixel gap in the middle of its span where every sample fell below the
25-grey-level threshold (a reflective patch on the floor there), so its
quadratic fit is anchored almost entirely by the two end clusters and is
not a trustworthy sagitta.

**ShuttleSet-03**: three of its four edges are comfortably inside the
gate, but the near baseline sits at -2.32 px, over the 1.5 px limit. Unlike
Amateur-3, this reading is not noisy: the raw offsets move smoothly and
monotonically from +1.0 px at one end, through -1.16 px at the middle,
back to +1.1 px at the other end, with a line-fit RMS residual of only
0.67 px. This looks like a genuine small bow (or a small corner-placement
offset in the E3 control) rather than a method artefact, and it alone is
enough to fail the gate on a court that is supposed to be a clean
reference.

## Table

| case_id | role | edge | status | clipped_length_px | n_survive | rms_residual_line_px | sagitta_px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gxBQ_window_00_frame_0 | GX0 (test) | far_baseline | ok | 472.00 | 11 | 6.9080 | 4.8139 |
| gxBQ_window_00_frame_0 | GX0 (test) | near_baseline | ok | 622.98 | 59 | 2.3319 | 1.7710 |
| gxBQ_window_00_frame_0 | GX0 (test) | left_sideline | ok | 847.66 | 48 | 2.3270 | 2.3424 |
| gxBQ_window_00_frame_0 | GX0 (test) | right_sideline | ok | 959.69 | 25 | 6.0333 | 3.4734 |
| am3_window_00_frame_0 | Amateur-3 (control) | far_baseline | ok | 654.50 | 33 | 6.9078 | 4.0283 |
| am3_window_00_frame_0 | Amateur-3 (control) | near_baseline | ok | 1031.77 | 59 | 2.9793 | 0.2759 |
| am3_window_00_frame_0 | Amateur-3 (control) | left_sideline | ok | 711.08 | 32 | 4.5503 | -5.4683 |
| am3_window_00_frame_0 | Amateur-3 (control) | right_sideline | ok | 1202.65 | 56 | 2.6439 | 4.0385 |
| shuttleset_03_scene_0016 | ShuttleSet-03 scene 16 (control) | far_baseline | ok | 323.83 | 53 | 1.2033 | -1.0306 |
| shuttleset_03_scene_0016 | ShuttleSet-03 scene 16 (control) | near_baseline | ok | 548.36 | 58 | 0.6718 | -2.3193 |
| shuttleset_03_scene_0016 | ShuttleSet-03 scene 16 (control) | left_sideline | ok | 298.68 | 59 | 0.1724 | -0.3248 |
| shuttleset_03_scene_0016 | ShuttleSet-03 scene 16 (control) | right_sideline | ok | 296.74 | 60 | 0.2101 | -0.5360 |

(clipped_length_px is the length of the edge after the 20-pixel margin
clip; n_survive is out of 60 sampled positions; sagitta_px is signed
positive when the bow points away from the image centre.)

## Answers to the brief's questions

Because the gate fails, these answers describe noisy observations, not
a confirmed distortion measurement. GX's sagittas (1.77 to 4.81 px) are
larger than three of ShuttleSet's four control readings (0.32 to 1.03 px),
about the same as ShuttleSet's worst edge (2.32 px), and smaller than
Amateur-3's noisiest edge (5.47 px, which is itself a method artefact, not
a real bow). So GX does not clearly stand out above what the method
already produces on a control it is supposed to pass. GX's sign is
consistent on every edge (all four sagittas are positive, meaning every
edge bows away from the image centre), which would be compatible with barrel
distortion rather than pincushion; for comparison, ShuttleSet's four
sagittas are all negative (a small, consistent pincushion-direction
signal), while Amateur-3's signs are mixed, matching its noisy readings.
The largest observed GX sagitta is 4.81 native pixels, on the far baseline.

These methods therefore establish no validated GX-specific upper bound.
Further distortion work is deferred for priority reasons, not because the
explanation has been disproved. No fix is proposed here.
