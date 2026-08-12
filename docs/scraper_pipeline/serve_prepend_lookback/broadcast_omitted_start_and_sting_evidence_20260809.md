# Broadcast-omitted rally starts and replay-sting evidence

## Decision supported by current evidence

The issue 28 result still rejects the tested raw-impulse plus central-pose
serve-lookback prototype. Human broadcast labels do not select its candidates
or match them to ShuttleSet frames. Removing questionable target rows cannot
turn any selected false positive into a target match.

The phrase `136 missed serves` is not sufficiently supported. The measured
quantity is 136 unmatched first ShuttleSet strokes where at least one later
stroke matched. Some first rows carry ShuttleSet's own quality flag and an
unknown stroke type. A separate visual annotation is required before reporting
a missed-visible-serve percentage.

Repeated broadcast stings are a plausible match-local replay-boundary signal.
The current timelines strongly support the pattern in `sset_01`. They do not
yet establish the same pattern in `sset_15` or `sset_21`.

## Sources and grain

The findings use these committed inputs:

- the per-rally committed results in
  [`data/serve_prepend_lookback_189c5af_20260808/`](data/serve_prepend_lookback_189c5af_20260808/);
- [`shots_master.csv`](../../../training/data/shuttleset/annotations/shots_master.csv);
- the raw ShuttleSet `setN.csv` files under
  [`training/data/shuttleset/annotations/set/`](../../../training/data/shuttleset/annotations/set/);
- the canonical human broadcast timelines under
  [`broadcast_nonstandard_camera_id/data/`](../broadcast_nonstandard_camera_id/data/); and
- the candidate and matching definition in
  [`serve_prepend_measurement.py`](../../../src/annotator/calibration/serve_prepend_measurement.py).
- the official ShuttleSet
  [`Movement Forecasting/data_cleaner.py`](https://github.com/wywyWang/CoachAI-Projects/blob/45517f7d4cb936b03f3eabf939cc7959d39226fe/Movement%20Forecasting/data_cleaner.py#L19-L23)
  treatment of non-null `flaw` rows.

The target grain is one ShuttleSet rally. The target filter is
`status == serve_missed_later_strokes_matched` in each committed-rally CSV.
The code currently assigns `serve_frame = rally.stroke_frames[0]` without
checking the raw stroke type, upstream `flaw` field, or visual visibility.

## First-stroke quality finding

For each target rally, exactly one complete row with `ball_round == 1` was
selected per `(set_id, rally)`. Its frame was required to equal the committed
`gt_serve_frame`. This matters because two `sset_01` rallies have ball rounds 1
and 2 at the same minimum frame.

| Video | Current target rows | First row `flaw=1` | First row type `unknown` |
| --- | ---: | ---: | ---: |
| `sset_01` | 63 | 2 | 1 |
| `sset_15` | 39 | 0 | 0 |
| `sset_21` | 34 | 24 | 24 |
| **Pooled** | **136** | **26** | **25** |

The pooled quality-flag share is 26 of 136, or 19.12%. The pooled unknown-type
share is 25 of 136, or 18.38%. In `sset_21`, 24 of 34 target rows, or 70.59%,
have both an unknown first-stroke type and `flaw=1`.

Across all 292 committed rallies, 54 contain a non-null flaw on at least one
raw row. They comprise 8 serve-matched rows, 26 target rows, and 20 whole-rally
misses. That 54-rally population matters only when comparing with ShuttleSet's
official whole-rally cleaning policy. It is not the denominator for the raw
issue 28 sensitivity below.

Every first row in this target has `ball_round=1`. All target first rows have
`server=1` in the raw set files. Those fields show ShuttleSet's intended rally
position. They do not prove that the service contact is visible in the video.
The `flaw` flag also does not prove broadcast omission. Official ShuttleSet
preprocessing drops a whole rally when any row has a non-null flaw, but the
dataset does not document a broadcast-specific meaning. Visual adjudication is
still required.

### Important reproduction trap

Select and require exactly one `ball_round == 1` row. Do not rely on the minimum
frame because two target rallies tie their first and second strokes. Do not use
pandas `GroupBy.first()`. That method takes the first non-null value
independently in each column and can assemble values from different strokes
when a service row contains null coordinates.

## Sensitivity of the issue 28 conclusion

Treating all 26 flaw-marked target rows as unusable would reduce the target
count from 136 to 110. The tested prototype would still recover zero targets.
Its target recovery would remain zero, and removing targets cannot convert one
of its 14 selected triggers into a true positive.

This is a sensitivity bound, not a corrected metric. Some flagged first rows
may show visible services. Some unflagged rows may have broadcast-omitted
starts. The human audit must supply the corrected denominator.

## Location relative to broadcast transitions

All 136 target first-stroke frames fall inside the current human `live` or
`live-non-standard` classes. None falls in `replay`, `cutaway`, or `other`.
That result does not prove visibility of the physical serve. A broadcast can
return after the serve, and ShuttleSet can place an uncertain first row in the
new live interval.

Counts close to a transition from `replay`, `cutaway`, or `other` into a live
class are:

| Video | Within 1 second | Within 3 seconds | Within 5 seconds | Within 10 seconds |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 25 | 51 | 56 | 59 |
| `sset_15` | 13 | 35 | 37 | 39 |
| `sset_21` | 9 | 17 | 20 | 26 |

Proximity alone is not evidence of omission. A normal broadcast also returns
to live setup shortly before a visible serve. These counts define efficient
review windows.

The 24 unknown and flaw-marked `sset_21` target frames are:

```text
21146, 21990, 23168, 24875, 25486, 27458, 31431, 31893,
40231, 40970, 44598, 49809, 50529, 63766, 64600, 66268,
68019, 69747, 71006, 72678, 73275, 86535, 87717, 92132
```

Three examples occur almost immediately after reviewed boundaries:

- 25486 is 11 frames after live starts at 25475 following replay.
- 64600 is 10 frames after live starts at 64590 following replay.
- 92132 is 12 frames after live starts at 92120 following cutaway.

These examples are priorities for visual review. They are not pre-adjudicated
as omitted serves.

## Replay-sting adjacency finding

For each canonical human `replay` interval, the adjacent human classes were
counted. `Other` is the class used for broadcast stings and transitions.

| Video | Replay intervals | `other` on both sides | `other` on either side |
| --- | ---: | ---: | ---: |
| `sset_01` | 62 | 57 (91.94%) | 61 (98.39%) |
| `sset_15` | 72 | 0 (0.00%) | 8 (11.11%) |
| `sset_21` | 45 | 0 (0.00%) | 9 (20.00%) |

In `sset_15`, 56 of 72 replay intervals are adjacent to `cutaway` on both
sides. In `sset_21`, replay commonly transitions from `cutaway` into `live`,
`live-non-standard`, or another `cutaway`.

The difference may reflect match-specific broadcast packages. It may also
reflect annotation granularity because `sset_15` and `sset_21` used
PySceneDetect scene boundaries as scaffolding. The current rows do not show
whether a brief sting was merged into an adjacent cutaway. Source-video review
is required.

## Existing replay-mask context

The production replay/off-rally mask unions court absence, perspective shift,
and shuttle velocity-drop signals. It has no repeated-sting matcher. The
existing `sset_01` audit measured 0.981 precision and 0.972 recall for the
duration-filtered mask. Court absence supplied all flagged frames in that
measurement. Perspective shift and velocity drop supplied none.

The high `sset_01` result shows that much replay footage is already excluded.
A sting signal must be measured as incremental evidence. Its likely value is
precise replay bracketing when replay footage still resembles the live court,
plus a reliable boundary for partial-rally entry after the broadcast returns.

## Required new truth

The complete timeline classes should remain unchanged in shape. Two separate
event-level tables are needed.

The rally-start table must record:

- whether the physical serve contact is visible, omitted, or uncertain;
- the visible serve frame when present;
- the first visible rally frame when the serve is omitted;
- the broadcast return frame and preceding class; and
- a note and confidence.

The replay-sting table must record:

- exact entry- and exit-sting bounds when present;
- whether both stings use the same visual template;
- the enclosed replay bounds;
- the state after the replay: setup, active rally, cutaway, or other; and
- a note and confidence.

The current evidence does not authorise a production replay-sting detector or
partial-rally rule.
