# Gap-bounded paint test check, 26 September 2026

This check tests one change to the scoring stage's paint test. The test now
measures each painted line on the full-size frame, and none of its samples
reaches past halfway to the next parallel line. The aim is to stop a court that
slips one painted line at the far end from scoring as well as the right court.
Everything else matches `../check_20260926_upright/` (commit `c5a7cfdb`),
including the upright-camera filter.

**Result: reverted.** The test fails the keep rule. It fixes neither gxBQ
slip, and it makes `gxBQ_window_00_frame_0` 0.36 m worse. It does fix
`am3_window_02_frame_17174`, from 0.55 to 0.21 m, and improves
`am3_window_00_frame_0`. It is 12% slower. On the gxBQ views the far lines are
too faint for any one-sample-at-a-time paint test: it removes the slipped
court's false far-end paint, but also the right court's real paint.

## Why

The net choice picks the eligible court with the best paint score plus a small
net-post bonus. The paint score cannot tell a far-end slip from the right
court:

- On `gxBQ_window_00_frame_689` and `gxBQ_window_03_frame_77876`, the winner
  sits about one painted line in at the far end. It wins by 0.0005 and 0.0066
- On `am3_window_02_frame_17174`, 189 eligible courts are closer to the hand
  marks than the winner

A predicted line collects paint from the neighbouring painted line. Today's
test runs on the half-size working image. For each sample it tries five centre
positions, up to 4 px either side of the predicted line, and compares each with
two side samples 6 px further out. A sample passes if its best centre is at
least 10 grey levels brighter than both sides. At the far end, the 4 cm stripes
are under a pixel wide and neighbouring lines only a few pixels apart. So a line
predicted on bare floor still reaches its neighbour's paint.

## The change

W5 scores each line as the mean, over its samples, of fragment support times
paint pass. Fragment support says whether a detected line fragment backs that
spot. Only the paint pass changes. The samples, their fragment support, the
person masks and the way lines combine into a court score all stay as they are.

- **Full-size frame.** The paint pass is measured on the native frame,
  converted to grey once per view
- **Gap bound.** From the candidate's own court, each sample gets the image
  distance to the nearest parallel painted line. The five centre positions
  reach at most a quarter of that distance, and no side sample sits more than
  half of it from the predicted line. Near the camera, where lines are far
  apart, this is today's test at full resolution
- **Pass bar.** The best centre must be brighter than both sides by at least a
  set number of grey levels. That number comes from a bare-floor measurement,
  described below, because a best-of-five brightness test passes on random
  texture far more often than half the time
- **Switch.** On by default (`Switches.gap_bounded_paint`).
  `run_views.py --fixed-paint-test` turns it off. Research scripts that call W5
  directly keep the old test

### Setting the pass bar

On each view with landmark hand marks, a court fitted to the hand marks gives
each painted line's true place. The new test runs along every true line, and
along the floor midway between each pair of neighbouring parallel lines. The
pass bar is the value that maximises the line pass rate minus the floor pass
rate, over all lines together. This rule was set before the measurement.

27 frames have landmark hand marks: the 10 in the 28 test views and 17 more
from the same videos. `pass_bar.py` writes `pass_bar.txt`. The bar comes out
at 9 grey levels, against today's 10. The difference barely moves between 7 and
10:

| Bar (grey levels) | Lines pass | Two far lines pass | Floor passes |
| ---: | ---: | ---: | ---: |
| 0 | 93% | 84% | 64% |
| 3 | 85% | 62% | 29% |
| 9 | 77% | 41% | 12% |

A bar of 0, pure polarity, cannot tell paint from floor: 64% of floor samples
pass. So any gain has to come from the native frame and the gap bound.

Before the run, W5's own code re-scored the 78 eligible courts saved for
`gxBQ_window_03_frame_77876` with the new test. The old test reproduced every
saved score exactly. The new one picks another slipped court, 1.49 m out before
the refit. With the new test, the right court's far baseline and far
long-service line score 0, the same as the slipped court's. So on this view the
far end no longer separates them.

## What was tried first

An offline re-score of the saved candidates tried a looser version. It sampled
evenly in floor metres, passed any sample brighter than both sides by any
amount, and dropped fragment support. It fixed both gxBQ slips. But on
`am2_window_01_frame_28019` it picked a court 0.5-0.9 m out along the whole
left side, because its left doubles line sat on the real left singles stripe
and scored full paint.

A red-team review (Codex GPT-6 Sol) found three problems:
- The trial changed where samples fall, not only the paint pass
- Its "chance scores zero" rescaling was wrong: random noise scored 0.45-0.49
  per line
- Its side samples could reach three quarters of the way to the next line, not
  half

It found the gxBQ_689 gain sensitive to sample placement. It recommended
exactly this narrower change: keep W5's samples and per-sample fragment support,
and replace only the paint pass with one that has a measured null response and a
literal half-gap bound. It also warned that 10 hand-marked views, several from
the same scenes, cannot show a general gain.

## How it will be judged

Set before the run. The measure is the largest hand-mark error in floor metres
after the final refit, on the 10 views with landmark hand marks, against
`../check_20260926_upright/`.

- **Keep** if `gxBQ_window_00_frame_689` and `gxBQ_window_03_frame_77876`
  come back to about their 25 September errors (0.32 and 0.35 m), and no other
  view gets more than 0.2 m worse
- **Otherwise revert** the build commit

A control view that gains a court is reported but does not on its own force a
revert. Renders of every view whose court changes go with the results. The time
cost comes from each view's `detect_seconds` against the upright check's
correctness run.

## What ran

Commit `2e30d4d9`, one run on Carmack with `run_paint_test.sh`: artefacts
and self-checks on, in `../check_20260925/correctness/groups.txt`'s 8 groups,
at most 8 processes. All 8 groups exited 0. `compare_paint_test.py` writes
`compare_paint_test.txt`.

## Results

The largest hand-mark error in floor metres, before and after the final
refit:

| View | Upright check | Gap-bounded test |
| --- | --- | --- |
| `am1_window_00_frame_54` | 0.31 / 0.30 | 0.41 / 0.30 |
| `am2_window_00_frame_150` | 0.32 / 0.24 | 0.27 / 0.23 |
| `am2_window_01_frame_28019` | 0.69 / 0.62 | 0.71 / 0.64 |
| `am3_window_00_frame_0` | 0.26 / 0.24 | 0.18 / 0.14 |
| `am3_window_02_frame_17174` | 0.59 / 0.55 | 0.25 / 0.21 |
| `gxBQ_window_00_frame_0` | 0.34 / 0.19 | 0.51 / 0.55 |
| `gxBQ_window_00_frame_5` | 0.23 / 0.14 | 0.23 / 0.14 |
| `gxBQ_window_00_frame_689` | 1.18 / 0.94 | 1.18 / 0.94 |
| `gxBQ_window_03_frame_77876` | 1.40 / 1.06 | 1.49 / 0.93 |
| `letterboxed_short_frame_78` | 0.18 / 0.12 | 0.18 / 0.12 |

The chosen court changes on 19 of the 28 views, mostly by under 4 native px.
The largest moves are `gxBQ_window_03_frame_77876` (8.1 px),
`gxBQ_window_00_frame_0` (7.1 px), `sset_21_gloiZ_gTJaE_frame_00000001`
(6.6 px) and `am3_window_02_frame_17174` (4.2 px). One control view,
`…00100347`, now gets no court instead of a wrong one: its new pick fails the
final fit (`rank_deficient`). The other 7 control views are unchanged.

The 28 views took 3,829 detect seconds against the upright check's 3,413, both
with self-checks on: 12% slower.

### Why it did not fix the gxBQ slips

The far lines on the gxBQ views are too faint to pass one sample at a time.
`per_view_floor.py` runs the test along the lines of a court fitted to the hand
marks. On the 7 gxBQ frames, only 5-13% of far-line samples pass at the bar of
9, while bare floor passes about 10%. So the new test takes away the slipped
court's false far-end paint, but the right court scores no better. On the am3
views, where 33-47% of far-line samples pass, it picks the better court.

A second red-team review (Codex GPT-6 Sol) checked the build:
- **Geometry bug.** The gap to the next line is measured to the matching point
  on that line, not along the sample's normal. Under perspective these differ,
  so 23.5% of side samples on `gxBQ_window_03_frame_77876`'s saved courts pass
  the true halfway point, by up to 1.39 times. They still stop short of the
  next stripe, and a brighter side only lowers the contrast. So the bug cannot
  lend a line its neighbour's paint. On that view's far lines the error is
  about 2%. A future version should intersect the normal with the projected
  neighbouring line
- **Not a sampling miss.** At the far end the five centre positions sit about
  0.6 px apart, so they do not step over the stripe. Even on the court fitted
  to the hand marks, only 3 and 6 of 64 far-line samples pass on
  `gxBQ_window_03_frame_77876`. The saved candidates sit about 1.6 px off those
  lines, which makes it worse
- **Lowering the bar does not rescue it.** At a bar of 3, the right court's far
  lines still score almost nothing, and the slipped court scores slightly more
- **Pass-bar caveat.** About 4% of the floor samples cross a perpendicular
  stripe. Without them the bar is still 9

### The pass bar

The old bar of 10 grey levels has no recorded basis; the W5 notes call it "the
historical contrast probe of 10". The measurement cannot tell 7 to 10 apart:
line-minus-floor is 0.642-0.644 across that range.

A per-view bar, set from each frame's own floor, was checked before building
(`per_view_floor.py`). Floors do differ: the contrast bare floor reaches 10% of
the time runs from 2.5 grey levels (letterboxed) to 29 (am3's wood floor). But
a per-view bar does slightly worse overall, 0.631 against 0.649. It raises the
bar on noisy floors and wipes out their far lines: on
`am3_window_02_frame_17174`, far-line passes fall from 34% to 1%. It does not
help the gxBQ views.

### What might work instead

Neither of these has been built or tested:
- **Average along the line instead of passing samples one at a time.** Score
  each line by its mean contrast along its length, minus what bare floor gives
  on the same measure. A faint line that is slightly brighter all along could
  show up where single samples do not. Check it first on the hand-marked
  lines against the midway floor, with means in place of pass rates
- **Fresh hand-marked views.** Sol recommends designing any new paint rule on
  views these checks have not used

## Files

- `paint_test/results/`, `paint_test/logs/`: per-view results and logs. Remote
  paths in the logs are replaced by `<run root>` and `<ShuttleSet root>`. The
  28 artefact files (189 MB) are in the run folder
  `court_detector_20260926_paint_test/` in the court-detector run root on
  Carmack
- `compare_paint_test.py`, `pass_bar.py`, `per_view_floor.py` and their `.txt`
  outputs. The last two use the reverted paint test, so run them at commit
  `55f7615e`
