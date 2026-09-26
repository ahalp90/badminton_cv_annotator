# Line-averaged paint check, 26 September 2026

This check tests one change to the court detector's final choice. Each
painted line now passes or fails its paint test on its average contrast along
its whole length, instead of sample by sample. The aim is to let the right
court's faint far lines count, so that a court slipped by one line at the far
end stops scoring as well as the right one.

**Status: third version under test.** Two versions failed the keep rule
below, each for a cause found afterwards:

- **v1** (commit `466114d3`, outputs in `v1/`) read each line only at its
  predicted place. Courts before the refit sit a few centimetres off the
  paint, so near-right courts failed their own lines. It made
  `gxBQ_window_00_frame_689` 0.63 m worse
- **v2** (outputs in `v2/`) added a search across the line, as W5's own test
  has. It fixed `gxBQ_window_03_frame_77876` (1.06 to 0.35 m) but made
  `am1_window_00_frame_54` 0.30 m worse. There the right court's near
  long-service line had no usable row, and v2 counted it as unpainted. It
  also added about 18 s a view on the laptop
- **v3** leaves such a line out, as W5 does, and reads a row every 4 cm
  instead of every 1 cm

The rest of this README describes v1's design; the version notes above give
the changes.

## Why

The net choice picks the eligible court with the best score: 90% of the
scoring stage's paint score, 10% of its geometry score, and a small bonus for
supported net posts (`../check_20260926_court_choice/README.md`).

The paint score passes or fails each sample on its own. A sample passes when
the grey level at the line is at least 10 grey levels brighter than on both
sides. On the gxBQ views only 5-13% of the true far lines' samples pass, about
as often as bare floor does (`../check_20260926_paint_test/README.md`). So the
right court's far lines score no better than a slipped court's, and
`gxBQ_window_03_frame_77876` keeps its far-end slip, 1.06 m out.

Averaging first should help. Noise in single samples cancels out along a
line's length, so a line only a few grey levels brighter than the floor can
still show.

## The change

`../line_paint.py` holds the test. `Switches.line_paint` turns it on (off by
default), as does `run_views.py --line-paint`. It acts only in the net choice,
on the courts that are already eligible.

- **Samples.** One sample every 1 cm of floor along each painted line, on the
  full-size frame. Each sample reads the grey level at the line and at a point
  on the floor either side of it. A sample is skipped if any of its three
  points is outside the frame, behind the camera or inside a person box
- **Side points.** Each sits 0.4 times the gap to the nearest parallel painted
  line away, at most 0.30 m. That is 0.18 m for the four sidelines, which sit
  0.46 m apart, and 0.30 m for every other line. So a side point never reaches
  the neighbouring line
- **Line contrast.** The line's average grey level minus the brighter side's
  average. A line must be brighter than both sides
- **Pass.** A line passes when its contrast is at least 1 grey level (set
  below)
- **Court score.** The scoring stage's paint score, with each line's pass
  decided this way. A line that passes counts its fragment support, the share
  of it that detected line fragments back; a line that fails counts 0. Lines
  are weighted by their length in the image within each direction, lengthwise
  and crosswise, and the weaker direction counts. A line the scoring stage
  sees but this test cannot measure counts as unpainted. The scoring stage
  instead leaves such a line out; this only happens on slivers at the frame's
  edge
- **Unchanged.** The 10% geometry share, the net-post bonus, the eligible
  courts and the stripe refit

### Setting the pass bar

On each view with landmark hand marks, a court fitted to the hand marks gives
each painted line's true place. The test runs along every true line, and
along the floor midway between each pair of neighbouring parallel lines. The
pass bar is the value, in steps of 0.5 grey levels, that maximises the true
lines' pass rate minus the floor lines' pass rate over all views together.
This rule was set before the measurement.

27 frames have landmark hand marks: the 10 in the 28 test views and 17 more
from the same videos. `line_bar.py` writes `line_bar.txt`. The bar comes out
at 1 grey level:

| Bar (grey levels) | True lines pass | Floor lines pass |
| ---: | ---: | ---: |
| 0 | 95% | 12% |
| 1 | 93% | 5% |
| 2 | 89% | 2% |
| 6 | 80% | 0% |

Anything from 1 to 2 grey levels does about equally well. For comparison, the
per-sample test at its best bar passed 77% of true-line samples and 12% of
floor samples.

On the gxBQ views the far lines' contrast is 0.8 to 2.7 grey levels, and the
floor lines' at most 1.8. So at a bar of 1, 12 of the 14 gxBQ far lines pass.
The two that fail are the far baselines of `gxBQ_window_00_frame_0` (0.9) and
`gxBQ_window_00_frame_5` (0.8). On the three `yellow_short` frames the far
lines stay invisible even averaged (-0.9 to 0.7); they sit 2.4 native px
apart.

`line_profiles.py` writes `line_profiles.txt`: the grey level across each
line, averaged along it. On the gxBQ views the far lines show a broad, low
hump of 4-6 grey levels about 0.4 m wide. Other lines show sharp ridges of
14-53 grey levels.

The bar is set on the same frames the keep rule judges, and on frames from
the same videos. A red-team review of the earlier paint test warned that such
views cannot show a general gain.

## What will run

A local replay, as in `../check_20260926_court_choice/`. The change acts only
after the scoring stage, so the upright check's saved results hold everything
the final choice needs. `replay_line_paint.py` rebuilds each view's scoring
context and calls the detector's own final-choice code once per arm, with
self-checks on:

| Arm | Net choice |
| --- | --- |
| `paint_only` | The scoring stage's paint score, as the upright check ran. Must reproduce the saved court |
| `blend` | Today's default: 10% geometry |
| `line_paint` | 10% geometry, with line-averaged paint |

Two control views cannot be replayed on the laptop, because their fits have
no unique answer (`../check_20260926_court_choice/README.md`). The one
Carmack run of the final default covers them.

## How it will be judged

Set before the run, as in the court-choice check, with `line_paint` against
`blend`. The measure is the largest hand-mark error in floor metres, after the
final refit, on the 10 views with landmark hand marks.

- **Keep** if at least one view is more than 0.2 m better and none is more
  than 0.2 m worse. A clearly wrong court on an unmarked view counts as worse
- **Otherwise** leave the switch off

A control view that gains a court is reported but does not on its own fail
the change. `compare_line_paint.py` also reports laptop seconds; the Carmack
run gives the real cost, which must stay under 10% of detect time.

## Files

- `line_profiles.py`, `line_bar.py` and their `.txt` outputs: the
  measurements on the hand-marked views
- `replay_line_paint.py`: the replay
- `compare_line_paint.py`: the comparison
