> The original scripts are [archived](../../archive/20260927_code/court_detector/check_20260926_line_paint/). Commands below use their historical paths.

> Dated experiment record. Old plans and approval requests below describe that run.
> Use [pickup](../../pickup.md) for current work and
> [the decisions](../../DETECTOR_DECISIONS.md#d19) for the later rulings.

# Line-averaged paint check, 26 September 2026

This check tests one change to the court detector's final choice. Each
painted line now passes or fails its paint test on its average contrast along
its whole length, instead of sample by sample. The aim is to let the right
court's faint far lines count, so that a court slipped by one line at the far
end stops scoring as well as the right one.

**Result: rejected. The detector keeps the 10% geometry blend.** Four
versions ran on the 10 views with landmark hand marks, and none held up. The
line-paint switch was taken out of the detector afterwards. Commit `2d69273a`
holds the last version's code (v4); the scripts here need it to rerun.

| Version | What changed | Outcome, after the refit |
| --- | --- | --- |
| v1 (`466114d3`, `v1/`) | Reads each line only at its predicted place; pass bar 1 grey level | `gxBQ_window_00_frame_689` 0.63 m worse |
| v2 (`v2/`) | Searches across the line, as W5's own test does; pass bar 4 | Fixed `gxBQ_window_03_frame_77876` (1.06 to 0.35 m), but `am1_window_00_frame_54` 0.30 m worse. About 18 s a view on the laptop |
| v3 (`5cb31733`, `v3/`) | Leaves out lines with no usable row, as W5 does; a row every 4 cm | Passed the keep rule's numbers, but slipped `am2_window_01_frame_28019` by a whole line at the near end. The user judged the am1 and am2 renders unusable |
| v4 (`2d69273a`, `v4/`) | No pass bar: each line counts its contrast as a share of the view's strongest line | Four views more than 0.2 m worse, none better. Total largest error 3.79 to 6.00 m |

## Why it failed

- **A fixed pass bar counts anything a little brighter than its sides as
  paint.** On `am2_window_01_frame_28019`, v3's slipped court put its
  baseline on the edge of the blue mat against the wooden floor. That edge
  reads 16 grey levels and passed, just as the real baseline's 93 did. The
  bar was set against bare floor midway between lines. Sol's review of v3
  had warned that a slipped court's lines land on other things: other paint,
  junctions and, as it turned out, edges
- **Pass or fail throws away how strong the paint is.** On
  `am1_window_00_frame_54` both courts passed the same seven lines, though
  the right court read stronger on several (28 against 15 grey levels on the
  right doubles sideline). The choice then fell to fragment support, 0.501
  to 0.497, and picked the wrong court
- **Raw strength favours near lines.** A view's strongest lines read 50-97
  grey levels, and far lines under 10 even where the court is right. So under
  v4 a court can give up faint real lines for slightly stronger readings on
  bright ones. On `am3_window_00_frame_0`, v4's pick reads its right sidelines
  at 34 and 29 grey levels against the blend court's 23 and 18. It reads its
  short-service lines at -0.2 and 4.5 against 13.7 and 20.0. It lands 0.56 m
  out against the blend's 0.24 m (`v4/line_passes.txt`)

The underlying problem: a line's contrast in grey levels depends on its
distance, the lighting and its width in pixels. So the numbers do not compare
across lines, and neither a fixed bar nor a share of the strongest line fixed
that on these views.

The keep rule also missed v3's slip on am2. It uses each view's largest error,
and the blend court already had one bad far mark at 0.62 m, so the slip showed
as only 0.14 m worse (0.62 to 0.76 m). The median error shows it: 0.07 to
0.37 m. `compare_line_paint.py` now reports the median too.

The one Carmack run, started before v3 was rejected, had two arms: the blend
default and v3. Both match the laptop replay on the 27 views it can replay, so
v3 passes the keep rule's numbers on Carmack too. Its time cannot be told apart
from Carmack's run-to-run noise
([Carmack run](../check_20260926_court_choice/README.md#carmack-run)).

The rest of this README is the plan as written before the first run, with v1's
design; the table above gives each later change.

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

The scripts need the line-paint code, which the detector no longer has. Run
them at commit `2d69273a` (v4), or at a version's own commit for its results.

- `line_profiles.py`, `line_bar.py` and their `.txt` outputs: the
  measurements on the hand-marked views. `line_bar.txt` is v1's bar; each
  version folder has its own
- `replay_line_paint.py`: the replay. Each version folder's `replay/` holds its
  output (v3 and v4)
- `compare_line_paint.py`: the comparison; `compare_line_paint.txt` in each
  version folder
- `line_passes.py`: each line's contrast and fragment support for three courts
  on a view; `line_passes.txt` in the version folders
- `run_carmack.sh`: the two-arm Carmack run, started at commit `cd3011b7`
- `compare_carmack.py`: compares that run's two arms, once it is copied back
