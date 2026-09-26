# Gap-bounded paint test check, 26 September 2026

This check tests one change to the scoring stage's paint test. The test now
measures each painted line on the full-size frame, and none of its samples
reaches past halfway to the next parallel line. The aim is to stop a court that
slips one painted line at the far end from scoring as well as the right court.
Everything else matches `../check_20260926_upright/` (commit `c5a7cfdb`),
including the upright-camera filter.

Status: planned. Results follow the run.

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

One run on Carmack, as step 1 of `../check_20260926_upright/run_upright.sh`:
artefacts and self-checks on, in `../check_20260925/correctness/groups.txt`'s 8
groups, at most 8 processes.
