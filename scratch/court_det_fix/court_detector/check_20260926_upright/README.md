> The original scripts are [archived](../../archive/20260927_code/court_detector/check_20260926_upright/). Commands below use their historical paths.

> Dated experiment record. Old plans and approval requests below describe that run.
> Use [pickup](../../pickup.md) for current work and
> [the decisions](../../DETECTOR_DECISIONS.md#d19) for the later rulings.

# Upright-camera filter check, 26 September 2026

The upright-camera filter cuts the joined detector's time by 42% on the 28
test views, from 6,434 s to 3,703 s. It is not accuracy-neutral. 17 of the 20
court views keep the same court. Of the three that change, one gets much
closer to its hand marks and two get about one painted line worse at the far
end. The filter removes only impossible courts. The two worse views come from
a weakness in the scoring stage, which the filter exposes. The filter does not
fix the control views' false courts.

## What the filter does

Each direction pair implies a horizon: the line through its two vanishing
points. The filter skips a pair whose horizon tilts more than 45 degrees,
because every court it builds needs a camera turned on its side. It also drops
a court that sits above its pair's horizon, because that needs an upside-down
camera. A horizon more than 10 image diagonals away, as for a camera looking
nearly straight down, passes both tests. The chosen courts of the 20 court
views imply camera rolls within 2.5 degrees.

It is on by default (`Switches.upright_camera`). `run_views.py
--any-camera-roll` turns it off.

## What ran

Commit `c5a7cfdb`, on Carmack, with at most 8 processes and one numerical
thread each. `run_upright.sh` gives the arguments.

1. **Filter on, artefacts and self-checks on**, in the 8 groups of
   `../check_20260925/correctness/groups.txt` (`upright/`). All 8 groups
   exited 0
2. **Timing**, filter on, self-checks off, one process per view
   (`timing_upright/`). All 28 views exited 0

The comparison is with `../check_20260925/`: its correctness results and its
`timing_no_checks` run. With the filter off, the code on the timed path is the
same as in that run. `compare_upright.py` writes `compare_upright.txt`.

## Results

| Step | 25 September (s) | Filter on (s) | Saved |
| --- | ---: | ---: | ---: |
| Whole detector | 6,434 | 3,703 | 42% |
| Court search, all fragments (G0) | 3,346 | 951 | 72% |
| Court search, paint fragments (G1) | 785 | 284 | 64% |
| Scoring (W5) | 1,837 | 1,968 | -7% |
| Line templates | 409 | 439 | -7% |

Scoring and line templates do the same work as before, so their 7% rise is
Carmack running slower that day. The saving is probably slightly larger than
42%. Per view, the 20 court views saved 45% and the 8 control views 1%. The
slowest view before, `gxBQ_window_00_frame_689`, went from 546 s to 161 s.
The six `shuttleset_03` broadcast views are now the slowest, at 202-264 s.
They saved 26-49%.

### Accuracy

17 of the 20 court views keep the same court. Three hard amateur views pick a
court 8-14 native px away. The table gives their hand-mark errors in floor
metres (median / largest). Each hand mark is mapped back onto the floor
through the court, and the error is how far it lands from where it should be.

| View | 25 September | Filter on |
| --- | --- | --- |
| `gxBQ_window_00_frame_0` | 0.11 / 1.00 | 0.05 / 0.19 |
| `gxBQ_window_00_frame_689` | 0.22 / 0.32 | 0.23 / 0.94 |
| `gxBQ_window_03_frame_77876` | 0.11 / 0.35 | 0.21 / 1.06 |

The gap between a baseline and the line inside it is 0.76 m. So the two worse
views slip about one line at the far end, and the renders show the same.
The other 7 views with landmark hand marks are unchanged, at 0.30 m or less.
The shuttleset views have only corner hand marks, so this measure does not
cover them.

Pixel errors hide these slips. At the far end, neighbouring lines can be only
a few pixels apart. On `am3_window_02_frame_17174`, 1 px up or down at the far
corner is 0.14 m on the floor. The pixel measure
(`reference_errors.txt` in the evidence folder below) had `gxBQ_window_00_frame_689` getting better.
Hand-mark clicks are magnified the same way. At the far end, one native pixel
of click error moves a mark by up to about 0.2 m on the floor, so errors up to
about 0.3 m there can be click error.

### Why two views got worse

The filter does not remove the old winners. On both views, the old winner is
still a candidate with an identical score. The new winner was already built
on 25 September, in its direction pair's own shortlist. But courts from
steep-horizon pairs, which need a sideways camera, held 253-256 of the 256
places in G0's overall list, so it never reached scoring. With those gone it
reaches scoring, and wins by 0.0005
(`gxBQ_window_00_frame_689`) and 0.0066 (`gxBQ_window_03_frame_77876`). The
same crowding out explains the one improvement: the better court on
`gxBQ_window_00_frame_0` was also shut out on 25 September.

The scoring stage cannot reliably tell a court with a far-end slip from the
right one. `am3_window_02_frame_17174` shows this without the filter. Before the
final refit its winner is 0.59 m out (0.55 m after). 189 other candidates that
pass the full-court checks do better. The best,
at 0.18 m, loses on paint score (0.559 against 0.581) and on the net bonus
(0.02 against 0.04). Fixing the scoring is separate work.

The evidence is in
`../../court_detector_optimisation_handover/claude_evidence/upright_camera/`:
`floor_errors.txt` for the table and where each new winner sat on 25
September, `saved_outputs.txt` for the places steep pairs held, and
`candidates_am3_window_02_frame_17174.txt` for am3.

### Control views

6 of 8 still get no court. `…00100347` still gets a wrong
court, a different one 973 px away. `…00014336` now gets a wrong court. Before,
it got none only because its final fit failed (`rank_deficient`). The filter
removes the old picks, which needed a sideways or upside-down camera, but the
next wrong court takes their place.

## Files

- `upright/results/`, `upright/logs/`, `timing_upright/results/`,
  `timing_upright/logs/`: per-view results and logs. Remote paths in the logs
  are replaced by `<run root>`
- `left_on_carmack.tsv`: the 28 artefact files (189 MB), left in the run
  folder `court_detector_20260926_upright/` in the court-detector run root
