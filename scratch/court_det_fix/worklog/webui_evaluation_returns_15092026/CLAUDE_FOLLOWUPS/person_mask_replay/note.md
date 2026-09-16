# Masking fragments inside person boxes on GX0, replayed

## Gate

Passed. With no masking, the replay reproduces the saved estimator exactly: the same 16 retained candidate IDs, the same working points and the same support masks as `vp_pruning_20260914/coverage/results/gxBQ_window_00_frame_0.json.gz`.

## Table

There are 724 fragments feeding the merge before any masking.

| rule | dropped | merged lines | selected directions | min angle to x-control (deg) | min angle to y-control (deg) | set_bound (working px) | best control fit (working px) | dropped fragments within 6px of a court edge |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 106 | 16 | 0.177489 | 1.794981 | 6.022569 | 6.798367 | n/a |
| A_both_endpoints | 55 | 105 | 16 | 0.056371 | 0.618692 | 6.022569 | 7.070301 | 1 |
| B_midpoint | 66 | 105 | 16 | 0.056371 | 0.000000 | 1.218743 | 1.498316 | 1 |

## Which offending structures each rule removes

### A_both_endpoints

- Row 31: survives. 4 of 4 member fragment(s) remain, now merged as line 31 (4 member(s), 153.6 px covered).
- Row 51: survives. 1 of 1 member fragment(s) remain, now merged as line 51 (1 member(s), 101.8 px covered).
- Row 61: removed. All 1 member fragment(s) ([499]) fell inside a person box and were dropped.

### B_midpoint

- Row 31: survives. 4 of 4 member fragment(s) remain, now merged as line 31 (4 member(s), 153.6 px covered).
- Row 51: survives. 1 of 1 member fragment(s) remain, now merged as line 51 (1 member(s), 101.8 px covered).
- Row 61: removed. All 1 member fragment(s) ([499]) fell inside a person box and were dropped.

## Does masking person boxes alone move GX0's selection toward the approved court

Both rules remove row 61 (the spectator's face); neither touches row 31 or row 51, which sit outside every person box on this frame. Rule A barely changes the outcome: set_bound stays at 6.023 working px (baseline 6.023) and the best control fit gets slightly worse, 7.070 against 6.798. Rule B, which drops more fragments, changes the coverage allocation enough that one of the 16 selected directions lands on almost exactly the same direction as the approved control's y-axis candidate (0.000 degrees away, against 1.795 at baseline), and both distance measures fall a long way: set_bound to 1.219 working px and the best control fit to 1.498 working px. That improvement comes from which candidate the greedy allocation happens to pick as a leader once the fragment mix changes, not from removing row 61 specifically, since row 61's own direction was never close to the control axes.

