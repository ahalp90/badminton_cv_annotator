# Player-size filter check, 26 September 2026

The player-size filter failed its keep rule and was taken out of the detector.
It does what it was built to do: it removes courts at an absurd scale, such as
courts built from the ceiling. But removing them let a worse court reach the
final choice on `am2_window_01_frame_28019`, which then slipped at the near
end. No view improved, and the run was not clearly faster.

## What the filter did

A court's homography maps floor metres to image pixels. Mapping a person's
box through it gives their width on the floor: the distance between the box's
bottom corners, each mapped onto the floor. A court at the wrong scale makes
the people giants or dwarfs. The filter skipped any court whose people came
out narrower than 0.2 m or wider than 3 m, as a median over the people it puts
on or within 1 m of the court. Courts with no one near stayed. It ran inside
each direction pair's court search, before full scoring, behind
`Switches.player_size` (off by default) and `run_views.py --player-size`.

A first version divided the box's pixel width by the court's largest floor
stretch at the feet. That stretch can point up the image, so it measured some
courts wrongly. Commit `4ddb96b0` fixed it before this run.

## What ran

Commit `6c787ea3`, on Carmack, with at most 8 processes. `run_carmack.sh` ran
two arms, the default detector and the default with the filter, in the 8
groups of `../check_20260925/correctness/groups.txt`. The arms alternated
group by group, so their timings shared one load. Artefacts and self-checks
were on. All 16 groups exited 0.

`compare_player_size.py` compares the arms and writes
`compare_player_size.txt`. The default arm reproduced the final 26 September
run exactly: the same pick and the same corners on all 28 views.

The keep rule, set before the run: keep the filter only if all hold.

- No hand-marked view's median or largest floor error worsens by more than
  0.1 m
- The user inspects renders of every view whose court changes
- No control view gains a court
- Some view improves, or the run gets faster

## Results

It fails on accuracy and on speed:

- **One view got worse.** `am2_window_01_frame_28019` picked a court 291
  native px away. Its floor error went from 0.07 / 0.62 m (median / largest)
  to 0.45 / 0.76 m. The render shows the court slipping at the near end
- **No view improved.** The other 9 views with landmark hand marks kept the
  same court. Three more views report a different chosen key but identical
  final corners, so their courts did not change
- **The controls were unchanged.** No control view gained or lost a court
- **The run was not clearly faster.** The filter fully scored 14% fewer
  courts, but the whole detector took 3,417 s against 3,591 s, 4.9% less.
  Per view, the two arms' times differ by up to about 70% in both directions,
  so 4.9% is within the noise of this run

### Why am2 got worse

The filter kept the right court: the default's winner is still in the filtered
arm's shortlist, with the same score. What it removed were junk courts: 134
of G0's 256 shortlist places and 68 of G1's. All but one put the people at
3-17 cm wide. The three best-scoring ones in G0 sit in the ceiling, built from
its beams and lights. With them gone, a court the default had built but
crowded out of G1's shortlist got in, at place 240. The final choice then
preferred it (0.400 against 0.378).

That court and the right one both put the players at about 1 m, so no width
test can separate them. The upright-camera filter exposed the same weakness
(`../check_20260926_upright/`): scoring cannot reliably tell a court that
slips one line at an end from the right one. So cleaner shortlists can move
results by chance, and here they moved one the wrong way.

## The search stage's score does not fix it

On am2 the search stage had it right. Each direction pair's search scores a
court when it builds its shortlist, and it scored the right court well above
the slipped one (0.326 against 0.267). The final choice then preferred the
slipped court, by 0.3377 against 0.3344 on paint and 0.56 against 0.37 on
fragment support.

So a screen added each court's search score to the final choice, on the 10
views with landmark hand marks (`search_score_screen.py`, output in
`search_score_screen.txt`). It uses the default arm's saved scores, with
floor errors before the final refit. Each search score is scaled by the best
one in the same search, since G0's scores run higher than G1's. Line-template
courts have no search score; the screen tried 0 and the view's lowest
shortlisted score for them. At weight 0 it reproduces today's choice on every
view.

It does not help:

- **The search score does not favour the best courts in general.** On
  `am3_window_02_frame_17174` it prefers the chosen, worse court (0.933
  against 0.875). On `gxBQ_window_00_frame_689` the best court ranks 100th
  by search score
- **Every weight that changes a pick makes the total largest error worse:**
  4.91 m today, 6.09 m at weight 0.05 and 6.77-8.22 m at 0.1 and above. At
  0.05, `gxBQ_window_00_frame_689` goes from 0.58 to 1.73 m

So the search score helped on am2 by chance.

## Combining the two directions differently does not fix it either

On am2 the slipped court beats the right one on both paint and fragment
support, and both get the same net bonus. So no blend of those two scores
can pick the right court. But the scoring stage measures each court in two
directions, lengthwise and crosswise, and the final choice keeps only the
weaker direction of each score.

A second screen tried two other ways to combine them, on the same saved
scores (`direction_screen.py`, output in `direction_screen.txt`). E1 blends
paint and fragment support within each direction, then takes the weaker
direction. E2 takes the harmonic mean of the two blended directions, so a
strong direction can make up for a weak one. Today's rule reproduces every
saved score and pick.

Neither holds up (floor error before the final refit, median / largest):

- **E1 breaks am2 in the default arm**, from 0.07 / 0.69 m to 0.66 / 0.99 m,
  and leaves the filtered arm's slip in place
- **E2 fixes the filtered arm's am2**, from 0.37 / 0.74 m to 0.07 / 0.69 m,
  and improves `am3_window_00_frame_0`, from 0.10 / 0.26 m to 0.07 / 0.18 m
- **But E2 slips a whole end on two gxBQ views:** `gxBQ_window_00_frame_0`
  from 0.34 to 0.97 m largest error, and `gxBQ_window_00_frame_689` from 0.58
  to 1.42 m. The error sits at one end of the court; the other end stays
  within 0.4 m. The final refit pulls lines onto the nearest stripe, so it
  cannot undo a slip onto the wrong line
- E2 also changes the court on six unmarked broadcast views, which nobody
  has looked at

E2's gain and its loss have the same cause. On `gxBQ_window_00_frame_0` the
slipped court reads far more paint in one direction than today's pick (0.555
against 0.424). Taking the weaker direction ignores that surplus; the
harmonic mean rewards it.

## Players' width on the 28 views

A rebuild of every court each direction pair builds, from the final 26
September run's inputs, measured the width each court implies
(`../../court_detector_optimisation_handover/claude_evidence/built_courts/summary.txt`,
with the upright-camera filter):

- The chosen courts put the players at 0.59-1.40 m
- 0.2-3 m would skip 13.6% of the 12,438,584 courts the pairs build. The
  Carmack run's filtered arm skipped 14.0%. The rebuild's count is a lower
  bound, read off a histogram
- The chosen courts and the net choice's top fives span 0.58-3.44 m. So at
  least one top-five court sits outside 0.2-3 m

## Files

The scripts need the filter, so run them at commit `6c787ea3`.

- `run_carmack.sh`: the two-arm run
- `compare_player_size.py` and its output, `compare_player_size.txt`
- `search_score_screen.py` and `direction_screen.py`, with their outputs in
  the matching `.txt` files. They read saved artefacts only, so they run at
  any commit
- `blend_default/` and `player_size/`: each arm's per-view results and group
  logs. Remote paths are replaced by `<run root>` and `<ShuttleSet root>`
- `left_on_carmack.tsv`: the 56 artefact files (399 MB), with sizes and MD5
  sums, left in the run folder `court_detector_20260926_player_size_fix/` in
  the court-detector run root
