# Court-choice check, 26 September 2026

This check tests two changes to how the detector picks its final court, alone
and together. Both aim at views where a court much closer to the hand marks is
already eligible but loses. Everything else matches
`../check_20260926_upright/` (commit `c5a7cfdb`), including the upright-camera
filter.

**Result: the geometry blend passes the keep rule; the top-15 refit fails,
alone and with the blend.** The blend fixes `gxBQ_window_00_frame_689`, from
0.94 to 0.32 m, and changes no other view. The refit fixes that view too but
makes two others about 0.7 m worse. After refitting, the rescore often ranks the
best refitted court well down. Keeping the blend, reverting the refit and one
Carmack run all wait for the project owner's approval.

## Why

The net choice picks the eligible court with the best paint score plus a small
net-post bonus. Only that court then gets the final stripe refit. On several
hand-marked views a much better court is eligible but ranks lower
(`good_court_ranks.py`, from the upright check's saved results):

| View | Chosen court's error | Best eligible | Where good courts rank |
| --- | ---: | ---: | --- |
| `gxBQ_window_00_frame_689` | 1.18 m | 0.43 m | 14th |
| `gxBQ_window_03_frame_77876` | 1.40 m | 0.41 m | 3rd |
| `am3_window_02_frame_17174` | 0.59 m | 0.17 m | 10th, 11th |
| `gxBQ_window_00_frame_0` | 0.34 m | 0.23 m | 5th |
| `am2_window_01_frame_28019` | 0.69 m | 0.46 m | 34th and lower |

Errors here are the largest hand-mark error in floor metres, before the final
refit. A good court is one within 0.1 m of the best eligible court.

The [paint-test check](../check_20260926_paint_test/README.md) showed that a
stricter paint test cannot fix this on the gxBQ views. Their far lines are too
faint to pass one sample at a time.

## The changes

W5, the scoring stage, gives each court two scores from the same samples along
its predicted lines:

- **Geometry score**: how much of each line detected line fragments back
- **Paint score**: the same, but a sample counts only if the spot is also at
  least 10 grey levels brighter than the floor either side

So the paint score is the geometry score with a paint check on top. Today the
final choice uses the paint score alone.

### Geometry blend

The final choice scores each court as 90% paint score plus 10% geometry score,
plus the net bonus. A fragment-backed spot that fails the paint test then
counts about 0.1 instead of 0. Switch: `Switches.geometry_weight`, and
`run_views.py --geometry-weight 0.1`.

A Codex red-team review (GPT-6 Sol) suggested this rule. Replaying the saved
choices at different weights (`geometry_weight_sweep.py`) found:
- 10% changes one pick: `gxBQ_window_00_frame_689` moves from a 1.18 m court
  to a 0.58 m one, before the refit
- 10% to 50% all make that fix. Past 10%, the other changes are 0.05-0.06 m,
  within hand-mark noise, and the number of changed picks grows from 1 to 10
- Above 50% results get worse. Geometry alone picks a 1.32 m court on
  `am2_window_00_frame_150`

So 10% makes the one clear fix with the fewest changed picks.

### Refit the top 15, then choose

The 15 best courts by the choice score each get the final stripe refit, not
only the winner. Each refitted court is then scored again with the same rule:
the same evidence score (paint, or the blend), plus the net bonus, measured on
the refitted court. The best valid refitted court wins. Switch:
`Switches.refit_top`, and `run_views.py --refit-top 15`.

On `gxBQ_window_03_frame_77876`, the better eligible court's far lines sit
about 1.6 native px from those of a court fitted to the hand marks (Sol's
measurement). The refit moves each court onto its stripes before the
comparison. It moves a
court only a few pixels, though, so it cannot turn a slipped court into the
right one. It can help only if the right court scores better once it sits on
its stripes. Top 15 reaches a good court on every view in the table above,
except `am2_window_01_frame_28019`. Each refit takes about 0.3 s, so 15 add
about 4 s to a view, about 3%.

If W5 ranked a view by geometry alone, because no court had any paint
evidence, the rescore uses the geometry score too. A refitted court without a
score in that criterion cannot be ranked, so it drops out.

### Defaults

Both switches are off by default: a weight of 0 and a refit of the top 1. This
reproduces today's detector. After the decision, the kept change becomes the
default and the code of any change that fails is reverted.

## What will run

Both changes act only after W5 scoring. The search, line templates and W5 do not
change, and repeat runs of them are bit-identical. So a replay of the final
choice from the upright check's saved results gives what a full run would.

`replay_court_choice.py` runs on the laptop. For each of the 28 views, it
rebuilds the scoring context from the view's frame, line fragments, person
boxes and the saved feet. It then calls the detector's own `choose_court` on the
saved W5 record, once per arm:

| Arm | Switches |
| --- | --- |
| `baseline` | none |
| `blend` | `geometry_weight=0.1` |
| `refit` | `refit_top=15` |
| `both` | `geometry_weight=0.1, refit_top=15` |

Self-checks are on, so each refit first replays its court's saved W5 fit to
within 0.0001 native px. The baseline arm must reproduce the upright check's
saved final court on all 28 views: the same chosen court and outcome, and
corners within 0.0001 native px. If it does not, the laptop's libraries differ
too much from Carmack's, and the arms go to Carmack instead.

The keep-or-revert decision waits for the project owner's approval. So does
one Carmack run of the kept configuration, which checks it end to end and
times it.

## How it will be judged

Set before the replay. The measure is the largest hand-mark error in floor
metres after the final refit, on the 10 views with landmark hand marks,
against the baseline arm.

- **An arm passes** if it makes at least one view more than 0.2 m better and
  no view more than 0.2 m worse. At the far end, one native pixel of hand-mark
  error moves a mark 0.18-0.22 m (Sol's measurement)
- **Court views without hand marks** are rendered wherever the court changes.
  A clearly wrong new court, such as an obvious one-line slip, counts as a
  view made more than 0.2 m worse
- **Control views** without a court: a control that gains a court is reported,
  but does not fail an arm on its own
- **If more than one arm passes**, the simplest is kept: the blend, then the
  refit, then both. A bigger arm is kept instead only if its total error over
  the 10 views is more than 0.2 m lower
- **Time** is the replay's seconds in `choose_court`, on the laptop. It
  decides nothing unless an arm costs more than 10%

## What ran

Commit `abc4318b`, with the replay and comparison scripts as committed here.
The replay ran on the laptop from the upright check's saved results.

It reproduces the upright check on 26 of the 28 views in all four arms. On
those views the baseline arm gives the saved final court within 0.0001 native
px, and every refitted court first replays its saved W5 fit within the same
tolerance. The baseline's final corners differ from the saved ones by at most
0.00000007 native px.

Two control views, which should get no court, cannot be replayed here:
- `sset_21_gloiZ_gTJaE_frame_00100347`, in all four arms: the upright check's
  wrong court there comes from a rank-deficient W5 fit
- `sset_21_gloiZ_gTJaE_frame_00014336`, in the two refit arms: two of its
  top 15 have rank-deficient W5 fits

A rank-deficient fit has no unique answer, so the laptop's libraries land
far from where Carmack's did. `fit_replay_scan.py` replays the W5 fit of every
court a refit arm could refit: 317 of 327 are within 0.0001 native px. All 10
misses are on these two views. W5 on Carmack recorded 9 of them as rank
deficient or with an invalid projection. The tenth is 0.00011 px off, just
over the tolerance. These two views are left out below and need Carmack.

`compare_court_choice.py` writes `compare_court_choice.txt`.

## Results

The largest hand-mark error in floor metres, before and after the final
refit:

| View | Baseline | Blend | Top-15 refit | Both |
| --- | --- | --- | --- | --- |
| `am1_window_00_frame_54` | 0.31 / 0.30 | 0.31 / 0.30 | 0.48 / 0.49 | 0.48 / 0.49 |
| `am2_window_00_frame_150` | 0.32 / 0.24 | 0.32 / 0.24 | 0.41 / 0.29 | 0.41 / 0.29 |
| `am2_window_01_frame_28019` | 0.69 / 0.62 | 0.69 / 0.62 | 0.81 / 0.77 | 0.81 / 0.77 |
| `am3_window_00_frame_0` | 0.26 / 0.24 | 0.26 / 0.24 | 0.31 / 0.26 | 0.31 / 0.26 |
| `am3_window_02_frame_17174` | 0.59 / 0.55 | 0.59 / 0.55 | 0.59 / 0.55 | 0.59 / 0.55 |
| `gxBQ_window_00_frame_0` | 0.34 / 0.19 | 0.34 / 0.19 | 0.84 / 0.92 | 0.84 / 0.92 |
| `gxBQ_window_00_frame_5` | 0.23 / 0.14 | 0.23 / 0.14 | 1.09 / 0.86 | 1.09 / 0.86 |
| `gxBQ_window_00_frame_689` | 1.18 / 0.94 | 0.58 / 0.32 | 0.43 / 0.14 | 0.43 / 0.14 |
| `gxBQ_window_03_frame_77876` | 1.40 / 1.06 | 1.40 / 1.06 | 1.38 / 1.08 | 1.38 / 1.08 |
| `letterboxed_short_frame_78` | 0.18 / 0.12 | 0.18 / 0.12 | 0.18 / 0.12 | 0.19 / 0.14 |
| **Total after refit** | 4.41 | 3.79 | 5.49 | 5.51 |

Against the keep rule:
- **Blend: passes.** One view is 0.62 m better and none is worse. It changes
  the court on no other replayed view. It costs no measurable time
- **Top-15 refit: fails.** `gxBQ_window_00_frame_0` is 0.73 m worse and
  `gxBQ_window_00_frame_5` 0.72 m worse. It changes the court on 18 of the 20
  court views, 9 of them by more than 2 native px. On `am2_window_01_frame_28019` it
  moves the court 291 px, to one whose near lines miss the paint by eye. It
  adds about 59 s over the 26 views on the laptop. On Carmack the refit takes
  about 0.3 s a court, so 14 more refits add about 4 s a view, roughly 3%
- **Both: fails**, for the same reasons as the refit

### Why the refit fails

The refit does make much better courts available. On 3 of the 10 views, the
best of the 15 refitted courts beats today's final court by more than 0.2 m:
- `gxBQ_window_00_frame_689`: 0.14 m against 0.94 m, and the rescore picks it
- `gxBQ_window_03_frame_77876`: 0.35 m against 1.06 m, but it places 12th
- `am3_window_02_frame_17174`: 0.21 m against 0.55 m, but it places 12th

Elsewhere the rescore moves away from good courts. On `gxBQ_window_00_frame_0`
the best refitted court places 8th, and on `gxBQ_window_00_frame_5` 3rd. So
refitting raises wrong courts' paint scores at least as much as the right
ones'.

The rescore measures refitted courts as W5 measured the unrefitted ones.
`footing_check.py` re-measures each hand-marked view's 15 best unrefitted
courts from their homographies alone, as the refit does. The paint scores
match W5's saved ones to within 0.0000000000001. The geometry scores differ by
up to 0.004 on five views. That touches only the combined arm, at a tenth of
the weight.

### Red-team review

A Codex review (GPT-6 Sol) of the build, the replay and the comparison agrees
with the verdict. Its report stays outside the repository. Its points:
- **The build is right at the defaults.** With a weight of 0 and a refit of the
  top 1, the detector picks and refits exactly as before
- **Top 15 tests fewer than 15 courts.** The shortlist ranks courts, and
  several are often children of the same parent. The refit starts from the
  parent, so those give the same refitted court. 32 of the 40 shortlists
  repeated a parent; `gxBQ_window_00_frame_5` had 9 distinct parents in 15.
  This does not change the verdict: on the views that got worse, the better
  refitted courts were in the shortlist and lost the rescore
- **Refitted courts are not re-gated.** Only the refit's own validity check
  applies, as for today's single winner. Six refitted courts failed the camera
  check; none won
- **If no refitted court has a score**, the code keeps the first court, where
  this README says unscored courts drop out. It never happened here
- **The replay is not a full run.** The two control views above cannot be
  replayed, and laptop time is not Carmack time. On
  `sset_21_gloiZ_gTJaE_frame_00100347`, the blend's pick before the refit is
  the same as today's
- The comparison now checks that it covers every view of the upright run

### Renders

The final court of each arm, after the refit, is drawn on every view where it
moved more than 2 native px (outside the repository, in the project owner's
perspective-image folders):
- `9_geometry_blend_final_26sep.png`: `gxBQ_window_00_frame_689`
- `10_top15_refit_final_26sep.png` and
  `11_blend_and_top15_refit_final_26sep.png`: 9 and 10 views

## Waiting for approval

- Make the blend the default (`geometry_weight=0.1`)
- Revert the top-15 refit switch
- One Carmack run of the new default on all 28 views, which also checks the
  two controls the laptop cannot replay, and times it

## Files

- `geometry_weight_sweep.py`, `good_court_ranks.py` and their `.txt` outputs:
  replays of the upright check's saved results, from before the build. Run them
  from the repository root with the saved artefacts folder as the argument
- `replay_court_choice.py`: the replay. `replay/<arm>/results/` and
  `replay/<arm>/artefacts/` are its output
- `compare_court_choice.py` and `compare_court_choice.txt`: the comparison,
  run on `replay/` and the upright check's run folder
- `fit_replay_scan.py`, `footing_check.py` and their `.txt` outputs: the
  replay's faithfulness checks

The upright check's saved artefacts (189 MB) are not in the repository. They
are in the upright check's run folder on Carmack.
