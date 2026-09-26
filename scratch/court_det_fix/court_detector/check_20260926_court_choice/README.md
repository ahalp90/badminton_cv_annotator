# Court-choice check, 26 September 2026

This check tests two changes to how the detector picks its final court, alone
and together. Both aim at views where a court much closer to the hand marks is
already eligible but loses. Everything else matches
`../check_20260926_upright/` (commit `c5a7cfdb`), including the upright-camera
filter.

**Result: not run yet.**

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
- **Time** is the replay's seconds in `choose_court`, on the laptop, as a
  share of the upright check's detect seconds. It decides nothing unless an arm
  costs more than 10%

## Files

- `geometry_weight_sweep.py`, `good_court_ranks.py` and their `.txt` outputs:
  replays of the upright check's saved results. Run them from the repository
  root with the saved artefacts folder as the argument
