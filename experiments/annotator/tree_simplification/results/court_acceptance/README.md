# Removing the court head-count rule

The annotator used to ignore a detected court unless exactly two people stood
on it in at least half of the scene's frames. We removed that rule so every
detected court is used. Removal alone lost fully correct rallies, because the
ignored scenes had been acting as breaks between neighbouring rallies.

Two additions recovered the loss:

- The contact model receives the number of standing people on the court as a
  feature, rather than as a hard rule.
- After rally selection, a predicted rally is split wherever accepted contacts
  are more than three seconds apart. Players are refitted within each piece,
  and the pieces keep the parent's outer bounds.

Since then, the annotator has gained a second rally selector that corrects the
split pieces. See [Correcting rallies after splitting](../../../post_split_correction/README.md).

Tables show five-seed means (seeds 0, 100, 200, 300 and 400) at a tolerance
of ten frames at 30 fps, scaled to each video's frame rate. Training used 32 videos. Validation has 8 videos and 668
labelled rallies; test has 46 videos and 3,327 scored rallies. Settings were
chosen on validation, with test results used as a sanity check. A fully correct rally has
every labelled contact, no extra contacts and the right players. The
[benchmark report](../../../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md)
defines the scoring.

## Removing the rule

| Pipeline | Validation fully correct | Validation player F1 | Test fully correct | Test player F1 |
|---|---:|---:|---:|---:|
| Old head-count rule | 352.0 | 89.786% | 1,963.2 | 91.763% |
| Rule removed, original models | 342.0 | 89.383% | 1,886.6 | 90.600% |
| Rule removed, all three models refitted | 338.0 | 89.232% | 1,874.8 | 90.447% |

Scenes that failed the rule were treated as having no court. Before rally
discovery, the annotator held the shuttle still in those frames, so they looked
like rest between rallies. Without the rule, those frames became ordinary play
and neighbouring rallies started to merge. The rally selector can add or drop
contacts but can't split a merged rally, so refitting didn't help.

## Choices made on validation

Fully correct validation rallies, out of 668, with a three-second split:

| Standing-person count feature | No split | Split after selection | Split before selection |
|---|---:|---:|---:|
| Without | 338.0 | 349.6 | 358.2 |
| With | 344.0 | **356.6** | 358.4 |

- **Split after selection.** Splitting before selection retrains the rally
  selector on split pieces. With the count feature, it scored only 1.8 rallies more,
  lost 16.4 previously correct rallies per seed, and lowered contact F1 from
  90.900% to 90.651% and player F1 from 89.900% to 89.614%. Splitting after
  selection lost no previously correct rallies and removed all 10.0 merged
  predictions per seed.
- **Use the actual count.** With a 2.5-second split, the actual count gave
  356.6 rallies. A hand-set mapping (0 people → 0, 1 → 0.5, 2 → 1, more → −1)
  gave 352.2, and no count gave 349.6. The count feature improved four of five
  seeds.
- **Three seconds.** A sweep with the original models, rule removed, showed
  short thresholds break real rallies: 2 seconds lost 168 previously correct rally/seed pairs,
  1.5 seconds lost 579. From 2.5 to 4 seconds, results were flat (351.0–351.8)
  with no losses. After refitting, 2.5 and 3 seconds tied at 356.6 and
  5 seconds gave 356.0. The selected threshold is 3 seconds.
- **One duration for high shots.** Raising the tracking-loss allowance for high
  shots from 2.5 to 3 seconds changed none of the 270 saved outputs (54
  validation and test videos, five seeds). Both limits now use three seconds.
- **Keep outer bounds.** Trimming pieces to their first and last hits scored the
  same, because the benchmark uses contact endpoints. Keeping the parent's bounds
  keeps the video after the last hit available for landing estimation.

| Validation | Contact F1 | Player F1 | Fully correct |
|---|---:|---:|---:|
| Old head-count rule | 90.897% | 89.786% | 352.0 |
| Selected: count feature, split after selection | 90.900% | 89.900% | 356.6 |

## Test check

| Pipeline | Fully correct | Contact F1 | Player F1 |
|---|---:|---:|---:|
| Old head-count rule | 1,963.2 | 94.184% | 91.763% |
| Rule removed, refitted | 1,874.8 | 93.421% | 90.447% |
| Plus three-second split | 1,985.2 | 93.421% | 91.260% |
| Plus count feature (selected) | 2,001.4 | 93.725% | 91.609% |

The selected pipeline gets 38.2 more fully correct test rallies than the old
rule. Contact F1 stays 0.459 points lower and player F1 0.154 points lower.
Per seed, it adds 453.0 false contacts and finds 82.8 more true ones. Splitting
gained 111.6 test rallies per seed and lost none that were already correct. It
does break some imperfect rallies: labelled rallies covered by more than one
prediction rose from 1.0 to 8.2 per seed.

After integrating the shared feature builder, a fresh five-seed fit gave
353.2 fully correct validation rallies and 1,992.0 test rallies. Those models
are the starting point for the [second-selector comparison](../../../post_split_correction/README.md).

The labels contain almost no quick restarts. Validation's shortest pause between
adjacent rallies is 9.7 seconds, training has one under 5 seconds (4.0), and
every short test pause touches an excluded rally. These data can't show how a
three-second split behaves on amateur footage with quick restarts.

## Data and code

- [summary.json.gz](summary.json.gz): per-seed and per-video results for the
  rule-removal comparison at ±5 and ±10 frames, with review-ranking measures.
- [count_and_split.json.gz](count_and_split.json.gz): per-seed contact and rally
  counts for every pipeline above, the validation split and count comparisons,
  the threshold sweep, the high-shot and outer-bound comparisons, and the short
  pauses found in the labels.

Commit `5f9e721e` removes the rule. The rule-removal harness is
[court_acceptance.py](../../court_acceptance.py); use its version at commit
`0d75074e` for the exact runs. Commit `92150b68`
adds the count feature and the split; `9205982a` sets both limits to three
seconds.
