# Video-model validation and test results

X3D-M records come first; [VideoMAE-B records](#videomae-b-pilot) follow.

Each directory under [`x3d_pilot/`](x3d_pilot/) is one training run, named
`{scheme}-contact{percent}-lr{rate}-seed{seed}`. We keep all 12 runs,
including the ones not selected, as the pilot's full record. The write-up is
[`../x3d_results.md`](../x3d_results.md).

## Files in each run

| File | Contents |
|---|---|
| `settings.json.gz` | Training settings: seed, batch size, learning rate and warm-up, original update budget and validation steps, contact share per batch, precision and label scheme. |
| `history.json.gz` | One entry per validation check, original 30,000 updates then continuation. Each has the update count, mean training loss since the previous check, validation loss, macro F1, a confusion matrix (rows true class, columns predicted; 0 no contact, 1 bottom player, 2 top player), event scores, the next learning rate and the interval's run time. |
| `continuation.json.gz` | Continuation schedule: start update, peak learning rate, half-cycle length, validation interval and patience. |
| `training_progress.json.gz` | Training-loss readings about every 30 seconds. The four scheme A seed 0 runs and the two scheme A 50%-contact seed 1 runs started before this logger existed, so their readings begin at 30,000 updates. See [Progress and recovery](../README.md#progress-and-recovery) for the fields. |

Event scores are keyed by timing tolerance in frames at 30 fps (`"10"` and
`"5"`), scaled to each video's frame rate. Each holds labelled, predicted,
matched and correct-player hit counts, rally and fully correct rally counts,
contact F1, correct-player F1 and the fully correct percentage. Every check
covers the same 668 validation rallies and 5,696 labelled hits.

The two selected checkpoints are in [`../weights/`](../weights/README.md).
Other checkpoints and per-frame probabilities are not included.

## Held-out test results

[`x3d_test/`](x3d_test/) contains `seed0/` and `seed1/`, evaluated using the
selected ±1-frame-positive, 50%-contact, LR 0.02 checkpoints at 25,000 and
35,000 updates respectively. Both cover 46 videos, 3,327 rallies and 37,184
labelled hits. Settings and checkpoint selection were fixed before testing.

| File in each seed directory | Contents |
|---|---|
| `events.json.gz` | Aggregate contact F1, correct-player F1 and fully correct rally scores at ±5 and ±10 frames, with supporting counts; decoded event predictions grouped by video and rally. |
| `frame_metrics.json.gz` | Test cross-entropy, macro F1 and frame confusion matrix, using the same class order as the validation histories. |

The [report](../x3d_results.md#held-out-test-comparison) compares the two seeds
with the five-seed supplied-boundary annotator and MonoTrack results.

## Other files

| File | Contents |
|---|---|
| `development_counts.json.gz` | For the training and validation splits: videos, rallies, labelled contacts, clip centres, videos by frame rate, and the number of top, bottom, background and ignored centres under each label scheme. |
| `x3d_timing.json.gz` | Throughput trials on an NVIDIA L40: seconds per training and validation batch and peak GPU memory, for the batch sizes and memory layouts tried. |

## Selected checkpoints

Each run keeps its lowest-validation-loss check, the earliest on a tie. Scores
are at ±10 frames. The selected configuration's two runs, in bold, are the
checkpoints used for the held-out test evaluation.

| Run | Selected (updates) | Last check (updates) | Validation loss | Fully correct rallies (%) | Correct-player F1 (%) |
|---|---:|---:|---:|---:|---:|
| A-contact25-lr0.005-seed0 | 25,000 | 40,000 | 0.1172 | 42.5 | 90.8 |
| A-contact25-lr0.005-seed1 | 25,000 | 40,000 | 0.1094 | 44.2 | 91.2 |
| A-contact25-lr0.02-seed0 | 25,000 | 40,000 | 0.1115 | 43.0 | 91.3 |
| A-contact25-lr0.02-seed1 | 25,000 | 40,000 | 0.1106 | 42.2 | 92.7 |
| A-contact50-lr0.005-seed0 | 25,000 | 40,000 | 0.1532 | 41.3 | 87.4 |
| A-contact50-lr0.005-seed1 | 35,000 | 50,000 | 0.1427 | 40.0 | 89.2 |
| **A-contact50-lr0.02-seed0** | **25,000** | 40,000 | 0.1330 | 48.1 | 92.2 |
| **A-contact50-lr0.02-seed1** | **35,000** | 50,000 | 0.1357 | 45.4 | 90.1 |
| B-contact25-lr0.005-seed0 | 30,000 | 45,000 | 0.1760 | 44.0 | 91.9 |
| B-contact25-lr0.005-seed1 | 30,000 | 45,000 | 0.1681 | 44.2 | 91.7 |
| B-contact50-lr0.02-seed0 | 25,000 | 40,000 | 0.1933 | 46.7 | 91.4 |
| B-contact50-lr0.02-seed1 | 15,000 | 30,000 | 0.2064 | 31.7 | 80.9 |

Validation loss compares within a scheme only, because schemes A and B label
different frames as contacts. B-contact50-lr0.02-seed1 had already gone three
checks without improvement at 30,000 updates, so its continuation stopped
before any further training.

## VideoMAE-B pilot

Each directory under [`videomae_pilot/`](videomae_pilot/) is one training run,
named `videomae-lr{rate}-seed{seed}`. Each holds the same four files as an X3D-M
run: `settings.json.gz`, `history.json.gz`, `continuation.json.gz` and
`training_progress.json.gz`. The settings add the VideoMAE fields described in
the [run guide](../VIDEOMAE.md). Every check covers the same 668 validation
rallies and 5,696 labelled hits as X3D-M. The write-up is
[`../videomae_results.md`](../videomae_results.md).

After all four runs finish, `videomae_pilot/selection.json.gz` records the
queue's learning-rate ranking and each run's selected checkpoint. Test outputs
for both selected seeds then go in `videomae_test/seed0/` and
`videomae_test/seed1/`, with the same two files as `x3d_test/`.

[`videomae_stride3/validation_scores.csv.gz`](videomae_stride3/validation_scores.csv.gz)
has one row per run, sampling, hitter rule and tolerance. Columns:

| Column | Meaning |
|---|---|
| `split`, `run`, `learning_rate`, `seed` | Which predictions were rescored |
| `best_step` | Update count of the lowest-validation-loss checkpoint that made the predictions |
| `stride`, `offset` | 1 and 0 for dense prediction; 3 and 0–2 for every third video frame, starting at that offset |
| `hitter_rule` | `raw` keeps each hit's own most likely hitter; `alternating` is the reported rule |
| `tolerance` | Timing allowance in frames at 30 fps, scaled to each video's frame rate |
| `labelled`, `predicted`, `matched`, `correct_player`, `rallies`, `fully_correct` | Counts behind the scores |
| `contact_recall`, `contact_precision`, `contact_f1`, `correct_player_f1`, `fully_correct_percent` | Scores in percent |

`test_scores.csv.gz` will use the same columns for the two test seeds.
[`videomae_timing/`](videomae_timing/) holds the batch-size throughput measurements.
Checkpoints and per-frame probabilities are not included.
