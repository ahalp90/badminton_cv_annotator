# X3D-M contact detection: validation and test results

**X3D-M gets 55.58% of test rallies fully correct with forced alternation,
compared with 59.99% for our annotator and 47.83% for MonoTrack.** Keeping the
model's raw hitter labels raises correct-player F1 from 90.48% to 96.39%, but
reduces fully correct rallies to 53.35%. Contact F1 stays at 96.94%.
These are five-seed means at ±10-frame tolerance.

We fine-tuned a Kinetics-400 X3D-M model to find racket hits and name the hitter,
with rally bounds supplied. Validation uses 8 videos, 668 rallies and 5,696
labelled hits. Test uses 46 videos, 3,327 rallies and 37,184 labelled hits.

## Validation selection

**Positives within ±1 frame, 50% contact clips per batch and learning rate 0.02
won the eight-configuration search.** We tested both ±1 and ±2-frame positives,
both 25% and 50% contact sampling, and learning rates 0.005 and 0.02, each with
seeds 0 and 1. With ±1-frame positives, centres exactly two frames from a hit
are excluded from the training loss unless another hit makes them positive.

Each run keeps its lowest-validation-cross-entropy checkpoint, earliest on a
tie. Mean fully correct validation rallies at ±10 frames with forced alternation
select the configuration; mean correct-player F1 breaks ties. All validation
checks use stride 3, offset 0. Seeds 2–4 use the winning configuration and do
not enter the configuration ranking. Test scores select nothing.

Table 1. Validation selection, ranked by two-seed means. Rally scores show
seed 0 / seed 1 / mean. Checkpoints show training updates for seed 0 / seed 1.

| Positive radius | Contact clips | Learning rate | Fully correct rallies (%) | Correct-player F1 (%), mean | Checkpoints |
|---|---:|---:|---:|---:|---:|
| ±1 frame | 50% | 0.02 | 50.15 / 49.70 / **49.93** | 92.15 | 25,000 / 35,000 |
| ±2 frames | 50% | 0.02 | 51.50 / 41.92 / **46.71** | 90.44 | 25,000 / 15,000 |
| ±2 frames | 25% | 0.02 | 44.91 / 47.75 / **46.33** | 92.90 | 25,000 / 30,000 |
| ±1 frame | 50% | 0.005 | 45.06 / 46.41 / **45.73** | 90.39 | 25,000 / 35,000 |
| ±2 frames | 25% | 0.005 | 45.81 / 45.36 / **45.58** | 92.46 | 50,000 / 30,000 |
| ±1 frame | 25% | 0.02 | 46.11 / 43.41 / **44.76** | 93.09 | 25,000 / 25,000 |
| ±1 frame | 25% | 0.005 | 44.01 / 44.31 / **44.16** | 91.47 | 25,000 / 25,000 |
| ±2 frames | 50% | 0.005 | 46.56 / 38.32 / **42.44** | 88.01 | 25,000 / 15,000 |

![X3D-M validation configuration comparison](writeup_figures/x3d_stride3_validation.png)

Figure 1. Validation at ±10 frames with forced alternation. Diamonds show means,
ticks show individual seeds and lines span the seeds.

## Test results

Table 2. Selected checkpoints at ±10 frames with forced alternation.

| Seed | Checkpoint (updates) | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |
|---:|---:|---:|---:|---:|
| 0 | 25,000 | 96.83 | 90.43 | 54.28 |
| 1 | 35,000 | 96.96 | 91.07 | 55.58 |
| 2 | 30,000 | 97.10 | 91.84 | 56.39 |
| 3 | 20,000 | 96.44 | 87.48 | 49.20 |
| 4 | 40,000 | 97.40 | 91.56 | 62.46 |

Table 3. Test comparison at ±10 frames. Values are seed means (minimum–maximum).
All systems receive rally bounds; our annotator and MonoTrack use their own
court detections and evaluate every frame. The video models use stride 3.

| System | Seeds | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |
|---|---:|---:|---:|---:|
| X3D-M, raw hitters | 5 | 96.94 (96.44–97.40) | 96.39 (95.77–96.83) | 53.35 (46.80–60.38) |
| X3D-M, forced alternation | 5 | 96.94 (96.44–97.40) | 90.48 (87.48–91.84) | 55.58 (49.20–62.46) |
| VideoMAE, raw hitters | 2 | 95.49 (95.06–95.92) | 94.93 (94.41–95.46) | 39.22 (34.63–43.82) |
| VideoMAE, forced alternation | 2 | 95.49 (95.06–95.92) | 84.41 (82.72–86.10) | 40.67 (36.46–44.88) |
| Our annotator, own courts | 5 | 96.84 (96.75–96.94) | 94.42 (94.07–94.76) | 59.99 (58.97–61.41) |
| MonoTrack, own courts | 5 | 96.04 (95.95–96.18) | 95.84 (95.69–95.99) | 47.83 (46.44–49.17) |

![Test comparison with raw and forced hitter labels](writeup_figures/stride3_test_comparison.png)

Figure 2. Test seed means, individual seeds and ranges at ±10 frames.

## Raw hitters and forced alternation

Forcing alternation adds 2.23 percentage points of fully correct test rallies,
but costs 5.92 points of correct-player F1. It repairs some complete rallies
while assigning the wrong hitter to more individual hits elsewhere. Raw X3D
hitters have the highest mean correct-player F1 in this comparison; our
annotator has the highest mean fully correct rally score.

Table 4. Selected-configuration seed means on validation and test.

| Split | Tolerance | Hitter rule | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |
|---|---|---|---:|---:|---:|
| Validation | ±10 frames | Raw | 95.88 | 95.08 | 48.77 |
| Validation | ±10 frames | Forced alternation | 95.88 | 92.23 | 49.46 |
| Validation | ±5 frames | Raw | 94.99 | 94.36 | 44.91 |
| Validation | ±5 frames | Forced alternation | 94.99 | 91.47 | 45.42 |
| Test | ±10 frames | Raw | 96.94 | 96.39 | 53.35 |
| Test | ±10 frames | Forced alternation | 96.94 | 90.48 | 55.58 |
| Test | ±5 frames | Raw | 95.73 | 95.33 | 47.41 |
| Test | ±5 frames | Forced alternation | 95.73 | 89.43 | 48.97 |

![X3D-M raw and forced hitters on validation and test](writeup_figures/x3d_stride3_hitters.png)

Figure 3. The same selected checkpoints under both hitter rules. Diamonds show
means, ticks show individual seeds and lines span the seeds.

## Method

The model predicts no contact, TOP hitter or BOT hitter for each evaluated clip
centre. Validation and test use every third video frame, starting at offset 0.
Skipped centres cannot produce hits; every labelled hit and rally still counts.
After choosing each centre's most likely class, we keep the strongest hit among
predictions within 6 frames at 30 fps, scaled for other frame rates.

Raw hitters keep the model's player labels. Forced alternation assigns a single
TOP/BOT alternating pattern to each rally by majority vote, using the predicted
probabilities to break tied votes. Contact times stay the same. A missed or extra
hit can put the alternating pattern out of step with the remaining hits.

A predicted hit matches a labelled hit within ±5 or ±10 frames at 30 fps, scaled
for other frame rates. Contact F1 is twice the number of matched hits divided by
the total number of labelled and predicted hits. Correct-player F1 uses only
matches with the right hitter in the numerator. These scores pool hits across
videos, so long rallies contribute more hits. Fully correct rallies have every
hit found, no extra hits and every hitter right; each rally counts equally.
We calculate each seed's score first, then take the mean across seeds. Ranges
are the lowest and highest observed seed scores.

Validation includes rallies marked as flawed in the source labels; test excludes
them. One test hit has an unknown hitter. It counts for contact scoring, earns no
correct-player credit and prevents its rally from being fully correct.

## Results and figures

The completed run records are in [results/stride3_20261105/](results/stride3_20261105/):

- [runs.json.gz](results/stride3_20261105/runs.json.gz): settings and all 240 validation checks across 23 runs.
- [selection.json.gz](results/stride3_20261105/selection.json.gz): configuration rankings from validation seeds 0 and 1.
- [scores.csv.gz](results/stride3_20261105/scores.csv.gz): selected checkpoints, both hitter rules, both splits and both tolerances.
- [data.json.gz](results/stride3_20261105/data.json.gz) and [predictions/](results/stride3_20261105/predictions/): rally labels and saved contact predictions.

Download the selected checkpoints through the [weights index](weights/README.md).
The comparison systems use the [MonoTrack benchmark summary](../post_split_correction/benchmark/contact_summary.json.gz).

Regenerate the figures from the repository root:

```bash
python3 experiments/annotator/video_models_benchmark/writeup_scripts/make_stride3_figures.py
```
