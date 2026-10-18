# Tree retune results

Run completed 18 October 2026.

**Result: the shipped settings stay.** The best setting on validation gained less on test than the spread between seeds.

grid-017, the best on validation, against shipped on the 46 test videos:

- correct-player F1: +0.13 points (per seed: −0.14 to +0.45)
- contact F1: −0.05 points
- fully correct rallies: −0.44 points

Shipped seed 0 matches the production model's test result exactly: 40,246 predicted contacts, 34,200 matched, 33,156 with the right player and 1,744 fully correct rallies.

Training the contact tree on all 40 videos instead of 32 raised fully correct rallies on test by 0.7–0.8 points with discovered rallies and 1.5–1.7 points with known rallies. Details: [32-video and 40-video contact trees](#32-video-and-40-video-contact-trees-on-test).

## Test results

Test set: 46 ShuttleSet22 videos, 3,327 labelled rallies, 37,184 labelled contacts.

How to read the tables:

- each cell: mean of five seeds, with the lowest and highest seed in brackets
- values are percentages
- tolerances are set at 30 fps and scaled to each video's frame rate
- counts are pooled across videos before rates are worked out

| Configuration | Contact F1 ±10 | Correct-player F1 ±10 | Fully correct rallies ±10 |
| --- | ---: | ---: | ---: |
| Shipped | 88.33 (88.27–88.41) | 85.48 (85.35–85.67) | 52.16 (51.73–52.42) |
| grid-017 | 88.28 (88.23–88.31) | 85.61 (85.45–85.83) | 51.72 (50.89–52.15) |

| Configuration | Contact F1 ±5 | Correct-player F1 ±5 | Fully correct rallies ±5 |
| --- | ---: | ---: | ---: |
| Shipped | 86.58 (86.48–86.69) | 83.90 (83.74–84.11) | 42.19 (41.75–42.77) |
| grid-017 | 86.41 (86.33–86.50) | 83.91 (83.79–84.19) | 41.33 (40.79–42.08) |

### Paired differences

Both configurations use the same seed offset in each row. Differences are grid-017 minus shipped at ±10, in percentage points, except where rallies are counted.

| Seed offset | Correct-player F1 | Contact F1 | Fully correct rallies |
| ---: | ---: | ---: | ---: |
| 0 | +0.07 | −0.06 | −51 (−1.53) |
| 100 | +0.19 | +0.01 | 0 (0.00) |
| 200 | −0.14 | −0.17 | −22 (−0.66) |
| 300 | +0.10 | −0.02 | −1 (−0.03) |
| 400 | +0.45 | −0.01 | +1 (+0.03) |
| Mean | +0.13 | −0.05 | −14.6 (−0.44) |

### Precision and recall at ±10

| Configuration | Contact precision | Contact recall | Correct-player precision | Correct-player recall |
| --- | ---: | ---: | ---: | ---: |
| Shipped | 84.93 (84.87–84.98) | 92.01 (91.95–92.14) | 82.19 (82.07–82.38) | 89.04 (88.89–89.29) |
| grid-017 | 84.80 (84.71–84.85) | 92.06 (92.02–92.13) | 82.24 (82.10–82.40) | 89.28 (89.09–89.56) |

### Hit-level errors at ±10

Each error is one contact to add, delete or reassign. Means are per run. Extra contacts include predictions outside labelled rallies.

| Configuration | Missed | Extra | Wrong player | Total |
| --- | ---: | ---: | ---: | ---: |
| Shipped | 2,971.6 (2,922–2,992) | 6,069.8 (6,046–6,095) | 1,103.6 (1,044–1,153) | 10,145.0 (10,045–10,217) |
| grid-017 | 2,952.2 (2,926–2,969) | 6,137.8 (6,115–6,177) | 1,033.8 (956–1,107) | 10,123.8 (10,040–10,179) |

Where grid-017 differs: about 70 fewer wrong-player contacts per run, but about 68 more extra contacts. The totals end up roughly even.

### Every seed at ±10

| Configuration | Seed offset | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | ---: | ---: | ---: | ---: |
| Shipped | 0 | 88.34 | 85.64 | 1,744 (52.42) |
| Shipped | 100 | 88.27 | 85.35 | 1,721 (51.73) |
| Shipped | 200 | 88.41 | 85.67 | 1,744 (52.42) |
| Shipped | 300 | 88.33 | 85.35 | 1,734 (52.12) |
| Shipped | 400 | 88.30 | 85.38 | 1,734 (52.12) |
| grid-017 | 0 | 88.28 | 85.71 | 1,693 (50.89) |
| grid-017 | 100 | 88.28 | 85.54 | 1,721 (51.73) |
| grid-017 | 200 | 88.23 | 85.53 | 1,722 (51.76) |
| grid-017 | 300 | 88.31 | 85.45 | 1,733 (52.09) |
| grid-017 | 400 | 88.29 | 85.83 | 1,735 (52.15) |

### Per video

Each value is that video's mean across the five seeds. The headline scores aren't averages of this table: they pool all videos within each run first.

| Metric | Videos: grid-017 higher | Equal | Videos: shipped higher |
| --- | ---: | ---: | ---: |
| Contact F1 | 16 | 0 | 30 |
| Correct-player F1 | 25 | 0 | 21 |
| Fully correct discovered rallies | 24 | 2 | 20 |

Values are percentages. Win/tie counts use unrounded values.

| Video | Contact F1: shipped | Contact F1: grid-017 | Correct-player F1: shipped | Correct-player F1: grid-017 | Fully correct discovered: shipped | Fully correct discovered: grid-017 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 93.40 | 93.15 | 90.21 | 89.67 | 51.43 | 48.31 |
| 9 | 93.28 | 93.47 | 91.48 | 91.13 | 46.67 | 46.98 |
| 10 | 81.57 | 81.29 | 79.06 | 79.29 | 47.76 | 48.57 |
| 11 | 82.27 | 82.07 | 80.04 | 80.31 | 33.18 | 32.73 |
| 12 | 77.25 | 77.13 | 71.37 | 71.27 | 45.25 | 41.97 |
| 13 | 89.88 | 90.20 | 86.44 | 89.17 | 43.49 | 38.84 |
| 16 | 96.62 | 96.81 | 94.37 | 94.93 | 50.83 | 53.61 |
| 17 | 88.27 | 88.14 | 68.10 | 72.83 | 25.75 | 24.11 |
| 18 | 93.64 | 93.36 | 92.93 | 92.78 | 74.00 | 73.00 |
| 19 | 88.40 | 88.20 | 86.71 | 85.88 | 53.47 | 53.47 |
| 20 | 72.11 | 72.39 | 71.41 | 71.75 | 40.93 | 50.23 |
| 21 | 74.76 | 74.53 | 72.15 | 72.24 | 41.60 | 38.13 |
| 22 | 81.49 | 81.53 | 80.75 | 80.83 | 56.50 | 60.00 |
| 23 | 87.96 | 88.10 | 86.35 | 86.46 | 56.32 | 59.08 |
| 24 | 68.61 | 68.64 | 68.06 | 68.44 | 47.74 | 52.26 |
| 25 | 74.45 | 74.41 | 71.18 | 71.22 | 37.88 | 36.94 |
| 26 | 88.12 | 87.80 | 84.88 | 84.22 | 44.11 | 44.66 |
| 27 | 89.15 | 88.78 | 88.31 | 87.90 | 65.45 | 64.00 |
| 28 | 85.01 | 85.21 | 84.04 | 84.31 | 53.17 | 55.61 |
| 29 | 86.62 | 86.54 | 84.25 | 84.20 | 45.45 | 43.64 |
| 30 | 96.43 | 96.36 | 94.82 | 94.08 | 60.31 | 61.23 |
| 31 | 94.11 | 94.03 | 93.76 | 93.78 | 70.59 | 70.59 |
| 32 | 90.92 | 90.78 | 90.05 | 90.22 | 51.53 | 52.20 |
| 33 | 94.97 | 95.08 | 94.30 | 94.25 | 64.89 | 68.30 |
| 34 | 96.14 | 96.08 | 93.78 | 93.72 | 55.61 | 54.63 |
| 35 | 92.42 | 92.29 | 91.12 | 90.72 | 63.55 | 61.61 |
| 36 | 74.36 | 74.45 | 73.31 | 73.40 | 44.00 | 48.00 |
| 37 | 91.55 | 91.88 | 90.27 | 90.44 | 62.00 | 55.64 |
| 38 | 79.13 | 78.84 | 72.25 | 72.35 | 33.17 | 31.46 |
| 39 | 95.13 | 94.81 | 93.93 | 93.42 | 58.93 | 53.60 |
| 40 | 91.86 | 91.62 | 90.41 | 89.95 | 62.47 | 58.88 |
| 41 | 95.96 | 95.79 | 95.18 | 94.93 | 70.13 | 60.52 |
| 42 | 70.38 | 70.80 | 56.74 | 57.41 | 27.93 | 29.31 |
| 43 | 82.96 | 82.86 | 81.09 | 81.01 | 60.98 | 61.22 |
| 44 | 94.10 | 94.11 | 91.03 | 91.22 | 53.47 | 55.05 |
| 46 | 85.74 | 85.72 | 83.03 | 83.08 | 48.62 | 49.85 |
| 47 | 94.09 | 94.05 | 90.95 | 90.90 | 48.37 | 49.18 |
| 48 | 91.62 | 91.48 | 89.31 | 89.59 | 46.67 | 47.27 |
| 49 | 94.71 | 94.62 | 93.23 | 93.36 | 55.70 | 56.26 |
| 50 | 92.29 | 92.50 | 90.10 | 90.60 | 51.43 | 52.00 |
| 51 | 91.27 | 91.17 | 89.23 | 88.95 | 59.26 | 58.52 |
| 52 | 94.44 | 94.36 | 92.23 | 91.98 | 51.67 | 52.78 |
| 53 | 95.35 | 95.36 | 92.40 | 92.26 | 46.84 | 44.47 |
| 54 | 91.27 | 91.07 | 89.44 | 88.57 | 55.14 | 53.71 |
| 55 | 85.56 | 85.45 | 83.66 | 83.78 | 57.25 | 58.82 |
| 57 | 90.04 | 90.26 | 88.73 | 88.76 | 61.59 | 63.81 |

## 32-video and 40-video contact trees on test

**The contact tree trained on all 40 videos did better on test than the 32-video one.** Fully correct rallies gained most. grid-017 still didn't beat shipped.

At ±10, 40-video tree minus 32-video tree:

- discovered rallies (A): +0.7 to +0.8 points of fully correct rallies, +0.1 to +0.2 points of correct-player F1
- known rallies (B and C): +1.5 to +1.7 points of fully correct rallies, +0.15 to +0.19 points of correct-player F1
- every seed improved on all three scores under B and C; under A, a few seeds dropped slightly

At ±10, grid-017 minus shipped on correct-player F1:

- discovered rallies: +0.07 points with the 32-video tree, +0.13 with the 40-video tree
- known rallies: −0.11 and −0.07 points
- every seed range includes zero

What differs between the two models:

- 32-video models: the ones scored on validation
- 40-video models: the test models in the tables above
- only the contact tree changes; both use the same 32-video sequence, serve and confidence models and the same seed offsets

Conditions:

- A: the annotator finds the rallies itself, using scene courts from fast-robust mode. This is the setup of the test results above.
- B: labelled rallies, padded by 1 second at each end, with video-robust courts.
- C: as B, with MonoTrack's courts.

B and C are the `ours_our_courts` and `ours_mono_courts` conditions of the [MonoTrack benchmark](../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup.md). They drop predictions outside the padded windows and judge each rally inside its window. Their scores run higher than A's for that reason, so A can't be compared with B or C directly.

Check: under B and C, shipped seed 0 with the 40-video tree matched the benchmark's production scores exactly.

Table format as above: 46 test videos, 3,327 labelled rallies, five seeds per row. Records: `results/test_rerun/`.

### Scores at ±10

| Condition | Configuration | Contact tree | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | ---: | ---: | ---: |
| A: discovered rallies | Shipped | 32 videos | 88.30 (88.22–88.38) | 85.37 (85.08–85.52) | 51.39 (50.74–52.06) |
| A: discovered rallies | Shipped | 40 videos | 88.33 (88.27–88.41) | 85.48 (85.35–85.67) | 52.16 (51.73–52.42) |
| A: discovered rallies | grid-017 | 32 videos | 88.23 (88.21–88.24) | 85.44 (85.29–85.59) | 51.02 (50.44–51.37) |
| A: discovered rallies | grid-017 | 40 videos | 88.28 (88.23–88.31) | 85.61 (85.45–85.83) | 51.72 (50.89–52.15) |
| B: known rallies, video-robust courts | Shipped | 32 videos | 96.59 (96.52–96.69) | 94.21 (94.08–94.31) | 59.96 (58.88–60.93) |
| B: known rallies, video-robust courts | Shipped | 40 videos | 96.72 (96.68–96.77) | 94.37 (94.28–94.48) | 61.50 (60.99–62.07) |
| B: known rallies, video-robust courts | grid-017 | 32 videos | 96.56 (96.52–96.61) | 94.10 (93.98–94.29) | 59.72 (59.60–59.93) |
| B: known rallies, video-robust courts | grid-017 | 40 videos | 96.71 (96.65–96.77) | 94.29 (94.21–94.44) | 61.40 (60.93–61.71) |
| C: known rallies, MonoTrack courts | Shipped | 32 videos | 96.53 (96.46–96.63) | 94.14 (94.01–94.24) | 59.84 (58.76–60.84) |
| C: known rallies, MonoTrack courts | Shipped | 40 videos | 96.66 (96.62–96.70) | 94.30 (94.20–94.41) | 61.35 (60.84–61.89) |
| C: known rallies, MonoTrack courts | grid-017 | 32 videos | 96.50 (96.46–96.55) | 94.04 (93.92–94.23) | 59.59 (59.48–59.78) |
| C: known rallies, MonoTrack courts | grid-017 | 40 videos | 96.66 (96.59–96.71) | 94.23 (94.16–94.37) | 61.27 (60.81–61.62) |

### Scores at ±5

| Condition | Configuration | Contact tree | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | ---: | ---: | ---: |
| A: discovered rallies | Shipped | 32 videos | 86.50 (86.38–86.58) | 83.74 (83.41–83.85) | 41.21 (40.61–41.78) |
| A: discovered rallies | Shipped | 40 videos | 86.58 (86.48–86.69) | 83.90 (83.74–84.11) | 42.19 (41.75–42.77) |
| A: discovered rallies | grid-017 | 32 videos | 86.37 (86.31–86.46) | 83.76 (83.58–83.85) | 40.85 (40.34–41.54) |
| A: discovered rallies | grid-017 | 40 videos | 86.41 (86.33–86.50) | 83.91 (83.79–84.19) | 41.33 (40.79–42.08) |
| B: known rallies, video-robust courts | Shipped | 32 videos | 94.70 (94.62–94.79) | 92.50 (92.31–92.63) | 48.79 (48.27–49.53) |
| B: known rallies, video-robust courts | Shipped | 40 videos | 94.87 (94.74–95.00) | 92.69 (92.59–92.78) | 50.17 (48.99–51.13) |
| B: known rallies, video-robust courts | grid-017 | 32 videos | 94.61 (94.47–94.76) | 92.32 (92.15–92.51) | 48.00 (47.07–49.20) |
| B: known rallies, video-robust courts | grid-017 | 40 videos | 94.78 (94.70–94.94) | 92.55 (92.47–92.65) | 49.58 (49.05–50.83) |
| C: known rallies, MonoTrack courts | Shipped | 32 videos | 94.63 (94.55–94.72) | 92.42 (92.24–92.55) | 48.67 (48.09–49.41) |
| C: known rallies, MonoTrack courts | Shipped | 40 videos | 94.80 (94.68–94.94) | 92.62 (92.51–92.69) | 50.04 (48.87–50.92) |
| C: known rallies, MonoTrack courts | grid-017 | 32 videos | 94.53 (94.40–94.69) | 92.25 (92.09–92.42) | 47.86 (46.89–49.11) |
| C: known rallies, MonoTrack courts | grid-017 | 40 videos | 94.72 (94.63–94.89) | 92.48 (92.39–92.58) | 49.49 (48.93–50.77) |

### 40-video minus 32-video tree at ±10

Paired by seed offset. Values are differences in points: the mean, with the lowest and highest seed in brackets.

| Condition | Configuration | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | ---: | ---: | ---: |
| A: discovered rallies | Shipped | +0.03 (−0.08 to +0.11) | +0.11 (−0.10 to +0.30) | +0.77 (+0.06 to +1.41) |
| A: discovered rallies | grid-017 | +0.05 (−0.01 to +0.10) | +0.17 (+0.02 to +0.26) | +0.70 (−0.03 to +1.65) |
| B: known rallies, video-robust courts | Shipped | +0.13 (+0.04 to +0.18) | +0.15 (+0.09 to +0.25) | +1.54 (+0.60 to +2.10) |
| B: known rallies, video-robust courts | grid-017 | +0.15 (+0.05 to +0.20) | +0.19 (+0.06 to +0.29) | +1.68 (+1.32 to +1.89) |
| C: known rallies, MonoTrack courts | Shipped | +0.13 (+0.05 to +0.18) | +0.16 (+0.09 to +0.25) | +1.51 (+0.57 to +2.07) |
| C: known rallies, MonoTrack courts | grid-017 | +0.15 (+0.05 to +0.21) | +0.19 (+0.07 to +0.28) | +1.68 (+1.32 to +1.89) |

### grid-017 minus shipped at ±10

Same format as the table above.

| Condition | Contact tree | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | ---: | ---: | ---: |
| A: discovered rallies | 32 videos | −0.07 (−0.14 to 0.00) | +0.07 (−0.17 to +0.20) | −0.37 (−1.08 to +0.42) |
| A: discovered rallies | 40 videos | −0.05 (−0.17 to +0.01) | +0.13 (−0.14 to +0.45) | −0.44 (−1.53 to +0.03) |
| B: known rallies, video-robust courts | 32 videos | −0.02 (−0.12 to +0.04) | −0.11 (−0.25 to +0.08) | −0.25 (−0.99 to +0.96) |
| B: known rallies, video-robust courts | 40 videos | 0.00 (−0.11 to +0.03) | −0.07 (−0.18 to +0.12) | −0.10 (−1.14 to +0.42) |
| C: known rallies, MonoTrack courts | 32 videos | −0.02 (−0.12 to +0.04) | −0.11 (−0.24 to +0.08) | −0.25 (−1.05 to +0.93) |
| C: known rallies, MonoTrack courts | 40 videos | 0.00 (−0.11 to +0.04) | −0.07 (−0.18 to +0.10) | −0.08 (−1.08 to +0.39) |

## What was tested

Question: do the trees do better when trained on the [tuned heuristic](../heuristic_only_tune/RESULTS.md)'s preprocessing settings?

The grid:

- four settings take either the shipped value or the tuned heuristic's value
- rest speed takes the shipped value and two values between it and the tuned heuristic's 1/60
- smoothing stays at the shipped three frames
- 48 combinations in total, one of which is the shipped setup

| Setting (30 fps base) | Shipped | Grid values | Tuned heuristic |
| --- | --- | --- | --- |
| `rest_speed` | 0.002 | 0.002, 1/240, 1/120 | 1/60 |
| `end_rest_frames` | 90 | 90, 54 | 54 |
| `gap_state_demotion_bound` | 75 | 75, 30 | 30 |
| `body_unit_half_window` | 12 | 12, 3 | 3 |
| `span_open` | `BACK_FILL` | `BACK_FILL`, `REGION_START` | `REGION_START` |

Only the preprocessing changes. For each setting, the trees are refit on that setting's evidence, with the shipped tree hyperparameters, seeds, training groups and fitting order. Settings file: [grid-48.json.gz](grid-48.json.gz).

## Selection

Data used for selection: the original ShuttleSet split. Test videos played no part.

- training: 32 videos, 2,691 rallies, 27,571 contacts
- validation: 8 videos, 668 rallies, 5,696 contacts

1. **Grid.** Each setting was fitted once on the 32 training videos and scored on them by cross-fitting. The videos fall into four groups of eight. Each video is scored by a contact tree and sequence models that left its group out. One indirect link remains: the sequence models train on contact predictions for the other groups, made by contact trees that saw the scored group. So the training score is held out stage by stage, not end to end.
2. **Baseline check.** The shipped refit matched all 32 training-video prediction streams of the earlier production refit exactly.
3. **Shortlist.** The top four settings by training correct-player F1 at ±10. Ties break on fully correct rallies, then correct-player F1 at ±5. Shipped is added if it isn't already in; here it ranked fourth.
4. **Repeats.** Four more seed offsets for each shortlisted setting: 100, 200, 300 and 400. An offset shifts every random seed in the contact, serve, sequence and confidence models. Offset 0 is the original grid fit.
5. **Choice.** Winner: highest mean validation correct-player F1 at ±10 over the five seeds. Ties break on mean fully correct rallies, then mean correct-player F1 at ±5, and shipped wins an exact tie. Training scores don't count at this step.
6. **Test.** Winner and shipped, all five seeds. For each seed, the contact tree was refit on all 40 original videos. The sequence, serve and confidence models stayed as the matching 32-video fits. No seed or setting was picked from test scores.

### Grid training scores

- Training correct-player F1 at ±10: 88.76% to 89.29%.
- Shipped: 89.27%, ranked fourth. The top four are within 0.02 points of each other.
- Only 26 distinct sets of per-video scores came out of the 48 settings. The gap-demotion bound made no difference in 22 of the 24 pairs that differ only in that setting.

Full table: end of this report.

### Validation

Values are percentages, except rallies. Means are over five seeds.

| Configuration | Training correct-player F1 ±10 | Validation correct-player F1 ±10 | Seed range | Seed SD | Validation fully correct rallies (of 668) |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid-017 | 89.04 | 84.92 | 84.49–85.20 | 0.27 | 282.4 |
| grid-021 | 89.04 | 84.92 | 84.49–85.20 | 0.27 | 282.4 |
| Shipped | 89.10 | 84.63 | 84.15–85.05 | 0.38 | 279.4 |
| grid-083 | 89.13 | 84.16 | 83.66–84.63 | 0.38 | 277.0 |

grid-017 and grid-021 are in effect the same setting. They differ only in the gap-demotion bound and made identical validation predictions at every seed. grid-017 is reported because it comes first in the frozen order.

grid-017's validation lead over shipped: 0.28 points (per seed: +1.05, −0.36, +0.52, 0.00, +0.21).

grid-017 settings: `rest_speed` 0.002, `end_rest_frames` 54, `gap_state_demotion_bound` 75, `body_unit_half_window` 12, `span_open` `REGION_START`.

IDs:

- grid-017: `a521305d102725a9673c7ca76934cc2eb42106ca61285b754913aee0c14bc762`
- shipped: `a322221a4890572e9065814a2e4e3e0ad033ba679e9b60a46ed38d9b639d7cd6`

## Scoring

- Matched contact: within ±10 or ±5 frames of a labelled contact.
- Correct-player contact: matched, with the right hitter.
- Contacts are counted across the whole video, so false positives outside labelled rallies lower precision.

- Fully correct rally: one discovered sequence holding exactly one complete labelled rally, with every contact and player right. Splitting or merging a rally fails. The annotator finds its own rally boundaries.

The [MonoTrack benchmark](../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup.md) scores rallies inside supplied labelled windows instead, so its rally percentages can't be compared with these.

Scoring code: `score_stream` in [heuristic_only_tune/data.py](../heuristic_only_tune/data.py).

## Compute

74 fits in total: 48 grid fits, 16 seed repeats and 10 final contact refits. Per grid fit: 16.7–17.0 GiB peak memory, 1.1–6.3 hours, one or two numerical threads.

The 32-video and 40-video comparison added 50 test runs with saved models and no fitting. They took about 40 minutes on 28 cores.

## Grid training table

Single seed, cross-fitted on the 32 training videos. Settings column order: rest speed, end-rest frames, gap-demotion bound, body half-window, span opening.

| Rank | Label | Settings | Correct-player F1 ±10 | Fully correct rallies ±10 | Correct-player F1 ±5 |
|---:|---|---|---:|---:|---:|
| 1 | grid-017 | rest 0.0020, end 54, gap 75, body 12, REGION_START | 89.29 | 1243 | 87.20 |
| 2 | grid-021 | rest 0.0020, end 54, gap 30, body 12, REGION_START | 89.29 | 1243 | 87.20 |
| 3 | grid-083 | rest 0.0083, end 54, gap 75, body 3, REGION_START | 89.27 | 1254 | 87.21 |
| 4 | shipped | shipped | 89.27 | 1229 | 87.23 |
| 5 | grid-004 | rest 0.0020, end 90, gap 30, body 12, BACK_FILL | 89.27 | 1229 | 87.23 |
| 6 | grid-016 | rest 0.0020, end 54, gap 75, body 12, BACK_FILL | 89.26 | 1240 | 87.26 |
| 7 | grid-020 | rest 0.0020, end 54, gap 30, body 12, BACK_FILL | 89.26 | 1240 | 87.26 |
| 8 | grid-080 | rest 0.0083, end 54, gap 75, body 12, BACK_FILL | 89.25 | 1247 | 87.03 |
| 9 | grid-084 | rest 0.0083, end 54, gap 30, body 12, BACK_FILL | 89.25 | 1247 | 87.03 |
| 10 | grid-049 | rest 0.0042, end 54, gap 75, body 12, REGION_START | 89.24 | 1251 | 87.14 |
| 11 | grid-053 | rest 0.0042, end 54, gap 30, body 12, REGION_START | 89.24 | 1251 | 87.14 |
| 12 | grid-085 | rest 0.0083, end 54, gap 30, body 12, REGION_START | 89.18 | 1247 | 87.08 |
| 13 | grid-081 | rest 0.0083, end 54, gap 75, body 12, REGION_START | 89.11 | 1232 | 86.93 |
| 14 | grid-001 | rest 0.0020, end 90, gap 75, body 12, REGION_START | 89.10 | 1222 | 87.10 |
| 15 | grid-005 | rest 0.0020, end 90, gap 30, body 12, REGION_START | 89.10 | 1222 | 87.10 |
| 16 | grid-050 | rest 0.0042, end 54, gap 75, body 3, BACK_FILL | 89.09 | 1244 | 86.97 |
| 17 | grid-054 | rest 0.0042, end 54, gap 30, body 3, BACK_FILL | 89.09 | 1244 | 86.97 |
| 18 | grid-082 | rest 0.0083, end 54, gap 75, body 3, BACK_FILL | 89.08 | 1235 | 86.86 |
| 19 | grid-086 | rest 0.0083, end 54, gap 30, body 3, BACK_FILL | 89.08 | 1235 | 86.86 |
| 20 | grid-032 | rest 0.0042, end 90, gap 75, body 12, BACK_FILL | 89.07 | 1243 | 87.04 |
| 21 | grid-036 | rest 0.0042, end 90, gap 30, body 12, BACK_FILL | 89.07 | 1243 | 87.04 |
| 22 | grid-051 | rest 0.0042, end 54, gap 75, body 3, REGION_START | 89.06 | 1244 | 86.87 |
| 23 | grid-055 | rest 0.0042, end 54, gap 30, body 3, REGION_START | 89.06 | 1244 | 86.87 |
| 24 | grid-048 | rest 0.0042, end 54, gap 75, body 12, BACK_FILL | 89.02 | 1243 | 87.02 |
| 25 | grid-052 | rest 0.0042, end 54, gap 30, body 12, BACK_FILL | 89.02 | 1243 | 87.02 |
| 26 | grid-087 | rest 0.0083, end 54, gap 30, body 3, REGION_START | 89.01 | 1221 | 87.01 |
| 27 | grid-002 | rest 0.0020, end 90, gap 75, body 3, BACK_FILL | 89.00 | 1229 | 86.93 |
| 28 | grid-006 | rest 0.0020, end 90, gap 30, body 3, BACK_FILL | 89.00 | 1229 | 86.93 |
| 29 | grid-035 | rest 0.0042, end 90, gap 75, body 3, REGION_START | 88.99 | 1239 | 87.04 |
| 30 | grid-039 | rest 0.0042, end 90, gap 30, body 3, REGION_START | 88.99 | 1239 | 87.04 |
| 31 | grid-033 | rest 0.0042, end 90, gap 75, body 12, REGION_START | 88.98 | 1228 | 86.99 |
| 32 | grid-037 | rest 0.0042, end 90, gap 30, body 12, REGION_START | 88.98 | 1228 | 86.99 |
| 33 | grid-065 | rest 0.0083, end 90, gap 75, body 12, REGION_START | 88.98 | 1214 | 86.86 |
| 34 | grid-069 | rest 0.0083, end 90, gap 30, body 12, REGION_START | 88.98 | 1214 | 86.86 |
| 35 | grid-066 | rest 0.0083, end 90, gap 75, body 3, BACK_FILL | 88.97 | 1237 | 86.90 |
| 36 | grid-070 | rest 0.0083, end 90, gap 30, body 3, BACK_FILL | 88.97 | 1237 | 86.90 |
| 37 | grid-019 | rest 0.0020, end 54, gap 75, body 3, REGION_START | 88.95 | 1233 | 86.99 |
| 38 | grid-023 | rest 0.0020, end 54, gap 30, body 3, REGION_START | 88.95 | 1233 | 86.99 |
| 39 | grid-018 | rest 0.0020, end 54, gap 75, body 3, BACK_FILL | 88.92 | 1253 | 86.97 |
| 40 | grid-022 | rest 0.0020, end 54, gap 30, body 3, BACK_FILL | 88.92 | 1253 | 86.97 |
| 41 | grid-067 | rest 0.0083, end 90, gap 75, body 3, REGION_START | 88.89 | 1216 | 86.85 |
| 42 | grid-071 | rest 0.0083, end 90, gap 30, body 3, REGION_START | 88.89 | 1216 | 86.85 |
| 43 | grid-003 | rest 0.0020, end 90, gap 75, body 3, REGION_START | 88.86 | 1240 | 86.88 |
| 44 | grid-007 | rest 0.0020, end 90, gap 30, body 3, REGION_START | 88.86 | 1240 | 86.88 |
| 45 | grid-034 | rest 0.0042, end 90, gap 75, body 3, BACK_FILL | 88.83 | 1221 | 86.80 |
| 46 | grid-038 | rest 0.0042, end 90, gap 30, body 3, BACK_FILL | 88.83 | 1221 | 86.80 |
| 47 | grid-064 | rest 0.0083, end 90, gap 75, body 12, BACK_FILL | 88.76 | 1207 | 86.71 |
| 48 | grid-068 | rest 0.0083, end 90, gap 30, body 12, BACK_FILL | 88.76 | 1207 | 86.71 |
