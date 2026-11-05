# X3D-M contact detection: validation and test results

We fine-tuned a Kinetics-400 X3D-M video model to mark when each racket hit
happens and which player made it, with the rally bounds given. The validation
pilot chose the training configuration on 8 videos: 668 rallies holding
5,696 labelled hits. The two selected seeds then completed the test
on 46 videos: 3,327 rallies holding 37,184 labelled hits.

## Outcome

**X3D-M gets 55.95% of test rallies fully correct at stride 3, offset 0 and ±10-frame tolerance,**
compared with 60.34% for our annotator and 47.83% for MonoTrack. Its correct-player
F1 is 89.95%, below both systems. The two X3D seeds span 52.42–59.48% fully
correct rallies; the comparison systems each use five seeds.

## Validation selection

**Positives within ±1 frame, 50% contact clips per batch and learning rate 0.02
were selected.**
The selected configuration's two seeds get 46.7% of validation rallies fully correct on average
(48.1% and 45.4%), with 91.1% correct-player F1 and 95.7% contact F1.
The next-best configuration averages 44.1% fully correct rallies.

Native stride-three validation of the same selected checkpoints raises fully
correct rallies to 50.15% (51.95% and 48.35%), with 92.51% correct-player F1
and 96.03% contact F1. Both runs evaluate 60,059 of 180,179 centres at offset 0.

The test evaluation used these two [checkpoints](weights/README.md)
as saved: seed 0 after 25,000 updates and seed 1 after 35,000. Settings and
checkpoints were fixed before testing. Each checkpoint minimised validation
cross-entropy; mean fully correct validation rallies selected the configuration,
with correct-player F1 breaking ties.

Table 1. All six configurations at ±10 frames, ranked by mean fully correct
rallies, then mean correct-player F1, using stride 1 and enforced alternation. Each run is scored at its
lowest-validation-loss checkpoint.

| Rank | Configuration | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean | Contact F1 (%), mean | Selected checkpoints |
|---:|---|---:|---:|---:|---|
| 1 | Positive centres ±1 frame, 50% contacts, LR 0.02 | 48.1 / 45.4 / **46.7** | 92.2 / 90.1 / **91.1** | 95.7 | 25k / 35k |
| 2 | Positive centres ±2 frames, 25% contacts, LR 0.005 | 44.0 / 44.2 / **44.1** | 91.9 / 91.7 / **91.8** | 95.5 | 30k / 30k |
| 3 | Positive centres ±1 frame, 25% contacts, LR 0.005 | 42.5 / 44.2 / **43.3** | 90.8 / 91.2 / **91.0** | 95.6 | 25k / 25k |
| 4 | Positive centres ±1 frame, 25% contacts, LR 0.02 | 43.0 / 42.2 / **42.6** | 91.3 / 92.7 / **92.0** | 95.6 | 25k / 25k |
| 5 | Positive centres ±1 frame, 50% contacts, LR 0.005 | 41.3 / 40.0 / **40.6** | 87.4 / 89.2 / **88.3** | 95.1 | 25k / 35k |
| 6 | Positive centres ±2 frames, 50% contacts, LR 0.02 | 46.7 / 31.7 / **39.2** | 91.4 / 80.9 / **86.2** | 94.0 | 25k / 15k |

![Dot plots of contact F1, correct-player F1 and fully correct rallies for the six configurations](writeup_figures/fig1_configurations.png)

Figure 1. The first six rows use stride 1; the final row evaluates the selected configuration natively at stride 3, offset 0. All use enforced alternation. Dots are the mean of seeds 0
and 1; black lines join the two seeds. Rows are grouped by positive-centre window, contact proportion and learning rate.

Continuation supplied the selected checkpoint in 2 of 12 runs, including
seed 1 of the winning configuration. The other ten were selected within
the original 30,000 updates.

![Training and validation loss curves for all 12 runs, with the selected checkpoints circled](writeup_figures/fig2_loss_curves.png)

Figure 2. Loss at each validation check for all 12 runs, including
continuation. Circles mark the selected checkpoints. Loss compares within a
row only, because the positive windows label different frames as contacts.

## Test comparison

All three systems receive the same rally bounds. Our annotator and MonoTrack
each use their own court estimates, corresponding to `annotator_our_courts`
and `monotrack_own_courts` in the
[comparison scores](../monotrack_benchmark/benchmark_results/discovery_context/primary/contact_summary.json.gz).
X3D uses RGB clips sampled at stride 3, offset 0. Timing tolerances are ±5 and ±10 frames at 30 fps, scaled
to each video's frame rate.

Table 2. Test scores in percent: seed mean (minimum–maximum). X3D uses two
seeds; our annotator and MonoTrack each use five. Ranges show observed seed
variation. X3D uses stride 3, offset 0 and enforced alternation.

| Tolerance | System | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |
|---|---|---:|---:|---:|
| ±10 frames | X3D-M | 96.90 (96.52–97.28) | 89.95 (88.36–91.54) | 55.95 (52.42–59.48) |
| ±10 frames | Our annotator, our courts | 96.75 (96.67–96.88) | 94.75 (94.48–95.02) | 60.34 (59.30–61.29) |
| ±10 frames | MonoTrack, its own courts | 96.04 (95.95–96.18) | 95.84 (95.69–95.99) | 47.83 (46.44–49.17) |
| ±5 frames | X3D-M | 95.72 (95.19–96.26) | 88.96 (87.27–90.65) | 49.46 (45.60–53.32) |
| ±5 frames | Our annotator, our courts | 95.01 (94.81–95.21) | 93.16 (92.99–93.30) | 50.08 (49.05–51.73) |
| ±5 frames | MonoTrack, its own courts | 94.86 (94.75–95.06) | 94.75 (94.59–94.97) | 40.26 (38.71–41.93) |

![Test contact F1, correct-player F1 and fully correct rallies for VideoMAE, X3D-M, our annotator and MonoTrack](writeup_figures/test_comparison.png)

Figure 3. Shared test comparison at ±10 frames, with enforced alternation. Both video models use stride 3, offset 0. Dots show seed means; black lines span the seeds.

X3D's contact F1 is close to both comparison systems, but assigning the hitter
remains a larger source of error. Our annotator has the highest mean share of
fully correct rallies at both tolerances. X3D's mean rally score exceeds
MonoTrack's for both seeds at both tolerances.

The existing
[MonoTrack report](../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md)
contains the test-population details, rally-discovery results and court-detector
comparison.

## Method

The model sees 16 frames around a centre frame and predicts one of three
classes for that frame: no contact, bottom-player contact or top-player
contact. The ±1-frame window labels frames within one frame of a labelled hit
as contacts and leaves frames exactly two away out of the loss. The ±2-frame
window labels frames within two frames as contacts. Contact share is the number of each batch's 16
clips centred on a contact: 4 at 25%, 8 at 50%.

The ±1-frame runs crossed contact shares of 25% and 50% with learning rates of
0.005 and 0.02, on seeds 0 and 1. We then reran the two best configurations
with the ±2-frame window on the same seeds, from fresh Kinetics-400 weights. Every run
trained the whole network with SGD for 30,000 updates (batch 16, 2,000 warm-up
updates, then cosine decay). Continuation followed on a repeating
learning-rate cycle, checking every 5,000 updates, and stopped after three
checks in a row without a lower validation loss, counting from the original
phase. Each run keeps the checkpoint with the lowest cross-entropy on the whole
validation set, the earliest on a tie.

Rally bounds run from 1 second before each rally's first labelled hit to 1
second after its last. Each frame takes its most likely class. Fixed rules then
keep the strongest predicted hit within 6 frames and make hitters alternate,
with no threshold search. A predicted hit matches a labelled hit within ±10
frames at 30 fps (one third of a second), scaled for other frame rates.

- **Contact F1** combines the share of predicted hits that match a labelled hit
  with the share of labelled hits found.
- **Correct-player F1** also needs the right hitter.
- **A fully correct rally** has every hit found, no extra hits and every hitter
  right.

See the [benchmark README](README.md) for the full method and commands.

## Evidence and figures

Settings and full validation histories for all 12 runs are in
[`results/x3d_pilot/`](results/x3d_pilot/), with an [index](results/README.md)
of the files and each run's selected checkpoint. Test event predictions and
dense scores for both seeds are in [`results/x3d_test/`](results/x3d_test/). Stride-three validation and test scores are in [`results/x3d_stride3/`](results/x3d_stride3/).
To redraw the two X3D pilot figures and print Table 1, run this from the repository root
in an environment with matplotlib, NumPy and pandas:

```bash
python experiments/annotator/video_models_benchmark/writeup_scripts/make_x3d_figures.py
```

Generate the shared test comparison with `writeup_scripts/make_videomae_figures.py`.

The stride-three test scores were derived from the saved dense probability arrays.
Validation at stride three used native inference of only the sampled centres.

Collect the compact scores before regenerating the plots:

```bash
PYTHONPATH=src:. python -m experiments.annotator.video_models_benchmark.writeup_scripts.collect_x3d_stride3 \
  --validation /path/to/native-validation-results \
  --test /path/to/rescored-test-results
```
