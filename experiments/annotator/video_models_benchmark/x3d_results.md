# X3D-M contact detection: validation pilot

We fine-tuned a Kinetics-400 X3D-M video model to mark when each racket hit
happens and which player made it, with the rally bounds given. This pilot
chose the training configuration on the 8 validation videos: 668 rallies
holding 5,696 labelled hits.

## Outcome

**Scheme A with 50% contact clips per batch and learning rate 0.02 was selected.**
Its two seeds get 46.7% of validation rallies fully correct on average
(48.1% and 45.4%), with 91.1% correct-player F1 and 95.7% contact F1.
The next-best configuration averages 44.1% fully correct rallies.

The held-out test evaluation uses these two [checkpoints](weights/README.md)
as saved: seed 0 after 25,000 updates and seed 1 after 35,000. This report
covers the completed validation pilot. The
[MonoTrack comparison](../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md)
describes the shared test population and scoring.

Table 1. All six configurations at ±10 frames, ranked by mean fully correct
rallies, then mean correct-player F1. Each run is scored at its
lowest-validation-loss checkpoint.

| Rank | Configuration | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean | Contact F1 (%), mean | Selected checkpoints |
|---:|---|---:|---:|---:|---|
| 1 | Scheme A, 50% contacts, LR 0.02 | 48.1 / 45.4 / **46.7** | 92.2 / 90.1 / **91.1** | 95.7 | 25k / 35k |
| 2 | Scheme B, 25% contacts, LR 0.005 | 44.0 / 44.2 / **44.1** | 91.9 / 91.7 / **91.8** | 95.5 | 30k / 30k |
| 3 | Scheme A, 25% contacts, LR 0.005 | 42.5 / 44.2 / **43.3** | 90.8 / 91.2 / **91.0** | 95.6 | 25k / 25k |
| 4 | Scheme A, 25% contacts, LR 0.02 | 43.0 / 42.2 / **42.6** | 91.3 / 92.7 / **92.0** | 95.6 | 25k / 25k |
| 5 | Scheme A, 50% contacts, LR 0.005 | 41.3 / 40.0 / **40.6** | 87.4 / 89.2 / **88.3** | 95.1 | 25k / 35k |
| 6 | Scheme B, 50% contacts, LR 0.02 | 46.7 / 31.7 / **39.2** | 91.4 / 80.9 / **86.2** | 94.0 | 25k / 15k |

![Dot plots of contact F1, correct-player F1 and fully correct rallies for the six configurations](writeup_figures/fig1_configurations.png)

Figure 1. The three scores for each configuration. Dots are the mean of seeds 0
and 1; black lines join the two seeds. Axes do not start at zero.

Continuation supplied the selected checkpoint in 2 of 12 runs, including
seed 1 of the winning configuration. The other ten were selected within
the original 30,000 updates.

![Training and validation loss curves for all 12 runs, with the selected checkpoints circled](writeup_figures/fig2_loss_curves.png)

Figure 2. Loss at each validation check for all 12 runs, including
continuation. Circles mark the selected checkpoints. Loss compares within a
row only, because the two schemes label different frames as contacts.

## Method

The model sees 16 frames around a centre frame and predicts one of three
classes for that frame: no contact, bottom-player contact or top-player
contact. Scheme A labels frames within one frame of a labelled hit as contacts
and leaves frames exactly two away out of the loss. Scheme B labels frames
within two frames as contacts. Contact share is the number of each batch's 16
clips centred on a contact: 4 at 25%, 8 at 50%.

The scheme A grid crossed contact shares of 25% and 50% with learning rates of
0.005 and 0.02, on seeds 0 and 1. Scheme B then reran the two best scheme A
configurations on the same seeds, from fresh Kinetics-400 weights. Every run
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
of the files and each run's selected checkpoint. To redraw both figures and
print Table 1, run this from the repository root in an environment with
matplotlib, NumPy and pandas:

```bash
python experiments/annotator/video_models_benchmark/writeup_scripts/make_x3d_figures.py
```
