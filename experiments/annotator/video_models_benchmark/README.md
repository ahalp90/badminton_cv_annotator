# Video-model contact benchmarks

This benchmark compares pretrained RGB video models with the badminton
annotator and MonoTrack. Models predict contact times and hitters inside
supplied rally windows, using the existing training, validation and test
splits.

## X3D-M experiment

The first experiment adapts Kinetics-400 X3D-M with a three-class classifier:

| Class | Meaning |
|---|---|
| 0 — NO_CONTACT | No contact near the target frame |
| 1 — BOT_CONTACT | Contact by the player at the bottom of the image |
| 2 — TOP_CONTACT | Contact by the player at the top of the image |

Each input contains 16 consecutive frames. Frames `t-8` through `t+7` predict
the target at frame `t`. The full image is resized proportionally and padded
to 224 × 224. Training augmentation varies between clips and remains
consistent across the frames within each clip.

Apply horizontal flip, random square crop and colour augmentation independently
with probability 0.5 each. The crop retains 80–100% of the padded image area
before resizing to 224 × 224. Brightness, contrast and saturation factors range
from 0.8 to 1.2. Apply these transformations on the GPU, consistently across
each clip. Omit MixUp, CutMix and perspective augmentation for the pilot.

Two training targets are compared:

| Scheme | Positive centres | Ignored centres |
|---|---|---|
| A | Within one frame of a contact | Two frames from a contact, unless another contact makes the centre positive |
| B | Within two frames of a contact | None |

Other centres inside the supplied rally windows are background. When both
players have contacts within the positive band, TOP takes precedence,
following HitNet's class-ID ordering. Both schemes use the original event
annotations for evaluation.

Fine-tune the full network from the pretrained weights, beginning with a short
linear learning-rate warm-up. Each epoch visits the eligible training windows
in shuffled order, followed by evaluation on the complete validation set.
Reduce the learning rate tenfold when validation loss plateaus and use early
stopping, retaining the best validation checkpoint. Use cross-entropy without
label smoothing and retain X3D's training-time classifier dropout of 0.5.

Start with batch size 16 and SGD with learning rate 0.02, momentum 0.9 and
weight decay `1e-5`. Warm up from learning rate 0.002; determine the duration
after measuring throughput. Use ReduceLROnPlateau patience 3 and early-stopping
patience 15, with a maximum of 1,000 epochs. Confirm these intervals against
measured epoch duration before training. The first pilot uses one seed and
no class weighting or background resampling; the final comparison uses five
seeds.

## Evaluation

Supplied windows extend one second before the first labelled contact and one
second after the last, using the stored benchmark bounds. Input clips can
read surrounding video frames for context.

At each centre frame, take argmax over the three classes and discard
NO_CONTACT predictions. Within each rally, greedily retain the strongest
player-class probabilities, suppressing candidates within six frames at 30 fps,
scaled to the source frame rate. Equal-score ties favour earlier frames.
Choose the alternating TOP/BOT sequence agreeing with more retained player
predictions; retain their original player labels if the two sequences tie.
These fixed rules follow the annotator's contact suppression and alternation.
No confidence-threshold or spacing search is used.

Select checkpoints by validation loss. Select experiment configurations by
fully correct validation rallies, with correct-player contact F1 breaking
ties, following the [MonoTrack benchmark](../monotrack_benchmark/README.md).
Report contact F1, correct-player F1 and fully correct rallies using the
existing scorer and its frame-rate-scaled timing tolerances.

## References

- [X3D paper](https://arxiv.org/abs/2004.04730) and
  [released X3D-M configuration](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/X3D_M.yaml)
- [E2E-Spot](https://arxiv.org/abs/2207.10213), including its centre-frame
  video-classification baseline, VC-Spot
