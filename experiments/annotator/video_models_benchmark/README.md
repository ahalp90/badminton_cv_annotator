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

Two training targets are compared:

| Scheme | Positive centres | Ignored centres |
|---|---|---|
| A | Within one frame of a contact | Two frames from a contact, unless another contact makes the centre positive |
| B | Within two frames of a contact | None |

Other centres inside the supplied rally windows are background. When both
players have contacts within the positive band, TOP takes precedence,
following HitNet's class-ID ordering. Both schemes use the original event
annotations for evaluation.

Train the new classifier with the pretrained network frozen, then fine-tune
the full network from the selected head-only checkpoint. Retain results from
both stages. Use cross-entropy without label smoothing and retain X3D's
training-time classifier dropout of 0.5.

## Evaluation

Supplied windows extend one second before the first labelled contact and one
second after the last, using the stored benchmark bounds. Input clips can
read surrounding video frames for context.

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
